import os
import time
import traceback

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request
from PIL import Image, ImageDraw, ImageFont

from src.system.camera import Camera, SimulatedCamera
from src.system.data_manager import DataManager
from src.system.face_processor import FaceProcessor
from src.system.services import (
    AuthenticationService,
    EncodingService,
    RegistrationService,
)
from src.utils.logger import setup_logger

# --- アプリケーション設定 ---
USE_REAL_CAMERA = True
REGISTRATION_PASSWORD = "704lIlac"  # 登録モードに入るためのパスワード

app = Flask(__name__)
logger = setup_logger(__name__)


# --- グローバル変数 (アプリケーションの状態管理) ---
class AppState:
    def __init__(self):
        self.mode = "AUTHENTICATING"
        self.captured_frame = None
        self.last_auth_result = {}


app_state = AppState()

# --- グローバルなサービスの初期化 ---
data_manager = DataManager()
face_processor = FaceProcessor()
auth_service = AuthenticationService(data_manager, face_processor, tolerance=0.55)
registration_service = RegistrationService(data_manager)
encoding_service = EncodingService(data_manager, face_processor)

# カメラの初期化
try:
    camera = Camera() if USE_REAL_CAMERA else SimulatedCamera()
except Exception as e:
    # カメラが初期化できない場合はアプリケーションを終了
    logger.error(
        f"Error initializing camera: {e}",
        extra={
            "error": str(e),
            "traceback": traceback.format_exc(),
        },
    )
    exit()

# --- UIとロジックに関する定数 ---
FRAME_WIDTH, FRAME_HEIGHT = 1280, 720
GUIDE_BOX_WIDTH, GUIDE_BOX_HEIGHT = 350, 450
GUIDE_BOX_RECT = (
    (FRAME_WIDTH - GUIDE_BOX_WIDTH) // 2,
    (FRAME_HEIGHT - GUIDE_BOX_HEIGHT) // 2,
    GUIDE_BOX_WIDTH,
    GUIDE_BOX_HEIGHT,
)
POSITION_THRESHOLD, SIZE_THRESHOLD = 50, 0.5
FONT_PATH = "ipaexg.ttf"
if not os.path.exists(FONT_PATH):
    logger.warning(
        "Warning: Font file not found. Japanese text may not display correctly.",
        extra={
            "font_path": FONT_PATH,
        },
    )
    FONT_PATH = ""


# --- ヘルパー関数 ---
def get_box_center(box):
    """
    矩形の中心を計算する
    """
    x, y, w, h = box
    return x + w // 2, y + h // 2


def get_box_area(box):
    """
    矩形の面積を計算する
    """
    _, _, w, h = box
    return w * h


def get_face_properties(location):
    """
    顔の位置とサイズを計算する
    """
    top, right, bottom, left = location
    w = right - left
    h = bottom - top
    area = w * h
    return (left, top, w, h), area


def calculate_face_metrics(face_box, face_area):
    """
    顔の位置と大きさを計算し、その指標を返す
    """
    guide_center = get_box_center(GUIDE_BOX_RECT)
    face_center = get_box_center(face_box)

    distance = np.linalg.norm(np.array(guide_center) - np.array(face_center))
    size_ratio = face_area / get_box_area(GUIDE_BOX_RECT)

    return distance, size_ratio


def draw_japanese_text(image, text, position, font_path, font_size, color):
    """
    Pillowを使って、OpenCVの画像に日本語を描画する
    """
    if not font_path:
        # font_pathが指定されていない場合はデフォルトのフォントを使用
        cv2.putText(
            image, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_size / 30, color, 2
        )
        return image
    try:
        font = ImageFont.truetype(font_path, font_size)
        img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        draw.text(position, text, font=font, fill=color)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except Exception:
        logger.warning(
            "Could not load font file. Falling back to default font.",
            extra={
                "font_path": font_path,
            },
        )
        cv2.putText(
            image, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_size / 30, color, 2
        )
        return image


# --- ビデオストリーミングのメインロジック ---
def generate_frames():
    """
    カメラからフレームを取得し、認証処理を行い、ストリーミングする
    """
    while True:
        # 現在のモードがフリーズ状態の場合、保持しているフレームを使い続ける
        if (app_state.mode == "REGISTRATION_FROZEN") and (
            app_state.captured_frame is not None
        ):
            frame = app_state.captured_frame.copy()
        else:
            frame = camera.get_frame()
            if frame is None:
                logger.warning("Failed to capture frame from camera.")
                time.sleep(0.1)
                # TODO: エラーメッセージを返す,リトライ機構, 再接続処理
                continue

        if app_state.mode == "AUTHENTICATING":
            # --- 認証モードのロジック ---
            handle_authentication_frame(frame)
            pass
        elif app_state.mode in ["REGISTRATION_SEARCHING", "REGISTRATION_FROZEN"]:
            # --- 登録モードのロジック ---
            handle_registration_frame(frame)

        _, buffer = cv2.imencode(".jpg", frame)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )


def handle_authentication_frame(frame):
    """
    認証モード時のフレーム処理と描画を行う
    """
    detected_faces = face_processor.detect_and_encode_faces(frame)
    default_color = (128, 128, 128)
    gx, gy, gw, gh = GUIDE_BOX_RECT
    cv2.rectangle(frame, (gx, gy), (gx + gw, gy + gh), default_color, 3)

    if not detected_faces:
        # 顔が検出されなかった場合
        pass
    else:
        largest_face = max(
            detected_faces, key=lambda f: get_face_properties(f["location"])[1]
        )
        face_box_coords, face_area = get_face_properties(largest_face["location"])
        left, top, w, h = face_box_coords
        distance, size_ratio = calculate_face_metrics(face_box_coords, face_area)
        # 認証処理事前チェック（顔の位置と大きさを確認）
        if distance <= POSITION_THRESHOLD and size_ratio >= SIZE_THRESHOLD:
            logger.info(
                "Face detected and ready for authentication.",
                extra={
                    "distance": distance,
                    "size_ratio": size_ratio,
                },
            )
            # 認証処理の前に先に対象フレームを返しておく
            auth_color = (255, 255, 255)
            auth_message = "認証中..."
            processing_frame = frame.copy()
            cv2.rectangle(
                processing_frame, (left, top), (left + w, top + h), auth_color, 3
            )
            processing_frame = draw_japanese_text(
                processing_frame,
                auth_message,
                (left, top - 40),
                FONT_PATH,
                30,
                auth_color,
            )
            _, buffer = cv2.imencode(".jpg", processing_frame)
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )
            # 認証処理
            logger.info("Starting face authentication.")
            auth_result = auth_service.authenticate_face(largest_face)
            logger.info(
                "Face authentication completed.",
                extra={
                    "user_name": auth_result[0]["name"],
                },
            )
            name = auth_result[0]["name"]

            result_color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            result_frame = frame.copy()
            cv2.rectangle(
                result_frame, (left, top), (left + w, top + h), result_color, 3
            )
            result_frame = draw_japanese_text(
                result_frame,
                f"結果: {name}",
                (left, top - 40),
                FONT_PATH,
                30,
                result_color,
            )

            end_time = time.time() + 3
            while time.time() < end_time:
                _, buffer = cv2.imencode(".jpg", result_frame)
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
                )
            return

        else:
            # 認証処理事前チェックに引っかからなかった場合フィードバックを描画
            feedback_color = (0, 255, 255)
            feedback_message = (
                "顔を枠の中央に"
                if distance > POSITION_THRESHOLD
                else "近づいてください"
            )
            cv2.rectangle(frame, (left, top), (left + w, top + h), feedback_color, 3)
            frame = draw_japanese_text(
                frame,
                feedback_message,
                (left, top - 40),
                FONT_PATH,
                30,
                feedback_color,
            )


def handle_registration_frame(frame):
    """
    登録モード時のフレーム処理と描画を行う
    """
    detected_faces = face_processor.detect_and_encode_faces(frame)
    if not detected_faces:
        app_state.captured_frame = None
        return

    largest_face = max(
        detected_faces, key=lambda f: get_face_properties(f["location"])[1]
    )
    face_box, face_area = get_face_properties(largest_face["location"])
    left, top, w, h = face_box

    distance, size_ratio = calculate_face_metrics(face_box, face_area)

    if (
        app_state.mode == "REGISTRATION_SEARCHING"
        and distance <= POSITION_THRESHOLD
        and size_ratio >= SIZE_THRESHOLD
    ):
        app_state.mode = "REGISTRATION_FROZEN"
        app_state.captured_frame = frame.copy()
        logger.info("Frame captured for registration.")

    if app_state.mode == "REGISTRATION_FROZEN":
        guide_color = (0, 255, 0)
        feedback_message = "この顔で登録します"
    else:  # REGISTRATION_SEARCHING
        guide_color = (0, 255, 255)
        feedback_message = (
            "顔を枠の中央に" if distance > POSITION_THRESHOLD else "近づいてください"
        )

    cv2.rectangle(frame, (left, top), (left + w, top + h), guide_color, 3)
    draw_japanese_text(frame, feedback_message, (left, top - 40), guide_color)


# --- APIエンドポイント ---
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/check_password", methods=["POST"])
def check_password():
    """パスワードを検証する"""
    password = request.form.get("password")
    if password == REGISTRATION_PASSWORD:
        return jsonify({"status": "ok"})
    return jsonify({"status": "error", "message": "パスワードが違います。"}), 401


@app.route("/start_registration", methods=["POST"])
def start_registration():
    """登録モードを開始する"""
    app_state.mode = "REGISTRATION_SEARCHING"
    logger.info("Mode changed to REGISTRATION_SEARCHING")
    return jsonify({"status": "ok"})


@app.route("/recapture", methods=["POST"])
def recapture():
    """キャプチャしたフレームを破棄し、再撮影モードに戻す"""
    app_state.mode = "REGISTRATION_SEARCHING"
    app_state.captured_frame = None
    logger.info("Recapturing. Mode changed to REGISTRATION_SEARCHING")
    return jsonify({"status": "ok"})


@app.route("/cancel_registration", methods=["POST"])
def cancel_registration():
    """登録をキャンセルし、認証モードに戻る"""
    app_state.mode = "AUTHENTICATING"
    app_state.captured_frame = None
    logger.info("Registration cancelled. Mode changed to AUTHENTICATING")
    return jsonify({"status": "ok"})


@app.route("/status")
def status():
    """フロントエンドに現在のアプリケーション状態を返す"""
    return jsonify(
        {
            "mode": app_state.mode,
            "isFrameCaptured": app_state.captured_frame is not None,
        }
    )


@app.route("/submit_registration", methods=["POST"])
def submit_registration():
    """ユーザー名を受け取り、登録を実行する"""
    user_name = request.form.get("name")
    if not user_name or app_state.captured_frame is None:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "名前またはキャプチャされたフレームがありません。",
                }
            ),
            400,
        )

    registration_service.register_new_user(
        name=user_name, images=[app_state.captured_frame]
    )
    encoding_service.build_encodings_from_dataset()
    auth_service.reload_knowledge()

    cancel_registration()  # 状態をリセットして認証モードに戻る
    return jsonify({"status": "ok", "message": f"{user_name}さんを登録しました。"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
