# /app.py

import os
import time
import traceback

import cv2
import numpy as np
from flask import Flask, Response, render_template, request
from PIL import Image, ImageDraw, ImageFont

from src.system.camera import Camera, SimulatedCamera
from src.system.data_manager import DataManager
from src.system.face_processor import FaceProcessor
from src.system.services import AuthenticationService
from src.utils.logger import setup_logger

# --- アプリケーション設定 ---
USE_REAL_CAMERA = True  # デバイスカメラの有無

app = Flask(__name__)
logger = setup_logger(__name__)


# アプリケーション起動時に一度だけ、各サービスをインスタンス化
data_manager = DataManager()
face_processor = FaceProcessor()
auth_service = AuthenticationService(data_manager, face_processor, tolerance=0.55)

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

logger.info("Flask application and services initialized.")


# --- UIとロジックに関する定数 ---
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
FONT_PATH = "ipaexg.ttf"  # 日本語フォントのパス
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


# --- Webカメラストリーミング ---
def generate_frames():
    """
    カメラからフレームを取得し、認証処理を行い、ストリーミングする
    """
    while True:
        frame = camera.get_frame()
        if frame is None:
            logger.warning("Failed to capture frame from camera.")
            # TODO: エラーメッセージを返す,リトライ機構, 再接続処理
            continue

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
                continue

            else:
                # 認証処理事前チェックに引っかからなかった場合フィードバックを描画
                feedback_color = (0, 255, 255)
                feedback_message = (
                    "顔を枠の中央に"
                    if distance > POSITION_THRESHOLD
                    else "近づいてください"
                )
                cv2.rectangle(
                    frame, (left, top), (left + w, top + h), feedback_color, 3
                )
                frame = draw_japanese_text(
                    frame,
                    feedback_message,
                    (left, top - 40),
                    FONT_PATH,
                    30,
                    feedback_color,
                )

        _, buffer = cv2.imencode(".jpg", frame)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )


@app.route("/")
def index():
    """
    メインページを表示する
    """
    logger.info(f"Request received for / from {request.remote_addr}")
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    """
    Webカメラストリーミングを開始する
    """
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    try:
        logger.info("Starting Flask server...")
        app.run(host="0.0.0.0", port=8080, debug=False)
    except Exception as e:
        logger.error(
            "Flask server failed to start",
            extra={"error": str(e), "traceback": traceback.format_exc()},
        )
    finally:
        camera.release()
        logger.info("Flask server shutting down.")
