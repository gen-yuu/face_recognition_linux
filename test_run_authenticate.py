# /run_batch_authenticate.py

import os

import cv2
import numpy as np

# srcパッケージから必要なクラスをインポート
from src.system.data_manager import DataManager
from src.system.face_processor import FaceProcessor
from src.system.services import AuthenticationService


def draw_results(frame: np.ndarray, results: list) -> np.ndarray:
    """
    認証結果をフレームに描画する
    """
    for result in results:
        top, right, bottom, left = result["box"]
        name = result["name"]
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    return frame


def main():

    # テストディレクトリ内の画像に対してバッチ顔認証を実行する

    INPUT_DIR = "test_auth_imgs"
    OUTPUT_DIR = "results_dir"

    print("Starting Batch Authentication Process")

    data_manager = DataManager()
    face_processor = FaceProcessor()
    auth_service = AuthenticationService(
        data_manager=data_manager, face_processor=face_processor
    )
    print("Services initialized.")

    # 出力ディレクトリの作成
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

    for root, _, files in os.walk(INPUT_DIR):
        for filename in files:
            # .pgmや他の画像形式を処理対象とする
            if not filename.lower().endswith((".pgm", ".jpg", ".png")):
                continue

            # 入力パスと出力パスを構築
            input_path = os.path.join(root, filename)

            # 出力先のディレクトリ構造を維持する
            relative_path = os.path.relpath(root, INPUT_DIR)
            output_subdir = os.path.join(OUTPUT_DIR, relative_path)
            if not os.path.exists(output_subdir):
                os.makedirs(output_subdir)
            output_path = os.path.join(
                output_subdir, os.path.splitext(filename)[0] + ".jpg"
            )

            print(f"Processing: {input_path}")

            # 画像を読み込み
            frame = cv2.imread(input_path)
            if frame is None:
                print("Error: Failed to read image.")
                continue

            # 認証を実行
            recognized_faces = auth_service.authenticate_frame(frame)

            if recognized_faces:
                print(
                    f"Found {len(recognized_faces)} face(s). Names: \
                        {[face['name'] for face in recognized_faces]}"
                )
            else:
                print("No faces detected.")

            # 結果を描画
            processed_frame = draw_results(frame, recognized_faces)

            # 結果を保存
            cv2.imwrite(output_path, processed_frame)
            print(f"Saved result to: {output_path}")

    print("Batch Authentication Process Finished")


if __name__ == "__main__":
    main()
