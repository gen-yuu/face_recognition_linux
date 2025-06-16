import cv2
import numpy as np

# from src.system.data_manager import DataManager
# from src.system.face_processor import FaceProcessor

# from src.system.services import AuthenticationService


def draw_results(frame: np.ndarray, results: list) -> np.ndarray:
    """
    認証結果をフレームに描画します。

    Args:
        frame (np.ndarray): 描画対象のフレーム。
        results (list): AuthenticationServiceから返された認証結果のリスト。

    Returns:
        np.ndarray: 名前とバウンディングボックスが描画されたフレーム。
    """
    for result in results:
        top, right, bottom, left = result["box"]
        name = result["name"]

        # 顔の周りにボックスを描画（認証成功なら緑、不明なら赤）
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        # 名前のラベルを描画
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    return frame


def main():
    """
    顔認証のアプリケーションフローを実行
    """
    print("Starting Real-time Authentication")

    # data_manager = DataManager()
    # face_processor = FaceProcessor()
    # auth_service = AuthenticationService(
    #     data_manager=data_manager, face_processor=face_processor, tolerance=0.6
    # )

    # --- 2. カメラの初期化 ---
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Camera opened. Press 'q' to quit.")

    # # --- 3. メインループ ---
    # while True:
    #     # フレームを1枚キャプチャ
    #     ret, frame = cap.read()
    #     if not ret:
    #         print("Error: Failed to capture frame.")
    #         break

    #     # 認証サービスにフレームを渡して、顔認証を実行
    #     recognized_faces = auth_service.authenticate_frame(frame)

    #     # 認証結果をフレームに描画
    #     processed_frame = draw_results(frame, recognized_faces)

    #     # 結果を表示
    #     cv2.imshow("Real-time Face Authentication", processed_frame)

    #     # 'q'キーが押されたらループを抜ける
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

    # # --- 4. クリーンアップ ---
    # cap.release()
    # cv2.destroyAllWindows()
    print("--- Application Closed ---")


if __name__ == "__main__":
    main()
