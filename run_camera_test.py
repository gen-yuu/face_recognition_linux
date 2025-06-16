import os
import shutil
import sys
import time

import cv2

# フレームを一時的に保存するdir
TEMP_FRAMES_DIR = os.path.join(os.path.expanduser("~"), "workspace", "temp_frames")

# 保存する最新のフレーム数
MAX_FRAMES_TO_KEEP = 100

# 古いフレームを削除する間隔
CLEANUP_INTERVAL_SECONDS = 5  # Check and cleanup every 5 seconds


def cleanup_old_frames(directory, max_frames):
    """
    Deletes the oldest frames in the specified directory to keep only max_frames.
    """
    try:
        # tmpdirにあるフレームを全て取得（sort by time）
        files = sorted(
            [
                os.path.join(directory, f)
                for f in os.listdir(directory)
                if f.endswith((".jpg", ".jpeg"))
            ],
            key=os.path.getmtime,
        )

        # 許容フレーム数以上あれば許容範囲まで削除
        if len(files) > max_frames:
            files_to_delete = files[: len(files) - max_frames]
            print(f"Cleaning up {len(files_to_delete)} old frames...")
            for f in files_to_delete:
                try:
                    os.remove(f)
                except OSError as e:
                    print(f"Error deleting file {f}: {e}")
    except FileNotFoundError:
        print(f"Cleanup directory '{directory}' not found. Skipping cleanup.")
    except Exception as e:
        print(f"An error occurred during cleanup: {e}")


def main():
    # 一時フレームディレクトの作成
    if not os.path.exists(TEMP_FRAMES_DIR):
        os.makedirs(TEMP_FRAMES_DIR)
        print(f"一時フレームディレクトリを作成しました: {TEMP_FRAMES_DIR}")
    else:
        # 起動時に既存のフレームをクリアしてクリーンな状態にする
        print(f"既存のフレームをクリアしています: {TEMP_FRAMES_DIR}")
        shutil.rmtree(TEMP_FRAMES_DIR)
        os.makedirs(TEMP_FRAMES_DIR)

    camera_index = 0
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(
            f"エラー: カメラインデックス {camera_index} のWebカメラを開けませんでした"
        )
        sys.exit(1)

    print(f"Webカメラがカメラインデックス {camera_index} で正常に開かれました。")
    print(f"フレームを以下のディレクトリに保存します: {TEMP_FRAMES_DIR}")
    print("ストリームを終了するには Ctrl+C を押してください。")

    frame_counter = 0
    last_cleanup_time = time.time()

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("エラー: フレームの取得に失敗しました。終了します。")
                break

            # ユニークなファイル名のためにミリ秒単位のタイムスタンプを使用
            timestamp = int(time.time() * 1000)
            frame_filename = os.path.join(TEMP_FRAMES_DIR, f"frame_{timestamp}.jpg")

            # フレームをJPEG画像として保存
            # 品質は調整可能（0-100、JPEGのデフォルトは95）
            cv2.imwrite(frame_filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_counter += 1

            # 古いフレームを定期的にクリーンアップ
            current_time = time.time()
            if current_time - last_cleanup_time > CLEANUP_INTERVAL_SECONDS:
                cleanup_old_frames(TEMP_FRAMES_DIR, MAX_FRAMES_TO_KEEP)
                last_cleanup_time = current_time

    except KeyboardInterrupt:
        print("\nユーザーによってストリームが停止されました (Ctrl+C)。")
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")
    finally:
        # リソースを解放
        cap.release()
        print("Webカメラのストリームを閉じました。")
        # TODO: webcam終了後tmpdirの削除
        # print(f"Removing all temporary frames from: {TEMP_FRAMES_DIR}")
        # shutil.rmtree(TEMP_FRAMES_DIR)


if __name__ == "__main__":
    main()
