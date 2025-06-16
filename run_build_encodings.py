from src.system.data_manager import DataManager
from src.system.face_processor import FaceProcessor
from src.system.services import EncodingService


def main():
    # データセットから顔のエンコーディングを構築する
    print("--- Starting Encoding Build Process ---")

    # テスト用の画像ディレクトリ（3人＊10imgs）
    data_manager = DataManager(dataset_path="test_user_imgs")

    face_processor = FaceProcessor()

    encoding_service = EncodingService(
        data_manager=data_manager, face_processor=face_processor
    )

    # --- 2. テスト用のメタデータを準備 ---
    # 'test_user_imgs'内のディレクトリ名と一致する一時的なメタデータを作成します。
    print("Creating temporary metadata for test users...")
    test_metadata = [
        {"user_id": "testuser01", "name": "Test_User 01", "registered_at": ""},
        {"user_id": "testuser02", "name": "Test_User 02", "registered_at": ""},
        {"user_id": "testuser03", "name": "Test_User 03", "registered_at": ""},
    ]
    data_manager.write_metadata(test_metadata)

    # エンコードサービスを実行して、encodings.pickleを構築
    encoding_service.build_encodings_from_dataset()

    print("Encoding Build Process Finished.")


if __name__ == "__main__":
    main()
