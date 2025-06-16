# /test_runner.py (プロジェクトルートに置く)

# srcパッケージから、絶対パスでインポートする
from src.system.data_manager import DataManager

if __name__ == "__main__":
    print("Testing DataManager...")

    # DataManagerをインスタンス化
    # パスがルートからの相対パスになることに注意
    dm = DataManager(
        dataset_path="dataset",
        metadata_path="metadata.json",
        encodings_path="encodings.pickle",
    )

    # 何かメソッドを呼び出してテスト
    metadata = dm.read_metadata()
    print("Successfully read metadata:")
    print(metadata)
