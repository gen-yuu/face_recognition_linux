import json
import os
import pickle
import traceback
from typing import Dict, List, Optional

import cv2
import numpy as np

from ..utils.logger import setup_logger


class DataManager:
    """
    ユーザーのメタデータ、顔画像、学習済みのエンコーディングデータを操作するクラス
    """

    def __init__(
        self,
        dataset_path: str = "dataset",
        metadata_path: str = "metadata.json",
        encodings_path: str = "encodings.pickle",
    ):
        """
        DataManagerクラスのコンストラクタ

        Args:
            dataset_path (str): ユーザーの顔画像を格納するディレクトリへのパス
            metadata_path (str): ユーザーのメタデータを格納するJSONファイルへのパス
            encodings_path (str): 顔のエンコーディングを格納するpickleファイルへのパス
        """
        self.dataset_path = dataset_path
        self.metadata_path = metadata_path
        self.encodings_path = encodings_path
        self.logger = setup_logger(__name__)

        # 存在しない場合は作成
        self._initialize_storage()
        self.logger.info("DataManager initialized.")

    def _initialize_storage(self):
        """
        必要なファイルやディレクトリを作成してストレージを初期化する
        """
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)
            self.logger.info(
                "Created dataset directory",
                extra={"dataset_path": self.dataset_path},
            )

        if not os.path.exists(self.metadata_path):
            self.write_metadata([])  # 空のメタデータファイルを作成
            self.logger.info(
                "Created metadata file",
                extra={"metadata_path": self.metadata_path},
            )

    def read_metadata(self) -> List[Dict]:
        """
        JSONファイルからユーザーのメタデータを読み込む

        Returns:
            List[Dict]: 各ユーザーを辞書として表現したリスト
                         ファイルが存在しない場合は空のリストを返す
        """
        try:
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(
                f"Failed to read or parse metadata file: {e}",
                extra={
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                },
            )
            return []

    def write_metadata(self, data: List[Dict]):
        """
        指定されたデータをメタデータJSONファイルに書き込む

        Args:
            data (List[Dict]): ファイルに書き込むユーザーデータの辞書のリスト
        """
        try:
            with open(self.metadata_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(
                "Failed to write to metadata file",
                extra={
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                },
            )

    def save_images_for_user(self, user_id: str, images: List[np.ndarray]) -> List[str]:
        """
        指定されたユーザーの顔画像を保存する

        Args:
            user_id (str): ユーザーの一意な識別子
            images (List[np.ndarray]): OpenCVのndarray形式の画像のリスト

        Returns:
            List[str]: 画像が保存されたファイルパスのリスト
        """
        user_dir = os.path.join(self.dataset_path, user_id)
        os.makedirs(user_dir, exist_ok=True)

        saved_paths = []
        for i, img in enumerate(images):
            # ファイル名を2桁の連番にフォーマット (例: 01.jpg, 02.jpg)
            img_path = os.path.join(user_dir, f"{i+1:02d}.jpg")
            cv2.imwrite(img_path, img)
            saved_paths.append(img_path)

        self.logger.info(f"Saved {len(saved_paths)} images for user_id: {user_id}")
        return saved_paths

    def save_encodings(self, encodings: List[np.ndarray], user_ids: List[str]):
        """
        顔のエンコーディングをユーザーIDと紐付けて保存する

        Args:
            encodings (List[np.ndarray]): 128次元の顔エンコーディングのリスト
            user_ids (List[str]): 各エンコーディングに対応するユーザーIDのリスト
        """
        data = {"encodings": encodings, "user_ids": user_ids}
        try:
            with open(self.encodings_path, "wb") as f:
                f.write(pickle.dumps(data))
            self.logger.info(
                f"Saved {len(encodings)} encodings",
                extra={"encodings_path": self.encodings_path},
            )
        except Exception as e:
            self.logger.error(
                "Failed to write to encodings file",
                extra={
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                },
            )

    def load_encodings(self) -> Optional[Dict[str, List]]:
        """
        pickleファイルから顔のエンコーディングを読み込む

        Returns:
            Optional[Dict[str, List]]: 'encodings'と'user_ids'のリストを含む辞書
        """
        if not os.path.exists(self.encodings_path):
            self.logger.warning("Encodings file not found. Please build it first.")
            return None

        try:
            with open(self.encodings_path, "rb") as f:
                data = pickle.load(f)
                self.logger.info(
                    f"Loaded {len(data.get('encodings', []))} encodings",
                    extra={"encodings_path": self.encodings_path},
                )
                return data
        except Exception as e:
            self.logger.error(
                "Failed to load or parse encodings file",
                extra={
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                },
            )
            return None

    def get_image_paths_for_user(self, user_id: str) -> List[str]:
        """
        指定されたユーザーの全ての画像ファイルパスを取得する

        Args:
            user_id (str): ユーザーの一意な識別子

        Returns:
            List[str]: 画像ファイルパスのリスト
        """
        user_dir = os.path.join(self.dataset_path, user_id)
        if not os.path.isdir(user_dir):
            return []

        return [os.path.join(user_dir, fname) for fname in os.listdir(user_dir)]


if __name__ == "__main__":
    manager = DataManager()
    manager.read_metadata()
