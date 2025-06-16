# src/system/services.py

import datetime
import uuid
from typing import Dict, List

import cv2
import numpy as np

import face_recognition

from ..utils.logger import setup_logger
from .data_manager import DataManager
from .face_processor import FaceProcessor


class RegistrationService:
    """
    顔登録を担当するクラス

    DataManagerを利用して、ユーザー情報の登録と画像データの保存を行う
    """

    def __init__(self, data_manager: DataManager):
        """
        RegistrationServiceのコンストラクタ

        Args:
            data_manager (DataManager): データ永続化を担当するDataManagerのインスタンス
        """
        self.data_manager = data_manager
        self.logger = setup_logger(__name__)
        self.logger.info("RegistrationService initialized.")

    def register_new_user(self, name: str, images: List[np.ndarray]) -> str:
        """
        新しいユーザーをシステムに登録する

        新しい一意なIDを生成し、メタデータと顔画像の両方をDataManagerを介して保存する

        Args:
            name (str): 登録するユーザーの名前。
            images (List[np.ndarray]): 登録する顔画像のリスト (OpenCV形式)

        Returns:
            str: 生成された新しいユーザーの一意なID (UUID)
        """
        if not name or not images:
            self.logger.error("Registration failed: name or images are empty.")
            raise ValueError

        # 新しいユーザーのための一意なIDを生成
        user_id = str(uuid.uuid4())
        self.logger.info(f"Generated new user_id: {user_id} for name: {name}")

        # 既存のメタデータを読み込み、新しいユーザー情報を追加
        metadata = self.data_manager.read_metadata()
        new_user_data = {
            "user_id": user_id,
            "name": name,
            "registered_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        metadata.append(new_user_data)

        # 更新したメタデータを書き戻す
        self.data_manager.write_metadata(metadata)
        self.logger.info(f"Appended new user to metadata for user_id: {user_id}")

        # 顔画像を保存(dataset/ユーザーID/.jpg)
        self.data_manager.save_images_for_user(user_id, images)

        self.logger.info(
            f"Successfully registered new user '{name}' with ID '{user_id}'."
        )
        return user_id


class EncodingService:
    """
    データセット全体の顔エンコード処理を担当するクラス
    """

    def __init__(self, data_manager: DataManager, face_processor: FaceProcessor):
        """
        EncodingServiceのコンストラクタ

        Args:
            data_manager (DataManager): データ永続化を担当するDataManagerのインスタンス
            face_processor (FaceProcessor): 顔処理を担当するFaceProcessorのインスタンス
        """
        self.data_manager = data_manager
        self.face_processor = face_processor
        self.logger = setup_logger(__name__)
        self.logger.info("EncodingService initialized.")

    def build_encodings_from_dataset(self):
        """
        データセット内の画像を処理し、エンコーディングを構築・保存する
        """
        self.logger.info("Starting to build encodings from dataset")

        metadata = self.data_manager.read_metadata()
        if not metadata:
            self.logger.warning("Metadata is empty. No users to encode.")
            return

        all_known_encodings = []
        all_known_user_ids = []

        for user_data in metadata:
            user_id = user_data["user_id"]
            user_name = user_data["name"]

            self.logger.info(f"Processing user: {user_name} (ID: {user_id})")

            # DataManagerからユーザーの画像パスを取得
            image_paths = self.data_manager.get_image_paths_for_user(user_id)
            if not image_paths:
                self.logger.warning(f"No images found for user {user_name}. Skipping.")
                continue

            for image_path in image_paths:
                image = cv2.imread(image_path)
                if image is None:
                    self.logger.error(f"Failed to read image: {image_path}")
                    continue

                # FaceProcessorを使って、画像内の全ての顔エンコーディングを取得
                encodings = self.face_processor.extract_encodings(image)

                # 顔が1つだけ検出された場合のみ、処理を続行する
                if len(encodings) == 1:
                    all_known_encodings.append(encodings[0])
                    all_known_user_ids.append(user_id)
                elif len(encodings) > 1:
                    self.logger.warning(
                        f"Image contains multiple faces. Skipping: {image_path}"
                    )
                else:
                    self.logger.warning(
                        f"No face detected in image. Skipping: {image_path}"
                    )

        if not all_known_encodings:
            self.logger.error("No valid encodings were generated. Aborting save.")
            return

        # 全ての有効のエンコーディングをファイルに保存
        self.data_manager.save_encodings(
            encodings=all_known_encodings, user_ids=all_known_user_ids
        )
        self.logger.info("Successfully built and saved all valid encodings.")


class AuthenticationService:
    """
    顔認証のロジックを担当するクラス
    """

    def __init__(
        self,
        data_manager: DataManager,
        face_processor: FaceProcessor,
        tolerance: float = 0.6,
    ):
        """
        AuthenticationServiceのコンストラクタ

        起動時に、認証に必要な学習済みデータをメモリにロードする

        Args:
            data_manager (DataManager): データ永続化を担当するインスタンス
            face_processor (FaceProcessor): 顔処理アルゴリズムを担当するインスタンス
            tolerance (float): 顔の類似度の閾値
        """
        self.data_manager = data_manager
        self.face_processor = face_processor
        self.tolerance = tolerance
        self.logger = setup_logger(__name__)

        self.known_encodings = []
        self.known_user_ids = []
        self.user_id_to_name_map = {}

        self._load_knowledge()
        self.logger.info("AuthenticationService initialized.")

    def _load_knowledge(self):
        """
        DataManagerを介して、認証に必要なデータをロードする
        """
        # 顔のエンコーディングをロード
        encoding_data = self.data_manager.load_encodings()
        if encoding_data:
            self.known_encodings = encoding_data.get("encodings", [])
            self.known_user_ids = encoding_data.get("user_ids", [])
            self.logger.info(f"Loaded {len(self.known_encodings)} known encodings.")
        else:
            self.logger.warning(
                "Could not load encodings. Authentication will not work."
            )

        # ユーザーIDと名前の対応辞書を作成
        metadata = self.data_manager.read_metadata()
        self.user_id_to_name_map = {user["user_id"]: user["name"] for user in metadata}
        self.logger.info(
            f"Loaded {len(self.user_id_to_name_map)} user metadata mappings.",
            extra={
                "user_count": len(self.user_id_to_name_map),
            },
        )

    def authenticate_face(self, face_data: dict) -> list:
        """
        単一の顔データを受け取り認証する

        Args:
            face_data (dict): 顔情報を含む辞書

        Returns:
            list: 認証結果を含む辞書のリスト
        """
        face_encoding = face_data["encoding"]
        box_location = face_data["location"]
        name = "Unknown"
        face_distances = face_recognition.face_distance(
            self.known_encodings, face_encoding
        )
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if face_distances[best_match_index] <= self.tolerance:
                user_id = self.known_user_ids[best_match_index]
                name = self.user_id_to_name_map.get(user_id, "Unknown")
        return [{"name": name, "box": box_location}]

    def authenticate_frame(self, frame: np.ndarray) -> List[Dict]:
        """
        単一のフレームを受け取り、顔認証を行い、結果を返す

        Args:
            frame (np.ndarray): カメラから取得したフレーム (BGR形式)

        Returns:
            List[Dict]: 認証結果を含む辞書のリスト
        """
        if not self.known_encodings:
            return []

        # フレームから顔の位置とエンコーディングを検出
        detected_faces = self.face_processor.detect_and_encode_faces(frame)

        name = "Unknown"
        recognized_faces = []
        # 抽出された顔情報を使って、認証
        for face_data in detected_faces:
            face_encoding = face_data["encoding"]
            box_location = face_data["location"]

            # 検出された顔と、学習済みの全ての顔との距離を計算
            face_distances = face_recognition.face_distance(
                self.known_encodings, face_encoding
            )

            if len(face_distances) > 0:
                # 最も距離が短い顔を探す
                best_match_index = np.argmin(face_distances)
                min_distance = face_distances[best_match_index]

                # その最短距離が閾値以下であれば、認証成功と判断
                if min_distance <= self.tolerance:
                    user_id = self.known_user_ids[best_match_index]
                    name = self.user_id_to_name_map.get(user_id, "Unknown")

            recognized_faces.append({"name": name, "box": box_location})
        self.logger.info(
            f"Authenticated {len(recognized_faces)} faces.", extra={"name": name}
        )
        return recognized_faces
