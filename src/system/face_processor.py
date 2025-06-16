from typing import Dict, List

import cv2
import numpy as np

import face_recognition

from ..utils.logger import setup_logger


class FaceProcessor:
    """
    顔検出、エンコードなどの画像処理を行うクラス
    """

    def __init__(self):
        """
        FaceProcessorのコンストラクタ
        """
        self.logger = setup_logger(__name__)
        self.logger.info("FaceProcessor initialized.")

    def extract_encodings(self, image: np.ndarray) -> List[np.ndarray]:
        """
        単一の画像から顔のエンコーディングを全て抽出する

        Args:
            image (np.ndarray): 顔を検出する対象の画像 (BGR形式)

        Returns:
            List[np.ndarray]: 検出された全ての顔のエンコーディングのリスト
                              顔が検出されなかった場合は空のリストを返す
        """
        # face_recognitionで処理するために、画像をBGRからRGBに変換
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 画像から全ての顔の位置を検出
        face_locations = face_recognition.face_locations(rgb_image, model="hog")
        if not face_locations:
            self.logger.warning("No faces found in the provided image.")
            return []

        # 検出された顔からエンコーディングを抽出
        encodings = face_recognition.face_encodings(rgb_image, face_locations)
        self.logger.info(f"Found {len(encodings)} face(s) in the image.")

        return encodings

    def detect_and_encode_faces(self, frame: np.ndarray) -> List[Dict]:
        """
        フレームから全ての顔を検出し、位置とエンコーディングを抽出する

        Args:
            frame (np.ndarray): カメラから取得したフレーム (BGR形式)

        Returns:
            List[Dict]: 検出された各顔の情報を含む辞書のリスト
                        例: [{"location": (top, right, bottom, left), "encoding": [...]}]
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        locations = face_recognition.face_locations(rgb_frame)
        encodings = face_recognition.face_encodings(rgb_frame, locations)

        results = []
        for loc, enc in zip(locations, encodings):
            results.append({"location": loc, "encoding": enc})

        if results:
            self.logger.info(f"Detected and encoded {len(results)} faces.")

        return results
