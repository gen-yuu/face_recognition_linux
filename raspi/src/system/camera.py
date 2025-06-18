import os
import random
from typing import Optional

import cv2
import numpy as np

from ....src.utils.logger import setup_logger


class Camera:
    """
    内蔵orUSBカメラを操作するクラス
    """

    def __init__(self, device_id: int = 0, width: int = 1280, height: int = 720):
        """
        カメラを初期化し、指定された解像度に設定する

        Args:
            device_id (int): カメラのデバイスID, 通常は0
            width (int): 要求するフレームの幅
            height (int): 要求するフレームの高さ
        """
        self.logger = setup_logger(__name__)
        self.cap = cv2.VideoCapture(device_id)

        if not self.cap.isOpened():
            self.logger.error(
                "Cannot open camera",
                extra={"device_id": device_id},
            )
            raise IOError

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        self.logger.info(
            "Camera Device initialized.",
            extra={
                "device_id": device_id,
                "requested_resolution": f"{width}x{height}",
                "actual_resolution": f"{int(actual_width)}x{int(actual_height)}",
            },
        )

    def get_frame(self) -> Optional[np.ndarray]:
        """
        カメラから1フレームをキャプチャして返す

        Returns:
            Optional[np.ndarray]: 成功した場合はキャプチャしたフレーム, 失敗した場合はNone
        """
        success, frame = self.cap.read()
        if not success:
            self.logger.warning("Failed to capture frame from camera.")
            return None
        self.logger.debug("Captured frame from camera.")
        return frame

    def release(self):
        """
        カメラリソースを解放する
        """
        if self.cap.isOpened():
            self.cap.release()
            self.logger.info("Camera Device released.")
