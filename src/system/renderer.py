import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..utils.logger import setup_logger


class FrameRenderer:
    """
    フレームへのUI要素の描画をするクラス
    """

    def __init__(self, font_path: str):
        """
        FrameRendererを初期化する

        Args:
            font_path (str): フォントファイルのパス
        """
        self.logger = setup_logger(__name__)
        self.font_path = font_path
        self.font = None
        if os.path.exists(font_path):
            try:
                self.font = ImageFont.truetype(font_path, 32)
            except Exception:
                self.logger.warning(f"Could not load font file: {font_path}")
        else:
            self.logger.warning(f"Font file not found: {font_path}")

    def _draw_text(self, image, text, position, color):
        """
        ヘルパー関数: Pillowを使って日本語を描画する。
        """
        if not self.font:
            # フォールバックとしてOpenCVのデフォルトフォントで描画
            cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            return image

        img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        draw.text(position, text, font=self.font, fill=color)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def draw_guide_box(self, frame, rect, color):
        """
        指定された色でガイド枠を描画する
        """
        x, y, w, h = rect
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
        return frame

    def draw_face_box(self, frame, location, message, color):
        """
        顔の周りに枠とフィードバックメッセージを描画する
        """
        top, right, bottom, left = location
        # 枠を描画
        cv2.rectangle(frame, (left, top), (right, bottom), color, 3)
        # メッセージを描画
        frame = self._draw_text(frame, message, (left, top - 40), color)
        return frame
