import os
import time
import traceback

import cv2
import numpy as np
from flask import Flask, Response, render_template, request
from PIL import Image, ImageDraw, ImageFont

from src.system.camera import Camera, SimulatedCamera
from src.system.data_manager import DataManager
from src.system.face_processor import FaceProcessor
from src.system.services import AuthenticationService
from src.utils.logger import setup_logger

app = Flask(__name__)
logger = setup_logger(__name__)

if __name__ == "__main__":
    try:
        logger.info("Starting Flask server...")
        app.run(host="0.0.0.0", port=8080, debug=False)
    except Exception as e:
        logger.error(
            "Flask server failed to start",
            extra={
                "error": str(e),
                "traceback": traceback.format_exc()
            },
        )
    finally:
        camera.release()
        logger.info("Flask server shutting down.")
