import os
import random

import cv2
import numpy as np


def create_test_frames(
    output_dir="test_webcam_frames",
    source_dir="source_faces",
    num_frames=20,
    frame_width=1280,
    frame_height=720,
    background_color=(128, 128, 128),  # Dark gray background
):
    """
    ソースディレクトリの顔画像を使い、指定した解像度のテスト用フレームを生成する
    """
    if not os.path.exists(source_dir) or not os.listdir(source_dir):
        print(f"Error: Source directory '{source_dir}' not found or is empty.")
        print("Please create it and add source face images.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    source_face_paths = [
        os.path.join(source_dir, f)
        for f in os.listdir(source_dir)
        if f.lower().endswith((".pgm", ".jpg", ".png"))
    ]
    if not source_face_paths:
        print(f"Error: No valid images found in '{source_dir}'.")
        return

    for i in range(num_frames):
        background = np.full(
            (frame_height, frame_width, 3), background_color, dtype=np.uint8
        )

        face_path = random.choice(source_face_paths)
        face_image = cv2.imread(face_path)
        if face_image is None:
            print(f"Warning: Could not read {face_path}. Skipping.")
            continue

        fh, fw, _ = face_image.shape

        scale = random.uniform(1.5, 4.0)
        new_w, new_h = int(fw * scale), int(fh * scale)
        resized_face = cv2.resize(face_image, (new_w, new_h))

        max_x = frame_width - new_w
        max_y = frame_height - new_h

        if max_x <= 0 or max_y <= 0:
            print("Warning: Resized face is too large for the frame. Skipping.")
            continue

        start_x = random.randint(0, max_x)
        start_y = random.randint(0, max_y)

        background[start_y : start_y + new_h, start_x : start_x + new_w] = resized_face

        output_path = os.path.join(output_dir, f"test_frame_{i+1:02d}.jpg")
        cv2.imwrite(output_path, background)
        print(f"Generated: {output_path}")

    print("\n--- Test Frame Generation Finished ---")


if __name__ == "__main__":
    create_test_frames()
