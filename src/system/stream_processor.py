import time

import cv2


class StreamProcessor:
    """
    カメラからの映像ストリームの処理フロー全体を管理するクラス。
    """

    def __init__(
        self, camera, face_processor, auth_service, renderer, app_state, config
    ):
        """
        StreamProcessorを初期化する

        Args:
            camera (Camera): カメラインスタンス
            face_processor (FaceProcessor): 顔認識プロセッサインスタンス
            auth_service (AuthenticationService): 認証サービスインスタンス
            renderer (Renderer): 描画プロセッサインスタンス
            app_state (AppState): アプリ状態インスタンス
        """
        self.camera = camera
        self.face_processor = face_processor
        self.auth_service = auth_service
        self.renderer = renderer
        self.app_state = app_state
        self.config = config

    def generate(self):
        """
        ビデオフレームを生成するジェネレータ関数
        """
        while True:
            if (self.app_state.mode == "REGISTRATION_FROZEN") and (
                self.app_state.captured_frame is not None
            ):
                frame = self.app_state.captured_frame.copy()
            else:
                frame = self.camera.get_frame()
                if frame is None:
                    time.sleep(0.1)
                    continue

            # 現在のモードに応じてフレームを処理・描画
            if self.app_state.mode == "AUTHENTICATING":
                frame = self._handle_authentication_frame(frame)
            elif self.app_state.mode in [
                "REGISTRATION_SEARCHING",
                "REGISTRATION_FROZEN",
            ]:
                frame = self._handle_registration_frame(frame)

            # フレームをJPEGにエンコードしてyield
            _, buffer = cv2.imencode(".jpg", frame)
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )

    def _handle_authentication_frame(self, frame) -> cv2.Mat:
        """
        認証モード時のフレーム処理

        Args:
            frame (numpy.ndarray): 入力フレーム

        Returns:
            numpy.ndarray: 処理済みのフレーム
        """
        detected_faces = self.face_processor.detect_and_encode_faces(frame)

        # デフォルトのガイド枠を描画
        frame = self.renderer.draw_guide_box(
            frame, self.config["GUIDE_BOX_RECT"], (128, 128, 128)
        )

        if not detected_faces:
            return frame

        largest_face = max(
            detected_faces,
            key=lambda f: self.config["get_face_properties"](f["location"])[1],
        )
        face_box_coords, face_area = self.config["get_face_properties"](
            largest_face["location"]
        )

        distance, size_ratio = self.config["calculate_face_metrics"](
            face_box_coords, face_area
        )

        if (
            distance <= self.config["POSITION_THRESHOLD"]
            and size_ratio >= self.config["SIZE_THRESHOLD"]
        ):
            # 認証処理へ (ここでは描画のみ)
            auth_result = self.auth_service.authenticate_face(largest_face)
            name = auth_result[0]["name"]
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            frame = self.renderer.draw_face_box(
                frame, largest_face["location"], f"結果: {name}", color
            )
        else:
            # 条件不足のフィードバック
            color = (0, 255, 255)
            message = (
                "顔を枠の中央に"
                if distance > self.config["POSITION_THRESHOLD"]
                else "近づいてください"
            )
            frame = self.renderer.draw_face_box(
                frame, largest_face["location"], message, color
            )

        return frame

    def _handle_registration_frame(self, frame) -> cv2.Mat:
        """
        登録モード時のフレーム処理

        Args:
            frame (numpy.ndarray): 入力フレーム

        Returns:
            numpy.ndarray: 処理済みのフレーム
        """
        detected_faces = self.face_processor.detect_and_encode_faces(frame)
        if not detected_faces:
            self.app_state.captured_frame = None
            return frame

        largest_face = max(
            detected_faces,
            key=lambda f: self.config["get_face_properties"](f["location"])[1],
        )
        face_box, face_area = self.config["get_face_properties"](
            largest_face["location"]
        )

        distance, size_ratio = self.config["calculate_face_metrics"](
            face_box, face_area
        )

        if (
            self.app_state.mode == "REGISTRATION_SEARCHING"
            and distance <= self.config["POSITION_THRESHOLD"]
            and size_ratio >= self.config["SIZE_THRESHOLD"]
        ):
            self.app_state.mode = "REGISTRATION_FROZEN"
            self.app_state.captured_frame = frame.copy()

        if self.app_state.mode == "REGISTRATION_FROZEN":
            color = (0, 255, 0)
            message = "この顔で登録します"
        else:  # REGISTRATION_SEARCHING
            color = (0, 255, 255)
            message = (
                "顔を枠の中央に"
                if distance > self.config["POSITION_THRESHOLD"]
                else "近づいてください"
            )

        return self.renderer.draw_face_box(
            frame, largest_face["location"], message, color
        )
