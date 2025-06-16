# /run_register.py

import numpy as np

# srcパッケージから必要なクラスをインポート
from src.system.data_manager import DataManager
from src.system.services import RegistrationService


def main():
    # 顔登録のアプリケーションフローを実行します。

    print("Starting User Registration")

    data_manager = DataManager()

    registration_service = RegistrationService(data_manager=data_manager)

    # ダミーデータを使用
    # 実際のアプリケーションでは、カメラから取得した画像やGUIから入力された名前を使用
    user_name_to_register = "Yuki Gennai"

    # 真っ黒なダミー画像を3枚作成
    dummy_image = np.zeros((200, 200, 3), dtype=np.uint8)
    images_to_register = [dummy_image, dummy_image, dummy_image]

    print(f"Attempting to register user: {user_name_to_register}")

    try:
        user_id = registration_service.register_new_user(
            name=user_name_to_register, images=images_to_register
        )
        print("\nRegistration Successful")
        print(f"  User Name: {user_name_to_register}")
        print(f"  User ID:   {user_id}")
    except ValueError as e:
        print(f"Registration Failed: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
