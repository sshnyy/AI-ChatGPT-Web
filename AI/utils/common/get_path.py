import os
from typing import Tuple


def get_save_model_path(save_path: str, save_model_name: str) -> Tuple[str, str]:
    save_model_path = os.path.join(save_path, save_model_name)
    print(f"Model Save Path: {save_path}")

    return save_model_path, save_path


def get_save_kfold_model_path(
    save_path: str, save_model_name: str, fold_num: int
) -> Tuple[str, str]:
    save_folder_path = os.path.join(save_path, str(fold_num + 1))

    if not os.path.exists(save_folder_path):
        os.mkdir(save_folder_path)

    save_model_path = os.path.join(save_folder_path, save_model_name)
    print(f"Model Save Path : {save_folder_path}")

    return save_model_path, save_folder_path
