from typing import List
import pandas as pd


def get_image_paths_from_csv(dataframe: pd.DataFrame) -> List:
    image_paths = dataframe["image"].tolist()
    return image_paths


def get_label_from_csv(dataframe: pd.DataFrame) -> List:
    labels = dataframe["label"].tolist()
    return labels
