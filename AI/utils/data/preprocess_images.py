import cv2
import os
from typing import Any
from glob import glob
from tqdm import tqdm


def resize_single_image(
    image_path: str,
    width: int,
    height: int,
    save_path: str,
    interpolation: Any,
    equalization: bool = False,
    count: int = 0,
):

    """
    이미지를 resize 해서 save path에 저장하는 함수입니다.
    equalization 옵션을 True로 줄 경우 Adaptive Histogram Equalization이 적용됩니다.
    interpolation: {0: INTER_NEAREST, 1: INTER_LINEAR, 2: INTER_CUBIC, 3: INTER_LANCZOS4, 4: INTER_AREA} / https://deep-learning-study.tistory.com/185
    """
    img = cv2.imread(image_path)

    if equalization:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, dsize=(width, height), interpolation=interpolation)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    save_image_path = os.path.join(save_path, f"{count}.jpg")
    cv2.imwrite(save_image_path, img)


if __name__ == "__main__":
    for i in tqdm(range(19)):
        os.mkdir(f"./data/dataset/{i}_resized")

        cnt = 0
        img_paths = sorted(glob(f"./data/dataset/{i}/*"))

        for img_path in tqdm(img_paths):
            img = cv2.imread(img_path)
            try:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except cv2.error:
                print(img_path)
                continue
            h, w, _ = img.shape
            if h >= 320 and w >= 320:
                resize_single_image(
                    image_path=img_path,
                    width=320,
                    height=320,
                    save_path=f"./data/dataset/{i}_resized",
                    interpolation=cv2.INTER_LANCZOS4,
                    equalization=False,
                    count=cnt,
                )
                cnt += 1
