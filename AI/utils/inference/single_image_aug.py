import cv2
from albumentations import (
    Compose,
    Resize,
    Normalize,
)
from albumentations.pytorch import ToTensorV2


def augment_single_image(img_path: str, img_size: int):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    augmentation = Compose(
        [
            Resize(img_size, img_size),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )

    augmented = augmentation(image=img)
    img = augmented["image"]

    return img
