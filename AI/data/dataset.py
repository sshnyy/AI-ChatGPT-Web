from typing import List, Optional
import cv2
import torch
import numpy as np
from albumentations import (
    Compose,
    RandomRotate90,
    HorizontalFlip,
    VerticalFlip,
    Resize,
    Normalize,
    MotionBlur,
    CenterCrop,
    CoarseDropout,
)
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from models.unet_model import SegmentationModel



class LandmarkDataset(Dataset):
    def __init__(
        self,
        img_paths: List,
        labels: Optional[List],
        img_size: int,
        is_training: bool = True,
        use_augmentation: bool = True,
        use_unet: bool = False,
        unet_model_path: Optional[str] = None,
    ):
        super(LandmarkDataset, self).__init__()
        self.img_paths = img_paths
        self.labels = labels
        self.img_size = img_size
        self.is_training = is_training
        self.use_augmentation = use_augmentation
        self.use_unet = use_unet
        if use_unet:
            # Load U-Net model
            arch = "Unet"
            encoder_name = "resnet34"
            in_channels = 3
            out_classes = 1
            self.unet_model = SegmentationModel(arch, encoder_name, in_channels=in_channels, out_classes=out_classes)
            self.unet_model = SegmentationModel.load_from_checkpoint(unet_model_path, arch=arch, encoder_name=encoder_name, in_channels=in_channels, out_classes=out_classes)
            self.unet_model.eval()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, item):
        img = cv2.imread(self.img_paths[item])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.use_unet:
            # Extract lesion using U-Net
            img = self._process_image_with_lesion_extraction(img)
        augmented = self._get_augmentations(self.use_augmentation, self.img_size)(image=img)
        img = augmented["image"]

        if self.is_training:
            label = self.labels[item]
            return {"input": img, "target": torch.tensor(label, dtype=torch.long)}
        return {"input": img}

    def _process_image_with_lesion_extraction(self, image: np.ndarray) -> np.ndarray:
        transform = Compose(
            [
                Resize(self.img_size, self.img_size),  # Ensure image is right size for U-Net
                ToTensorV2()
            ]
        )
        image_tensor = transform(image=image)["image"].unsqueeze(0)
        with torch.no_grad():
            predictions = self.unet_model(image_tensor)
        predicted_masks = predictions.sigmoid().cpu()
        lesion_part = np.transpose(image_tensor[0].numpy(), (1, 2, 0)) * np.repeat(predicted_masks[0].squeeze().numpy()[:, :, np.newaxis], 3, axis=2)
        return (lesion_part * 255).astype(np.uint8)  # Convert back to 8-bit


    @staticmethod
    def _get_augmentations(use_augmentation: bool, img_size: int) -> Compose:
        if use_augmentation:
            return Compose(
                [
                    # RandomRotate90(p=0.5),
                    Resize(img_size, img_size),
                    HorizontalFlip(p=0.5),
                    VerticalFlip(p=0.5),
                    # CoarseDropout(always_apply=False, p=0.5, max_holes=8, max_height=8, max_width=8, min_holes=8, min_height=8, min_width=8),
                    # MotionBlur(p=0.5),
                    # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                    
                ]
            )
        else:
            return Compose(
                [
                    Resize(img_size, img_size),
                    # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ]
            )
