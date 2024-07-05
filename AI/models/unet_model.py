
import lightning as pl

import numpy as np
import pandas as pd
import seaborn as sns

import os
import torch
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp
from PIL import Image
from pprint import pprint
from torch.utils.data import DataLoader


class SegmentationModel(pl.LightningModule):

    def __init__(self, arch, encoder_name, in_channels, out_classes, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, **kwargs
        )

        # preprocessing parameteres for image
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # dice loss as loss function for binary image segmentation
        self.loss_fn = smp.losses.DiceLoss(mode='binary', from_logits=True)

    def forward(self, image):
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def _step(self, batch, stage):
        
        image = batch[0]        # Shape of the image : (batch_size, num_channels, height, width)
        mask = batch[1]

        logits_mask = self.forward(image)
        loss = self.loss_fn(logits_mask, mask)

        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")
        return {"loss": loss, "tp": tp, "fp": fp, "fn": fn, "tn": tn,}

    def _epoch_end(self, outputs, stage):

        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # micro-imagewise - first calculate IoU score for each image and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            }
        
        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        output = self._step(batch, "train")
        if not hasattr(self, "train_outputs"):
            self.train_outputs = []
        self.train_outputs.append(output)
        return output

    def on_train_epoch_end(self):
        outputs = self.train_outputs
        self._epoch_end(outputs, "train")
        self.train_outputs = []

    def validation_step(self, batch, batch_idx):
        output = self._step(batch, "valid")
        if not hasattr(self, "valid_outputs"):
            self.valid_outputs = []
        self.valid_outputs.append(output)
        return output

    def on_validation_epoch_end(self):
        outputs = self.valid_outputs
        self._epoch_end(outputs, "valid")
        self.valid_outputs = []

    def test_step(self, batch, batch_idx):
        output = self._step(batch, "test")
        if not hasattr(self, "test_outputs"):
            self.test_outputs = []
        self.test_outputs.append(output)
        return output

    def test_epoch_end(self):
        outputs = self.test_outputs
        self._epoch_end(outputs, "test")
        self.test_outputs = []
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)


def process_image_mask(image_path: str) -> (torch.Tensor):
    """
    주어진 이미지와 마스크 경로를 사용하여 이미지 및 마스크를 처리합니다.
    
    Args:
    - image_path (str): 처리할 이미지의 경로.
    - mask_path (str): 처리할 마스크의 경로.

    Returns:
    - tuple: 처리된 이미지와 마스크의 Tensor.
    """
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # 이미지 처리
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0)  # 추가적인 배치 차원을 추가합니다.

    with torch.no_grad():
        predictions = model(image_tensor.to(model.device))

    # 예측 결과를 CPU로 가져오고 sigmoid를 적용하여 확률을 얻기
    predicted_masks = predictions.sigmoid().cpu()
    # 병변 부분만 추출
    lesion_part = np.transpose(image_tensor[0].numpy(), (1, 2, 0)) * np.repeat(predicted_masks[0].squeeze().numpy()[:, :, np.newaxis], 3, axis=2)
    return torch.from_numpy(np.transpose(lesion_part, (2, 0, 1)))  # CxHxW 포맷으로 변환

def extract_and_save_lesion(image_path, save_path, model):
    """
    주어진 이미지 경로의 이미지에서 병변 부분을 추출하고 지정된 경로에 이미지를 저장합니다.
    """
    image_tensor = process_image_mask(image_path)
    with torch.no_grad():
        predictions = model(image_tensor.unsqueeze(0).to(model.device))

    # 예측 결과를 CPU로 가져오고 sigmoid를 적용하여 확률을 얻기
    predicted_masks = predictions.sigmoid().cpu()
    
    # 병변 부분만 추출
    lesion_part = np.transpose(image_tensor.numpy(), (1, 2, 0)) * np.repeat(predicted_masks[0].squeeze().numpy()[:, :, np.newaxis], 3, axis=2)
    
    # 이미지 저장
    plt.imshow(lesion_part)
    plt.axis('off')  # 축 제거
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)  # 이미지 저장
    plt.close()  # 그림 닫기 (화면에 출력하지 않음)

if __name__ == "__main__":

    # 모델을 초기화할 때 사용된 인자
    arch = "Unet"
    encoder_name = "resnet34"
    in_channels = 3
    out_classes = 1
    model = SegmentationModel(arch, encoder_name , in_channels=in_channels, out_classes=out_classes)
    # 모델 불러오기
    model = SegmentationModel.load_from_checkpoint("/home/suyeon/code/capstone3/DL/models/unet/model/model_checkpoint.ckpt", arch=arch, encoder_name=encoder_name, in_channels=in_channels, out_classes=out_classes)
    model.eval()
    n_cpu = os.cpu_count()
        # df에서 이미지 경로 불러오기
    df = pd.read_csv("data/dataset/train.csv")
    
    # 저장할 디렉토리 지정 (만약 없으면 생성)
    save_directory = "/home/suyeon/code/capstone3/DL/image/lesion_parts/"
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # df['image']에 저장된 모든 이미지 경로의 이미지에서 병변 부분을 추출하고 저장
    for image_path in df['image']:
        # 이미지 파일 이름 추출
        file_name = os.path.basename(image_path)
        save_path = os.path.join(save_directory, file_name)
        extract_and_save_lesion(image_path, save_path, model)
        
    # image_path = "/home/suyeon/code/capstone3/DL/image/ISIC_0029307.jpg"

    # # 단일 이미지를 모델에 전달합니다.
    # image_tensor = process_image_mask(image_path).unsqueeze(0)  # 추가적인 배치 차원을 추가합니다.
    # with torch.no_grad():
    #     predictions = model(image_tensor.to(model.device))

    # # 예측 결과를 CPU로 가져오고 sigmoid를 적용하여 확률을 얻기
    # predicted_masks = predictions.sigmoid().cpu()
    # # 병변 부분만 추출
    # lesion_part = np.transpose(image_tensor[0].numpy(), (1, 2, 0)) * np.repeat(predicted_masks[0].squeeze().numpy()[:, :, np.newaxis], 3, axis=2)

    # 이미지 저장
    # plt.imshow(lesion_part)
    # plt.axis('off')  # 축 제거
    # plt.savefig("/home/suyeon/code/capstone3/DL/image/lesion_part_extracted1.png", bbox_inches='tight', pad_inches=0)  # 이미지 저장
    # plt.close()  # 그림 닫기 (화면에 출력하지 않음)

