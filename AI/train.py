import os
import argparse
import shutil
from glob import glob
import pandas as pd

from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from torchinfo import summary

from utils.common.constant import (
    LABEL_DICT,
    SAVE_LOSS_BASED_MODEL_NAME,
    SAVE_ACC_BASED_MODEL_NAME,
    SAVE_F1_BASED_MODEL_NAME,
)
from utils.common.fix_seed import seed_everything
from utils.common.translation import str2bool
from utils.common.get_path import get_save_model_path
from utils.common.load_data_from_dataframe import (
    get_image_paths_from_csv,
    get_label_from_csv,
)

from models.runners.training_runner import TrainingRunner

from data.data_loader import get_data_loader
from data.dataset import LandmarkDataset

from models.model import SkinlesionModel


def run(train_img_paths, train_labels, valid_img_paths, valid_labels, args, device):
    train_dataset = LandmarkDataset(
        train_img_paths,
        train_labels,
        args.img_size,
        is_training=True,
        use_augmentation=True,
        use_unet=args.use_unet,
        unet_model_path=args.unet_checkpoint,
    )
    valid_dataset = LandmarkDataset(
        valid_img_paths,
        valid_labels,
        args.img_size,
        is_training=True,
        use_augmentation=False,
        use_unet=args.use_unet,
        unet_model_path=args.unet_checkpoint,
    )
    # ==========================================================================
    train_dataloader, valid_dataloader = get_data_loader(
        args, train_dataset, valid_dataset, test_dataset=None, training=True
    )
    # ==========================================================================
    model = SkinlesionModel(**args.__dict__).to(device)
    summary(model, (32, 3, 224, 224))
    # ==========================================================================
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=args.betas,
        eps=args.eps,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=args.eta_min)
    loss_func = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    # ==========================================================================
    save_loss_based_model_path, save_loss_based_folder_path = get_save_model_path(
        os.path.join(args.save_path, "loss"), SAVE_LOSS_BASED_MODEL_NAME
    )
    save_f1_based_model_path, save_f1_based_folder_path = get_save_model_path(
        os.path.join(args.save_path, "f1"), SAVE_F1_BASED_MODEL_NAME
    )
    save_acc_based_model_path, save_acc_based_folder_path = get_save_model_path(
        os.path.join(args.save_path, "acc"), SAVE_ACC_BASED_MODEL_NAME
    )
    # ==========================================================================
    train_runner = TrainingRunner(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_func=loss_func,
        device=device,
        max_grad_norm=args.max_grad_norm,
    )
    # ==========================================================================
    prev_valid_loss: float = 1e4
    prev_valid_acc: float = 1e-4
    prev_valid_f1: float = 1e-4
    t_loss, t_acc, t_f1 = [], [], []
    v_loss, v_acc, v_f1 = [], [], []

    for epoch in range(args.epochs):
        print(f"Epoch : {epoch + 1}")
        train_loss, train_acc, train_precision, train_recall, train_f1_score = train_runner.run(
            train_dataloader,
            epoch + 1,
            mixup=args.mixup,
            mixup_epochs=args.mixup_epochs,
            cutmix=args.cutmix,
            cutmix_epochs=args.cutmix_epochs,
        )
        t_loss.append(train_loss)
        t_acc.append(train_acc)
        t_f1.append(train_f1_score)
        print(
            f"Train loss : {train_loss}, Train acc : {train_acc}, F1-score : {train_f1_score}"
        )

        valid_loss, valid_acc, valid_precision, valid_recall, valid_f1_score = train_runner.run(
            valid_dataloader,
            epoch + 1,
            training=False,
            mixup=False,
            mixup_epochs=None,
            cutmix=False,
            cutmix_epochs=None,
        )
        v_loss.append(valid_loss)
        v_acc.append(valid_acc)
        v_f1.append(valid_f1_score)
        print(
            f"Valid loss : {valid_loss}, Valid acc : {valid_acc}, F1-score : {valid_f1_score}"
        )

        train_runner.save_graph(
            save_loss_based_folder_path,
            t_loss,
            t_acc,
            t_f1,
            v_loss,
            v_acc,
            v_f1,
        )

        if prev_valid_loss > valid_loss:
            prev_valid_loss = valid_loss

            train_runner.save_model(save_path=save_loss_based_model_path)
            train_runner.save_result(
                epoch,
                save_loss_based_folder_path,
                train_loss,
                valid_loss,
                train_acc,
                valid_acc,
                train_precision,
                valid_precision,
                train_recall,
                valid_recall,
                train_f1_score,
                valid_f1_score,
                args,
            )
            print("Save Best Loss Model and Graph")

        if prev_valid_acc < valid_acc:
            prev_valid_acc = valid_acc
            train_runner.save_model(save_path=save_acc_based_model_path)
            train_runner.save_result(
                epoch,
                save_acc_based_folder_path,
                train_loss,
                valid_loss,
                train_acc,
                valid_acc,
                train_precision,
                valid_precision,
                train_recall,
                valid_recall,
                train_f1_score,
                valid_f1_score,
                args,
            )
            print("Save Best ACC Model and Graph")

        if prev_valid_f1 < valid_f1_score:
            prev_valid_f1 = valid_f1_score
            train_runner.save_model(save_path=save_f1_based_model_path)
            train_runner.save_result(
                epoch,
                save_f1_based_folder_path,
                train_loss,
                valid_loss,
                train_acc,
                valid_acc,
                train_precision,
                valid_precision,
                train_recall,
                valid_recall,
                train_f1_score,
                valid_f1_score,
                args,
            )
            print("Save Best F1 Model and Graph")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--pin_memory", type=str2bool, default="True")
    parser.add_argument("--persistent_workers", type=str2bool, default="True")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--betas", type=tuple, default=(0.9, 0.999))
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--T_max", type=int, default=10)
    parser.add_argument("--eta_min", type=float, default=1e-6)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--mixup", type=str2bool, default="False")
    parser.add_argument("--mixup_epochs", type=int, nargs="?")
    parser.add_argument("--cutmix", type=str2bool, default="False")
    parser.add_argument("--cutmix_epochs", type=int, nargs="?")
    parser.add_argument("--save_path", type=str, default="./models/saved_model/")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--backbone", type=str, default="resnet50d")
    parser.add_argument("--num_classes", type=str, default=len(LABEL_DICT))
    parser.add_argument(
        "--train_csv_path", type=str, default="./data/dataset/train.csv"
    )
    parser.add_argument("--use-unet", type=str2bool ,default="False") # Unet 을 사용하여 학습 ( 속도 느림 )
    parser.add_argument("--predict_mask", type=str2bool ,default="False") # 예측시 입력받은 이미지에 대한 마스크 생성 여부
    parser.add_argument("--unet-checkpoint", type=str, default="/home/suyeon/code/capstone3/DL/models/unet/model/model_checkpoint.ckpt")
    

    args = parser.parse_args()

    # ===========================================================================
    seed_everything(seed=args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.device}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ===========================================================================
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    num_folder = len(glob(args.save_path + "*"))
    args.save_path = os.path.join(args.save_path, str(num_folder + 1))
    print(f"Save Path: {args.save_path}")

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    shutil.copyfile("data/dataset.py", os.path.join(args.save_path, "dataset.py"))

    os.mkdir(os.path.join(args.save_path, "loss"))
    os.mkdir(os.path.join(args.save_path, "acc"))
    os.mkdir(os.path.join(args.save_path, "f1"))
    # ===========================================================================
    train_dataframe = pd.read_csv(args.train_csv_path)

    img_paths = get_image_paths_from_csv(train_dataframe)
    labels = get_label_from_csv(train_dataframe)
    # ===========================================================================
    train_img_paths, valid_img_paths, train_labels, valid_labels = train_test_split(
        img_paths,
        labels,
        test_size=0.2,
        shuffle=True,
        stratify=labels,
        random_state=args.seed,
    )
    print(train_img_paths[:5], valid_img_paths[:5])
    print(train_labels[:5], valid_labels[:5])
    run(train_img_paths, train_labels, valid_img_paths, valid_labels, args, device)
