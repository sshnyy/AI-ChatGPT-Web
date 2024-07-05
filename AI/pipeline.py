import os
import argparse
import numpy as np
import torch
import pandas as pd

from tqdm import tqdm
from inference import predict

from utils.common.fix_seed import seed_everything
from utils.common.constant import LABEL_DICT
from utils.common.translation import str2bool


def pipe(args, device,img_path):

    # Download Image From Firebase Storage
    image_saved_path = img_path # 이미지 경로 

    # Predict Value by downloaded imagl
    prediction = predict(
        image_saved_path, args, args.model_saved_path, device, voting=False
    )
    
    return prediction


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--model_saved_path",
        type=str,
        default="./models/saved_model/1/f1/f1_best.pt",
    )
    parser.add_argument("--backbone", type=str, default="resnet50d")
    parser.add_argument(
        "--tta", type=str2bool, default="False", help="test time augmentation"
    )
    parser.add_argument("--num_classes", type=str, default=len(LABEL_DICT))
    parser.add_argument("--use-unet", type=str2bool ,default="False") # Unet 을 사용하여 학습 ( 속도 느림 )
    parser.add_argument("--predict_mask", type=str2bool ,default="False") # 예측시 입력받은 이미지에 대한 마스크 생성 여부
    parser.add_argument("--unet-checkpoint", type=str, default="/home/suyeon/code/capstone3/DL/models/unet/model/model_checkpoint.ckpt")
    args = parser.parse_args()
    # ===========================================================================
    seed_everything(seed=args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.device}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ===========================================================================

    LABEL_IDX = {i:v for v,i in LABEL_DICT.items()}
    preds = []

    # test = pd.read_csv('./test.csv')
    # submit = pd.read_csv('./sample_submission.csv')
    # print(len(test['img_path']))

    # for idx in tqdm(range(len(test['img_path']))):
    #     img_path = test['img_path'][idx]
    #     answer = LABEL_IDX[pipe(args, device,img_path)]
    #     # print(idx,img_path,":",answer)
    #     preds.append(answer)
    # submit['label'] = preds
    # submit.to_csv('./submit/submit_cutmix_8.csv', index=False)

