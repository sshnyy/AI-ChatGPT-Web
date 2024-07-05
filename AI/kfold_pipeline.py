
import os
import argparse
from glob import glob
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from inference import predict

from utils.common.fix_seed import seed_everything
from utils.common.constant import LABEL_DICT
from utils.common.translation import str2bool


def kfold_pipe(args, device,img_path):


    # Download Image From Firebase Storage
    image_saved_path = img_path # 이미지 경로 

    infer_results = []

    model_saved_paths = sorted(glob(args.kfold_model_saved_path))
    # print(model_saved_paths)

    for fold_num, model_saved_path in enumerate(model_saved_paths):
        # print("=" * 100)
        # print(f"Model trained fold : {fold_num + 1}")
        # print(f"Saved Model path : {model_saved_path}")
        infer_result = predict(
            image_saved_path, args, model_saved_path, device, voting=True
        )
        infer_results.append(infer_result)

    prediction = (sum([infer_results[i] for i in range(args.num_folds)]))
    prediction = prediction / args.num_folds
    prediction = np.argmax(prediction)
    prediction = int(prediction)
    return prediction




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--kfold_model_saved_path",
        type=str,
        default="./models/saved_model/8/f1/*/f1_best.pt",
    )
    parser.add_argument("--backbone", type=str, default="resnet50d")
    parser.add_argument(
        "--tta", type=str2bool, default="False", help="test time augmentation"
    )
    parser.add_argument("--num_classes", type=str, default=len(LABEL_DICT))
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--save_title", type=str, default="22(10fold_rX)")
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
    #     answer = LABEL_IDX[kfold_pipe(args, device,img_path)]
    #     # print(idx,img_path,":",answer)
    #     preds.append(answer)
    # submit['label'] = preds
    # submit_path = './submit/kfold_submit_' + args.save_title + ".csv"
    # submit.to_csv(submit_path, index=False)
