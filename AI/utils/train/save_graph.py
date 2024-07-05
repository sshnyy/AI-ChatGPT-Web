import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from ..common.constant import LABEL_DICT


def save_confusion_matrix(labels, predictions, save_path):
    try:
        matrix = confusion_matrix(labels, predictions)
        data_frame = pd.DataFrame(matrix, columns=range(len(LABEL_DICT.keys())))

        plt.figure(figsize=(10, 10))
        sns.heatmap(
            data_frame,
            cmap="Blues",
            annot_kws={"size": 8},
            annot=True,
            linecolor="grey",
            linewidths=0.3,
        )
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.xlabel("Prediction")
        plt.ylabel("Answer")

        plt.savefig(os.path.join(save_path, "confusion_matrix.png"))
    except:
        plt.figure(figsize=(10, 10))
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.xlabel("Prediction")
        plt.ylabel("Answer")
        plt.savefig(os.path.join(save_path, "confusion_matrix.png"))



def save_loss_graph(train_loss, valid_loss, save_path):
    plt.figure(figsize=(10, 7))
    plt.grid()
    plt.plot(train_loss, label="train_loss")
    plt.plot(valid_loss, label="valid_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Loss", fontsize=15)
    plt.legend()
    plt.savefig(os.path.join(save_path, "Loss.png"))


def save_acc_graph(train_acc, valid_acc, save_path):
    plt.figure(figsize=(10, 7))
    plt.grid()
    plt.plot(train_acc, label="train_loss")
    plt.plot(valid_acc, label="valid_loss")
    plt.xlabel("epoch")
    plt.ylabel("Acc")
    plt.title("Accuracy", fontsize=15)
    plt.legend()
    plt.savefig(os.path.join(save_path, "Acc.png"))


def save_f1_graph(train_f1, valid_f1, save_path):
    plt.figure(figsize=(10, 7))
    plt.grid()
    plt.plot(train_f1, label="train_f1")
    plt.plot(valid_f1, label="valid_f1")
    plt.xlabel("epoch")
    plt.ylabel("F1")
    plt.title("F1 Score", fontsize=15)
    plt.legend()
    plt.savefig(os.path.join(save_path, "F1.png"))
