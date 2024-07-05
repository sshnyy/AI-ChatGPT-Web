import os
import json
from typing import Dict, List, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.train.mixup import mixup_data, mixup_criterion
from utils.train.cutmix import cutmix_data

from utils.train.metrics import metric
from utils.train.save_graph import *

from .runner import Runner


class TrainingRunner(Runner):
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim,
        scheduler: torch.optim.lr_scheduler,
        loss_func: torch.optim,
        device: torch.device,
        max_grad_norm: float,
    ):
        super().__init__(model, optimizer, scheduler, loss_func, device, max_grad_norm)
        self._valid_predict: List = []
        self._valid_label: List = []

    def forward(self, item):
            inp = item["input"].to(self._device)
            target = item["target"].to(self._device)
            output = self._model.forward(inp.float())
            train_acc, train_precision, train_recall, train_f1, prediction, label = metric(output, target)

            return self._loss_func(output, target), train_acc, train_precision, train_recall, train_f1, prediction, label

    def _mixup_forward(self, item):
        inp = item["input"].to(self._device)
        target = item["target"].to(self._device)

        inp, target_a, target_b, lam = mixup_data(inp, target, self._device)
        output = self._model.forward(inp)

        train_acc, train_f1, prediction, label = metric(output, target)

        return (
            mixup_criterion(self._loss_func, output, target_a, target_b, lam),
            train_acc,
            train_f1,
            prediction,
            label,
        )

    def _cutmix_forward(self, item):
        inp = item["input"].to(self._device)
        target = item["target"].to(self._device)

        inp, target_a, target_b, lam = cutmix_data(inp, target, alpha=1.0)
        output = self._model.forward(inp)

        train_acc, train_f1, prediction, label = metric(output, target)

        loss = lam * self._loss_func(output, target_a) + (1 - lam) * self._loss_func(
            output, target_b
        )
        return (loss, train_acc, train_f1, prediction, label)

    def _backward(self, loss):
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._max_grad_norm)
        self._optimizer.step()
        self._scheduler.step()

    def run(
        self,
        data_loader: DataLoader,
        epoch: int,
        training: bool = True,
        mixup: bool = False,
        mixup_epochs: Optional[int] = None,
        cutmix: bool = False,
        cutmix_epochs: Optional[int] = None,
    ):
        total_loss: float = 0.0
        total_acc: float = 0.0
        total_f1: float = 0.0
        total_precision: float = 0.0
        total_recall: float = 0.0
        total_batch: int = 0
        train_batch: int = 0
        if training:
            if mixup and epoch < mixup_epochs:
                print("=" * 25 + f"Epoch {epoch} Train with Mixup" + "=" * 25)
            elif cutmix and epoch < cutmix_epochs:
                print("=" * 25 + f"Epoch {epoch} Train with Cutmix" + "=" * 25)
            else:
                print("=" * 25 + f"Epoch {epoch} Train" + "=" * 25)

            self._model.train()
            for item in tqdm(data_loader):
                self._optimizer.zero_grad()

                if mixup and epoch < mixup_epochs:
                    loss, acc, f1, _, _ = self._mixup_forward(item)
                elif cutmix and epoch < cutmix_epochs:
                    loss, acc, f1, _, _ = self._cutmix_forward(item)
                else:
                    loss, acc, precision, recall, f1, _, _ = self.forward(item)

                total_loss += loss.item()
                total_acc += acc
                total_precision += precision
                total_recall += recall
                total_f1 += f1
                total_batch += 1
                train_batch += 1
                self._backward(loss)

            return (
                round((total_loss / total_batch), 4),
                round((total_acc / total_batch), 4),
                round((total_precision / total_batch), 4),
                round((total_recall / total_batch), 4),
                round((total_f1 / total_batch), 4),
            )

        else:
            print("=" * 25 + f"Epoch {epoch} Valid" + "=" * 25)
            self._model.eval()
            with torch.no_grad():
                for item in tqdm(data_loader):
                    loss, acc, precision, recall, f1, prediction, label = self.forward(item)

                    self._valid_predict.extend(prediction)
                    self._valid_label.extend(label)

                    total_loss += loss.item()
                    total_acc += acc
                    total_precision += precision
                    total_recall += recall
                    total_f1 += f1
                    total_batch += 1

        return (
                round((total_loss / total_batch), 4),
                round((total_acc / total_batch), 4),
                round((total_precision / total_batch), 4),
                round((total_recall / total_batch), 4),
                round((total_f1 / total_batch), 4),
        )

    def save_model(self, save_path):
        torch.save(
            {
                "model": self._model.state_dict(),
                "optimizer": self._optimizer.state_dict(),
                "scheduler": self._scheduler.state_dict(),
            },
            save_path,
        )

    @staticmethod
    def save_result(
        epoch: int,
        save_folder_path: str,
        train_loss: float,
        valid_loss: float,
        train_acc: float,
        valid_acc: float,
        train_precision: float,
        valid_precision: float,
        train_recall: float,
        valid_recall: float,
        train_f1: float,
        valid_f1: float,
        args,
    ):
        if epoch == 0:  # save only once
            save_json_path = os.path.join(save_folder_path, "model_spec.json")
            with open(save_json_path, "w") as json_file:
                save_json = args.__dict__
                json.dump(save_json, json_file)

        save_result_path = os.path.join(save_folder_path, "best_result.json")
        with open(save_result_path, "w") as json_file:
            save_result_dict: Dict = {
                "best_epoch": epoch + 1,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "train_acc": train_acc,
                "valid_acc": valid_acc,
                "train_precision": train_precision,
                "valid_precision": valid_precision,
                "train_recall": train_recall,
                "valid_recall": valid_recall,
                "train_f1": train_f1,
                "valid_f1": valid_f1,
            }

            json.dump(save_result_dict, json_file)

    def save_graph(
        self,
        save_folder_path: str,
        train_loss: List,
        train_acc: List,
        train_f1: List,
        valid_loss: List,
        valid_acc: List,
        valid_f1: List,
    ):
        save_loss_graph(train_loss, valid_loss, save_folder_path)
        save_acc_graph(train_acc, valid_acc, save_folder_path)
        save_f1_graph(train_f1, valid_f1, save_folder_path)
        save_confusion_matrix(self._valid_predict, self._valid_label, save_folder_path)
