from typing import Optional
from abc import ABCMeta, abstractmethod
import torch
from torch.utils.data import DataLoader
from torch import nn


class Runner(metaclass=ABCMeta):
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim,
        scheduler: torch.optim.lr_scheduler,
        loss_func: torch.nn,
        device: Optional[torch.device],
        max_grad_norm: float,
    ):
        self._model = model
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._loss_func = loss_func
        self._device = device
        self._max_grad_norm = max_grad_norm

    @abstractmethod
    def forward(self, item):
        pass

    @abstractmethod
    def run(
        self,
        data_loader: DataLoader,
        epoch: Optional[int],
        training: bool,
        mixup: bool,
        mixup_epochs: Optional[int],
        cutmix: bool,
        cutmix_epochs: Optional[int],
    ):
        pass
