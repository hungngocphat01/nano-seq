from abc import ABC, abstractmethod

import torch

from nano_seq.data.dataset import NetInput


class BaseTask(ABC):
    @abstractmethod
    def prepare(self):
        """
        Initialize data loader and model

        Returns
        -------
        tuple[train_iter, valid_iter, model]
        """

    @abstractmethod
    def get_net_input(self, sample) -> NetInput:
        """
        Construct neural network input from data returned by collator

        Args
        ----
        sample: any
            data sample returned from collator of the task

        Return
        ------
        NetInput
            dataclass containing neural network input
        """

    @abstractmethod
    def train_step(
        self,
        net_input: NetInput,
        label: torch.Tensor,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
    ) -> dict:
        """
        Function to be called from trainer on one training batch.
        Pass the input to the NN, compute the loss and return logging metrics.
        """

    @abstractmethod
    def eval_step(
        self,
        net_input: NetInput,
        label: torch.Tensor,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
    ) -> dict:
        """
        Function to be called from trainer on one eval batch.
        Pass the input to the NN, compute the loss and return logging metrics.
        """
