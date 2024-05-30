from typing import Iterable, Optional
import torch
from torch import nn

from nano_seq.utils.logger import Logger
from nano_seq.task.base import BaseTask


class Trainer:
    def __init__(
        self,
        train_iter: Iterable,
        eval_iter: Iterable,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
        criterion: nn.Module,
        task: BaseTask,
        model: torch.nn.Module,
        logger: Logger,
    ):
        self.train_iter = train_iter
        self.eval_iter = eval_iter
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.criterion = criterion
        self.task = task
        self.model = model
        self.logger = logger

    def train_epoch(self):
        for batch_idx, sample in enumerate(iter(self.train_iter)):
            _, label = sample
            net_input = self.task.get_net_input(sample)

            logs = self.task.train_step(net_input, label, self.model, self.optimizer, self.criterion)
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            self.logger.write_train(batch_idx, **logs)

    def eval_epoch(self):
        for batch_idx, sample in enumerate(iter(self.eval_iter)):
            _, label = sample
            net_input = self.task.get_net_input(sample)

            with torch.no_grad():
                logs = self.task.eval_step(net_input, label, self.model, self.criterion)

            self.logger.write_eval(batch_idx, **logs)

    def start_train(self, num_epochs: int):
        for i in range(num_epochs):
            self.logger.epoch = i + 1

            self.train_epoch()
            self.eval_epoch()
