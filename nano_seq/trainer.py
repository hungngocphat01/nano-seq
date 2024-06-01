import math
import os
from typing import Optional
import torch
from torch import nn

from nano_seq.data.collator import BaseCollator
from nano_seq.utils.logger import Logger
from nano_seq.task.base import BaseTask
from nano_seq.model.base import NetInput


class Trainer:
    def __init__(
        self,
        train_iter: BaseCollator,
        eval_iter: BaseCollator,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
        criterion: nn.Module,
        task: BaseTask,
        model: torch.nn.Module,
        logger: Logger,
        checkpoint_path: Optional[str] = None,
    ):
        self.train_iter = train_iter
        self.eval_iter = eval_iter
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.criterion = criterion
        self.task = task
        self.model = model
        self.logger = logger
        self.chkpt_path = checkpoint_path
        self._current_epoch = 0

    def train_epoch(self):
        for batch_idx, sample in enumerate(iter(self.train_iter)):
            _, label = sample
            net_input = self.task.get_net_input(sample)

            logs = self.task.train_step(net_input, label, self.model, self.optimizer, self.criterion)
            logs.update(self.training_metrics(net_input))

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

    def start_train(self, target_epoch: int, save_chkpt_every: Optional[int] = None):
        """
        Start the training process for an additional `num_epochs`.

        This function has a side effect: it will set the logger's stdout handler's `_step_epoch`
        property to show the progress bar while training, if the stdout handler is enabled.
        """
        if "stdout" in self.logger.handlers:
            batches_per_epoch = math.ceil(len(self.train_iter) / self.train_iter.bsz)
            setattr(self.logger.handlers["stdout"], "_step_epoch", batches_per_epoch)

        for i in range(self._current_epoch, target_epoch):
            self.logger.epoch = i + 1
            self._current_epoch = i

            self.train_epoch()
            self.eval_epoch()

            if save_chkpt_every and self.chkpt_path and (i + 1) % save_chkpt_every == 0:
                self.save_checkpoint(
                    os.path.join(self.chkpt_path, f"epoch_{i + 1}.pt")
                )

    def get_state_dict(self):
        state_dict = {
            key: getattr(self, key).state_dict
            for key in ["model", "optimizer", "lr_scheduler"]
        }
        state_dict.update({
            "epoch": self._current_epoch
        })

    def load_state_dict(self, state_dict: dict):
        for key in ["model", "optimizer", "lr_scheduler"]:
            getattr(self, key).load_state_dict(state_dict.get(key))
        self._current_epoch = int(state_dict.get("epoch") or 0)

    def save_checkpoint(self, path: str):
        torch.save(self.get_state_dict(), path)

    def load_checkpoint(self, path: str):
        self.load_state_dict(torch.load(path))

    def training_metrics(self, net_input: NetInput):
        lr = self.lr_scheduler.get_last_lr() if self.lr_scheduler is not None else 0
        num_toks = net_input.num_input_toks()

        return {
            "lr": lr,
            "tokens_per_batch": num_toks,
            "vram": torch.cuda.mem_get_info()[0]
        }
