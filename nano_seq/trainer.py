import os
import re
import glob
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
        self.train_iter.shuffle()

        for batch_idx, sample in enumerate(iter(self.train_iter)):
            _, label = sample
            net_input = self.task.get_net_input(sample)

            logs = self.task.train_step(net_input, label, self.model, self.optimizer, self.criterion)

            if batch_idx % 100 == 0:
                logs.update(self.training_metrics(net_input))

            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            if batch_idx % 10 == 0:
                self.logger.write_train(batch_idx, **logs)

    def eval_epoch(self):
        for batch_idx, sample in enumerate(iter(self.eval_iter)):
            _, label = sample
            net_input = self.task.get_net_input(sample)

            with torch.no_grad():
                logs = self.task.eval_step(net_input, label, self.model, self.criterion)

            self.logger.write_eval(batch_idx, **logs)

    def start_train(self, target_epoch: int, save_chkpt_every: Optional[int] = None, keep_last: Optional[int] = None):
        """
        Start the training process for an additional `num_epochs`.

        This function has a side effect: it will set the logger's training_info["step_epoch"]
        property to enable calculation of global steps in some log handlers
        """
        self.logger.training_info["step_epoch"] = len(self.train_iter)

        for i in range(self._current_epoch, target_epoch):
            self.logger.training_info["epoch"] = i + 1
            self._current_epoch = i

            self.train_epoch()
            self.eval_epoch()

            if save_chkpt_every and self.chkpt_path and (i + 1) % save_chkpt_every == 0:
                self.save_checkpoint(os.path.join(self.chkpt_path, f"epoch_{i + 1}.pt"))

                glob_pattern = os.path.join(self.chkpt_path, "epoch_*.pt")
                regex = r"epoch_(\d+)\.pt"
                if keep_last is not None and len(glob.glob(glob_pattern)) > keep_last:
                    delete_old_checkpoint(glob_pattern, regex)

    def get_state_dict(self):
        state_dict = {key: getattr(self, key).state_dict() for key in ["model", "optimizer", "lr_scheduler"]}
        state_dict.update({"epoch": self._current_epoch})

    def load_state_dict(self, state_dict: dict):
        for key in ["model", "optimizer", "lr_scheduler"]:
            getattr(self, key).load_state_dict(state_dict.get(key))
        self._current_epoch = int(state_dict.get("epoch") or 0)

    def save_checkpoint(self, path: str):
        torch.save(self.get_state_dict(), path)

    def load_checkpoint(self, path: str):
        self.load_state_dict(torch.load(path))

    def training_metrics(self, net_input: NetInput):
        lr = self.lr_scheduler.get_last_lr()[0] if self.lr_scheduler is not None else 0
        num_toks = net_input.num_input_toks()

        return {"lr": lr, "tokens_per_batch": num_toks, "samples_per_batch": len(getattr(net_input, "x"))}


def delete_old_checkpoint(glob_pattern: str, num_extract_regex: str):
    oldest_file = list(
        sorted(glob.glob(glob_pattern), key=lambda x: re.search(num_extract_regex, x).group(1), reverse=True)
    )[0]
    os.remove(oldest_file)
