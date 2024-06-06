import math
import torch
from torch.optim.lr_scheduler import LRScheduler


def get_lr(d_model: int, warm_up_steps: int, step: int):
    return math.pow(d_model, -0.5) * min(math.pow(step, -0.5), step * math.pow(warm_up_steps, -1.5))


class TransformerLRScheduler(LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        d_model: int,
        multiplier: float,
        warm_up_steps: int,
        last_epoch: int = -1,
        verbose: bool = False,
    ):
        self.d_model = d_model
        self.warm_up_steps = warm_up_steps
        self.num_param_groups = len(optimizer.param_groups)
        self.multiplier = multiplier

        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        return [
            get_lr(self.d_model, self.warm_up_steps, self._step_count) * self.multiplier    # type: ignore
        ] * self.num_param_groups
