from abc import ABC, abstractmethod
from collections import defaultdict


class Logger(ABC):
    @abstractmethod
    def write_train(self, batch_idx: int, **kwargs):
        pass

    @abstractmethod
    def write_eval(self, **kwargs):
        pass


class SimpleLogger(Logger):
    def __init__(self):
        self.step = 0
        self.epoch = 0
        self.train_metrics = []
        self.eval_metrics = []

    def write_train(self, batch_idx: int, **kwargs):
        self.train_metrics.append({"epoch": self.epoch, "batch_idx": batch_idx, "step": self.step, **kwargs})
        self.step += 1

        log_str = "Epoch {} | Step {} ".format(self.epoch, self.step)
        for name, value in kwargs.items():
            log_str += "| {}: {} ".format(name, value)
        print(log_str)

    def write_eval(self, **kwargs):
        self.eval_metrics.append({"epoch": self.epoch, "step": self.step, **kwargs})

        log_str = "Epoch {} | Step {} ".format(self.epoch, self.step)
        for name, value in kwargs.items():
            log_str += "| {}: {} ".format(name, value)
        print(log_str)
