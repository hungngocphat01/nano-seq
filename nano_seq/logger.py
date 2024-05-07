from abc import ABC, abstractmethod
from collections import defaultdict


class Logger(ABC):
    def __init__(self):
        self.step = 0
        self._epoch = 1

    @property
    def epoch(self):
        return self._epoch

    @epoch.setter
    def epoch(self, value: int):
        pass

    @abstractmethod
    def write_train(self, batch_idx: int, **kwargs):
        pass

    @abstractmethod
    def write_eval(self, batch_idx: int, **kwargs):
        pass


class SimpleLogger(Logger):
    def __init__(self):
        super().__init__()
        self.train_metrics = {}
        self.eval_metrics = {}

    @property
    def epoch(self):
        return self._epoch

    @epoch.setter
    def epoch(self, value: int):
        self._epoch = value

        first_value = {"step": -1, "value": 0}
        self.train_metrics[value] = defaultdict(lambda: [first_value])
        self.eval_metrics[value] = defaultdict(lambda: [first_value])

    def _write(self, container, batch_idx, **kwargs):
        moving_avg = {}
        for name, value in kwargs.items():
            metric_arr = container[self.epoch][name]
            avg_value = (metric_arr[-1]["value"] * batch_idx + value) / (batch_idx + 1)
            moving_avg[name] = avg_value
            metric_arr.append(
                {
                    "step": self.step,
                    "value": avg_value,
                }
            )
        return moving_avg

    def write_train(self, batch_idx: int, **kwargs):
        moving_avg = self._write(self.train_metrics, batch_idx, **kwargs)
        self.step += 1
        return moving_avg

    def write_eval(self, batch_idx, **kwargs):
        moving_avg = self._write(self.train_metrics, batch_idx, **kwargs)
        return moving_avg


class PrintLogger(SimpleLogger):
    def write_train(self, batch_idx: int, **kwargs):
        ma = super().write_train(batch_idx, **kwargs)

        log_str = "Epoch {} | Step {} ".format(self.epoch, self.step)
        for name, value in ma.items():
            log_str += "| {}: {} ".format(name, value)
        print(log_str)

        return ma

    def write_eval(self, batch_idx, **kwargs):
        ma = super().write_eval(batch_idx, **kwargs)

        log_str = "Validation | Epoch {} | Step {} ".format(self.epoch, self.step)
        for name, value in ma.items():
            log_str += "| {}: {} ".format(name, value)
        print(log_str)

        return ma
