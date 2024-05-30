from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Optional

from tqdm import tqdm


class LogHandler(ABC):
    def __init__(self, name: str):
        """
        Interface for handlers (sinks) of metrics
        
        Args
        ----
        name: str
            name of this handler (for ease of access from client)
        """
        self.name = name
        self.logger = None

    @abstractmethod
    def write(self, split: str, step: int, epoch: int, data: dict):
        pass

    def set_logger(self, logger: "Logger") -> "LogHandler":
        self.logger = logger
        return self


class Logger:
    def __init__(self, handlers: Optional[list[LogHandler]] = None, stdout: bool = True):
        """
        Handling log and distribute to log sinks

        Args
        ----
        handlers: list[LogHandler]
            optional list of log sinks
        stdout: bool
            display a progress bar during training and evaluation
        """
        # default in-memory log sink
        container_handler = ContainerLogHandler("container", ["train", "eval"])
        handlers = [container_handler, *(handlers or [])]

        if stdout:
            handlers.append(StdoutLogHandler("stdout", container_handler.container))

        self.handlers = {handler.name: handler.set_logger(self) for handler in handlers} if handlers is not None else {}

        self.epoch = 1

    def write(self, split: str, batch_idx: int, **kwargs):
        for handler in self.handlers.values():
            handler.write(split, batch_idx, self.epoch, kwargs)

    def write_train(self, batch_idx: int, **kwargs):
        self.write("train", batch_idx, **kwargs)

    def write_eval(self, batch_idx: int, **kwargs):
        self.write("eval", batch_idx, **kwargs)


class LogContainer:
    def __init__(self, splits: list[str]):
        """
        Container to store logged metrics in-memory

        Args
        ----
        splits: list[str]
            list of split names (e.g. train, valid1, valid2, ...) to store the logs

        Note
        ----
        Log is stored in the _container attribute of the object in the format of

        dict:
            key: split name
            value: dict
                key: epoch number
                value: list of step-wise metric dict
                    key: metric name
                    value: value
        """
        # container to store step-wise metrics
        self._container = {split: defaultdict(list) for split in splits}

        # to efficiently calculate the running averaged metrics
        self._avg = {split: defaultdict(lambda: defaultdict(lambda: 0)) for split in splits}

    def write(self, split: str, step: int, epoch: int, data: dict):
        self._container[split][epoch].append(data)

        for metric_name, value in data.items():
            current_value = self._avg[split][epoch][metric_name]
            new_value = (current_value * step + value) / (step + 1)

            self._avg[split][epoch][metric_name] = new_value


class ContainerLogHandler(LogHandler):
    def __init__(self, name: str, splits: list[str]):
        """
        `LogHandler` interface adapter for `LogContainer`
        """
        super().__init__(name)
        self.container = LogContainer(splits)

    def write(self, *args, **kwargs):
        self.container.write(*args, **kwargs)


class StdoutLogHandler(LogHandler):
    def __init__(self, name: str, container: LogContainer):
        """
        Creates a tqdm progress bar every epoch and print logged metrics every 10 steps

        Args
        ----
        name: str
            name of the logger
        container: LogContainer
            reference to the underlying container used in the logger's ContainerLogHandler
        """
        super().__init__(name)
        self.tqdm = None
        self.eval_tqdm = None
        self.container = container
        self._step_epoch = 0

    def write(self, split: str, step: int, epoch: int, *args):
        if self.tqdm is None:
            self.tqdm = tqdm(total=self._step_epoch, position=0)

        if split == "train":
            self.tqdm.update(step - self.tqdm.n)
            if step % 10 == 0:
                self.tqdm.set_description(f"Epoch {epoch}")
                self.tqdm.set_postfix(self._fmt("train", epoch))

        elif split == "eval":
            self.eval_tqdm = self.eval_tqdm or tqdm(
                total=self._step_epoch, position=1, desc=f"Eval {epoch}", leave=False
            )
            self.eval_tqdm.update(step - self.eval_tqdm.n)
            if step % 10 == 0:
                self.eval_tqdm.set_postfix(self._fmt("eval", epoch))

    def _fmt(self, split: str, epoch: int) -> dict:
        return {k: round(v, 4) for k, v in self.container._avg[split][epoch].items()}
