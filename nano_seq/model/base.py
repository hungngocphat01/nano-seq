from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict


@dataclass
class NetInput(ABC):
    def asdict(self):
        return asdict(self)

    @abstractmethod
    def num_input_toks(self) -> int:
        pass


__all__ = [
    "NetInput"
]
