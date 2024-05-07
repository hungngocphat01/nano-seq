from abc import ABC
from dataclasses import dataclass, asdict


@dataclass
class NetInput(ABC):
    def asdict(self):
        return asdict(self)


__all__ = [
    "NetInput"
]
