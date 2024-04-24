from abc import ABC, abstractmethod


class BaseTask(ABC):
    @abstractmethod
    def prepare(self):
        """
        Initialize data loader and model
        
        Returns
        -------
        tuple[train_iter, valid_iter, model]
        """
