from typing import Type
import torch
from .dataset import ClassificationDataset
from .utils import collate_batch


class ClassificationCollator:
    def __init__(self, dataset: ClassificationDataset, batch_size: int, device, padding="left"):
        self.i = 0
        self.bsz = batch_size
        self.dataset = dataset
        self.device = device

        dictionary = dataset.dictionary

        self.eos = dictionary.eos
        self.sos = dictionary.sos
        self.pad = dictionary.pad
        self.padding = padding

    def __len__(self):
        return len(self.dataset)

    def range(self) -> tuple[int, int]:
        """
        The current batch range
        """
        return (self.i, min(self.i + self.bsz, len(self.dataset)))

    def collate(self, batch: list[tuple[list[int], int]]):
        """
        Args
        ----
        batch:
            a list of samples in the batch, with each sample is a tuple of (x, y).
            each x is an array of token ids, each y is the class index.
            
        Return
        ------
        (X, Y)
            where X [bsz, max_len] contains the batched and padded source sequences.
            Y [bsz,] contains class indices.
        """
        x = [sample[0] for sample in batch]
        y = [sample[1] for sample in batch]

        batched_x = torch.tensor(
            collate_batch(x, eos_idx=self.eos, sos_idx=self.sos, pad_idx=self.pad, pad=self.padding)
        )
        batched_y = torch.tensor(y)

        return batched_x, batched_y

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        n = len(self.dataset)
        if self.i >= n:
            raise StopIteration("Dataset has ended")

        begin, end = self.range()
        batch = self.dataset[begin:end]  # type: ignore
        self.i += self.bsz

        return self.collate(batch)       # type: ignore
