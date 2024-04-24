from typing import Iterable
import torch
from .dataset import ClassificationDataset
from .utils import pad_sequence

def collate_batch(data: list[list[int]], eos_idx: int, sos_idx: int, pad_idx: int, pad="left"):
    """
    Create padding for a batch

    Args
    ----
    data:
        list of sentences in the batch. each sentence is a list of token ids
    eos_idx:
        the token id to insert at the beginning of every sentence
    sos_idx:
        the token id to insert at the end of every sentence
    pad_idx:
        the token id to insert as padding tokens
    pad:
        padding strategy, "left" or "right"

    Return
    ------
    list[list[int]]
        padded batch of sentences with equal size
    """
    assert pad in ["left", "right"], "Padding must be either left or right. Invalid value: {}".format(pad)

    # insert sos, eos and calculate max_len
    max_len = 0
    processed_data = []
    for sample in data:
        processed_sample = [sos_idx, *sample, eos_idx]
        processed_data.append(processed_sample)
        max_len = max(max_len, len(processed_sample))

    # batch by max_len
    batched_data = []
    for sent in processed_data:
        batched_data.append(pad_sequence(sent, pad_idx, max_len, pad))

    return batched_data


class ClassificationCollator(Iterable):
    def __init__(self, dataset: ClassificationDataset, batch_size: int, padding="left"):
        self.i = 0
        self.bsz = batch_size
        self.dataset = dataset

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

        return self.collate(batch)  # type: ignore


__all__ = [
    "ClassificationCollator"
]
