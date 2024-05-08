from abc import abstractmethod
from argparse import Namespace
from typing import Iterable
import torch
from .dataset import ClassificationDataset, LanguagePairDataset
from .utils import pad_sequence


def collate_batch(data: list[list[int]], eos: int, sos: int, pad: int, padding="right", append_sos=False):
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
    # assert pad in ["left", "right"], "Padding must be either left or right. Invalid value: {}".format(pad)
    assert padding == "right", "Please use right padding (pad='right') for the time being"

    # insert sos, eos and calculate max_len
    max_len = 0
    processed_data = []
    for sample in data:
        processed_sample = [*sample, eos] if not append_sos else [sos, *sample, eos]

        processed_data.append(processed_sample)
        max_len = max(max_len, len(processed_sample))

    # batch by max_len
    batched_data = []
    for sent in processed_data:
        batched_data.append(pad_sequence(sent, pad, max_len, padding))

    return batched_data


class BaseCollator(Iterable):
    def __init__(self, dataset, batch_size: int):
        self.i = 0
        self.dataset = dataset
        self.bsz = batch_size

    def __len__(self):
        return len(self.dataset)

    def range(self) -> tuple[int, int]:
        """
        The current batch range
        """
        return (self.i, min(self.i + self.bsz, len(self.dataset)))

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

    @abstractmethod
    def collate(self, batch):
        pass


class ClassificationCollator(BaseCollator):
    def __init__(self, dataset: ClassificationDataset, batch_size: int, padding="right"):
        super().__init__(dataset, batch_size)

        dictionary = dataset.dictionary
        self.eos = dictionary.eos
        self.sos = dictionary.sos
        self.pad = dictionary.pad
        self.padding = padding

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

        batched_x = torch.tensor(collate_batch(x, eos=self.eos, sos=self.sos, pad=self.pad, padding=self.padding))
        batched_y = torch.tensor(y)

        return batched_x, batched_y


class LanguagePairCollator(BaseCollator):
    def __init__(self, dataset: LanguagePairDataset, batch_size: int, src_padding="right", tgt_padding="right"):
        super().__init__(dataset, batch_size)

        self.src = Namespace(
            dictionary=dataset.src_dict,
            eos=dataset.src_dict.eos,
            sos=dataset.src_dict.sos,
            pad=dataset.src_dict.pad,
            padding=src_padding,
        )

        self.tgt = Namespace(
            dictionary=dataset.tgt_dict,
            eos=dataset.tgt_dict.eos,
            sos=dataset.tgt_dict.sos,
            pad=dataset.tgt_dict.pad,
            padding=tgt_padding,
        )

    def collate(self, batch):
        """
        Args
        ----
        batch:
            a list of samples in the batch, with each sample is a tuple of (x, y).
            each x and y is the source and target sequence, respectively.

        Return
        ------
        (X, X_dec, Y)
            X [bsz, max_len] contains source sequences.
            X_dec [bsz, max_len] contains the target sequences shifted right by prepending SOS.
            Y [bsz, max_len] contains the target sequences as-is.

            All padded so that every sequence has the same length.
        """
        x = [sample[0] for sample in batch]
        y = [sample[1] for sample in batch]

        batched_x = torch.tensor(
            collate_batch(x, eos=self.src.eos, sos=self.src.sos, pad=self.src.pad, padding=self.src.padding)
        )

        batched_y = torch.tensor(
            collate_batch(y, eos=self.tgt.eos, sos=self.tgt.sos, pad=self.tgt.pad, padding=self.tgt.padding)
        )

        # Input to decoder (output shifted right by SOS)
        batched_x_dec = torch.tensor(
            collate_batch(
                y,
                eos=self.tgt.eos,
                sos=self.tgt.sos,
                pad=self.tgt.pad,
                padding=self.tgt.padding,
                append_sos=True,
            )
        )

        return batched_x, batched_x_dec, batched_y


__all__ = ["ClassificationCollator", "LanguagePairCollator"]
