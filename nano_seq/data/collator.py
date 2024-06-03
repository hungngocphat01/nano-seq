import random
from abc import abstractmethod
from collections import namedtuple
from argparse import Namespace
from typing import Iterable
import torch
from .dataset import ClassificationDataset, LanguagePairDataset
from .utils import pad_sequence

DataSample = namedtuple("DataSample", "idx len")
BatchItem = namedtuple("BatchItem", "idx num_samples num_toks")

def collate_batch(
    data: list[list[int]], eos: int, sos: int, pad: int, padding="right", prepend_sos=False, append_eos=False
):
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
        if prepend_sos:
            sample = [sos] + sample
        if append_eos:
            sample = sample + [eos]

        processed_data.append(sample)
        max_len = max(max_len, len(sample))

    # batch by max_len
    batched_data = []
    for sent in processed_data:
        batched_data.append(pad_sequence(sent, pad, max_len, padding))

    return batched_data


def _build_batches(num_tokens: list[DataSample], batch_size: int) -> list[BatchItem]:
    batches = []

    current_num_sents = 0
    current_num_toks = 0
    current_start = num_tokens[0].idx

    for sample in num_tokens:
        current_num_sents += 1
        current_num_toks += sample.len

        if current_num_toks > batch_size:
            batches.append(
                BatchItem(current_start, current_num_sents - 1, current_num_toks - sample.len)
            )
            current_num_sents = 1
            current_start = sample.idx
            current_num_toks = sample.len

    # final batch, smaller than max batch size
    if batches[-1].idx != current_start:
        batches.append(
            BatchItem(current_start, current_num_sents, current_num_toks)
        )

    return batches


class BaseCollator(Iterable):
    def __init__(self, dataset, batch_size: int):
        self.dataset = dataset
        # number of source tokens per batch
        self.bsz = batch_size
        self.batch_mappings: list[BatchItem] = _build_batches(
            self._count_num_tokens(self.dataset),
            batch_size
        )
        self.batch_mapping_iter = None

    def shuffle(self):
        random.shuffle(self.batch_mappings)

    def _count_num_tokens(self, dataset):
        # tuple of index, length
        return [DataSample(i, len(x)) for i, (x, _) in enumerate(dataset)]

    def __iter__(self):
        self.batch_mapping_iter = iter(self.batch_mappings)
        return self

    def __len__(self):
        return len(self.batch_mappings)

    def __next__(self):
        assert self.batch_mapping_iter is not None, "Please call iter() on this collator before iterating it"
        batch = next(self.batch_mapping_iter)
        return self.collate(
            self.dataset[batch.idx : batch.idx + batch.num_samples]
        )    # type: ignore

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
            collate_batch(
                y, eos=self.tgt.eos, sos=self.tgt.sos, pad=self.tgt.pad, padding=self.tgt.padding, append_eos=True
            )
        )

        # Input to decoder (output shifted right by SOS)
        batched_x_dec = torch.tensor(
            collate_batch(
                y,
                eos=self.tgt.eos,
                sos=self.tgt.sos,
                pad=self.tgt.pad,
                padding=self.tgt.padding,
                prepend_sos=True,
            )
        )

        return (batched_x, batched_x_dec), batched_y


__all__ = ["ClassificationCollator", "LanguagePairCollator"]
