from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from nano_seq.data.base import NetInput

from .dictionary import Dictionary


class ClassificationDataset(Dataset):
    def __init__(self, src_list: list[str], tgt_list: list[int] | list[str], dictionary: Dictionary):
        """
        Dataset for sentence classification

        Args
        ----
        src_list: list[str]
            list of source sentences
        tgt_list: list[int]
            list of label indices
        dictionary: nano_seq.data.dictionary.Dictionary
            dictionary of source language
        """
        self.data_path = src_list
        self.dictionary = dictionary

        assert len(src_list) == len(tgt_list), "Source and target list must have equal length"

        self.data = [
            (dictionary.encode(x.split()), int(y)) for x, y in tqdm(zip(src_list, tgt_list), desc="Loading dataset")
        ]

    def __getitem__(self, idx: int):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    @classmethod
    def from_text_file(cls, src_path: str, tgt_path: str, dictionary: Dictionary):
        """
        Load the dataset from filesystem

        The dataset must contain two text files
            src: newline-separated list of space-tokenized input sequences
            tgt: newline-separated list of label indices

        Args
        ----
        src_path: str
            path of source text file
        tgt_path: str
            path of target text file
        dictionary: nano_seq.data.dictionary.Dictionary
            dictionary of source language
        """
        with open(src_path, "rt", encoding="utf-8") as src_f, open(tgt_path, "rt", encoding="utf-8") as tgt_f:
            src_lines = [line.strip() for line in src_f.readlines()]
            tgt_lines = [line.strip() for line in tgt_f.readlines()]

        return cls(src_lines, tgt_lines, dictionary)


@dataclass
class ClassificationNetInput(NetInput):
    x: torch.Tensor
    mask: torch.Tensor


__all__ = [
    "ClassificationDataset",
    "ClassificationNetInput"
]
