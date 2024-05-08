from torch.utils.data import Dataset
from tqdm import tqdm

from .dictionary import Dictionary


class ClassificationDataset(Dataset):
    def __init__(self, src_list: list[str], tgt_list: list[int] | list[str], dictionary: Dictionary):
        """
        Dataset for sentence classification

        Args
        ----
        src_list: list[str]
            list of space-tokenized source sentences
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



class LanguagePairDataset(Dataset):
    def __init__(self, src_list: list[str], tgt_list: list[str], src_dict: Dictionary, tgt_dict: Dictionary):
        """
        Dataset for bilingual translation

        Args
        ----
        src_list: list[str]
            list of space-tokenized source sentences
        tgt_list: list[str]
            list of space-tokenized target sentences
        src_dict: nano_seq.data.dictionary.Dictionary
            dictionary of source language
        tgt_dict: nano_seq.data.dictionary.Dictionary
            dictionary of target language
        """
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

        self.data = [
            (src_dict.encode(src_sent.split()), tgt_dict.encode(tgt_sent.split()))
            for (src_sent, tgt_sent) in tqdm(zip(src_list, tgt_list), desc="Loading and encoding data")
        ]

    def __getitem__(self, index: int):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    @classmethod
    def from_text_file(cls, src_path: str, tgt_path: str, src_dict: Dictionary, tgt_dict: Dictionary):
        with open(src_path, "rt", encoding="utf-8") as src_f, open(tgt_path, "rt", encoding="utf-8") as tgt_f:
            src_lines = [line.strip() for line in src_f.readlines()]
            tgt_lines = [line.strip() for line in tgt_f.readlines()]

        return cls(src_lines, tgt_lines, src_dict, tgt_dict)


__all__ = [
    "ClassificationDataset",
    "LanguagePairDataset"
]
