import os
from torch.utils.data import Dataset
from tqdm import tqdm

from .dictionary import Dictionary


class MonolingualDataset(Dataset):
    def __init__(self, sents: list[str], dictionary: Dictionary, lang_code: str):
        self.dictionary = dictionary
        self.lang = lang_code
        self.data = [dictionary.encode(sent.split()) for sent in tqdm(sents, desc=f"Loading dataset for {lang_code}")]

    def __getitem__(self, index: int) -> list[int]:
        return self.data[index]

    @classmethod
    def from_text_file(cls, path: str, lang_code: str, dictionary: Dictionary):
        with open(os.path.join(path, f"{lang_code}.txt"), "rt", encoding="utf-8") as f:
            lines = [l.strip() for l in f.readlines()]
        return cls(lines, dictionary, lang_code)

    def __len__(self):
        return len(self.data)


class LanguagePairDataset(Dataset):
    def __init__(
        self,
        src_dataset: MonolingualDataset,
        tgt_dataset: MonolingualDataset,
        sort_by_length: bool,
    ):
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
        self.src_dict = src_dataset.dictionary
        self.tgt_dict = tgt_dataset.dictionary

        self.src_data = src_dataset
        self.tgt_data = tgt_dataset

        if sort_by_length:
            sorted_pairs = sorted(list(zip(self.src_data.data, self.tgt_data.data)), key=lambda x: len(x[1]))
            self.src_data.data, self.tgt_data.data = list(zip(*sorted_pairs))

    def __getitem__(self, index: int):
        return self.src_data[index], self.tgt_data[index]

    def __len__(self):
        return len(self.src_data)

    @classmethod
    def from_text_file(
        cls, data_path: str, src_lang: str, tgt_lang: str, src_dict: Dictionary, tgt_dict: Dictionary, sort_by_length: bool
    ):
        src_data = MonolingualDataset.from_text_file(data_path, src_lang, src_dict)
        tgt_data = MonolingualDataset.from_text_file(data_path, tgt_lang, tgt_dict)
        return cls(src_data, tgt_data, sort_by_length)


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

        self.data = list(sorted(filter(lambda x: len(x[0]) >= 2, self.data), key=lambda x: len(x[0])))

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


__all__ = ["ClassificationDataset", "LanguagePairDataset", "MonolingualDataset"]
