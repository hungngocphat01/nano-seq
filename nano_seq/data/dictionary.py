from tqdm import tqdm
from .const import PAD, SOS, EOS, UNK


class Dictionary:
    def __init__(self, pad_idx: int = PAD, sos_id: int = SOS, eos_id: int = EOS, unk_id: int = UNK):
        """
        A utility class for conversion of text to token id and vice versa

        Args
        ----
        pad_idx: int
            token id of the padding positions when collating a sequence batch
        sos_id: int
            start of sentence token id
        eos_id: int
            end of sentence token id
        unk_id: int
            out-of-vocabulary token id
        """
        self.unk = unk_id
        self.sos = sos_id
        self.eos = eos_id
        self.pad = pad_idx

        self._tok2id = {"<s>": sos_id, "</s>": eos_id, "<unk>": unk_id, "<pad>": pad_idx}
        self._id2tok = self._create_id2tok(self._tok2id)

    def __len__(self):
        return max(self._id2tok.keys()) + 1

    def _create_id2tok(self, tok2id: dict):
        return {id: tok for tok, id in tok2id.items()}

    def initialize(self, tok2id: dict):
        self._tok2id.update(tok2id)
        self._id2tok.update(self._create_id2tok(self._tok2id))

    def tok2id(self, tok: str):
        try:
            return self._tok2id[tok]
        except KeyError:
            return self.unk

    def id2tok(self, id: int):
        try:
            return self._id2tok[id]
        except KeyError:
            return "<unk>"

    def encode(self, sentence: list[str]):
        """
        Convert a list of text token to id
        """
        return [self.tok2id(tok) for tok in sentence]

    def decode(self, toks: list[int]):
        """
        Convert a list of token id to text
        """
        return [self.id2tok(tok) for tok in toks]

    @classmethod
    def from_spm(cls, spm_dict_path: str, pad_idx: int = PAD, sos_id: int = SOS, eos_id: int = EOS, unk_id: int = UNK):
        """
        Factory method to create a dictionary from an spm .vocab file
        """
        d = cls(pad_idx, sos_id, eos_id, unk_id)

        # number of reserved tokens
        offset = max(d._id2tok.keys())

        tok2id = {}
        with open(spm_dict_path, "rt", encoding="utf-8") as f:
            for i, line in tqdm(enumerate(f), desc="Loading dictionary"):
                token, _ = line.strip().split("\t")

                # reserved tokens must not be redefined
                if token in d._tok2id:
                    continue

                # continue numbering
                tok2id[token] = offset + i

        d.initialize(tok2id)
        return d

__all__ = [
    "Dictionary"
]
