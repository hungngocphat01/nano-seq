import os
from dataclasses import dataclass, field
from typing import Any

import torch
from torch.nn.modules import Module

from nano_seq.data import Dictionary
from nano_seq.data.collator import LanguagePairCollator
from nano_seq.data.const import PAD
from nano_seq.data.dataset import LanguagePairDataset
from nano_seq.data.utils import get_decoder_mask, get_padding_mask
from nano_seq.model.translation import TranslationModel, TranslationNetInput
from nano_seq.task.base import BaseTask
from nano_seq.utils.search import batched_greedy


@dataclass
class TranslationConfig:
    embed_dims: int
    num_heads: int
    ffn_project_dims: int
    encoder_layers: int
    encoder_dropout: float
    decoder_layers: int
    decoder_dropout: float

    train_path: str
    valid_path: str
    dict_path: str
    shared_dict: bool
    shared_embedding: bool

    src_lang: str
    tgt_lang: str

    batch_size: int
    left_pad_src: bool = field(default=False)
    left_pad_tgt: bool = field(default=False)

    pad_idx: int = field(default=PAD)


def get_padding_str(left_pad: bool):
    if left_pad:
        return "left"
    return "right"


def load_langpair_dictionary(dict_path: str, src_lang: str, tgt_lang: str, shared_dictionary=False):
    if not shared_dictionary:
        src_dict = Dictionary.from_spm(os.path.join(dict_path, f"{src_lang}.vocab"))
        tgt_dict = Dictionary.from_spm(os.path.join(dict_path, f"{tgt_lang}.vocab"))
        return src_dict, tgt_dict

    shared_dict = Dictionary.from_spm(os.path.join(dict_path, "dictionary.vocab"))
    return shared_dict, shared_dict


class TranslationTask(BaseTask):
    def __init__(self, cfg: TranslationConfig):
        self.cfg = cfg

    def prepare(self):
        cfg = self.cfg

        src_dict, tgt_dict = load_langpair_dictionary(cfg.dict_path, cfg.src_lang, cfg.tgt_lang, cfg.shared_dict)
        train_iter, valid_iter = self._load_dataset(src_dict, tgt_dict)

        model = TranslationModel.from_cfg(cfg, len(src_dict), len(tgt_dict))

        return train_iter, valid_iter, model

    def get_net_input(self, batch) -> TranslationNetInput:
        (x_enc, x_dec), _ = batch

        return TranslationNetInput(
            x=x_enc, x_dec=x_dec, enc_mask=get_padding_mask(x_enc), dec_mask=get_decoder_mask(x_dec)
        )

    def _load_dataset(self, src_dict: Dictionary, tgt_dict: Dictionary):
        cfg = self.cfg

        def load_collator_split(prefix: str):
            dataset = LanguagePairDataset.from_text_file(prefix, cfg.src_lang, cfg.tgt_lang, src_dict, tgt_dict, True)
            return LanguagePairCollator(
                dataset, cfg.batch_size, get_padding_str(cfg.left_pad_src), get_padding_str(cfg.left_pad_tgt)
            )

        return load_collator_split(cfg.train_path), load_collator_split(cfg.valid_path)

    def train_step(
        self,
        net_input: TranslationNetInput,
        label: torch.Tensor,
        model: TranslationModel,
        optimizer: torch.optim.Optimizer,
        criterion,
    ) -> dict:
        """
        Function to call for one training batch

        Args
        ----
        net_input:
            neural network input, defined by batch iterator
        label:
            1d tensor of shape [bsz]
        model:
            the neural network module

        Returns
        -------
        dict
            logging metrics
        """
        model.train()

        optimizer.zero_grad()
        logits = model(**net_input.asdict())
        loss = criterion(logits, label)
        loss.backward()

        return {"loss": loss.detach().item()}

    def eval_step(
        self, net_input: TranslationNetInput, label: torch.Tensor, model: TranslationModel, criterion: Module
    ) -> dict:
        model.eval()

        with torch.no_grad():
            logits = model(**net_input.asdict())
            loss = criterion(logits, label)

        return {"loss": loss.item()}

    def decode(self, model: TranslationModel, tgt_dict: Dictionary, batch: Any) -> list[str]:
        model = model.eval()
        net_input = self.get_net_input(batch)

        with torch.no_grad():
            h_encoder = model.encoder(net_input.x, net_input.enc_mask)

        enc_mask = net_input.enc_mask
        out_tensor = batched_greedy(model, h_encoder, enc_mask, tgt_dict)
        return tgt_dict.decode_batch(out_tensor.cpu().numpy())
