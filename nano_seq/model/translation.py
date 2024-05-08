from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from torch import Tensor, nn

from nano_seq.model.base import NetInput
from nano_seq.data.const import PAD
from nano_seq.module.transformer import TransformerDecoder, TransformerEmbedding, TransformerEncoder


class TranslationModel(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        enc_layers: int,
        dec_layers: int,
        enc_dropout: float,
        dec_dropout: float,
        src_vocab_size: int,
        tgt_vocab_size: int,
        pad_idx: int = PAD,
        shared_embedding: bool = False,
    ):
        super().__init__()
        self.pad_idx = pad_idx

        if shared_embedding:
            assert (
                src_vocab_size == tgt_vocab_size
            ), "Target and source vocab size must be identical if embedding is shared"
            emb = TransformerEmbedding(d_model, src_vocab_size, 5000, pad_idx)
        else:
            emb = None

        kwargs: Dict[str, Any] = dict(
            d_model=d_model,
            n_heads=n_heads,
            padding_idx=pad_idx,
            embedding=emb,
        )

        self.encoder = TransformerEncoder(**kwargs, layers=enc_layers, dropout=enc_dropout, vocab_size=src_vocab_size)
        self.decoder = TransformerDecoder(**kwargs, layers=dec_layers, dropout=dec_dropout, vocab_size=tgt_vocab_size)

    def forward(self, x: torch.Tensor, x_dec: Optional[torch.Tensor], enc_mask: torch.Tensor, dec_mask: torch.Tensor):
        h_encoder = self.encoder(x, enc_mask)
        h_decoder = self.decoder(x_dec, h_encoder, enc_mask, dec_mask)

        return h_decoder

    @classmethod
    def from_cfg(cls, cfg, src_vocab_size: int, tgt_vocab_size: int):
        return cls(
            d_model=cfg.embed_dims,
            n_heads=cfg.num_heads,
            enc_layers=cfg.encoder_layers,
            dec_layers=cfg.decoder_layers,
            enc_dropout=cfg.encoder_dropout,
            dec_dropout=cfg.decoder_dropout,
            pad_idx=cfg.pad_idx,
            shared_embedding=cfg.shared_embedding,
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
        )


@dataclass
class TranslationNetInput(NetInput):
    x_enc: Tensor
    x_dec: Tensor
    enc_mask: Tensor
    dec_mask: Tensor
