from dataclasses import dataclass


@dataclass
class TransformerEncoderConfig:
    n_heads: int
    d_model: int
    layers: int
    dropout: float
