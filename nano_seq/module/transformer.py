import math
import torch
from torch import nn
from torch.nn import functional as F


class MultiheadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        self.d_head = self.d_model // self.n_heads
        assert self.d_head * self.n_heads == self.d_model

        # transform on big matrix then split == split then transform on small matrices
        # in the paper: does the second way, but the first way is easier to implement
        self.qkv_linears = nn.ModuleList([nn.Linear(self.d_model, self.d_model) for _ in range(4)])
        self.ffn_out = nn.Linear(self.d_model, self.d_model)
        self.softmax = nn.Softmax(dim=-1)

    def attention(self, q, k, v, mask=None):
        """
        Compute multi-head attention score
        Attention(Q, K, V) = Softmax(Q @ K.T / sqrt(d_head)) @ V

        Args
        ----
        q, k, v: tensor
            shape [bsz, n_head, len, d_head]
        mask:
            as described in forward method

        Returns
        -------
        tensor of shape [bsz, n_head, len_q, d_head]
        """
        # q @ k.t: [bsz, n_heads, len_q, len_k]
        q_kt = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_head)

        if mask is not None:
            q_kt = q_kt.masked_fill(mask == 0, -1e8)

        raw_score = self.softmax(q_kt)
        score = torch.matmul(raw_score, v)

        return score, raw_score

    def _head_split(self, q, k, v):
        # head spliting
        # after linear transformation: [bsz, max_len, d_head]
        # after reshape: [bsz, max_len, n_heads, d_head]
        # after transpose: [bsz, n_heads, max_len, d_head]
        bsz = q.size(0)

        return [
            f(x).reshape(bsz, -1, self.n_heads, self.d_head).transpose(1, 2)
            for f, x in zip(self.qkv_linears, (q, k, v))
        ]

    def forward(self, q, k, v, mask=None):
        """
        Compute multi-head attention

        Args
        ----
        q: tensor
            shape [bsz, len_q, d_model]
        k, v: tensor
            shape [bsz, len_k, d_model]
        mask: tensor
            shape [bsz, 1, 1, len_k] if used in encoder or cross attention
            shape [bsz, 1, len_q, len_k] if used in decoder attention

        Returns
        -------
        tuple[tensor, tensor]
            multi-head attention: shape [bsz, len_q, d_model]
            raw attention: shape [bsz, n_head, len_q, len_k]
        """
        bsz = q.size(0)

        q, k, v = self._head_split(q, k, v)
        attn, raw_attn = self.attention(q, k, v, mask)

        # "concat" heads step
        # attn has shape [bsz, n_head, len_q, d_head]
        # after permute: [bsz, len_q, n_head, d_head]
        # after reshape: [bsz, len_q, d_model]
        attn = attn.permute(0, 2, 1, 3)
        attn = attn.reshape(bsz, -1, self.d_model)

        # final transform
        return self.ffn_out(attn), raw_attn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len=5000):
        super().__init__()
        pos = torch.stack([torch.arange(max_len)] * d_model, dim=1)
        pe = torch.zeros(max_len, d_model)
        i = torch.arange(d_model)
        pe[:, 0::2] = torch.cos(pos[:, 0::2] / torch.pow(torch.tensor(10000), i[0::2] / d_model))
        pe[:, 1::2] = torch.sin(pos[:, 1::2] / torch.pow(torch.tensor(10000), (i[1::2] - 1) / d_model))

        # shape [max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args
        ----
        x: tensor
            shape [bsz, max_len]
        """

        return self.pe[: x.size(1), :]


class TransformerEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, max_len=5000, padding_idx=0):
        super().__init__()
        self.d_model = d_model
        self.pos = PositionalEncoding(d_model, max_len)
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)

    def forward(self, x):
        """
        Args
        ----
        x: tensor
            integer tensor of shape [bsz, max_len]
        """
        return self.emb(x) + self.pos(x)


class FeedforwardNet(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, activation=F.relu):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.activation = activation

    def forward(self, x):
        return self.activation(self.linear(x))


class TransformerLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        self.ffn = FeedforwardNet(d_model, d_model)
        self.attention = MultiheadAttention(d_model, n_heads)
        self.layer_norm = nn.ModuleList([nn.LayerNorm(d_model), nn.LayerNorm(d_model)])


class EncoderLayer(TransformerLayer):
    def forward(self, x, mask):
        """
        Args
        ----
        x: tensor
            shape [bsz, src_len, d_model]
        mask: tensor
            shape [bsz, 1, 1, src_len]
        """
        x_attn, _ = self.attention(x, x, x, mask)
        x_norm1 = self.layer_norm[0](x_attn)
        x_add_norm1 = x + x_norm1

        x_ffn = self.ffn(x_add_norm1)
        x_norm2 = self.layer_norm[1](x_ffn)
        x_add_norm2 = x_add_norm1 + x_norm2

        return x_add_norm2


class DecoderLayer(TransformerLayer):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__(d_model, n_heads)

        self.cross_attention = MultiheadAttention(d_model, n_heads)
        self.layer_norm.append(nn.LayerNorm(d_model))

    def forward(self, x, x_enc, mask, mask_enc):
        """
        Args
        ----
        x: tensor
            shape [bsz, tgt_len, d_model]
        x_enc: tensor
            shape [bsz, src_len, d_model]
        mask: tensor
            shape [bsz, 1, tgt_len, tgt_len]
        mask_enc: tensor
            shape [bsz, 1, 1, src_len]
        """
        x_attn, _ = self.attention(x, x, x, mask)
        x_norm1 = self.layer_norm[0](x_attn)
        x_add_norm1 = x_norm1 + x

        x_cross, _ = self.cross_attention(x_add_norm1, x_enc, x_enc, mask_enc)
        x_norm2 = self.layer_norm[1](x_cross)
        x_add_norm2 = x_norm2 + x_add_norm1

        x_ffn = self.ffn(x_add_norm2)
        x_norm3 = self.layer_norm[2](x_ffn)
        x_add_norm3 = x_norm3 + x_add_norm2

        return x_add_norm3


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        layers: int,
        vocab_size: int,
        dropout: float = 0.1,
        max_len=5000,
        padding_idx=0,
    ):
        super().__init__()
        self.emb = TransformerEmbedding(d_model, vocab_size, max_len, padding_idx)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads) for _ in range(layers)])
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        """
        Args
        ----
        x: tensor
            integer tensor of shape [bsz, max_len]
        mask: tensor
            shape [bsz, 1, 1, max_len]
        """
        x = self.emb(x)
        for layer in self.layers:
            x = x + self.dropout(layer(x, mask))
        return x


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        layers: int,
        vocab_size: int,
        dropout: float = 0.1,
        max_len=5000,
        padding_idx=-1,
    ):
        super().__init__()
        self.emb = TransformerEmbedding(d_model, vocab_size, max_len, padding_idx)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads) for _ in range(layers)])
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_enc, mask, mask_enc):
        """
        Args
        ----
        x: tensor
            shape [bsz, tgt_len, d_model]
        x_enc: tensor
            shape [bsz, src_len, d_model]
        mask: tensor
            shape [bsz, 1, tgt_len, tgt_len]
        mask_enc: tensor
            shape [bsz, 1, 1, src_len]
        """
        x = self.emb(x)
        for layer in self.layers:
            x = x + self.dropout(layer(x, x_enc, mask, mask_enc))
        return x
