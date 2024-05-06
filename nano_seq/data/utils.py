from numpy import pad
import torch
from .const import PAD


def pad_sequence(seq: list[int], pad_idx: int, max_len: int, pad="left"):
    """
    Create padding for a single sequence sample

    Args
    ----
    sent:
        list of token ids of the sentence
    pad_idx:
        id of padding index
    max_len:
        the target length to pad to
    pad:
        padding strategy, "left" or "right"

    Return
    ------
    list[int]
        padded list of tokens
    """
    num_pad_toks = max_len - len(seq)
    pad_segment = [pad_idx] * num_pad_toks

    if pad == "left":
        return pad_segment + seq
    elif pad == "right":
        return seq + pad_segment


def get_padding_mask(x: torch.Tensor, pad_idx=PAD):
    """
    Args
    ----
    x: tensor of shape [bsz, max_len]

    Return
    ------
    tensor of shape [bsz, 1, 1, max_len]
        with value 1 where the token is not pad
    """
    assert x.ndim == 2, "input must have size [bsz, max_len]. got only {} dimensions".format(x.ndim)
    return (x != pad_idx).unsqueeze(1).unsqueeze(1).long()


def get_future_mask(x: torch.Tensor, padding = "left"):
    """
    Args
    ----
    x: tensor
        shape [bsz, max_len]

    Return
    ------
    tensor of shape [1, 1, max_len, max_len]
        with value 1 where the token is not pad
    """
    assert x.ndim == 2, "input tensor must have size [bsz, max_len]. got only {} dimensions".format(x.ndim)
    max_len = x.size(1)

    mask = torch.tril(torch.ones((max_len, max_len)))
    if padding == "left":
        mask = torch.flip(mask, dims=[1])

    return mask.unsqueeze(0).unsqueeze(0).long()


def get_decoder_mask(x: torch.Tensor, padding = "left", pad_idx=PAD):
    """
    Args
    ----
    x: tensor
        shape [bsz, max_len]

    Return
    ------
    tensor of shape [bsz, 1, max_len, max_len]
        with value 1 where the token is not pad
    """
    padding_mask = get_padding_mask(x, pad_idx)
    future_mask = get_future_mask(x, padding)

    # 0 (padded) takes precedence
    return torch.minimum(padding_mask, future_mask)
