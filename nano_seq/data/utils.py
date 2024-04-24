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


def get_encoder_mask(x: torch.Tensor, pad_idx=PAD):
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
