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


def collate_batch(data: list[list[int]], eos_idx: int, sos_idx: int, pad_idx: int, pad="left"):
    """
    Create padding for a batch

    Args
    ----
    data:
        list of sentences in the batch. each sentence is a list of token ids
    eos_idx:
        the token id to insert at the beginning of every sentence
    sos_idx:
        the token id to insert at the end of every sentence
    pad_idx:
        the token id to insert as padding tokens
    pad:
        padding strategy, "left" or "right"

    Return
    ------
    list[list[int]]
        padded batch of sentences with equal size
    """
    assert pad in ["left", "right"], "Padding must be either left or right. Invalid value: {}".format(pad)

    # insert sos, eos and calculate max_len
    max_len = 0
    processed_data = []
    for sample in data:
        processed_sample = [sos_idx, *sample, eos_idx]
        processed_data.append(processed_sample)
        max_len = max(max_len, len(processed_sample))

    # batch by max_len
    batched_data = []
    for sent in processed_data:
        batched_data.append(pad_sequence(sent, pad_idx, max_len, pad))

    return batched_data


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
