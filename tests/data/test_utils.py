import pytest
import torch

from nano_seq.data.utils import pad_sequence, get_padding_mask, get_future_mask, get_decoder_mask


@pytest.fixture
def padded_input():
    return torch.Tensor([[7, 7, 7, 1, 2, 3], [7, 7, 1, 2, 3, 4], [7, 1, 2, 3, 4, 5], [7, 7, 7, 7, 1, 2]]).long()


@pytest.fixture
def expected_decoder_mask_left():
    return (
        torch.tensor(
            [
                [0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 1],
                [0, 0, 0, 1, 1, 1],
                [0, 0, 0, 1, 1, 1],
                [0, 0, 0, 1, 1, 1],
                [0, 0, 0, 1, 1, 1],
            ]
        )
        .unsqueeze(0)
        .unsqueeze(0)
        .long()
    )


def test_pad_sequence_left():
    out = pad_sequence([1, 2, 3], pad_idx=5, max_len=6, pad="left")
    assert out == [5, 5, 5, 1, 2, 3]


def test_pad_sequence_right():
    out = pad_sequence([1, 2, 3], pad_idx=5, max_len=6, pad="right")
    assert out == [1, 2, 3, 5, 5, 5]


def test_get_padding_mask(padded_input):
    mask = get_padding_mask(padded_input, pad_idx=7)
    expected_mask = (
        torch.Tensor([[0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1], [0, 0, 0, 0, 1, 1]])
        .unsqueeze(1)
        .unsqueeze(1)
    )

    assert mask.ndim == 4
    assert torch.equal(mask, expected_mask)


def test_get_future_mask_left(padded_input):
    mask = get_future_mask(padded_input, padding="left")

    assert mask.ndim == 4
    assert mask.size(2) == mask.size(3) == len(padded_input[0])

    expected_mask = torch.flip(torch.tril(torch.ones((6, 6))), dims=[1]).unsqueeze(0).unsqueeze(0).long()

    assert torch.equal(expected_mask, mask)


def test_get_decoder_mask_left(padded_input, expected_decoder_mask_left):
    mask = get_decoder_mask(padded_input, pad_idx=7)
    assert torch.equal(mask[0], expected_decoder_mask_left[0])


def test_get_future_mask_right(padded_input):
    # Does not need to flip input, since future mask only need the dimensions, not the data
    mask = get_future_mask(padded_input, padding="right")
    expected_mask = torch.tril(torch.ones((6, 6))).unsqueeze(0).unsqueeze(0).long()
    assert torch.equal(expected_mask, mask)


def test_get_decoder_mask_right(padded_input, expected_decoder_mask_left):
    flip_input = torch.flip(padded_input, dims=[1])
    mask = get_decoder_mask(flip_input, pad_idx=7, padding="right")
    assert torch.equal(mask[0], torch.flip(expected_decoder_mask_left[0], dims=[2]))
