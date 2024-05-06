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


@pytest.mark.parametrize("pad,expected", [("left", [5, 5, 5, 1, 2, 3]), ("right", [1, 2, 3, 5, 5, 5])])
def test_pad_sequence(pad: str, expected: list):
    out = pad_sequence([1, 2, 3], pad_idx=5, max_len=6, pad=pad)
    assert out == expected


def test_get_padding_mask(padded_input):
    mask = get_padding_mask(padded_input, pad_idx=7)
    expected_mask = (
        torch.Tensor([[0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1], [0, 0, 0, 0, 1, 1]])
        .unsqueeze(1)
        .unsqueeze(1)
    )

    assert mask.ndim == 4
    assert torch.equal(mask, expected_mask)

# Future mask (and thus decoder mask) for left padding is broken. Do not use for now

@pytest.mark.parametrize("padding,expected_mask", [
    ("right", torch.tril(torch.ones(6, 6))),
])
def test_get_future_mask(padding, padded_input, expected_mask):
    if padding == "right":
        padded_input = torch.flip(padded_input, dims=[1])

    mask = get_future_mask(padded_input, padding=padding)

    assert mask.ndim == 4
    assert mask.size(2) == mask.size(3) == len(padded_input[0])

    expected_mask = expected_mask.unsqueeze(0).unsqueeze(0).long()
    assert torch.equal(expected_mask, mask)


@pytest.mark.parametrize("padding", ["right"])
def test_get_decoder_mask(padding, padded_input, expected_decoder_mask_left):
    expected_mask = expected_decoder_mask_left[0]

    if padding == "right":
        padded_input = torch.flip(padded_input, dims=[1])
        expected_mask = torch.flip(expected_mask, dims=[2])

    mask = get_decoder_mask(padded_input, pad_idx=7, padding=padding)
    assert torch.equal(mask[0], expected_mask)
