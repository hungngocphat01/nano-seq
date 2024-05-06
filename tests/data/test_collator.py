import pytest
from nano_seq.data.collator import collate_batch


@pytest.fixture
def non_pad_input():
    return [
        [1, 2, 3],
        [1, 2, 3, 4],
        [1, 2, 3, 4, 5, 6],
        [1]
    ]


def test_collate_left_pad(non_pad_input):
    E = 100
    S = 101
    P = 99
    batch = collate_batch(non_pad_input, eos_idx=E, sos_idx=S, pad_idx=P, pad="left")

    assert batch == [
        [P, P, P, S, 1, 2, 3, E],
        [P, P, S, 1, 2, 3, 4, E],
        [S, 1, 2, 3, 4, 5, 6, E],
        [P, P, P, P, P, S, 1, E]
    ]


def test_collate_right_pad(non_pad_input):
    E = 100
    S = 101
    P = 99
    batch = collate_batch(non_pad_input, eos_idx=E, sos_idx=S, pad_idx=P, pad="right")

    assert batch == [
        [S, 1, 2, 3, E, P, P, P],
        [S, 1, 2, 3, 4, E, P, P],
        [S, 1, 2, 3, 4, 5, 6, E],
        [S, 1, E, P, P, P, P, P]
    ]
