import pytest
from nano_seq.data.collator import collate_batch

E = 100
S = 101
P = 99

@pytest.fixture
def non_pad_input():
    return [
        [1, 2, 3],
        [1, 2, 3, 4],
        [1, 2, 3, 4, 5, 6],
        [1]
    ]


# def test_collate_left_pad(non_pad_input):
#     batch = collate_batch(non_pad_input, eos=E, sos=S, pad=P, padding="left")

#     assert batch == [
#         [P, P, P, 1, 2, 3, E],
#         [P, P, 1, 2, 3, 4, E],
#         [1, 2, 3, 4, 5, 6, E],
#         [P, P, P, P, P, 1, E]
#     ]


@pytest.mark.parametrize(
    "prepend_sos,expected_output", [
        (
            False,
            [[1, 2, 3, E, P, P, P],
             [1, 2, 3, 4, E, P, P],
             [1, 2, 3, 4, 5, 6, E],
             [1, E, P, P, P, P, P]]
        ),
        (
            True,
            [[S, 1, 2, 3, E, P, P, P],
             [S, 1, 2, 3, 4, E, P, P],
             [S, 1, 2, 3, 4, 5, 6, E],
             [S, 1, E, P, P, P, P, P]]
        )
    ]
)
def test_collate_right_pad(non_pad_input, prepend_sos, expected_output):

    batch = collate_batch(non_pad_input, eos=E, sos=S, pad=P, padding="right", append_sos=prepend_sos)

    assert batch == expected_output
