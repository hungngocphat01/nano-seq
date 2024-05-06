import pytest
import torch

from nano_seq.module.transformer import MultiheadAttention


BSZ = 2
MAX_LEN_SRC = 7
MAX_LEN_TGT = 6
D_MODEL = 8
N_HEAD = 4


class TestEncoderAttention:
    @pytest.fixture
    def attention(self):
        return MultiheadAttention(D_MODEL, N_HEAD).eval()

    @pytest.fixture
    def encoder_mask(self):
        return torch.tensor([
            [1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0, 0, 0]
        ]).unsqueeze(1).unsqueeze(1).long()


    @pytest.mark.parametrize("query_length", [MAX_LEN_SRC, MAX_LEN_TGT])
    def test_encoder_attention(self, attention, encoder_mask, query_length):
        # Test function for both encoder self attention and cross attention
        q = torch.rand(BSZ, query_length, D_MODEL)
        k = v = torch.rand(BSZ, MAX_LEN_SRC, D_MODEL)

        # Assert output size
        out, raw_attn = attention(q, k, v, encoder_mask)
        assert out.shape == torch.Size([BSZ, query_length, D_MODEL])
        assert raw_attn.shape == torch.Size([BSZ, N_HEAD, query_length, MAX_LEN_SRC])

        for i in range(N_HEAD):
            # Padded index must have zero attention
            assert torch.count_nonzero(raw_attn[0, i, :, -2:]) == 0
            assert torch.count_nonzero(raw_attn[1, i, :, -3:]) == 0


class TestDecoderAttention:
    @pytest.fixture
    def attention(self):
        return MultiheadAttention(D_MODEL, N_HEAD).eval()

    @pytest.fixture
    def decoder_mask(self):
        return torch.tensor([
            [[1, 0, 0, 0, 0, 0],    # last 2 tokens are padding
             [1, 1, 0, 0, 0, 0],
             [1, 1, 1, 0, 0, 0],
             [1, 1, 1, 1, 0, 0],
             [1, 1, 1, 1, 0, 0],
             [1, 1, 1, 1, 0, 0]],
            [[1, 0, 0, 0, 0, 0],    # last 3 tokens are padding
             [1, 1, 0, 0, 0, 0],
             [1, 1, 1, 0, 0, 0],
             [1, 1, 1, 0, 0, 0],
             [1, 1, 1, 0, 0, 0],
             [1, 1, 1, 0, 0, 0]],
        ]).unsqueeze(1)

    def test_decoder_attention(self, attention, decoder_mask):
        q = k = v = torch.rand(BSZ, MAX_LEN_TGT, D_MODEL)

        # Assert output size
        out, raw_attn = attention(q, k, v, decoder_mask)
        assert out.shape == torch.Size([BSZ, MAX_LEN_TGT, D_MODEL])
        assert raw_attn.shape == torch.Size([BSZ, N_HEAD, MAX_LEN_TGT, MAX_LEN_TGT])

        for i in range(N_HEAD):
            assert torch.equal(
                torch.ceil(raw_attn[0, i]).long(), decoder_mask[0, 0],
            )
            assert torch.equal(
                torch.ceil(raw_attn[1, i]).long(), decoder_mask[1, 0],
            )
