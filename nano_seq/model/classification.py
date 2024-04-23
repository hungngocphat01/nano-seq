from torch import Tensor, nn
from torch.nn import functional as F

from nano_seq.data.const import PAD
from nano_seq.module.transformer import TransformerEncoder


class EncoderClassificationModel(nn.Module):
    def __init__(
        self, n_class: int, d_model: int, n_heads: int, layers: int, dropout: float, vocab_size: int, pad_idx: int = PAD
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.n_class = n_class
        self.encoder = TransformerEncoder(d_model, n_heads, layers, vocab_size, dropout=dropout, padding_idx=pad_idx)
        self.linear = nn.Linear(d_model, n_class)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: Tensor, mask: Tensor):
        """
        Args
        ----
        x: tensor
            shape [bsz, max_len]
        mask: tensor
            shape [bsz, 1, 1, max_len]

        Return
        ------
        tuple[tensor, tensor]
            - tensor of shape [bsz, n_class] where each column is logits
            - tensor of shape [bsz,] where each position is the predicted class index
        """

        x = self.encoder(x, mask)

        # masked average along each dimension
        # encoder output: [bsz, len, d] -> [bsz, d]

        # mask [bsz, 1, 1, len] -> [bsz, len, 1]
        out_mask = mask.squeeze(1).transpose(-1, -2)
        x_masked = x.masked_fill(out_mask == 0, 0)
        masked_sum = x_masked.sum(dim=1)
        masked_denom = out_mask.sum(dim=1)

        # [bsz, d] -> [bsz, n_class]
        x = masked_sum / masked_denom
        x = self.linear(x)

        y_hat = F.softmax(x, dim=1)
        y_hat_class = y_hat.argmax(dim=1)

        return x, y_hat_class
