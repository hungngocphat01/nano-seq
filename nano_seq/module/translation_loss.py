import torch
from torch import nn
from nano_seq.data.const import PAD


class TranslationLoss(nn.Module):
    def __init__(self, padding_idx: int = PAD, label_smoothing: float = 0.1):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=padding_idx, label_smoothing=label_smoothing)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """
        Compute the machine translation loss for one batch using multi-class cross-entropy loss

        Args
        ----
        y_true: tensor
            of shape [bsz, max_len, ]
        """
        y_true = y_true.ravel()
        y_pred = y_pred.reshape(-1, y_pred.shape[-1])

        return self.cross_entropy(y_pred, y_true)
