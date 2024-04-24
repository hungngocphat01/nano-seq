from torch import Tensor


def accuracy(y_true: Tensor, y_pred: Tensor):
    return (y_true.long() == y_pred.long()).float().mean().item()
