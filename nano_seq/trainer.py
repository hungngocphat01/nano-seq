import torch
from torch import nn

from nano_seq.logger import Logger


def train(
    epochs: int,
    model: nn.Module,
    train_iter,
    valid_iter,
    optimizer: torch.optim.Optimizer,
    criterion,
    train_func,
    valid_func,
    logger: Logger
):
    for i in range(epochs):
        logger.epoch = i + 1

        model.train()
        train_func(train_iter, model, optimizer, criterion, logger)

        model.eval()
        valid_func(valid_iter, model, logger)
