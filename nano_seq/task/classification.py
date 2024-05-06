from dataclasses import dataclass
import functools
import os

import torch
from torch.optim import Optimizer

from nano_seq.data import ClassificationCollator, ClassificationDataset, Dictionary
from nano_seq.data.utils import get_padding_mask
from nano_seq.logger import Logger
from nano_seq.model.classification import EncoderClassificationModel
from nano_seq.task.base import BaseTask
from nano_seq.utils.metrics import accuracy


class ClassificationTask(BaseTask):
    def __init__(self, args):
        self.args = args

    def prepare(self):
        # Load dictionary and data
        args = self.args
        dictionary = Dictionary.from_spm(self.args.spm_dict_path)
        train_iter, valid_iter = self._load_data(dictionary)
        clas_train, clas_valid = self._infer_n_class(train_iter, valid_iter)

        assert (
            clas_valid <= clas_train
        ), "Number of classes in valid set must not be greater than train set. Found {} > {}".format(
            clas_valid, clas_train
        )

        # Create model
        model = EncoderClassificationModel(
            clas_train,
            args.embed_dims,
            args.num_heads,
            args.encoder_layers,
            args.encoder_dropout,
            len(dictionary),
            dictionary.pad,
        )

        return train_iter, valid_iter, model

    def _infer_n_class(self, train_iter: ClassificationCollator, valid_iter: ClassificationCollator) -> tuple[int, int]:
        """Count the number of classes in train and valid set"""

        def max_label(sample):
            y = sample[1]
            return torch.max(y).item()

        def count_label(iterable):
            m = -1
            for batch in iter(iterable):
                m = max(m, max_label(batch))
            return int(m + 1)

        return count_label(train_iter), count_label(valid_iter)

    def _load_data(self, dictionary: Dictionary):
        args = self.args
        train_dataset = ClassificationDataset.from_text_file(
            os.path.join(args.train_path, "src.txt"), os.path.join(args.train_path, "tgt.txt"), dictionary
        )
        valid_dataset = ClassificationDataset.from_text_file(
            os.path.join(args.valid_path, "src.txt"), os.path.join(args.valid_path, "tgt.txt"), dictionary
        )

        train_iter = ClassificationCollator(
            train_dataset, args.batch_size, padding="left" if args.left_pad_src is True else "right"
        )
        valid_iter = ClassificationCollator(
            valid_dataset, args.batch_size, padding="left" if args.left_pad_src is True else "right"
        )

        return train_iter, valid_iter

    def train_step(
        self, train_loader, model: EncoderClassificationModel, optimizer: Optimizer, criterion, logger: Logger
    ):
        for batch_idx, data in enumerate(iter(train_loader)):
            x, y = data

            optimizer.zero_grad()
            enc_mask = get_padding_mask(x, model.pad_idx)
            logits, y_hat = model(x, enc_mask)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            batch_acc = accuracy(y, y_hat.detach())
            logger.write_train(batch_idx, loss=loss.item(), accuracy=batch_acc)

    def eval_step(self, valid_iter, model: EncoderClassificationModel, criterion, logger: Logger):
        for batch_idx, data in enumerate(iter(valid_iter)):
            x, y = data

            with torch.no_grad():
                enc_mask = get_padding_mask(x, model.pad_idx)
                logits, y_hat = model(x, enc_mask)
                loss = criterion(logits, y)

            batch_acc = accuracy(y, y_hat)

            logger.write_eval(batch_idx, loss=loss.item(), accuracy=batch_acc)

    def inference_step(self, infer_iter, model):
        pass


@dataclass
class Config:
    embed_dims: int
    batch_size: int
    num_heads: int
    encoder_layers: int
    encoder_dropout: float
    spm_dict_path: str
    left_pad_src: bool
    train_path: str
    valid_path: str
