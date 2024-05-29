from dataclasses import dataclass, field
import os

import torch

from nano_seq.data import ClassificationCollator, ClassificationDataset, Dictionary
from nano_seq.data.const import PAD
from nano_seq.data.utils import get_padding_mask
from nano_seq.model.classification import EncoderClassificationModel, ClassificationNetInput
from nano_seq.task.base import BaseTask
from nano_seq.utils.metrics import accuracy

@dataclass
class ClassificationConfig:
    embed_dims: int
    num_heads: int
    ffn_project_dims: int
    encoder_layers: int
    encoder_dropout: float

    train_path: str
    valid_path: str
    spm_dict_path: str
    batch_size: int
    left_pad_src: bool = field(default=False)
    pad_idx: int = field(default=PAD)


class ClassificationTask(BaseTask):
    def __init__(self, cfg: ClassificationConfig):
        self.cfg = cfg

    def prepare(self):
        # Load dictionary and data
        cfg = self.cfg
        dictionary = Dictionary.from_spm(self.cfg.spm_dict_path)
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
            cfg.embed_dims,
            cfg.num_heads,
            cfg.ffn_project_dims,
            cfg.encoder_layers,
            cfg.encoder_dropout,
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
        cfg = self.cfg
        train_dataset = ClassificationDataset.from_text_file(
            os.path.join(cfg.train_path, "src.txt"), os.path.join(cfg.train_path, "tgt.txt"), dictionary
        )
        valid_dataset = ClassificationDataset.from_text_file(
            os.path.join(cfg.valid_path, "src.txt"), os.path.join(cfg.valid_path, "tgt.txt"), dictionary
        )

        train_iter = ClassificationCollator(
            train_dataset, cfg.batch_size, padding="left" if cfg.left_pad_src is True else "right"
        )
        valid_iter = ClassificationCollator(
            valid_dataset, cfg.batch_size, padding="left" if cfg.left_pad_src is True else "right"
        )

        return train_iter, valid_iter

    def get_net_input(self, batch) -> ClassificationNetInput:
        x, _ = batch
        return ClassificationNetInput(x=x, mask=get_padding_mask(x, self.cfg.pad_idx))

    def train_step(
        self,
        net_input: ClassificationNetInput,
        label: torch.Tensor,
        model: EncoderClassificationModel,
        optimizer: torch.optim.Optimizer,
        criterion,
    ) -> dict:
        """
        Function to call for one training batch

        Args
        ----
        net_input:
            neural network input, defined by batch iterator
        label:
            1d tensor of shape [bsz]
        model:
            the neural network module

        Returns
        -------
        dict
            logging metrics
        """
        model.train()

        optimizer.zero_grad()
        logits, y_hat = model(**net_input.asdict())
        loss = criterion(logits, label)
        loss.backward()

        batch_acc = accuracy(label, y_hat.detach())

        return {"loss": loss.detach().item(), "accuracy": batch_acc}

    def eval_step(
        self, net_input: ClassificationNetInput, label: torch.Tensor, model: EncoderClassificationModel, criterion
    ) -> dict:

        with torch.no_grad():
            logits, y_hat = model(**net_input.asdict())
            loss = criterion(logits, label)

        batch_acc = accuracy(label, y_hat)

        return {"loss": loss.item(), "accuracy": batch_acc}

    def inference_step(self, infer_iter, model):
        pass
