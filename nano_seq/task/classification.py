from torch.optim import Optimizer

from nano_seq.data.utils import get_encoder_mask
from nano_seq.model.classification import EncoderClassificationModel


class ClassificationTask:
    def __init__(self):
        self.moving_avg = {"loss": 0.0, "accuracy": 0.0}
        # it's just cummulative sum of non-averaged metrics
        self.cum_metrics = {}

    def _update_moving_avg(self, step: int, **kwargs):
        for key, value in kwargs.items():
            self.cum_metrics[key] += float(value)
            self.moving_avg[key] = self.cum_metrics[key] / (step + 1)

    def train_step(self, train_loader, model: EncoderClassificationModel, optimizer: Optimizer, criterion, epoch: int):
        batches_per_epoch = len(train_loader) // train_loader.bsz

        for batch_idx, data in iter(train_loader):
            x, y = data
            step = epoch * batches_per_epoch + batch_idx

            optimizer.zero_grad()
            enc_mask = get_encoder_mask(x, model.pad_idx)
            logits, y_hat = model(x, enc_mask)
            loss = criterion(logits, y)
            loss.backward()

            batch_acc = self.accuracy(y, y_hat.detach())
            self._update_moving_avg(step, loss=loss.item(), accuracy=batch_acc)

    def accuracy(self, y, y_hat):
        return (y.long() == y_hat.long()).float().mean().item()

    def eval_step(self, valid_iter, model, optimizer, criterion):
        pass

    def inference_step(self, infer_iter, model):
        pass
