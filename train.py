import argparse
import yaml

import torch
import wandb

from nano_seq.trainer import Trainer
from nano_seq.task.translation import TranslationConfig, TranslationTask
from nano_seq.utils.lr_scheduler import TransformerLRScheduler
from nano_seq.utils.logger import Logger, WandBLogHandler
from nano_seq.module.translation_loss import TranslationLoss


def assign_default_params(config: dict):
    defaults = {
        "adam_betas": (0.9, 0.98),
        "adam_eps": "1e-9",
        "lr_multiplier": 1,
        "warm_up_steps": 12000,
        "label_smoothing": 0.1,
        "chkpt_save_every": 1,
        "chkpt_keep_last": None,
        "wandb_project": None,
    }

    for k, v in defaults.items():
        if k not in config:
            config = {**config, k: v}

    return config


def main(args):
    # Read config
    with open(args.config, "rt", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
        train_cfg = assign_default_params(config_dict["training"])
        cfg = TranslationConfig(**config_dict["task"])

    # Create task, load data and model
    task = TranslationTask(cfg)
    train_iter, eval_iter, model = task.prepare()

    optimizer = torch.optim.Adam(params=model.parameters(), betas=train_cfg["adam_betas"], eps=float(train_cfg["adam_eps"]))
    lr_scheduler = TransformerLRScheduler(
        optimizer, cfg.embed_dims, train_cfg["lr_multiplier"], train_cfg["warm_up_steps"]
    )
    criterion = TranslationLoss(cfg.pad_idx, train_cfg["label_smoothing"])

    # Wandb support
    if (wandb_project := train_cfg["wandb_project"]) is not None:
        wandb_run = wandb.init(project=wandb_project)
        handlers = [WandBLogHandler("wandb", wandb_run)]  # type: ignore
    else:
        handlers = []

    logger = Logger(handlers)  # type: ignore

    # Start training
    trainer = Trainer(
        train_iter=train_iter,
        eval_iter=eval_iter,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        criterion=criterion,
        task=task,
        model=model,
        logger=logger,
        checkpoint_path=args.chkpt_path,
    )

    if args.chkpt_load:
        trainer.load_checkpoint(args.chkpt_load)

    trainer.start_train(train_cfg["epochs"], train_cfg["chkpt_save_every"], train_cfg["chkpt_keep_last"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, help="YAML config file")
    parser.add_argument("--chkpt-path", type=str, help="Directory to save the checkpoint", required=True)
    parser.add_argument("--chkpt-load", type=str, help="Checkpoint to continue training from", required=False)
