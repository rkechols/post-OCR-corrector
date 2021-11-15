import argparse
import json
import os
import sys
from typing import Dict

import pytorch_lightning as pl
import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from model import CHECKPOINT_DIR_NAME, HYPERPARAMS_FILE_NAME, TENSORBOARD_DIR_NAME
from model.neural_corrector import NeuralCorrector
from util import DEFAULT_ENCODING


seed_everything(42, workers=True)  # reproducibility


CUDA_COUNT = torch.cuda.device_count()


def load_hparams(model_dir: str) -> Dict:
    with open(os.path.join(model_dir, HYPERPARAMS_FILE_NAME), "r", encoding=DEFAULT_ENCODING) as hparams_file:
        hparams = json.loads(hparams_file.read())
    return hparams


def set_batch_size(model: NeuralCorrector, num_gpus: int):
    print("finding batch size...")
    trainer = pl.Trainer(
        auto_scale_batch_size="binsearch",
        gpus=num_gpus,
        auto_select_gpus=True
    )
    trainer.tune(model)
    print(f"\nselected batch size: {model.batch_size}\n")


def set_learning_rate(model: NeuralCorrector, num_gpus: int):
    print("finding learning rate...")
    trainer = pl.Trainer(
        auto_lr_find=True,
        gpus=num_gpus,
        auto_select_gpus=True
    )
    trainer.tune(model)
    print(f"\nselected learning rate: {model.lr}\n")


def train(data_dir: str, model_dir: str, num_cpus: int, num_gpus: int, checkpoint: str = None):
    hparams = load_hparams(model_dir)
    model = NeuralCorrector(data_dir, num_cpus, **hparams)

    if "batch_size" not in hparams:
        set_batch_size(model, num_gpus)
    if "lr" not in hparams:
        set_learning_rate(model, num_gpus)

    log_dir = os.path.abspath(os.path.join(model_dir, TENSORBOARD_DIR_NAME))
    checkpoint_dir = os.path.abspath(os.path.join(model_dir, CHECKPOINT_DIR_NAME))

    trainer = pl.Trainer(
        logger=TensorBoardLogger(save_dir=log_dir, name=""),
        checkpoint_callback=True,  # we use the custom one below
        callbacks=[
            ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename="neural_corrector-epoch{epoch}-step{step}-val_loss{ptl/val_loss:.2f}",
                monitor="ptl/val_loss",
                save_top_k=1,
                mode="min",
                auto_insert_metric_name=False
            ),
            EarlyStopping(
                monitor="ptl/val_loss",
                min_delta=0.01,
                patience=3,  # if we have 3 validations in a row that improve loss by less than 0.01, stop training
                mode="min",
                stopping_threshold=0.0,
                divergence_threshold=6.0  # random answers give about 7.2, and a small amount of training quickly gets it to around 1.5
            )
        ],
        gradient_clip_val=hparams["gradient_clip_val"],
        gpus=num_gpus,
        auto_select_gpus=True,
        max_time="00:23:59:59",  # "DD:HH:MM:SS" format
        limit_val_batches=50,
        val_check_interval=500,
        num_sanity_val_steps=1,
        resume_from_checkpoint=checkpoint,
        deterministic=True,  # reproducibility
        terminate_on_nan=True,
        stochastic_weight_avg=True
    )
    trainer.fit(model)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("corpus_dir", type=str, help="File path to the directory containing the corpus to train on.")
    arg_parser.add_argument("--model-dir", type=str, required=True, help="File path to the directory where to save model info.")
    arg_parser.add_argument("--gpus", type=int, default=CUDA_COUNT, help="Max number of GPUs to use (defaults to no limit).")
    arg_parser.add_argument("--cpu-limit", type=int, default=None, help="Max number of CPU processors to use.")
    arg_parser.add_argument("--checkpoint", type=str, default=None, help="Path to a checkpoint to resume from")
    args = arg_parser.parse_args()
    corpus_dir = args.corpus_dir
    model_dir_ = args.model_dir
    gpus_ = args.gpus
    cpu_limit_ = args.cpu_limit
    checkpoint_ = args.checkpoint

    if gpus_ > CUDA_COUNT:
        print(f"WARNING: provided GPU count '{gpus_}' is greater than available count '{CUDA_COUNT}'; clipping value", file=sys.stderr)
        gpus_ = CUDA_COUNT
    gpus_ = max(gpus_, 0)  # make sure we don't get a negative number...?

    if cpu_limit_ is None:  # use all we've got
        cpus_ = os.cpu_count()
    else:
        cpus_ = min(max(cpu_limit_, 1), os.cpu_count())  # clip the provided number between 1 and os.cpu_count(), inclusive
    if cpus_ == 1:
        cpus_ = 0  # DataLoader expects 0 if we're not doing extra workers

    train(corpus_dir, model_dir_, cpus_, gpus_, checkpoint_)
