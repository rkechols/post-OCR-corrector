import argparse
import json
import os
from math import ceil

import pytorch_lightning as pl
import ray
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from ray import tune
from ray.tune import CLIReporter
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.schedulers import ASHAScheduler

from corpus import ALL_CHARS_FILE_NAME
from model import HYPERPARAMS_FILE_NAME
from model.neural_corrector import NeuralCorrector
from util import DEFAULT_ENCODING


def train_mini(config, data_dir: str, alphabet_size: int,
               num_cpus: float = 1, num_gpus: float = 0,
               checkpoint_dir: str = None):
    # model = NeuralCorrector(data_dir, alphabet_size, ceil(num_cpus), **config)
    model = NeuralCorrector(data_dir, alphabet_size, 0, **config)
    trainer = pl.Trainer(
        max_epochs=1,
        val_check_interval=100,
        gpus=ceil(num_gpus),  # if fractional GPUs passed in, convert to int
        logger=TensorBoardLogger(save_dir=tune.get_trial_dir(), name="", version="."),
        enable_progress_bar=False,
        callbacks=[
            TuneReportCallback({"loss": "ptl/val_loss"}, on="validation_end")
        ],
    )
    trainer.fit(model)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("corpus_dir", type=str, help="File path to the directory containing the corpus to train on.")
    arg_parser.add_argument("--model-dir", type=str, required=True, help="File path to the directory where to save model info.")
    arg_parser.add_argument("--gpus", type=int, default=torch.cuda.device_count(), help="Max number of GPUs to use (defaults to no limit).")
    arg_parser.add_argument("--cpus", type=int, default=os.cpu_count(), help="Max number of CPU processors to use (defaults to no limit).")
    arg_parser.add_argument("--n-concurrent", type=int, default=4, help="Number of trials to run simultaneously.")
    args = arg_parser.parse_args()
    corpus_dir = args.corpus_dir
    model_dir = args.model_dir
    gpus_ = args.gpus
    cpus_ = args.cpus
    n_concurrent = args.n_concurrent

    gpus_per_trial = gpus_ / n_concurrent
    cpus_per_trial = cpus_ / n_concurrent

    os.makedirs(model_dir, exist_ok=True)

    with open(os.path.join(corpus_dir, ALL_CHARS_FILE_NAME), "r", encoding=DEFAULT_ENCODING) as chars_file:
        all_chars = chars_file.read().replace("\n", "")
    alphabet_size_ = len(all_chars)

    ray.init(num_cpus=cpus_, num_gpus=gpus_)
    print(f"ray cluster: {ray.cluster_resources()}")

    search_space = {
        "d_model": tune.choice([128, 256, 512]),
        "n_head": tune.choice([4, 8]),
        "n_encoder_layers": tune.choice(list(range(3, 7))),
        "n_decoder_layers": tune.choice(list(range(3, 7))),
        "d_linear": tune.choice([256, 512, 1024, 2048]),
        "dropout": tune.choice([0.0, 0.05, 0.1, 0.3]),
        "layer_norm_eps": tune.loguniform(1e-6, 1e-4),
        "label_smoothing": tune.choice([0.0, 0.05, 0.1]),
    }

    analysis = tune.run(
        tune.with_parameters(train_mini, data_dir=corpus_dir, alphabet_size=alphabet_size_, num_cpus=cpus_per_trial, num_gpus=gpus_per_trial),
        resources_per_trial={
            "cpu": cpus_per_trial,
            "gpu": gpus_per_trial
        },
        metric="loss",
        mode="min",
        config=search_space,
        scheduler=ASHAScheduler(
            time_attr="training_iteration",
            max_t=1000,
            grace_period=25,
            reduction_factor=2
        ),
        progress_reporter=CLIReporter(
            parameter_columns=list(search_space.keys()),
            metric_columns=["loss", "training_iteration"]
        ),
        name="hparams_tune_asha"
    )

    best_config = analysis.best_config  # get best trial's hyperparameters
    print("BEST CONFIGURATION FOUND:")
    for k, v in best_config.items():
        print(f"  {k} = {v}")
    hparams_file_path = os.path.join(model_dir, HYPERPARAMS_FILE_NAME)
    with open(hparams_file_path, "w", encoding=DEFAULT_ENCODING) as hparams_file:
        print(json.dumps(best_config, indent=2), file=hparams_file)
    print(f"\nConfig saved to {hparams_file_path}")
