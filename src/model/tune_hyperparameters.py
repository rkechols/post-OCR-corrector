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

from model import HYPERPARAMS_FILE_NAME
from model.neural_corrector import NeuralCorrector
from util import DEFAULT_ENCODING
from util.data_functions import get_alphabet


def train_mini(config, data_dir: str,
               num_cpus: float = 1, num_gpus: float = 0,
               checkpoint_dir: str = None):
    model = NeuralCorrector(data_dir, ceil(num_cpus), **config)
    trainer = pl.Trainer(
        logger=TensorBoardLogger(save_dir=tune.get_trial_dir(), name="", version="."),
        callbacks=[
            TuneReportCallback({"loss": "ptl/val_loss"}, on="validation_end")
        ],
        gradient_clip_val=0.5,
        gpus=ceil(num_gpus),  # if fractional GPUs passed in, convert to int
        progress_bar_refresh_rate=0,
        max_epochs=1,
        limit_val_batches=20,
        val_check_interval=200,
        num_sanity_val_steps=0,
        terminate_on_nan=True
    )
    trainer.fit(model)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("corpus_dir", type=str, help="File path to the directory containing the corpus to train on.")
    arg_parser.add_argument("--model-dir", type=str, required=True, help="File path to the directory where to save model info.")
    arg_parser.add_argument("--gpus", type=int, default=torch.cuda.device_count(), help="Max number of GPUs to use (defaults to no limit).")
    arg_parser.add_argument("--cpus", type=int, default=os.cpu_count(), help="Max number of CPU processors to use (defaults to no limit).")
    arg_parser.add_argument("--n-concurrent", type=int, default=4, help="Number of trials to run simultaneously.")
    arg_parser.add_argument("--n-total", type=int, default=20, help="Total number of trials to run.")
    args = arg_parser.parse_args()
    corpus_dir = args.corpus_dir
    model_dir = args.model_dir
    gpus_ = args.gpus
    cpus_ = args.cpus
    n_concurrent = args.n_concurrent
    n_total = args.n_total

    gpus_per_trial = gpus_ / n_concurrent
    if gpus_per_trial > 1:
        gpus_per_trial = int(gpus_per_trial)
    cpus_per_trial = cpus_ / n_concurrent
    if cpus_per_trial > 1:
        cpus_per_trial = int(cpus_per_trial)
    os.makedirs(model_dir, exist_ok=True)

    all_chars = get_alphabet(corpus_dir)
    alphabet_size_ = len(all_chars)

    ray.init(num_cpus=cpus_, num_gpus=gpus_)
    print(f"ray cluster: {ray.cluster_resources()}")

    search_space = {
        "d_model": tune.choice([128, 256, 512]),
        "n_head": tune.choice([4, 8]),
        "n_layers": tune.choice(list(range(3, 7))),
        "d_linear": tune.choice([256, 512, 1024, 2048]),
        "dropout": tune.choice([0.0, 0.05, 0.1, 0.3]),
        "layer_norm_eps": tune.loguniform(1e-6, 1e-4),
        "label_smoothing": tune.choice([0.0, 0.05, 0.1]),
    }

    analysis = tune.run(
        tune.with_parameters(
            train_mini,
            data_dir=os.path.abspath(corpus_dir),  # needs full path, apparently
            num_cpus=cpus_per_trial,
            num_gpus=gpus_per_trial
        ),
        resources_per_trial={
            "cpu": cpus_per_trial,
            "gpu": gpus_per_trial
        },
        metric="loss",
        mode="min",
        config=search_space,
        num_samples=n_total,
        scheduler=ASHAScheduler(
            time_attr="training_iteration",
            max_t=10_000,
            grace_period=200,
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
