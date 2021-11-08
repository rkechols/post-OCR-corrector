import argparse
import os
import sys
from typing import Dict

import torch
from ray import tune
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from corpus import ALL_CHARS_FILE_NAME, DEFAULT_ENCODING
from corpus.corrector_dataset import CorrectorDataset
from model.neural_corrector import NeuralCorrector
from util.data_functions import collate_sequences


def train_mini(config: Dict):
    model = NeuralCorrector(**config)
    pass


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("corpus_dir", type=str, help="File path to the directory containing the corpus to train on.")
    arg_parser.add_argument("--model-dir", type=str, required=True, help="File path to the directory where to save model info.")
    arg_parser.add_argument("--cuda", type=int, default=None, help="Index of the CUDA device (GPU) to use.")
    arg_parser.add_argument("--cpu-limit", type=int, default=None, help="Max number of CPU processors to use.")
    args = arg_parser.parse_args()
    corpus_dir = args.corpus_dir
    model_dir = args.model_dir
    cuda_index = args.cuda
    cpu_limit = args.cpu_limit

    os.makedirs(model_dir, exist_ok=True)

    if cuda_index is None:
        device_ = torch.device("cpu")
    elif cuda_index >= (cuda_count := torch.cuda.device_count()) or cuda_index < 0:
        print(f"ERROR: provided cuda index '{cuda_index}' is not valid (available count = {cuda_count}); defaulting to CPU", file=sys.stderr)
        device_ = torch.device("cpu")
    else:
        device_ = torch.device(f"cuda:{cuda_index}")

    if cpu_limit is None:  # use all we've got
        cpus = max(os.cpu_count(), 1)
    else:
        cpus = min(max(cpu_limit, 1), os.cpu_count())  # clip the provided number between 1 and os.cpu_count(), inclusive
    if cpus == 1:
        cpus = 0  # DataLoader expects 0 if we're not doing extra workers

    dataset_train = CorrectorDataset(corpus_dir, split="train", tensors_out=True)
    analysis = tune.run(
        train_mini,
        config={

        }
    )
