import argparse
import os
import sys

import torch
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader
from tqdm import tqdm

from corpus.corrector_dataset import CorrectorDataset
from model.neural_corrector import NeuralCorrector
from util.edit_distance import normalized_edit_distance


def evaluate(data_dir: str, model_path: str, num_cpus: int, device: torch.device, test: bool = False):
    seed_everything(42, workers=True)  # reproducibility

    # prep model
    model = NeuralCorrector.load_from_checkpoint(
        model_path,  # most hyperparameters are saved here, but we can override a few
        cpus=num_cpus,
        show_warnings=False
    )
    model.eval()
    model = model.to(device)
    # prep data
    if test:
        dataset = CorrectorDataset(data_dir, split="test")
    else:
        dataset = CorrectorDataset(data_dir, split="validation")
    data_loader = DataLoader(dataset, batch_size=model.batch_size, num_workers=num_cpus)
    # make a place to save results
    scores_out = list()
    # run data through until interrupted or completed
    try:
        with torch.no_grad():
            with tqdm(total=len(dataset)) as progress:
                for x_batch, y_batch in data_loader:
                    y_hat_batch = model.correct(x_batch)
                    for y, y_hat in zip(y_batch, y_hat_batch):
                        if y == y_hat:  # shortcut
                            scores_out.append(0)
                        elif len(y) == 0:
                            print("y was length 0; skipping", file=sys.stderr)
                        else:
                            scores_out.append(normalized_edit_distance(y_hat, y, banded=False))
                        progress.update()
    except KeyboardInterrupt:
        print("\nKEYBOARD INTERRUPT - terminating evaluation\n")
    n = len(scores_out)
    avg_distance = sum(scores_out) / n
    percent_perfect = scores_out.count(0) / n
    print(f"Average edit distance: {avg_distance:.2f}")
    print(f"Percent perfect: {100 * percent_perfect:.2f}%")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("corpus_dir", type=str, help="File path to the directory containing the corpus to train on.")
    arg_parser.add_argument("--model-path", type=str, required=True, help="File path to the checkpoint file to load the model from")
    arg_parser.add_argument("--cpu-limit", type=int, default=None, help="Max number of CPU processors to use.")
    arg_parser.add_argument("--cuda", type=int, default=None, help="Index of the CUDA device (GPU) to use.")
    arg_parser.add_argument("--test", action="store_true", help="Uses the final test set, not validation.")
    args = arg_parser.parse_args()
    corpus_dir = args.corpus_dir
    model_path_ = args.model_path
    cpu_limit_ = args.cpu_limit
    cuda_index = args.cuda
    test_ = args.test

    if cpu_limit_ is None:  # use all we've got
        cpus_ = os.cpu_count()
    else:
        cpus_ = min(max(cpu_limit_, 1), os.cpu_count())  # clip the provided number between 1 and os.cpu_count(), inclusive
    if cpus_ == 1:
        cpus_ = 0  # DataLoader expects 0 if we're not doing extra workers

    if cuda_index is None:
        device_ = torch.device("cpu")
    elif cuda_index >= (cuda_count := torch.cuda.device_count()) or cuda_index < 0:
        print(f"WARNING: provided cuda index '{cuda_index}' is not valid (available count = {cuda_count}); defaulting to CPU", file=sys.stderr)
        device_ = torch.device("cpu")
    else:
        device_ = torch.device(f"cuda:{cuda_index}")

    evaluate(corpus_dir, model_path_, cpus_, device_, test_)
