import argparse
import os
import multiprocessing as mp
from typing import Tuple

from tqdm import tqdm
from math import ceil

from corpus.corrector_dataset import CorrectorDataset
from util.edit_distance import normalized_edit_distance


def norm_edit_distance_mp(pair: Tuple[str, str]) -> float:
    return normalized_edit_distance(*pair)


def evaluate(dataset: CorrectorDataset, n_cpus: int = 1):
    n = len(dataset)
    all_scores = list()
    try:
        if n_cpus <= 1:  # mp not worth it
            for i in tqdm(range(n)):
                text_messy, text_clean = dataset[i]
                score = normalized_edit_distance(text_messy, text_clean)
                all_scores.append(score)
        else:
            with mp.Pool(n_cpus) as pool:
                generator = (dataset[i] for i in range(n))
                with tqdm(total=n) as progress:
                    for score in pool.imap_unordered(
                            norm_edit_distance_mp,
                            generator,
                            chunksize=50
                    ):
                        all_scores.append(score)
                        progress.update()
    except KeyboardInterrupt:
        print("KEYBOARD INTERRUPT - terminating evaluation")
    # give results
    n = len(all_scores)  # maybe we didn't do all of them
    avg_score = sum(all_scores) / n
    print(f"Number of sentences evaluated: {n}")
    print(f"Average edit distance: {avg_score:.2f}")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("corpus_dir", type=str, help="File path to the directory containing the corpus to train on.")
    arg_parser.add_argument("--cpu-limit", type=int, default=None, help="Max number of CPU processors to use.")
    args = arg_parser.parse_args()
    corpus_dir = args.corpus_dir
    cpu_limit_ = args.cpu_limit

    if cpu_limit_ is None:  # use all we've got
        cpus = max(os.cpu_count() - 1, 1)
    else:
        cpus = min(max(cpu_limit_, 1), os.cpu_count())  # clip the provided number between 1 and os.cpu_count(), inclusive

    dataset_test = CorrectorDataset(corpus_dir, split="test")
    evaluate(dataset_test, cpus)
