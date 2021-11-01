import argparse
import csv
import json
import math
import os
import re
import string
import sys
from typing import Literal, Tuple

from torch.utils.data import Dataset
from tqdm import tqdm

from corpus import CORPUS_PLAIN_FILE_NAME, DEFAULT_ENCODING, GOOD_CHARS_FILE_NAME, SPLIT_FILE_NAME
from corpus.corrector_dataset import CorrectorDataset
from corpus.make_split_csv import BYTE_INDEX_CLEAN_STR, SPLIT_CSV_HEADER, SPLIT_STR
from util.data_functions import get_line
from util.edit_distance import edit_distance


WHITESPACE_RE = re.compile(r"\s")


class DictionaryCorrectorDataset(Dataset):
    def __init__(self, data_dir: str, split: Literal["train", "validation", "test"]):
        print(f"Loading {self.__class__.__name__}, split='{split}'", file=sys.stderr)
        self.clean_corpus_path = os.path.join(data_dir, CORPUS_PLAIN_FILE_NAME)
        split_csv_path = os.path.join(data_dir, SPLIT_FILE_NAME)
        self.byte_indices = list()
        with open(split_csv_path, "r", encoding=DEFAULT_ENCODING, newline="") as split_file:
            csv_reader = csv.DictReader(split_file)
            assert csv_reader.fieldnames == SPLIT_CSV_HEADER, f"{split_csv_path} had unexpected header: {csv_reader.fieldnames}"
            for row in csv_reader:
                if row[SPLIT_STR] == split:
                    self.byte_indices.append(int(row[BYTE_INDEX_CLEAN_STR]))

    def __len__(self) -> int:
        return len(self.byte_indices)

    def __getitem__(self, index: int) -> str:
        byte_index_clean = self.byte_indices[index]
        text_clean = get_line(self.clean_corpus_path, byte_index_clean)
        return text_clean


class DictionaryCorrector:
    def __init__(self, min_frequency: int = 2, good_chars: str = string.ascii_lowercase + string.ascii_uppercase):
        self.min_frequency = min_frequency
        self._good_chars = WHITESPACE_RE.sub("", good_chars)
        self.vocabulary = dict()

    def train(self, data: DictionaryCorrectorDataset):
        print(f"{self.__class__.__name__} training progress:", file=sys.stderr)
        for i in tqdm(range(len(data))):
            text = data[i]
            for token in text.strip().split():
                try:
                    self.vocabulary[token] += 1
                except KeyError:
                    self.vocabulary[token] = 1

    def prune(self, min_frequency: int = None):
        if min_frequency is None:
            min_frequency = self.min_frequency
        print(f"Calculating prunings for min_frequency = {min_frequency}", file=sys.stderr)
        to_prune = list()
        for token, frequency in tqdm(self.vocabulary.items()):
            if frequency < min_frequency:
                to_prune.append(token)
        n_to_prune = len(to_prune)
        n_original = len(self.vocabulary)
        print(f"Pruning {n_to_prune} of {n_original} word forms ({n_original - n_to_prune} will remain)", file=sys.stderr)
        for token in tqdm(to_prune):
            del self.vocabulary[token]

    def _in_vocab(self, token: str) -> bool:
        # don't count words below the minimum frequency
        return token in self.vocabulary and self.vocabulary[token] >= self.min_frequency

    def __call__(self, to_correct: str) -> str:  # inference
        to_return = list()
        for raw_token in to_correct.strip().split():  # split by whitespace
            raw_token_size = len(raw_token)
            if self._in_vocab(raw_token):  # it's in our vocab; no edit
                to_return.append(raw_token)
            else:  # not recognized; find the word that's closest by edit distance
                best_token = None
                best_score = None
                best_frequency = None
                for real_token, frequency in sorted(self.vocabulary.items(), key=lambda x: abs(len(x[1] - raw_token_size))):
                    if best_score is not None and abs(len(real_token) - raw_token_size) > best_score:
                        break  # not possible to get a better edit score from this word; too many letters need to be added or deleted just to match the length
                    if frequency < self.min_frequency:
                        continue  # this word happens so rarely we won't count it as being in the vocabulary
                    score = edit_distance(raw_token, real_token)
                    if best_score is None or score < best_score or (score == best_score and frequency > best_frequency):  # use frequency to break ties on edit distance
                        best_token = real_token
                        best_score = score
                        best_frequency = frequency
                to_return.append(best_token)
        return " ".join(to_return)

    def evaluate(self, dataset: CorrectorDataset) -> Tuple[float, float]:
        print(f"{self.__class__.__name__} evaluating...", file=sys.stderr)
        n = len(dataset)
        distance_total = 0
        n_perfect = 0
        for i in tqdm(range(n)):
            text_messy, text_clean = dataset[i]
            text_out = self(text_messy)
            if text_out == text_clean:  # nailed it
                n_perfect += 1
            else:  # measure how far off it was
                distance = edit_distance(text_out, text_clean)
                distance_norm = distance / len(text_clean)
                distance_total += distance_norm
        avg_distance = distance_total / n
        return avg_distance, n_perfect / n

    def save(self, file_path: str):
        print(f"Saving {self.__class__.__name__} to file {file_path}", file=sys.stderr)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding=DEFAULT_ENCODING) as file:
            print(json.dumps(self.__dict__, indent=2), file=file)

    @classmethod
    def load(cls, file_path: str):  # -> DictionaryCorrector:
        print(f"Loading {cls.__name__} from file {file_path}", file=sys.stderr)
        with open(file_path, "r", encoding=DEFAULT_ENCODING) as file:
            json_obj = json.loads(file.read())
        assert isinstance(json_obj, dict), f"root json type needs to be a dict"
        to_return = DictionaryCorrector()
        assert set(to_return.__dict__.keys()) == set(json_obj.keys()), "keys of the loaded json dict do not match the needed keys"
        to_return.__dict__ = json_obj
        return to_return


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("corpus_dir", type=str, help="File path to the directory containing the corpus to train on.")
    args = arg_parser.parse_args()
    corpus_dir = args.corpus_dir

    models_dir = os.path.join("data", "models", "dictionary_corrector")

    # get good chars
    with open(os.path.join(corpus_dir, GOOD_CHARS_FILE_NAME), "r", encoding=DEFAULT_ENCODING) as chars_file:
        good_chars_ = chars_file.read().replace("\n", "")  # these chars will be used to generate edits

    dataset_train = DictionaryCorrectorDataset(corpus_dir, split="train")
    corrector = DictionaryCorrector(min_frequency=1, good_chars=good_chars_)
    corrector.train(dataset_train)
    # corrector = DictionaryCorrector.load("data/models/dictionary_corrector/dictionary_corrector-min_1.json")

    dataset_val = CorrectorDataset(corpus_dir, split="validation")

    best_model_avg = None
    best_model_path = None

    # estimate a min_freq value that will give us just the top 1000 words
    freq_for_top_1000 = max(corrector.vocabulary.values()) / 1000

    try:
        for power in range(math.ceil(math.log2(freq_for_top_1000))):
            min_freq = 2 ** power
            print("----------")
            print(f"min_frequency = {min_freq}")
            corrector.min_frequency = min_freq
            corrector.prune()
            this_model_path = os.path.join(models_dir, f"dictionary_corrector-min_{min_freq}.json")
            corrector.save(this_model_path)
            print("Evaluating on validation set...")
            average_distance, percent_perfect = corrector.evaluate(dataset_val)
            print(f"Average edit distance: {average_distance:.2f}")
            print(f"Percent perfect: {100 * percent_perfect:.2f}%")
            if best_model_avg is None or average_distance < best_model_avg:
                print("(Best model so far!)")
                best_model_avg = average_distance
                best_model_path = this_model_path
    except KeyboardInterrupt as e:
        if best_model_path is None:
            raise e
        print("\nINTERRUPTED - skipping to final evaluation on 'test' set\n", file=sys.stderr)

    print("----------")
    print(f"Loading best model: {best_model_path}")
    corrector = DictionaryCorrector.load(best_model_path)
    print("Evaluating on test set...")
    dataset_test = CorrectorDataset(corpus_dir, split="test")
    average_distance, percent_perfect = corrector.evaluate(dataset_val)
    print(f"Average edit distance: {average_distance:.2f}")
    print(f"Percent perfect: {100 * percent_perfect:.2f}%")
