import csv
import json
import os
import sys
import time

from torch.utils.data import Dataset
from tqdm import tqdm

from corpus import CORPUS_PLAIN_FILE_NAME, DEFAULT_ENCODING, SPLIT_FILE_NAME
from corpus.make_split_csv import BYTE_INDEX_CLEAN_STR, SPLIT_CSV_HEADER, SPLIT_STR, SPLIT_TEST
from util.data_functions import get_line
from util.edit_distance import edit_distance


class DictionaryCorrectorDataset(Dataset):
    def __init__(self, data_dir: str, test: bool = False):
        self.clean_corpus_path = os.path.join(data_dir, CORPUS_PLAIN_FILE_NAME)
        split_csv_path = os.path.join(data_dir, SPLIT_FILE_NAME)
        self.byte_indices = list()
        with open(split_csv_path, "r", encoding=DEFAULT_ENCODING, newline="") as split_file:
            csv_reader = csv.DictReader(split_file)
            assert csv_reader.fieldnames == SPLIT_CSV_HEADER, f"{split_csv_path} had unexpected header: {csv_reader.fieldnames}"
            for row in csv_reader:
                if (row[SPLIT_STR] == SPLIT_TEST) == test:
                    self.byte_indices.append(int(row[BYTE_INDEX_CLEAN_STR]))

    def __len__(self) -> int:
        return len(self.byte_indices)

    def __getitem__(self, index: int) -> str:
        byte_index_clean = self.byte_indices[index]
        text_clean = get_line(self.clean_corpus_path, byte_index_clean)
        return text_clean


class DictionaryCorrector:
    def __init__(self, min_frequency: int = 2):
        self.min_frequency = min_frequency
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

    def __call__(self, to_correct: str) -> str:  # inference
        to_return = list()
        for raw_token in to_correct.strip().split():  # split by whitespace
            raw_token_size = len(raw_token)
            if raw_token in self.vocabulary and self.vocabulary[raw_token] >= self.min_frequency:  # it's in our vocab; no edit
                to_return.append(raw_token)
            else:  # not recognized; find the word that's closest by edit distance
                best_token = None
                best_score = None
                best_frequency = None
                for real_token, frequency in self.vocabulary.items():
                    if frequency < self.min_frequency:
                        continue  # this word happens so rarely we won't count it as being in the vocabulary
                    if best_score is not None and abs(len(real_token) - raw_token_size) > best_score:
                        continue  # not possible to get a better edit score from this word; too many letters need to be added or deleted just to match the length
                    score = edit_distance(raw_token, real_token)
                    if best_score is None or score < best_score or (score == best_score and frequency > best_frequency):  # use frequency to break ties on edit distance
                        best_token = real_token
                        best_score = score
                        best_frequency = frequency
                to_return.append(best_token)
        return " ".join(to_return)

    def save(self, file_path: str):
        print(f"Saving {self.__class__.__name__} to file {file_path}...", file=sys.stderr)
        with open(file_path, "w", encoding=DEFAULT_ENCODING) as file:
            print(json.dumps(self.__dict__, indent=2), file=file)

    @classmethod
    def load(cls, file_path: str):  # -> DictionaryCorrector:
        print(f"Loading {cls.__name__} from file {file_path}...", file=sys.stderr)
        with open(file_path, "r", encoding=DEFAULT_ENCODING) as file:
            json_obj = json.loads(file.read())
        assert isinstance(json_obj, dict), f"root json type needs to be a dict"
        to_return = DictionaryCorrector()
        assert set(to_return.__dict__.keys()) == set(json_obj.keys()), "keys of the loaded json dict do not match the needed keys"
        to_return.__dict__ = json_obj
        return to_return


if __name__ == "__main__":
    # dataset = DictionaryCorrectorDataset("data/corpus/srWaC", test=True)
    # corrector = DictionaryCorrector()
    # corrector.train(dataset)
    #
    # print("Sentence in:")
    # test_sentence = "Ov mi je najbola stvar."
    # print(test_sentence)
    #
    # print("Sentence out:")
    # out_sentence = corrector(test_sentence)
    # print(out_sentence)
    #
    # corrector.save("data/base_corrector.json")
    # del corrector

    corrector = DictionaryCorrector.load("data/base_corrector.json")
    # corrector.prune()
    corrector.min_frequency = 1
    del corrector.vocabulary["najbola"]  # hax so we get the full run time

    test_sentence = "Ov mi je najbola stvar."
    print("Sentence in:")
    print(test_sentence)

    time_start = time.time()
    out_sentence = corrector(test_sentence)
    time_end = time.time()
    print("Sentence out:")
    print(out_sentence)
    print(f"Time elapsed: {time_end - time_start:.2f} seconds")
