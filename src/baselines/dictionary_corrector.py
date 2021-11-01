import csv
import os
import sys

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
    def __init__(self):
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

    def __call__(self, to_correct: str) -> str:  # inference
        to_return = list()
        for raw_token in to_correct.strip().split():  # split by whitespace
            if raw_token in self.vocabulary:  # it's in our vocab; no edit
                to_return.append(raw_token)
            else:  # not recognized; find the word that's closest by edit distance
                best_token = None
                best_score = None
                best_frequency = None
                for real_token, frequency in self.vocabulary.items():
                    score = edit_distance(raw_token, real_token)
                    if best_score is None or score < best_score or (score == best_score and frequency > best_frequency):
                        best_token = real_token
                        best_score = score
                        best_frequency = frequency
                to_return.append(best_token)
        return " ".join(to_return)
