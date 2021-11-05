import csv
import os
import sys
import time
from typing import Literal, Tuple

import numpy as np
from torch.utils.data import Dataset

from corpus import ALL_CHARS_FILE_NAME, CORPUS_MESSY_FILE_NAME, CORPUS_PLAIN_FILE_NAME, DEFAULT_ENCODING, SPLIT_FILE_NAME
from corpus.make_split_csv import BYTE_INDEX_CLEAN_STR, BYTE_INDEX_MESSY_STR, SPLIT_CSV_HEADER, SPLIT_NAMES, SPLIT_STR
from util.data_functions import get_line, text_to_tensor


class CorrectorDataset(Dataset):
    def __init__(self, data_dir: str, split: Literal["train", "validation", "test"], tensors_out: bool = False):
        print(f"Loading {self.__class__.__name__}, split='{split}'", file=sys.stderr)
        self.clean_corpus_path = os.path.join(data_dir, CORPUS_PLAIN_FILE_NAME)
        self.messy_corpus_path = os.path.join(data_dir, CORPUS_MESSY_FILE_NAME)
        split_csv_path = os.path.join(data_dir, SPLIT_FILE_NAME)
        byte_indices = list()
        with open(split_csv_path, "r", encoding=DEFAULT_ENCODING, newline="") as split_file:
            csv_reader = csv.DictReader(split_file)
            assert csv_reader.fieldnames == SPLIT_CSV_HEADER, f"{split_csv_path} had unexpected header: {csv_reader.fieldnames}"
            for row in csv_reader:
                if row[SPLIT_STR] == split:
                    byte_indices.append((int(row[BYTE_INDEX_CLEAN_STR]), int(row[BYTE_INDEX_MESSY_STR])))
        self.byte_indices = np.array(byte_indices)  # numpy array so that when the dataset is copied by the workers, it doesn't copy this whole array at every access
        with open(os.path.join(data_dir, ALL_CHARS_FILE_NAME), "r", encoding=DEFAULT_ENCODING) as chars_file:
            self._all_chars = chars_file.read().replace("\n", "")
        self.tensors_out = tensors_out

    @property
    def alphabet_size(self) -> int:
        return len(self._all_chars)

    @property
    def alphabet(self) -> str:
        return self._all_chars

    def __len__(self) -> int:
        return len(self.byte_indices)

    def __getitem__(self, index: int) -> Tuple[str, str]:
        byte_index_clean, byte_index_messy = self.byte_indices[index]
        text_clean = get_line(self.clean_corpus_path, byte_index_clean)
        text_messy = get_line(self.messy_corpus_path, byte_index_messy)
        if self.tensors_out:
            text_messy = text_to_tensor(text_messy, self._all_chars)
            text_clean = text_to_tensor(text_clean, self._all_chars)
        return text_messy, text_clean


if __name__ == "__main__":
    n_total = 0
    for split_name in SPLIT_NAMES:
        time_start = time.time()
        dataset = CorrectorDataset("data/corpus/srWaC/", split=split_name)
        time_end = time.time()
        time_diff = time_end - time_start
        print(f"Time to load '{split_name}': {time_diff:.2f} seconds")
        this_size = len(dataset)
        n_total += this_size
        print("SIZE:", this_size)
        print("----------")
    print("SIZE OF ALL:", n_total)

    dataset = CorrectorDataset("data/corpus/srWaC/", split="test")

    same_count = 0
    n_tests = 100

    time_start = time.time()
    for i_ in range(n_tests):
        messy, correct = dataset[i_]
        if messy == correct:
            same_count += 1
    time_end = time.time()

    print("SAME COUNT:", same_count, f"(from sample of {n_tests})")

    time_diff = time_end - time_start
    time_per_item = time_diff / n_tests
    print(f"AVG TIME TO READ EACH ITEM: {time_per_item:.4f} seconds")
    expected_total = time_per_item * n_total
    print(f"EXPECTED TIME TO READ ALL: {expected_total:.2f} seconds")
    expected_minutes = expected_total / 60
    print(f"EXPECTED TIME TO READ ALL: {expected_minutes:.2f} minutes")
    expected_hours = expected_minutes / 60
    print(f"EXPECTED TIME TO READ ALL: {expected_hours:.2f} hours")
    expected_days = expected_hours / 24
    print(f"EXPECTED TIME TO READ ALL: {expected_days:.2f} days")
