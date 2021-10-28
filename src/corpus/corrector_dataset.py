import random
import re
import time
from typing import Tuple

import pandas as pd
from torch.utils.data import Dataset

from corpus import DEFAULT_ENCODING


WHITESPACE_RE = re.compile(r"\s+")


def mutilate_string(text: str) -> str:
    pass


class CorrectorDataset(Dataset):
    def __init__(self, split_csv_path: str, plain_txt_path: str):
        self.split_frame = pd.read_csv(split_csv_path)
        self.plain_txt_path = plain_txt_path

    def __len__(self) -> int:
        return len(self.split_frame)

    def __getitem__(self, index: int) -> Tuple[str, str]:
        byte_index, split = self.split_frame.iloc[index]
        with open(self.plain_txt_path, "r", encoding=DEFAULT_ENCODING) as corpus_file:
            corpus_file.seek(byte_index)
            chars = list()
            while True:  # read to the end of the line
                char = corpus_file.read(1)
                if char == "" or char == "\n":
                    break
                chars.append(char)
        text_correct = "".join(chars)
        text_messy = mutilate_string(text_correct)
        return text_messy, text_correct


if __name__ == "__main__":
    time_start = time.time()
    dataset = CorrectorDataset("data/corpus/serbian/split.csv", "data/corpus/serbian/srWaC-plain.txt")
    time_end = time.time()
    time_diff = time_end - time_start
    print(f"TIME TO LOAD: {time_diff:.2f} seconds")
    n_total = len(dataset)
    print("SIZE:", n_total)
    same_count = 0
    time_start = time.time()
    n_tests = 100
    for i in range(n_tests):
        messy, correct = dataset[i]
        if messy == correct:
            same_count += 1
    time_end = time.time()
    time_diff = time_end - time_start
    time_per_item = time_diff / n_tests
    print("SAME COUNT:", same_count)
    print(f"AVG TIME TO LOAD EACH ITEM: {time_per_item:.4f} seconds")
    expected_total = time_per_item * n_total
    print(f"EXPECTED TIME TO LOAD ALL: {expected_total:.2f} seconds")
    expected_minutes = expected_total / 60
    print(f"EXPECTED TIME TO LOAD ALL: {expected_minutes:.2f} minutes")
    expected_hours = expected_minutes / 60
    print(f"EXPECTED TIME TO LOAD ALL: {expected_hours:.2f} hours")
    expected_days = expected_hours / 24
    print(f"EXPECTED TIME TO LOAD ALL: {expected_days:.2f} days")
