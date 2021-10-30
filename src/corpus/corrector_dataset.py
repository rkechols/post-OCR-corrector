import csv
import random
import re
import time
from enum import auto, Enum
from typing import Literal, Tuple

from torch.utils.data import Dataset

from corpus import DEFAULT_ENCODING
from corpus.make_split_csv import BYTE_INDEX_STR, SPLIT_CSV_HEADER, SPLIT_STR


WHITESPACE_RE = re.compile(r"\s+")

EDIT_CHANCE = 0.15


class EditType(Enum):
    DELETE = auto()
    CHANGE = auto()
    INSERT = auto()


N_EDIT_TYPES = len(EditType) + 1


class CorrectorDataset(Dataset):
    def __init__(self, split_csv_path: str, plain_txt_path: str, good_chars: str, split: Literal["train", "validation", "test"]):
        self.byte_indices = list()
        with open(split_csv_path, "r", encoding=DEFAULT_ENCODING, newline="") as split_file:
            csv_reader = csv.DictReader(split_file)
            assert csv_reader.fieldnames == SPLIT_CSV_HEADER, f"{split_csv_path} had unexpected header: {csv_reader.fieldnames}"
            for row in csv_reader:
                if row[SPLIT_STR] == split:
                    self.byte_indices.append(row[BYTE_INDEX_STR])
        self.plain_txt_path = plain_txt_path
        self.good_chars = good_chars

    def __len__(self) -> int:
        return len(self.byte_indices)

    def __getitem__(self, index: int) -> Tuple[str, str]:
        byte_index = self.byte_indices[index]
        with open(self.plain_txt_path, "r", encoding=DEFAULT_ENCODING) as corpus_file:
            corpus_file.seek(byte_index)
            chars = list()
            while True:  # read to the end of the line
                char = corpus_file.read(1)
                if char == "" or char == "\n":
                    break
                chars.append(char)
        text_correct = "".join(chars)
        text_messy = self._mutilate_string(text_correct)
        return text_messy, text_correct

    def _mutilate_string(self, text: str) -> str:
        n = len(text)
        to_return = list()
        i = 0
        while i < n:
            if random.uniform(0.0, 1.0) <= EDIT_CHANCE:  # yes edit
                if random.randrange(0, N_EDIT_TYPES) == 0:
                    pass
                else:
                    for _ in range(random.choice([1, 2])):  # edit 1 or 2 chars right here
                        edit_type = random.choice(list(EditType))
                        if edit_type == EditType.DELETE:
                            # just increment the index without copying characters
                            i += 1
                        elif edit_type == EditType.CHANGE:
                            # pick a replacement character
                            new_char = random.choice(self.good_chars)
                            to_return.append(new_char)
                            i += 1  # move past the real one
                        elif edit_type == EditType.INSERT:
                            # pick a character to add
                            new_char = random.choice(self.good_chars)
                            to_return.append(new_char)
                            # don't increment `i` so the real char will actually be added
                        # else: UNKNOWN
            else:  # no edit
                to_return.append(text[i])
                i += 1
        # we could also insert at the end
        if random.uniform(0.0, 1.0) <= EDIT_CHANCE:
            for _ in range(random.choice([1, 2])):  # edit 1 or 2 chars right here
                # pick a character to add
                new_char = random.choice(self.good_chars)
                to_return.append(new_char)
        # return the new version of the string
        return "".join(to_return)


if __name__ == "__main__":
    with open("data/corpus/serbian/good_chars.txt", "r", encoding=DEFAULT_ENCODING) as good_chars_file:
        good_chars_ = good_chars_file.read()
    good_chars_ = good_chars_.replace("\n", "")  # \n is never a "good char", but it may be in the file if they put it on multiple lines

    time_start = time.time()
    dataset = CorrectorDataset("data/corpus/serbian/split.csv", "data/corpus/serbian/srWaC-plain.txt", good_chars_, split="test")
    time_end = time.time()

    time_diff = time_end - time_start
    print(f"TIME TO LOAD: {time_diff:.2f} seconds")
    n_total = len(dataset)
    print("SIZE:", n_total)

    same_count = 0
    n_tests = 100

    time_start = time.time()
    for i_ in range(n_tests):
        messy, correct = dataset[i_]
        if messy == correct:
            same_count += 1
    time_end = time.time()

    print("SAME COUNT:", same_count, f"(of {n_tests})")

    time_diff = time_end - time_start
    time_per_item = time_diff / n_tests
    print(f"AVG TIME TO LOAD EACH ITEM: {time_per_item:.4f} seconds")
    expected_total = time_per_item * n_total
    print(f"EXPECTED TIME TO LOAD ALL: {expected_total:.2f} seconds")
    expected_minutes = expected_total / 60
    print(f"EXPECTED TIME TO LOAD ALL: {expected_minutes:.2f} minutes")
    expected_hours = expected_minutes / 60
    print(f"EXPECTED TIME TO LOAD ALL: {expected_hours:.2f} hours")
    expected_days = expected_hours / 24
    print(f"EXPECTED TIME TO LOAD ALL: {expected_days:.2f} days")
