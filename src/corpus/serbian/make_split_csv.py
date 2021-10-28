import csv
import os

from corpus import DEFAULT_ENCODING
from corpus.serbian.to_plain_text import CORPUS_PLAIN_FILE_PATH


CORPUS_SPLIT_CSV = os.path.join(os.path.dirname(CORPUS_PLAIN_FILE_PATH), "split.csv")
SPLIT_TRAIN = "train"
SPLIT_VAL = "validation"
SPLIT_TEST = "test"


def pick_split(line_num: int) -> str:
    # last digit of the line number:
    # 1 through 8 -> train
    # 9 -> validation
    # 0 -> test
    remainder = line_num % 10
    if remainder == 9:
        return SPLIT_VAL
    elif remainder == 0:
        return SPLIT_TEST
    else:
        return SPLIT_TRAIN


if __name__ == "__main__":
    with open(CORPUS_PLAIN_FILE_PATH, "r", encoding=DEFAULT_ENCODING) as corpus_file:
        with open(CORPUS_SPLIT_CSV, "w", encoding=DEFAULT_ENCODING, newline="") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["byte_index", "split"])
            line_num_ = 1
            cursor_value = corpus_file.tell()
            just_saw_newline = False
            while True:
                char = corpus_file.read(1)  # one char (even if it's multiple bytes)
                if char == "":
                    break  # no more chars to read
                elif char == "\n":
                    just_saw_newline = True
                    # pick a split and write down this line
                    split_str = pick_split(line_num_)
                    csv_writer.writerow([cursor_value, split_str])
                    # prepare for the next line
                    line_num_ += 1
                    cursor_value = corpus_file.tell()
                else:
                    just_saw_newline = False
            if not just_saw_newline:  # no final newline; we're missing the last line
                split_str = pick_split(line_num_)
                csv_writer.writerow([cursor_value, split_str])

    print(f"Done creating {CORPUS_SPLIT_CSV}")
