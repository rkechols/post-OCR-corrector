import argparse
import csv
import os

from tqdm import tqdm

from corpus import CORPUS_MESSY_FILE_NAME, CORPUS_PLAIN_FILE_NAME, DEFAULT_ENCODING, SPLIT_FILE_NAME


SPLIT_TRAIN = "train"
SPLIT_VAL = "validation"
SPLIT_TEST = "test"
SPLIT_NAMES = [SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST]


BYTE_INDEX_CLEAN_STR = "byte_index_clean"
BYTE_INDEX_MESSY_STR = "byte_index_messy"
SPLIT_STR = "split"
SPLIT_CSV_HEADER = [BYTE_INDEX_CLEAN_STR, BYTE_INDEX_MESSY_STR, SPLIT_STR]


def pick_split(line_num: int) -> str:
    # last digit of the line number determines split
    # 80% train, 10% validation, 10% test
    remainder = line_num % 10
    if remainder == 9:
        return SPLIT_VAL
    elif remainder == 0:
        return SPLIT_TEST
    else:
        return SPLIT_TRAIN


def line_byte_indices(file):
    cursor_value = file.tell()
    just_saw_newline = False
    while True:
        char = file.read(1)  # one char (even if it's multiple bytes)
        if char == "":
            break  # no more chars to read
        elif char == "\n":
            just_saw_newline = True
            # send out the byte-index for the start of this line we just finished
            yield cursor_value
            # prepare for the next line
            cursor_value = file.tell()
        else:
            just_saw_newline = False
    if not just_saw_newline:  # no final newline; we're missing the last line
        yield cursor_value


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("corpus_dir", type=str, help="File path to the directory containing the corpus to index and split.")
    args = arg_parser.parse_args()
    corpus_dir = args.corpus_dir

    corpus_clean_path = os.path.join(corpus_dir, CORPUS_PLAIN_FILE_NAME)
    corpus_messy_path = os.path.join(corpus_dir, CORPUS_MESSY_FILE_NAME)

    print(f"Clean corpus file: {corpus_clean_path}")
    print(f"Messy corpus file: {corpus_messy_path}")

    corpus_split_csv_path = os.path.join(corpus_dir, SPLIT_FILE_NAME)

    print(f"Indexing corpus files and determining train/val/test splits...")

    with open(corpus_clean_path, "r", encoding=DEFAULT_ENCODING) as corpus_clean_file:
        with open(corpus_messy_path, "r", encoding=DEFAULT_ENCODING) as corpus_messy_file:
            with open(corpus_split_csv_path, "w", encoding=DEFAULT_ENCODING, newline="") as csv_file:
                # open a csv writer and write the header row
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(SPLIT_CSV_HEADER)
                # create generators that give the byte-index of the start of each line
                bytes_clean = line_byte_indices(corpus_clean_file)
                bytes_messy = line_byte_indices(corpus_messy_file)
                for line_num_, (byte_clean, byte_messy) in tqdm(enumerate(zip(bytes_clean, bytes_messy), start=1)):
                    split_str = pick_split(line_num_)  # pick a split for this pair to belong to
                    csv_writer.writerow([byte_clean, byte_messy, split_str])

    print(f"Done creating {corpus_split_csv_path}")
