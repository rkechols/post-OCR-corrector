import argparse
import os
import random
from enum import auto, Enum

from tqdm import tqdm

from corpus import CORPUS_MESSY_FILE_NAME, CORPUS_PLAIN_FILE_NAME
from util import DEFAULT_ENCODING
from util.data_functions import get_alphabet


class EditType(Enum):
    DELETE = auto()
    CHANGE = auto()
    INSERT = auto()
    SWAP = auto()


EDIT_CHANCE = 0.12
N_EDIT_TYPES = len(EditType)
INSERT_CHANCE = EDIT_CHANCE / N_EDIT_TYPES


def mutilate_string(text: str, good_chars: str) -> str:
    n = len(text)
    to_return = list()
    i = 0
    while i < n:
        if random.uniform(0.0, 1.0) < EDIT_CHANCE:  # yes edit
            edit_type = random.choice(list(EditType))
            if edit_type == EditType.DELETE:
                # just increment the index without copying characters
                i += 1
            elif edit_type == EditType.CHANGE:
                # pick a replacement character
                new_char = random.choice(good_chars)
                to_return.append(new_char)
                i += 1  # move past the real one
            elif edit_type == EditType.INSERT:
                # pick a character to add
                new_char = random.choice(good_chars)
                to_return.append(new_char)
                # don't increment `i` so the real char will actually be added
            elif edit_type == EditType.SWAP:
                if i + 1 < n:
                    to_return.append(text[i + 1])  # second comes first
                else:  # nothing to actually grab
                    to_return.append(" ")
                to_return.append(text[i])
                i += 2  # we used 2 characters
            # else: UNKNOWN
        else:  # no edit
            to_return.append(text[i])
            i += 1
    # we could also insert at the end
    while random.uniform(0.0, 1.0) < INSERT_CHANCE:
        # pick a character to add
        new_char = random.choice(good_chars)
        to_return.append(new_char)
    # return the new version of the string
    return "".join(to_return)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("corpus_dir", type=str, help="File path to the directory containing the corpus to make a messy version of.")
    args = arg_parser.parse_args()
    corpus_dir = args.corpus_dir

    corpus_path = os.path.join(corpus_dir, CORPUS_PLAIN_FILE_NAME)

    good_chars_ = get_alphabet(corpus_dir, only_select_chars=True)

    print(f"Creating a mutilated/messy version of {corpus_path}")

    messy_path = os.path.join(corpus_dir, CORPUS_MESSY_FILE_NAME)

    with open(corpus_path, "r", encoding=DEFAULT_ENCODING) as corpus_file:
        with open(messy_path, "w", encoding=DEFAULT_ENCODING) as messy_file:
            for line in tqdm(corpus_file):
                messy_line = mutilate_string(line.strip(), good_chars_)
                print(messy_line, file=messy_file)

    print(f"Messy corpus written to {messy_path}")
