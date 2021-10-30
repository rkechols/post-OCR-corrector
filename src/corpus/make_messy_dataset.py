import argparse
import os
import random
from enum import auto, Enum

from tqdm import tqdm

from corpus import DEFAULT_ENCODING


class EditType(Enum):
    DELETE = auto()
    CHANGE = auto()
    INSERT = auto()


EDIT_CHANCE = 0.12
N_EDIT_TYPES = len(EditType) + 1
INSERT_CHANCE = EDIT_CHANCE / N_EDIT_TYPES


def mutilate_string(text: str, good_chars: str) -> str:
    n = len(text)
    to_return = list()
    i = 0
    while i < n:
        if random.uniform(0.0, 1.0) <= EDIT_CHANCE:  # yes edit
            if random.randrange(0, N_EDIT_TYPES) == 0:
                pass
            else:
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
                # else: UNKNOWN
        else:  # no edit
            to_return.append(text[i])
            i += 1
    # we could also insert at the end
    while random.uniform(0.0, 1.0) <= INSERT_CHANCE:
        # pick a character to add
        new_char = random.choice(good_chars)
        to_return.append(new_char)
    # return the new version of the string
    return "".join(to_return)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("corpus_path", type=str, help="File path to the plain-text file containing the corpus to make a messy version of.")
    arg_parser.add_argument("--good-chars-path", type=str, required=True, help="File path to the text file containing the 'good'/'standard' characters of the corpus.")
    args = arg_parser.parse_args()
    corpus_path = args.corpus_path
    good_chars_path = args.good_chars_path

    with open(good_chars_path, "r", encoding=DEFAULT_ENCODING) as good_chars_file:
        good_chars_ = good_chars_file.read()
    good_chars_ = good_chars_.replace("\n", "")  # \n is never a "good char", but it may be in the file if they put it on multiple lines

    print(f"Creating a mutilated/messy version of {corpus_path}")

    messy_path = os.path.join(os.path.dirname(corpus_path), "corpus-messy.txt")

    with open(corpus_path, "r", encoding=DEFAULT_ENCODING) as corpus_file:
        with open(messy_path, "w", encoding=DEFAULT_ENCODING) as messy_file:
            for line in tqdm(corpus_file):
                messy_line = mutilate_string(line.strip(), good_chars_)
                print(messy_line, file=messy_file)

    print(f"Messy corpus written to {messy_path}")
