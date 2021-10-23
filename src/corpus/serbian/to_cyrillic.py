import argparse
import os

from tqdm import tqdm

from corpus.serbian import CORPUS_DIR, DEFAULT_ENCODING
from corpus.serbian.srwac import SrWaC


CYRILLIC_CORPUS_FILE_PATH = os.path.join(CORPUS_DIR, "srWaC-cyrillic.txt")


def to_cyrillic(text: str) -> str:
    to_return = list()
    for char in text:
        if char == "a":
            char_c = ""  # TODO
        else:
            char_c = char
        to_return.append(char_c)
    return "".join(to_return)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("srwac-path", type=str, help="File path to the srWaC directory which contains the 6 XML files.")
    args = arg_parser.parse_args()

    print(f"Loading corpus from {args.srwac_path} ...")
    corpus = SrWaC(args.srwac_path)

    print(f"Converting corpus to Cyrillic...")
    with open(CYRILLIC_CORPUS_FILE_PATH, "w", encoding=DEFAULT_ENCODING) as out_file:
        for sentence in tqdm(corpus):
            print(to_cyrillic(sentence), file=out_file)

    print(f"Cyrillic corpus saved to {CYRILLIC_CORPUS_FILE_PATH}")
