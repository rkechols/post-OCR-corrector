import argparse
import os
from typing import List

from tqdm import tqdm

from corpus import DEFAULT_ENCODING
from corpus.serbian import CORPUS_DIR
from corpus.serbian.srwac import NO_SPACE_TAG, SrWaC


CORPUS_PLAIN_FILE_PATH = os.path.join(CORPUS_DIR, "srWaC-plain.txt")


def sentence_to_string(tokens: List[str]) -> str:
    to_return = list()
    saw_joiner = False
    for token in tokens:
        if token == NO_SPACE_TAG:
            saw_joiner = True
            continue
        if saw_joiner:
            saw_joiner = False
            to_return[-1] += token
        else:
            to_return.append(token)
    return " ".join(to_return)


def clean_text(text: str, allowed_chars: str) -> str:
    to_return = list()
    for char in text:
        if char in allowed_chars:
            to_return.append(char)
        else:
            if len(to_return) != 0 and not to_return[-1].isspace():
                to_return.append(" ")
    return "".join(to_return).strip()


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("srwac_path", type=str, help="File path to the srWaC directory which contains the 6 XML files.")
    arg_parser.add_argument("--good-chars-path", type=str, required=True, help="File path to the plain text file which contains the entire set of allowed characters.")
    args = arg_parser.parse_args()
    srwac_path = args.srwac_path
    good_chars_path = args.good_chars_path

    with open(good_chars_path, "r", encoding=DEFAULT_ENCODING) as good_chars_file:
        good_chars = good_chars_file.read()
    good_chars = good_chars.replace("\n", "")

    print(f"Loading corpus from {srwac_path} ...")
    corpus = SrWaC(srwac_path)

    print(f"Converting corpus to plain text...")
    longest = 0
    with open(CORPUS_PLAIN_FILE_PATH, "w", encoding=DEFAULT_ENCODING) as out_file:
        for sentence in tqdm(corpus):
            sentence_str = sentence_to_string(sentence)
            sentence_str = clean_text(sentence_str, good_chars)
            longest = max(longest, len(sentence_str))
            print(sentence_str, file=out_file)

    print(f"Plain text corpus saved to {CORPUS_PLAIN_FILE_PATH}")
    print(f"Longest sentence was {longest} characters.")
