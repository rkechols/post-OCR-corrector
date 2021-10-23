import argparse
import os

from tqdm import tqdm

from corpus.serbian import CORPUS_DIR, DEFAULT_ENCODING
from corpus.serbian.srwac import SrWaC


CORPUS_PLAIN_FILE_PATH = os.path.join(CORPUS_DIR, "srWaC-plain.txt")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("srwac_path", type=str, help="File path to the srWaC directory which contains the 6 XML files.")
    args = arg_parser.parse_args()

    print(f"Loading corpus from {args.srwac_path} ...")
    corpus = SrWaC(args.srwac_path)

    print(f"Converting corpus to plain text...")
    with open(CORPUS_PLAIN_FILE_PATH, "w", encoding=DEFAULT_ENCODING) as out_file:
        for sentence in tqdm(corpus):
            print(" ".join(sentence), file=out_file)

    print(f"Plain text corpus saved to {CORPUS_PLAIN_FILE_PATH}")
