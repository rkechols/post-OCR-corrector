import argparse
import os

from corpus import DEFAULT_ENCODING


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("corpus_path", type=str, help="File path to the plain-text file containing the corpus to index and split.")
    args = arg_parser.parse_args()
    corpus_path = args.corpus_path

    print(f"Collecting chars from {corpus_path}")

    all_chars = set()
    with open(corpus_path, "r", encoding=DEFAULT_ENCODING) as corpus_file:
        while True:
            char = corpus_file.read(1)
            if char == "":  # no more to read
                break
            all_chars.add(char)

    all_chars_path = os.path.join(os.path.dirname(corpus_path), "all_chars.txt")

    with open(all_chars_path, "w", encoding=DEFAULT_ENCODING) as all_chars_file:
        for char in all_chars:
            all_chars_file.write(char)

    print(f"All chars written to {all_chars_path}")
