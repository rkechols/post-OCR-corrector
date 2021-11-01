import argparse
import os

from corpus import ALL_CHARS_FILE_NAME, CORPUS_PLAIN_FILE_NAME, DEFAULT_ENCODING


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("corpus_dir", type=str, help="File path to the directory containing the corpus to collect chars from.")
    args = arg_parser.parse_args()
    corpus_dir = args.corpus_dir

    corpus_path = os.path.join(corpus_dir, CORPUS_PLAIN_FILE_NAME)
    print(f"Collecting chars from {corpus_path}")

    all_chars = set()
    with open(corpus_path, "r", encoding=DEFAULT_ENCODING) as corpus_file:
        while True:
            char = corpus_file.read(1)
            if char == "":  # no more to read
                break
            all_chars.add(char)

    all_chars_path = os.path.join(corpus_dir, ALL_CHARS_FILE_NAME)

    with open(all_chars_path, "w", encoding=DEFAULT_ENCODING) as all_chars_file:
        for char in sorted(all_chars):
            all_chars_file.write(char)

    print(f"All chars written to {all_chars_path}")
