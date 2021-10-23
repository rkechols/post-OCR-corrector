import argparse
import multiprocessing as mp
import os
import time
from typing import Tuple

from tqdm import tqdm

from corpus import DEFAULT_ENCODING


# make sure the images dir exists, but is empty
IMAGES_DIR = os.path.join("data", "corpus", "images")
os.makedirs(IMAGES_DIR, exist_ok=True)
for file_name_ in os.listdir(IMAGES_DIR):
    file_path_ = os.path.join(IMAGES_DIR, file_name_)
    if os.path.isfile(file_path_):
        os.remove(file_path_)


def number_to_filename(i: int) -> str:
    # formats i to have 8 digits by adding leading zeroes if needed
    # (8 digits because there are 25,636,542 sentences)
    return f"sentence_{i:08d}.png"


def sentence_to_image(index_sentence: Tuple[int, str]):
    index, sentence = index_sentence
    time.sleep(0.01)  # TODO: actually make the image and save it to disk


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("plain_text_path", type=str, help="File path to the plain text file to create images from.")
    arg_parser.add_argument("--n-total", type=int, default=None, help="File path to the plain text file to create images from.")
    args = arg_parser.parse_args()

    print(f"Converting lines of {args.plain_text_path} to images...")

    with open(args.plain_text_path, "r", encoding=DEFAULT_ENCODING) as corpus_file:
        sentence_generator = ((i_, line.strip()) for i_, line in enumerate(corpus_file, start=1) if line.strip() != "")
        cpus = os.cpu_count()
        if cpus <= 2:  # multiprocessing isn't worth it
            for tup in tqdm(sentence_generator, total=args.n_total):
                sentence_to_image(tup)
        else:  # cpus >= 3
            with mp.Pool(cpus - 1) as pool:
                for _ in tqdm(pool.imap_unordered(sentence_to_image, sentence_generator), total=args.n_total):
                    pass

    print(f"Created images of text in directory {IMAGES_DIR}")
