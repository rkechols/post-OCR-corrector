import argparse
import multiprocessing as mp
import os
import random
from typing import Tuple

import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFont
from skimage.util import random_noise
from tqdm import tqdm

from corpus import DEFAULT_ENCODING
from util.fonts import ALL_FONTS


HEX_WHITE = "#FFFFFF"

IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512
MARGIN_HORIZONTAL = 16
MARGIN_VERTICAL = 16


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
    # make a canvas
    image = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), color=HEX_WHITE)
    draw = ImageDraw.Draw(image)
    # pick some random attributes
    text_color = random.randrange(32)
    font = ImageFont.truetype(random.choice(ALL_FONTS), random.randrange(10, 17))
    # figure out where to put newlines
    tokens = sentence.split()
    lines = [[tokens[0]]]
    for token in tokens[1:]:
        # check if this token will fit on the most recent line
        if draw.textsize(" ".join(lines[-1] + [token]), font=font)[0] > IMAGE_WIDTH - (2 * MARGIN_HORIZONTAL):
            # nope; add it to a new line
            lines.append([token])
        else:
            # sure; put it in
            lines[-1].append(token)
    wrapped_sentence = "\n".join([" ".join(line) for line in lines])
    # check the height
    bottom = draw.multiline_textbbox((MARGIN_HORIZONTAL, MARGIN_VERTICAL), wrapped_sentence, font=font)[3]
    # re-do the canvas to be the right height
    image = Image.new("RGB", (IMAGE_WIDTH, bottom + MARGIN_VERTICAL), color=HEX_WHITE)
    draw = ImageDraw.Draw(image)
    # actually put the text on
    draw.multiline_text((MARGIN_HORIZONTAL, MARGIN_VERTICAL), wrapped_sentence, fill=(text_color, text_color, text_color), font=font)
    # tilt the image a bit
    image = image.rotate(random.randrange(-3, 4), PIL.Image.BILINEAR, expand=True, fillcolor=HEX_WHITE)
    # add random noise
    noise_level = random.uniform(0, 0.15)
    np_image = np.asarray(image)
    np_image = random_noise(np_image, mode="gaussian", var=(noise_level ** 2))
    np_image = (255 * np_image).astype(np.uint8)
    image = Image.fromarray(np_image)
    # save the image to disk
    image.save(os.path.join(IMAGES_DIR, number_to_filename(index)))


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("plain_text_path", type=str, help="File path to the plain text file to create images from.")
    arg_parser.add_argument("--n-total", type=int, default=None, help="Number of sentences being turned into images.")
    arg_parser.add_argument("--cpu-limit", type=int, default=(os.cpu_count() - 1), help="Maximum number of CPU cores allowed to use.")
    args = arg_parser.parse_args()

    print(f"Converting lines of {args.plain_text_path} to images...")

    with open(args.plain_text_path, "r", encoding=DEFAULT_ENCODING) as corpus_file:
        sentence_generator = ((i_, line.strip()) for i_, line in enumerate(corpus_file, start=1) if line.strip() != "")
        cpus_available = max(min(os.cpu_count(), args.cpu_limit), 1)  # the outer max() is in case we get a non-positive number for the CPU limit
        if cpus_available <= 2:  # multiprocessing isn't worth it
            for tup in tqdm(sentence_generator, total=args.n_total):
                sentence_to_image(tup)
        else:  # cpus_available >= 3
            with mp.Pool(cpus_available - 1) as pool:
                for _ in tqdm(pool.imap_unordered(sentence_to_image, sentence_generator), total=args.n_total):
                    pass

    print(f"Created images of text in directory {IMAGES_DIR}")
