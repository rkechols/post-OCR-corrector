import random

import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFont
from skimage.util import random_noise

from util.fonts import ALL_FONTS


HEX_WHITE = "#FFFFFF"

IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512
MARGIN_HORIZONTAL = 16
MARGIN_VERTICAL = 16


def sentence_to_image(sentence: str) -> Image.Image:
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
    return image
