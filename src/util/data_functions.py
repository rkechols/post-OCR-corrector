import torch
from torch import Tensor

from corpus import DEFAULT_ENCODING


def get_line(file_path: str, byte_index: int) -> str:
    chars = list()
    with open(file_path, "r", encoding=DEFAULT_ENCODING) as file:
        file.seek(byte_index)
        while True:  # read to the end of the line
            char = file.read(1)
            if char == "" or char == "\n":
                break
            if char.isspace():  # collapse all blocks of whitespace to a single space
                if len(chars) == 0 or chars[-1].isspace():
                    continue  # don't start with whitespace or put one space after another
                else:
                    chars.append(" ")
            else:  # not whitespace; act normal
                chars.append(char)
    return "".join(chars)


def text_to_tensor(text: str, all_chars: str) -> Tensor:
    one_hot_tensor = torch.zeros((len(text), len(all_chars) + 1))  # +1 is for unknown
    for i, index in enumerate(all_chars.find(char) for char in text):  # if char is not found, `find` gives -1
        one_hot_tensor[i, index] = 1
    return one_hot_tensor
