import os
from typing import Iterable, List, Set, Tuple

import torch
from torch import Tensor

from corpus import ALL_CHARS_FILE_NAME, GOOD_CHARS_FILE_NAME
from util import DEFAULT_ENCODING


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
    unknown_index = len(all_chars)
    tensor_out = torch.empty(len(text), dtype=torch.long)
    for i, index in enumerate(all_chars.find(char) for char in text):
        if index == -1:  # if char is not found (unknown), `find` gives -1
            tensor_out[i] = unknown_index
        else:  # regular char
            tensor_out[i] = index
    return tensor_out


def collate_single_column(data: Iterable[Tensor]) -> Tensor:
    # find the longest sequence length
    x_size = max(x.shape[0] for x in data)
    x_stack = list()
    for x in data:
        # does it need padding?
        x_len_diff = x_size - x.shape[0]
        if x_len_diff > 0:
            x_stack.append(torch.cat([x, torch.full((x_len_diff,), -1)], dim=0))
        else:
            x_stack.append(x)
    x_batch = torch.stack(x_stack, dim=1)  # sequence first, batch second
    return x_batch


def collate_sequences(data_pairs: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor]:
    x_batch = collate_single_column(iter(x for x, _ in data_pairs))
    y_batch = collate_single_column(iter(y for _, y in data_pairs))
    return x_batch, y_batch


def get_alphabet(data_dir: str, only_select_chars: bool = False) -> str:
    char_file_name = GOOD_CHARS_FILE_NAME if only_select_chars else ALL_CHARS_FILE_NAME
    with open(os.path.join(data_dir, char_file_name), "r", encoding=DEFAULT_ENCODING) as chars_file:
        all_chars = chars_file.read().replace("\n", "")  # \n is never in the alphabet, but it may be in the file if they put it on multiple lines
    return all_chars


def get_whitespace_indices(data_dir: str) -> Set[int]:
    alphabet = get_alphabet(data_dir)
    to_return = set()
    for index, char in enumerate(alphabet):
        if char.isspace():
            to_return.add(index)
    return to_return
