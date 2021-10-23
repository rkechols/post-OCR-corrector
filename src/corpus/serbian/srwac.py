import os
import re
import sys


EXPECTED_FILES = {f"srWaC1.1.0{i}.xml" for i in range(1, 7)}

SENTENCE_OPEN_RE = re.compile("<s>")
SENTENCE_CLOSE_RE = re.compile("</s>")
TOKEN_RE = re.compile(r"(\S+)(?:\s+\S+){3}")


class SrWaC:
    def __init__(self, dir_path: str):
        self.dir_path = dir_path
        found_files = os.listdir(dir_path)
        if set(found_files) != EXPECTED_FILES:
            print("WARNING: the files found do not match the expected files", file=sys.stderr)
        self.file_paths = [os.path.join(self.dir_path, file_name) for file_name in found_files]
        self.file_paths.sort()
        self.n = 0
        for file_path in self.file_paths:
            with open(file_path, "r", encoding="utf-8") as file:
                for line in file:
                    self.n += len(SENTENCE_OPEN_RE.findall(line))

    def __len__(self) -> int:
        return self.n

    def __iter__(self):
        for file_path in self.file_paths:
            with open(file_path, "r", encoding="utf-8") as file:
                in_sentence = False
                sentence = list()
                for line in file:
                    if SENTENCE_OPEN_RE.search(line):
                        in_sentence = True
                        sentence = list()
                    elif SENTENCE_CLOSE_RE.search(line):
                        in_sentence = False
                        yield sentence
                    elif in_sentence:
                        token_match = TOKEN_RE.search(line)
                        if token_match is not None:
                            sentence.append(token_match.group(1))
