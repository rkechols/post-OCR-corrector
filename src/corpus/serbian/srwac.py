import os
import re
import sys


EXPECTED_FILES = {f"srWaC1.1.0{i}.xml" for i in range(1, 7)}

SENTENCE_OPEN = "<s>"
SENTENCE_CLOSE = "</s>"
NO_SPACE_TAG = "<g/>"
TOKEN_RE = re.compile(r"(\S+)(?:\s+\S+){3}")


class SrWaC:
    def __init__(self, dir_path: str):
        self.dir_path = dir_path
        found_files = os.listdir(dir_path)
        if set(found_files) != EXPECTED_FILES:
            print("WARNING: the file names found do not match the expected file names", file=sys.stderr)
        self.file_paths = [os.path.join(self.dir_path, file_name) for file_name in found_files]
        self.file_paths.sort()

    def __iter__(self):
        for file_path in self.file_paths:
            with open(file_path, "r", encoding="utf-8") as file:
                in_sentence = False
                sentence = list()
                for line_ in file:
                    line = line_.strip()
                    if line == SENTENCE_OPEN:
                        if in_sentence:
                            print("WARNING: nested sentence?", file=sys.stderr)
                        in_sentence = True
                        sentence = list()
                    elif line == SENTENCE_CLOSE:
                        if not in_sentence:
                            print("WARNING: sentence close without open?", file=sys.stderr)
                        in_sentence = False
                        yield sentence
                    elif in_sentence:
                        if line == NO_SPACE_TAG:
                            sentence.append(NO_SPACE_TAG)
                        else:
                            token_match = TOKEN_RE.fullmatch(line)
                            if token_match is not None:
                                sentence.append(token_match.group(1))
                            else:
                                print("WARNING: unexpected line within a sentence", file=sys.stderr)
                                print(line, file=sys.stderr)
