import os
import sys


EXPECTED_FILES = {f"srWaC1.1.0{i}.xml" for i in range(1, 7)}


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
            pass  # TODO: iterate and increment n

    def __len__(self) -> int:
        return self.n

    def __iter__(self):
        for file_path in self.file_paths:
            pass  # TODO: iterate and yield
