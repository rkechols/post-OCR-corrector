# Post-OCR Corrector

Deep Learning tool to correct text extracted from a document with OCR.

Final project for BYU LING 581 "NLP".

This code was developed using Python 3.8.10 and may not work correctly on other versions.
Scripts are intended to be OS-agnostic, but were developed on Linux Ubuntu 20.04.3 LTS.

Commands listed in this document use bash syntax. Adjustments may be needed to use other command shells.

All scripts are intended to be run with this directory (the repository root directory) as the working directory,
but with the `src` directory as the Python path. For example, to run `src/train.py`, use this command:
```shell
PYTHONPATH=src python src/train.py 
```

Or, to run scripts without needing `PYTHONPATH=src` at the start each time, use this command to save it for your session:
```shell
export PYTHONPATH=src
```


## Dependencies

### Python Packages

Python packages that need to be installed are listed in `requirements.txt` and can be installed with this command:
```shell
pip install -r requirements.txt
```

### Tesseract OCR

Google's [Tesseract OCR](https://opensource.google/projects/tesseract) needs to be installed.
One simple way to install is with `apt`:
```shell
sudo apt install tesseract-ocr -y
```


## Workflow

The following is an outline of the expected workflow.

### 1. Generate Training Data

#### 1. XML to Plain Text

`src/corpus/serbian/to_plain_text.py`: Converts the Serbian Corpus (srWaC1.1) from XML format to plain text.

Usage:
```shell
to_plain_text.py [-h] srwac_path
```
- `srwac_path`: the path to the directory of the original srWaC1.1 corpus.

#### 2. Plain Text to Images

`src/corpus/to_images.py`: Converts a plain text corpus file into a set of image file, one image per line.

Usage:
```shell
to_images.py [-h] [--n-total N_TOTAL] plain_text_path
```
- `plain_text_path`: the path to the plain text corpus file.
- `N_TOTAL`: (optional) the number of lines in the plain text corpus file. Used to estimate remaining time before completion.
