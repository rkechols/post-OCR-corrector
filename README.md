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

### Fonts Listing

In order to type text into images, a listing of paths to `.ttf` files is needed. On Linux that can be created with these commands:
```shell
mkdir -p data
fc-list | grep ttf > data/fonts.txt
```

It is recommended that the listed fonts be legible to a normal human, and capable of handling any UTF-8 characters.


## Workflow

The following is an outline of the expected workflow.

For help regarding usage for any script, call the script with the `-h` flag:
```shell
python my_script.py -h
```

### 1. Generate Training Data

1. XML to Plain Text: `src/corpus/serbian/to_plain_text.py`  
    Converts the Serbian Corpus (srWaC1.1) from XML format to plain text.

2. Index and Split Corpus: `src/corpus/make_split_csv.py`  
    Creates a CSV file with byte-indices (for use with `seek`) for the start of each line, as well as which dataset split (train, validation, test) the line belongs to.
