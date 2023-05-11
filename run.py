import os
from collections import defaultdict
import numpy as np
import heapq

DIMENSION = 1000

if __name__ == '__main__':

    dataset_dir = "./dataset"
    tmp_dir = "./tmp"

    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    filenames = [filename for filename in os.listdir(dataset_dir) if filename.endswith(".pdf")]

    for filename in filenames:
        filepath = os.path.join(dataset_dir, filename)
        outpath = os.path.join(tmp_dir, filename[:-4]+"-OCR.txt")
        # Run OCR
        if not os.path.exists(outpath)
            exit_code = os.system(f'ocrmypdf --sidecar {outpath} --force-ocr --deskew --clean --output-type=none {filepath} - > sdout.txt')
            if exit_code = 0:
                print(f'OCR {filename} successful')
            else:
                print(f'OCR {filename} unsuccessful')
        # Run clean script
        inpath = os.path.join(tmp_dir, filename[:-4]+"-OCR.txt")
        outpath = os.path.join(tmp_dir, filename[:-4]+"-OCR-clean.txt")
        if not os.path.exists(outpath)
            exit_code = os.system(f'python clean.py -i {inpath} -o {outpath}')
            if exit_code = 0:
                print(f'Cleaned {filename} successfuly')
            else:
                print(f'Cleaned {filename} unsuccessfuly')
        # Run spell check
        inpath = os.path.join(tmp_dir, filename[:-4]+"-OCR-clean.txt")
        outpath = os.path.join(tmp_dir, filename[:-4]+"-OCR-mistakes.txt")
        if not os.path.exists(outpath)
            exit_code = os.system(f'aspell --lang=en --mode=none --list < {inpath} > {outpath}')
            if exit_code = 0:
                print(f'Spell checked {filename} successfuly')
            else:
                print(f'Spell checked {filename} unsuccessfuly')

    # Create full dictionary of spelling mistakes
    mistakes = defaultdict(int)
    for filename in filenames:
        filepath = os.path.join(tmp_dir, filename[:-4]+"-OCR-mistakes.txt")
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                mistakes[line] += 1

    # Filter the mistakes to only keep the most common mistakes
    filtered_items = heapq.nlargest(DIMENSION, mistakes.items(), key=lambda x: x[1])
    # create a new dictionary with the filtered key-value pairs
    filtered_mistakes = dict(filtered_items)
    mistake_list = list(filtered_mistakes.keys())

    # One hot encoding function - order of list doesn't matter
    one_hot_encode = lambda x: np.array([float(x == label) for label in mistake_list], dtype=np.float32)

    # Create the multihot encodings for each file
    encodings = {filename : np.zeros(DIMENSION, dtype=np.float32) for filename in filenames}
    for filename in filenames:
        filepath = os.path.join(tmp_dir, filename[:-4]+"-OCR-mistakes.txt")
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                encodings[filename] += one_hot_encode(line)

    # TODO clustering & visualisation




    

