import os
from collections import defaultdict
import numpy as np
import heapq
from sklearn import metrics
from time import time

DIMENSION = 1000

if __name__ == '__main__':

    dataset_dir = "./dataset"
    tmp_dir = "./tmp"

    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    filenames = ([filename for filename in os.listdir(dataset_dir) if filename.endswith(".pdf")]).sort()

    for filename in filenames:
        filepath = os.path.join(dataset_dir, filename)
        outpath = os.path.join(tmp_dir, filename[:-4]+"-OCR.txt")
        # Run OCR
        if not os.path.exists(outpath):
            exit_code = os.system(f'ocrmypdf --sidecar {outpath} --force-ocr --deskew --clean --output-type=none {filepath} - > sdout.txt')
            if exit_code == 0:
                print(f'OCR {filename} successful')
            else:
                print(f'OCR {filename} unsuccessful')
        # Run clean script
        inpath = os.path.join(tmp_dir, filename[:-4]+"-OCR.txt")
        outpath = os.path.join(tmp_dir, filename[:-4]+"-OCR-clean.txt")
        if not os.path.exists(outpath):
            exit_code = os.system(f'python clean.py -i {inpath} -o {outpath}')
            if exit_code == 0:
                print(f'Cleaned {filename} successfuly')
            else:
                print(f'Cleaned {filename} unsuccessfuly')
        # Run spell check
        inpath = os.path.join(tmp_dir, filename[:-4]+"-OCR-clean.txt")
        outpath = os.path.join(tmp_dir, filename[:-4]+"-OCR-mistakes.txt")
        if not os.path.exists(outpath):
            exit_code = os.system(f'aspell --lang=en --mode=none --list < {inpath} > {outpath}')
            if exit_code == 0:
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

    # Create dataset labels 
    label_dict = {'3': 0, '57':1, '885':2, '2248':3}
    labels = {filename : label_dict[filename.split('-')[2]] for filename in filenames}

    evaluations = []
    evaluations_std = []


    def fit_and_evaluate(km, X, name=None, n_runs=5):
        name = km.__class__.__name__ if name is None else name

        train_times = []
        scores = defaultdict(list)
        for seed in range(n_runs):
            km.set_params(random_state=seed)
            t0 = time()
            km.fit(X)
            train_times.append(time() - t0)
            scores["Homogeneity"].append(metrics.homogeneity_score(labels, km.labels_))
            scores["Completeness"].append(metrics.completeness_score(labels, km.labels_))
            scores["V-measure"].append(metrics.v_measure_score(labels, km.labels_))
            scores["Adjusted Rand-Index"].append(
                metrics.adjusted_rand_score(labels, km.labels_)
            )
            scores["Silhouette Coefficient"].append(
                metrics.silhouette_score(X, km.labels_, sample_size=2000)
            )
        train_times = np.asarray(train_times)

        print(f"clustering done in {train_times.mean():.2f} ± {train_times.std():.2f} s ")
        evaluation = {
            "estimator": name,
            "train_time": train_times.mean(),
        }
        evaluation_std = {
            "estimator": name,
            "train_time": train_times.std(),
        }
        for score_name, score_values in scores.items():
            mean_score, std_score = np.mean(score_values), np.std(score_values)
            print(f"{score_name}: {mean_score:.3f} ± {std_score:.3f}")
            evaluation[score_name] = mean_score
            evaluation_std[score_name] = std_score
        evaluations.append(evaluation)
        evaluations_std.append(evaluation_std)

    



    

