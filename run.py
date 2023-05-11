import os
from collections import defaultdict
import numpy as np
import heapq
from sklearn import metrics
from time import time
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
import pandas as pd
import matplotlib.pyplot as plt

DIMENSION = 1000

if __name__ == '__main__':

    dataset_dir = "./dataset"
    tmp_dir = "./tmp"

    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    filenames = [filename for filename in os.listdir(dataset_dir) if filename.endswith(".pdf")]
    filenames.sort()

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
    mistake_counts = defaultdict(int)
    mistake_doc_total = defaultdict(int)
    for filename in filenames:
        filepath = os.path.join(tmp_dir, filename[:-4]+"-OCR-mistakes.txt")
        mistake_counts_copy = mistake_counts.copy()
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                mistake_counts[line] += 1
        for key, value in mistake_counts_copy.items():
            if value < mistake_counts[key]:
                mistake_doc_total[key] += 1

    for key, value in mistake_doc_total.items():
        # Remove mistakes that appear in >50% of documents
        if value > 0.5 * len(filenames):
            mistake_counts[key] = 0
        # Remove mistakes that appear in <5% of documents
        if value < 0.05 * len(filenames):
                mistake_counts[key] = 0
    
    # Filter the mistakes to only keep the most common mistakes
    filtered_items = heapq.nlargest(DIMENSION, mistake_counts.items(), key=lambda x: x[1])
    # create a new dictionary with the filtered key-value pairs
    filtered_mistakes = dict(filtered_items)
    mistake_list = list(filtered_mistakes.keys())

    # One hot encoding function - order of list doesn't matter
    one_hot_encode = lambda x: np.array([float(x == label) for label in mistake_list], dtype=np.float32)

    # Create the multihot encodings for each file
    encodings_dict = {filename : np.zeros(DIMENSION, dtype=np.float32) for filename in filenames}
    for filename in filenames:
        filepath = os.path.join(tmp_dir, filename[:-4]+"-OCR-mistakes.txt")
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                encodings_dict[filename] += one_hot_encode(line)
    X = np.array([encodings_dict[filename] for filename in filenames])

    print(f"n_samples: {X.shape[0]}, n_features: {X.shape[1]}")
    print(f"{np.sum(X) / np.prod(X.shape):.3f}")

    # Create dataset labels 
    labels_dict = {filename : filename.split('-')[2] for filename in filenames}
    labels = [labels_dict[filename] for filename in filenames]

    # Following code is mostly lifted from https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html
    unique_labels, category_sizes = np.unique(labels, return_counts=True)
    true_k = unique_labels.shape[0]

    print(f"{len(X)} documents - {true_k} categories")

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

    kmeans = KMeans(
        n_clusters=true_k,
        max_iter=100,
        n_init=5,
    )

    fit_and_evaluate(kmeans, X, name="KMeans\non mistake vectors")

    # Dimensionality reduction with LSA
    
    lsa = make_pipeline(TruncatedSVD(n_components=100), Normalizer(copy=False))
    X_lsa = lsa.fit_transform(X)
    explained_variance = lsa[0].explained_variance_ratio_.sum()

    print(f"Explained variance of the SVD step: {explained_variance * 100:.1f}%")
    kmeans = KMeans(
        n_clusters=true_k,
        max_iter=100,
        n_init=5,
    )

    fit_and_evaluate(kmeans, X_lsa, name="KMeans\nwith LSA on mistake vectors")

    original_space_centroids = lsa[0].inverse_transform(kmeans.cluster_centers_)
    order_centroids = original_space_centroids.argsort()[:, ::-1]
    terms = mistake_list

    for i in range(true_k):
        print(f"Cluster {i}: ", end="")
        for ind in order_centroids[i, :10]:
            print(f"{terms[ind]} ", end="")
        print()


    fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(16, 6), sharey=True)

    df = pd.DataFrame(evaluations[::-1]).set_index("estimator")
    df_std = pd.DataFrame(evaluations_std[::-1]).set_index("estimator")

    df.drop(
        ["train_time"],
        axis="columns",
    ).plot.barh(ax=ax0, xerr=df_std)
    ax0.set_xlabel("Clustering scores")
    ax0.set_ylabel("")

    df["train_time"].plot.barh(ax=ax1, xerr=df_std["train_time"])
    ax1.set_xlabel("Clustering time (s)")
    plt.tight_layout()
    plt.savefig(f'./clustering.png', dpi=300)
    



    

