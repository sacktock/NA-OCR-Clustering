# National Archives OCR Clustering Project

This is a proof of concept implementation of a document clustering pipeline. Our key aim is to cluster documents that have similar OCR mistakes, to aid with croud sourcing and so that tools that help alleviate these mistakes can be applied on batches of documents rather than on a one to one basis.

## Idea

The main idea is to extract text with the best off-the-shelf optical character recognition (OCR) tools. Then apply a spell checking tool on the extrated text to create a one-hot vector for each of the mistakes. Clustering with k-means (or similar) on multi-hot vector encodings of the documents should cluster documents with similar mistakes and we hypothesise from similar eras or by similar scribes/wiritng styles.

## Proposed Pipeline

- Apply off-the-shelf OCR
- Clean extracted text
- Spell check the extracted text
- [Optional] For better clustering, remove mistakes that occur in >50% of documents and <5% of documents.
- Create one-hot vector encodings of the most common spelling mistakes.
- Create multi-hot vector for each document (sum of one-hot vectors).
- [Optional] Perform dimensionality reduction with LSA (or similar).
- K-means (or similar) clustering on the multi-hot vectors.
- Apply evaluation metrics and visualisation tools.

## Dataset

Aquired from [http://discovery.nationalarchives.gov.uk/details/r/C12122](http://discovery.nationalarchives.gov.uk/details/r/C12122) the dataset consists of 60 wills and testament documents of scanned handwritten text. The dataset is further divided into 4 subgroubs with 15 documents each, where each subgroup contains docuemnts of a similar era.

## Required Tools

- [ocrmypdf](https://ocrmypdf.readthedocs.io/en/latest/index.html)
- [aspell](http://aspell.net/)
```
sudo apt-get update -y
sudo apt-get install -y aspell
```

## Running

```
python run.py
```

## Results

### Evaluation Metrics

![Evaluation Metrics](https://github.com/sacktock/NA-OCR-Clustering/blob/main/clustering.png)

### Most common mistakes
Cluster 0: fhe ane fone aud ond ard Ane Fhe fre dnd

Cluster 1: aud aus tte ote oud ree Aud ieee ated ete

Cluster 2: mee fre Ces aes ete ont aie ene poe fone

Cluster 3: ote Ree bok Referenceprob Sher fae Sele tal ces panies

## Acks

- [document clustering](https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html)