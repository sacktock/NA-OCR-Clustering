# National Archives OCR Clustering Project

This is a proof of concept implementation of a document clustering pipeline. Our key aim is to cluster documents that have similar OCR mistakes, to aid with croud sourcing and so that tools that help alleviate these mistakes can be applied on batches of documents rather than on a one to one basis.

## Idea

The main idea is to extract text with the best off-the-shelf optical character recognition (OCR) tools. Then apply a spell checking tool on the extrated text to create a one-hot vector for each of the mistakes. Clustering with k-means (or similar) on multi-hot vector encodings of the documents should cluster documents with similar mistakes and we hypothesise from similar eras or by similar scribes/wiritng styles.

## Proposed Pipeline

Apply OCR -> Clean up extracted text -> spell check text document -> create one-hot vector encodings for each of the mistakes (using most common spelling mistakes in the dataset) -> cerate multi-hot vector for each document -> k-means clustering on the dataset -> visualisation tools or run batch scripts on the clusters
 
## Dataset

Aquired from [http://discovery.nationalarchives.gov.uk/details/r/C12122](http://discovery.nationalarchives.gov.uk/details/r/C12122) the dataset consists of 60 wills and testament documents of scanned handwritten text. The dataset is further divided into 4 subgroubs with 15 documents each, where each subgroup contains docuemnts of a similar era.

## Required Tools

- [ocrmypdf](https://ocrmypdf.readthedocs.io/en/latest/index.html)
- [aspell](http://aspell.net/)
```
sudo apt-get update -y
sudo apt-get install -y aspell
```