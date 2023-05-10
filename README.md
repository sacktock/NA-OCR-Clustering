# National Archives OCR Clustering Project

This is a proof of concept implementation of a document clustering pipeline. Our key aim is to cluster documents that have similar OCR mistakes, to aid with croud sourcing and so that tools that help alleviate these mistakes can be applied on batches of documents rather than on a one to one basis.

## Idea

The main idea is to extract text with the best off-the-shelf optical character recognition (OCR) tools. Then apply a spell checking tool on the extrated text to create a one-hot vector of mistakes. Clustering with k-means (or similar) on these one-hot vectors should cluster documents with similar mistakes and we hypothesise from similar eras or by similar scribes/wiritng styles.

## Proposed Pipeline

Apply OCR -> Clean up extracted text -> spell check text document -> create one-hot vector encodings for each of the documents (using most common spelling mistakes in the dataset) -> k-means clustering on the dataset -> visualisation tools or run batch scripts on the clusters
 
## Required Tools

- [ocrmypdf](https://ocrmypdf.readthedocs.io/en/latest/index.html)