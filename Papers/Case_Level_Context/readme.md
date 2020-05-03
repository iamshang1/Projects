## Using case-level context to classify cancer pathology reports

This repo contains the code used in the paper *Using case-level context to classify cancer pathology reports*.

### Requirements

 - Python 3.6 or higher
 - Tensorflow 1.15

### Instructions for Use

See scripts for example usage. Each model requires input data in the form of `Iterable[Iterable[np.ndarray(dim=input_size)]]`
where the outer iterable represents a list of cases, the inner iterable represents list of documents for each case, 
and each document is represented by an n-dimensional document embedding. Each model requires labels in the form of
`Iterable[Iterable[int]]`, where the outer iterable represents a list of cases, and the inner iterable 
represents a list of labels for each document in the case.

Note that currently, the `predict()` method for each model returns a single list that flattens the predicted labels for all input documents.

### Acknowledgements
This work has been supported in part by the Joint Design of Advanced Computing Solutions for Cancer (JDACS4C) program established by the U.S. Department of Energy (DOE) and the National Cancer Institute (NCI) of the National Institutes of Health.
