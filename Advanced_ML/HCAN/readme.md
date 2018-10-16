## Hierarchical Convolutional Attention Networks for Text Classification

This repo contains some of the code used for the paper [https://aclanthology.info/papers/W18-3002/w18-3002](*Hierarchical Convolutional Attention Networks for Text Classification*). 

### Requirements

 - Python 3.6 or higher
 - Tensorflow 1.5 or higher
 - Gensim 3.4 or higher

### Instructions to Run Models

Download the Yelp Reviews dataset from [https://www.yelp.com/dataset](https://www.yelp.com/dataset)

Use the following command to preprocess the Yelp reviews:
 - python feature_extraction_yelp.py \<path to Yelp json file\>

Use one or more of the following commands to run the desired model:
 - python tf_cnn.py
 - python tf_han.py
 - python tf_hcan.py
 - python traditional_ml.py

### Acknowledgements
This work has been supported in part by the Joint Design of Advanced Computing Solutions for Cancer (JDACS4C) program established by the U.S. Department of Energy (DOE) and the National Cancer Institute (NCI) of the National Institutes of Health.
