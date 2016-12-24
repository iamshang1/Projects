#Word2Vec

This exercise applies Word2Vec on the 20 newsgroups corpus to convert the 50k most common words into
dense numerical vectors. Word2Vec is an unsupervised learning technique that converts words to a vector
of numerical embeddings based on the context in which words appear. Words that commonly appear
in the same context will have similar embeddings, while words that seldomly appear in the same
context will have very different embeddings.

The gensim Python module is used to perform Word2Vec. In addition, a custom Theano implementation was
also tested, which had comparable performance but took significantly longer to train.

### Predicted Topics

Below is a visualization of the 500 most common words in the 20 newsgroup corpus in 2-dimensional space. 
TSNE is used to reduce the dimensionality from 50 to 2. We see that words that are semantically similar
are located close to each other.

![plot](https://github.com/iamshang1/Projects/blob/master/Basic_ML/Word2Vec/plot.png)