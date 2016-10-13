# Document Similarity

This exercise uses tf-idf vectorization with principal component analysis to map the similarity between text documents. For this
exercise, we analyze the entire corpus of Shakespeare's plays (37 plays). These plays are separated into three categories:
comedy, tragedy, and history.

Each play is put through a term frequency - inverse document frequency vectorizor that turns the play into a numerical vector
of features. These numerical features reflect the log count of unique words in each play multiplied by the log inverse count of
documents that each word belongs to. The standard set of English stop words from the Python nltk package are used so that common
English words such as "and" and "the" are ignored.

The dimensionality of the final feature vectors for the plays is equal to the number of distinct words (minus stop words) in the corpus --
in this case 9658. To visualize this data, we use principal component analysis to reduce the data to two dimensions, and the result is 
displayed below.

### Results

![corpus](https://github.com/iamshang1/Projects/blob/master/Basic_ML/Document_Similarity/corpus.png)

We see that there are distinct linguistic differences between Shakespeare's comedies, tragedies, and histories in terms of word usage.
We also see that some of Shakespeare's comedies, tragedies, and histories share similar vector space (after dimensionality reduction),
indicating that these plays use similar language even though they belong to different classes.