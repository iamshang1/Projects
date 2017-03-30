# Sentiment Analysis

This exercise utilizes SentiWordNet and Naive Bayes to classify text reviews into positive and negative sentiment
classes. The Amazon Fine Food Reviews dataset from Kaggle is used for this exercise. This dataset
has been pre-pruned to only include 1-star (negative) and 5-star (positive) reviews.

### Using SentiWordNet Only

SentiWordNet is a sentiment dictionary that maps words in the English dictionary to a positivity score,
negativity score, and objectivity score. Because the same word can be used in different parts of speech
and in different contexts, each word in SentiWordNet is associated with a part of speech and can have
multiple entries that account for the context in which the word is used.

For this exercise, each word in a review is tagged with a part of speech using the Python NLTK package.
The word is then looked up within SentiWordNet to find the positivity or negativity score associated with
the word. For this exercise, word-sense disambiguation is not used to determine the context of the word;
if the word has multiple entries in SentiWordNet, the average score across all entries is used.

If the cumulative positivity score of the review is greater than the negativity score, the review is
classified as positive. If the cumulative negativity score is greater than the positivity score, the review
is classified as negative. 

**Results**

Our dataset has 65581 positive reviews. Of these, SentiWordNet classified 76.64% as positive.

Our dataset has 52268 negative reviews. Of these, SentiWordNet classified 69.47% as negative.

### Using Naive Bayes Only

The Naive Bayes algorithm is a commonly used algorithm in classification tasks related to natural language
processing because it is simple to implement and yields solid results. This algorithm examines a corpus of text
and determines the probability of a word belonging to each class. A new document is then classified based on
the prior probabilities of the words in the document. Naive Bayes assumes that the occurence of any word is 
independent from the occurence of any other word, which is rarely true. However, Naive Bayes performs well in
practice despite this assumption.

Unlike the SentiWordNet, which as already been pre-trained, Naive Bayes requires training data to learn which
words are associated with positive reviews and which words are associated with negative reviews. Therefore, 
we split our dataset into training and testing sets as follows:

 - 40000 positive reviews and 40000 negative reviews in training set
 - 25581 positive reviews in positive review test set
 - 12268 negative reviews in negative review test set
 
A term frequency-inverse document frequency vectorizer is used to transform the train and test datasets into
vectors of float values. In addition, the standard set of English stop words (based on the Python nltk package)
is used to filter common words such as "the" and "and" from the reviews. Finally, words that appear less than
three times in the dataset are omitted.
 
**Results**

Our positive review test set has 25581 positive reviews. Of these, Naive Bayes classified 91.04% as positive.

Our negative review test set has 12268 negative reviews. Of these, Naive Bayes classified 88.98% as negative.

### Combining SentiWordNet with Naive Bayes

Finally, we combine SentiWordNet and Naive Bayes into a single classifier by combining their predictions.
From SentiWordNet, we take the positivity/(positivity+negativity) for each review. From Naive Bayes, we take
the probability that each review belongs to the positive class. These two values are then used as the input features
in a logistic regression classifier.

We use the same test/train split as that used in the Naive Bayes only case.

**Results**

Our positive review test set has 25581 positive reviews. Of these, the combined classifier classified 91.18% as positive.

Our negative review test set has 12268 negative reviews. Of these, the combined classifier classified 90.19% as negative.