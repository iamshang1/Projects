#Deep Learning for Text Classification
This exercise compares the effectiveness of different deep learning models
for text classification with traditional machine learning text classifiers.
We compare the performance of Naive Bayes, XGBoost, Convolutional Neural
Networks, and Hierachical Attention Networks.

We use the Yelp reviews dataset for this exercise. This dataset can be obtained
online from [https://www.yelp.com/dataset_challenge/dataset](https://www.yelp.com/dataset_challenge/dataset). 
To reduce the size of the dataset (and therefore training time), we only use reviews from 2013, 
which yields approximately 470,000 documents. The goal of the classifiers is to 
predict the number of stars for a review based on the text from the review.

##Naive Bayes
Naive Bayes is a traditional machine learning classifier commonly used for text
classification. Naive Bayes first uses a training corpus to find the probability
of each word or phrase belonging to each class. For a new document, Naive Bayes multiplies
(or log sums) the precalculated probabilities of every word in the document and chooses
the class with the highest probability.

For our exercise, we removed punctuation and stop words and then applied TFIDF
vectorization on the resulting unigrams and bigrams. The TFIDF features were then
fed into the classifier. Naive bayes achieved a test set accuracy of 50.68%.

##XGBoost
XGBoost is an extremely powerful machine learning method based on gradient-boosted
trees. In XGBoost, additional decision trees are added to the model with each
new decision tree fitting to the residuals of the sum of all previous decision
trees. XGBoost has a reputation for being both accurate and fast, and it has been
used to win many Kaggle competitions.

For our exercise, we used the same feature set as Naive Bayes. We ran XGBoost with
a max depth of 5 and 300 estimators. XGBoost achieved a test set accuracy of 57.36%.

##Convolutional Neural Networks
Convolutional neural networks are traditionally used for image recognition but
have been adapted for text classification. First, words are converted to embedding
vectors using Word2Vec. These are fed into the CNN as features. The CNN uses a single
convolutional layer with a window size of three, four, and/or five words followed
by a single maxpool layer. These layers are designed to find the most impactful word
segments or phrases in a document in relation to the classes. The output of the maxpool
layer is then fed into a softmax classifier.

For our exercise, we use gensim's word2vec to create 200-dimension word embeddings
for the dataset. We use a three-word, four-word, and five-word window, each with
100 feature maps, and applied 50% dropout to the softmax layer to reduce overfitting.
Our CNN achieved a test set accuracy of 60.15%.

##Heirarchical Attention Networks
A heirarchical attention network ([http://www.aclweb.org/anthology/N16-1174](http://www.aclweb.org/anthology/N16-1174)) 
is a deep learning model composed of bidirectional LSTMs/GRUs with attention
mechanisms. The model has two "heirarchies". The lower heirarchy takes in one
sentence at a time, broken into word embeddings. This heirarchy outputs a
weighted sentence embedding based on the words in the sentence that are most 
relevant to the classification. The upper heirachy takes in one document at a
time, broken into the sentence embeddings from the lower heirarchy. This heirarchy
outputs a weighted document embedding based on the sentences in the document that
are most relevant to the classification. This final document embedding is then
fed into a softmax classifier.

For our exercise, we start with the same gensim word embeddings used for the CNN. Our HAN model
uses 50 LSTM units per heirarchy layer and an attention embedding size of 100.
Our HAN achieved a test set accuracy of 69.19%

##Instructions to Run Models
First, download the Yelp reviews dataset from [https://www.yelp.com/dataset_challenge/dataset](https://www.yelp.com/dataset_challenge/dataset)
and extract the .json file. Then run the following:

 - feature_extraction.py \<path to Yelp json\>
 - naive_bayes.py
 - xg-boost.py
 - th_cnn.py
 - th_han.py
