# Topic Modeling

This exercise uses Latent Dirichlet allocation to perform topic modeling on the 20Newsgroups 
corpus from the Usenet newsgroups collection.

Latent Dirichlet allocation is an unsupervised learning technique that attempts to guess the topics
discussed within a corpus of text documents based on how words appear together in the documents. If words
commonly appear together in many documents, those words will be placed within the same topic.

The gensim Python module is used to perform LDA for this exercise, and it is used to separate the 
20Newsgroups corpus into 20 topics.

### True Topics

Below are the actual topics in the 20Newsgroups corpus:
 - Computing - Graphics
 - Computing - Miscellaneous
 - Computing - PC Hardware
 - Computing - Mac Hardware
 - Computing - Windows
 - Miscellaneous - For Sale
 - Recreation - Automotive
 - Recreation - Motorcycles
 - Recreation - Baseball
 - Recreation - Hockey
 - Science - Cryptography
 - Science - Electronics
 - Science - Medicine
 - Science - Space
 - Politics - Gun Control
 - Politics - Middle East Policy
 - Politics - Miscellaneous
 - Religion - Atheism
 - Religion - Christianity
 - Religion - Miscellaneous

### Predicted Topics

The most common topic seems to be about Christian morality:

![wordcloud1](https://github.com/iamshang1/Projects/blob/master/ML_Exercises/Topic_Modeling/wordcloud1.png)

The second most common topic is also about Christianity, but seems to involve science vs faith:

![wordcloud2](https://github.com/iamshang1/Projects/blob/master/ML_Exercises/Topic_Modeling/wordcloud2.png)

The third most common topic seems to be about security in the Middle East:

![wordcloud3](https://github.com/iamshang1/Projects/blob/master/ML_Exercises/Topic_Modeling/wordcloud3.png)