# Spam Detection

This exercise uses Naive Bayes to detect if emails are spam or ham. 

The Enron email dataset is used for this exercise. This dataset contains 16383 spam emails and 16363 ham emails.

For preprocessing, the words in each email are first stemmed using the NLTK stemming package.
Then, the stemmed emails are vectorized using Scikit Learn's term frequency-inverse document frequency vectorizer.
The tf-idf vectorizer uses the standard list of English stop words and an n-gram range of 2.

Because word occurence is not normally distributed, multinomial naive bayes is used for classification.

### Results

Mean Cross Validation Classification Accuracy: 99.15%

Mean Cross Validation Spam Precision: 99.42%

Mean Cross Validation Spam Recall: 98.88%

Mean Cross Validation Spam F1 Score: 99.15%

### Word Stems Most Indicative of Ham

enron, ect, subject, hou, hou ect, 2000, ect ect, 2001, schedul, vinc, thank, deal,
pm, attach, gas, 01, cc, know, meet, com, 10, date, hpl, messag, let

### Word Stems Most Indicative of Spam

subject, http, com, compani, email, softwar, click, www, price, onlin, money, save,
mail, free, offer, inform, http www, want, order, best, secur, time, remov, busi, adob

**Note:**

We see that some word stems appear in both the list of Ham words and Spam words. This is because
these words appear often in both spam and ham emails.