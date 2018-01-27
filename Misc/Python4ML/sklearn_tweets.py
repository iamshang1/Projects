import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

#import data
data = pd.read_csv('airline_tweets.csv')
print(data)

#show one column
print(data['text'])

#show two columns
print(data[['text','airline_sentiment']])

#show one record
print(data.loc[0,:])

#show some more records
print(data.loc[0:5,:])

#show one column of one record
print(data.loc[0,'text'])

#select columns
data = data[['tweet_id','text','airline_sentiment','airline']]
print(data)

#convert sentiment into numbers
def sentiment2int(sentiment):
    if sentiment == 'positive':
        return 1
    elif sentiment == 'neutral':
        return 0
    elif sentiment == 'negative':
        return -1
    else:
        return np.NaN

data['rating'] = data['airline_sentiment'].apply(sentiment2int)
print(data)
        
#alternatively, use encoder
encoder = LabelEncoder()
encoder.fit(data['airline_sentiment'])
data['encoded'] = encoder.transform(data['airline_sentiment'])
print(data)
        
#average sentiment of airlines
filter = data['airline'] == 'Virgin America'
virgin = data[filter]
print(virgin['rating'].mean())
        
#tfidf
vectorizer = TfidfVectorizer(min_df=3, stop_words='english',ngram_range=(1, 2))
vectorizer.fit(data['text'])
X = vectorizer.transform(data['text'])

#get labels
y = np.array(data['rating'])

print(X)
print(X.shape)
print(y)
print(y.shape)

#test train split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=1234)
print(X_train.shape)
print(X_test.shape)

#naive bayes
nb = MultinomialNB()
nb.fit(X_train,y_train)
print(nb.score(X_test,y_test))
nb_preds = nb.predict(X_test)
print(nb_preds)

#logistic regression
lr = LogisticRegression(penalty='l1',C=1)
lr.fit(X_train,y_train)
print(lr.score(X_test,y_test))
lr_preds = lr.predict(X_test)
print(lr_preds)

#random forest
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train,y_train)
print(rf.score(X_test,y_test))
rf_preds = rf.predict(X_test)
print(rf_preds)