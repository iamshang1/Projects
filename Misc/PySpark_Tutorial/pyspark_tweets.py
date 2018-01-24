from pyspark.sql import SparkSession
from pyspark.sql import Column
from pyspark.sql.types import *
from pyspark.sql.functions import udf
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, NGram, ChiSqSelector
from pyspark.ml.classification import NaiveBayes, RandomForestClassifier
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re

#start spark session
spark = SparkSession\
        .builder\
        .appName("TweetClassification")\
        .getOrCreate()

#import csv
df = spark.read.csv("airline_tweets.csv",header=True,inferSchema=True)

#show dataframe
df.show()

#count total records
print("total records:", df.count())

#show schema
df.printSchema()

#show one column
df.select('text').show(n=5,truncate=False)

#show one record
df.where(df['tweet_id'] == '570306133677760513').show()

#casting
df = df.withColumn("airline_sentiment_confidence", df["airline_sentiment_confidence"].cast("float"))
df = df.withColumn("negativereason_confidence", df["negativereason_confidence"].cast("float"))
df.printSchema()

#remove rows missing rating or tweets
print("total records:", df.count())
df = df.where(df['airline_sentiment'].isNotNull())
print("records with sentiment:", df.count())
df = df.where(df['text'].isNotNull())
print("records with sentiment and tweet text:", df.count())

#rearrange or drop columns
reduced_df = df.select("tweet_id","text","airline_sentiment","airline")
reduced_df.show()
reduced_df.printSchema()

#user defined functions
def label_encoder(sentiment):
    if sentiment == 'positive':
        return 1
    elif sentiment == 'negative':
        return -1
    elif sentiment == 'neutral':
        return 0
    else:
        raise Exception('invalid sentiment')
label_encoder_udf = udf(label_encoder, IntegerType())
reduced_df = reduced_df.withColumn("airline_sentiment", label_encoder_udf("airline_sentiment"))
reduced_df.show()
reduced_df.printSchema()

#summary statistics
reduced_df.filter(reduced_df['airline'] == 'Virgin America').show()
virgin_count = reduced_df.filter(reduced_df['airline'] == 'Virgin America').count()
print("Virgin American Tweets:", virgin_count)

reduced_df.filter(reduced_df['airline'] == 'Virgin America').agg({"airline_sentiment": "avg"}).show()
reduced_df.groupBy("airline").count().show()
reduced_df.groupBy("airline").agg({"airline_sentiment": "avg"}).show()

#remove neutral tweets
print("before removing neutral tweets:", reduced_df.count())
reduced_df = reduced_df.where(reduced_df['airline_sentiment'] != 0)
print("after removing neutral tweets:", reduced_df.count())
def label_encoder(sentiment):
    if sentiment == 1:
        return 1
    elif sentiment == -1:
        return 0
label_encoder_udf = udf(label_encoder, IntegerType())
reduced_df = reduced_df.withColumn("airline_sentiment", label_encoder_udf("airline_sentiment"))
reduced_df.show()

#count positive and negative tweets
positive = reduced_df.where(reduced_df['airline_sentiment'] == 1).count()
negative = reduced_df.where(reduced_df['airline_sentiment'] == 0).count()
print("positive reviews:", positive)
print("negative reviews:", negative)
print("baseline score:", negative/(positive+negative))

#clean text
def clean(text):
    text = text.lower()
    text = re.sub("'", '', text)
    text = re.sub('[^\w_]+', ' ', text)
    return text.lstrip()
    
clean_udf = udf(clean, StringType())
reduced_df = reduced_df.withColumn("clean_text", clean_udf("text"))
reduced_df.show()

#tokenize words
tokenizer = Tokenizer(inputCol="clean_text", outputCol="tokens")
reduced_df = tokenizer.transform(reduced_df)
reduced_df.show()
reduced_df.printSchema()

#stop and stem
stopwords = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")
def stop_stem(tokens):
    stemmed = [stemmer.stem(word) for word in tokens if word not in stopwords]
    return stemmed

stop_stem_udf = udf(stop_stem, ArrayType(StringType()))
reduced_df = reduced_df.withColumn("tokens", stop_stem_udf("tokens"))
reduced_df.show()
reduced_df.printSchema()

#tfidf
hashingTF = HashingTF(inputCol="tokens", outputCol="term_freq")
reduced_df = hashingTF.transform(reduced_df)
reduced_df.show()
idf = IDF(inputCol="term_freq", outputCol="tfidf", minDocFreq=5)
idfModel = idf.fit(reduced_df)
reduced_df = idfModel.transform(reduced_df)
reduced_df.show()

#test train split
train,test = reduced_df.select("tweet_id","tfidf","airline_sentiment").randomSplit([0.8, 0.2],seed=1234)
print("train samples:", train.count())
print("test samples:",test.count())

#apply naive bayes
nb = NaiveBayes(featuresCol="tfidf", labelCol="airline_sentiment", predictionCol="NB_pred",
                probabilityCol="NB_prob", rawPredictionCol="NB_rawPred")
nbModel = nb.fit(train)
test = nbModel.transform(test)
test.show()

#get test accuracy
total = test.count()
correct = test.where(test['airline_sentiment'] == test['NB_pred']).count()
print("naive bayes unigrams test accuracy:", correct/total)

#try bigrams
reduced_df = reduced_df.select("tweet_id","airline_sentiment","tokens")
ngram = NGram(n=2, inputCol="tokens", outputCol="ngrams")
reduced_df = ngram.transform(reduced_df)
reduced_df.show()

#rerun tfidf
hashingTF = HashingTF(inputCol="ngrams", outputCol="term_freq")
reduced_df = hashingTF.transform(reduced_df)
reduced_df.show()
idf = IDF(inputCol="term_freq", outputCol="tfidf", minDocFreq=5)
idfModel = idf.fit(reduced_df)
reduced_df = idfModel.transform(reduced_df)
reduced_df.show()

#rerun test train split (using same seed)
train,test = reduced_df.select("tweet_id","tfidf","airline_sentiment").randomSplit([0.8, 0.2],seed=1234)

#rerun naive bayes
nb = NaiveBayes(featuresCol="tfidf", labelCol="airline_sentiment", predictionCol="NB_pred",
                probabilityCol="NB_prob", rawPredictionCol="NB_rawPred")
nbModel = nb.fit(train)
test = nbModel.transform(test)
test.show()

#test accuracy
total = test.count()
correct = test.where(test['airline_sentiment'] == test['NB_pred']).count()
print("naive bayes bigrams test accuracy:", correct/total)

#close spark
spark.stop()