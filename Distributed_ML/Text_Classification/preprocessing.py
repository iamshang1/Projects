from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import *
from pyspark.rdd import RDD
from pyspark.mllib.linalg import SparseVector, VectorUDT, Vectors
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re
import numpy as np

#data preprocessing class
class preprocessor(object):
    '''
    preprocessor for Reuters news articles corpus

    parameters:
      - bigrams: bool
        if set to True, will use bigrams and unigrams in term frequency or tf-idf (default)
        if set to False, will only use unigrams in term frequency or tf-idf
      - min_df: int
        minimum number of times a word or bigram must appear in document to be included as a feature (default 2)
      - stemming: bool
        if set to True, words are stemmed when tokenized (default)
        if set to False, words are not stemmed when tokenized
      - tfidf: bool
        if set to True, tf-idf vectorization is used (default)
        if set to False, term frequency vectorization is used
        
    methods:
      - transform(X,y)
        convert text and labels into tf-idf dataframe for classifier
        parameters:
          - X: pyspark rdd
            rdd with text as rows
          - y: pyspark rdd (optional)
            rdd with labels as rows
    
    requires following non-standard python packages
      - nltk.corpus.stopwords
      - nltk.stem.snowball.SnowballStemmer
      - numpy
    '''
    def __init__(self,bigrams=True,min_df=3,stemming=True,tfidf=True):
        self.regex = re.compile('[^a-zA-Z ]')
        self.stop = set(stopwords.words('english'))
        self.stemmer = SnowballStemmer("english")
        self.bigrams = bigrams
        self.min_df = min_df
        self.stemming = stemming
        self.tfidf = tfidf
    
    def _tokenize(self, row):
        '''
        clean texts by removing special and non-chars
        stems each word and removes stop words
        return list of tokenized words for each row
        '''
        chars = re.sub(r'-|&quot;|&amp;',' ',row) #replace dashes, quotes, and ampersands
        chars = self.regex.sub('',chars) # remove nonchars
        wordlist = str(chars).split()
        if self.stemming:
            wordlist = [self.stemmer.stem(word.lower()) for word in wordlist if word.lower() not in self.stop] # stem and remove stopwords
        else:
            wordlist = [word.lower() for word in wordlist if word.lower() not in self.stop]
        
        #create bigrams if enabled
        if self.bigrams:
            bigrams = []
            for i in range(len(wordlist)-1):
                bigrams.append(wordlist[i]+" "+wordlist[i+1])
            wordlist = wordlist + bigrams
            
        return wordlist
                
    def _term_frequency(self,row):
        '''
        convert row of word token se into sparse vector of term frequencies
        '''
        sparse_dic = {}
        for word in row:
            if word in self.dictionary:
                if self.dictionary[word] in sparse_dic:
                    sparse_dic[self.dictionary[word]] += 1.
                else:
                    sparse_dic[self.dictionary[word]] = 1.
        tf = SparseVector(len(self.dictionary),sparse_dic)
        return tf
                
    def _tf_idf(self,row):
        '''
        convert row of word token counts into sparse vector of tfidf frequencies
        '''
        sparse_dic = {}
        df_dic = {}
        for word in row:
            if word in self.dictionary:
                if self.dictionary[word] in sparse_dic:
                    sparse_dic[self.dictionary[word]] += 1.
                else:
                    sparse_dic[self.dictionary[word]] = 1.
                if word in self.doc_freq:
                    df_dic[self.dictionary[word]] = self.doc_freq[word]
                else:
                    df_dic[self.dictionary[word]] = 1
        for key in sparse_dic:
            sparse_dic[key] = (1+np.log(sparse_dic[key]))*(np.log10(float(self.doc_count)/df_dic[key]))                    
        tfidf = SparseVector(len(self.dictionary),sparse_dic)
        return tfidf
       
    def transform(self,X,y=None,train=True):
        '''
        convert input RDDs into dataframe of features and labels
        '''
        #check input type
        if type(X) != RDD:
            raise TypeError("Arguments must be pySpark RDDs")
        if y and type(y) != RDD:
            raise TypeError("Arguments must be pySpark RDDs")            

        #word tokenization
        X = X.map(self._tokenize).cache()
        
        #create dictionary of words
        if train:
            self.dictionary = X.flatMap(lambda word: word).map(lambda word: (word,1)).reduceByKey(lambda acc, w: acc + w).filter(lambda x: x[1]>=self.min_df).collectAsMap()
            self.dictionary = dict(zip(self.dictionary,xrange(len(self.dictionary))))
            
            #populate word count dictionary
            if self.tfidf:
                self.doc_freq = X.map(lambda wordlist: set(wordlist)).flatMap(lambda word: word).map(lambda word: (word,1)).reduceByKey(lambda acc, w: acc + w).filter(lambda x: x[1]>=2).collectAsMap()
                self.doc_count = X.count()

        #create word vectors
        if self.tfidf:
            X = X.map(self._tf_idf)
        else:
            X = X.map(self._term_frequency)
        
        #check if labels exist
        if y:
            #combine X and y into single dataframe
            X = X.zipWithIndex().map(lambda r: (r[1],r[0]))
            y = y.zipWithIndex().map(lambda r: (r[1],r[0]))
            data = X.join(y).map(lambda r: r[1])            
            df = data.toDF(['features','label'])
            
            #one hot encoding for labels
            CCAT = udf(lambda l: 1 if "CCAT" in l else 0, IntegerType())
            ECAT = udf(lambda l: 1 if "ECAT" in l else 0, IntegerType())
            GCAT = udf(lambda l: 1 if "GCAT" in l else 0, IntegerType())
            MCAT = udf(lambda l: 1 if "MCAT" in l else 0, IntegerType())
            df = df.withColumn("CCAT",CCAT(df['label']))
            df = df.withColumn("ECAT",ECAT(df['label']))
            df = df.withColumn("GCAT",GCAT(df['label']))
            df = df.withColumn("MCAT",MCAT(df['label']))
            
            df = df.select('features','CCAT','ECAT','GCAT','MCAT')
        
        else:
            X = X.map(lambda row: [row])
            schema = StructType([StructField("features", VectorUDT(), True)])
            df = X.toDF(schema)
            
        return df

if __name__ == '__main__':        

    #initialize spark session
    spark = SparkSession\
            .builder\
            .appName("Test")\
            .config('spark.sql.warehouse.dir', 'file:///C:/')\
            .getOrCreate()
    sc = spark.sparkContext

    #load data
    X_file = "./data/X_train_vsmall.txt"
    y_file = "./data/y_train_vsmall.txt"
    data = sc.textFile(X_file)
    labels = sc.textFile(y_file)
    
    #process data    
    p1 = preprocessor(bigrams=True,stemming=True,tfidf=True)
    df = p1.transform(data,labels)
  
    print df.show()
    print df.printSchema()
    print df.select('features').rdd.first()

    #save to parquet
    try:
        df.write.save("./data/df_train_vsmall.parquet")
    except:
        pass
    
    #test without labels
    X_test = "./data/X_test_vsmall.txt"
    data = sc.textFile(X_file)
    p2 = preprocessor(bigrams=False,stemming=True,tfidf=True)
    df_no_y = p2.transform(data)
    
    print df_no_y.show()
    print df_no_y.printSchema()
    print df_no_y.select('features').rdd.first()

    try:
        df_no_y.write.save("./data/df_test_vsmall.parquet")
    except:
        pass
    
    spark.stop()