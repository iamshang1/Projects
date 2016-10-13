from pyspark.sql import SparkSession
from pyspark.ml.linalg import SparseVector, VectorUDT, Vectors
from pyspark.sql.types import *
from pyspark.rdd import RDD, PipelinedRDD
import numpy as np
import subprocess
import os
import time

#preprocessor for bytes files
class preprocessor_bytes(object):
    '''
    preprocessor for .bytes files in Microsoft Malware Classification Challenge
    generates the following features:
      - counts of hexademical bigrams (e.g., '53 8F')

    parameters:
      - most_common_bigrams: boolean
        whether or not to filter out everything but the top 1000 bigrams from each class (default True)
        
    methods:
      - transform(X_rdd, y_rdd)
        extract features from .bytes files
        parameters:
          - X_rdd: pyspark rdd
            rdd with hashes as rows
          - y_rdd: pyspark rdd (optional)
            rdd with labels as rows
        outputs:
          - spark dataframe with features and label for each row
    
    requires following non-standard python packages
      - numpy
    '''
    def __init__(self,most_common_bigrams=True):
        self.most_common_bigrams = most_common_bigrams
        self.num_features = 65536
        if self.most_common_bigrams:
            self.bigrams = list(np.load('./data/most_common_bigrams.npy'))
            self.num_features = len(self.bigrams)
    
    def _term_frequency(self,row):
        '''
        get term frequency of 4-hexadecimal-character words
        '''

        #use s3 cli to get binary from hash file
        cmd = 'aws s3 cp s3://eds-uga-csci8360/data/project2/binaries/%s.bytes data/binaries/%s.bytes' % (row,row)
        subprocess.call(cmd, shell=True)
        print 'downloading binary from s3: %s' % row
       
        #tokenize file
        path = '/home/ec2-user/eatingnails-project2/data/binaries/' + row + '.bytes'
        while not os.path.isfile(path):
            time.sleep(1)
            cmd = 'aws s3 cp s3://eds-uga-csci8360/data/project2/binaries/%s.bytes data/binaries/%s.bytes' % (row,row)
            subprocess.call(cmd, shell=True)
            print 're-downloading binary from s3: %s' % row
        with open(path,'r') as f:
            tokens = [word for word in f.read().replace('\n', ' ').split() if len(word)==2 and word!="??"] 
       
        #del file for space        
        cmd = 'rm data/binaries/%s.bytes' % (row)
        push=subprocess.Popen(cmd, shell=True, stdout = subprocess.PIPE)

        #get hex bigrams and convert hex to int
        tf = np.zeros(65536)
        for idx in xrange(len(tokens)-1):
            word = tokens[idx]+tokens[idx+1]
            converted = int(word,16)
            tf[converted] += 1
        if self.most_common_bigrams:
            tf = tf[self.bigrams]
        return (row, tf)

    def transform(self,X_rdd,y_rdd=None):
        '''
        given X RDD (and optionally y RDD), output dataframe with term frequency feature vector and labels
        '''    
        #check input type
        if type(X_rdd) != RDD:
            raise TypeError("Arguments must be pySpark RDDs")
        if y_rdd and type(y_rdd) != RDD:
            raise TypeError("Arguments must be pySpark RDDs")
        
        #get term frequencies
        X = X_rdd.map(self._term_frequency).cache()
        
        #convert to sparse
        X = X.map(lambda (hash,features): (hash,SparseVector(self.num_features,np.nonzero(features)[0],features[features>0])))

        #check if labels exist
        if y_rdd:
            #combine X and y into single dataframe
            X = X.zipWithIndex().map(lambda r: (r[1],r[0]))
            y = y_rdd.zipWithIndex().map(lambda r: (r[1],r[0]))
            data = X.join(y).map(lambda (idx,((hash,features),label)): (hash,features,label))
            schema = StructType([StructField('hash',StringType(),True),StructField('features',VectorUDT(),True),StructField('label',StringType(),True)])
            data = data.toDF(schema)
            data = data.withColumn('label',data.label.cast(DoubleType()))
        
        else:
            schema = StructType([StructField('hash',StringType(),True),StructField("features", VectorUDT(), True)])
            data = X.toDF(schema)
            
        return data

if __name__ == '__main__':        

    #initialize spark session
    spark = SparkSession\
            .builder\
            .appName("Test")\
            .getOrCreate()
    sc = spark.sparkContext
    
    #paths to training data
    X_file = "./data/X_train_small.txt"
    y_file = "./data/y_train_small.txt"
    X_file = sc.textFile(X_file,20)
    y_file = sc.textFile(y_file,20)
    
    #preprocess data
    preprocessor1 = preprocessor_bytes()
    data = preprocessor1.transform(X_file,y_file)
    
    print data.show()

    #save to parquet
    try:
        data.write.save("./data/train_small_bytes.parquet")
    except:
        pass
        
    #paths to test data
    X_file = "./data/X_test_small.txt"
    y_file = "./data/y_test_small.txt"
    X_file = sc.textFile(X_file,20)
    y_file = sc.textFile(y_file,20)
    
    #preprocess data
    preprocessor2 = preprocessor_bytes()
    data = preprocessor2.transform(X_file,y_file)
    
    print data.show()
    
    #save to parquet
    try:
        data.write.save("./data/test_small_bytes.parquet")
    except:
        pass