from pyspark.sql import SparkSession
from pyspark.ml.linalg import SparseVector, VectorUDT, Vectors
from pyspark.sql.types import *
from pyspark.rdd import RDD, PipelinedRDD
import numpy as np
import re
import subprocess
import os
import time

#preprocessor for asm files
class preprocessor_asm(object):
    '''
    preprocessor for .asm files in Microsoft Malware Classification Challenge
    generates the following features:
      - count of lines associated with each prefix (e.g. HEADER, idata, rdata)
      - bigram counts of common opcode commands (e.g., 'push pop' or 'mov jmp')
      - counts of .dlls affected
      - counts of __stdcall, FUNCTION, and call commands
      - counts of other special commands and datatypes, such as dwords and references to db

    parameters:
      - min_df: int
        minimum number of times a word or bigram must appear in training corpus to be included as a feature (default 30)
        
    methods:       
      - transform(X_rdd, y_rdd)
        extract features from .asm files
        parameters:
          - X_rdd: pyspark rdd
            rdd with hashes as rows
          - y_rdd: pyspark rdd (optional)
            rdd with labels as rows
          - train: boolean
            whether or not metadata is the training set. If true, preprocessor will build a dictionary of words
            to include in the features based on the min_df set (default True)
        outputs:
          - spark dataframe with features and label for each row
    
    requires following non-standard python packages
      - numpy
    '''
    def __init__(self,min_df=50):
        self.min_df = min_df

    def _tokenize(self,row):
        '''
        convert raw text into tokens
        '''
        
        #use s3 cli to get binary from hash file
        cmd = 'aws s3 cp s3://eds-uga-csci8360/data/project2/metadata/%s.asm data/metadata/%s.asm' % (row,row)
        subprocess.call(cmd, shell=True)
        print 'downloading binary from s3: %s' % row
       
        #tokenize file
        path = '/home/ec2-user/eatingnails-project2/data/metadata/' + row + '.asm'
        while not os.path.isfile(path):
            time.sleep(1)
            cmd = 'aws s3 cp s3://eds-uga-csci8360/data/project2/metadata/%s.asm data/metadata/%s.asm' % (row,row)
            subprocess.call(cmd, shell=True)
            print 're-downloading binary from s3: %s' % row
        with open(path,'r') as f:
            tokens = re.sub(r'\n|\r|\t',' ',f.read()).split()
        
        #del file for space        
        cmd = 'rm data/metadata/%s.asm' % (row)
        push=subprocess.Popen(cmd, shell=True, stdout = subprocess.PIPE)
        
        #filter out important words
        prefixes = ['HEADER:','.text:','.Pav:','.idata:','.data:','.bss:','.rdata:','.edata:','.rsrc:','.tls:','.reloc:']
        opcodes = ['jmp', 'mov', 'retf', 'push', 'pop', 'xor', 'retn', 'nop', 'sub', 'inc', 'dec', 'add', \
                   'imul', 'xchg', 'or', 'shr', 'cmp', 'call', 'shl', 'ror', 'rol', 'jnb']
        keywords = ['.dll','std::',':dword']
        keywords2 = ['FUNCTION','call']
        opcode_list = []
        filtered = []
        filtered.append('HEADER:') # add Header because first word of each file is skipped
        for i in xrange(1,len(tokens)-1): #skip first and last word so we can check previous and following word during iter
            if any(tokens[i]==opcode for opcode in opcodes):
                for opcode in opcodes:
                    if tokens[i]==opcode:
                        filtered.append(opcode)
                        opcode_list.append(opcode)
                        break
            if any(prefix in tokens[i] for prefix in prefixes):
                for prefix in prefixes:
                    if prefix in tokens[i]:
                        filtered.append(prefix)
                        break
            elif any(keyword in tokens[i] for keyword in keywords):
                filtered.append(tokens[i])
            elif any(tokens[i]==keyword for keyword in keywords2):
                bigram = tokens[i] + ' ' + tokens[i+1]
                filtered.append(bigram)
            elif tokens[i] == '__stdcall':
                bigram = tokens[i] + ' ' + tokens[i+1].partition("(")[0]
                filtered.append(bigram)
                filtered.append(tokens[i-1])
            elif tokens[i] == 'db' and tokens[i+1][0] == "'":
                bigram = tokens[i] + ' ' + tokens[i+1]
                filtered.append(bigram)

        #add opcode bigrams
        for i in range(len(opcode_list)-1):
            bigram = opcode_list[i] + ' ' + opcode_list[i+1]
            filtered.append(bigram)

        return (row,filtered)
    
    def _term_frequency(self,row):
        '''
        convert row of word tokens into sparse vector of terms frequencies
        '''
        sparse_dic = {}
        df_dic = {}
        for word in row[1]:
            if word in self.dictionary:
                if self.dictionary[word] in sparse_dic:
                    sparse_dic[self.dictionary[word]] += 1.
                else:
                    sparse_dic[self.dictionary[word]] = 1.         
        tf = SparseVector(len(self.dictionary),sparse_dic)
        return (row[0],tf)

    def transform(self,X_rdd,y_rdd=None,train=True):
        '''
        given X RDD (and optionally y RDD), output dataframe with term frequency feature vector and labels
        '''    
        #check input type
        if type(X_rdd) != RDD:
            raise TypeError("Arguments must be pySpark RDDs")
        if y_rdd and type(y_rdd) != RDD:
            raise TypeError("Arguments must be pySpark RDDs")
        
        #word tokenization
        X = X_rdd.map(self._tokenize).cache()
        
        #create dictionary of words
        if train:
            self.dictionary = X.map(lambda row: row[1]).flatMap(lambda word: word).map(lambda word: (word,1)).reduceByKey(lambda acc, w: acc + w).filter(lambda x: x[1]>=self.min_df).collectAsMap()
            self.dictionary = dict(zip(self.dictionary,xrange(len(self.dictionary))))

        #create word vectors
        X = X.map(self._term_frequency)
        
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
    preprocessor1 = preprocessor_asm()
    data = preprocessor1.transform(X_file,y_file)
    
    print data.show()

    #save to parquet
    try:
        data.write.save("./data/train_small_asm.parquet")
    except:
        pass
        
    #paths to test data
    X_file = "./data/X_test_small.txt"
    y_file = "./data/y_test_small.txt"
    X_file = sc.textFile(X_file,20)
    y_file = sc.textFile(y_file,20)
    
    #preprocess data   
    data = preprocessor1.transform(X_file,y_file,train=False)
    
    print data.show()
    
    #save to parquet
    try:
        data.write.save("./data/test_small_asm.parquet")
    except:
        pass