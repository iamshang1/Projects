from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import *
import numpy as np

      
#nb classifier
class naive_bayes(object):
    def __init__(self):
        self.num_docs = 0
        self.num_features = 0

    def fit(self, dataframe, featurescol='features', labelcol='label'):

        # get the size of vocabulary / doc numbers
        if not self.num_docs or not self.num_features:
            data = dataframe.select(featurescol).rdd
            self.num_docs = data.count()
            self.num_features = len(data.first()[0].toArray())

        # count doc numbers in label class / prior likelihood
        data = dataframe.filter(dataframe[labelcol]==1)
        self.prior = float(data.count())/float(self.num_docs)

        # calculate post likelihood
        sum_bycol = data.select(featurescol).rdd.aggregate(np.zeros(self.num_features),lambda acc,row:acc+row[0].toArray(),lambda acc,row:acc+row)
        sum_total = sum_bycol.sum()

        # calculate the final prob
        self.wordprobs = np.log((sum_bycol + 1)/(sum_total + self.num_features))
        
    def predict(self,df,featurecol='features',outputcol='prediction'):
        dot = udf(self._dot,FloatType())
        data = df.withColumn(outputcol,dot(featurecol))
        return data

    def _dot(self,features):
        f = features.toArray()        
        d = float(np.log(self.prior) + np.dot(f,self.wordprobs))
        return d

if __name__ == '__main__':
        
	#initialize spark session
	spark = SparkSession\
	        .builder\
	        .appName("Test")\
	        .config('spark.sql.warehouse.dir', 'file:///C:/')\
	        .getOrCreate()
	sc = spark.sparkContext

	#load saved dataframe
	df = spark.read.load("./data/df_train_vsmall.parquet")     
	df_test = spark.read.load("./data/df_test_vsmall.parquet") 

	#fit nb
	nb = naive_bayes()
	nb.fit(df,labelcol='GCAT')

	out = nb.predict(df_test)
	print out.show()

	spark.stop()