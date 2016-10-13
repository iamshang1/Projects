from preprocessing import preprocessor
from naive_bayes import naive_bayes
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import *
import numpy as np

#initialize spark session
spark = SparkSession\
        .builder\
        .appName("Pipeline_Naive_Bayes")\
        .config("spark.driver.maxResultSize", "3g")\
        .getOrCreate()
sc = spark.sparkContext

#load data
X_train_file = "./data/X_train_large.txt"
y_train_file = "./data/y_train_large.txt"
train_data = sc.textFile(X_train_file)
train_labels = sc.textFile(y_train_file)

X_test_file = "./data/X_test_large.txt"
test_data = sc.textFile(X_test_file)

#process data
preprocessor = preprocessor(bigrams=True,stemming=True,tfidf=True,min_df=3)
train = preprocessor.transform(train_data,train_labels)
test = preprocessor.transform(test_data,train=False)

#fit nb
nb = naive_bayes()
nb.fit(train,labelcol='CCAT')
test = nb.predict(test,outputcol='CCAT_nb')

nb.fit(train,labelcol='ECAT')
test = nb.predict(test,outputcol='ECAT_nb')

nb.fit(train,labelcol='GCAT')
test = nb.predict(test,outputcol='GCAT_nb')

nb.fit(train,labelcol='MCAT')
test = nb.predict(test,outputcol='MCAT_nb')

#select category with highest nb output
def select_output(CCAT_nb,ECAT_nb,GCAT_nb,MCAT_nb):
    if np.argmax([CCAT_nb,ECAT_nb,GCAT_nb,MCAT_nb]) == 0:
        return "CCAT"
    if np.argmax([CCAT_nb,ECAT_nb,GCAT_nb,MCAT_nb]) == 1:
        return "ECAT"
    if np.argmax([CCAT_nb,ECAT_nb,GCAT_nb,MCAT_nb]) == 2:
        return "GCAT"
    if np.argmax([CCAT_nb,ECAT_nb,GCAT_nb,MCAT_nb]) == 3:
        return "MCAT"

select_output = udf(select_output,StringType())
test = test.withColumn("prediction",select_output('CCAT_nb','ECAT_nb','GCAT_nb','MCAT_nb'))

print test.show()
print test.printSchema()

test.select('prediction').toPandas().to_csv('prediction.csv',header=False,index=False)

spark.stop()