from preprocessing import preprocessor
from naive_bayes import naive_bayes
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import *
import numpy as np

#initialize spark session
spark = SparkSession\
        .builder\
        .appName("Test")\
        .config('spark.sql.warehouse.dir', 'file:///C:/')\
        .getOrCreate()
sc = spark.sparkContext

#load data
X_train_file = "./data/X_train_small.txt"
y_train_file = "./data/y_train_small.txt"
train_data = sc.textFile(X_train_file)
train_labels = sc.textFile(y_train_file)

X_test_file = "./data/X_test_small.txt"
y_test_file = "./data/y_test_small.txt"
test_data = sc.textFile(X_test_file)
test_labels = sc.textFile(y_test_file)

#process data
preprocessor = preprocessor(bigrams=True,stemming=True,tfidf=True,min_df=4)
train = preprocessor.transform(train_data,train_labels)
test = preprocessor.transform(test_data,test_labels,train=False)

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

#test set accuracy
def accuracy(prediction,CCAT,ECAT,GCAT,MCAT):
    if prediction == 'CCAT' and CCAT == 1:
        return 1
    elif prediction == 'ECAT' and ECAT == 1:
        return 1
    elif prediction == 'GCAT' and GCAT == 1:
        return 1
    elif prediction == 'MCAT' and MCAT == 1:
        return 1
    else:
        return 0
        
accuracy = udf(accuracy,IntegerType())
test = test.withColumn("accuracy",accuracy('prediction','CCAT','ECAT','GCAT','MCAT'))
testAcc = test.agg({"accuracy": "sum"}).first()[0]/float(test.count()) * 100
print 'Test Set Accuracy: ' + str(testAcc) + '%%'

spark.stop()