#Spark Text Classfication

##Overview

The following repository runs text classification on the Reuters news articles corpus using Naive Bayes
classification and optionally Logistic Regression. Each article is classified into the one category that
it most likely belongs to. Only the following categories are considered:

 - CCAT: Corporate/Industrial
 - ECAT: Economics
 - GCAT: Government/Social
 - MCAT: Markets

The package includes the following scripts. These scripts have been designed to run on a distributed cluster using PySpark.

 - naive_bayes.py - class definition for naive bayes classifier
 - pipeline_nb.py - pipeline for testing naive bayes classifier on small dataset
 - pipeline_nb_large.py - pipeline for predicting full dataset using naive bayes classifier
 - preprocessing.py - class definition for data preprocessor

##Model Details

###Data Preprocessing

The following data preprocessing steps are applied to the raw input data. These steps are applied in the order listed:

- Eliminate all punctuation, lowercase all words.
- Remove stopwords. The python NLTK stopwords corpus is used as the set of stop words to remove.
- Stem words. The python NLTK snowball stemmer is used to stem words.
- Create tokens using unigrams and bigrams of word combinations.
- Remove any tokens that appear in the training corpus less than three times.
- Convert tokens into numerical vector using based on term frequency - inverse document frequency.

The final numerical vector is used as the input to the naive bayes model.

###Naive Bayes Model

The Naive Bayes model uses standard Naive Bayes classification with Laplace (add one) smoothing and log probabilities for numerical
stability. Because each article can have multiple labels, four separate Naive Bayes models are trained, one for each category.
For each article, the category with the highest log probability sum of its input vector is chosen as the category for the article. 
 
##Instructions for Running Naive Bayes Model

Flintrock was used to launch Amazon EC2 instances to run our program. The following arguments were used to launch Flintrock:

flintrock launch *cluster name* --num-slaves 4 --no-install-hdfs --spark-version 2.0.0 --ec2-key-name *key name* --ec2-identity-file 
*path to identity file* --ec2-user ec2-user --ec2-region us-west-1 --ec2-ami ami-31490d51 --ec2-instance-type m4.2xlarge

Once the EC2 machines are set up, each machine must have the following Python packages installed:
- numpy
- pandas
- nltk

In addition, the NLTK package requires additional setup. On each machine, you must open the Python shell, import nltk, then run
the nltk.download() command and download the 'book' corpora.

Next, run the following commands on each machine from the /home/ec2-user directory to get the necessary project files:

- git clone https://github.com/eds-uga/team-1-project1.git (clone repository)
- cd team-1-project1/data (navigate to data directory)
- wget https://s3.amazonaws.com/eds-uga-csci8360/data/project1/X_train_large.txt (download full X_train set)
- wget https://s3.amazonaws.com/eds-uga-csci8360/data/project1/y_train_large.txt (download full y_train set)
- wget https://s3.amazonaws.com/eds-uga-csci8360/data/project1/X_test_large.txt (download full X_test set)

Finally, run the following from the /home/ec2-user/team-1-project1 directory on the Master:

- spark-submit --master *Master Spark IP* --deploy-mode client --driver-memory 30g --executor-memory 7g --num-executors 16 --executor-cores 2 --files preprocessing.py,naive_bayes.py pipeline_nb_large.py