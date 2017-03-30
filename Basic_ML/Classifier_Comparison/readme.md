# Digits Classifier Comparison

This exercise compares the speed and accuracy of different classification algorithms on different datasets.

The following classifiers are tested:
 - Naive Bayes
 - Logistic Regression
 - Support Vector Machine using Linear Kernel
 - Support Vector Machine using Radial Basis Function Kernel
 - Random Forest with 100 Trees
 - Adaboost
 - K-Nearest Neighbors using Euclidean Distance
 - K-Nearest Neighbors using Cosine Distance

The following datasets are used for testing:
 - Titanic Survival Dataset (2 classes, 714 samples, 5 features each)
 - MAGIC Gamma Telescope Dataset (2 classes, 19020 samples, 10 features each)
 - Sci-Kit Learn Digits Dataset (10 classes, 1797 samples, 64 features each)
 - MNIST Digits Dataset (10 classes, 60000 samples, 784 features each)

To test for training time, a timer is started before the classifier is created and ended after the classifer is trained.

To test for model accuracy, each dataset is split into a train set consisting of 75% of the samples and a test set containing 25% of the samples.

### Results: Speed

The following table displays the training time (in seconds) of each classification algorithm on each dataset:

|Dataset|Naive Bayes|Logistic|Linear SVM|RBF SVM|Random Forest|Adaboost|Euclidean KNN|Cosine KNN|
|-------|:---------:|:------:|:--------:|:-----:|:-----------:|:------:|:-----------:|:--------:|
|Titanic|0.00       |0.00    |0.03      |0.02   |0.23         |0.12    |0.01         |0.00      |
|MAGIC  |0.01       |0.04    |0.54      |5.53   |5.85         |1.56    |0.51         |1.57      |
|Digits |0.03       |0.15    |0.13      |0.36   |0.39         |2.07    |0.09         |0.02      |
|MNIST  |2.78       |93.52   |98.84     |3253.43|53.42        |932.29  |885.99       |37.36     |

### Results: Accuracy

The following table displays the test set accuracy of each classification algorithm on each dataset:

|Dataset|Naive Bayes|Logistic|Linear SVM|RBF SVM|Random Forest|Adaboost|Euclidean KNN|Cosine KNN|
|-------|:---------:|:------:|:--------:|:-----:|:-----------:|:------:|:-----------:|:--------:|
|Titanic|79.33      |81.01   |80.45     |83.24  |80.45        |81.01   |78.77        |80.45     |
|MAGIC  |72.85      |79.03   |78.72     |87.11  |87.80        |84.02   |83.68        |84.08     |
|Digits |86.00      |97.33   |96.22     |99.78  |98.22        |97.11   |98.67        |98.89     |
|MNIST  |82.72      |92.00   |91.55     |96.09  |96.63        |88.64   |96.64        |97.14     |

### Generalizations

 - Naive Bayes is consistently one of the quickest algorithms, but has the lowest accuracy across the datasets
 - Logistic Regression, Random Forests and K-Nearest Neighbors all perform fairly well (sometimes outperforming more time-expensive algorithms) with fast training speeds
 - Support Vector Machine with Radial Basis Function Kernel performs consistently well in accuracy but takes the longest to train