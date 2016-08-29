# Feature Selection

This exercise compares the effectiveness of different methods of feature selection on a randomly generated dataset.

The dataset generated has 20000 samples and 30 features. Only 10 of the 30 features contain information useful for prediction.

A support vector machine with an radial bias function kernel is used to fit the data after each feature selection method.

The following feature selection methods are tested:
 - Univariate Feature Selection
 - Recursive Feature Selection
 - L1 Lasso Feature Selection
 - Tree-Based Feature Selection
 - Principal Component Analysis

## Results

#### Dataset before any feature selection preprocessing

Training Time: 19.3738

Test Accuracy: 96.15

#### Univariate Feature Selection

![eda1](https://github.com/iamshang1/Projects/blob/master/ML_Exercises/Feature_Selection/ufs.png)

The following statistics are for the model selected using the elbow method:

Features Selected: 12

Preprocessing Time: 0.0084

Training Time: 13.1206

Test Accuracy: 95.90

#### Recursive Feature Selection

![eda1](https://github.com/iamshang1/Projects/blob/master/ML_Exercises/Feature_Selection/rfe.png)

The following statistics are for the model selected using the elbow method:

Features Selected: 10

Preprocessing Time: 1.5056

Training Time: 5.7457

Test Accuracy: 96.75

#### L1 Lasso Feature Selection

![eda1](https://github.com/iamshang1/Projects/blob/master/ML_Exercises/Feature_Selection/l1.png)

The following statistics are for the model selected using the elbow method:

Features Selected: 11

Preprocessing Time: 0.1402

Training Time: 11.9274

Test Accuracy: 95.35

#### Tree-Based Feature Selection

![eda1](https://github.com/iamshang1/Projects/blob/master/ML_Exercises/Feature_Selection/tree.png)

The following statistics are for the model selected using the elbow method:

Features Selected: 9

Preprocessing Time: 0.1509

Training Time: 4.3877

Test Accuracy: 96.85

#### Principal Component Analysis

![eda1](https://github.com/iamshang1/Projects/blob/master/ML_Exercises/Feature_Selection/pca.png)

The following statistics are for the model selected using the elbow method:

Features Selected: 11

Preprocessing Time: 0.0266

Training Time: 20.3374

Test Accuracy: 96.20

#### Bringing everything together

|                  |No Preprocessing|Univariate|Recursive|L1 Lasso|Tree Based|PCA     |
|------------------|:--------------:|:--------:|:-------:|:------:|:--------:|:------:|
|Features Selected |30              |12        |10       |11      |9         |11      |
|Preprocessing Time|0.0000          |0.0084    |1.5056   |0.1402  |0.1509    |0.0266  |
|Training Time     |19.3738         |13.1206   |5.7457   |11.9274 |4.3877    |20.3374 |
|Test Accuracy     |96.15           |95.90     |96.75    |95.35   |96.85     |96.20   |

## Conclusions

 - For datasets with uncorrelated features, using feature elimination methods can significantly reduce the time required to train a model without reducing test accuracy.
 - Recursive Feature Elimination and Tree Based Feature Selection yielded the greatest increases in training speed.
 - Principal Component Analysis does not reduce the model training time. This may be because PCA retains the variance of all data including that from junk features.