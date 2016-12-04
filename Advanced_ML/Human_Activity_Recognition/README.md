#Human Activity Recognition

##Overview
Human activity recognition is a type of classification problem that involves attaching 
monitoring devices (e.g. accelerometers, heart rate monitors) to a person while that 
person is performing different activities. This study compares the performance of different
classiﬁers on a human activity recognition dataset.

##Files
The following scripts are used for this project:

 - preprocessing.py - separate the raw accelerometer data by subject and activity
 - visualization.py - plot the raw accelerometer data for a specific category of activity
 - RF/record_fetcher_between_subject.py - create input features for between-subject testing for RF/XGBoost
 - RF/record_fetcher_within_subject.py - create input features for within-subject testing for RF/XGBoost
 - RF/rf.py - random forest classifier model
 - RF/xgb.py - XGBoost classifier model
 - LSTM/record_fetcher_between_subject.py - create input features for between-subject testing for LSTM
 - LSTM/record_fetcher_within_subject.py - create input features for within-subject testing for LSTM
 - LSTM/lstm_between_subject - LSTM classifier model for between-subject testing
 - LSTM/lstm_within_subject - LSTM classifier model for within-subject testing
 - LSTM/record_fetcher_between_subject_raw.py - create input features for between-subject testing for conv-LSTM
 - LSTM/record_fetcher_within_subject_raw.py - create input features for within-subject testing for conv-LSTM
 - LSTM/convlstm_between_subject - conv-LSTM classifier model for between-subject testing
 - LSTM/convlstm_within_subject - conv-LSTM classifier model for within-subject testing
 
Note that the dataset used for this project is not publically available. For queries regarding
this dataset, contact Frederick Maier at fmaier@uga.edu.

##Feature Engineering
Many of the classifiers used in this study are unable to take in the raw accelerometer data directly
as input features. Therefore, we had to hand-engineer features from the raw data to feed into our classifiers.

The primary method used for feature engineering consisted of generating summary statistics over a time period.
For example, the raw accelerometer data was split into ten-second windows, and summary statistics were then 
calculated over each window. The following statistics were calculated for the readings on each axis (X,Y,Z), 
the combined magnitude of all three axis (calculated as sqrt(X^2+Y^2+Z^2)), and the first differential of each
axis and magnitude:

 - Mean and standard deviation
 - 10,25,50,75,90 percentiles
 - Min, max, range, interquartile range
 - Number of median crossings
 - Correlation with other axes

The following normalized demographic data was also included as input features:

 - Continuous: age, height, sex, weight, bmi
 - Binary: overweight, obese

The summary statistics and demographic data was concatenated to form 109 input features that could
be fed directly into a classifier.

##Models
In our study we tested the performance of the following four classifiers:

 - Random Forests
 - XGBoost
 - LSTM networks
 - Convolutional LSTM networks

All models were tested at two levels of granularity. At the first level, activities were
classified as ambulatory or non-ambulatory. At the second level, activities were classified
as non-ambulatory, walking, running, upstairs, or downstairs. At both levels, classifiers
were tested using both within-subject test/train splitting and between-subject test/train
splitting. This yields a total of four test accuracies per model:

 - Within-subject 2-category classification
 - Within-subject 5-category classification
 - Between-subject 2-category classification
 - Between-subject 5-category classification

For additional information about these models, see [paper.pdf](https://github.com/eds-uga/shang-final-project/blob/master/Paper.pdf).
 
##Results
XGBoost had the best performance across all categories of evaluation. LSTMs performed well
in within-subject test/train splitting, doing only slightly worse than XGBoost, but overfit
in between-subject test/train splitting even with simplified models. Convolutional-LSTMs
remained competitive with the other models without any need for human feature engineering,
and achieved the second-best score in between-subject 5 category classification, which is
arguably the most difficult classification in this experiment.

![results](https://github.com/eds-uga/shang-final-project/blob/master/results.png)

##Instructions to Run Models

Before running any of the models, run the following commands to preprocess the dataset:

 - python preprocessing.py \<PATH_TO_DEMOGRAPHICS_CSV\> \<PATH_TO_ACCELEROMETER_DATA_FOLDER\>
 - (optional visualization) python visualization.py <"nonambulatory", "walking", "running", "upstairs", or "downstairs">
 
To run the random forest model, use the following commands:
 - python RF/record_fetcher_between_subject.py <1 for 2-category labels, 0 for 5-category labels>
 - python RF/record_fetcher_within_subject.py <1 for 2-category labels, 0 for 5-category labels>
 - python RF/rf.py

To run the xgboost model, use the following commands:
 - python RF/record_fetcher_between_subject.py <1 for 2-category labels, 0 for 5-category labels>
 - python RF/record_fetcher_within_subject.py <1 for 2-category labels, 0 for 5-category labels>
 - python RF/xgb.py
 
To run the LSTM model for between-subject testing, use the following commands:
 - python LSTM/record_fetcher_between_subject.py <1 for 2-category labels, 0 for 5-category labels>
 - python LSTM/lstm_between_subject.py <1 for 2-category labels, 0 for 5-category labels>
 
To run the LSTM model for within-subject testing, use the following commands:
 - python LSTM/record_fetcher_within_subject.py <1 for 2-category labels, 0 for 5-category labels>
 - python LSTM/lstm_within_subject.py <1 for 2-category labels, 0 for 5-category labels>
 
To run the convolution LSTM model for between-subject testing, use the following commands:
 - python LSTM/record_fetcher_between_subject_raw.py <1 for 2-category labels, 0 for 5-category labels>
 - python LSTM/convlstm_between_subject.py <1 for 2-category labels, 0 for 5-category labels>
 
To run the convolution LSTM model for within-subject testing, use the following commands:
 - python LSTM/record_fetcher_within_subject_raw.py <1 for 2-category labels, 0 for 5-category labels>
 - python LSTM/convlstm_within_subject.py <1 for 2-category labels, 0 for 5-category labels>