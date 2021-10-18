# Credit_Risk_Analysis

## 1.0 Project Overview
The aim of this project is to use `LoanStats_2019Q1.csv` data to evaluate 3 machine learning models by using resampling to determine which is model
is better for predicting credit risk.
Next a comparison is made for two new machine learning models that reduce bias, to predict credit risk.

## 2.0 Resources
Python 3.7.10
Jupyter notebook 6.3.0

## 3.0 Coding Steps
In general the coding consists of the following steps
1. Importing libraries for for Numpy and Panda functions,
2. Importing libraries from sklearn, imblearn
3. Read in data from the csv data
4. Convert the columns to binary data
5. Drop the target column `loan_status`
6. Assign remaining columns to X as `feature`
7. Split the data into training and testing data 
8. Oversampling- `RandomOverSampler`
9. Train the logistic Regression model using resampled data, `LogisticRegression`
10. Fit the model `.fit`
11 .Get predicted y-value, `.predict`
12. Calculate the accuracy score `balanced_accuracy_score`
13. Display the confusion matrix, `confusion_matrix`
14. Print the imbalanced classification report, `classification_report_imbalanced`
15. Print the metrics classification report `metrics.classification_report`
 
For the SMOTE oversampling, Undersampling, and Combination sampling, steps 9-15 are repeated with changes to functions where applicable.
For the Ensemble Algorithms, use `BalanceRandomForestClassifier` to resample the data with 100 estimators, and fit the model, and follow the steps 
## 3.0 Results
### 3.1 Results-Sampling

![image](https://user-images.githubusercontent.com/85843030/137648423-aa792756-bb6c-4cd6-ae27-55e351e2d406.png)

<ins>Observations Naive Random Oversampling:</ins>
1. Total samples used: 17205
2. Out of 101 samples, 71 were predicted correctly to be high risk loans, 30 were False Negative (high risk loan but they are not)
3. 6334 were predicted correctly to be low risk loans, False Positive
4. 10770 were predicted incorrectly to be low risk loans, hence True Negative


<ins>Observations SMOTE Oversampling:</ins>
1. Total samples used: 17205
2. Out of 101 samples, 64 were predicted correctly to be high risk loans and
   37 were predicted wrongly to be high risk loans,hence False Negative
3. 5286 were predicted correctly to be low risk loans, False Positive
4. 11818 were predicted incorrectly to be low risk loans, hence True Negative


<ins>Observations Cluster Centroids Undersampling:</ins>
1. Total samples used: 17205
2. Out of 101 samples, 70 were predicted correctly to be high risk loans, 31 were False Negative (high risk loan but they are not)
3. 10324 were predicted correctly to be low risk loans, False Positive
4. 6780 were predicted incorrectly to be low risk loans, hence True Negative


<ins>Observations Combination Sampling:</ins>
1. Total samples used: 17205
2. Out of 101 samples, 78 were predicted correctly to be high risk loans, 23 were False Negative (high risk loan but they are not)
3. 7187 were predicted correctly to be low risk loans, False Positive
4. 9917 were predicted incorrectly to be low risk loans, hence True Negative




### 3.2 Results-Ensemble Classifiers

![image](https://user-images.githubusercontent.com/85843030/137651568-f7e44c5b-2135-4ed3-ab98-128c800c7e0d.png)

![image](https://user-images.githubusercontent.com/85843030/137652181-a6056e10-b52b-4697-a9fa-6f67f0eaeae9.png)

<ins>Observations Balanced Random Forest Classifier:</ins>
1. Total samples used: 17205
2. Out of 101 samples, 71 were predicted correctly to be high risk loans, 30 were False Negative (high risk loan but they are not)
3. 2153 were predicted correctly to be low risk loans, False Positive
4. 14951 were predicted incorrectly to be low risk loans, hence True Negative


<ins>Observations EasyEnsemble Classifier:</ins>
1. Total samples used: 17205
2. Out of 101 samples, 93 were predicted correctly to be high risk loans, 8 were False Negative (high risk loan but they are not)
3. 983 were predicted correctly to be low risk loans, False Positive
4. 16121 were predicted incorrectly to be low risk loans, hence True Negative

## 4.0 Summary

### 4.1 Summary-Sampling

![image](https://user-images.githubusercontent.com/85843030/137652237-1d74d41e-a61f-4e5b-90cc-515601210660.png)

<ins>Observations Summary-Sampling</ins>
1. Lowest balanced accuracy score is 54% using Under Sampling
2. Highest balanced accuracy score is 68% using Combination Sampling
3. All models have 0.01 precision
4.  The highest Sensitivity score is 77% for combination Sampling
5.  The lowest Sensitivity score is 63% for the SMOTE-Oversampling
6.  Highest f1-score is 0.012 for the SMOTE Oversampling
7.  The lowest f1-score is for the Cluster Centroid Undersampling





### 4.2 Summary-Ensemble Classifiers

![image](https://user-images.githubusercontent.com/85843030/137790478-dd211d71-57e8-418d-b6eb-8a2afd573838.png)



<ins>Observations Summary-Ensample Classifiers</ins>
1. Lowest balanced accuracy score is 79% for Blanced Random Forest Classifier
2. Highest balance accurace score is 93% for the Easy Ensemble Classifier
3. Highest sensitivity score `rec` is 92% for Easy Ensemble Classifier
4. Highest precision score `pre` is 0.09 for Easy Ensemble Classifier
5. Highest f1-score is 0.16 for the Easy Ensemble Classifier





