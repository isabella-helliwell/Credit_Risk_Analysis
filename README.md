# Credit_Risk_Analysis

## 1.0 Project Overview
The aim of this project is to use `LoanStats_2019Q1.csv` data to evaluate 3 machine learning models by using resampling to determine which is model
is better for predicting credit risk.

## 2.0 Resources
Python 3.7.10
Jupyter notebook 6.3.0

## 3.0 Coding Steps
In general the coding consists of the following steps
- 1. Importing libraries for for Numpy and Panda functions,
- 2. Importing libraries from sklearn, imblearn
- 3. Read in data from the csv data
- 4. Convert the columns to binary data
* 5. Drop the target column `loan_status`
* 6. Assign remaining columns to X as `feature`
* 7. Split the data into training and testing data 
* 8. Oversampling- `RandomOverSampler`
* 9. Train the logistic Regression model using resampled data, `LogisticRegression`
* 10. Fit the model `.fit`
* 11 .Get predicted y-value, `.predict`
* 12. Calculate the accuracy score `balanced_accuracy_score`
* 13. Display the confusion matrix, `confusion_matrix`
* 14. Print the imbalanced classification report, `classification_report_imbalanced`
* 15. Print the metrics classification report `metrics.classification_report`
 
For the SMOTE oversampling, Undersampling, and Combination sampling, steps 9-15 to be completed
## 3.0 Results
The results of the Resampling Models are shown below in Table

