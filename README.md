# Credit_Risk_Analysis

## 1.0 Project Overview
The aim of this project is to use `LoanStats_2019Q1.csv` data to evaluate 3 machine learning models by using resampling to determine which is model
is better for predicting credit risk.

## 2.0 Resources
Python 3.7.10
Jupyter notebook 6.3.0

## 3.0 Coding Steps
In general the coding consists of the following steps
* importing libraries for for Numpy and Panda functions,
* importing libraries from sklearn, imblearn
* read in data from the csv data
* convert the columns to binary data
* drop the target column `loan_status`
* assign remaining columns to X as `feature`
* Split the data into training and testing data 
* Oversampling- `RandomOverSampler`
* Train the logistic Regression model using resampled data, `LogisticRegression`
* Fit the model `.fit`
* Get predicted y-value, `.predict`
* Calculate the accuracy score `balanced_accuracy_score` 

## 3.0 Results
The results of the Resampling Models are shown below in Table

