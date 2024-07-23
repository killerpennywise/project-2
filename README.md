# AI Bootcamp Group Project 2

## Bank Account Fraud Analysis

### Description
In attempting to determine fraud in credit card applications, the Bank Account Fraud (BAF)'s main dataset allowed us to perform data analysis, data preprocessing and fraud detection ML modeling.

### Goal
Create a ML model that can test data to determine fraud in credit card applications with a high level of accuracy.

#### Dataset
[Bank Account Fraud Dataset Suite (NeurlPS)](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022?select=Base.csv)
*The dataset was resampled to 200,000 rows due to github file size limitations.*

### Required Git Installations
> - pip install pandas
> - pip install scikit-learn
> - pip install matplotlib
> - pip install imbalanced-learn
  
### Dependencies
* import pandas as pd
* from sklearn.preprocessing import StandardScaler, OneHotEncoder
* from sklearn.compose import ColumnTransformer
* from sklearn.pipeline import Pipeline
* from sklearn.impute import SimpleImputer
* from sklearn.decomposition import PCA
* import seaborn as sns
* import matplotlib as plt
* from sklearn.model_selection import train_test_split
* from sklearn.metrics import balanced_accuracy_score, accuracy_score, classification_report
* from imblearn.over_sampling import SMOTE
* from sklearn.ensemble import RandomForestClassifier

### Installation
Steps to install and use:
* Open github [repository](https://github.com/killerpennywise/project-2/tree/main)
* Click green code button
* Scroll down to download zip file
* Open local downloads folder and right click project-2-main.zip file
* Click and extract all to desired location
* Open newly extracted project-2-main folder
* Run the data_prep file
* Next, run the model file

### Preprocessing
*  Base dataframe created to assess data file including null values, data types and column names
*  Processed value counts on ```y``` to determine data distribution for bias
*  'Objects' were separated into another dataframe for OneHotEncoding including:
  - payment_type
  - employment_status
  - housing_status
  - source
  - device_os
*  Dataframes were merged into a new dataframe with the encoded data
*  ```y``` was dropped
*  Data was normalized utilizing StandardScaler
*  Data was fit to a new dataframe created
*  Correlation graph created on data before PCA was applied
*  PCA model applied for feature selection and includes an explained variance ratio array
*  Correlation graph created on data post-PCA processing and reduced dataframe saved

### Modeling
*  From the PCA.csv file, a new dataframe is created
*  Features are separated from ```y```
*  The dataset was split to training and testing with 20% set for the testing size and a random state of 42
*  The RandomForestClassifier is initiated with 100 estimators and a random state of 42
*  RandomForestClassifier is applied to the training data
*  Predict was applied to the data to calculate and print a testing, training and model accuracy scores
   - Testing Score: 50%
   - Training Score: 99.9%
   - Model Accuracy Score: 98.8%
* A classification report was then printed for further evaluation
  
  >                   precision    recall  f1-score   support
  >
  >                0       0.99      1.00      0.99     39537
  >                1       0.00      0.00      0.00       463
  >
  >         accuracy                           0.99     40000
  >        macro avg       0.49      0.50      0.50     40000
  >     weighted avg       0.98      0.99      0.98     40000

*  Synthetic Minority Over-sampling TEchnique (SMOTE) to balance to dataset
*  This data was resampled to balance Fraud and Non-Fraud transactions
*  The balanced data was split to training and testing with 20% set for the testing size and a random state of 42
*  RandomForestClassifier was trained on the resampled data
*  Predict was applied to the testing data to calculate and print a testing accuracy score
*  Predict was applied to the training data to calculate and print a training accuracy score
*  A classification report was then printed for further evaluation
*  Lastly, the data was split into training and testing data without SMOTE
   - This achieved a Testing Score of 99.7%
*  The final classification report demostrates a more balanced model and improved performance
  >                   precision    recall  f1-score   support
  >  
  >                0       1.00      0.98      0.99     39627
  >                1       0.98      1.00      0.99     39481
  >         accuracy                           0.99     79108
  >        macro avg       0.99      0.99      0.99     79108
  >     weighted avg       0.99      0.99      0.99     79108

### Credit
- Jack Hoffmann
- Nicholas Merz
- Mark Moore
- Angelina Prema

### Dataset License
CC BY-NC-SA
   *Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International*
