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
* Base dataframe created to assess data file including null values, data types and column names
* Processed value counts on y to determine data distribution for bias
* 'Objects' were separated into another dataframe for OneHotEncoding including:
  - payment_type
  - employment_status
  - housing_status
  - source
  - device_os
*  Dataframes were merged into a new dataframe with the encoded data
*  y was dropped
*  Data was normalized utilizing StandardScaler
*  Data was fit to a new dataframe created
*  Correlation graph created on data before PCA was applied
*  PCA model applied for feature selection and includes an explained variance ratio array
*  Correlation graph created on data post-PCA processing and reduced dataframe saved

### Modeling


### Credit
- Jack Hoffmann
- Nicholas Merz
- Mark Moore
- Angelina Prema

### Dataset License
CC BY-NC-SA
   *Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International*

