# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 09:38:37 2019

@author: jflygare
"""

#XGBoost Tutorial
#https://cambridgespark.com/getting-started-with-xgboost/

import pandas as pd
import os
import xgboost as xgb

os.chdir('C:/Users/jflygare/Documents/ML_Projects/Python/')


df = pd.read_excel('datasets/default of credit card clients.xls', header=1, index_col=0)
df.head()
df.shape




def process_categorical_features(df): 
    dummies_education = pd.get_dummies(df.EDUCATION, 
    prefix="EDUCATION", drop_first=True) 
    dummies_marriage = pd.get_dummies(df.MARRIAGE, CSt20xup!
    prefix="MARRIAGE", drop_first=True) 
    df.drop(["EDUCATION", "MARRIAGE"], axis=1, inplace=True)
    return pd.concat([df, dummies_education, dummies_marriage], axis=1)
df = process_categorical_features(df)
df.head()

y = df['default payment next month']
X = df[[col for col in df.columns if col!="default payment next month"]]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.25, random_state=42)
print("Size of train dataset: {} rows".format(X_train.shape[0]))
print("Size of test dataset: {} rows".format(X_test.shape[0]))

import xgboost as xgb

