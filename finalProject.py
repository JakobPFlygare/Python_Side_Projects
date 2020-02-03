# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 13:14:08 2019

@author: jflygare
"""

## Load packages and change working directory
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import numpy.random as nr
import math
from dateutil.relativedelta import relativedelta
import datetime
import os
import scipy.stats as ss

# make plots appear inline in the notebook
%matplotlib inline 

os.chdir('C:/Users/jflygare/Documents/datasets for courses/ML_course_Python/Final Project/')

import myFunctions

## Read files

dfAveMonthSpend = pd.read_csv("AW_AveMonthSpend.csv")
dfBikeBuyer = pd.read_csv("AW_BikeBuyer.csv")
dfCustomers= pd.read_csv("AdvWorksCusts.csv")

##Summary statistics for first set of questions

dfAveMonthSpend.describe()
dfAveMonthSpend["AveMonthSpend"].median()
        
dfBikeBuyer["BikeBuyer"].value_counts()
myFunctions.plot_bars(dfBikeBuyer,["BikeBuyer"])

dfCustomers.describe()
dfCustomers['YearlyIncome'].groupby(dfCustomers['Occupation']).describe()
dfCustomers['YearlyIncome'].groupby(dfCustomers['Occupation']).median()

dfCustomers = pd.merge(dfCustomers,
                        dfAveMonthSpend[["CustomerID","AveMonthSpend"]],
                        left_on = "CustomerID",
                        right_on = "CustomerID",
                        how = "left")

dfCustomers.dtypes
#Age + Gender vs Spending
def ageGroups(birthdate):
    end_date = datetime.date(1998,1,1)
    #start_date = datetime.datetime.strptime(birthdate, '%Y-%m-%d')
    start_date = datetime.datetime.strptime(birthdate, '%m/%d/%Y')
    age = relativedelta(end_date, start_date).years
    if age > 55: 
        return '>55'
    elif age >= 25 and age <= 45: 
        return '25-45'
    elif age < 25: 
        return '<25'
    else: return 'other'
    
dfCustomers["ageGroup"] = dfCustomers["BirthDate"].map(ageGroups)
cols = ["AveMonthSpend","ageGroup","Gender"]
monthSpend_Age = dfCustomers[cols].groupby(['ageGroup','Gender']).mean()

#Marital statuts  vs spending
dfCustomers['AveMonthSpend'].groupby(dfCustomers["MaritalStatus"]).median()

#nr of cars vs spending
cars = np.where(dfCustomers['NumberCarsOwned'] >= 3 , '>3', 
                np.where(dfCustomers['NumberCarsOwned'] == 0,'0','1-2'))

dfCustomers.groupby(cars)['AveMonthSpend'].median()

#Male vs female
dfCustomers.groupby("Gender")["AveMonthSpend"].median()
dfCustomers.groupby("Gender")["AveMonthSpend"].describe()

#Children at home
kids = np.where(dfCustomers["NumberChildrenAtHome"] > 0,'>0 kids','0 kids')
dfCustomers.groupby(kids)['AveMonthSpend'].median()

## Mergin in if they bought bikes
dfCustomers = pd.merge(dfCustomers,
                        dfBikeBuyer,
                        left_on = "CustomerID",
                        right_on = "CustomerID",
                        how = "left")
#Bikebuyer vs income
dfCustomers.groupby("BikeBuyer")["YearlyIncome"].median()
dfCustomers.groupby("BikeBuyer")["NumberCarsOwned"].median()

#Occupation for bike buyers
myFunctions.plot_bars(dfCustomers.loc[dfCustomers['BikeBuyer'] == 1],
                      ["Occupation"])

#Gender vs bike buyers
dfCustomers.groupby("Gender")["BikeBuyer"].mean()
#Marital status vs bike buyer
dfCustomers.groupby("MaritalStatus")["BikeBuyer"].mean()


######################### CLASSFICATION ################################
# Objective - Classification model if customer is a bike buyer
# Note: No information on AveMonthSpend can be used
dfTest = pd.read_csv("AW_Test.csv")
originalTrain = dfCustomers
originalTest = dfTest

dfCustomers.dtypes

#Class imbalance
bike_buyers = dfCustomers['BikeBuyer'].value_counts()
print(bike_buyers)

#Dropping obvious columns and where there's a lot of NA's
dfCustomers.isna().sum()
dfTest.isna().sum()

colDrop = ['CustomerID', 'Title','FirstName','MiddleName','LastName',
           'Suffix','AddressLine1','AddressLine2','PhoneNumber']

dfCustomers = dfCustomers.drop(columns=colDrop)

## Visualizing data

#Numerical features
num_cols = ['NumberCarsOwned', 'NumberChildrenAtHome', 'TotalChildren',
            'YearlyIncome']
myFunctions.plot_box(dfCustomers, num_cols,'BikeBuyer')
myFunctions.plot_violin(dfCustomers,num_cols,'BikeBuyer')
myFunctions.plot_histogram(dfCustomers,num_cols)

#Categorical features
dfCustomers.dtypes
cat_cols = ['City','StateProvinceName','CountryRegionName',
            'Education','Occupation','Gender','MaritalStatus',
            'HomeOwnerFlag','ageGroup']

myFunctions.plot_frequency(dfCustomers,cat_cols,"BikeBuyer",labels = ["no Bike","Bike"])
dfCustomers.dtypes

## Data preparation

#Dropping variables which contains too many variations
colDrop2 = ['City', 'StateProvinceName','PostalCode','BirthDate']
colDropAll = colDrop + colDrop2
dfCustomers = dfCustomers.drop(columns=colDrop2)
dfCustomers = dfCustomers.drop(columns = ['dummy'])
dfCustomers = dfCustomers.drop(columns = ['AveMonthSpend'])


dfCustomers['logIncome'] = np.log(dfCustomers['YearlyIncome'])
myFunctions.plot_histogram(dfCustomers,['logIncome'])

#Logistic regression
from sklearn import preprocessing
import sklearn.model_selection as ms
from sklearn import linear_model
import sklearn.metrics as sklm

labels = np.array(dfCustomers['BikeBuyer'])

cat_cols = ['CountryRegionName','Education','Occupation','Gender',
            'MaritalStatus']

Features = myFunctions.encode_string(dfCustomers['ageGroup'])

for col in cat_cols:
    temp = myFunctions.encode_string(dfCustomers[col])
    Features = np.concatenate([Features, temp], axis = 1)

print(Features.shape)
print(Features[:2, :])  

#Numeric features -> numpy arrays
num_features = ['NumberCarsOwned','NumberChildrenAtHome','TotalChildren',
                'YearlyIncome','logIncome']
Features = np.concatenate([Features, np.array(dfCustomers[num_features])], axis = 1)
print(Features.shape)
print(Features[:2, :])   

## Randomly sample cases to create independent training and test data
nr.seed(9988)
indx = range(Features.shape[0])
indx = ms.train_test_split(indx, test_size = 0.3)
X_train = Features[indx[0],:]
y_train = np.ravel(labels[indx[0]])
X_test = Features[indx[1],:]
y_test = np.ravel(labels[indx[1]])

#Scale numeric features
scaler = preprocessing.StandardScaler().fit(X_train[:,24:])
X_train[:,24:] = scaler.transform(X_train[:,24:])
X_test[:,24:] = scaler.transform(X_test[:,24:])
X_train[:2,]

# Create the logistic model
logistic_mod = linear_model.LogisticRegression() 
logistic_mod.fit(X_train, y_train)

print(logistic_mod.intercept_)
print(logistic_mod.coef_)

probabilities = logistic_mod.predict_proba(X_test)
print(probabilities[:15,:])


scores = myFunctions.score_model(probabilities, 0.5)
print(np.array(scores[:15]))
print(y_test[:15])

myFunctions.print_metrics(y_test, scores)  

myFunctions.plot_auc(y_test, probabilities)    

# Weighted model due to class imbalance
logistic_mod = linear_model.LogisticRegression(class_weight = {0:0.5, 1:0.5}) 
logistic_mod.fit(X_train, y_train)

probabilities = logistic_mod.predict_proba(X_test)
print(probabilities[:15,:])

scores = myFunctions.score_model(probabilities, 0.5)
myFunctions.print_metrics_logreg(y_test, scores)  
myFunctions.plot_auc(y_test, probabilities)  

thresholds = [0.5,0.45, 0.40, 0.35, 0.3, 0.25]
for t in thresholds:
    myFunctions.test_threshold(probabilities, y_test, t)
    
## Introducing the test data set  
dfTest['ageGroup'] = dfTest["BirthDate"].map(ageGroups)
dfTest['logIncome'] = np.log(dfTest['YearlyIncome'])    
dfTest = dfTest.drop(columns=colDropAll)

cat_cols = ['CountryRegionName','Education','Occupation','Gender',
            'MaritalStatus']

Features = myFunctions.encode_string(dfTest['ageGroup'])

for col in cat_cols:
    temp = myFunctions.encode_string(dfTest[col])
    Features = np.concatenate([Features, temp], axis = 1)

print(Features.shape)
print(Features[:2, :])  

#Numeric features -> numpy arrays
num_features = ['NumberCarsOwned','NumberChildrenAtHome','TotalChildren',
                'YearlyIncome','logIncome']
Features = np.concatenate([Features, np.array(dfTest[num_features])], axis = 1)
print(Features.shape)
print(Features[:2, :])   

X_test2 = Features[:,:]
X_test2[:,24:] = scaler.transform(X_test2[:,24:])

probabilities = logistic_mod.predict_proba(X_test2)
print(probabilities[:15,:])


scores = myFunctions.score_model(probabilities, 0.5)
print(np.array(scores[:15]))
    

## Regression model
dfSpend = originalTrain.drop(columns=colDropAll)
dfSpend = dfSpend.drop(columns=['BikeBuyer'])
dfSpend['logSpend'] = np.log(dfSpend['AveMonthSpend'])
myFunctions.plot_histogram(dfSpend,['AveMonthSpend','logSpend'])
dfSpend = dfSpend.drop(columns=['AveMonthSpend'])
dfSpend['logIncome'] = np.log(dfSpend['YearlyIncome'])


#Copying code from above to transform dataset
cat_cols = ['CountryRegionName','Education','Occupation','Gender',
            'MaritalStatus','ageGroup']


Features = myFunctions.featureTransformation(dfSpend,cat_cols,num_features)

# Randomly sample cases to create independent training and test data
labels = np.array(dfSpend['logSpend'])
nr.seed(9988)
indx = range(Features.shape[0])
indx = ms.train_test_split(indx, test_size = 0.3)
X_train = Features[indx[0],:]
y_train = np.ravel(labels[indx[0]])
X_test = Features[indx[1],:]
y_test = np.ravel(labels[indx[1]])

#Scale numeric features
scaler = preprocessing.StandardScaler().fit(X_train[:,24:])
X_train[:,24:] = scaler.transform(X_train[:,24:])
X_test[:,24:] = scaler.transform(X_test[:,24:])
X_train[:2,]

## define and fit the linear regression model
lin_mod = linear_model.LinearRegression()
lin_mod.fit(X_train, y_train)

print(lin_mod.intercept_)
print(lin_mod.coef_)

y_score = lin_mod.predict(X_test) 

#plotten funkar inte
myFunctions.plot_regression(X_test, y_score, y_test)

myFunctions.print_metrics_linreg(y_test, y_score, 2)    
myFunctions.hist_resids(y_test,y_score)
myFunctions.resid_qq(y_test,y_score)

## Applying on test data
X_test2 = myFunctions.featureTransformation(dfTest,cat_cols,num_features)
X_test2[:,24:] = scaler.transform(X_test2[:,24:])


y2_score = lin_mod.predict(X_test2) 
expScore = np.round(np.exp(y2_score))