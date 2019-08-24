# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 18:03:39 2019

@author: iriqu
"""

import pandas as pd
import numpy as np


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


import matplotlib.pyplot as plt
import seaborn as sns


df_train = pd.read_csv("train.csv", encoding ="latin-1", sep=",", header=0)


############ **fUNCIONES**############
def find_missing_values(df, columns):
    missing_vals ={}
    print('Número de missing values para cada columna')
    df_lenght=len(df)
    for column in columns:
        total_column_values=df[column].value_counts().sum()
        missing_vals[column]= df_lenght-total_column_values
    return missing_vals    

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def root_mean_squared_log_error(y_true,y_pred):
    y_true, y_pred = np.array(np.log(y_true)), np.array(np.log(y_pred))
    return np.sqrt(np.mean(np.power((y_true - y_pred),2)))

############ **fUNCIONES**##############
 
############## **EDA** #################

#print(df_train.columns)
#  df_train.info() ; df_train.head()
#usar diccionario de missing values por columna    
#missing_values= find_missing_values(df_train, columns=df_train.columns)

############## **EDA** #################

############# **MODELO**################

y = df_train.loc[:,"SalePrice"]

x = df_train.loc[:,["LotArea","GrLivArea", "GarageArea", "PoolArea"]]

# **ALTERNATIVA** Y = df_train[df_train.columns[-1]] 

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3, random_state=4)

reg=LinearRegression()
reg.fit(x_train,y_train)

predictions=reg.predict(x_test)
print('Coefficients: \n', reg.coef_)
print('Variance score: %.2f' % r2_score(y_test, predictions))
print("Mean squared error: %.2f"% mean_squared_error(y_test, predictions))
print("MAPE: %.2f"% mean_absolute_percentage_error(y_test, predictions))
print("RMSE(log) Kaggle measurement: %.5f"% root_mean_squared_log_error(y_test, predictions))


###### BIAS/VARIANCE THEORY ******ACCURACY****** ########

print('Training score: {}'.format(reg.score(x_train, y_train)))
print('Test score: {}'.format(reg.score(x_test, y_test)))

steps = [
    ('scalar', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2)),
    ('model', LinearRegression())
]

pipeline = Pipeline(steps)

pipeline.fit(x_train, y_train)

'Modelo con pipeline'

print('Training score: {}'.format(pipeline.score(x_train, y_train)))
print('Test score: {}'.format(pipeline.score(x_test, y_test)))