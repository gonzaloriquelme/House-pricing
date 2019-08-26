# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 23:41:55 2019

@author: iriqu
"""

import pandas as pd
import numpy as np

df_train = pd.read_csv("train.csv", encoding ="latin-1", sep=",", header=0)


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

