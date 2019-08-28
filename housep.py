# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 18:03:39 2019

@author: iriqu
"""

import pandas as pd
import warnings
warnings.filterwarnings('ignore') #Elimina advertencias sobre traspaso de int64 a float64 por normalizacion


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


from funciones_ayuda import mean_absolute_percentage_error, root_mean_squared_log_error


df_train = pd.read_csv("train.csv", encoding ="latin-1", sep=",", header=0)


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

########## L2 Regularization or Ridge Regression #########

steps = [
    ('scalar', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2)),
    ('model', Ridge(alpha=3, fit_intercept=True)) ## alpha {0.1 to infinity}
]

ridge_pipe = Pipeline(steps)

ridge_pipe.fit(x_train, y_train)

print('Modelo Ridge')

print('Training score: {}'.format(ridge_pipe.score(x_train, y_train)))
print('Test score: {}'.format(ridge_pipe.score(x_test, y_test)))

########## L1 Regularization or Lasso Regression #########

steps = [
    ('scalar', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2)),
    ('model', Lasso(alpha=0.2, fit_intercept=True)) ## alpha {0.1 to 1}
]

lasso_pipe = Pipeline(steps)

lasso_pipe.fit(x_train, y_train)

print('Modelo Lasso')

print('Training score: {}'.format(lasso_pipe.score(x_train, y_train)))
print('Test score: {}'.format(lasso_pipe.score(x_test, y_test)))