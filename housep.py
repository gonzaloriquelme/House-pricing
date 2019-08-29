# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 18:03:39 2019

@author: iriqu
"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from funciones_ayuda import find_missing_values
from scipy.stats import norm
from scipy import stats
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


import xlsxwriter

df_train = pd.read_csv("train.csv", encoding ="latin-1", sep=",", header=0)



############# **MODELO**################

y = df_train.SalePrice
y=np.log(y)
colsnum=['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
#x_cat = df_train[colscat]
#x_cat=pd.get_dummies(x_cat)
x_num=df_train[colsnum]
x=x_num

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3, random_state=4)

reg=LinearRegression()
reg.fit(x_train,y_train)

predictions=reg.predict(x_test)
print('Coefficients: \n', reg.coef_)
print('Variance score(R^2): %.2f' % r2_score(y_test, predictions))
print("Mean squared error: %.2f"% mean_squared_error(y_test, predictions))
print("MAPE: %.2f"% mean_absolute_percentage_error(y_test, predictions))
print("RMSE(log) Kaggle measurement: %.5f" % root_mean_squared_log_error(y_test, predictions))


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


########Predicciones finales##########

df_test = pd.read_csv("test.csv",encoding="latin-1",sep=",",header=0)

cols=['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
x=df_test[cols]
 
x=x.fillna(x.median()) #se rellena con mediana (necesario para kaggle)

#x=x.dropna()#si al menos 1 columna tiene NaN se elimina la fila
ides=df_train.Id
predicciones=np.exp(ridge_pipe.predict(x))

#### Crear archivo para Kaggle####
workbook = xlsxwriter.Workbook('submit.xlsx')
worksheet = workbook.add_worksheet()  
worksheet.write(0,0,'Id')
worksheet.write(0,1,'SalePrice')
row=1
col=0
for i in range(1,len(predicciones)):
    worksheet.write(row,col,i)
    row=row+1
row=1
col=1    
for item in predicciones:
    worksheet.write(row,col,item)
    row=row+1
workbook.close()

   
    