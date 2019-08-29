# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 18:03:39 2019

@author: iriqu
"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore') #Elimina advertencias sobre traspaso de int64 a float64 por normalizacion
import xlsxwriter
from scipy.stats import norm
from scipy import stats
from funciones_ayuda import mean_absolute_percentage_error, root_mean_squared_log_error


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler


#####Cargar base de datos########

df_train = pd.read_csv("train.csv", encoding ="latin-1", sep=",", header=0)

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
print("RMSLE: %.5f"% np.sqrt(mean_squared_log_error(y_test,predictions)))


###### BIAS/VARIANCE THEORY ******ACCURACY****** ########

print('Training score: {}'.format(reg.score(x_train, y_train)))
print('Test score: {}'.format(reg.score(x_test, y_test)))

##### Feature Scaler 
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)#se requiere incluir fit solamente la primera vez que se utiliza

##### Polynomial features
pf=PolynomialFeatures(degree=2)
x_train=pf.fit_transform(x_train)
x_test=pf.transform(x_test)#misma idea


########## L2 Regularization or Ridge Regression 
########## Alpha {0.1 to infinity} 

ridge=RidgeCV(alphas = [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60], fit_intercept=True) 

ridge.fit(x_train, y_train)

alpha=ridge.alpha_
print('Best alpha', alpha)

print("Try again for more precision with alphas centered around " + str(alpha))
ridge = RidgeCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85, 
                          alpha * .9, alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15,
                          alpha * 1.25, alpha * 1.3, alpha * 1.35, alpha * 1.4],cv = 5, fit_intercept=True) 

ridge.fit(x_train, y_train)
alpha = ridge.alpha_
print("Best alpha :", alpha)

print("Try again for more precision with alphas centered around " + str(alpha))
ridge = RidgeCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85, 
                          alpha * .9, alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15,
                          alpha * 1.25, alpha * 1.3, alpha * 1.35, alpha * 1.4],cv = 5, fit_intercept=True) 

ridge.fit(x_train, y_train)
alpha = ridge.alpha_
print("Best alpha :", alpha)

predictions = ridge.predict(x_test)

print('Variance score(R^2): %.2f' % r2_score(y_test, predictions))
print("Mean squared error: %.2f"% mean_squared_error(y_test, predictions))
print("MAPE: %.2f"% mean_absolute_percentage_error(y_test, predictions))
print("RMSLE: %.5f"% np.sqrt(mean_squared_log_error(y_test,predictions)))
print('Modelo Ridge')

print('Training score: {}'.format(ridge.score(x_train, y_train)))
print('Test score: {}'.format(ridge.score(x_test, y_test)))

########Predicciones finales##########

df_test = pd.read_csv("test.csv",encoding="latin-1",sep=",",header=0)

cols=['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
x=df_test[cols]
 
x=x.fillna(x.median()) #se rellena con mediana (necesario para kaggle)
x=sc.transform(x)
x=pf.transform(x)

#x=x.dropna()#si al menos 1 columna tiene NaN se elimina la fila
ides=df_train.Id
predictions=np.exp(ridge.predict(x))

#### Crear archivo para Kaggle####
workbook = xlsxwriter.Workbook('submit.xlsx')
worksheet = workbook.add_worksheet()  
worksheet.write(0,0,'Id')
worksheet.write(0,1,'SalePrice')
row=1
col=0
for item in ides:
    if item==1460:
        break
    worksheet.write(row,col,item+1460)
    row=row+1
row=1
col=1    
for item in predictions:
    worksheet.write(row,col,item)
    row=row+1
workbook.close()


df=pd.read_excel('submit.xlsx')
df.to_csv('submit.csv', encoding='utf-8', index=False)
#submission=pd.read_csv('submit.csv')
#output=df({'Id':submission.Id,'SalePrice':submission.SalePrice  })

   
    