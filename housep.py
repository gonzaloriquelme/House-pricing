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
from scipy.stats import norm,skew
from scipy import stats
from funciones_ayuda import mean_absolute_percentage_error, root_mean_squared_log_error
from tabulate import tabulate

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor



### Se cargan datos TRAIN y se crea columna identificadora
df_train = pd.read_csv("train.csv", encoding ="latin-1", sep=",", header=0)
df_train['istrain']=1

# Se cargan datos TEST y se crea columna identificadora
df_test = pd.read_csv("test.csv",encoding="latin-1",sep=",",header=0)
df_test['istrain']=0

# Se guarda Y y se transforma
y = df_train.SalePrice
y=np.log(y)

# Se unen TRAIN y TEST axis=0 en filas
all_data=pd.concat([df_train,df_test],axis=0)
all_data=all_data.drop(columns='SalePrice') 


### Vista de missing values y reemplazo por valores
nulltrain=all_data.isnull().sum()
nulltrain=nulltrain[nulltrain>0]
nulltrain.sort_values(ascending=False)
for item in nulltrain.index:
    if all_data[item].dtype=='O':
        all_data[item]=all_data[item].fillna('None')
    else:
        all_data[item]=all_data[item].fillna(0)
        
## LabelEncoder        
        
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(all_data[c].values)) 
    all_data[c] = lbl.transform(list(all_data[c].values))        
        
### Se agrega TotalSF

all_data['TotalBsmtSF']=all_data['TotalBsmtSF'].astype('int64')

all_data['TotalSF']= all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']         

## Arreglar ciertas variables

all_data['TotalSF']= all_data['TotalSF'].astype('float64') 

all_data['BsmtFinSF1']=all_data['BsmtFinSF1'].astype('float64')

all_data['BsmtFinSF2']=all_data['BsmtFinSF2'].astype('float64')  

all_data['BsmtUnfSF']=all_data['BsmtUnfSF'].astype('float64')

all_data['GarageArea']=all_data['GarageArea'].astype('float64')
         
all_data['LotFrontage']=all_data['LotFrontage'].astype('float64')

all_data['MasVnrArea']=all_data['MasVnrArea'].astype('float64')
##  Revision Skewness 
# Se separan variables númericas de las no-númericas(Puede contener categóricas)

colsnum=all_data.select_dtypes(exclude = ["object"]).columns
x_num=all_data[colsnum]

corrmat=pd.concat([x_num[x_num['istrain']==1],y],axis=1).corr()
corrmat['SalePrice'].sort_values(ascending=False)##Revisar correlacion de TotBsmtSF
 

## Se descartan las variables que no tienen relacion lineal con SalePrice

dropvar=corrmat['SalePrice'][abs(corrmat['SalePrice'])<0.5].index.values
x_num=x_num.drop(columns=dropvar)

### Ahora se revisa Skewness

skewness = x_num.apply(lambda x: skew(x))
skewness.sort_values(ascending=False)
skewness = skewness[abs(skewness)>0.75]
    
# Se redefine el conjunto skewness y se aplica la transformacion    
x_num[skewness.index] = np.log1p(x_num[skewness.index]) #Revisar 


## Separación de variables no númericas, categóricas
colscat=all_data.select_dtypes(include = ["object"]).columns
x_cat = all_data[colscat]
x_cat=pd.get_dummies(x_cat, prefix=colscat)


## Se vuelven a separar TRAIN y TEST
all_data_pre=pd.concat([x_num,x_cat],axis=1)
x_train_=all_data_pre[all_data_pre['istrain']==1]
x_train_=x_train_.drop(columns='istrain')
x_test_=all_data_pre[all_data_pre['istrain']==0]
x_test_=x_test_.drop(columns='istrain')


# Separación de TRAIN -- Train=Train ; Test=CrossValidation    
x_train,x_test,y_train,y_test = train_test_split(x_train_,y,test_size=0.3, random_state=4)

# Se crea la regresión y se ajustan los parámetros
reg=LinearRegression()
reg.fit(x_train,y_train)

predictions=reg.predict(x_test)

columnastabla=['Modelo','Training R^2', 'Test/Validation R^2', 'RMSLE','RMSE']
Linear=['RL',
        "%.4f" % reg.score(x_train,y_train),
        "%.4f" % reg.score(x_test,y_test),
        "%.4f" % np.sqrt(mean_squared_log_error(y_test,predictions)),
        "%.4f" % np.sqrt(mean_squared_error(y_test, predictions))]


########## L2 Regularization or Ridge Regression 
########## Alpha {0.1 to infinity} 

ridge=RidgeCV(alphas = [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60], fit_intercept=True) 

ridge.fit(x_train, y_train)

alpha=ridge.alpha_
print('Best alpha',"%.2f" % alpha)

print("Try again for more precision with alphas centered around " + "%.2f" %alpha)
ridge = RidgeCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85, 
                          alpha * .9, alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15,
                          alpha * 1.25, alpha * 1.3, alpha * 1.35, alpha * 1.4],cv = 5, fit_intercept=True) 

ridge.fit(x_train, y_train)
alpha = ridge.alpha_
print("Best alpha :","%.2f" % alpha)

print("Try again for more precision with alphas centered around " + "%.2f" % alpha)
ridge = RidgeCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85, 
                          alpha * .9, alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15,
                          alpha * 1.25, alpha * 1.3, alpha * 1.35, alpha * 1.4],cv = 5, fit_intercept=True) 

ridge.fit(x_train, y_train)
alpha = ridge.alpha_
print("Best alpha :","%.2f" % alpha)

test_pre = ridge.predict(x_test)
train_pre = ridge.predict(x_train)


coef = pd.Series(ridge.coef_)

print("Ridge picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")

Ridge=['RL-Ridge:'+"%.2f" % alpha,
        "%.4f" % ridge.score(x_train,y_train),
        "%.4f" % ridge.score(x_test,y_test),
        "%.4f" % np.sqrt(mean_squared_log_error(y_test,test_pre)),
        "%.4f" % np.sqrt(mean_squared_error(y_test, test_pre))]


### Gradient Boosting Regression ###
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)
GBoost.fit(x_train,y_train)
test_pre=GBoost.predict(x_test)
train_pre=GBoost.predict(x_train)


GradBoost=['GBoost-R',
        "%.4f" % GBoost.score(x_train,y_train),
        "%.4f" % GBoost.score(x_test,y_test),
        "%.4f" % np.sqrt(mean_squared_log_error(y_test,test_pre)),
        "%.4f" % np.sqrt(mean_squared_error(y_test, test_pre))]

#### Tabla con Resultados ######

Resultados=pd.DataFrame(np.array([Linear,Ridge,GradBoost]),columns=columnastabla)
print(tabulate(Resultados,headers='keys',tablefmt='fancy_grid'))
########Predicciones finales##########


#x=sc.transform(x)
#x=pf.transform(x)

ides=df_train.Id
predictions=np.exp(ridge.predict(x_test_))

#### Crear archivo para Kaggle####
workbook = xlsxwriter.Workbook('submit.xlsx') #revisar esto debería definir las dummies por su definicion
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

   
    