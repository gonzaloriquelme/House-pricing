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
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from StackingClass import StackingAveragedModels



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

## Tipos de variables

pd.set_option('display.max_rows', 81)
all_data.dtypes.sort_values(ascending=False)

## Object
## Por lo general, todas son categoricas en forma de string
all_data.select_dtypes(include=['O']).columns
## Numeric int64 y float64
## Se revisan variables numericas que deben ser categoricas astype('str')
all_data.select_dtypes(exclude=['O']).columns

### Se agrega TotalSF

all_data['TotalSF']= all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']         


## Casos moda y mediana
        
all_data['MSZoning']=all_data['MSZoning'].fillna(all_data['MSZoning'].mode())

all_data['LotFrontage']=all_data['LotFrontage'].fillna(all_data['LotFrontage'].median())

all_data['Functional']=all_data['Functional'].fillna(all_data['Functional'].mode())            

all_data['GarageYrBlt']=all_data['GarageYrBlt'].fillna(all_data['GarageYrBlt'].median())

all_data['Exterior1st']=all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode())
  
all_data['Exterior2nd']=all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode())

all_data['BsmtFinSF1']=all_data['BsmtFinSF1'].fillna(all_data['BsmtFinSF1'].median())

all_data['BsmtFinSF2']=all_data['BsmtFinSF2'].fillna(all_data['BsmtFinSF2'].median())

all_data['BsmtUnfSF']=all_data['BsmtUnfSF'].fillna(all_data['BsmtUnfSF'].median())

all_data['TotalBsmtSF']=all_data['TotalBsmtSF'].fillna(all_data['TotalBsmtSF'].median())

all_data['Electrical']=all_data['Electrical'].fillna(all_data['Electrical'].mode())

all_data['KitchenQual']=all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode())

all_data['GarageCars']=all_data['GarageCars'].fillna(all_data['GarageCars'].mode())

all_data['GarageArea']=all_data['GarageArea'].fillna(all_data['GarageArea'].mode())

all_data['SaleType']=all_data['SaleType'].fillna(all_data['SaleType'].mode())

### Vista de missing values y reemplazo por valores
## Antes de eliminar missing values se revisa si NAN significa 0 o None (otra categoría)
nulltrain=all_data.isnull().sum()
nulltrain=nulltrain[nulltrain>0]
percent=nulltrain*100/len(all_data)
missingtable=pd.concat([nulltrain,percent,all_data.dtypes],axis=1)
missingtable=missingtable.rename(columns={0:'#',1:'%',2:'dtype'})
missingtable=missingtable[missingtable['#']>0]
missingtable.sort_values(by='%',ascending=False)

for item in nulltrain.index:
    if all_data[item].dtype=='O':
        data = pd.concat([y, df_train[item]], axis=1)
        f, ax = plt.subplots(figsize=(8, 6))
        fig = sns.boxplot(x=item, y="SalePrice", data=data)
        plt.show()


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
        
##  Revision Skewness 
# Se separan variables númericas de las no-númericas(Puede contener categóricas)

colsnum=all_data.select_dtypes(exclude = ["object"]).columns
x_num=all_data[colsnum]

corrmat=pd.concat([x_num[x_num['istrain']==1],y],axis=1).corr()
corrmat['SalePrice'].sort_values(ascending=False)##Revisar correlacion de TotBsmtSF
 

## Se descartan las variables que no tienen relacion lineal con SalePrice

#dropvar=corrmat['SalePrice'][abs(corrmat['SalePrice'])<0.5].index.values
#x_num=x_num.drop(columns=dropvar)

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

columnastabla=['Modelo','Training R^2', 'Test/Validation R^2']
Linear=['RL',
        "%.4f" % reg.score(x_train,y_train),
        "%.4f" % reg.score(x_test,y_test)]


########## L2 Regularization or Ridge Regression 
########## Alpha {0.1 to infinity} 


#ridge = GridSearchCV(Ridge(fit_intercept=True),
#                     param_grid={'alpha':np.linspace(7,8,10)}, cv=5)

ridge= Ridge(alpha=7, fit_intercept=True)

ridge.fit(x_train,y_train)


Rridge=['RL-Ridge',
        "%.4f" % ridge.score(x_train,y_train),
        "%.4f" % ridge.score(x_test,y_test)]


### Gradient Boosting Regression ###

#GBoost = GridSearchCV(GradientBoostingRegressor(n_estimators=3000,learning_rate=0.05,
#                                                max_features='sqrt',min_samples_leaf=15, 
#                                                min_samples_split=10,loss='huber',
#                                                 random_state =5),
#param_grid={'max_depth':[5,6]},cv=5)
    
GBoost = GradientBoostingRegressor(n_estimators=3000,learning_rate=0.05,
                                                max_features='sqrt',min_samples_leaf=15, 
                                                min_samples_split=10,loss='huber',
                                                 random_state =5, max_depth=5)    

GBoost.fit(x_train,y_train)


GradBoost=['GBoost-R',
        "%.4f" % GBoost.score(x_train,y_train),
        "%.4f" % GBoost.score(x_test,y_test)]

### Lasso Regression ###

#lasso = GridSearchCV(Lasso(random_state=1),
#                   param_grid={'alpha':np.logspace(-5,-4,10)},cv=5)

lasso = Lasso(alpha=1e-05, random_state=1)


lasso.fit(x_train,y_train)


RLasso=['RL-Lasso',
        "%.4f" % lasso.score(x_train,y_train),
        "%.4f" % lasso.score(x_test,y_test)]

### Elastic Net Regression ###


#ENet = GridSearchCV(ElasticNet(random_state=3),param_grid={'alpha':np.logspace(-6,-2,10),
#                    'l1_ratio':np.linspace(0.5,0.9,5)}, cv=5)

ENet = ElasticNet(alpha=0.00046,l1_ratio=0.9,random_state=3)

ENet.fit(x_train,y_train)


ElasticN=['RL-ENet',
        "%.4f" % ENet.score(x_train,y_train),
        "%.4f" % ENet.score(x_test,y_test)]

### Kernel Ridge Regression ###

#KRidge = GridSearchCV(KernelRidge(kernel='polynomial',degree=2,coef0=2.5),
#                      param_grid={'alpha':np.linspace(0.7,0.9,10),
#                                  'gamma': np.logspace(-5,-3,10)}, cv=5)

KRidge = KernelRidge(kernel='polynomial', degree=2, coef0=2.5, alpha=0.9 ,gamma=1e-04)

KRidge.fit(x_train,y_train)


Kernelridge=['RL-KRidge',
        "%.4f" % KRidge.score(x_train,y_train),
        "%.4f" % KRidge.score(x_test,y_test)]

#### Stacking Averaged Models ####

Stack = StackingAveragedModels(base_models=(ENet, GBoost, ridge, KRidge), meta_model=lasso)

Stack.fit(x_train,y_train)

Stacking=['Stack',
        "%.4f" % Stack.score(x_train,y_train),
        "%.4f" % Stack.score(x_test,y_test)]

#### Tabla con Resultados ######

Resultados=pd.DataFrame(np.array([Linear,Rridge,GradBoost,RLasso,ElasticN,Kernelridge,Stacking]),columns=columnastabla)
print(tabulate(Resultados,headers='keys',tablefmt='fancy_grid'))

########Predicciones finales##########


#x=sc.transform(x)
#x=pf.transform(x)

ides=df_train.Id
predictions=np.exp(Stack.predict(x_test_))

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

   
    