# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 22:56:34 2019

@author: iriqu
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#from funciones_ayuda import find_missing_values
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm,skew
from scipy import stats

## Cargar datos TRAIN ###

df_train = pd.read_csv("train.csv", encoding ="latin-1", sep=",", header=0)

## Códigos BÁSICOS VISTA GENERAL DE LOS DATOS
print(df_train.columns) #Entrega todos los nombres de las variables
    
df_train.info() #Muestra el tipo (int64, object,etc) nombre columna, non null data
    
df_train.head() #En este caso con muchas columnas no es muy útil (muestra las 10 primeras filas)    

df_train['SalePrice'].describe() #entrega medidas de tendencia central y dispersión

sns.distplot(df_train['SalePrice']) #histograma o distribución de los datos ordenados

print("Skewness: %f" % df_train['SalePrice'].skew()) # Simetría entre  {-0,5 y 0,5} es bastante simetrico, |skew|>1 es muy skewed
print("Kurtosis: %f" % df_train['SalePrice'].kurt()) # Kurt=3 normal Kurt>3 mucho outlier Kurt<3 poco outlier

y = df_train.SalePrice
y=np.log(y)
### Vista de NULL VALUES

nulltrain=df_train.isnull().sum()
nulltrain=nulltrain[nulltrain>0]
nulltrain.sort_values(ascending=False)

### Se toman las variables NÚMERICAS (puede incluir categóricas)
colsnum=df_train.select_dtypes(exclude = ["object"]).columns
colsnum= colsnum.drop('SalePrice')
x_num=df_train[colsnum]

### Se ordenan según skewness
skewness = x_num.apply(lambda x: skew(x))
skewness.sort_values(ascending=False)
skewness = skewness[abs(skewness)>0.5]

#Analizar todos los histogramas y scatter de x_num con SalePrice
for item in skewness.index.values:
    sns.distplot(x_num[item])
    plt.show()
    data =pd.concat([df_train.SalePrice,x_num[item]],axis=1)
    data.plot.scatter(y='SalePrice',x=item)
    plt.show()
    
# Separación de variables NO-NÚMERICAS/CATEGÓRICAS
colscat=df_train.select_dtypes(include = ["object"]).columns
x_cat = df_train[colscat]
# Analizar Boxplots de x_cat con SalePrice
for item in colscat:
    data = pd.concat([df_train['SalePrice'], df_train[item]], axis=1)
    f, ax = plt.subplots(figsize=(8, 6))
    fig = sns.boxplot(x=item, y="SalePrice", data=data)
    plt.show()    

#correlation matrix
corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True, cbar=True);

#saleprice correlation matrix
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values, cmap='RdYlGn')
plt.show()   

#Ordenamos la matriz de corr con la columna de SalePrice
#Luego se imprime la columna
corrmat.sort_values(['SalePrice'],ascending=False,inplace=True) #Corresponde a una accion sobre corrmat
print(corrmat.SalePrice)

categorical_features = df_train.select_dtypes(include=['object']).columns #Separa los int64 de string
numerical_features = df_train.select_dtypes(exclude = ['object']).columns
numerical_features = numerical_features.drop("SalePrice")
train_num = df_train[numerical_features]
train_cat = df_train[categorical_features]

##### OTROS CÓDIGOS ###########
#scatterplot
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(df_train[cols], size = 2.5)
plt.show();

#missing data
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)

#dealing with missing data
df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index,1)
df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)
df_train.isnull().sum().max() #just checking that there's no missing data missing...

#Univariate analysis
#standardizing data
saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:,np.newaxis]);
low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)
#Bivariate analysis
#bivariate analysis saleprice/grlivarea
var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#deleting points
#En scatterplot se enceuntran dos outliers con valores altos de GrLivArea
df_train.sort_values(by = 'GrLivArea', ascending = False)[:2]
df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)
df_train = df_train.drop(df_train[df_train['Id'] == 524].index)
#bivariate analysis saleprice/grlivarea
var = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#histogram and normal probability plot
sns.distplot(df_train['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)
#applying log transformation
df_train['SalePrice'] = np.log(df_train['SalePrice'])
#transformed histogram and normal probability plot
sns.distplot(df_train['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)
#histogram and normal probability plot
sns.distplot(df_train['GrLivArea'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['GrLivArea'], plot=plt)
df_train['GrLivArea'] = np.log(df_train['GrLivArea'])
#histogram and normal probability plot
sns.distplot(df_train['TotalBsmtSF'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['TotalBsmtSF'], plot=plt)

#create column for new variable (one is enough because it's a binary categorical feature)
#if area>0 it gets 1, for area==0 it gets 0
df_train['HasBsmt'] = pd.Series(len(df_train['TotalBsmtSF']), index=df_train.index)
df_train['HasBsmt'] = 0 
df_train.loc[df_train['TotalBsmtSF']>0,'HasBsmt'] = 1
#transform data
df_train.loc[df_train['HasBsmt']==1,'TotalBsmtSF'] = np.log(df_train['TotalBsmtSF'])
#histogram and normal probability plot
sns.distplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)


#Checking for homoscedasticity
#scatter plot
plt.scatter(df_train['GrLivArea'], df_train['SalePrice']);
plt.scatter(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], df_train[df_train['TotalBsmtSF']>0]['SalePrice']);
#convert categorical variable into dummy
df_train = pd.get_dummies(df_train)