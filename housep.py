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


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler


### Se cargan datos TRAIN
df_train = pd.read_csv("train.csv", encoding ="latin-1", sep=",", header=0)

### Eliminar variables con missing values
nulltrain=df_train.isnull().sum()
nulltrain=nulltrain[nulltrain>0]
nulltrain.sort_values(ascending=False)
df_train = df_train.drop(columns=nulltrain.index.values)

# Se separan variables númericas de las no-númericas(Puede contener categóricas)
colsnum=df_train.select_dtypes(exclude = ["object"]).columns
colsnum= colsnum.drop('SalePrice')
x_num=df_train[colsnum]

#x_num=x_num.fillna(x_num.median())
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
    
# Variables que tienen una relacion interesante con SalePrice    
# GrLivArea 1stFlrSF 2ndFlrSF TotalBsmtSF BsmtFinSF1 YearBuilt
# Se elimina el resto de las variables spuestamente skewed
dropvar = ['MSSubClass','LotArea','OverallCond','YearRemodAdd','BsmtFinSF1',
           'BsmtFinSF2','BsmtUnfSF','LowQualFinSF','BsmtFullBath','BsmtHalfBath',
           'HalfBath','KitchenAbvGr','TotRmsAbvGrd','Fireplaces','WoodDeckSF',
           'OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea',
           'MiscVal']    
x_num = x_num.drop(columns=dropvar)
skewness = x_num.apply(lambda x: skew(x))
skewness.sort_values(ascending=False)
skewness = skewness[abs(skewness)>0.5]
# Se redefine el conjunto skewness y se aplica la transformacion    
x_num[skewness.index] = np.log1p(x_num[skewness.index])

# Separación de variables no númericas, categóricas
colscat=df_train.select_dtypes(include = ["object"]).columns
x_cat = df_train[colscat]
# Analizar Boxplots de x_cat con SalePrice
for item in colscat:
    data = pd.concat([df_train['SalePrice'], df_train[item]], axis=1)
    f, ax = plt.subplots(figsize=(8, 6))
    fig = sns.boxplot(x=item, y="SalePrice", data=data)
    plt.show()

# Se cargan datos TEST
df_test = pd.read_csv("test.csv",encoding="latin-1",sep=",",header=0)

# Se eliminan las mismas columnas/variables eliminadas en TRAIN
df_test = df_test.drop(columns=nulltrain.index.values)
df_test =df_test.drop(columns=dropvar)

# Se toman las variables númericas, imputan medianas y aplica log(skew[train])
colsnum_test=df_test.select_dtypes(exclude = ["object"]).columns
x_num_test=df_test[colsnum_test]
x_num_test=x_num_test.fillna(x_num_test.median())  
x_num_test[skewness.index] = np.log1p(x_num_test[skewness.index])

# Se toman las variables no númericas
colscat_test=df_test.select_dtypes(include = ["object"]).columns
x_cat_test = df_test[colscat_test]

# Se agrupan las variables
x_train_=pd.concat([x_num,x_cat],axis=1)
x_train_['istest']=0
x_test_=pd.concat([x_num_test,x_cat_test],axis=1)
x_test_['istest']=1
tempo=pd.concat([x_train_,x_test_],axis=0)
tempo=pd.get_dummies(tempo, prefix=colscat)
x_train_=tempo[tempo['istest']==0]
x_test_=tempo[tempo['istest']==1]
###Transformar a distribucion normal SalePrice
y = df_train.SalePrice
y=np.log(y)

# Separación de TRAIN -- Train=Train ; Test=CrossValidation
x_train,x_test,y_train,y_test = train_test_split(x_train_,y,test_size=0.3, random_state=4)

# Se crea la regresión y se ajustan los parámetros
reg=LinearRegression()
reg.fit(x_train,y_train)

predictions=reg.predict(x_test)
print("Modelo regresión lineal sin regularización")
#print('Coefficients: \n', reg.coef_)
print("MAPE: %.2f"% mean_absolute_percentage_error(y_test, predictions),"%")
print("RMSE: %.2f"% np.sqrt(mean_squared_error(y_test, predictions)))
print("RMSLE: %.5f"% np.sqrt(mean_squared_log_error(y_test,predictions)))


###### BIAS/VARIANCE THEORY ******ACCURACY****** ########

print('Training (R^2) score: {}'.format(reg.score(x_train, y_train)))
print('Test/Validation (R^2) score: {}'.format(reg.score(x_test, y_test)))

##### Feature Scaler 
#sc=StandardScaler()
#x_train=sc.fit_transform(x_train)
#x_test=sc.transform(x_test)#se requiere incluir fit solamente la primera vez que se utiliza

##### Polynomial features
#pf=PolynomialFeatures(degree=2)
#x_train=pf.fit_transform(x_train)
#x_test=pf.transform(x_test)#misma idea


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

test_pre = ridge.predict(x_test)
train_pre = ridge.predict(x_train)

coef = pd.Series(ridge.coef_)

#plot between predicted values and residuals
plt.scatter(train_pre, train_pre - y_train, c = "blue",  label = "Training data")
plt.scatter(test_pre,test_pre - y_test, c = "black",  label = "Validation data")
plt.title("Linear regression")
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.legend(loc = "upper left")
plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = "red")
plt.show()

# Plot predictions - Real values
plt.scatter(train_pre, y_train, c = "blue",  label = "Training data")
plt.scatter(test_pre, y_test, c = "black",  label = "Validation data")
plt.title("Linear regression with Ridge regularization")
plt.xlabel("Predicted values")
plt.ylabel("Real values")
plt.legend(loc = "upper left")
plt.plot([10.5, 13.5], [10.5, 13.5], c = "red")
plt.show()

print("Ridge picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")

print("MAPE: %.2f"% mean_absolute_percentage_error(y_test, test_pre),"%")
print("RMSE: %.2f"% np.sqrt(mean_squared_error(y_test, test_pre)))
print("RMSLE: %.5f"% np.sqrt(mean_squared_log_error(y_test,test_pre)))
print('Modelo Ridge')

print('Training (R^2) score: {}'.format(ridge.score(x_train, y_train)))
print('Test/Validation (R^2) score: {}'.format(ridge.score(x_test, y_test)))

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

   
    