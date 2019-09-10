# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 23:35:28 2019

@author: iriqu
"""

import xlsxwriter
import pandas as pd

df_train = pd.read_csv("train.csv", encoding ="latin-1", sep=",", header=0)
df_test = pd.read_csv('test.csv', encoding='latin-1', sep=',', header=0)

all_data=pd.concat([df_train,df_test],axis=0)
workbook = xlsxwriter.Workbook('variables_1.xlsx')
worksheet = workbook.add_worksheet()    
### Creamos Excel con variables###
c = ['Variable', 'Tipo', 'Descripción', 'Num/Cat/Numcat','Nan values','Nan treatment']   
a= df_train.columns    
v= []    
for i in range(len(a)):
    v.append(a[i])
row=0
col=0
for item in c:
    worksheet.write(row,col, item)
    col=col+1
col=0
row=1
for variable in v:
    worksheet.write(row,col, variable)
    row=row+1
col=1
row=1
for variable in v:
    worksheet.write(row,col, str(df_train[variable].dtype))        
    row=row+1
col=4
row=1    
for variable in v:
    n=all_data.isnull().sum()[variable]
    p=round(all_data.isnull().sum()[variable]/len(all_data)*100,3)
    worksheet.write(row,col, str(n)+'('+str(p)+'%'+')') 
    row=row+1
    
workbook.close()    