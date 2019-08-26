# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 23:35:28 2019

@author: iriqu
"""

import xlsxwriter
import pandas as pd

df_train = pd.read_csv("train.csv", encoding ="latin-1", sep=",", header=0)

workbook = xlsxwriter.Workbook('variables_1.xlsx')
worksheet = workbook.add_worksheet()    
### Creamos Excel con variables###
c = ['Variable', 'Type', 'Segment', 'Expectation', 'Conclusion', 'Comments']   
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
    
workbook.close()    