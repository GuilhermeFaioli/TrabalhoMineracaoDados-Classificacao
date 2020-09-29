# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 18:29:15 2020

@author: guilh
"""

import pandas as pd
import numpy as np
import seaborn as sns

base = pd.read_csv('BaseDeDados.csv')
#print(base.head())
#pd.set_option('display.max_columns', None)
#print(base.describe())
#print(base.loc[pd.isnull(base['FEBRE'])])
#classe = linha 64 CLASSI_FIN
from sklearn.impute import SimpleImputer
#simpleImputer =  SimpleImputer(missing_values=np.nan, strategy='most_frequent')

New_Data = base

#simpleImputer = simpleImputer.fit(New_Data)
#New_Data = simpleImputer.transform(New_Data)

#New_Data = New_Data.dropna(axis = 1)

#from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler()
#New_Data = scaler.fit_transform(New_Data)

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
New_Data.iloc[:, 1] = labelencoder.fit_transform(New_Data.iloc[:, 1])
New_Data.iloc[:, 3] = labelencoder.fit_transform(New_Data.iloc[:, 3])
New_Data.iloc[:, 4] = labelencoder.fit_transform(New_Data.iloc[:, 4])
New_Data.iloc[:, 6] = labelencoder.fit_transform(New_Data.iloc[:, 6])
New_Data.iloc[:, 8] = labelencoder.fit_transform(New_Data.iloc[:, 8])
New_Data.iloc[:, 13] = labelencoder.fit_transform(New_Data.iloc[:, 13])
New_Data.iloc[:, 15] = labelencoder.fit_transform(New_Data.iloc[:, 15].astype(str))
New_Data.iloc[:, 16] = labelencoder.fit_transform(New_Data.iloc[:, 16].astype(str))
New_Data.iloc[:, 30] = labelencoder.fit_transform(New_Data.iloc[:, 30].astype(str))
New_Data.iloc[:, 32] = labelencoder.fit_transform(New_Data.iloc[:, 32])
New_Data.iloc[:, 48] = labelencoder.fit_transform(New_Data.iloc[:, 48].astype(str))
New_Data.iloc[:, 49] = labelencoder.fit_transform(New_Data.iloc[:, 49].astype(str))
New_Data.iloc[:, 50] = labelencoder.fit_transform(New_Data.iloc[:, 50].astype(str))
New_Data.iloc[:, 52] = labelencoder.fit_transform(New_Data.iloc[:, 52].astype(str))
New_Data.iloc[:, 54] = labelencoder.fit_transform(New_Data.iloc[:, 54].astype(str))
New_Data.iloc[:, 55] = labelencoder.fit_transform(New_Data.iloc[:, 55].astype(str))
New_Data.iloc[:, 56] = labelencoder.fit_transform(New_Data.iloc[:, 56].astype(str))
New_Data.iloc[:, 61] = labelencoder.fit_transform(New_Data.iloc[:, 61].astype(str))
New_Data.iloc[:, 68] = labelencoder.fit_transform(New_Data.iloc[:, 68].astype(str))
New_Data.iloc[:, 71] = labelencoder.fit_transform(New_Data.iloc[:, 71].astype(str))
New_Data = New_Data.drop('Unnamed: 63', 1)

#from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler()
#New_Data = scaler.fit_transform(New_Data)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler() 
scaled_values = scaler.fit_transform(New_Data) 
New_Data.loc[:,:] = scaled_values

New_Data.to_csv('BaseDeDadosNormalizada.csv')
