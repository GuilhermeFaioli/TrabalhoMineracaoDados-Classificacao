# -- coding: utf-8 --
"""
Created on Sat Sep 26 18:29:15 2020
@author: guilh
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Importando algumas funções para este código
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix

base = pd.read_csv('BaseDeDados.csv')

from sklearn.impute import SimpleImputer


New_Data = base

classe = base.iloc[:, 63].values
New_Data = New_Data.drop('CLASSI_FIN', 1)


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
New_Data.iloc[:, 66] = labelencoder.fit_transform(New_Data.iloc[:, 67].astype(str))
New_Data.iloc[:, 69] = labelencoder.fit_transform(New_Data.iloc[:, 70].astype(str))




from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler() 
scaled_values = scaler.fit_transform(New_Data) 
New_Data.loc[:,:] = scaled_values

New_Data['CLASSI_FIN'] = classe
New_Data = New_Data.fillna(-1)

New_Data.to_csv('BaseDeDadosNormalizada.csv')

# o arquivo abaixo executa 

teste = pd.read_csv('BaseDeDadosNormalizada.csv')

# Definindo as colunas 2 e 3 como atributos descritivos
X = teste.iloc[:, [21,22,23,24,25,26,27,28,29,30,31,32,33]].values

# Definindo a coluna 4 como atributo Classe (Preditivo)
y = teste.iloc[:, 79].values

# Separando o conjunto de dados em conjunto de treinamento e de teste

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)




# Gerando o Classificador com os dados de treinamento
classifier = KNeighborsClassifier(n_neighbors = 3)
classifier.fit(X_train, y_train)

# # Realizando a Predição das Classes dos dados do conjunto de teste 
y_pred = classifier.predict(X_test)

# # Gerando a Matriz de Confusão com os dados de teste
cm = confusion_matrix(y_test, y_pred)

print('Matriz de confusão', cm)


New_Data.to_csv('BaseDeDadosNormalizada.csv')



sns.heatmap(cm, center=True)
plt.show()