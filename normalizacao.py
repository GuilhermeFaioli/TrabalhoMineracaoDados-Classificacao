# -- coding: utf-8 --
"""
Created on Sat Sep 26 18:29:15 2020
@author: Fabiana Barreto Pereira, Guilherme Ferreira Faioli Lima, Junior Reis dos Santos, Rafaela Silva Miranda e Yasmine de Melo Leite
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from report import Report as rp

# gráficos
import matplotlib.pyplot as plt
import time
from matplotlib.colors import ListedColormap
from sklearn.metrics import roc_curve, auc, plot_roc_curve
from sklearn.preprocessing import label_binarize
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


base = pd.read_csv('BaseDeDados.csv')


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
New_Data.iloc[:, 15] = labelencoder.fit_transform(
    New_Data.iloc[:, 15].astype(str))
New_Data.iloc[:, 16] = labelencoder.fit_transform(
    New_Data.iloc[:, 16].astype(str))
New_Data.iloc[:, 30] = labelencoder.fit_transform(
    New_Data.iloc[:, 30].astype(str))
New_Data.iloc[:, 32] = labelencoder.fit_transform(New_Data.iloc[:, 32])
New_Data.iloc[:, 48] = labelencoder.fit_transform(
    New_Data.iloc[:, 48].astype(str))
New_Data.iloc[:, 49] = labelencoder.fit_transform(
    New_Data.iloc[:, 49].astype(str))
New_Data.iloc[:, 50] = labelencoder.fit_transform(
    New_Data.iloc[:, 50].astype(str))
New_Data.iloc[:, 52] = labelencoder.fit_transform(
    New_Data.iloc[:, 52].astype(str))
New_Data.iloc[:, 54] = labelencoder.fit_transform(
    New_Data.iloc[:, 54].astype(str))
New_Data.iloc[:, 55] = labelencoder.fit_transform(
    New_Data.iloc[:, 55].astype(str))
New_Data.iloc[:, 56] = labelencoder.fit_transform(
    New_Data.iloc[:, 56].astype(str))
New_Data.iloc[:, 61] = labelencoder.fit_transform(
    New_Data.iloc[:, 61].astype(str))
New_Data.iloc[:, 66] = labelencoder.fit_transform(
    New_Data.iloc[:, 67].astype(str))
New_Data.iloc[:, 69] = labelencoder.fit_transform(
    New_Data.iloc[:, 70].astype(str))

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaled_values = scaler.fit_transform(New_Data)
New_Data.loc[:, :] = scaled_values

New_Data['CLASSI_FIN'] = classe
New_Data = New_Data.fillna(-1)
filter_class = New_Data['CLASSI_FIN'] > -1
New_Data = New_Data[filter_class]
New_Data.to_csv('BaseDeDadosNormalizada.csv')

# o arquivo abaixo executa

teste = pd.read_csv('BaseDeDadosNormalizada.csv')


# Definindo as colunas como atributos descritivos
X = teste.iloc[:, 1:78].values

# Definindo a coluna 79 como atributo Classe (Preditivo)
y = teste.iloc[:, 79].values

# Separando o conjunto de dados em conjunto de treinamento e de teste

test_size = 0.25

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=0)


# testes para diferentes números de vizinhos (k)

precisionsMicro = []
precisionsWeighted = []
recallMicro = []
recallWeighted = []
nNeighbors = [3, 5, 7, 9]
graphics = []  # lista de gráficos gerados
n_rows = teste.count()[0]
n_rows_test = round(n_rows * test_size)
n_rows_train = n_rows - n_rows_test
n_columns = len(X[0])+1

report = rp('Relatorio.pdf')
report.setNumberColumnsAndRows(n_columns,n_rows,n_rows_train,n_rows_test)

# grafico
# IDENTIFICAR DESBALANCEAMENTO DOS DADOS
target_count = teste['CLASSI_FIN'].value_counts()
target_count = target_count.sort_index()
target_count[1]
target_count[2]
target_count[3]
target_count[4]
target_count[5]


fig, ax = plt.subplots(figsize=(8, 4))
altura = []
for i in target_count:
    altura.append(i)
posicao = []
for i in range(5):
    posicao.append(i)
target_count.plot(kind='bar', title='Balanceamento (classes)',
                  color=['#1F77B4', '#FF7F0E',
                         '#1F77B4', '#FF7F0E', '#1F77B4'],
                  )
# coloca o valor associado a cada barra
for i in range(5):
    ax.text(x=posicao[i], y=altura[i], s=target_count[i+1],
            fontsize=10, ha='center', va='bottom')

# salvando imagem
fig = plt.gcf()
nameFig = 'fig/balanceamento.jpg'
fig.savefig(nameFig, format='jpg')
plt.close()
graphics.append(nameFig)
# fim grafico


for k in nNeighbors:
    # Gerando o Classificador com os dados de treinamento

    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(X_train, y_train)

    # Realizando a Predição das Classes dos dados do conjunto de teste
    y_pred = classifier.predict(X_test)
    y_score = classifier.predict_proba(X_test)

    colors = ["blue", "brown", "green", "yellow",
              "red", "beige", "tomato", "orange", "turquoise", "pink"]

    # precisão micro (Precisão global)
    precision = precision_score(
        y_test, y_pred, average='micro', zero_division=0)
    precisionsMicro.append(round(precision, 2))

    # precisão weighted (Média Ponderada)
    precision = precision_score(
        y_test, y_pred, average='weighted', zero_division=0)
    precisionsWeighted.append(round(precision, 2))

    # erro por classe
    precisions = precision_score(
        y_test, y_pred, average=None, zero_division=0)
    err = [(1-x) for x in precisions]
    classes = [1, 2, 3, 4, 5]
    plt.figure(figsize=(10, 5))
    plt.margins(0.1,0.1)
    plt.subplot(121)

    s = 600
    for i in range(len(classes)):
        plt.scatter(classes[i], err[i], s, c=colors[i %
                                                    len(colors)], alpha=0.5, marker='.', edgecolors='black')
    plt.xlabel("Classes")
    plt.ylabel("erro")
    plt.xticks([1, 2, 3, 4, 5])
    plt.title('Erro por classes')
    plt.legend(loc='best', labels=['Influenza', 'Outro vírus respiratório', 'Outro agente etiológico',
                                   'Não especificado', 'COVID-19'])

    # recall
    recall = recall_score(y_test, y_pred, average=None, zero_division=0)
    classes = [1, 2, 3, 4, 5]
    plt.subplot(122)

    s = 600
    for i in range(len(classes)):
        plt.scatter(classes[i], recall[i], s, c=colors[i %
                                                       len(colors)], alpha=0.5, marker='.', edgecolors='black')
    plt.xlabel("Classes")
    plt.ylabel("recall")
    plt.xticks([1, 2, 3, 4, 5])
    plt.title('Recall')
    plt.legend(loc='best', labels=['Influenza', 'Outro vírus respiratório', 'Outro agente etiológico',
                                   'Não especificado', 'COVID-19'])

    # salvando imagem - erro por classes e recall

    plt.tight_layout()
    nameFig = "fig/Erro e Recall_{}.jpg".format(k)
    plt.savefig(nameFig, format='jpg')
    graphics.append(nameFig)
    plt.clf()

    #  Curva Característica de Operação do Receptor (Curva ROC)

    # Binarizar a saída
    y = label_binarize(y_test, classes=classes)
    n_classes = y.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Calcular a curva ROC micro-média e a área ROC
    fpr["micro"], tpr["micro"], _ = roc_curve(y.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Primeiro agregue todas as taxas de falsos positivos
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Em seguida, interpolar todas as curvas ROC nestes pontos
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finalmente calcule a média e calcule a AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Trace todas as curvas ROC
    lw = 2  # largura da linha
    plt.figure(figsize=(6.5, 6.5))
    plt.plot(fpr["micro"], tpr["micro"],
             label='Curva ROC micro-média (area = {0:0.2f})'
             ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='Curva ROC macro-média (area = {0:0.2f})'
             ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue',
                    'beige', 'green', 'red', 'orange'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='Curva ROC - classe {0} (area = {1:0.2f})'
                 ''.format(i+1, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de falso positivo')
    plt.ylabel('Taxa de verdadeiro positivo')
    plt.title('Característica Operacional do Receptor para multiclasse')
    plt.legend(loc="best")
    fig = plt.gcf()
    nameFig = "fig/ROC_{}.jpg".format(k)
    fig.savefig(nameFig, format='jpg')
    graphics.append(nameFig)
    plt.clf()
    plt.close()

    print("k: "+str(k))

    # matriz de confusão
    class_names = ['inflz', 'outro_virus', 'outro_ag',
                   'sem_esp', 'COVID']

    np.set_printoptions(precision=2)
    # Plota a matriz de confusão não normalizada
    titles_options = [("Matriz de Confusão, Sem Normalização", None),
                      ("Matriz de Confusão Normalizada", 'true')]
    aux = 0
    for title, normalize in titles_options:
        # salvando imagem - matriz de confusão não normalizada
        fig = plt.gcf()
        nameFig = "fig/matriz_confusao_{}.jpg".format(k)
        fig.savefig(nameFig, format='jpg')
        if aux==0:
            graphics.append(nameFig)

        # normaliza a matriz
        disp = plot_confusion_matrix(classifier, X_test, y_test,
                                     display_labels=class_names,
                                     cmap=plt.cm.Blues,
                                     normalize=normalize)
        disp.ax_.set_title(title)

        print(title)
        print(disp.confusion_matrix)

        # salvando imagem - matriz de confusão normalizada
        fig = plt.gcf()
        nameFig = "fig/matriz_confusao_normalizada_{}.jpg".format(k)
        fig.savefig(nameFig, format='jpg')
        if aux==0:
            graphics.append(nameFig)
        aux = 1


labels = [str(x) for x in nNeighbors]

x = np.arange(len(labels))  # localização das labels
width = 0.35  # largura das barras


# gráfico precisão micro e weighted

fig, ax = plt.subplots(figsize=(8, 4))
fig.subplots_adjust(left=0.115, right=0.88)
rects = [[], []]
rects[0] = ax.bar(x - width/2, precisionsMicro, width, label='Micro (global)',
                  align='center', edgecolor="black")
rects[1] = ax.bar(x + width/2, precisionsWeighted, width,
                  label='Ponderada', align='center', edgecolor="black")
ax.set_xticks(x)
ax.set_xlabel('k')
ax.set_ylabel('precisão')
ax.set_title('Precisão Micro e Ponderada')
ax.set_xticklabels(labels)
ax.margins(0.05, 0.5)
ax.legend(loc='best')

# coloca o valor associado a cada barra
for i in range(2):
    for c in rects[i]:
        height = c.get_height()
        ax.annotate('{}'.format(height),
                    xy=(c.get_x() + c.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
fig.tight_layout()
# salvando imagem
nameFig = 'fig/precisao.jpg'
fig.savefig(nameFig, format='jpg')
plt.close()
graphics.append(nameFig)

#passar figuras para o relatório
report.generateReport(graphics,nNeighbors)
