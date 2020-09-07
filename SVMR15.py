import os, time, pandas, numpy, colorama, math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
datos=pandas.read_csv('R15.csv',  header=0, delimiter=';', usecols=range(0,3))#columnas del dataset
datos=pd.DataFrame(datos)
print(datos)
X=datos.iloc[: , 0:3]
Xnorm=(X-X.min())/(X.max()-X.min()) #estandarizacion de los datos
X=Xnorm
y=datos.clases
x_train, x_test, y_train, y_test=train_test_split(X, y, random_state=42, test_size= 0.20)
clf=SVC(kernel= 'poly', C=1,).fit(x_train, y_train)
print(X)
prediccion=clf.predict(x_test)
print('Accuracy' , accuracy_score(y_test, prediccion))
print('F1' , f1_score(y_test, prediccion, average='micro'))
print('Precision' , precision_score(y_test, prediccion , average='macro'))
print('Recall' , recall_score(y_test, prediccion, average='weighted'))