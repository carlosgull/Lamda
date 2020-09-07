import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
datos=pd.read_csv('R15_Modelo1.csv',  header=0, delimiter=';', usecols=range(0,3))
df=pd.DataFrame(datos)
print(df)

df1=df.drop(['INDEX'], axis=1)#elimino la columna de la etiqueta
print(df1)

#dfnorm=(df1-df1.min())/(df1.max()-df1.min()) #estandarizacion de los datos

#dfnorm['clusteres']=df['clases'] #agrego nuevamente las etiquetas de la data data original 
pca=PCA(n_components=2)#Modelo de componentes principales
pca_iris=pca.fit_transform(df1) #Contiene las coordenadas de los individuos esto implica que se hace la transformación lineal 

pca_iris_df=pd.DataFrame(data=pca_iris, columns= ['Componente_1', 'Componente_2']) #creación del data frame con las coordenadas de los dos componentes 
print(pca_iris_df)
pca_nombres_iris=pd.concat([pca_iris_df, df[['INDEX']]], axis=1) #concatenar las etiquetas con los individuos
pca_nombres_iris['INDEX']=pca_nombres_iris['INDEX'].round(0).astype(int)
print(pca_nombres_iris)
#CREACIÓN DEL GRÁFICO
fig=plt.figure(figsize=(10,10))
ax=fig.add_subplot(1,1,1)
ax.set_xlabel('Componente 1', fontsize=15)
ax.set_ylabel('Componente 2', fontsize=15)
ax.set_title('R15 LAMDA Semisupervisado', fontsize=20) #magenta es el grupo 4
color_theme= np.array([ 'red', 'orange', 'blue', 'brown', 'black', 'green', 'gray', 'purple', 'salmon', 'yellow', 'cyan', 'red', 'chocolate', 'tomato', 'violet', 'firebrick'])    
ax.scatter(x=pca_nombres_iris.Componente_1 , y= pca_nombres_iris.Componente_2, c=color_theme[pca_nombres_iris.INDEX], s= 20, marker='.')
plt.show()
