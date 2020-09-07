import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
datos=pd.read_csv('R15.csv',  header=0, delimiter=';', usecols=range(0,3))
df=pd.DataFrame(datos)
v1= df['v1'].values
v2= df['v2'].values
X=np.array(list(zip(v1,v2))) #transformo los datos en una matriz
df1=df.drop(['clases'], axis=1)#elimino la columna de la etiqueta
dfnorm=(df1-df1.min())/(df1.max()-df1.min()) #estandarizacion de los datos
#dfnorm['clusteres']=df['clases'] #agrego nuevamente las etiquetas de la data data original 
pca=PCA(n_components=2)#Modelo de componentes principales
pca_iris=pca.fit_transform(dfnorm) #Contiene las coordenadas de los individuos esto implica que se hace la transformación lineal 
print(dfnorm)
dfnorm['clusteres']=df['clases'] #agrego nuevamente las etiquetas de la data data original 
pca_iris_df=pd.DataFrame(data=pca_iris, columns= ['Componente_1', 'Componente_2']) #creación del data frame con las coordenadas de los dos componentes 
pca_nombres_iris=pd.concat([pca_iris_df, dfnorm[['clusteres']]], axis=1) #concatenar las etiquetas con los individuos
print(pca_nombres_iris)
#CREACIÓN DEL GRÁFICO
fig=plt.figure(figsize=(10,10))
ax=fig.add_subplot(1,1,1)
ax.set_xlabel('Componente 1', fontsize=15)
ax.set_ylabel('Componente 2', fontsize=15)
ax.set_title('R15 Original', fontsize=20)
color_theme= np.array(['red', 'blue', 'orange', 'black', 'pink', 'green', 'brown', 'purple', 'yellow', 'gray', 'black', 'blue', 'red', 'orange', 'purple', 'brown' ])    
ax.scatter(x=pca_nombres_iris.Componente_1 , y= pca_nombres_iris.Componente_2, c=color_theme[pca_nombres_iris.clusteres], s= 20, marker='.')
plt.show()
