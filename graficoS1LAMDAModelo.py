import pandas as pd 
from sklearn  import preprocessing
import numpy as np 
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
descriptores=2
datos=pd.read_csv('C:/Python/Doctorado/LAMDAClusS1Modelo.csv', header=0, delimiter=',')#Importación de datos
print(datos)
datos.iloc[1]=datos.iloc[1].astype(float)# los datos en reales
datos.iloc[2]=datos.iloc[2].astype(float)
df=pd.DataFrame(datos)# DataFrame Original
df1= df.iloc[:, [1, 2]]# data lista para aplicar el ACP
print(df1)
df_scaled=preprocessing.scale(df1) #con esta funcion se estandarizan los datos a una normal estandar
print('Data normalizada')
print(df_scaled)
media=df_scaled.mean(axis=0)
desv=df_scaled.std(axis=0)
print(media)
print(desv)
pca=PCA(n_components=2)# creo el modelo 
pca_s1_modelo=pca.fit_transform(df_scaled) #Contiene las coordenadas de los individuos esto implica que se hace la transformación lineal
pca_s1_df=pd.DataFrame(data=pca_s1_modelo, columns= ['Componente1', 'Componente2']) #creación del data frame con las coordenadas de los dos componentes 
pca_nombres_s1=pd.concat([pca_s1_df, df[['Cluster']]], axis=1)#Agrego la columna de la etiqueta de los datos originales
pca_nombres_s1['Cluster']=pca_nombres_s1['Cluster'].astype(int)# convierto la columna de la etiqueta en valores enteros 
fig=plt.figure(figsize=(10,10))
ax=fig.add_subplot(1,1,1)
ax.set_xlabel('Componente 1', fontsize=15)
ax.set_ylabel('Componente 2', fontsize=15)
ax.set_title('Data S1 LAMDA Clustering', fontsize=20)
color_theme= np.array(['red', 'blue', 'black', 'red', 'orange', 'green', 'brown', 'purple'  , 'black' , 'maroon', 'gold', 'violet'])
ax.scatter(x=pca_s1_df.Componente1 , y= pca_s1_df.Componente2, c=color_theme[pca_nombres_s1.Cluster], s=40, marker='.')
plt.savefig("S1LAMDAClus.png", transparent=True,  bbox_inches='tight', dpi=400)
plt.show()
