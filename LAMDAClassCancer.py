import pandas as pd
import numpy as np 
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

datos=pd.read_csv('C:/Doctorado/Data/BrestCancer.csv',  header=0, delimiter=';', usecols=range(0,10))#columnas del dataset
df=pd.DataFrame(datos)
clases=2
descriptores=9
alfa=0.9
mad=np.zeros((clases,descriptores))
matrizprom=np.zeros((clases,descriptores))
gad=np.zeros((clases,descriptores))
variables=df.dtypes #Explorando el tipo de variables del dataset
print(variables)
for j in range(0, descriptores):
    for i in range(0,len(df)):
        df.loc[i, repr(j)+repr(j)]=((df.loc[i,repr(j)]-df[repr(j)].min())/(df[repr(j)].max()-df[repr(j)].min()))
print(df)
dfnorm=df
dfnormorig=df #Este dataframe me servirá para graficar los datos originales
cont=0
for i in range(0, descriptores):
    dfnorm=dfnorm.drop([repr(i)], axis=1) #Elimino los valores de las variables no estandarizadas
print(dfnorm)

#Calculo de la matriz de promedios por clases y descriptores
for i in range(1, clases+1):
    grupo=dfnorm['clases']==i
    grupofiltrado=dfnorm[grupo]#grupofiltrado es el conjunto de datos filtrados por clases
    #print(grupofiltrado)
    for j in range(0, descriptores):
        matrizprom[i-1][j]=grupofiltrado[repr(j)+repr(j)].mean()
#print(matrizprom)

#Calculo del Grado de Adecuación Marginal MAD
for i in range(0, clases):
    for j in range(0, descriptores):
        for k in range (0, len(dfnorm)):
            dfnorm.loc[k, 'MAD'+repr(i)+repr(j)]= matrizprom[i][j]**(dfnorm.loc[k,repr(j)+repr(j)])*(1-matrizprom[i][j])**(1-dfnorm.loc[k,repr(j)+repr(j)])
           
dfnorm['GADNIC']=0.5
#Calculo del Grado de Adecuación Global
for i in range(0, clases):
    for j in range(0, len(dfnorm)):
        dfnorm.loc[j,'MaximoC'+repr(i)]=max(dfnorm.loc[j,'MAD'+repr(i)+'0'],dfnorm.loc[j,'MAD'+repr(i)+'1'], dfnorm.loc[j,'MAD'+repr(i)+'2'] , dfnorm.loc[j,'MAD'+repr(i)+'3'], dfnorm.loc[j,'MAD'+repr(i)+'4'], dfnorm.loc[j,'MAD'+repr(i)+'5'], dfnorm.loc[j,'MAD'+repr(i)+'6'], dfnorm.loc[j,'MAD'+repr(i)+'7'], dfnorm.loc[j,'MAD'+repr(i)+'8'])
        dfnorm.loc[j,'MinimoC'+repr(i)]=min(dfnorm.loc[j,'MAD'+repr(i)+'0'],dfnorm.loc[j,'MAD'+repr(i)+'1'], dfnorm.loc[j,'MAD'+repr(i)+'2'] , dfnorm.loc[j,'MAD'+repr(i)+'3'], dfnorm.loc[j,'MAD'+repr(i)+'4'], dfnorm.loc[j,'MAD'+repr(i)+'5'], dfnorm.loc[j,'MAD'+repr(i)+'6'], dfnorm.loc[j,'MAD'+repr(i)+'7'], dfnorm.loc[j,'MAD'+repr(i)+'8'])
        dfnorm.loc[j,'GAD'+repr(i)]=alfa*dfnorm.loc[j,'MinimoC'+repr(i)]+(1-alfa)*dfnorm.loc[j,'MaximoC'+repr(i)]
for i in range(0,len(dfnorm)):
    dfnorm.loc[i,'index']=max(dfnorm.loc[i,'GAD0'],dfnorm.loc[i,'GAD1'], dfnorm.loc[i,'GADNIC'])
    if dfnorm.loc[i,'index']==dfnorm.loc[i,'GADNIC']:
        dfnorm.loc[i,'clase1']=99
    elif (dfnorm.loc[i,'index']==dfnorm.loc[i,'GAD0']):
        dfnorm.loc[i,'clase1']=1
    else:
        dfnorm.loc[i,'clase1']=2
ruta='C:\Python\doctorado\LAMDACancerClasificacion.csv'#EXPORTACIÓN DE LOS RESULTADOS
dfnorm.to_csv(ruta)
#print('este es el ultimo dataaset')
dflisto= dfnorm.iloc[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 36]]#Este DataFrame esta listo para hacer grafico y bondad de ajuste
#print(dflisto)
dflisto['clase1']=dflisto['clase1'].astype(int)#convierto la column de la clasificaciòn en valores enteros
#print(dflisto['clase1'])
accuracy1= accuracy_score(dflisto['clases'], dflisto['clase1'])
f1_1= f1_score(dflisto['clases'], dflisto['clase1'], average= 'micro')
precision= precision_score(dflisto['clases'], dflisto['clase1'], average='macro')
print('El valor de F1 Score es : ' , f1_1)
print('la accuracy es : ' , accuracy1)
print('La precisión es : ' , precision)

"""********************AQUI TERMINA LAMDA CLASIFICACIÓN CON LAS MÉTRICAS***************************************************************************"""

"""********************GRAFICO DE LOS DATOS ORIGINALES*************************************************************************************"""
#dfnormorig=dfnormorig.drop(['clases'], axis=1)#borro la columna de la etiqueta
dfnormorig=dfnormorig.drop(['0' , '1', '2', '3', '4', '5', '6', '7', '8'], axis=1)#borro la columna de descriptor 1 no estandarizados
print('dataset original estandarizado')
print(dfnormorig)
pca=PCA(n_components=2)
#Cuando se crea el modelo, tiene  que ser con la columna de la etiqueta mosca con esto
pca_brest=pca.fit_transform(dfnormorig) #Contiene las coordenadas de los individuos esto implica que se hace la transformación lineal
pca_BrestLAMDA_df=pd.DataFrame(data=pca_brest, columns= ['Componente_1', 'Componente_2']) #creación del data frame con las coordenadas de los dos componentes 
pca_nombres_brest=pd.concat([pca_BrestLAMDA_df, dfnorm[['clases']]], axis=1)#Agrego la columna de la etiqueta
print(pca_BrestLAMDA_df)
print(pca_nombres_brest)
fig=plt.figure(figsize=(10,10))
ax=fig.add_subplot(1,1,1)
ax.set_xlabel('Componente 1', fontsize=15)
ax.set_ylabel('Componente 2', fontsize=15)
ax.set_title('Data Brest Cancer Original', fontsize=20)
color_theme= np.array(['red', 'red', 'blue'])
ax.scatter(x=pca_BrestLAMDA_df.Componente_1 , y= pca_BrestLAMDA_df.Componente_2, c=color_theme[pca_nombres_brest.clases], s= 10, marker='.')
plt.show()
""" *********************************TERMINA GRAFICO ORIGINAL****************************************************************************"""

"""***************************GRAFICO CON LA CLASIFICACIÓN DEL MODELO LAMDA ORIGINAL*******************************************************"""
#Se trabaja con el dataframe dflisto que contiene solo las variables y las clases arrojadas por el modelo y por los datos
#print(dflisto) 
dflisto1=dflisto #Voy a trabajar con dflisto1 por cuestión de las etiquetas
dflisto1=dflisto1.drop(['clases'], axis=1)#borro la columna de la etiqueta original y le dejo la etiqueta del modelo
dflisto1=dflisto1.drop(dflisto1[dflisto1['clase1']==99].index)#elimnino los registros 'NIC' de la columna clase del modelo LAMDA
dflisto1=dflisto1.reset_index()# cuadno se eliminan valores, en esta caso el nic, hay que resetear el índice para no tener problemas
dflisto1=dflisto1.drop(['index'], axis=1)#borro la columna del index anterior
ruta2='C:\Python\doctorado\LAMDACancerClasificacion_modelo.csv'#EXPORTACIÓN DE LOS RESULTADOS
dflisto1.to_csv(ruta2)
pca=PCA(n_components=2)
#print('data grafico echar ojo')
#print(dflisto1)
pca_brest_modelo=pca.fit_transform(dflisto1) #Contiene las coordenadas de los individuos esto implica que se hace la transformación lineal
pca_BrestLAMDA1_df=pd.DataFrame(data=pca_brest_modelo, columns= ['Componente1', 'Componente2']) #creación del data frame con las coordenadas de los dos componentes 
pca_nombres_brest1=pd.concat([pca_BrestLAMDA1_df, dflisto1[['clase1']]], axis=1)#Agrego la columna de la etiqueta arrojada por el modelo
print('data del componente')
print(pca_brest_modelo)
print(pca_BrestLAMDA1_df)
print(pca_nombres_brest1)
fig=plt.figure(figsize=(10,10))
ax=fig.add_subplot(1,1,1)
ax.set_xlabel('Componente 1', fontsize=15)
ax.set_ylabel('Componente 2', fontsize=15)
ax.set_title('Data Brest Cancer Clasificación LAMDA', fontsize=20)
color_theme= np.array(['red', 'red', 'blue'])
ax.scatter(x=pca_BrestLAMDA1_df.Componente1 , y= pca_BrestLAMDA1_df.Componente2, c=color_theme[pca_nombres_brest1.clase1], s= 10, marker='.')
plt.show()
"""**********************************************************FIN DEL PROGRAMA*******************************************"""
