"""Codificacion del algoritmo lamda clustering primera versión para el archivo de datos Aggregation"""
import os, time, pandas, numpy, colorama, math
from numpy.core.fromnumeric import ravel
import numpy as np 
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
#Hay que tener claro las caracteristicas del archivo, si es csv son archivos separados por (,)
#en el delimitador hay que colocar \t para indicarle que se separan por tabulador
datos=pandas.read_csv('C:/Python/electrico/ConsConsolidado.csv', delimiter=';', header=0)#columnas del dataset
#creo un dataframe que me permitirá un mejor manejo de la estructura
df=pandas.DataFrame(datos)#Dataset original con la etiqueta
print(df)
"""
df1=df.drop(['clases'], axis=1)# elimino la columna de las clases y trabajo con el Dataframe df1
df1=(df1-df1.min())/(df1.max()-df1.min()) #normalizo la data de acuerdo al criterio maximo minimo
print(df1) #Dataframe normalizado para trabajar 788 individuos y 2 descriptores 7 grupos supuestamente
k=1 #indica el cluster
n1=0 #indica el número de observaciones del cluster 1
df1['PromNIC']=0.5 #promedio de la clase no informativa
df1['n'+repr(k)]=0 #número de elementos del cluster k
alfa=0.9
df1['tnic'] =1 /   (1+ (1- 0.5)/0.5 +( (1- 0.5) / (0.5)   ))
df1['snic']= 1 - (1/(1+                  ( 0.5/(1-0.5)+                                         0.5/(1-0.5))))
df1['GADNIC']=alfa*df1['tnic']+(1-alfa)*df1['snic']

####### **************************EVALUaCION DE LA PRIMERA OBSERVACION *******************************################################################
df1.loc[0, 'promi'+repr(k)+ '1']= df1.loc[0, '1'] #AL SER LA PRIMERA OBSERVACION EL PROMEDIO INICIAL ES IGUAL A LA OBSERVACION S
df1.loc[0, 'promi'+repr(k)+ '2']= df1.loc[0, '2']
df1.loc[0, 'promf'+repr(k) + '1' ]= df1.loc[0, '1']  #promedio final de cluster 1 descriptor 1 
df1.loc[0, 'promf'+repr(k) +'2' ]= df1.loc[0, '2'] #promedio final de cluster 1 descriptor 2
alfa=0.9
df1.loc[0,'Cluster']=1
df1['n'+repr(k)]=1 #NUMERO DE ELEMENTO DEL CLUSTER
df1['POS'+repr(k)]=int(0) #POSICION DE LA ÚLTIMA OBSERVACION DEL CLUSTER PARA LUEGO SABER DONDE ESTA EL PROMEDIO DEL CLUSTER
contador=0
df1['dnb']=0.06
for l in range(1,len(df1)): 
    for h in range(1, k+1): #hace referencia al cluster
        df1.loc[l, 'promi'+repr(h)+ '1']= df1.loc[df1.loc[0,'POS'+repr(k)], 'promi'+repr(h)+'1']
        df1.loc[l, 'promi'+repr(h)+ '2']= df1.loc[df1.loc[0,'POS'+repr(k)], 'promi'+repr(h)+'2']
        df1.loc[l, 'promf'+repr(h) + '1' ]= df1.loc[df1.loc[0,'POS'+repr(k)], 'promi' +repr(h) +'1'] + (df1.loc[l, '1'] - df1.loc[df1.loc[0,'POS'+repr(k)], 'promi'+repr(h) +'1'])/(df1.loc[df1.loc[0,'POS'+repr(k)], 'n'+repr(h)]+1) #promedio final de cluster 1 descriptor 1 
        df1.loc[l, 'promf'+repr(h) +'2' ]= df1.loc[df1.loc[0,'POS'+repr(k)], 'promi'+repr(h) +'2'] + (df1.loc[l, '2'] - df1.loc[df1.loc[0,'POS'+repr(k)], 'promi' +repr(h) +'2'])/(df1.loc[df1.loc[0,'POS'+repr(k)], 'n'+repr(h)]+1)       #promedio final de cluster 1 descriptor 2
        df1.loc[l, 'CMAD'+repr(h) + '1' ]=   1/ (1+abs((df1.loc[l, '1']-df1.loc[l, 'promf'+repr(h) + '1'])))
        df1.loc[l, 'CMAD'+repr(h) + '2' ]=  1/(1+abs((df1.loc[l, '2']-df1.loc[l, 'promf'+repr(h) + '2'])))
        df1.loc[l, 'dist'+repr(h)]=(abs(df1.loc[l, '1']-df1.loc[l, 'promf'+repr(h) + '1'])+ abs(df1.loc[l, '2']-df1.loc[l, 'promf'+repr(h) + '2']))/2#CALCULO DE LA DISTANCIA DEL INDIVIDUO X AL CENTROIDE DE CADA CLUSTER ECUACION (13)
        df1.loc[l, 'dist1'+repr(h)]=abs(df1.loc[l, 'dist'+repr(h)]-df1.loc[l, 'dnb'])
        df1.loc[l, 'k'+repr(h)]= df1.loc[l, 'dnb'] / ( df1.loc[l,'dnb']+ df1.loc[l, 'dist1'+repr(h)])
        if df1.loc[l,'dist'+repr(h)]<df1.loc[l,'dnb']:
            df1.loc[l, 'k'+repr(h)]=1
        df1.loc[l, 'RMAD'+repr(h)+ '1']= df1.loc[l, 'k'+repr(h)]* df1.loc[l,'CMAD'+repr(h)+'1']
        df1.loc[l, 'RMAD'+repr(h)+ '2']= df1.loc[l, 'k'+repr(h)]* df1.loc[l, 'CMAD'+repr(h)+'2']
        df1.loc[l, 'MaxRMADs'+repr(h)]=max([ df1.loc[l, 'RMAD'+repr(h) + '1' ],  df1.loc[l, 'RMAD'+repr(h) + '2' ]])
        df1.loc[l, 'MinRMADs'+repr(h)]=min([ df1.loc[l, 'RMAD'+repr(h) + '1' ],  df1.loc[l, 'RMAD'+repr(h) + '2' ]])
        df1.loc[l,'t'+repr(h)]=  1/ (1+ ((1- df1.loc[l,'MaxRMADs'+repr(h)])/ df1.loc[l,'MaxRMADs'+repr(h)]) + ((1- df1.loc[l,'MinRMADs'+repr(h)])/ df1.loc[l,'MinRMADs'+repr(h)]))
        df1.loc[l, 's'+repr(h)]= 1- (  1/ (1+ ( df1.loc[l, 'MaxRMADs'+repr(h)]/ (1- df1.loc[l, 'MaxRMADs'+repr(h)]) +    df1.loc[l,'MinRMADs'+repr(h)]/ (1- df1.loc[l,'MinRMADs'+repr(h)]))))
        df1.loc[l,'GAD'+repr(h)]= alfa*df1.loc[l, 't'+repr(h)]+(1-alfa)*df1.loc[l,'s'+repr(h)]
        mayor= df1.loc[l,'GAD1']
        poscluster=1
    for i in range(1, k+1):
        if mayor<df1.loc[l, 'GAD'+repr(i)]:
            mayor=df1.loc[l, 'GAD'+repr(i)]
            poscluster=i #NÚMERO DEL CLUSTER EN EVALUACION 
    index=max([mayor,df1.loc[0,'GADNIC']])
    if index==mayor:
        df1.loc[l,'Cluster']=poscluster
        df1['POS'+repr(poscluster)]=int(l)
        df1['n'+repr(poscluster)]= df1.loc[0,'n'+repr(poscluster)]+1
        df1.loc[l,'n']= df1.loc[0,'n'+repr(k)]  #ojo con esta sentencia
        df1.loc[l,'promi' + repr(poscluster) +'1']=df1.loc[l, 'promf'+repr(poscluster) + '1']
        df1.loc[l,'prom1']=df1.loc[l, 'promi'+repr(poscluster) + '1']
        df1.loc[l,'promi' + repr(poscluster) +'2']=df1.loc[l, 'promf'+repr(poscluster) + '2']
        df1.loc[l,'prom2']=df1.loc[l, 'promi'+repr(poscluster) + '2']
    else:
        k+=1
        df1.loc[l,'Cluster']=k
        df1['POS'+repr(k)]=int(l)
        df1['n'+repr(k)]=1
        df1.loc[l,'n']=df1.loc[0,'n'+repr(k)]
        df1.loc[l,'promi' + repr(k) +'1']=df1.loc[l, '1']
        df1.loc[l,'prom1']=df1.loc[df1.loc[0,'POS'+repr(poscluster)],'promi' + repr(k) +'1']       # Promedio final cluster k descriiptor 1  
        df1.loc[l,'promi' + repr(k) +'2']=df1.loc[l, '2']
        df1.loc[l,'prom2']=df1.loc[df1.loc[0,'POS'+repr(poscluster)],'promi' + repr(k) +'2'] # Promedio final cluster k descriiptor 2 

#print(df1)
#GADNIC1=numpy.average(mgad[:, 0]) para calcular los promedios 
contador=0
for z in (1, k):
    for contador in (0, df1.loc[0,'n'+repr(z)]):
        df1.loc[contador,'n']= int(df1.loc[0,'n'+repr(z)])
        contador+=1

#CREACION DE GRUPOS PARA LUEGO FUSIONARLOS SEGÙN MERGIN PROCESS 





datasalida=df1[[ 'prom1', 'prom2', '1','2','Cluster','n', 'n1', 'n2', 'n3' ,'promi11', 'promi12', 'POS1']]
#Los clusters deben ser , creados por grupos según Definición 5 Hay que hacerlos

ruta='C:\Python\doctorado\LAMDARDAggregationModelo.csv'#EXPORTACIÓN DE LOS RESULTADOS
datasalida.to_csv(ruta)
print(datasalida)

def compacta(clu): #Función que calcula la compactación del cluster vecino (clu) de tamaño (nc), segùn Definición 7 LAMDA-RD
    suma=0
    for j in (0, len(clu)-1):
        for k in (j+1, len(clu)):
            for i in (1, 2): #Número de descriptores
                suma+=abs(clu.loc[j,repr(i)]- clu.loc[k,repr(i)])

"""
""" 
""" """ ***************************************************************"""
"""CREACIÓN DEL GRAFICO DEL CONJUNTO DE DATOS ORIGINAL
EN ESTE CASO SE PUEDE HACER YA QUE SE CUENTA CON LAS ETIQUETAS PERO NO 
SIEMPRE SUCEDE"""
"""
df2=df1.iloc[:, [0, 1]]# data lista para aplicar el ACP
print(df2)
pca=PCA(n_components=2)# creo el modelo 
pca_aggre_modelo=pca.fit_transform(df2)
pca_aggre_df=pandas.DataFrame(data=pca_aggre_modelo, columns= ['Componente1', 'Componente2']) #creación del data frame con las coordenadas de los dos componentes 
pca_nombres_aggre=pandas.concat([pca_aggre_df, df[['clases']]], axis=1)#Agrego la columna de la etiqueta de los datos originales
pca_nombres_aggre['clases']=pca_nombres_aggre['clases'].astype(int)# convierto la columna de la etiqueta en valores enteros 
fig=plt.figure(figsize=(10,10))
ax=fig.add_subplot(1,1,1)
ax.set_xlabel('Componente 1', fontsize=15)
ax.set_ylabel('Componente 2', fontsize=15)
ax.set_title('Data Aggregation Original', fontsize=20)
color_theme= np.array(['red', 'blue', 'black', 'red', 'orange', 'green', 'brown', 'purple'])
ax.scatter(x=pca_aggre_df.Componente1 , y= pca_aggre_df.Componente2, c=color_theme[pca_nombres_aggre.clases], s=40, marker='.')
plt.savefig("AggregationOrig.png", bbox_inches='tight')
plt.show()"""
""" **************GRAFICO DEL MODELO  LAMDA CLASIFICACION*********************************************"""
"""
pca_nombres_aggre1=pandas.concat([pca_aggre_df, datasalida[['Cluster']]], axis=1)#Agrego la columna de la etiqueta del modelo
pca_nombres_aggre1['Cluster']=pca_nombres_aggre1['Cluster'].astype(int)# convierto la columna de la etiqueta en valores enteros 
fig=plt.figure(figsize=(10,10))
ax=fig.add_subplot(1,1,1)
ax.set_xlabel('Componente 1', fontsize=15)
ax.set_ylabel('Componente 2', fontsize=15)
ax.set_title('Data Aggregation LAMDA Clasificación', fontsize=20)
color_theme= np.array(['red', 'blue', 'black', 'red', 'orange', 'green', 'brown', 'purple'])
ax.scatter(x=pca_aggre_df.Componente1 , y= pca_aggre_df.Componente2, c=color_theme[pca_nombres_aggre1.Cluster], s=40, marker='.')
plt.savefig("AggregationLAMDAClas.png", transparent=True,  bbox_inches='tight', dpi=300)
plt.show()"""
