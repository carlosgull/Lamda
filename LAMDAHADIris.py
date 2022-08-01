import os, time, pandas, numpy, colorama, math
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
datos=pandas.read_csv('IrisHAD.csv',  header=0, delimiter='\t', usecols=range(0,5))#importacion del dataset las columnas estan etiquetadas con numeros para mayor comodidad
print(datos)
df=pandas.DataFrame(datos)#CREACION DE DATA FRAME PARA LA MANIPULACIÓN DE LOS DATOS
clases=3#CLASES DEL DATASET
descriptores=4#DESCRIPTORES DEL DATASET
prom=numpy.zeros((clases,descriptores)) #MATRIZ DE PROMEDIOS
mgad=numpy.zeros((clases,clases))
alfa=0.9
os.system('cls')
df1=df
df1=df1.drop(['clases'], axis=1)#Elimino la columna de las etiquetas
#print(df1) #dataset de trabajo sin las etiquetas
#print(df) #dataset original con etiquetas
df1=(df1-df1.min())/(df1.max()-df1.min()) #normalizo la data de acuerdo al criterio maximo minimo
df1=pd.concat([df1, df[['clases']]], axis=1)#Agrego la columna de la etiqueta de los datos originales
print(df1)
for k in range(1,descriptores+1): #MCREACIÓN DE LA matriz de promedios 
    for h in range(1, clases+1): #NÚMERO DE CLASES
        filtro=0
        obs=0
        for l in range(0,len(df1)):
            if df1.loc[l,'clases']==h:
                filtro+=df1.loc[l,repr(k)]
                obs+=1
        filtro=(filtro)/obs
        prom[h-1][k-1]=filtro
print(prom)#imprime la matriz de promedios clases X descriptores
for i in range(0,len(df1)):#calculo del MAD
    for j in range(1,clases+1): #CLASES
        for k in range(1,descriptores+1): #DESCRIPTORES
            df1.loc[i, 'MAD'+repr(j)+repr(k)]=prom[j-1][k-1]**df1.loc[i,repr(k)]*(1-prom[j-1][k-1] )**(1-df1.loc[i,repr(k)])
    
for i in range(0,len(df1)):#OPERADORES DEL T Y S COMBINACIONES LINEALES DE LOS MADS EN ESTE CASO MAXIMO Y MÍNIMO
    for j in range(1,clases+1): #CLASES
        df1.loc[i,'MaxC'+repr(j)]= max(df1.loc[i, 'MAD'+repr(j)+'1'], df1.loc[i, 'MAD'+repr(j)+'2'], df1.loc[i, 'MAD'+repr(j)+'3'], df1.loc[i, 'MAD'+repr(j)+'4'])
        df1.loc[i,'MinC'+repr(j)]= min(df1.loc[i, 'MAD'+repr(j)+'1'], df1.loc[i, 'MAD'+repr(j)+'2'], df1.loc[i, 'MAD'+repr(j)+'3'], df1.loc[i, 'MAD'+repr(j)+'4'])
        df1.loc[i,'GAD'+repr(j)]= alfa*df1.loc[i,'MinC'+repr(j)]+(1-alfa)*df1.loc[i,'MaxC'+repr(j)]
c=0
for l in range (1,clases+1): #CLASES
    for j in range (1,clases+1):#CLASES
        suma=0; obs1=0
        for i in range(0,len(df)):
            if l==df1.loc[i,'clases']:
                suma+=df1.loc[i,'GAD'+repr(j)]
                obs1+=1
        promed=suma/obs1
        mgad[j-1][c]=promed
    c+=1

GADNIC1=numpy.average(mgad[:, 0])
GADNIC2=numpy.average(mgad[:, 1])
GADNIC3=numpy.average(mgad[:, 2])

print('matriz magad', mgad)

for i in range(0,clases):#CALCULO DEL ADGAD EN FUNCIÓN A LOS PROMEDIOS MGAD
    for j in range(0,clases):
        for k in range(0,len(df1)):
            df1.loc[k, 'ADGAD'+repr(i+1)+repr(j+1)]= mgad[j][i]**df1.loc[k,'GAD'+repr(j+1)]*(1-mgad[j][i] )**(1-df1.loc[k,'GAD'+repr(j+1)] )

for j in range(1,clases+1):#CALCULO DEL HAD EN FUNCIÓN A LOS PROMEDIOS ADGAD
    for k in range(0,len(df1)):
        df1.loc[k, 'HAD'+repr(j)]= df1.loc[k,'ADGAD'+repr(j)+'1']+ df1.loc[k,'ADGAD'+repr(j)+'2']+df1.loc[k,'ADGAD'+repr(j)+'3']
vector=numpy.zeros(clases)#este vector se crea para obtener el segundo mayor HAD
#DETERMINACIÓN DEL MAXIMO HAD

for k in range(0,len(df1)):      
    df1.loc[k,'MAXHAD']= max(df1.loc[k,'HAD1'], df1.loc[k,'HAD2'], df1.loc[k,'HAD3']) 
nc1=0; nc2=0; nc3=0 #ESTOS VALORES CONTIENEN EL NÚMERO DE OBSERVACIONES POR CLASES
for k in range(0,len(df1)):
    if df1.loc[k,'MAXHAD']== df1.loc[k,'HAD1']:
        df1.loc[k,'EI']=1
        if df1.loc[k, 'GAD1']> GADNIC1:
            df1.loc[k,'INDEX']=1
            nc1+=1#DETERMINA EL NÚMERO DE OBSERVACIONES EN LA CLASE 1
        else:
            df1.loc[k,'INDEX']=99    
    elif df1.loc[k,'MAXHAD']== df1.loc[k,'HAD2']:
            df1.loc[k,'EI']=2
            if df1.loc[k, 'GAD2']> GADNIC2:
                df1.loc[k,'INDEX']=2
                nc2+=1
            else:
                df1.loc[k,'INDEX']=99
    elif df1.loc[k,'MAXHAD']==df1.loc[k,'HAD3']:
            df1.loc[k,'EI']=3
            if df1.loc[k, 'GAD3']> GADNIC3:
                df1.loc[k,'INDEX']=3
                nc3+=1
            else:
                df1.loc[k,'INDEX']=99
dflisto= df1.loc[:, ['1', '2', '3', '4', 'clases', 'INDEX']]#Este DataFrame esta listo para hacer grafico y bondad de ajuste
ruta='C:\Python\doctorado\LAMDAHADIris.csv'#EXPORTACIÓN DE LOS RESULTADOS   
dflisto.to_csv(ruta)

accuracy1= accuracy_score(dflisto['clases'], dflisto['INDEX'])
f1_1= f1_score(dflisto['clases'], dflisto['INDEX'], average= 'micro')
recall1=recall_score(dflisto['clases'], dflisto['INDEX'], average= 'weighted')
precision= precision_score(dflisto['clases'], dflisto['INDEX'], average='macro')
print('la accuracy es : ' , accuracy1)
print('El valor de F1 Score es : ' , f1_1)
print('REcall es : ' , recall1)
print('La precisión es : ' , precision)
""" ****************** CREACION DEL GRAFICO LAMDA HAD****************"""
dfgrafico= dflisto# dflisto contienen la data con las etiquetas originales (clases) y las del modelo (INDEX)
print(dfgrafico)
dfgrafico= dfgrafico.drop(['clases'],axis=1) #Elimino la etiqueta clases
dfgrafico= dfgrafico.drop(['INDEX'], axis=1) #Elimino la etiqueta clases
print(dfgrafico)
pca=PCA(n_components=2)#Modelo de componentes principales
pca_1=pca.fit_transform(dfgrafico) #Contiene las coordenadas
pca_df=pd.DataFrame(data=pca_1, columns= ['Componente_1', 'Componente_2']) #creación del data frame con las coordenadas de los dos componentes 
pca_nombres=pd.concat([pca_df, dflisto[['INDEX']]], axis=1) #concatenar las etiquetas del modelo con los individuos
pca_nombres['INDEX']=pca_nombres['INDEX'].astype(int)
#CREACIÓN DEL GRÁFICO DEL MODELO LAMDA-HAD
fig=plt.figure(figsize=(10,10))
ax=fig.add_subplot(1,1,1)
ax.set_xlabel('Componente 1', fontsize=15)
ax.set_ylabel('Componente 2', fontsize=15)
ax.set_title('Iris LAMDA-HAD', fontsize=20)
color_theme= np.array(['red', 'blue', 'red', 'purple'])    
ax.scatter(x=pca_nombres.Componente_1 , y= pca_nombres.Componente_2, c=color_theme[pca_nombres.INDEX], s= 20, marker='.')
plt.show()