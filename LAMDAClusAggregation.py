"""Codificacion del algoritmo lamda clustering primera versión para el archivo de datos Aggregation"""
import os, time, pandas, numpy, colorama, math
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
#Hay que tener claro las caracteristicas del archivo, si es csv son archivos separados por (,)
#en el delimitador hay que colocar \t para indicarle que se separan por tabulador
datos=pandas.read_csv('C:/Python/doctorado/Aggregation1.csv', delimiter='\t')#columnas del dataset
#creo un dataframe que me permitirá un mejor manejo de la estructura
df=pandas.DataFrame(datos)
print(df)
df1=df.drop(['clases'], axis=1)# elimino la columna de las clases y trabajo con el Dataframe df1
df1=(df1-df1.min())/(df1.max()-df1.min()) #normalizo la data de acuerdo al criterio maximo minimo
print(df1) #Dataframe normalizado para trabajar 788 individuos y 2 descriptores
k=1 #indica el cluster
n1=0 #indica el número de observaciones del cluster 1
df1['PromNIC']=0.5 #promedio de la clase no informativa
df1['n'+repr(k)]=0 #número de elementos del cluster k
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
for l in range(1,len(df1)): 
    for h in range(1, k+1): #hace referencia al cluster
        df1.loc[l, 'promi'+repr(h)+ '1']= df1.loc[df1.loc[0,'POS'+repr(k)], 'promi'+repr(h)+'1']
        df1.loc[l, 'promi'+repr(h)+ '2']= df1.loc[df1.loc[0,'POS'+repr(k)], 'promi'+repr(h)+'2']
        df1.loc[l, 'promf'+repr(h) + '1' ]= df1.loc[df1.loc[0,'POS'+repr(k)], 'promi' +repr(h) +'1'] + (df1.loc[l, '1'] - df1.loc[df1.loc[0,'POS'+repr(k)], 'promi'+repr(h) +'1'])/(df1.loc[df1.loc[0,'POS'+repr(k)], 'n'+repr(h)]+1) #promedio final de cluster 1 descriptor 1 
        df1.loc[l, 'promf'+repr(h) +'2' ]= df1.loc[df1.loc[0,'POS'+repr(k)], 'promi'+repr(h) +'2'] + (df1.loc[l, '2'] - df1.loc[df1.loc[0,'POS'+repr(k)], 'promi' +repr(h) +'2'])/(df1.loc[df1.loc[0,'POS'+repr(k)], 'n'+repr(h)]+1)       #promedio final de cluster 1 descriptor 2
        df1.loc[l, 'MAD'+repr(h) +'1' ]=  df1.loc[l, 'promf'+repr(h) +'1' ]**(df1.loc[l, '1'])*(1- df1.loc[l, 'promf'+repr(h) +'1' ])**(1-df1.loc[l, '1'])
        df1.loc[l, 'MAD'+repr(h) +'2' ]=  df1.loc[l, 'promf'+repr(h) +'2' ]**(df1.loc[l, '2'])*(1- df1.loc[l, 'promf'+repr(h) +'2' ])**(1-df1.loc[l, '2'])
        df1.loc[l, 'MaxMAD'+repr(h)]=max([ df1.loc[l, 'MAD'+repr(h) + '1' ],  df1.loc[l, 'MAD'+repr(h) + '2' ]])
        df1.loc[l, 'MinMAD'+repr(h)]=min([ df1.loc[l, 'MAD'+repr(h) + '1' ],  df1.loc[l, 'MAD'+repr(h) + '2' ]])
        df1.loc[l, 't']= 1/ (1+  (      ((1-df1.loc[l, 'MinMAD'+repr(h)])/(df1.loc[l, 'MinMAD'+repr(h)])) +  ((1-df1.loc[l, 'MaxMAD'+repr(h)])/(df1.loc[l, 'MaxMAD'+repr(h)]))         ))
        df1.loc[l, 's']= 1- (1/ (1+  (      ((df1.loc[l, 'MinMAD'+repr(h)])/(1-df1.loc[l, 'MinMAD'+repr(h)])) +  ((df1.loc[l, 'MaxMAD'+repr(h)])/(1-df1.loc[l, 'MaxMAD'+repr(h)]))         )))
        df1.loc[l, 'GAD'+repr(h)]= alfa*df1.loc[l, 't'] + (1-alfa)*df1.loc[l, 's']
        df1.loc[l, 'tNIC']=1/ (1+  (      ((1-0.5)/(0.5)) +  ((1-0.5)/(0.5))         ))
        df1.loc[l, 'sNIC']= 1- (1/ (1+  (      ((0.5)/(1-0.5)) +  ((0.5)/(1-0.5))         )))
        df1.loc[l, 'GADNIC']= alfa*df1.loc[l, 'tNIC'] + (1-alfa)*df1.loc[l, 'sNIC']
        mayor= df1.loc[l,'GAD1']
        poscluster=1
    for i in range(1, k+1):#Calculo del elemento mayor en una fila
        if mayor<df1.loc[l, 'GAD'+repr(i)]:
            mayor=df1.loc[l, 'GAD'+repr(i)]
            poscluster=i #NÚMERO DEL CLUSTER EN EVALUACION 
    index=max([mayor,df1.loc[l, 'GADNIC']])
    if (index==mayor):
        contador+=1
        print('hola', contador)
        df1.loc[l,'Cluster']=poscluster
        df1['POS'+repr(poscluster)]=int(l)
        df1['n'+repr(poscluster)]= df1.loc[0,'n'+repr(poscluster)]+1
        df1.loc[l,'promi' + repr(poscluster) +'1']=df1.loc[l, 'promf'+repr(poscluster) + '1']
        df1.loc[l,'promi' + repr(poscluster) +'2']=df1.loc[l, 'promf'+repr(poscluster) + '2']
    else:
        contador+=1
        print('chao', contador)
        k+=1
        df1.loc[l,'Cluster']=k
        df1['POS'+repr(k)]=int(l)
        df1['n'+repr(k)]=1
        df1.loc[l,'promi' + repr(k) +'1']=df1.loc[l, '1']
        df1.loc[l,'promi' + repr(k) +'2']=df1.loc[l, '2']    
print(df1)
datasalida=df1[['1','2', 'Cluster']] 
print(datasalida)