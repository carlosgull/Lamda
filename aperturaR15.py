"""Este código realiza la importación de un archivo que se encuentra estructurado como  una matriz de datos para
ello es importante el uso de la libreria pandas, la misma tiene que ser importada, ya que no se encuentra en e
el paquete estandard de python """
import os, time, pandas, numpy, colorama, math
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

"""es importante conocer el nombre del archivo que se importará así como también la dirección donde 
se encuentra el mismo"""
print(os.getcwd()) 
archivo=open("carlos.csv","w")
#Hay que tener claro las caracteristicas del archivo, si es csv son archivos separados por (,)
#en el delimitador hay que colocar \t para indicarle que se separan por tabulador
datos=pandas.read_csv('R15.csv',  header=0, delimiter=';', usecols=range(0,3))#columnas del dataset
#creo un dataframe que me permitirá un mejor manejo de la estructura
df=pandas.DataFrame(datos)
clases=15
descriptores=2
mad=numpy.zeros((clases,descriptores))
matrizprom=numpy.zeros((clases,descriptores))
gad=numpy.zeros((clases,descriptores))
print(mad)
print(matrizprom)
print(df) #imprimo ahora el dataframe
#creo tantas collumnas como descriptores una columna nueva para normalizar cada uno de los descriptores
for i in range(0,len(df)):
    df.loc[i, 1]=((df.loc[i,'v1']-df['v1'].min())/(df['v1'].max()-df['v1'].min()))
    df.loc[i, 2]=((df.loc[i,'v2']-df['v2'].min())/(df['v2'].max()-df['v2'].min()))
print(df)
#calcular el grado de adecuacion marginal (MAD) para el descriptor normalizado
#se calcula el promedio del descriptor por cada clase (ro) Ecuacion (3)
grupo=df['clases']==1#filtro el data frame con la clase que me interesas
grupo.head()
grupof=df[grupo] #creo un nuveo dataframe con los datos comopletos pero filtrados
print(grupof)
prom11= grupof[1].mean()#calculo el promedio del data frame filtrado promedio del descriptor 1 de la clase 1 
prom12= grupof[2].mean()#calculo el promedio del data frame filtrado promedio del descriptor 1 de la clase 1 

print('el promedio del primer descriptor es : ' , prom11 , prom12) #promedio de la clase 1

#ahora se calcula el MAD Ecuación 3 es un valor para cada registro i dentro de la clase, eso 
#implica que se tiene que crear una columna adicional con el valor del MAD para cada descriptor
print(grupof)
#Necesito calcular el promedio por clase y por descriptor para luego almacenarlo en la matriz
cont=0
for i in range(1, clases+1):
    grupof2=df['clases']==i
    grupofiltrado=df[grupof2]
    print(grupofiltrado)
 
    print(i)
    for j in range(0, descriptores):
        matrizprom[cont][j]=grupofiltrado[j+1].mean()
    cont+=1
print('matriz de los promedios es : ' , matrizprom)
#ahora falta calcular el mad con ec 3 o 5 para cada individuo
#tengo que crear el calculo del MAD para cada descriptor haciendo uso de la matriz de los promedios
aux=0
for i in range(1, clases+1):
    for j in range(0,descriptores):
        mad[aux][j]= matrizprom[aux][j]
    aux+=1
print('la matriz mad es la siguiente : ' , mad)
print(df)
for k in range(1,3): #la k representa el número de descriptores
    print(k)
    for j in range(0,15): #la j representa el número de clases
        for i in range(0,len(df)): #la i representa el número de observaciones 
            df.loc[i,('MAD'+repr(k))]=matrizprom[j][k-1]**(df.loc[i,k])*(1-matrizprom[j][k-1])**(1-df.loc[i,k])
print(df)
df.head()
print(df['clases'])

#calculo del Indice de Adecuación Marginal MAD
tclases=[40] #VECTOR QUE CONTIENE EL NÚMERO DE OBSERVACIONES POR CLASES
for k in range(0,2): # k representa el número de descriptores
    o=0
    for j in range(0,14): # represnta el número de clases
        o=0
        tamano=tclases[0]
        #mosca con la lineaa siguiente inicialmente el lazo llega a 50 pero cancer es desbalanceado
        for i in range(0,tamano): #represente el total de observaciones por clases
            if df.loc[o,'clases']==j+1:
                df.loc[o,('MAD'+repr(k+1))]=matrizprom[j][k]**(df.loc[o,k+1])*(1-matrizprom[j][k])**(1-(df.loc[o,k+1]))
            o+=1
os.system('cls')
#ahora se tiene que crear la clase no informativa, con la ecuación 3 o 5 pero con promedio =0.5 y para cada descriptor
for i in range(0,len(df)):
    df.loc[i, 'MADNIC']=0.5**(df.loc[i,1])*0.5**(1-df.loc[i,1])
#ahora se tiene que calcular el GAD dentro de cada clase tiene que estar en una matriz
b=0
for i in range(0,len(df)):
    df.loc[b,'GADX']=max(df.loc[b,'MAD1'],df.loc[b,'MAD2'])
    b+=1
print(df)
print(matrizprom)
#calculo del MAD definitivo
for j in range(1,16): #clases
    for k in range(1,3): #descriptores
        for i in range(0,len(df)): #observaciones
            df.loc[i,('MAD'+repr(j)+ repr(k))]=matrizprom[j-1][k-1]**df.loc[i,k]*(1- matrizprom[j-1][k-1])**(1-df.loc[i,k])
df['GADNIC']=0.5

for j in range(1,16):
    for i in range(0,len(df)):
        df.loc[i,'MaximoC'+repr(j)]=max(df.loc[i,'MAD'+repr(j)+'1'],df.loc[i,'MAD'+repr(j)+'2'] )
        df.loc[i,'MinimoC'+repr(j)]=min(df.loc[i,'MAD'+repr(j)+'1'],df.loc[i,'MAD'+repr(j)+'2'])
        df.loc[i,'GAD'+repr(j)]=0.9*df.loc[i,'MinimoC'+repr(j)]+(1-0.9)*df.loc[i,'MaximoC'+repr(j)]
for j in range(1,16):
    for i in range(0,len(df)):
        df.loc[i,'index']=max(df.loc[i,'GAD1'],df.loc[i,'GAD2'], df.loc[i,'GADNIC'])
        if df.loc[i,'index']==0.5:
            df.loc[i,'clase']=9
        elif (df.loc[i,'index']==df.loc[i,'GAD1']):
            df.loc[i,'clase']=1
        elif (df.loc[i,'index']==df.loc[i,'GAD2']):
            df.loc[i,'clase']=2
      
#GADNIC1=numpy.avg(mgad[:,0])
#dflisto=pandas.DataFrame([ df[1], df[2],df[3],df[4], df['clase']])
#dflisto=pandas.DataFrame.transpose(dflisto)
print(df)
#dflisto=dflisto.drop(dflisto[dflisto['clase']=='NIC'].index)#elimnino los registros 'NIC' de la columna clase
"""pca=PCA(n_components=2)
dflisto1=dflisto.drop(['clase'], axis=1)#borro la columna de etiqueta
pca_iris=pca.fit_transform(dflisto1) #Contiene las coordenadas de los individuos esto implica que se hace la transformación lineal 
pca_irisLAMDA_df=pandas.DataFrame(data=pca_iris, columns= ['Componente_1', 'Componente_2']) #creación del data frame con las coordenadas de los dos componentes 
pca_nombres_iris=pandas.concat([pca_irisLAMDA_df, dflisto[['clase']]], axis=1)
pca_irisLAMDA_df=pca_nombres_iris.drop(pca_nombres_iris[pca_nombres_iris['clase']=='NIC'].index)
print(pca_irisLAMDA_df)
fig=plt.figure(figsize=(10,10))
ax=fig.add_subplot(1,1,1)
ax.set_xlabel('Componente 1', fontsize=15)
ax.set_ylabel('Componente 2', fontsize=15)
ax.set_title('Componentes Principales', fontsize=20)
color_theme= numpy.array(['red', 'blue', 'black', 'green'])
ax.scatter(x=pca_irisLAMDA_df.Componente_1 , y= pca_irisLAMDA_df.Componente_2,  s= 10, marker='*')
plt.show()
print(pca_irisLAMDA_df)
pca_irisLAMDA_df.to_csv('salidairisLAMDA9.csv', header=True,index= False, sep= ';')
ruta='C:\Python\Python37-32\doctorado\LAMDAIRisClasificacion.csv'#EXPORTACIÓN DE LOS RESULTADOS
df.to_csv(ruta)"""
recall1=recall_score(df['clases'], df['clase'], average= 'weighted')
accuracy1= accuracy_score(df['clases'], df['clase'])
f1_1= f1_score(df['clases'], df['clase'], average= 'micro')
precision= precision_score(df['clases'], df['clase'], average='macro')
print('la accuracy es : ' , accuracy1)
print('El valor de F1 Score es : ' , f1_1)
print('La precisión es : ' , precision)
print('REcall es : ' , recall1)