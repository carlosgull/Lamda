import os, time, pandas, numpy, colorama, math
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
datos=pandas.read_csv('R15.csv',  header=0, delimiter=';', usecols=range(0,3))#columnas del dataset
df=pandas.DataFrame(datos)#CREACION DE DATA FRAME PARA LA MANIPULACIÓN DE LOS DATOS
clases=15#CLASES DEL DATASET
descriptores=2#DESCRIPTORES DEL DATASET
prom=numpy.zeros((clases,descriptores)) #MATRIZ DE PROMEDIOS
mgad=numpy.zeros((clases,clases))
alfa=0.9
os.system('cls')
print(df)
for i in range(0,len(df)):#ESTANDARIZACIÓN DE DATOS 
	df.loc[i, 1]=((df.loc[i,'v1']-df['v1'].min())/(df['v1'].max()-df['v1'].min()))
	df.loc[i, 2]=((df.loc[i,'v2']-df['v2'].min())/(df['v2'].max()-df['v2'].min()))
print(df)
for k in range(1,3): #MCREACIÓN DE LA matriz de promedios
    for h in range(1, 16): #NÚMERO DE CLASES
        filtro=0
        obs=0
        for l in range(0,len(df)):
            if df.loc[l,'clases']==h:
                filtro+=df.loc[l,k]
                obs+=1
        filtro=(filtro)/obs
        prom[h-1][k-1]=filtro
#print(prom)#imprime la matriz de promedios
for i in range(0,len(df)):#calculo del MAD
    for j in range(1,16): #CLASES
        for k in range(1,3): #DESCRIPTORES
            df.loc[i, 'MAD'+repr(j)+repr(k)]=prom[j-1][k-1]**df.loc[i,k]*(1-prom[j-1][k-1] )**(1-df.loc[i,k])
for i in range(0,len(df)):#OPERADORES DEL T Y S COMBINACIONES LINEALES DE LOS MADS EN ESTE CASO MAXIMO Y MÍNIMO
    for j in range(1,16): #CLASES
        df.loc[i,'MaxC'+repr(j)]= max(df.loc[i, 'MAD'+repr(j)+'1'], df.loc[i, 'MAD'+repr(j)+'2'])
        df.loc[i,'MinC'+repr(j)]= min(df.loc[i, 'MAD'+repr(j)+'1'], df.loc[i, 'MAD'+repr(j)+'2'])
        df.loc[i,'GAD'+repr(j)]= alfa*df.loc[i,'MinC'+repr(j)]+(1-alfa)*df.loc[i,'MaxC'+repr(j)]
c=0
for l in range (1,16): #CLASES
    for j in range (1,16):#CLASES
        suma=0; obs1=0
        for i in range(0,len(df)):
            if l==df.loc[i,'clases']:
                suma+=df.loc[i,'GAD'+repr(j)]
                obs1+=1
        promed=suma/obs1
        mgad[j-1][c]=promed
    c+=1

GADNIC1=numpy.average(mgad[:, 0])
GADNIC2=numpy.average(mgad[:, 1])
GADNIC3=numpy.average(mgad[:, 2])
GADNIC4=numpy.average(mgad[:, 3])
GADNIC5=numpy.average(mgad[:, 4])
GADNIC6=numpy.average(mgad[:, 5])
GADNIC7=numpy.average(mgad[:, 6])
GADNIC8=numpy.average(mgad[:, 7])
GADNIC9=numpy.average(mgad[:, 8])
GADNIC10=numpy.average(mgad[:, 9])
GADNIC11=numpy.average(mgad[:, 10])
GADNIC12=numpy.average(mgad[:, 11])
GADNIC13=numpy.average(mgad[:, 12])
GADNIC14=numpy.average(mgad[:, 13])
GADNIC15=numpy.average(mgad[:, 14])

print('matriz magad', mgad)
for i in range(0,15):#CALCULO DEL ADGAD EN FUNCIÓN A LOS PROMEDIOS MGAD
    for j in range(0,15):
        for k in range(0,len(df)):
            df.loc[k, 'ADGAD'+repr(i+1)+repr(j+1)]= mgad[j][i]**df.loc[k,'GAD'+repr(j+1)]*(1-mgad[j][i] )**(1-df.loc[k,'GAD'+repr(j+1)] )
for j in range(1,16):#CALCULO DEL HAD EN FUNCIÓN A LOS PROMEDIOS ADGAD
    for k in range(0,len(df)):
        df.loc[k, 'HAD'+repr(j)]= df.loc[k,'ADGAD'+repr(j)+'1']+ df.loc[k,'ADGAD'+repr(j)+'2']+df.loc[k,'ADGAD'+repr(j)+'3']+df.loc[k,'ADGAD'+repr(j)+'4']+df.loc[k,'ADGAD'+repr(j)+'5']+df.loc[k,'ADGAD'+repr(j)+'6']+df.loc[k,'ADGAD'+repr(j)+'7']+df.loc[k,'ADGAD'+repr(j)+'8']+df.loc[k,'ADGAD'+repr(j)+'9']+ df.loc[k,'ADGAD'+repr(j)+'10']+df.loc[k,'ADGAD'+repr(j)+'11']+df.loc[k,'ADGAD'+repr(j)+'12']+df.loc[k,'ADGAD'+repr(j)+'13']+ df.loc[k,'ADGAD'+repr(j)+'14']+ df.loc[k,'ADGAD'+repr(j)+'15']
vector=numpy.zeros(clases)#este vector se crea para obtener el segundo mayor HAD
#*******CALCULOS NECESARIOS PARA EL VECINO MÁS CERCANO*******#
for k in range(0,len(df)):      
    df.loc[k, 'MAXHAD']= max(df.loc[k,'HAD1'], df.loc[k,'HAD2'], df.loc[k,'HAD3'], df.loc[k,'HAD4'], df.loc[k,'HAD5'], df.loc[k,'HAD6'], df.loc[k,'HAD7'], df.loc[k,'HAD8'], df.loc[k,'HAD9'], df.loc[k,'HAD10'], df.loc[k,'HAD11'], df.loc[k,'HAD12'], df.loc[k,'HAD13'], df.loc[k,'HAD14'], df.loc[k,'HAD15']) 
    for z in range(1,16):
        vector[z-1]=df.loc[k, 'HAD'+ repr(z)]#ALMACENAMIENTO DE LOS VALORES EN EL VECTOR
    v=numpy.sort(vector)
    df.loc[k, 'vectorSEgundo']=v[1]
    df.loc[k, 'Ivecino']=v[1]#contiene el HAD DEL VECINO MÁS CERCANO
    for g in range(1,16):
        if df.loc[k,'Ivecino']== df.loc[k, 'HAD'+repr(g)]:
            df.loc[k, 'vecino']=g
#***AQUI TERMINA EL CALCULO DEL VECINO MÁS CERCANO*****# 
nc1=0; nc2=0; nc3=0; nc4=0; nc5=0; nc6=0; nc7=0; nc8=0; nc9=0; nc10=0; nc11=0; nc12=0; nc13=0; nc14=0; nc15=0; 
#ESTOS VALORES CONTIENEN EL NÚMERO DE OBSERVACIONES POR CLASES
for k in range(0,len(df)):
    if df.loc[k,'MAXHAD']==df.loc[k,'HAD1']:
        df.loc[k,'EI']=1
        if df.loc[k, 'GAD1']> GADNIC1:
            df.loc[k,'INDEX']=1
            nc1+=1#DETERMINA EL NÚMERO DE OBSERVACIONES EN LA CLASE 1
        else:
            df.loc[k,'INDEX']=99    
    elif df.loc[k,'MAXHAD']==df.loc[k,'HAD2']:
            df.loc[k,'EI']=2
            if df.loc[k, 'GAD2']> GADNIC2:
                df.loc[k,'INDEX']=2
                nc2+=1
            else:
                df.loc[k,'INDEX']=99
    elif df.loc[k,'MAXHAD']==df.loc[k,'HAD3']:
            df.loc[k,'EI']=3
            if df.loc[k, 'GAD3']> GADNIC3:
                df.loc[k,'INDEX']=3
                nc3+=1
            else:
                df.loc[k,'INDEX']=99
    elif  df.loc[k,'MAXHAD']==df.loc[k,'HAD4']:
            df.loc[k,'EI']=4
            if df.loc[k, 'GAD4']> GADNIC4:
                df.loc[k,'INDEX']=4
                nc4+=1
            else:
                df.loc[k,'INDEX']=99        
    elif  df.loc[k,'MAXHAD']==df.loc[k,'HAD5']:
            df.loc[k,'EI']=5
            if df.loc[k, 'GAD5']> GADNIC5:
                df.loc[k,'INDEX']=5
                nc5+=1
            else:
                df.loc[k,'INDEX']=99
    elif  df.loc[k,'MAXHAD']==df.loc[k,'HAD6']:
            df.loc[k,'EI']=6
            if df.loc[k, 'GAD6']> GADNIC6:
                df.loc[k,'INDEX']=6
                nc6+=1
            else:
                df.loc[k,'INDEX']=99   
    elif  df.loc[k,'MAXHAD']==df.loc[k,'HAD7']:
            df.loc[k,'EI']=7
            if df.loc[k, 'GAD7']> GADNIC7:
                df.loc[k,'INDEX']=7
                nc7+=1
            else:
                df.loc[k,'INDEX']=99   
    elif  df.loc[k,'MAXHAD']==df.loc[k,'HAD8']:
            df.loc[k,'EI']=8
            if df.loc[k, 'GAD8']> GADNIC8:
                df.loc[k,'INDEX']=8
                nc8+=1
            else:
                df.loc[k,'INDEX']=99   
    elif  df.loc[k,'MAXHAD']==df.loc[k,'HAD9']:
            df.loc[k,'EI']=9
            if df.loc[k, 'GAD9']> GADNIC9:
                df.loc[k,'INDEX']=9
                nc9+=1
            else:
                df.loc[k,'INDEX']=99   
    elif  df.loc[k,'MAXHAD']==df.loc[k,'HAD10']:
            df.loc[k,'EI']=10
            if df.loc[k, 'GAD10']> GADNIC10:
                df.loc[k,'INDEX']=10
                nc10+=1
            else:
                df.loc[k,'INDEX']=99  
    elif  df.loc[k,'MAXHAD']==df.loc[k,'HAD11']:
            df.loc[k,'EI']=11
            if df.loc[k, 'GAD11']> GADNIC11:
                df.loc[k,'INDEX']=11
                nc11+=1
            else:
                df.loc[k,'INDEX']=99   
    elif  df.loc[k,'MAXHAD']==df.loc[k,'HAD11']:
            df.loc[k,'EI']=11
            if df.loc[k, 'GAD11']> GADNIC11:
                df.loc[k,'INDEX']=11
                nc11+=1
            else:
                df.loc[k,'INDEX']=99   
    elif  df.loc[k,'MAXHAD']==df.loc[k,'HAD12']:
            df.loc[k,'EI']=12
            if df.loc[k, 'GAD12']> GADNIC12:
                df.loc[k,'INDEX']=12
                nc12+=1
            else:
                df.loc[k,'INDEX']=99   
    elif  df.loc[k,'MAXHAD']==df.loc[k,'HAD13']:
            df.loc[k,'EI']=13
            if df.loc[k, 'GAD13']> GADNIC13:
                df.loc[k,'INDEX']=13
                nc13+=1
            else:
                df.loc[k,'INDEX']=99   
    elif  df.loc[k,'MAXHAD']==df.loc[k,'HAD14']:
            df.loc[k,'EI']=14
            if df.loc[k, 'GAD14']> GADNIC14:
                df.loc[k,'INDEX']=14
                nc14+=1
            else:
                df.loc[k,'INDEX']=99   
    elif  df.loc[k,'MAXHAD']==df.loc[k,'HAD15']:
            df.loc[k,'EI']=15
            if df.loc[k, 'GAD15']> GADNIC15:
                df.loc[k,'INDEX']=15
                nc15+=1
            else:
                df.loc[k,'INDEX']=99   
    
df['n1']=nc1; df['n2']=nc2; df['n3']=nc3; df['n4']=nc4; df['n5']=nc5; df['n6']=nc6; df['n7']=nc7; df['n8']=nc8; 
df['n9']=nc9; df['n10']=nc10; df['n11']=nc11; df['n12']=nc12; df['n13']=nc13; df['n14']=nc14; df['n15']=nc15; 

df['prom11']=prom[0][0]#CREACION DE LAS COLUMNAS QUE CONTIENE EL PROMEDIO 
df['prom12']=prom[0][1]#Clase 1 descriptor 2
df['prom21']=prom[1][0]
df['prom22']=prom[1][1]
df['prom31']=prom[2][0]
df['prom32']=prom[2][1]
df['prom41']=prom[3][0]
df['prom42']=prom[3][1]
df['prom51']=prom[4][0]
df['prom52']=prom[4][1]
df['prom61']=prom[5][0]
df['prom62']=prom[5][1]
df['prom71']=prom[6][0]
df['prom72']=prom[6][1]
df['prom81']=prom[7][0]
df['prom82']=prom[7][1]
df['prom91']=prom[8][0]
df['prom92']=prom[8][1]
df['prom101']=prom[9][0]
df['prom102']=prom[9][1]
df['prom111']=prom[10][0]
df['prom112']=prom[10][1]
df['prom121']=prom[11][0]
df['prom122']=prom[11][1]
df['prom131']=prom[12][0]
df['prom132']=prom[12][1]
df['prom141']=prom[13][0]
df['prom142']=prom[13][1]
df['prom151']=prom[14][0]
df['prom152']=prom[14][1]
df = df.dropna(axis=0, subset=['INDEX'])#elimino las filas donde el index es un valor vacio, no se puede clasificar
#CREACIÓN DE LOS DATASETS PARA CREAR LAS CLASES CONFORMADAS

df1=pandas.DataFrame(df[[1, 2,  'INDEX' , 'prom11', 'prom12', 'HAD1', 'n1', 'vecino', 'clases']]   [df['INDEX']==1])
df2=pandas.DataFrame(df[[1, 2,  'INDEX' , 'prom21', 'prom22', 'HAD2', 'n2', 'vecino' , 'clases']]   [df['INDEX']==2])
df3=pandas.DataFrame(df[[1, 2,  'INDEX' , 'prom31', 'prom32', 'HAD3', 'n3', 'vecino', 'clases']]   [df['INDEX']==3])
df4=pandas.DataFrame(df[[1, 2,  'INDEX' , 'prom41', 'prom42', 'HAD4', 'n4', 'vecino' , 'clases']]   [df['INDEX']==4])
df5=pandas.DataFrame(df[[1, 2,  'INDEX' , 'prom51', 'prom52', 'HAD5', 'n5', 'vecino', 'clases']]   [df['INDEX']==5])
df6=pandas.DataFrame(df[[1, 2,  'INDEX' , 'prom61', 'prom62', 'HAD6', 'n6', 'vecino', 'clases']]   [df['INDEX']==6])
df7=pandas.DataFrame(df[[1, 2,  'INDEX' , 'prom71', 'prom72', 'HAD7', 'n7', 'vecino', 'clases']]   [df['INDEX']==7])
df8=pandas.DataFrame(df[[1, 2,  'INDEX' , 'prom81', 'prom82', 'HAD8', 'n8', 'vecino', 'clases']]   [df['INDEX']==8])
df9=pandas.DataFrame(df[[1, 2,  'INDEX' , 'prom91', 'prom92', 'HAD9', 'n9', 'vecino', 'clases']]   [df['INDEX']==9])
df10=pandas.DataFrame(df[[1, 2,  'INDEX' , 'prom101', 'prom102', 'HAD10', 'n10', 'vecino', 'clases']]   [df['INDEX']==10])
df11=pandas.DataFrame(df[[1, 2,  'INDEX' , 'prom111', 'prom112', 'HAD11', 'n11', 'vecino', 'clases']]   [df['INDEX']==11])
df12=pandas.DataFrame(df[[1, 2,  'INDEX' , 'prom121', 'prom122', 'HAD12', 'n12', 'vecino', 'clases']]   [df['INDEX']==12])
df13=pandas.DataFrame(df[[1, 2,  'INDEX' , 'prom131', 'prom132', 'HAD13', 'n13', 'vecino', 'clases']]   [df['INDEX']==13])
df14=pandas.DataFrame(df[[1, 2,  'INDEX' , 'prom141', 'prom142', 'HAD14', 'n14', 'vecino', 'clases']]   [df['INDEX']==14])
df15=pandas.DataFrame(df[[1, 2,  'INDEX' , 'prom151', 'prom152', 'HAD15', 'n15', 'vecino', 'clases']]   [df['INDEX']==15])
dft=pandas.DataFrame(df[[1, 2, 'INDEX' , 'vecino' , 'clases' ]])
print('esto es para probra columas', df4.iloc[0, 0])
print('esto es para probra columas', df4.iloc[0, 1])

print('esto es para probra columas', df4.iloc[0, 2])
print('esto es para probra columas', df4.iloc[0, 3])
print('esto es para probra columas', df4.iloc[0, 4])
print('esto es para probra columas', df4.iloc[0, 5])
print('esto es para probra columas', df4.iloc[0, 6])
print('esto es para probra columas', df4.iloc[0, 7])
print('imprime registro' , df4.loc[0 : 0])
print(df3)
df3.info()
print(df4)
df4.info()

#FUNCIÓN PARA FUSIÓN DE GRUPOS CLASES FUSIÓN ENTRE X y Y
def fusion(x,y, z, w): # recibe 1 data frame por clase y arroja el dataframe completo (w)
    individuos=0
    for i in range(0, len(x)):
        for j in range(0, len(y)):
            if (abs(x.iloc[i, 5]-y.iloc[j,5])<0.01):
                individuos+=1
    densidad=individuos/(x.iloc[0,6]+y.iloc[0,6])
    dr1= len(x)/(len(y)+len(x)+len(z))
    dr2= len(y)/(len(y)+len(x)+len(z))
    proden=(dr1+dr2)/2
    if (densidad>= proden):
        print('Se ejecuto una fusion')
        for d in range(0,len(w)):
            if (w.iloc[d, 2]==2):
                w.iloc[d, 2 ]=1
    else:
        print('no hay fusion entre 1 y 2')
    return w #dataframe con la fusion 
        #FIN DE LA FUSION DE GRUPOS*************#
#MIGRACIÓN DE GRUPOS DEL GRUPO K HACIA EL GRUPO H 
#grupo k, grupo w, datacompleta w, número del grupo 1 num1, numero del grupo 2 num2
#la migrración es de miembros de k para h """

def migracion(k, h ):#recibe dos grupos y arroja el dataframe completo en caso de una migracion
    nv=0; nm=0; PGADV=0; PGADM=0
    for i in range (0, len(k)-2):
	    for j in range(i+1, len(k)-1):
		    if abs(k.iloc[i, 5]- k.iloc[j, 5])<3:
			    PGADV+=(abs(k.iloc[i,5]- k.iloc[j,5]))
			    nv+=1	
    for g in range (0, len(h)-2):#Ecuación 10
        for p in range(g+1, len(h)-1):
            if abs(h.iloc[g, 5]-h.iloc[p, 5]<3): 
                PGADM+=abs(h.iloc[g,5]-h.iloc[p, 5])
                nm+=1
    if ((PGADV<PGADM)): #and (nv>=nm)):
        for i in range (0, len(k)-2): #Ejecución de la migración miembros de k hacia h 
            for j in range(i+1, len(k)-1):
                if (abs(k.iloc[i, 5]- k.iloc[j, 5])<3): #esta sentencia originalmente estaba < 0.1
                    k.iloc[i, 2]=h.iloc[0, 2]
                    h=h.append(k.loc[i :i ], ignore_index=True) #Agrego el nuevo registro al dataframe h
                    #k=k.drop([i], axis=0)
                    k=k.drop(k.index[[i]], axis =0)
                    #h.iloc[0, 6]+=1
                    #h['n4']=h.iloc[0,6]
                    #k.iloc[0, 6]-=1   
                    #k['n3']= k.iloc[0,6]                
        print('se ejecuto una migracion')
    return k, h 
#****FIN DEL PROCESO DE MIGRACION ********************* 
df33, df34=migracion(df3,df4) 
#df22, df44=migracion(df2,df4)
#df55, df4_3=migracion(df5,df4)
#df88, df4_4=migracion(df8,df4)

print('nuevos datasets')
print(df33)
df33.info()
print(df34)
df34.info()

def compacta(a):#calcula la compactación de cada dataframe
    suma=0
    for j in range(1, 3):
        for i in range(0, len(a)):
            suma+=abs(a.iloc[i, j-1 ]-a.iloc[0, j+2]) #diferencia entre el valor y el promedio
    compactacion=suma/(len(a)*2)#número total de diferencias realizadas se multiplica por el número de descriptores
    return compactacion#fin del proceso de compactación 

yy2=compacta(df2)
print(yy2)
yy3=compacta(df3)
print(yy3)
yy4=compacta(df4)
print(yy4)
yy5=compacta(df5)
print(yy5)
yy6=compacta(df6)
print(yy6)
yy7=compacta(df7)
print(yy7)
yy8=compacta(df8)
print(yy8)
yy9=compacta(df9)
print(yy9)
yy10=compacta(df10)
print(yy10)
yy11=compacta(df11)
print(yy11)
yy12=compacta(df12)
print(yy12)
yy13=compacta(df13)
print(yy13)
yy14=compacta(df14)
print(yy14)
yy15=compacta(df15)
print(yy15)

desvi=[yy2,yy3,yy4,yy5,yy6,yy7,yy8,yy9,yy10,yy11,yy12,yy13,yy14,yy15, ]#vector que contiene la compactacion 
#ssprom= (yy1+yy2)/2 #compactacion promedio
promedios=numpy.average(desvi)
print('Promedio ' , promedios)
desviacion=(numpy.std(desvi)/math.sqrt(14))
print('Desviacion ' , desviacion)
#u dataset t compactacion p promedio z desviacion 
def particiona(u, t, p, z):
    delta= p+5*z
    if (t>delta):
        clustering= KMeans(n_clusters=2, max_iter=10000)
        clustering.fit(u)
        u['K-Means']=clustering.labels_
        print('Se ejecuto una partición')    	
    else:
        print('el dataset : ' , u , 'no particiona')
    return u    
particiona(df4, yy4, promedios, desviacion)
print(df4)

ruta1='C:\Python\Python37-32\doctorado\LAMDAHADIris_HAD4.csv'#EXPORTACIÓN DE LOS RESULTADOS
ruta5='C:\Python\Python37-32\doctorado\LAMDAHADR15Todos.csv'#EXPORTACIÓN DE LOS RESULTADOS

dft.to_csv(ruta5)
df4.to_csv(ruta1)

accuracy1= accuracy_score(dft['clases'], dft['INDEX'])
f1_1= f1_score(dft['clases'], dft['INDEX'], average= 'micro')
recall1=recall_score(dft['clases'], dft['INDEX'], average= 'weighted')
precision= precision_score(dft['clases'], dft['INDEX'], average='macro')
print('la accuracy es : ' , accuracy1)
print('El valor de F1 Score es : ' , f1_1)
print('REcall es : ' , recall1)
print('La precisión es : ' , precision)

#CREACION DEL GRÁFICO EN R15 
v1= dft[1].values
v2= dft[2].values
X=np.array(list(zip(v1,v2))) #transformo los datos en una matriz
df1=dft.drop(['INDEX'], axis=1)#elimino la columna de la etiqueta
df1=dft.drop(['vecino'], axis=1)#elimino la columna de la etiqueta
pca=PCA(n_components=2)#Modelo de componentes principales
pca_iris=pca.fit_transform(dft) #Contiene las coordena
pca_iris_df=pd.DataFrame(data=pca_iris, columns= ['Componente_1', 'Componente_2']) #creación del data frame con las coordenadas de los dos componentes 
pca_nombres_iris=pd.concat([pca_iris_df, dft[['INDEX']]], axis=1) #concatenar las etiquetas con los individuos

#CREACIÓN DEL GRÁFICO

fig=plt.figure(figsize=(10,10))
ax=fig.add_subplot(1,1,1)
ax.set_xlabel('Componente 1', fontsize=15)
ax.set_ylabel('Componente 2', fontsize=15)
ax.set_title('R15 Original', fontsize=20)
color_theme= np.array(['red', 'blue', 'red', 'black', 'pink', 'orange', 'brown', 'purple', 'yellow', 'red', 'green', 'gray', 'yellow', 'black', 'purple', 'red' ])    
ax.scatter(x=pca_nombres_iris.Componente_1 , y= pca_nombres_iris.Componente_2, c=color_theme[pca_nombres_iris.INDEX], s= 20, marker='.')
plt.show()
print(dft)
print(df2)