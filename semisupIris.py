#Código del algoritmo híbrido semisupervisado basado en LAMDA.
"""ESte código aunque es semisupervisado eeta diseñado para que funcione 
en un principio con la data Iris es decir exclusivo para clasificacion 
y con la data aggregation en el caso de clustering"""
import os, numpy, pandas, time #librerias necesarias
from colorama import *
os.system('cls') #limpiar pantalla
datos=pandas.read_csv('IrisHybrid2.csv', delimiter=';') #importación del dataset Iris 4 descriptores y 150 datos
df=pandas.DataFrame(datos) #creación del dataframe de datos
a=0
for i in range(0, len(df)):#Recorre toda la matriz de datos
    if df.loc[i, 'y1']==0: #AGRUPAMIENTO
        k=1 #inidca el cluster
        n1=0 #indica el número de observaciones del cluster 1
        for i in range(0,len(df)): #estandarización de los datos se crean tantas columnas como descriptores
            df.loc[i, 11]=((df.loc[i, '0']-df['0'].min())/(df['0'].max()-df['0'].min()))
            df.loc[i, 22]=((df.loc[i,'1']-df['1'].min())/(df['1'].max()-df['1'].min()))
        df['PromNIC']=0.5 #promedio de la clase no informativa
        df['n'+repr(k)]=0 #número de elementos del cluster k 
        ####### **************************EVALUaCION DE LA PRIMERA OBSERVACION *******************************################################################
        df.loc[0, 'promi'+repr(k)+ '1']= df.loc[0, 11] #AL SER LA PRIMERA OBSERVACION EL PROMEDIO INICIAL ES IGUAL A LA OBSERVACION S
        df.loc[0, 'promi'+repr(k)+ '2']= df.loc[0, 22]
        df.loc[0, 'promf'+repr(k) + '1' ]= df.loc[0, 11]  #promedio final de cluster 1 descriptor 1 
        df.loc[0, 'promf'+repr(k) +'2' ]= df.loc[0, 22] #promedio final de cluster 1 descriptor 2
        alfa=0.9
        df['tnic'] =1 /   (1+ (1- 0.5)/0.5 +( (1- 0.5) / (0.5)   ))
        df['snic']= 1 - (1/(1+                  ( 0.5/(1-0.5)+                                         0.5/(1-0.5))))
        df['GADNIC']=alfa*df['tnic']+(1-alfa)*df['snic']
        df.loc[0,'Cluster']=1
        df['n'+repr(k)]=1 #NUMERO DE ELEMENTO DEL CLUSTER
        df['POS'+repr(k)]=int(0) #POSICION DE LA ÚLTIMA OBSERVACION DEL CLUSTER PARA LUEGO SABER DONDE ESTA EL PROMEDIO DEL CLUSTER
        df['dnb']=0.06
        for l in range(1,len(df)):
            for h in range(1, k+1):
                df.loc[l, 'promi'+repr(h)+ '1']= df.loc[df.loc[0,'POS'+repr(k)], 'promi'+repr(h)+'1']
                df.loc[l, 'promi'+repr(h)+ '2']= df.loc[df.loc[0,'POS'+repr(k)], 'promi'+repr(h)+'2']
                df.loc[l, 'promf'+repr(h) + '1' ]= df.loc[df.loc[0,'POS'+repr(k)], 'promi' +repr(h) +'1'] + (df.loc[l, 11] - df.loc[df.loc[0,'POS'+repr(k)], 'promi'+repr(h) +'1'])/(df.loc[df.loc[0,'POS'+repr(k)], 'n'+repr(h)]+1) #promedio final de cluster 1 descriptor 1 
                df.loc[l, 'promf'+repr(h) +'2' ]= df.loc[df.loc[0,'POS'+repr(k)], 'promi'+repr(h) +'2'] + (df.loc[l, 22] - df.loc[df.loc[0,'POS'+repr(k)], 'promi' +repr(h) +'2'])/(df.loc[df.loc[0,'POS'+repr(k)], 'n'+repr(h)]+1)       #promedio final de cluster 1 descriptor 2
                df.loc[l, 'CMAD'+repr(h) + '1' ]=   1/ (1+abs((df.loc[l, 11]-df.loc[l, 'promf'+repr(h) + '1'])))
                df.loc[l, 'CMAD'+repr(h) + '2' ]=  1/(1+abs((df.loc[l, 22]-df.loc[l, 'promf'+repr(h) + '2'])))
                df.loc[l, 'dist'+repr(h)]=(abs(df.loc[l, 11]-df.loc[l, 'promf'+repr(h) + '1'])+ abs(df.loc[l, 22]-df.loc[l, 'promf'+repr(h) + '2']))/2#CALCULO DE LA DISTANCIA DEL INDIVIDUO X AL CENTROIDE DE CADA CLUSTER ECUACION (13)
                df.loc[l, 'dist1'+repr(h)]=abs(df.loc[l, 'dist'+repr(h)]-df.loc[l, 'dnb'])
                df.loc[l, 'k'+repr(h)]= df.loc[l, 'dnb'] / ( df.loc[l,'dnb']+ df.loc[l, 'dist1'+repr(h)])
                if df.loc[l,'dist'+repr(h)]<df.loc[l,'dnb']:
                    df.loc[l, 'k'+repr(h)]=1
                df.loc[l, 'RMAD'+repr(h)+ '1']= df.loc[l, 'k'+repr(h)]* df.loc[l,'CMAD'+repr(h)+'1']
                df.loc[l, 'RMAD'+repr(h)+ '2']= df.loc[l, 'k'+repr(h)]* df.loc[l, 'CMAD'+repr(h)+'2']
                df.loc[l, 'MaxRMADs'+repr(h)]=max([ df.loc[l, 'RMAD'+repr(h) + '1' ],  df.loc[l, 'RMAD'+repr(h) + '2' ]])
                df.loc[l, 'MinRMADs'+repr(h)]=min([ df.loc[l, 'RMAD'+repr(h) + '1' ],  df.loc[l, 'RMAD'+repr(h) + '2' ]])
                df.loc[l,'t'+repr(h)]=  1/ (1+ ((1- df.loc[l,'MaxRMADs'+repr(h)])/ df.loc[l,'MaxRMADs'+repr(h)]) + ((1- df.loc[l,'MinRMADs'+repr(h)])/ df.loc[l,'MinRMADs'+repr(h)]))
                df.loc[l, 's'+repr(h)]= 1- (  1/ (1+ ( df.loc[l, 'MaxRMADs'+repr(h)]/ (1- df.loc[l, 'MaxRMADs'+repr(h)]) +    df.loc[l,'MinRMADs'+repr(h)]/ (1- df.loc[l,'MinRMADs'+repr(h)]))))
                df.loc[l,'GAD'+repr(h)]= alfa*df.loc[l, 't'+repr(h)]+(1-alfa)*df.loc[l,'s'+repr(h)]
            mayor= df.loc[l,'GAD1']
            poscluster=1
            for i in range(1, k+1):
                if mayor<df.loc[l, 'GAD'+repr(i)]:
                        mayor=df.loc[l, 'GAD'+repr(i)]
                        poscluster=i #NÚMERO DEL CLUSTER EN EVALUACION 
            index=max([mayor,df.loc[0,'GADNIC']])
            if index==mayor:
                df.loc[l,'Cluster']=poscluster
                df['POS'+repr(poscluster)]=int(l)
                df['n'+repr(poscluster)]= df.loc[0,'n'+repr(poscluster)]+1
                df.loc[l,'promi' + repr(poscluster) +'1']=df.loc[l, 'promf'+repr(poscluster) + '1']
                df.loc[l,'promi' + repr(poscluster) +'2']=df.loc[l, 'promf'+repr(poscluster) + '2']
            else:
                k+=1
                df.loc[l,'Cluster']=k
                df['POS'+repr(k)]=int(l)
                df['n'+repr(k)]=1
                df.loc[l,'promi' + repr(k) +'1']=df.loc[l, 11]
                df.loc[l,'promi' + repr(k) +'2']=df.loc[l, 22]
        #CALCULO DEL GAD DE CADA CLUSTER PARA VERIFICAR CUAL ES EL VECINO MÁS CERCANO CREO UNA COLUMNA LLAMADA GADA
        VGAD=numpy.zeros(k) #VECTOR QUE CONTIENE LOS GAD DE CADA CLUSTER
        clus=numpy.zeros(k) #vECTOR QUE CONTIENE EL IDENTIFICADOR DEL CLUSTERS
        for i in range(1, k+1):
            df['GADA'+repr(i)]=df.loc[int(df.loc[0,'POS'+repr(i)]), 'GAD'+repr(i)]
            VGAD[i-1]=df.loc[0,'GADA'+repr(i)]
            print(Cursor.FORWARD(10)+Cursor.DOWN(1),  VGAD[i-1])
            clus[i-1]=i
            print(Cursor.FORWARD(30)+Cursor.UP(1),  clus[i-1])
        #ORDENAMIENTO DEL VECTOR PARA VERIFICAR EL VECINO MAS CERCANO DE ACUERDO AL VALOR DEL GAD
        for i in range(0, k-1):
            for j in range(i+1, k):
                if VGAD[i]>VGAD[j]:
                    aux=VGAD[i]
                    aux2=clus[i]
                    VGAD[i]=VGAD[j]
                    clus[i]=clus[j]
                    VGAD[j]=aux
                    clus[j]=aux2
        for i in range(0, k):
            print(Cursor.FORWARD(10)+Cursor.DOWN(1),  VGAD[i-1])
            print(Cursor.FORWARD(30)+Cursor.UP(1),  clus[i-1])
        print('Cluster vecinos')        
        for i in range(0,k):#IMPRESION DE LOS CLUSTERS VECINOS
            print(clus[i], ' ')
        #CALCULO DE LA COMPACTACION DE CADA CLUSTER POR LA SUMA DE CUADRADOS DEL CLUSTER RESPECTO AL PROMEDIO DEL CLUSTER
        j=0
        c=numpy.zeros(k) #Vector que contiene la compactación de cada cluster la compactación es de acuerdo a cada descriptor
        for h in range(1,k+1):
            j+=1
            suma1=0; suma2=0
            for i in range(0, len(df)):
                if j== df.loc[i,'Cluster']:
                    suma1+= abs(df.loc[i, 11]-df.loc[int(df.loc[0,'POS'+repr(h)]), 'promf'+repr(h)+ '1' ])
                    #print(df.loc[i, 11], '   ',  df.loc[int(df.loc[0,'POS'+repr(h)]), 'promf'+repr(h)+ '1' ])
                    #print('diferencia: ', suma1)
                    suma2+= abs(df.loc[i, 22]-df.loc[int(df.loc[0,'POS'+repr(h)]), 'promf'+repr(h)+ '2' ])
                    #print(df.loc[i, 22], '   ',  df.loc[int(df.loc[0,'POS'+repr(h)]), 'promf'+repr(h)+ '2' ])
                    #print('diferencia2 : ' , suma2)
        #print('suma 1 : ' , suma1 , 'Suma 2  ' , suma2 , ' observaciones : ' , df.loc[0, 'n'+repr(h)])
            tot=(suma1+suma2)/df.loc[0, 'n'+repr(h)]
            c[j-1]=tot
        print('Compactacion es de : ' , c)
        for i in range(1 , k+1):
            print('Compact de ' , i ,  ' : ' , c[i-1])
        #EVALUACION DE LOS CLUSTER A FUSIONAR PRIMER PAR A FUSIONAR ES CLUSTER 1 Y CLUSTER 7 SEGÚN DEFINICION 8 
        #VERIFICAR CUANTOS ELEMENTOS SON MENORES A DISTANCIA, POR CADA CLUSTER
        #CREACION DE TANTOS DATA FRAMES COMO CLUSTER SE HAYAN FORMADO
        #os.system('cls')
        #CADA UNO DE LOS DATAFRAMES CONTIENE LOS CLUSTERS QUE SE HAN FORMADO
        #os.system('cls')

        df1=pandas.DataFrame(df[['promf11','promf12', 'n1', 'Cluster', 11, 22, 'POS1' ]]     [df['Cluster']==1])
        df2=df[['promf11','promf12', 'n2', 'Cluster', 11, 22, 'POS2']]     [df['Cluster']==2]
        df3=df[['promf11','promf12', 'n3', 'Cluster', 11, 22, 'POS3']]     [df['Cluster']==3]
        df4=df[['promf11','promf12', 'n4', 'Cluster', 11, 22, 'POS4']]     [df['Cluster']==4]
        df5=df[['promf11','promf12', 'n5', 'Cluster', 11, 22, 'POS5']]     [df['Cluster']==5]
        df6=df[['promf11','promf12', 'n6', 'Cluster', 11, 22, 'POS6']]     [df['Cluster']==6]
        df7=df[['promf11','promf12', 'n7', 'Cluster', 11, 22, 'POS7']]     [df['Cluster']==7]
        df8=df[['promf11','promf12', 'n8', 'Cluster', 11, 22, 'POS8']]     [df['Cluster']==8]
        df9=df[['promf11','promf12', 'n9', 'Cluster', 11, 22, 'POS9']]     [df['Cluster']==9]
        df10=df[['promf11','promf12', 'n10', 'Cluster', 11, 22, 'POS10']]     [df['Cluster']==10]
        df11=df[['promf11','promf12', 'n11', 'Cluster', 11, 22, 'POS11']]     [df['Cluster']==11]
        df12=df[['promf11','promf12', 'n12', 'Cluster', 11, 22, 'POS12']]     [df['Cluster']==12]
        df13=df[['promf11','promf12', 'n13', 'Cluster', 11, 22, 'POS13']]     [df['Cluster']==13]
        df14=df[['promf11','promf12', 'n14', 'Cluster', 11, 22, 'POS14']]     [df['Cluster']==14]
        df15=df[['promf11','promf12', 'n15', 'Cluster', 11, 22, 'POS15']]     [df['Cluster']==15]
        print(df1 , df7)
        ruta='C:\Python\Python37-32\doctorado\salidaRD.csv'
        df.to_csv(ruta)
    else: #Agrupamiento según LAMDA HAD (CLASIFICACIÓN)
        clases=3#CLASES DEL DATASET en el caso de iris
        descriptores=4#DESCRIPTORES DEL DATASET
        prom=numpy.zeros((clases,descriptores)) #MATRIZ DE PROMEDIOS
        mgad=numpy.zeros((clases,clases))
        alfa=0.9
        os.system('cls')
        for i in range(0,len(df)):#ESTANDARIZACIÓN DE DATOS 
            df.loc[i, 1]=((df.loc[i,'largosepa']-df['largosepa'].min())/(df['largosepa'].max()-df['largosepa'].min()))
            df.loc[i, 2]=((df.loc[i,'anchosepa']-df['anchosepa'].min())/(df['anchosepa'].max()-df['anchosepa'].min()))
            df.loc[i, 3]=((df.loc[i,'largopeta']-df['largopeta'].min())/(df['largopeta'].max()-df['largopeta'].min()))
            df.loc[i, 4]=((df.loc[i,'anchopeta']-df['anchopeta'].min())/(df['anchopeta'].max()-df['anchopeta'].min()))
        for k in range(1,5): #MCREACIÓN DE LA matriz de promedios
            for h in range(1, 4):
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
            for j in range(1,4):
                for k in range(1,5):
                    df.loc[i, 'MAD'+repr(j)+repr(k)]=prom[j-1][k-1]**df.loc[i,k]*(1-prom[j-1][k-1] )**(1-df.loc[i,k])
        for i in range(0,len(df)):#OPERADORES DEL T Y S COMBINACIONES LINEALES DE LOS MADS EN ESTE CASO MAXIMO Y MÍNIMO
            for j in range(1,4):
                df.loc[i,'MaxC'+repr(j)]= max(df.loc[i, 'MAD'+repr(j)+'1'], df.loc[i, 'MAD'+repr(j)+'2'], df.loc[i, 'MAD'+repr(j)+'3'], df.loc[i, 'MAD'+repr(j)+'4'])
                df.loc[i,'MinC'+repr(j)]= min(df.loc[i, 'MAD'+repr(j)+'1'], df.loc[i, 'MAD'+repr(j)+'2'], df.loc[i, 'MAD'+repr(j)+'3'], df.loc[i, 'MAD'+repr(j)+'4'])
                df.loc[i,'GAD'+repr(j)]= alfa*df.loc[i,'MinC'+repr(j)]+(1-alfa)*df.loc[i,'MaxC'+repr(j)]
        for k in range(1,4):#CALCULO DE LOS PROMEDIOS MGAD
            j=1
            c=0
            while j<=3:
                suma=0
                obs1=0
                for i in range(0,len(df)):
                    if j==df.loc[i,'clases']:
                        suma+=df.loc[i,'GAD'+repr(k)]
                        obs1+=1
                promed=suma/obs1
                mgad[k-1][c]=promed
                j+=1
                c+=1
        GADNIC1=numpy.average(mgad[:, 0])
        GADNIC2=numpy.average(mgad[:, 1])
        GADNIC3=numpy.average(mgad[:, 2])
        for i in range(0,3):#CALCULO DEL ADGAD EN FUNCIÓN A LOS PROMEDIOS MGAD
            for j in range(0,3):
                for k in range(0,len(df)):
                    df.loc[k, 'ADGAD'+repr(i+1)+repr(j+1)]= mgad[i][j]**df.loc[k,'GAD'+repr(j+1)]*(1-mgad[i][j] )**(1-df.loc[k,'GAD'+repr(j+1)] )
        for j in range(1,4):#CALCULO DEL HAD EN FUNCIÓN A LOS PROMEDIOS ADGAD
            for k in range(0,len(df)):
                df.loc[k, 'HAD'+repr(j)]= df.loc[k,'ADGAD'+repr(j)+'1']+ df.loc[k,'ADGAD'+repr(j)+'2']+df.loc[k,'ADGAD'+repr(j)+'3']
        vector=numpy.zeros(clases) #este vector se crea para obtener el segundo mayor HAD
        #*******CALCULOS NECESARIOS PARA EL VECINO MÁS CERCANO*******#
        for k in range(0,len(df)):#SE DEBE CALCULAR EL SEGUNDO MÁXIMO VALOR DE HAD QUE PERMITA SABER CUAL ES EL VECINO MÁS CERCACNO      
            df.loc[k, 'MAXHAD']= max(df.loc[k,'HAD1'], df.loc[k,'HAD2'], df.loc[k,'HAD3']) #Aqui es donde se guardan el MAXIMO GRADO DE ADECUACIÓN GLOBAL
            for z in range(1,4):
                vector[z-1]=df.loc[k, 'HAD'+ repr(z)]#ALMACENAMIENTO DE LOS VALORES EN EL VECTOR
            v=numpy.sort(vector)
            df.loc[k, 'vectorSEgundo']=v[1]
            df.loc[k, 'Ivecino']=v[1]#contiene el HAD DEL VECINO MÁS CERCANO
            for g in range(1,4):
                if df.loc[k,'Ivecino']== df.loc[k, 'HAD'+repr(g)]:
                    df.loc[k, 'vecino']=g
        #***AQUI TERMINA EL CALCULO DEL VECINO MÁS CERCANO*****# 
        nc1=0; nc2=0; nc3=0 #ESTOS VALORES CONTIENEN EL NÚMERO DE OBSERVACIONES POR CLASES
        
        for k in range(0,len(df)):
            if df.loc[k,'MAXHAD']==df.loc[k,'HAD1']:
                df.loc[k,'EI']=1
                if df.loc[k, 'GAD1']> GADNIC1:
                    df.loc[k,'INDEX']=1
                    nc1+=1#DETERMINA EL NÚMERO DE OBSERVACIONES EN LA CLASE 1
                else:
                    df.loc[k,'INDEX']='NIC'    
            elif df.loc[k,'MAXHAD']==df.loc[k,'HAD2']:
                    df.loc[k,'EI']=2
                    if df.loc[k, 'GAD2']> GADNIC2:
                        df.loc[k,'INDEX']=2
                        nc2+=1
                
                    else:
                        df.loc[k,'INDEX']='NIC'            
            else:
                    df.loc[k,'EI']=3
                    if df.loc[k, 'GAD3']> GADNIC3:
                        df.loc[k,'INDEX']=3
                        nc3+=1
                
                    else:
                        df.loc[k,'INDEX']='NIC'
        df['n1']=nc1; df['n2']=nc2; df['n3']=nc3
        df['prom11']=prom[0][0]#CREACION DE LAS COLUMNAS QUE CONTIENE EL PROMEDIO 
        df['prom12']=prom[0][1]
        df['prom13']=prom[0][2]
        df['prom14']=prom[0][3]
        df['prom21']=prom[1][0]
        df['prom22']=prom[1][1]
        df['prom23']=prom[1][2]
        df['prom24']=prom[1][3]
        df['prom31']=prom[2][0]
        df['prom32']=prom[2][1]
        df['prom33']=prom[2][2]
        df['prom34']=prom[2][3]
        #CREACIÓN DE LOS DATASETS PARA CREAR LAS CLASES CONFORMADAS
        df1=pandas.DataFrame(df[[1, 2, 3, 4, 'INDEX' , 'prom11', 'prom12', 'prom13', 'prom14', 'HAD1', 'n1', 'vecino']]   [df['INDEX']==1])
        df2=pandas.DataFrame(df[[1, 2, 3, 4, 'INDEX' , 'prom21', 'prom22', 'prom23', 'prom24', 'HAD2', 'n2' , 'vecino']]   [df['INDEX']==2])
        df3=pandas.DataFrame(df[[1, 2, 3, 4, 'INDEX' , 'prom31', 'prom32', 'prom33', 'prom34', 'n3', 'HAD3', 'vecino' ]]   [df['INDEX']==3])
        #df1_1= [df1, prom[0][0], prom[0][1], prom[0][2], prom[0][3], nc1]
        #data_s=pandas.DataFrame(df1_1)

        #FUNCIÓN PARA FUSIÓN DE GRUPOS CLASES
        def fusion(x,y):
            for i in range(0, len(x)):
                for j in range(0, len(y)):
                    if (abs(x.iloc[i, 9]-y.iloc[j,9])<0.03):
                        individuos+=1
            densidad=individuos/(x.iloc[0,10]+y.iloc[0,10])
            dr1=

        
        
        individuos=0
        xx=df2.iloc[ 0, 9 ]
        print(xx)
        #determinacion del area de solapamiento y cardinalidad
        for i in range(0, int(df1.iloc[0,10])):
            for t in range(0, int(df2.iloc[0, 10])):
                if (abs(df1.iloc[i,9]- df2.iloc[t,9])<0.03):
                    individuos+=1
        densidad= individuos/(df1.iloc[0,10]+df2.iloc[0,10])
        dr1=df.loc[0,'n1']/(df.loc[0,'n1']+df.loc[0,'n2']+df.loc[0,'n3'])
        dr2=df.loc[0,'n2']/(df.loc[0,'n1']+df.loc[0,'n2']+df.loc[0,'n3'])
        proden=(dr1+dr2)/2 #Ecuacion 7 
        if (densidad>= proden):
            print('Se ejecuto una fusion')
            for d in range(0,len(df)):
                if df.loc[d,'INDEX']==2:
                    df.loc[d,'INDEX']=1
        else:
            print('no hay fusion entre 1 y 2')
        #FIN DE LA F    USION DE GRUPOS*************#
        #MIGRACION DE INDIVIDUOS DE UN GRUPO A OTRO****************"""
        ruta1='C:\Python\Python37-32\doctorado\LAMDASemisupIrisHAD_1.csv'#EXPORTACIÓN DE LOS RESULTADOS
        ruta2='C:\Python\Python37-32\doctorado\LAMDASemisupIrisHAD_2.csv'#EXPORTACIÓN DE LOS RESULTADOS
        ruta3='C:\Python\Python37-32\doctorado\LAMDASemisupIrisHAD_3.csv'#EXPORTACIÓN DE LOS RESULTADOS
        df1.to_csv(ruta1)
        df2.to_csv(ruta2)
        df3.to_csv(ruta3)
        print(df1)
        print(df2)
        print(df3)
        print(df)





