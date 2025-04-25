import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # Para leer archivos
import geopandas as gpd # Para hacer cosas geográficas
import seaborn as sns # Para hacer plots lindos
import networkx as nx # Construcción de la red en NetworkX
import scipy as sp

def construye_adyacencia(D,m): 
    # Función que construye la matriz de adyacencia del grafo de museos
    # D matriz de distancias, m cantidad de links por nodo
    # Retorna la matriz de adyacencia como un numpy.
    D = D.copy()
    l = [] # Lista para guardar las filas
    for fila in D: # recorriendo las filas, anexamos vectores lógicos
        l.append(fila<=fila[np.argsort(fila)[m]] ) # En realidad, elegimos todos los nodos que estén a una distancia menor o igual a la del m-esimo más cercano
    A = np.asarray(l).astype(int) # Convertimos a entero
    np.fill_diagonal(A,0) # Borramos diagonal para eliminar autolinks
    return(A)

def calcularLU(A):
    # matriz es una matriz de NxN
    # Retorna la factorización LU a través de una lista con dos matrices L y U de NxN.
    # Completar! Have fun
    m=A.shape[0]
    n=A.shape[1]
    Ac = A.copy()

    if m!=n:
        print('Matriz no cuadrada')
        return

    ## desde aqui -- CODIGO A COMPLETAR
    for i in range(m-1):
        pivote = Ac[i,i]
        for j in range(i+1,m):
            factor = Ac[j,i]/pivote
            for k in range(i,m):
                Ac[j,k] = Ac[j,k] - Ac[i,k] * factor
            Ac[j,i] = factor
    ## hasta aqui

    L = np.tril(Ac,-1) + np.eye(A.shape[0])
    U = np.triu(Ac)

    return L, U

def inversaPorLU(B):
  n = B.shape[0]
  L, U = calcularLU(B)
  B_inv = np.zeros(shape=(n,n))
  I = np.eye(n)
  for i in range(n):
    e_i = I[:,i]
    y = sp.linalg.solve_triangular(L,e_i,lower=True)
    B_inv[:,i] = sp.linalg.solve_triangular(U,y)
  return B_inv

def calcula_matriz_C(A):
    # Función para calcular la matriz de trancisiones C
    # A: Matriz de adyacencia
    # Retorna la matriz C
    Kinv = np.eye(A.shape[0])*(1/np.sum(A[0,:])) # Calcula inversa de la matriz K, que tiene en su diagonal la suma por filas de A
    C = A.transpose()@Kinv # Calcula C multiplicando Kinv y A traspuesta
    return C

def calcula_pagerank(A,alfa):
    # Función para calcular PageRank usando LU
    # A: Matriz de adyacencia
    # d: coeficientes de damping
    # Retorna: Un vector p con los coeficientes de page rank de cada museo
    C = calcula_matriz_C(A)
    N = A.shape[0] # Obtenemos el número de museos N a partir de la estructura de la matriz A
    M = N/alfa * (np.eye(N)-(1-alfa)*C)
    L, U = calcularLU(M) # Calculamos descomposición LU a partir de C y d
    b = np.ones(N) # Vector de 1s, multiplicado por el coeficiente correspondiente usando d y N.
    Up = sp.linalg.solve_triangular(L,b,lower=True) # Primera inversión usando L
    p = sp.linalg.solve_triangular(U,Up) # Segunda inversión usando U
    return p

def calcula_matriz_C_continua(D):
    # Función para calcular la matriz de trancisiones C
    # D: matriz de distancia
    # Retorna la matriz C en versión continua
    Dc = D.copy()
    np.fill_diagonal(Dc,1)
    F = 1/Dc
    np.fill_diagonal(F,0)
    sumas_filas = np.sum(F, axis=1)  # Vector con la suma de cada fila
    K = np.diag(sumas_filas)         # Matriz diagonal con las sumas

    Kinv = np.diag(1 / sumas_filas)  # Inversa de cada elemento diagonal
    # Calcula inversa de la matriz K, que tiene en su diagonal la suma por filas de F
    C = F.transpose()@Kinv # Calcula C multiplicando Kinv y F
    return C

def calcula_B(C,cantidad_de_visitas):
    # Recibe la matriz T de transiciones, y calcula la matriz B que representa la relación entre el total de visitas y el número inicial de visitantes
    # suponiendo que cada visitante realizó cantidad_de_visitas pasos
    # C: Matriz de transiciones
    # cantidad_de_visitas: Cantidad de pasos en la red dado por los visitantes. Indicado como r en el enunciado
    # Retorna:Una matriz B que vincula la cantidad de visitas w con la cantidad de primeras visitas v
    B = np.eye(C.shape[0])
    Cp = np.eye(C.shape[0])
    for i in range(1,cantidad_de_visitas):
        Cp = Cp@C
        B += Cp
        # Sumamos las matrices de transición para cada cantidad de pasos
    return B

def analiza_top_museos(diccionario_pagerank, nombre_parametro="m"):
    # Función para analizar museos que alguna vez estuvieron en el top 3
    # Primero identificamos todos los museos que aparecieron en el top 3 al menos una vez
    museos_destacados = set()
    for vector in diccionario_pagerank.values():
        indices_top3 = np.argsort(vector)[-3:][::-1]
        museos_destacados.update(indices_top3)

    # Para cada museo en museos_destacados, registramos su ranking y PageRank en todos los parámetros
    filas = []
    for id_museo in museos_destacados:
        for valor_parametro, vector in diccionario_pagerank.items():
            posicion = np.argsort(vector)[::-1].tolist().index(id_museo) + 1
            filas.append({
                nombre_parametro: valor_parametro,
                "PageRank": vector[id_museo],
                "Museo": f"Museo {id_museo}",
                "Ranking": posicion,
                "EnTop3": posicion <= 3
            })
    
    return pd.DataFrame(filas)

def grafica_evolucion(df_resultados, nombre_parametro, valor_fijo):
    # Función para graficar evolución del PageRank de los museos destacados
    plt.figure(figsize=(12,6))
    eje = plt.gca()

    for nombre_museo in df_resultados["Museo"].unique():
        datos_museo = df_resultados[df_resultados["Museo"] == nombre_museo]
        estilo = "-" if datos_museo["EnTop3"].any() else "--"
        eje.plot(
            datos_museo[nombre_parametro],
            datos_museo["PageRank"],
            marker="o",
            linestyle=estilo,
            label=nombre_museo
        )

    eje.set_xlabel(nombre_parametro)
    eje.set_ylabel("PageRank")
    eje.set_title(
        f"Evolución de PageRank\n"
        f"(Museos que estuvieron en el Top 3, {nombre_parametro} variable, "
        f"{'alfa' if nombre_parametro == 'm' else 'm'}={valor_fijo})"
    )
    eje.grid(True)
    eje.legend(bbox_to_anchor=(1.1, 1), loc="upper left")
    plt.tight_layout()
    plt.show()