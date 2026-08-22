
import numpy as np
import matplotlib.pyplot as plt

#Ejercicio 1
def esCuadrada(A):
    if A.ndim != 2:
        return False
    filas,columnas = A.shape
    if filas != columnas:
        return False
    return True

#Ejercicio 2
def triangSup(A):
    if not esCuadrada(A):
        return None
    filas, columnas = A.shape
    U = np.copy(A)
    for i in range(filas):
        for j in range(columnas):
            if i>= j:
                U[i][j] = 0
    return U

#Ejercicio 3
def triangInf(A):
    if not esCuadrada(A):
        return None
    filas, columnas = A.shape
    U = np.copy(A)
    for i in range(filas):
        for j in range(columnas):
            if i<= j:
                U[i][j] = 0
    return U

#Ejercicio 4
def diagonal(A):
    D = np.zeros(A.shape)
    filas = D.shape[0]
    for i in range(filas):
        D[i][i] = A[i][i]
    return D

#Ejercicio 5
def traza(A):
    suma = 0
    filas = A.shape[0]
    for i in range(filas):
        suma += A[i][i]
    return suma

#Ejercicio 6
def traspuesta(A):
    T = np.copy(A)
    filas, columnas = A.shape
    for i in range(filas):
        for j in range(columnas):
            T[i][j] = A[j][i]
    return T


#Ejercicio 7
def esSimetrica(A):
    filas, columnas = A.shape
    for i in range(filas):
        for j in range(columnas):
            if A[i][j] != A[j][i]:
                return False
    return True

#Ejercicio 8
def calcularAx(A,x):
    b = np.zeros(A.shape[0])
    filas, columnas = A.shape
    for i in range(filas):
        suma = 0
        for j in range(columnas):
            suma += A[i][j] * x[j]
        b[i] = suma
    return b

#Ejercicio 9
def intercambiarFilas(A,i,j):
    U = np.copy(A)
    columnas = A.shape[1]
    for k in range(columnas):
        U[i][k], U[j][k] = U[j][k], U[i][k]
    return U

#Ejercicio 10
def sumar_fila_multiplo(A,i,j,s):
    U = np.copy(A)
    columnas = A.shape[1]
    U[i] += U[j]*s
    return U

#Ejercicio 11
def esDiagonalmenteDominante(A):
    fila, columna = A.shape
    for i in range(fila):
        suma = 0
        for j in range(columna):
            suma += A[i][j]
        if abs(A[i][i]) < abs(suma):
            return False

    return True

#Ejercicio 12
def matrizCirculante(v):
    largo = len(v)
    filas = []
    for i in range(largo):
        rotada = v
        if i > 0:
            rotada = v[-i:] + v[:-i]
        filas.append(rotada)
    M = np.array(filas)
    return M

#Ejercicio 13
def matrizVandermonde(v):
    n = len(v)
    M = np.zeros((n,n))
    filas = n
    columnas = filas
    for i in range(columnas):
        for j in range(filas):
            M[j][i] = v[j]**i
    return M 



#Ejercicio 14

## (Fk+2,Fk+1) = M(Fk+1,Fk)



def numeroAureo(n):
    M = np.array([[1,1],[1,0]])

    phis = []
    v = np.array([1,0]) # [F1,F0]
    M_power = np.eye(2)
    for i in range(1,n+1):
        M_power = M_power @ M
        phi = M_power[0,0]/M_power[0,1]
        print('Valor de phi: ', phi, 'para n = ',i)
        phis.append(phi)

    n_vals = np.linspace(1,n,n)
    plt.figure()
    plt.plot(n_vals, phis)
    plt.xlabel("n")
    plt.ylabel("Aproximación de φ")
    plt.show()
    
     

def multiplicarMatrices(A,B):
    filasA,columnasA = A.shape
    filasB,columnasB = B.shape
    if columnasA != filasB:
        return None
    M = np.zeros((filasA,columnasB))
    for i in range(filasA):
        for j in range(columnasB):
            suma = 0
            for k in range(filasB):
                suma += A[i][k] * B[k][j]
            M[i][j] = suma
    return M

#Ejercicio 15
def matrizFibonacci(n):
    M = np.zeros((n,n))
    M[0][0],M[0][1] = 0, 1

    for i in range(n):
        for j in range(n):
            if j < 2: 
                if i > 0:
                    if j == 0:
                        M[i][j] = M[i-1][n-1]+M[i-1][n-2]
                    if j == 1:
                        M[i][j] = M[i][j-1]+M[i-1][n-1]
            else:
                M[i][j] = M[i][j-1] + M[i][j-2]

    return M

#Ejercicio 16
def matrizHilbert(n):
    M = np.zeros((n,n))
    filas, columnas = M.shape
    for i in range(filas):
        for j in range(columnas):
            M[i][j] = 1/(i+j+1)

    return M 

#Ejercicio 17
def calcularPolinomios():
    x = np.linspace(-1,1,100)
    V = matrizVandermonde(x) 
    V = V[:, :11]
    p1 = np.array([-1, 1, -1, 1, -1, 1, 0, 0, 0, 0, 0])   # grado 5, relleno hasta 10
    p2 = np.array([3, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])       # grado 2, relleno hasta 10
    p3 = np.array([-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])      # grado 10

    M = np.array([p1, p2, p3])
    y = M @ V.T

    plt.plot(x, y[0], label='p1')
    plt.plot(x, y[1], label='p2')
    plt.plot(x, y[2], label='p3')
    plt.legend()
    plt.show()


#Ejercicio 18
def row_echelon(M):
    """ 
        Retorna la Matriz Escalonada por Filas 
    """
    A = np.copy(M)
    if (issubclass(A.dtype.type, np.integer)):
        A = A.astype(float)
    # Si A no tiene filas o columnas, ya esta escalonada
    f, c = A.shape
    if f == 0 or c == 0:
        return A
    # buscamos primer elemento no nulo de la primera columna
    i = 0
    max_val = abs(A[i,0])
    max_fila = i
    while i < f:
        if abs(A[i,0]) > max_val:
            max_val = abs(A[i,0])
            max_fila = 0
        i += 1

    if max_val == 0: #el valor absoluto mayor es 0, entonces entra aca
        # si todos los elementos de la primera columna son ceros
        # escalonamos filas desde la segunda columna
        B = row_echelon(A[:,1:])
        # y volvemos a agregar la primera columna de zeros
        return np.block([A[:,:1], B])
    # intercambiamos filas i <-> 0, pues el primer cero aparece en la fila i
    if max_fila > 0:
        A[[0,max_fila],:] = A[[max_fila,0],:]
    # PASO DE TRIANGULACION GAUSSIANA:
    # a las filas subsiguientes les restamos un multiplo de la primera
    A[1:,:] -= (A[0,:] / A[0,0]) * A[1:,0:1]
    # escalonamos desde la segunda fila y segunda columna en adelante
    B = row_echelon(A[1:,1:])
    # reconstruimos la matriz por bloques adosando a B la primera fila 
    # y la primera columna (de ceros)
    return np.block([ [A[:1,:]], [ A[1:,:1], B] ])

def main():
    #Definición de matriz
    A = np.array([[1, -2,3], [-4, -4,2],[10,-2,1]])
    B = np.copy(A)
    print('Matriz A\n', A)
    print('Es cuadrada?\n', esCuadrada(A))
    print('Triangular superior:\n',  triangSup(A))
    print('Triangular inferior:\n',  triangInf(A))
    print('Diagonal:\n',  diagonal(A))
    print('Traza de A:\n',traza(A))
    print('Traspuesta de A:\n',traspuesta(A))
    print('Es simetrica A:', esSimetrica(A))
    x = np.array([2,2,2])
    print('Producto A y x:', calcularAx(A,x))
    print('Intercambio filas 1 y 2 de A:\n', intercambiarFilas(A,0,1))
    print('A la fila 1 le suma la fila 2 multiplicada por 3:\n', sumar_fila_multiplo(A,0,1,3))
    print('Es diagonalmente dominante:' , esDiagonalmenteDominante(A))
    v = [1,2,3,4,5]
    print('Filas rotadas para vector:',v,'son:\n',matrizCirculante(v))
    print('La matriz Vandermonde de:',v,'es:\n',matrizVandermonde(v))
    print('Matriz Fibonacci para 6:\n', matrizFibonacci(6))
    print('Matriz de Hilbert para n=5 es:\n', matrizHilbert(5))
    print('Matriz escalonada A:\n',row_echelon(A))
    #calcularPolinomios()
    numeroAureo(15)

    exit()

if __name__ == "__main__":
    main()