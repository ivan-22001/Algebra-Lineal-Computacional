import numpy as np

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

resultado = matrizFibonacci(3)
print(resultado)