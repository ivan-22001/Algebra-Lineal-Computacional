
def solveL(L,b):
   
    n = L.shape[0]
    
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i]
        for j in range(i):
            y[i] -= L[i,j] * y[j]
    return y

def solveU(U,y):
    
    n = U.shape[0]
    x = np.zeros(n)

    for i in range(n-1,-1,-1):
        x[i] = y[i]
        for j in range(i+1,n):
            x[j] -= U[i,j] * y[j]
        x[i] /= U[i,i]
    return x
