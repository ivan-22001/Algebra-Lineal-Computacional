import numpy as np


def error(x,y):
  return abs(np.float64(x)-np.float64(y))

def error_relativo(x,y):
  return error(x,y)/abs(np.float64(x))

def matricesIguales(A,B):
  if A.shape != B.shape:
    return False
  
  fil, col = A.shape

  for i in range(fil):
    for j in range(col):
      if abs(A[i,j] - B[i,j]) > 1e-9:
       return False
  return True



def esSimetrica(A):
  fil, col = A.shape

  for i in range(fil):
    for j in range(col):
      if abs(A[i,j] - A[j,i]) > 1e-9:
       return False
  return True
