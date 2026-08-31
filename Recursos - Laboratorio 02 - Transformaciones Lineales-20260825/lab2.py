import numpy as np
import math


def calcularAx(A,x):
    b = np.zeros(A.shape[0])
    filas, columnas = A.shape
    for i in range(filas):
        suma = 0
        for j in range(columnas):
            suma += A[i][j] * x[j]
        b[i] = suma
    return b

def rota(theta):
  R = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
  return R

def escala(s):
    E = np.eye(len(s))
    n = len(s)
    for i in range(n):
       for j in range(n):
          if i == j:
             E[i][i] = s[i]
    return E

def rota_y_escala(theta, s):
  return escala(s)@rota(theta)

def afin(theta,s,b):
  M = rota_y_escala(theta,s)
  T = np.eye(3)
  T[:2,:2] = M
  T[:2,2] = b
  return T

def trans_afin(v,theta,s,b):
   M = afin(theta,s,b)
   v = np.array([v[0],v[1],1])
   return (calcularAx(M,v))[:2].flatten()