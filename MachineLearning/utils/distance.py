import numpy as np
import numpy.linalg as la
import scipy.spatial as spt
import timy

@timy.timer(ident='naiveEDM', loops=10)
def naiveEDM(X):
    m, n = X.shape
    D = np.zeros((m, m))
    for i in range(m):
        for j in range(i+1, m):
            D[i,j] = la.norm(X[i, :] - X[j, :])**2
            D[j,i] = D[i,j]
    return D

@timy.timer(ident='dotEDM', loops=10)
def dotEDM(X):
    m, n = X.shape
    D = np.zeros((m, m))
    for i in range(m):
        for j in range(i+1, m):
            d = X[i, :] - X[j, :]
            D[i, j] = np.dot(d, d.T)
            D[j, i] = D[i, j]
    return D
    
@timy.timer(ident='matrixEDM', loops=10)
def matrixEDM(X):
    m, n = X.shape
    G = np.dot(X, X.T)
    H = np.tile(np.diag(G), (m,1))
    return H + H.T - 2*G
