import numpy as np
import numpy.linalg as la
import timy

@timy.timer(ident='naiveNearestNeighbor', loops=10)
def naiveNearestNeighbor(X, q):
    m, n = X.shape
    minindx = 0
    mindist = np.inf
    for i in range(m):
        dist = la.norm(X[i,:] - q)
        if dist <= mindist:
            mindist = dist
            minindx = i
    return X[minindx]

@timy.timer(ident='vectorizedNearestNeighbor', loops=10)
def vectorizedNearestNeighbor(X, q):
    m, n = X.shape
    return X[np.argmin(np.sum((X - q)**2, axis=1))]


def kNearestNeighborSearch(X, q, k):
    m, n = X.shape
    sorted_inds = np.argsort(np.sum((X - q)**2, axis=1))
    return sorted_inds[:k]



