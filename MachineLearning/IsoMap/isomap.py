import numpy as np
from ..KNearestNeighbor.knn import kNearestNeighborSearch
from ..MDS.mds import mds, mds2
from ..utils.distance import matrixEDM
from ..PCA.pca import pca
from scipy.sparse.csgraph import shortest_path
from sklearn import manifold
from sklearn.decomposition import KernelPCA

def delta_matrix(X, D, k):
    m, n = X.shape
    d = np.full((m, m), np.inf)
    np.fill_diagonal(d, 0)
    for i in range(m):
        neighbors = kNearestNeighborSearch(X, X[i], k+1)
        #d[i, neighbors] = D[i, neighbors]
        for ne in neighbors:
            d[i, ne] = D[i, ne]
    return np.minimum(d, d.T)

def shortest_pathFW(d):
    m = d.shape[0]
    for k in range(m):
        for i in range(m):
            for j in range(m):
                if d[i, j] > d[i, k] + d[k, j]:
                    d[i, j] = d[i, k] + d[k, j]
    return d

def isomap(X, k, p):
    D = matrixEDM(X)**0.5
    d_matrix = delta_matrix(X, D, k)
    #d_matrix = manifold.Isomap(n_neighbors=15, n_components=2).fit(X).dist_matrix_
    #d = shortest_pathFW(d_matrix)
    d = shortest_path(d_matrix)
    #print(d)

    #y = pca(d, p)
    d = d**2
    y = mds(d, p)
    #kernelpca = KernelPCA(n_components=2, kernel="precomputed")
    #y = kernelpca.fit_transform(d)
    return y




