import numpy as np
from ..utils.distance import matrixEDM
from ..PCA.pca import pca

def mds(D, p):
    """
    Classical multidimensional scaling (MDS)

    Parameters
    ----------
    D : (m, m) array
        Symmetric distance matrix.

    Returns
    -------
    Y : (m, p) array
        Configuration matrix. Each column represents a dimension. Only the
        p dimensions corresponding to positive eigenvalues of G_tilda are returned.
        Note that each dimension is only determined up to an overall sign,
        corresponding to a reflection.

    e : (m,) array
        Eigenvalues of G_tilda.

    """

    #D = matrixEDM(X)

    # Number of points
    m = D.shape[0]

    # Centering matrix
    P = np.eye(m) - np.ones((m, m))/m

    # YY^T
    G_tilda = -0.5 * P.dot(D).dot(P)
    G_tilda = (G_tilda + G_tilda.T)/2

    #return pca(G_tilda, 2)

    # Diagonalize
    evals, evecs = np.linalg.eigh(G_tilda)

    # Sort by eigenvalue in descending order
    idx   = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:,idx]

    # Compute the coordinates using positive-eigenvalued components only
    w, = np.where(evals > 0)
    #print(w)
    w = range(p)
    L  = np.diag(np.sqrt(evals[w]))
    V  = evecs[:,w]
    #print(L)
    #print(V)
    Y  = V.dot(L)
    #Y = np.dot(L, V.T).T
    #print(L)
    #print(evals**0.5)
    #print(V.shape)
    return Y
    #return np.dot(evecs, evals.T)
    #return evecs[:p,:] * np.sqrt(evals[:p])

def mds2(X, dim=2):
    """
    Compute classical (metric) MDS embedding of a given dimension
    based on a matrix of distances d.
    """
    d = matrixEDM(X)**0.5
    n = d.shape[0]
    A = -d**2/2.
    a = A.mean(0)

    # Compute matrix of
    # similarities based on distances

    B = (A - a[np.newaxis,:] - a[:,np.newaxis] + a.mean())

    # Lowrank (rank=dim) approximation of B

    l, v = np.linalg.eigh(B) # this can be much more efficient for
                             # large B!

    # reorder the eigenvalues / eigenvectors
    a = l.argsort()[::-1]
    A = v[:,a[:dim]] * np.sqrt(l[a[:dim]])
    return A
