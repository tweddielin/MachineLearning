import numpy as np
import numpy.linalg as la

def pca(X, p):
    X -= X.mean(axis=0)
    C = np.cov(X.T)
    evals, evecs = la.eigh(C)
    idx   = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:,idx]

    evecs = evecs[:, :p]
    evals = evals[:p]

    #return evecs, evals
    return np.dot(X, evecs)
