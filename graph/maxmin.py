import numpy as np
import scipy.sparse as sp

# XXX weights == 0 are different from no edge! Use inf?
def _maxmin_dense(adj):
    '''
    Compute max-min closure on adj -- dense algorithm
    '''
    _adj = adj.todense()
    N = _adj.shape[0]
    dty = adj.dtype
    D = np.ones((N,N), dty) * np.inf
    for i in xrange(N):
        for j in xrange(N):
            dij = -1
            for k in xrange(N):
                aik = _adj[i,k]
                akj = _adj[k,j]
                if (aik > 0) and (akj > 0):
                    kmin = min(aik, akj)
                    if kmin > dij:
                        dij = kmin
            if dij > -1:
                D[i, j] = dij
    return np.matrix(D)

if __name__ == '__main__':
    from numpy.random import seed
    seed(10)
    adj = sp.rand(1000, 1000, density=.1)
    d = _maxmin_dense(adj)
