import numpy as np

cimport numpy as cnp
cimport cython

cdef extern from "math.h":
    double INFINITY

@cython.boundscheck(False)
@cython.wraparound(False)
def _maxmin_naive(object A):
    '''
    See `maxmin.maxmin_naive`. Cythonized version. Doesn't work on CSR sparse
    matrices (`scipy.sparse.csr_matrix`).
    '''
    cdef:
        int N, M
        cnp.ndarray[cnp.double_t, ndim=2] _A = np.asarray(A)
        cnp.ndarray[cnp.double_t, ndim=2] AP = np.zeros(A.shape, A.dtype)
        int i,j
        cnp.double_t max_ij, aik, akj, min_k
    N = A.shape[0]
    M = A.shape[1]
    for i in xrange(N):
        for j in xrange(M):
            max_ij = - INFINITY
            for k in xrange(N):
                aik = _A[i,k]
                akj = _A[k,j]
                min_k = min(aik, akj)
                if min_k > max_ij:
                    max_ij = min_k
            AP[i, j] = max_ij
    return AP
