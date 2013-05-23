import numpy as np
import scipy.sparse as sp

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

@cython.boundscheck(False)
@cython.wraparound(False)
def _maxmin_sparse(object A):
    '''
    See `maxmin.maxmin_sparse`. Cythonized version. Will convert argument to
    sparse CSR matrix (see `scipy.sparse.csr_matrix`).
    '''
    cdef:
        cnp.ndarray[cnp.int_t, ndim=1] A_indptr, A_indices, At_indptr, At_indices
        cnp.ndarray[cnp.double_t, ndim=1] A_data, At_data
        object AP_data, AP_indptr, AP_indices
        int N, M, i, j, ii, jj, ik, kj, iimax, jjmax, innz, iptr
        double max_ij, min_k

    if not sp.isspmatrix_csr(A):
        A = sp.csr_matrix(A)
    N = A.shape[0] 
    M = A.shape[1]
    # output array
    AP_indptr = [0]
    AP_indices = []
    AP_data = []
    iptr = 0
    At = A.transpose().tocsr() # transpose of A in CSR format
    A_indptr = A.indptr
    A_indices = A.indices
    A_data = A.data
    At_indptr = At.indptr
    At_indices = At.indices
    At_data = At.data

    for i in xrange(N):
    
        innz = 0

        for j in xrange(M):
            
            # ii is the index of the first non-zero element value (in A.data)
            # and column index (in A.indices) of the the i-th row
            ii = A_indptr[i] 
            iimax = A_indptr[i + 1]

            # jj is the index of the first non-zero element value (in At.data)
            # and column (that is, row) index (in A.indices) of the the j-th row
            # (that is, column).
            jj = At_indptr[j]
            jjmax = At_indptr[j + 1]

            max_ij = 0.
            
            while (ii < iimax) and (jj < jjmax):
                
                ik = A_indices[ii]
                kj = At_indices[jj]

                if ik == kj: 
                    # same element, apply min
                    min_k = min(A_data[ii], At_data[jj])
                    # update the maximum so far
                    if min_k > max_ij:
                        max_ij = min_k
                    ii += 1
                    jj += 1

                elif ik > kj: 
                    # the row element (in A) corresponding to kj is zero,
                    # hence min_k is zero. Advance only jj.
                    jj += 1

                else: # ik < kj
                    # the column elment (in At) corresponding to ik is zero,
                    # hence min_k is zero. Advance only ii.
                    ii += 1

            if max_ij:
                AP_data.append(max_ij)
                AP_indices.append(j)
                innz += 1

        iptr += innz
        AP_indptr.append(iptr)


    # return in CSR format
    return sp.csr_matrix((np.asarray(AP_data), AP_indices, AP_indptr), (N, N))
