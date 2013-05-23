import numpy as np
import scipy.sparse as sp

from _maxmin import _maxmin_naive

def maxmin_naive(A):
    '''
    Compute max-min product on A:

    [ AP ]_ij = max_k min ( A_ik, A_kj )
    
    This is the naive algorithm that runs in O(n^3). It should be used only
    for testing with small matrices. Works both on dense and CSR sparse
    matrices.
    '''
    N, M = A.shape
    AP = np.zeros(A.shape, A.dtype)
    for i in xrange(N):
        for j in xrange(M):
            max_ij = - np.inf
            for k in xrange(N):
                aik = A[i,k]
                akj = A[k,j]
                min_k = min(aik, akj)
                if min_k > max_ij:
                    max_ij = min_k
            AP[i, j] = max_ij
    return AP

def maxmin_sparse(A):
    '''
    Implementation for CSR sparse matrix (see `scipy.sparse.csr_matrix`)
    '''
    # TODO: move following two lines to future `maxmin` function
    if not sp.isspmatrix_csr(A):
        A = sp.csr_matrix(A)
    N, M = A.shape
    AP = sp.dok_matrix(A.shape, A.dtype)
    At = A.transpose().tocsr() # transpose of A in CSR format

    for i in xrange(N):
    
        for j in xrange(M):
            
            # ii is the index of the first non-zero element value (in A.data)
            # and column index (in A.indices) of the the i-th row
            ii = A.indptr[i] 
            iimax = A.indptr[i + 1]

            # jj is the index of the first non-zero element value (in At.data)
            # and column (that is, row) index (in A.indices) of the the j-th row
            # (that is, column).
            jj = At.indptr[j]
            jjmax = At.indptr[j + 1]

            max_ij = 0.
            
            while (ii < iimax) and (jj < jjmax):
                
                ik = A.indices[ii]
                kj = At.indices[jj]

                if ik == kj: 
                    # same element, apply min
                    min_k = min(A.data[ii], At.data[jj])
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
                AP[i,j] = max_ij

    # return in CSR format
    return AP.tocsr()




