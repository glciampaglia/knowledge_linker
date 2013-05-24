import numpy as np
import scipy.sparse as sp
from multiprocessing import Pool, Array, cpu_count
from ctypes import c_int, c_double

# TODO understand why sometimes _init_worker raises a warning complaining that
# the indices array has dtype float64. This happens intermittently.

# Global variables each worker needs

_indptr = None
_indices = None
_data = None
_A = None

# Each worker process is initialized with this function

def _init_worker(indptr, indices, data, shape):
    '''
    See `pmaxmin`. This is the worker initialization function.
    '''
    global _indptr, _indices, _data, _A
    _indptr = np.frombuffer(indptr.get_obj(), dtype=np.int32)
    _indices = np.frombuffer(indices.get_obj(), dtype=np.int32)
    _data = np.frombuffer(data.get_obj())
    _A = sp.csr_matrix((_data, _indices, _indptr), shape)

def _maxmin_worker(a_b):
    '''
    See `pmaxmin`. This is the map function each worker executes
    '''
    global _A
    a, b = a_b
    # return also the first index to help re-arrange the result
    return a, maxmin(_A, a, b)

# Parallel version

def pmaxmin(A, splits=None, nprocs=None):
    '''
    See `maxmin`. Parallel version. Splits the rows of A in even intervals and
    distribute them to a pool of workers. 

    Parameter
    ---------
    A       - a 2D array, matrix, or CSR matrix
    splits  - integer; split the rows of A in equal intervals. If not provided, each
              worker will be assigned exactly an interval. If `split` is not
              a divisor of the number of rows of A, the last interval will be
              equal to the remainder of the division. 
    nprocs  - integer; number of workers to spawn.
    '''
    if nprocs is None:
        nprocs = cpu_count()
    if splits is None:
        splits = nprocs
    if not isinstance(splits, int):
        raise TypeError('expecting an integer number of splits')
    N = A.shape[0]
    if not sp.isspmatrix_csr(A):
        A = sp.csr_matrix(A)
    chunk_size = N / splits
    breaks = [(i, i + chunk_size) for i in xrange(0, N, chunk_size)]
    if splits % N != 0:
        a, b = breaks[-1]
        breaks[-1] = (a, N)

    # Wrap the indptr/indices and data arrays of the CSR matrix into shared
    # memory arrays and pass them to the initialization function of the workers
    indptr = Array(c_int, A.indptr)
    indices = Array(c_int, A.indices)
    data = Array(c_double, A.data)
    initargs = (indptr, indices, data, A.shape)

    # create the pool, call map, reassemble result
    pool = Pool(processes=nprocs, initializer=_init_worker, initargs=initargs)
    result = pool.map(_maxmin_worker, breaks)
    chunks = zip(*sorted(result, key=lambda k : k[0]))[1]
    return sp.vstack(chunks).tocsr()

# Frontend function. 

def maxmin(A, a=None, b=None, sparse=False):
    '''
    Compute the max-min product of A with itself:

    [ AP ]_ij = max_k min ( A_ik, A_kj )

    Parameters
    ----------
    A       - A 2D square ndarray, matrix or sparse (CSR) matrix (see
              `scipy.sparse`). The sparse implementation will be used
              automatically for sparse matrices.
    a,b     - optional integers; compute only the max-min product between
              A[a:b,:] and A.T 
    sparse  - if True, transforms A to CSR matrix format and use the sparse
              implementation.

    Return
    ------
    A CSR sparse matrix if the sparse implementation is used, otherwise a numpy
    matrix.
    '''
    if A.ndim != 2:
        raise ValueError('expecting 2D array or matrix')
    N, M = A.shape
    if N != M:
        raise ValueError('argument must be a square array')
    if a is not None:
        if (a < 0) or (a > N):
            raise ValueError('a cannot be smaller nor larger than axis dim')
    if b is not None:
        if (b < 0) or (b > N):
            raise ValueError('b cannot be smaller nor larger than axis dim')
    if (a is not None) and (b is not None):
        if a > b:
            raise ValueError('a must be less or equal b')
    if sp.isspmatrix(A) or sparse:
        if not sp.isspmatrix_csr(A):
            A = sp.csr_matrix(A)
        return maxmin_sparse(A, a, b)
    else:
        return np.matrix(maxmin_naive(A, a, b))


# These functions don't perform checks on the argument, so don't use these
# functions directly. Use the frontend instead.

def _maxmin_naive(A, a=None, b=None):
    '''
    See `maxmin`. This is the naive algorithm that runs in O(n^3). It should be
    used only for testing with small matrices. Works both on dense and CSR
    sparse matrices. 

    Don't use these functions directly. Use the frontend instead.
    '''
    N = A.shape[0]
    if a is None: 
        a = 0
    if b is None: 
        b = N
    Nout = b - a
    AP = np.zeros((Nout, N), A.dtype)
    for i in xrange(Nout):
        ih = a + i
        for j in xrange(N):
            max_ij = 0.
            for k in xrange(N):
                aik = A[ih, k]
                akj = A[k, j]
                min_k = min(aik, akj)
                if min_k > max_ij:
                    max_ij = min_k
            AP[i, j] = max_ij
    return AP

def _maxmin_sparse(A, a=None, b=None):
    '''
    Implementation for CSR sparse matrix (see `scipy.sparse.csr_matrix`)
    '''
    if not sp.isspmatrix_csr(A):
        raise ValueError('expecting a sparse CSR matrix')

    N = A.shape[0]
    if a is None: 
        a = 0
    if b is None: 
        b = N
    Nout = b - a

    # AP is the output matrix, At is A in compressed sparse column format (CSC)
    AP = sp.dok_matrix((Nout, N), A.dtype)
    At = A.tocsc()

    for i in xrange(Nout):
    
        ih = a + i
        for j in xrange(N):
            
            # ii is the index of the first non-zero element value (in A.data)
            # and column index (in A.indices) of the the i-th row
            ii = A.indptr[ih] 
            iimax = A.indptr[ih + 1]

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

# try importing the fast C implementations first, otherwise use the Python
# versions provided in this module as a fallback
try:
    from cmaxmin import c_maxmin_naive as maxmin_naive,\
            c_maxmin_sparse as maxmin_sparse
except ImportError:
    import warnings
    warnings.warn('Could not import fast C implementation!')
    maxmin_naive = _maxmin_naive
    maxmin_sparse = _maxmin_sparse

if __name__ == '__main__':
    from time import time
    np.random.seed(10)
    B = sp.rand(8000, 8000, 1e-4, 'csr')
    print 'Testing maxmin on a matrix of shape %s with nnz = %d:' % (B.shape,
            B.getnnz())

    tic = time()
    C = pmaxmin(B, 10, 10)
    toc = time()
    print '* parallel version executed in %.2e seconds' % (toc - tic)

    tic = time()
    D = maxmin(B)
    toc = time()
    print '* serial version executed in %.2e seconds' % (toc - tic)
