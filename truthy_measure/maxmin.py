'''
maxmin
======

This module provides functions to compute the max-min (i.e. ultra-metric)
transitive closure on a similarity (weights $\in [0,1]$) graph. These functions
compute the closure on the whole graph. The notion of max-min similarity is akin
to bottleneck capacity, and these functions can thus be seen used for solving
the all-pairs shortest bottleneck path problem (APSBP).

There are two classes of algorithms implemented in this module: approaches based
on matrix multiplication, and graph traversal algorithms. Matrix multiplication
methods are guaranteed to converge only on undirected graphs or on directed
acyclical graphs (DAG). For directed graphs with cycles you can use a graph
traversal algorithms.

## Module contents

### Maxmin closure / bottleneck path
* maxmin_closure
    Max-min transitive closure via matrix multiplication, user function. This
    function uses the max-min multiplication function `maxmin` resp. `pmaxmin` to
    compute the transitive closure sequentially or in parallel, respectively.
* bottleneckpaths/cbottleneckpaths
    Single source Bottleneck paths (i.e. max-min closure for a single source).
    This is a modification of Dikstra's shortest path algorithm for computing
    the bottleneck/maxmin paths. It works on directed graphs with cycles.
    Returns the distance to all connected nodes and the paths. Note that this
    function is pure Python and thus very slow. Use the Cythonized version
    `cbottleneckpaths`, which accepts only CSR matrices.
* parallel_bottleneckpaths
    All-pairs Bottleneck paths. This is the parallel implementation for large
    graphs. The job is split on a pool of worker processes, and it is also
    possible to specify a starting index and an offset for limiting only to a
    subset of sources. Data are written to disk in a directory tree (see
    `truthy_measure.dirtree.DirTree`).

### Maxmin matrix multiplication
* maxmin
    Max-min matrix multiplication (one step), sequential version. Two
    implementation are provided, depending on whether a regular numpy array (or
    matrix), or a scipy sparse matrix is passed. See below `_maxmin_naive` and
    `_maxmin_sparse`.
* pmaxmin
    Max-min matrix multiplication (one step), parallel (process-based) version.
* _maxmin_naive
    Basic (i.e. O(N^3)) matrix multiplication. This function exists only for
    testing purpose. A fast Cython implementation is used instead.
* _maxmin_sparse
    Matrix multiplication on compressed sparse row matrix. This function exists
    only for testing purpose. A fast Cython implementation is used instead.

### Helper functions
* _maximum_csr_safe
    Element-wise maximum for CSR sparse matrices.
* _allclose_csr
    Replacement of `numpy.allclose` for CSR sparse matrices.
'''

from __future__ import division
import os
import sys
import numpy as np
import scipy.sparse as sp
from multiprocessing import Pool, Array, cpu_count, current_process
from ctypes import c_int, c_double
from contextlib import closing
from datetime import datetime
from itertools import izip
from heapq import heappush, heappop, heapify

now = datetime.now

from .utils import coo_dtype
from .cmaxmin import c_maximum_csr # see below for other imports
from .cmaxmin import bottleneckpaths as cbottleneckpaths

# Dijkstra

def bottleneckpaths(A, source):
    '''
    Finds the smallest bottleneck paths in a directed network
    '''
    N = A.shape[0]
    certain = np.zeros(N, dtype=np.bool)
    items = {} # handles to the items inserted in the queue
    Q = [] # heap queue
    # populate the queue
    for node in xrange(N):
        if node == source:
            sim = 1.
        else:
            sim = 0.0
        dist = (1.0 + sim) ** -1
        item = [dist, node, -1]
        items[node] = item
        heappush(Q, item)
    # compute bottleneck distances and predecessor information
    while len(Q):
        node_item = heappop(Q)
        dist, node, pred = node_item
        certain[node] = True
        neighbors, = np.where(A[node])
        for neighbor in neighbors:
            if not certain[neighbor]:
                neighbor_item = items[neighbor]
                neigh_dist = neighbor_item[0]
                w = (1.0 + A[node, neighbor]) ** -1
                d = max(w, dist)
                if d < neigh_dist:
                    neighbor_item[0] = d
                    neighbor_item[2] = node
                    heapify(Q)
    # generate paths
    bott_dists = []
    paths = []
    for node in xrange(N):
        item = items[node]
        if item[2] == -1: # disconnected node
            bott_dists.append(-1)
            paths.append(np.empty(0, dtype=np.int))
        else:
            bdist = item[0] ** -1 - 1.0
            bott_dists.append(bdist)
            path = []
            i = node
            while i != source:
                path.insert(0, i)
                i = items[i][2]
            path.insert(0, source)
            paths.append(np.asarray(path))
    return bott_dists, paths

maxchunksize = 100000
max_tasks_per_worker = 500
log_out = 'bottleneck_dists-{proc:0{width}d}.log'
log_out_start = 'bottleneck_dists_{start:0{width1}d}-{{proc:0{{width}}d}}.log'
logline = "{now}: worker-{proc:0{width}d}: source {source} completed."

_nprocs = None
_logpath = None
_dirtree = None

def _bottleneck_worker(n):
    global _A, _dirtree, _logpath, _logf, _nprocs, digits_procs
    worker_id, = current_process()._identity
    logpath = _logpath.format(proc=worker_id, width=digits_procs)
    outpath = _dirtree.getleaf(n)
    with \
            closing(open(outpath, 'w')) as outf,\
            closing(open(logpath, 'a', 1)) as logf:
        dists, paths = cbottleneckpaths(_A, n, outf)
        logf.write(logline.format(now=now(), source=n, proc=worker_id,
                width=digits_procs) + '\n')

def _init_worker_dirtree(nprocs, logpath, dirtree, indptr, indices, data,
        shape):
    global _dirtree, _logpath, _nprocs, digits_procs, digits_rows
    _nprocs = nprocs
    digits_procs = int(np.ceil(np.log10(_nprocs)))
    _logpath = logpath
    _dirtree = dirtree
    _init_worker(indptr, indices, data, shape)

def parallel_bottleneckpaths(A, dirtree, start=None, offset=None, nprocs=None):
    '''
    Computes the all-pairs bottleneck paths for matrix A and saves the results
    in a compressed tar archive called `bottlenecks.tar.gz`. Each member of the
    tar archive is a compressed NumPy binary archive containing a distance
    vector and an array for each path length, containing all paths of that given
    path length. The directory tree of the TAR archive is managed by DirTree
    instance `dirtree`. All intermediate files and directories are deleted.

    If `start` and `offset` parameters are passed, then only indices between
    `start` and `start + offset` are computed, and the resulting TAR file is
    called `bottlenecks_<start>.tar.gz`. The directory tree used to store
    intermediate result is NOT deleted (since other processes might still need
    it), but the uncompressed output files are.

    Parameters
    ----------
    A : array_like
        NxN adjacency matrix, will be converted to CSR format.

    dirtree : a `truthy_measure.utils.DirTree` instance
        The directory tree object used to generate the directory tree in which
        the results are stored.

    start : int
        optional; see above.

    offset : int
        optional, but if `start` is passed than offset is expected too. See
        above.

    nprocs : int
        optional; number of processes to spawn. Default is 90% of available
        CPUs/cores.
    '''
    N = A.shape[0]
    digits = int(np.ceil(np.log10(N)))
    if start is None:
        fromi = 0
        toi = N
    else:
        assert offset is not None
        assert 0 <= offset <= N
        assert start >= 0
        fromi = start
        toi = start + offset
    if nprocs is None:
        # by default use 90% of available processors or 2, whichever the
        # largest.
        nprocs = max(int(0.9 * cpu_count()), 2)
    # allocate array to be passed as shared memory
    A = sp.csr_matrix(A)
    indptr = Array(c_int, A.indptr)
    indices = Array(c_int, A.indices)
    data = Array(c_double, A.data)
    if start is None:
        logpath = log_out
    else:
        logpath = log_out_start.format(start=start, width1=digits)
    initargs = (nprocs, logpath, dirtree, indptr, indices, data, A.shape)
    print '{}: launching pool of {} workers.'.format(now(), nprocs)
    pool = Pool(processes=nprocs, initializer=_init_worker_dirtree,
            initargs=initargs, maxtasksperchild=max_tasks_per_worker)
    with closing(pool):
        pool.map(_bottleneck_worker, xrange(fromi, toi))
    pool.join()
    print '{}: done'.format(now())

# Matrix multiplication

def maxmin_closure(A, parallel=False, maxiter=1000, quiet=False,
        dumpiter=None, **kwrds):
    '''
    Computes the max-min product closure. This algorithm is based matrix
    operation and is guaranteed to converge only in the case of undirected
    graphs or directed acyclical graphs (DAG).

    Parameters
    ----------
    A : array_like
        an NxN adjacency matrix. Can be either sparse or dense.
    parallel : bool
        if True, the parallel maxmin is used.
    maxiter  : integer
        maximum number of iterations for the closure loop. Will warn if the
        maximum number of iterations is reached without convergence.
    quiet : bool
        if True, will not print the current time at each iteration.
    dumpiter : bool
        if True, will dump to file `closure_<iter>.npy` the intermediate matrix
        computed at each iteration.

    Additional keyword arguments are passed to p/maxmin.

    Returns
    -------
    closure : array_like
        The max-min, or ultra-metric, closure. This is also equal to the
        all-pairs bottleneck paths. Zero entries correspond to disconnected
        pairs, i.e. null capacity paths. If parallel is True, returns a matrix
        in compressed sparse row format (CSR). See `scipy.sparse`.
    '''
    if parallel:
        A = sp.csr_matrix(A)
        AP = pmaxmin(A, **kwrds)
    else:
        AP = maxmin(A, **kwrds)
    AP = _maximum_csr_safe(A, AP)
    iterations = 1
    if dumpiter:
        _AP = AP.tocoo()
        fn = 'closure_%d.npy' % iterations
        np.save(fn, np.fromiter(izip(_AP.row, _AP.col, _AP.data), coo_dtype,
            len(_AP.data)))
        if not quiet:
            print 'Intermediate matrix saved to %s.' % fn
    while not _allclose_csr(A, AP) and iterations < maxiter:
        A = AP.copy()
        if parallel:
            AP = pmaxmin(A, **kwrds)
        else:
            AP = maxmin(A, **kwrds)
        AP = _maximum_csr_safe(A, AP)
        iterations += 1
        if dumpiter:
            _AP = AP.tocoo()
            fn = 'closure_%d.npy' % iterations
            np.save(fn, np.fromiter(izip(_AP.row, _AP.col, _AP.data), coo_dtype,
                len(_AP.data)))
            if not quiet:
                print 'Intermediate matrix saved to %s.' % fn
        if not quiet:
            print '%s: iteration %d completed.' % (datetime.now(), iterations +
                    1)
    if not _allclose_csr(A, AP):
        print 'Closure did not converge in %d iterations!' % maxiter
    else:
        print 'Closure converged after %d iterations.' % iterations
    return AP

def maxmin(A, a=None, b=None, sparse=False):
    '''
    Compute the max-min product of A with itself:

    [ AP ]_ij = max_k min ( [ A ]_ik, [ A ]_kj )

    Parameters
    ----------
    A : array_like
        A 2D square ndarray, matrix or sparse (CSR) matrix (see `scipy.sparse`).
        The sparse implementation will be used automatically for sparse
        matrices.
    a,b : integer
        optional; compute only the max-min product between A[a:b,:] and A.T
    sparse : bool
        if True, transforms A to CSR matrix format and use the sparse
        implementation.

    Return
    ------
    A' : array_like
        The max-min product of A with itself. A CSR sparse matrix will be
        returned if the sparse implementation is used, otherwise a numpy matrix.
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

# Global variables used by _maxmin_worker (see below)

_indptr = None
_indices = None
_data = None
_A = None

# Pool functions

def _init_worker(indptr, indices, data, shape):
    '''
    See `pmaxmin`. This is the worker initialization function.
    '''
    global _indptr, _indices, _data, _A
    _indptr = np.frombuffer(indptr.get_obj(), dtype=np.int32)
    _indices = np.frombuffer(indices.get_obj(), dtype=np.int32)
    _data = np.frombuffer(data.get_obj())
    _A = sp.csr_matrix((_data, _indices.astype('int32'), _indptr), shape)

def _maxmin_worker(a_b):
    '''
    See `pmaxmin`. This is the map function each worker executes
    '''
    global _A
    a, b = a_b
    # return also the first index to help re-arrange the result
    return maxmin(_A, a, b)

# TODO switch from processes to threads, refactor the mmclosure_matmul, move to
# Cython and release the GIL like this:
#
# with nogil:
#   <do stuff>
def pmaxmin(A, splits=None, nprocs=None):
    '''
    See `maxmin`. Parallel version. Splits the rows of A in even intervals and
    distribute them to a pool of workers.

    Parameters
    ----------
    A : array_like
        a 2D array, matrix, or CSR matrix representing an NxN adjacency matrix
    splits : integer
        split the rows of A in equal intervals. If not provided, each worker
        will be assigned exactly an interval. If `split` is not an integer
        divisor of the number of rows of A, the last interval will be equal to
        the remainder of the integer division.
    nprocs : integer
        number of workers to spawn.

    Returns
    -------
    maxmin : `scipy.sparse.csr_matrix`
        The maxmin composition of A with itself. See `maxmin`.
    '''
    N = A.shape[0]
    if nprocs is None:
        nprocs = cpu_count()
        nprocs -= nprocs // 10
        nprocs = max(nprocs, 2)

    # check splits
    if splits is None:
        if N > nprocs:
            splits = nprocs
        else:
            splits = N
    else:
        if not isinstance(splits, int):
            raise TypeError('expecting an integer number of splits')
        if splits > N:
            raise ValueError('too many splits for %d rows: %d' % (N, splits))
    if not sp.isspmatrix_csr(A):
        A = sp.csr_matrix(A)

    chunk_size = int(np.ceil(N / splits))
    breaks = [(i, min(i + chunk_size, N)) for i in xrange(0, N, chunk_size)]

    # Wrap the indptr/indices and data arrays of the CSR matrix into shared
    # memory arrays and pass them to the initialization function of the workers
    # NOTE: this introduces overhead, as it copies each array to a new memory
    # location.
    indptr = Array(c_int, A.indptr)
    indices = Array(c_int, A.indices)
    data = Array(c_double, A.data)
    initargs = (indptr, indices, data, A.shape)

    # create the pool; this will initialize the workers with the shared memory
    # arrays
    pool = Pool(processes=nprocs, initializer=_init_worker, initargs=initargs)

    with closing(pool):

        # call map, reassemble result
        chunks = pool.map(_maxmin_worker, breaks)
        AP = sp.vstack(chunks).tocsr()

    # wait for worker processes to terminate
    pool.join()
    return AP

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

def _maximum_csr_safe(A, B):
    '''
    Safe version of `numpy.maximum` for CSR matrices
    '''
    # fall back on numpy's default if both matrices are dense
    if not sp.isspmatrix(A) and not sp.isspmatrix(B):
        return np.maximum(A, B)

    # if one of the two inputs is sparse and the other is dense, convert the
    # latter to sparse
    if not sp.isspmatrix_csr(A):
        A = sp.csr_matrix(A)
    if not sp.isspmatrix_csr(B):
        B = sp.csr_matrix(B)

    return c_maximum_csr(A, B)

def _allclose_csr(A, B, **kwrds):
    '''
    CSR matrices-safe equivalent of allclose. Additional keyword are passed to
    allclose. See `numpy.allclose`. Will call the numpy version if passed dense
    matrices.
    '''
    # fall back on numpy's allclose if both matrices are dense
    if not sp.isspmatrix(A) and not sp.isspmatrix(B):
        return np.allclose(A, B)

    if not sp.isspmatrix_csr(A):
        A = sp.csr_matrix(A)

    if not sp.isspmatrix_csr(B):
        B = sp.csr_matrix(B)

    # check indices
    indices_all = np.all(A.indices == B.indices)
    if not indices_all:
        return False

    # check indices pointers
    indptr_all = np.all(A.indptr == B.indptr)
    if not indptr_all:
        return False

    # check data
    return np.allclose(A.data, B.data, **kwrds)

# try importing the fast C implementations first, otherwise use the Python
# versions provided in this module as a fallback
try:
    from .cmaxmin import c_maxmin_naive as maxmin_naive,\
            c_maxmin_sparse as maxmin_sparse
except ImportError:
    import warnings
    warnings.warn('Could not import fast C implementation!')
    maxmin_naive = _maxmin_naive
    maxmin_sparse = _maxmin_sparse
