from __future__ import division
import os
import sys
import numpy as np
import scipy.sparse as sp
from multiprocessing import Pool, Array, cpu_count
from ctypes import c_int, c_double
from contextlib import closing
import warnings
from datetime import datetime
from itertools import izip
from collections import defaultdict
from array import array

from .utils import coo_dtype
from .cmaxmin import c_maximum_csr # see below for other imports

# TODO understand why sometimes _init_worker raises a warning complaining that
# the indices array has dtype float64. This happens intermittently. In the
# meanwhile, just ignore it.

warnings.filterwarnings('ignore', 
        message='.*',
        module='scipy\.sparse\.compressed.*',
        lineno=122)

# Format warnings nicely

def _showwarning(message, category, filename, lineno, line=None):
    filename = os.path.basename(__file__)
    warning = category.__name__
    print >> sys.stderr, '{}:{}: {}: {}'.format(filename, lineno, warning, message)

warnings.showwarning = _showwarning

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
    _A = sp.csr_matrix((_data, _indices.astype('int32'), _indptr), shape)

def _maxmin_worker(a_b):
    '''
    See `pmaxmin`. This is the map function each worker executes
    '''
    global _A
    a, b = a_b
    # return also the first index to help re-arrange the result
    return maxmin(_A, a, b)

# Parallel version

# TODO switch from processes to threads, refactor the productclosure, move to
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

# maxmin closure for directed networks with cycles. Recursive implementation.

def maxmin_closure_cycles_recursive(A):
    '''
    Maxmin transitive closure. Recursive implementation. If A is large, this
    implementation will likely raise RuntimeError for reaching the maximum
    recursion depth.

    See `maxmin_closure_cycles` for parameters, return value, etc.
    '''
    AT = np.zeros(A.shape)
    for row, col, weight in _maxmin_closure_cycles_recursive(A):
        AT[row, col] = weight
    return AT

def _maxmin_closure_cycles_recursive(A):
    '''
    Maxmin transitive closure, recursive implementation. Returns an iterator
    over COO tuples.

    NOTE: The frontend `maxmin_closure_cycles_recursive` constructs a 2-D array
    out of it.
    '''
    # the full successors set
    def _succ(node):
        r = root[node]
        s = succ[r] # only SCC roots
        return np.where(reduce(_ndor, map(_rooteq, s)))[0]
    def search(node, target, min_weight, weights):
        if node == target: # append minimum of path
            if min_weight != maxval:
                weights.append(min_weight)
        if target not in _succ(node): # prune branch
            return
        for neighbor in A.rows[node]:
            if explored[(node, neighbor)]:
                continue
            explored[(node, neighbor)] = True
            w = float(A[node, neighbor]) # copy value
            if w < min_weight:
                search(neighbor, target, w, weights)
            else:
                search(neighbor, target, min_weight, weights)
    maxval = np.inf
    A = sp.lil_matrix(A)
    root, succ = closure_cycles(A)
    _ndor = np.ndarray.__or__ # element-wise or (x|y)
    _rooteq = root.__eq__ 
    for source in xrange(A.shape[0]):
        for target in _succ(source):
            explored = defaultdict(bool)
            weights = [] # for each path
            search(source, target, maxval, weights)
            yield (source, target, max(weights))

# maxmin closure for directed networks with cycles. Iterative implementation.

def maxmin_closure_cycles(A):
    '''
    Maxmin transitive closure.

    Parameters
    ----------
    A : array_like
        A NumPy 2D array/matrix or a SciPy sparse matrix. This is the adjacency
        matrix of the graph.

    Returns
    -------
    AT : 2-D array
        the max-min, or ultra-metric transitive closure of A. This is also equal
        to the all-pairs bottleneck paths. Zero entries correspond to
        disconnected pairs, i.e. null capacities.

    Notes
    ----- 
    For each source, and for each target in the successors of the source, this
    function performs a depth-first search of target, propagating the minimum
    weight along the DFS branch, and pruning whenever a node that does not have
    `target` among its successors is entered. Whenever the target node is
    reached, the minimum weight found along the path is added to a list of
    minima. When the DFS search tree is exhausted, the maximum weight is
    extract. The successors are computed using `closure_cycles`.
    '''
    AT = np.zeros(A.shape)
    for row, col, weight in _maxmin_closure_cycles(A):
        AT[row, col] = weight
    return AT

def _maxmin_closure_cycles(A):
    '''
    Maxmin transitive closure. Returns an iterator over COO tuples.

    NOTE: The frontend `maxmin_closure_cycles` constructs a 2-D array out of it.
    '''
    def _succ(node):
        r = root[node]
        s = succ[r] # only SCC roots
        return np.where(reduce(_ndor, map(_rooteq, s)))[0]
    maxval = np.inf
    A = sp.lil_matrix(A)
    root, succ = closure_cycles(A)
    _ndor = np.ndarray.__or__ # element-wise or (x|y)
    _rooteq = root.__eq__ 
    for source in xrange(A.shape[0]):
        for target in _succ(source):
            explored = defaultdict(bool)
            dfs_stack = [(source, maxval)]
            weights = [] # weight of each branch
            while dfs_stack:
                node, min_weight = dfs_stack[-1]
                if node == target: # append minimum of path
                    if min_weight != maxval:
                        weights.append(min_weight)
                if target not in _succ(node): # prune
                    dfs_stack.pop()
                    continue
                backtracking = True
                for neighbor in A.rows[node]:
                    if explored[(node, neighbor)]:
                        continue
                    backtracking = False
                    explored[(node, neighbor)] = True
                    w = float(A[node, neighbor]) # value copy
                    if w < min_weight:
                        dfs_stack.append((neighbor, w))
                    else:
                        dfs_stack.append((neighbor, min_weight))
                    break # break to visit neighbor
                # node has no neighbors, or all outgoing edges already explored
                if backtracking: 
                    dfs_stack.pop()
            yield (source, target, max(weights))


# Transitive closure for cyclical directed graphs. Recursive implementation.

_dfs_order = 0 # this must be a module-level global

def closure_cycles_recursive(adj):
    '''
    Transitive closure for directed graphs with cycles. Original recursive
    implementation. See `closure_cycles` for more details.
    
    Note
    ----
    This implementation is not suited for large graphs, as it will likely reach
    the maximum recursion depth and throw a RuntimeError. 
    '''
    global _dfs_order
    _dfs_order = 0
    # for min comparison
    def _order(node):
        return order[node]
    # dfs visiting function
    def visit(node):
        global _dfs_order
        visited[node] = True
        order[node] = _dfs_order
        _dfs_order += 1
        root[node] = node
        for neigh_node in adj.rows[node]:
            if not visited[neigh_node]:
                visit(neigh_node)
            if not in_scc[root[neigh_node]]:
                root[node] = min(root[node], root[neigh_node], key=_order)
            else:
                local_roots[node].update((root[neigh_node],))
        tmp = set()
        for cand_root in local_roots[node]:
            tmp.update(set((cand_root,)).union(succ[cand_root]))
        succ[root[node]].update(tmp) 
        if root[node] == node:
            succ[node].update((node,))
            if len(stack) and order[stack[-1]] >= order[node]:
                while True:
                    comp_node = stack.pop()
                    in_scc[comp_node] = True
                    if comp_node != node:
                        succ[node].update(succ[comp_node])
                        succ[comp_node] = succ[node] # ref copy only
                    if len(stack) == 0 or order[stack[-1]] < order[node]:
                        break 
            else:
                in_scc[node] = True
        else:
            if root[node] not in stack:
                stack.append(root[node])
    # main function
    adj = sp.lil_matrix(adj)
    order = array('i', (0 for i in xrange(adj.shape[0])))
    root = array('i', (-1 for i in xrange(adj.shape[0])))
    stack = []
    in_scc = defaultdict(bool) # default value : False
    visited = defaultdict(bool)
    local_roots = defaultdict(set)
    succ = defaultdict(set)
    for node in xrange(adj.shape[0]):
        if not visited[node]:
            visit(node)
    return np.frombuffer(root, dtype=np.int32), dict(succ)

# Transitive closure for directed cyclical graphs. Iterative implementation.

# XXX: does not always compute the correct SCC set!

def closure_cycles(adj):
    '''
    Transitive closure for directed graphs with cycles. Iterative implementation
    based on the algorithm by Nuutila et Soisalon-Soininen [1].

    Arguments
    ---------
    adj : array_like
        the adjacency matrix of the graph

    Returns
    -------
    root : instance of `array.array`
        for each node, the root of the SCC of that node.
    succ : dict of lists
        for each root of an SCC, the list of strongly connected components that
        can be reached by that node. Each SCC is identified by its root node.

    Note
    ----
    In the original paper [1], the algorithm builds the full set of successors
    of each SCC. Here, the function returns only the set of roots of strongly
    connected successor components.

    References
    ----------
    .. [1] On finding the strongly connected components in a directed graph.
       E. Nuutila and E. Soisalon-Soininen.
       Information Processing Letters 49(1): 9-14, (1994)
    '''
    def _order(node):
        return dfs_order[node]
    dfs_counter = 0
    dfs_stack = []
    adj = sp.lil_matrix(adj)
    dfs_order = array('i', (-1 for i in xrange(adj.shape[0])))
    root = array('i', (-1 for i in xrange(adj.shape[0])))
    scc_stack = []
    in_scc = defaultdict(bool)
    local_roots = defaultdict(set)
    succ = defaultdict(set)
    for source in xrange(adj.shape[0]):
        if dfs_order[source] < 0:
            # start a new depth-first traversal from source
            dfs_stack = [source]
        else:
            # we have already traversed this source
            continue
        while dfs_stack:
            # the top of dfs_stack holds the current node
            node = dfs_stack[-1]
            if dfs_order[node] < 0:
                # we are visiting a new node.
                dfs_order[node] = dfs_counter
                dfs_counter += 1
                root[node] = node
            # go through all the neighbors, until we find a non-visited node. If
            # we find one, put it on top of the stack and break to next iteration.
            # If no neighbors exist, or all neighbors have been already visited,
            # backtrack.
            backtracking = True
            for neighbor_node in adj.rows[node]:
                if dfs_order[neighbor_node] < 0:
                    dfs_stack.append(neighbor_node)
                    backtracking = False
                    break # will visit neighbor_node at next iteration
                if not in_scc[root[neighbor_node]]:
                    root[node] = min(root[node], root[neighbor_node], key=_order)
                else:
                    local_roots[node].update((root[neighbor_node],))
            if backtracking:
                # all neighbors have been visited at this point
                tmp = set()
                for cand_root in local_roots[node]:
                    tmp.update(set((cand_root,)).union(succ[cand_root]))
                succ[root[node]].update(tmp)
                if root[node] == node:
                    succ[node].update((node,))
                    if len(scc_stack) and dfs_order[scc_stack[-1]] >= dfs_order[node]:
                        while True:
                            comp_node = scc_stack.pop()
                            in_scc[comp_node] = True
                            if comp_node != node:
                                succ[node].update(succ[comp_node])
                                succ[comp_node] = succ[node] # ref copy only
                            if len(scc_stack) == 0 or\
                                    dfs_order[scc_stack[-1]] < dfs_order[node]:
                                break
                    else:
                        in_scc[node] = True
                else:
                    if root[node] not in scc_stack:
                        scc_stack.append(root[node])
                # clear the current node from the top of the DFS stack.
                dfs_stack.pop()
    return np.frombuffer(root, dtype=np.int32), dict(succ)

def productclosure(A, parallel=False, maxiter=1000, quiet=False, dumpiter=None,
        **kwrds):
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

# Frontend function. 

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

# Run a quick test if ran like a script

if __name__ == '__main__':
    from time import time
    np.random.seed(10)
    B = sp.rand(5, 5, .2, 'csr')

    print 'Testing maxmin on a matrix of shape %s with nnz = %d:' % (B.shape,
            B.getnnz())

    tic = time()
    B1 = pmaxmin(B, 2, 2)
    toc = time()
    print '* parallel version executed in %.2e seconds' % (toc - tic)

    tic = time()
    B2 = maxmin(B)
    toc = time()
    print '* serial version executed in %.2e seconds' % (toc - tic)

    print 'Testing maxmin product closure on a matrix of shape %s with'\
            ' nnz = %d:' % (B.shape, B.getnnz())

    tic = time()
    Cl1 = productclosure(B, splits=2, nprocs=2, maxiter=10, parallel=True)
    toc = time()
    print '* parallel version executed in %.2e seconds' % (toc - tic)

    tic = time()
    Cl2 = productclosure(B, maxiter=10)
    toc = time()
    print '* serial version executed in %.2e seconds' % (toc - tic)

    assert _allclose_csr(Cl1, Cl2)
