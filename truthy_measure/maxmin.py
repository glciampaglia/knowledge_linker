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

### Maxmin closure
* mmclosure_matmul
    Max-min transitive closure via matrix multiplication, user function. This
    function uses the max-min multiplication function `maxmin` resp. `pmaxmin` to
    compute the transitive closure sequentially or in parallel, respectively.
* mmclosure_dfs
    Max-min transitive closure, based on depth-first traversal with pruning.
    For each source, only targets that are reachable are searched. Pruning is
    performed using the information on the successors of a node, computed with
    `closure_recursive`.
* mmclosure_dfsrec
    Same as `mmclosure_dfs`, except that depth-first traversal
    is implemented iteratively.
* itermmclosure_dfs
* itermmclosure_dfsrec
    These are the actual function that computes the closure; they both return an
    iterator over all node pairs with non-zero weight.

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

### Transitive closure
* closure_recursive
    Transitive closure based on the algorithm by Nuutila et Soisalon-Soininen
    (1994). Compute (strongly) connected components and successors sets.
    Recursive implementation. For larger graphs, use the iterative version
    `closure`. The successors matrix may be stored to disk.
* closure
    Iterative version of the above.

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
from multiprocessing import Pool, Array, cpu_count
from ctypes import c_int, c_double
from contextlib import closing
import warnings
from datetime import datetime
from itertools import izip, product
from collections import defaultdict
from array import array
from tables import BoolAtom, Filters, open_file
from tempfile import NamedTemporaryFile
from progressbar import ProgressBar, Bar, AdaptiveETA, Percentage
from operator import itemgetter

from .utils import coo_dtype, Cache
from .cmaxmin import c_maximum_csr # see below for other imports

# for closure/closure_recursive
CHUNKSIZE = 1000

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

# max-min transitive closure based on DFS

class ReachablePairsIter(object):
    '''
    Instances of this class are iterator that, for each source, yield (source,
    target) pairs where target is reachable from source according to the succ
    matrix.
    '''
    def __init__(self, sources, roots, succ):
        '''
        Parameters
        ----------
        sources : sequence of ints
            The sources
        roots : array_like
            A 1D array of root labels (see `closure`)
        succ : array_like
            A 2D bool matrix representing the "successor" relation
        '''
        try:
            len(sources)
        except TypeError:
            raise ValueError("sources must be a sequence")
        self.sources = sources
        self.roots = roots
        self.succ = succ
        self._len = sum([succ[roots[i],:].sum() for i in sources])
    def __len__(self):
        return self._len
    def __iter__(self):
        for s in self.sources:
            for t in xrange(len(self.roots)):
                if self.succ[self.roots[s], self.roots[t]]:
                    yield s, t

class ProductIter(object):
    '''
    Wrapper around `itertools.product` with len() method.
    '''
    def __init__(self, *sequences):
        '''
        Parameters
        ----------

        *sequences : sequence of sequences

            Each sequeunce must support len().
        '''
        self.sequences = sequences
        self._len = reduce(int.__mul__, map(len, self.sequences))
    def __iter__(self):
        return product(*self.sequences)
    def __len__(self):
        return self._len

def _dfs_items(sources, targets, n, succ, roots, progress):
    '''
    Produces input (source, target) pairs for DFS search and related progress
    bar object.
    '''
    if succ is not None:
        if roots is None:
            roots = np.arange((n,), dtype=np.int32)
        else:
            roots = np.ravel(roots)
        items = ReachablePairsIter(sources, roots, succ)
    else:
        if targets is not None:
            items = zip(sources, targets)
        else:
            items = ProductIter(sources, xrange(n))
    if progress:
        widgets = ['[Maxmin closure] ', AdaptiveETA(), Bar(), Percentage()]
        pbar = ProgressBar(widgets=widgets)
        items = pbar(items)
    return items

def itermmclosure_dfs(a, sources, targets=None, succ=None, roots=None,
        progress=False, cachesize=1000):
    '''
    Max-min transitive closure (iterative). Performs a DFS search from source to
    target, caching results as they are computed.

    Parameters
    ----------
    a : array_like
        an (n,n) adjacency matrix
    sources : sequence of int
        the source nodes
    targets : iterable of int
        the target nodes; optional. If not provided, for each source will test
        all n possible targets.
    succ : array_like
        optional; an (n,n) boolean array (or matrix) indicating successors
        relation succ[i, j] == True iff j is a successor of i.
    roots : array_like
        optional; an (n,) array of labels (roots) of strongly connected
        components. If passed, then only the rows and columns of `succ`
        corresponding to the elements in `roots` are consulted.
    progress : bool
        optional; if True, a progress bar will be shown.

    Returns
    -------
    An iterator over source, target, max-min weight. Only pairs with non-zero
    weight are returned.

    Notes
    -----
    The algorithm traverses the graph in depth-first order, propagating the
    minimum so far. The generated path does not contain duplicates (with the
    possible exception of source and target, if they are the same). When
    backtracking from a node, the maximum of all the minima coming from the
    neighbors is chosen, and propagated back.
    '''
    n = a.shape[0]
    a = sp.lil_matrix(a)
    items = _dfs_items(sources, targets, n, succ, roots, progress)
    get0 = itemgetter(0) # e.g. lambda k : k[0]
    usecache = cachesize > 0
    if usecache:
        cache = Cache(cachesize)
    for s, t in items:
        max_weight = -1 # the max-min over all paths so far
        max_path = None # the nodes in the path of max_weight
        path = set() # nodes visited along the current path
        # local context: current node, target node, minimum along the path, last
        # neighbor explored at current node. Exploration starts from the source
        # with inf and no neighbor explored (-1)
        dfs_stack = [(s, t, np.inf, -1)]
        path.add(s)
        while dfs_stack:
            node, target, min_so_far, last_neighbor = dfs_stack[-1]
            backtracking = True
            for neighbor in a.rows[node]:
                # skip all neighbors already explored to pick up from where we
                # left off
                if neighbor <= last_neighbor:
                    continue
                dfs_stack[-1] = (node, target, min_so_far, neighbor)
                # prune if target is not reachable through neighbor
                if succ is not None:
                    if not succ[roots[neighbor], roots[target]]:
                        continue
                w = float(a[node, neighbor]) # copy value
                if neighbor == target:
                    path.add(target) # close the path
                    max_weight, max_path = max(
                            (max_weight, max_path),
                            (min(min_so_far, w), path), key=get0)
                    continue
                if usecache and (neighbor, target) in cache:
                    cached_weight, cached_path = cache[(neighbor, target)]
                    if not cached_path.intersection(path):
                        max_weight, max_path = max((max_weight, max_path),\
                                (min(min_so_far, w, cached_weight), \
                                path.union(cached_path)), key=get0)
                        continue
                if neighbor not in path:
                    dfs_stack.append((neighbor, target, min(w, min_so_far), -1))
                    backtracking = False
                    path.add(node)
                    break
            if backtracking:
                path.discard(node)
                dfs_stack.pop()
        if max_weight > -1:
            yield s, t, max_weight
            if usecache:
                cache[(s,t)] = (max_weight, max_path)

def mmclosure_dfs(a, succ=None, roots=None, progress=False, cachesize=1000):
    '''
    Max-min closure by simple DFS traversals. Returns a sparse matrix.
    '''
    A = sp.lil_matrix(a.shape)
    for i,j,w in itermmclosure_dfs(a, xrange(a.shape[0]), succ=succ,
            roots=roots, progress=progress, cachesize=cachesize):
        A[i,j] = w
    return A

def itermmclosure_dfsrec(a, sources, targets=None, succ=None, roots=None,
        progress=False):
    '''
    Recursive version of `itermmclosure_simplesearch`. Not suitable for large
    graphs.
    '''
    def search(node, target, min_so_far):
        if node != target:
            visited.add(node)
        if node == target and min_so_far < np.inf:
            return min_so_far
        max_weight = -1
        for neighbor in a.rows[node]:
            if succ is not None and not succ[roots[neighbor], roots[target]]:
                continue # prune
            if (node, neighbor) in explored or neighbor in visited:
                # to avoid getting stuck inside cycles
                continue
            explored.add((node, neighbor))
            w = float(a[node, neighbor]) # copy value
            if w < min_so_far:
                m = search(neighbor, target, w)
            else:
                m = search(neighbor, target, min_so_far)
            if m is not None and m > max_weight:
                max_weight = m
        visited.discard(node)
        if max_weight > -1:
            if max_weight < min_so_far:
                return max_weight
            else:
                return min_so_far
    n = a.shape[0]
    a = sp.lil_matrix(a)
    items = _dfs_items(sources, targets, n, succ, roots, progress)
    for s, t in items:
        explored = set() # traversed edges
        visited = set() # nodes traversed along the path
        m = search(s, t, np.inf) # None if t is not reachable from s
        if m is not None:
            yield s, t, m

def mmclosure_dfsrec(a):
    '''
    Max-min closure by simple recursive DFS traversals. Returns a sparse matrix.
    '''
    A = sp.lil_matrix(a.shape)
    for i,j,w in itermmclosure_dfsrec(a, xrange(a.shape[0])):
        A[i,j] = w
    return A

# Transitive closure for cyclical directed graphs. Recursive implementation.

def _mk_succ(outpath, shape, ondisk=False):
    '''
    Creates the chunked array that will hold the successors
    '''
    # create CArray on disk
    if outpath is None:
        outfile = NamedTemporaryFile(suffix='.h5', delete=True)
        outpath = outfile.name
    if ondisk:
        h5f = open_file(outpath, 'w')
    else:
        # this will create an in-memory file, which will be synced to disk when
        # closed
        h5f = open_file(outpath, 'w', driver="H5FD_CORE")
    atom = BoolAtom()
    filters = Filters(complevel=5, complib='zlib')
    succ = h5f.create_carray(h5f.root, 'succ', atom, shape,
            filters=filters, chunkshape=(1, CHUNKSIZE))
    return succ

def _mk_sources(sources, n, progress):
    '''
    Creates the sources sequence for the transitive closure functions
    '''
    if sources is None:
        sources = xrange(n) # explore the whole graph
    pbar = None
    if progress:
        widgets = ['[Transitive closure] ', AdaptiveETA(), Bar(), Percentage()]
        pbar = ProgressBar(maxval=2 * n, widgets=widgets)
        pbar.start()
    return sources, pbar

_dfs_order = 0 # this must be a module-level global

def closure_recursive(adj, sources=None, ondisk=False, outpath=None,
        progress=False):
    '''
    Transitive closure for directed graphs with cycles. Original recursive
    implementation. See `closure` for more details.

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
        if progress:
            pbar.update(2 * _dfs_order - 1)
        for neigh_node in adj.rows[node]:
            if not visited[neigh_node]:
                visit(neigh_node)
            if not in_scc[root[neigh_node]]:
                root[node] = min(root[node], root[neigh_node], key=_order)
            else:
                local_roots[node].update((root[neigh_node],))
        for cand_root in local_roots[node]:
            succ[root[node], :] += succ[cand_root, :]
            succ[root[node], cand_root] = True
        del local_roots[node]
        if root[node] == node:
            succ[node, node] = True
            if len(stack) and order[stack[-1]] >= order[node]:
                while True:
                    comp_node = stack.pop()
                    in_stack[comp_node] = False
                    in_scc[comp_node] = True
                    if comp_node != node:
                        succ[node, :] += succ[comp_node, :]
#                         succ[comp_node, :] = False
                    if len(stack) == 0 or order[stack[-1]] < order[node]:
                        break
            else:
                in_scc[node] = True
        else:
            if not in_stack[root[node]]:
                stack.append(root[node])
                in_stack[root[node]] = True
            succ[root[node], node] = True
        if progress:
            pbar.update(2 * _dfs_order)
    # main function
    adj = sp.lil_matrix(adj)
    order = array('i', (0 for i in xrange(adj.shape[0])))
    root = array('i', (-1 for i in xrange(adj.shape[0])))
    stack = []
    in_stack = array('B', (0 for i in xrange(adj.shape[0])))
    in_scc = defaultdict(bool) # default value : False
    visited = defaultdict(bool)
    local_roots = defaultdict(set)
    succ = _mk_succ(outpath, adj.shape, ondisk)
    sources, progress = _mk_sources(sources, adj.shape[0], progress)
    for node in sources:
        if not visited[node]:
            visit(node)
    if progress:
        pbar.finish()
    return np.frombuffer(root, dtype=np.int32), succ, outpath

# Transitive closure for directed cyclical graphs. Iterative implementation.

def closure(adj, sources=None, ondisk=False, outpath=None, progress=False):
    '''
    Transitive closure for directed graphs with cycles. Iterative implementation
    based on the algorithm by Nuutila et Soisalon-Soininen [1].

    Arguments
    ---------
    adj : array_like
        the adjacency matrix of the graph
    source : sequence or array_like
        a sequence of source nodes. Roots and successors will be computed only
        starting from these nodes.
    ondisk : bool
        if True, will store the successors matrix to disk.
    outpath : string
        optional; specify the path to the output file used for storing the
        successors matrix on disk.
    progress : bool
        if True, prints information about progress of the computation.

    Returns
    -------
    root : instance of `array.array`
        for each node, the root of the SCC of that node.
    succ : dict of lists
        for each root of an SCC, the list of SCCs (other than itself) that
        can be reached by that node. Each SCC is identified by its root node.
    outpath : string
        optional; if outpath was not passed, the path to an open temporary file.
        The file will be delete upon closing it.

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
    in_stack = array('B', (0 for i in xrange(adj.shape[0])))
    scc_stack = []
    in_scc = defaultdict(bool)
    local_roots = defaultdict(set)
    succ = _mk_succ(outpath, adj.shape, ondisk)
    sources, pbar = _mk_sources(sources, adj.shape[0], progress)
    for source in sources:
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
                if progress:
                    pbar.update(2 * dfs_counter - 1)
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
                    succ[root[node], :] += succ[cand_root, :]
                    succ[root[node], cand_root] = True
                del local_roots[node]
                if root[node] == node:
                    succ[node, node] = True
                    if len(scc_stack) and \
                            dfs_order[scc_stack[-1]] >= dfs_order[node]:
                        while True:
                            comp_node = scc_stack.pop()
                            in_stack[comp_node] = False
                            in_scc[comp_node] = True
                            if comp_node != node:
                                succ[node, :] += succ[comp_node, :]
#                                 succ[comp_node, :] = False
                            if len(scc_stack) == 0 or\
                                    dfs_order[scc_stack[-1]] < dfs_order[node]:
                                break
                    else:
                        in_scc[node] = True
                else:
                    if not in_stack[root[node]]:
                        scc_stack.append(root[node])
                        in_stack[root[node]] = True
                    succ[root[node], node] = True
                # clear the current node from the top of the DFS stack.
                dfs_stack.pop()
                if progress:
                    pbar.update(2 * dfs_counter)
    if progress:
        pbar.finish()
    return np.frombuffer(root, dtype=np.int32), succ, outpath

def mmclosure_matmul(A, parallel=False, maxiter=1000, quiet=False,
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
    Cl1 = mmclosure_matmul(B, splits=2, nprocs=2, maxiter=10,
            parallel=True)
    toc = time()
    print '* parallel version executed in %.2e seconds' % (toc - tic)

    tic = time()
    Cl2 = mmclosure_matmul(B, maxiter=10)
    toc = time()
    print '* serial version executed in %.2e seconds' % (toc - tic)

    assert _allclose_csr(Cl1, Cl2)
