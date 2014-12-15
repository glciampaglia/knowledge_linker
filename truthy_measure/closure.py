""" Dijkstra-based path finding algorithms for epistemic closures.
Dijkstra-based path finding algorithm for computing the transitive closure on
similarity/proximity graphs.

Nomenclature
============

closure

:   for source, target problems

closuress

:   for single-source problems

closureap

:   for all-pairs problems; uses multiprocessing.

epclosure,
epclosuress,

:   closure computed only on intermediate nodes, with the additional constraint
    that direct neighbors have similarity 1.

"""

import sys
import numpy as np
import scipy.sparse as sp
from ctypes import c_int, c_double
from contextlib import closing
from datetime import datetime
from heapq import heappush, heappop, heapify
from multiprocessing import Pool, Array, cpu_count, current_process
from operator import itemgetter

now = datetime.now

# package imports
from ._closure import cclosuress, cclosure


def dombit1(a, b):
    """ Dombi T-conorm with lambda = 1.

    Returns a float between 0 and 1.

    >>> dombit1(0, 0)
    0.0
    >>> dombit1(1, 1)
    1.0
    >>> dombit1(0.5, 0.5)
    0.3333333333333333

    """
    if (a < 0) or (a > 1) or (b < 0) or (b > 1):
        raise ValueError('not in unit interval: {}, {}'.format(a, b))
    if a == b == 0:
        return 0.0
    else:
        return float(a * b) / (a + b - a * b)

# This dictionary maps metric kinds to pairs (conjf, disjf) where:
#
# 1. conjf takes two inputs, a and b, and a `key` callable argument and return
#    one of the two arguments based on the values key(a) and key(b).
# 2. disjf takes two float inputs and returns a float.

_metricfuncs = {
    'ultrametric': (max, min),
    'metric': (max, dombit1)
}


def closure(A, source, target, kind='ultrametric'):
    """ Source-target metric closure via Dijkstra path finding.

    Note that this function is pure Python and thus very slow. Use the
    Cythonized version `cclosure`, which accepts only CSR matrices.

    This always returns the paths.

    """
    cap, paths = closuress(A, source, kind=kind)
    return cap[target], paths[target]


def closuress(A, source, kind='ultrametric'):
    """ Single source metric closure via Dijkstra path finding.

    Parameters
    ----------

    A : array_like
        NxN adjacency matrix

    source : int
        the source node

    kind : str
        the kind of closure to compute; either 'metric' or 'ultrametric'
        (default).

    Note that this function is pure Python and thus very slow. Use the
    Cythonized version `cclosuress`, which accepts only CSR matrices.

    This always returns the paths.

    >>> A = np.asarray([
    ...     [0., .1, .0],
    ...     [0., 0., .2],
    ...     [0., 0., 0.]])
    >>> c, _ = closuress(A, 0)
    >>> c
    [1.0, 0.10000000000000001, 0.10000000000000001]

    """
    keyf = itemgetter(0)
    if kind not in _metricfuncs:
        raise ValueError('unknown metric: {}'.format(kind))
    disjf, conjf = _metricfuncs[kind]
    A = sp.csr_matrix(A)
    N = A.shape[0]
    certain = np.zeros(N, dtype=np.bool)
    items = {}  # handles to the items inserted in the queue
    Q = []  # heap queue
    # populate the queue
    for node in xrange(N):
        if node == source:
            cap = 1.
            item = [- cap, node, node]
        else:
            cap = 0.0
            item = [- cap, node, -1]
        items[node] = item
        heappush(Q, item)
    # compute spanning tree
    while len(Q):
        node_item = heappop(Q)
        cap, node, _ = node_item
        cap = - cap
        certain[node] = True
        neighbors = A.getrow(node).indices
        for neighbor in neighbors:
            if not certain[neighbor]:
                neighbor_item = items[neighbor]
                neigh_curr_cap = - neighbor_item[0]
                neigh_curr_pred = neighbor_item[2]
                current = (neigh_curr_cap, neigh_curr_pred)
                neigh_cand_cap = conjf(A[node, neighbor], cap)
                neigh_cand_pred = node
                candidate = (neigh_cand_cap, neigh_cand_pred)
                try:
                    new_cap, new_pred = disjf(candidate, current, key=keyf)
                except TypeError:
                    raise ValueError('disjunction must accept `key` param.')
                if new_cap != neigh_curr_cap:
                    neighbor_item[0] = - new_cap
                    neighbor_item[2] = new_pred
                    heapify(Q)
    # generate paths based on the spanning tree
    bott_caps = []
    paths = []
    for node in xrange(N):
        item = items[node]
        print item
        if item[2] == -1:  # disconnected node
            bott_caps.append(0.0)
            paths.append(np.empty(0, dtype=np.int))
        else:
            bott_caps.append(- item[0])
            path = []
            i = node
            while i != source:
                path.insert(0, i)
                i = items[i][2]
            path.insert(0, source)
            paths.append(np.asarray(path))
    return bott_caps, paths

maxchunksize = 100000
max_tasks_per_worker = 500
log_out = 'closure-{proc:0{width}d}.log'
log_out_start = 'closure_{start:0{width1}d}-{{proc:0{{width}}d}}.log'
logline = "{now}: worker-{proc:0{width}d}: source {source} completed."

_A = None
_nprocs = None
_logpath = None
_dirtree = None
_kind = None
digits_procs = 2


def _closure_worker(n):
    global _A, _dirtree, _logpath, _logf, _nprocs, _kind, digits_procs
    worker_id, = current_process()._identity
    logpath = _logpath.format(proc=worker_id, width=digits_procs)
    outpath = _dirtree.getleaf(n)
    with \
            closing(open(outpath, 'w')) as outf, \
            closing(open(logpath, 'a', 1)) as logf:
        cclosuress(_A, n, outf, kind=_kind)
        logf.write(logline.format(now=now(), source=n, proc=worker_id,
                                  width=digits_procs) + '\n')


def _init_worker_dirtree(kind, nprocs, logpath, dirtree, indptr, indices, data,
                         shape):
    global _dirtree, _logpath, _nprocs, digits_procs, digits_rows
    _nprocs = nprocs
    digits_procs = int(np.ceil(np.log10(_nprocs)))
    _logpath = logpath
    _dirtree = dirtree
    _init_worker(kind, indptr, indices, data, shape)


def _init_worker(kind, indptr, indices, data, shape):
    """ See `pmaxmin`. This is the worker initialization function.  """
    global _kind, _indptr, _indices, _data, _A
    _kind = kind
    _indptr = np.frombuffer(indptr.get_obj(), dtype=np.int32)
    _indices = np.frombuffer(indices.get_obj(), dtype=np.int32)
    _data = np.frombuffer(data.get_obj())
    _A = sp.csr_matrix((_data, _indices.astype('int32'), _indptr), shape)


def _fromto(start, offset, N):
    """
    Translate start and offset to slice indices.

    If offset overflows the axis lenght N, it is capped to N.
    """
    if start is None:
        fromi = 0
        toi = N
    else:
        assert offset is not None
        assert offset >= 0
        assert start >= 0
        fromi = start
        toi = min(start + offset, N)
    return fromi, toi


def _nprocs(nprocs):
    if nprocs is None:
        # by default use 90% of available processors or 2, whichever the
        # largest.
        return max(int(0.9 * cpu_count()), 2)
    else:
        return nprocs


def closureap(A, dirtree, start=None, offset=None, nprocs=None,
              kind='ultrametric'):
    """
    All-pairs metric closure via path-finding. Computes the closure of a graph
    represented by adjacency matrix A and saves the results for each row in
    separate files organized in a directory tree (see
    `truthy_measure.dirtree`).

    If `start` and `offset` parameters are passed, then only rows between
    `start` and `start + offset` are computed.

    Parameters
    ----------
    A : array_like
        NxN adjacency matrix, will be converted to CSR format.

    dirtree : a `truthy_measure.utils.DirTree` instance
        The directory tree object used to generate the directory tree in which
        the results are stored.

    start : int, offset : int
        optional; compute only rows from start up to offset.

    nprocs : int
        optional; number of processes to spawn. Default is 90% of available
        CPUs/cores.

    kind : str
        the kind of closure, either 'ultrametric' (default) or 'metric'.

    """
    N = A.shape[0]
    digits = int(np.ceil(np.log10(N)))
    fromi, toi = _fromto(start, offset, N)
    nprocs = _nprocs(nprocs)
    # allocate array to be passed as shared memory
    A = sp.csr_matrix(A)
    indptr = Array(c_int, A.indptr)
    indices = Array(c_int, A.indices)
    data = Array(c_double, A.data)
    if start is None:
        logpath = log_out
    else:
        logpath = log_out_start.format(start=start, width1=digits)
    initargs = (kind, nprocs, logpath, dirtree, indptr, indices, data, A.shape)
    print '{}: launching pool of {} workers.'.format(now(), nprocs)
    pool = Pool(processes=nprocs, initializer=_init_worker_dirtree,
                initargs=initargs, maxtasksperchild=max_tasks_per_worker)
    with closing(pool):
        pool.map(_closure_worker, xrange(fromi, toi))
    pool.join()
    print '{}: done'.format(now())


def epclosure(A, source, target, B=None, retpath=False, kind='ultrametric'):
    """
    Source target "epistemic" closure. Python implementation.

    See `epclosuress` for parameters.

    """
    cap, paths = epclosuress(A, source, B=B, retpaths=retpath, kind=kind)
    if retpath:
        return cap[target], paths[target]
    else:
        return cap[target], None


def epclosuress(A, source, B=None, kind='ultrametric', retpaths=False):
    """
    Single-source "epistemic" closure. Python implementation.

    Parameters
    ----------

    A : array_like
        Adjacency matrix. Will be converted to CSR

    source : int
        The source node

    B : array_like
        Optional; a copy of A in CSC format. Useful in loops to avoid
        converting A at every iteration.

    retpaths : bool
        if True, return paths, else, and empty list. Default: False.

    kind : str the type of closure to compute: either 'metric' or 'ultrametric'
        (default).

    """
    # ensure A is CSR
    A = sp.csr_matrix(A)
    if B is None:
        B = A.tocsc()
    else:
        # ensure B is CSC
        B = sp.csc_matrix(B)
    N = A.shape[0]
    _caps, _paths = cclosuress(A, source, kind=kind, retpaths=retpaths)
    _caps = np.asarray(_caps)
    caps = np.empty(N)
    paths = []
    s_neighbors = set(A.getrow(source).indices)
    s_reachables = set(np.where(_caps)[0])
    for target in xrange(N):
        if target in s_neighbors or target == source:
            caps[target] = 1.0
            if retpaths:
                paths.append(np.empty(0))
        elif _caps[target] > 0.0:
            # target must have at least one neighbor that is also reachable
            # from source. In case the graph is directed, we take the
            # in-neighbors.
            t_neighbors = set(B.getcol(target).indices)
            t_neighbors.intersection_update(s_reachables)
            t_neighbors = np.asarray(sorted(t_neighbors))
            t_neighbors_cap = _caps[t_neighbors]
            imax = t_neighbors[np.argmax(t_neighbors_cap)]
            caps[target] = _caps[imax]
            if retpaths:
                paths.append(np.hstack([_paths[imax], target]))
        else:
            # target is not reachable from source
            caps[target] = 0.0
            if retpaths:
                paths.append(np.empty(0))
    return caps, paths


def _backbone_worker(n):
    global _A, _kind
    d0 = np.ravel(_A[n].todense())  # original
    d1, _ = cclosuress(_A, n, kind=_kind)  # closed
    B, = np.where((d0 > 0.0) & (d0 == d1))
    return [(n, b) for b in B]


# TODO: add test function.
def backbone(A, kind='ultrametric', start=None, offset=None, nprocs=None):
    """ Compute the graph backbone.

    The graph backbone is the set of edges whose weight does not change after
    the closure operation. These edges respect the triangular inequality (kind
    = 'metric') or the maxmin inequality (kind = 'ultrametric'). And are
    therefore part of the shortest/bottleneck paths of the graph.

    Parameters
    ----------

    A : array_like
        Adjacency matrix. Will be converted to CSR

    kind : str
        the type of closure to compute: either 'metric' or 'ultrametric'
        (default).

    start : int
        Optional; only compute the closure on the submatrix starting at this
        index. Default is 0.

    offset : int
        Optional; only compute the closure on the submatrix ending at this
        offset. The default up to N, where A is an (N, N) matrix.

    nprocs : int
        Optional; distribute the computation over `nprocs` workers. Default is
        90% of the available CPUs/cores.

    Returns
    -------
    A scipy.sparse.coo_matrix.
    """
    A = sp.csr_matrix(A)
    N = A.shape[0]
    fromi, toi = _fromto(start, offset, N)
    nprocs = _nprocs(nprocs)
    indptr = Array(c_int, A.indptr)
    indices = Array(c_int, A.indices)
    data = Array(c_double, A.data)
    initargs = (kind, indptr, indices, data, A.shape)
    print '{}: launching pool of {} workers.'.format(now(), nprocs)
    pool = Pool(processes=nprocs, initializer=_init_worker,
                initargs=initargs, maxtasksperchild=max_tasks_per_worker)
    try:
        with closing(pool):
            result = pool.map_async(_backbone_worker, xrange(fromi, toi))
            while not result.ready():
                result.wait(1)
        pool.join()
        if result.successful():
            coords = result.get()
        else:
            print >> sys.stderr, "There was an error in the pool."
            sys.exit(2)  # ERROR occurred
    except KeyboardInterrupt:
        print "^C"
        pool.terminate()
        sys.exit(1)  # SIGINT received
    print '{}: done'.format(now())
    coords = np.asarray(reduce(list.__add__, coords))
    d = np.ones(len(coords))
    B = sp.coo_matrix((d, (coords[:, 0], coords[:, 1])), shape=A.shape)
    return B
