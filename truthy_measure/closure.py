'''
Dijkstra-based path finding algorithm for computing the metric closure on either
similarity/proximity graphs or distance graphs.

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
epclosureap

:   closure computed only on intermediate nodes, with the additional constraint
    that direct neighbors have similarity 1.

'''

import os
import sys
import numpy as np
import scipy.sparse as sp
from ctypes import c_int, c_double
from contextlib import closing
from datetime import datetime
from heapq import heappush, heappop, heapify
from multiprocessing import Pool, Array, cpu_count, current_process

now = datetime.now

# package imports
from ._closure import cclosuress, cclosure
from .maxmin import _init_worker

def closure(A, source, target):
    '''
    Source-target metric closure via Dijkstra path finding. 
    
    Note that this function is pure Python and thus very slow. Use the
    Cythonized version `cclosure`, which accepts only CSR matrices.

    This always returns the paths.
    '''
    cap, paths = closuress(A, source)
    return cap[target], paths[target]

def closuress(A, source):
    '''
    Single source metric closure via Dijkstra path finding. 
    
    Note that this function is pure Python and thus very slow. Use the
    Cythonized version `cclosuress`, which accepts only CSR matrices.

    This always returns the paths.
    '''
    A = sp.csr_matrix(A)
    N = A.shape[0]
    certain = np.zeros(N, dtype=np.bool)
    items = {} # handles to the items inserted in the queue
    Q = [] # heap queue
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
        cap, node, pred = node_item
        cap = - cap
        certain[node] = True
        neighbors = A.getrow(node).indices
        for neighbor in neighbors:
            if not certain[neighbor]:
                neighbor_item = items[neighbor]
                neigh_cap = - neighbor_item[0]
                w = A[node, neighbor]
                d = min(w, cap)
                if d > neigh_cap:
                    neighbor_item[0] = - d
                    neighbor_item[2] = node
                    heapify(Q)
    # generate paths
    bott_caps = []
    paths = []
    for node in xrange(N):
        item = items[node]
        if item[2] == -1: # disconnected node
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

_nprocs = None
_logpath = None
_dirtree = None

def _closure_worker(n):
    global _A, _dirtree, _logpath, _logf, _nprocs, digits_procs
    worker_id, = current_process()._identity
    logpath = _logpath.format(proc=worker_id, width=digits_procs)
    outpath = _dirtree.getleaf(n)
    with \
            closing(open(outpath, 'w')) as outf,\
            closing(open(logpath, 'a', 1)) as logf:
        dists, paths = cclosuress(_A, n, outf)
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

def _init_worker(indptr, indices, data, shape):
    '''
    See `pmaxmin`. This is the worker initialization function.
    '''
    global _indptr, _indices, _data, _A
    _indptr = np.frombuffer(indptr.get_obj(), dtype=np.int32)
    _indices = np.frombuffer(indices.get_obj(), dtype=np.int32)
    _data = np.frombuffer(data.get_obj())
    _A = sp.csr_matrix((_data, _indices.astype('int32'), _indptr), shape)

def closureap(A, dirtree, start=None, offset=None, nprocs=None):
    '''
    All-pairs metric closure via path-finding. Computes the closure of a graph
    represented by adjacency matrix A and saves the results for each row in
    separate files organized in a directory tree (see `truthy_measure.dirtree`). 

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
        pool.map(_closure_worker, xrange(fromi, toi))
    pool.join()
    print '{}: done'.format(now())

def epclosure(A, source, target, B=None, closurefunc=None, **kwargs):
    '''
    Source target "epistemic" closure. Python implementation.

    See `epclosuress` for parameters.

    Note: always returns paths.
    '''
    cap, paths = epclosuress(A, source, B=B, closurefunc=closurefunc, **kwargs)
    return cap[target], paths[target]

def epclosuress(A, source, B=None, closurefunc=None, **kwargs):
    '''
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

    closurefunc : func
        Optional; an alternative closure function. By default,
        `truthy_measure.closure.closuress` will be used. Additional keyword
        arguments are passed to closurefunc.

    Note: always returns paths.
    '''
    # ensure A is CSR
    A = sp.csr_matrix(A)
    if B is None:
        B = A.tocsc()
    else:
        # ensure B is CSC
        B = sp.csc_matrix(B)
    N = A.shape[0]
    if closurefunc:
        _caps, _paths = closurefunc(A, source, **kwargs)
    else:
        _caps, _paths = closuress(A, source)
    _caps = np.asarray(_caps)
    caps = np.empty(N)
    paths = []
    s_neighbors = set(A.getrow(source).indices)
    s_reachables = set(np.where(_caps)[0])
    for target in xrange(N):
        if target in s_neighbors or target == source:
            caps[target] = 1.0
            paths.append(np.empty(0))
        elif _caps[target] > 0.0:
            # target must have at least one neighbor that is also reachable from
            # source. In case the graph is directed, we take the in-neighbors.
            t_neighbors = set(B.getcol(target).indices)
            t_neighbors.intersection_update(s_reachables)
            t_neighbors = np.asarray(sorted(t_neighbors))
            t_neighbors_cap = _caps[t_neighbors]
            imax = t_neighbors[np.argmax(t_neighbors_cap)]
            caps[target] = _caps[imax]
            paths.append(_paths[imax])
        else:
            # target is not reachable from source
            caps[target] = 0.0
            paths.append(np.empty(0))
    return caps, paths

def epclosureap(A, source):
    pass
