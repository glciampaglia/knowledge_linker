'''
Dijkstra-based path finding algorithm for computing the metric closure on
similarity/proximity graphs 
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
from ._closure import bottleneckpaths as cbottleneckpaths
from .maxmin import _init_worker

def bottleneckpaths(A, source):
    '''
    Single source Bottleneck paths (i.e. max-min closure for a single source).
    This is a modification of Dikstra's shortest path algorithm for computing
    the bottleneck/maxmin paths. Returns the distance to all connected nodes and
    the paths. Note that this function is pure Python and thus very slow. Use
    the Cythonized version `cbottleneckpaths`, which accepts only CSR matrices.
    '''
    N = A.shape[0]
    certain = np.zeros(N, dtype=np.bool)
    items = {} # handles to the items inserted in the queue
    Q = [] # heap queue
    # populate the queue
    for node in xrange(N):
        if node == source:
            cap = 1.
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
        neighbors, = np.where(A[node])
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
            bott_caps.append(0)
            paths.append(np.empty(0, dtype=np.int))
        else:
            bott_caps.append(item[0])
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
    Computes the all-pairs bottleneck paths for matrix and saves the results for
    each row in separate files organized in a directory tree (see
    `truthy_measure.dirtree`). 

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
