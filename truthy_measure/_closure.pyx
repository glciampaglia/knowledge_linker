# cython: profile=False

import numpy as np
import scipy.sparse as sp
from cython.parallel import parallel, prange
from tempfile import NamedTemporaryFile
from struct import pack

# cimports
cimport numpy as cnp
cimport cython
from libc.stdlib cimport malloc, free, calloc
from libc.stdio cimport printf
from .heap cimport FastUpdateBinaryHeap

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef object bottleneckpaths(object A, int source, object f = None, int retpaths = 0):
    '''
    Computes the bottleneck distances from `source` on the directed graph
    represented by adjacency matrix `A`, and optionally write them to open file
    `f`. Return the distances and optionally the associated bottleneck paths.

    A bottleneck path corresponds to the path that maximizes the minimum weight
    on its edges. It is also known as the max-min co-norm, or ultra-metric
    distance.

    Parameters
    ----------
    A : array_like
        NxN weighted adjancency matrix, will be converted to compressed sparse
        row format. Weights are double floats.

    source : int
        The source node.

    f : open file

    retpaths : int
        optional; compute and return the paths to each connected node, or an
        empty array for disconnected pairs. Default: do not compute nor return
        paths.

    Returns
    -------
    dists : (N,) double float ndarray
        the bottleneck distances from source to all other nodes in the graph, or
        -1 if the two nodes are disconnected. 

    paths : list of ndarrays
        optional; the associated path of nodes (excluding the source).
    '''
    A = sp.csr_matrix(A)
    cdef:
        int [:] A_indptr = A.indptr
        int [:] A_indices = A.indices
        double [:] A_data = A.data
        int N = A.shape[0]
        int i
        int flag = 0
        int cnt = 0
        MetricPathPtr paths
        MetricPath path
        cnp.ndarray[cnp.double_t] dists = np.empty(N, dtype=np.double)
        object pathslist = []
    if f is not None:
        flag = 1
        f.write(pack("ii", source, 0)) # provisional count
    paths = _bottleneckpaths(N, A_indptr, A_indices, A_data, source, retpaths)
    for i in xrange(N):
        path = paths[i]
        if path.found:
            dists[i] = path.distance
            if flag:
                f.write(pack("id", i, path.distance))
                cnt += 1
            if retpaths:
                pathslist.append(np.asarray((<int [:path.length]>path.vertices).copy()))
                free(<void *>path.vertices)
        else:
            dists[i] = -1
            if retpaths:
                pathslist.append(np.empty(0, dtype=np.int))
    if flag and cnt:
        f.seek(sizeof(int))
        f.write(pack('i', cnt))
    free(<void *> paths)
    return (dists, pathslist)

# we push the inverse of the similarity to fake a max-heap: this means we
# compute the min-max.
@cython.boundscheck(False)
@cython.wraparound(False)
cdef MetricPathPtr _bottleneckpaths(
        int N,
        int [:] indptr,
        int[:] indices,
        double [:] data,
        int source,
        int retpaths):
    cdef:
        FastUpdateBinaryHeap Q = FastUpdateBinaryHeap(N, N)
        int * P = init_intarray(N, -1)
        int * certain = init_intarray(N, 0)
        int * tmp = init_intarray(N, -1) # stores paths in reverse order
        double * dists = <double *> malloc(N * sizeof(double))
        int * neighbors = NULL
        int node, i, hopscnt
        int N_neigh
        double sim, dist, w, d, neigh_dist
        MetricPathPtr paths = <MetricPathPtr> malloc(N * sizeof(MetricPath))
        MetricPath path
    # populate the queue
    for node in xrange(N):
        if node == source:
            sim = 1.
        else:
            sim = 0.0
        dists[node] = sim
        dist = (1.0 + sim) ** -1
        Q.push_fast(dist, node)
    # compute bottleneck distances and predecessor information
    while Q.count:
        dist = Q.pop_fast()
        node = Q._popped_ref
        certain[node] = True
        dists[node] = dist
        N_neigh = _csr_neighbors(node, indices, indptr, &neighbors)
        for i in xrange(N_neigh):
            neighbor = neighbors[i]
            if not certain[neighbor]:
                neigh_dist = Q.value_of_fast(neighbor)
                w = data[indptr[node] + i] # i.e. A[node, neigh_node]
                w = (1.0 + w) ** -1
                d = max(w, dist)
                if d < neigh_dist:
                    Q.push_if_lower_fast(d, neighbor) # will only update
                    P[neighbor] = node
        free(<void *> neighbors)
        neighbors = NULL
    # generate paths
    for node in xrange(N):
        path = paths[node]
        path.vertices = NULL
        if P[node] == -1:
            path.found = 0
            path.distance = -1
            path.length = -1
        else:
            path.found = 1
            dist = dists[node]
            bdist = dist ** -1 - 1.0
            path.distance = bdist
            if retpaths:
                hopscnt = 0
                i = node
                while i != source:
                    tmp[hopscnt] = i
                    hopscnt += 1
                    i = P[i]
                path.length = hopscnt + 1
                path.vertices = <int *>calloc(hopscnt + 1, sizeof(int))
                path.vertices[0] = source
                for i in xrange(hopscnt):
                    path.vertices[hopscnt - i] = tmp[i]
        paths[node] = path
    free(<void *>tmp)
    free(<void *>P)
    free(<void *>certain)
    free(<void *>dists)
    return paths

## BFS for shortest path (by number of hops)

cpdef reachables(object A, int source):
    A = sp.csr_matrix(A)
    cdef:
        int N = A.shape[0]
        int [:] A_indices = A.indices
        int [:] A_indptr = A.indptr
        int [:] reachables = np.zeros((N,), dtype=np.int32)
        object items
        int i
    _shortestpath(N, A_indptr, A_indices, source, source, reachables, 0)
    items, = np.where(reachables)
    return items

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef reachablesmany(object A, int [:] sources, int mmap = 0):
    A = sp.csr_matrix(A)
    cdef:
        int N = A.shape[0]
        int M = sources.shape[0]
        int [:] A_indices = A.indices
        int [:] A_indptr = A.indptr
        int [:, :] reachables
        object items
        int i
    f = None
    if mmap:
        f = NamedTemporaryFile()
        reachables = np.memmap(f, shape=(M, N), dtype=np.int32)
    else:
        reachables = np.zeros((M, N), dtype=np.int32)
    try:
        with nogil, parallel():
            for i in prange(M, schedule='guided'):
                _shortestpath(N, A_indptr, A_indices, sources[i], sources[i],
                        reachables[i], 0)
        items, = zip(*map(np.where, reachables))
        return items
    finally:
        if f:
            f.close() # delete temporary file

@cython.boundscheck(False)
@cython.wraparound(False)
def shortestpathmany(object A, int [:] sources, int [:] targets):
    A = sp.csr_matrix(A)
    if sources.shape[0] != targets.shape[0]:
        raise ValueError("sources/targets mismatch")
    cdef:
        size_t N = A.shape[0], M = sources.shape[0]
        PathPtr paths
        Path path
        object pathlist = []
        int [:] A_indices = A.indices
        int [:] A_indptr = A.indptr
        int i, j
        int [:] reachable = np.zeros((N,), dtype=np.int32)
    paths = <PathPtr> malloc(M * sizeof(Path))
    # parallel part
    with nogil, parallel():
        for i in prange(M, schedule='guided'):
            for j in xrange(N):
                reachable[j] = 0
            paths[i] = _shortestpath(N, A_indptr, A_indices, sources[i],
                    targets[i], reachable, 1)
    # pack results in a Python list
    for i in xrange(M):
        path = paths[i]
        if path.found:
            # copy allocated memory to allow later to free up the paths pointer
            pathlist.append(
                    np.asarray((<int [:path.length + 1]> path.vertices).copy()))
            free(<void *>paths.vertices)
        else:
            # create an empty array for disconnected vertices
            pathlist.append(np.empty(0, dtype=np.int32))
    free(<void *> paths)
    return pathlist

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray shortestpath(object A, int source, int target):
    '''
    Return the shortest path from source to target. A is converted to CSR
    format.

    Parameters
    ----------
    A : N x N array_like
        The adjacency matrix of the graph; will be converted to compressed
        sparse row format.
    source : int
        The source node.
    target : int
        The target node.

    Returns
    -------
    path : array_like
        an array of length dist with the path between source and target, else an
        empty array.

    Notes
    -----
    Executes a BFS from source and stops as soon as target is found. Worst case:
    searches the whole graph.
    '''
    A = sp.csr_matrix(A)
    cdef:
        size_t N = A.shape[0]
        Path path
        cnp.ndarray[cnp.int32_t] retpath
        int [:] A_indices = A.indices
        int [:] A_indptr = A.indptr
        int [:] reachable = np.zeros((N,), dtype=np.int32)
    path = _shortestpath(N, A_indptr, A_indices, source, target, reachable, 1)
    if path.found:
        retpath = np.asarray(<int [:path.length + 1]> path.vertices)
    else:
        retpath = np.empty(0, dtype=np.int32)
    return retpath

# The actual BFS function
@cython.boundscheck(False)
@cython.wraparound(False)
cdef Path _shortestpath(
        size_t N,
        int [:] A_indptr,
        int [:] A_indices,
        int source,
        int target,
        int [:] reached,
        int stoponfound) nogil:
    '''
    The actual BFS function for computing the shortest path

    Parameters
    ----------
    N : size_t
        The number of nodes in the graph.

    A_indptr, A_indices: int memoryviews
        The array of row index pointers and column pointers from the CSR object
        holding the adjacency matrix of the graph.

    source, target: int

    reached: pointer to int
        Pre-allocated array storing 1 for each reached node. The function must
        be called with `stoponfound = 0` in order to correctly compute this
        information.

    stoponfound: int memoryview
        integer flag, if 1, stop as soon as target is found, else, continue
        until the whole graph has been explored. This is needed if wanting to
        compute the reachability set of source.

    Returns
    -------
    path : a Path struct
    '''
    global neigh_buf
    cdef:
        int * P, * D, * Q, * visited  # predecessors, distance vectors, fifo queue
        int * neigh = NULL # neighbors
        Path path # return struct
        size_t N_neigh, Nd
        int i, ii, readi = 0, writei = 1, d = 0, nodei, neighi, found
        int breakflag = 0
    # initialize P, D and Q
    P = init_intarray(N, -1)
    D = init_intarray(N, -1)
    Q = init_intarray(N, -1)
    found = 0
    D[source] = 0
    Q[readi] = source
    while writei < N and readi < writei and not found:
        d += 1
        Nd = writei - readi # number of elements at distance d from source
        for i in xrange(Nd):
            nodei = Q[i + readi]
            N_neigh = _csr_neighbors(nodei, A_indices, A_indptr, &neigh)
            for ii in xrange(N_neigh):
                neighi = neigh[ii]
                if not reached[neighi]: # Found new node
                    reached[neighi] = 1
                    D[neighi] = d
                    P[neighi] = nodei
                    Q[writei] = neighi
                    writei += 1
                if neighi == target:
                    found = 1
                    if stoponfound:
                        breakflag = 1
                        break
            free(<void *>neigh)
            neigh = NULL
            if breakflag:
                break
        readi += Nd
    if found:
        path.length = d
        path.found = 1
        path.vertices = init_intarray(d + 1, -1)
        path.vertices[d] = target
        nodei = target
        for i in xrange(d):
            path.vertices[d - i - 1] = P[nodei]
            nodei = P[nodei]
    else:
        path.length = 0
        path.found = 0
        path.vertices = NULL
    free(<void *> P)
    free(<void *> D)
    free(<void *> Q)
    return path

