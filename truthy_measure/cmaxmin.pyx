# cython: profile=False

import numpy as np
import scipy.sparse as sp
from collections import defaultdict
from tables import BoolAtom
from cython.parallel import parallel, prange
from tempfile import NamedTemporaryFile
from heapq import heappush, heappop, heapreplace

# cimports
cimport numpy as cnp
cimport cython
from libc.math cimport fmin
from libc.stdlib cimport realloc, malloc, abort, free
from libc.string cimport memset
from libc.stdio cimport printf
from libc.float cimport DBL_MAX

cpdef object bottleneckpaths(object A, int source):
    A = sp.csr_matrix(A)
    cdef:
        int [:] A_indptr = A.indptr
        int [:] A_indices = A.indices
        double [:] A_data = A.data
        int N = A.shape[0]
    return _bottleneckpaths(N, A_indptr, A_indices, A_data, source)

# we push the inverse of the similarity to fake a max-heap
cdef _bottleneckpaths(
        int N, 
        int [:] indptr, 
        int[:] indices, 
        double [:] data,
        int source):
    cdef:
        object items = {}, item
        object Q = []
        int * neighbors = NULL
        int node, i
        int N_neigh
        double sim, dist, w, d, neigh_dist
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
        if neighbors != NULL:
            free(<void *>neighbors)
        neighbors = _csr_neighbors(node, indices, indptr)
        N_neigh = indptr[node + 1] - indptr[node]
        for i in xrange(N_neigh):
            neighbor = neighbors[i]
            neighbor_item = items[neighbor]
            neigh_dist = neighbor_item[0]
            w = data[indptr[node] + i] # i.e. A[node, neigh_node]
            w = (1.0 + w) ** -1
            d = max(w, dist)
            if d < neigh_dist:
                neighbor_item[0] = d
                neighbor_item[2] = node
                heapreplace(Q, neighbor_item)
    # generate paths
    bott_dists = []
    paths = []
    for node in xrange(N):
        item = items[node]
        if item[2] == -1: # disconnected node
            continue
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

ctypedef struct MetricPath:
    size_t length
    int * vertices
    int found
    double distance

ctypedef MetricPath * MetricPathPtr

# parallel version, multiple source-target pairs
@cython.boundscheck(False)
@cython.wraparound(False)
def maxminclosuremany(object A, int [:] sources, int [:] targets):
    A = sp.csr_matrix(A)
    if sources.shape[0] != targets.shape[0]:
        raise ValueError("sources/targets mismatch")
    cdef:
        size_t N = A.shape[0], M = sources.shape[0]
        MetricPathPtr paths
        MetricPath path
        object pathlist = []
        double [:] distances = np.zeros((M,), dtype=np.float64)
        int [:] A_indices = A.indices
        int [:] A_indptr = A.indptr
        double [:] A_data = A.data
        int i
    paths = <MetricPathPtr> malloc(M * sizeof(MetricPath))
    # parallel section
    with nogil, parallel():
        for i in prange(M, schedule='guided'):
            paths[i] = _maxminclosure(N, A_indptr, A_indices, A_data,
                    sources[i], targets[i])
    # pack results in a Python list
    for i in xrange(M):
        path = paths[i]
        if path.found:
            pathlist.append(
                    np.asarray((<int [:path.length + 1]> path.vertices).copy()))
            distances[i] = path.distance
        else:
            pathlist.append(np.empty(0, dtype=np.int32))
            distances[i] = -1
    free(<void *> paths)
    return pathlist, np.asarray(distances)

# single source-target pair
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef object maxminclosure(object A, int source, int target):
    A = sp.csr_matrix(A)
    cdef:
        size_t N = A.shape[0]
        MetricPath path
        double distance
        cnp.ndarray[cnp.int32_t] retpath
        int [:] A_indices = A.indices
        int [:] A_indptr = A.indptr
        double [:] A_data = A.data
    path = _maxminclosure(N, A_indptr, A_indices, A_data, source, target)
    if path.found:
        retpath = np.asarray(<int [:path.length + 1]> path.vertices)
        distance = path.distance
    else:
        retpath = np.empty(0, dtype=np.int32)
        distance = -1
    return (retpath, distance)

ctypedef struct StackElem:
    int node
    int last_neighbor
    double minsofar

ctypedef StackElem * StackPtr

ctypedef struct Stack:
    StackPtr elements
    int top
    int size

cdef inline void init_stack(int n, Stack * stack) nogil:
    cdef void * buf
    buf = malloc(n* sizeof(StackElem))
    if buf == NULL:
        abort()
    stack.elements = <StackPtr> buf
    stack.top = -1
    stack.size = n

cdef inline void push(StackElem elem, Stack * stack) nogil:
    stack.top += 1
    stack.elements[stack.top] = elem

cdef inline StackElem pop(Stack * stack) nogil:
    cdef StackElem elem
    elem = stack.elements[stack.top]
    stack.top -= 1
    return elem

cdef inline StackElem newelem(
        int node,
        int last_neighbor,
        double minsofar) nogil:
    cdef StackElem elem
    elem.node = node # the current node
    elem.last_neighbor = last_neighbor # the last neighbor visited so far
    elem.minsofar = minsofar # the minimum along the path up to node
    return elem

# the actual search function
@cython.boundscheck(False)
@cython.wraparound(False)
cdef MetricPath _maxminclosure(
        int N,
        int [:] indptr,
        int [:] indices,
        double [:] data,
        int source,
        int target
        ) nogil:
    cdef:
        int i, j, neigh_node, backtracking, N_neigh
        double w, m, maxsofar = -1
        int * neigh = NULL, * inpath # neighbors, predecessors, in-path
        Stack stack # DFS stack
        StackElem curr, top
        MetricPath path
    path.length = 0
    path.found = 0
    path.vertices = NULL
    path.distance = -1
    init_stack(N, &stack)
    inpath = init_intarray(N, 0)
    push(newelem(source, -1, DBL_MAX), &stack)
    inpath[source] = 1
    while stack.top >= 0:
        curr = stack.elements[stack.top]
        backtracking = 1
        if neigh != NULL:
            free(<void *> neigh)
        neigh = _csr_neighbors(curr.node, indices, indptr)
        N_neigh = indptr[curr.node + 1] - indptr[curr.node]
        for i in xrange(N_neigh):
            neigh_node = neigh[i]
            if neigh_node <= curr.last_neighbor:
                # skip already visited neighbor
                continue
            curr.last_neighbor = neigh_node
            stack.elements[stack.top] = curr
            w = data[indptr[curr.node] + i] # i.e. A[node, neigh_node]
            if w < maxsofar:
                # prune path
                continue
            m = fmin(curr.minsofar, w)
            if neigh_node == target:
                # update maximum so far
                inpath[neigh_node] = 1
                if maxsofar < m:
                    maxsofar = m
                    path.distance = maxsofar
                    path.length = stack.top
                    path.found = 1
                    path.vertices = <int *> realloc(<void *>path.vertices,
                            (path.length + 1) * sizeof(int))
                    if path.vertices == NULL:
                        abort()
                    for j in xrange(path.length):
                        path.vertices[j] = stack.elements[j].node
                    path.vertices[path.length] = target
                continue
            if not inpath[neigh_node]:
                push(newelem(neigh_node, -1, m), &stack)
                backtracking = 0
                inpath[curr.node] = 1
                break
        if backtracking:
            pop(&stack)
            inpath[curr.node] = 0
    free(<void *> inpath)
    free(<void *> stack.elements)
    return path

## BFS for shortest path (by number of hops)

ctypedef struct Path:
    size_t length
    int * vertices
    int found

ctypedef Path * PathPtr

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
        int * neigh # neighbors
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
            neigh = _csr_neighbors(nodei, A_indices, A_indptr)
            N_neigh = A_indptr[nodei + 1] - A_indptr[nodei]
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

@cython.profile(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline int * init_intarray(size_t n, int val) nogil:
    '''
    Allocates memory for holding n int values, and initialize them to val.
    Caller is responsible for free-ing up the memory.
    '''
    cdef:
        void * buf
        int * ret
    buf = malloc(n * sizeof(int))
    if buf == NULL:
        abort()
    memset(buf, val, n * sizeof(int))
    ret = <int *> buf
    return ret

@cython.profile(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline int * _csr_neighbors(int row, int [:] indices, int [:] indptr) nogil:
    '''
    Returns the neighbors of a row for a CSR adjacency matrix. Caller is
    responsible to `free` allocated memory at the end.
    '''
    cdef size_t n
    cdef int i, I, II
    cdef void * buf
    cdef int * res
    I = indptr[row]
    II = indptr[row + 1]
    n = II - I
    buf = malloc(n * sizeof(int))
    if buf == NULL:
        abort()
    res = <int *> buf
    for i in xrange(n):
        res[i] = indices[I + i]
    return res

## Maxmin matrix multiplication

@cython.boundscheck(False)
@cython.wraparound(False)
def c_maxmin_naive(object A, object a=None, object b=None):
    '''
    See `maxmin.maxmin_naive`. Cythonized version. Doesn't work on CSR sparse
    matrices (`scipy.sparse.csr_matrix`).
    '''
    cdef:
        int N
        cnp.ndarray[cnp.double_t, ndim=2] _A = np.asarray(A)
        cnp.ndarray[cnp.double_t, ndim=2] AP
        int i,j,ih
        cnp.double_t max_ij, aik, akj, min_k
    N = A.shape[0]
    if a is None:
        a = 0
    if b is None:
        b = N
    Nout = <int>(b - a)
    AP = np.zeros((Nout, N), A.dtype)
    for i in xrange(Nout):
        ih = <int>a + i
        for j in xrange(N):
            max_ij = 0.
            for k in xrange(N):
                aik = _A[ih,k]
                akj = _A[k,j]
                min_k = min(aik, akj)
                if min_k > max_ij:
                    max_ij = min_k
            AP[i, j] = max_ij
    return AP

@cython.boundscheck(False)
@cython.wraparound(False)
def c_maxmin_sparse(object A, object a=None, object b=None):
    '''
    See `maxmin.maxmin_sparse`. Cythonized version. Requires as argument a
    sparse CSR matrix (see `scipy.sparse.csr_matrix`).
    '''
    cdef:
        cnp.ndarray[int, ndim=1] A_indptr, A_indices, At_indptr, At_indices
        cnp.ndarray[cnp.double_t, ndim=1] A_data, At_data
        object AP_data, AP_indptr, AP_indices
        int N, Nout, i, ih, j, ii, jj, ik, kj, iimax, jjmax, innz, iptr
        double max_ij, min_k

    if not sp.isspmatrix_csr(A):
        raise ValueError('expecting a sparse CSR matrix')

    N = A.shape[0]
    if a is None:
        a = 0
    if b is None:
        b = N
    Nout = <int>(b - a)

    # build output matrix directly in compressed sparse row format. These are
    # the index pointers, indices, and data lists for the output matrix
    AP_indptr = [0]
    AP_indices = []
    AP_data = []
    iptr = 0

    # At is A in compressed sparse column format
    At = A.tocsc()
    A_indptr = A.indptr
    A_indices = A.indices
    A_data = A.data
    At_indptr = At.indptr
    At_indices = At.indices
    At_data = At.data

    for i in xrange(Nout):

        # innz keeps track of the number of non-zero elements on the i-th output row in
        # this iteration; ih is the index corresponding to i in the input matrix
        innz = 0
        ih = <int>a + i

        for j in xrange(N):

            # ii is the index of the first non-zero element value (in A.data)
            # and column index (in A.indices) of the the i-th row
            ii = A_indptr[ih]
            iimax = A_indptr[ih + 1]

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
                # add value and column index to data/indices lists, increment
                # number non-zero elements. Python list appends are fast!
                AP_data.append(max_ij)
                AP_indices.append(j)
                innz += 1

        # update indptr list
        iptr += innz
        AP_indptr.append(iptr)

    # return in CSR format
    return sp.csr_matrix((np.asarray(AP_data), AP_indices, AP_indptr), (Nout, N))

def c_maximum_csr(object A, object B):
    '''
    Equivalent of numpy.maximum for CSR matrices.
    '''
    cdef:
        cnp.ndarray[int, ndim=1] A_indptr, A_indices, B_indptr, B_indices
        cnp.ndarray[cnp.double_t, ndim=1] A_data, B_data
        object Out_indptr, Out_indices, Out_data
        int k, kptr, knnz, ii, jj, iimax, jjmax, icol, jcol, N, M, ik, jk

    # check input is CSR
    if not sp.isspmatrix_csr(A) or not sp.isspmatrix_csr(B):
        raise ValueError('expecting a CSR matrix')

    N = A.shape[0]
    M = A.shape[1]
    A_indptr = A.indptr
    A_indices = A.indices
    A_data = A.data
    B_indptr = B.indptr
    B_indices= B.indices
    B_data = B.data
    Out_indptr = [0]
    Out_indices = []
    Out_data = []
    kptr = 0

    for k in xrange(N):
        ii = A_indptr[k]
        jj = B_indptr[k]
        iimax = A_indptr[k + 1]
        jjmax = B_indptr[k + 1]
        knnz = 0

        if (ii == iimax) and (jj == jjmax):
            # both rows in A and B are empty, skip this row
            Out_indptr.append(kptr)
            continue

        elif (ii == iimax) ^ (jj == jjmax):
            # either one of A and B (but not both) has non-zero elements

            if (ii == iimax):
                # the k-th row in A is empty, add the elements of B
                knnz = jjmax - jj
                for jk in xrange(knnz):
                    jcol = B_indices[jk + jj]
                    Out_indices.append(jcol)
                    Out_data.append(B_data[jk + jj])
                kptr += knnz
                Out_indptr.append(kptr)

            else:
                # the k-th row in B is empty, add the elements of A
                knnz = iimax - ii
                for ik in xrange(knnz):
                    icol = A_indices[ik + ii]
                    Out_indices.append(icol)
                    Out_data.append(A_data[ik + ii])
                kptr += knnz
                Out_indptr.append(kptr)

        else:
            # the k-th rows in both B and A contain non-zero elements.

            # First, scan both index pointers simultaneously. Will break as soon
            # as all non-zero elements on one row are exhausted
            while ii < iimax and jj < jjmax:

                icol = A_indices[ii]
                jcol = B_indices[jj]

                # both elements non-zero, add max
                if icol == jcol:
                    Out_data.append(max(A_data[ii], B_data[jj]))
                    Out_indices.append(icol)
                    ii += 1
                    jj += 1

                # A-element is zero, add B-element
                elif icol > jcol:
                    Out_data.append(B_data[jj])
                    Out_indices.append(jcol)
                    jj += 1

                # B-element is zero, add A-element
                else:
                    Out_data.append(A_data[ii])
                    Out_indices.append(icol)
                    ii += 1

                knnz += 1

            # Second, complete by adding the elements left in the other row, if
            # any. By the previous loop condition, no more than one of these two
            # loops will execute.
            for ik in xrange(iimax - ii):
                icol = A_indices[ik + ii]
                Out_data.append(A_data[ik + ii])
                Out_indices.append(icol)
                knnz += 1

            for jk in xrange(jjmax - jj):
                jcol = B_indices[jk + jj]
                Out_data.append(B_data[jk + jj])
                Out_indices.append(jcol)
                knnz += 1

            # update the output index pointers array
            kptr += knnz
            Out_indptr.append(kptr)

    return sp.csr_matrix((np.asarray(Out_data), Out_indices, Out_indptr), (N, M))
