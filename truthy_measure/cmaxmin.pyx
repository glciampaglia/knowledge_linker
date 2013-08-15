# cython: profile=False

import numpy as np
import scipy.sparse as sp
from collections import defaultdict
from tables import BoolAtom

from .utils import mkcarray, CHUNKSHAPE

# cimports
cimport numpy as cnp
cimport cython

# TODO change _csr_neighbors to use C array as return type
# TODO --> add nogil to _csr_neighbors
# TODO instead of np.empty, use malloc to allocate C arrays
# TODO --> add nogil to shortestpath
# TODO --> --> write parallel shortestpath with cython.parallel.prange

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
    dist : int
        The distance between source and target, -1 if they are not in the same
        connected component.
    path : array_like
        If dist > -1, an array of length dist with the path between source and
        target, else an empty array.

    Notes
    -----
    Executes a BFS from source and stops as soon as target is found. Worst case:
    searches the whole graph.
    '''
    A = sp.csr_matrix(A)
    cdef int [:] A_indices = A.indices
    cdef int [:] A_indptr = A.indptr
    cdef int [:] neigh
    cdef cnp.ndarray[ndim=1, dtype=cnp.int32_t] path
    cdef int N = A.shape[0], N_neigh, Nd
    cdef int [:] P = np.empty(N, dtype=np.int32) # Predecessors
    cdef int [:] D = np.empty(N, dtype=np.int32) # Distance vector
    cdef int [:] Q = np.empty(N, dtype=np.int32) # FIFO queue
    cdef int i, ii, readi = 0, writei = 1, d = 0, nodei, neighi
    # initialize P, D and Q
    for i in xrange(N):
        D[i] = -1
        P[i] = -1
    D[source] = 0
    Q[readi] = source
    found = 0
    while writei < N and not found:
        d += 1
        Nd = writei - readi # number of elements at distance d from source
        for i in xrange(Nd):
            nodei = Q[i + readi]
            neigh = _csr_neighbors(nodei, A_indices, A_indptr)
            N_neigh = len(neigh)
            for ii in xrange(N_neigh):
                neighi = neigh[ii]
                if D[neighi] == -1: # Found new node
                    D[neighi] = d
                    P[neighi] = nodei
                    Q[writei] = neighi
                    writei += 1
                if neighi == target:
                    found = 1
                    break
        readi += Nd
    if found:
        path = np.empty(d + 1, dtype=np.int32)
        path[0] = source
        path[d] = target
        nodei = target
        for i in xrange(d - 1):
            path[d - i - 1] = P[nodei]
            nodei = P[nodei]
    else:
        path = np.empty(0, dtype=np.int32)
    return path

@cython.profile(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline int [:] _csr_neighbors(int row, int [:] indices, int [:] indptr):
    '''
    Returns the neighbors of a row for a CSR adjacency matrix
    '''
    cdef int n, i, I, II
    cdef int [:] res
    I = indptr[row]
    II = indptr[row + 1]
    n = II - I
    res = np.empty((n,), dtype=np.int)
    for i in xrange(n):
        res[i] = indices[i + I]
    return res

cdef int _dfs_order

# recursive function
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline int _closure_visit(
        int node,
        int [:] adj_indices,
        int [:] adj_indptr,
        int [:] order,
        int [:] root,
        int [:] in_scc,
        int [:] in_stack,
        int [:] visited,
        object stack,
        object succ,
        object local_roots
        ) except -1:
    cdef int n_neigh, neigh_node, cand_root, n, i, r_node, comp_node, st_top,\
            n_stack
    cdef int [:] neighbors = _csr_neighbors(node, adj_indices, adj_indptr)
    global _dfs_order
    n = len(adj_indptr) - 1
    visited[node] = True
    order[node] = _dfs_order
    _dfs_order += 1
    root[node] = node
    n_neigh = len(neighbors)
    for neigh_node in xrange(n_neigh):
        if not visited[neigh_node]:
            _closure_visit(neigh_node, adj_indices, adj_indptr, order, root,
                    in_scc, in_stack, visited, stack, succ, local_roots)
        if not in_scc[root[neigh_node]]:
            if order[neigh_node] < order[node]:
                root[node] = root[neigh_node]
        else:
            local_roots[node].add(root[neigh_node])
    r_node = root[node]
    for cand_root in local_roots[node]:
        for i in xrange(n):
            if succ[cand_root, i]:
                succ[r_node, i] = True
        succ[r_node, cand_root] = True
    del local_roots[node]
    if r_node == node:
        succ[r_node, node] = True
        n_stack = len(stack)
        if n_stack > 0:
            st_top = stack[n_stack]
        if n_stack and order[st_top] >= order[node]:
            while True:
                comp_node = stack.pop()
                in_stack[comp_node] = False
                in_scc[comp_node] = True
                if comp_node != node:
                    for i in xrange(n):
                        if succ[comp_node, i]:
                            succ[node, i] = True
                n_stack = len(stack)
                if n_stack > 0:
                    st_top = stack[n_stack]
                if len(stack) == 0 or order[st_top] < order[node]:
                    break
        else:
            in_scc[node] = True
    else:
        if not in_stack[r_node]:
            stack.append(r_node)
            in_stack[r_node] = True
        succ[r_node, node] = True
    return 0

@cython.boundscheck(False)
@cython.wraparound(False)
def c_closure_rec(adj, sources=None, ondisk=False, outpath=None):
    global _dfs_order
    # main function
    adj = sp.csr_matrix(adj)
    _dfs_order = 0
    cdef:
        int n = adj.shape[0]
        int node, i, n_sources
        int [:] _sources
        int [:] adj_indices = adj.indices
        int [:] adj_indptr = adj.indptr
        int [:] order = np.zeros(n, dtype=np.int32)
        int [:] root = np.zeros(n, dtype=np.int32) - 1
        int [:] in_scc = np.zeros(n, dtype=np.int32)
        int [:] in_stack = np.zeros(n, dtype=np.int32)
        int [:] visited = np.zeros(n, dtype=np.int32)
        object stack = []
        object succ = defaultdict(bool)
        object local_roots = defaultdict(set)
    succ = mkcarray('succ', adj.shape, CHUNKSHAPE, BoolAtom(), outpath=outpath,
            ondisk=ondisk)
    if sources is None:
        n_sources = n
        _sources = np.arange(n)
    else:
        n_sources = len(sources)
        _sources = np.asarray(sources)
    for i in xrange(n_sources):
        node = _sources[i]
        if not visited[node]:
            _closure_visit(node, adj_indices, adj_indptr, order, root, in_scc,
                    in_stack, visited, stack, succ, local_roots)
    return (np.asarray(root), succ)

# iterative version
@cython.boundscheck(False)
@cython.wraparound(False)
def c_closure(adj, sources=None, ondisk=False, outpath=None):
    # main function
    adj = sp.csr_matrix(adj)
    cdef:
        int dfs_order = 0
        int n = adj.shape[0]
        int node, i, n_sources
        int [:] _sources, neighbors
        int [:] adj_indices = adj.indices
        int [:] adj_indptr = adj.indptr
        int [:] order = np.zeros(n, dtype=np.int32)
        int [:] root = np.zeros(n, dtype=np.int32) - 1
        int [:] in_scc = np.zeros(n, dtype=np.int32)
        int [:] in_stack = np.zeros(n, dtype=np.int32)
        int [:] visited = np.zeros(n, dtype=np.int32)
        int backtracking
        object stack = []
        object dfs_stack = []
        object succ = mkcarray('succ', adj.shape, CHUNKSHAPE, BoolAtom(),
                outpath=outpath, ondisk=ondisk)
        object local_roots = defaultdict(set)
    cdef int n_neigh, neigh_node, cand_root, r_node, comp_node, st_top,\
            n_stack
    if sources is None:
        n_sources = n
        _sources = np.arange(n)
    else:
        n_sources = len(sources)
        _sources = np.asarray(sources)
    for i in xrange(n_sources):
        node = _sources[i]
        if not visited[node]:
            dfs_stack = [node]
        while dfs_stack:
            node = dfs_stack[-1]
            visited[node] = True
            order[node] = dfs_order
            dfs_order += 1
            root[node] = node
            backtracking = 1
            neighbors = _csr_neighbors(node, adj_indices, adj_indptr)
            n_neigh = len(neighbors)
            for neigh_node in xrange(n_neigh):
                if not visited[neigh_node]:
                    dfs_stack.append(neigh_node)
                    backtracking = False
                    break
                if not in_scc[root[neigh_node]]:
                    if order[neigh_node] < order[node]:
                        root[node] = root[neigh_node]
                else:
                    local_roots[node].add(root[neigh_node])
            if backtracking:
                r_node = root[node]
                for cand_root in local_roots[node]:
                    for i in xrange(n):
                        if succ[cand_root, i]:
                            succ[r_node, i] = True
                    succ[r_node, cand_root] = True
                del local_roots[node]
                if r_node == node:
                    succ[r_node, node] = True
                    n_stack = len(stack)
                    if n_stack > 0:
                        st_top = stack[-1]
                    if len(stack) and order[st_top] >= order[node]:
                        while True:
                            comp_node = stack.pop()
                            in_stack[comp_node] = False
                            in_scc[comp_node] = True
                            if comp_node != node:
                                for i in xrange(n):
                                    if succ[comp_node, i]:
                                        succ[node, i] = True
                            n_stack = len(stack)
                            if n_stack:
                                st_top = stack[-1]
                            if n_stack == 0 or order[st_top] < order[node]:
                                break
                    else:
                        in_scc[node] = True
                else:
                    if not in_stack[r_node]:
                        stack.append(r_node)
                        in_stack[r_node] = True
                    succ[r_node, node] = True
                # clear the current node from the top of the DFS stack.
                dfs_stack.pop()
    return (np.asarray(root), succ)

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
