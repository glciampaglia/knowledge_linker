#   Copyright 2016 The Trustees of Indiana University.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

# cython: profile=False

import numpy as np
import scipy.sparse as sp

# cimports
cimport numpy as cnp
cimport cython

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
