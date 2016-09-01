import os
import numpy as np
import scipy.sparse as sp

from cStringIO import StringIO
from itertools import izip, chain, repeat, groupby
from contextlib import closing

# dtype for saving COO sparse matrices
coo_dtype = np.dtype([
    ('row', np.int32),
    ('col', np.int32),
    ('weight', np.float)])


def arrayfile(data_file, shape, descr, fortran=False):
    '''
    Returns an array that is memory-mapped to an NPY (v1.0) file

    Arguments
    ---------
    data_file :
        a file-like object opened with write mode compatible to NumPy's
        memory-mapped array types (see `numpy.memmap`). It is responsibility of
        the caller to close the file.

    shape : tuple
        shape of the ndarray.

    descr : str
        a typecode str (see `array` of `numpy.dtype`). Will be converted to a
        NumPy dtype.

    fortran : bool
        optional; if True, the array uses Fortran data order. Default: use C
        order.
    '''
    from numpy.lib import format
    header = {
        'descr': descr,
        'fortran_order': fortran,
        'shape': shape
    }
    cio = StringIO()
    format.write_array_header_1_0(cio, header)  # write header here first
    format.write_array_header_1_0(data_file, header)  # write header
    cio.seek(0)
    offset = len(cio.readline())  # get offset
    return np.memmap(data_file, dtype=np.dtype(descr), mode=data_file.mode,
                     shape=shape,
                     offset=offset)


def disttosim(x):
    '''
    Transforms a vector non-negative integer distances x to
    proximity/similarity weights in the [0,1] interval:
                                         1
                                   s = -----
                                       x + 1
    Parameters
    ----------
    x : array_like
        an array of non-negative distances.
    '''
    return (x + 1) ** -1


def indegree(adj):
    '''
    Computes the in-degree of each node.

    Parameters
    ----------
    adj : sparse matrix
        the adjacency matrix, in sparse format (see `scipy.sparse`).

    Returns
    -------
    indeg : 1-D array
        the in-degree of each node
    '''
    adj = adj.tocsc()
    indegree = adj.sum(axis=0)
    return np.asarray(indegree).flatten()


def logindegree(adj):
    '''
    Computes the log(in-degree) of each node.

    Parameters
    ----------
    adj : sparse matrix
        the adjacency matrix, in sparse format (see `scipy.sparse`).

    Returns
    -------
    indeg : 1-D array
        the log(in-degree + 1) of each node
    '''
    d = indegree(adj)
    return np.log(d)


def make_symmetric(A):
    '''
    Transforms a matrix, not necessary triangular, to symmetric

    Parameters
    ----------
    A : array_like
        The matrix

    Returns : CSR sparse matrix
        The symmetric matrix
    '''
    G = sp.csr_matrix(A)
    n = G.shape[0]
    G2 = G.transpose()
    G3 = G + G2
    i, j, v = sp.find(G.multiply(G2))
    v = np.sqrt(v)
    N = sp.csr_matrix((v, (i, j)), shape=(n, n))
    Gsym = G3 - N
    return Gsym


def recstosparse(coords, shape=None, fmt='csr'):
    '''
    Returns a sparse adjancency matrix from a records array of (col, row,
    weights)

    Parameters
    ----------
    coords : array_likey
        either a recarray or a 2d ndarray. If recarray, fields must be named:
        `col`, `row`, and `weight`.
    shape : tuple
        the shape of the array, optional.
    fmt : string
        the sparse matrix format to use. See `scipy.sparse`. Default: csr.

    Returns
    -------
    adj : see sparse matrix (see `scipy.sparse`)
        the adjacency matrix in sparse matrix format.
    '''
    if coords.dtype.names is not None:
        # recarray
        irow = coords['row']
        icol = coords['col']
        w = coords['weight']
    else:
        # plain ndarray
        if coords.ndims != 2:
            raise ValueError('expecting a 2-d array or a recarray')
        if coords.shape[1] != 3:
            raise ValueError('expecting three columns (row, col, weights)')
        irow = coords[:, 0]
        icol = coords[:, 1]
        w = coords[:, 2]
    adj = sp.coo_matrix((w, (irow, icol)), shape=shape)
    if fmt == 'coo':
        return adj
    else:
        return adj.asformat(fmt)


def weighted(m, weightfunc=indegree, undirected=False):
    '''
    Transform a directed adjancency matrix to a weighted, directed, adjacency
    matrix, with edge weights computed as the in-degree of the incoming vertex
    (default), transformed to similarity scores.

    Parameters
    ----------
    m : `scipy.sparse.spmatrix`
        the input adjancency matrix
    weightfunc : function object
        a function that take an adjancency matrix and returns an array of
        weights.

    Returns
    -------
    adj : `scipy.sparse.csr_matrix`
        the weighted adjancency matrix. The matrix represents a directed
        network.
    '''
    # ensure input is in COO format
    m = sp.coo_matrix(m)
    if undirected:
        # transform to symmetric
        m = make_symmetric(m).tocoo()
    # compute in-degrees
    dist = weightfunc(m)
    # transform to similarity scores
    sim = disttosim(dist)
    # create CSR matrix
    return sp.coo_matrix((sim[m.col], (m.row, m.col)), shape=m.shape).tocsr()


WEIGHT_FUNCTIONS = {
    'degree': indegree,
    'logdegree': logindegree
}

def make_weighted(path, N, weight='degree', undirected=False):
    '''
    Loads a (row, col, weight) records array from path and returns a weighted
    CSR adjancency matrix.

    Parameters
    ----------
    path : string
        path to recarray numpy binary file.
    N : integer
        number of nodes in the graph
    undirected : bool
        if True, return the weight for an undirected network.
    '''
    # load coordinates from file.
    # coords is a recarray with records (row, col, weights)
    coords = np.load(path)
    shape = (N,) * 2
    # create sparse COO matrix
    adj = recstosparse(coords, shape, 'coo')
    weightfunc = WEIGHT_FUNCTIONS[weight]
    return weighted(adj, weightfunc, undirected=undirected)


def dict_of_dicts_to_sparse(dd, num, shape, kind):
    '''
    Transforms a dict of dicts to a records array in COOrdinates format.

    Parameters
    ----------
    dd : dict of dicts

    num : int
        the number of coordinate entries

    shape: tuple
        the output matrix shape

    kind : string
        a `scipy.sparse` matrix code (e.g, 'csr', 'coo', etc.)

    Returns
    -------
    A : scipy.sparse.spmatrix
        a `scipy.sparse` matrix
    '''

    def coorditer(dd):
        for s in dd:
            for t in dd[s]:
                yield s, t, (dd[s][t] or 1.)
    C = np.fromiter(coorditer(dd), dtype=coo_dtype, count=num)
    A = sp.coo_matrix((C['weight'], (C['row'], C['col'])), shape=shape)
    return A.asformat(kind)


def dict_of_dicts_to_ndarray(dd, size):
    '''
    Transforms a dict of dicts to 2-D array

    Parameters
    ----------
    dd : dict of dicts
        a dictionary of dictionary; a dictionary with len(dd) is mapped to each
        key in dd.
    size : tuple
        dimensions of the output array

    Returns
    -------
    a 2-D ndarray. The dtype is inferred from the first element. Missing
    elements are equal to zero.
    '''
    if len(size) != 2:
        raise ValueError('can return only 2-D arrays')
    dty = type(dd.itervalues().next().itervalues().next())
    tmp = np.zeros(size, dtype=dty)
    for irow in dd:
        d = dd[irow]
        for icol in d:
            tmp[irow, icol] = d[icol]
    return tmp


def loadadjcsr(path):
    '''
    Loads a CSR matrix from a NPZ file containing `indices`, `indptr`, and
    (optionally) `data` entries. If `data` is missing all non-zero elements are
    set to 1
    '''
    a = np.load(path)
    indices = a['indices']
    indptr = a['indptr']
    n_rows = len(indptr) - 1
    n_data = len(indices)
    if 'data' not in a.files:
        data = np.ones((n_data,), dtype=np.float64)
    return sp.csr_matrix((data, indices, indptr), shape=(n_rows, n_rows))


def chainrepeat(sources, times):
    '''
    utility function that does the same as using `chain` + `repeat` from the
    itertools module.
    '''
    count = sum(times)
    c = chain(*(repeat(s, t) for s, t in izip(sources, times)))
    return np.fromiter(c, count=count, dtype=np.int32)


def group(data, key, keypattern='{}'):
    '''
    group unsorted data by key and return them in a dictionary mapping key
    values to data groups. `key` is a function that applies to each element of
    data.
    '''
    mapping = {}
    for k, datagroup in groupby(sorted(data, key=key), key):
        mapping[keypattern.format(k)] = np.asarray(list(datagroup))
    return mapping


def load_csmatrix(path, fmt='csr'):
    """
    Return as CSR/CSC matrix with data structures memory-mapped onto disk.

    Parameters
    ==========

    path : str
        Path to a location on disk under which to find the
        {data/indices/indptr/shape}.npy files.

    fmt : str
        any of 'csr' or 'csc'. Default: csr.

    Returns
    =======

    adj : `scipy.sparse.csr_matrix` or `scipy.sparse.csc_matrix` A compressed
    sparse row/column matrix.

    Notes
    =====
    The structures data/indices/indptr and
    `numpy.memmap` objects open in copy-on-write mode (changes take place in
    memory but are not written back to disk, see `numpy.memmap`).
    """
    if fmt not in ['csc', 'csr']:
        raise ValueError("expecting either csc or csr: {}".format(fmt))
    # one
    data = np.load(os.path.join(path, 'data.npy'), mmap_mode='c')
    indices = np.load(os.path.join(path, 'indices.npy'), mmap_mode='c')
    indptr = np.load(os.path.join(path, 'indptr.npy'), mmap_mode='c')
    shape = np.load(os.path.join(path, 'shape.npy'))
    if fmt == 'csr':
        A = sp.csr_matrix((data, indices, indptr), shape=shape)
    else:
        A = sp.csc_matrix((data, indices, indptr), shape=shape)
    A.has_sorted_indices = True
    return A


def tocsc_memmap(A, dir='/tmp'):
    """
    Same as `scipy.sparse.csr.csr_matrix.tocsc` but using memory-mapped arrays.

    Parameters
    ----------
    A : scipy.sparse.csr_matrix
        The input CSR matrix

    dir : string
        The {data/indices/indptr}.npy files will be created under this path

    Returns
    -------
    B : scipy.sparse.csc_matrix
        The input matrix in CSC format

    Notes
    -----
    The files `{data/indices/indptr}.npy` will be created under `dir` (default:
    `/tmp`), together with file `shape.npy`. This file contains the value of
    B.shape. This is for compatibility purpose with `load_csr`.

    The returned matrix B has memmaps open in read-write mode and it is your
    responsibility to close them.
    """
    indptr_path = os.path.join(dir, 'indptr.npy')
    indices_path = os.path.join(dir, 'indices.npy')
    data_path = os.path.join(dir, 'data.npy')
    indptr_f = open(indptr_path, 'w+')
    indices_f = open(indices_path, 'w+')
    data_f = open(data_path, 'w+')
    with closing(indptr_f), closing(indices_f), closing(data_f):
        indptr = arrayfile(indptr_f, (A.shape[1] + 1,), '<i4')
        indices = arrayfile(indices_f, (A.nnz,), '<i4')
        data = arrayfile(data_f, (A.nnz,), '<f8')
        sp.csr.csr_tocsc(A.shape[0], A.shape[1],
                         A.indptr, A.indices, A.data,
                         indptr, indices, data)
    np.save(os.path.join(dir, 'shape.npy'), A.shape)
    return load_csmatrix(dir, 'csc')


def save_to_npy(A, dir):
    """
    Save a CSR to NPY format

    Parameters
    ==========

    A : scipy.sparse.csr_matrix

    dir : string
        path to a folder
    """
    A = sp.csr_matrix(A)
    np.save(os.path.join(dir, 'data.npy'), A.data)
    np.save(os.path.join(dir, 'indices.npy'), A.indices)
    np.save(os.path.join(dir, 'indptr.npy'), A.indptr)
    np.save(os.path.join(dir, 'shape.npy'), A.shape)
    print 'CSR matrix saved to {}/{{data,indices,indptr,shape}}.npy'.format(dir)


