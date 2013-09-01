import numpy as np
import scipy.sparse as sp
from collections import defaultdict
from cStringIO import StringIO
from tempfile import NamedTemporaryFile
from itertools import izip, chain, repeat, groupby, product
from progressbar import ProgressBar, Bar, AdaptiveETA, Percentage
from tables import Float64Atom, Filters, open_file

from .dirtree import DirTree

# dtype for saving COO sparse matrices
coo_dtype = np.dtype([('row', np.int32), ('col', np.int32), ('weight', np.float)])

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

def dfs_items(sources, targets, n, succ, roots, progress):
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

def arrayfile(data_file, shape, descr, fortran=False):
    '''
    Returns an array that is memory-mapped to an NPY (v1.0) file

    Arguments
    ---------
    data_file - a file-like object opened with write mode
    shape - shape of the ndarray
    descr - any argument that numpy.dtype() can take
    fortran - if True, the array uses Fortran data order, otherwise C order
    '''
    from numpy.lib import format
    header = {
        'descr' : descr,
        'fortran_order' : fortran,
        'shape' : shape
        }
    preamble = '\x93NUMPY\x01\x00'
    data_file.write(preamble)
    cio = StringIO()
    format.write_array_header_1_0(cio, header) # write header here first
    format.write_array_header_1_0(data_file, header) # write header
    cio.seek(0)
    offset = len(preamble) + len(cio.readline()) # get offset
    return np.memmap(data_file, dtype=np.dtype(descr), mode=data_file.mode,
            shape=shape, offset=offset)

def disttosim(x):
    '''
    Transforms a vector non-negative integer distances x to proximity/similarity
    weights in the [0,1] interval:
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
        irow = coords[:,0]
        icol = coords[:,1]
        w = coords[:,2]
    adj = sp.coo_matrix((w, (irow, icol)), shape=shape)
    return adj.asformat(fmt)

def make_weighted(path, N):
    '''
    Return a weighted adjacency matrix, with edge weights computed as the
    in-degree of the incoming vertex, transformed to similarity scores.

    Parameters
    ----------
    path : string
        path to data file.
    N : integer
        number of nodes

    Returns
    -------
    adj : `scipy.sparse.csr_matrix`
        the weighted adjancency matrix
    '''
    # load coordinates from file.
    # coords is a recarray with records (row, col, weights)
    coords = np.load(path)
    # shortcuts
    irow = coords['row']
    icol = coords['col']
    shape = (N,) * 2
    # create sparse adjacency matrix
    adj = recstosparse(coords, shape)
    # computes distances based on in-degrees
    dist = indegree(adj)
    # transform distances to similarity scores
    sim = disttosim(dist)
    # assign the weight to each edge (the weight of an edge is the in-degree of
    # the incoming vertex, translated to a similarity score)
    weights = sim[icol]
    # recreate the sparse matrix with weights and convert to CSR format
    adj = sp.coo_matrix((weights, (irow, icol)), shape=shape)
    adj = adj.tocsr()
    return adj

def dict_of_dicts_to_coo(dd, num=-1):
    '''
    Transforms a dict of dicts to a records array in COOrdinates format.

    Parameters
    ----------
    dd : dict of dicts

    num : int
        the number of coordinate entries

    Returns
    -------
    coo : recarray
        A records array with fields: row, col, and data. This can be used to
        create a sparse matrix. See `coo_dtype`, `scipy.sparse.coo_matrix`.
    '''

    def coorditer(dd):
        for irow in sorted(dd):
            d = dd[irow]
            for icol in sorted(d):
                yield irow, icol, d[icol]
    return np.fromiter(coorditer(dd), coo_dtype, count=num)

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
    a = np.load(path)
    indices = a['indices']
    indptr = a['indptr']
    n_rows = len(indptr) - 1
    n_data = len(indices)
    if 'data' not in a.files:
        data = np.ones((n_data,), dtype=np.float64)
    return sp.csr_matrix((data, indices, indptr), shape=(n_rows, n_rows))

def chainrepeat(sources, times):
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

def mkcarray(fn, shape, name, chunksize, num=1):
    '''
    Create a HDF5 file at `fn` containing `num` compressed chunked arrays of double
    floats,  accessible under `/<name>`. The chunks of the array have shape `(1,
    chunksize)`. Returns the created file and the chunked array.
    '''
    atom = Float64Atom()
    filters = Filters(complevel=5, complib='zlib')
    h5f = open_file(fn, 'w')
    a = h5f.create_carray(h5f.root, name, atom, shape, filters=filters,
            chunkshape=(1, chunksize))
    return h5f, a
