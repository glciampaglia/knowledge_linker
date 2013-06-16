import numpy as np
import scipy.sparse as sp

# dtype for saving COO sparse matrices
coo_dtype = np.dtype([('row', np.int32), ('col', np.int32), ('weight', np.float)])

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

def dict_of_dicts_to_ndarray(dd):
    '''
    Transforms a dict of dicts to 2-D array

    Parameters
    ----------
    dd : dict of dicts
        a dictionary of dictionary; a dictionary with len(dd) is mapped to each
        key in dd.

    Returns
    -------
    a 2-D ndarray
    '''
    def _sorted_values(d):
        for k in sorted(d):
            yield d[k]
    return np.asarray([ list(_sorted_values(d)) for d in _sorted_values(dd) ])
