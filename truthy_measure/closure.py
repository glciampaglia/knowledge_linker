#!/usr/bin/env python

import numpy as np
import scipy.sparse as sp
from argparse import ArgumentParser
from datetime import datetime
from itertools import izip

from .maxmin import productclosure
from .utils import coo_dtype

_now = datetime.now

def disttosim(x):
    '''
    transforms a vector non-negative integer distances x to proximity/similarity
    weights in the [0,1] interval:
          1
    s = -----
        x + 1
    '''
    return (x + 1) ** -1

def indegree(adj):
    '''
    computes the in-degree of each node
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
    coords - either a recarray or a 2d ndarray. If recarray, fields must be
             named: `col`, `row`, and `weight`.
    shape  - the shape of the array, optional.
    fmt    - the sparse matrix format to use. See `scipy.sparse`. Default:
             csr. 
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
    path - path to data file
    N    - number of nodes
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

if __name__ == '__main__':

    parser = ArgumentParser(description=__doc__)
    parser.add_argument('data_path', metavar='data', help='Graph data')
    parser.add_argument('nodes', type=int, help='number of nodes')
    parser.add_argument('output', help='output file')
    parser.add_argument('-p', '--procs', type=int, metavar='NUM', 
            help='use %(metavar)s process for computing the transitive closure')
    parser.add_argument('-i', '--intermediate', action='store_true', 
            help='save intermediate matrix to file')

    args = parser.parse_args()

    # read adjacency matrix from file and compute weights
    print "{}: read data from {}.".format(_now(), args.data_path)
    print "{}: adjacency matrix created.".format(_now())
    adj = make_weighted(args.data_path, args.nodes)
    print "{}: in-degree weights computed. Computing closure...".format(_now())

    # compute transitive closure
    if args.procs is not None:
        adjt = productclosure(adj, parallel=True, splits=args.procs,
                nprocs=args.procs, dumpiter=args.intermediate)
    else:
        adjt = productclosure(adj, dumpiter=args.intermediate) 

    print "{}: closure algorithm completed.".format(_now())

    # save to file as records array
    adjt = adjt.tocoo()
    np.save(args.output, np.fromiter(izip(adjt.row, adjt.col, adjt.data),
        coo_dtype, len(adjt.row))) 
    print "{}: closure graph saved to {}.".format(_now(), args.output)
