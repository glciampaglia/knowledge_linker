#!/usr/bin/env python

import numpy as np
import scipy.sparse as sp
from argparse import ArgumentParser

def _to_similarity(x):
    '''
    transform into proximity/similarity weights \in [0,1]
    '''
    return (x + 1) ** -1

def _indegree_weights(adj):
    icol = adj.col
    adj = adj.tocsc()
    indegree = adj.sum(axis=0)
    indegree = np.asarray(indegree).flatten()
    weights = _to_similarity(indegree)
    return weights[icol]
    
def weights(adj, kind='indegree'):
    '''
    Computes proximity/similarity weights

    Parameters
    ----------
    adj     - a sparse matrix
    kind    - the type of weights definition. Right now only `indegree' is
              available

    Returns the similarity 
    '''
    if kind == 'indegree':
        return _indegree_weights(adj)
    else:
        raise ValueError('unknonw weight kind: {}'.format(kind))

def main(args):
    # load coordinates from file. 
    # coords is a recarray with records (row, col, weights)
    coords = np.load(args.data_path)
    
    # create sparse matrix
    irow = coords['row']
    icol = coords['col']
    w = coords['weight']
    shp = (args.nodes,) * 2
    adj = sp.coo_matrix((w, (irow, icol)), shp)

    # compute weights
    weights = weights(adj)

    # recreate sparse matrix with weights and convert to CSR format
    adj = sp.coo_matrix((weights, (irow, icol)), shp)
    adj = adj.tocsr()

if __name__ == '__main__':

    parser = ArgumentParser(description=__doc__)
    parser.add_argument('data_path', metavar='data', help='Graph data')
    parser.add_argument('nodes', type=int, help='number of nodes')

    args = parser.parse_args()

