#!/usr/bin/env python
''' Computes distance weights as in-degree of nodes '''

import numpy as np
import scipy.sparse as sp
import networkx as nx
from argparse import ArgumentParser

if __name__ == '__main__':

    parser = ArgumentParser(description=__doc__)
    parser.add_argument('data_path', metavar='data', help='Graph data')
    parser.add_argument('nodes', type=int, help='number of nodes')

    args = parser.parse_args()

    a = np.load(args.data_path, mmap_mode='r+')

    # adj[i,:] = (outnode, innode, weight)
    adj = sp.coo_matrix((a[:,2], (a[:,0], a[:,1])), (args.nodes,)*2)

    adj = adj.tocsc()
    w = adj.sum(axis=1)
    adj[:,2] = w
    print 'weights saved to file: {}'.format(args.data_path)

