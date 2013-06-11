#!/usr/bin/env python

from argparse import ArgumentParser
import numpy as np
import scipy.sparse as sp
import networkx as nx

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('datapath', metavar='data', help='the graph file')
    parser.add_argument('numnodes', type=int, help='number of nodes', metavar='nodes')
    args = parser.parse_args()
    data = np.load(args.datapath)
    shape = (args.numnodes,) * 2
    adj = sp.coo_matrix((data['weight'], (data['row'], data['col'])), shape)
    g = nx.DiGraph(adj)
    isDAG = nx.is_directed_acyclic_graph(g)
    nV = g.number_of_nodes()
    nE = g.number_of_edges()
    print 'data: %s' % args.datapath
    print 'number of nodes: %d' % nV
    print 'number of edges: %d' % nE
    print 'is DAG? %s' % isDAG
    print 'edge density: %.2e' % (float(nE) / nV ** 2)
    print 'avg. degree: %.2e' % (float(nE) / nV)
    print 'number of self-loops: %d' % g.number_of_selfloops()
