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

""" Test suite for closure functions. """

import os
from glob import glob
from time import time
import numpy as np
import scipy.sparse as sp
from nose.tools import nottest
from functools import partial

import networkx as nx
from itertools import combinations

import knowledge_linker.algorithms.closure as clo
from knowledge_linker.utils import coo_dtype, disttosim, weighted
from knowledge_linker.inout.dirtree import DirTree, fromdirtree

# tests for normal closure

def test_closure_big():
    """ closure on large graph + speed test. """
    np.random.seed(100)
    N = 500
    thresh = 0.1
    A = sp.rand(N, N, thresh, 'csr')
    A = np.asarray(A.todense())
    source = 0
    tic = time()
    proxs, _ = clo.closuress(A, source)
    toc = time()
    py_time = toc - tic
    tic = time()
    proxs2, _ = clo.cclosuress(A, source)
    toc = time()
    cy_time = toc - tic
    assert np.allclose(proxs, proxs2)
    assert py_time > cy_time, \
        'python: {:.2g} s, cython: {:.2g} s.'.format(py_time, cy_time)


def test_closure_small():
    """ closure on small graph. """
    A = np.asarray([
        [0.0, 0.1, 0.0, 0.2],
        [0.0, 0.0, 0.3, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.3, 0.0, 0.0]
    ])
    # ultrametric
    source = 0
    proxs, paths = clo.closuress(A, source)
    proxs2, paths2 = clo.cclosuress(A, source, retpaths=1)
    assert np.allclose(proxs, proxs2)
    for p1, p2 in zip(paths, paths2):
        assert np.all(p1 == p2)
    # metric
    proxs, paths = clo.closuress(A, source, kind='metric')
    proxs2, paths2 = clo.cclosuress(A, source, retpaths=1, kind='metric')
    assert np.allclose(proxs, proxs2)
    for p1, p2 in zip(paths, paths2):
        assert np.all(p1 == p2)

@nottest
def test_closure_rand():
    """ closure on E-R random graph. """
    np.random.seed(21)
    N = 10
    sparsity = 0.3
    A = sp.rand(N, N, sparsity, 'csr')
    # ultrametric
    pyss = partial(clo.closuress, A)
    cyss = partial(clo.cclosuress, A, retpaths=1)
    proxs1, paths1 = zip(*map(pyss, xrange(N)))
    proxs1 = np.asarray(proxs1)
    paths1 = reduce(list.__add__, paths1)
    proxs2, paths2 = zip(*map(cyss, xrange(N)))
    proxs2 = np.asarray(proxs2)
    paths2 = reduce(list.__add__, paths2)
    assert np.allclose(proxs1, proxs2)
    # XXX known issue, see https://github.com/glciampaglia/knowledge_linker/issues/1
    for p1, p2 in zip(paths1, paths2):
        assert np.all(p1 == p2)
    # metric
    pyss = partial(clo.closuress, A, kind='metric')
    cyss = partial(clo.cclosuress, A, retpaths=1, kind='metric')
    proxs1, paths1 = zip(*map(pyss, xrange(N)))
    proxs1 = np.asarray(proxs1)
    paths1 = reduce(list.__add__, paths1)
    proxs2, paths2 = zip(*map(cyss, xrange(N)))
    proxs2 = np.asarray(proxs2)
    paths2 = reduce(list.__add__, paths2)
    assert np.allclose(proxs1, proxs2)
    for p1, p2 in zip(paths1, paths2):
        assert np.all(p1 == p2)


def test_closureap():
    """ Correctedness of all-pairs parallel closure. """
    np.random.seed(100)
    dt = DirTree('test', (2, 5, 10), root='test_parallel')
    N = 100
    thresh = 0.1
    A = sp.rand(N, N, thresh, 'csr')
    nnz = A.getnnz()
    sparsity = float(nnz) / N ** 2
    print 'Number of nnz = {}, sparsity = {:g}'.format(nnz, sparsity)
    A = np.asarray(A.todense())
    clo.closureap(A, dt)
    coords = np.asarray(fromdirtree(dt, N), dtype=coo_dtype)
    coo = (coords['weight'], (coords['row'], coords['col']))
    B = np.asarray(sp.coo_matrix(coo, shape=(N, N)).todense())
    rows = []
    for row in xrange(N):
        r, _ = clo.cclosuress(A, row)
        rows.append(r)
    C = np.asarray(rows)
    assert np.allclose(B, C)
    # cleanup
    for logpath in glob('closure-*.log'):
        os.remove(logpath)


def test_closure():
    """ Correctedness of s-t closure function. """
    np.random.seed(20)
    N = 10
    A = sp.rand(N, N, 1e-2, 'csr')
    source, target = np.random.randint(0, N, 2)
    # metric
    cap1, path1 = clo.closure(A, source, target)
    cap2, path2 = clo.cclosure(A, source, target, retpath=1)
    assert cap1 == cap2
    assert np.all(path1 == path2)
    # ultrametric
    cap1, path1 = clo.closure(A, source, target, kind='metric')
    cap2, path2 = clo.cclosure(A, source, target, retpath=1, kind='metric')
    assert cap1 == cap2
    assert np.all(path1 == path2)


def test_backbone():
    """ (ultra)metric backbone extraction. """
    N = 10
    A = np.zeros((N, N))
    center = 0
    A[center, :] = 1.0
    A[:, center] = 1.0
    B = clo.backbone(A).todense()
    assert np.allclose(B, A)


# test for epclosure* functions

@nottest
def run_test(G, expect):
    N = G.shape[0]
    clofunc = partial(clo.epclosuress, G, retpaths=True)
    o, p = zip(*map(clofunc, xrange(N)))
    o = np.round(o, 2)
    # check capacities match
    assert np.allclose(o, expect)
    # check paths match with computed capacities
    for s, t in np.ndindex(G.shape):
        if (s == t) or G[s, t] > 0 or (o[s, t] == 0):
            # path must be empty
            assert len(p[s][t]) == 0
        else:
            # minimum on path must correspond to computed capacity
            path = p[s][t]
            weights = np.ravel(G[path[:-1], path[1:]])[:-1]
            weights = np.round(weights, 2)
            assert o[s, t] == np.min(weights)


def test_graph1_maxmin():
    """ max-min epistemic closure on an arbitraty graph (ex. #1).  """
    G = np.matrix([
        [0.,  1.,  0.,  0.,  0.,  1.,  0.,  0.],
        [1.,  0.,  1.,  0.,  0.,  0.,  1.,  0.],
        [0.,  1.,  0.,  1.,  0.,  0.,  0.,  0.],
        [0.,  0.,  1.,  0.,  1.,  1.,  1.,  0.],
        [0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.],
        [1.,  0.,  0.,  1.,  0.,  0.,  0.,  1.],
        [0.,  1.,  0.,  1.,  0.,  0.,  0.,  0.],
        [0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.]], dtype=np.double)
    G = weighted(G, undirected=True)
    expect = np.matrix([
        [1.,    1.,    0.25,  0.25,  0.2,   1.,    0.25,  0.25],
        [1.,    1.,    1.,    0.33,  0.2,   0.33,  1.,    0.25],
        [0.25,  1.,    1.,    1.,    0.2,   0.25,  0.25,  0.25],
        [0.25,  0.33,  1.,    1.,    1.,    1.,    1.,    0.25],
        [0.2,   0.2,   0.2,   1.,    1.,    0.2,   0.2,   0.20],
        [1.,    0.33,  0.25,  1.,    0.2,   1.,    0.25,  1.00],
        [0.25,  1.,    0.25,  1.,    0.2,   0.25,  1.,    0.25],
        [0.25,  0.25,  0.25,  0.25,  0.2,   1.,    0.25,  1.00]])
    run_test(G, expect)


def test_graph2_maxmin():
    """ max-min epistemic closure on an arbitraty graph (ex. #2). """
    data = np.ones(12, dtype=np.double)
    ptr = np.array([0, 3, 6, 9, 10, 11, 12])
    idx = np.array([1, 2, 3, 0, 2, 4, 0, 1, 5, 0, 1, 2])
    N = 6
    G = sp.csr_matrix((data, idx, ptr), shape=(N, N))
    G = weighted(G, undirected=True)
    expect = np.matrix([
        [1.,    1.,    1.,    1.,    0.25,  0.25],
        [1.,    1.,    1.,    0.25,  1.,    0.25],
        [1.,    1.,    1.,    0.25,  0.25,  1.00],
        [1.,    0.25,  0.25,  1.,    0.25,  0.25],
        [0.25,  1.,    0.25,  0.25,  1.,    0.25],
        [0.25,  0.25,  1.,    0.25,  0.25,  1.00]])
    run_test(G, expect)


def test_cycle_graph_maxmin():
    """ max-min epistemic closure on a 4-cycle. """
    G = np.matrix([[False,  True, False, False,  True],
                   [True,  False,  True, False, False],
                   [False,  True, False,  True, False],
                   [False, False,  True, False,  True],
                   [True,  False, False,  True, False]])
    G = sp.csr_matrix(G, dtype=np.double)
    G = weighted(G, undirected=True)
    expect = np.matrix([
        [1.,    1.,    0.33,  0.33,  1.00],
        [1.,    1.,    1.,    0.33,  0.33],
        [0.33,  1.,    1.,    1.,    0.33],
        [0.33,  0.33,  1.,    1.,    1.00],
        [1.,    0.33,  0.33,  1.,    1.00]])
    run_test(G, expect)


def test_grid_graph_maxmin():
    """ max-min epistemic closure on a grid. """
    G = np.matrix([
        [False, False, True,  True,  False,  True],
        [False, False, False, True,  False,  True],
        [True,  False, False, False, True,  False],
        [True,  True,  False, False, False, False],
        [False, False, True,  False, False,  True],
        [True,  True,  False, False, True,  False]])
    G = sp.csr_matrix(G, dtype=np.double)
    G = weighted(G, undirected=True)
    expect = np.matrix([
        [1.,   0.33, 1.,   1.,   0.33, 1.00],
        [0.33, 1.,   0.25, 1.,   0.25, 1.00],
        [1.,   0.25, 1.,   0.25, 1.,   0.33],
        [1.,   1.,   0.25, 1.,   0.25, 0.33],
        [0.33, 0.25, 1.,   0.25, 1.,   1.00],
        [1.,   1.,   0.33, 0.33, 1.,   1.00]])
    run_test(G, expect)


def test_balanced_tree_maxmin():
    """ max-min epistemic closure on a balanced tree with branching factor 3
    and depth 2.

    """
    G = np.matrix([
        [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
    G = sp.csr_matrix(G, dtype=np.double)
    G = weighted(G, undirected=True)
    expect = np.matrix([
        [1.,  1.,   1.,   1.,   0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
        [1.,  1.,   0.25, 0.25, 1.,  1.,  1.,  0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
        [1.,  0.25, 1.,   0.25, 0.2, 0.2, 0.2, 1.,  1.,  1.,  0.2, 0.2, 0.2],
        [1.,  0.25, 0.25, 1.,   0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 1.,  1.,  1.0],
        [0.2, 1.,   0.2,  0.2,  1.,  0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
        [0.2, 1.,   0.2,  0.2,  0.2, 1.,  0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
        [0.2, 1.,   0.2,  0.2,  0.2, 0.2, 1.,  0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
        [0.2, 0.2,  1.,   0.2,  0.2, 0.2, 0.2, 1.,  0.2, 0.2, 0.2, 0.2, 0.2],
        [0.2, 0.2,  1.,   0.2,  0.2, 0.2, 0.2, 0.2, 1.,  0.2, 0.2, 0.2, 0.2],
        [0.2, 0.2,  1.,   0.2,  0.2, 0.2, 0.2, 0.2, 0.2, 1.,  0.2, 0.2, 0.2],
        [0.2, 0.2,  0.2,  1.,   0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 1.,  0.2, 0.2],
        [0.2, 0.2,  0.2,  1.,   0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 1.,  0.2],
        [0.2, 0.2,  0.2,  1.,   0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 1.0]
    ])
    run_test(G, expect)


def test_graph4_maxmin():
    """ max-min epistemic closure on an arbitraty graph (ex. #4). """
    G = np.matrix([
        [0, 0, 0, 1, 0, 1],
        [0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0]])
    G = sp.csr_matrix(G, dtype=np.double)
    G = weighted(G, undirected=True)
    expect = np.matrix([
        [1.,    0.25,  0.25,  1.,    0.33,  1.00],
        [0.25,  1.,    1.,    0.25,  0.25,  0.33],
        [0.25,  1.,    1.,    0.25,  0.25,  1.00],
        [1.,    0.25,  0.25,  1.,    1.,    0.33],
        [0.33,  0.25,  0.25,  1.,    1.,    1.00],
        [1.,    0.33,  1.,    0.33,  1.,    1.00]])
    run_test(G, expect)


def test_graph5_maxmin():
    """ max-min epistemic closure on an arbitraty graph (ex. #5). """
    G = np.matrix([
        [0, 0, 1, 1, 1],
        [0, 0, 1, 1, 0],
        [1, 1, 0, 0, 0],
        [1, 1, 0, 0, 1],
        [1, 0, 0, 1, 0]])
    G = sp.csr_matrix(G, dtype=np.double)
    G = weighted(G, undirected=True)
    expect = np.matrix([
        [1.,   0.33, 1.,   1.,   1.00],
        [0.33, 1.,   1.,   1.,   0.25],
        [1.,   1.,   1.,   0.33, 0.25],
        [1.,   1.,   0.33, 1.,   1.00],
        [1.,   0.25, 0.25, 1.,   1.00]])
    run_test(G, expect)


def test_closure_and_cclosure_against_networkx():
    """ Test 'clusure' and 'cclosure' on 'metric' againt the NetworkX shortest_path (Rion's testing) """

    G = nx.Graph()
    G.add_nodes_from([0,1,2,3,4])
    G.add_edges_from([(0,1), (1,2), (2,3), (3,4)], weight=0.1)
    G.add_edges_from([(0,4)], weight=0.8)

    # Extract Adjacency Matrix from G
    A = nx.adjacency_matrix(G)
    # Transform distance into a similarity
    x = np.ravel(A[A > 0])
    A[A > 0] = (1.0 / (x + 1.0))

    for n1, n2 in combinations(G.nodes(),2):

        # Tests all three methods of computing all shortest paths ('closure','cclosure', and 'nx.all_shortest_paths')
        c_dist, c_paths = clo.closure(A, source=n1, target=n2, kind='metric')
        c_paths = [n for n in c_paths] # convers numbers to letters

        cc_dist, cc_paths = clo.cclosure(A, source=n1, target=n2, retpath=1, kind='metric')
        cc_paths = [n for n in cc_paths] if cc_paths is not None else ''

        nx_paths = list(nx.all_shortest_paths(G, source=n1, target=n2, weight='weight'))[0]

        assert nx_paths == c_paths, "NetworkX and Python 'closure' differ"
        assert nx_paths == cc_paths, "NetworkX and Cython 'cclosure' differ"
        assert c_paths == cc_paths, "Python 'closure' and Cython 'cclosure' differ"


def test_backbone_metric():
    """ Test 'backbone' metric againt NetworkX (Rion's testing) """

    #


    # Creates a simple graph that will have distance edges that are Equal OR Bigger than original edges
    G = nx.Graph()
    G.add_nodes_from([0,1,2,3,4,5,6,7])
    G.add_edges_from([ (0,1), (0,2), (0,3), (1,2), (1,3), (2,3) ], weight=1. )
    G.add_edges_from([ (4,5), (4,6), (4,7), (5,6), (5,7), (6,7) ], weight=1. )
    G.add_edges_from([ (0,4) ], weight=4. )
    G.add_edges_from([ (2,6) ], weight=0.5 )

    # Instanciates a Distance Graph
    DG = nx.Graph()
    DG.add_nodes_from(G.nodes())

    # Instanciantes the Backbone SubGraph
    BG = nx.Graph()
    BG.add_nodes_from(G.nodes())

    # Extract Adjacency Matrix from G
    A = nx.adjacency_matrix(G)
    A = sp.csr_matrix(A)
    # Transform distance into a similarity
    A.data = disttosim(A.data)

    # Calculate Distance Graph
    for n1, n2 in combinations(G.nodes(),2):
        # Computes all possible shortest paths
        paths = [p for p in nx.all_shortest_paths(G, source=n1, target=n2, weight='weight')]

        # Calculate segment weights (min, sum)
        for path in paths:
            path_weights = []
            for seg1,seg2 in zip(path[:-1], path[1:]):

                seg_weight = G[seg1][seg2]['weight']
                if seg_weight > 0:
                    path_weights.append(seg_weight)

            sum_path_weights = sum(path_weights)
            DG.add_edge(n1, n2, {'weight_sum':sum_path_weights})

    # Build Backbone SubGraph
    for n1,n2, data in DG.edges(data=True):
        dm_sum_weight = data['weight_sum']

        try:
            g_weight = G[n1][n2]['weight']
        except:
            pass
        else:

            # Identify backbone SubGraph
            if (g_weight == dm_sum_weight):
                BG.add_edge(n1, n2, {'weight_sum':1.})

    # Adjacency Matrix of the Backbone SubGraph
    M_BG = nx.adjacency_matrix(BG, weight='weight_sum')

    # Calculate Backbone to compare
    M_B = clo.backbone(A, kind='metric', start=None, offset=None, nprocs=1)

    assert np.all(M_BG == M_B.todense()).all(), "Backbone and NetworkX Backbone differ"

