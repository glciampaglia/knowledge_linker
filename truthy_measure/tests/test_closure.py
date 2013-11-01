import os
from glob import glob
from time import time
import numpy as np
import scipy.sparse as sp
from nose.tools import raises, nottest
from functools import partial

import truthy_measure.closure as clo
from truthy_measure.utils import DirTree, coo_dtype, fromdirtree
from truthy_measure.utils import make_weighted, weighted_undir

## tests for normal closure

def test_closure_big():
    '''
    closure on large graph + speed test
    '''
    np.random.seed(100)
    N = 500
    thresh = 0.1
    A = sp.rand(N, N, thresh, 'csr') 
    nnz = A.getnnz()
    sparsity = float(nnz) / N ** 2
    A = np.asarray(A.todense())
    source = 0
    tic = time()
    dists, paths = clo.closuress(A, source)
    toc = time()
    py_time = toc - tic
    tic = time()
    dists2, paths2 = clo.cclosuress(A, source)
    toc = time()
    cy_time = toc - tic
    assert np.allclose(dists, dists2)
    assert py_time > cy_time, \
            'python: {:.2g} s, cython: {:.2g} s.'.format(py_time, cy_time)

def test_closure_small():
    '''
    closure on small graph
    '''
    A = np.asarray([
        [0.0, 0.1, 0.0, 0.2],
        [0.0, 0.0, 0.3, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.3, 0.0, 0.0]
        ])
    source = 0
    dists, paths = clo.closuress(A, source)
    dists2, paths2 = clo.cclosuress(A, source, retpaths=1)
    assert np.allclose(dists, dists2)
    for p1, p2 in zip(paths, paths2):
        assert np.all(p1 == p2)

def test_closure_rand():
    '''
    closure on E-R random graph
    '''
    np.random.seed(21)
    N = 10
    sparsity = 0.3
    A = sp.rand(N, N, sparsity, 'csr')
    pyss = partial(clo.closuress, A)
    cyss = partial(clo.cclosuress, A, retpaths = 1)
    dists1, paths1 = zip(*map(pyss, xrange(N)))
    dists1 = np.asarray(dists1)
    paths1 = reduce(list.__add__, paths1)
    dists2, paths2 = zip(*map(cyss, xrange(N)))
    dists2 = np.asarray(dists2)
    paths2 = reduce(list.__add__, paths2)
    assert np.allclose(dists1, dists2)
    for p1, p2 in zip(paths1, paths2):
        assert np.all(p1 == p2)

def test_closureap():
    '''
    Correctedness of all-pairs parallel closure
    '''
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
    B = np.asarray(sp.coo_matrix((coords['weight'], (coords['row'], coords['col'])),
        shape=(N,N)).todense())
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
    np.random.seed(20)
    N = 10
    A = sp.rand(N, N, 1e-2, 'csr')
    source, target = np.random.randint(0, N, 2)
    cap1, path1 = clo.closure(A, source, target)
    cap2, path2 = clo.cclosure(A, source, target, retpath = 1)
    assert cap1 == cap2
    assert np.all(path1 == path2)

## test for epclosure* functions

@nottest
def run_test(G, expect):
    N = G.shape[0]
    pyfunc = partial(clo.epclosuress, G)
    cyfunc = partial(clo.epclosuress, G, closurefunc=clo.cclosuress, retpaths=1)
    o, p = zip(*map(pyfunc, xrange(N)))
    o = np.round(o, 2)
    co, cp = zip(*map(cyfunc, xrange(N)))
    co = np.round(co, 2)
    # check capacities match
    assert np.allclose(o, expect)
    assert np.allclose(co, expect)
    flags = (o > 0) & (o < 1)
    nonemptyi = np.where(flags)
    emptyi = np.where(np.logical_not(flags))
    # check paths match with computed capacities
    for s, t in np.ndindex(G.shape):
        if (s == t) or G[s, t] > 0 or (o[s,t] == 0):
            # path must be empty
            assert len(p[s][t]) == 0
            assert len(cp[s][t]) == 0
        else:
            # minimum on path must correspond to computed capacity
            path = p[s][t]
            weights = np.ravel(G[path[:-1], path[1:]])[:-1]
            weights = np.round(weights, 2)
            assert o[s, t] == np.min(weights)

def test_graph1_maxmin():
    """
    max-min epistemic closure on an arbitraty graph (ex. #1)
    """
    G = np.matrix([
        [ 0.,  1.,  0.,  0.,  0.,  1.,  0.,  0.],
        [ 1.,  0.,  1.,  0.,  0.,  0.,  1.,  0.],
        [ 0.,  1.,  0.,  1.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  1.,  0.,  1.,  1.,  1.,  0.],
        [ 0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.],
        [ 1.,  0.,  0.,  1.,  0.,  0.,  0.,  1.],
        [ 0.,  1.,  0.,  1.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.]], dtype=np.double)
    G = weighted_undir(G)
    expect = np.matrix([
        [ 1.  ,  1.  ,  0.25,  0.25,  0.2 ,  1.  ,  0.25,  0.25],
        [ 1.  ,  1.  ,  1.  ,  0.33,  0.2 ,  0.33,  1.  ,  0.25],
        [ 0.25,  1.  ,  1.  ,  1.  ,  0.2 ,  0.25,  0.25,  0.25],
        [ 0.25,  0.33,  1.  ,  1.  ,  1.  ,  1.  ,  1.  ,  0.25],
        [ 0.2 ,  0.2 ,  0.2 ,  1.  ,  1.  ,  0.2 ,  0.2 ,  0.2 ],
        [ 1.  ,  0.33,  0.25,  1.  ,  0.2 ,  1.  ,  0.25,  1.  ],
        [ 0.25,  1.  ,  0.25,  1.  ,  0.2 ,  0.25,  1.  ,  0.25],
        [ 0.25,  0.25,  0.25,  0.25,  0.2 ,  1.  ,  0.25,  1.  ]])
    run_test(G, expect)

def test_graph2_maxmin():
    """
    max-min epistemic closure on an arbitraty graph (ex. #2)
    """
    data = np.ones(12, dtype=np.double)
    ptr = np.array([0,3,6,9,10,11,12])
    idx = np.array([1,2,3,0,2,4,0,1,5,0,1,2])
    N = 6
    G = sp.csr_matrix((data,idx,ptr),shape=(N,N))
    G = weighted_undir(G)
    expect = np.matrix([
        [ 1.  ,  1.  ,  1.  ,  1.  ,  0.25,  0.25],
        [ 1.  ,  1.  ,  1.  ,  0.25,  1.  ,  0.25],
        [ 1.  ,  1.  ,  1.  ,  0.25,  0.25,  1.  ],
        [ 1.  ,  0.25,  0.25,  1.  ,  0.25,  0.25],
        [ 0.25,  1.  ,  0.25,  0.25,  1.  ,  0.25],
        [ 0.25,  0.25,  1.  ,  0.25,  0.25,  1.  ]])
    run_test(G, expect)
    
def test_cycle_graph_maxmin():
    """
    max-min epistemic closure on a 4-cycle
    """
    N = 5
    G = np.matrix([[False,  True, False, False,  True],
                    [ True, False,  True, False, False],
                    [False,  True, False,  True, False],
                    [False, False,  True, False,  True],
                    [ True, False, False,  True, False]])
    G = sp.csr_matrix(G, dtype=np.double)
    G = weighted_undir(G)
    output = []
    expect = np.matrix([
        [ 1.  ,  1.  ,  0.33,  0.33,  1.  ],
        [ 1.  ,  1.  ,  1.  ,  0.33,  0.33],
        [ 0.33,  1.  ,  1.  ,  1.  ,  0.33],
        [ 0.33,  0.33,  1.  ,  1.  ,  1.  ],
        [ 1.  ,  0.33,  0.33,  1.  ,  1.  ]])
    run_test(G, expect)

def test_grid_graph_maxmin():
    """
    max-min epistemic closure on a grid
    """
    G = np.matrix([
        [False, False,  True,  True, False,  True],
        [False, False, False,  True, False,  True],
        [ True, False, False, False,  True, False],
        [ True,  True, False, False, False, False],
        [False, False,  True, False, False,  True],
        [ True,  True, False, False,  True, False]])
    G = sp.csr_matrix(G, dtype=np.double)
    G = weighted_undir(G)
    expect = np.matrix([
        [ 1.  ,  0.33,  1.  ,  1.  ,  0.33,  1.  ],
        [ 0.33,  1.  ,  0.25,  1.  ,  0.25,  1.  ],
        [ 1.  ,  0.25,  1.  ,  0.25,  1.  ,  0.33],
        [ 1.  ,  1.  ,  0.25,  1.  ,  0.25,  0.33],
        [ 0.33,  0.25,  1.  ,  0.25,  1.  ,  1.  ],
        [ 1.  ,  1.  ,  0.33,  0.33,  1.  ,  1.  ]])
    run_test(G, expect)

def test_balanced_tree_maxmin():
    """
    max-min epistemic closure on a balanced tree with branching factor 3 and depth 2
    """
    G = np.matrix([
        [False,True,True,True,False,False,False,False,False,False,False,False,False],
        [True,False,False,False,True,True,True,False,False,False,False,False,False],
        [True,False,False,False,False,False,False,True,True,True,False,False,False],
        [True,False,False,False,False,False,False,False,False,False,True,True,True],
        [False,True,False,False,False,False,False,False,False,False,False,False,False],
        [False,True,False,False,False,False,False,False,False,False,False,False,False],
        [False,True,False,False,False,False,False,False,False,False,False,False,False],
        [False,False,True,False,False,False,False,False,False,False,False,False,False],
        [False,False,True,False,False,False,False,False,False,False,False,False,False],
        [False,False,True,False,False,False,False,False,False,False,False,False,False],
        [False,False,False,True,False,False,False,False,False,False,False,False,False],
        [False,False,False,True,False,False,False,False,False,False,False,False,False],
        [False,False,False,True,False,False,False,False,False,False,False,False,False]
        ])
    G = sp.csr_matrix(G, dtype=np.double)
    G = weighted_undir(G)
    expect = np.matrix([
        [ 1. , 1.  , 1.  , 1.  , 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
        [ 1. , 1.  , 0.25, 0.25, 1. , 1. , 1. , 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
        [ 1. , 0.25, 1.  , 0.25, 0.2, 0.2, 0.2, 1. , 1. , 1. , 0.2, 0.2, 0.2],
        [ 1. , 0.25, 0.25, 1.  , 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 1. , 1. , 1. ],
        [ 0.2, 1.  , 0.2 , 0.2 , 1. , 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
        [ 0.2, 1.  , 0.2 , 0.2 , 0.2, 1. , 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
        [ 0.2, 1.  , 0.2 , 0.2 , 0.2, 0.2, 1. , 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
        [ 0.2, 0.2 , 1.  , 0.2 , 0.2, 0.2, 0.2, 1. , 0.2, 0.2, 0.2, 0.2, 0.2],
        [ 0.2, 0.2 , 1.  , 0.2 , 0.2, 0.2, 0.2, 0.2, 1. , 0.2, 0.2, 0.2, 0.2],
        [ 0.2, 0.2 , 1.  , 0.2 , 0.2, 0.2, 0.2, 0.2, 0.2, 1. , 0.2, 0.2, 0.2],
        [ 0.2, 0.2 , 0.2 , 1.  , 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 1. , 0.2, 0.2],
        [ 0.2, 0.2 , 0.2 , 1.  , 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 1. , 0.2],
        [ 0.2, 0.2 , 0.2 , 1.  , 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 1. ]])
    run_test(G, expect)

def test_graph4_maxmin():
    """
    max-min epistemic closure on an arbitraty graph (ex. #4)
    """
    G = np.matrix([
        [False, False, False,  True, False,  True],
        [False, False,  True, False, False, False],
        [False,  True, False, False, False,  True],
        [ True, False, False, False,  True, False],
        [False, False, False,  True, False,  True],
        [ True, False,  True, False,  True, False]])
    G = sp.csr_matrix(G, dtype=np.double)
    G = weighted_undir(G)
    expect = np.matrix([
        [ 1.  ,  0.25,  0.25,  1.  ,  0.33,  1.  ],
        [ 0.25,  1.  ,  1.  ,  0.25,  0.25,  0.33],
        [ 0.25,  1.  ,  1.  ,  0.25,  0.25,  1.  ],
        [ 1.  ,  0.25,  0.25,  1.  ,  1.  ,  0.33],
        [ 0.33,  0.25,  0.25,  1.  ,  1.  ,  1.  ],
        [ 1.  ,  0.33,  1.  ,  0.33,  1.  ,  1.  ]])
    run_test(G, expect)

def test_graph5_maxmin():
    """
    max-min epistemic closure on an arbitraty graph (ex. #5)
    """
    G = np.matrix([
        [False, False,  True,  True,  True],
        [False, False,  True,  True, False],
        [ True,  True, False, False, False],
        [ True,  True, False, False,  True],
        [ True, False, False,  True, False]])
    G = sp.csr_matrix(G, dtype=np.double)
    G = weighted_undir(G)
    expect = np.matrix([
        [ 1.  ,  0.33,  1.  ,  1.  ,  1.  ],
        [ 0.33,  1.  ,  1.  ,  1.  ,  0.25],
        [ 1.  ,  1.  ,  1.  ,  0.33,  0.25],
        [ 1.  ,  1.  ,  0.33,  1.  ,  1.  ],
        [ 1.  ,  0.25,  0.25,  1.  ,  1.  ]])
    run_test(G, expect)

