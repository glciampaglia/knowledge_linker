import scipy.sparse as scsp
import numpy as np
from nose.tools import nottest

from truthy_measure.utils import make_weighted, weighted_undir
from truthy_measure.maxmin_node import *

@nottest
def test_graph3():
    numVertices=10
    r = np.array([0,0,0,0,0,0,1,1,1,2,2,3,3,3,3,4,5,5,5,5,5,6,6,7,7,7,7,8,8,9])
    c = np.array([1,3,5,7,8,9,0,2,3,1,3,0,1,2,5,5,0,3,4,6,7,5,7,0,5,6,8,0,7,0])
    data = np.array([.33,.25,.2,.25,.5,-1,.16,.5,.25,.33,.25,.16,.33,.5,.2,.2,\
                    .16,.25,-1,.5,.25,.2,.25,.16,.2,.5,.25,.16,.25,.16])
    G = scsp.csr_matrix((data,(r,c)), shape=(numVertices,numVertices))
    G = weighted_undir(G, undirected=True)
    output = []
    for s in xrange(numVertices):
        row_output = []
        for t in xrange(numVertices):
            if s!=t:
                cap = bottlenecknodest(G,s,t)
                row_output.append(cap)
            else:
                row_output.append(0)
        output.append(row_output)
    o = np.matrix(output)
    return o
    #assert np.allclose(o,expect)

@nottest
def run_test(G, expect):
    o, _ = bottlenecknodefull(G)
    co, _ = cbottlenecknodefull(G)
    o = np.round(o, 2)
    co = np.round(o, 2)
    assert np.allclose(o, expect)
    assert np.allclose(co, expect)

def test_graph1():
    """
    Node-based Dijkstra on an arbitraty graph (ex. #1)
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

def test_graph2():
    """
    Node-based Dijkstra on an arbitraty graph (ex. #2)
    """
    data = np.ones(12, dtype=np.double)
    ptr = np.array([0,3,6,9,10,11,12])
    idx = np.array([1,2,3,0,2,4,0,1,5,0,1,2])
    N = 6
    G = scsp.csr_matrix((data,idx,ptr),shape=(N,N))
    G = weighted_undir(G)
    expect = np.matrix([
        [ 1.  ,  1.  ,  1.  ,  1.  ,  0.25,  0.25],
        [ 1.  ,  1.  ,  1.  ,  0.25,  1.  ,  0.25],
        [ 1.  ,  1.  ,  1.  ,  0.25,  0.25,  1.  ],
        [ 1.  ,  0.25,  0.25,  1.  ,  0.25,  0.25],
        [ 0.25,  1.  ,  0.25,  0.25,  1.  ,  0.25],
        [ 0.25,  0.25,  1.  ,  0.25,  0.25,  1.  ]])
    run_test(G, expect)
    
def test_cycle_graph():
    """
    Node-based Dijkstra on a 4-cycle
    """
    N = 5
    G = np.matrix([[False,  True, False, False,  True],
                    [ True, False,  True, False, False],
                    [False,  True, False,  True, False],
                    [False, False,  True, False,  True],
                    [ True, False, False,  True, False]])
    G = scsp.csr_matrix(G, dtype=np.double)
    G = weighted_undir(G)
    output = []
    expect = np.matrix([
        [ 1.  ,  1.  ,  0.33,  0.33,  1.  ],
        [ 1.  ,  1.  ,  1.  ,  0.33,  0.33],
        [ 0.33,  1.  ,  1.  ,  1.  ,  0.33],
        [ 0.33,  0.33,  1.  ,  1.  ,  1.  ],
        [ 1.  ,  0.33,  0.33,  1.  ,  1.  ]])
    run_test(G, expect)

def test_grid_graph():
    """
    Node-based Dijkstra on a grid
    """
    G = np.matrix([
        [False, False,  True,  True, False,  True],
        [False, False, False,  True, False,  True],
        [ True, False, False, False,  True, False],
        [ True,  True, False, False, False, False],
        [False, False,  True, False, False,  True],
        [ True,  True, False, False,  True, False]])
    G = scsp.csr_matrix(G, dtype=np.double)
    G = weighted_undir(G)
    expect = np.matrix([
        [ 1.  ,  0.33,  1.  ,  1.  ,  0.33,  1.  ],
        [ 0.33,  1.  ,  0.25,  1.  ,  0.25,  1.  ],
        [ 1.  ,  0.25,  1.  ,  0.25,  1.  ,  0.33],
        [ 1.  ,  1.  ,  0.25,  1.  ,  0.25,  0.33],
        [ 0.33,  0.25,  1.  ,  0.25,  1.  ,  1.  ],
        [ 1.  ,  1.  ,  0.33,  0.33,  1.  ,  1.  ]])
    run_test(G, expect)

def test_balanced_tree():
    """
    Node-based Dijkstra on a balanced tree with branching factor 3 and depth 2
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
    G = scsp.csr_matrix(G, dtype=np.double)
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

def test_graph4():
    """
    Node-based Dijkstra on an arbitraty graph (ex. #4)
    """
    G = np.matrix([
        [False, False, False,  True, False,  True],
        [False, False,  True, False, False, False],
        [False,  True, False, False, False,  True],
        [ True, False, False, False,  True, False],
        [False, False, False,  True, False,  True],
        [ True, False,  True, False,  True, False]])
    G = scsp.csr_matrix(G, dtype=np.double)
    G = weighted_undir(G)
    expect = np.matrix([
        [ 1.  ,  0.25,  0.25,  1.  ,  0.33,  1.  ],
        [ 0.25,  1.  ,  1.  ,  0.25,  0.25,  0.33],
        [ 0.25,  1.  ,  1.  ,  0.25,  0.25,  1.  ],
        [ 1.  ,  0.25,  0.25,  1.  ,  1.  ,  0.33],
        [ 0.33,  0.25,  0.25,  1.  ,  1.  ,  1.  ],
        [ 1.  ,  0.33,  1.  ,  0.33,  1.  ,  1.  ]])
    run_test(G, expect)

def test_graph5():
    """
    Node-based Dijkstra on an arbitraty graph (ex. #5)
    """
    G = np.matrix([
        [False, False,  True,  True,  True],
        [False, False,  True,  True, False],
        [ True,  True, False, False, False],
        [ True,  True, False, False,  True],
        [ True, False, False,  True, False]])
    G = scsp.csr_matrix(G, dtype=np.double)
    G = weighted_undir(G)
    expect = np.matrix([
        [ 1.  ,  0.33,  1.  ,  1.  ,  1.  ],
        [ 0.33,  1.  ,  1.  ,  1.  ,  0.25],
        [ 1.  ,  1.  ,  1.  ,  0.33,  0.25],
        [ 1.  ,  1.  ,  0.33,  1.  ,  1.  ],
        [ 1.  ,  0.25,  0.25,  1.  ,  1.  ]])
    run_test(G, expect)

