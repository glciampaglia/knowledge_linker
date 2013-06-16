import numpy as np
import scipy.sparse as sp
from nose.tools import raises
import warnings

# local imports
from truthy_measure.maxmin import *
from truthy_measure.maxmin import _maxmin_naive, _maxmin_sparse 
from truthy_measure.cmaxmin import *
from truthy_measure.utils import dict_of_dicts_to_ndarray

def test_naive():
    A = np.random.rand(5, 5)
    AP = _maxmin_naive(A)
    AP2 = c_maxmin_naive(A) 
    assert np.array_equal(AP, AP2)

def test_naive_subset():
    a = 1
    b = 3
    A = np.random.rand(5, 5)
    AP = _maxmin_naive(A, a, b)
    AP2 = c_maxmin_naive(A, a, b) 
    assert np.array_equal(AP, AP2)

@raises(ValueError)
def test_naive_sparse():
    A = sp.rand(5, 5, .2, 'csr')
    AP = _maxmin_naive(A)
    AP2 = c_maxmin_naive(A) # expects ndarray type
    assert np.array_equal(AP, AP2)

def test_sparse():
    A = sp.rand(5, 5, .2, 'csr')
    AP = _maxmin_naive(A)
    AP2 = _maxmin_sparse(A)
    assert np.array_equal(AP, AP2.todense())

    AP3 = c_maxmin_sparse(A)
    assert np.array_equal(AP, AP3.todense())

def test_sparse_subset():
    a = 1
    b = 3
    A = sp.rand(5, 5, .2, 'csr')
    AP = _maxmin_naive(A, a, b)
    AP2 = _maxmin_sparse(A, a, b)
    assert np.array_equal(AP, AP2.todense())

    AP3 = c_maxmin_sparse(A, a, b)
    assert np.array_equal(AP, AP3.todense())

def test_frontend():
    A = sp.rand(5, 5, .2, 'csr')
    AP = _maxmin_naive(A)
    AP2 = maxmin(A)
    assert np.array_equal(AP, AP2.todense())

def test_parallel():
    A = sp.rand(5, 5, .2, 'csr')
    AP = maxmin(A)
    AP2 = pmaxmin(A, nprocs=2)
    assert np.array_equal(AP.todense(), AP2.todense())

def test_parallel_is_faster():
    from time import time
    B = sp.rand(4000, 4000, 1e-4, 'csr')

    tic = time()
    C = pmaxmin(B, 10, 10)
    toc = time()
    time_parallel = toc - tic

    tic = time()
    D = maxmin(B)
    toc = time()
    time_serial = toc - tic
    assert time_serial > time_parallel, "parallel version slower than serial!"

def test_maximum_csr():
    A = sp.rand(5, 5, .2, 'csr')
    B = sp.rand(5, 5, .2, 'csr')
    C1 = np.maximum(A.todense(), B.todense())
    C2 = c_maximum_csr(A, B).todense()
    assert np.array_equal(C1, C2)

def test_closure_cycle_2():
    # length 2 cycle
    C2 = np.array([
        [0.0, 0.5, 0.0], 
        [0.0, 0.0, 0.1], 
        [0.2, 0.0, 0.0]
        ])
    C2T = np.array([    
        [0.1, 0.5, 0.1],
        [0.1, 0.1, 0.1],
        [0.2, 0.2, 0.1]
        ])
    res = productclosure(C2, maxiter=100)
    print res
    assert np.allclose(res, C2T) 

def test_transitive_closure():
    '''
    Test recursive vs non-recursive implementation of closure_cycles
    '''
    B = sp.rand(10, 10, .2, 'csr')
    root1, succ1 = closure_cycles(B)
    root2, succ2 = closure_cycles_recursive(B)
    for i in xrange(B.shape[0]):
        if succ1[root1[i]] != succ2[root2[i]]:
            raise AssertionError("successors sets differ.")

def test_closure():
    B = sp.rand(10, 10, .2, 'csr')
    with warnings.catch_warnings():
        # most likely it won't converge, so we ignore the warning
        warnings.simplefilter("ignore")
        Cl1 = productclosure(B, splits=2, nprocs=2, maxiter=10, parallel=True)
        Cl2 = productclosure(B, maxiter=100)
        assert np.allclose(Cl1.todense(), Cl2.todense())

def test_maxmin_cycles_iterative():
    A = np.random.random_sample((5,5))
    res1 = maxmin_closure_cycles(A)
    res2 = maxmin_closure_cycles_recursive(A)
    assert np.allclose(res1, res2)

# on simple cycles, the matrix multiplication and the graph traversal algorithms
# give the same correct answer

def test_maxmin_c3():
    '''
    length 3 cycle
    '''
    C3 = np.array([
        [0.0, 0.5, 0.0], 
        [0.0, 0.0, 0.1], 
        [0.2, 0.0, 0.0] 
        ])
    C3T = np.array([
        [0.1, 0.5, 0.1],
        [0.1, 0.1, 0.1],
        [0.2, 0.2, 0.1]
        ])
    res1 = maxmin_closure_cycles(C3) # graph traversal
    res2 = productclosure(C3, maxiter=100) # matrix multiplication
    assert np.allclose(res1, res2)
    assert np.allclose(res1, C3T)

def test_closure_c4():
    '''
    length 4 cycle
    '''
    C4 = np.array([
            [0.0, 0.5, 0.0, 0.0], 
            [0.0, 0.0, 0.2, 0.0],
            [0.0, 0.0, 0.0, 0.4],
            [0.1, 0.0, 0.0, 0.0]
            ])
    C4T = np.array([
            [0.1, 0.5, 0.2, 0.2], 
            [0.1, 0.1, 0.2, 0.2],
            [0.1, 0.1, 0.1, 0.4],
            [0.1, 0.1, 0.1, 0.1]
            ])
    res1 = maxmin_closure_cycles(C4) # graph traversal
    res2 = productclosure(C4, maxiter=100) # matrix multiplication
    assert np.allclose(res1, res2)
    assert np.allclose(res1, C4T)
