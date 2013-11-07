from contextlib import closing, nested
from tempfile import mktemp
import numpy as np
import scipy.sparse as sp
from nose.tools import raises, nottest
import warnings

# local imports
from truthy_measure.maxmin import *
from truthy_measure.maxmin import _maxmin_naive, _maxmin_sparse 
from truthy_measure._maxmin import *

def test_naive():
    '''
    Test naive implementation Python vs Cython.
    '''
    A = np.random.rand(5, 5)
    AP = _maxmin_naive(A)
    AP2 = c_maxmin_naive(A) 
    assert np.array_equal(AP, AP2)

def test_naive_subset():
    '''
    Test naive implementation Python vs Cython with source/target parameters.
    '''
    a = 1
    b = 3
    A = np.random.rand(5, 5)
    AP = _maxmin_naive(A, a, b)
    AP2 = c_maxmin_naive(A, a, b) 
    assert np.array_equal(AP, AP2)

@raises(ValueError)
def test_naive_sparse():
    '''
    Test parameter checking cythonized maxmin product.
    '''
    A = sp.rand(5, 5, .2, 'csr')
    AP = _maxmin_naive(A)
    AP2 = c_maxmin_naive(A) # expects ndarray type
    assert np.array_equal(AP, AP2)

def test_sparse():
    '''
    Test maxmin product sparse implementation.
    '''
    A = sp.rand(5, 5, .2, 'csr')
    AP = _maxmin_naive(A)
    AP2 = _maxmin_sparse(A)
    assert np.array_equal(AP, AP2.todense())

    AP3 = c_maxmin_sparse(A)
    assert np.array_equal(AP, AP3.todense())

def test_sparse_subset():
    '''
    Test maxmin product with source/target parameters.
    '''
    a = 1
    b = 3
    A = sp.rand(5, 5, .2, 'csr')
    AP = _maxmin_naive(A, a, b)
    AP2 = _maxmin_sparse(A, a, b)
    assert np.array_equal(AP, AP2.todense())

    AP3 = c_maxmin_sparse(A, a, b)
    assert np.array_equal(AP, AP3.todense())

def test_frontend():
    '''
    Test maxmin product frontend.
    '''
    A = sp.rand(5, 5, .2, 'csr')
    AP = _maxmin_naive(A)
    AP2 = maxmin(A)
    assert np.array_equal(AP, AP2.todense())

def test_parallel():
    '''
    Test maxmin product parallel frontend.
    '''
    A = sp.rand(5, 5, .2, 'csr')
    AP = maxmin(A)
    AP2 = pmaxmin(A, nprocs=2)
    assert np.array_equal(AP.todense(), AP2.todense())

def test_maximum_csr():
    '''
    Test element-wise maximum on CSR matrix.
    '''
    A = sp.rand(5, 5, .2, 'csr')
    B = sp.rand(5, 5, .2, 'csr')
    C1 = np.maximum(A.todense(), B.todense())
    C2 = c_maximum_csr(A, B).todense()
    assert np.array_equal(C1, C2)

def test_closure_cycle_2():
    '''
    Test maxmin matrix multiplication on length-2 cycle.
    '''
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
    res = maxmin_closure(C2, maxiter=100)
    print res
    assert np.allclose(res, C2T) 

def test_matmul_closure():
    '''
    Test sequential vs parallel matrix multiplication transitive closure.
    '''
    B = sp.rand(10, 10, .2, 'csr')
    with warnings.catch_warnings():
        # most likely it won't converge, so we ignore the warning
        warnings.simplefilter("ignore")
        Cl1 = maxmin_closure(B, splits=2, nprocs=2, maxiter=10,
                parallel=True) 
        Cl2 = maxmin_closure(B, maxiter=100)
        assert np.allclose(Cl1.todense(), Cl2.todense())

# on simple cycles, the matrix multiplication and the graph traversal algorithms
# give the same correct answer

def test_closure_c3():
    '''
    Test correctedness of transitive closure on length 3 cycle.
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
    res = maxmin_closure(C3, maxiter=100) # matrix multiplication
    assert np.allclose(res, C3T)

def test_closure_c4():
    '''
    Test correctedness of transitive closure on length 4 cycle.
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
    res = maxmin_closure(C4, maxiter=100) # matrix multiplication
    assert np.allclose(res, C4T)

def test_closure_comb():
    '''
    Test correctedness of maxmin closure on a comb graph
    '''
    A = np.array([
        [0.0, 0.3, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.2, 0.1, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.1],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0]
        ])
    AT = np.array([
        [0.0, 0.3, 0.2, 0.1, 0.1],
        [0.0, 0.0, 0.2, 0.1, 0.1],
        [0.0, 0.0, 0.0, 0.0, 0.1],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0]
        ])
    res = maxmin_closure(A, maxiter=100) # matrix multiplication
    assert np.allclose(res, AT)

def test_closure_two_paths():
    '''
    Test correctedness of maxmin closure on a two disjoint paths graph
    '''
    A = np.array([
        [0.0, 0.3, 0.3, 0.0],
        [0.0, 0.1, 0.0, 0.2],
        [0.0, 0.0, 0.0, 0.4],
        [0.0, 0.0, 0.0, 0.0]
        ])
    AT = np.array([
        [0.0, 0.3, 0.3, 0.3],
        [0.0, 0.1, 0.0, 0.2],
        [0.0, 0.0, 0.0, 0.4],
        [0.0, 0.0, 0.0, 0.0]
        ])
    res = maxmin_closure(A) # matrix multiplication
    assert np.allclose(res, AT)

def test_closure_cycle_path():
    '''
    Test correctedness of maxmin closure on the following graph

     ------>-----
    |            |
    1 -- > 0 --> 2
    |            |
     ------<-----
    '''
    A = np.array([[ 0.  ,  0.  ,  0.09],
                  [ 0.01,  0.  ,  0.61],
                  [ 0.  ,  0.98,  0.  ]])

    AT = np.array([[ 0.01,  0.09,  0.09],
                   [ 0.01,  0.61,  0.61],
                   [ 0.01,  0.98,  0.61]])
    B = maxmin_closure(A)
    assert np.allclose(AT, B)
