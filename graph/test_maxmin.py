import numpy as np
import scipy.sparse as sp
from nose.tools import raises

# local imports
from maxmin import _maxmin_naive, _maxmin_sparse, maxmin, pmaxmin
from cmaxmin import c_maxmin_naive, c_maxmin_sparse, c_maximum_csr

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
    np.random.seed(10)
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

def test_closure():
    pass
