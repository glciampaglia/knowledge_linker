import numpy as np
import scipy.sparse as sp
from nose.tools import raises

# local imports
from maxmin import _maxmin_naive, _maxmin_sparse
from cmaxmin import c_maxmin_naive, c_maxmin_sparse

def test_naive():
    A = np.random.rand(5, 5)
    AP = _maxmin_naive(A)
    AP2 = c_maxmin_naive(A) 
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

