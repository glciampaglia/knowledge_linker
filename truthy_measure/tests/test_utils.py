from nose.tools import raises
from truthy_measure.utils import *

def test_cache():
    c = Cache(2)
    c[0] = 0
    c[1] = 1
    assert len(c) == 2
    c[2] = 2
    assert len(c) == 2

