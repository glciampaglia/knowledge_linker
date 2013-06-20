from nose.tools import raises
from truthy_measure.utils import *

def test_refdict_base():
    rd = RefDict()
    rd[1] = 1
    assert rd[1] == 1
    rd[1] = 2
    assert rd[1] == 2
    assert len(rd._values) == 1
    assert len(rd._values[2]) == 1
    assert 1 not in rd._values

def test_refdict_dups():
    rd = RefDict()
    for i in xrange(10):
        rd[i] = i % 2
    assert len(rd) == 10
    assert len(rd._values) == 2

@raises(TypeError)
def test_refdict_unhashable():
    rd = RefDict()
    rd[1] = list()

def test_refdict_deletion():
    rd = RefDict()
    rd[1] = 1
    assert len(rd) == 1
    del rd[1]
    assert len(rd) == 0
    assert len(rd._values) == 0

