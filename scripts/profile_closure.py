import pyximport
pyximport.install()
import os
import numpy as np
import pstats, cProfile
from truthy_measure.utils import coo_dtype, recstosparse
from truthy_measure.maxmin import closure
from truthy_measure.cmaxmin import c_closure, c_closure_rec


N = 6463
path = os.path.expanduser('~/data/dbpedia/adjacency_test30k.npy')
coords = np.load(path)
A = recstosparse(coords, (N, N))

cProfile.runctx("root1, succ1, outpath = closure(A, ondisk=True, outpath='test.h5')", 
        globals(), locals(), "Profile.prof")
s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()

# %timeit root1, succ1 = c_closure(A)
# %timeit root2, succ2 = c_closure_rec(A)
# %timeit root3, succ3, outpath = closure(A)
# assert np.allclose(root1, root2)
# assert np.allclose(root1, root3)
