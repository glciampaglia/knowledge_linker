from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from numpy import get_include

_incl = [ get_include() ]

ext_modules = [
        Extension("_maxmin", ["_maxmin.pyx"], include_dirs=_incl)
        ]

setup(
        name="maxmin Cythonized function",
        cmdclass={'build_ext' : build_ext},
        ext_modules=ext_modules
        )
