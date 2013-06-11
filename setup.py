from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from numpy import get_include

_incl = [ get_include() ]

ext_modules = [
        Extension("truthy_measure.cmaxmin", ["truthy_measure/cmaxmin.pyx"], include_dirs=_incl)
        ]

setup(
        name="truthy_measure",
        cmdclass={'build_ext' : build_ext},
        ext_modules=ext_modules
        )
