from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from numpy import get_include

_incl = [ get_include() ]

setup(
        name="truthy_measure",
        description='Graph-theoretic measures of truthiness',
        version='0.0.1pre',
        author='Giovanni Luca Ciampaglia',
        author_email='gciampag@indiana.edu',
        packages=['truthy_measure'],
        cmdclass={'build_ext' : build_ext},
        ext_modules=[
            Extension("truthy_measure.cmaxmin", ["truthy_measure/cmaxmin.pyx"],
                include_dirs=_incl, 
                extra_compile_args=['-fopenmp'],
                extra_link_args=['-fopenmp']) 
            ],
        scripts = [
            'scripts/closure.py',
            'scripts/cycles.py',
            'scripts/ontoparse.py',
            'scripts/test_dag.py',
            'scripts/prep.py',
            ]
        )
