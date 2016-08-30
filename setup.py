""" Knowledge Linker Setup Script """

import os
import argparse
from setuptools import setup, Extension
from numpy import get_include

_incl = [get_include()]

kwargs = dict(
    name="knowledge_linker",
    description='Computational fact-checking from knowledge networks',
    version='0.1rc0',
    author='Giovanni Luca Ciampaglia and others (see CONTRIBUTORS.md)',
    author_email='gciampag@indiana.edu',
    packages=['knowledge_linker'],
    ext_modules=[
        Extension("knowledge_linker.algorithms.heap",
                  ["knowledge_linker/algorithms/heap.c"],
                  include_dirs=_incl),
        Extension("knowledge_linker.algorithms._maxmin",
                  ["knowledge_linker/algorithms/_maxmin.c"],
                  include_dirs=_incl,
                  extra_compile_args=['-fopenmp'],
                  extra_link_args=['-fopenmp']),
        Extension("knowledge_linker.algorithms._closure",
                  ["knowledge_linker/algorithms/_closure.c"],
                  include_dirs=_incl,
                  extra_compile_args=['-fopenmp'],
                  extra_link_args=['-fopenmp']),
    ],
    test_suite='nose.collector',
    tests_require='nose',
    include_package_data=True,
    install_requires=[
        'numpy',
        'scipy',
        'networkx',
        'nose >= 1.3.7',
    ],
    extras_require={
        'plotting': ['matplotlib'],
        'tensor': ['scikit-tensor','scikit-learn']
    }
#    scripts=[
#        'scripts/closure.py',
#        'scripts/ontoparse.py',
#        'scripts/prep.py',
#    ]
)

parser = argparse.ArgumentParser(description=__file__,
                                 add_help=False)
parser.add_argument('--cython', action='store_true',
                    help='Run Cython')


def replaceext(s, a='c', b='pyx'):
    """
    If the extension of the filename s is a, replace it with b.

    E.g. hello_world.c -> hello_world.pyx

    Note that comparison is case-insensitive.
    """
    sep = os.path.extsep
    fn, ext = os.path.splitext(s)
    # splitext leaves the separator `.`
    if ext.lower() == (sep + a).lower():
        return fn + sep + b
    else:
        return s

if __name__ == '__main__':
    args, rest = parser.parse_known_args()
    kwargs['script_args'] = rest

    # Run Cython on the .pyx files
    if args.cython:
        try:
            import Cython.Build
        except ImportError:
            import sys
            print >> sys.stderr, "Could not import Cython." \
                " Check that it is installed properly!"
            sys.exit(1)
        for x in kwargs['ext_modules']:
            x.sources = map(replaceext, x.sources)
        kwargs['ext_modules'] = Cython.Build.cythonize(kwargs['ext_modules'])

    setup(**kwargs)
