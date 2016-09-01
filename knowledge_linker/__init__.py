# Package API
from . import utils
from . import io
from . import algorithms
from .algorithms.closure import *
from .algorithms.maxmin import *
from .io.dirtree import *
from .io.ntriples import *


def _initialize():
    import warnings

    # filter out some harmless warnings about SciPy's CSR matrix format.
    warnings.filterwarnings('ignore',
            message='.*',
            module='scipy\.sparse\.compressed.*',
            lineno=122)


    # Format warnings nicely
    def _showwarning(message, category, filename, lineno, line=None):
        import sys
        warning = category.__name__
        print >> sys.stderr
        print >> sys.stderr, '>> {}: {}'.format(warning, message)
        print >> sys.stderr

    warnings.showwarning = _showwarning

_initialize()
