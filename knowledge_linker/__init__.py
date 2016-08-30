import sys
import warnings

# filter out some harmless warnings about SciPy's CSR matrix format.
warnings.filterwarnings('ignore',
        message='.*',
        module='scipy\.sparse\.compressed.*',
        lineno=122)


# Format warnings nicely
def _showwarning(message, category, filename, lineno, line=None):
    warning = category.__name__
    print >> sys.stderr
    print >> sys.stderr, '>> {}: {}'.format(warning, message)
    print >> sys.stderr

warnings.showwarning = _showwarning
