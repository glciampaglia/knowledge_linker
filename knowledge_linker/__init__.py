#   Copyright 2016 The Trustees of Indiana University.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

#   Copyright 2016 The Trustees of Indiana University.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

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
