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

""" Main console entry point. """

import argparse

from . import backbone
from . import linkpred
from . import batch
from . import confmatrix

_sub = {}
for mod in [backbone, linkpred, batch, confmatrix]:
    k = mod.__name__.split('.')[-1]
    _sub[k] = (mod.populate_parser, mod.main, mod.__doc__)


def main():
    descr = 'Knowledge linker: computational fact checking from knowledge networks'
    parser = argparse.ArgumentParser(description=descr)
    subparsers = parser.add_subparsers()
    for k, (populate_parser, main, doc) in _sub.items():
        subp = subparsers.add_parser(k, help=doc)
        populate_parser(subp)
        subp.set_defaults(func=main)
    args = parser.parse_args()
    args.func(args)
