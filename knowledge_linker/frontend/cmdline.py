""" Main console entry point. """

import argparse

from . import backbone
from . import linkpred
from . import batch
from . import confmatrix

_sub = {}
for mod in [backbone, linkpred, batch, confmatrix]:
    k = mod.__name__.split('.')[-1]
    _sub[k] = (mod.populate_parser, mod.main)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers()
    for k, (populate_parser, main) in _sub.items():
        subp = subparsers.add_parser(k)
        populate_parser(subp)
        subp.set_defaults(func=main)
    args = parser.parse_args()
    args.func(args)
