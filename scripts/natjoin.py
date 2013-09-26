#!/usr/bin/env python

''' Perform a natural join of the first field of all input files '''

import os
from argparse import ArgumentParser
from itertools import product

parser = ArgumentParser(description=__doc__)
parser.add_argument('paths', metavar='FILE', nargs='+')
parser.add_argument('-d', '--delimiter', default='\t', 
        help='field delimiter (default: %(default)s')
parser.add_argument('-s', '--skip-lines', default=0, type=int)

def fieldsiter(f, sep='\t', field=0, skip=0):
    for i in xrange(skip):
        f.readline()
    for line in f:
        yield line.split(sep)[field]

if __name__ == '__main__':
    args = parser.parse_args()
    files = [ open(path) for path in args.paths ]
    try:
        fields = [ fieldsiter(f, args.delimiter, 0, args.skip_lines) for f in files ]
        for comb in product(*fields):
            print args.delimiter.join(comb)
    except IOError, e:
        if e.errno == os.errno.EPIPE:
            pass
    finally:
        for f in filter(None, files):
            f.close()
