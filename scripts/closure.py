#!/usr/bin/env python

import numpy as np
import scipy.sparse as sp
from argparse import ArgumentParser
from datetime import datetime
from itertools import izip

from knowledge_linker.maxmin import mmclosure_matmul
from knowledge_linker.utils import make_weighted

_now = datetime.now

if __name__ == '__main__':

    parser = ArgumentParser(description=__doc__)
    parser.add_argument('data_path', metavar='data', help='Graph data')
    parser.add_argument('nodes', type=int, help='number of nodes')
    parser.add_argument('output', help='output file')
    parser.add_argument('-p', '--procs', type=int, metavar='NUM',
            help='use %(metavar)s process for computing the transitive closure')
    parser.add_argument('-i', '--intermediate', action='store_true',
            help='save intermediate matrix to file')
    args = parser.parse_args()

    # read adjacency matrix from file and compute weights
    print "{}: read data from {}.".format(_now(), args.data_path)
    print "{}: adjacency matrix created.".format(_now())
    adj = make_weighted(args.data_path, args.nodes)
    print "{}: in-degree weights computed. Computing closure...".format(_now())

    # compute transitive closure
    if args.procs is not None:
        adjt = mmclosure_matmul(adj, parallel=True, splits=args.procs,
                nprocs=args.procs, dumpiter=args.intermediate)
    else:
        adjt = mmclosure_matmul(adj, dumpiter=args.intermediate)
    print "{}: closure algorithm completed.".format(_now())

    # save to file as records array
    adjt = adjt.tocoo()
    np.save(args.output, np.fromiter(izip(adjt.row, adjt.col, adjt.data),
        coo_dtype, len(adjt.row)))
    print "{}: closure graph saved to {}.".format(_now(), args.output)
