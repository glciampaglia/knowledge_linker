""" Metric closure backbone """

import os
import sys
from scipy.io import mmwrite
from datetime import datetime

from ..utils import make_weighted, WEIGHT_FUNCTIONS
from .. import backbone

from scipy.sparse import SparseEfficiencyWarning
from warnings import filterwarnings

filterwarnings('ignore', category=SparseEfficiencyWarning)

now = datetime.now


def populate_parser(parser):
    parser.add_argument('nspath', metavar='ns', help='namespace abbreviations')
    parser.add_argument('adjpath', metavar='graph', help='adjacency matrix')
    parser.add_argument('outpath', metavar='output',
                        help='matrix market output file')
    parser.add_argument('N', metavar='num-nodes', type=int, help='number of nodes')
    parser.add_argument('-n', '--nprocs', type=int, help='number of processes')
    parser.add_argument('-u', '--undirected', action='store_true',
                        help='use the undirected network')
    parser.add_argument('-k', '--kind', default='ultrametric',
                        choices=['ultrametric', 'metric'],
                        help='the kind of proximity metric')
    parser.add_argument('-w', '--weight', choices=WEIGHT_FUNCTIONS,
                        default='degree',
                        help='Weight type (default: %(default)s)')
    parser.add_argument('-s', '--start', type=int, help='optional start index')
    parser.add_argument('-o', '--offset', type=int, help='offset (required with -s)')


def main(ns):
    ## print bookkeeping information
    ns.edgetype = 'undirected' if ns.undirected else 'directed'
    print >> sys.stderr, """
{now}: computing backbone:
    metric: {ns.kind}
    edges:  {ns.edgetype}
    weight: {ns.weight}
    ns:     {ns.nspath}
    graph:  {ns.adjpath}
    start:  {ns.start}
    offset: {ns.offset}""".format(now=now(), ns=ns)

    ## load adjacency matrix
    print >> sys.stderr, '{}: reading graph..'.format(now())
    A = make_weighted(os.path.expanduser(ns.adjpath), ns.N,
                      undirected=ns.undirected, weight=ns.weight)

    ## compute backbone
    B = backbone(A, kind=ns.kind, nprocs=ns.nprocs, start=ns.start,
                 offset=ns.offset)

    ## write output to matrix market file
    comment = '{ns.edgetype} {ns.weight} {ns.kind} backbone of {ns.adjpath}'.format(ns=ns)
    mmwrite(os.path.expanduser(ns.outpath), B, comment)
    print >> sys.stderr, '{}: backbone written to {}'.format(now(), ns.outpath)
