#!/usr/bin/env python

""" Epistemic closure batch processing script. """

import re
import os
import sys
from argparse import ArgumentParser
from contextlib import closing
import numpy as np
import pandas as pd
import scipy.sparse as sp
from datetime import datetime
from multiprocessing import Pool, cpu_count

from knowledge_linker.utils import make_weighted, WEIGHT_FUNCTIONS, load_csmatrix
from knowledge_linker.closure import epclosure

from scipy.sparse import SparseEfficiencyWarning
from warnings import filterwarnings

filterwarnings('ignore', category=SparseEfficiencyWarning)

now = datetime.now
WORKER_DATA = {}
max_tasks_per_worker = 500

parser = ArgumentParser(description=__doc__)
parser.add_argument('nodespath', metavar='uris', help='node uris '
                    '(or path to HDF5 store)')
parser.add_argument('adjpath', metavar='graph', help='adjacency matrix '
                    '(or path to CSR/CSC data)')
parser.add_argument('inputpath', metavar='input', help='input file')
parser.add_argument('outputpath', metavar='output', help='output file')
parser.add_argument('-n', '--nprocs', type=int, help='number of processes')
parser.add_argument('-u', '--undirected', action='store_true',
                    help='use the undirected network')
parser.add_argument('-k', '--kind', default='ultrametric',
                    choices=['ultrametric', 'metric'],
                    help='the kind of proximity metric')
parser.add_argument('-w', '--weight', choices=WEIGHT_FUNCTIONS,
                    default='degree',
                    help='Weight type (default: %(default)s)')
parser.add_argument('-N', '--no-closure', action='store_true',
                    help='Do not compute closure, '
                    'return weight on direct edge')


def _init_worker(A, B, kind):
    global WORKER_DATA
    WORKER_DATA['A'] = A
    WORKER_DATA['B'] = B
    WORKER_DATA['kind'] = kind
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _worker(args):
    global WORKER_DATA
    A = WORKER_DATA['A']
    B = WORKER_DATA['B']
    kind = WORKER_DATA['kind']
    source, target = args
    try:
        s = A[source, target]
        if s > 0:
            A[source, target] = 0.
            B[source, target] = 0.
        D, path = epclosure(A, source, target, B=B, kind=kind, retpath=True)
        return D, path, s > 0
    finally:
        if s > 0:
            A[source, target] = s
            B[source, target] = s


def linkpred(A, sources, targets, B=None, nprocs=None, kind='ultrametric'):
    """
    Link prediction by epistemic closure

    Parameters
    ----------
    A : array_like
        Adjacency matrix.

    sources : array_like
        Source node IDs

    targets : array_like or dict or list of lists
        Target node IDs

    nprocs : int
        The number of processes to use. Default: 90% of the available CPUs or
        2, whichever the largest.

    kind : string
        The metric type. See `knowledge_linker.closure.epclosuress`.

    Returns
    -------

    simil : array_like
        Array of similarity values

    """
    if nprocs is None:
        # by default use 90% of available processors or 2, whichever the
        # largest.
        nprocs = max(int(0.9 * cpu_count()), 2)
    A = sp.csr_matrix(A)
    if B is not None:
        B = sp.csc_matrix(B)
    else:
        B = A.tocsc()
    initargs = (A, B, kind)
    print >> sys.stderr, \
        '{}: launching pool of {} workers.'.format(now(), nprocs)
    pool = Pool(processes=nprocs, initializer=_init_worker,
                initargs=initargs, maxtasksperchild=max_tasks_per_worker)
    try:
        with closing(pool):
            result = pool.map_async(_worker, zip(sources, targets))
            while not result.ready():
                result.wait(1)
        pool.join()
        if result.successful():
            data = result.get()
        else:
            print >> sys.stderr, "There was an error in the pool."
            sys.exit(2)  # ERROR occurred
    except KeyboardInterrupt:
        print "^C"
        pool.terminate()
        sys.exit(1)  # SIGINT received
    D, paths, removed = zip(*data)
    print >> sys.stderr, '{}: done'.format(now())
    return np.asarray(D), paths, np.asarray(removed)


def main(ns):
    ## print bookkeeping information
    ns.edgetype = 'undirected' if ns.undirected else 'directed'
    print >> sys.stderr, """
{now}:
    metric: {ns.kind}
    edges:  {ns.edgetype}
    weight: {ns.weight}
    nodes:  {ns.nodespath}
    graph:  {ns.adjpath}
    input:  {ns.inputpath}
    output: {ns.outputpath}""".format(now=now(), ns=ns)

    ## load nodes list
    ext = os.path.splitext(ns.nodespath)[1]
    # load from HDFStore
    # re matches .hdf .hdf5, .h5, .h5f (and upper case combos)
    m = re.match('^\.h((df)?5?|5f)$', ext, re.IGNORECASE)
    nodes_in_store = m is not None
    if nodes_in_store:
        print >> sys.stderr, '{}: reading nodes from HDFStore..'.format(now())
        store = pd.HDFStore(ns.nodespath)
        nodes = store.get_storer('/entities').table
    # load from text file
    else:
        print >> sys.stderr, '{}: reading nodes from text file..'.format(now())
        with closing(open(ns.nodespath)) as f:
            nodes = f.readlines()

    N = len(nodes)

    ## load adjacency matrix
    if os.path.isdir(ns.adjpath):
        print >> sys.stderr, '{}: reading graph from npy file..'.format(now())
        A = load_csmatrix(os.path.join(ns.adjpath, 'csr'))
        B = load_csmatrix(os.path.join(ns.adjpath, 'csc'), fmt='csc')
    else:
        print >> sys.stderr, '{}: reading graph from text file..'.format(now())
        A = make_weighted(ns.adjpath, N, undirected=ns.undirected,
                          weight=ns.weight)
        B = None

    ## read inputs
    print >> sys.stderr, '{}: reading input..'.format(now())
    names = (
        'sfid',  # source FreeBase ID
        'sid',  # source DBpedia ID
        'stitle',  # source title
        'tfid',  # target FreeBase ID
        'tid',  # target DBpedia ID
        'ttitle',  # target title
        'rating',  # rating tuple
    )

    rateconv = lambda k: tuple(str.split(k, ','))
    convs = {'rating': rateconv}
    df = pd.read_csv(ns.inputpath, names=names, converters=convs, sep=' ')
    if ns.no_closure:
        simil = [A[df.sid[k], df.tid[k]] for k in xrange(len(df))]
        rem = False
        paths = [[] for i in xrange(len(df))]
    else:
        simil, paths, rem = linkpred(A, df.sid, df.tid, B=B, nprocs=ns.nprocs,
                                     kind=ns.kind)
    df['simil'] = simil
    df['rem'] = rem
    if nodes_in_store:
        # nodes is a PyTable Table object; its rows can be referenced by ID as
        # a normal list but it will return the entire record
        df['paths'] = [[nodes[i][2] for i in p] for p in paths]
    else:
        df['paths'] = [[nodes[i] for i in p] for p in paths]
    df.to_json(ns.outputpath)

if __name__ == '__main__':

    args = parser.parse_args()
    main(args)
