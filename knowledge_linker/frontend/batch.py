""" Batch processing script """

import os
import sys
from contextlib import closing
import numpy as np
import scipy.sparse as sp
import pandas as pd
from datetime import datetime
from multiprocessing import Pool, Array, cpu_count
from ctypes import c_int, c_double

from ..io.ntriples import NodesIndex
from ..utils import make_weighted, WEIGHT_FUNCTIONS
from ..algorithms.closure import _init_worker as _clo_init_worker, \
    epclosuress

now = datetime.now


def populate_parser(parser):
    parser.add_argument('nspath', metavar='ns', help='namespace abbreviations')
    parser.add_argument('nodespath', metavar='uris', help='node uris')
    parser.add_argument('adjpath', metavar='graph', help='adjacency matrix')
    parser.add_argument('sourcespath', metavar='sources', help='input sources')
    parser.add_argument('targetspath', metavar='targets', help='input targets')
    parser.add_argument('-S', '--skip', type=int, default=0, metavar='LINES')
    parser.add_argument('-n', '--nprocs', type=int, help='number of processes')
    parser.add_argument('-u', '--undirected', action='store_true',
                        help='use the undirected network')
    parser.add_argument('-k', '--kind', default='ultrametric',
                        choices=['ultrametric', 'metric'],
                        help='the kind of proximity metric')
    parser.add_argument('-N', '--no-closure', action='store_true',
                        help='Do not compute a closure, use the base graph')
    parser.add_argument('-w', '--weight', choices=WEIGHT_FUNCTIONS,
                        default='degree',
                        help='Weight type (default: %(default)s)')
    parser.add_argument('-s', '--sep', default=',',
                        help='field separator (default: ,)')
    parser.add_argument('-H', '--header', nargs='+', help='Column names',
                        dest='names')

_B = None


def _init_worker(kind, indptr, indices, data, shape):
    global _B
    _clo_init_worker(kind, indptr, indices, data, shape)
    from knowledge_linker.closure import _A
    _B = _A.tocsc()


def _worker(source):
    global _B
    from knowledge_linker.closure import _A, _kind  # global
    D, _ = epclosuress(_A, source, B=_B, kind=_kind)
    return D

max_tasks_per_worker = 500


def islistoflists(a):
    """
    Check if argument is a list of lists.

    NOTE: If non-empty, will check only the first element.

    """
    try:
        if len(a):
            return isinstance(a[0], list)
        return isinstance(a, list)
    except TypeError:
        return False


def _check_inputs(targets, sources):
    """ Check inputs.
    """
    if not (isinstance(targets, np.ndarray)
            or isinstance(targets, dict)
            or islistoflists(targets)):
        raise ValueError('targets must be either ndarray, dict or '
                         'list of lists.')
    if isinstance(targets, list) and len(sources) != len(targets):
        raise ValueError('sources and targets must have same len')


def _make_return(D, targets, sources):
    """ Make return values.
    """
    if isinstance(targets, np.ndarray):
        return D[:, targets]
    elif isinstance(targets, dict):
        c = {}
        for s in sources:
            try:
                c[s] = D[s, targets[s]]
            except KeyError:
                c[s] = D[s, :]
        return c
    elif islistoflists(targets):
        c = []
        for i in xrange(len(sources)):
            c.append(D[sources[i], targets[i]])
        return c
    else:
        # this should *never* happen, per the initial check.
        raise RuntimeError('targets is instance of unknown class.')


def epclosurebatch(A, sources, targets, nprocs=None, kind='ultrametric'):
    """
    Compute a batch of epistemic jobs, one for each source.

    Parameters
    ----------
    A : array_like
        Adjacency matrix.

    sources : array_like
        Batch of source nodes.

    targets : array_like or dict or list of lists
        The target nodes. Accepts several formats:

        If array_like, then for each source return only the values of targets.

        If dict, then must for each source s return only the targets associated
        to s (i.e. targets[s]). If KeyError is raised will return all targets.

        If list of lists, then must have the same length of `sources`, and for
        the i-th source will return only the i-th list of targets (i.e.
        targets[i]).

    nprocs : int
        The number of processes to use. Default: 90% of the available CPUs or
        2, whichever the largest.

    kind : string
        The metric type. See `knowledge_linker.closure.epclosuress`.

    Returns
    -------

    closure : array_like or dict or list of lists
        The epistemic closure values. Format is the same as the one of the
        `targets` parameter.

    """
    _check_inputs(targets, sources)
    if nprocs is None:
        # by default use 90% of available processors or 2, whichever the
        # largest.
        nprocs = max(int(0.9 * cpu_count()), 2)
    # allocate array to be passed as shared memory
    print >> sys.stderr, \
        '{}: copying graph data to shared memory.'.format(now())
    A = sp.csr_matrix(A)
    indptr = Array(c_int, A.indptr)
    indices = Array(c_int, A.indices)
    data = Array(c_double, A.data)
    initargs = (kind, indptr, indices, data, A.shape)
    print >> sys.stderr, \
        '{}: launching pool of {} workers.'.format(now(), nprocs)
    pool = Pool(processes=nprocs, initializer=_init_worker,
                initargs=initargs, maxtasksperchild=max_tasks_per_worker)
    with closing(pool):
        D = pool.map(_worker, sources)
    pool.join()
    print >> sys.stderr, '{}: done'.format(now())
    return _make_return(np.asarray(D), targets, sources)


def epnoclosure(A, sources, targets):
    """
    Compute the baseline on the original graph (i.e. no closure is performed).

    See `epclosurebatch` for parameters and return value format.
    """
    _check_inputs(targets, sources)
    N = A[sources, :]
    if sp.isspmatrix(N):
        N = N.todense()
    return _make_return(N, targets, sources)


def main(ns):
    ## print bookkeeping information
    ns.edgetype = 'undirected' if ns.undirected else 'directed'
    print >> sys.stderr, """
{now}:
    no closure: {ns.no_closure}
    metric: {ns.kind}
    edges:  {ns.edgetype}
    weight: {ns.weight}
    ns:     {ns.nspath}
    nodes:  {ns.nodespath}
    graph:  {ns.adjpath}
    source: {ns.sourcespath}
    target: {ns.targetspath}""".format(now=now(), ns=ns)

    ## load index
    print >> sys.stderr, '{}: reading URIs..'.format(now())
    ni = NodesIndex(os.path.expanduser(ns.nodespath),
                    os.path.expanduser(ns.nspath))

    ## load adjacency matrix
    print >> sys.stderr, '{}: reading graph..'.format(now())
    N = len(ni)
    A = make_weighted(ns.adjpath, N, undirected=ns.undirected,
                      weight=ns.weight)

    ## read sources
    print >> sys.stderr, '{}: reading sources..'.format(now())
    sf = pd.read_csv(os.path.expanduser(ns.sourcespath), sep=ns.sep,
                     names=ns.names)
    if 'node_id' not in sf:
        sf['node_id'] = list(ni.tonodemany(sf['node_title']))
    if sf['node_id'].isnull().any():
        print >> sys.stderr, "Missing sources:"
        print >> sys.stderr, sf[sf['node_id'].isnull()]['node_title']
    sf = sf.dropna()
    sources = sf['node_id'].values

    ## read targets
    tf = pd.read_csv(os.path.expanduser(ns.targetspath), sep=' ',
                     names=ns.names)
    if 'node_id' not in tf:
        tf['node_id'] = list(ni.tonodemany(tf['node_title']))
    if tf['node_id'].isnull().any():
        print >> sys.stderr, "Missing targets:"
        print >> sys.stderr, tf[tf['node_id'].isnull()]['node_title']
    tf = tf.dropna()
    targets = tf['node_id'].values

    ## compute closure
    if ns.no_closure:
        ## compute noclosure
        print >> sys.stderr, '{}: computing noclosure..'.format(now())
        B = epnoclosure(A, sources, targets)
    else:
        ## compute closure
        print >> sys.stderr, '{}: computing closure..'.format(now())
        B = epclosurebatch(A, sources, targets, nprocs=ns.nprocs,
                           kind=ns.kind)

    ## write output to CSV
    colnames = tf['node_title'].map(lambda k: k.split('/')[-1])
    outf = pd.concat([sf, pd.DataFrame(B, columns=colnames)], axis=1)
    outf.to_csv(sys.stdout, encoding='utf-8')
