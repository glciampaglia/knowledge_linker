""" Compute confusion matrices using edge removal. """

import sys
from contextlib import closing
import numpy as np
import pandas as pd
import scipy.sparse as sp
from datetime import datetime
import traceback
from multiprocessing import Pool, cpu_count

from ..utils import make_weighted, WEIGHT_FUNCTIONS
from ..algorithms.closure import epclosure, epclosuress

from scipy.sparse import SparseEfficiencyWarning
from warnings import filterwarnings

filterwarnings('ignore', category=SparseEfficiencyWarning)

now = datetime.now
WORKER_DATA = {}
max_tasks_per_worker = 500


def populate_parser(parser):
    parser.add_argument('nodespath', metavar='uris', help='node uris')
    parser.add_argument('adjpath', metavar='graph', help='adjacency matrix')
    parser.add_argument('sourcespath', metavar='source', help='sources input file')
    parser.add_argument('targetspath', metavar='target', help='targets input file')
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


def _init_worker(A, kind, targets):
    global WORKER_DATA
    B = A.tocsc()
    WORKER_DATA['A'] = A
    WORKER_DATA['B'] = B
    WORKER_DATA['kind'] = kind
    WORKER_DATA['targets'] = targets
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _worker(source):
    try:
        global WORKER_DATA
        A = WORKER_DATA['A']
        B = WORKER_DATA['B']
        kind = WORKER_DATA['kind']
        targets = WORKER_DATA['targets']
        # first, compute closure to all targets
        D, _ = epclosuress(A, source, B=B, kind=kind)
        D = D[targets]
        # then, check if any element needs to be recomputed without its edge
        # removal
        idx, = np.where(D == 1.0)  # direct neighbors, by definition
        for i in idx:
            target = targets[i]
            s = A[source, target]
            A[source, target] = 0.
            B[source, target] = 0.
            A.eliminate_zeros()
            B.eliminate_zeros()
            d, _ = epclosure(A, source, target, B=B, kind=kind)
            D[i] = d
            A[source, target] = s
            B[source, target] = s
        return D
    except Exception:
        val, ty, tb = sys.exc_info()
        traceback.print_tb(tb)
        raise


def confmatrix(A, sources, targets, nprocs=None, kind='ultrametric'):
    """
    Confusion matrix with edge removal

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
    # allocate array to be passed as shared memory
    print >> sys.stderr, \
        '{}: copying graph data to shared memory.'.format(now())
    A = sp.csr_matrix(A)
    initargs = (A, kind, targets)
    print >> sys.stderr, \
        '{}: launching pool of {} workers.'.format(now(), nprocs)
    pool = Pool(processes=nprocs, initializer=_init_worker,
                initargs=initargs, maxtasksperchild=max_tasks_per_worker)
    try:
        with closing(pool):
            result = pool.map_async(_worker, sources)
            while not result.ready():
                result.wait(1)
        pool.join()
        if result.successful():
            print >> sys.stderr, '{}: done'.format(now())
            return result.get()
        else:
            print >> sys.stderr, "{}: There was an error in "\
                "the pool.".format(now())
            sys.exit(2)  # ERROR occurred
    except KeyboardInterrupt:
        print "^C"
        pool.terminate()
        sys.exit(1)  # SIGINT received


def main(ns):
    ## print bookkeeping information
    ns.edgetype = 'undirected' if ns.undirected else 'directed'
    print >> sys.stderr, """
{now}:
    metric:   {ns.kind}
    edges:    {ns.edgetype}
    weight:   {ns.weight}
    nodes:    {ns.nodespath}
    graph:    {ns.adjpath}
    sources:  {ns.sourcespath}
    targets:  {ns.targetspath}
    output:   {ns.outputpath}""".format(now=now(), ns=ns)

    ## load nodes list
    print >> sys.stderr, '{}: reading nodes..'.format(now())
    with closing(open(ns.nodespath)) as f:
        nodes = f.readlines()

    ## load adjacency matrix
    print >> sys.stderr, '{}: reading graph..'.format(now())
    A = make_weighted(ns.adjpath, len(nodes), undirected=ns.undirected,
                      weight=ns.weight)

    ## read inputs
    print >> sys.stderr, '{}: reading input..'.format(now())
    names = (
        'node_id',  # source DBpedia ID
        'node_title',  # source title
    )

    sf = pd.read_csv(ns.sourcespath, names=names, sep=' ')
    tf = pd.read_csv(ns.targetspath, names=names, sep=' ')
    B = confmatrix(A, sf.node_id, tf.node_id.values, nprocs=ns.nprocs,
                   kind=ns.kind)
    colnames = tf['node_title'].map(lambda k: k.split('/')[-1])
    outf = pd.concat([sf, pd.DataFrame(B, columns=colnames)], axis=1)
    outf.to_csv(ns.outputpath, encoding='utf-8')
