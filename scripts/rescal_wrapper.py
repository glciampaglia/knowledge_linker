"""
A Python module to compute RESCAL factorization, a latent variable model
for relational data such as Semantic Web data.

For details, please see: 

"A three-way model for collective learning on multi-relational data."
by Maximilian Nickel et. al (ICML 2011).

"""

import os
import sys
import argparse
import types
import numpy as np
import matplotlib.pyplot as plt

from time import time
from scipy import interp
from itertools import izip
from sktensor import sptensor
from sktensor import rescal
from sklearn.cross_validation import KFold
from sklearn.metrics import average_precision_score, precision_recall_curve
from matplotlib.backends.backend_pdf import PdfPages

MULTIGRAPH_PATH = "../../truthy_data/iudata/edges.txt"
DATASET = 'iudata'
OUTPATH = '../../truthy_data/iudata/'
DELIM = ' '
NETWORK = None

def sptensor_to_list(X):
    "Converts a 'sptensor' in a list of matrices."
    from scipy.sparse import lil_matrix
    if X.ndim != 3:
        raise ValueError('Only third-order tensors are supported \
                        (ndim=%d)' % X.ndim)
    if X.shape[0] != X.shape[1]:
        raise ValueError('First and second mode must be of identical length')
    N = X.shape[0]
    K = X.shape[2]
    res = [lil_matrix((N, N)) for _ in range(K)]
    for n in range(len(X.vals)):
        res[X.subs[2][n]][X.subs[0][n], X.subs[1][n]] = X.vals[n]
    return res

def readNetwork(path):
    """
    Returns a sparse tensor representation of the input graph.

    The input data is arranged in a (subject, object, predicate) format.
    """
    if not os.path.exists(path):
        raise Exception('No such file exists.')
    print "\nReading network from '{}'...".format(path)
    coo = []
    nodes = set()
    relations = set()
    with open(path) as f:
        for line in f:
            coo_item = [int(x) for x in line.strip().split(DELIM)]
            if not coo:
                dim = len(coo_item)
                coo = [[] for i in xrange(dim)]

            for i, item in enumerate(coo_item):
                coo[i].append(item)
                if i in (0, 1) and item not in nodes:
                    nodes.add(item)
                if i == 2 and item not in relations:
                    relations.add(item)
    edge_count = len(coo[0]) # No. of entries/triples
    node_count = len(nodes) # No. of nodes
    relations_count = len(relations) # No. of relations
    shape = (node_count, node_count, relations_count)
    print "Network read complete.\n"
    return coo, shape, edge_count

def create_sptensor(path, full=False, nFold=5):
    """
    Tensor generator function for the graph specified by path and delimited 
    by DELIM. If 'full' is True, a sparse tensor from complete data is returned.
    Otherwise, train and test sparse tensors are returned on each call. 'nFold'
    specifies how many test sets to create for cross-validation.
    """
    global NETWORK
    if NETWORK is None:
        NETWORK = readNetwork(path) 
    else:
        print "** Note: Using cached network."
    coo, shpe, edge_count = NETWORK
    if full:
        print "Creating FULL sparse tensor.."
        S = sptensor(tuple(coo), np.ones(edge_count), 
                        shape=shpe, dtype=np.float64)
        print "------------ Full Tensor -------------"
        print "#Nodes: %s" % shpe[0]
        print "#Relations: %s" % shpe[2]
        print "#Triples/Edges: %s" % edge_count
        print "Shape of graph's tensor: {}".format(S.shape)
        print "--------------------------------------\n"
        yield S
    else:
        kf = KFold(edge_count, n_folds=nFold, shuffle=True)
        for i, (train, test) in enumerate(kf):
            print "\nFold %d" % (i + 1)
            print "Creating Train, Test sparse tensors..."
            train_coo = []
            test_coo = []
            for i in xrange(len(coo)):
                train_coo.append(np.array(coo[i])[train])
                test_coo.append(np.array(coo[i])[test])
            S_train = sptensor(tuple(train_coo), 
                                np.ones(len(train_coo[0])),
                                shape=shpe,
                                dtype=np.float64)
            S_test = sptensor(tuple(test_coo), 
                                np.ones(len(test_coo[0])),
                                shape=shpe,
                                dtype=np.float64)
            print "#NNZ Train:{}, Test:{}".format(len(S_train.vals), 
                    len(S_test.vals))
            yield S_train, S_test

def run_rescal(T, dataset, rank, outpath=os.path.curdir, 
                save=True, display=False):
    """
    Performs RESCAL factorization on the input sparse tensor and 
    returns the estimated parameters, namely matrix A, list of 
    frontal Rks and other useful information of the model estimation process
    such as number of iterations, fit value, etc. Since the matrices A and Rks
    are generally dense, they are saved on disk at 'outpath' directory. 

    See: 'A Three-Way Model for Collective Learning on Multi-Relational Data'
    by Maximilian Nickel (ICML 2011).
    
    Parameters:
    -----------
    T: sptensor
        A sparse tensor representation. See 'sktensor' library by Maximilian 
        Nickel.
    
    dataset: str
        Name of the dataset represented by tensor T.

    rank: int or list
        If rank is an 'int', a factorization of that rank is obtained. 
        (for large data)

        If rank is a 'list', series of factorizations are tried for values in
        the list, and the 'right' among them is chosen.
        (for experimental purposes)

    outpath: str
        Path of the directory where matrices A and Rks have to be stored.

    save: bool
        Whether to save the model (matrix A and matrices Rks) to disk,
        or return as-is. Default is True.

        ** Note: If save is True, matrix A is saved in .npy format. Also, 
        each frontal slice Rk is vectorized and forms a row of a matrix 
        which is then saved in .npy format. If save is False, matrix A
        and list of frontal Rks are returned as-is.

    display: bool
        Whether or not to display model fit details. Defaults to False.

    Returns:
    --------
    ret: dict
        A dictionary of the estimated parameters from RESCAL factorization.
        The keys include 'rank', A', 'Rks', 'fit', 'iter'. 
        And the values include absolute path to the files saved on disk. 
        This applies only for matrix A and Rks. 
    """
    print "Converting to list of frontal slices.."
    frontal_slices = sptensor_to_list(T)
    print "Rank: ", rank
    r = rank
    A, frontal_Rk, fval, itr, exectimes = rescal.als(frontal_slices, r)
    
    # save RESCAL model
    if save:
        if not os.path.exists(outpath):
            os.mkdir(outpath)
        base = dataset + '_rank' + str(r)
        A_path = os.path.join(outpath, base + '_A.npy')
        Rks_path = os.path.join(outpath, base + '_Rk.npy')
        frontal_Rk = np.array([rk.ravel() for rk in frontal_Rk])
        print "Saving RESCAL.."
        print "-> 'A' matrix {} at: {}".format(A.shape, A_path)
        print "-> 'R' core tensor {}, one relation per row at: {}"\
                    .format(frontal_Rk.shape, Rks_path)
        np.save(A_path, A)
        np.save(Rks_path, frontal_Rk)   

        return {'rank': r, 'A': A_path, 'Rks': Rks_path, 
            'iter': itr, 'fit': fval}

    return {'rank': r, 'A': A, 'Rks': frontal_Rk, 'iter': itr, 'fit': fval}

def model_selection(rank, nFold=5, save_plot=True):
    """
    Performs RESCAL model selection by cross-validation (CV).
    
    Parameters:
    -----------
    nFold: int
        Number of cross_validation folds. Default: 5.

    rank: int or iterable
        Rank of desired RESCAL factorization. It could be an integer or a
        list/tuple containing the distinct ranks to try.

    """
    def _cross_validation(r):
        precision = dict()
        recall = dict()
        auc_prs = dict()
        ranks = dict()
        best_fold = -1
        tensor_gen = create_sptensor(MULTIGRAPH_PATH, nFold=nFold)
        print "Computing RESCAL factorization for rank {}".format(r)

        # perform K-Fold cross-validation
        for fold, (Train, Test) in enumerate(tensor_gen):
            ret = run_rescal(Train, DATASET, outpath=OUTPATH, 
                                rank=r, save=False, display=True)
            A, Rks, ranks[fold] = ret['A'], ret['Rks'], ret['rank']

            # test on test set
            print "Testing peformance...",
            pred_scores = np.empty(len(Test.subs[0]), dtype=np.float64)
            subs, objs, preds = Test.subs
            test_triples = izip(subs, objs, preds)
            for i, (s, o, p) in enumerate(test_triples):
                A_sub = A[s,].reshape((1, len(A[s,])))
                A_obj = A[o,].reshape((len(A[o,]), 1))
                pred_scores[i] = np.dot(np.dot(A_sub, Rks[p]), A_obj)
            # print "True labels: ", Test.vals
            # print "Prediction scores: ", pred_scores

            # compute precision-recall, and AUC-PR
            precision[fold], recall[fold], _ = precision_recall_curve(Test.vals, 
                                                                    pred_scores)
            auc_prs[fold] = average_precision_score(Test.vals, pred_scores)
            if auc_prs.get(best_fold) is None \
                or auc_prs[best_fold] < auc_prs[fold]:
                best_fold = fold
            print "Done."
        
        avg_auc_pr = np.mean(auc_prs.values())
        print "\nAUC-PR @rank {}:".format(r)
        print "fold\tAUC-PR(fold)"
        for fold in xrange(nFold):
            print "{}\t{}".format(fold + 1, auc_prs[fold])
        print "Average AUC-PR @rank {}: {}".format(r, avg_auc_pr)
        print "Cross-validation complete for rank {}.".format(r)
        print "--------------------\n"
        return avg_auc_pr

    if isinstance(rank, list): # search best_rank from among the input list
        performance = []
        for rk in rank:
            perf = _cross_validation(rk)
            if perf is None:
                break
            performance.append((rk, perf))

        # Performance stats
        best = sorted(performance, key=lambda x: x[1])[::-1][0]
        best_rank, best_auc_pr = best
        metrics_path = os.path.join(OUTPATH, DATASET \
                        + '_bestrank_' + str(best_rank) + '_metrics.txt')
        print "Saving metrics at '{}'".format(metrics_path)
        with open(metrics_path, 'w') as f:
            f.write('Rank, AUC-PR\n')
            print "Rank AUC-PR"
            for r, avg_auc_pr in performance:
                print r, avg_auc_pr
                f.write('{},{}\n'.format(r, avg_auc_pr))
        print "Best (rank, AUC-PR): ({}, {})".format(best_rank, best_auc_pr)
    else: # fixed rank
        auc_pr = _cross_validation(rank)
        print "Rank: {}, AUC-PR: {}".format(rank, auc_pr)

def compute_full_rescal(rank):
    """
    Computes RESCAL factorization model on entire data. Saves the 'A'
    matrix and core tensor 'R' to disk.
    
    Parameters:
    -----------
    rank: int or iterable
        Rank of factorization. Usually, an integer between 1 and number of nodes.
        If iterable, the values in it are tried.
    
    """
    tensor_gen = create_sptensor(MULTIGRAPH_PATH, full=True)
    tensor = tensor_gen.next()
    print "Computing RESCAL factorization.."
    ret = run_rescal(tensor, DATASET, rank=rank, outpath=OUTPATH, 
                            save=True, display=True)
    A, Rks = ret['A'], ret['Rks']
    print "RESCAL factorization complete."

if __name__ == '__main__':
    """ e.g. cmd call

    -> (No model selection, full factorization)
    python rescal_wrapper.py 
        -name iudata -path ../../truthy_data/iudata/edges.txt 
        -rank 3 -outpath ../../truthy_data/iudata/

    -> (Model selection: FOAF)
    python rescal_wrapper.py -name foaf 
        -path ~/Projects/truthy_data/foafpub-umbc-2005-feb/adjacency.txt 
        -outpath ~/Projects/truthy_data/foafpub-umbc-2005-feb/rescal/ 
        -nFold 5 -rank 5

    -> (Model selection: iudata)
    python rescal_wrapper.py -name iudata 
        -path ../../truthy_data/iudata/edges.txt 
        -outpath ../../truthy_data/iudata/ 
        -full True -rank 4 -nFold 2

    -> (Multiple ranks: iudata)
    python rescal_wrapper.py -name iudata 
        -path ../../truthy_data/iudata/edges.txt 
        -outpath ../../truthy_data/iudata/ 
        -full True -rank 4 5 6 -nFold 2

    -> (Multiple ranks: FOAF)
    python rescal_wrapper.py -name foaf 
        -path ~/Projects/truthy_data/foafpub-umbc-2005-feb/adjacency.txt 
        -outpath ~/Projects/truthy_data/foafpub-umbc-2005-feb/rescal/ 
        -nFold 5
        -rank 10  20  30  40  50  60  70  80  90 100

    """
    if len(sys.argv) == 1:
        ranks = list(np.arange(7) + 1) # maxrank = 8-1 = 7 for iudata
        model_selection(ranks)
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('-name', metavar=DATASET, type=str, 
                            dest='name', help='Name of dataset', required=True)
        parser.add_argument('-path', metavar=MULTIGRAPH_PATH, type=str,
                            dest='path', help='Path to input graph file',
                            required=True)
        parser.add_argument('-rank', metavar='5', nargs='+', type=int,
                            dest='rank', help='Factorization rank', 
                            required=True)
        parser.add_argument('-nFold', metavar='5', type=int, 
                            dest='nFold', help='#Folds for CV')
        parser.add_argument('-delim', metavar='', type=str, 
                            dest='delim', help='Input file delimiter', 
                            default=' ')
        parser.add_argument('-full', metavar='True', type=bool, 
                            dest='full', help='Factorization of complete data?', 
                            default=True)
        parser.add_argument('-outpath', metavar='', type=str, 
                            dest='outpath', help='Index of class column', 
                            required=True)
                            
        args = parser.parse_args()
        
        
        DATASET = args.name
        MULTIGRAPH_PATH = args.path
        OUTPATH = args.outpath
        DELIM = args.delim.decode('string_escape')

        
        if args.nFold is not None:
            t1 = time()
            model_selection(args.rank, nFold=args.nFold)
            print "\nTime taken: {} secs.".format(time() - t1)
        else:
            for r in args.rank:
                t1 = time()
                compute_full_rescal(r)
                print "Time taken: {} secs.\n".format(time() - t1)

    print "\nDone!\n"
        