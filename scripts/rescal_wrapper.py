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

def sptensor_to_list(X):
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
    coo, shpe, edge_count = readNetwork(path) 
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
            print "Fold %d" % (i + 1)
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

def run_rescal(T, dataset, rank=None, outpath=os.path.curdir, 
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

    rank: None, int or list
        If None, "right" rank is chosen based on the quality of fit (expensive)
        by doing a grid search between 1 and number of nodes. 
        (for small data)
        
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
    frontal_slices = sptensor_to_list(T)
    if (rank is None) or isinstance(rank, list):
        print "** Rank not supplied. Model for 'best' rank will be saved."
        if rank is None:
            r_trials = np.arange(T.shape[0]-1) + 1
        else:
            r_trials = rank
        best_rank = 1
        best_fit = -1.0
        best_A = None
        best_frontal_Rk = None
        best_itr = None
        for r in r_trials:
            A, frontal_Rk, fval, itr, exectimes = rescal.als(frontal_slices, r)
            if display:
                print "Rank: {}, Fit: {}, Iter: {}".format(r, fval, itr)
            if fval >= best_fit:
                best_fit = fval
                best_rank = r
                best_A = A
                best_frontal_Rk = frontal_Rk
                best_itr = itr
        print "**Best: Rank: {}, Fit: {}, Iter: {}".format(best_rank, 
                                                            best_fit, best_itr)
        r, A, frontal_Rk, fval, itr = best_rank, best_A, \
                    best_frontal_Rk, best_fit, best_itr
    else:
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

def model_selection(nFold=5, rank=None, save_plot=True):
    """
    Performs RESCAL model selection by cross-validation (CV).
    
    Parameters:
    -----------
    nFold: int
        Number of cross_validation folds. Default: 5.

    rank: int
        Rank of desired RESCAL factorization.

    save_plot: bool
        Whether to save (Precision-Recall) curve plots while testing each folds.

    """
    tensor_gen = create_sptensor(MULTIGRAPH_PATH, nFold=nFold)
    precision = dict()
    recall = dict()
    auc_pr = dict()

    # perform K-Fold cross-validation
    for fold, (Train, Test) in enumerate(tensor_gen):
        print "Computing RESCAL factorization.."
        ret = run_rescal(Train, DATASET, outpath=OUTPATH, 
                            rank=rank, save=False, display=True)
        A, Rks = ret['A'], ret['Rks']
        print "RESCAL factorization complete."

        # test predictions
        print "Testing.."
        predictions = np.empty(len(Test.subs[0]), dtype=np.float64)
        subs, objs, preds = Test.subs
        test_triples = izip(subs, objs, preds)
        for i, (s, o, p) in enumerate(test_triples):
            A_sub = A[s,].reshape((1, len(A[s,])))
            A_obj = A[o,].reshape((len(A[o,]), 1))
            predictions[i] = np.dot(np.dot(A_sub, Rks[p]), A_obj)
        print "True labels: ", Test.vals
        print "Prediction scores: ", predictions

        # compute precision-recall, and AUC-PR
        print "Computing performance stats.."
        precision[fold], recall[fold], _ = precision_recall_curve(Test.vals, 
                                                                predictions)
        auc_pr[fold] = average_precision_score(Test.vals, predictions)

        print "--------------------\n"

    # Performance stats
    print "\nOverall performance:"
    print "AUC-PRs: ", auc_pr.values()
    print "Average AUC-PR: {}".format(np.mean(auc_pr.values()))

    # Save plot
    if save_plot is not None:
        plot_path = os.path.join(OUTPATH, DATASET + '_pr.pdf')
        pdf = PdfPages(plot_path)
        print "\nSaving PR-curve at '{}'..".format(plot_path)
        plt.clf()
        for fold in xrange(nFold):
            plt.plot(recall[fold], precision[fold],
                 label='Fold %d (AUC-PR = %0.3f)' % (fold + 1, auc_pr[fold]))

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title("'{}': Precision-Recall curves for each fold.".format(DATASET))
        plt.legend(loc="lower left")
        plt.savefig(pdf, format='pdf')
        pdf.close()

def compute_full_rescal(rank):
    """
    Computes RESCAL factorization model on entire data. Saves the 'A'
    matrix and core tensor 'R' to disk.
    
    Parameters:
    -----------
    rank: int
        Rank of factorization. An integer between 1 and number of nodes.
        If None, a grid search is performed to select the 'best' rank.
    
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

    -> python rescal_wrapper.py 
        -name iudata -path ../../truthy_data/iudata/edges.txt 
        -rank 3 -outpath ../../truthy_data/iudata/

    -> python rescal_wrapper.py -name foaf 
        -path ~/Projects/truthy_data/foafpub-umbc-2005-feb/adjacency.txt 
        -outpath ~/Projects/truthy_data/foafpub-umbc-2005-feb/rescal/ 
        -nFold 5

    -> python rescal_wrapper.py -name iudata 
        -path ../../truthy_data/iudata/edges.txt 
        -outpath ../../truthy_data/iudata/ 
        -full True -rank 4 -nFold 2

    """
    if len(sys.argv) == 1:
        model_selection()
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('-name', metavar=DATASET, type=str, 
                            dest='name', help='Name of dataset', required=True)
        parser.add_argument('-path', metavar=MULTIGRAPH_PATH, type=str,
                            dest='path', help='Path to input graph file',
                            required=True)
        parser.add_argument('-rank', metavar='5', type=int,
                            dest='rank', help='Factorization rank')
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

        t1 = time()
        if args.nFold is not None:
            model_selection(nFold=args.nFold, rank=args.rank)
        else:
            compute_full_rescal(args.rank)
        print "\nTime taken: {} secs.".format(time() - t1)

    print "\nDone!\n"
        