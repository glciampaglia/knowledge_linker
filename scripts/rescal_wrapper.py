"""
A Python module to compute RESCAL factorization, a latent variable model
for relational data such as Semantic Web data.

For details, please see: 

"A three-way model for collective learning on multi-relational data."
by Maximilian Nickel et. al (ICML 2011).

Usage notes:
There are two choices: 1) Compute factorization on entire data (full), 
2) Perform model selection by cross-validation. To perform model 
selection by cross_validation, use the '-nFold' option. If '-full' is specified,
'-nFold' is ignored. By default, models are NOT saved during cross-validation. 
If you want to save them, add '-savemodel'. By default, all models are saved 
when '-full' is specified.


"""

import os
import sys
import argparse
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import multiprocessing as mp

from numpy.linalg import norm
from time import time
from scipy import interp
from itertools import izip
from sktensor import sptensor
from sktensor import rescal
from sklearn.cross_validation import KFold
from sklearn.metrics import auc, average_precision_score, precision_recall_curve
from matplotlib.backends.backend_pdf import PdfPages

MULTIGRAPH_PATH = "../../truthy_data/iudata/edges.txt"
DATASET = 'iudata'
OUTPATH = '../../truthy_data/iudata/'
DELIM = ' '
NETWORK = None
MAX_TRIALS = 3

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
	for n in xrange(len(X.vals)):
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

def unravel(arr, shape):
	"""
	Converts a flat array into a tuple of coordinate arrays.
	"""
	i, j, k = shape
	coo = [[], [], []]
	arr = np.array(arr)
	if np.any(arr >= np.prod(shape)):
		raise ValueError('Out of bounds.')
	for ele in arr:
		org = ele
		slice_no = int(ele/(i*j))
		if slice_no >= k:
			raise ValueError('Frontal slice out of bounds.')
		coo[2].append(slice_no)
		ele = ele - slice_no * (i * j)
		row_no = int(ele/j)
		if row_no >= i:
			raise ValueError('Row out of bounds.')
		coo[0].append(row_no)
		col_no = ele - row_no * j
		if col_no >= j:
			raise ValueError('Column out of bounds.')
		coo[1].append(col_no)
	coo = tuple([np.array(e) for e in coo])
	return coo

def ravel(coo, shape):
	"""
	Converts a tuple of coordinate arrays into a flat array.
	"""
	if len(coo) != 3:
		raise ValueError('Incorrect input format.')
	if len(shape) != 3:
		raise ValueError('Only works for tensors of order 3.')
	i, j, k = shape
	coo = [np.array(e) for e in coo]
	r, c, d = coo
	if np.any(r >= i) or np.any(c >= j) or np.any(d >= k):
		raise ValueError('Out of bounds.')
	idx = []
	for a, b, e in zip(r, c, d):
		idx.append((a * j) + b + (e * i * j))
	return np.array(idx)

def get_test(folds, pos, shpe, sample_size):
	"""
	Returns positive and negative examples for a test fold.
	"""
	total_entries = np.product(shpe)	
	np.random.shuffle(pos) # shuffle positives
	pos = pos[:sample_size] 
	fsz = int(total_entries/folds)
	offset = 0
	sample_size = min(fsz, sample_size)
	for _ in xrange(folds):
		idx = np.random.randint(offset, offset + fsz, size=sample_size * 10)
		neg_idx = np.setdiff1d(idx, pos)
		np.random.shuffle(neg_idx) # shuffle negatives
		offset += fsz
		neg = neg_idx[:sample_size]
		test_idx = np.unique(np.array(list(pos) + list(neg)))
		test_idx = unravel(test_idx, shpe)
		yield test_idx

def create_sptensor(path, nFold=None, mode='CWA'):
	"""
	Tensor generator function for the graph specified by path and delimited 
	by DELIM. If 'nFold' is None, a sparse tensor from complete data is returned.
	Otherwise, train and test sparse tensors are returned on each call. 'nFold'
	specifies how many test sets to create for cross-validation.

	** IMPORTANT: This function primarily follows a 'Closed World Assumption'
	(CWA), i.e. any entries in the tensor that are 0 are considered to be 
	false.
	"""
	global NETWORK
	if NETWORK is None:
		print "Creating FULL sparse tensor.."
		coo, shpe, edge_count = readNetwork(path) 
		T = sptensor(tuple(coo), np.ones(edge_count), 
					shape=shpe, dtype=np.float64)
		print "Converting to list of frontal LIL slices.."
		t1 = time()
		S = sptensor_to_list(T)
		print "Time taken for LIL list building ", (time() - t1)
		# assert len(T.vals) == sum([S[i].nnz for i in xrange(len(S))])
		del T # NOTE: Sparse tensor is deleted, only frontal slices are kept.
		NETWORK = S, shpe, edge_count

		print "------------ Full Tensor -------------"
		print "#Nodes: %s" % shpe[0]
		print "#Relations: %s" % shpe[2]
		print "#Triples/Edges: %s" % edge_count
		print "Shape of graph's tensor: {}".format(shpe)
		print "--------------------------------------\n"
	else:
		print "** Note: Using cached network."
		S, shpe, edge_count = NETWORK
		print "Shape: ", shpe, "NNZ: ", edge_count
	
	if nFold is not None:
		if mode == 'CWA':
			print "Before pos."
			t1 = time()
			pos = np.empty(edge_count, dtype=np.int64)
			off = 0
			for k in xrange(len(S)):
				cur = S[k]
				sk_idx = np.ravel_multi_index(sp.find(cur)[:2], cur.shape) \
							+ k*np.prod(cur.shape)
				pos[off:off+len(sk_idx)] = sk_idx
				off += len(sk_idx)
			print "after pos. ", (time() - t1)
			kf = get_test(nFold, pos, shpe, int(len(pos)/nFold))
		elif mode == 'LCWA':
			pass

		print "Starting now.."
		t1 = time()
		for f, test_idx in enumerate(kf):
			print "building test:", (time() - t1)
			t1 = time()
			S_test = np.empty((len(test_idx[0]), 4), dtype=np.int64)

			# mask: set value of indices to be tested to zero
			# ground truth: S_test, each row contains test_triple and label
			for i in xrange(len(test_idx[0])):
				S_test[i, :3] = np.array([test_idx[j][i] for j in xrange(3)]) # test triple
				S_test[i, 3] = S[test_idx[2][i]][test_idx[0][i], test_idx[1][i]] # label
				S[test_idx[2][i]][test_idx[0][i], test_idx[1][i]] = 0 # mask
			print "Time to build test:", (time() - t1)
			yield S, S_test

			# unmask to return to original tensor
			for i in np.where(S_test[:, 3] == 1)[0]:
				i, j, k = S_test[i, :3]
				S[k][i, j] = 1
				# assert ravel([[i],[j],[k]], shpe) in pos

			t1 = time()
	else:
		yield S, None

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

	rank: int
		A factorization of that rank is obtained. 
		(for large data)

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
	for i in xrange(MAX_TRIALS):
		try:
			# A, frontal_Rk, fval, itr, exectimes = rescal.als(T, rank, conv=1e-3)
			A, frontal_Rk, fval, itr, exectimes = rescal.als(T, rank, 
								conv=1e-3, lambda_A=10, lambda_R=10)
			break
		except Exception, e:
			print "Error: {}".format(sys.exc_info()[0]) 
		finally:
			print "Max trials ({}) complete. Stopping execution.".format(MAX_TRIALS)
	
	
	# save RESCAL model
	if save:
		if not os.path.exists(outpath):
			os.mkdir(outpath)
		base = dataset + '_rank' + str(rank)
		A_path = os.path.join(outpath, base + '_A.npy')
		Rks_path = os.path.join(outpath, base + '_Rk.npy')
		print "Saving RESCAL.."
		print "-> 'A' matrix {} at: {}".format(A.shape, A_path)
		print "-> 'R' core tensor {}, one relation per row at: {}"\
					.format((len(frontal_Rk), frontal_Rk[0].shape[0] ** 2), 
								Rks_path)
		np.save(A_path, A)
		np.save(Rks_path, np.array([rk.ravel() for rk in frontal_Rk]))   

	return {'rank': rank, 'A': A, 'Rks': frontal_Rk, 'iter': itr, 'fit': fval}

def model_selection(rank, nFold=5, save_model=False, parallel=False):
	"""
	Performs RESCAL model selection by cross-validation (CV).

	Parameters:
	-----------
	nFold: int
		Number of cross_validation folds. Default: 5.

	rank: int or iterable
		Rank of desired RESCAL factorization. It could be an integer or a
		list/tuple containing the distinct ranks to try.

	save_model: bool
		Whether to save the model matrix A and core tensor R.

	parallel: bool
		Whether to run 'run_rescal' in parallel. Cross-validation is 
		still performed sequentially however due to potentially huge
		memory demands. Defaults to False.

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
		for fold, (T, GT) in enumerate(tensor_gen):
			print "Fold {}..".format(fold + 1)
			print "Factorization.."
			ret = run_rescal(T, DATASET, outpath=OUTPATH, 
								rank=r, save=save_model, display=True)
			A, Rks, ranks[fold] = ret['A'], ret['Rks'], ret['rank']
			
			# # predict on test set
			# print "Testing.."
			# idx = np.unique(list(GT[:,0]) + list(GT[:, 1]))
			# P = np.zeros((len(idx), len(idx), len(T)))
			# A_sub = A[idx,:]
			# for k in xrange(len(T)):
			# 	P[:, :, k] = np.dot(A_sub, np.dot(Rks[k], A_sub.T))
			# # print 'Prediction scores: \n', P
			
			# # normalize predictions
			# nrm = norm(P, axis=2) + 1e-07 # to avoid errors due to zero
			# for k in xrange(P.shape[2]):
			# 	P[:, :, k] = np.round_(P[:, :, k]/nrm, decimals=3)
			# new_idx = [[], [], []]
			# for i, j, k in zip(GT[:, 0], GT[:, 1], GT[:, 2]):
			# 	new_idx[0].append(np.where(idx == i)[0][0])
			# 	new_idx[1].append(np.where(idx == j)[0][0])
			# 	new_idx[2].append(k)
			# new_idx = np.array(new_idx)
			# pred_scores2 = P[(new_idx[0], new_idx[1], new_idx[2])]
			# # print 'Normalized scores: \n', P

			pred_scores = np.empty(GT.shape[0], dtype=np.float64)
			for z, (i, j, k) in enumerate(izip(GT[:, 0], GT[:, 1], GT[:, 2])):
				scores = np.empty(len(Rks), dtype=np.float64)
				for l in xrange(len(Rks)):
					i_vec = A[i,:].reshape((1, r))
					j_vec = A[j,:].reshape((r, 1))
					scores[l] = np.dot(i_vec, np.dot(Rks[l].reshape((r, r)), j_vec))[0,0]
				pred_scores[z] = np.round_(scores[k]/(norm(scores) + 1e-07), decimals=3)
			# assert np.allclose(pred_scores2, pred_scores)


			# compute precision-recall, and AUC-PR
			print "Metrics..",
			precision[fold], recall[fold], _ = precision_recall_curve(GT[:, 3], 
																	pred_scores)
			auc_prs[fold] = average_precision_score(GT[:, 3], pred_scores)
			if auc_prs.get(best_fold) is None \
				or auc_prs[best_fold] < auc_prs[fold]:
				best_fold = fold
			print "AUC: {}\n".format(auc_prs[fold])
		
		avg_auc_pr = np.mean(auc_prs.values())
		print "\nAUC-PR @rank {}:".format(r)
		print "fold\tAUC-PR(fold)"
		for fold in xrange(nFold):
			print "{}\t{}".format(fold + 1, auc_prs[fold])
		print "Avg. AUC-PR @rank {}: {}".format(r, avg_auc_pr)
		print "Cross-validation complete for rank {}.".format(r)
		print "--------------------\n"
		return avg_auc_pr

	if isinstance(rank, list): # search best_rank from among the input list
		performance = []
		if parallel:
			pass
			# pool = mp.Pool(mp.cpu_count())
			# result = pool.map_async(_cross_validation, rank)
			# pool.close()
			# pool.join()
			# if result.ready():
			#     print result.get()
		else:
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
	print ""

def compute_full_rescal(rank):
	"""
	Computes RESCAL factorization model on entire data. Saves the 'A'
	matrix and core tensor 'R' to disk. This function is meant for one-off
	RESCAL computation, when model selection is prohibitive or a specific
	rank factorization is desired for whatever reasons. 
	
	Parameters:
	-----------
	rank: int or iterable
		Rank of factorization. Usually, an integer between 1 and number of nodes.
		If iterable, the values in it are tried.
	
	"""
	tensor_gen = create_sptensor(MULTIGRAPH_PATH)
	T, _ = tensor_gen.next()
	print "Computing full RESCAL factorization for rank {}".format(rank)
	ret = run_rescal(T, DATASET, rank=rank, outpath=OUTPATH, 
							save=True, display=True)
	A, Rks = ret['A'], ret['Rks']
	print "RESCAL factorization complete.\n"

def main():
	print "Starting RESCAL"
	compute_full_rescal(5)

if __name__ == '__main__':
	"""  e.g. cmd call

	-> (No model selection, full factorization)
	python rescal_wrapper.py 
		-name iudata -path ../../truthy_data/iudata/edges.txt 
		-rank 3 -outpath ../../truthy_data/iudata/

	-> (Model selection: FOAF)
	python rescal_wrapper.py -name foaf 
		-path ../../truthy_data/foafpub-umbc-2005-feb/adjacency.txt 
		-outpath ../../truthy_data/foafpub-umbc-2005-feb/rescal/ 
		-nFold 5 -rank 5

	-> (Model selection: iudata)
	python rescal_wrapper.py -name iudata 
		-path ../../truthy_data/iudata/edges.txt 
		-outpath ../../truthy_data/iudata/ 
		-full -rank 4 -nFold 2

	-> (Multiple ranks: iudata)
	python rescal_wrapper.py -name iudata 
		-path ../../truthy_data/iudata/edges.txt 
		-outpath ../../truthy_data/iudata/ 
		-full -rank 4 5 6 -nFold 2 -savemodel

	-> (Multiple ranks: FOAF)
	python rescal_wrapper.py -name foaf 
		-path ../../truthy_data/foafpub-umbc-2005-feb/adjacency.txt 
		-outpath ../../truthy_data/foafpub-umbc-2005-feb/rescal/ 
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
		parser.add_argument('-full', action='store_true',
							dest='full', help='Factorization of complete data?')
		parser.add_argument('-savemodel', action='store_true',
							dest='savemodel', help='Save model (A & Rk)?')
		parser.add_argument('-outpath', metavar='', type=str, 
							dest='outpath', help='Path for storing model', 
							required=True)
							
		args = parser.parse_args()
		
		
		DATASET = args.name
		MULTIGRAPH_PATH = args.path
		OUTPATH = args.outpath
		DELIM = args.delim.decode('string_escape')

		
		if args.nFold is not None:
			t1 = time()
			model_selection(args.rank, nFold=args.nFold, 
							save_model=args.savemodel)
			print "\nTime taken: {} secs.".format(time() - t1)
		else:
			for r in args.rank:
				t1 = time()
				compute_full_rescal(r)
				print "Time taken: {} secs.\n".format(time() - t1)

	print "\nDone!\n"
		
		