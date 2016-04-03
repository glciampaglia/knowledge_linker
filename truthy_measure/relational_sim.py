import os
import sys
import numpy as np
import codecs 
import argparse

from scipy.spatial.distance import cdist

def rel_distance(m, measure='cosine'):
	"""
	Returns a distance matrix between rows of the input matrix m.
	
	Parameters:
	-----------
	m: array
		A numpy dense array

	measure: str
		The measure to use to compute distance between two row vectors of m.
		
	Returns:
	--------
	ms: array
		A distance matrix for m. Symmetric.
	"""
	if m is None:
		raise ValueError('Matrix required to compute distance.')
	if measure == 'cosine':
		ms = cdist(m, m, 'cosine')
	elif measure == 'euclidean':
		ms = cdist(m, m, 'euclidean')
	else:
		pass
	return ms

def dist_to_sim(m, measure='cosine'):
	"""
	Returns a similarity matrix for given distance matrix
	after considering the measure used to compute the distance
	values.
	"""
	if m is None:
		raise ValueError('Distance matrix required.')
	if measure == 'cosine':
		ms = np.abs( 1. - m )
	elif measure == 'euclidean':
		ms = np.abs( 1. - m/np.max(m, axis=1) ) # row-normalized by max
	return ms

if __name__ == '__main__':
	"""
	python relational_sim.py 
		-rescalcore ../../../truthy_data/iudata/rescal/iudata_rank5_Rk.npy 
		-measure euclidean -outpath ../../../truthy_data/iudata/rescal/
	"""
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('-rescalcore', type=str, required=True,
			dest='Rk', help='Path to RESCAL Rk core tensor (numpy file)')
	parser.add_argument('-measure', type=str, required=True,
			dest='measure', help='Similarity measure to be used (e.g. euclidean, cosine)')
	parser.add_argument('-outpath', type=str, required=True,
			dest='outpath', help='Path to save relational similarity matrix.')
						
	args = parser.parse_args()
	
	if not os.path.exists(args.Rk):
		raise ValueError('RESCAL core tensor Rk file does not exist.')
	if args.measure not in ('euclidean', 'cosine'):
		print "Only euclidean or cosine meaure currently supported."
		sys.exit()
	elif not os.path.exists(args.outpath):
		raise ValueError('Output file path does not exist. Please check.')

	fname = os.path.splitext(os.path.basename(os.path.abspath(args.Rk)))[0]
	fname += '_{}.npy'.format(args.measure)
	outfile = os.path.join(args.outpath, fname)
	
	# remove output file if it already exists.
	if os.path.exists(outfile):
		os.remove(outfile)
		print "** Removed file {}".format(outfile)

	print "Computing relational similarity matrix for {}".format(args.Rk)
	# read relations' row vectors to compute similarity
	A = np.load(args.Rk)
	print "Relational matrix: ", A.shape

	print "Computing {} similarity..".format(args.measure)
	sim_mat = dist_to_sim(rel_distance(A, args.measure), args.measure)
	np.save(outfile, sim_mat)
	print "{} similarity matrix saved: '{}'".format(args.measure, outfile)
