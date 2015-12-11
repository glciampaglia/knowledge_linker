import os
import sys
import numpy as np

from scipy.sparse import csr_matrix

class csr_tensor(object):
	"""
	A class representing a list of CSR matrices, each representing
	a frontal slice of a tensor. For example, this tensor could be a representation
	for a multi-graph or longitudinal graph.
	"""

	def __init__(self):
		self.csr_list = []
		self.shape = None
		self.nnz = 0

	def __eq__(self, other):
		pass

	def __getitem__(self, idx):
		"""
		Simplistic implementation: requires all indices to be specified
		"""
		# print idx
		if len(idx) != 3: 
			raise ValueError('Incorrect number of indices provided.')
		i, j, k = idx
		if not ((i >= 0 and i < self.shape[0])
			and (j >= 0 and j < self.shape[1])
			and (k >= 0 and k < self.shape[2])):
			raise IndexError('Index out of bounds. Check specified indices.')
		return self.csr_list[k][i, j]

	def __setitem__(self, idx, val):
		if len(idx) != 3:
			raise ValueError('Incorrect number of indices provided.')
		i, j, k = idx
		if not ((i >= 0 and i < self.shape[0])
			and (j >= 0 and j < self.shape[1])
			and (k >= 0 and k < self.shape[2])):
			raise IndexError('Index out of bounds. Check specified indices.')
		self.csr_list[k][i, j] = val


	def fromRecs(self, path, shape=None, delim=' ', valued=False):
		"""
		Reads records from file specified by 'path' 
		and creates a csr_tensor. 

		* Note: This performs only single pass over data. 
		Assumes records are sorted by third order index. 
		
		Parameters:
		-----------
		path: str
			Path of the file from which records are to be read.

		shape: tuple
			Dimensions of the desired tensor. e.g. (3, 4, 5)

		delim: str
			Delimiter used in the specified file.

		valued: bool
			Indicates whether its an adjacency tensor or valued tensor.
			'False' indicates adjacency tensor.
		
		"""
		if not shape:
			raise ValueError('Shape not provided.')
		self.shape = shape
		if os.path.exists(path):
			data, indices, indptr = [], [], np.zeros(self.shape[0] + 1, dtype=np.int8)
			prev_i = 0
			nnz = 0
			prev_k = 0
			with open(path) as f:
				for z, line in enumerate(f):
					coords = line.strip().split(delim)
					nnz += 1
					if (valued and len(coords) != 4) or (not valued and len(coords) != 3):
						print "Incorrect number of elements on line {}.".format(z)
						raise ValueError('Only tensors of order 3 are currently supported.')
					if valued:
						i, j, k = [int(x) for x in coords[:3]]
						v = float(coords[3])
					else:
						i, j, k = [int(x) for x in coords]
						v = 1.0 # adjacency

					# create & append a CSR to the list
					if k != prev_k:
						prev_k = k
						indptr[prev_i + 1:] = nnz - 1
						M = csr_matrix((data, indices, indptr), \
								shape=self.shape[:2], dtype=np.float64)
						self.csr_list.append(M)
						self.nnz += M.nnz

						# reset
						data, indices, indptr = [], [], np.zeros(self.shape[0] + 1, \
												dtype=np.int8)
						prev_i = 0
						nnz = 1

					if i != prev_i:
						indptr[prev_i + 1] = nnz - 1
						prev_i = i
					data.append(v)
					indices.append(j)

				# add last frontal slice
				indptr[prev_i + 1:] = nnz
				M = csr_matrix((data, indices, indptr), \
						shape=self.shape[:2], dtype=np.float64)
				self.csr_list.append(M)
				self.nnz += M.nnz
		else:
			raise ValueError('Path %s does not exist.' % path)

	def __str__(self):
		if max(self.shape) > 10:
			print type(csr_tensor)
		else:
			for mat in self.csr_list:
				print "=" * 20
				print mat.todense()
		return ""

	def get_shape(self):
		"""
		Returns the shape of the tensor.
		"""
		return self.shape

	def getrow(self, idx, slice, axis=0):
		"""
		Returns a copy of the row corresponding to 'idx' from 'slice'
		along 'axis'. If axis=0 (frontal slices), this returns
		row from 'slice'-th frontal slice. Likewise, axis=1 means
		horizontal slices, and axis=2 means lateral slices.
		
		Parameters:
		-----------
		idx: int
			Index in the chosen matrix whose row is to be returned.

		slice: int
			Index of the slice (matrix) to be used.

		axis: int
			Takes value 0, 1 or 2 indicating frontal, horizontal and lateral
			slices of the tensor.
		
		Returns:
		--------
		ret: scipy.sparse.csr.csr_matrix
			Returns a copy of row 'idx' of the matrix, as a (1 x n) CSR matrix (row vector).
		"""
		# check bounds
		if axis == 0:
			if not (slice >= 0 and slice < self.shape[2]):
				raise IndexError('Slice out of bounds. Should be less than %d' 
						% self.shape[2])
			if not (idx >= 0 and idx < self.shape[0]):
				raise IndexError('Index out of bounds: Should be less than %d' 
						% self.shape[0])
			return self.csr_list[slice].getrow(idx)
		elif axis == 1:
			if not (slice >= 0 and slice < self.shape[0]):
				raise IndexError('Slice out of bounds. Should be less than %d' 
						% self.shape[0])
			if not (idx >= 0 and idx < self.shape[1]):
				raise IndexError('Index out of bounds. Should be less than %d' 
						% self.shape[1])
			data, indices, indptr = [], [], []
			for i, mat in enumerate(self.csr_list):
				if mat[slice, idx] != 0:
					data.append(mat[slice, idx])
					indices.append(i)
			indptr = [0, len(indices)]
			ret = csr_matrix((np.array(data), np.array(indices), np.array(indptr)), \
					shape=(1, self.shape[2]), dtype=np.float64)
			return ret
		elif axis == 2:
			if not (slice >= 0 and slice < self.shape[1]):
				raise IndexError('Slice out of bounds. Should be less than %d' 
						% self.shape[1])
			if not (idx >= 0 and idx < self.shape[0]):
				raise IndexError('Index out of bounds. Should be less than %d' 
						% self.shape[0])
			data, indices, indptr = [], [], []
			for i, mat in enumerate(self.csr_list):
				if mat[idx, slice] != 0:
					data.append(mat[idx, slice])
					indices.append(i)
			indptr = [0, len(indices)]
			ret = csr_matrix((np.array(data), np.array(indices), 
				np.array(indptr)), shape=(1, self.shape[2]), dtype=np.float64)
			return ret
		else:
			raise ValueError('Incorrect value for axis. It can be 0, 1 or 2.')

	def getnnz(self, axis=None):
		"""
		Count of explicitly stored values: for whole matrix (axis=None), 
		for a frontal (axis=0), horizontal (axis=1) or lateral slice (axis=2)
		
		Parameters:
		-----------
		axis: int
			Axis along which NNZs are desired.
		
		Returns:
		--------
		ret: int or array_like
			Integer if for the whole matrix, and array for other axes.
		"""
		if not axis:
			return self.nnz
		elif axis == 0:
			res = np.zeros(self.shape[2])
			for i, mat in enumerate(self.csr_list):
				res[i] = mat.getnnz()
			return res
		elif axis == 1:
			res = np.zeros(self.shape[0])
			for mat in self.csr_list:
				res += mat.getnnz(axis=1) # i'th row of each matrix
			return res
		elif axis == 2:
			res = np.zeros(self.shape[1])
			for mat in self.csr_list:
				res += mat.getnnz(0) # j'th column of each matrix
			return res
		else:
			raise ValueError('Only third order tensors supported. \
				Try axis=None, 0, 1 or 2.')

	def getslice(self, idx, axis=0):
		"""
		Returns the slice indexed by idx. frontal, horizontal or lateral slices
		are specified by axis option. 
		
		Parameters:
		-----------
		idx: int
			Index of the slice along axis.

		axis: int
			Takes value 0 (frontal), 1 (horizontal) or 2 (lateral).
		
		Returns:
		--------
		mat: csr_matrix
			A sparse CSR matrix representing the slice.
		"""
		if not (axis in (0, 1, 2)):
			raise ValueError('Incorrect axis specified. \
				Valid values include 0, 1, or 2.')
		if axis == 0:
			if not (idx >= 0 and idx < self.shape[2]):
				raise IndexError('Index out of bounds.')
			return self.csr_list[idx]
		else:
			raise NotImplementedError('Slicing for axis 1 and 2 not implemented.')

	def set_shape(self, shape):
		if self.shape is not None:
			raise NotImplementedError('Did you mean reshape? Tensor reshaping \
				not implemented.')
		self.shape = shape

	def setslice(self, idx, mat, axis=0):
		"""
		Sets the slice specified by mat as the idx'th slice 
		in the tensor, along 'axis'.

		Parameters:
		-----------
		idx: int
			Index at which the slice needs to be set.

		mat: scipy.sparse.csr.csr_matrix
			A CSR matrix representing the slice to be set.

		axis: int 
			Axis along which slice needs to be set. Takes values 0, 1 or 2.
			0 = frontal slice, 1 = horizontal slice, 2 = lateral slice
		
		"""
		if not (axis in (0, 1, 2)):
			raise ValueError('Incorrect axis specified. Valid values \
				include 0, 1, or 2.')
		if axis == 0:
			if not (idx >= 0 and idx < self.shape[2]):
				raise IndexError('Index out of bounds.')
			self.csr_list.insert(idx, csr_matrix(mat))
		else:
			raise NotImplementedError('Slicing for axis 1 and 2 not implemented.')

def main():
	path = '../../truthy_data/iudata/edges.txt'
	shape = (8, 8, 6)

	# path = '../../../truthy_data/Sample_graphs/example_graph.txt'
	# shape = (3, 3, 2)
	
	T = csr_tensor()
	T.fromRecs(path, shape=shape)
	# print T.getslice(4).todense()
	T.setslice(0, T.getslice(1))
	print T
	T[2,3,4] = 5
	print T[2,3,4]


if __name__ == '__main__':
	main()
	

