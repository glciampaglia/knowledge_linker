#!/usr/bin/env python

''' Converts data from N-triple format to binary format. See docstrings in the
module. '''

from __future__ import division
import re
import sys
import errno
import numpy as np
from gzip import GzipFile
from StringIO import StringIO
from argparse import ArgumentParser
from contextlib import closing, nested
from collections import deque, OrderedDict
from itertools import groupby
from operator import itemgetter
from time import time
from datetime import timedelta
from codecs import EncodedFile

from truthy_measure.utils import coo_dtype

namespaces = {}

def arrayfile(data_file, shape, descr, fortran=False):
    ''' 
    Returns an array that is memory-mapped to an NPY (v1.0) file

    Arguments
    ---------
    data_file - a file-like object opened with write mode
    shape - shape of the ndarray
    descr - any argument that numpy.dtype() can take
    fortran - if True, the array uses Fortran data order, otherwise C order
    '''
    from numpy.lib import format
    header = { 
        'descr' : descr, 
        'fortran_order' : fortran, 
        'shape' : shape
        }
    preamble = '\x93NUMPY\x01\x00'
    data_file.write(preamble)
    cio = StringIO()
    format.write_array_header_1_0(cio, header) # write header here first
    format.write_array_header_1_0(data_file, header) # write header
    cio.seek(0) 
    offset = len(preamble) + len(cio.readline()) # get offset 
    return np.memmap(data_file, dtype=np.dtype(descr), mode=data_file.mode,
            shape=shape, offset=offset)

def itertriples(path):
    ''' 
    iterates over an N-triples file returning triples as tuples 
    
    Parameters
    ----------
    path - path to N-triples file
    '''
    if path.endswith('.gz'):
        ntfile = GzipFile(path)
    else:
        ntfile = open(path)
    ntfile = EncodedFile(ntfile, 'utf-8')
    with closing(ntfile):
        for line_no, line in enumerate(ntfile):
            if line.startswith('#'):
                continue
            # remove trailing newline and dot
            line = line.strip().strip('.').strip()
            # the first two whitespaces are guaranteed to split the line
            # correctly. The trailing part may be a property containing
            # whitespaces, so using str.split is not viable.
            s1 = line.find(' ')
            s2 = line.find(' ', s1 + 1)
            triple = line[:s1], line[s1 + 1:s2], line[s2 + 1:]
            yield triple

def iterabbrv(triples, abbreviations, properties=False):
    ''' 
    returns an iterator over n-triples, with namespaces inside URI abbreviated
    to their "canonical" form. 

    Parameters
    ----------
    triples     - an iterator over n-triples as tuples
    abbreviated - a mapping of namespaces to abbreviations
    properties  - boolean; if true, yield also properties. Default is no
                  properties
    '''
    x = re.compile('({})'.format('|'.join(abbreviations.keys())))
    for triple in triples:
        abbrvtriple = []
        triple_has_property = False
        for item in triple:

            is_property = False

            # detect whether item is an entity or a property 
            if item.startswith('<'): 
                # URI-based entity: e.g. <http://www.w3.org/..>, try to
                # abbreviate it
                item = item[1:-1]
            elif item.endswith('>'): 
                # typed property: property^^<URI>, where <URI> is same as above,
                # try to abbreviate URI and then recompose with ^^
                is_property = True
                triple_has_property = True
                prop, item = item.split('^^') 
                item = item[1:-1]
            else: 
                # normal property (e.g. "Rome"@en), no abbreviation possible
                abbrvtriple.append(item)
                triple_has_property = True
                continue

            # substitute namespace with abbreviation
            m = x.match(item)
            if m is not None:
                matchedns = m.group()
                abbrvns = abbreviations[matchedns]
                item = x.sub(abbrvns + ':', item)
            
            # recompose the items of the form property^^<URI>
            if is_property:
                item = '^^'.join((prop, item))

            abbrvtriple.append(item)

        # skip properties by default
        if triple_has_property and not properties:
            continue

        yield tuple(abbrvtriple)

def _readns(path):
    global namespaces
    namespaces.clear()
    with closing(open(path)) as nsfile:
        for line in nsfile:
            ns, code = line.strip().split('\t')
            namespaces[ns] = code
    return namespaces

def _first_pass(path, properties=False):
    '''
    Returns an ordered mapping (see `collections.OrderedDict`) of abbreviated
    entity URIs to vertex IDs, and writes them to file `nodes.txt`.
    '''
    global namespaces
    _node = '{} {}\n'
    vertices = set()
    triplesiter = iterabbrv(itertriples(path), namespaces, properties)
    num_triples = 0
    for triple in triplesiter:
        out_entity, predicate, in_entity = triple
        vertices.add(out_entity)
        vertices.add(in_entity)
        num_triples += 1
    vertexmap = OrderedDict(( (entity, i) for i, entity in
        enumerate(sorted(vertices)) ))
    with closing(open('nodes.txt', 'w')) as nodesfile:
        for k, v in vertexmap.iteritems():
            nodesfile.write(_node.format(v, k))
    return vertexmap, num_triples

def _second_pass(path, vertexmap, num_triples, properties):
    '''
    Prints the edges list to file `edges.txt` and the coordinate list of the
    adjacency matrix to file `adjacency.npy`. 
    
    The adjacency matrix is written in coordinate format, e.g.: 
    
        adj[k] = i, j, 1    where e_k = (v_i, v_j)
        
    This is format is suitable for opening the file as a sparse matrix (see
    `scipy.sparse`).

    The edges list has the form: 

        k, predicate-list

    where k is the corresponding index in the adjacency matrix, and
    Predicates are written as attributes, collapsing all parallel arcs (the
    attribute is a
    comma-separated list of all predicates). Also writes the adjacency matrix in COO format to a NPY file to disk.
    '''
    _edge = '{} {}\n'
    triplesiter = iterabbrv(itertriples(path), namespaces, properties)
    edgesfile = open('edges.txt', 'w')
    data = []
    with closing(edgesfile):
        i = 0 
        for key, subiter in groupby(triplesiter, itemgetter(0,2)):
            out_entity, in_entity = key
            predicates = [ p for (oe, p, ie) in subiter ]
            out_vertex = vertexmap[out_entity]
            in_vertex = vertexmap[in_entity]
            attributes = ','.join(predicates)
            edgesfile.write(_edge.format(i, attributes))
            # default weight is 1
            data.append((int(out_vertex), int(in_vertex), 1.0))
            i += 1
    np.save('adjacency.npy', np.asarray(data, dtype=coo_dtype))

if __name__ == '__main__':

    parser = ArgumentParser(description=__doc__)
    parser.add_argument('ns_file', metavar='namespaces', help='tab-separated list of namespace codes')
    parser.add_argument('nt_file', metavar='ntriples', help='N-Triples file')
    parser.add_argument('-p', '--properties', action='store_true', 
            help='print properties')

    args = parser.parse_args()
    _readns(args.ns_file)
    sys.stdout = EncodedFile(sys.stdout, 'utf-8')

    try:
        tic = time()

        vertexmap, num_triples = _first_pass(args.nt_file, args.properties)
        _second_pass(args.nt_file, vertexmap, num_triples, args.properties)

        toc = time()

        etime = timedelta(seconds=round(toc - tic))
        speed = num_triples / (toc - tic)
        print >> sys.stderr, 'info: {:d} triples processed in {} '\
                '({:.2f} triple/s)'.format(num_triples, etime, speed)
    except IOError, e:
        if e.errno == errno.EPIPE: # broken pipe
            sys.exit(0)
        raise

