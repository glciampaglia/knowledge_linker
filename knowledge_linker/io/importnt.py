#   Copyright 2016 The Trustees of Indiana University.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

''' Converts data from N-triple format to binary format. See docstrings in the
module. '''

from __future__ import division
import os
import sys
import errno
import numpy as np
from argparse import ArgumentParser
from contextlib import closing, nested
from collections import OrderedDict
from itertools import groupby
from operator import itemgetter
from time import time
from datetime import timedelta
from codecs import EncodedFile

from .ntriples import *
from ..utils import coo_dtype

namespaces = {}

def _first_pass(path, properties=False, destination=os.path.curdir):
    '''
    Returns an ordered mapping (see `collections.OrderedDict`) of abbreviated
    entity URIs to vertex IDs, and writes them to file `nodes.txt`. Also writes
    the abbreviated triples to `triples_abbrev.nt`.
    '''
    global namespaces
    _node = '{} {}\n'
    vertices = set()
    triplesiter = iterabbrv(itertriples(path), namespaces, properties)
    num_triples = 0
    abbrevpath = os.path.join(destination, 'triples_abbrev.nt')
    with closing(open(abbrevpath, 'w')) as abbrevfile:
        for triple in triplesiter:
            out_entity, predicate, in_entity = triple
            vertices.add(out_entity)
            vertices.add(in_entity)
            num_triples += 1
            print >> abbrevfile, '{} {} {} .'.format(*triple)
    vertexmap = OrderedDict(( (entity, i) for i, entity in
        enumerate(sorted(vertices)) ))
    nodespath = os.path.join(destination, 'nodes.txt')
    with closing(open(nodespath, 'w')) as nodesfile:
        for k, v in vertexmap.iteritems():
            nodesfile.write(_node.format(v, k))
    print >> sys.stderr, 'info: abbreviated n-triples written to '\
            '{}'.format(abbrevpath)
    print >> sys.stderr, 'info: nodes written to {}'.format(nodespath)
    return vertexmap, num_triples

def _second_pass(path, vertexmap, num_triples, properties,
        destination=os.path.curdir):
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
    attribute is a comma-separated list of all predicates). Also writes the
    adjacency matrix in COO format to a NPY file to disk.
    '''
    _edge = '{} {}\n'
    triplesiter = iterabbrv(itertriples(path), namespaces, properties)
    edgespath = os.path.join(destination, 'edges.txt')
    edgesfile = open(edgespath, 'w')
    data = []
    with closing(edgesfile):
        i = 0
        # we group by source AND destination to make sure we group all parallel
        # edges in a single one. The predicates are saved as attributes. This
        # assumes that the input file is already sorted by source, destination!
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
    print >> sys.stderr, 'info: edges written to {}'.format(edgespath)
    adjpath = os.path.join(destination, 'adjacency.npy')
    np.save(adjpath, np.asarray(data, dtype=coo_dtype))
    print >> sys.stderr, 'info: adj written to {}'.format(adjpath)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('ns_file', metavar='namespaces', help='tab-separated list of namespace codes')
    parser.add_argument('nt_file', metavar='ntriples', help='N-Triples file')
    parser.add_argument('-p', '--properties', action='store_true',
            help='print properties')
    parser.add_argument('-D', '--destination', help='destination path')
    args = parser.parse_args()
    print
    print 'WARNING: the n-triples file must be already sorted by source,'\
            ' destination!'
    print
    namespaces = readns(args.ns_file)
    sys.stdout = EncodedFile(sys.stdout, 'utf-8')
    # expand destination path, check it is not an existing file, create it in
    # case it does not exist
    args.destination = os.path.expanduser(os.path.expandvars(args.destination))
    if os.path.exists(args.destination) and not os.path.isdir(args.destination):
        print >> sys.stderr, 'error: not a directory: '\
                '{}'.format(args.destination)
        sys.exit(1)
    elif not os.path.exists(args.destination):
        os.mkdir(args.destination)
        print >> sys.stderr, 'info: created {}'.format(args.destination)
    try:
        tic = time()
        vertexmap, num_triples = _first_pass(args.nt_file, args.properties,
                args.destination)
        _second_pass(args.nt_file, vertexmap, num_triples, args.properties,
                args.destination)
        toc = time()
        etime = timedelta(seconds=round(toc - tic))
        speed = num_triples / (toc - tic)
        print >> sys.stderr, 'info: {:d} triples processed in {} '\
                '({:.2f} triple/s)'.format(num_triples, etime, speed)
    except IOError, e:
        if e.errno == errno.EPIPE: # broken pipe
            sys.exit(0)
        raise
