#!/usr/bin/env python

''' Converts data from N-triple format to GML '''

from __future__ import division
import re
import sys
import errno
from argparse import ArgumentParser
from contextlib import closing
from collections import deque, OrderedDict
from itertools import groupby
from operator import itemgetter
from time import time
from datetime import timedelta
from codecs import EncodedFile
from urllib import unquote_plus

namespaces = {}

def itertriples(path):
    ''' 
    iterates over an N-triples file returning triples as tuples 
    
    Parameters
    ----------
    path - path to N-triples file
    '''
    with closing(EncodedFile(open(path), 'utf-8')) as ntfile:
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

_preamble = '''graph [

    directed 1
    label "DBPedia"'''

_epilog = ']'

_node = '''
    node [
        id {}
        label "{}"
    ]'''

_edge = '''
    edge [
        source {}
        target {}
        label "{}"
    ]'''

def _first_pass(path, properties=False):
    '''
    Returns an ordered mapping (see `collections.OrderedDict`) of abbreviated
    entity URIs to vertex IDs
    '''
    global namespaces
    vertices = set()
    triplesiter = iterabbrv(itertriples(path), namespaces, properties)
    triple_no = 0
    for triple_no, triple in enumerate(triplesiter):
        out_entity, predicate, in_entity = triple
        vertices.add(out_entity)
        vertices.add(in_entity)
    vertexmap = OrderedDict(( (entity, i) for i, entity in
        enumerate(sorted(vertices)) ))
    for k, v in vertexmap.iteritems():
        print _node.format(v, unquote_plus(k))
    return vertexmap, triple_no + 1

def _second_pass(path, vertexmap, properties):
    '''
    Prints the edges list, with predicates as attributes, and collapsing all
    parallel arcs (the attribute is a comma-separated list of all predicates)
    '''
    triplesiter = iterabbrv(itertriples(path), namespaces, properties)
    for key, subiter in groupby(triplesiter, itemgetter(0,2)):
        out_entity, in_entity = key
        predicates = [ p for (oe, p, ie) in subiter ]
        out_vertex = vertexmap[out_entity]
        in_vertex = vertexmap[in_entity]
        attributes = ','.join(predicates)
        print _edge.format(out_vertex, in_vertex, attributes)

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

        print _preamble
        vertexmap, num_triples = _first_pass(args.nt_file, args.properties)
        _second_pass(args.nt_file, vertexmap, args.properties)
        print _epilog

        toc = time()

        etime = timedelta(seconds=round(toc - tic))
        speed = num_triples / (toc - tic)
        print >> sys.stderr, 'info: {:d} triples processed in {} '\
                '({:.2f} triple/s)'.format(num_triples, etime, speed)
    except IOError, e:
        if e.errno == errno.EPIPE: # broken pipe
            sys.exit(0)


