''' Utilities for dealing with nt files '''

import re
from contextlib import closing
from codecs import EncodedFile
from gzip import GzipFile

def itertriples(path):
    '''
    Iterates over an N-triples file returning triples as tuples.

    Parameters
    ----------
    path : string
        path to N-triples file.
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
    Iterator over n-triples, with namespaces abbreviated to their "canonical"
    form (e.g. rdf:, rdfs:, dbpedia:, etc)

    Parameters
    ----------
    triples : sequence
        An iterator over n-triples as tuples.
    abbreviated : mapping
        A mapping of namespaces to abbreviations.
    properties : bool
        If true, yield also properties. Default is no properties.
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

