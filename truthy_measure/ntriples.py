''' Utilities for dealing with nt files '''

import re
from contextlib import closing
from codecs import EncodedFile
from gzip import GzipFile
from itertools import imap
from operator import methodcaller

def readns(path):
    '''
    Returns a dictionary mapping full namespaces URIs to abbreviated names
    '''
    with closing(open(path)) as f:
        return dict(imap(methodcaller('split','\t'), imap(str.strip, f)))

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
    patterns = map(re.compile, abbreviations)
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
            # substitute namespace with abbreviation. Abbreviate with the
            # longest matching namespace URI
            matches = filter(None, [ x.match(item) for x in patterns ])
            groups = [ m.group() for m in matches ]
            if len(groups):
                max_len, max_group, max_match = max(zip(map(len, groups),
                    groups, matches))
                abbrev = abbreviations[max_group]
                item = max_match.re.sub(abbrev + ':', item)
            # recompose the items of the form property^^<URI>
            if is_property:
                item = '^^'.join((prop, item))
            abbrvtriple.append(item)
        # skip properties by default
        if triple_has_property and not properties:
            continue
        yield tuple(abbrvtriple)

