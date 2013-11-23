""" Utilities for dealing with nt files and RDF URIs. """
import re
from contextlib import closing
from codecs import EncodedFile
from gzip import GzipFile
from collections import OrderedDict


class NodesIndex(object):

    """ URI to node mapping, URI abbreviation, etc.  """

    def __init__(self, path, nspath):
        """ Instantiate a NodesIndex object.

        Arguments
        ---------
        path : str
            path to a file with the a list of abbreviated URIs

        nspath : str
            path to a file with a list of namespace abbreviation mappings

        """
        with closing(open(path)) as f:
            # do not use csv.reader, it might mess up with commas!
            lines = (line.strip() for line in f)
            self.uri2node = OrderedDict(((u, n) for n, u in enumerate(lines)))
        self.ns = self.readns(nspath)
        self.x = re.compile('({})'.format('|'.join(self.ns.keys())))

    def __len__(self):
        return len(self.uri2node)

    def tonodeone(self, fulluri):
        """ Return int node ID from full URI. """
        try:
            u = self.abbreviateone(fulluri)
        except ValueError:
            raise KeyError(fulluri)
        return self.uri2node[u]

    def tonodemany(self, fulluris):
        """ Convert sequence of full URIs to iterator over node IDs. """
        for uri in fulluris:
            yield self.tonodeone(uri)

    def tonodefile(self, path):
        """ Convert URIs from file to node IDs.

        Parameters
        ----------
        path : str
            path to a file with a list of URIs

        Returns
        -------
        a list of node IDs.

        """
        with closing(open(path)) as f:
            return list(self.tonodemany((l.strip() for l in f)))

    def abbreviateone(self, fulluri):
        """ Return abbreviated URI. """
        m = self.x.match(fulluri)
        if m is not None:
            matchedns = m.group()
            abbrvns = self.ns[matchedns]
            return self.x.sub(abbrvns + ':', fulluri)
        else:
            raise ValueError('No abbreviation: {}'.format(fulluri))

    def abbreviatemany(self, fulluris):
        """ Iterator over abbreviated URIs. """
        for uri in fulluris:
            yield self.abbreviateone(uri)

    @staticmethod
    def readns(path):
        """ Read a file with abbreviation mappings and return a dict. """
        with closing(open(path)) as f:
            items = (tuple(l.strip().split()) for l in f)
            return dict(items)


def itertriples(path):
    '''
    Iterates over an N-triples file returning triples as tuples.

    Parameters
    ----------
    path : string
        path to N-triples file.
    '''
    if isinstance(path, file):
        ntfile = path
    else:
        if path.endswith('.gz'):
            ntfile = GzipFile(path)
        else:
            ntfile = open(path)
    ntfile = EncodedFile(ntfile, 'utf-8')
    with closing(ntfile):
        for line in ntfile:
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
                # typed property: property^^<URI>, where <URI> is same as
                # above, try to abbreviate URI and then recompose with ^^
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
