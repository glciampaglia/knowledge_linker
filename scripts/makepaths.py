from os.path import expanduser
from truthy_measure.utils import make_weighted
from truthy_measure.ntriples import NodesIndex
from truthy_measure.closure import cclosure
#from truthy_measure._closure import shortestpath

nodespath = expanduser('~/data/dbpedia/filtered/nodes_uris.txt')
adjpath = expanduser('~/data/dbpedia/filtered/adjacency.npy')
nspath = expanduser('~/data/dbpedia/allns.tsv')
source = 'http://dbpedia.org/resource/Barack_Obama'
target = 'http://dbpedia.org/resource/Islam'
undirected = False
weight = 'degree'
kind = 'metric'

ni = NodesIndex(nodespath, nspath)
N = len(ni)
ni.node2uri = dict(((v, k) for k, v in ni.uri2node.iteritems()))
A = make_weighted(adjpath, N, undirected=undirected,
                  weight=weight)
source_node = ni.tonodeone(source)
target_node = ni.tonodeone(target)
print 'Starting'
for i in xrange(10):
    print i
    _, path = cclosure(A, source_node, target_node, retpath=True,
                       kind=kind)
#    path = shortestpath(A, source_node, target_node)
for node in path:
    print ni.node2uri[node]
