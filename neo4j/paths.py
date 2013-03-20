import sys
from datetime import datetime
from argparse import ArgumentParser, FileType
from py2neo import neo4j, gremlin
from pprint import pprint

SCRIPT = """
source = 'U %s';
target = 'U %s';
maxlen = %d;
v = g.idx('vertices')[[value:source]];
v.as('x').bothE.bothV.loop('x'){it.loops <= maxlen}.filter{it.value == target}.simplePath.paths{it.value}{it.p}
"""

NEO_URL = "http://carl.cs.indiana.edu:7474/db/data/"

def paths(source, target, predicate=None, maxlen=1):
    """Takes an rdf triple and performs inference or traversal and continuous value between 0 and 1
    """
    global SCRIPT, NEO_URL

    g = neo4j.GraphDatabaseService(NEO_URL)
 
    result = gremlin.execute(SCRIPT % (source, target, maxlen), g)
    return result 

def iteratepathlen(ns):
    '''
    calls paths repeatedly with maxlen argument between minlen and maxlen,
    timing the computation and writing the results to file
    '''
    print >> ns.output, "length time paths"
    for l in xrange(1, ns.len + 1):
        # print length
        print >> ns.output, l,

        # paths from source to target following out-edges
        start_time = datetime.now()
        result = paths(ns.source, ns.target, maxlen=l)
        stop_time = datetime.now()
        print >> ns.output, stop_time - start_time,
        print >> ns.output, len(result),

        if ns.debug:
            print >> sys.stderr, "# len = %d" % l
            pprint(result, stream=sys.stderr)
            print >> sys.stderr, ""
            sys.stderr.flush()

if __name__ == "__main__":
    def_source = "http://dbpedia.org/resource/Barack_Obama"
    def_target =  "http://dbpedia.org/resource/Joe_Biden"
    pred = "rdf:type"
    parser = ArgumentParser()
    parser.add_argument('-s', '--source', help='the source entity (def: %(default)s)', 
            default=def_source)
    parser.add_argument('-t', '--target', help='the target entity (def: %(default)s)', 
            default=def_target)
    parser.add_argument('-l', '--len', help='the maximum path length to test'\
            ' (default: %(default)s)', type=int, default=1)
    parser.add_argument('-o', '--output', help='output file',
            type=FileType('w'))
    parser.add_argument('-d', '--debug', help='print the paths to console',
            action='store_true')
    
    ns = parser.parse_args()
    iteratepathlen(ns)
