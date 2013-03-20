import sys
from datetime import datetime
from argparse import ArgumentParser, FileType
from py2neo import neo4j, gremlin
from pprint import pprint

def paths(source, target, predicate=None, maxlen=1):
    """Takes an rdf triple and performs inference or traversal and continuous value between 0 and 1
    """

    g = neo4j.GraphDatabaseService("http://carl.cs.indiana.edu:7474/db/data/")
 
# v.as('x').bothE.except(y).aggregate(y).bothV.loop('x'){it.loops < %d}{it.object.value == 'U %s'}.paths{it.value}{it.p}

    script = """
    v = g.idx('vertices')[[value:'U %s']]
    y = []
    v.as('x').outE.inV.loop('x'){it.loops <= %d}.filter{it.value == 'U %s'}.paths{it.value}{it.p}
    """

    result = gremlin.execute(script % (source, maxlen, target), g)
    return result 

def iteratepathlen(ns):
    '''
    calls paths repeatedly with maxlen argument between minlen and maxlen,
    timing the computation and writing the results to file
    '''
    print >> ns.output, "length time_in time_out time_both paths_in paths_out paths_both"
    for l in xrange(1, ns.len + 1):
        print >> ns.output, l,

        # paths from source to target following out-edges
        start_time = datetime.now()
        result = paths(ns.source, ns.target, maxlen=l)
        stop_time = datetime.now()
        print >> ns.output, stop_time - start_time,

        # paths from target to source following out-edges (equivalent to source
        # -> target following in-edges)
        start_time = datetime.now()
        result2 = paths(ns.target, ns.source, maxlen=l)
        stop_time = datetime.now()
        print >> ns.output, stop_time - start_time,
        
        # paths following both-edges
        print >> ns.output, "NA",

        # number of paths (in, out, both)
        print >> ns.output, len(result),
        print >> ns.output, len(result2),
        print >> ns.output, "NA"

        if ns.debug:
            print >> sys.stderr, "# len = %d" % l
            pprint(result, stream=sys.stderr)
            pprint(result2, stream=sys.stderr)
            print >> sys.stderr, ""

if __name__ == "__main__":
    def_source = "http://dbpedia.org/resource/Barack_Obama"
    def_target =  "http://dbpedia.org/resource/Joe_Biden"
    pred = "rdf:type"
    parser = ArgumentParser()
    parser.add_argument('-s', '--source', help='the source entity (def: %(default)s)', default=def_source)
    parser.add_argument('-t', '--target', help='the target entity (def: %(default)s)', default=def_target)
    parser.add_argument('-l', '--len', help='the maximum path length to test'\
            ' (default: %(default)s)', type=int, default=3)
    parser.add_argument('-o', '--output', help='output file',
            type=FileType('w'))
    parser.add_argument('-d', '--debug', help='print the paths to console',
            action='store_true')
    
    ns = parser.parse_args()
    iteratepathlen(ns)
