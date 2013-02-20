from py2neo import neo4j, gremlin

def truthiness(entity, predicate, object):
    """Takes an rdf triple and performs inference or traversal and continuous value between 0 and 1
    """

    g = neo4j.GraphDatabaseService("http://carl.cs.indiana.edu:7474/db/data/")
    
    script = """
    v = g.idx(T.v)[[value:'U http://dbpedia.org/resource/Barack_Obama']]
    v
    """
    
    result = gremlin.execute(script, g)

    return result

if __name__ == "__main__":
    entity = "http://dbpedia.org/resource/Barack_Obama"
    pred = "rdf:type"
    o =  "http://dbpedia.org/resource/President_of_the_United_States"
    result = truthiness(entity, pred, o)
    print result
