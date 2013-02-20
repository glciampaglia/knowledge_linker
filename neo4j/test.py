from py2neo import neo4j, gremlin

g = neo4j.GraphDatabaseService("http://carl.cs.indiana.edu:7474/db/data/")

script = """
v = g.idx(T.v)[[value:'U http://dbpedia.org/resource/Barack_Obama']]
v
"""

result = gremlin.execute(script, g)
print result
