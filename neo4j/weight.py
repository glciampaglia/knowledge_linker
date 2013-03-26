from __future__ import division
from neo4j import GraphDatabase

script = '''
START root=node:vertices('U http://dbpedia.org/resource/Barack_Obama')
RETURN root
'''

STORE_DIR='/u/truthy/neo4j/dbpedia4neo/db/'
MAX = 10

if __name__ == '__main__':
    try:
        db = GraphDatabase(STORE_DIR)
        with db.transaction:
            for i,node in enumerate(db.nodes):
                if i == MAX:
                    break
                if not node.hasProperty('value'):
                    continue
                value = node['value']
                print 'node {}'.format(value),
                rels = list(node.getRelationships())
                if len(rels):
                    weight = 1 / len(rels)
                    node['weight'] = weight
                    print 'has weight {}'.format(node['weight'])
                else:
                    print 'has no relationships'
    finally:
        db.shutdown()

