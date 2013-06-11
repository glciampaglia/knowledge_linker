from truthy_measure.maxmin import *
import networkx as nx

# maximum path length is 1

def max_path_len(a):
    g = nx.DiGraph(a)
    paths = nx.all_pairs_shortest_path_length(g)
    return max(reduce(list.__add__,
        map(dict.values, paths.values())))

print
print '-' * 80
print 

dag2 = np.array([
    [0.0, 0.1, 0.3, 0.4, 0.0],
    [0.0, 0.0, 0.1, 0.2, 0.8],
    [0.0, 0.0, 0.0, 0.1, 0.7],
    [0.0, 0.0, 0.0, 0.0, 0.6],
    [0.0, 0.0, 0.0, 0.0, 0.0]
    ])

c2 = productclosure(dag2)
print 'graph ='
print dag2
print 'closure ='
print c2
print 'max (shortest) path length: %d' % max_path_len(dag2)
print 'closure is identical to graph: %s' % np.all(dag2 == c2)

print
print '-' * 80
print 

# maximum path length is 3

dag3 = np.array([
    [0.0, 0.3, 0.0, 0.0],
    [0.0, 0.0, 0.2, 0.0],
    [0.0, 0.0, 0.0, 0.1],
    [0.0, 0.0, 0.0, 0.0]
    ])

c3 = productclosure(dag3)
print 'graph ='
print dag3
print 'closure ='
print c3
print 'max (shortest) path length: %d' % max_path_len(dag3)
print 'closure is identical to graph: %s' % np.all(dag3 == c3)
