from time import time
import numpy as np
from graph_tool import Graph
from graph_tool.run_action import inline


def add_edges(graph, edges):
    '''
    Batch edges insertion. Pure Python implementation.

    Parameters
    ----------

    graph : graph_tool.Graph
        The graph to which nodes will be added. Vertices must have been already
        created.
    edges : list of int tuples
        A list of (i, j) vertex indices.
    '''
    for v, w in edges:
        graph.add_edge(v, w)


_add_code = '''
using namespace boost;

typedef adjacency_list<vecS, vecS, bidirectionalS> Graph _graph = graph;

for (int i = 0; i < n; i++)
{
    add_edge(edges[i, 0], edges[i, 1], _graph);
}
'''

def weave_add_edges(graph, edges):
    '''
    Batch edges insertion. Weave implementation.
    '''
    n = len(edges)
    edges = np.asarray(edges)
    inline(_add_code, ['graph', 'n', 'edges'], debug=True)


if __name__ == '__main__':

    N = 20
    M = 100
    graph1 = Graph()
    graph1.add_vertex(n=N)
    graph2 = Graph(graph1)
    edges = np.random.randint(0, N, (M, 2))

    print 'Python mode'
    tic = time()
    add_edges(graph1, edges)
    toc = time()
    print 'Python mode added {} edges in {} seconds'.format(M, toc - tic)

    print 'Weave mode'
    tic = time()
    weave_add_edges(graph2, edges)
    toc = time()
    print 'Weave mode added {} edges in {} seconds'.format(M, toc - tic)


