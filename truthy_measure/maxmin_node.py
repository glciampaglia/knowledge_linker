import numpy as np
from heapq import heappush, heappop, heapify

from truthy_measure.cmaxmin_node import \
        bottlenecknodefull as cbottlenecknodefull,\
        bottlenecknodest as cbottlenecknodest,\
        bottlenecknode as cbottlenecknode

def bottlenecknodefull(G, retpath=False):
    rows = []
    paths = []
    N = G.shape[0]
    for s in xrange(N):
        row, p = bottlenecknode(G, s, retpath=retpath)
        rows.append(row)
        paths.append(p)
    return np.asarray(rows), paths

def bottlenecknode(G, s, targets=None, retpath=False):
    '''
    Return the bottleneck capacity and paths from node s. If no targets are
    specified, will return the capacities for all possible targets in the graph.
    '''
    N = G.shape[0]
    if targets is None:
        targets = xrange(N)
    capacities = []
    paths = []
    for t in targets:
        c, p = bottlenecknodest(G, s, t, retpath=retpath)
        paths.append(p)
        capacities.append(c)
    capacities = np.asarray(capacities)
    return capacities, paths

def bottlenecknodest(G, s, t, retpath=False):
    """
    Returns the bottleneck capacity and path between source node s and 
    target node t for a given graph G.
    """
    def _neighbors(node, G):
        return G.indices[G.indptr[node]:G.indptr[node+1]]
    Q = []
    path = []
    connected = set([s])
    s_nbrs = _neighbors(s, G)
    if s==t or t in s_nbrs:
        path = []
        bottleneck = 1.0
    else:
        items = {} # contains 3-tuple: distance, node, predecessor
        # init the heap
        for n in xrange(G.shape[0]):
            if n==s:
                dist = 1.0
            else:
                dist = 0.0
            items[n] = [-dist, n, -1]
            heappush(Q, items[n])
        while t not in connected or len(Q) > 0:
            dist_n, n, _ = heappop(Q)
            nbrs = _neighbors(n, G)
            if len(nbrs)>0:
                for nbr in nbrs:
                    if nbr not in connected:
                        dist_nbr, _, pred = items[nbr]
                        if nbr==t:
                            items[nbr][0] = -max([-dist_nbr, -dist_n])
                            if -dist_n > -dist_nbr:
                                items[nbr][2] = n
                        else:
                            if min([-dist_n, G[n, nbr]]) > -dist_nbr:
                                items[nbr][0] = -min([-dist_n, G[n, nbr]])
                                items[nbr][2] = n
                heapify(Q)
            connected.add(n)
        bottleneck = -items[t][0]
        if retpath:
            # trace the path
            tracenode = t
            path.insert(0, t)
            while True:
                predecessor = items[tracenode][2]
                if predecessor >= 0:
                    path.insert(0, predecessor)
                    tracenode = predecessor
                    if tracenode == s:
                        break
                else:
                    path = []
                    break
    if retpath == True:
        return bottleneck, np.asarray(path)
    else:
        return bottleneck, None
