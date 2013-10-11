#import pyximport; pyximport.install()
import scipy.sparse as scp
import numpy as np
from heapq import heappush, heappop, heapify

def bottleneck_dijkstra_undirected(G,s,t,path_needed=False):
    """
    Returns the bottleneck capacity and path between source node s and 
    target node t for a given graph G.
    """
    Q = []
    path = []
    visited = set([s])
    _,s_nbrs,_ = scp.find(G[s,:])
    if s==t or t in s_nbrs:
        if s==t:
            path.append(t)
        else:
            path = [s,t]
        bottleneck = 1
    else:
        items = {} # contains 3-tuple: distance, node, predecessor
        for n in xrange(G.shape[0]):
            if n==s:
                dist = 1
            else:
                dist = 0
            items[n] = [-dist,n,-1]
            heappush(Q,items[n])
    
        while t not in visited:
            dist_n,n,_ = heappop(Q)
            _,nbrs,_ = scp.find(G[n,:])
    
            if len(nbrs)>0:
                for nbr in nbrs:
                    dist_nbr,_,pred = items[nbr]
                    if nbr==t:
                        items[nbr][0] = -max([-dist_nbr,-dist_n])
                        if -dist_n > -dist_nbr:
                            items[nbr][2] = n
                    else:
                        if min([-dist_n,G[n,nbr]]) > -dist_nbr:
                            items[nbr][0] = -min([-dist_n,G[n,nbr]])
                            items[nbr][2] = n
                heapify(Q)
            visited = visited | {n}
        bottleneck = -items[t][0]
        
        # trace the path
        tracenode = t
        path.insert(0,t)
        while True:
            path.insert(0,items[tracenode][2])
            tracenode = items[tracenode][2]
            if tracenode == s:
                break
    if path_needed == True:
        return bottleneck, path 
    else:
        return bottleneck
