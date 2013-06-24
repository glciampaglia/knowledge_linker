Project logbook
===============

Somehow I figured out that instead of writing long, abstruse emails to my collaborators,
it's simpler to write them here. 

## Tue Jun 18 18:52:27 EDT 2013

The code for doing the maxmin transitive closure on the cyclical graphs works
correctly and it's now running on the full dbpedia graph on Smithers. I took me
some time to adapt the algorithm by Nuutila et Soisalon-Soininen because it was
choking on the memory usage: the set of successor sets (i.e. for any node, the
list of node that can be reached by paths of finite length from that node) is
just too big to be stored, both on memory and on disk. Luckily I could trade
memory for computation time by computing only the set of reachable SCCs from
each SCC. You may ask why we need to compute such a monster and it's simply
because in this way we can prune a path whenever we end up in a node from which
it is impossible to reach the target (and also because we also limit the
all-pairs path computations only to the nodes that we can actually reach).

In the meanwhile, to understand a bit more about the successors set, after
Sandro's suggestion I also plotted the SCC size distribution, which is attached
(CDF and log-binned PDF). Note that the slopes do not match very much (for the
CDF is roughly 2 while for the log-binned density it's roughly 4). It may be a
problem due to the binning but anyway what is important is the large number of
SCCs of size 1. It's almost 95% of the nodes. This could be a problem if many of
those SCCs are actually reachable, because then we don't save much by storing
only the SCC root rather than the whole SCC, but my hunch is that most of them
are probably not reachable (e.g. leaves of the DAG hierarchy). 

## Wed Jun 19 14:01:12 EDT 2013

Bad news. It still takes all the memory on smithers (a total of 144GB). I am
going to compute the successors for 1000 at random and see what the size
distribution of these sets is. 

## Thu Jun 20 18:57:00 EDT 2013

The analysis of the successors shows that, on a sample of 132K successor sets
collected from starting the traversal at 1000 random source nodes, only a
small minority of the successors sets (3.4%) has more than 100k elements. These
are *SCC successor sets*, that is, I store only the roots of the SCC that can be
reached by any SCC. Still, it is not clear what is taking most of the space:
whether the vast majority of small sets, or the small minority of large sets.
Also, that starting from 1000 sources we collected 132K successor sets, that is,
we encountered 132K SCCs, is interesting in itself. 

My hypothesis is that the network has a huge bowtie structure, and that there is
a component (perhaps a huge one), that: 1) has a large set of successors, and 2)
has a large set of predecessors. That means that, for the exception of the
themselves, many SCCs share the same set of successors. This intuition was
confirmed by looking at the set themselves. I dumped them to a text file and
looked at the largest sets. They are indeed identical (again, with the exception
of itself, that is for any vertex/component $v$, $v \in succ(v)$).

If this is the case, then there is room for reducing the usage of memory. I
implemented a Python dictionary class that stores references to values, without
keeping duplicates. I then modified the code of `transitive_closure` so that now
the successors set of a SCC root does not contain itself and integrated the new
dictionary class. I did a test running the same successor sets sampling code,
but starting from 100 sources only (it takes less time than with 1000 and I had
tested that too with the old code). The result is that roughly 124K SCC are
encountered and the same number of successor sets is collected, but now only
3200 sets are effectively stored in memory, i.e. it stores 97% less set objects
than before. How this translates to normal memory is difficult to compute, but I
plotted the number of duplicates as a function of the cardinality of the sets
and there are good savings even for the very big sets. 

I relaunched the closure script on the full graph. Keep fingers crossed!

## Fri Jun 21 11:57:29 EDT 2013

Unfortunately it did not work. The script took up all the memory on smithers.
This made the tweet filtering process running on smithers crash at around 7.44
AM today. Bruce rebooted smithers and now the data collection is back online. On
my side, the script was still in `closure_cycles`.

This means that there is no way to keep in memory the successor sets of all
SCCs. I have to change the algorithm so that it does without the full set of all
successor sets, and instead uses only the successors set of the current source.
In particular, we cannot prune the DFS search tree when we visit nodes that do
not lead to the target. But instead of running a full DFS for each target, we
can compute the max-min closure of all the successors at once, i.e. each node is
a target. This should be efficient both in terms of memory and computation, and
should be easily parallelizable. 

The above idea could be even turned to a dynamic programming problem if, before
computing the closure of a node, we have managed to compute the closure of its
successors. But to do this we need to know the dependency structure of the
succession among SCCs, i.e. we need to compute the graph of the SCCs. This is a
directed acyclic multigraph in which each node corresponds to an SCC from the
original graph, and there is an edge between two distinct SCCs for each edge
that connected nodes in them. Analysing this graph will also let us answer my
previous question about the bow-tie structure of the DBpedia graph.

Did an experiment with saving the DBpedia adjancency graph as a sparse file,
which could then be opened with mmap. The estimated size for storing 23M weights
(as 64 bits floats) would be **70GB**. This is because each block with a signe
non-null value must be written in full. With a block size of 4K this should make
sense. PyTables has a `CArray` class that saves the data as a B-tree (chuncked
array) and that should be able to apply compression to the data chuncks. Worth
giving it a look!

These are the results of diameter analysis on the full DBpedia graph:

> Is the graph a strongly connected component? False
> The number of connected components in the graph is 2873970
> Is the graph weakly connected? True
> The number of connected components in the graph is 1

## Sat Jun 22 22:53:33 EDT 2013

I wrote a script for converting the adjacency matrix from NumPy format to
PyTable's `CArray` but I had to give up because it was incredibly slow at
writing to disk. It seems that there is a problem with the way the slices are
written to disk, because with the `test_30k` data it goes decently fast (4000
edges/s), while with the full matrix it drops to just 4 edges/s. Written a post
on pytables-users, let's see if I get a reply.

Also, killed the networkx script for computing the diameter: after 24h it had
covered 57K source! The maximum distance was nonetheless equal to *260* at that
point -- quite a lot. Tried looking at alternative Python libraries, but
graph\_tool doesn't have a decent way to load an adjacency matrix, and scipy's
sparse graph routines do not seem to work properly. Maybe try iGraph?

## Sun Jun 23 16:59:15 EDT 2013

Today started compiling graph\_tool on smithers. Converted the adjancecy list
(no weights) to GraphML format, with good results in terms of speed and and
memory occupancy, though I believe most of the speedup comes from my using of
the laptop, which is equipped with an SSD disk. If everything works out on
Lenny, I will pimp up the diameter script to compute it in parallel with
graph\_tool though I will probably have to install OpenMP as well. 

Update: it compiled and I relaunched the diameter script. The code work in
parallel out of the box, which was a pleasant surprise. Be careful about the
priority, as it seemed to take all the CPUs and had to renice it immediately!

## Mon Jun 24 12:27:26 EDT 2013

Fixed the speed problem with PyTables: apparently the default chunkshape is
still too large for the I/O buffer. Using row blocks of size 100 with
medium compression level gives very good speeds and a 1.5GB file, compared to
the 70GB file you would get with an uncompressed sparse UNIX file holding the
raw NumPy binary array. Horray!

The diameter code with graph\_tool is running faster than NetworkX (100K sources
explored vs 57K in probably half to two thirds of the time), and so far the
diameter is still stuck at 260.
