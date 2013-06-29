Project logbook
===============

_Somehow I figured out that, for the purpose of recapping what I did every day,
instead of addressing my collaborators with long, abstruse emails that they
won't reply to, it's simpler to put everything here._

### Tue Jun 18 18:52:27 EDT 2013

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

### Wed Jun 19 14:01:12 EDT 2013

Bad news. It still takes all the memory on smithers (a total of 144GB). I am
going to compute the successors for 1000 at random and see what the size
distribution of these sets is. 

### Thu Jun 20 18:57:00 EDT 2013

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

### Fri Jun 21 11:57:29 EDT 2013

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

### Sat Jun 22 22:53:33 EDT 2013

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

### Sun Jun 23 16:59:15 EDT 2013

Today started compiling graph\_tool on smithers. Converted the adjancecy list
(no weights) to GraphML format, with good results in terms of speed and and
memory occupancy, though I believe most of the speedup comes from my using of
the laptop, which is equipped with an SSD disk. If everything works out on
Lenny, I will pimp up the diameter script to compute it in parallel with
graph\_tool though I will probably have to install OpenMP as well. 

Update: it compiled and I relaunched the diameter script. The code work in
parallel out of the box, which was a pleasant surprise. Be careful about the
priority, as it seemed to take all the CPUs and had to renice it immediately!

### Mon Jun 24 12:27:26 EDT 2013

Fixed the speed problem with PyTables: apparently the default chunkshape is
still too large for the I/O buffer. Using row blocks of size 100 with
medium compression level gives very good speeds and a 1.5GB file, compared to
the 70GB file you would get with an uncompressed sparse UNIX file holding the
raw NumPy binary array. Horray!

The diameter code with graph\_tool is running faster than NetworkX (100K sources
explored vs 57K in probably half to two thirds of the time), and so far the
diameter is still stuck at 260.

Read [the DBpedia paper][DBpedia2009]. Weird averaging for in-degree: should be
46\.25 for the mapping-based network if computed as number of edges over number
of vertices. Instead they average only on the vertices with at least one
incoming edge. So the figure should be larger than my estimate, and instead what
they report is smaller (11.03)! The density is not so large, but need to check
on the Newman if that level is OK for small-world networks. The node with the
highest in-degree is that of the United States. Also, the in-degree distribution
looks more like a log-normal, once you draw in the plot a straight line as a
guide to the eye.

Wrote a script to sample the pseudo-diameter of the graph and launched it on
Lenny. Draws 10000 sources without replacement and computes the pseudo-diameter.
So far stuck at D = 148, which is much smaller than 260.

[DBpedia2009]: http://dx.doi.org/10.1016/j.websem.2009.07.002 

### Tue Jun 25 17:45:33 EDT 2013

The pseudo-diameter computed over 10^5 random sources (without replacement) is
256\. I know already that this number is smaller than the actual diameter found
so far by the full exhaustive search, which as of today is still at 260. Also,
I went through the output and there were only 8 cases where the pseudo-diameter
was > 200. Most cases are either 0, 1 or some number around 140.

Wrote a script to create the macro-graph of the strongly connected components,
that is, this graph has vertex for each SCC in the original graph and an edge
between two nodes whether the corresponding SCCs are connected by at least one
edge. I am storing the size of the SCC, as well as the number of links between
nodes in the SCC, on the nodes using a BGL `PropertyMap` (BGL = Boost Graph
Library); the edges are instead weighted by the number of between-components
links. I'll try to visualize this graph or, as Sandro suggested, list the
components that have the highest value of the product of the out-degree with the
in-degree, to see if it is true that there is a bow-tie structure. Should also
compute the percentage of nodes in the largest component, both directed and
undirected. Must read the chapter about largest components in the Newman.

### Wed Jun 26 22:58:38 EDT 2013

The script that I had launched on Lenny for computing the SCC graph (which, btw,
according to Wikipedia is called the *condensation* of the graph) got first
slowed down tonight by a job launched by Onur, and later unexplicably killed. In
the meanwhile, Smithers, on which the other job for completely enumerating the
diameter of the graph was running, has been reported to be unresponsive, and
has been reboot by Bruce, since the data collection from twitter was likely
down. Bummer. 

Spoke to Sandro, who suggested to sample a subgraph of the condensation, and
visualize it, just to get an idea of how the graph looks like. I illustrated him
the idea of computing the ultrametric (actually, any) closure with a dynamic
programming scheme, and he said it makes sense to him. Just need to find some
time to implement it before I leave for Switzerland.

### Thu Jun 27 19:12:00 EDT 2013

Started implementing the dynamic programming scheme. We had a guest today so
could not do much besides jotting down some pseudo-code and erasing the old code
and starting the general structure of the function. The compressed sparse graph
routines package from SciPy (`scipy.sparse.csgraph`) has a graph for the
connected components, which is implemented in C++, so I am gonna use that
directly. Also, wrote a script for dumping to disk the adjacency matrix of the
SCC graph -- the *condensation*, though I do not like the term and will probably
stop using it. Launched it with an ETA of 30' (thanks to the `progressbar`
package!). In the meanwhile, the `graph-tool`-based script that I relaunched
yesterday night is still halfway through. Funny how `graph_tool`, which is a
super-optimized package, incurs in these silly bottlenecks. The classic problem
with one-man projects is that if you don't fit in the workflow of the creator
then the software becomes almost useless.

### Fri Jun 28 21:03:43 EDT 2013

Last day before my Summer break. Plotted the distribution of between
in/out-degree and within component degree of the graph of the strongly connected
components, as well as the standard in/out-degree distribution of the original
graph. Let's start with the original graph. The out-degree looks Poisson with
average equal to ten (10) neighbors. This make all in all sense, considered that
the number of attributes of each infobox does not vary dramatically, and that
the depth of the DBpedia ontology is not huge. The CDF of the in-degree,
instead, seem to scale down straight for almost seven orders of magnitude with a
slope of -1, which would correspond to a power-law exponent of -2 -- if it is
actually a power-law at all, of course. This is interesting but sounds not
terribly surprising if one considers that the is-a graph is already closed. But
still, even adding the mapping-based properties the scaling looks fairly
straight, so both I and Sandro were intrigued by it.

When we look at the SCCs, things change quite a bit. I looked at three
quantities: the first two are the total between-component in- and out-degree,
that is, the count of all edges spanning different components, and the
within-component degree, that is, the count of all edges that connect nodes
within the same component. The latter quantity is bounded below by the size of
the component itself, since you need at least N edges to close a cycle among N
nodes, and is the same if one counts the incoming or outgoing edges, therefore I
call it simply "degree". The result is that, while the between-component
in-degree doesn't change much, the between-component out-degree increases in
what looks like a multi-modal distribution up to roughly k = 10^4, plus a single
observation at roughly 10^6, that is, a "monster" component with something
roughly like 1M out edges to other components. The within-component in-degree
scales somewhat similarly to the out-degree, again with a single data point at
roughly 900K within component edges. On the other side of the range, there are
also other interesting things going on at the head of the distribution, but
anyway the first question was obviously to see if these three outliers, one per
distribution, actually corresponded to the same component. I made a scatter plot
in which all the three quantities are related, using the between component
degrees as x/y coordinates, and the within component degree as the size of the
points, and the plot confirmed this intuition!

So we have a big component (which nonetheless amounts to just a 3% of the total
number of nodes), that works as a big exchange node. A very small bowtie, it
seems, though I would need more analysis to see if that's effectively the case.
Also, there a lot of nodes with in-degree equal to zero or one, which end up
being trivial components, and that's why the trick of storing only the component
roots was not saving us much memory. It would be good to understand at what
level of the is-a hierarchy these guys connect, just to make sure that they
don't correspond to one or more big sequence of edges (a filament?).
