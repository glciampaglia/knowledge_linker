Project logbook
===============

_I frequently have to send updates to Johan, Luis, Fil, and Sandro about my
progresses with the project. Since I cannot usually meet with all of them at
once, I used to send, every few days, an email with the updates to everyone. The
result was that I would often send long, abstruse messages, full of technical
details that made sense only to me. I was in fact trying to recap for myself
what I had done every day. So eventually I just figured out that it would be simpler
to write everything in a log book, and send them only short updates. That's why
this file exists._

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
has a large set of predecessors. That means that there is a lot of overlap
between the successor sets of many SCCs. This intuition was confirmed by looking
at the set themselves. I dumped them to a text file and looked at the largest
sets. They are indeed identical (again, with the exception of itself, that is
for any vertex/component $v$, $v \in succ(v)$).

If this is the case, then there is room for reducing the usage of memory. I
implemented a Python dictionary class that stores references to values, instead
of values, that is, when two objects have the same value, only one is
effectively stored and the second is stored as a reference to the first. I then
modified the code of `transitive_closure` so that now the set of successors of a
SCC (really, its root) does not contain the root itself (for the above reason
that the successor sets of two different SCCs will at least differ in the SCC
themselves), and integrated the new dictionary class. I did a test running the
same successor sets sampling code, but starting from 100 sources only (it takes
less time than with 1000 and I had tested that too with the old code). The
result is that roughly 124K SCC are encountered and the same number of successor
sets is collected, but now only 3200 sets are effectively stored in memory, i.e.
it stores 97% less set objects than before. How this translates to normal memory
is difficult to compute, but I plotted the number of duplicates as a function of
the cardinality of the sets and there are good savings even for the very big
sets. 

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

    Is the graph a strongly connected component? False
    The number of connected components in the graph is 2873970
    Is the graph weakly connected? True
    The number of connected components in the graph is 1

### Sat Jun 22 22:53:33 EDT 2013

I wrote a script for converting the adjacency matrix from NumPy format to
PyTable's `CArray` but I had to give up because it was incredibly slow at
writing to disk. It seems that there is a problem with the way the slices are
written to disk, because with the `test_30k` data it goes decently fast (4000
edges/s), while with the full matrix it drops to just 4 edges/s. Written a post
on pytables-users, let's see if I get a reply.

Also, killed the networkx script for computing the diameter: after 24h it had
covered only 57K source! The maximum distance was nonetheless equal to *260* at
that point -- quite a lot. Tried looking at alternative Python libraries, but
graph-tool doesn't have a decent way to load an adjacency matrix, and scipy's
sparse graph routines do not seem to work properly. Maybe try iGraph?

### Sun Jun 23 16:59:15 EDT 2013

Today started compiling graph-tool on smithers. Converted the adjancecy list
(no weights) to GraphML format, with good results in terms of speed and memory
occupancy, though I believe most of the speedup comes from my using of the
laptop, which is equipped with an SSD disk. If everything works out on Lenny, I
will pimp up the diameter script to compute it in parallel with graph-tool
though I will probably have to install OpenMP as well. 

Update: it compiled and I relaunched the diameter script. The code work in
parallel out of the box, which was a pleasant surprise. Be careful about the
priority, as it seemed to take all the CPUs and had to renice it immediately!

### Mon Jun 24 12:27:26 EDT 2013

Fixed the speed problem with PyTables: apparently the default chunkshape is
still too large for the I/O buffer. Using row blocks of size 100 with medium
compression level gives very good speeds and results in a 1.5GB file, compared
to the 70GB file you would get with an uncompressed sparse UNIX file holding the
raw NumPy binary array. Horray!

The diameter code with graph-tool is running faster than NetworkX (100K sources
explored vs 57K in probably half to two thirds of the time), and so far the
diameter is still stuck at 260.

Read [the DBpedia paper][DBpedia2009]. Weird averaging for in-degree: my
estimate was 46\.25 for the mapping-based network (computed as number of edges
over number of vertices). They chose instead to compute the average only on
vertices with at least one incoming edge. Thus the value should be larger than
my estimate, and instead they report a smaller figure (11.03)! The density is not
so large, but need to check on the Newman if that level is OK for small-world
networks. The node with the highest in-degree is that of the United States.
Also, the in-degree distribution looks more like a log-normal, once you draw in
the plot a straight line as a guide to the eye.

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
that is, the graph whose nodes are the SCC of the original graph and in which
there is an edge between two nodes if the corresponding SCCs are connected by at
least one edge. I am storing the size of the SCC, as well as the number of links
between nodes in the SCC, on the nodes using a BGL `PropertyMap` (BGL = Boost
Graph Library); the edges are instead weighted by the number of
between-components links. I'll try to visualize this graph or, as Sandro
suggested, list the components that have the highest value of the product of the
out-degree with the in-degree, to see if it is true that there is a bow-tie
structure. Should also compute the percentage of nodes in the largest component,
both directed and undirected. Must read the chapter about largest components in
the Newman.

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
stop using it. Launched it with an ETA of 30' (estimate thanks to the
`progressbar` package!). In the meanwhile, the graph-tool script (the one for
the SCC graph) that I relaunched yesterday night is still halfway through. Funny
how `graph_tool`, which is a super-optimized package, incurs in these silly
bottlenecks. The classic problem with one-man projects is that if you don't fit
in the workflow of the creator then the software becomes almost useless.

### Fri Jun 28 21:03:43 EDT 2013

Last day before my Summer break. Plotted various quantities related to node
degree for both the original and the SCC graph. Let's start with the original
graph. 

For the original graph, I plotted the in- and out-degree distribution. The
out-degree looks Poisson with average equal to ten (10) neighbors. This make all
in all sense, considered that the number of attributes of each infobox does not
vary dramatically, and that the depth of the DBpedia ontology, which is an upper
bound to the number of additional outgoing links added by the closure of the
*is-a* graph, is not huge. The CDF of the in-degree, instead, seems to scale for
almost seven orders of magnitude with a slope of -1, which would correspond to a
power-law exponent of -2 -- if it is actually a power-law at all, of course.
This is interesting but sounds not terribly surprising if one considers that the
is-a graph is already closed. But still, even adding the mapping-based
properties the scaling looks fairly straight, so both I and Sandro were
intrigued by it.

For the SCC graph, I plotted three quantities: the first two are the total
between-component in- and out-degree, that is, the count of all edges spanning
different components, and the within-component degree, that is, the count of all
edges that connect nodes within the same component. Compared with the previous
plots, things change quite a bit. The latter quantity is bounded below by the
size of the component itself, since you need at least N edges to close a cycle
among N nodes, and is the same if one counts the incoming or outgoing edges,
therefore I call it simply "degree". The result is that, while the
between-component in-degree doesn't change much, the between-component
out-degree increases in what looks like a multi-modal distribution up to roughly
k = 10^4, plus a single observation at roughly 10^6, that is, a "monster"
component with something roughly like 1M out edges to other components. The
within-component in-degree scales somewhat similarly to the out-degree, again
with a single data point at roughly 900K within component edges. On the other
side of the range, there are also other interesting things going on at the head
of the distribution, but anyway the first question was obviously to see if these
three outliers, one per distribution, actually corresponded to the same
component. I made a scatter plot in which all the three quantities are related,
using the between component degrees as x/y coordinates, and the within component
degree as the size of the points, and the plot confirmed this intuition!

So we have a big component (which nonetheless amounts to just a 3% of the total
number of nodes), that works as a big exchange node. A very small bowtie, it
seems, though I would need more analysis to see if that's effectively the case.
Also, there a lot of nodes with in-degree equal to zero or one, which end up
being trivial components, and that's why the trick of storing only the component
roots was not saving us much memory. It would be good to understand at what
level of the is-a hierarchy these guys connect, just to make sure that they
don't correspond to one or more big sequence of edges (a filament?).

## Mon Jul 22 22:31:56 EDT 2013

Back from vacations. Had a phone conversation with Mo Moadeli from
[DEEMOK](http://deemok.com), a NYC-based startup that is trying to build a
platform for political discussions that would feature a number of automatic
fact-checking tools. He sounded very interested to the stuff I am doing. Will
follow-up with the rest of his team next week and try to see if, besides the
obvious relatedness of what we are both trying to do, there are real
opportunities for mutual collaborations with them.

Organized a nice birthday party, really a lunch, back in Rome, for a bunch of
friends. Trying to describe what I am doing in this project to them, and today
to Mo, I came to the conclusion that there is no point in trying the compute the
maxmin distance over all possible pairs (that is, the all-pairs shortest
bottleneck path, even though this applies, as always to any kind of metric, not
just the ultra-metric). What's really interesting for us, in fact, is to see if
certain statements have a measure of truthiness at all, and to do this we don't
need to compute the whole closure of the graph, but just the maxmin similarity
for candidate statements vs average statements. Also, more than the actual
values, it would be more interesting to see how the values change when you start
to remove edges, the idea of robustness Luis was talking about. If I start to
disconnect the graph, I am preventing an hypothetical agent from using certain
connections when trying to "resolve" the statement. Statements such as _X is a
Y_ could have a higher truthiness if, as the number of removed edges decreases,
their similarity decreases slower than the average. 

From the technical point of view. The DFS would just work, without having the
need of implementing the dynamic programming algorithm, which requires to keep a
huge table either in memory or on disk. I could even use a cache to avoid
wasting time recomputing over and over the same intermediate results.

## Tue Jul 23 17:42:09 EDT 2013

Today went again through these notes, just to get back into the problem, and it
worked great. Keeping these notes is proving to be useful, at last! Spent most
of the time today following the summer interns and wrote a simple reverse
geocoding script for Bryce, so could not devote much time to implementing
yesterday's ideas. A tornardo struck south of the city around 2.30pm and we
all had to go downstair and wait 20' until the emergence was over. There do not
seem to be any big damages, at least from what I see from my office window.

## Tue Jul 30 20:22:16 EDT 2013

Worked most of the last week supporting Kehontas and Bryce to finalize their
summer project and produce the poster for the final presentation in Indy.

Yesterday evening had a hangout with Mo Moadeli from Deemok and the rest of his
team. They were obviously very interested in understanding more about the
project and see if they can leverage anything of what we do or will do (the
future tense is more appropriate here) for their product. Not sure about the real
possibities for collaborating now, since they don't seem to have any data at
hand, but suggested them to talk to Fil anyway and see. 

Had finally time yesterday and today to code the DFS. It took me more than
expected and several iterations of the algorithm (the original idea was not
really correct), but, at last, keeping in memory the set of nodes that
constitute the paths did the trick. Shouldn't be a problem in terms of memory
footprint (at worst $O(n)$), but probably will have to cythonize it to speed it
up. Will do some tests tomorrow.

Sandro advised not to throw away the implementation based on the transitive
closure -- Implementation which, btw, I discovered to be not correct while
implementing the DFS yesterday -- but the good news is that I can just plug in the
new DFS code and it should work -- provided that I can find a way to store the
information about the successors sets on disk and access it with ease. Other
task for tomorrow will be to test PyTables or just plain numpy array files.

Bought a bed from a classical music student and an entertainment system (DVD, CD
player, 5+1 speakers) from a SOIC staffer who's leaving soon.

## Wed Jul 31 19:17:40 EDT 2013

In the morning reorganized the function names in maxmin.py. In the afternoon,
modified the transitive closure function so that it can store the successors
sets to disk. The changes introduced break the integration with the maxmin
closure functions. Will run a test later today to see if now it is feasible to
compute the closure of the full DBPedia graph. If this works, must integrate the
code from the DFS search of yesterday with the branch pruning. Also must make
sure that the full successors set is iterated upon, not just the SCC roots.

## Mon Aug  5 19:38:30 EDT 2013

Skipped a few entries last week so I am summing all up here.

Last week worked on fixing a bug in the transitive closure algorithm and
integrating it with PyTables so that the closure graph can be saved to disk.
Also added a progress bar and ETA, and it looks like that it would take close to
12 days to finish computing the closure. Launched it on Friday on Lenny but had
to kill it today because the GFS was running out of space again, though it seems
that my script it was not the main cause the fillup. Queued it today on Quarry
with maximum wall time (14 days!) and will start in two days.

Last Friday worked on fixing a bug on the DFS functions and in integrating the
cache. Today finished rewriting the iterative DFS and wrote some more tests.
Tested the effect of the cache but found no speedup on random graphs up to 5000
nodes and sparsity equal 20%. Could be due to the ordering of the computation:
ideally first computing the sources with higher betweenness should increase the
ratio of cache hits, though should actually profile this more accurately and not
use random graphs for doing the benchmarking. The current implementation of the
cache is also slow, by the way.

Also, found out that `graph_tool` provides a python wrapper to the transitive
closure function from the BGL. Tried to see if the closure graph would fit in
memory on smithers (which should have something around 192GB or RAM) with no
luck -- called it off (i.e. killed the process) at around 75% of memory
consumption to avoid making the machine unresponsive. In theory, modifying the
function to store the graph on disk would solve all my problems, but the time it
would take me to learn how to write decent C++ code would probably even out with
the slowness of Python. 

## Tue Aug  6 17:15:25 EDT 2013

Today looked at how to recover from the raw data files the actual ontological
network without the closure. Recall that the dbpedia instance types file is
already closed, plus it includes links to external classes that are equivalent
to classes in the dbpedia ontology (i.e. dbo:Person and foaf:Person). It should
be possible to recover the ontology without transforming it in a taxonomy (i.e.
keep cases when a resource has multiple classes) in a fairly simple way.

## Wed Aug  7 18:56:13 EDT 2013

Finished the script for filtering the instance-type data files. The script
filters out classes from external ontologies (such as FOAF or schema.org) and
extracts the original taxonomy from the existing graph distributed by the
DBPedia team, which is closed. People from the DBPedia-users mailing list told
me there should have been a few instances belonging to multiple classes, that
is, the ontology should be a DAG and not a tree, but I could not find any. This
process eliminates roughly 80% of the is-a edges from the instance-type file.

Recreated the data file, the network has 3,141,896 nodes and 12,898,752 edges.
Compared with the previous version (3,141,881 nodes and 26,914,521 edges) the
difference of 17 nodes is due to the FOAF, bibo, and schema.org classes that
have been removed to 54 classes from the DBPedia ontology namespace that for
some reason had been lost.

## Fri Aug  9 19:24:00 EDT 2013

Skipped another log entry (Thursday). Several updates:
1. Filtered network still has power-law (like) distribution of in-degree, and
   Poisson like of out-degree. The filtering did not affect that too much.
   Good. Loaded the new files on Quarry. The process running there is too slow
   and I will kill it (but read on).
2. Dumped to file the contents of the strongly connected components. Many
   interesting thing. The large component is a generic group, while smaller
   components are all topical groups, e.g. Lost episodes, rivers (there are more
   than one group for that -- maybe continents?), human anatomy, etc. Also some
   groups seem to have smaller subgroups, e.g. all glutei nerves, etc. Showed
   this to Sandro. Plan is to run shortest paths from source components to
   owl:Thing to see where the paths run through.
3. Worked on cythonizing the transitive closure function. Spent much time on the
   recursive function, with horrible results: it was actually slower than the
   Python version. Did not despair and thought that recursion might be the real
   problem, and that inline hints where probably dismissed by the compiler, so
   quickly translated the iterative version and bam! 8ms per loop vs 6s on the
   test30k graph! There are some glitches in the results but it's probably
   nothing serious.
4. Also figured out that the test for the root nodes in the stack was
   implemented as a membership test on Python list, which is horribly slow.
   Changed that to a test on an array. Hopefully all these improvements should
   make the thing work on the full graph. Need to add the PyTables support and
   see if the gains from Cython are not all wasted by the I/O to disk.

## Sat Aug 10 21:20:48 EDT 2013

Sadly, discovered today that the speed-up obtained yesterday was due to a bug I
had introduced in hastily translating the iterative implementation of the
closure function -- having copy-pasted the code from the recursive version, the
neighbors function was being called only once at the initial cdef time. So back
to 6s. The real bad news though are that integrating PyTables in the cython code
results in horrible performance degradation, which make the work of the last two
days utterly useless. No idea why this occurs, and don't want to waste more time
into this. The only good news is that having introduced a smarter membership
test for the stack dropped the overall performance of the pure Python
implementation of more than 50% on the test30k graph, from 18s to 8s. Played a
bit with the chunkshape, and it seems that single-row chunks of size 10k produce
reasonably sized files at the best speed. At this point the only option is to
relaunch the script on Lenny on the filtered graph.

## Wed Aug 14 18:27:08 EDT 2013

Relaunched the transitive closure on Lenny over the weekend (Saturday night).
The computation is currently 1/5th through. Modified the script for printing the
SCCs to include entities' out degree and sorting them in descending order of out
degree. Also produced an extra file with all components of size greater than one
and the member of maximum out degree. In the large component this is the United
State, with roughly 185k out edges. Jotted down (in Cython!) a shortest path
function that will serve for tracing the paths from the source components to
`owl:Thing`. Presently not working (segfaults) but should be just a simple bug.

## Thu Aug 15 19:12:48 EDT 2013

Tonight Lenny went out of memory multiple times and the OOM killer thread ended
up killing the transitive closure script. The culprit was a script by Diego.
Will move my computation to snowball, in the hope that there no one will hammer
the machine too much. Finished implementing the shortest path algorithm in
Cython. Need to use C arrays in it so that I can release the GIL and can run the
code in parallel.

## Fri Aug 16 18:28:55 EDT 2013

Worked again on the code for the shortest paths, and finished the parallel
implementation. Learned how to use Cython's parallel facilities, which are based
upon OpenMP, and even had a brushup of C memory management. The code is now
running on Lenny and will save all paths from all sources (nodes with zero
in-degree) to owl:Thing to a file for later analysis. Haven't restarted the
transitive closure code yet, but have figured out that perhaps using the
approach based on topological sort could help. In particular, the successors
matrix of the condensation can be split in a diagonal part, whose rows
corresponds to source node only, and another matrix that will have an empty
block, whose columsn correspond to sink nodes only (node with 0 out-degree) and
a relatively dense part, whose row and nodes corresponds to all intermediate
nodes (that is, SCCs). There are 1,642,002 sinks and 782,016 sources. The total
number of SCCs is 2,873,985. This means that the actual sparsity coefficient of
the matrix is at most 30%. Still a lot: if I wanted to represent the matrix
as dense, it would require 2,5TB. On the other hand, could be doable on Quarry
to mmap it as a binary numpy array file, in uncompressed format, which would
increase the speed quite a bit! Have to think about it and maybe talk about it
with Fil.

## Fri Aug 23 18:43:58 EDT 2013

Spent most of the week reading papers for the work on the ultimatum game, so
could devote little time to this, but still managed to expanded the script for
the shortest paths to compute the betweenness centrality and the path length
distribution, and did some preliminary experiments with memory-mapped arrays and
Cython typed memoryviews (which seem to work flawlessly). 

On Thursday, read the paper by Steyvers and Tenenbaum about the statistical
structure of three classic semantic networks: WordNet, Rotgen Thesaurus, and the
Free Association Netword (IIRC). The first two are undirected network while the
third is originally directed, but they analysed the undirected case as well. We
should really focus on the undirected case because the edge directionality can
be too constraining, despite my initial idea to take it into account. 

On Friday, had a meeting with Fil and Sandro, which got again too much into the
implementation details, anyway the resolve is to stop wasting time on making the
transitive closure algorithm (the Nuutila one) compute the whole successors
matrix. Instead, for each source, just perform a first BFS step to gather all
successors, and then launch as many depth-first traversals as targets. Fil was
still pushing to use a breadth-first approach but I think that my DF approach is
better, besides, I have the code working. Spent the afternoon converting the
depth-first traversal code (`truthy_measure.maxmin.mmclosure_dfs`) into Cython,
mimicking the API chosen from the shortest path function. Fixed all compilation
errors thrown by Cython, but the code still throws a segfault, so will need to
do some debugging, but nothing crazy.

Also, from the meeting it seems that I will get a grad student to help me on
this, and perhaps also somebody to aid with the infocycle project.

## Mon Aug 26 21:30:06 EDT 2013

Worked on the Cython implementation of the reachability functions via BFS and on
the maxmin closure. Still some bugs in the way the parallel function of the
maxmin closure returns value, but improved a lot the code. Met with Prashant and
described him broadly the project and its possible applications in terms of
automatic fact checking.

## Tue Aug 27 14:50:58 EDT 2013

Fixed the bug with the parallel function, and wrote a script that splits the
computation over chunks of sources, applies the BFS from each source to find all
reachable nodes, and launches a DFS traversal for computing the maxmin. Also,
added a pruning condition that should improve performances. Made some tests on a
random network with comparable sparsity coefficient and it seems to go decently
fast, though the complexity depends crucially on the number of nodes, with
a sparsity coefficient of 1e-2 it was taking forever. Perhaps a branching
factor? At any rate, launched the script on Lenny to get an idea of the possible
ETA. In case will move it over Big Red2 if more power is needed.

Did some more research on ways to use Dijkstra instead of a simple DFS, and
found a nice technical report by [Kaibel and Peinhardt](Kaibel2006) explaining
the algorithms for the widest path problem (the name under which the transitive
closure is known in graph theory). Assigned it to Prashant as a reading and to
implement the algorithm for the undirected case.

[Kaibel2006]: http://www.zib.de/Publications/Reports/ZR-06-22.pdf "Kaibel,
Volker; Peinhardt, Matthias A. F. (2006), On the bottleneck shortest path
problem, ZIB-Report 06-22, Konrad-Zuse-Zentrum für Informationstechnik Berlin"

## Wed Aug 28 16:39:31 EDT 2013

Today implemented a modified version of Dijkstra to compute the maxmin paths. It
works, and is fast enough on the random graph with sparsity coefficient where
the previous brute-force-ish attempts where taking forever. Since the algorithm
uses a binary heap, needed to find a C or C++ implementation of binary heaps (or
priority queues). Tried to get Boosts's heap library to work but with no success
(must really start to study C++ seriously) and finally found a Cython
implementation posted on SciPy User.

## Thu Aug 29 19:49:04 EDT 2013

Finished cythonizing the code written yesterday and integrating the binary heap
code by Almar Klein found on SciPy-User. This code provides a binary heap
structure as a Cython extension type. Also discovered a silly bug in my
implementation of Dijkstra that would update nodes after they had been popped
from the heap. Tested on a 10^4 x 10^4 random graph with sparsity coefficient
0\.1, which yields roughly the same number of edges of the DBPedia graph. The
pure Python version took 5 minutes to execute, while the Cython version just 3
seconds, a fat 100x speedup. Wrote a parallel all-pairs function reusing the
multi-process code written for the matrix multiplication (as it turns out,
extension types cannot fully release the GIL, so the code cannot parallelized
using OpenMP, which requires the GIL to be released), and launched it on
snowball.

## Fri Aug 30 17:22:13 EDT 2013

Found this morning that a few processes had accumulated a lot of memory and had
sent snowball into heavy swapping throught the night. Terminated the job per
Rob's request, and started investigating the source of the memory leaks. Found
several leaks (thanks to Valgrind) and fixed them. Unfortunately, despite the
these being fixed, memory usage on snowball would still increase, admittedly
more slowly now, to the point of making again the machine unresponsive. Perhaps
an undetected leak? Or too many workers at once? Decided eventually to move the
computation on Quarry. (Disappointingly, Big Red II still does not have BLAS nor
ATLAS installed, so impossible to install SciPy quickly via pip. Need to email
the support team about this.) Launched a first batch of 50 jobs, each of 1000
input nodes (of the graph, that is) each. Machines on Quarry are much less fat
than snowball (only 8GB or 16GB of RAM), so hope that smaller chunks will finish
before saturating the main memory of the nodes.

Also implemented a class that manages a directory tree, so that the output files
do not all cram the same inode. 
