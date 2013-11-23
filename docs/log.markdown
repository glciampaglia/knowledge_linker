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

Last update: added a switch for the generation of the paths and set it off by
default, and relaunched the job on snowball. Memory consumption seems constant
now.

## Sat Aug 31 20:42:54 EDT 2013

This morning got an email by Rob warning that my job on snowball had filled up
the scratch partition on snowball. This had caused the job to terminate (after
some 27000 rows computed, btw). Moved the data (1.8TB) to the HPSS. I am not
going to use snowball anymore for this. In the meanwhile, and predictably I have
to admit, the job on quarry, which was still running the leaky code that
computes the paths, had sent all nodes into heavy swapping. Had tried to delete
the jobs in the night but had gotten only about half of them down when the queue
manager had gone down (not sure if related to my attempts). This morning mailed
the UITS people to fix that, and could kill the remaning jobs. Realized that by
default the distance arrays returned by the dijkstra function have -1 for
disconnected pairs, so the array files dumped to disk have all blocks full.
Modified the code to pass to the frontend function in `cmaxmin` a PyTables
chunked array, and to save only the non-negative values returned by the
algorithm. Also, to avoid issues with writing in parallel to the same HDF5 file
(PyTables does not support parallel writes), added some extra logic to have each
process save to a separate `.h5` file. Chunking the job into 50 batches (the
maximum that can be submitted at once on Quarry), each job has to compute 64000
rows. Resubmitted the first 5 jobs only just to see how big they get with this
new setup, and if not too big (currently I am using 25% of my 10TB quota on the
data capacitor), should be able to launch the rest later tonight or tomorrow
morning.

## Mon Sep  2 17:18:33 EDT 2013

PyTables still too slow for being useful! Changed to `memmap` objects but turns
out that cannot create files larger than 2GB in this way due to a limitation of
Python's mmap module.. Will dump data to disk each row a separate file and a
sequence of column indices and data values. Bummer.

## Tue Sep  3 21:43:28 EDT 2013

Implemented the simplest binary format for dumping data to file: one file per
row, two long integers (source and number of non-zero elements), and then pairs of
column coordinate and bottleneck distance value (double). Discovered a bug in
the DirTree class, rewritten it to produce more meaningul paths. Removed all
code that saved to file the actual paths, which for now we don't use due memory
leak in the Cython code. Overall simplified a lot the code for handling the pool
of workers. Launched once single job (1/50th of the whole thing) on an
interactive node on Quarry (q0148) just too see how far it goes in 24 hours and
how much space will it take.

## Wed Sep  4 19:42:32 EDT 2013

Yesterday's computation completed at 11:53 am, roughly 14 hours, and produces
185\.4GB of data. This means that the whole uncompressed graph would range
around 10TB. Decided to run the computation in three main chunks of 15 jobs
each, and modified the code to have the files added to a separate TAR archive
(one for each worker) immediately and removed from the directory tree. Tested
creating a gzipped TAR archive and in 3 hours it was roughly at half of the
64,000 files of job 0000000. With pgzip it should be OK to concatenate and
compresse all the archives generated in each chunk and finally create a single
TAR archive with the whole data.

## Thu Sep  5 21:51:03 EDT 2013

This morning found 5 jobs killed for weird I/O errors. Got in touch with the
uits people, who found it might have been a communication error with the data
capacitor. They are investigating. In the meanwhile, discovered that the
remaining processes did not terminate in the allotted time: analyzed the logs,
and found that appending to TAR archives becomes progressively slower until
basically the processes where at a grinding halt.
So decided to ditch TAR files again and to just leave the files in the directory
tree, and resubmitted the jobs. At this point, rather than storing the whole
matrix, somewhere, it makes more sense to just start analyzing the data
in batches. This makes more sense than wasting more time and energy trying to
compress all the data.

## Sat Sep  7 15:12:14 EDT 2013

Yesterday all 16 jobs terminated, with job #0 taking considerably more time than
the others. Disk usage is 2358.54 GB. Considered that 16 jobs accounted for
1,024,000 sources, this means that the whole graph should not take more than
\7.5 TB, and so it could fit on scratch without hitting the quota. Submitted
another batch of 16 jobs. Estimated start should be in 2 days.

## Mon Sep 16 09:35:13 CEST 2013

Checked jobs on Quarry. 48 out of 50 completed correctly, 2 failed again with
I/O errors. Relaunched them this morning, ETA for job start is one day.

## Tue Sep 17 11:48:53 CEST 2013

The two last jobs completed successfully.

## Tue Sep 24 20:55:40 EDT 2013

Before leaving for Barcelona (that was two weekends ago), pulled from DBPedia
data about all U.S. democrats and republicans born after 1940, as well as all
ideologies (including a separate subset of all -isms). There 2,392 democrats,
1,923 republicans, and 820 ideologies -- of which 331 -isms. 

Wrote a simple sequential script for computing the bottleneck distances between
each politician and all ideologies, and launched it on Lenny. The plan is to see
whether the two classes separate. 

## Thu Sep 26 19:07:20 EDT 2013

Computed the bottleneck distances. Plotted the data using two dimensionality
reduction techniques, PCA and LDA. Used 2 components, which for PCA explain >
96% of the variance. The data look like a perfectly shaped parabola. Printed the
top/bottom 10 dimensions of the two eigenvectors of the matrix, but found no
clear pattern. Rewrote the script to also plot 1) random noise (per Sandro's
request, to make sure there are no bugs in the code), and 2) random
source/target pairs. Launched it on Lenny.

## Fri Sep 27 18:07:32 EDT 2013

Made also the plots for the random noise (which works as expected) and for
random s/t pairs. The latter produces a parabola-like segment too. Also, found a
small bug in the way the eigenvectors were printed and now there seems to be a
more discernible pattern of ideologies, but still not very satisfactory. Why do
the points all stay in a 1D subspace? Guess could be that either the columns are
strongly dependent on each other, or that some other regularities imposed by the
fact that we are looking at distances on a graph is at work. To understand this,
the next step will be to look at the same bottleneck distances with the inverse
in-degree weighting scheme, but computed on a random graph.
random network.

## Sun Oct 13 05:14:28 EDT 2013

Several updates:

* Had a long meeting with Fil, and reconsidered several tenets of the current
  approach. In particular, moved to the undirected case and to node-based paths
  instead of edge-based paths, and decided to include base cases so that direct
  neighbors have alway truthy equal to 1. Also, the truth should not depend on
  the value of the weight of the last node, but only of the weights of the
  intermediate node. This amounts to taking an axiomatic approach for the
  desired truth function.
* Once we have this, we can use nearest neighbors to make classification and
  compute F1 scores, so that we can quantify the goodness of each approach. The
  idea, which was the original idea actuall, is to test several metrics (maxmin,
  diffusion, etc.) and several node weights. Right still using the inverse
  degree of the node.
* Had Prashant implement a prototype of the node-based Dijkstra, and then
  integrated it within the package, together with tests that he computed by
  hand.
* Implemented the Cython version and launched it on the politician data (this
  time using all ideologies) on Big Red II.
* Will have to present on next Monday so hopefully there will be some
  interesting results.

## Sun Oct 13 19:00:23 EDT 2013

Found today that the queue accepts only about 20 jobs to run at the same time,
and the majority of jobs was blocked. Of the job that had complemted, the
results did not look particularly good: all capacities are pretty much the same,
but could be that it computed only a few poorly connected individuals.

Changed the script that generates the PBS script to use MPMD mode, essentially
one big job that requires multiple nodes instead of many jobs each on one node.
After wrestling with the batch system found that I could submit 70 jobs each
with 51000 lines each using 47 hours of wall time. 51000 lines should take
roughly 46 hours. The job is now queued and is estimated to start tomorrow. 

This means that I won't have anything to present tomorrow but perhaps it's
better like this.

## Tue Oct 15 19:09:21 EDT 2013

Yesterday presented the current state of the progress at the weekly NaN meeting
and got green light from Johan about the changes agreed with Fil. Showed the
presentation today to Sandro, who could not attend yesterday, and agreed that we
are still doing something that just works, rather than something principled.
Meanwhile, the job on Big Red II had started yesterday and had ran for a shorter
time than estimated, until it terminated due to a bug in aprun. Fixed the
problem with it and tested carefully that it works. Resubmitted the job: it
should complete at the latest by Friday morning/early afternoon, depending on
the estimate.

As a side note, yesterday evening stumbled upon the Washington Post's [Truth
Teller](http://www.washingtonpost.com/blogs/ask-the-post/wp/2013/09/25/announcing-truth-teller-beta-a-better-way-to-watch-political-speech/)
uu
, a prototype for an automatic real-time fact checker. It does speech
recognition of a video, and then tries to match sequences of the transcript to a
database of statements, which have been previously labeled as either true or
false (or maybe?). [More in
detail](http://www.knightfoundation.org/blogs/knightblog/2013/1/29/debuting-truth-teller-washington-post-real-time-lie-detection-service-your-service-not-quite-yet/):
    
    We are transcribing videos using Microsoft Audio Video indexing service
    (MAVIS) technology. MAVIS is a Windows Azure application which uses Deep
    Neural Net (DNN) based speech recognition technology to convert audio
    signals into words. Using this service, we are extracting audio from videos
    and saving the information in our Lucene search index as a transcript. We
    are then looking for the facts in the transcription. Finding distinct
    phrases to match is difficult. Instead, we are focusing on patterns.

    We are using approximate string matching, or a fuzzy string searching
    algorithm. We are implemented a modified version Rabin-Karp using
    Levenshtein distance algorithm. This will be modified to recognize
    paraphrasing and negative connotations in the future.

The prototype they have features just two videos, but it does
the real thing live, which is neat. At first I thought it was vaporware, but it
appears to be legit. Whether they can actually produce and maintain a database
of curated statements is another thing, but it makes a lot of sense to them,
since they have so much unstructured data to use.

## Tue Oct 22 21:58:14 EDT 2013

Upon completion of the job, discovered that the results were almost completely
all equal to the same extremely small value (3.3e-5). This prompted
investigatiosn and eventually led me to discover a bug, that was due to a
misunderstanding of the algorithm's implementation. What is annoying is that the
bug had slipped through the test suite because the test suite had itself a bug:
a silly cut&paste bug that made all consistency checks between the Cython and
the Python version pass by default. 

Fixed that plus other, related bugs and resubmitted the job in Big Red II, this
time limiting the number of jobs to 64 so that all job can go in the queue at
once. Incidentally, took a look at the paths generated on the politicians data,
and it seems that maxmin tends to create very long chains. This makes sense,
since the hubs, have high betweenness, are bottlenecks, and so having to avoid
them one ends up increasing the number of hops. Funny that one such path, a
sequence of more than 100 nodes, even included nodes such as Mars Volta and the
singer of Neurosis, two of my favorite bands.

## Thu Oct 24 19:46:52 EDT 2013

Refactoring the package to have a single function for computing the distance
closure. Also, figured out a way to compute in just one pass the similarity for
all targets, instead of calling Dijkstra for each target. Results are coming
from Big Red II and they look OK. 

## Thu Oct 31 19:42:08 EDT 2013

The job has completed and the results have been backed up on tape on the SDA.
Took a look at the raw data, and the values now seem OK. Figured out a way to
compute the closure restricted only on the intermediate nodes that does not
require to launch a full Dijkstra for each target: simply launch the normal
closure, then look at all in-neighbors of the target that are also reachable
from the source, and take the max (or whatever it is) among those. Implemented
this, and also started a major refactoring of the package in order to
accommodate the min-plus (i.e. the classic Dijkstra) under the same
implementation. The harmonic mean, on which Prashant is working is probably
going to end up on its own implementation. 

Discussed with Fil how to perform the calibration: essentially perform a simple
classification with Nearest Neighbors, and compute the F1. Prashant asked to use
the politicians data for his machine learning course and will implement other
algorithms, but in R.

## Thu Nov  7 12:50:14 EST 2013

Implemented the one-pass method for computing what now I call "epistemic"
closure (as opposed to the normal metric closure), which is essentially the
normal closure computed only on intermediate nodes and with the additional
constraint that direct neighbors (i.e. available knowledge) have maximal
similarity (or minimal distance).

## Sat Nov  9 23:01:55 EST 2013

Integrated the one-pass algorithm into the package and implemented also the
metric version, using the Dombi t-conorm with $\lambda = 1$, which is the
equivalent of the classic Dijkstra algorithm, but on proximity graphs. Tested
it on Big Red II and to compute one single-source problem, (i.e. one row) it
takes approximately 3 minutes, which means that the wall time is cut down from
more than 3 days to just 3h30'! 

Submitted three job arrays, each consisting of an array of 64 simple jobs:

1. (id = 185915): metric closure, directed graph
2. (id = 185916): ultrametric closure, undirected graph
3. (id = 185917): metric closure, undirected graph

## Sun Nov 10 15:41:29 EST 2013

__Update__ on jobs launched:

1. (id = 185915) all jobs except #55 terminated. Job #55 was canceled.
2. (id = 185916) jobs exceeded walltime (5h) and were all terminated; some
   late-starters were canceled.
3. (id = 185917) jobs exceeded walltime (5h) and were all terminated; some
   late-starters were canceled.

Some jobs that started late were still appearing in the queue with status
"Canceled":

	189515[55]
	189516[35]
	189516[38]
	189516[46]
	189517[2]
	189517[8]

I terminated them. 

__Update #2__: relaunched job 189515[55] as 191513[55].
__Update #3 (17:33:36)__: job 191513[55] terminated, renamed as 189515[55] and
added to TAR file. Moved tar file to
`~/data/dbpedia/politicians/dbpedia-189515.tar.gz`
__Update #4 (17:37)__: launched: 

* ultrametric undirected, job 191538[], walltime 12h,
* metric undirected as, job 191539[], walltime 12h.

__Update #5__: wrote Nearest Neighbors script for calibration.

## Mon Nov 11 11:53:33 EST 2013

Killed jobs 191538 and 191539 for multiple issues: 

* jobs were failing at startup
* not enough walltime (!)

__Postmortem__: Yesterday I uninstalled the local build of scipy on Quarry
because it would fail importing `scipy.spatial` (which was required by the
`scikits-learn.neighbor`, the nearest neighbors module), and reverted to the
system installation provided in the system's `python` module. Unfortunately on
BigRedII the `python` module does not include scipy, which caused the jobs
submitted yesterday to crash at startup b/c.

__Update 14:50__: Resubmitted on BR2 after recompiling numpy and scipy (but
still `scipy.spatial` does not work):

* metric, undirected, job array 192597[], walltime 24h,
* ultrametric, undirected, job array 192598[], walltime 24h.

__Update 19:23__: Resubmitted jobs in the serial queue:
* metric, undirected, job array 192648[], walltime 24h,
* ultrametric, undirected, job array 192649[], walltime 24h.

## Tue Nov 12 10:22:57 EST 2013

Jobs still in queue on Br2, the first job should start soon.

__10:51__: Went to open the tar archive with the results of job
189515, and discovered that I had only added the error log, not the output;
also, I had removed the original files. *Bummer!1#%}$~!*

Relaunched the job (metric closure, directed graph on BigRed2 (id = 192833[],
array -t 0-64).

__13:40__: BR2 had issues with starting jobs and the was not launching them. Now
apparently the issue was fixed.

__17:00__: The first 28 jobs from 192648[] have started.

__17:06__: Computed F1 using NN (k=20, uniform weights, 10-fold CV). Directed
ultrametric F1 = 0.38 +/- 0.07

## Wed Nov 13 10:09:38 EST 2013

Jobs are running but BR2 is still somehow congested. Hopefully 192648[]
(undirected metric) shall be end of this week. Also: on the undirected graph the
walltime is approximately 16h.

__12:29__: Completed presentation for CASCI group meeting, adding result about
ultrametric.

## Thu Nov 14 14:49:11 EST 2013

__14:54__: checked progress of jobs on BigRed2. Two jobs still running (192648
and 192833), ETA for the last one in the queue is Saturday afternoon/evening.
Job 192649 completed, but found three jobs (ARRAYID=8,9,10) to have been killed
by the OOM. They were among the earliest to start after the scheduler had been
restarted (see entry 11-12), so perhaps there was some sort of congestion
problem. 

__15:07__: Restarted job with -t 8-10, -l ppn=1:nodes=32 and walltime = 18h
Job ID: 197498 (ultrametric, undirected). Job running right away, skipping rest
of queue (perhaps b/c of lower walltime?).

## Fri Nov 15 10:04:38 EST 2013

Job 197498 completed. Check file size (OK) and added the files to the TAR
archive for 192649 (renamed).

Job 192833 completed. Seen several jobs failing. Will check into this after the
twitter-truthy meeting.

__19:33__: updated with Fil and Sandro (and Onur and Emilio). Reported about F1
score (which is not the accuracy, as previously thought) and seems the result
doesn't look bad, but need to compute precision and recall separately. Several
suggestions (use random forests instead of NN, establish baseline, compare other
characteristics of the sample to see if task is too easy, compute within- and
between-class similarity, start thinking of case study for validation).

All jobs on BR2 completed, will check how many failed and need to be restarted
later in the w/e. 

## Sun Nov 17 17:26:24 EST 2013

Recap of jobs:

* 192648[] -t 0-63 (undirected, metric): jobs 16,32,36-40,47-49 aborted for
  various reasons. Average walltime for other jobs: 16h.
* 192649[] -t 0-63 (undirected, ultrametric): all jobs completed.
* 192833[] -t 0-63 (directed, metric); jobs 3,32,35-37,41,45,47,5,51,52,57,60
  failed for various reasons (OOM killer among others). Average walltime for
  other jobs: 1h23'.
* 197498[] -t 8-10 (undirected, ultrametric -- repost of 192649[], files have
  been renamed and integrated into the TAR archive for 192649)

__18:00__: relaunched failed jobs,

* Undirected, metric: 198733[] -t 16,32,36-40,47-49, walltime 18h. ETA start:
  1d from now (Monday evening).
* Directed, metric: 198735[] -t 3,32,35-37,41,45,47,5,51,52,57,60, walltime 4h.
  Jobs are running.

## Mon Nov 18 11:32:58 EST 2013
All jobs in 198735[] completed regularly. Array 198733[] has still three job
running, which should complete soon, and has one job (198733[32]) which has been
put on hold. Opened a ticket to have it released.

__12:19__: the three jobs in 198733[] have completed. Only 198733[32] still to
go.

__16:12__: heard back for HPS: job is now in queue.

## Tue Nov 19 10:25:01 EST 2013

Job 198733[32] completed.

## Sat Nov 23 17:51:20 EST 2013

Archived output of 198733[] and 198735[].

Archived output of 198733[] and 198735[] as tar.gz files in
~/data/dbpedia/politicians.

Updated files for jobs 16,32,36-40,47-49 in dbpedia.192648.tar.gz using those
from dbpedia.198733.tar.gz and archived updated file in
~/data/dbpedia/politicians.

Updated files for jobs 3,32,35-37,41,45,47,5,51,52,57,60 in
dbpedia.192833.tar.gz using those in dbpedia.198735.tar.gz and archived file in
~/data/dbpedia/politicians. 

__18:34__. Double-checked actual files in the archive. The data are thus stored
as follows:

* ultrametric, undirected: ~/data/dbpedia/politicians/192649.tar.gz
* metric, directed ~/data/dbpedia/politicians/192833.tar.gz
* metric, undirected: ~/data/dbpedia/politicians/192648.tar.gz

__18:36__. Launched backup to SDA.
