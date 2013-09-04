import numpy as np
import os

def _dfsiter(path, *children):
    '''
    Traverses the tree in the depth-first order and yields the node path to each
    visited element.
    '''
    lvl = len(path)
    yield tuple(path)
    if lvl == len(children):
        return
    for n in xrange(children[lvl]):
        path.append(n)
        for ppath in _dfsiter(path, *children):
            yield ppath
        path.pop()

_dir = '{prefix}{digit:0{width}d}'
_base = '{prefix}{digit:0{width}d}{suffix}'

class DirTree(object):
    '''
    A directory tree structure with heterogeneous arity.
    '''
    def __init__(self, prefix, children, suffix='.dat', root='root', createdirs=True):
        '''
        Parameters
        ----------

        prefix : string
            All paths are prefixed with this string.

        children: sequence of ints
            Creates a directory tree rooted at path given by `root` with levels
            specified by the `children` array: level ``i`` had ``children[i]``
            children. ``children[-1]`` specifies the arity of the leaves.

        suffix : string
            The suffix of the leafs (i.e. files) of the tree.

        root : path
            The path to the root of the tree.

        createdirs : bool.
            If True, actually create the directories. Note: this is not thread-safe.
        '''
        if len(children) == 0:
            raise ValueError("need at least one level")
        self.children = np.asarray(children)
        self.root = root
        self.prefix = prefix
        self.suffix = suffix
        self.depth = len(children)
        self._cap = np.cumprod(self.children[::-1])[::-1]
        self.capacity = self._cap[0]
        self._den = self._cap / self.children
        self.width = int(np.ceil(np.log10(self.capacity)))
        if createdirs:
            self._mktree()
    def _treepath(self, logpath):
        if len(logpath) == self.depth: # a leaf
            p = np.asarray(logpath[:-1])
            leaf = logpath[-1]
        else:
            leaf = None
            p = np.asarray(logpath)
        d = len(p)
        labels = np.cumsum(p * self._den[:d])
        fmtfun = lambda k, p, d : k.format(prefix=p, digit=d,
                width=self.width)
        path = map(fmtfun, [_dir] * d, [self.prefix ] * d, labels)
        if leaf is not None:
            basename = _base.format(prefix=self.prefix, digit=leaf,
                    width=self.width, suffix=self.suffix)
            path = path + [ basename ]
        path.insert(0, self.root)
        return os.path.join(*path)
    def walk(self, dirsonly=False, filesonly=False):
        '''
        Depth-first iterator over tree element paths. If dirs is True, only
        intermediate nodes (i.e. directories) are iterated upon.
        '''
        if dirsonly and filesonly:
            raise ValueError("cannot pass both dirsonly and filesonly")
        for nodes in _dfsiter([], *self.children):
            if len(nodes) == self.depth:
                if dirsonly:
                    continue
                nodes = list(nodes)
                nodes[-1] = np.sum(nodes[:-1] * self._den[:-1]) + nodes[-1]
            elif filesonly:
                continue
            yield self._treepath(nodes)
    def _mktree(self):
        '''
        Creates the directory structure
        '''
        for path in self.walk(dirsonly=True):
            if not os.path.exists(path):
                os.mkdir(path)
            elif not os.path.isdir(path):
                raise RuntimeError('not a directory: {}'.format(path))
    def _logicalpath(self, value):
        if value >= self.capacity:
            raise ValueError("Beyond capacity: {}".format(value))
        v = value
        path = []
        for d in self._den:
            path.append(v // d)
            v %= d
        if v != 0:
            raise ValueError(value)
        path[-1] = value
        return tuple(path)
    def getintnode(self, value, depth):
        '''
        Return the intermediate node at give depth on the path to the leaf
        associated to `value`.
        '''
        if depth > self.depth:
            raise ValueError('tree has {} levels: {}'.format(self.depth, depth))
        lp = self._logicalpath(value)
        return self._treepath(lp[:depth])
    def getleaf(self, value):
        '''
        returns the path of the leaf associated to value in the directory tree
        '''
        return self._treepath(self._logicalpath(value))
