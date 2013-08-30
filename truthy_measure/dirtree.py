import os

def _dfsiter(path, *arities):
    '''
    Traverses the tree in the depth-first order and yields the node path to each
    visited element.
    '''
    lvl = len(path)
    yield tuple(path)
    if lvl == len(arities):
        return
    for n in xrange(arities[lvl]):
        path.append(n)
        for ppath in _dfsiter(path, *arities):
            yield ppath
        path.pop()

class DirTree(object):
    '''
    A directory tree structure with heterogeneous arity
    '''
    def __init__(self, arities, capacity=1, root=os.curdir, prefix='', createdirs=True):
        '''
        Creates a directory tree rooted at `root` with `len(arities)` levels,
        where the i-th level has arity `arities[i]`. Leaves can hold `capacity`
        files.
        '''
        self.arities = tuple(arities)
        self.levels = len(arities)
        self.capacity = capacity
        self.maxval = reduce(int.__mul__, arities) * capacity - 1
        self.root = root
        self.prefix = prefix
        self._powers = [ n ** i for i,n in enumerate(arities) ]
        if createdirs:
            self._mktree()
    def _treepath(self, nodes):
        '''
        Returns the path in the directory tree associated to the give node
        sequence `nodes`.
        '''
        path = []
        for i in xrange(len(nodes)):
            tmp = nodes[:i+1]
            tmp = map(str, tmp)
            tmp = '-'.join(tmp)
            path.append(tmp)
        if path:
            path = map(lambda k : self.prefix + k, path)
            path = os.path.join(*path)
        else:
            path = ''
        return os.path.join(self.root, path)
    def walk(self):
        '''
        Depth-first iterator over tree element paths
        '''
        for nodes in _dfsiter([], *self.arities):
            yield self._treepath(nodes)
    def _mktree(self):
        '''
        Creates the directory structure
        '''
        for path in self.walk():
            if not os.path.exists(path):
                os.mkdir(path)
            elif not os.path.isdir(path):
                raise RuntimeError('not a directory: {}'.format(path))
    def getleaf(self, value):
        '''
        returns the path of the leaf associated to value in the directory tree
        '''
        if value > self.maxval:
            raise ValueError("Beyond capacity: {}".format(value))
        nodes = []
        v = value // self.capacity
        for p in reversed(self._powers):
            nodes.append(v // p)
            v %= p
        return self._treepath(nodes)
