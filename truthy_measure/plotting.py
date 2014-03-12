""" Plotting tools. """

import numpy as np
import matplotlib.pyplot as plt
from itertools import groupby
from operator import itemgetter


def plot_cdf(x, **kwargs):
    """
    Add a log-log CDF plot to the current axes.

    Arguments
    ---------
    x : array_like
        The data to plot

    Additional keyword arguments are passed to `matplotlib.pyplot.loglog`.
    Returns a matplotlib axes object.

    This version tries collapse the ranks from multiple, identical observations
    into their midpoint. This produces smaller figures.

    """
    N = float(len(x))
    x.sort()
    t = []
    for x, chunk in groupby(enumerate(x, 1), itemgetter(1)):
        xranks, _ = zip(*list(chunk))
        t.append((float(x), xranks[0] + len(xranks) / 2.))
    t = np.asarray(t)
    if 'ax' not in kwargs:
        ax = plt.gca()
    else:
        ax = kwargs.pop('ax')
    ax.loglog(t[:, 0], (N - t[:, 1]) / N, 'ow', **kwargs)
    return ax

def plot_pdf_log2(x, nbins=10, **kwargs):
    '''
    Adds a log-log PDF plot to the current axes. The PDF is binned with
    logarithmic binning of base 2.

    Arguments
    ---------
    x : array_like
        The data to plot
    nbins : integer
        The number of bins to take

    Additional keyword arguments are passed to `matplotlib.pyplot.loglog`.
    '''
    x = np.asarray(x)
    exp_max = np.ceil(np.log2(x.max()))
    bins = np.logspace(0, exp_max, exp_max + 1, base=2)
    fig = plt.gcf()
    ax = plt.gca()
    hist, _ = np.histogram(x, bins=bins)
    binsize = np.diff(np.asfarray(bins))
    hist = hist / binsize
    ax.loglog(bins[1:], hist, 'ow', **kwargs)
    return ax

