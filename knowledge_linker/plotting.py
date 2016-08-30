""" Plotting tools. Requires optional dependency Matplotlib. """

import numpy as np
import matplotlib.pyplot as plt
from itertools import groupby
from operator import itemgetter


def plot_cdf(x, copy=True, fractional=True, **kwargs):
    """
    Add a log-log CCDF plot to the current axes.

    Arguments
    ---------
    x : array_like
        The data to plot

    copy : boolean
        copy input array in a new object before sorting it. If data is a *very*
        large, the copy can avoided by passing False to this parameter.

    fractional : boolean
        compress the data by means of fractional ranking. This collapses the
        ranks from multiple, identical observations into their midpoint, thus
        producing smaller figures. Note that the resulting plot will NOT be the
        exact CCDF function, but an approximation.

    Additional keyword arguments are passed to `matplotlib.pyplot.loglog`.
    Returns a matplotlib axes object.

    """
    N = float(len(x))
    if copy:
        x = x.copy()
    x.sort()
    if fractional:
        t = []
        for x, chunk in groupby(enumerate(x, 1), itemgetter(1)):
            xranks, _ = zip(*list(chunk))
            t.append((float(x), xranks[0] + np.ptp(xranks) / 2.0))
        t = np.asarray(t)
    else:
        t = np.c_[np.asfarray(x), np.arange(N) + 1]
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
    ax = plt.gca()
    hist, _ = np.histogram(x, bins=bins)
    binsize = np.diff(np.asfarray(bins))
    hist = hist / binsize
    ax.loglog(bins[1:], hist, 'ow', **kwargs)
    return ax

if __name__ == '__main__':
    from scipy.stats import zipf
    x = zipf.rvs(2, size=10000)
    plt.figure(figsize=(9, 3))
    plt.subplot(131)
    plot_cdf(x, fractional=False)
    plt.title("CCDF")
    ylim = plt.ylim()

    plt.subplot(132)
    plot_cdf(x)
    plt.title("CCDF with fractional ranks")
    plt.ylim(*ylim)

    plt.subplot(133)
    plot_pdf_log2(x)
    plt.title("PDF")

    plt.tight_layout()
    plt.show()
