import numpy as np
import matplotlib.pyplot as plt

def plot_cdf(x, **kwargs):
    '''
    Adds a log-log CDF plot to the current axes.

    Arguments
    ---------
    x : array_like
        The data to plot

    Additional keyword arguments are passed to `matplotlib.pyplot.loglog`.
    '''
    N = len(x)
    x.sort()
    r = np.arange(N, 0, -1, dtype=float) / N
    fig = plt.gcf()
    ax = plt.gca()
    ax.loglog(x, r, 'ow', **kwargs)
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

