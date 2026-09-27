import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np


def colors_from_vals(vs, q=None, cmap="viridis", vmin=None, vmax=None):
    if isinstance(q,list):
        v_min = np.quantile(vs, q[0])
        v_max = np.quantile(vs, q[1])
    elif isinstance(q,float):
        v_min = np.quantile(vs, q)
        v_max = np.max(vs)
    else:
        v_min = np.min(vs)
        v_max = np.max(vs)

    if vmax is not None: v_max = vmax
    if vmin is not None: v_min = vmin

    cm  = getattr(plt.cm, cmap)
    vs_ = np.clip(vs, v_min, v_max)
    cs  = cm(plt.Normalize()(vs_))
    return cs

def line_collection(a, b, c=None, linewidth=1, **kwargs):
    lines = np.column_stack((a, b)).reshape(-1, 2, 2)
    lc = LineCollection(lines, colors=c, linewidths=linewidth, **kwargs)
    return lc

def plot_segs(segs, c="k", linewidth=1, ax=None,  **kwargs):
    if ax is None: ax = plt.gca()
    n = 10
    segs = segs.reshape(-1,2,2)
    a = segs[:,0]
    b = segs[:,1]
    lc = line_collection(a, b, linewidth=linewidth, **kwargs)
    lc.set_colors(c)
    ax.add_collection(lc)

