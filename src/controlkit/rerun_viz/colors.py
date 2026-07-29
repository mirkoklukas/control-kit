"""Palette and scalar-to-color mapping.

Pure numpy -- no rerun. Lifts out of this package unchanged if a second drawing
backend ever wants the same colors.
"""

import numpy as np

BODY_COLOR = (66, 135, 245)
SEGMENT_COLOR = (90, 90, 90)
CONTACT_COLOR = (255, 180, 0)
MOMENT_COLOR = (150, 80, 240)
EDGE_COLOR = (40, 40, 40)


def colors_from_values(values, *, cmap="viridis", vmin=None, vmax=None):
    """Map scalars to RGB through a matplotlib colormap.

    Args:
        values: (N,) scalars.
        cmap: matplotlib colormap name.
        vmin: lower bound (default: data min).
        vmax: upper bound (default: data max).

    Returns:
        (N, 3) uint8 RGB. Constant input maps to mid-colormap.
    """
    import matplotlib
    from matplotlib.colors import Normalize

    v = np.asarray(values, dtype=float).reshape(-1)
    lo = v.min() if vmin is None else vmin
    hi = v.max() if vmax is None else vmax
    rgba = matplotlib.colormaps[cmap](Normalize(lo, hi)(v))   # (N, 4) float in [0, 1]
    return (rgba[:, :3] * 255).astype(np.uint8)


def as_per_item(color, n):
    """Broadcast a single RGB(A) or a per-item array to (n, k).

    Args:
        color: one RGB(A) tuple, or an (n, 3/4) array.
        n: number of items.

    Returns:
        (n, k) color array.
    """
    c = np.asarray(color)
    return np.tile(c, (n, 1)) if c.ndim == 1 else c
