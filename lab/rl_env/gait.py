"""Text gait diagram from a per-step planted array: which feet are down, when.

:func:`strips`: one line per foot (``x`` planted, ``.`` not), time along the line,
wrapped into blocks separated by an empty line.
"""
import numpy as np


def _bin(planted, k):
    """Planted per bin of ``k`` steps: a foot counts as planted when it is for at
    least half the bin. Returns (n_bins, 4) bool."""
    n = len(planted) // k * k
    return planted[:n].reshape(-1, k, planted.shape[1]).mean(1) >= 0.5


def strips(planted, dt, *, bin_s=None, width=100) -> str:
    """One line per foot, one character per ``bin_s`` seconds, ``width`` characters
    per block.

    Args:
        planted: (T, 4) bool, per step and foot.
        dt: seconds per step.
        bin_s: seconds per character (default: one step).
        width: characters per block (a block is ``width * bin_s`` seconds).

    Returns:
        The diagram as a multi-line string.
    """
    k = max(1, round((bin_s or dt) / dt))
    b = _bin(np.asarray(planted, bool), k)
    s = k * dt
    out = []
    for i0 in range(0, len(b), width):
        blk = b[i0:i0 + width]
        out.append(f"t {i0 * s:5.2f} .. {(i0 + len(blk)) * s:5.2f} s   ({s:.2f} s per char)")
        for f in range(b.shape[1]):
            out.append(f"  foot{f}  " + "".join("x" if p else "." for p in blk[:, f]))
        out.append("")
    return "\n".join(out)
