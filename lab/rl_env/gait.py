"""Text gait diagrams from a per-step planted array: which feet are down, when.

:func:`strips`: one line per foot (``x`` planted, ``.`` not), time along the line,
wrapped into blocks separated by an empty line.
:func:`rows`: one line per step, time down the page -- the feet side by side and
how many are planted.

Both take ``min_support`` (the env's ``min_support``, e.g. 3 for a crawl) to mark
where fewer feet are planted than that.
"""
import numpy as np


def _bin(planted, k):
    """Planted per bin of ``k`` steps: a foot counts as planted when it is for at
    least half the bin. Returns (n_bins, 4) bool."""
    n = len(planted) // k * k
    return planted[:n].reshape(-1, k, planted.shape[1]).mean(1) >= 0.5


def strips(planted, dt, *, bin_s=None, width=100, min_support=None) -> str:
    """One line per foot, one character per ``bin_s`` seconds, ``width`` characters
    per block.

    Args:
        planted: (T, 4) bool, per step and foot.
        dt: seconds per step.
        bin_s: seconds per character (default: one step).
        width: characters per block (a block is ``width * bin_s`` seconds).
        min_support: if given, a ``short`` line under the feet marks with ``!`` the
            characters where fewer feet than this are planted (in a block that
            has any).

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

        # short = "".join("!" if min_support is not None and n < min_support else " "
                        # for n in blk.sum(1)).rstrip()
        # if short:                                    # only in a block that has some
            # out.append("  short  " + short)
        out.append("")
    return "\n".join(out)


def rows(planted, dt, *, seconds=None, min_support=None) -> str:
    """One line per step: its time, each foot (``x`` planted, ``.`` not) and how
    many are planted.

    Args:
        planted: (T, 4) bool, per step and foot.
        dt: seconds per step.
        seconds: only the first this many seconds (default: all of it).
        min_support: if given, a step with fewer feet planted than this is marked
            ``< min_support``.

    Returns:
        The table as a multi-line string, a header line first.
    """
    p = np.asarray(planted, bool)
    if seconds is not None:
        p = p[:max(1, round(seconds / dt))]
    feet = " ".join(f"f{f}" for f in range(p.shape[1]))
    out = [f"     t  {feet}   n"]
    for i, step in enumerate(p):
        n = int(step.sum())
        mark = f"  < {min_support}" if min_support is not None and n < min_support else ""
        out.append(f"{i * dt:6.2f}  " + " ".join(f" {'x' if x else '.'}" for x in step)
                   + f"   {n}{mark}")
    return "\n".join(out)
