# simulation/geometry/draw_utils.py
import numpy as np
from matplotlib.collections import LineCollection

def draw_cup(ax, cup_segments, color="k", lw=2.0, zorder=5):
    """Draws the cup as a LineCollection and returns the artist."""
    if not cup_segments:
        return None
    segs = [np.vstack([seg.a, seg.b]) for seg in cup_segments]  # each (2,2)
    lc = LineCollection(segs, colors=color, linewidths=lw, zorder=zorder)
    ax.add_collection(lc)
    return lc


def impulses_split(contacts, lam):
    """
    Returns, per contact i:
      dict with keys: 'p' (2,), 'n' (2,), 't' (2,), 'jn' (scalar), 'jt' (scalar)
    """
    out = []
    for i, c in enumerate(contacts):
        out.append({
            'p': np.asarray(c['p']),
            'n': np.asarray(c['n']),
            't': np.asarray(c['t']),
            'jn': float(lam[2*i]),        # λ_n
            'jt': float(lam[2*i + 1]),    # λ_t (can be ±)
        })
    return out

def ensure_quivers(ax, count, existing=None, color='r'):
    """
    Make sure we have exactly 'count' quiver artists on axes 'ax'.
    Each quiver shows one arrow at one point.
    Returns the list of quivers (length == count).
    """
    qs = [] if existing is None else list(existing)
    while len(qs) < count:
        q = ax.quiver([0.0], [0.0], [0.0], [0.0],
                      angles='xy', scale_units='xy', scale=1.0,
                      color=color, width=0.006)
        qs.append(q)
    # hide extras if any
    for j in range(count, len(qs)):
        qs[j].set_offsets([[np.nan, np.nan]])
        qs[j].set_UVC([0.0], [0.0])
    return qs[:count]
