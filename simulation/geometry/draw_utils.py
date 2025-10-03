# simulation/geometry/draw_utils.py
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle, Circle
from matplotlib.transforms import Affine2D


class DrawUtils:

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


    def draw_walls(ax, wall_segments, color="k", lw=2.0, zorder=5):
        """Draws the walls as a LineCollection and returns the artist."""
        if not wall_segments:
            return None
        segs = [np.vstack([seg.a, seg.b]) for seg in wall_segments]  # each (2,2)
        lc = LineCollection(segs, colors=color, linewidths=lw, zorder=zorder)
        ax.add_collection(lc)
        return lc
    @staticmethod
    def make_box_patch(ax, box, fc='C1', ec='k', lw=1.2, zorder=2):
        """
        Create a Rectangle patch centered at box.q[:2] with rotation box.q[2].
        The patch is defined in its local frame at (0,0) and transformed.
        """
        w = float(box.shape.width)
        h = float(box.shape.height)
        rect = Rectangle((-w/2, -h/2), w, h, fc=fc, ec=ec, lw=lw, zorder=zorder)
        # initial transform
        cx, cy, th = float(box.q[0]), float(box.q[1]), float(box.q[2])
        rect.set_transform(Affine2D().rotate_around(0.0, 0.0, th).translate(cx, cy) + ax.transData)
        ax.add_patch(rect)
        return rect

    @staticmethod
    def update_box_patch(ax, rect: Rectangle, box):
        w = float(box.shape.width); h = float(box.shape.height)
        rect.set_width(w); rect.set_height(h)
        rect.set_xy((-w/2, -h/2))  # keep patch centered at local origin
        cx, cy, th = float(box.q[0]), float(box.q[1]), float(box.q[2])
        rect.set_transform(Affine2D().rotate_around(0.0, 0.0, th).translate(cx, cy) + ax.transData)

    # -------- Robot (point) --------
    @staticmethod
    def make_robot_patch(ax, robot, radius=0.015, fc='C0', ec='k', lw=1.2, zorder=3):
        circ = Circle((float(robot.q[0]), float(robot.q[1])), radius, fc=fc, ec=ec, lw=lw, zorder=zorder)
        ax.add_patch(circ)
        return circ

    @staticmethod
    def update_robot_patch(circle: Circle, robot):
        circle.center = (float(robot.q[0]), float(robot.q[1]))

    # -------- Goal marker --------
    @staticmethod
    def make_goal_patch(ax, goal_xy, radius=0.018, fc='none', ec='C2', lw=2.0, zorder=1):
        goal = Circle((float(goal_xy[0]), float(goal_xy[1])), radius,
                      fc=fc, ec=ec, lw=lw, linestyle='--', zorder=zorder)
        ax.add_patch(goal)
        return goal