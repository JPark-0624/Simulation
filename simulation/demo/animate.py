# at module top
ANIM = None
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation
from collections import deque


from simulation.physics.rigid_body2d import RigidBody2D
from simulation.geometry.cup_builder import build_cup
from simulation.engine.world import World
from simulation.geometry.draw_utils import draw_cup, impulses_split, ensure_quivers


def run_animation():
    
    h = 1/1000.0   

    cup = build_cup(x_left=-0.6, x_right=0.6, y_bottom=0.0, height=0.8)

    # make as many balls as you want
    balls = [
        RigidBody2D(mass=1.0, inertia=0.01, q=[-0.25, 0.45, 0.0], v=[2.0, 2.0, 0.0], radius=0.1),
        RigidBody2D(mass=1.0, inertia=0.01, q=[+0.25, 0.45, 0.0], v=[-2.0, -2.0, 0.0], radius=0.1),
        # add more here if you like
    ]

    mu, beta, restitution = 0.6, 0.25, 0.7
    world = World(bodies=balls, cup_segments=cup, mu=mu, beta=beta, restitution=restitution)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-0.8, 0.8)
    ax.set_ylim(-0.1, 1.2)
    ax.set_title("Ball in a Cup — Anitescu CCP (QP via OSQP)")
    ax.grid(True, alpha=0.3)

    # ball patches
    patches = []
    patches2 = []
    colors = ['C0','C1','C2','C3','C4']
    for i, b in enumerate(balls):
        p = Circle((b.q[0], b.q[1]), b.r, fc=colors[i % len(colors)], ec='k', lw=1.2)
        p2 = Circle((b.q[0]+b.r*math.cos(b.q[2]), b.q[1] + b.r*math.sin(b.q[2])), 0.01, fc='k', ec='k', lw=1.2)  # to show rotation
        ax.add_patch(p)
        ax.add_patch(p2)
        patches.append(p)
        patches2.append(p2)

    substeps_per_frame = 4

    draw_cup(ax, cup)

    # visual scale for arrow length (purely visual)
    IMP_SCALE = 40

    # lists of quivers that we will reuse each frame
    quivers_n = []  # red (normal)
    quivers_t = []  # blue (tangential)

    def update(_):
    # run substeps but only keep the LAST contacts/impulses to draw
        contacts = []; lam = None
        for _ in range(substeps_per_frame):
            contacts, lam = world.step(h)

        artists = []

        # update body patches (positions and heading dots)
        for p, p2, b in zip(patches, patches2, world.bodies):
            p.center = (b.q[0], b.q[1])
            p2.center = (b.q[0] + b.r * math.cos(b.q[2]),
                        b.q[1] + b.r * math.sin(b.q[2]))
            artists.extend([p, p2])

        # draw impulses only if we have contacts & lam
        if lam is not None and len(contacts) > 0:
            idata = impulses_split(contacts, lam)  # [{'p','n','t','jn','jt'}, ...]
            # ensure the right number of arrows (reuse existing, hide extras)
            quivers_n[:] = ensure_quivers(ax, len(idata), existing=quivers_n, color='r')  # normal (red)
            quivers_t[:] = ensure_quivers(ax, len(idata), existing=quivers_t, color='b')  # tangential (blue)

            # update the arrows for each contact
            for qn, qt, d in zip(quivers_n, quivers_t, idata):
                p = d['p']
                jn_vec = d['jn'] * d['n'] * IMP_SCALE
                jt_vec = d['jt'] * d['t'] * IMP_SCALE
                qn.set_offsets([p]); qn.set_UVC([jn_vec[0]], [jn_vec[1]])
                qt.set_offsets([p]); qt.set_UVC([jt_vec[0]], [jt_vec[1]])
            artists.extend(quivers_n)
            artists.extend(quivers_t)
        else:
            # no contacts: hide any existing arrows
            quivers_n[:] = ensure_quivers(ax, 0, existing=quivers_n, color='r')
            quivers_t[:] = ensure_quivers(ax, 0, existing=quivers_t, color='b')

        return artists   # important if blit=True

    
    global ANIM
    ANIM = FuncAnimation(fig, update, frames=1000, interval=1000/60, blit=False)
    plt.show()




if __name__ == "__main__":
    run_animation()
