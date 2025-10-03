# at module top
ANIM = None
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation
from collections import deque


from simulation.physics.rigid_body2d import RigidBody2D
# from simulation.engine.world import World
from simulation.engine.world_Boxpush import World_Boxpush
from simulation.geometry.draw_utils import DrawUtils
from simulation.geometry.env_builder import build_walls
from simulation.physics.body import makeSquareBox, makePointRobot
from simulation.control.dot_robot_controller import DotRobotController


# ----------------------------- params --------------------------------
H                = 1.0 / 1000.0        # simulation step
SUBSTEPS_PER_FRAME = 4                 # physics steps per animation frame
MU, BETA, REST   = 0.6, 0.25, 0.7      # friction, Baumgarte, restitution
WORLD_MIN, WORLD_MAX = (0.0, 0.0), (1.0, 1.0)  # axes limits (match your walls)

GOAL_X, GOAL_Y   = 0.15, 0.80          # goal for the robot
BOX_INIT         = dict(mass=1.0, width=0.10, height=0.10,
                        q0=[0.15, 0.50, 0.0], v0=[0.0, 0.0, 0.0])
ROBOT_INIT       = dict(q0_xy=[0.15, 0.05], v0_xy=[0.0, 0.0])
ROBOT_DRAW_RADIUS = 0.015
GOAL_DRAW_RADIUS  = 0.020
IMP_SCALE         = 40.0               # visual scale for impulse quivers

# -------------------------- main animation ---------------------------


def run_animation_Boxpush():
    
    wallSegments = build_walls(x_lim=WORLD_MIN[0:1]+(WORLD_MAX[0],),
                            y_lim=WORLD_MIN[1:2]+(WORLD_MAX[1],))
    box   = makeSquareBox(**BOX_INIT)
    robot = makePointRobot(**ROBOT_INIT)
    controller = DotRobotController(robot, mu=MU)
    controller.setTarget(np.array([GOAL_X, GOAL_Y]))

    # world = World(bodies=balls, cup_segments=cup, mu=mu, beta=beta, restitution=restitution)

    world = World_Boxpush(box=box, robot=robot, wall_segments=wallSegments,
                        mu=MU, beta=BETA, restitution=REST)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(WORLD_MIN[0], WORLD_MAX[0])
    ax.set_ylim(WORLD_MIN[1], WORLD_MAX[1])
    ax.set_title("Box Pushing (point robot → OBB)")
    ax.grid(True, alpha=0.25)

    DrawUtils.draw_walls(ax, wallSegments)
    boxPatch   = DrawUtils.make_box_patch(ax, box)
    robotPatch = DrawUtils.make_robot_patch(ax, robot, radius=ROBOT_DRAW_RADIUS)
    goalPatch  = DrawUtils.make_goal_patch(ax, [GOAL_X, GOAL_Y], radius=GOAL_DRAW_RADIUS)

    # Impulse quivers (normal=red, tangent=blue)
    quiversN, quiversT = [], []

    # Optional traces
    SHOW_TRACES   = True
    TRACE_LEN_MAX = 400
    robotTraceXs, robotTraceYs = [], []
    boxTraceXs,   boxTraceYs   = [], []
    robotTraceLine = ax.plot([], [], lw=1.0, color='C0', alpha=0.6)[0] if SHOW_TRACES else None
    boxTraceLine   = ax.plot([], [], lw=1.0, color='C1', alpha=0.6)[0] if SHOW_TRACES else None

    # Overlay text for quick debugging
    textOverlay = ax.text(0.02, 0.98, "", transform=ax.transAxes, va='top', ha='left', fontsize=9)


    def stepAndDraw(_frameIdx):
        # 4) substeps of physics
        lastContacts, lastLam = [], None
        for _ in range(SUBSTEPS_PER_FRAME):
            lastContacts, lastLam = world.step(H, controller=controller)

        # 5) update patches (OBB rotation is handled inside DrawUtils)
        DrawUtils.update_box_patch(ax, boxPatch, box)
        DrawUtils.update_robot_patch(robotPatch, robot)

        artists = [boxPatch, robotPatch, goalPatch]

        # 6) impulses visualization (if any contact)
        if (lastLam is not None) and (len(lastContacts) > 0) and (np.size(lastLam) > 0):
            idata = DrawUtils.impulses_split(lastContacts, lastLam)  # [{'p','n','t','jn','jt'}, ...]
            quiversN[:] = DrawUtils.ensure_quivers(ax, len(idata), existing=quiversN, color='r')
            quiversT[:] = DrawUtils.ensure_quivers(ax, len(idata), existing=quiversT, color='b')

            for qn, qt, d in zip(quiversN, quiversT, idata):
                p = d['p']
                jnVec = d['jn'] * d['n'] * IMP_SCALE
                jtVec = d['jt'] * d['t'] * IMP_SCALE
                qn.set_offsets([p]); qn.set_UVC([jnVec[0]], [jnVec[1]])
                qt.set_offsets([p]); qt.set_UVC([jtVec[0]], [jtVec[1]])

            artists.extend(quiversN); artists.extend(quiversT)
        else:
            # ensure zero quivers when no contact
            quiversN[:] = DrawUtils.ensure_quivers(ax, 0, existing=quiversN, color='r')
            quiversT[:] = DrawUtils.ensure_quivers(ax, 0, existing=quiversT, color='b')

        # 7) traces (optional)
        if SHOW_TRACES:
            robotTraceXs.append(robot.q[0]); robotTraceYs.append(robot.q[1])
            boxTraceXs.append(box.q[0]);     boxTraceYs.append(box.q[1])
            if len(robotTraceXs) > TRACE_LEN_MAX:
                robotTraceXs.pop(0); robotTraceYs.pop(0)
                boxTraceXs.pop(0);   boxTraceYs.pop(0)
            robotTraceLine.set_data(robotTraceXs, robotTraceYs)
            boxTraceLine.set_data(boxTraceXs, boxTraceYs)
            artists.append(robotTraceLine); artists.append(boxTraceLine)

        # 8) overlay quick stats
        textOverlay.set_text(
            f"robot@({robot.q[0]:.3f},{robot.q[1]:.3f})  "
            f"box@({box.q[0]:.3f},{box.q[1]:.3f},{box.q[2]:.2f} rad)\n"
            f"v_r=({robot.v[0]:.3f},{robot.v[1]:.3f})  "
            f"v_b=({box.v[0]:.3f},{box.v[1]:.3f},{box.v[2]:.3f})"
        )
        artists.append(textOverlay)

        return artists

    anim = FuncAnimation(fig, stepAndDraw, frames=1000, interval=1000/60, blit=False)
    plt.show()
    return anim



if __name__ == "__main__":
    run_animation_Boxpush()
