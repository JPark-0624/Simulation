# simulation/engine/integrator.py
import numpy as np

# Delegates to the unified World pipeline
from simulation.engine.world import World

def step_world(world: World, h: float):
    """
    Single substep using the unified CCP->QP pipeline implemented in World.
    This function exists so demos (and external code) can 'step' without
    needing to know World internals.
    """
    return world.step(h)


# --- Back-compat thin wrapper (uses World under the hood) --------------------
from simulation.physics.rigid_body2d import RigidBody2D  # for type hints (optional)

def step_ball_in_cup(ball: RigidBody2D,
                     cup_segments,
                     h: float,
                     g: float = 9.81,
                     mu: float = 0.6,
                     beta: float = 0.05,
                     restitution: float = 0.0):
    """
    Backward-compatible single-ball step that *does not* assemble contacts.
    It wraps the new World so old demos keep working.

    NOTE:
      - No tangential bias (gamma_t) here; if you want damping/projection,
        implement it inside World.step() for consistency across N bodies.
    """
    # Build a temporary world with one body and the given cup
    tmp_world = World(
        bodies=[ball],
        cup_segments=cup_segments,
        g=g,
        mu=mu,
        beta=beta,
        restitution=restitution,
    )

    # Do one unified step (this will collect contacts and solve the QP)
    contacts, _ = tmp_world.step(h)

    # tmp_world writes ball.v and integrates ball.q in-place, so nothing else to do
    return contacts
