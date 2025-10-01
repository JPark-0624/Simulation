import numpy as np

class RigidBody2D:
    def __init__(self, mass, inertia, q, v, radius=None):
        self.m = float(mass)
        self.I = float(inertia)
        self.q = np.array(q, dtype=float)  # [x, y, theta]
        self.v = np.array(v, dtype=float)  # [vx, vy, omega]
        self.r = None if radius is None else float(radius)

    @property
    def M(self):
        return np.diag([self.m, self.m, self.I])
