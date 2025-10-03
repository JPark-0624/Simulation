import numpy as np
from dataclasses import dataclass

def rot2(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s],
                     [s,  c]])

def hat2(v: np.ndarray) -> np.ndarray:
    # 2D "perp" operator for torque: a⊥ b = a_x*b_y - a_y*b_x
    # not a full 3D hat, but useful: perp(v) @ w = v_x*w_y - v_y*w_x
    return np.array([[0., -1.],
                     [1.,  0.]])

class Shape2D:
    def signedDistance(self, body, pWorld: np.ndarray):
        raise NotImplementedError

class PointShape(Shape2D):
    # single dot robot: geometric point (zero radius)
    def signedDistance(self, body, pWorld):
        # distance from point to itself is exactly position error
        # For completeness, return phi=||p - p_r||, outward normal, closest point.
        pr = body.q[:2]
        d = pWorld - pr
        dist = np.linalg.norm(d)
        if dist == 0:
            n = np.array([1.0, 0.0])
        else:
            n = d / dist
        return dist, n, pr  # closest point is the robot itself

@dataclass
class BoxShape(Shape2D):
    width: float
    height: float

    @property
    def halfExtents(self):
        return np.array([0.5*self.width, 0.5*self.height])

    def signedDistance(self, body, pWorld: np.ndarray):
        """
        Signed distance from a point to an oriented box (OBB).
        phi >= 0 outside; phi == 0 on boundary; negative inside.
        Also returns outward normal at the closest point and that point in world.
        """
        R = rot2(body.q[2])
        pBody = R.T @ (pWorld - body.q[:2])        # world -> body
        a = self.halfExtents
        # distance to AABB in body frame
        q = np.abs(pBody) - a
        outside = np.maximum(q, 0.0)
        phi = np.linalg.norm(outside)
        inside_margin = np.maximum(q[0], q[1])
        if inside_margin <= 0:  # inside box: negative distance
            phi = inside_margin

        # closest point on the box in body frame
        cpBody = np.clip(pBody, -a, a)

        # outward normal in body frame
        eps = 1e-12
        if phi > eps:  # outside: gradient from outside vector
            nBody = np.zeros(2)
            mask = q > 0
            if mask.any():
                nBody[mask] = np.sign(pBody[mask])
                nBody = nBody / (np.linalg.norm(nBody) + eps)
            else:
                # on a corner exactly: use direction from cp to point
                dBody = pBody - cpBody
                nBody = dBody / (np.linalg.norm(dBody) + eps)
        else:
            # inside: normal points to the nearest face
            k = np.argmax(np.abs(pBody) - a)  # axis with largest penetration
            nBody = np.zeros(2); nBody[k] = np.sign(pBody[k])

        nWorld = R @ nBody
        cpWorld = body.q[:2] + R @ cpBody
        return phi, nWorld, cpWorld

class RigidBody2D:
    def __init__(self, mass, inertia, q, v, shape: Shape2D | None = None):
        self.m = float(mass)
        self.I = float(inertia)
        self.q = np.array(q, dtype=float)  # [x, y, theta]
        self.v = np.array(v, dtype=float)  # [vx, vy, omega]
        self.shape = shape

    @property
    def M(self):
        return np.diag([self.m, self.m, self.I])

    def bodyPointToWorld(self, rBody: np.ndarray) -> np.ndarray:
        R = rot2(self.q[2]); return self.q[:2] + R @ rBody

    def worldPointToBody(self, pWorld: np.ndarray) -> np.ndarray:
        R = rot2(self.q[2]); return R.T @ (pWorld - self.q[:2])

    def contactJacobianAt(self, rBody: np.ndarray) -> np.ndarray:
        """
        2x3 Jacobian mapping body generalized velocity v=[vx,vy,ω] to
        point velocity at world point R rBody + q[:2].
            v_point = J * v
        """
        R = rot2(self.q[2])
        rWorld = R @ rBody
        J = np.array([[1., 0., -rWorld[1]],
                      [0., 1.,  rWorld[0]]])
        return J

# --- Example constructors ---

def makeSquareBox(mass: float, width: float, height: float, q0, v0):
    I = mass * (width**2 + height**2) / 12.0
    shape = BoxShape(width, height)
    return RigidBody2D(mass, I, q0, v0, shape)

def makePointRobot(q0_xy, v0_xy=None):
    # Kinematic default: store as a "body" with zero inertia; controller decides dynamics.
    q = np.array([q0_xy[0], q0_xy[1], 0.0])
    v = np.array([0.0, 0.0, 0.0]) if v0_xy is None else np.array([v0_xy[0], v0_xy[1], 0.0])
    return RigidBody2D(mass=1.0, inertia=1.0, q=q, v=v, shape=PointShape())
