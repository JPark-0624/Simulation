# simulation/control/dot_robot_controller.py
import numpy as np

from simulation.physics.body import RigidBody2D

def perp(v: np.ndarray) -> np.ndarray:
    return np.array([v[1], -v[0]])

def unit(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x)
    return x / (n + eps)

class DotRobotController:
    """
    Force-controller for a single dot robot (RigidBody2D) interacting with boxes.
    Usage in animate_Boxpush.py:
        robot  = makePointRobot(q0_xy=[0.0, 0.0], v0_xy=[0.0, 0.0])
        ctrl   = DotRobotController(robot, mu=0.6)
        ctrl.setTarget(np.array([goalX, goalY]))
        ...
        u_xy = ctrl.computeForce(boxes, dt)     # 2D force to apply on the robot
        # EITHER inject into v_free (preferred) OR:
        applyRobotForce(robot, u_xy, dt)        # minimal fallback integrator
    """
    def __init__(self,
                 robot: RigidBody2D,            # RigidBody2D (the dot)
                 mu: float,
                 kpFree=20.0, kdFree=8.0,   # free-space PD gains
                 kn=2e3, dn=60.0,           # normal impedance gains
                 kt=1e3, dtg=40.0,          # tangential impedance gains
                 uMax=200.0):
        self.robot = robot
        self.mu = float(mu)
        self.kpFree, self.kdFree = float(kpFree), float(kdFree)
        self.kn, self.dn = float(kn), float(dn)
        self.kt, self.dtg = float(kt), float(dtg)
        self.uMax = float(uMax)

        self.pTarget = None
        self.vTarget = np.zeros(2)
        self.preloadMin = 1.0   # minimal desired normal force when engaged

    # ---------- public API ----------
    def setTarget(self, pTarget_xy: np.ndarray, vTarget_xy: np.ndarray | None = None):
        self.pTarget = np.array(pTarget_xy, dtype=float)
        if vTarget_xy is not None:
            self.vTarget = np.array(vTarget_xy, dtype=float)

    def computeForce(self, boxes: list, dt: float) -> np.ndarray:
        """
        Returns a 2D force u_xy to apply to the robot BEFORE contact impulses.
        """
        assert self.pTarget is not None, "Call setTarget(...) before computeForce"

        pr = self.robot.q[:2]
        vr = self.robot.v[:2]

        # 1) Pick the closest box and compute contact frame (phi, n, cp)
        box, (phi, n, pC) = self._closestBoxAndContact(boxes, pr)
        t = perp(n)

        # 2) Relative velocity at contact point (box point minus robot)
        vBoxPoint = self._velOfBoxPoint(box, pC)
        vRel = vBoxPoint - vr
        vn, vt = float(n @ vRel), float(t @ vRel)

        # 3) Mode selection
        if phi > 0.0:
            # Free-space: PD to target (world frame)
            e  = pr - self.pTarget
            ed = vr - self.vTarget
            aDes = -self.kpFree * e - self.kdFree * ed
            u = self.robot.m * aDes
            return np.clip(u, -self.uMax, self.uMax)

        # In contact: impedance in contact frame, then friction-cone projection
        an_star = - self.kn * phi - self.dn * vn          # compress to maintain small preload
        at_star = - self.kt * 0.0 - self.dtg * vt         # drive vt -> 0 (prefer stick)

        lamN_star = max(self.preloadMin, self.robot.m * an_star)
        lamT_star = self.robot.m * at_star

        lamN = max(0.0, lamN_star)
        lamT = np.clip(lamT_star, -self.mu * lamN, +self.mu * lamN)

        fContact = lamN * n + lamT * t
        u = fContact                                   # m a_r = u - f_c → choose u ≈ f_c
        return np.clip(u, -self.uMax, self.uMax)

    # ---------- internals ----------
    def _closestBoxAndContact(self, boxes: list, pRobot: np.ndarray):
        best = None
        bestBox = None
        for b in boxes:
            try:
                phi, n, pC = b.shape.signedDistance(b, pRobot)
            except Exception:
                # Fallback if shape not present; treat as no contact
                d = pRobot - b.q[:2]
                dist = np.linalg.norm(d)
                n = np.array([1.0, 0.0]) if dist < 1e-12 else d / (dist + 1e-12)
                pC = pRobot.copy()
                phi = dist  # "far"
            if (best is None) or (phi < best[0]):
                best = (float(phi), unit(n), np.array(pC, dtype=float))
                bestBox = b
        assert bestBox is not None, "No boxes provided to controller."
        return bestBox, best

    def _velOfBoxPoint(self, box, pWorld: np.ndarray) -> np.ndarray:
        # Jp maps [vx, vy, ω] -> v_point in world: [[1,0,-r_y],[0,1,r_x]]
        rBody  = box.worldPointToBody(pWorld)
        c, s   = np.cos(box.q[2]), np.sin(box.q[2])
        R      = np.array([[c, -s],[s, c]])
        rWorld = R @ rBody
        Jp     = np.array([[1.0, 0.0, -rWorld[1]],
                           [0.0, 1.0,  rWorld[0]]])
        return Jp @ box.v

# ---- optional helper if your world doesn't inject into v_free yet ----
def applyRobotForce(robot, u_xy: np.ndarray, h: float):
    """
    Minimal semi-implicit update for the robot *before* contact impulses.
    Prefer injecting into v_free in your world; use this only as a quick fallback.
    """
    a = u_xy / robot.m
    robot.v[:2] += a * h
    robot.q[:2] += robot.v[:2] * h
