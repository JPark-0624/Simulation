import numpy as np
from dataclasses import dataclass, field

from simulation.physics.rigid_body2d import RigidBody2D
from simulation.collision.detect2d import box_vs_segment_contact_obb, circle_vs_segment_contact, circle_circle_contact, point_vs_obb_contact
from simulation.collision.jacobian2d import body_point_jacobians
from simulation.engine.solver_ccp_qp import build_qp_from_contacts, solve_ccp_qp
from simulation.engine.debug_checks import check_ccp_solution

@dataclass
class World_Boxpush:
    box: RigidBody2D                     # RigidBody2D
    robot: RigidBody2D                   # single RigidBody2D
    wall_segments: list           # list of Segment
    g: float = 0               # gravity = 0 this example is on a 2D plane (no gravity)
    mu: float = 0.6
    beta: float = 0.05
    restitution: float = 0.0

    # ------- utilities -------
    @property
    def bodies(self):
        return [self.box, self.robot] #unified order : [box, robot]
    
    @property
    def N(self): return len(self.bodies)
    
    def _blockdiag_M(self):
        """3N x 3N block diagonal mass matrix."""
        M = np.zeros((3*self.N, 3*self.N))
        for i, b in enumerate(self.bodies):
            M[3*i:3*i+3, 3*i:3*i+3] = b.M
        return M

    def _stack_v(self):
        """3N vector from per-body velocities."""
        v = np.zeros(3*self.N)
        for i, b in enumerate(self.bodies):
            v[3*i:3*i+3] = b.v
        return v

    def _apply_v(self, v):
        """write back 3N vector to bodies."""
        for i, b in enumerate(self.bodies):
            b.v = v[3*i:3*i+3]

    def _integrate_positions(self, h):
        for b in self.bodies:
            b.q[:2] += h * b.v[:2]
            b.q[2]  += h * b.v[2]

    def _stack_q(self):
        q = np.zeros(3*self.N)
        for i, b in enumerate(self.bodies):
            q[3*i:3*i+3] = b.q
        return q

    def _apply_q(self, qVec):
        for i, b in enumerate(self.bodies):
            b.q = qVec[3*i:3*i+3]

    def _injectRobotForceIntoVfree(self, v_free: np.ndarray, h: float, u_xy: np.ndarray):
        """
        Adds h * M^{-1} * tau to the robot's slice of v_free.
        tau = [Fx, Fy, 0], robot global index = len(self.box) + robotIdx.
        """
        if u_xy is None:
            return
        # 1) compute the correct global slice for the robot
        iRobot = len(self.bodies)-1             # [boxes..., robots...]
        start, end = 3 * iRobot, 3 * iRobot + 3

        # 2) build tau and apply via the robot's mass matrix
        tau = np.array([float(u_xy[0]), float(u_xy[1]), 0.0], dtype=float)
        Mi  = self.robot.M                    # 3x3

        # 3) in-place update of v_free on the robot slice
        v_free[start:end] += h * np.linalg.solve(Mi, tau)


    # ------- contact assembly helpers -------
    def _globalize_rows_single(self, i_body, Jn_loc, Jt_loc):
        """Pad 1x3 local rows into 1x(3N) at body i."""
        Jn = np.zeros((1, 3*self.N))
        Jt = np.zeros((1, 3*self.N))
        Jn[0, 3*i_body:3*i_body+3] = Jn_loc
        Jt[0, 3*i_body:3*i_body+3] = Jt_loc
        return Jn, Jt

    def _globalize_rows_pair(self, i, j, Jn_i, Jt_i, Jn_j, Jt_j):
        """Place two bodies’ 1x3 rows into 1x(3N)."""
        Jn = np.zeros((1, 3*self.N))
        Jt = np.zeros((1, 3*self.N))
        Jn[0, 3*i:3*i+3] = Jn_i;  Jt[0, 3*i:3*i+3] = Jt_i
        Jn[0, 3*j:3*j+3] = Jn_j;  Jt[0, 3*j:3*j+3] = Jt_j
        return Jn, Jt

    def _gather_contacts(self):
        """
        Build contact list with global Jacobian rows for:
        - box vs wall segment (bilateral env contact)
        - point robot vs box (unilateral object contact)
        Each entry: {'p','n','t','phi','Jn','Jt'} where Jn/Jt are (1 x 3N).
        """
        contacts = []
        N = self.N
        iRobot = len(self.bodies) -1 # global index of the single robot in [boxes..., robots...]
        boxes = [self.box]
        # ---------- (A) Box vs Wall (bilateral) ----------
        for iBox, b in enumerate(boxes):
            # Skip if no box shape
            if not hasattr(b, "shape") or b.shape is None:
                continue
            # OBB parameters
            center = b.q[:2]
            theta  = float(b.q[2])
            half_extents = b.shape.halfExtents if hasattr(b.shape, "halfExtents") else None
            if half_extents is None:
                continue

            for seg in self.wall_segments:
                c = box_vs_segment_contact_obb(center, half_extents, theta, seg, sep_tol=0.0)
                if c is None:
                    continue  # clearly separated

                n, t, pC, phi = c['n'], c['t'], c['p'], float(c['phi'])
                if phi > 0.0:
                    continue  # positive gap → no contact row

                # Box Jacobian at contact point
                rBody = b.worldPointToBody(pC)
                Jp_box = b.contactJacobianAt(rBody)       # 2x3
                Jn_box = (n @ Jp_box).reshape(1, 3)       # 1x3
                Jt_box = (t @ Jp_box).reshape(1, 3)       # 1x3

                # Globalize rows: only the box moves (wall static)
                Jn = np.zeros((1, 3*N)); Jt = np.zeros((1, 3*N))
                Jn[0, 3*iBox:3*iBox+3] = Jn_box
                Jt[0, 3*iBox:3*iBox+3] = Jt_box

                contacts.append({'p': pC, 'n': n, 't': t, 'phi': phi, 'Jn': Jn, 'Jt': Jt})

        # ---------- (B) Point Robot vs Box (unilateral) ----------
        if self.robot is not None:
            r = self.robot
            pR = r.q[:2]

            for iBox, b in enumerate(boxes):
                if not hasattr(b, "shape") or b.shape is None:
                    continue
                center = b.q[:2]
                theta  = float(b.q[2])
                half_extents = b.shape.halfExtents if hasattr(b.shape, "halfExtents") else None
                if half_extents is None:
                    continue

                c = point_vs_obb_contact(pR, center, half_extents, theta, sep_tol=0.0)
                if c is None:
                    continue

                n, t, pC, phi = c['n'], c['t'], c['p'], float(c['phi'])
                if phi > 0.0:
                    continue  # positive gap → no unilateral contact

                # Box side rows
                rBody   = b.worldPointToBody(pC)
                Jp_box  = b.contactJacobianAt(rBody)         # 2x3
                Jn_box  = (n @ Jp_box).reshape(1, 3)         # 1x3
                Jt_box  = (t @ Jp_box).reshape(1, 3)         # 1x3

                # Robot (point) rows: velocity of contact point is robot v[:2]
                # so Jacobian wrt [vx,vy,ω] is [1,0,0; 0,1,0] → project to n,t and apply action–reaction sign
                Jn_robot = np.array([ -n[0], -n[1], 0.0 ], dtype=float).reshape(1, 3)
                Jt_robot = np.array([ -t[0], -t[1], 0.0 ], dtype=float).reshape(1, 3)

                # Globalize: fill box slice and robot slice
                Jn = np.zeros((1, 3*N)); Jt = np.zeros((1, 3*N))
                Jn[0, 3*iBox:3*iBox+3] = Jn_box
                Jt[0, 3*iBox:3*iBox+3] = Jt_box
                Jn[0, 3*iRobot:3*iRobot+3] = Jn_robot
                Jt[0, 3*iRobot:3*iRobot+3] = Jt_robot

                contacts.append({'p': pC, 'n': n, 't': t, 'phi': phi, 'Jn': Jn, 'Jt': Jt})

        return contacts


    # ------- one substep -------
    def step(self, h, controller=None):
        # free motion (gravity only)
        M = self._blockdiag_M()
        v = self._stack_v()
        q = self._stack_q()
        v_free = v.copy()

        # for i in range(self.N): # there is no gravity in this 2D plane example
        #     v_free[3*i+1] += -h * self.g

        u_xy = None
        if controller is not None:
            # accept either .computeControl(world, h) or .computeForce(boxes, h)
            if hasattr(controller, "computeControl"):
                u_xy = controller.computeControl(self, h)
            else:
                u_xy = controller.computeForce([self.box], h)
        self._injectRobotForceIntoVfree(v_free, h, u_xy)
        # contacts
        contacts = self._gather_contacts()
        lam = np.zeros(0, dtype=float)  # default if no contacts

        # build + solve QP (reuses your existing builder)
        qp = build_qp_from_contacts(M, v_free, contacts, h,
                                    mu=self.mu, beta=self.beta, restitution=self.restitution)
        if qp is None:
            v_post = v_free
        else:
            P, q, A, l, u, G = qp
            lam = solve_ccp_qp(P, q, A, l, u, G)

            # --- after building P,q,A,l,u ---
            assert A.shape[0] == l.shape[0] == u.shape[0], \
                f"Row mismatch: A{A.shape}, l{l.shape}, u{u.shape}"

            J_rows = []
            for c in contacts:
                J_rows.extend([c['Jn'], c['Jt']])
            J = np.vstack(J_rows) if J_rows else np.zeros((0, 3*self.N))
            v_post = v_free + np.linalg.inv(M) @ (J.T @ lam)

        # write back + integrate
        self._apply_v(v_post)
        self._integrate_positions(h)

        return contacts, lam
