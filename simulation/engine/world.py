import numpy as np
from dataclasses import dataclass, field

from simulation.collision.detect2d import circle_vs_segment_contact, circle_circle_contact
from simulation.collision.jacobian2d import body_point_jacobians
from simulation.engine.solver_ccp_qp import build_qp_from_contacts, solve_ccp_qp
from simulation.engine.debug_checks import check_ccp_solution

@dataclass
class World:
    bodies: list                     # list of RigidBody2D
    cup_segments: list               # list of Segment
    g: float = 9.81               # gravity
    mu: float = 0.6
    beta: float = 0.05
    restitution: float = 0.0

    # ------- utilities -------
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
        """Return list of contact dicts with global Jn/Jt (1 x 3N)."""
        contacts = []

        # body–cup
        for i, b in enumerate(self.bodies):
            for seg in self.cup_segments:
                c = circle_vs_segment_contact(b.q[:2], b.r, seg)
                if c is None: 
                    continue
                Jn_loc, Jt_loc = body_point_jacobians(b.q, c['p'], c['n'], c['t'])
                Jn, Jt = self._globalize_rows_single(i, Jn_loc, Jt_loc)
                c.update({'Jn': Jn, 'Jt': Jt})
                contacts.append(c)

        # body–body (all unordered pairs)
        for i in range(self.N):
            for j in range(i+1, self.N):
                bi, bj = self.bodies[i], self.bodies[j]
                cc = circle_circle_contact(bi.q[:2], bi.r, bj.q[:2], bj.r)
                if cc is None: 
                    continue
                # action–reaction: flip n,t for body j
                Jn_i, Jt_i = body_point_jacobians(bi.q, cc['p'],  cc['n'],  cc['t'])
                Jn_j, Jt_j = body_point_jacobians(bj.q, cc['p'], -cc['n'], -cc['t'])
                Jn, Jt = self._globalize_rows_pair(i, j, Jn_i, Jt_i, Jn_j, Jt_j)
                cc.update({'Jn': Jn, 'Jt': Jt})
                contacts.append(cc)

        return contacts


    # ------- one substep -------
    def step(self, h):
        # free motion (gravity only)
        v_free = self._stack_v().copy()
        for i in range(self.N):
            v_free[3*i+1] += -h * self.g

        # contacts
        contacts = self._gather_contacts()
        lam = np.zeros(0, dtype=float)  # default if no contacts

        # dynamics matrices
        M = self._blockdiag_M()

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

            # Constraint residual check (primal feasibility)
            x = np.zeros_like(q)  # placeholder; replace with 'lam' after solve to re-check
            x = lam
            Ax = A @ x
            lower_viol = np.maximum(l - Ax, 0.0)
            upper_viol = np.maximum(Ax - u, 0.0)
            # print("max lower viol:", float(lower_viol.max()),
            #     "| max upper viol:", float(upper_viol.max()))
            # apply impulses with exactly the same J used in the builder
            J_rows = []
            for c in contacts:
                J_rows.extend([c['Jn'], c['Jt']])
            J = np.vstack(J_rows) if J_rows else np.zeros((0, 3*self.N))
            v_post = v_free + np.linalg.inv(M) @ (J.T @ lam)
            Minv = np.linalg.inv(M)
            
            
            # optional debug checks
            # ok = check_ccp_solution(
            #     contacts, v_free, v_post, lam,
            #     mu=self.mu, beta=self.beta, restitution=self.restitution, h=h,
            #     print_each=True,
            #     kkt=(G, q)  # only if you return them from builder; else pass None
            # )
            # if not ok:
            #     print(">>> DEBUG: some contact checks failed — inspect rows above.")

        # write back + integrate
        self._apply_v(v_post)
        self._integrate_positions(h)

        return contacts, lam
