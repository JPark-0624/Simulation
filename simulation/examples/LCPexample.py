import numpy as np
import scipy.sparse as sp
import osqp

def solve_lcp_via_qp(M, q, verbose=False):
    """
    Solve the monotone LCP:
        w = M z + q,   z >= 0,  w >= 0,  z^T w = 0
    by the equivalent QP:
        minimize 0.5 z^T M z + q^T z   subject to z >= 0
    Returns (z, w, info_dict).
    """
    M = np.array(M, dtype=float)
    q = np.array(q, dtype=float).reshape(-1)
    n = len(q)

    # QP in OSQP form: min 0.5 z^T P z + q^T z  s.t. l <= A z <= u
    P = sp.csc_matrix(M)
    A = sp.eye(n, format="csc")
    l = np.zeros(n)
    u = np.full(n, np.inf)

    prob = osqp.OSQP()
    prob.setup(P=P, q=q.copy(), A=A, l=l, u=u, verbose=verbose)
    res = prob.solve()

    if res.info.status_val not in (1, 2):  # 1=solved, 2=solved inaccurate
        raise RuntimeError(f"OSQP failed: {res.info.status}")

    z = res.x
    w = M @ z + q

    # KKT / LCP checks
    kkt_stationarity = np.linalg.norm(M @ z + q - w, ord=np.inf)  # here w is the dual for bound z>=0
    primal_feas = float(np.min(z))
    dual_feas   = float(np.min(w))
    comp        = float(z @ w)

    obj = 0.5 * z @ (M @ z) + q @ z

    info = dict(obj=obj,
                kkt_stationarity_inf_norm=kkt_stationarity,
                primal_feasible=(primal_feas >= -1e-10),
                dual_feasible=(dual_feas   >= -1e-10),
                complementarity=comp)

    return z, w, info

if __name__ == "__main__":
    # --- Example 1: SPD M, strictly interior solution (inactive bounds) ---
    # LCP: w = M z + q, z >= 0, w >= 0, z^T w = 0
    M = np.array([[2.0, 1.0],
                  [1.0, 2.0]])  # SPD (monotone LCP)
    q = np.array([-5.0, -6.0])  # chosen so the unconstrained minimizer is positive

    z, w, info = solve_lcp_via_qp(M, q)
    print("=== Example 1 (interior) ===")
    print("M=\n", M, "\nq=", q)
    print("z* =", z)
    print("w* =", w, "   (should be ~0 for interior case)")
    print("Objective =", info["obj"])
    print("Primal feasible (z>=0)?", info["primal_feasible"])
    print("Dual feasible (w>=0)?  ", info["dual_feasible"])
    print("Complementarity z^T w =", info["complementarity"])
    print("Stationarity ||Mz+q-w||_inf =", info["kkt_stationarity_inf_norm"])
    print()

    # --- Example 2: Active bound (some z_i = 0) to see complementarity at the boundary ---
    # Pick q so that one component wants to go negative; projection hits z_i = 0
    M2 = np.array([[2.0, 0.0],
                   [0.0, 1.0]])
    q2 = np.array([+1.0, -2.0])  # unconstrained minimizer solves M2 z + q2 = 0 -> z = [-0.5, 2.0]
                                 # After z>=0, we expect z1=0, z2=2
    z2, w2, info2 = solve_lcp_via_qp(M2, q2)
    print("=== Example 2 (active bound) ===")
    print("M=\n", M2, "\nq=", q2)
    print("z* =", z2, "   (expect ~[0, 2])")
    print("w* =", w2, "   (w1>0 active; w2~0 interior)")
    print("Objective =", info2["obj"])
    print("Primal feasible (z>=0)?", info2["primal_feasible"])
    print("Dual feasible (w>=0)?  ", info2["dual_feasible"])
    print("Complementarity z^T w =", info2["complementarity"])
    print("Stationarity ||Mz+q-w||_inf =", info2["kkt_stationarity_inf_norm"])
