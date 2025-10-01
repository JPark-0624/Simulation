import numpy as np
import scipy.sparse as sp
import osqp

def choose_tangent_signs(contacts, v_free, eps=1e-4):
    """
    For each contact:
      - compute v_t^free = Jt @ v_free
      - choose s in {+1,-1} with hysteresis
      - flip Jt and the geometric tangent t accordingly
      - store s and v_t^free in the contact dict (for debugging/plotting)
    Returns contacts (mutated in-place).
    """
    for c in contacts:
        Jt = c['Jt']
        vt_free = float(Jt @ v_free)

        # previous sign (if you cached it last frame), else 0 (unknown)
        s_prev = c.get('t_sign', 0)

        if vt_free < -eps:
            s = +1
        elif vt_free > +eps:
            s = -1
        else:
            # hysteresis: keep previous if we have one; otherwise default +1
            s = s_prev if s_prev in (+1, -1) else +1

        # flip Jacobian row and geometric tangent
        c['Jt'] = s * Jt
        if 't' in c:
            c['t'] = s * c['t']

        # keep for downstream
        c['t_sign']  = s
        c['vt_free'] = vt_free

    return contacts


def build_qp_from_contacts(M, v_free, contacts, h, mu=0.6, beta=0.25, restitution=0.0, alpha_t=0.1):
    """Anitescu CCP as QP:
       min 0.5 λ^T G λ + b^T λ
       s.t. λ_n >= 0, |λ_t| <= μ λ_n and λ_t >= 0.
       lambda λ is in friction cone
       A * λ_c + b is in velocity cone
       where G = J M^{-1} J^T, b = J v_free + gamma
       gamma is velocity-level bias (stabilization + restitution)
       J is stacked contact Jacobian, each contact has (Jn, Jt) rows
       λ is stacked contact impulses, each contact has (λ_n, λ_t) components
       m = number of contacts     
       """
    m = len(contacts)
    if m == 0:
        return None


    # Stack J and velocity-level bias gamma
    J_rows, gamma_list = [], []
    for c in contacts:
        Jn, Jt = c['Jn'], c['Jt']
        # Baumgarte stabilization + restitution
        vn = float(Jn @ v_free)   
        phi = float(c['phi'])     # 
        
        gn = -(beta / h) * max(phi, 0.0) + restitution * min(vn, 0.0)
        
        gamma_list += [gn, 0.0]
        
        J_rows.extend([Jn, Jt])

    J = np.vstack(J_rows)                         # (2m, 3)
    gamma = np.array(gamma_list).reshape(-1, 1)   # (2m, 1)

    Minv = np.linalg.inv(M)
    G = J @ Minv @ J.T             # in ref paper, A = J M^{-1} J^T, here G = A. 
    b = (J @ v_free.reshape(-1,1) + gamma).reshape(-1)  # (2m,) # contain only bias term

    # QP matrices for OSQP
    # slight Tikhonov reg for conditioning
    P = sp.csc_matrix(G + 1e-10 * np.eye(G.shape[0]))
    q = b.copy()


    ### To Yifan, I tried to define the tangent bounds both case of 
    # 1) λ_t >=0 and 2) λ_t <= μ λ_n and changed the sign of Jt accordingly,
    # but the result was not good. So I just defined the bounds as below.
    # 2) |λ_t| ≤ μ λ_n  →  ( +λ_t - μ λ_n ≤ 0 ) & ( -λ_t - μ λ_n ≤ 0 )
    # Bounds for λ components  
    # order: [λ_n1, λ_t1, λ_n2, λ_t2, ...]
   
    dim = 2*m

    A_bounds = sp.eye(dim, format='csc')
    l = np.full(dim, -np.inf)
    u = np.full(dim,  np.inf)

    # λ_n ≥ 0,  λ_t free sign
    for i in range(m):
        l[2*i]   = 0.0      # normal >= 0
        # l[2*i+1] 

    # |λ_t| ≤ μ λ_n  →  ( +λ_t - μ λ_n ≤ 0 ) & ( -λ_t - μ λ_n ≤ 0 )
    rows, cols, data = [], [], []
    for i in range(m):
        idx_n, idx_t = 2*i, 2*i+1

        # +λ_t - μ λ_n ≤ 0
        rows += [2*i,   2*i]
        cols += [idx_t, idx_n]
        data += [1.0,   -mu]

        # -λ_t - μ λ_n ≤ 0
        rows += [2*i+1,   2*i+1]
        cols += [idx_t,   idx_n]
        data += [-1.0,    -mu]

    A_cone = sp.coo_matrix((data, (rows, cols)), shape=(2*m, dim)).tocsr()

    A = sp.vstack([A_bounds, A_cone], format='csc')
    l = np.concatenate([l, np.full(2*m, -np.inf)])
    u = np.concatenate([u, np.zeros(2*m)])


    return P, q, A, l, u, G

def solve_ccp_qp(P, q, A, l, u, G):
    prob = osqp.OSQP()
    prob.setup(
        P=P, q=q, A=A, l=l, u=u,
        polish=False,                 # try to land exactly on the bounds
        eps_abs=1e-7, eps_rel=1e-7,  # tighter feasibility/optimality
        max_iter=20000,
        verbose=False
    )
    res = prob.solve()
    if res.info.status_val not in (1, 2):
        raise RuntimeError(f"OSQP failed: {res.info.status}")
    return res.x  # (optionally also return res if you want y=dual)

