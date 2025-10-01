# simulation/engine/debug_checks.py
import numpy as np

# debug_checks.py 상단 어딘가에 추가
def _safe_fmt(x):
    try:
        return f"{float(x):+.3e}"
    except Exception:
        return "n/a"

def _resolve_vt_components(contact, v_free):
    """
    choose_tangent_signs를 쓰지 않아도 vt_free(orig)/vt_free(flipped)를 최대한 복원.
    - orig : Jt_raw가 있으면 그것으로, 없으면 현재 Jt로 계산
    - flipped : vt_free_after가 있으면 그 값, 없으면 현재 Jt로부터 계산
    """
    Jt_raw = contact.get('Jt_raw', None)
    Jt_cur = contact.get('Jt', None)

    vt_orig = None
    if Jt_raw is not None:
        vt_orig = float(Jt_raw @ v_free)
    elif Jt_cur is not None:
        vt_orig = float(Jt_cur @ v_free)

    vt_after = contact.get('vt_free_after', None)
    if vt_after is None and Jt_cur is not None:
        vt_after = float(Jt_cur @ v_free)

    # 부호 선택을 쓰지 않는다면 s는 +1로 가정
    s = int(contact.get('t_sign', +1))
    return vt_orig, vt_after, s



def check_ccp_solution(contacts, v_free, v_post, lam, mu, beta, restitution, h, *,
                       print_each=True, kkt=None):
    """
    Verifies, per contact k:
      1) Dissipation in tangential direction:
            λ_t * (s * v_t^free) <= 0    where s = c['t_sign'], v_t^free = c['vt_free']
      2) Friction strip:
            0 <= λ_t <= μ λ_n
      3) Normal CCP complementarity (velocity-level):
            s_n = v_n^+ + beta*max(phi,0)/h + restitution*min(v_n^-,0) >= 0
            λ_n >= 0
            λ_n * s_n ≈ 0

    Also prints a KKT stationarity infinity-norm if provided via kkt=(G, q).
    Returns True if all contacts pass within tolerances, else False.
    """
    tol = 1e-8
    ok_all = True

    # convenience
    def v_of(row, v):
        return float(row @ v)

    # optional KKT residual: stationarity = G λ + q
    if kkt is not None:
        G, q = kkt
        stat = G @ lam + q
        kkt_inf = float(np.linalg.norm(stat, ord=np.inf))
        if print_each:
            print(f"[KKT] ||Gλ + q||_inf = {kkt_inf:.3e}")

    # loop contacts
    for k, c in enumerate(contacts):
        Jn, Jt = c['Jn'], c['Jt']                 # NOTE: Jt is already flipped by your builder
        ln = float(lam[2*k])
        lt = float(lam[2*k+1])

        vn0 = v_of(Jn, v_free)
        vt0_free_orig = c.get('vt_free', None)    # BEFORE flip
        s = c.get('t_sign', +1)
        vt0 = s * vt0_free_orig if vt0_free_orig is not None else v_of(Jt, v_free)

        vn1 = v_of(Jn, v_post)
        vt1 = v_of(Jt, v_post)

        # 1) Dissipation: λ_t * (s * v_t^free) <= 0
        diss = lt * vt0
        diss_ok = (diss <= tol)

        # 2) Friction strip: 0 <= λ_t <= μ λ_n
        strip_ok = (lt >= -tol) and (lt <= mu*ln + 1e-7)

        # 3) Normal CCP (velocity-level)
        phi_pos = max(c['phi'], 0.0)
        g_n = -beta * (phi_pos) / h - restitution * min(vn0, 0.0)
        s_n = vn1 + g_n            
        prod = ln * s_n
        comp_ok = (s_n >= -1e-8) and (ln >= -1e-12) and (abs(prod) <= 1e-6)

        if print_each:
            print(f"[c{k}] vt_free(orig)={vt0_free_orig:+.3e} s={s:+d} -> vt_free(flipped)={vt0:+.3e} "
                  f"| ln={ln:+.3e} lt={lt:+.3e} | vn0={vn0:+.3e} vn1={vn1:+.3e}")
            print(f"     Dissip: lt*vt_free'={diss:+.3e}  -> {'OK' if diss_ok else 'BAD'}")
            print(f"     Strip : 0<=lt<=μln ? 0<={lt:.3e}<={mu*ln:.3e} -> {'OK' if strip_ok else 'BAD'}")
            print(f"[c{k}] CCP   : s_n={s_n:+.3e}, ln*s_n={prod:+.3e} -> {'OK' if comp_ok else 'BAD'}")
        ok = diss_ok and strip_ok and comp_ok
        ok_all = ok_all and ok

    return ok_all
