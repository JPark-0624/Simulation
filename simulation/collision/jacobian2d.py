import numpy as np

def body_point_jacobians(ball_q, contact_point_world, n, t):
    x, y, th = ball_q
    r_w = contact_point_world - np.array([x, y], dtype=float)

    r_cross_n = r_w[0]*n[1] - r_w[1]*n[0]
    r_cross_t = r_w[0]*t[1] - r_w[1]*t[0]

    Jn = np.array([[n[0], n[1], r_cross_n]])
    Jt = np.array([[t[0], t[1], r_cross_t]])
    return Jn, Jt
