import numpy as np

def closest_point_on_segment(p, a, b):
    ab = b - a
    t = float(np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-12))
    t = min(1.0, max(0.0, t))
    return a + t * ab, t

def circle_vs_segment_contact(circle_center, r, segment, sep_tol=0.0):
    
    n = segment.n
    t = segment.t
    # t = np.array([-n[1], n[0]])

    p_closest, _ = closest_point_on_segment(circle_center, segment.a, segment.b)
    signed = float(np.dot(circle_center - p_closest, n))   
    phi = signed - r                                      

    if phi > sep_tol:
        return None

    
    p = circle_center - n * r

    return dict(p=p, n=n, t=t, phi=phi)





def circle_circle_contact(x1, r1, x2, r2, tol=1e-6, sep_tol=0.0):
    """
    Return None if clearly separated, else a contact dict.
    phi (signed gap) = dist - (r1 + r2); phi >= 0 => separated.
    """
    d = x2 - x1
    dist = np.linalg.norm(d)

    if dist < tol:
        # Centers coincide; choose a stable arbitrary normal.
        n = np.array([1.0, 0.0])
        phi = dist - (r1 + r2)      # keep the sign convention consistent
    else:
        n = d / dist
        phi = dist - (r1 + r2)

    # Use a small separation tolerance to avoid chattering (optional)
    if phi > sep_tol:
        return None  # not in contact region

    # Build an orthonormal basis (n, t)
    t = np.array([n[1], -n[0]])   #right-handed t->n order 

    # Surface points
    p1 = x1 + r1 * n
    p2 = x2 - r2 * n
    p  = 0.5 * (p1 + p2)

    # flip normal and tangent vectors
    # the contact force on body 1 is -n, -t
    n = -n
    t = -t

    return dict(p1=p1, p2=p2, p=p, n=n, t=t, phi=phi)

