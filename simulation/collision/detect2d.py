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


def rot2(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s],
                     [s,  c]])

def obb_vertices(center: np.ndarray, half_extents: np.ndarray, theta: float) -> np.ndarray:
    """
    Returns the 4 world-space vertices of an oriented box (counter-clockwise).
    half_extents: (ax, ay) = (width/2, height/2)
    """
    a = np.asarray(half_extents, dtype=float)
    R = rot2(theta)
    # body-frame corners
    corners = np.array([[ a[0],  a[1]],
                        [-a[0],  a[1]],
                        [-a[0], -a[1]],
                        [ a[0], -a[1]]], dtype=float)
    return center + (R @ corners.T).T  # (4,2)

def obb_signed_distance_point(p_world: np.ndarray,
                              center: np.ndarray,
                              half_extents: np.ndarray,
                              theta: float):
    """
    Signed distance from a point to an OBB, outward normal at the closest point,
    and the closest point on the box boundary (all in world frame).
      phi >= 0 outside; phi == 0 on boundary; phi < 0 inside.
    """
    a = np.asarray(half_extents, dtype=float)
    R = rot2(theta)
    # world -> body
    p_body = R.T @ (p_world - center)
    q = np.abs(p_body) - a

    outside = np.maximum(q, 0.0)
    phi = np.linalg.norm(outside)
    inside_margin = np.max(q)  # <= 0 if inside

    # closest point on box in body frame
    cp_body = np.clip(p_body, -a, a)

    eps = 1e-12
    if inside_margin <= 0.0:
        # point is inside: negative distance (amount to exit along nearest face)
        phi = inside_margin
        # normal toward nearest face
        k = int(np.argmax(np.abs(p_body) - a))
        n_body = np.array([0.0, 0.0])
        n_body[k] = np.sign(p_body[k])
    else:
        # outside: gradient from outside vector
        if phi < eps:
            # exactly on edge/corner: fall back to direction from cp to point
            d_body = p_body - cp_body
            n_body = d_body / (np.linalg.norm(d_body) + eps)
        else:
            n_body = np.zeros(2)
            mask = q > 0
            if mask.any():
                n_body[mask] = np.sign(p_body[mask])
                n_body = n_body / (np.linalg.norm(n_body) + eps)
            else:
                # numerical guard
                d_body = p_body - cp_body
                n_body = d_body / (np.linalg.norm(d_body) + eps)

    # back to world
    n_world = R @ n_body
    cp_world = center + R @ cp_body
    # tangent (right-handed)
    t_world = np.array([ n_world[1], -n_world[0] ], dtype=float)

    n = -n_world  # flip normal to point from box -> point
    t = -t_world  # flip tangent accordingly

    return float(phi), n, t, cp_world

def point_vs_obb_contact(p_world: np.ndarray,
                         center: np.ndarray,
                         half_extents: np.ndarray,
                         theta: float,
                         sep_tol: float = 0.0):
    """
    Contact between a geometric point and an oriented box (OBB).
    Normal points from BOX -> POINT. Returns None if separated by > sep_tol.
    Dict keys: p (contact point on box boundary), n, t, phi
    """
    phi, n, t, pC = obb_signed_distance_point(p_world, center, half_extents, theta)
    if phi > sep_tol:
        return None
    # contact point is the closest point on the box surface
    return dict(p=pC, n=n, t=t, phi=phi)

def box_vs_segment_contact_obb(center: np.ndarray,
                               half_extents: np.ndarray,
                               theta: float,
                               segment,
                               sep_tol: float = 0.0):
    """
    Bilateral environment contact: OBB vs wall segment with known outward (environment->object) normal 'segment.n'.
    We test the 4 box vertices against the segment and take the *most penetrating* one.
    Returns None if clearly separated (max phi > sep_tol).
    Dict keys: p (point on the box surface closest to segment), n (= segment.n), t (= segment.t), phi
    """
    n = segment.n
    t = segment.t
    verts = obb_vertices(center, half_extents, theta)

    best_phi = None
    best_p  = None
    best_cp = None

    for vtx in verts:
        # closest point on segment to this vertex
        cp, _ = closest_point_on_segment(vtx, segment.a, segment.b)
        # signed distance using the segment normal (positive if outside)
        signed = float(np.dot(vtx - cp, n))
        phi = signed  # for a "box vs segment", treat vertex as the feature
        if (best_phi is None) or (phi < best_phi):
            best_phi = phi
            best_p   = vtx
            best_cp  = cp

    if best_phi is None or best_phi > sep_tol:
        return None

    # For reporting, use the point on the box boundary that is closest to the segment:
    # here 'best_p' already lies on the box; the actual contact point is where the normal ray meets the segment.
    # We'll return the *box-side* point (best_p). This matches your circle_vs_segment convention (p on the body).
    return dict(p=best_p, n=n, t=t, phi=float(best_phi))


