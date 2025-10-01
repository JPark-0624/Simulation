import numpy as np

class Segment:
    """Line segment with inward unit normal n and tangent t (right-handed)."""
    def __init__(self, a, b, n_inward):
        self.a = np.array(a, float)
        self.b = np.array(b, float)
        n = np.array(n_inward, float)
        self.n = n / (np.linalg.norm(n) + 1e-12)
        # self.t = np.array([-self.n[1], self.n[0]], float)
        self.t = np.array([self.n[1], -self.n[0]], float)  # right-handed t->n order 

