import numpy as np
import scipy.sparse as sp
import osqp

# QP: minimize 0.5 x^T H x + f^T x  subject to x >= 0
H = np.array([[2.0, 0.0],
              [0.0, 1.0]])
f = np.array([-2.0, -4.0])

# OSQP form: minimize 0.5 x^T P x + q^T x  s.t. l <= A x <= u
P = sp.csc_matrix(H)
q = f.copy()

# x >= 0  -> A = I, l = 0, u = +inf
n = H.shape[0]
A = sp.eye(n, format="csc")
l = np.zeros(n)
u = np.full(n, np.inf)

prob = osqp.OSQP()
prob.setup(P=P, q=q, A=A, l=l, u=u, verbose=False)
res = prob.solve()

x = res.x
obj = 0.5 * x @ H @ x + f @ x

print("x* =", x)
print("objective =", obj)

# ---- KKT / LCP check for nonnegativity case ----
# Stationarity: Hx + f - λ = 0 with λ >= 0
lam = H @ x + f
print("lambda =", lam)

# Primal/dual feasibility + complementarity
print("x >= 0 ?          ", np.all(x >= -1e-10))
print("lambda >= 0 ?     ", np.all(lam >= -1e-10))
print("x^T lambda (≈0) = ", float(x @ lam))

# LCP view: w = Hx + f, z = x with w>=0, z>=0, z^T w = 0
w = lam
z = x
print("LCP check -> min(w)=", w.min(), " min(z)=", z.min(), " z^T w =", float(z @ w))
