#!/usr/bin/env python3
"""Derives the constants of the three-stage Radau IIA method that `RadauIia` uses, exactly with
sympy and printed to 36 digits with mpmath:

    python3 gen_radau.py

* A, b = A's last row, c: the collocation method on the zeros of the Radau polynomial
  x^2 (x - 1) differentiated (c = (4 -+ sqrt6)/10, 1), a[i][j] = integral of the j-th Lagrange
  basis polynomial from 0 to c[i].
* The eigenvalues of A^-1: one real, gamma, and a complex pair alpha +- i beta, and a real
  transformation T with T^-1 A^-1 T = [[gamma, 0, 0], [0, alpha, -beta], [0, beta, alpha]]: the
  columns of T are the real eigenvector and the real and imaginary parts of the eigenvector of
  alpha - i beta, each scaled so that its last component is 1 (real part) and 0 (imaginary part).
  The scaling is a choice; any T that block-diagonalises A^-1 serves.
* The embedded formula of order 3 of Hairer and Wanner (Solving ODEs II, IV.8, (8.19)):
  y^ = y0 + h (gamma0 f(x0, y0) + sum b^_i f(Y_i)), gamma0 = 1/gamma, of order 3. Its difference
  from the solution is gamma0 h f0 + sum e_j Z_j with e = (b^ - b)^T A^-1 (Z_j = Y_j - y0), and
  e = gamma0 * dd with the dd printed.
* The tests check every relation again in floating point (radau.rs).
"""
import sympy as sp
import mpmath

mpmath.mp.dps = 50
s6 = sp.sqrt(6)
c = [(4 - s6) / 10, (4 + s6) / 10, sp.Integer(1)]
x = sp.symbols("x")


def lagrange(j):
    p = sp.Integer(1)
    for m in range(3):
        if m != j:
            p *= (x - c[m]) / (c[j] - c[m])
    return sp.expand(p)


A = sp.Matrix(3, 3, lambda i, j: sp.nsimplify(sp.simplify(sp.integrate(lagrange(j), (x, 0, c[i])))))
A = A.applyfunc(sp.radsimp)
b = A.row(2)
Ainv = A.inv().applyfunc(sp.radsimp)

# The eigenvalues of A^-1, from its characteristic polynomial (a cubic with one real root).
lam = sp.symbols("lam")
charpoly = sp.Poly((Ainv - lam * sp.eye(3)).det(), lam)
cube3 = sp.root(3, 3)
gamma = 3 + cube3**2 - cube3
alpha = 3 + (cube3 - cube3**2) / 2
beta = (sp.root(3, 6) ** 5 + sp.root(3, 6) ** 7) / 2
for root in (gamma, alpha + sp.I * beta, alpha - sp.I * beta):
    v = sp.N(charpoly.eval(root), 60)
    assert abs(complex(v)) < 1e-50, v


def eigenvector(mu):
    """The eigenvector of A^-1 for mu with last component 1, solving the first two rows of
    (A^-1 - mu I) v = 0 with v[2] = 1."""
    M = (Ainv - mu * sp.eye(3))
    v0, v1 = sp.symbols("v0 v1")
    sol = sp.solve([M[0, 0] * v0 + M[0, 1] * v1 + M[0, 2], M[1, 0] * v0 + M[1, 1] * v1 + M[1, 2]],
                   [v0, v1], dict=True)[0]
    return [sol[v0], sol[v1], sp.Integer(1)]


def num(e):
    return mpmath.mpf(str(sp.N(e, 60)))


def numc(e):
    z = complex(sp.N(e, 60))
    re, im = sp.N(sp.re(e), 60), sp.N(sp.im(e), 60)
    return mpmath.mpc(str(re), str(im))


vr = eigenvector(gamma)
vc = eigenvector(alpha - sp.I * beta)
Tm = mpmath.matrix(3, 3)
for i in range(3):
    Tm[i, 0] = num(vr[i])
    z = numc(vc[i])
    Tm[i, 1] = z.real
    Tm[i, 2] = z.imag
TIm = Tm ** -1
Am = mpmath.matrix([[num(A[i, j]) for j in range(3)] for i in range(3)])
Lam = TIm * (Am ** -1) * Tm
want = mpmath.matrix([[num(gamma), 0, 0], [0, num(alpha), -num(beta)], [0, num(beta), num(alpha)]])
assert mpmath.norm(Lam - want) < mpmath.mpf(10) ** -40, Lam

# The embedded formula of order 3 with b^0 = gamma0 = 1/gamma: sum over (0, c1, c2, c3) of
# b^_i c_i^(k-1) = 1/k for k = 1, 2, 3; the stages are those of the collocation method, whose
# stage order 3 reduces every condition of order <= 3 to these.
g0 = 1 / gamma
bh = sp.symbols("bh1 bh2 bh3")
eqs = [g0 * (1 if k == 1 else 0) + sum(bh[i] * c[i] ** (k - 1) for i in range(3)) - sp.Rational(1, k)
       for k in (1, 2, 3)]
sol = sp.solve(eqs, bh, dict=True)[0]
bhat = sp.Matrix([[sol[bh[0]], sol[bh[1]], sol[bh[2]]]])
e = ((bhat - b) * Ainv).applyfunc(sp.radsimp)
dd = (e / g0).applyfunc(lambda v: sp.nsimplify(sp.radsimp(sp.simplify(v)), [s6]))
want_dd = [-(13 + 7 * s6) / 3, (-13 + 7 * s6) / 3, sp.Rational(-1, 3)]
for got, w in zip(dd, want_dd):
    assert abs(sp.N(got - w, 60)) < 1e-45, (got, w)


def show(name, v):
    print(f"{name} = {mpmath.nstr(v, 36, min_fixed=-1, max_fixed=1)}")


print("# c")
for i in range(3):
    show(f"c[{i}]", num(c[i]))
print("# A (exact):", A.tolist())
for i in range(3):
    for j in range(3):
        show(f"A[{i}][{j}]", num(A[i, j]))
print("# eigenvalues of A^-1")
show("gamma", num(gamma))
show("alpha", num(alpha))
show("beta", num(beta))
print("# T (columns: real eigenvector, real and imaginary part of that of alpha - i beta)")
for i in range(3):
    for j in range(3):
        show(f"T[{i}][{j}]", Tm[i, j])
print("# T^-1")
for i in range(3):
    for j in range(3):
        show(f"TI[{i}][{j}]", TIm[i, j])
print("# error estimate: dd (exact)", list(dd))
for i in range(3):
    show(f"dd[{i}]", num(dd[i]))
