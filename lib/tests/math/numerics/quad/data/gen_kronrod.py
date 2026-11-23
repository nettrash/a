"""The Gauss-Kronrod pairs G7-K15 and G10-K21, computed independently at 60 digits.

The Kronrod extension of the n-point Gauss-Legendre rule adds the n + 1 zeros of the Stieltjes
polynomial E_{n+1}, the polynomial of degree n + 1 orthogonal to P_n(x) x^k on [-1, 1] for every
k <= n. Here E_{n+1} = sum_k c_k P_k(x), c_{n+1} = 1, is solved for from those n + 1 conditions in
the Legendre basis, its zeros found by Newton's method from the zeros of P_{n+1}, and the weights of
all 2n + 1 nodes solved for from the moment equations sum_i w_i P_k(x_i) = 2 [k = 0], k <= 2n; the
script then checks that the rule integrates every monomial up to degree 3n + 1 exactly. The Gauss
weights are 2 / ((1 - x^2) P_n'(x)^2) at the zeros of P_n.

Writes kronrod_ref.txt next to this script: `G7K15|G10K21 u kronrod_weight gauss_weight|-`, one line
a node of non-negative abscissa, largest first, 40 significant digits.

Run: python3 gen_kronrod.py   (mpmath; a few seconds)
"""
import os
import mpmath as mp

mp.mp.dps = 60
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kronrod_ref.txt")


def legendre_values(n, x):
    """P_0(x) .. P_n(x)."""
    p = [mp.mpf(1), x]
    for k in range(1, n):
        p.append(((2 * k + 1) * x * p[k] - k * p[k - 1]) / (k + 1))
    return p[: n + 1]


def gauss(n):
    nodes = [mp.cos(mp.pi * (4 * k - 1) / (4 * n + 2)) for k in range(1, n + 1)]
    out = []
    for x in nodes:
        for _ in range(100):
            p = legendre_values(n, x)
            dp = n * (x * p[n] - p[n - 1]) / (x * x - 1)
            step = p[n] / dp
            x -= step
            if abs(step) < mp.mpf(10) ** -55:
                break
        p = legendre_values(n, x)
        dp = n * (x * p[n] - p[n - 1]) / (x * x - 1)
        out.append((x, 2 / ((1 - x * x) * dp * dp)))
    return out


def inner(f, g):
    """The integral over [-1, 1] of the product of two polynomials given by coefficient lists."""
    prod = [mp.mpf(0)] * (len(f) + len(g) - 1)
    for i, a in enumerate(f):
        for j, b in enumerate(g):
            prod[i + j] += a * b
    return sum(c * (2 / mp.mpf(k + 1) if k % 2 == 0 else 0) for k, c in enumerate(prod))


def legendre_coefficients(n):
    """The monomial coefficients of P_0 .. P_n."""
    p = [[mp.mpf(1)], [mp.mpf(0), mp.mpf(1)]]
    for k in range(1, n):
        a = [mp.mpf(0)] + [(2 * k + 1) * c for c in p[k]]
        b = [k * c for c in p[k - 1]] + [mp.mpf(0)] * 2
        p.append([(x - y) / (k + 1) for x, y in zip(a, b)])
    return p[: n + 1]


def kronrod(n):
    pc = legendre_coefficients(n + 1)
    # E_{n+1} = P_{n+1} + sum_{k<=n} c_k P_k with <P_n x^j, E> = 0 for j = 0..n.
    pn = pc[n]
    rows, rhs = [], []
    for j in range(n + 1):
        w = [mp.mpf(0)] * j + pn  # P_n(x) x^j
        rows.append([inner(w, pc[k]) for k in range(n + 1)])
        rhs.append(-inner(w, pc[n + 1]))
    c = mp.lu_solve(mp.matrix(rows), mp.matrix(rhs))
    e = [mp.mpf(0)] * (n + 2)
    for k in range(n + 1):
        for i, v in enumerate(pc[k]):
            e[i] += c[k] * v
    for i, v in enumerate(pc[n + 1]):
        e[i] += v
    poly = lambda x: sum(v * x**i for i, v in enumerate(e))
    dpoly = lambda x: sum(i * v * x ** (i - 1) for i, v in enumerate(e) if i)
    # Its zeros interlace with those of P_n: start from the zeros of P_{n+1}.
    zeros = []
    for x, _ in gauss(n + 1):
        for _ in range(200):
            step = poly(x) / dpoly(x)
            x -= step
            if abs(step) < mp.mpf(10) ** -55:
                break
        zeros.append(x)
    g = gauss(n)
    nodes = sorted([x for x, _ in g] + zeros, reverse=True)
    m = len(nodes)
    a = mp.matrix(m, m)
    b = mp.matrix(m, 1)
    for k in range(m):
        for i, x in enumerate(nodes):
            a[k, i] = legendre_values(m, x)[k]
        b[k] = 2 if k == 0 else 0
    w = mp.lu_solve(a, b)
    # Exactness up to degree 3n + 1, and not beyond by accident of the check.
    for k in range(3 * n + 2):
        got = sum(w[i] * nodes[i] ** k for i in range(m))
        want = mp.mpf(2) / (k + 1) if k % 2 == 0 else mp.mpf(0)
        assert abs(got - want) < mp.mpf(10) ** -45, (n, k, got, want)
    gw = {mp.nstr(x, 50): wt for x, wt in g}
    rows = []
    for i, x in enumerate(nodes):
        if x < -mp.mpf(10) ** -40:
            continue
        x = abs(x) if abs(x) < mp.mpf(10) ** -40 else x
        key = mp.nstr(x, 50)
        gauss_w = next((v for kx, v in gw.items() if abs(mp.mpf(kx) - x) < mp.mpf(10) ** -40), None)
        rows.append((x, w[i], gauss_w))
    return rows


def main():
    lines = ["# Gauss-Kronrod pairs from gen_kronrod.py: name, node, Kronrod weight, Gauss weight"]
    for name, n in (("G7K15", 7), ("G10K21", 10)):
        for x, wk, wg in kronrod(n):
            x = mp.mpf(0) if abs(x) < mp.mpf(10) ** -40 else x
            lines.append(" ".join([name, mp.nstr(x, 40), mp.nstr(wk, 40),
                                   mp.nstr(wg, 40) if wg is not None else "-"]))
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
