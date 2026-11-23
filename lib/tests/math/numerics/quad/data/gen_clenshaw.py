"""Reference nodes and weights of the Clenshaw-Curtis rules of N = 32, 64, 128, 256 on [-1, 1].

The weights are found from the definition of an interpolatory rule, not from the cosine series
the library sums: the rule on the N + 1 points x_k = cos(k pi / N) that integrates T_0 ... T_N
exactly, sum_k w_k T_j(x_k) = int_{-1}^{1} T_j = 2 / (1 - j^2) for even j and 0 for odd j, solved
in mpmath at 40 digits. The nodes are cos(k pi / N) at 40 digits.

Writes clenshaw_ref.txt next to this script: `N k node weight`, k = 0 at x = 1.

Run: python3 gen_clenshaw.py   (mpmath; about a minute)
"""
import os
import mpmath as mp

mp.mp.dps = 40
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clenshaw_ref.txt")


def rule(n):
    x = [mp.cos(mp.pi * k / n) for k in range(n + 1)]
    a = mp.matrix(n + 1, n + 1)
    b = mp.matrix(n + 1, 1)
    for j in range(n + 1):
        # T_j(cos t) = cos(j t)
        for k in range(n + 1):
            a[j, k] = mp.cos(j * mp.pi * k / n)
        b[j] = mp.mpf(2) / (1 - j * j) if j % 2 == 0 else mp.mpf(0)
    w = mp.lu_solve(a, b)
    # The rule is exact beyond its interpolation degree for the odd T_{N+1} by symmetry.
    assert abs(sum(w) - 2) < mp.mpf(10) ** -30
    return x, w


def main():
    lines = ["# Clenshaw-Curtis rules on [-1, 1] from gen_clenshaw.py: N k node weight"]
    for n in (32, 64, 128, 256):
        x, w = rule(n)
        for k in range(n + 1):
            node = "0" if 2 * k == n else mp.nstr(x[k], 35)
            lines.append(f"{n} {k} {node} {mp.nstr(w[k], 35)}")
        print(n, "done", flush=True)
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
