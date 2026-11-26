#!/usr/bin/env python3
"""Bases that fail a condition of LLL, or of deep insertions, by a hair: for tests/math/lattice/l2.rs
and deep.rs (the constants there are this script's output).

With δ = 1 − 2⁻⁵³, the largest float below 1, and A = 2^k, δ·A² = A² − A²/2⁵³ is an integer, and
a vector of squared norm N = δ·A² − 2 is shorter than √δ·A by a relative 2⁻²ᵏ: floating point of
fewer than about 2k bits rounds N to δ·A² and sees no failure. So:

* lovasz(k): b₀ = (A, 0, 0), b₁ = (x, y, z) with x small, x² + y² + z² = N: size-reduced, and
  Lovász's condition |b*₁|² + μ²·|b*₀|² = N ≥ δ·|b*₀|² fails by 2.
* deep(k): b₀ = (A, 0, 0, 0, 0), b₁ = (0, u, v, 0, 0) with δ·A² ≤ u² + v² ≤ A² − 3 − x₀², and
  b₂ = (x₀, 0, 0, y, z) with x₀ small and |b₂|² = N: LLL-reduced (Lovász's condition holds at
  both places: |π₁(b₂)|² = N − x₀² ≥ δ·(u² + v²)), but the deep exchange condition
  |π₀(b₂)|² = |b₂|² ≥ δ·|b*₀|² fails by 2, so b₂ belongs in front.

Each sum y² + z² (and u² + v²) is a prime p ≡ 1 (mod 4), split by Cornacchia's algorithm (Hermite–Serret: Euclid's
algorithm on p and a square root of −1 modulo p, stopped below √p).

Usage: python3 gen_l2_boundary.py
"""
import random
from math import isqrt


def is_prime(n, rounds=40):
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    rng = random.Random(n)
    for _ in range(rounds):
        a = rng.randrange(2, n - 1)
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def two_squares(p):
    """(y, z) with y² + z² = p, for a prime p ≡ 1 (mod 4)."""
    c = 2
    while True:
        r = pow(c, (p - 1) // 4, p)
        if r * r % p == p - 1:
            break
        c += 1
    a, b, limit = p, r, isqrt(p)
    while b > limit:
        a, b = b, a % b
    y = b
    z = isqrt(p - y * y)
    assert y * y + z * z == p
    return y, z


def target(k):
    a2 = 1 << (2 * k)
    return a2 - (a2 >> 53) - 2  # δ·A² − 2


def lovasz(k):
    n = target(k)
    for x in range(1, 10**6):
        p = n - x * x
        if p % 4 == 1 and is_prime(p):
            y, z = two_squares(p)
            return [[1 << k, 0, 0], [x, y, z]]


def deep(k):
    n = target(k)
    for x0 in range(1, 10**6):
        p = n - x0 * x0
        if p % 4 == 1 and is_prime(p):
            y, z = two_squares(p)
            break
    m = (1 << (2 * k)) - 3 - x0 * x0
    while not (m % 4 == 1 and is_prime(m)):
        m -= 1
    u, v = two_squares(m)
    return [[1 << k, 0, 0, 0, 0], [0, u, v, 0, 0], [x0, 0, 0, y, z]]


def check(b, k, deep_case):
    from fractions import Fraction as F
    delta = F(2**53 - 1, 2**53)
    a2 = 1 << (2 * k)
    assert sum(x * x for x in b[-1]) == delta * a2 - 2
    if deep_case:
        m = sum(x * x for x in b[1])
        assert m >= delta * a2
        assert sum(x * x for x in b[2]) - b[2][0] ** 2 >= delta * m


def rust(name, b):
    rows = ",\n".join("        &[" + ", ".join(f'"{x}"' for x in r) + "]" for r in b)
    return f"const {name}: &[&[&str]] = &[\n{rows},\n];"


for k in (35, 200, 500):
    b = lovasz(k)
    check(b, k, False)
    print(rust(f"LOVASZ_{k}", b))
for k in (35, 200, 500):
    b = deep(k)
    check(b, k, True)
    print(rust(f"DEEP_{k}", b))
