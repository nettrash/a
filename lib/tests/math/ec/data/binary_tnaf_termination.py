# Checks that the width-w tau-adic non-adjacent form of src/math/ec/binary/koblitz.rs ends for
# every element of Z[tau], for mu = 1 and -1 and the widths 2 to 8 it can take.
#
#   python3 binary_tnaf_termination.py
#
# The digit set is the one the crate computes: alpha_u = u mods tau^w by Solinas's rounding. Each
# step takes rho to (rho - alpha) / tau, of norm N(rho - alpha) / 2, so sqrt(N) goes to at most
# (sqrt(N(rho)) + c) / sqrt(2), c the largest sqrt(N(alpha_u)), and falls strictly while sqrt(N(rho))
# > c / (sqrt(2) - 1). The norm r0^2 + mu r0 r1 + 2 r1^2 is at least (3 - sqrt(2)) / 2 (r0^2 + r1^2),
# which bounds that disc by a box; the script runs the algorithm from every element of a larger box
# and checks that each run ends, so that every run does. It also checks that each alpha_u is
# congruent to u modulo tau^w (beta + gamma t_w = u modulo 2^w), and prints the digit sets and the
# largest |beta| and |gamma|, which the crate's comment states.
import math


def lucas_u(mu, k):
    u0, u1 = 0, 1
    for _ in range(k):
        u0, u1 = u1, mu * u1 - 2 * u0
    return u0


def norm(a, b, mu):
    return a * a + mu * a * b + 2 * b * b


def round_z(n0, n1, d, mu):
    f0, f1 = (2 * n0 + d) // (2 * d), (2 * n1 + d) // (2 * d)
    e0, e1 = n0 - f0 * d, n1 - f1 * d
    h0 = h1 = 0
    e = 2 * e0 + mu * e1
    if e >= d:
        if e0 - 3 * mu * e1 < -d:
            h1 = mu
        else:
            h0 = 1
    elif e0 + 4 * mu * e1 >= 2 * d:
        h1 = mu
    if e < -d:
        if e0 - 3 * mu * e1 >= d:
            h1 = -mu
        else:
            h0 = -1
    elif e0 + 4 * mu * e1 < -2 * d:
        h1 = -mu
    return f0 + h0, f1 + h1


def reduce(k, d0, d1, mu):
    s0 = d0 + mu * d1
    q0, q1 = round_z(k * s0, -k * d1, norm(d0, d1, mu), mu)
    return k - q0 * d0 + 2 * q1 * d1, -(q0 * d1 + q1 * d0 + mu * q1 * d1)


def digit_set(mu, w):
    uw, uw1 = lucas_u(mu, w), lucas_u(mu, w - 1)
    tw = (2 * uw1 * pow(uw, -1, 1 << w)) % (1 << w)
    assert (tw * tw - mu * tw + 2) % (1 << w) == 0
    alphas = [reduce(u, -2 * uw1, uw, mu) for u in range(1, 1 << (w - 1), 2)]
    for u, (b, g) in zip(range(1, 1 << (w - 1), 2), alphas):
        assert (b + g * tw - u) % (1 << w) == 0
    return tw, alphas


def tnaf_length(r0, r1, mu, w, tw, alphas, limit):
    n = 0
    while r0 or r1:
        if r0 & 1:
            v = (r0 + r1 * tw) % (1 << w)
            u = v - (1 << w) if v >= 1 << (w - 1) else v
            b, g = alphas[(abs(u) - 1) // 2]
            if u > 0:
                r0, r1 = r0 - b, r1 - g
            else:
                r0, r1 = r0 + b, r1 + g
        r0, r1 = r1 + mu * (r0 // 2), -(r0 // 2)
        n += 1
        if n > limit:
            return None
    return n


largest = 0
for mu in (1, -1):
    for w in range(2, 9):
        tw, alphas = digit_set(mu, w)
        largest = max([largest] + [max(abs(b), abs(g)) for b, g in alphas])
        c = max(math.sqrt(norm(b, g, mu)) for b, g in alphas)
        radius = c / (math.sqrt(2) - 1) / math.sqrt((3 - math.sqrt(2)) / 2)
        box = int(radius) + 8
        for r0 in range(-box, box + 1):
            for r1 in range(-box, box + 1):
                assert tnaf_length(r0, r1, mu, w, tw, alphas, 10000) is not None, (mu, w, r0, r1)
        print('mu %2d  w %d  t_w %3d  box %3d  alphas %s' % (mu, w, tw, box, alphas))
print('largest |beta|, |gamma|:', largest)
print('every run ends')
