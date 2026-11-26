"""The first outputs of xoshiro256** (Blackman and Vigna) seeded by SplitMix64, for
lib/tests/math/numerics/quad/monte_carlo.rs: an implementation of the published reference algorithms
(https://prng.di.unimi.it/xoshiro256starstar.c, splitmix64.c), independent of the library's.

Prints, for the seeds 1 and 42, the first 8 outputs shifted right by 12 bits (the 52 bits the
library makes a uniform number in (0, 1) of: (k + 1/2) / 2^52).

Run: python3 gen_xoshiro.py
"""
M = (1 << 64) - 1


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & M
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M
    return state, z ^ (z >> 31)


def rotl(x, k):
    return ((x << k) | (x >> (64 - k))) & M


def xoshiro(seed, count):
    s, sm = [], seed
    for _ in range(4):
        sm, v = splitmix64(sm)
        s.append(v)
    out = []
    for _ in range(count):
        result = (rotl((s[1] * 5) & M, 7) * 9) & M
        t = (s[1] << 17) & M
        s[2] ^= s[0]
        s[3] ^= s[1]
        s[1] ^= s[2]
        s[0] ^= s[3]
        s[2] ^= t
        s[3] = rotl(s[3], 45)
        out.append(result >> 12)
    return out


if __name__ == "__main__":
    for seed in (1, 42):
        print(seed, xoshiro(seed, 8))
