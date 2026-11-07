[A](../../../README.md) › [Math](../../../README.md#math)

# Factorisation

[Numbers](numbers.md) factor into primes with `a.factor()`, which gives a `Factorization`: the primes in increasing order with their exponents. The methods it chooses from are public in `a::math::numbers::factor`: `trial` (trial division), `rho` (Pollard's rho with Brent's cycle finding), `pm1` and `pp1` (Pollard's `p − 1` and Williams's `p + 1`), `ecm` (Lenstra's elliptic curve method) and `siqs` (the [quadratic sieve](siqs.md)). `a.perfect_power()` tells whether `a` is `b^k` for some `k >= 2`, with the largest `k`.

- `factor` gives the factorisation of the absolute value of `a`, in the byte order of `a`: none for 1 and −1; it panics for 0. The primes are probable primes by [`is_probable_prime`](prime.md), exact below `2^64`. It always ends: the elliptic curves go on with growing bounds until they find a factor, in a time that grows with the second largest prime factor.
- `Factorization::new(factors, order)` takes primes and exponents in any order, sorts them, merges equal primes and drops exponents of 0; it panics for a "prime" below 2 and for exponents that add up past `u32::MAX`, and does not test the primes. `factors()`, `into_factors()` and `byte_order()` give them back, `value()` the number they make, by a balanced tree of products.
- `trial(n, bound)` gives the primes up to `bound` that divide `|n|`, with their exponents, in increasing order, and the cofactor left, `|n|` divided by them, in the byte order of `n`; for 0, no primes and the cofactor 0. Once the cofactor is below the square of the next prime it is 1 or prime, and is given among the primes when it is at most `bound`: `trial(n, u64::MAX)` factors any `n` below `2^64` completely.
- `rho(n, iterations)`, `pm1(n, b1, b2)`, `pp1(n, b1, b2)` and `ecm(n, b1, b2, curves)` give a factor of `|n|` other than 1 and `|n|`, in the byte order of `n`, or `None`: always `None` for 0 to 3 and for primes (which [`is_probable_prime`](prime.md) tells at once), and `Some(2)` for an even number above 2. The factor need not be prime. Each is deterministic: the same arguments give the same factor.
- `perfect_power` gives `Some((b, k))` when `|a| = |b|^k` with `k >= 2` the largest such exponent, `b` with the sign of `a`, so that `k` is odd for a negative `a` (`−64 = (−4)^3`); `None` for 0, 1 and −1 and every other number, as PARI's `ispower` does. `is_perfect_power` tells whether it is `Some`.
- Each magnitude is read in its own byte order, and zero bytes above its most significant non-zero byte are ignored; an empty vector is zero, and a magnitude of zero is zero whatever `negative` says. The numbers are left as they were.

## Trial division

The odd primes below 8192 are those of the primality test's tables: a long number is reduced modulo products of consecutive primes that fit in a limb, by the division's two-limbs-by-one steps with a reciprocal, four products a pass in independent chains, and each prime of a product is tested on the remainder by one multiplication, as `x·p^(−1) mod 2^64 <= (2^64 − 1)/p` exactly when `p` divides `x`. A prime that divides is taken out by Hensel's exact division, a limb at a time from the bottom, whose carry out of the top tells whether it divided again. Once the number fits in a limb, each prime is tried on it directly, the quotient by `p` of a multiple of `p` being `x·p^(−1) mod 2^64`, until the prime's square passes the number. The primes from 8192 to `2^20` come from a table of the same kind made on the first use of such a bound, 81,000 primes with their inverses and their products; primes beyond, from the crate's segmented sieve ([Prime sieves and counting](sieve.md)), in products that fit in a limb.

## Pollard's rho

Pollard's method (1975) iterates `y ↦ y² + c` modulo `n`: modulo each prime `p` of `n` the sequence repeats after about `√p` steps, and the greatest common divisor of `n` and the difference of two terms that agree modulo `p` is a multiple of `p`. Brent's cycle finding (1980) compares `x = y_(2^i − 1)` with the next `2^i` terms, and the differences are multiplied together, 128 of them between two greatest common divisors; when one is `n`, the batch is gone over again from its start, a greatest common divisor a step, and when that still gives `n` the next `c` is taken. The terms are residues in Montgomery's form, `y·R ↦ (y·R)²/R + c`, which is the sequence of `y ↦ y² + c/R`, as good a constant as `c`; a step is a square, an addition, a subtraction and a product.

## p − 1 and p + 1

The first stage of both, and of the elliptic curve method, takes the exponent `E`, the product of the largest powers of the primes up to `B1` that are at most `B1`, about `1.44·B1` bits, in chunks of at most `2^22` bits, each computed by a balanced tree of products of limbs packed with as many prime powers as fit; up to `B1 = 10^8` the chunks are kept for every curve or starting value of a call.

- Pollard's `p − 1` (1974): when `p − 1` divides `E`, `3^E = 1` modulo `p`. `3^E` is computed from the top bit of `E` with a square for each bit and, for each one, a multiplication by 3, which is two additions; a later chunk takes a power by a sliding window.
- Williams's `p + 1` (1982): the Lucas sequence `V_m(P_0)`, `V_0 = 2`, `V_1 = P_0`, is `α^m + α^(−m)` for a root `α` of `X² − P_0·X + 1`, of an order dividing `p + 1` when `P_0² − 4` is not a square modulo `p` (and `p − 1` when it is): `V_E − 2` is then 0 modulo `p` when the order divides `E`. `V_E` is computed by the ladder, a square and a product a bit, through the chunks as `V_(ab)(P) = V_b(V_a(P))`; from 9 limbs, by Montgomery's PRAC chain for each prime instead (1983), about `1.52·log2 p` products and squares against `2·log2 p`, the best of ten multipliers near the golden ratio chosen for each prime. The starting values `2/7`, `6/5` and `6`, whose `P_0² − 4` are `−3`, `−1` and `2` times squares, are tried in turn: for a prime with `p + 1` smooth, one of them takes `p + 1` with a chance of 7 in 8.

When the greatest common divisor of a chunk is `n`, every prime was caught at once: the chunk is gone over again a prime at a time from its start, with a greatest common divisor each.

## The elliptic curve method

Lenstra's method (1987) on Montgomery's curves `B·y² = x³ + A·x² + x` (1987), with Suyama's parametrisation (1985): for `σ`, `u = σ² − 5`, `v = 4σ`, the point `(u³ : v³)` and `(A + 2)/4 = (v − u)³(3u + v)/(16·u³·v)`, whose group orders modulo every prime are multiples of 12. These are GMP-ECM's `-param 0` curves; the curves of a call are those of `σ = 6, 7, 8, …` in turn, so that a factorisation is reproducible, and GMP-ECM's `-sigma 0:σ` gives the same curve. Both stages compute on `x` and `z` alone.

- Stage 1: `Q = E·P` by Montgomery's ladder, `(R0, R1) ↦ (2·R0, R0 + R1)` or `(R0 + R1, 2·R1)` with `R1 − R0 = P` all along, the bit choosing which point is doubled by swapping the two without a branch. With `P` affine, `z = 1`, a step is 4 squares and 6 products (the formulas of RFC 7748, with `(A + 2)/4`). The point is taken to affine after each chunk of `E` by one inversion, which a factor of `n` common to `z` makes impossible: the greatest common divisor is the factor, and when it is `n` the chunk is gone over again a prime at a time. Up to eight limbs, two curves go through stage 1 together, step by step: in registers a product waits for the ones before it, and the other curve's fill the wait; the result is that of the two one after the other.
- Stage 2: the continuation below, on `x`: the baby steps `j·Q` by the chain `(j + 2)·Q = j·Q + 2·Q`, the giant steps `k·D·Q` by `(k + 1)·D·Q = k·D·Q + D·Q`, both taken to affine together by Montgomery's trick, a block of 128 giant steps at a time, so that each pair `(k, j)` costs one product, of `x_(kD) − x_j`.

With the same curves and bounds, stage 1 here finds a prime exactly when the point's order modulo it divides `E`; GMP-ECM's finds those (giving `n` itself when it catches every prime at once, where this one goes over the chunk again) and a few more, as a PRAC chain whose intermediate multiple of the point is the identity modulo `p` makes its result the identity too. On 2900 semiprimes of 64 to 1100 bits with a prime of 30 or 34 bits, one curve of `σ = 6` at `B1` of 2000 or 5000, the two found the same factor in 506 cases, neither in 2369, and differed in 25; the point orders that PARI's `ellorder` gave for those checked bore this out.

## The second stage

The primes `q` from `B1` to `B2`, by Montgomery's baby-step giant-step continuation with prime pairing (1987): each `q` is `k·D ± j` for its nearest multiple `k·D` of a fixed `D` and an odd `j < D/2` prime to `D`. For a function `x_m` of the multiple `m` of the point (or the power) that tells `m·P` from `−m·P` no more than `P` from `−P` does, `x_(kD) − x_j` is 0 modulo a prime `p` when `(kD − j)·P` or `(kD + j)·P` is the identity modulo `p`: one product catches both, and serves two primes when both are prime. The products are accumulated in four chains, so that their latencies overlap, and a greatest common divisor at the end of each block of giant steps tells whether a prime was caught (a block that gives `n` is gone over again a product at a time).

- `D` is chosen among 6, 30, 60, 210, 420, 2310, 4620, 30030 and 60060 by the cost of the baby steps (about `D/4` additions and `φ(D)/2` kept), of the giant steps, and of the products, fewer as more of them serve two primes; when `D/2` passes `B1`, the primes up to `D/2` are taken in stage 1.
- Which `j` each giant step needs is a bitmap per step, made once from the primes of `(B1, B2]`, which the crate's segmented sieve gives ([Prime sieves and counting](sieve.md)), and kept for every curve or starting value when it takes at most 32 MB.
- For `p − 1` and `p + 1`, `x_m = V_m`: `V_(m+n) = V_m·V_n − V_(m−n)` gives the baby steps and the giant steps with a product each; for `p − 1`, `V_1 = x + 1/x` for `x = 3^E`.

## Perfect powers

`x = c^K` with `K` the largest exponent is a `p`-th power for a prime `p` exactly when `p` divides `K`, with the root `c^(K/p)`, so the primes are tried in increasing order, each as long as it gives a root: their product is `K`.

- The exponents of 2 and of the odd primes to 47 in `x` are multiples of `K`. One pass of the number gives its remainders by the product of the squares of the odd primes to 29 and by the product of the primes from 31 to 47, in two independent chains: a prime to 29 that divides `x` only once makes `K = 1` at once, which settles almost every number. When the exponents found (counted up to 64) have a common divisor `g`, only the primes dividing `g` are tried.
- Each prime `p` is tried up to the bits of `x` over `log2` of its smallest possible prime factor (53 when none is below): first by residues, a `p`-th power being one modulo every prime: modulo 64 and the primes to 47 for squares, the primes `q ≡ 1 (mod p)` to 47 for `p = 3, 5, 7, 11`, from the remainders already known, and up to four primes `q ≡ 1 (mod p)` below `2^16` for larger `p`; then, when the root has at most 40 bits, by the floating-point root of the top 64 bits, exact to 1, whose `p`-th power modulo `2^64` is compared with the low limb; otherwise by [`nth_root`](root.md) or `isqrt_rem` and an exact check.
- Below `2^128` the same is done in registers, without reading the number into limbs, the roots by the root's own routine for `u128`.

## The factorisation

`factor` chooses the method by size, as PARI, FLINT and yafu do:

- Trial division by the primes below 8192. Each cofactor then takes the exact test below `2^64` or Baillie–PSW above, the test for a perfect power, and the search for a factor, and both parts go round again.
- One limb: Hart's one line method (2012) up to 40 bits, `s = ⌈√(480·n·i)⌉` for `i = 1, 2, …` until `s² mod 480·n` is a square `t²`, with the factor `gcd(s − t, n)`; then the elliptic curve method from `B1 ≈ 64·1.08^(bits − 44)`, after four curves on each of the three smaller bounds, 8 curves a level, `B1` growing by 1.25 a level, `B2 = 50·B1`.
- Two limbs: a run of 4096 steps of rho, then curves from `B1 ≈ 450·1.0583^(bits − 64)`, the bound for a factor of half the bits, after four curves on each of the three smaller bounds, `max(8, B1/250)` curves a level, `B1` growing by 1.25 a level, `B2 = 100·B1`.
- More limbs: 4096 steps of rho, `p − 1` from 45 digits (`B1 = 10^5`, `B2 = 10^7`), then GMP-ECM's levels for factors of 10, 12, 15, 20, … digits (`B1` from 150 to `2.9·10^9`), `B2 = 100·B1`.
- The quadratic sieve takes the cofactors of 28 to 110 digits once the curves have made factors of 30 % of their digits unlikely (yafu's 4/13).

The stages of the bounds that recur, the exponents of the first and the plans of the second, are kept for the life of the program once made (up to `B1` of about `5·10^5` for two limbs and `10^6` beyond, a few megabytes at most), so that the factorisation of a small number costs its curves alone.

## The arithmetic

All the methods share one arithmetic, residues modulo the odd `n` in Montgomery's form, fully or lazily reduced: in registers up to eight limbs, the powers' arithmetic (one limb, two limbs, and three to eight by the coarsely integrated operand scanning), with residues kept below `2n` and no subtraction after each product when `n` is below `R/4` (the top limb below `2^62`); two limbs then take REDC a limb at a time, 8 products against 12; nine limbs and more in memory, the multiplication's kernels and REDC by the kernel's passes, or by products from 100 limbs. The operations write their result to a residue of their own, or in place, so that each method is written once for all the sizes. Inversions are done by the extended Euclidean algorithm in registers for one limb, the binary one for two, and [`mod_inverse`](gcd.md) beyond.

## Measured

Measured on an Apple M4 Max, one thread, with nothing else running, each contender interleaved with its reference in the same process and the medians of 3 to 7 rounds given; GMP-ECM 7.0.7 (configured "Tuned for default", generic `MODMULN` arithmetic), FLINT 3.6.0, GMP 6.3.0, PARI/GP 2.19.0 and msieve 1.53 from Homebrew.

One elliptic curve against GMP-ECM's `-param 0` curve of the same `σ` (6): stage 1 at `B1`, and stage 2 to `B2 = 100·B1` for both, and to GMP-ECM's default `B2` (170 to 1000 times `B1`: 1.87·10^6, 1.27·10^7, 1.29·10^8 and 1.05·10^9), for semiprimes of the bits given that the curves do not factor. Ours per curve is the time of four curves less that of two, so that the exponent of stage 1 and the plan of stage 2, made once a call, are left out:

| Bits | B1 | stage 1 | GMP-ECM | stage 2, `100·B1` | GMP-ECM, `100·B1` | GMP-ECM, default `B2` |
| --- | --- | --- | --- | --- | --- | --- |
| 124 | 11,000 | 0.40 ms | 3.19 ms | 0.26 ms | 1.97 ms | 2.27 ms |
| 128 | 11,000 | 0.52 ms | 3.43 ms | 0.36 ms | 2.46 ms | 2.87 ms |
| 190 | 11,000 | 1.04 ms | 3.62 ms | 0.66 ms | 3.00 ms | 3.57 ms |
| 256 | 11,000 | 2.16 ms | 5.07 ms | 1.29 ms | 3.52 ms | 4.29 ms |
| 380 | 11,000 | 4.40 ms | 6.29 ms | 2.26 ms | 4.29 ms | 5.33 ms |
| 512 | 11,000 | 8.57 ms | 9.71 ms | 4.34 ms | 6.08 ms | 7.63 ms |
| 1024 | 11,000 | 23.3 ms | 27.1 ms | 11.8 ms | 13.8 ms | 17.3 ms |
| 124 | 50,000 | 1.80 ms | 14.2 ms | 1.01 ms | 5.11 ms | 8.79 ms |
| 128 | 50,000 | 2.33 ms | 16.4 ms | 1.43 ms | 6.35 ms | 10.5 ms |
| 190 | 50,000 | 4.74 ms | 16.3 ms | 2.80 ms | 7.49 ms | 13.0 ms |
| 256 | 50,000 | 9.86 ms | 23.1 ms | 4.69 ms | 8.47 ms | 15.0 ms |
| 380 | 50,000 | 20.2 ms | 28.3 ms | 9.50 ms | 10.6 ms | 20.1 ms |
| 512 | 50,000 | 38.2 ms | 41.7 ms | 16.9 ms | 14.7 ms | 27.2 ms |
| 1024 | 50,000 | 106 ms | 121 ms | 44.5 ms | 31.4 ms | 61.8 ms |
| 124 | 250,000 | 9.13 ms | 70.0 ms | 4.22 ms | 15.6 ms | 33.9 ms |
| 256 | 250,000 | 50.6 ms | 116 ms | 22.5 ms | 25.3 ms | 53.9 ms |
| 512 | 250,000 | 192 ms | 213 ms | 81.1 ms | 41.8 ms | 98.4 ms |
| 1024 | 250,000 | 532 ms | 602 ms | 206 ms | 94.9 ms | 224 ms |
| 256 | 10^6 | 202 ms | 472 ms | 84.3 ms | 53.3 ms | 211 ms |
| 512 | 10^6 | 777 ms | 843 ms | 296 ms | 98.0 ms | 396 ms |

- Stage 1: from 8 times faster than GMP-ECM at 124 bits to 1.09 to 1.16 times at 512 and 1024 bits: the residues in registers up to eight limbs, lazily reduced below 126 bits and whenever the top limb is below `2^62` (124, 190, 380 bits against 128, 256, 512), two curves interleaved, and the ladder on the whole exponent, 4 squares and 6 products a bit with an affine difference, against GMP-ECM's PRAC chain for each prime with `mpz` calls.
- Stage 2 to the same `B2`: faster where `B2` is small for the modulus, 2.7 to 7.6 times at `B1 = 11,000` up to 256 bits and 1.8 to 5.1 times at 50,000, and slower where it is large: about the same at 256 bits and `B1 = 250,000`, 1.6 times slower at `10^6`, and from 512 bits 1.15 to 3 times slower from `B1 = 50,000` on. GMP-ECM's continuation by fast polynomial arithmetic (Montgomery and Kruppa's) costs about `√B2` products times logarithms, this one about a product for each prime; GMP-ECM's default `B2`, 1.7 to 10 times further, takes 1.15 to 4 times its own time to `100·B1`.
- A whole curve here (both stages to `100·B1`) costs less than GMP-ECM's up to 380 bits, and to 1024 bits at `B1 = 11,000`; about the same at 512 and 1024 bits and `B1 = 50,000`; beyond, GMP-ECM's curve to its default `B2` finds more for its time.

Against FLINT's `fmpz_factor_ecm`, which takes its own random curves, a curve with both stages to `100·B1` (here four curves less two, there ten less five):

| Bits | B1 | a curve | FLINT |
| --- | --- | --- | --- |
| 128 | 11,000 | 0.87 ms | 7.16 ms |
| 128 | 50,000 | 3.77 ms | 30.1 ms |
| 256 | 11,000 | 3.52 ms | 14.9 ms |
| 256 | 50,000 | 15.0 ms | 69.4 ms |
| 512 | 11,000 | 12.9 ms | 27.4 ms |
| 512 | 50,000 | 55.3 ms | 119 ms |
| 1024 | 11,000 | 34.9 ms | 79.3 ms |
| 1024 | 50,000 | 152 ms | 346 ms |

From 2.1 to 8.2 times faster than FLINT.

`p − 1` and `p + 1` against GMP-ECM's `-pm1` (`x_0 = 3`) and `-pp1` (its own starting value), stage 1 at `B1` and stage 2 to `100·B1` and to GMP-ECM's default (here, the plan of stage 2 is made by each call and counted, from the primes of the crate's sieve); `p + 1` here tries three starting values, GMP-ECM one:

| Method | Bits | B1 | stage 1 | GMP-ECM | stage 2, `100·B1` | GMP-ECM, `100·B1` | GMP-ECM, default |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `p − 1` | 128 | 10^5 | 1.59 ms | 2.86 ms | 4.40 ms | 3.39 ms | 4.92 ms |
| `p − 1` | 128 | 10^6 | 16.6 ms | 27.2 ms | 38.3 ms | 8.09 ms | 30.7 ms |
| `p − 1` | 256 | 10^6 | 27.2 ms | 41.9 ms | 93.2 ms | 12.9 ms | 51.9 ms |
| `p − 1` | 512 | 10^6 | 75.9 ms | 88.8 ms | 313 ms | 22.9 ms | 92.6 ms |
| `p − 1` | 1024 | 10^6 | 211 ms | 250 ms | 762 ms | 57.1 ms | 226 ms |
| `p + 1`, 3 values | 128 | 10^6 | 54.1 ms | 58.4 ms | 90.2 ms | 11.1 ms | 45.7 ms |
| `p + 1`, 3 values | 256 | 10^6 | 133 ms | 84.3 ms | 248 ms | 16.2 ms | 68.9 ms |
| `p + 1`, 3 values | 512 | 10^6 | 484 ms | 159 ms | 910 ms | 32.7 ms | 137 ms |
| `p + 1`, 3 values | 1024 | 10^6 | 1.11 s | 447 ms | 2.24 s | 83.7 ms | 345 ms |

- Stage 1 of `p − 1` is 1.17 to 1.8 times faster than GMP-ECM's, a square a bit and two additions for a one; that of `p + 1`, a starting value at a time, 3.2 times faster at 128 bits, 1.9 at 256, 0.98 at 512 (the ladder) and 1.2 at 1024 bits (PRAC).
- Stage 2 is where GMP-ECM's fast polynomial continuation makes it 1.3 to 14 times faster at the same `B2` for `p − 1`, and 2.7 to 9.3 times for `p + 1` a starting value at a time: this continuation is the baby-step giant-step one with prime pairing, a product for each pair of a giant step and a baby step, and the plan is counted in each call.

Pollard's rho against FLINT's `fmpz_factor_pollard_brent` (one try, run again until it finds), for semiprimes of 128 bits with a prime of the bits given, the mean of 20 (10 for the last two rows):

| Factor bits | rho | FLINT |
| --- | --- | --- |
| 20 | 19.5 µs | 40.7 µs |
| 24 | 68.8 µs | 182 µs |
| 28 | 301 µs | 762 µs |
| 32 | 979 µs | 2.73 ms |
| 36 | 4.97 ms | 13.3 ms |
| 40 | 17.6 ms | 39.0 ms |

From 2.1 to 2.8 times faster: Montgomery's form in registers, a square, an addition, a subtraction and a product a step, 128 products between two greatest common divisors.

Factorisations below `2^64` against FLINT's `n_factor` and PARI's `factor` (the mean over sets of 1000 to 4000 numbers: random 64-bit numbers, and products of two random primes of the bits given):

| Numbers | `factor` | FLINT `n_factor` | PARI |
| --- | --- | --- | --- |
| random of 64 bits | 5.5 µs | 10.4 µs | 21.0 µs |
| primes of 16 and 16 bits | 1.0 µs | 1.7 µs | 1.7 µs |
| 20 and 20 | 4.3 µs | 6.7 µs | 5.5 µs |
| 24 and 24 | 7.8 µs | 18.5 µs | 30.9 µs |
| 28 and 28 | 17.4 µs | 65.1 µs | 47.8 µs |
| 32 and 32 | 28.3 µs | 251 µs | 90.1 µs |
| 20 and 44 | 7.6 µs | 9.1 µs | 90.0 µs |

From 1.2 to 9 times faster than FLINT and 1.3 to 12 times faster than PARI: Hart's method below 40 bits, then curves in one limb, two at a time, with bounds for the size, the stages of their bounds made once for the program.

Whole factorisations against PARI's `factor`, FLINT's `fmpz_factor` and msieve (a process each, its own start of 3 ms taken out; it is meant for large numbers), the mean and the median over 24 numbers of each kind: balanced semiprimes (two primes of half the digits) and random numbers:

| Numbers | Digits | `factor` mean, median | PARI | FLINT | msieve |
| --- | --- | --- | --- | --- | --- |
| semiprimes | 20 | 0.12, 0.11 ms | 0.16, 0.16 ms | 1.03, 0.72 ms | 1.8 ms |
| semiprimes | 24 | 0.36, 0.29 ms | 0.39, 0.39 ms | 2.38, 2.40 ms | 2.0 ms |
| semiprimes | 28 | 1.40, 1.41 ms | 0.85, 0.86 ms | 2.27, 2.29 ms | 12 ms |
| semiprimes | 32 | 2.02, 2.27 ms | 2.02, 2.05 ms | 4.08, 3.93 ms | 19 ms |
| semiprimes | 36 | 4.48, 4.49 ms | 5.08, 5.19 ms | 8.85, 8.63 ms | 25 ms |
| semiprimes | 40 | 9.32, 9.49 ms | 12.3, 12.5 ms | 20.3, 19.1 ms | 39 ms |
| semiprimes | 44 | 21.4, 23.1 ms | 39.5, 39.5 ms | 43.0, 40.5 ms | 56 ms |
| semiprimes | 50 | 78, 80 ms | 130, 132 ms | 239, 239 ms | 161 ms |
| semiprimes | 60 | 0.54, 0.55 s | 1.45, 1.46 s | 2.09, 2.13 s | 1.44 s |
| random | 20 | 12.6, 6.7 µs | 27.6, 8.4 µs | 112, 4.4 µs | 0.1 ms |
| random | 30 | 74, 27 µs | 248, 195 µs | 1.50, 1.83 ms | 1.9 ms |
| random | 40 | 0.90, 0.16 ms | 2.51, 1.91 ms | 4.73, 4.02 ms | 10.7 ms |
| random | 50 | 12.9, 1.8 ms | 35.3, 16.5 ms | 57.3, 30.2 ms | |
| random | 60 | 42.1, 11.6 ms | 105, 46 ms | 177, 71 ms | |

- Up to 24 digits, where the curves are the method, `factor` is 1.1 to 1.4 times faster than PARI and 7 to 9 times faster than FLINT on average; at 28 digits, where the quadratic sieve takes over after short curves, PARI's MPQS is 1.6 times faster; from 32 digits, with the quadratic sieve, as fast as PARI at 32 and 1.13 to 2.7 times faster from 36 to 60 digits, 1.6 to 3.9 times faster than FLINT from 28, and 2.1 to 9 times faster than msieve.
- Random numbers, whose cofactors are mostly small or prime, 2.2 to 3.4 times faster than PARI on average and 1.25 to 12 times at the median, and faster than FLINT and msieve at every size.

Before the quadratic sieve was in, with other work running on the machine, the same semiprimes by PARI's `factorint(n, 1)`, without its quadratic sieve, so that its curves find the factors, against `factor`'s curves alone (the median):

| Digits | 24 | 28 | 32 | 36 | 40 | 44 | 50 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `factor` | 0.33 ms | 1.61 ms | 5.00 ms | 8.06 ms | 78 ms | 172 ms | 2.27 s |
| PARI's curves | 12.0 ms | 46.0 ms | 130 ms | 689 ms | 1.83 s | 5.44 s | 28.4 s |

From 12 to 85 times faster with the curves alone.

Perfect powers against GMP's `mpz_perfect_power_p` (which only tells whether), FLINT's `fmpz_is_perfect_power` and PARI's `ispower` (its interpreter's cost of about 50 ns a call included), on random odd numbers, primes, cubes of primes, and squares of primes plus 2:

| Input | `perfect_power` | GMP | FLINT | PARI |
| --- | --- | --- | --- | --- |
| random odd, 64 bits | 7 ns | 40 ns | 8 ns | 291 ns |
| prime, 64 bits | 49 ns | 320 ns | 38 ns | 102 ns |
| cube of 21 bits | 51 ns | 241 ns | 51 ns | 132 ns |
| square of 32 bits plus 2 | 7 ns | 38 ns | 8 ns | 56 ns |
| random odd, 256 bits | 96 ns | 145 ns | 88 ns | 7.4 µs |
| prime, 256 bits | 270 ns | 500 ns | 1.8 µs | 7.6 µs |
| cube of 85 bits | 364 ns | 300 ns | 747 ns | 4.1 µs |
| square of 128 bits plus 2 | 31 ns | 57 ns | 42 ns | 192 ns |
| random odd, 1024 bits | 157 ns | 160 ns | 252 ns | 327 ns |
| prime, 1024 bits | 1.1 µs | 2.0 µs | 12.1 µs | 22.7 µs |
| cube of 341 bits | 956 ns | 863 ns | 2.7 µs | 12.1 µs |
| square of 512 bits plus 2 | 74 ns | 189 ns | 79 ns | 1.0 µs |
| random odd, 4096 bits | 302 ns | 743 ns | 267 ns | 4.7 µs |
| prime, 4096 bits | 1.6 µs | 2.0 µs | 11.8 µs | 22.4 µs |
| cube of 1365 bits | 3.6 µs | 3.6 µs | 7.6 µs | 35.9 µs |
| square of 2048 bits plus 2 | 284 ns | 612 ns | 225 ns | 504 ns |

Faster than GMP but for the cubes of 85 bits and more (up to 1.21 times slower), which take the root and its power where GMP takes a 2-adic root; as fast as FLINT, up to 1.3 times slower on some (primes of 64 bits, near-square numbers of 2048 bits, random ones of 4096) and 6.7 to 11 times faster on primes of 256 bits and more, which take every candidate exponent.

Trial division against FLINT's `fmpz_factor_trial_range` with the same primes, of numbers without small factors, so that every prime is tried:

| Bits | to 1000 | FLINT | to 8191 | FLINT | to 10^5 | FLINT | to 10^6 | FLINT |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 64 | 128 ns | 323 ns | 571 ns | 1.8 µs | 5.0 µs | 16.4 µs | 39.1 µs | 134 µs |
| 256 | 186 ns | 613 ns | 1.1 µs | 3.5 µs | 11.8 µs | 33.3 µs | 106 µs | 251 µs |
| 1024 | 446 ns | 2.1 µs | 2.9 µs | 12.6 µs | 35.6 µs | 120 µs | 347 µs | 956 µs |
| 4096 | 1.4 µs | 5.8 µs | 10.7 µs | 36.0 µs | 132 µs | 331 µs | 1.34 ms | 2.76 ms |

From 2.1 to 4.7 times faster: the remainders by products of primes four at a time, and a multiplication for each prime, against a remainder for each prime.

## Examples

Rust usage: factorisations, the methods and perfect powers:

```rust
use a::math::numbers::{ByteOrder, Factorization, Number, factor};

fn main() {
  // F7 = 2^128 + 1 = 59649589127497217 · 5704689200685129054721 (Morrison and Brillhart, 1970).
  let f7 = &(&Number::new_from_u8(1, ByteOrder::BigEndian) << 128usize) + 1u8;
  let f = f7.factor();
  let primes: Vec<String> = f.factors().iter().map(|(p, _)| p.to_string()).collect();
  assert_eq!(primes, ["59649589127497217", "5704689200685129054721"]);
  assert_eq!(f.value(), f7);

  // Exponents, signs and byte orders: -(2^10 · 3^5 · 101).
  let n = Number::new_from_i64(-(1024 * 243 * 101), ByteOrder::LittleEndian);
  let f = n.factor();
  let w: Vec<(u64, u32)> = f.factors().iter().map(|(p, e)| (u64::try_from(p).unwrap(), *e)).collect();
  assert_eq!(w, [(2, 10), (3, 5), (101, 1)]);
  assert!(f.byte_order() == ByteOrder::LittleEndian);
  let g = Factorization::new(vec![(Number::new_from_u8(101, ByteOrder::BigEndian), 1)], ByteOrder::BigEndian);
  assert_eq!(g.value(), 101u8);

  // Trial division: the primes up to the bound, and the cofactor.
  let n = Number::new_from_u64(1001 * 1_000_003, ByteOrder::BigEndian);
  assert_eq!(factor::trial(&n, 100), (vec![(7, 1), (11, 1), (13, 1)], Number::new_from_u32(1_000_003, ByteOrder::BigEndian)));

  // The methods: a factor, or nothing for a prime.
  let n = Number::new_from_u64(1_000_003 * 1_000_033, ByteOrder::BigEndian);
  let f = factor::rho(&n, 10_000).unwrap();
  assert!(f == 1_000_003u32 || f == 1_000_033u32);
  // 1000033 - 1 = 2^5 · 3 · 11 · 947: p - 1 with B1 = 1000 finds it.
  assert_eq!(factor::pm1(&n, 1000, 0).unwrap(), 1_000_033u32);
  // 1000099 + 1 = 2^2 · 5^2 · 73 · 137: p + 1 with B1 = 200.
  let m = Number::new_from_u64(1_000_099 * 1_000_117, ByteOrder::BigEndian);
  assert_eq!(factor::pp1(&m, 200, 0).unwrap(), 1_000_099u32);
  let e = factor::ecm(&n, 300, 30_000, 20).unwrap();
  assert!(e == 1_000_003u32 || e == 1_000_033u32);
  assert!(factor::ecm(&Number::new_from_u64(1_000_000_007, ByteOrder::BigEndian), 300, 30_000, 20).is_none());

  // Perfect powers: the largest exponent, odd for a negative number.
  let n = Number::new_from_u64(1 << 60, ByteOrder::BigEndian);
  assert_eq!(n.perfect_power(), Some((Number::new_from_u8(2, ByteOrder::BigEndian), 60)));
  // -2^60 = (-16)^15: 15 is the odd part of 60.
  let n = Number::new_from_i64(-(1 << 60), ByteOrder::BigEndian);
  assert_eq!(n.perfect_power(), Some((Number::new_from_i64(-16, ByteOrder::BigEndian), 15)));
  assert!(!Number::new_from_u32(1_000_001, ByteOrder::BigEndian).is_perfect_power());
}
```

## Efficiency

- Time complexity: trial division `O(n·g)` limb operations for `n` limbs and `g` products of primes, and a multiplication for each prime; rho about `√p` steps for the smallest prime `p`, each two products of residues; `p − 1` about `1.44·B1` squares, `p + 1` `2.9·B1` products (`2.2·B1` by PRAC), and an elliptic curve `14.4·B1` products in stage 1; the second stages about `0.8·(π(B2) − π(B1))` products and the sieving of `(B1, B2]`, once a call; a product of residues `O(n²)` below 100 limbs and about 2.7 products of `n` limbs beyond; perfect powers a pass or two of the number for most, and up to `log2 x / 5.7` roots for the few that pass the residues
- Space complexity: the residues of the method, in registers up to eight limbs; the exponent of stage 1 (`1.44·B1` bits, in chunks of `2^22` bits, kept up to `B1 = 10^8`), the baby steps (`φ(D)/2` residues) and a block of 128 giant steps, and the plan of stage 2 (a bit per baby step a giant step, kept up to 32 MB); the factorisation's kept stages, a few megabytes at most; the table of the primes to `2^20`, 2.6 MB, made on the first trial division beyond 8192
- Soundness: the bytes beyond a vector's length, which may be uninitialised, are never read; every residue lives in a vector of its own or in registers, and the kernels' working memory is checked against its plan in builds with debug assertions; the numbers are left as they were
