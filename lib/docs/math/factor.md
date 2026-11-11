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

The odd primes below 8192 are those of the primality test's tables: a long number is reduced modulo products of consecutive primes that fit in a limb, by the division's two-limbs-by-one steps with a reciprocal, eight products a pass in independent chains (for two limbs one step each, the high limb reduced first by a subtraction, as the products are normalized), and the primes of a product are tested on its remainder together, by one multiplication each, as `x·p^(−1) mod 2^64 <= (2^64 − 1)/p` exactly when `p` divides `x`; only a product with a prime that divides is gone over one prime at a time. A prime that divides is taken out by Hensel's exact division, a limb at a time from the bottom, whose carry out of the top tells whether it divided again. Once the number fits in a limb, the primes are tried on it directly, eight at a time without a branch while they are within the bound and their squares below the number, the quotient by `p` of a multiple of `p` being `x·p^(−1) mod 2^64`, until the prime's square passes the number. The primes from 8192 to `2^20` come from a table of the same kind made on the first use of such a bound, 81,000 primes with their inverses and their products; primes beyond, from the crate's segmented sieve ([Prime sieves and counting](sieve.md)), in products that fit in a limb.

## Pollard's rho

Pollard's method (1975) iterates `y ↦ y² + c` modulo `n`: modulo each prime `p` of `n` the sequence repeats after about `√p` steps, and the greatest common divisor of `n` and the difference of two terms that agree modulo `p` is a multiple of `p`. Brent's cycle finding (1980) compares `x = y_(2^i − 1)` with the next `2^i` terms, and the differences are multiplied together, 128 of them between two greatest common divisors; when one is `n`, the batch is gone over again from its start, a greatest common divisor a step, and when that still gives `n` the next `c` is taken. The terms are residues in Montgomery's form, `y·R ↦ (y·R)²/R + c`, which is the sequence of `y ↦ y² + c/R`, as good a constant as `c`; a step is a square, an addition, a subtraction and a product.

## p − 1 and p + 1

The first stage of both, and of the elliptic curve method, takes the exponent `E`, the product of the largest powers of the primes up to `B1` that are at most `B1`, about `1.44·B1` bits, in chunks of at most `2^22` bits, each computed by a balanced tree of products of limbs packed with as many prime powers as fit; up to `B1 = 10^8` the chunks are kept for every curve or starting value of a call.

- Pollard's `p − 1` (1974): when `p − 1` divides `E`, `3^E = 1` modulo `p`. `3^E` is computed from the top bit of `E` with a square for each bit and, for each one, a multiplication by 3, which is two additions; a later chunk takes a power by a sliding window.
- Williams's `p + 1` (1982): the Lucas sequence `V_m(P_0)`, `V_0 = 2`, `V_1 = P_0`, is `α^m + α^(−m)` for a root `α` of `X² − P_0·X + 1`, of an order dividing `p + 1` when `P_0² − 4` is not a square modulo `p` (and `p − 1` when it is): `V_E − 2` is then 0 modulo `p` when the order divides `E`. Below 6 limbs `V_E` is computed by the ladder, a square and a product a bit, through the chunks as `V_(ab)(P) = V_b(V_a(P))`, the two independent of each other; from 6 limbs, where a product is bound by the additions it takes rather than by its latency, so that the ladder's two cost twice one, by Montgomery's PRAC chain for each prime instead (1983), about `1.52·log2 p` products and squares, with the best of ten multipliers near the golden ratio for each prime. Choosing them takes about 0.4 µs a prime, more than the chains save in registers, so they are chosen once for the bound and kept, a byte a prime, for the other starting values and the next calls with the bound (up to `B1 = 10^7`, 665 KB; beyond, the ladder below 9 limbs). The starting values `2/7`, `6/5` and `6`, whose `P_0² − 4` are `−3`, `−1` and `2` times squares, are tried in turn: for a prime with `p + 1` smooth, one of them takes `p + 1` with a chance of 7 in 8.

When the greatest common divisor of a chunk is `n`, every prime was caught at once: the chunk is gone over again a prime at a time from its start, with a greatest common divisor each.

## The elliptic curve method

Lenstra's method (1987) on Montgomery's curves `B·y² = x³ + A·x² + x` (1987), with the parametrisation of GMP-ECM's `-param 2` (Bouvier and Zimmermann): for `σ >= 2`, the point `σ·(−3, 3)` on `y² = x³ + 36` gives `x₃ = (3x + y + 6)/(2(y − 3))` and the curve of `(A + 2)/4 = (1 − x₃)³(3x₃ + 1)/(16·x₃³)`, which is Suyama's (1985) of `u/v = x₃`, with the starting point `x = 2`, on the curve with a point of order 6: the group orders modulo every prime are multiples of 12 and as likely to be smooth as those of Suyama's own starting point (GMP-ECM's `-param 0`; one curve of each found a factor of 32 bits in 31.8 % and 31.2 % of 3000 semiprimes at `B1 = 300`, `B2 = 15,000`). The curves of a call are those of `σ = 2, 3, 4, …` in turn, so that a factorisation is reproducible, and GMP-ECM's `-sigma 2:σ` gives the same curve. When `σ·(−3, 3)` is at infinity modulo every prime of `n` at once (the point has the order `σ` modulo each), the multiples of the point, and so the curves, would only repeat modulo each prime from there, and the formulas would keep the point at infinity: the curves of that `σ` and after are GMP-ECM's `-param 1` ones instead, `(A + 2)/4 = σ²/2^64` from `x = 2` (`-sigma 1:σ`), so that `factor` goes on with new curves. On 14,000 semiprimes of 64 to 1100 bits, stage 1 of the first one to four curves found the same factor as GMP-ECM's, curve by curve, in every case but 53: 44 where GMP-ECM caught both primes at once (and gave `n`), and 9 where it found a prime whose curve's point had twice the order of a divisor of `E` (its conversion to Weierstrass's form divides by `y`, which is then 0; here the second stage finds those). Both stages compute on `x` and `z` alone.

- Stage 1: `Q = E·P` by Montgomery's ladder, `(R0, R1) ↦ (2·R0, R0 + R1)` or `(R0 + R1, 2·R1)` with `R1 − R0 = P` all along, the bit choosing which point is doubled by swapping the two without a branch. With `P = (2 : 1)` the addition's product by `P`'s `x` is a doubling: a step is 4 squares and 5 products (the formulas of RFC 7748, with `(A + 2)/4`). The point is taken to affine after each chunk of `E` (later chunks from the affine `x`, with its product) by one inversion, which a factor of `n` common to `z` makes impossible: the greatest common divisor is the factor, and when it is `n` the chunk is gone over again a prime at a time. In registers the curves go through stage 1 in blocks, operation by operation, so that one curve's products fill the latency of another's: four at a time for one and two limbs, two for three limbs, one from four (where one curve keeps the multipliers busy) and in memory; the first block of each level of the factorisation takes two, so that a curve likely to finish alone does not wait for three others. A block's curves are made together, a point addition on `y² = x³ + 36` each and one inversion for the block, which also tells when a multiple of the point is at infinity modulo a prime of `n`. Whatever the blocks, the factor given is the first curve's to find one.
- Stage 2: one of the continuations below, on `x`: the baby steps `j·Q` by the chain `(j + 2)·Q = j·Q + 2·Q`, the giant steps `k·D·Q` by `(k + 1)·D·Q = k·D·Q + D·Q`, taken to affine by Montgomery's trick a block at a time; then either a product for each pair `(k, j)`, of `x_(kD) − x_j`, or the polynomials of the baby steps' and the giant steps' `x`.

With the same curves and bounds, stage 1 here finds a prime exactly when the point's order modulo it divides `E`.

## The second stage

The primes `q` from `B1` to `B2`, by the one of three continuations whose estimated time is least, from the measured costs of a product of residues and of polynomials for the modulus's length: Montgomery's baby-step giant-step continuation with prime pairing for the shorter ranges, and for the longer ones the continuations by fast polynomial arithmetic, Montgomery and Kruppa's for `p − 1` and `p + 1`, and the FFT continuation for the curves. The ones by polynomials take every number prime to their `P` or `D` from `B1` to a little past `B2`, primes and others, so that they find every prime the baby-step giant-step one does, and sometimes more; a call is no less deterministic, as the choice is made from the bounds and the modulus's length alone.

- Baby-step giant-step (1987): each `q` is `k·D ± j` for its nearest multiple `k·D` of a fixed `D` and an odd `j < D/2` prime to `D`. For a function `x_m` of the multiple `m` of the point (or the power) that tells `m·P` from `−m·P` no more than `P` from `−P` does, `x_(kD) − x_j` is 0 modulo a prime `p` when `(kD − j)·P` or `(kD + j)·P` is the identity modulo `p`: one product catches both, and serves two primes when both are prime. The products are accumulated in four chains, so that their latencies overlap, and a greatest common divisor at the end of each block of giant steps tells whether a prime was caught. `D` is chosen among 6, 30, 60, 210, 420, 2310, 4620, 30030 and 60060 by the cost of the baby steps, the giant steps and the products; when `D/2` passes `B1`, the primes up to `D/2` are taken in stage 1. Which `j` each giant step needs is a bitmap per step, made from the primes of `(B1, B2]`, which the crate's segmented sieve gives ([Prime sieves and counting](sieve.md)), on the first use of the plan, and kept for every curve or starting value when it takes at most 32 MB. For `p − 1` and `p + 1`, `x_m = V_m`: `V_(m+n) = V_m·V_n − V_(m−n)` gives the baby steps and the giant steps with a product each; for `p − 1`, `V_1 = x + 1/x` for `x = 3^E`.
- Montgomery and Kruppa ("An FFT extension to the P−1 factoring algorithm", 2010), for `p − 1` and `p + 1`, as GMP-ECM's `pm1fs2.c`: for an odd `P` made of the primes 3 to 29 and symmetric sets `S_1` and `S_2` whose sums `k_1 + k_2` give each class prime to `P` once, the reciprocal polynomial `f(x) = ∏_(k ∈ S_1) (x − b^(2k))` is evaluated, for each `k_2` of `S_2`, at the `nr` points `x_0·r^(2i)`, `x_0 = b^(2k_2 + (2m_1 + 1)·P)`, `r = b^P`: `f` there has the factor `b^((2m + 1)·P + 2(k_1 + k_2)) − 1` for each `k_1`, so the exponents tried are every odd number prime to `P` in a range that holds `(B1, B2]`. By `2ij = (i + j)² − i² − j²`, the values at a geometric progression are, up to units, the middle coefficients of one product of polynomials, of `h_j = f_j·r^(−j²)` (`s_1 + 1` coefficients) by `g_t = x_0^t·r^(t²)` (`s_1 + nr`). The sets are sums of arithmetic progressions of prime lengths, the classes prime to `q` being `(P/q)·{−(q − 2), …, −1, 1, …, q − 2}`, whose progression of length `q − 1` splits by the factors of `q − 1`; `f` is made from them, `f_(A+T)(x) = ∏_(t ∈ T) f_A(x·b^(−2t))`, the longest progressions first, about one product of polynomials of `s_1` coefficients in all. `P` (from 511 candidates), the split of the sets and `nr` are those of the least estimated time, under 16 MiB for the numbers of a product's factors; for the longest ranges, where no split near the best one fits (from about `B2 = 10^12` at 16 limbs and `10^14` at 4), the largest that fits, up to `B2` from about `3·10^14` (16 limbs) to `3·10^15` (one limb). For `p + 1`, `b` is the root `β = (V + √Δ)/2` of `X² − V·X + 1` in `Z_n[√Δ]`, `Δ = V² − 4`, of norm 1, whose powers have `V_m/2` as their rational coordinate: `f` has rational coefficients, its pairs of copies `f_A(x·c)·f_A(x/c)` being `U² − Δ·W²`; `h` and `g` have two coordinates, made by Montgomery's recurrence for powers of norm 1 with quadratic exponents (GMP-ECM's), five products a term, and of each value only the rational coordinate, `h_x·g_x + Δ·h_y·g_y`, which is 0 modulo `p` with the value.
- The FFT continuation for the curves (Montgomery's thesis, 1992; Zimmermann and Dodson, "20 years of ECM", 2006; GMP-ECM's `ecm2.c`): for `D = 210·t`, `2310·t` or `30030·t` (`t` made of the primes to 11, `D/2` at most `B1`), the baby steps' `x(j·Q)` for the `d = φ(D)/2` odd `j < D/2` prime to `D` are the roots of `F`, the giant steps' `x(k·D·Q)`, `d` of them a block, the roots of `G`; `H = ∏ G mod F`, and the product of the values of `H` at the roots of `F` is that of all the differences `x(j·Q) − x(k·D·Q)`. `F` is the root of a subproduct tree of linear factors, which is kept; the remainders take the inverse of the reversed `F` modulo `X^d`, by Newton's iteration; and the values at the roots of `F`, by the transposed algorithm of Bostan, Lecerf and Schost ("Tellegen's principle into practice", 2003): at the root the reversal of the low `d` coefficients of `rev(F)^(−1)·rev(H)`, then down the tree, at a node of children `M_1` and `M_2` of `d_1` and `d_2` leaves, the coefficients `d_2 … d_1 + d_2` of `M_2·u` to the first and `d_1 … d_1 + d_2` of `M_1·u` to the second, the values at the leaves.
- The products of polynomials modulo `n` are those of numbers (Kronecker's substitution): the coefficients, residues of `k` limbs, are laid `2k + 1` limbs apart, so that each coefficient of the product, a sum of products below `4n²`, fits its place; the crate's multiplication takes them, by its number-theoretic transforms when long; each coefficient wanted is reduced by REDC over `k + 1` limbs. That is the exact product of the coefficients in Montgomery's form for `R' = B^(k + 1)`, the form the curves' polynomials use (their roots taken to it by a product with the residue of `B`), so that monic polynomials stay monic; for `p ± 1` the factor `B^(−1)` it leaves is a unit, as are the others the values carry. Products of at most 6 coefficients in the shorter polynomial are made coefficient by coefficient, the products of limbs summed before one REDC each.
- When the product of a block, an evaluation or a curve's values is 0 modulo `n`, the baby-step giant-step continuation goes over the block again a product at a time; Montgomery and Kruppa's goes over the evaluation's values one at a time, and a value that is 0 over its factors, and when the first factor that is not a unit is 0 (its exponent can be a multiple of the orders of several primes, as `q_1·q_2` up to `B2`), hands the stage to the baby-step giant-step continuation, which goes up the primes; the FFT continuation hands the curve to the baby-step giant-step one, as it does when its baby steps or a block of its giant steps cannot be taken to affine modulo any prime of `n` (its blocks are of `d` giant steps, up to 77,760, against 128, and its greatest common divisor comes at the end, so that the other may still tell the primes apart).
- The parameters of the continuations by polynomials are chosen when first asked for and kept with the plan, by the limbs of the modulus, so that the stages the factorisation keeps for its levels choose them once.

## Perfect powers

`x = c^K` with `K` the largest exponent is a `p`-th power for a prime `p` exactly when `p` divides `K`, with the root `c^(K/p)`, so the primes are tried in increasing order, each as long as it gives a root: their product is `K`.

- The exponents of 2 and of the odd primes to 47 in `x` are multiples of `K`. One pass of the number gives its remainders by the product of the squares of the odd primes to 29 and by the product of the primes from 31 to 47, in two independent chains, and from them the remainders by each prime, by constant divisors: a prime to 29 that divides `x` only once makes `K = 1` at once, which settles almost every number. When the exponents found (counted up to 64) have a common divisor `g`, only the primes dividing `g` are tried.
- Each prime `p` is tried up to the bits of `x` over `log2` of its smallest possible prime factor (53 when none is below): first by residues, a `p`-th power being one modulo every prime: modulo 64 and the primes to 47 for squares, the primes `q ≡ 1 (mod p)` to 47 for `p = 3, 5, 7, 11`, from the remainders already known, and up to four primes `q ≡ 1 (mod p)` below `2^16` for larger `p`; then, when the root has at most 40 bits, by the floating-point root from `log2 x`, computed once for all `p`: as `log2 x` is known to `2^−52·log2 x`, the root of a power is within `2^−6.9` of an integer, so a root further than `2^−6` from one rules the power out without a product, and an integer root is checked modulo `2^64`, then exactly; otherwise by [`nth_root`](root.md) or `isqrt_rem` and an exact check.
- Below `2^64` the same is done on words, the remainders by constant divisors and the roots in floating point (`sqrt`, `cbrt` or a power), checked exactly; below `2^128`, in registers, the roots by the root's own routine for `u128`.

## The factorisation

`factor` chooses the method by size, as PARI, FLINT and yafu do:

- Trial division by the primes below 8192. Each cofactor then takes the exact test below `2^64` or Baillie–PSW above, the test for a perfect power, and the search for a factor, and both parts go round again; a cofactor that comes down to a limb goes on without dividing by the small primes again.
- One limb: Hart's one line method (2012) up to 40 bits, `s = ⌈√(480·n·i)⌉` for `i = 1, 2, …` until `s² mod 480·n` is a square `t²`, with the factor `gcd(s − t, n)`; then the elliptic curve method from `B1 ≈ 64·1.08^(bits − 44)`, after two curves on each of the three smaller bounds, 8 curves a level, `B1` growing by 1.25 a level, `B2 = 35·B1`.
- Two limbs: a run of 2048 steps of rho, then curves from `B1 ≈ 450·1.0583^(bits − 64)`, the bound for a factor of half the bits, after four curves on each of the three smaller bounds, `max(8, B1/250)` curves a level, `B1` growing by 1.25 a level, `B2 = 100·B1`; when the quadratic sieve is to come, from `B1 = 191` instead, four curves on each bound below the one for the size, until the sieve takes over.
- More limbs: 4096 steps of rho, `p − 1` from 45 digits (`B1 = 10^5`, `B2 = 10^7`), then GMP-ECM's levels for factors of 10, 12, 15, 20, … digits (`B1` from 150 to `2.9·10^9`), `B2 = 100·B1` up to 20 digits and 200, 400 and 800 times `B1` at 25, 30 and 35 digits (800 times beyond): the continuation by polynomials makes a long second stage cheap against the first, and each of those levels takes GMP-ECM's curves times the ratio of one curve's chances at `100·B1` and at its `B2` (GMP-ECM's model), 251, 501 and 1119 curves at 25 to 35 digits for 300, 700 and 1800, so that it finds its factors as often in less time.
- The quadratic sieve takes the cofactors of 24 to 110 digits once the curves have made factors of 30 % of their digits unlikely (yafu's 4/13).

The search for a factor of a cofactor runs in one arithmetic modulo it, rho, `p − 1` and all the levels of the curves, which are taken in turn from one sequence. The stages of the bounds that recur, the exponents of the first and the plans of the second, are kept for the life of the program once made (up to `B1` of about `5·10^5` for two limbs and `10^6` beyond, a few megabytes at most: the longer second stages of the levels from 25 digits would keep bitmaps of 0.14, 1.2 and 9.6 MB only if they took the baby-step giant-step continuation, which the cofactors that reach them, of 84 digits and more, do not), so that the factorisation of a small number costs its curves alone.

## The arithmetic

All the methods share one arithmetic, residues modulo the odd `n` in Montgomery's form, fully or lazily reduced: in registers up to eight limbs, the [powers](pow.md)' arithmetic (one limb, two limbs, and three to eight by the coarsely integrated operand scanning; on aarch64 by its kernels in inline assembly, whose squares take each product of two different limbs once), with residues kept below `2n` and no subtraction after each product when `n` is below `R/4` (the top limb below `2^62`; on the other targets two limbs then take REDC a limb at a time, 8 products against 12); nine limbs and more in memory: up to sixteen on aarch64 by the powers' kernels of residues in memory, the window of the rows in registers, and beyond, and on the other targets, the multiplication's kernels and REDC by the kernel's passes, or by products from 100 limbs. The operations write their result to a residue of their own, or in place, so that each method is written once for all the sizes. Inversions are done by the extended Euclidean algorithm in registers for one limb, the binary one for two, and [`mod_inverse`](gcd.md) beyond.

## Measured

Measured on an Apple M4 Max, one thread, each contender interleaved with its reference in the same process and the medians of 3 to 7 rounds given, with nothing else running unless a table says otherwise; GMP-ECM 7.0.7 (configured "Tuned for default", generic `MODMULN` arithmetic), FLINT 3.6.0, GMP 6.3.0, PARI/GP 2.19.0 and msieve 1.53 from Homebrew.

Stage 1 of one curve of this version (GMP-ECM's `-param 2` curves, `x = 2`, curves in blocks), measured on 2026-11-11 on the quiet machine (load 3.8 to 5.3 on its 16 cores, falling from the builds just before; the medians of 7 rounds at `B1 = 11,000` and 5 at 50,000), each row interleaved in one process with the curves of 0.2.32 (Suyama's, `σ = 6`, two at a time) and with GMP-ECM's `-param 0` (Suyama's) and `-param 1` (its default for these sizes and the faster of the two: `d = s²/2^64`, `x = 2`, one product a bit by a word), per curve, the time of four curves less that of two, on semiprimes the curves do not factor:

| Bits | B1 | stage 1 | 0.2.32 | GMP-ECM `-param 0` | GMP-ECM `-param 1` | both stages to `100·B1` | 0.2.32 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 124 | 11,000 | 0.270 ms | 0.427 ms | 3.32 ms | 2.10 ms | 0.491 ms | 0.673 ms |
| 128 | 11,000 | 0.374 ms | 0.535 ms | 3.61 ms | 2.42 ms | 0.677 ms | 0.926 ms |
| 190 | 11,000 | 0.747 ms | 1.08 ms | 3.69 ms | 2.62 ms | 1.34 ms | 1.80 ms |
| 256 | 11,000 | 1.67 ms | 2.33 ms | 5.42 ms | 3.91 ms | 2.60 ms | 3.67 ms |
| 380 | 11,000 | 2.29 ms | 4.51 ms | 6.41 ms | 5.09 ms | 4.08 ms | 7.09 ms |
| 512 | 11,000 | 4.31 ms | 8.78 ms | 9.61 ms | 7.70 ms | 8.03 ms | 13.3 ms |
| 1024 | 11,000 | 14.6 ms | 23.3 ms | 27.0 ms | 22.9 ms | 23.0 ms | 35.1 ms |
| 124 | 50,000 | 1.17 ms | 1.86 ms | 14.4 ms | 9.30 ms | 2.00 ms | 2.87 ms |
| 256 | 50,000 | 7.73 ms | 10.4 ms | 23.7 ms | 16.9 ms | 11.2 ms | 15.5 ms |
| 512 | 50,000 | 19.6 ms | 39.9 ms | 43.1 ms | 34.4 ms | 30.2 ms | 58.0 ms |
| 1024 | 50,000 | 66.3 ms | 106 ms | 121 ms | 102 ms | 90.0 ms | 153 ms |

- Stage 1 is 1.34 to 2.04 times faster than in 0.2.32, and both stages to `100·B1` 1.34 to 1.92 times (a product less a bit from `x = 2`, four curves at a time for one and two limbs, and the arithmetic's kernels in inline assembly, above), the most at 380 and 512 bits (six and eight limbs, 1.97 to 2.04 times); the gain is least at 256 bits (four limbs with the top one above `2^62`, which are not lazily reduced), 1.34 and 1.39 times. Stage 1 is faster than GMP-ECM's default curves (`-param 1`) at every size, 7.8 and 7.9 times at 124 bits, 1.76 and 1.79 at 512 and 1.55 and 1.57 at 1024, where 0.2.32 was up to 1.16 times slower.
- One curve of these and one of Suyama's found factors of 32 bits as often (31.8 % and 31.2 % of 3000 semiprimes at `B1 = 300`, `B2 = 15,000`), and in stage 1 the curves of `σ = 2, 3, …` found the factors GMP-ECM's `-sigma 2:σ` found (above).

The times above are of the crate built with fat link-time optimisation and one codegen unit. With cargo's default release profile (no link-time optimisation, 16 codegen units), the same numbers, each build in a process of its own right after the table above (load 3.0 to 3.8, the medians of nine rounds; ratio = this version's time over 0.2.32's):

| Bits | B1 | stage 1, fat LTO | 0.2.32, fat LTO | ratio | stage 1, default profile | 0.2.32, default profile | ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 256 | 11,000 | 1.62 ms | 2.23 ms | 0.725 | 1.61 ms | 2.21 ms | 0.729 |
| 512 | 11,000 | 4.32 ms | 8.55 ms | 0.505 | 4.33 ms | 7.37 ms | 0.588 |
| 256 | 50,000 | 7.45 ms | 10.0 ms | 0.742 | 7.45 ms | 10.2 ms | 0.732 |
| 512 | 50,000 | 19.3 ms | 38.8 ms | 0.497 | 19.4 ms | 33.2 ms | 0.584 |

This version's stage 1 takes the same time with either profile (within 1 %), as does 0.2.32's at 256 bits; at 512 bits 0.2.32's is 1.16 and 1.17 times faster with the default profile, so that the gain there is 1.70 and 1.71 times with it, against 1.98 and 2.01 with fat link-time optimisation.

The second stage, on the stage 1 of this version (GMP-ECM's `-param 2` curves from `σ = 2`, the curves above), one elliptic curve against GMP-ECM's `-param 0` curve of `σ = 6` (Suyama's): stage 1 at `B1`, and stage 2 to `B2 = 100·B1` for both: here by the continuation chosen, and by 0.2.32's (the column "baby-step giant-step": that continuation alone, on 0.2.32's own stage 1, Suyama's curves of `σ = 6`, and its arithmetic), for semiprimes of the bits given that the curves do not factor. Ours per curve is the time of four curves less that of two, so that the exponent of stage 1 and the plan of stage 2, made once a call, are left out. Measured on 2026-11-11 on the quiet machine (load 1.5 to 2.0 on its 16 cores; one thread, every process with `MallocNanoZone` unset, the harness built with fat LTO and one codegen unit; the medians of 5 rounds), each row interleaved in one process with 0.2.32 and GMP-ECM:

| Bits | B1 | stage 1 | GMP-ECM | stage 2, `100·B1` | baby-step giant-step | GMP-ECM, `100·B1` |
| --- | --- | --- | --- | --- | --- | --- |
| 124 | 11,000 | 0.27 ms | 3.35 ms | 0.24 ms | 0.24 ms | 1.97 ms |
| 128 | 11,000 | 0.37 ms | 3.75 ms | 0.28 ms | 0.43 ms | 2.61 ms |
| 190 | 11,000 | 0.684 ms | 3.61 ms | 0.56 ms | 0.66 ms | 3.01 ms |
| 256 | 11,000 | 1.56 ms | 5.15 ms | 0.861 ms | 1.27 ms | 3.53 ms |
| 380 | 11,000 | 2.27 ms | 6.30 ms | 1.70 ms | 2.32 ms | 3.99 ms |
| 512 | 11,000 | 4.05 ms | 9.53 ms | 3.91 ms | 4.76 ms | 6.07 ms |
| 1024 | 11,000 | 14.6 ms | 26.8 ms | 8.11 ms | 11.8 ms | 13.3 ms |
| 124 | 50,000 | 1.09 ms | 14.4 ms | 0.943 ms | 1.05 ms | 5.30 ms |
| 190 | 50,000 | 3.53 ms | 16.5 ms | 2.25 ms | 2.67 ms | 7.52 ms |
| 256 | 50,000 | 7.37 ms | 22.3 ms | 3.25 ms | 5.04 ms | 8.34 ms |
| 380 | 50,000 | 10.5 ms | 28.4 ms | 7.72 ms | 9.27 ms | 10.5 ms |
| 512 | 50,000 | 19.6 ms | 41.4 ms | 10.7 ms | 16.7 ms | 14.9 ms |
| 1024 | 50,000 | 65.7 ms | 121 ms | 23.3 ms | 46.8 ms | 31.7 ms |
| 256 | 250,000 | 37.0 ms | 116 ms | 11.2 ms | 23.5 ms | 23.9 ms |
| 512 | 250,000 | 96.1 ms | 208 ms | 22.5 ms | 77.8 ms | 41.2 ms |
| 1024 | 250,000 | 325 ms | 595 ms | 53.3 ms | 204 ms | 93.5 ms |
| 256 | 10^6 | 146 ms | 466 ms | 26.5 ms | 82.2 ms | 53.3 ms |
| 512 | 10^6 | 386 ms | 837 ms | 50.3 ms | 289 ms | 95.8 ms |

- Stage 1: from 12 and 13 times faster than GMP-ECM's `-param 0` curve at 124 bits to 2.1 to 2.4 times at 512 bits and 1.83 and 1.84 times at 1024, against GMP-ECM's PRAC chain for each prime with `mpz` calls; and 1.34 to 2.07 times faster than 0.2.32's.
- Stage 2 to the same `B2`: up to 256 bits at `B1` to 50,000, and at 380 bits and 11,000, the baby-step giant-step continuation is the one chosen, 2.3 to 9.2 times faster than GMP-ECM, and from as fast as 0.2.32's (0.99 times at 124 bits and 11,000, a microsecond and a half) to 1.55 times faster, the same continuation on the arithmetic above; at 380 bits from `B1 = 50,000`, from 512 bits, and at 256 bits from `B1 = 250,000`, the FFT continuation, 1.2 to 5.7 times faster than 0.2.32's baby-step giant-step one and 1.36 to 2.1 times faster than GMP-ECM, which was up to 3.0 times faster than 0.2.32 (512 bits, `B1 = 10^6`).

Against FLINT's `fmpz_factor_ecm`, which takes its own random curves, a curve with both stages to `100·B1` (here four curves less two, there ten less five), on semiprimes that four curves of this version and of 0.2.32 and GMP-ECM's `-param 0` curve do not factor, measured on 2026-11-11 on the quiet machine (load 1.5 to 1.8, the same conditions as above), interleaved with 0.2.32:

| Bits | B1 | a curve | FLINT |
| --- | --- | --- | --- |
| 128 | 11,000 | 0.65 ms | 6.78 ms |
| 128 | 50,000 | 2.73 ms | 29.5 ms |
| 256 | 11,000 | 2.44 ms | 14.7 ms |
| 256 | 50,000 | 10.5 ms | 67.8 ms |
| 512 | 11,000 | 7.86 ms | 27.1 ms |
| 512 | 50,000 | 29.7 ms | 118 ms |
| 1024 | 11,000 | 22.8 ms | 79.4 ms |
| 1024 | 50,000 | 89.2 ms | 348 ms |

From 3.4 to 10.8 times faster than FLINT, and 1.33 to 1.86 times faster than 0.2.32 (0.87 to 151 ms in the same rounds).

`p − 1` and `p + 1` against GMP-ECM's `-pm1` (`x_0 = 3`) and `-pp1` (its own starting value), stage 1 at `B1` and stage 2 to `100·B1`, and 0.2.32 ("before": the baby-step giant-step continuation, the ladder for `p + 1` below 9 limbs), for semiprimes neither finds; here each call makes its plan of stage 2, counted; `p + 1` here tries three starting values, GMP-ECM one. Measured on 2026-11-11 on the quiet machine (load 1.6 to 2.3, the same conditions as above), interleaved:

| Method | Bits | B1 | stage 1 | before | GMP-ECM | stage 2, `100·B1` | before | GMP-ECM, `100·B1` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `p − 1` | 128 | 10^5 | 1.43 ms | 1.50 ms | 2.79 ms | 0.84 ms | 4.36 ms | 3.31 ms |
| `p − 1` | 128 | 10^6 | 15.3 ms | 16.3 ms | 26.7 ms | 2.52 ms | 37.4 ms | 8.05 ms |
| `p − 1` | 256 | 10^5 | 2.18 ms | 2.60 ms | 4.33 ms | 1.40 ms | 10.6 ms | 4.94 ms |
| `p − 1` | 256 | 10^6 | 22.9 ms | 27.9 ms | 45.3 ms | 4.50 ms | 95.7 ms | 12.6 ms |
| `p − 1` | 512 | 10^5 | 4.63 ms | 7.41 ms | 8.90 ms | 2.54 ms | 35.1 ms | 8.40 ms |
| `p − 1` | 512 | 10^6 | 47.0 ms | 74.5 ms | 84.6 ms | 8.91 ms | 305 ms | 22.5 ms |
| `p − 1` | 1024 | 10^5 | 14.0 ms | 21.1 ms | 25.6 ms | 6.00 ms | 85.4 ms | 18.7 ms |
| `p − 1` | 1024 | 10^6 | 142 ms | 210 ms | 248 ms | 22.5 ms | 754 ms | 53.6 ms |
| `p + 1`, 3 values | 128 | 10^5 | 4.24 ms | 5.08 ms | 6.36 ms | 3.74 ms | 9.28 ms | 3.72 ms |
| `p + 1`, 3 values | 128 | 10^6 | 42.7 ms | 51.1 ms | 56.8 ms | 12.9 ms | 81.8 ms | 11.3 ms |
| `p + 1`, 3 values | 256 | 10^5 | 9.78 ms | 13.3 ms | 8.88 ms | 7.67 ms | 28.5 ms | 5.50 ms |
| `p + 1`, 3 values | 256 | 10^6 | 99.5 ms | 136 ms | 84.3 ms | 27.0 ms | 257 ms | 16.1 ms |
| `p + 1`, 3 values | 512 | 10^5 | 23.5 ms | 47.6 ms | 16.4 ms | 14.7 ms | 101 ms | 10.2 ms |
| `p + 1`, 3 values | 512 | 10^6 | 236 ms | 475 ms | 155 ms | 56.3 ms | 885 ms | 33.0 ms |
| `p + 1`, 3 values | 1024 | 10^5 | 72.9 ms | 110 ms | 45.4 ms | 34.7 ms | 255 ms | 26.1 ms |
| `p + 1`, 3 values | 1024 | 10^6 | 0.733 s | 1.11 s | 439 ms | 128 ms | 2.25 s | 82.7 ms |

- Stage 1 of `p − 1` is 1.75 to 1.98 times faster than GMP-ECM's, a square a bit and two additions for a one, and 1.05 to 1.6 times faster than 0.2.32's, the most from 512 bits (the arithmetic above). That of `p + 1` costs, a starting value at a time, about the same as `p − 1`'s at 128 bits (0.93 and 0.99 times) and 1.45 to 1.73 times it from 256 bits: the ladder's square and product a bit overlap where a product waits on its latency (two limbs) and add up where it is bound by its additions (from about six), where PRAC's chains now take about 1.5 products a bit; with the arithmetic it is 1.20 to 2.03 times faster than 0.2.32's, the most at 512 bits. A starting value is 1.80 to 4.5 times faster than GMP-ECM's, so that the three take 0.67 to 1.67 times GMP-ECM's one.
- Stage 2 of `p − 1`, by Montgomery and Kruppa's continuation, is 5.2 to 34 times faster than 0.2.32's and 2.4 to 3.9 times faster than GMP-ECM's, which uses the same method with its own transforms; that of `p + 1`, two products of polynomials an evaluation where `p − 1` takes one, 2.5 to 17.5 times faster than 0.2.32's, and 1.76 to 3.0 times faster than GMP-ECM's a starting value at a time, so that the three take 1.0 to 1.7 times GMP-ECM's one.

Pollard's rho against FLINT's `fmpz_factor_pollard_brent` (one try, run again until it finds), for semiprimes of 128 bits with a prime of the bits given, the mean of 20 (10 for the last two rows), measured on 2026-11-11 on the quiet machine (load 1.4):

| Factor bits | rho | FLINT |
| --- | --- | --- |
| 20 | 24.2 µs | 40.6 µs |
| 24 | 62.9 µs | 185 µs |
| 28 | 347 µs | 693 µs |
| 32 | 1.16 ms | 2.75 ms |
| 36 | 4.68 ms | 10.4 ms |
| 40 | 14.7 ms | 37.4 ms |

From 1.7 to 2.9 times faster: Montgomery's form in registers, a square, an addition, a subtraction and a product a step, 128 products between two greatest common divisors.

Factorisations below `2^64` against FLINT's `n_factor` (in the same process) and PARI's `factor` (by gp on the same sets: each set factored over and over until a pass takes 200 ms, the best of three such passes, so that every number is fresh, as ours and FLINT's are timed; factoring each number over and over, as the earlier measurements of this page did, flatters PARI by about 1.5 times: in this run, each of the first 200 numbers of a set over and over until 50 ms took 0.61 to 0.69 of the times below), the mean over sets of 1000 to 4000 numbers (random 64-bit numbers, and products of two random primes of the bits given), measured on 2026-11-11 on the quiet machine (load 2.9 to 3.0, and 1.6 to 1.7 for PARI; the medians of 7 rounds):

| Numbers | `factor` | 0.2.32 | FLINT `n_factor` | PARI |
| --- | --- | --- | --- | --- |
| random of 64 bits | 5.19 µs | 5.43 µs | 10.4 µs | 27.8 µs |
| primes of 16 and 16 bits | 0.82 µs | 992 ns | 1.74 µs | 2.6 µs |
| 20 and 20 | 4.17 µs | 4.32 µs | 6.73 µs | 8.9 µs |
| 24 and 24 | 7.30 µs | 7.70 µs | 18.4 µs | 49.8 µs |
| 28 and 28 | 15.8 µs | 17.3 µs | 65.0 µs | 75.2 µs |
| 32 and 32 | 25.6 µs | 28.2 µs | 251 µs | 133 µs |
| 20 and 44 | 7.11 µs | 7.50 µs | 9.15 µs | 133 µs |

From 1.29 to 9.8 times faster than FLINT and 2.1 to 19 times faster than PARI, 1.04 to 1.22 times faster than 0.2.32: Hart's method below 40 bits, then curves in one limb, four at a time after a first block of two, from one sequence for all the levels, with bounds for the size and `B2 = 35·B1`, the stages of their bounds made once for the program.

Whole factorisations against PARI's `factor` and FLINT's `fmpz_factor`, the mean and the median over the 24 numbers of each kind of 2026-11-07 (balanced semiprimes, two primes of half the digits, and random numbers), measured on 2026-11-11 on the quiet machine (load 1.7 to 2.9, the medians of 5 rounds, 3 at 60 digits), `factor` interleaved with 0.2.32 and FLINT in one process, PARI in its own on the same numbers (load 1.4 to 1.6), each number factored over and over until 50 ms pass (the method that flatters PARI on words, above; not measured here for longer numbers):

| Numbers | Digits | `factor` mean, median | 0.2.32 | PARI | FLINT |
| --- | --- | --- | --- | --- | --- |
| semiprimes | 20 | 0.104, 0.0907 ms | 0.111, 0.102 ms | 0.160, 0.159 ms | 1.02, 0.64 ms |
| semiprimes | 24 | 0.211, 0.237 ms | 0.355, 0.283 ms | 0.382, 0.379 ms | 2.30, 2.40 ms |
| semiprimes | 28 | 0.542, 0.597 ms | 1.33, 1.32 ms | 0.83, 0.83 ms | 2.22, 2.19 ms |
| semiprimes | 32 | 1.20, 1.20 ms | 1.98, 2.21 ms | 1.98, 2.00 ms | 4.07, 3.90 ms |
| semiprimes | 36 | 2.34, 2.35 ms | 4.38, 4.41 ms | 4.99, 5.00 ms | 8.67, 8.42 ms |
| semiprimes | 40 | 5.14, 5.18 ms | 9.18, 9.26 ms | 12.0, 12.4 ms | 20.4, 19.3 ms |
| semiprimes | 44 | 11.5, 11.9 ms | 20.8, 22.6 ms | 38.9, 39.0 ms | 42.1, 39.7 ms |
| semiprimes | 50 | 46, 46 ms | 77, 78 ms | 129, 131 ms | 236, 236 ms |
| semiprimes | 60 | 0.36, 0.36 s | 0.53, 0.54 s | 1.44, 1.44 s | 2.05, 2.09 s |
| random | 20 | 5.0, 3.7 µs | 7.9, 4.9 µs | 27.3, 8.2 µs | 107, 3.0 µs |
| random | 30 | 79, 39 µs | 63, 25 µs | 244, 191 µs | 1.44, 1.78 ms |
| random | 40 | 0.39, 0.20 ms | 0.86, 0.16 ms | 2.49, 1.89 ms | 4.59, 3.91 ms |
| random | 50 | 5.41, 1.9 ms | 12.6, 1.8 ms | 34.8, 16.0 ms | 56.0, 30.1 ms |
| random | 60 | 26, 3.6 ms | 41, 11 ms | 104, 45 ms | 173, 68.3 ms |

Sets of 24 numbers are too few for the curves: each number takes a geometric number of curves, and this version and 0.2.32 take different sequences of curves, so that single numbers, and the means and medians of 24, move far more than the code does: the random numbers of 30 digits took 1.24 and 1.56 times as long as with 0.2.32 here (mean and median), and 0.62 and 0.90 of its time on the 400 below. On larger sets of the same kinds (PARI's `random` with seeds 20261108 to 20261110), measured on 2026-11-11, interleaved, on the quiet machine (load 1.4 to 1.7, the medians of 3 rounds; the means over the sets of 400 and of 2000 numbers of the same digits differ by up to 5 %, the curves' luck):

| Numbers | Digits | numbers | `factor` mean, median | 0.2.32 | ratio of means, medians |
| --- | --- | --- | --- | --- | --- |
| semiprimes | 20 | 400 | 68, 51 µs | 88.5, 70 µs | 0.76, 0.72 |
| semiprimes | 22 | 400 | 145, 110 µs | 180, 146 µs | 0.80, 0.75 |
| semiprimes | 24 | 400 | 172, 170 µs | 314, 234 µs | 0.55, 0.72 |
| semiprimes | 26 | 400 | 280, 289 µs | 553, 386 µs | 0.51, 0.75 |
| semiprimes | 22 | 2000 | 148, 111 µs | 188, 146 µs | 0.79, 0.76 |
| semiprimes | 24 | 2000 | 181, 170 µs | 327, 246 µs | 0.55, 0.69 |
| semiprimes | 26 | 2000 | 282, 290 µs | 573, 384 µs | 0.49, 0.75 |
| semiprimes | 28 | 160 | 0.445, 0.46 ms | 1.09, 1.14 ms | 0.41, 0.40 |
| semiprimes | 30 | 160 | 0.689, 0.699 ms | 1.35, 1.39 ms | 0.51, 0.50 |
| semiprimes | 32 | 160 | 1.02, 1.03 ms | 1.90, 1.93 ms | 0.53, 0.54 |
| semiprimes | 36 | 64 | 2.06, 2.02 ms | 3.86, 3.83 ms | 0.53, 0.53 |
| semiprimes | 40 | 64 | 4.97, 5.11 ms | 8.38, 8.54 ms | 0.59, 0.60 |
| random | 20 | 400 | 7.0, 2.5 µs | 7.8, 3.0 µs | 0.89, 0.84 |
| random | 25 | 400 | 28.6, 7.4 µs | 35.5, 8.4 µs | 0.81, 0.88 |
| random | 30 | 400 | 84.5, 21 µs | 136, 24 µs | 0.62, 0.90 |
| random | 35 | 400 | 198, 49 µs | 376, 72 µs | 0.53, 0.67 |
| random | 40 | 400 | 463, 72.5 µs | 775, 105 µs | 0.60, 0.69 |
| random | 50 | 200 | 3.74, 1.31 ms | 6.97, 1.49 ms | 0.54, 0.88 |
| random | 60 | 200 | 26.6, 2.86 ms | 41.5, 6.42 ms | 0.64, 0.44 |

- Balanced semiprimes, with the quadratic sieve from 24 digits (0.2.32 took it from 28): `factor` is faster than PARI at every size, 1.5 to 4.0 times on the mean and 1.4 to 4.0 at the median, the least at 20 and 28 digits (1.54 and 1.53 times on the mean), the most at 44 and 60 (3.4 and 4.0 times); and 3.4 to 11 times faster than FLINT (3.3 to 10 at the median). Against 0.2.32, the larger sets give 1.25 to 2.4 times on the means and 1.3 to 2.5 at the medians, the most from 24 to 32 digits on the means (1.8 to 2.4 times), the least at 20 and 22 digits (1.25 to 1.3), where the curves still take the numbers.
- Random numbers, whose cofactors are mostly small or prime, 3.1 to 6.4 times faster than PARI on average and 2.2 to 13 times at the median, and faster than FLINT at every size but at the median of 20 digits (3.7 µs against 3.0); against 0.2.32, on the larger sets, 1.12 to 1.9 times on the means and 1.11 to 2.2 at the medians.

Before the quadratic sieve was in, with other work running on the machine, the same semiprimes by PARI's `factorint(n, 1)`, without its quadratic sieve, so that its curves find the factors, against `factor`'s curves alone (the median):

| Digits | 24 | 28 | 32 | 36 | 40 | 44 | 50 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `factor` | 0.33 ms | 1.61 ms | 5.00 ms | 8.06 ms | 78 ms | 172 ms | 2.27 s |
| PARI's curves | 12.0 ms | 46.0 ms | 130 ms | 689 ms | 1.83 s | 5.44 s | 28.4 s |

From 12 to 85 times faster with the curves alone.

The levels' second stages, measured on 2026-11-11: the curves each level took, traced through `factor` for products of a prime of 25 digits with one of 33 (2100 of them) and of 30 digits with one of 45 (300), costed at each size by the time of one curve there (its stages kept, the bounds interleaved, the machine quiet), with `B2` of 200, 400 and 800 times `B1` at 25, 30 and 35 digits against `100·B1` on every level (the mean and the median):

| Limbs | 6 | 8 | 12 | 16 |
| --- | --- | --- | --- | --- |
| prime of 25 digits | 0.93, 0.98 | 0.92, 0.95 | 0.90, 0.94 | 0.88, 0.94 |
| prime of 30 digits | 0.95, 0.93 | 0.94, 0.92 | 0.90, 0.88 | 0.88, 0.87 |

The 90 % intervals of the means are ±3 % at 25 digits and ±8 % at 30. `factor` itself, both at once on the same numbers (load 5 to 40): 0.82 of the time on 100 products of 12 limbs with a prime of 25 digits, 1.02 on 200 of 6 limbs and on 200 of 8, whose primes are unlucky for the longer bounds (the traces of the same primes give 0.99 and 0.98, and 0.82 for those of 12 limbs; one number's time moves by up to 10 times with the sequence of curves). Longer still (`400·B1` at 25 digits, `800·B1` at 30) took 1.00 to 0.90 and 1.00 to 0.88 of the time, more than these bounds at every size. Factors of 20 digits take as long as before (0.98 to 1.00 on 3000 of them), and the numbers below 84 digits, which go to the quadratic sieve before the level of 25 digits, are untouched. A shorter `B2 = 50·B1` at 12 and 15 digits saved at most 2 % on factors of 15 and 20 digits, and for the fine levels of two limbs 2 to 8 % on balanced semiprimes of 20 to 23 digits but cost 3 % on products of a prime of 8 to 11 digits in the sieve's range, so those stay at `100·B1`.

Perfect powers against GMP's `mpz_perfect_power_p` (which only tells whether) and FLINT's `fmpz_is_perfect_power` (which gives one exponent, not the largest), the mean over 64 numbers of each shape, each timed over 300 calls, measured on 2026-11-11 on the quiet machine (load 1.4 to 1.5), interleaved with 0.2.32: random odd numbers, primes, cubes and squares of primes, and squares of primes plus 2:

| Input | `perfect_power` | 0.2.32 | GMP | FLINT |
| --- | --- | --- | --- | --- |
| random odd, 64 bits | 11.2 ns | 19.7 ns | 110 ns | 15.7 ns |
| prime, 64 bits | 21.6 ns | 49.4 ns | 330 ns | 17.2 ns |
| cube of 21 bits | 35.6 ns | 48.7 ns | 260 ns | 23.9 ns |
| square of 32 bits | 37.0 ns | 52.0 ns | 126 ns | 23.2 ns |
| square of 32 bits plus 2 | 8.0 ns | 11.7 ns | 80.4 ns | 12.0 ns |
| random odd, 128 bits | 48.2 ns | 56.0 ns | 114 ns | 181 ns |
| prime, 128 bits | 154 ns | 189 ns | 378 ns | 866 ns |
| square of 64 bits | 60.0 ns | 79.7 ns | 144 ns | 490 ns |
| random odd, 256 bits | 75.0 ns | 144 ns | 181 ns | 438 ns |
| prime, 256 bits | 91.0 ns | 271 ns | 533 ns | 1.79 µs |
| cube of 85 bits | 298 ns | 341 ns | 337 ns | 810 ns |
| square of 128 bits | 161 ns | 231 ns | 192 ns | 680 ns |
| random odd, 1024 bits | 229 ns | 399 ns | 512 ns | 2.21 µs |
| prime, 1024 bits | 517 ns | 1.17 µs | 2.03 µs | 11.8 µs |
| cube of 341 bits | 690 ns | 937 ns | 617 ns | 2.68 µs |
| square of 512 bits | 442 ns | 804 ns | 422 ns | 2.31 µs |
| square of 512 bits plus 2 | 123 ns | 115 ns | 281 ns | 290 ns |
| random odd, 4096 bits | 1.49 µs | 2.00 µs | 2.35 µs | 20.3 µs |
| cube of 1365 bits | 2.76 µs | 3.63 µs | 3.65 µs | 7.67 µs |
| square of 2048 bits | 2.38 µs | 3.55 µs | 3.41 µs | 6.42 µs |
| square of 2048 bits plus 2 | 454 ns | 452 ns | 1.29 µs | 589 ns |

1.1 to 3.0 times faster than 0.2.32 but on the squares of 512 and 2048 bits plus 2 (1.07 times slower and as fast, a few nanoseconds); faster than GMP and FLINT on every shape from 128 bits but the squares of 512 bits and the cubes of 341 bits against GMP (1.05 and 1.12 times slower), which takes a 2-adic root and, as FLINT does, stops at the first exponent, where `perfect_power` goes on with the root for the largest; below `2^64`, FLINT is faster on primes and powers (1.3 to 1.6 times), the powers' time here including the allocation of the root.

Trial division against FLINT's `fmpz_factor_trial_range` with the same primes, of numbers without small factors, so that every prime is tried, measured on 2026-11-11 on the quiet machine (load 1.4), interleaved with 0.2.32:

| Bits | to 1000 | 0.2.32 | FLINT | to 8191 | 0.2.32 | FLINT | to 10^5 | FLINT | to 10^6 | FLINT |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 64 | 69 ns | 136 ns | 349 ns | 201 ns | 526 ns | 1.78 µs | 4.86 µs | 16.5 µs | 38.8 µs | 133 µs |
| 128 | 134 ns | 146 ns | 445 ns | 768 ns | 819 ns | 2.56 µs | 8.53 µs | 23.9 µs | 75.5 µs | 195 µs |
| 256 | 170 ns | 187 ns | 608 ns | 1.01 µs | 1.09 µs | 3.56 µs | 11.9 µs | 33.4 µs | 109 µs | 270 µs |
| 1024 | 327 ns | 401 ns | 2.12 µs | 2.24 µs | 2.99 µs | 12.4 µs | 33.2 µs | 122 µs | 353 µs | 989 µs |
| 4096 | 957 ns | 1.39 µs | 5.93 µs | 7.63 µs | 10.6 µs | 35.8 µs | 133 µs | 336 µs | 1.33 ms | 2.73 ms |

From 2.1 to 8.8 times faster than FLINT, and to 1000 and 8191 1.07 to 2.6 times faster than 0.2.32: the remainders by products of primes eight at a time and their primes tested together, and for a word eight primes at a time, against a remainder for each prime.

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

- Time complexity: trial division `O(n·g)` limb operations for `n` limbs and `g` products of primes, and a multiplication for each prime; rho about `√p` steps for the smallest prime `p`, each two products of residues; `p − 1` about `1.44·B1` squares, `p + 1` `2.9·B1` products (`2.2·B1` by PRAC), and an elliptic curve `13·B1` products in stage 1 (4 squares and 5 products a bit); the baby-step giant-step continuation about `0.8·(π(B2) − π(B1))` products and the sieving of `(B1, B2]`, once a plan, and the continuations by polynomials `O(√B2)` products of residues and products of polynomials of `O(√B2)` coefficients, `O(√B2·log B2)` products of numbers a block for the curves, the products of numbers `O(m log m)` for `m` limbs; a product of residues `O(n²)` below 100 limbs and about 2.7 products of `n` limbs beyond; perfect powers a pass or two of the number for most, and up to `log2 x / 5.7` roots for the few that pass the residues
- Space complexity: the residues of the method, in registers up to eight limbs; the exponent of stage 1 (`1.44·B1` bits, in chunks of `2^22` bits, kept up to `B1 = 10^8`), the baby steps (`φ(D)/2` residues) and a block of 128 giant steps, and the plan of stage 2 (a bit per baby step a giant step, kept up to 32 MB, and the parameters of the continuations by polynomials, about 8 KB when asked for); the continuations by polynomials, a few polynomials of `O(√B2)` coefficients and the numbers of their products, whose factors are at most 16 MiB each (the curves' tree of `F`, `d` coefficients a level, at most 64 MiB), the product twice that, and the multiplication's working memory: at the longest ranges about 200 MB in all for the curves (`B2 = 10^5·B1`, four limbs) and up to about 330 MB for Montgomery and Kruppa's (`B2 = 10^13`, four limbs), whatever `B2`, as the products are capped (past those bounds the baby-step giant-step continuation is taken); the candidates for Montgomery and Kruppa's `P`, 511 of them, about 20 KB, and the multipliers of PRAC's chains for the last `B1` of `p + 1`, a byte a prime up to `10^7`, kept for the program; the factorisation's kept stages, a few megabytes at most; the table of the primes to `2^20`, 2.6 MB, made on the first trial division beyond 8192
- Soundness: the bytes beyond a vector's length, which may be uninitialised, are never read; every residue lives in a vector of its own or in registers, and the kernels' working memory is checked against its plan in builds with debug assertions; the numbers are left as they were
