[A](../../../README.md) › [Math](../../../README.md#math)

# Factorisation

[Numbers](numbers.md) factor into primes with `a.factor()`, which gives a `Factorization`: the primes in increasing order with their exponents. The methods it chooses from are public in `a::math::numbers::factor`: `trial` (trial division), `rho` (Pollard's rho with Brent's cycle finding), `pm1` and `pp1` (Pollard's `p − 1` and Williams's `p + 1`), `ecm` (Lenstra's elliptic curve method) and `siqs` (the [quadratic sieve](siqs.md)). `a.perfect_power()` tells whether `a` is `b^k` for some `k >= 2`, with the largest `k`.

- `factor` gives the factorisation of the absolute value of `a`, in the byte order of `a`: none for 1 and −1; it panics for 0. The primes are probable primes by [`is_probable_prime`](prime.md), exact below `2^64`. It always ends: the elliptic curves go on with growing bounds until they find a factor, in a time that grows with the second largest prime factor.
- `Factorization::new(factors, order)` takes primes and exponents in any order, sorts them, merges equal primes and drops exponents of 0; it panics for a "prime" below 2 and for exponents that add up past `u32::MAX`, and does not test the primes. `factors()`, `into_factors()` and `byte_order()` give them back, `value()` the number they make: in limbs on the stack while each product has room for 46 limbs, the product so far and the next factor together, checked before each product (each prime read once from its bytes, a word's powers in pieces of a word, the number written once at the end), and by a balanced tree of products when one has not (so a value of 46 limbs may take either: a prime of 23 limbs times one of 24 takes the tree).
- `trial(n, bound)` gives the primes up to `bound` that divide `|n|`, with their exponents, in increasing order, and the cofactor left, `|n|` divided by them, in the byte order of `n`; for 0, no primes and the cofactor 0. Once the cofactor is below the square of the next prime it is 1 or prime, and is given among the primes when it is at most `bound`: `trial(n, u64::MAX)` factors any `n` below `2^64` completely.
- `rho(n, iterations)`, `pm1(n, b1, b2)`, `pp1(n, b1, b2)` and `ecm(n, b1, b2, curves)` give a factor of `|n|` other than 1 and `|n|`, in the byte order of `n`, or `None`: always `None` for 0 to 3 and for primes (which [`is_probable_prime`](prime.md) tells at once), and `Some(2)` for an even number above 2. The factor need not be prime. Each is deterministic: the same arguments give the same factor.
- `perfect_power` gives `Some((b, k))` when `|a| = |b|^k` with `k >= 2` the largest such exponent, `b` with the sign of `a`, so that `k` is odd for a negative `a` (`−64 = (−4)^3`); `None` for 0, 1 and −1 and every other number, as PARI's `ispower` does. `is_perfect_power` tells whether it is `Some` (so `false` for 0, 1 and −1, as PARI answers), and stops at the first exponent it finds, without forming the roots of the others, as GMP's `mpz_perfect_power_p` does; GMP's function, unlike this, counts 0, 1 and −1 as powers.
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
- The estimates are those measured. For the curves, a prime of the baby-step giant-step continuation costs 0.25 of a product of residues at one limb, 0.38 at two, 0.6 at three to eight, 0.52 at 9 to 16 and 0.8 from 17 (the four chains overlap, and a pair serves two primes; from 17 limbs a product in memory takes longer than its estimate), its share of the sieve (0.17 ns a number, once a plan) is counted for eight curves, and the FFT continuation costs 0.85 of the sum of its parts. Measured on 2026-11-14 on an Apple M4 Max, each continuation forced, stage 2 alone, on 428 cells of 1 to 32 limbs, `B1` from 2000 to 250,000 and `B2` from `5·10^4` to `5·10^8`, the choice now takes the faster in all but 15, at most 9 % slower (two limbs about `B2 = 10^8`, where the lazily reduced residues cross later than the others); before, it took the polynomials 2 to 8 times too early in `B2` at 3 to 16 limbs and more than 5 times at one and two, the slower in 114 cells and up to 2.7 times slower. The crossovers measured, for `B1` from 11,000 (from 17 limbs, for `B1 = 2000`):

| Limbs | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 and 10 | 12 | 14 | 16 | 17 and 18 | 20 | 24 | 28 | 32 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `B2` | above `5·10^7` | `5·10^7` (`1.2·10^8` lazily reduced) | `1.5·10^7` to `1.9·10^7` | `1.3·10^7` to `1.5·10^7` | `7·10^6` | `6·10^6` | `5·10^6` | `4·10^6` | `4.5·10^6` | `2.5·10^6` | `1.8·10^6` | `1.1·10^6` to `2.1·10^6` | `4.6·10^5` | `3.7·10^5` | `2.6·10^5` | `2.1·10^5` | `1.7·10^5` |
| before | `7·10^6` | `8·10^6` | `7·10^6` | `5.5·10^6` | `3.5·10^6` | `2·10^6` to `3·10^6` | `1.2·10^6` | `10^6` | below `10^6` | below `6.5·10^5` | below `5·10^5` | `5·10^5` to `10^6` | `4·10^5` | `3.4·10^5` | `2.8·10^5` | `2.2·10^5` | `1.8·10^5` |

- For `p − 1` and `p + 1` the products of the baby-step giant-step continuation also cost less than their estimate and the sieve more, which a call makes once; the two cancel near the crossovers, where the choice takes the faster in 607 of 620 such cells (at most 1.37 times slower, `p + 1` at one to four limbs about `B2 = 10^6`), as well as estimates made true to the measurements do, so that choice is unchanged.

## Perfect powers

`x = c^K` with `K` the largest exponent is a `p`-th power for a prime `p` exactly when `p` divides `K`, with the root `c^(K/p)`, so the primes are tried in increasing order, each as long as it gives a root: their product is `K`.

- The exponents of 2 and of the odd primes to 47 in `x` are multiples of `K`. One pass of the number gives its remainders by the product of the squares of the odd primes to 29 and by the product of the primes from 31 to 47, in two independent chains, and from them the remainders by each prime, by constant divisors: a prime to 29 that divides `x` only once makes `K = 1` at once, which settles almost every number. When the exponents found (counted up to 64) have a common divisor `g`, only the primes dividing `g` are tried.
- Each prime `p` is tried up to the bits of `x` over `log2` of its smallest possible prime factor (53 when none is below): first by residues, a `p`-th power being one modulo every prime: modulo 64 and the primes to 47 for squares, the primes `q ≡ 1 (mod p)` to 47 for `p = 3, 5, 7, 11`, from the remainders already known, and up to four primes `q ≡ 1 (mod p)` below `2^16` for larger `p`; then, when the root has at most 40 bits, by the floating-point root from `log2 x`, computed once for all `p`: as `log2 x` is known to `2^−52·log2 x`, the root of a power is within `2^−6.9` of an integer, so a root further than `2^−6` from one rules the power out without a product, and an integer root is checked modulo `2^64`, then exactly; otherwise by [`nth_root`](root.md) or `isqrt_rem` and an exact check.
- Below `2^64` the same is done on words, the remainders by constant divisors and the roots in floating point (`sqrt`, `cbrt` or a power), checked exactly; below `2^128`, in registers, the roots by the root's own routine for `u128`.
- `is_perfect_power` takes the same steps and stops at the first prime `p` that gives a root: the root for that `p`, which shows the power, is formed, but not the search past it, nor the root for the largest exponent that `perfect_power` gives. On the squares and cubes measured below it takes 20.9 ns where `perfect_power` takes 43.1 for the square of a prime of 32 bits, 617 ns against 780 for the cube of one of 341 bits and 963 ns against 2.53 µs for the square of one of 2048 bits, and as long on numbers that are no powers.

## The factorisation

`factor` chooses the method by size, as PARI, FLINT and yafu do:

- Trial division by the primes below 8192. Each cofactor then takes the exact test below `2^64` or Baillie–PSW above, the test for a perfect power, and the search for a factor, and both parts go round again; a cofactor that comes down to a limb goes on without dividing by the small primes again.
- One limb: Hart's one line method (2012) up to 40 bits, `s = ⌈√(480·n·i)⌉` for `i = 1, 2, …` until `s² mod 480·n` is a square `t²`, with the factor `gcd(s − t, n)`; then the elliptic curve method from `B1 ≈ 64·1.08^(bits − 44)`, after two curves on each of the three smaller bounds, 8 curves a level, `B1` growing by 1.25 a level, `B2 = 35·B1`.
- Two limbs: a run of 2048 steps of rho, then curves from `B1 ≈ 450·1.0583^(bits − 64)`, the bound for a factor of half the bits, after four curves on each of the three smaller bounds, `max(8, B1/250)` curves a level, `B1` growing by 1.25 a level, `B2 = 100·B1`; when the quadratic sieve is to come, from `B1 = 191` instead, four curves on each bound below the one for the size, until the sieve takes over.
- More limbs: 4096 steps of rho, `p − 1` from 45 digits (`B1 = 10^5`, `B2 = 10^7`), then GMP-ECM's levels for factors of 10, 12, 15, 20, … digits (`B1` from 150 to `2.9·10^9`), `B2 = 100·B1` up to 20 digits and 200, 400 and 800 times `B1` at 25, 30 and 35 digits (800 times beyond): the continuation by polynomials makes a long second stage cheap against the first, and each of those levels takes GMP-ECM's curves times the ratio of one curve's chances at `100·B1` and at its `B2` (GMP-ECM's model), 251, 501 and 1119 curves at 25 to 35 digits for 300, 700 and 1800, so that it finds its factors as often in less time.
- The quadratic sieve takes the cofactors of 24 to 110 digits once the curves have made factors of 30 % of their digits unlikely (yafu's 4/13).

The search for a factor of a cofactor runs in one arithmetic modulo it, rho, `p − 1` and all the levels of the curves, which are taken in turn from one sequence. The stages of the bounds that recur, the exponents of the first and the plans of the second, are kept for the life of the program once made (up to `B1` of about `5·10^5` for two limbs and `10^6` beyond, a few megabytes at most: the longer second stages of the levels from 25 digits would keep bitmaps of 0.14, 1.2 and 9.6 MB only if they took the baby-step giant-step continuation, which the cofactors that reach them, of 84 digits and more, do not), so that the factorisation of a small number costs its curves alone.

## The arithmetic

All the methods share one arithmetic, residues modulo the odd `n` in Montgomery's form, fully or lazily reduced: in registers up to eight limbs, the [powers](pow.md)' arithmetic (one limb, two limbs, and three to eight by the coarsely integrated operand scanning; on aarch64 by its kernels in inline assembly, whose squares take each product of two different limbs once), with residues kept below `2n` and no subtraction after each product when `n` is below `R/4` (the top limb below `2^62`; on the other targets two limbs then take REDC a limb at a time, 8 products against 12); nine limbs and more in memory: up to sixteen on aarch64 by the powers' kernels of residues in memory, the window of the rows in registers, and beyond, and on the other targets, the multiplication's kernels and REDC by the kernel's passes, or by products from 86 limbs. The operations write their result to a residue of their own, or in place, so that each method is written once for all the sizes. Inversions are done by the extended Euclidean algorithm in registers for one limb, the binary one for two, and [`mod_inverse`](gcd.md) beyond.

## Measured

Measured on 2026-11-26 on an Apple M4 Max with nothing else running, by the harness in the repository's `bench/nt` (its method in `bench/README.md`), on one thread: this crate as a user builds it, by rustc 1.100 stable in cargo's release profile (no LTO, 16 codegen units, no `target-cpu`); the Rust references in the same profile, machine-factor 0.2.5 and malachite 0.15.0; the C references as Homebrew builds them, GMP 6.3.0, FLINT 3.6.0, PARI 2.19.1 and GMP-ECM 7.0.7, called through shims compiled with `-O3 -mcpu=native`. Each figure is the median of 15 samples of at least a millisecond (fewer for the longest calls, 3 at the least), in time per call, this crate and every reference alternating sample by sample in one process, each sample on the same numbers for all, after a check that all give the same results (PARI's rho, whose factor may validly differ, excepted). A ratio is this crate's time over that of the reference: below 1, this crate is faster.

One elliptic curve of `ecm`, two to eight curves a call timed per curve, against GMP-ECM's `-param 1` curve, its default here, one curve a call with its parameters and its product for stage 1 kept from call to call, and FLINT's `fmpz_factor_ecm`, its own curves, four or eight a call; on semiprimes that the curves do not factor, so that every curve runs its stages in full:

| Bits | B1 | stage 1 | GMP-ECM | ratio | both stages, `B2 = 100·B1` | GMP-ECM | ratio | FLINT | ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 128 | 11,000 | 386 µs | 2.32 ms | 0.167 | 705 µs | 4.94 ms | 0.143 | 7.15 ms | 0.0986 |
| 128 | 50,000 | 1.80 ms | 10.6 ms | 0.169 | 3.00 ms | 16.5 ms | 0.182 | 30.4 ms | 0.0986 |
| 128 | 250,000 | 9.05 ms | 50.7 ms | 0.179 | 14.4 ms | 64.2 ms | 0.225 | – | – |
| 256 | 11,000 | 1.58 ms | 3.44 ms | 0.460 | 2.52 ms | 7.17 ms | 0.351 | 14.9 ms | 0.169 |
| 256 | 50,000 | 7.25 ms | 15.6 ms | 0.466 | 10.8 ms | 24.0 ms | 0.451 | 67.0 ms | 0.161 |
| 256 | 250,000 | 36.5 ms | 78.4 ms | 0.465 | 47.9 ms | 103 ms | 0.466 | – | – |
| 256 | 1,000,000 |  |  |  | 173 ms | 371 ms | 0.466 | – | – |
| 512 | 11,000 | 4.38 ms | 7.24 ms | 0.605 | 7.21 ms | 13.7 ms | 0.527 | 27.6 ms | 0.261 |
| 512 | 50,000 | 19.8 ms | 32.8 ms | 0.604 | 30.3 ms | 47.7 ms | 0.634 | 125 ms | 0.242 |
| 512 | 250,000 | 99.0 ms | 164 ms | 0.604 | 122 ms | 206 ms | 0.593 | – | – |
| 512 | 1,000,000 |  |  |  | 451 ms | 757 ms | 0.596 | – | – |
| 1024 | 11,000 | 14.5 ms | 22.5 ms | 0.644 | 22.9 ms | 36.3 ms | 0.631 | 78.4 ms | 0.292 |
| 1024 | 50,000 | 65.5 ms | 102 ms | 0.645 | 89.0 ms | 132 ms | 0.673 | 345 ms | 0.258 |
| 1024 | 250,000 | 329 ms | 510 ms | 0.644 | 381 ms | 594 ms | 0.641 | – | – |

And with GMP-ECM's own bounds for the levels of factors of 20 to 40 digits:

| Bits | factors of | B1 | B2 | a curve | GMP-ECM | ratio |
| --- | --- | --- | --- | --- | --- | --- |
| 256 | 20 digits | 11,000 | 1,900,000 | 3.11 ms | 8.63 ms | 0.360 |
| 256 | 25 digits | 50,000 | 13,000,000 | 15.5 ms | 33.2 ms | 0.469 |
| 256 | 30 digits | 250,000 | 130,000,000 | 67.8 ms | 144 ms | 0.471 |
| 256 | 35 digits | 1,000,000 | 1,000,000,000 | 240 ms | 531 ms | 0.453 |
| 256 | 40 digits | 3,000,000 | 5,700,000,000 | 717 ms | 1.47 s | 0.489 |
| 512 | 20 digits | 11,000 | 1,900,000 | 8.97 ms | 16.7 ms | 0.537 |
| 512 | 25 digits | 50,000 | 13,000,000 | 36.9 ms | 64.4 ms | 0.573 |
| 512 | 30 digits | 250,000 | 130,000,000 | 163 ms | 281 ms | 0.580 |
| 512 | 35 digits | 1,000,000 | 1,000,000,000 | 583 ms | 1.05 s | 0.554 |
| 512 | 40 digits | 3,000,000 | 5,700,000,000 | 1.73 s | 2.90 s | 0.598 |
| 1024 | 20 digits | 11,000 | 1,900,000 | 26.9 ms | 43.4 ms | 0.620 |
| 1024 | 25 digits | 50,000 | 13,000,000 | 103 ms | 171 ms | 0.603 |
| 1024 | 30 digits | 250,000 | 130,000,000 | 470 ms | 766 ms | 0.614 |

- Stage 1 takes 0.167 to 0.179 of GMP-ECM's time at 128 bits, two limbs in registers, 0.460 to 0.466 at 256 bits and 0.604 to 0.645 from 512 bits; both stages to `100·B1` 0.143 to 0.673 of GMP-ECM's time and 0.0986 to 0.292 of FLINT's; at GMP-ECM's levels 0.360 to 0.620 of its time.

`p − 1` and `p + 1` against GMP-ECM's `-pm1` (`x_0 = 3`) and `-pp1`, the latter with the same three starting values as `pp1` (2/7, 6/5 and 6), and FLINT's `fmpz_factor_pp1` (with 3, 5 and 7, and `B2_sqrt = sqrt(100·B1)`), on semiprimes that none of them factors:

| Method | Bits | B1 | stage 1 | GMP-ECM | ratio | both stages, `B2 = 100·B1` | GMP-ECM | ratio | FLINT | ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `p − 1` | 128 | `10^5` | 1.41 ms | 2.78 ms | 0.506 | 2.19 ms | 6.41 ms | 0.341 | – | – |
| `p − 1` | 128 | `10^6` | 15.2 ms | 27.7 ms | 0.549 | 17.9 ms | 36.0 ms | 0.496 | – | – |
| `p − 1` | 256 | `10^5` | 2.12 ms | 3.90 ms | 0.543 | 3.59 ms | 9.27 ms | 0.387 | – | – |
| `p − 1` | 256 | `10^6` | 22.9 ms | 40.7 ms | 0.562 | 27.6 ms | 54.3 ms | 0.508 | – | – |
| `p − 1` | 512 | `10^5` | 4.52 ms | 8.23 ms | 0.549 | 7.17 ms | 17.0 ms | 0.423 | – | – |
| `p − 1` | 512 | `10^6` | 46.9 ms | 84.3 ms | 0.556 | 56.3 ms | 108 ms | 0.523 | – | – |
| `p − 1` | 1024 | `10^5` | 14.1 ms | 24.6 ms | 0.573 | 20.0 ms | 44.0 ms | 0.455 | – | – |
| `p − 1` | 1024 | `10^6` | 141 ms | 246 ms | 0.573 | 163 ms | 301 ms | 0.542 | – | – |
| `p + 1`, 3 values | 128 | `10^5` | 3.69 ms | 17.4 ms | 0.212 | 7.40 ms | 29.8 ms | 0.248 | 27.6 ms | 0.268 |
| `p + 1`, 3 values | 128 | `10^6` | 37.9 ms | 172 ms | 0.221 | 50.5 ms | 211 ms | 0.240 | 160 ms | 0.316 |
| `p + 1`, 3 values | 256 | `10^5` | 9.55 ms | 24.9 ms | 0.384 | 17.4 ms | 42.0 ms | 0.415 | 69.0 ms | 0.253 |
| `p + 1`, 3 values | 256 | `10^6` | 97.7 ms | 246 ms | 0.398 | 124 ms | 295 ms | 0.420 | 424 ms | 0.293 |
| `p + 1`, 3 values | 512 | `10^5` | 23.1 ms | 45.9 ms | 0.504 | 37.7 ms | 77.9 ms | 0.484 | 125 ms | 0.303 |
| `p + 1`, 3 values | 512 | `10^6` | 233 ms | 460 ms | 0.506 | 289 ms | 557 ms | 0.519 | 746 ms | 0.388 |
| `p + 1`, 3 values | 1024 | `10^5` | 72.4 ms | 131 ms | 0.555 | 107 ms | 210 ms | 0.510 | 321 ms | 0.334 |
| `p + 1`, 3 values | 1024 | `10^6` | 725 ms | 1.30 s | 0.556 | 856 ms | 1.56 s | 0.548 | 2.08 s | 0.412 |

- `p − 1`: stage 1 0.506 to 0.573 of GMP-ECM's time, a square a bit and two additions for a one; both stages 0.341 to 0.542, by Montgomery and Kruppa's continuation, which GMP-ECM takes too, on its own transforms.
- `p + 1` with three starting values: stage 1 0.212 to 0.556 of GMP-ECM's time and both stages 0.240 to 0.548, the smallest ratios at 128 bits; 0.253 to 0.412 of FLINT's.

Pollard's rho against FLINT's `fmpz_factor_pollard_brent` (one try of `2^26` steps, again until it finds) and PARI's `Z_pollardbrent` (seeds 1, 2, … until it finds), for semiprimes of 128 bits with a prime of the bits given, the mean over the set:

| Factor bits | numbers | `rho` | FLINT | ratio | PARI | ratio |
| --- | --- | --- | --- | --- | --- | --- |
| 20 | 16 | 19.3 µs | 35.0 µs | 0.550 | 124 µs | 0.156 |
| 24 | 16 | 68.9 µs | 181 µs | 0.380 | 500 µs | 0.138 |
| 28 | 16 | 250 µs | 704 µs | 0.355 | 2.01 ms | 0.124 |
| 32 | 16 | 1.18 ms | 2.48 ms | 0.477 | 10.4 ms | 0.114 |
| 36 | 8 | 3.66 ms | 9.79 ms | 0.374 | 29.5 ms | 0.124 |
| 40 | 8 | 12.2 ms | 26.2 ms | 0.465 | 131 ms | 0.0928 |

0.355 to 0.550 of FLINT's time and 0.0928 to 0.156 of PARI's: Montgomery's form in registers, a square, an addition, a subtraction and a product a step, 128 products between two greatest common divisors.

Factorisations below `2^64` against FLINT's `n_factor`, machine-factor's `factorize`, PARI's `Z_factor` (each number fresh) and malachite's `u64::factor` (a port of FLINT's `n_factor`), the same numbers for all in each sample, the mean over random 64-bit numbers and over products of two random primes of the bits given:

| Numbers | `factor` | FLINT `n_factor` | machine-factor | PARI | malachite | ratio | fastest |
| --- | --- | --- | --- | --- | --- | --- | --- |
| random of 64 bits | 4.96 µs | 11.6 µs | 11.9 µs | 27.6 µs | 73.1 µs | 0.427 | FLINT |
| primes of 16 and 16 bits | 850 ns | 1.65 µs | 2.69 µs | 2.55 µs | 21.3 µs | 0.515 | FLINT |
| 20 and 20 | 4.29 µs | 6.59 µs | 7.89 µs | 8.86 µs | 31.7 µs | 0.651 | FLINT |
| 24 and 24 | 7.26 µs | 18.9 µs | 22.0 µs | 48.8 µs | 60.2 µs | 0.385 | FLINT |
| 28 and 28 | 15.6 µs | 66.0 µs | 70.9 µs | 72.2 µs | 211 µs | 0.236 | FLINT |
| 32 and 32 | 26.1 µs | 258 µs | 256 µs | 129 µs | 472 µs | 0.202 | PARI |
| 20 and 44 | 7.28 µs | 8.92 µs | 8.46 µs | 130 µs | 235 µs | 0.861 | machine-factor |

0.202 to 0.861 of the fastest reference's time: Hart's method below 40 bits, then curves in one limb, four at a time after a first block of two, from one sequence for all the levels, with bounds for the size and `B2 = 35·B1`, the stages of their bounds made once for the program.

Whole factorisations against PARI's `Z_factor` and FLINT's `fmpz_factor`, each set factored whole in every sample, the mean time a number given: balanced semiprimes, two random primes of half the digits; random numbers; and smooth numbers, twice a product of random primes of the bits given, as the `p − 1` of the primes of `discrete_log`'s measurements on the [modular arithmetic](modular.md)'s page:

| Numbers | `factor`, mean | PARI | ratio | FLINT | ratio |
| --- | --- | --- | --- | --- | --- |
| balanced semiprimes, 20 digits, mean of 48 | 72.1 µs | 185 µs | 0.389 | 437 µs | 0.165 |
| balanced semiprimes, 24 digits, mean of 48 | 185 µs | 435 µs | 0.426 | 2.34 ms | 0.0794 |
| balanced semiprimes, 28 digits, mean of 48 | 469 µs | 818 µs | 0.573 | 2.06 ms | 0.227 |
| balanced semiprimes, 32 digits, mean of 32 | 1.02 ms | 1.85 ms | 0.549 | 3.76 ms | 0.270 |
| balanced semiprimes, 36 digits, mean of 24 | 2.05 ms | 4.03 ms | 0.508 | 7.12 ms | 0.288 |
| balanced semiprimes, 40 digits, mean of 16 | 5.07 ms | 12.2 ms | 0.416 | 17.2 ms | 0.295 |
| balanced semiprimes, 44 digits, mean of 12 | 9.71 ms | 36.4 ms | 0.267 | 47.2 ms | 0.206 |
| balanced semiprimes, 50 digits, mean of 6 | 38.3 ms | 107 ms | 0.357 | 184 ms | 0.208 |
| balanced semiprimes, 60 digits, mean of 3 | 374 ms | 1.30 s | 0.288 | 2.82 s | 0.132 |
| random, 20 digits, mean of 256 | 5.67 µs | 39.4 µs | 0.144 | 34.4 µs | 0.165 |
| random, 30 digits, mean of 256 | 83.7 µs | 334 µs | 0.250 | 1.59 ms | 0.0527 |
| random, 40 digits, mean of 96 | 744 µs | 2.68 ms | 0.278 | 4.67 ms | 0.159 |
| random, 50 digits, mean of 32 | 5.71 ms | 25.8 ms | 0.221 | 34.5 ms | 0.165 |
| random, 60 digits, mean of 16 | 31.0 ms | 157 ms | 0.197 | 218 ms | 0.142 |
| 2 times primes of 20 bits, 64 bits, mean of 64 | 7.95 µs | 43.7 µs | 0.182 | 12.3 µs | 0.648 |
| 2 times primes of 20 bits, 128 bits, mean of 32 | 53.9 µs | 384 µs | 0.141 | 382 µs | 0.141 |
| 2 times primes of 32 bits, 128 bits, mean of 16 | 182 µs | 4.68 ms | 0.0389 | 7.41 ms | 0.0245 |
| 2 times primes of 24 bits, 256 bits, mean of 16 | 1.40 ms | 2.60 ms | 0.538 | 1.30 ms | 1.08 |
| 2 times primes of 32 bits, 256 bits, mean of 8 | 3.90 ms | 26.7 ms | 0.146 | 5.31 ms | 0.735 |
| 2 times primes of 28 bits, 512 bits, mean of 4 | 19.8 ms | 24.6 ms | 0.805 | 13.4 ms | 1.48 |
| 2 times primes of 40 bits, 512 bits, mean of 2 | 24.7 ms | 809 ms | 0.0306 | 43.8 ms | 0.565 |

- Balanced semiprimes, with the quadratic sieve from 24 digits: 0.267 to 0.573 of PARI's time (the closest at 28 and 32 digits, 0.573 and 0.549), and 0.0794 to 0.295 of FLINT's.
- Random numbers, whose cofactors are mostly small or prime: 0.144 to 0.278 of PARI's time and 0.0527 to 0.165 of FLINT's.
- Smooth numbers: 0.0306 to 0.805 of PARI's time, and 0.0245 to 0.735 of FLINT's but for the two sets of the most primes, of 24 bits in 256 and of 28 bits in 512 (below).

Perfect powers against PARI's `ispower(n, , &r)` and malachite's `express_as_power`, which give the largest exponent and its root, as `perfect_power` does, and `is_perfect_power` against GMP's `mpz_perfect_power_p`, which only tells whether; FLINT's `fmpz_is_perfect_power` gives one exponent and its root, not the largest, so it does less and is shown apart; the mean over 64 numbers of each shape, random odd numbers, primes, cubes and squares of primes, and squares plus 2:

| Input | `perfect_power` | PARI | malachite | ratio | FLINT, one exponent (ratio) | `is_perfect_power` | GMP | ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random odd, 64 bits | 12.8 ns | 40.1 ns | 46.2 ns | 0.320 | 13.2 ns (0.969) | 12.7 ns | 111 ns | 0.115 |
| random odd, 128 bits | 57.8 ns | 1.11 µs | – | 0.0522 | 203 ns (0.284) | 56.9 ns | 135 ns | 0.423 |
| random odd, 256 bits | 74.2 ns | 2.30 µs | – | 0.0323 | 547 ns (0.136) | 73.5 ns | 211 ns | 0.349 |
| random odd, 1024 bits | 233 ns | 4.69 µs | – | 0.0496 | 1.41 µs (0.165) | 233 ns | 399 ns | 0.584 |
| random odd, 4096 bits | 1.47 µs | 13.8 µs | – | 0.107 | 19.2 µs (0.0767) | 1.47 µs | 2.08 µs | 0.704 |
| prime, 64 bits | 27.2 ns | 74.2 ns | 339 ns | 0.366 | 20.3 ns (1.34) | 26.5 ns | 343 ns | 0.0774 |
| prime, 128 bits | 161 ns | 4.84 µs | – | 0.0331 | 977 ns (0.164) | 158 ns | 378 ns | 0.419 |
| prime, 256 bits | 108 ns | 7.92 µs | – | 0.0136 | 1.97 µs (0.0550) | 107 ns | 532 ns | 0.202 |
| prime, 1024 bits | 577 ns | 23.6 µs | – | 0.0245 | 12.6 µs (0.0459) | 573 ns | 2.06 µs | 0.279 |
| cube of a prime of 21 bits | 46.0 ns | 92.3 ns | 25.3 ns | 1.82 | 34.2 ns (1.35) | 22.8 ns | 248 ns | 0.0917 |
| cube of a prime of 85 bits | 362 ns | 4.25 µs | – | 0.0852 | 874 ns (0.415) | 298 ns | 329 ns | 0.907 |
| cube of a prime of 341 bits | 780 ns | 12.3 µs | – | 0.0633 | 2.74 µs (0.285) | 617 ns | 608 ns | 1.01 |
| cube of a prime of 1365 bits | 3.20 µs | 35.8 µs | – | 0.0892 | 7.56 µs (0.423) | 2.04 µs | 3.95 µs | 0.517 |
| square of a prime of 32 bits | 43.1 ns | 76.0 ns | 24.5 ns | 1.76 | 21.6 ns (2.00) | 20.9 ns | 133 ns | 0.157 |
| square of a prime of 64 bits | 71.4 ns | 270 ns | – | 0.265 | 507 ns (0.141) | 33.3 ns | 155 ns | 0.215 |
| square of a prime of 128 bits | 187 ns | 4.94 µs | – | 0.0378 | 691 ns (0.270) | 122 ns | 204 ns | 0.599 |
| square of a prime of 512 bits | 523 ns | 14.2 µs | – | 0.0368 | 2.32 µs (0.225) | 302 ns | 454 ns | 0.665 |
| square of a prime of 2048 bits | 2.53 µs | 42.3 µs | – | 0.0598 | 6.47 µs (0.391) | 963 ns | 3.46 µs | 0.278 |
| square of 32 bits plus 2 | 10.2 ns | 30.1 ns | 20.9 ns | 0.487 | 12.8 ns (0.794) | 9.99 ns | 91.7 ns | 0.109 |
| square of 512 bits plus 2 | 101 ns | 209 ns | – | 0.480 | 145 ns (0.695) | 103 ns | 207 ns | 0.496 |
| square of 2048 bits plus 2 | 470 ns | 1.33 µs | – | 0.353 | 717 ns (0.655) | 476 ns | 1.17 µs | 0.406 |

- `perfect_power` takes 0.0136 to 0.487 of the faster of PARI's and malachite's time but on the powers of words (below); `is_perfect_power`, which stops at the first exponent it finds, 0.0774 to 0.907 of GMP's time, and as long at the cube of a prime of 341 bits (1.01).
- FLINT's `fmpz_is_perfect_power`, which answers less, is faster on the primes and powers of words (`perfect_power` takes 1.34 to 2.00 times its time there); from 128 bits `perfect_power` takes 0.0459 to 0.695 of it.

Trial division against FLINT's `fmpz_factor_trial_range` with the same primes, of numbers without small factors, so that every prime is tried:

| Bits | to 1000 | FLINT | ratio | to 8191 | FLINT | ratio | to `10^5` | FLINT | ratio | to `10^6` | FLINT | ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 64 | 64.8 ns | 362 ns | 0.179 | 221 ns | 1.81 µs | 0.122 | 4.74 µs | 16.5 µs | 0.288 | 39.3 µs | 151 µs | 0.260 |
| 128 | 122 ns | 479 ns | 0.254 | 756 ns | 2.57 µs | 0.294 | 8.61 µs | 23.9 µs | 0.360 | 75.0 µs | 194 µs | 0.385 |
| 256 | 158 ns | 647 ns | 0.243 | 989 ns | 3.75 µs | 0.264 | 12.0 µs | 34.8 µs | 0.344 | 108 µs | 287 µs | 0.378 |
| 1024 | 334 ns | 2.08 µs | 0.161 | 2.27 µs | 12.3 µs | 0.185 | 35.2 µs | 117 µs | 0.302 | 339 µs | 929 µs | 0.365 |
| 4096 | 999 ns | 5.82 µs | 0.172 | 7.78 µs | 35.4 µs | 0.220 | 132 µs | 336 µs | 0.393 | 1.30 ms | 2.74 ms | 0.474 |


0.122 to 0.474 of FLINT's time: the remainders by products of primes eight at a time and their primes tested together, and for a word eight primes at a time, against a remainder for each prime.

Where the references are faster, this crate's time over the fastest same-work reference's above 1.05:

- **Numbers with many primes of 24 to 28 bits.** `factor` of twice a product of primes of 24 bits, 256 bits in all, 1.08 times FLINT's `fmpz_factor`, and of primes of 28 bits, 512 bits in all, 1.48 times (PARI's 0.538 and 0.805): the driver finds these primes one curve run at a time, and starts again on each cofactor, where a pass of rho or `p − 1` over the whole cofactor, or curves whose stage 1 catches several primes at once, could take many together.
- **Powers of words.** `perfect_power` of the cube of a prime of 21 bits and of the square of one of 32 bits, 1.82 and 1.76 times malachite's `express_as_power` (46.0 ns against 25.3 and 43.1 ns against 24.5): the root's vector, about a third of the time, and residue tests that a word could take more cheaply.

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

- Time complexity: trial division `O(n·g)` limb operations for `n` limbs and `g` products of primes, and a multiplication for each prime; rho about `√p` steps for the smallest prime `p`, each two products of residues; `p − 1` about `1.44·B1` squares, `p + 1` `2.9·B1` products (`2.2·B1` by PRAC), and an elliptic curve `13·B1` products in stage 1 (4 squares and 5 products a bit); the baby-step giant-step continuation about `0.8·(π(B2) − π(B1))` products and the sieving of `(B1, B2]`, once a plan, and the continuations by polynomials `O(√B2)` products of residues and products of polynomials of `O(√B2)` coefficients, `O(√B2·log B2)` products of numbers a block for the curves, the products of numbers `O(m log m)` for `m` limbs; a product of residues `O(n²)` below 86 limbs and about 2.5 products of `n` limbs beyond (2.7 from the transforms' threshold); perfect powers a pass or two of the number for most, and up to `log2 x / 5.7` roots for the few that pass the residues
- Space complexity: the residues of the method, in registers up to eight limbs; the exponent of stage 1 (`1.44·B1` bits, in chunks of `2^22` bits, kept up to `B1 = 10^8`), the baby steps (`φ(D)/2` residues) and a block of 128 giant steps, and the plan of stage 2 (a bit per baby step a giant step, kept up to 32 MB, and the parameters of the continuations by polynomials, about 8 KB when asked for); the continuations by polynomials, a few polynomials of `O(√B2)` coefficients and the numbers of their products, whose factors are at most 16 MiB each (the curves' tree of `F`, `d` coefficients a level, at most 64 MiB), the product twice that, and the multiplication's working memory: at the longest ranges about 200 MB in all for the curves (`B2 = 10^5·B1`, four limbs) and up to about 330 MB for Montgomery and Kruppa's (`B2 = 10^13`, four limbs), whatever `B2`, as the products are capped (past those bounds the baby-step giant-step continuation is taken); the candidates for Montgomery and Kruppa's `P`, 511 of them, about 20 KB, and the multipliers of PRAC's chains for the last `B1` of `p + 1`, a byte a prime up to `10^7`, kept for the program; the factorisation's kept stages, a few megabytes at most; the table of the primes to `2^20`, 2.6 MB, made on the first trial division beyond 8192
- Soundness: the bytes beyond a vector's length, which may be uninitialised, are never read; every residue lives in a vector of its own or in registers, and the kernels' working memory is checked against its plan in builds with debug assertions; the numbers are left as they were
