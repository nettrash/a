[A](../../../README.md) › [Math](../../../README.md#math)

# Sequences

[Numbers](numbers.md) give five integer sequences: `Number::fibonacci(n, order)` and `Number::lucas(n, order)`, the Fibonacci and Lucas numbers; `Number::lucas_sequence(&p, &q, n)` and `Number::lucas_sequence_mod(&p, &q, &n, &m)`, the Lucas sequences `U_n(P, Q)` and `V_n(P, Q)`, exactly and modulo a number; `Number::primorial(n, order)`, the product of the primes up to `n`; and `Number::partitions(n, order)`, the number of partitions of `n`.

- `F_0 = 0`, `F_1 = 1`, `L_0 = 2`, `L_1 = 1`, and `X_(k+2) = X_(k+1) + X_k`; `U_0 = 0`, `U_1 = 1`, `V_0 = 2`, `V_1 = P`, and `X_(k+2) = P X_(k+1) − Q X_k`, for integers `P` and `Q` of any size and sign. `n#` is 1 for `n < 2`; `p(0) = 1`. The indices of `fibonacci`, `lucas`, `lucas_sequence`, `primorial` and `partitions` are `u64`s, that of `lucas_sequence_mod` a number of any size.
- The results of `fibonacci`, `lucas`, `primorial` and `partitions` are in the byte order `order`; those of `lucas_sequence` in the byte order of `p`, and of `lucas_sequence_mod` in that of `m`, each in `0..m` (Euclid's remainders). Every result is in the shortest form, in a new vector of exactly its length.
- `lucas_sequence_mod` panics when `m` is zero or negative, or `n` negative; nothing else panics but when a result or its working memory cannot be allocated, as for any vector: `F_n` takes `0.0868 n` bytes, `n#` about `0.18 n`, and `p(n)` about `0.46 √n`, with the segmented sieve's segments and its primes up to `√n` for `n#`, and a sieve of `4N` bytes for the `N` terms of `p(n)`'s series.
- Each input is read in its own byte order, and may be non-canonical (zero bytes above the top, an empty vector for zero, a negative zero); the inputs are left as they were.

## Fibonacci and Lucas numbers

By fast doubling, as GMP's `mpz_fib_ui` and `mpz_lucnum_ui` compute them:

- Below `2^128`, from tables: `F_n` up to 186, `L_n` up to 184.
- Up to `F_373` and `L_368`, one product of two numbers from the tables, in registers: `F_(2k) = F_k L_k`, `F_(2k+1) = F_(k+2) L_(k−1) + 2(−1)^k`, `L_(2k) = L_k² − 2(−1)^k` and `L_(2k+1) = L_k L_(k+1) − (−1)^k`.
- Beyond, `F_k` and `F_(k−1)` are doubled from the top bits of `k` that the table holds down to its last bit, with two squares a bit: `F_(2k−1) = F_k² + F_(k−1)²`, `F_(2k+1) = 4F_k² − F_(k−1)² + 2(−1)^k` and `F_(2k) = F_(2k+1) − F_(2k−1)`, the `±2` or-ed into a low limb that is 0 or 1 modulo 4, as GMP's `mpn_fib2_ui` does. The first doubling, from the table's pair of two limbs, is in registers; each one after it forms the two values it keeps in one pass over the squares' limbs, three carry chains at once (`x + y`, `4x − y` and their difference): on 64-bit aarch64 an assembly kernel of blocks of four limbs, whose chains hand the carry flag over by `cmp` and `cset` as the addition's kernels do (about 1.3 cycles a limb on an Apple M4 Max, against 4.5 for the loop that LLVM makes of the portable Rust), and that portable loop elsewhere and under Miri.
- The last step of `F_n`, `n = 2k` or `2k + 1`, is one product instead of two squares: `F_(2k) = F_k (F_k + 2F_(k−1))` and `F_(2k+1) = (2F_k + F_(k−1))(2F_k − F_(k−1)) + 2(−1)^k`, its linear parts in one loop up to 32 limbs and by the addition's kernels beyond.
- `L_n` takes the zero bits at the bottom of `n` as squares, `L_(2k) = L_k² − 2(−1)^k`, from the tables or from the odd part `2k + 1`, `L_(2k+1) = 5F_(k−1)(2F_k + F_(k−1)) − 4(−1)^k`.

Squares and products of up to eight limbs are straight-line kernels generated for each length (`sequences/small.py`): the operands in registers, every column of a product summed in a three-limb accumulator (Comba's order), a square's products of two different limbs once, then doubled with the squares of the limbs added; measured on an Apple M4 Max, a square of 4 limbs took 14 cycles against 27 for GMP's `mpn_sqr`, of 8 limbs 47 against 68, and a product of 8 by 8 79 against 87. Squares of up to 48 limbs are the multiplication's schoolbook kernel, called directly, and longer ones its Karatsuba's and Toom-Cook's methods and number-theoretic transforms. Up to 8 KiB, the working memory is on the stack.

## Lucas sequences

`lucas_sequence` doubles `U_k` and `U_(k−1)` with two squares a bit, as the Fibonacci numbers are, the identities taking `Q^k` beside them:

- `U_(2k−1) = U_k² − Q U_(k−1)²`, from `U_(m+n) = U_m U_(n+1) − Q U_(m−1) U_n`;
- `U_(2k+1) = (P² − 3Q) U_k² − Q² U_(k−1)² + 2Q^k`, from the same with `U_(k+1) = P U_k − Q U_(k−1)` and `U_k² − P U_k U_(k−1) + Q U_(k−1)² = Q^(k−1)`;
- `U_(2k) = (U_(2k+1) + Q U_(2k−1)) / P`, the recurrence, a division that is exact;
- then `V_n = P U_n − 2Q U_(n−1)`.

For `|P|` and `|Q|` below `2^30` the doubling runs on limbs: its first steps in `i128`s while the bit lengths of the values and of the coefficients keep every product below `2^125`, `P U_(2k)` divided by `P` by the inverse of its odd part modulo `2^128`; then the squares by the kernels of the Fibonacci numbers above (and `Q^k` squared beside them unless `Q = ±1`), and each of the two values that a bit keeps in one pass over the squares' limbs: the linear combination in an `i128` a limb, its coefficients `P² − 3Q`, `P² − 2Q`, `Q²` and `2Q²` below `2^61`, and for `U_(2k)` the exact division by `P` in the same pass, by Hensel's method from the low limb up, then a shift for the power of 2 in `P`. The buffers hold `M^(n+1)`, `M = (|P| + √(P² + 4|Q|))/2`, a bound of `|U_k|` by induction. Longer `P` and `Q` take the same steps on numbers. `P = ±1, Q = −1` are the Fibonacci and Lucas numbers, taken from the doubling above (`L_n = F_n + 2F_(n−1)`), and so are `P = ±3, Q = 1`, `U_n(3, 1) = F_(2n)` and `V_n(3, 1) = L_(2n)`; `Q = 0` (`U_n = P^(n−1)`), `P = 0` (`U_(2j+1) = (−Q)^j`, `V_(2j) = 2(−Q)^j`, the others 0) and `P² = 4Q` (`U_n = n c^(n−1)`, `V_n = 2c^n` for `P = 2c`) have closed forms.

## Lucas sequences modulo a number

`lucas_sequence_mod` splits the modulus as `m = 2^z m'` with `m'` odd. Modulo `m'` the residues are in Montgomery's form, with the arithmetic of the [Baillie–PSW test](prime.md): one and two limbs in registers, three to seven in registers by the coarsely integrated operand scanning, eight and more in limbs with REDC; modulo `2^z` they are the low limbs of products, as the [powers](pow.md) modulo an even number take them. The two results are put together by the Chinese remainder theorem. The index's bits are taken from the top by one of four ladders, chosen by the parameters and the modulus's size:

- `U_k` and `U_(k+1)`, for any `P` and `Q`: `U_(2k) = U_k (2U_(k+1) − P U_k)`, `U_(2k+1) = U_(k+1)² − Q U_k²` and `U_(2k+2) = U_(k+1)(P U_(k+1) − 2Q U_k)`, a product and two squares a bit, the products by `P` and `Q` by additions when they are at most 4 in absolute value; then `V_n = 2U_(n+1) − P U_n`. This is the ladder modulo `2^z`, and modulo `m'` when the others do not apply.
- `U_k` and `V_k`, for `Q = ±1` and `|P| <= 4` and a short index: `U_(2k) = U_k V_k`, `V_(2k) = V_k² − 2Q^k`, and for a one bit `U_(2k+1) = (P U_(2k) + V_(2k))/2` and `V_(2k+1) = (D U_(2k) + P V_(2k))/2`, `D = P² − 4Q`: a product and a square a bit, the rest additions and halvings.
- `V_k` and `V_(k+1)`, for `Q = ±1`: `V_(2k) = V_k² − 2Q^k`, `V_(2k+1) = V_k V_(k+1) − P Q^k`, `V_(2k+2) = V_(k+1)² − 2Q^(k+1)`, a product and a square a bit, as Montgomery's ladder for Williams's method `p + 1` takes them; then `U_n = (2V_(n+1) − P V_n) / D`, by the inverse of `D` modulo `m'` (by the binary method on one limb, by the numbers' gcd beyond). From 48 bits of index modulo one limb, 80 modulo two and 96 beyond, the inverse costs less than the third product the other ladders take.
- `V_k`, `V_(k+1)` and `Q^k`, for a `Q` of more than 4 in absolute value modulo three limbs and more: the same with `Q^(2k) = (Q^k)²` and `Q^(2k+1) = Q^k Q^(k+1)`.

When `D` shares a factor with `m'`, the first ladder is taken. The bits are taken by branches: computing both halves of a step and selecting the results without a branch took 20 to 35 % longer modulo 64 bits, even with random indices.

## Primorials

As GMP's `mpz_primorial_ui`, the primes multiplied by balanced product trees:

- Up to 102, from a table: 101# is the last below `2^128`.
- Below 8192, from the [primality test](prime.md)'s table of the products of consecutive odd primes that fit in a limb, which needs no sieve: those of the groups whose primes are all at most `n`, and the primes up to `n` of the next one.
- Beyond, the same products below 8192, then the odd primes from the crate's segmented sieve ([Prime sieves and counting](sieve.md)), as many to a limb as fit (`checked_mul`).
- The limbs are multiplied a limb at a time up to 16, and by the factorials' balanced product tree beyond, whose longest products take Karatsuba's and Toom-Cook's methods and the transforms; the product is then shifted by a bit for the prime 2.

## Partition numbers

Up to 416 (`p(416)` is the last below `2^64`), from a table of Euler's pentagonal recurrence computed at compile time. Beyond, by Hardy, Ramanujan and Rademacher's series, as Johansson implements it ("Efficient implementation of the Hardy–Ramanujan–Rademacher formula", 2012):

`p(n) = Σ_(k=1)^N T_k + R(n, N)`, `T_k = √(3/k) (4/(24n − 1)) A_k(n) U(C/k)`, `C = (π/6) √(24n − 1)`, `U(x) = cosh x − sinh(x)/x`,

with Rademacher's bound `|R(n, N)| < 44π²/(225√3) N^(−1/2) + (π√2/75) (N/(n − 1))^(1/2) sinh((π/N) √(2n/3))`. `N` is the least number of terms for which an upper bound of it, computed in doubles rounded upwards, is at most 1/4. Each term is computed with a bound of its error, at a precision that falls as `k` grows, and the terms are summed in fixed point with 40 bits below the point; the sum is rounded to the nearest integer once the bounds of all the errors and of the remainder add up to less than 1/2 (with the margins of the schedule they always do; a failed check would raise the precision and start again).

- `A_k(n)`, a sum of `φ(k)` roots of unity with Dedekind sums, is a product of at most one cosine for each prime power of `k`, a square root and a sign, by Whiteman's factorisation (1956): `A_(k1 k2)(n) = A_(k1)(n1) A_(k2)(n2)` for coprime `k1` and `k2` and residues `n1`, `n2` that congruences give, and closed forms for prime powers, which take modular square roots and Jacobi symbols. The parts of 2 and 3 are split off one after the other with their residues; the factors `p^λ` of the larger primes need none, as the congruences give `1 − 24n_p ≡ (1 − 24n)(k/p^λ)^(−2) mod p^λ`: the factor's root is `r (k/p^λ)^(−1)` for a root `r` of `1 − 24n` modulo `p^λ`, the same for every `k` that `p` divides. That root is found once for each prime, by Tonelli and Shanks the first time a term needs it, and kept in the slot of the prime in the sieve of the smallest prime factors (no memory beyond the sieve's); a prime power's is lifted from it by Hensel's method. The arithmetic is in `u64`s (every modulus is below `2^35`, and below `2^32` for any `n` of practical size). Both forms were checked against Selberg's formula `A_k(n) = √(k/3) Σ (−1)^l cos(π(6l + 1)/(6k))`, over `(3l² + l)/2 ≡ −n mod k`, for every `k` below 600 and every `n`.
- The terms whose bound in double precision is small enough, most of them, are computed in doubles, without the platform's `exp` and `cos`, whose accuracy Rust does not specify: polynomials on reduced arguments with IEEE 754's correctly rounded operations, with error bounds of 16 and 12 units of the last place, and the argument's error carried through `e^x`. They are accumulated exactly in an `i128`. The bounds that choose each term's precision take their logarithms from the doubles' exponents and mantissas, from above or from below as the bound needs, not from the system's `log2`.
- Terms of up to 1248 bits are computed in fixed point on arrays of up to 23 limbs on the stack, without allocations: `e^x = 2^q e^r` with `r = x − q ln 2` and a table of `ln 2`, `e^r − 1` from `r 2^−s` by its Taylor series and `s` doublings `w² + 2w`; each cosine from its versine `1 − cos y`, by its series on `y 2^−s` and doublings `4v − 2v²`; each series by rectangular splitting (Smith, 1989), its blocks cut so that the product of their divisors fits a limb, multiplications by them and one division a block. Each value carries a bound of its absolute error. The arithmetic is generic in the length of its arrays: from 2 to 8 limbs the length is a constant, its loops unrolled and its products the generated kernels above, which give the top half of a product directly; terms of 9 to 23 limbs take arrays of 16 and 23 limbs, the length at run time, and the schoolbook kernel; from 9 limbs, cosines with the denominators below come from numbers. `√(3/k)` times the square roots of `A_k(n)` is `√(3 sp/(k sq))`, which takes four values (3, 3/2, 1 and 1/2, by the factors of 2 and 3 of `k`): each is computed once a call, to the longest of its terms, and cut to the others, on limbs and on numbers alike.
- Longer terms are computed in floating point on [numbers](numbers.md), each value with a bound of its relative error. `π` and `ln 2` come from tables of 16,384 bits, and beyond from the [floating numbers](float.md)' constants: `π` from the Chudnovskys' series by binary splitting, its numerator and denominator cut to the bits the quotient needs, `ln 2` from its series, each kept for the process at the longest precision computed so far (up to `2^27` bits, 16 MiB, behind a mutex) and cut from there for shorter ones; `√(24n − 1)` is the numbers' integer square root. `e^C` is computed once: reduced by `ln 2` at every precision (`e^C = 2^q e^r`, which saves the doublings of `C`'s integer bits; beyond the table the first call at a precision pays the series of `ln 2` once), then by its series on `r 2^−s` with rectangular splitting, the blocks' divisors multiplied in and divided out once, and `s = w^(1/3)` doublings `w² + 2w` for `w` bits. For each `k = 2^j m ≥ 2`, `m` odd, whose term is on numbers, `e^(C/k)` is the `m`-th root of `e^C` by Newton's iteration for `a^(−1/m)`, which takes no division, the precision doubled at each step from the start's less the bits of the step's quadratic constant `(m + 1)/2` (without them, the steps fell short of their precision by a shortfall that doubled at each step, and `p(27 10^9)` and `p(10^11)` failed their bounds in 0.2.32), its error bounded from a final residual (or `e^(C/m)` by its own series, when that takes fewer products), then `j` square roots, each of which halves the relative error. `1/x = k/C` from one division. Cosines `cos(π a/b)` with a reduced denominator `b = 2^s 3^t q`, `q` 1 or 5, come from closed forms and the half- and triple-angle formulas, as Arb's `_arb_cos_pi_fmpq_algebraic` takes them: after the octant's reduction, `1/2`, `√2/2`, `√3/2` and `(√5 ± 1)/4` for `b` up to 6, `√((1 ± cos(2θ))/2)` for an even `b` (with `2 bits(b) + 4` more bits a level, for the cancellation of `1 − cos`), and for an odd multiple of 3 the root near its double of the cubic `4c³ − 3c = cos(3θ)` by Newton's method, with guard bits for its conditioning `12c² − 3`, which vanishes at `θ = π/3` (a denominator `3^j` has angles within `π/3^j` of it), its error bounded from the residual and checked to be that root; on an Apple M4 Max, `cos(7π/18)` to 3908 bits went from 225,000 cycles to about 20,000. Other denominators up to 512 come from Newton's method on a Chebyshev polynomial that has a simple root there (`T_(b/2)` for an even denominator, `U_(b−1)` for an odd one), from a double's start, the error bounded from the residual and Markov's inequality; the others from the versine's series.

FLINT's `partitions_fmpz_ui` does the same with Arb's ball arithmetic, which bounds every error as it computes; Johansson's paper bounds them beforehand.

## Measured

Measured on 2026-11-26 on an Apple M4 Max with nothing else running, by the harness in the repository's `bench/nt` (its method in `bench/README.md`), on one thread: this crate as a user builds it, by rustc 1.100 stable in cargo's release profile (no LTO, 16 codegen units, no `target-cpu`); malachite 0.15.0 in the same profile; GMP 6.3.0, FLINT 3.6.0 and PARI 2.19.1 as Homebrew builds them, called through shims compiled with `-O3 -mcpu=native`. Each figure is the median of 15 samples of at least a millisecond, in time per call, this crate and every reference alternating sample by sample in one process, after a check that all give the same values. GMP and FLINT write into results allocated before; this crate, as malachite, returns a new number, whose allocation is in its times. A ratio is this crate's time over that of the fastest reference that does the same work: below 1, this crate is faster.

Fibonacci numbers, against GMP's `mpz_fib_ui`, FLINT's `fmpz_fib_ui` and malachite's `Natural::fibonacci`:

| n | `fibonacci` | GMP | FLINT | malachite | ratio | fastest |
| --- | --- | --- | --- | --- | --- | --- |
| 100 | 10.8 ns | 10.5 ns | 2.92 ns | 38.5 ns | 3.70 | FLINT |
| 187 | 14.5 ns | 25.4 ns | 24.8 ns | 55.4 ns | 0.586 | FLINT |
| 300 | 14.2 ns | 27.5 ns | 27.8 ns | 63.2 ns | 0.517 | GMP |
| 374 | 30.1 ns | 38.6 ns | 37.9 ns | 79.1 ns | 0.794 | FLINT |
| 500 | 27.4 ns | 42.4 ns | 41.6 ns | 94.2 ns | 0.658 | FLINT |
| 748 | 43.4 ns | 58.0 ns | 55.4 ns | 105 ns | 0.783 | FLINT |
| 1000 | 47.5 ns | 75.7 ns | 75.3 ns | 151 ns | 0.630 | FLINT |
| 1500 | 84.6 ns | 102 ns | 101 ns | 193 ns | 0.836 | FLINT |
| 2000 | 99.3 ns | 141 ns | 140 ns | 265 ns | 0.712 | FLINT |
| 3000 | 206 ns | 233 ns | 232 ns | 442 ns | 0.887 | FLINT |
| `10^4` | 1.10 µs | 1.29 µs | 1.29 µs | 2.01 µs | 0.855 | FLINT |
| `10^5` | 31.9 µs | 41.5 µs | 42.5 µs | 66.1 µs | 0.770 | GMP |
| `10^6` | 451 µs | 929 µs | 937 µs | 736 µs | 0.613 | malachite |
| `10^7` | 5.69 ms | 13.5 ms | 13.5 ms | 8.42 ms | 0.675 | malachite |

- `F_n` from `n = 187` in 0.517 to 0.887 of the fastest reference's time: one product of two tables' numbers in registers up to `F_373`, then the register kernels, the first doubling in registers and the one-pass linear steps; from `10^5`, where the squares are the transforms', 0.613 to 0.770.

Lucas numbers, against GMP's `mpz_lucnum_ui` and malachite's `Natural::lucas_number`:

| n | `lucas` | GMP | malachite | ratio |
| --- | --- | --- | --- | --- |
| 100 | 10.6 ns | 10.2 ns | 30.1 ns | 1.04 |
| 185 | 14.9 ns | 11.7 ns | 39.3 ns | 1.28 |
| 300 | 15.0 ns | 21.0 ns | 49.6 ns | 0.713 |
| 370 | 25.2 ns | 24.6 ns | 57.4 ns | 1.02 |
| 500 | 26.4 ns | 37.9 ns | 84.4 ns | 0.698 |
| 740 | 36.6 ns | 47.8 ns | 92.7 ns | 0.767 |
| 1000 | 45.6 ns | 66.8 ns | 129 ns | 0.683 |
| 3000 | 181 ns | 200 ns | 373 ns | 0.907 |
| `10^4` | 1.01 µs | 1.14 µs | 1.80 µs | 0.883 |
| `10^5` | 27.8 µs | 35.9 µs | 51.5 µs | 0.775 |
| `10^6` | 363 µs | 743 µs | 592 µs | 0.614 |

- `L_n` from 300 in 0.614 to 0.907 of the fastest reference's time but at 370, as fast as GMP (1.02); at 100, 1.04 times GMP's time.

Lucas sequences, exactly: `U_n(1, −1), V_n(1, −1)` against GMP's `mpz_fib2_ui(n)` with `L_n = F_n + 2F_(n−1)`, and `(3, 1)`, `F_2n, L_2n`, against the same at `2n`; the general parameters have no reference and are given for comparison:

| n | (1, −1) | GMP | ratio | (3, 1) | GMP at 2n | ratio | (2, −1) | (5, 3) | (100003, −999331) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | 33.4 ns | 5.88 ns | 5.68 | 33.7 ns | 6.10 ns | 5.52 | 51.0 ns | 52.0 ns | 112 ns |
| 100 | 29.8 ns | 17.7 ns | 1.69 | 41.0 ns | 33.3 ns | 1.23 | 118 ns | 111 ns | 405 ns |
| 1000 | 82.9 ns | 102 ns | 0.815 | 157 ns | 192 ns | 0.818 | 285 ns | 560 ns | 8.13 µs |
| `10^4` | 1.47 µs | 1.83 µs | 0.803 | 4.43 µs | 5.38 µs | 0.824 | 4.31 µs | 10.8 µs | 164 µs |
| `10^5` | 41.7 µs | 57.3 µs | 0.729 | 94.2 µs | 162 µs | 0.580 | 87.1 µs | 186 µs | 1.84 ms |

- From `n = 1000` 0.580 to 0.824 of GMP's time; `(3, 1)` the even Fibonacci numbers' time.

Lucas sequences modulo a prime of the bits given, with an index as long: `U_n` and `V_n` against FLINT's `fmpz_lucas_chain_full` (and `fmpz_lucas_chain` for `Q = 1`), which give `V_n` and `V_(n+1)` only; FLINT does less, without `U_n`, so its times are shown and not counted:

| Bits | (1, −1) | FLINT | (3, 1) | FLINT, `Q = 1` | (1, −5) | FLINT | random P, Q | FLINT |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 64 | 419 ns | 5.35 µs | 424 ns | 602 ns | 423 ns | 6.00 µs | 463 ns | 6.36 µs |
| 128 | 1.11 µs | 11.8 µs | 1.08 µs | 2.16 µs | 2.16 µs | 13.0 µs | 2.35 µs | 14.0 µs |
| 256 | 5.27 µs | 42.8 µs | 5.09 µs | 17.0 µs | 7.87 µs | 51.3 µs | 10.4 µs | 52.2 µs |
| 512 | 35.6 µs | 158 µs | 35.5 µs | 59.7 µs | 58.8 µs | 198 µs | 72.2 µs | 205 µs |
| 1024 | 209 µs | 875 µs | 206 µs | 363 µs | 363 µs | 1.11 ms | 470 µs | 1.19 ms |
| 2048 | 1.99 ms | 5.10 ms | 1.98 ms | 2.40 ms | 3.51 ms | 6.49 ms | 4.57 ms | 6.91 ms |
| 4096 | 14.4 ms | 30.1 ms | 14.4 ms | 15.2 ms | 25.3 ms | 38.9 ms | 32.8 ms | 41.9 ms |

- 0.0704 to 0.784 of the time of FLINT's `fmpz_lucas_chain_full`, and 0.299 to 0.948 of its `fmpz_lucas_chain` for `Q = 1`, whose ladder is the third one here: FLINT reduces each product by a division, where Montgomery's form takes none, and in registers up to seven limbs.

Primorials, against GMP's `mpz_primorial_ui`, FLINT's `fmpz_primorial` (which keeps its table of primes between calls) and malachite's `Natural::primorial`:

| n | `primorial` | GMP | FLINT | malachite | ratio | fastest |
| --- | --- | --- | --- | --- | --- | --- |
| 100 | 13.6 ns | 32.1 ns | 47.3 ns | 60.5 ns | 0.424 | GMP |
| 1000 | 194 ns | 360 ns | 474 ns | 440 ns | 0.539 | GMP |
| `10^4` | 6.42 µs | 9.52 µs | 7.86 µs | 11.4 µs | 0.817 | FLINT |
| `10^5` | 185 µs | 249 µs | 219 µs | 277 µs | 0.846 | FLINT |
| `10^6` | 3.10 ms | 5.99 ms | 4.54 ms | 5.39 ms | 0.682 | FLINT |
| `10^7` | 47.2 ms | 103 ms | 68.5 ms | 76.7 ms | 0.689 | FLINT |

- 0.424 to 0.846 of the fastest reference's time: below 8192 the products of the tabulated groups of primes need no sieve, beyond the primes come from the segmented sieve, and the longest products take the transforms.

Partition numbers, against FLINT's `partitions_fmpz_ui`, which keeps `π`, the logarithms of small primes and other tables between calls, as this crate keeps its constants for the process (above), and PARI's `numbpart`; from `10^7` also FLINT with its caches cleared before each call (`flint_cleanup`), which then does more and is shown apart:

| n | `partitions` | FLINT | ratio | PARI | FLINT, caches cleared |
| --- | --- | --- | --- | --- | --- |
| 417 | 2.69 µs | 2.78 µs | 0.968 | 19.6 µs | – |
| 500 | 2.57 µs | 3.63 µs | 0.707 | 20.0 µs | – |
| 1000 | 3.59 µs | 10.6 µs | 0.338 | 26.1 µs | – |
| 2000 | 4.51 µs | 13.5 µs | 0.335 | 33.2 µs | – |
| `10^4` | 9.71 µs | 23.0 µs | 0.422 | 66.7 µs | – |
| `10^5` | 31.5 µs | 64.2 µs | 0.490 | 228 µs | – |
| `10^6` | 137 µs | 200 µs | 0.683 | 1.05 ms | – |
| `10^7` | 710 µs | 738 µs | 0.961 | 5.40 ms | 1.26 ms |
| `3·10^7` | 1.81 ms | 1.82 ms | 0.994 | 18.6 ms | 2.80 ms |
| `10^8` | 3.69 ms | 3.29 ms | 1.12 | 52.1 ms | 5.58 ms |
| `10^9` | 19.2 ms | 16.4 ms | 1.17 | – | 29.5 ms |

- From 417, the first index after the table, to `3·10^7`, 0.335 to 0.994 of FLINT's time: `A_k(n)` with the roots kept per prime, the terms of 2 to 8 limbs in registers in their own instances, `√(3 sp/(k sq))` once a call, and on numbers the reduction of `e^C` by `ln 2`, `e^(C/2^j)` by square roots and the cosines by the half- and triple-angle formulas. PARI's `numbpart` takes longer at every size (this crate 0.0708 to 0.145 of its time); FLINT with its caches cleared, 0.564 to 0.662.

Where the references are faster, this crate's time over the fastest same-work reference's above 1.05:

- **Small terms.** `fibonacci(100)` 3.70 times FLINT's time (10.8 ns against 2.92; GMP's 10.5), `lucas(185)` 1.28 times GMP's (14.9 ns against 11.7), and `lucas_sequence` at `n = 10` and 100, 1.23 to 5.68 times GMP's `mpz_fib2_ui` (33.4 ns against 5.88 for `(1, −1)` at 10): the terms come from tables up to `F_186` and `L_184`, and what is left is the result's vector, two for a Lucas sequence, which the references, writing into numbers allocated before, do not pay.
- **Partitions from `10^8`.** 1.12 and 1.17 times FLINT's time at `10^8` and `10^9`: the first term's exponential, of some `3.7 √n` bits, is the [floating numbers](float.md)' `exp` at 37 to 117 kbit, where Arb's, with the logarithms of small primes cached for its reduction, is faster.

## Examples

```rust
use a::math::numbers::{ByteOrder, Number};

fn main() {
  let o = ByteOrder::BigEndian;
  // F_100 = 354,224,848,179,261,915,075 = 0x13_33DB_76A7_C594_BFC3, L_100 = F_99 + F_101.
  let f = Number::fibonacci(100, o);
  assert_eq!(f.value, [0x13, 0x33, 0xDB, 0x76, 0xA7, 0xC5, 0x94, 0xBF, 0xC3]);
  let l = Number::lucas(100, o);
  assert_eq!(l, &Number::fibonacci(99, o) + &Number::fibonacci(101, o));
  // L_n^2 - 5 F_n^2 = 4 (-1)^n.
  assert_eq!(&(&l * &l) - &(&(&f * &f) * 5u8), Number::new_from_u8(4, o));

  // The Pell numbers U_n(2, -1) and the companion Pell numbers V_n(2, -1): 2378 and 6726.
  let (u, v) = Number::lucas_sequence(&Number::new_from_i32(2, o), &Number::new_from_i32(-1, o), 10);
  assert_eq!((u.value, v.value), (vec![0x09, 0x4A], vec![0x1A, 0x46]));

  // 30# = 2 3 5 7 11 13 17 19 23 29 = 6,469,693,230 = 0x01_819F_AF2E.
  assert_eq!(Number::primorial(30, o).value, [0x01, 0x81, 0x9F, 0xAF, 0x2E]);

  // p(100) = 190,569,292 and p(1000) = 24,061,467,864,032,622,473,692,149,727,991.
  assert_eq!(Number::partitions(100, o).value, 190_569_292u32.to_be_bytes());
  let p = Number::partitions(1000, ByteOrder::LittleEndian);
  assert_eq!(p.to_string(), "24061467864032622473692149727991");
}
```

Modulo a number, with an index of any size: Fibonacci numbers modulo a prime repeat with a period dividing `p − 1` when `p` is ±1 modulo 5, and a Lucas probable prime `n` has `U_(n+1)(P, Q) = 0 mod n` when `D = P² − 4Q` is not a square modulo `n`:

```rust
use a::math::numbers::{ByteOrder, Number};

fn main() {
  let o = ByteOrder::LittleEndian;
  let one = Number::new_from_i32(1, o);
  let minus_one = Number::new_from_i32(-1, o);
  // 10^9 + 9 is prime and 9 modulo 10: F_(k (p - 1) + 7) = F_7 = 13 modulo p.
  let p = Number::new_from_u64(1_000_000_009, o);
  let n = &(&(&p - 1u8) * &Number::new_from_u64(1 << 40, o)) + 7u8;
  let (u, v) = Number::lucas_sequence_mod(&one, &minus_one, &n, &p);
  assert_eq!(u, Number::new_from_u8(13, o));
  assert_eq!(v, Number::lucas(7, o)); // 29

  // 2^127 - 1 is prime and D = 5 is not a square modulo it.
  let m = &(&one << 127usize) - 1u8;
  let (u, _) = Number::lucas_sequence_mod(&one, &minus_one, &(&m + 1u8), &m);
  assert!(u.is_zero());
}
```

## Efficiency

- Time complexity: `F_n`, `L_n` and `lucas_sequence` in O(M(N)) for their N = 0.69n bits (`lucas_sequence`: N = n log2 max(|α|, |β|) for the roots α, β of x² − Px + Q, and the bits of `Q^n`), the time of their last squares, doubling at each step; `lucas_sequence_mod` in O(b M(m)) for an index of b bits and a modulus of m limbs, 2 to 4.5 products a bit; `n#` in O(M(N) log N) for its N = 1.44n bits, plus O(n log log n) for the sieve; `p(n)` in O(M(N) N^(1/3)) for its N = 3.7√n bits (the first term's exponential), plus about `0.25 √n` terms of falling precision
- Space complexity: the result, and twice to six times its limbs of working memory (`lucas_sequence` six buffers of the bound `M^(n+1)` of its values), on the stack up to 8 KiB; the sieves of `n#` (its segments and the primes up to `√n`) and of `p(n)`'s terms (4 bytes a term, which also keep the roots of `1 − 24n` modulo the primes)
- Soundness: every limb is written before it is read, and every byte of a result before the vector's length is set; the errors of `p(n)`'s terms are bounded as they are computed, and the result rounded only when their sum leaves less than 1/2; the inputs are left as they were
