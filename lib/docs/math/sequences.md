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
- Longer terms are computed in floating point on [numbers](numbers.md), each value with a bound of its relative error. `π` and `ln 2` come from tables of 16,384 bits, `π` beyond from the Chudnovskys' series by binary splitting, its numerator and denominator cut to the bits the quotient needs, kept for the process at the longest precision computed so far (up to `2^22` bits, 512 KiB, behind a mutex) and cut from there for shorter ones; `√(24n − 1)` is the numbers' integer square root. `e^C` is computed once: reduced by `ln 2` within the table's precision (`e^C = 2^q e^r`, which saves the doublings of `C`'s integer bits), then by its series on `r 2^−s` with rectangular splitting, the blocks' divisors multiplied in and divided out once, and `s = w^(1/3)` doublings `w² + 2w` for `w` bits. For each `k = 2^j m ≥ 2`, `m` odd, whose term is on numbers, `e^(C/k)` is the `m`-th root of `e^C` by Newton's iteration for `a^(−1/m)`, which takes no division, the precision doubled at each step from the start's less the bits of the step's quadratic constant `(m + 1)/2` (without them, the steps fell short of their precision by a shortfall that doubled at each step, and `p(27 10^9)` and `p(10^11)` failed their bounds in 0.2.32), its error bounded from a final residual (or `e^(C/m)` by its own series, when that takes fewer products), then `j` square roots, each of which halves the relative error. `1/x = k/C` from one division. Cosines `cos(π a/b)` with a reduced denominator `b = 2^s 3^t q`, `q` 1 or 5, come from closed forms and the half- and triple-angle formulas, as Arb's `_arb_cos_pi_fmpq_algebraic` takes them: after the octant's reduction, `1/2`, `√2/2`, `√3/2` and `(√5 ± 1)/4` for `b` up to 6, `√((1 ± cos(2θ))/2)` for an even `b` (with `2 bits(b) + 4` more bits a level, for the cancellation of `1 − cos`), and for an odd multiple of 3 the root near its double of the cubic `4c³ − 3c = cos(3θ)` by Newton's method, with guard bits for its conditioning `12c² − 3`, which vanishes at `θ = π/3` (a denominator `3^j` has angles within `π/3^j` of it), its error bounded from the residual and checked to be that root; on an Apple M4 Max, `cos(7π/18)` to 3908 bits went from 225,000 cycles to about 20,000. Other denominators up to 512 come from Newton's method on a Chebyshev polynomial that has a simple root there (`T_(b/2)` for an even denominator, `U_(b−1)` for an odd one), from a double's start, the error bounded from the residual and Markov's inequality; the others from the versine's series.

FLINT's `partitions_fmpz_ui` does the same with Arb's ball arithmetic, which bounds every error as it computes; Johansson's paper bounds them beforehand.

## Measured

Measured on an Apple M4 Max on 2026-11-11 on a quiet machine (the one-minute load average at 1.9 to 7.0 of its 16 cores at the steps' starts and ends, falling from the harness's build just before; no process but the window server above 20 % of a CPU), single-threaded, every process with `MallocNanoZone` unset, the harness built with fat LTO and one codegen unit, in cycles per call: interleaved rounds, the contenders' order rotating each round and each round's cycles from the clock measured just before it, each cell timed in batches of calls of at least 4 µs over four consecutive indices `n` to `n + 3` in turn, so that no branch is learnt; each cell is the median of 27 rounds over three passes (21 for the Lucas sequences); this code against the previous version (0.2.32) in the same binary, and against GMP 6.3 and FLINT 3.6 writing into results allocated before. The allocator is macOS's default; with `MallocNanoZone=0` in the environment, as this machine's shells set it, each new vector costs 26 to 37 cycles more (`F_100` 62 cycles against 33, `F_1000` 226 against 189, and a vector of 9 bytes allocated, written and freed 62 against 36), which only the results of a few limbs feel. A cell's medians in the three passes lie 3.4 % apart in the median of the cells, and ratios move by a few percent between passes.

Fibonacci numbers, against GMP's `mpz_fib_ui` and FLINT's `fmpz_fib_ui`:

| n | `fibonacci` | 0.2.32 | GMP | FLINT |
| --- | --- | --- | --- | --- |
| 100 | 33 | 33 | 36 | 8 |
| 187 | 50 | 58 | 90 | 84 |
| 300 | 49 | 57 | 103 | 96 |
| 374 | 119 | 204 | 158 | 153 |
| 500 | 114 | 215 | 179 | 171 |
| 748 | 188 | 356 | 243 | 235 |
| 1000 | 189 | 433 | 319 | 316 |
| 1500 | 337 | 561 | 434 | 431 |
| 2000 | 417 | 714 | 615 | 605 |
| 3000 | 867 | 1123 | 1020 | 1012 |
| 10^4 | 4812 | 5091 | 5560 | 5579 |
| 10^5 | 142,414 | 144,295 | 185,843 | 187,704 |
| 10^6 | 1.99 M | 2.00 M | 4.17 M | 4.21 M |
| 10^7 | 24.9 M | 25.1 M | 58.3 M | 58.4 M |

Lucas numbers, against GMP's `mpz_lucnum_ui`:

| n | `lucas` | 0.2.32 | GMP |
| --- | --- | --- | --- |
| 100 | 32 | 33 | 33 |
| 185 | 53 | 99 | 48 |
| 300 | 55 | 132 | 84 |
| 370 | 100 | 171 | 106 |
| 500 | 110 | 230 | 160 |
| 740 | 157 | 308 | 206 |
| 1000 | 191 | 393 | 292 |
| 3000 | 775 | 1024 | 864 |
| 10^4 | 4211 | 4457 | 5040 |
| 10^5 | 125,818 | 124,788 | 159,334 |
| 10^6 | 1.73 M | 1.71 M | 3.53 M |

- `F_n` from `n = 187` in 0.43 to 0.87 of GMP's time; from 374 to 2000, where 0.2.32 took 1.2 to 1.5 times GMP's here, in 0.59 to 0.78 of GMP's and 0.44 to 0.60 of 0.2.32's: one product of two tables' numbers in registers up to `F_373`, then the register kernels, the first doubling in registers and the one-pass linear steps. Below, a table's number costs the allocation of its vector, which GMP and FLINT, writing into results allocated before, do not pay: about all of the 33 cycles of `F_100`, as a vector of 9 bytes allocated, written and freed takes 36.
- `L_n` from 300 in 0.49 to 0.95 of GMP's time (0.99 at 368, measured and not shown), at 185 1.12 times its time (53 cycles against 48, of which the vector's 36).
- From `10^5`, where the squares are the transforms', as 0.2.32 (0.99 to 1.01 of its time), 0.43 to 0.79 of GMP's time; the rows of 10^5 to 10^7 moved by up to 3 % between passes, either way.

Lucas sequences, exactly: `U_n(1, −1), V_n(1, −1)` against `mpz_fib2_ui(n)` and `L_n = F_n + 2F_(n−1)`, and `(3, 1)`, `F_2n, L_2n`, against the same at `2n`; the general parameters for comparison:

| n | (1, −1) | GMP | (3, 1) | 0.2.32 | GMP at 2n | (2, −1) | 0.2.32 | (5, 3) | 0.2.32 | (100003, −999331) | 0.2.32 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | 100 | 25 | 98 | 803 | 25 | 192 | 789 | 191 | 763 | 416 | 1588 |
| 100 | 98 | 71 | 147 | 1481 | 143 | 452 | 1179 | 446 | 1696 | 1685 | 6367 |
| 1000 | 322 | 416 | 654 | 4417 | 822 | 1190 | 4416 | 2433 | 8460 | 35,639 | 47,699 |
| 10^4 | 6245 | 7914 | 18,440 | 30,472 | 24,081 | 19,162 | 27,516 | 47,745 | 61,097 | 739,444 | 793,904 |
| 10^5 | 187,739 | 252,747 | 429,022 | 500,593 | 681,968 | 387,663 | 429,294 | 839,397 | 898,242 | 8.41 M | 8.73 M |

- On limbs, the general parameters take 0.24 to 0.38 of 0.2.32's time up to `n = 1000` (0.75 for the six-digit ones at 1000, where the products are long) and 0.70 to 0.96 from `10^4` (where 0.2.32's allocations are a smaller part); `(3, 1)` the even Fibonacci numbers' time, 0.63 to 0.80 of GMP's from `n = 1000`. At 10 and 100 two results' allocations are most of the time.

Lucas sequences modulo a prime of the bits given, with an index as long: `U_n` and `V_n` against FLINT's `fmpz_lucas_chain_full` (and `fmpz_lucas_chain` for `Q = 1`), which give `V_n` and `V_(n+1)` only:

| Bits | (1, −1) | FLINT | (3, 1) | FLINT | (1, −5) | FLINT | random P, Q | FLINT |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 64 | 1860 | 25,098 | 1854 | 2718 | 1863 | 27,506 | 1997 | 26,848 |
| 128 | 4962 | 53,000 | 4734 | 9290 | 9765 | 58,254 | 10,795 | 61,667 |
| 256 | 23,038 | 190,540 | 22,791 | 75,892 | 33,574 | 225,498 | 42,864 | 231,813 |
| 512 | 148,704 | 710,668 | 151,018 | 260,749 | 257,968 | 872,234 | 313,910 | 898,184 |
| 1024 | 910,520 | 3.77 M | 918,820 | 1.63 M | 1.63 M | 4.91 M | 2.09 M | 5.22 M |
| 2048 | 8.80 M | 22.6 M | 8.75 M | 10.7 M | 15.4 M | 28.6 M | 20.0 M | 30.3 M |
| 4096 | 63.6 M | 132 M | 62.9 M | 66.5 M | 112 M | 174 M | 144 M | 184 M |

- 0.07 to 0.94 of FLINT's time, with `U_n` as well as `V_n`: FLINT reduces each product by a division, where Montgomery's form takes none, and in registers up to seven limbs; for `Q = 1` FLINT's ladder is the same as the third one here, and the difference is the reduction's, smallest from 2048 bits.
- The ladders are 0.2.32's, but run on the Baillie–PSW test's Montgomery arithmetic as it is now: from 128 to 1024 bits in 0.57 to 0.87 of 0.2.32's time in the same runs, and as long at 64 bits and from 2048 bits.

Primorials, against GMP's `mpz_primorial_ui` and FLINT's `fmpz_primorial` (which keeps its table of primes between calls); the code is 0.2.32's, its primes from the segmented sieve, and took 0.96 to 1.00 of 0.2.32's time in the same runs:

| n | LE | BE | GMP | FLINT |
| --- | --- | --- | --- | --- |
| 100 | 50 | 50 | 135 | 208 |
| 1000 | 930 | 932 | 1503 | 2116 |
| 10^4 | 27,783 | 27,944 | 45,532 | 34,599 |
| 10^5 | 778,949 | 778,741 | 1.11 M | 960,712 |
| 10^6 | 13.3 M | 13.3 M | 26.4 M | 20.1 M |
| 10^7 | 207 M | 207 M | 463 M | 304 M |

- 0.45 to 0.70 of GMP's time from 1000 up, and 0.37 at 100; 0.24 to 0.81 of FLINT's: below 8192 the products of the tabulated groups of primes need no sieve, beyond the primes come from the segmented sieve, and the longest products take the transforms.

Partition numbers, against FLINT's `partitions_fmpz_ui`, which keeps `π`, the logarithms of small primes and other tables between calls, and in milliseconds also with its caches cleared before each call (`flint_cleanup`):

| n | `partitions` | 0.2.32 | FLINT |
| --- | --- | --- | --- |
| 417 | 11,300 | 18,682 | 12,502 |
| 500 | 11,040 | 18,082 | 16,306 |
| 1000 | 15,636 | 28,072 | 45,926 |
| 2000 | 19,639 | 38,747 | 60,486 |
| 10^4 | 42,859 | 87,204 | 103,117 |
| 10^5 | 141,966 | 293,701 | 290,089 |
| 10^6 | 600,552 | 1.14 M | 872,645 |
| 10^7 | 2.93 M | 5.21 M | 3.23 M |

| n | `partitions` | 0.2.32 | FLINT | FLINT, caches cleared |
| --- | --- | --- | --- | --- |
| 3 10^7 | 1.73 ms | 2.96 ms | 1.70 ms | 2.69 ms |
| 10^8 | 3.81 ms | 5.86 ms | 3.17 ms | 5.54 ms |
| 10^9 | 20.2 ms | 28.7 ms | 16.6 ms | 29.8 ms |

(The milliseconds are the wall time of a call, from runs of their own, with the `π` kept between calls: the median of 9 rounds over three passes, each round a run of 0.3 s, 1.5 s from `10^8`, for each contender.)

- 0.48 to 0.61 of 0.2.32's time up to `10^7` (0.58 to 0.70 beyond), and faster than FLINT with its caches from 417, the first index after the table, to `10^7` (0.32 to 0.91 of its time): `A_k(n)` with the roots kept per prime, the terms of 2 to 8 limbs in registers in their own instances, `√(3 sp/(k sq))` once a call, and on numbers the reduction of `e^C` by `ln 2`, `e^(C/2^j)` by square roots and the cosines by the half- and triple-angle formulas.
- At `3 10^7` as fast as FLINT with its caches (1.02 times its time), from `10^8` 1.2 times its time, and 0.64 to 0.69 of it with them cleared: there the first term's exponential, of `3.7 √n` bits, is 46 % of the time at `10^9`, and Arb's, with the logarithms of 13 small primes cached for its reduction, is about 3.6 times faster (at 3740 bits 55,700 cycles against about 200,000 here, at 11,742 bits 357,000 against 1.3 M; this code's figures, and the shares of the time, from an instrumented copy in an earlier measurement); the terms on numbers are 30 %, their cosines with odd denominators other than multiples of 3 and 5 by series or by Chebyshev's polynomials.

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
