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
- `F_k` and `F_(k−1)` are doubled from the top bits of `k` that the table holds down to its last bit, with two squares a bit: `F_(2k−1) = F_k² + F_(k−1)²`, `F_(2k+1) = 4F_k² − F_(k−1)² + 2(−1)^k` and `F_(2k) = F_(2k+1) − F_(2k−1)`, the `±2` or-ed into a low limb that is 0 or 1 modulo 4, as GMP's `mpn_fib2_ui` does.
- The last step of `F_n`, `n = 2k` or `2k + 1`, is one product instead of two squares: `F_(2k) = F_k (F_k + 2F_(k−1))` and `F_(2k+1) = (2F_k + F_(k−1))(2F_k − F_(k−1)) + 2(−1)^k`; up to `n = 373`, from the table's pair in registers.
- `L_n` takes the zero bits at the bottom of `n` as squares, `L_(2k) = L_k² − 2(−1)^k`, from the table or from the odd part `2k + 1`, `L_(2k+1) = 5F_(k−1)(2F_k + F_(k−1)) − 4(−1)^k`.

The squares and products are the [multiplication](mul.md)'s: Karatsuba's and Toom-Cook's methods, and its number-theoretic transforms from about 360 limbs on aarch64. Up to 8 KiB, the working memory is on the stack.

## Lucas sequences

`lucas_sequence` doubles `U_k` and `U_(k−1)` with two squares a bit, as the Fibonacci numbers are, the identities taking `Q^k` beside them:

- `U_(2k−1) = U_k² − Q U_(k−1)²`, from `U_(m+n) = U_m U_(n+1) − Q U_(m−1) U_n`;
- `U_(2k+1) = (P² − 3Q) U_k² − Q² U_(k−1)² + 2Q^k`, from the same with `U_(k+1) = P U_k − Q U_(k−1)` and `U_k² − P U_k U_(k−1) + Q U_(k−1)² = Q^(k−1)`;
- `U_(2k) = (U_(2k+1) + Q U_(2k−1)) / P`, the recurrence, a division that is exact;
- then `V_n = P U_n − 2Q U_(n−1)`.

`Q^k` is a sign for `Q = ±1`; the first steps are taken in `i128`s while every value fits, the rest on numbers. `P = ±1, Q = −1` are the Fibonacci and Lucas numbers, taken from the doubling above (`L_n = F_n + 2F_(n−1)`), and `Q = 0` (`U_n = P^(n−1)`), `P = 0` (`U_(2j+1) = (−Q)^j`, `V_(2j) = 2(−Q)^j`, the others 0) and `P² = 4Q` (`U_n = n c^(n−1)`, `V_n = 2c^n` for `P = 2c`) have closed forms.

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

- `A_k(n)`, a sum of `φ(k)` roots of unity with Dedekind sums, is a product of at most one cosine for each prime power of `k`, a square root and a sign, by Whiteman's factorisation (1956): `A_(k1 k2)(n) = A_(k1)(n1) A_(k2)(n2)` for coprime `k1` and `k2` and residues `n1`, `n2` that congruences give, and closed forms for prime powers, which take modular square roots (Tonelli and Shanks, then Hensel's lifting) and Jacobi symbols. The smallest prime factors of every `k` come from one sieve. The factorisation was checked against Selberg's formula `A_k(n) = √(k/3) Σ (−1)^l cos(π(6l + 1)/(6k))`, over `(3l² + l)/2 ≡ −n mod k`, for every `k` below 600 and every `n`.
- The terms whose bound in double precision is small enough, most of them, are computed in doubles, without the platform's `exp` and `cos`, whose accuracy Rust does not specify: polynomials on reduced arguments with IEEE 754's correctly rounded operations, with error bounds of 16 and 12 units of the last place, and the argument's error carried through `e^x`. They are accumulated exactly in an `i128`.
- Terms of up to 1248 bits are computed in fixed point on arrays of up to 23 limbs on the stack, without allocations: `e^x = 2^q e^r` with `r = x − q ln 2` and a table of `ln 2`, `e^r − 1` from `r 2^−s` by its Taylor series and `s` doublings `w (w + 2)`; each cosine from its versine `1 − cos y`, by its series on `y 2^−s` and doublings `2v(2 − v)`; each series by rectangular splitting (Smith, 1989), its blocks cut so that the product of their divisors fits a limb, multiplications by them and one division a block. Each value carries a bound of its absolute error.
- Longer terms are computed in floating point on [numbers](numbers.md), each value with a bound of its relative error. `π` comes from a table of 16,384 bits, and beyond from the Chudnovskys' series by binary splitting, its numerator and denominator cut to the bits the quotient needs; `√(24n − 1)` is the numbers' integer square root. `e^C` is computed once, by its series on `C 2^−s` with rectangular splitting, the blocks' divisors multiplied in and divided out once, and `s` doublings `w² + 2w`, `s` the bits of `C`'s integer part and `w^(1/3)` more for `w` bits; for each `k ≥ 2` whose term is on numbers, `e^(−C/k)` is the `k`-th root of `e^C` by Newton's iteration for `a^(−1/k)`, which takes no division, its error bounded from a final residual (or `e^(C/k)` by its own series, when that takes fewer products). `1/x = k/C` from one division. Cosines of a reduced denominator up to 512 at high precision come from Newton's method on a Chebyshev polynomial that has a simple root there (`T_(b/2)` for an even denominator, `U_(b−1)` for an odd one), from a double's start, the error bounded from the residual and Markov's inequality; the others from the versine's series.

FLINT's `partitions_fmpz_ui` does the same with Arb's ball arithmetic, which bounds every error as it computes; Johansson's paper bounds them beforehand.

## Measured

Measured on an Apple M4 Max with other work running on the machine (interleaved runs, the median of 5 or 7, in cycles per call), the results in the byte order given, against GMP 6.3 and FLINT 3.6 into results allocated before, and PARI/GP 2.19, single-threaded.

Fibonacci and Lucas numbers, against GMP's `mpz_fib_ui` and `mpz_lucnum_ui` and FLINT's `fmpz_fib_ui`:

| n | `fibonacci` LE | BE | GMP | FLINT | `lucas` LE | BE | GMP |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 100 | 141 | 141 | 96 | 83 | 142 | 141 | 90 |
| 1000 | 528 | 538 | 331 | 330 | 301 | 298 | 186 |
| 10^4 | 4813 | 4830 | 5266 | 5255 | 3012 | 3016 | 3535 |
| 10^5 | 134,810 | 135,602 | 172,586 | 172,186 | 85,692 | 85,370 | 117,958 |
| 10^6 | 1.89 M | 1.89 M | 3.89 M | 3.89 M | 1.13 M | 1.13 M | 2.32 M |
| 10^7 | 23.6 M | 23.6 M | 55.1 M | 55.4 M | 14.0 M | 14.1 M | 33.5 M |

- From `n = 10^4`, 9 % less than GMP's time with Karatsuba's products, and from `10^5`, where the last squares and the product reach the [multiplication](mul.md)'s number-theoretic transforms, 0.43 to 0.78 of it (`F_(10^7)` has 868 KB).
- Below, 1.5 to 1.6 times GMP's time: the result's allocation, about 50 cycles of `malloc` and `free`, and the calls of the kernels for products of a few limbs, which GMP's assembly makes cheaper; up to `n = 373` the last step is in registers.

Lucas sequences, exactly: `U_n(1, −1), V_n(1, −1)` against `mpz_fib_ui(n)` and `mpz_lucnum_ui(n)`, and `U_n(3, 1), V_n(3, 1) = F_2n, L_2n` against them at `2n`; the Pell numbers `(2, −1)` and `(5, 3)` for comparison:

| n | (1, −1) | GMP | (3, 1) | GMP at 2n | (2, −1) | (5, 3) |
| --- | --- | --- | --- | --- | --- | --- |
| 100 | 179 | 106 | 1910 | 148 | 1169 | 2267 |
| 1000 | 687 | 506 | 5188 | 958 | 5344 | 9345 |
| 10^4 | 6206 | 8818 | 29,593 | 27,259 | 27,678 | 58,837 |
| 10^5 | 174,232 | 286,820 | 474,698 | 781,085 | 408,349 | 854,524 |
| 10^6 | 2.32 M | 6.24 M | 5.58 M | 13.4 M | 5.28 M | 10.5 M |

- From `n = 10^5`, 0.37 to 0.61 of GMP's time for the same numbers; below `10^4` the general parameters take operations on numbers, each with its allocation, and are 5 to 13 times GMP's specialised functions; `(1, −1)` takes the Fibonacci numbers' doubling, 1.4 to 1.7 times GMP's two calls below `10^4`.

Lucas sequences modulo a prime of the bits given, with an index as long: `U_n` and `V_n` against FLINT's `fmpz_lucas_chain_full` (and `fmpz_lucas_chain` for `Q = 1`), which give `V_n` and `V_(n+1)` only:

| Bits | (1, −1) | FLINT | (3, 1) | FLINT | (1, −5) | FLINT | random P, Q | FLINT |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 64 | 1958 | 20,994 | 1928 | 2797 | 1910 | 21,263 | 2074 | 22,350 |
| 128 | 5660 | 41,308 | 5486 | 8269 | 11,629 | 48,541 | 12,704 | 55,721 |
| 256 | 28,890 | 154,589 | 28,541 | 60,126 | 47,153 | 181,390 | 61,266 | 222,357 |
| 512 | 233,989 | 568,461 | 223,855 | 247,360 | 392,514 | 706,419 | 487,662 | 872,646 |
| 1024 | 1.19 M | 3.09 M | 1.17 M | 1.49 M | 2.01 M | 3.61 M | 3.71 M | 5.02 M |
| 2048 | 8.32 M | 18.2 M | 8.15 M | 9.84 M | 14.4 M | 21.4 M | 18.5 M | 28.8 M |
| 4096 | 58.9 M | 108 M | 58.6 M | 62.4 M | 103 M | 123 M | 135 M | 184 M |

- 0.09 to 0.94 of FLINT's time, with `U_n` as well as `V_n`: FLINT reduces each product by a division, where Montgomery's form takes none, and in registers up to seven limbs; for `Q = 1` FLINT's ladder is the same as the third one here, and the difference is the reduction's, smallest from 2048 bits.

Primorials, against GMP's `mpz_primorial_ui` and FLINT's `fmpz_primorial` (which keeps its table of primes between calls), measured again with nothing else running once the primes came from the segmented sieve:

| n | LE | BE | GMP | FLINT |
| --- | --- | --- | --- | --- |
| 100 | 145 | 143 | 144 | 220 |
| 1000 | 1033 | 1049 | 1547 | 2194 |
| 10^4 | 28,691 | 28,753 | 41,976 | 35,124 |
| 10^5 | 785,649 | 786,407 | 1.11 M | 958,736 |
| 10^6 | 13.6 M | 13.4 M | 26.7 M | 19.9 M |
| 10^7 | 207 M | 207 M | 458 M | 304 M |

- 0.45 to 0.71 of GMP's time from 1000 up, and the same at 100; 0.47 to 0.82 of FLINT's: below 8192 the products of the tabulated groups of primes need no sieve, beyond the primes come from the segmented sieve, 4 to 12 % faster than the factorials' sieve they first came from, and the longest products take the transforms.

Partition numbers, against FLINT's `partitions_fmpz_ui`, which keeps `π` and other constants and tables between calls, also with its caches cleared before each call (`flint_cleanup`), and PARI's `numbpart`, in microseconds:

| n | `partitions` | FLINT | FLINT, caches cleared | PARI |
| --- | --- | --- | --- | --- |
| 1000 | 7.31 | 10.9 | 75.7 | 32 |
| 10^4 | 21.0 | 22.9 | 85.8 | 66 |
| 10^5 | 80.7 | 82.6 | 147 | 300 |
| 10^6 | 283 | 209 | 275 | 1100 |
| 10^7 | 1218 | 729 | 1254 | 6000 |
| 10^8 | 6402 | 3409 | 5977 | 56,600 |
| 10^9 | 31,230 | 17,880 | 31,980 | 1,041,000 |

- Up to `10^5`, as fast as FLINT or faster (0.67 to 0.98 of its time): most terms are in doubles, and the rest on limbs on the stack, without allocations.
- From `10^6`, 1.35 to 1.9 times FLINT's time with its constants kept, and as fast as FLINT computing them again (0.97 to 1.07): at `10^9`, `π` to 117,000 bits takes 2.4 ms here, `e^C` 10.8 ms (Arb's `arb_exp`, with its caches cleared, 16.4 ms), and the 94 terms on numbers about 9 ms, half of it their cosines.
- 4 to 33 times faster than PARI.

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
- Space complexity: the result, and twice to five times its limbs of working memory, on the stack up to 8 KiB; the sieves of `n#` (its segments and the primes up to `√n`) and of `p(n)`'s terms (4 bytes a term)
- Soundness: every limb is written before it is read, and every byte of a result before the vector's length is set; the errors of `p(n)`'s terms are bounded as they are computed, and the result rounded only when their sum leaves less than 1/2; the inputs are left as they were
