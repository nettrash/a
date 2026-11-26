[A](../../../README.md) › [Math](../../../README.md#math)

# Arithmetic functions

A [factorisation](factor.md) gives the multiplicative functions of the number it factors, and [numbers](numbers.md) give them of their absolute value, by factoring it first: `divisors`, `divisor_count`, `divisor_sigma(k)`, `totient`, `carmichael`, `mobius`, `is_squarefree` and `radical`, on `Factorization` and on `Number` alike.

- `divisors` gives every positive divisor, 1 and the number included, in increasing order: `d(n)` of them, which `divisor_count` counts, the product of `e + 1` over the primes.
- `divisor_sigma(k)` is `σ_k(n)`, the sum of the `k`-th powers of the divisors: `σ_0` is `d(n)`, `σ_1` the sum of the divisors.
- `totient` is Euler's `φ(n)`, the number of integers from 1 to `n` prime to `n`; `carmichael` is Carmichael's `λ(n)`, the exponent of the group of units modulo `n`, the smallest `m >= 1` with `a^m = 1 (mod n)` for every `a` prime to `n`, which `φ(n)` is a multiple of.
- `mobius` is Möbius's `μ(n)`: 0 when the square of a prime divides `n`, otherwise `(-1)^k` for its `k` primes. `is_squarefree` is whether `μ(n)` is not 0. `radical` is the product of the distinct primes.
- On the factorisation of 1, which has no primes, `divisors` is `[1]`, and every function is 1 (`μ(1) = 1`, `λ(1) = 1`, 1 is square-free). On a `Factorization` the primes are taken as given: a composite one makes every function wrong.
- On a `Number`, the functions are those of its absolute value, as PARI gives them (PARI's `factorback` of the primes of `-12` is `-6`; `radical` is 6). Zero has no factorisation: every function of zero panics, with `zero has no factorisation`, but `is_squarefree`, which is `false` for zero, as PARI's `issquarefree(0)`; PARI's `eulerphi(0)` is 2, the units of the integers, and here `totient` of zero panics.
- Every result is in the byte order of the factorisation, or of `self`, and in the shortest form. Each magnitude is read in its own byte order, and zero bytes above its most significant non-zero byte are ignored; the number is left as it was.

## Divisors

The divisors of `n / p^e` (a sorted list `D`) give those of `n` as `S_e`, where `S_0 = D` and `S_j` is `D` merged with `p·S_(j−1)`, both sorted: each `S_j` is the divisors whose power of `p` is at most `j`, and those with a power of at least one are `p` times `S_(j−1)`. Each merge is linear, so the list comes out in order with no sorting; the primes of the largest exponents come first, while the list is still short, which keeps the merges' `e(e + 3)/2` passes cheap. Below `2^64` in words, below `2^128` in pairs of words.

Longer divisors, from 40 of them, are made in limbs, each the product of one made before it and a prime, written to the end of one buffer by the schoolbook kernels, with its logarithm in base 2, the sum of its parent's and the prime's, as a key. The divisors are sorted by the keys, floats compared as the integers of their bits, and each run of keys closer together than their rounding errors could be (such as two consecutive primes of 120 bits, whose logarithms no float tells apart) is sorted again by the limbs; then each divisor is written as a number once, in that order. Fewer divisors, whose time the buffers and the keys would pass (1.24 to 1.84 times the products of numbers for 4 to 24 of them), and primes of 23 limbs or more take numbers' products, sorted.

## The other functions

- `φ(n) = ∏ p^(e−1)(p − 1)`, `σ_k(p^e) = 1 + p^k + … + p^(ek)` by Horner's rule from `p^k + 1`, and the radical: products of a few factors each, taken in limbs, in two buffers on the stack of 46 limbs (the products below twice Karatsuba's threshold are the schoolbook kernels', which need no working memory), each prime read straight from its bytes, a word's powers in pieces of a word, and the number written once at the end. Those products one after another cost the square of their number, so a power of more than 20 of them by a word, 14 by two limbs or 6 by nine (`s²·l/2` up to about 200 for `s` products by `l` limbs), and Horner's rule of more than 12 steps for a sum that fits in 128 bits or 24 otherwise, take the numbers' way below instead, whose powers are by squaring (in limbs `σ_1(2^100)` took 2.6 times their time, `σ_1(2^300)` 7.3 times, `σ_40` of a prime of 64 bits 2.1 times). Longer products: on numbers, up to 16 factors one at a time into one vector, beyond up a balanced tree; `σ_k` then in 128 bits when `p^(ke)` fits there, and as `(p^(k(e+1)) − 1)/(p^k − 1)` by one power and an exact division for exponents above 8.
- `λ(n) = lcm λ(p^e)` with `λ(p^e) = p^(e−1)(p − 1)` for an odd prime and 1, 2 and `2^(e−2)` for 2, 4 and `2^e`, `e >= 3`; the least common multiples by the [greatest common divisor](gcd.md), up a balanced tree.
- A `Number` below `2^64` is factored in words, and its functions computed in words: trial division by the odd primes below 512, each tested by one multiplication by its inverse modulo `2^64`; then the strong tests to the bases 2, 7 and 61 below `2^32`, which no composite below 4,759,123,141 passes (Jaeschke 1993), their three powers to the one exponent taken side by side, and the [Baillie–PSW test](prime.md) above, exact below `2^64`; a perfect square split by its root; and the search for a factor that [`factor`](factor.md) makes in one limb, Hart's one line method up to 40 bits and the elliptic curve method in registers beyond. Longer numbers are factored by [`factor`](factor.md).
- `mobius` and `is_squarefree` of a word do not factor it all: the trial division stops at the first square of a prime, which more than a third of all numbers have below 512; what is left has no prime below 512, so below `512²` it is a prime, a square of its root is no square-free number, and below `512³` a composite is two distinct primes; only beyond are its factors searched for, each split checked for a common factor, a square, at once.

## Measured

Measured on 2026-11-26 on an Apple M4 Max with nothing else running, by the harness in the repository's `bench/nt` (its method in `bench/README.md`), on one thread: this crate as a user builds it, by rustc 1.100 stable in cargo's release profile (no LTO, 16 codegen units, no `target-cpu`), against FLINT 3.6.0 and PARI 2.19.1 as Homebrew builds them, called through shims compiled with `-O3 -mcpu=native`. Each figure is the median of 15 samples of at least a millisecond, in time per call, this crate and the references alternating sample by sample in one process, each sample on the same inputs for all, after a check that all give the same results. A ratio is this crate's time over that of the fastest reference: below 1, this crate is faster.

The functions of a factorisation, given, against FLINT's `fmpz_factor_euler_phi`, `fmpz_factor_divisor_sigma` (with `k = 0` for the count) and `fmpz_factor_moebius_mu`, and PARI's `eulerphi`, `numdiv`, `sumdivk`, `moebius`, `divisors` and `factorback` of the factorisation (of its primes for the radical); a factorisation of 6 random primes of 64 to 200 bits, every third one squared (144 divisors); `value` is the number the [factorisation](factor.md) makes:

| Function | a | FLINT | PARI | ratio |
| --- | --- | --- | --- | --- |
| `totient` | 162 ns | 158 ns | 370 ns | 1.02 |
| `divisor_count` | 15.6 ns | 11.8 ns | 19.7 ns | 1.32 |
| `divisor_sigma(1)` | 200 ns | 481 ns | 183 ns | 1.10 |
| `divisor_sigma(2)` | 404 ns | 764 ns | 385 ns | 1.05 |
| `carmichael` | 3.03 µs | – | – | – |
| `mobius` | 2.46 ns | 1.19 ns | 14.1 ns | 2.08 |
| `divisors` | 6.35 µs | – | 5.75 µs | 1.11 |
| `radical` | 103 ns | – | 91.7 ns | 1.13 |
| `value` | 145 ns | – | 181 ns | 0.802 |

The divisors of highly composite numbers, in words: those of the factorisation against PARI's `divisors` of it, and those of the number, which factors it first, against FLINT's `arith_divisors`, which does too:

| Number | divisors | factored | PARI | ratio | from the number | FLINT | ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 720720 | 240 | 3.20 µs | 5.13 µs | 0.623 | 3.25 µs | 5.14 µs | 0.632 |
| 2^6 3^4 5^2 7^2 11 13 17 19 23 | 10,080 | 151 µs | 435 µs | 0.347 | 165 µs | 455 µs | 0.363 |
| 897,612,484,786,617,600 | 103,680 | 1.78 ms | 7.01 ms | 0.254 | 1.82 ms | 6.29 ms | 0.289 |

The functions of random words, which factor them, the same words for all in each sample, against FLINT's `n_euler_phi`, `n_moebius_mu` and `n_is_squarefree` and PARI's `eulerphi`, `moebius`, `Z_issquarefree`, `sumdiv` and `numdiv`; and of random numbers of 128 bits, against FLINT's `fmpz_factor` followed by the function of the factorisation and PARI's `eulerphi` and `moebius` (which stops at a square factor):

| Function | a | FLINT | PARI | ratio |
| --- | --- | --- | --- | --- |
| `totient` | 5.12 µs | 9.38 µs | 24.5 µs | 0.545 |
| `mobius` | 3.30 µs | 6.50 µs | 16.8 µs | 0.508 |
| `is_squarefree` | 3.32 µs | 6.64 µs | 16.8 µs | 0.501 |
| `divisor_sigma(1)` | 5.30 µs | – | 24.2 µs | 0.219 |
| `divisor_count` | 5.18 µs | – | 23.8 µs | 0.218 |
| `carmichael` | 5.14 µs | – | – | – |
| `radical` | 5.17 µs | – | – | – |
| `divisors` | 5.83 µs | – | – | – |
| `totient`, 128 bits | 403 µs | 3.70 ms | 1.68 ms | 0.240 |
| `mobius`, 128 bits | 407 µs | 3.71 ms | 1.40 ms | 0.290 |

- The words' functions take 0.218 to 0.545 of the fastest reference's time: the factorisation in words, and for `mobius` and `is_squarefree` the stop at the first square factor and the short cuts for what is left; those of 128 bits 0.240 and 0.290. The divisors in order by merges take 0.254 to 0.632 of the reference's time.
- Of a factorisation given: `value` 0.802 of PARI's `factorback`, from the products in limbs on the stack; `totient` as fast as FLINT's `fmpz` arithmetic in place (1.02; 0.437 of PARI's time) and `divisor_sigma(2)` as PARI (1.05; 0.529 of FLINT's time); `divisor_sigma(1)` 0.416 of FLINT's time and `mobius` 0.174 of PARI's.

Where the references are faster, this crate's time over the fastest reference's above 1.05:

- **The small functions of a factorisation.** `divisor_count` 1.32 times FLINT's time (15.6 ns against 11.8), `divisor_sigma(1)` 1.10 times PARI's, `divisors` 1.11 times and `radical` 1.13 times PARI's: each result is a new number, or a vector of new numbers, where FLINT writes into an `fmpz` in place and PARI onto its stack, with no allocation for each result. `mobius`, 2.08 times FLINT's time (2.46 ns against 1.19), allocates nothing: what is left at that size is a call that the default profile may not inline across the crate and the walk over the factorisation's pairs of a number and an exponent, where FLINT's exponents are an array of their own; it was not taken apart further.

## Examples

```rust
use a::math::numbers::{ByteOrder, Factorization, Number};

let n = |x| Number::new_from_i64(x, ByteOrder::BigEndian);

// 720 = 2^4 3^2 5: 5 * 3 * 2 divisors, in increasing order.
let d = n(720).divisors();
assert_eq!(d.len(), 30);
assert_eq!((d[0].clone(), d[29].clone()), (n(1), n(720)));
assert!(d.windows(2).all(|w| w[0] < w[1]));
assert_eq!(n(720).divisor_count(), n(30));
// 28 is perfect: the sum of its divisors is twice itself.
assert_eq!(n(28).divisor_sigma(1), n(56));
assert_eq!(n(-12).divisor_sigma(2), n(1 + 4 + 9 + 16 + 36 + 144));

// The totient and Carmichael's function of 561 = 3 11 17, the first Carmichael number:
// λ = lcm(2, 10, 16) = 80 divides 560, so a^560 = 1 for every a prime to 561.
assert_eq!(n(561).totient(), n(320));
assert_eq!(n(561).carmichael(), n(80));
assert_eq!(n(2).pow_mod(&n(560), &n(561)), n(1));

assert_eq!(n(30).mobius(), -1);
assert_eq!(n(-6).mobius(), 1);
assert_eq!(n(18).mobius(), 0);
assert!(!n(18).is_squarefree() && !n(0).is_squarefree());
assert_eq!(n(-72).radical(), n(6));

// From a factorisation, with long primes: 2^127 - 1 squared times 3.
let p = (n(1) << 127usize) - 1u8;
let f = Factorization::new(vec![(p.clone(), 2), (n(3), 1)], ByteOrder::BigEndian);
assert_eq!(f.divisor_count(), n(6));
assert_eq!(f.totient(), &(&p - 1u8) * &p * 2u8);
assert_eq!(f.radical(), &p * 3u8);
assert_eq!(f.divisors()[1], n(3));
```

## Efficiency

- Time complexity: the [factorisation](factor.md) for a `Number`; then `divisors` `O(d(n))` products and comparisons (`O(d(n) log d(n))` for numbers beyond `2^128`); the others `O(k)` products and greatest common divisors of the numbers' lengths for `k` primes, the long products by balanced trees; below `2^64`, all in words
- Space complexity: the divisors and the factorisation, and for long divisors their limbs in one buffer and a key each; the products in limbs on the stack (two buffers of 46 limbs), beyond that one vector for a product of up to 16 factors
- Soundness: the products in limbs keep their buffers uninitialised (`MaybeUninit`) and read only the limbs the kernels or the reads have written, and call the multiplication's schoolbook kernels with no working memory, which their plans need none of at these lengths; the number is left as it was
