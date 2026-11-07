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

The divisors of `n / p^e` (a sorted list `D`) give those of `n` as `S_e`, where `S_0 = D` and `S_j` is `D` merged with `p·S_(j−1)`, both sorted: each `S_j` is the divisors whose power of `p` is at most `j`, and those with a power of at least one are `p` times `S_(j−1)`. Each merge is linear, so the list comes out in order with no sorting; the primes of the largest exponents come first, while the list is still short, which keeps the merges' `e(e + 3)/2` passes cheap. Below `2^64` in words, below `2^128` in pairs of words; longer divisors are each the product of one before it and a prime, sorted once at the end, which takes fewer long products and copies than merging long numbers.

## The other functions

- `σ_k(p^e) = 1 + p^k + … + p^(ek)` by Horner's rule, in 128 bits when `p^(ke)` fits there, otherwise on numbers, and as `(p^(k(e+1)) − 1)/(p^k − 1)` by one power and an exact division for exponents above 8; `σ_1(p) = p + 1`.
- `φ(n) = ∏ p^(e−1)(p − 1)`, `λ(n) = lcm λ(p^e)` with `λ(p^e) = p^(e−1)(p − 1)` for an odd prime and 1, 2 and `2^(e−2)` for 2, 4 and `2^e`, `e >= 3`; the least common multiples by the [greatest common divisor](gcd.md), up a balanced tree.
- Products: up to 16 factors one at a time into one vector of the product's length, allocated once; beyond, up a balanced tree, whose products have operands of about equal lengths, which the [multiplication](mul.md)'s fast methods want.
- A `Number` below `2^64` is factored in words, and its functions computed in words: trial division by the odd primes below 512, each tested by one multiplication by its inverse modulo `2^64`; then the strong tests to the bases 2, 7 and 61 below `2^32`, which no composite below 4,759,123,141 passes (Jaeschke 1993), and the [Baillie–PSW test](prime.md) above, exact below `2^64`; a perfect square split by its root; and the search for a factor that [`factor`](factor.md) makes in one limb, Hart's one line method up to 40 bits and the elliptic curve method in registers beyond. Longer numbers are factored by [`factor`](factor.md).

Measured on an Apple M4 Max with nothing else running, single-threaded (interleaved runs, the median of five, in cycles per call by the harness's estimate of the clock), against FLINT 3.6 (`fmpz_factor_euler_phi`, `fmpz_factor_moebius_mu`, `fmpz_factor_divisor_sigma`) and PARI 2.19 (`eulerphi`, `numdiv`, `sigma`, `moebius`, `divisors`, `factorback`) given the same factorisations, for 16 numbers of 6 random primes of 64 to 200 bits, every third one squared (144 divisors):

| Function | a | FLINT | PARI |
| --- | --- | --- | --- |
| `totient` | 1079 | 385 | 896 |
| `divisor_count` | 52 | – | 54 |
| `divisor_sigma(1)` | 926 | 1298 | 466 |
| `divisor_sigma(2)` | 2057 | 2086 | 998 |
| `carmichael` | 7697 | – | – |
| `mobius` | 3 | 6 | 25 |
| `divisors` | 25,300 | – | 12,468 |
| `radical` | 289 | – | 243 |

These are a few products of long numbers each: FLINT's `fmpz` arithmetic in place and PARI's stack take fewer allocations than numbers in vectors of their own, up to 2.8 times less time; `divisor_sigma(1)` is 1.4 times faster than FLINT. PARI has no function for λ but `znstar`, which builds the generators too (308 million cycles here).

The divisors of highly composite numbers, in words, against PARI's `divisors` of the factorisation and FLINT's `arith_divisors` (which factors the number; the factorisation is a few hundred cycles of that):

| Number | Divisors | a, factored | a, from the number | PARI | FLINT |
| --- | --- | --- | --- | --- | --- |
| 720720 | 240 | 11,776 | 11,907 | 15,591 | 14,638 |
| 2^6 3^4 5^2 7^2 11 13 17 19 23 | 10,080 | 451,129 | 457,591 | 1.28 M | 1.29 M |
| 897,612,484,786,617,600 | 103,680 | 5.09 M | 5.22 M | 17.8 M | 18.0 M |

From 1.2 to 3.5 times faster: the merges in order against generating and sorting (PARI's `divisors`), and words against `fmpz`s.

The functions of random words, which factor them, against FLINT's `n_euler_phi`, `n_moebius_mu` and `n_is_squarefree`: `totient` 12,462 cycles against 15,124, `mobius` 12,434 against 7859 and `is_squarefree` 12,536 against 7862, the time of the factorisation of a word by `factor`'s methods for one limb, where FLINT's `n_moebius_mu` and `n_is_squarefree` stop at the first square factor.

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
- Space complexity: the divisors and the factorisation; the products' vectors, one for a product of up to 16 factors
- Soundness: no `unsafe` code of its own; the number is left as it was
