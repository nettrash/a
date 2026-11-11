[A](../../../README.md) › [Math](../../../README.md#math)

# Squares modulo a number

[Numbers](numbers.md) tell squares modulo a number and find their roots: `a.jacobi(&n)` gives the Jacobi symbol for an odd positive `n`, `a.kronecker(&n)` the Kronecker symbol for any `n`, `a.sqrt_mod(&m)` the smallest square root of `a` modulo any positive `m`, `a.sqrt_mod_all(&m)` all of them, and `Number::cornacchia(&d, &m)` the primitive solutions of `x² + d·y² = m`. A [factorisation](factor.md) of the modulus gives the same without factoring it: `f.sqrt_mod(&a)`, `f.sqrt_mod_all(&a)` and `f.cornacchia(&d)`.

- `jacobi` gives 0 when `a` and `n` have a common factor, otherwise 1 or −1. For a prime `n` it is the Legendre symbol: 1 for a non-zero square modulo `n`, −1 for a non-square. For a composite `n`, 1 does not make `a` a square, though −1 still rules it out. It panics, with `the modulus must be odd and positive`, for an even, zero or negative `n`, at the caller's line.
- `kronecker` extends it to every `n`, as Cohen's algorithm 1.4.10 and PARI, GMP and FLINT do: `(a/2)` is 0 for an even `a`, 1 for `a = ±1 (mod 8)` and −1 for `a = ±3 (mod 8)`; `(a/−1)` is −1 for a negative `a` and 1 otherwise; `(a/0)` is 1 for `a = ±1` and 0 otherwise.
- `sqrt_mod` gives the smallest `r >= 0` with `r² = a (mod m)`, or `None` when `a` is not a square modulo `m`; zero is its own root, and modulo 1 every number is 0. Up to version 0.2.31 it took an odd prime modulus only and gave the smaller of the two roots; it is the same for an odd prime.
- `sqrt_mod_all` gives every root in `0..m`, in increasing order, none when there is none. A square that shares factors with `m` may have many: 0 modulo `p^e` has `p^⌊e/2⌋` of them, which it lists all of.
- `cornacchia` gives the solutions with `x, y >= 0` and `gcd(x, y) = 1` in increasing order of `x`, none when there is none, for any `d >= 1` and `m >= 1`: for `d = 1` both `(x, y)` and `(y, x)`. PARI's `qfbcornacchia` takes a prime `m` (or four times one) and gives one solution; its `qfbsolve` gives one solution of each orbit of the form's automorphisms.
- `sqrt_mod`, `sqrt_mod_all` and `cornacchia` factor the modulus, by [`factor`](factor.md): for a prime modulus, the [primality test](prime.md) takes most of the time; the forms on a `Factorization` take the primes as given (a composite one gives wrong roots or none). They panic for a zero or negative modulus (`the modulus must be positive`), and `cornacchia` for a zero or negative `d` (`d must be positive`).
- The results are in the byte order of `self` (of `a` for a factorisation's roots, of `m` or of the factorisation for `cornacchia`), in the shortest form. Each magnitude is read in its own byte order, and zero bytes above its most significant non-zero byte are ignored; the operands are left as they were. Built on [`pow_mod`](pow.md), none of them is constant-time, and they are not meant for secret values.

## The Jacobi and Kronecker symbols

The Kronecker symbol takes the factors of two of `n` and its sign out, by the rules above, and is then the Jacobi symbol.

- Words: the binary method, the factors of two of the numerator taken out (each turns the sign when the denominator is 3 or 5 modulo 8, bit 1 of `n ^ n/2`), then the larger of two odd numbers less the smaller, swapped by reciprocity (which turns the sign when both are 3 modulo 4, bit 1 of their `and`). The swap is a select, and the shift is counted on the difference before its sign is made positive, which has the same zero bits, so that a step is four instructions deep.
- Two limbs to six: the same steps on all the limbs at once, until both numbers fit in one limb less, then on one less, down to a word. The magnitude of a negative difference is the complement of its limbs above the lowest and the negation of the lowest (which is not zero but once in `2^64`, taken apart), so the steps do not wait for a carry across the limbs; on aarch64 they are in assembly, the greatest common divisor's `gcd_22` steps widened to the numbers' limbs, with the sign carried beside them, off the critical path (on fresh numbers of three and four limbs 2.4 and 3 times as fast as the same steps in Rust). Each step reduces about 1.4 bits in 7 to 10 cycles, which beats Lehmer's steps up to six limbs.
- Seven limbs and more: Lehmer's method with the symbol carried through the quotients, after Möller ("Efficient computation of the Jacobi symbol", 2008), whose algorithm GMP's `mpn_jacobi_n` takes. Euclid's algorithm reduces one number by a multiple of the other at each step; the symbol is `(−1)^e·(x/y)` for the current pair, `y` the one that plays the denominator, which is always odd: reducing the numerator by multiples of the denominator leaves it as it is; reducing the denominator by an odd numerator first swaps their parts by reciprocity; and reducing it by an even numerator `n` takes, for each subtraction, `(n/d) = (−1)^((d−1)/2)·s(d − n, d)·(n/(d − n))`, `s` the sign of reciprocity. All of that depends only on the quotient modulo 4 and on both numbers modulo 4, a state of six bits whose 512 transitions are a table built at compile time, so the symbol follows the quotients that the [greatest common divisor](gcd.md)'s hgcd2 finds from the top two limbs, applied to the whole numbers as a matrix in one pass; the steps hgcd2 cannot take are a subtraction and a division. The quotients below 8 are three conditional subtractions, whatever the size of the divisor, and the larger ones the hardware's division of the top 64 bits, corrected by at most two subtractions. When the numbers are down to six limbs, the binary steps take them.
- Up to 16 limbs the numbers are read into limbs on the stack, and a numerator below twice the modulus is reduced by one subtraction; longer ones are in vectors.

## Square roots

- Modulo an odd prime `p`: for `p = 3 (mod 4)` the single power `a^((p+1)/4)`; for `p = 5 (mod 8)` Atkin's method (1992), one power too: with `v = (2a)^((p−5)/8)` and `i = 2a·v²`, a square root of −1 when `a` is a square, `a·v·(i − 1)` is a root; otherwise, with `p − 1 = q·2^s`, Tonelli and Shanks's method: `x = a^((q+1)/2)` has `x² = a·b` for `b = a^q`, of order dividing `2^s`, and each step multiplies `x` and `b` by powers of `c = z^q`, `z` the first non-square, until `b` is 1, up to `s²/2` products. The powers `a^((q−1)/2)` and `z^((q−1)/2)` have the same exponent and are taken side by side, the processor overlapping their products, and `c = z·(z^((q−1)/2))²`; below `2^64` the non-square is found by reciprocity, `(z/p) = (p/z)`, from `p mod z` and the squares modulo the primes below 64. For `s > 24`, Cipolla's method (1903) instead: with `w = t² − a` not a square, `(t + ω)^((p+1)/2)` in the field of `u + v·ω`, `ω² = w`, is a root, about four and a half powers whatever `s`. In Montgomery's form, in the ring that fits the prime's length, the exponents in limbs; below `2^64` all in a register, the powers from the low bit up, in pairs of bits, so that the chain of squares is the only one on the critical path. The root is checked.
- Modulo `p^e`: Hensel's lifting, Newton's steps `x − (x² − a)/(2x)`, each doubling the power of `p` the root is right modulo; modulo `2^e`, `a` odd: one root modulo 2, two modulo 4 for `a = 1 (mod 4)` and four modulo `2^k`, `k >= 3`, for `a = 1 (mod 8)`, `±x` and `±x + 2^(k−1)`, `x` found a bit at a time in a word, by the 2-adic Newton's steps beyond.
- `a = p^v·u` with `u` prime to `p`: no root for an odd `v < e`; for `v >= e` (`a = 0`) the multiples of `p^⌈e/2⌉`; otherwise `p^(v/2)` times the roots of `u` modulo `p^(e−v)`, each with every multiple of `d = p^(e − v/2)` added. The roots modulo `p^e` are so the residues of a few numbers `b` modulo `d`.
- Modulo `m = ∏ p^e`: the roots are the solutions of the Chinese remainder theorem for every choice of one `b` modulo each `d`, `Σ b_i·c_i mod D` with the theorem's coefficients `c_i` for `D = ∏ d`, plus every multiple of `D` below `m`. `sqrt_mod_all` forms them all; `sqrt_mod` the smallest by trying every choice, or by meeting in the middle beyond 4096 of them, as the first 26 odd primes have `2^26`: the choices split in two halves whose sums `x` and `y` are formed apart, the `y`s sorted, and for each `x` the smallest `x + y − D` with `y >= D − x`, by a binary search, or else `x + min y`. Below `2^64` without a vector: the parts in an array, each a few roots, and the sums of the terms `b_i·c_i` depth first, one addition modulo `D` each.
- An odd modulus whose Jacobi symbol is −1 is answered without factoring, and `a = 0` too.

## Cornacchia's algorithm

Cornacchia (1908): a primitive solution of `x² + d·y² = m` has `x = r·y (mod m)` for a square root `r` of `−d` modulo `m`, and Euclid's algorithm on `m` and `r` stops at the first remainder `x` with `x² <= m`; `(m − x²)/d` is then `y²` when there is a solution for `r` at all. Every root `r <= m/2` is taken (`r` and `m − r` give the same solution), from `sqrt_mod_all`, so that a composite `m` has all its solutions. The first steps of Euclid's algorithm are Lehmer's, while the remainders are 128 bits above the bound, by the hgcd2 of the Jacobi symbol: every step it takes leaves both numbers above the bound. Below `2^64`, all in words.

Measured on 2026-11-11 on an Apple M4 Max on the quiet machine (the load average at 3.4 to 5.2 of its 16 cores, from the system's own services; the clock at 4.45 to 4.51 GHz; one thread, PARI's too, every process with `MallocNanoZone` unset, under the system's default allocator, the harness built with fat LTO and one codegen unit), in cycles per call by the harness's estimate of the clock: interleaved with the previous version (0.2.32) and the references, in three passes of seven rounds, each cell the median of the 21, each round on fresh inputs of each contender's own, in an order that turns with the round. The Kronecker symbol of random numbers of the bits given, the bottom one odd, against GMP 6.3's `mpz_kronecker` and FLINT 3.6's `fmpz_kronecker`, first on pools of up to 256 inputs that each measurement goes over again and again, as the earlier measurements did, then on pools of up to 16,384, too many for the processor's branch predictors to learn:

| Bits | a | 0.2.32 | GMP | FLINT |
| --- | --- | --- | --- | --- |
| 64 | 272 | 272 | 312 | 314 |
| 128 | 643 | 865 | 656 | 662 |
| 192 | 1252 | 2190 | 1774 | 1766 |
| 256 | 1715 | 2968 | 2630 | 2779 |
| 384 | 2827 | 4761 | 4665 | 4874 |
| 512 | 4753 | 6662 | 6959 | 7400 |
| 640 | 6515 | 8722 | 9623 | 9792 |
| 768 | 8452 | 11,008 | 11,956 | 12,071 |
| 1024 | 12,427 | 15,023 | 16,359 | 16,596 |
| 2048 | 30,530 | 32,406 | 35,010 | 35,723 |
| 4096 | 72,089 | 72,787 | 77,152 | 78,309 |
| 16,384 | 470,010 | 469,889 | 465,040 | 466,434 |

On fresh inputs, in the same passes:

| Bits | a | 0.2.32 | GMP | FLINT |
| --- | --- | --- | --- | --- |
| 64 | 295 | 291 | 324 | 331 |
| 128 | 670 | 875 | 989 | 1000 |
| 192 | 1263 | 2525 | 2495 | 2508 |
| 256 | 1711 | 3469 | 3566 | 3565 |
| 384 | 2837 | 5429 | 5807 | 5749 |
| 512 | 5099 | 7365 | 7998 | 8009 |
| 1024 | 13,115 | 15,594 | 17,090 | 17,048 |
| 4096 | 73,163 | 74,493 | 80,376 | 80,355 |
| 16,384 | 468,461 | 476,204 | 476,523 | 476,436 |

Faster than GMP and FLINT at every size but 16,384 bits on the inputs gone over again (1 % behind both there, and 0.98 of their time on fresh inputs, where all are Lehmer's method): 0.61 to 0.76 of the best one's time from 192 to 1024 bits, where the binary steps in assembly replaced Lehmer's up to six limbs and take its numbers below them (1.21 to 1.75 times as fast as the previous version), and at 128 bits 0.98 of GMP's time (1.35 times as fast as the previous version). On fresh inputs GMP's and FLINT's branches, which their predictors no longer learn, cost them up to a half more (1.28 to 1.51 times their time on the inputs gone over again from 128 to 256 bits, 1.03 and 1.04 at 1024): 0.48 to 0.51 of their time from 192 to 384 bits, 0.68 at 128, 0.64 to 0.77 from 512 to 1024.

Square roots modulo primes, the factorisation's `sqrt_mod`, which does not factor the modulus, and `sqrt_mod`, which does (the primality test is most of its time), against FLINT's `fmpz_sqrtmod` and PARI's `Fp_sqrt`, which take the modulus as prime; for words also FLINT's `n_sqrtmod` (at 63 bits for the row of 64):

| Prime | a, factored | 0.2.32, factored | a | 0.2.32 | FLINT | PARI |
| --- | --- | --- | --- | --- | --- | --- |
| 32 bits | 553 | 892 | 1503 | 2658 | 1304 (`n_sqrtmod` 1275) | 660 |
| 64 bits | 1009 | 1664 | 4308 | 4926 | 3219 (`n_sqrtmod` 2587) | 2781 |
| 128 bits | 7905 | 11,276 | 27,302 | 33,164 | 21,487 | 21,853 |
| 256 bits | 22,022 | 35,338 | 86,287 | 123,396 | 65,840 | 81,901 |
| 1024 bits | 618,185 | 987,307 | 2.30 M | 3.34 M | 1.39 M | 1.90 M |
| 2^255 − 19 | 20,615 | 35,750 | 75,204 | 110,805 | 57,779 | 57,621 |
| 2^521 − 1 | 91,574 | 145,975 | 342,410 | 554,487 | 157,984 | 218,804 |
| P-224 (`s = 96`) | 61,760 | 82,796 | 115,877 | 152,837 | 768,049 | 250,674 |

Known to be prime, faster than FLINT and PARI at every size, 0.25 to 0.58 of the faster one's time from 64 bits on, and PARI's `Fl_sqrt` at 32 bits now too (0.84 of its time; 1.35 times slower before): the powers of a word from the low bit up, Atkin's single power for `p = 5 (mod 8)` (2^255 − 19 1.73 times faster than before, the other primes 1.34 to 1.65 times), Tonelli and Shanks's two powers side by side, and no vector. With the primality test of the unfactored form, 1.14 to 1.77 times faster than before (1.77 at 32 bits), its three strong tests' powers taken side by side.

All the roots modulo words with a few prime factors, from their factorisation, against FLINT's `n_sqrtmodn`, and with the factorisation, against PARI's `Zn_sqrt`, which gives one root:

| Modulus | factored | 0.2.32 | `sqrt_mod_all` | `sqrt_mod` | FLINT, factored | PARI, one root |
| --- | --- | --- | --- | --- | --- | --- |
| 2 primes of 32 bits | 2289 | 3536 | 140,752 | 139,330 | 3485 | 654,942 |
| 4 primes of 16 bits | 3560 | 4940 | 40,875 | 39,912 | 3743 | 58,175 |
| 2^20 times a prime of 40 bits | 2142 | 3021 | 4168 | 3660 | 2750 | 9134 |

From the factorisation, 0.66 to 0.95 of FLINT's time (before 1.01 times it for two primes of 32 bits, 1.32 and 1.10 times it for the others); with the factorisation of the modulus by [`factor`](factor.md), most of the time for two primes of 32 bits.

Cornacchia's algorithm modulo primes the form represents, against PARI's `qfbcornacchia`, which takes the modulus as prime:

| Equation | a, factored | 0.2.32, factored | a | PARI |
| --- | --- | --- | --- | --- |
| x² + y², 64 bits | 1919 | 3450 | 4934 | 6965 |
| x² + y², 128 bits | 14,815 | 21,702 | 34,402 | 25,506 |
| x² + y², 192 bits | 27,972 | 44,891 | 65,761 | 46,175 |
| x² + y², 256 bits | 43,170 | 67,522 | 106,487 | 71,479 |
| x² + 2y², 256 bits | 42,883 | 55,663 | 107,713 | 72,644 |
| x² + 7y², 1024 bits | 678,899 | 1.09 M | 2.39 M | 1.61 M |

Known to be prime, 0.28 to 0.61 of PARI's time (0.50 to 0.97 of it before); with the primality test of `Number::cornacchia`, 0.71 of PARI's time at 64 bits and 1.35 to 1.49 times slower beyond.

## Examples

```rust
use a::math::numbers::{ByteOrder, Factorization, Number};

let n = |x| Number::new_from_i64(x, ByteOrder::BigEndian);

// Squares modulo 13: 1, 4, 9, 3, 12, 10.
assert_eq!(n(10).jacobi(&n(13)), 1);
assert_eq!(n(5).jacobi(&n(13)), -1);
assert_eq!(n(26).jacobi(&n(13)), 0);
// The Kronecker symbol for even, negative and zero moduli.
assert_eq!(n(5).kronecker(&n(8)), -1);
assert_eq!(n(-3).kronecker(&n(-5)), 1);
assert_eq!(n(-1).kronecker(&n(0)), 1);

// 6^2 = 36 = 10 modulo 13, and the other root is 7.
assert_eq!(n(10).sqrt_mod(&n(13)), Some(n(6)));
assert_eq!(n(5).sqrt_mod(&n(13)), None);
// Modulo composite numbers and powers of two: the smallest root, and all of them.
assert_eq!(n(4).sqrt_mod(&n(15)), Some(n(2)));
let all: Vec<Number> = [2, 7, 8, 13].iter().map(|&x| n(x)).collect();
assert_eq!(n(4).sqrt_mod_all(&n(15)), all);
assert_eq!(n(17).sqrt_mod(&n(1 << 20)), Some(n(206_569)));
assert_eq!(n(0).sqrt_mod_all(&n(27)), [n(0), n(9), n(18)]);

// The prime of Curve25519: 2 is not a square modulo it, -1 is.
let p = (n(1) << 255usize) - 19u8;
assert_eq!(n(2).jacobi(&p), -1);
let root = (&p - 1u8).sqrt_mod(&p).unwrap();
assert_eq!((&root * &root).rem_euclid(&p), &p - 1u8);
// Known to be prime: no primality test.
let f = Factorization::new(vec![(p.clone(), 1)], ByteOrder::BigEndian);
assert_eq!(f.sqrt_mod(&(&p - 1u8)), Some(root));

// x^2 + y^2 = 65 four ways, x^2 + 2 y^2 = 1009 one.
assert_eq!(Number::cornacchia(&n(1), &n(65)), [(n(1), n(8)), (n(4), n(7)), (n(7), n(4)), (n(8), n(1))]);
assert_eq!(Number::cornacchia(&n(2), &n(1009)), [(n(19), n(18))]);
```

## Efficiency

- Time complexity: `jacobi` and `kronecker` O(n²) in the limbs, as Lehmer's greatest common divisor; `sqrt_mod` the factorisation of `m` (a [primality test](prime.md) for a prime), then about one `pow_mod` modulo each prime and `O(s²)` products for `p − 1 = q·2^s`, `s <= 24`, about four and a half beyond, `O(log e)` steps of Hensel's lifting modulo `p^e`, and the Chinese remainder theorem's sums for every choice of the roots modulo the prime powers (`O(2^(k/2)·k)` for `k` odd primes by meeting in the middle); `cornacchia` that and one Euclid's algorithm on numbers of the modulus's length for each pair of roots
- Space complexity: the vectors of the operands, of the roots modulo the prime powers and of their sums; `sqrt_mod_all` its result
- Time complexity, below `2^64` and modulo primes of a word: `sqrt_mod` and `sqrt_mod_all` with no vector but the result's, and the roots modulo a composite word one addition each
- Soundness: the binary steps of the Jacobi symbol on two to six limbs are inline assembly on 64-bit little-endian aarch64, on registers only, with the same steps in Rust elsewhere and under Miri; the rings of residues use the powers' Montgomery kernels through their working memory plans, and the remainders' scratch is on the stack for short numbers; the operands are left as they were
