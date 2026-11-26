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
- Seven limbs to 23: Lehmer's method with the symbol carried through the quotients, after Möller ("Efficient computation of the Jacobi symbol", 2008), whose algorithm GMP's `mpn_jacobi_n` takes. Euclid's algorithm reduces one number by a multiple of the other at each step; the symbol is `(−1)^e·(x/y)` for the current pair, `y` the one that plays the denominator, which is always odd: reducing the numerator by multiples of the denominator leaves it as it is; reducing the denominator by an odd numerator first swaps their parts by reciprocity; and reducing it by an even numerator `n` takes, for each subtraction, `(n/d) = (−1)^((d−1)/2)·s(d − n, d)·(n/(d − n))`, `s` the sign of reciprocity. All of that depends only on the quotient modulo 4 and on both numbers modulo 4, a state of six bits whose 512 transitions are a table built at compile time, so the symbol follows the quotients that the [greatest common divisor](gcd.md)'s hgcd2 finds from the top two limbs, applied to the whole numbers as a matrix in one pass; the steps hgcd2 cannot take are a subtraction and a division. The quotients below 8 are three conditional subtractions, whatever the size of the divisor, and the larger ones the hardware's division of the top 64 bits, corrected by at most two subtractions. When the numbers are down to six limbs, the binary steps take them.
- From 24 limbs: the same symbol carried through the [greatest common divisor](gcd.md)'s own reductions, which tell each quotient to the state: hgcd2's matrices applied by its kernels, and from 800 limbs its subquadratic half gcd of the top third of the numbers, its matrix applied to the whole numbers, over and over, `O(M(n) log n)`, as GMP's `mpn_jacobi_n` does from its gcd's threshold. Below 24 limbs Lehmer's steps are the faster (see the measurements below), which is why these start there.
- Up to 16 limbs the numbers are read into limbs on the stack, and a numerator below twice the modulus is reduced by one subtraction; longer ones are in vectors.

## Square roots

- Modulo an odd prime `p`: for `p = 3 (mod 4)` the single power `a^((p+1)/4)`; for `p = 5 (mod 8)` Atkin's method (1992), one power too: with `v = (2a)^((p−5)/8)` and `i = 2a·v²`, a square root of −1 when `a` is a square, `a·v·(i − 1)` is a root; otherwise, with `p − 1 = q·2^s`, Tonelli and Shanks's method: `x = a^((q+1)/2)` has `x² = a·b` for `b = a^q`, of order dividing `2^s`, and each step multiplies `x` and `b` by powers of `c = z^q`, `z` the first non-square, until `b` is 1, up to `s²/2` products. The powers `a^((q−1)/2)` and `z^((q−1)/2)` have the same exponent and are taken side by side, the processor overlapping their products, and `c = z·(z^((q−1)/2))²`; below `2^64` the non-square is found by reciprocity, `(z/p) = (p/z)`, from `p mod z` and the squares modulo the primes below 64. For `s > 24`, Cipolla's method (1903) instead: with `w = t² − a` not a square, `(t + ω)^((p+1)/2)` in the field of `u + v·ω`, `ω² = w`, is a root, about four and a half powers whatever `s`. In Montgomery's form, in the ring that fits the prime's length, the exponents in limbs; below `2^64` all in a register, the powers from the low bit up, in pairs of bits, so that the chain of squares is the only one on the critical path. The root is checked.
- Modulo `p^e`: Hensel's lifting, Newton's steps `x − (x² − a)/(2x)`, each doubling the power of `p` the root is right modulo; modulo `2^e`, `a` odd: one root modulo 2, two modulo 4 for `a = 1 (mod 4)` and four modulo `2^k`, `k >= 3`, for `a = 1 (mod 8)`, `±x` and `±x + 2^(k−1)`, `x` found a bit at a time in a word, by the 2-adic Newton's steps beyond.
- `a = p^v·u` with `u` prime to `p`: no root for an odd `v < e`; for `v >= e` (`a = 0`) the multiples of `p^⌈e/2⌉`; otherwise `p^(v/2)` times the roots of `u` modulo `p^(e−v)`, each with every multiple of `d = p^(e − v/2)` added. The roots modulo `p^e` are so the residues of a few numbers `b` modulo `d`.
- Modulo `m = ∏ p^e`: the roots are the solutions of the Chinese remainder theorem for every choice of one `b` modulo each `d`, `Σ b_i·c_i mod D` with the theorem's coefficients `c_i` for `D = ∏ d`, plus every multiple of `D` below `m`. `sqrt_mod_all` forms them all; `sqrt_mod` the smallest by trying every choice, or by meeting in the middle beyond 4096 of them, as the first 26 odd primes have `2^26`: the choices split in two halves whose sums `x` and `y` are formed apart, the `y`s sorted, and for each `x` the smallest `x + y − D` with `y >= D − x`, by a binary search, or else `x + min y`. Below `2^64` without a vector: the parts in an array, each a few roots, and the sums of the terms `b_i·c_i` depth first, one addition modulo `D` each.
- An odd modulus whose Jacobi symbol is −1 is answered without factoring, and `a = 0` too.

## Cornacchia's algorithm

Cornacchia (1908): a primitive solution of `x² + d·y² = m` has `x = r·y (mod m)` for a square root `r` of `−d` modulo `m`, and Euclid's algorithm on `m` and `r` stops at the first remainder `x` with `x² <= m`; `(m − x²)/d` is then `y²` when there is a solution for `r` at all. Every root `r <= m/2` is taken (`r` and `m − r` give the same solution), from `sqrt_mod_all`, so that a composite `m` has all its solutions. The first steps of Euclid's algorithm are Lehmer's, while the remainders are 128 bits above the bound, by the hgcd2 of the Jacobi symbol: every step it takes leaves both numbers above the bound. Below `2^64`, all in words.

## Measured

Measured on 2026-11-26 on an Apple M4 Max with nothing else running, by the harnesses in the repository's `bench/nt` and `bench/numbers` (their method in `bench/README.md`), on one thread: this crate as a user builds it, by rustc 1.100 stable in cargo's release profile (no LTO, 16 codegen units, no `target-cpu`); malachite 0.15.0 in the same profile; GMP 6.3.0, FLINT 3.6.0 and PARI 2.19.1 as Homebrew builds them, called through shims compiled with `-O3 -mcpu=native`. Each figure is the median of 15 samples of at least a millisecond (21 for the symbols by limbs), in time per call, this crate and every reference alternating sample by sample in one process, each sample on the same inputs for all, after a check that all give the same symbols and roots. A ratio is this crate's time over that of the fastest reference that does the same work: below 1, this crate is faster.

The Kronecker and Jacobi symbols of random numbers of the bits given, the bottom one odd, against GMP's `mpz_kronecker` and `mpz_jacobi`, FLINT's `fmpz_kronecker` and `fmpz_jacobi`, malachite's and PARI's `kronecker`, on pools of fresh inputs (16,384 pairs up to 1024 bits, 2048 to 4096 bits and 256 at 16,384), too many for the processor's branch predictors to learn:

| Bits | `kronecker` | GMP | FLINT | malachite | PARI | ratio | `jacobi` | ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 64 | 59.5 ns | 64.6 ns | 66.6 ns | 166 ns | 185 ns | 0.922 | 56.1 ns | 0.869 |
| 128 | 133 ns | 201 ns | 197 ns | 311 ns | 1.03 µs | 0.672 |  |  |
| 192 | 250 ns | 500 ns | 488 ns | 542 ns | 1.90 µs | 0.512 |  |  |
| 256 | 346 ns | 712 ns | 707 ns | 733 ns | 2.81 µs | 0.490 | 310 ns | 0.439 |
| 384 | 569 ns | 1.17 µs | 1.16 µs | 1.12 µs | 4.67 µs | 0.511 |  |  |
| 512 | 1.02 µs | 1.59 µs | 1.57 µs | 1.47 µs | 6.50 µs | 0.696 |  |  |
| 640 | 1.38 µs | 2.04 µs | 2.03 µs | 1.84 µs | 8.35 µs | 0.751 |  |  |
| 768 | 1.77 µs | 2.50 µs | 2.47 µs | 2.21 µs | 10.3 µs | 0.804 |  |  |
| 1,024 | 2.56 µs | 3.40 µs | 3.37 µs | 3.00 µs | 14.0 µs | 0.853 | 2.53 µs | 0.853 |
| 2,048 | 6.22 µs | 7.36 µs | 7.33 µs | 6.38 µs | 31.4 µs | 0.975 |  |  |
| 4,096 | 13.1 µs | 16.2 µs | 16.0 µs | 14.2 µs | 74.8 µs | 0.926 | 12.8 µs | 0.923 |
| 16,384 | 73.8 µs | 96.0 µs | 94.4 µs | 88.0 µs | 617 µs | 0.839 |  |  |

And by limbs, against GMP's `mpz_jacobi`: `jacobi`, and the gcd's reductions with the symbol carried through them alone, which `jacobi` takes from 24 limbs, both timed by a probe that compiles the crate's source, so as to reach the internal kernel:

| Limbs | `jacobi` | GMP | ratio | the steps alone | ratio |
| --- | --- | --- | --- | --- | --- |
| 8 (512 bits) | 803 ns | 1.04 µs | 0.774 | 980 ns | 0.946 |
| 16 (1,024 bits) | 2.07 µs | 2.22 µs | 0.931 | 2.15 µs | 0.950 |
| 32 (2,048 bits) | 4.69 µs | 4.91 µs | 0.954 | 4.49 µs | 0.915 |
| 64 (4,096 bits) | 10.2 µs | 10.8 µs | 0.946 | 9.98 µs | 0.923 |
| 128 (8,192 bits) | 24.0 µs | 26.9 µs | 0.894 | 23.7 µs | 0.879 |
| 256 (16,384 bits) | 64.1 µs | 74.8 µs | 0.856 | 62.8 µs | 0.838 |
| 512 (32,768 bits) | 187 µs | 222 µs | 0.843 | 184 µs | 0.828 |
| 1024 (65,536 bits) | 563 µs | 670 µs | 0.839 | 555 µs | 0.827 |
| 2048 (131,072 bits) | 1.60 ms | 2.03 ms | 0.788 | 1.58 ms | 0.780 |

- 0.439 to 0.975 of the fastest reference's time at every size: the binary steps in assembly up to six limbs, Lehmer's steps to 23, and from 24 limbs the gcd's reductions, from 800 limbs its half gcd, where GMP's `mpn_jacobi_n` takes its own from its gcd's threshold (0.788 of its time at 2048 limbs). Below 24 limbs Lehmer's steps are the faster: at 8 and 16 limbs the gcd's reductions take 980 ns and 2.15 µs where `jacobi` takes 803 ns and 2.07 µs.

Square roots modulo primes: the factorisation's `sqrt_mod`, which takes the prime as given, against FLINT's `fmpz_sqrtmod` (and `n_sqrtmod` for a word) and PARI's `Fp_sqrt`, which take the modulus as prime; and `sqrt_mod`, which tests the modulus by Baillie–PSW and factors it, against the same, which then do less: its ratio, in parentheses, is not counted:

| Prime | factored | FLINT | PARI | ratio | `sqrt_mod` | (ratio) |
| --- | --- | --- | --- | --- | --- | --- |
| 32 bits | 115 ns | 251 ns (`n_sqrtmod` 242 ns) | 136 ns | 0.847 | 299 ns | 2.19 |
| 64 bits | 210 ns | 710 ns | 579 ns | 0.362 | 820 ns | 1.42 |
| 128 bits | 1.63 µs | 4.41 µs | 4.59 µs | 0.370 | 4.59 µs | 1.02 |
| 256 bits | 4.21 µs | 11.6 µs | 14.8 µs | 0.361 | 14.5 µs | 1.25 |
| 1024 bits | 133 µs | 267 µs | 400 µs | 0.498 | 489 µs | 1.81 |
| `2^255 − 19` | 4.19 µs | 12.2 µs | 12.0 µs | 0.349 | 15.1 µs | 1.28 |
| `2^521 − 1` | 19.5 µs | 34.0 µs | 46.4 µs | 0.573 | 72.0 µs | 2.11 |
| P-224 | 13.1 µs | 157 µs | 51.4 µs | 0.255 | 23.2 µs | 0.450 |

- Known to be prime, 0.255 to 0.847 of the fastest reference's time: the powers of a word from the low bit up, Atkin's single power for `p = 5 (mod 8)` (`2^255 − 19`), Tonelli and Shanks's two powers side by side, and no vector; P-224, whose `p − 1` has `2^96`, takes Cipolla's method, 0.255 of PARI's time. With the primality test of the unfactored form, which the references do not run, 1.02 to 2.19 times their time but at P-224.

All the roots modulo words with a few prime factors: `sqrt_mod_all`, with the factorisation and factoring the modulus, against FLINT's `n_factor` and `n_sqrtmodn`; and `sqrt_mod` against PARI's `Zn_sqrt` with the factorisation and without, which gives one root, not the smallest, so it does less, and its ratio, in parentheses, is not counted:

| Modulus | `sqrt_mod_all`, factored | `sqrt_mod_all` | FLINT | ratio | `sqrt_mod`, factored | PARI, one root | (ratio) | `sqrt_mod` | PARI, one root | (ratio) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 primes of 32 bits | 367 ns | 27.6 µs | 263 µs | 0.105 | 289 ns | 448 ns | 0.645 | 28.4 µs | 133 µs | 0.214 |
| 4 primes of 16 bits | 616 ns | 8.03 µs | 2.78 µs | 2.89 | 345 ns | 537 ns | 0.643 | 7.80 µs | 11.9 µs | 0.657 |
| `2^20` times a prime of 40 bits | 392 ns | 753 ns | 720 ns | 1.05 | 266 ns | 631 ns | 0.422 | 624 ns | 1.63 µs | 0.383 |

Cornacchia's algorithm modulo primes the form represents: the factorisation's `cornacchia`, which takes the prime as given, against PARI's `qfbcornacchia`, which takes it as prime and gives one solution (of the two for `d = 1`); and `Number::cornacchia`, which tests and factors the modulus, which PARI does not, so its ratio, in parentheses, is not counted:

| Equation | factored | PARI | ratio | `cornacchia` | PARI | (ratio) |
| --- | --- | --- | --- | --- | --- | --- |
| x² + y², prime of 64 bits | 396 ns | 1.34 µs | 0.296 | 1.01 µs | 1.33 µs | 0.757 |
| x² + y², prime of 128 bits | 3.01 µs | 5.38 µs | 0.560 | 6.59 µs | 5.37 µs | 1.23 |
| x² + y², prime of 192 bits | 5.87 µs | 9.82 µs | 0.598 | 13.4 µs | 9.82 µs | 1.36 |
| x² + y², prime of 256 bits | 8.90 µs | 15.0 µs | 0.594 | 22.1 µs | 15.1 µs | 1.46 |
| x² + 2y², prime of 256 bits | 8.86 µs | 15.0 µs | 0.591 | 21.6 µs | 15.1 µs | 1.43 |
| x² + 7y², prime of 1024 bits | 155 µs | 384 µs | 0.405 | 521 µs | 389 µs | 1.34 |


- Known to be prime, 0.296 to 0.598 of PARI's time; with the primality test of `Number::cornacchia`, 0.757 of PARI's time at 64 bits and 1.23 to 1.46 times it beyond.

Where the references are faster, this crate's time over the fastest same-work reference's above 1.05:

- **All the roots modulo four primes of 16 bits.** `sqrt_mod_all` 2.89 times FLINT's `n_factor` and `n_sqrtmodn` (8.03 µs against 2.78; from the factorisation, 616 ns): the factorisation of the word, which trial-divides to 512 only and splits the rest one factor at a time, by Hart's method and the curves, where `n_factor`'s longer trial division finds the four primes at once. Modulo `2^20` times a prime of 40 bits, 1.05 times FLINT's time.

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

- Time complexity: `jacobi` and `kronecker` `O(M(n) log n)` in the limbs from 800 limbs, as the subquadratic greatest common divisor, O(n²) below; `sqrt_mod` the factorisation of `m` (a [primality test](prime.md) for a prime), then about one `pow_mod` modulo each prime and `O(s²)` products for `p − 1 = q·2^s`, `s <= 24`, about four and a half beyond, `O(log e)` steps of Hensel's lifting modulo `p^e`, and the Chinese remainder theorem's sums for every choice of the roots modulo the prime powers (`O(2^(k/2)·k)` for `k` odd primes by meeting in the middle); `cornacchia` that and one Euclid's algorithm on numbers of the modulus's length for each pair of roots
- Space complexity: the vectors of the operands, of the roots modulo the prime powers and of their sums; `sqrt_mod_all` its result
- Time complexity, below `2^64` and modulo primes of a word: `sqrt_mod` and `sqrt_mod_all` with no vector but the result's, and the roots modulo a composite word one addition each
- Soundness: the binary steps of the Jacobi symbol on two to six limbs are inline assembly on 64-bit little-endian aarch64, on registers only, with the same steps in Rust elsewhere and under Miri; the rings of residues use the powers' Montgomery kernels through their working memory plans, and the remainders' scratch is on the stack for short numbers; the operands are left as they were
