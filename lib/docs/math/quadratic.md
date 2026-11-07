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

- Words: the binary method, the factors of two of the numerator taken out (each turns the sign when the denominator is 3 or 5 modulo 8, bit 1 of `n ^ n/2`), then the larger of two odd numbers less the smaller, swapped by reciprocity (which turns the sign when both are 3 modulo 4, bit 1 of their `and`). The swap is a select, and the shift is counted on the difference before its sign is made positive, which has the same zero bits, so that a step is four instructions deep: 220 cycles for random words against 560 for the plain loop. On two limbs the same steps on the limbs apart, until one number has one limb, which then takes the other modulo it.
- Three limbs and more: Lehmer's method with the symbol carried through the quotients, after Möller ("Efficient computation of the Jacobi symbol", 2008), whose algorithm GMP's `mpn_jacobi_n` takes. Euclid's algorithm reduces one number by a multiple of the other at each step; the symbol is `(−1)^e·(x/y)` for the current pair, `y` the one that plays the denominator, which is always odd: reducing the numerator by multiples of the denominator leaves it as it is; reducing the denominator by an odd numerator first swaps their parts by reciprocity; and reducing it by an even numerator `n` takes, for each subtraction, `(n/d) = (−1)^((d−1)/2)·s(d − n, d)·(n/(d − n))`, `s` the sign of reciprocity. All of that depends only on the quotient modulo 4 and on both numbers modulo 4, a state of six bits whose 512 transitions are a table built at compile time, so the symbol follows the quotients that the [greatest common divisor](gcd.md)'s hgcd2 finds from the top two limbs, applied to the whole numbers as a matrix in one pass; the steps hgcd2 cannot take are a subtraction and a division. The whole numbers are needed only at the end.

## Square roots

- Modulo an odd prime `p`: for `p = 3 (mod 4)` the single power `a^((p+1)/4)`; otherwise, with `p − 1 = q·2^s`, Tonelli and Shanks's method: `x = a^((q+1)/2)` has `x² = a·b` for `b = a^q`, of order dividing `2^s`, and each step multiplies `x` and `b` by powers of `c = z^q`, `z` the first non-square (by the Jacobi symbol), until `b` is 1, up to `s²/2` products. For `s > 24`, Cipolla's method (1903) instead: with `w = t² − a` not a square, `(t + ω)^((p+1)/2)` in the field of `u + v·ω`, `ω² = w`, is a root, about four and a half powers whatever `s` (the prime of P-224, `s = 96`, took 68,000 cycles against 655,000 for FLINT's Tonelli–Shanks). In Montgomery's form, in the ring that fits the prime's length, all in a register below `2^64`. The root is checked.
- Modulo `p^e`: Hensel's lifting, Newton's steps `x − (x² − a)/(2x)`, each doubling the power of `p` the root is right modulo; modulo `2^e`, `a` odd: one root modulo 2, two modulo 4 for `a = 1 (mod 4)` and four modulo `2^k`, `k >= 3`, for `a = 1 (mod 8)`, `±x` and `±x + 2^(k−1)`, `x` found a bit at a time in a word, by the 2-adic Newton's steps beyond.
- `a = p^v·u` with `u` prime to `p`: no root for an odd `v < e`; for `v >= e` (`a = 0`) the multiples of `p^⌈e/2⌉`; otherwise `p^(v/2)` times the roots of `u` modulo `p^(e−v)`, each with every multiple of `d = p^(e − v/2)` added. The roots modulo `p^e` are so the residues of a few numbers `b` modulo `d`.
- Modulo `m = ∏ p^e`: the roots are the solutions of the Chinese remainder theorem for every choice of one `b` modulo each `d`, `Σ b_i·c_i mod D` with the theorem's coefficients `c_i` for `D = ∏ d`, plus every multiple of `D` below `m`. `sqrt_mod_all` forms them all; `sqrt_mod` the smallest by trying every choice, or by meeting in the middle beyond 4096 of them, as the first 26 odd primes have `2^26`: the choices split in two halves whose sums `x` and `y` are formed apart, the `y`s sorted, and for each `x` the smallest `x + y − D` with `y >= D − x`, by a binary search, or else `x + min y`.
- An odd modulus whose Jacobi symbol is −1 is answered without factoring, and `a = 0` too.

## Cornacchia's algorithm

Cornacchia (1908): a primitive solution of `x² + d·y² = m` has `x = r·y (mod m)` for a square root `r` of `−d` modulo `m`, and Euclid's algorithm on `m` and `r` stops at the first remainder `x` with `x² <= m`; `(m − x²)/d` is then `y²` when there is a solution for `r` at all. Every root `r <= m/2` is taken (`r` and `m − r` give the same solution), from `sqrt_mod_all`, so that a composite `m` has all its solutions. The first steps of Euclid's algorithm are Lehmer's, while the remainders are 128 bits above the bound, by the hgcd2 of the Jacobi symbol: every step it takes leaves both numbers above the bound. Below `2^64`, all in words.

Measured on an Apple M4 Max with nothing else running, single-threaded (interleaved runs, the median of five, in cycles per call by the harness's estimate of the clock). The Kronecker symbol of random numbers of the bits given, the bottom one odd, against GMP 6.3's `mpz_kronecker`, FLINT 3.6's `fmpz_kronecker` and PARI 2.19's `kronecker`:

| Bits | a | GMP | FLINT | PARI |
| --- | --- | --- | --- | --- |
| 64 | 222 | 218 | 215 | 613 |
| 128 | 498 | 369 | 375 | 2920 |
| 256 | 1771 | 1401 | 1409 | 8011 |
| 512 | 3760 | 3836 | 3673 | 18,835 |
| 1024 | 8049 | 9246 | 8979 | 40,970 |
| 2048 | 17,479 | 20,183 | 19,769 | 91,363 |
| 4096 | 41,069 | 44,361 | 43,469 | 217,848 |
| 16,384 | 277,350 | 270,652 | 265,135 | 1.81 M |

From 1024 to 4096 bits 1.08 to 1.15 times faster than GMP, whose method it is, as the greatest common divisor's hgcd2 is faster than GMP's; within 3 % at 64, 512 and 16,384 bits; 1.26 times slower at 256 bits and 1.35 times at 128, where the binary method on two limbs is not as quick as GMP's. Up to version 0.2.31 the binary method on limbs took 2.9 times as long at 128 bits, 3.2 times at 1024 and 16 times at 16,384.

Square roots modulo primes: `sqrt_mod`, which factors the modulus (the primality test is most of its time), and the factorisation's `sqrt_mod`, which does not, against FLINT's `fmpz_sqrtmod` and PARI's `Fp_sqrt`, which take the modulus as prime; for words also FLINT's `n_sqrtmod` (at 63 bits for the row of 64):

| Prime | a | a, factored | FLINT | PARI |
| --- | --- | --- | --- | --- |
| 32 bits | 1479 | 563 | 695 (`n_sqrtmod` 630) | 395 |
| 64 bits | 2740 | 991 | 1519 (`n_sqrtmod` 1380) | 1589 |
| 128 bits | 17,001 | 7140 | 12,899 | 13,142 |
| 256 bits | 54,438 | 16,287 | 29,842 | 40,995 |
| 1024 bits | 2.11 M | 639,978 | 924,152 | 1.18 M |
| 2^255 − 19 | 67,646 | 22,944 | 36,734 | 35,195 |
| 2^521 − 1 | 340,304 | 89,229 | 96,956 | 135,352 |
| P-224 (`s = 96`) | 87,176 | 45,424 | 456,826 | 149,740 |

Known to be prime, 1.1 to 2.5 times faster than FLINT and PARI, and 10 times FLINT's Tonelli–Shanks for P-224's prime by Cipolla's method, but for primes of 32 bits, where PARI's `Fl_sqrt` is 1.43 times faster. With the primality test, up to 3.7 times slower.

All the roots modulo words with a few prime factors, from their factorisation, against FLINT's `n_sqrtmodn`, and with the factorisation, against PARI's `Zn_sqrt`, which gives one root:

| Modulus | `sqrt_mod_all` | factored | `sqrt_mod` | FLINT, factored | PARI, one root |
| --- | --- | --- | --- | --- | --- |
| 2 primes of 32 bits | 69,611 | 1774 | 69,199 | 1915 | 366,753 |
| 4 primes of 16 bits | 27,692 | 2796 | 26,816 | 1978 | 34,095 |
| 2^20 times a prime of 40 bits | 2824 | 1812 | 2445 | 1595 | 4737 |

From the factorisation, within 1.14 to 1.41 times FLINT's, and faster for two primes of 32 bits; with the factorisation of the modulus by [`factor`](factor.md), all the roots 1.2 to 5.3 times faster than PARI's one.

Cornacchia's algorithm modulo primes the form represents, against PARI's `qfbcornacchia`, which takes the modulus as prime:

| Equation | a | a, factored | PARI |
| --- | --- | --- | --- |
| x² + y², 64 bits | 3681 | 2065 | 4117 |
| x² + y², 128 bits | 24,654 | 14,521 | 14,633 |
| x² + y², 256 bits | 75,919 | 39,668 | 39,704 |
| x² + 2y², 256 bits | 70,224 | 34,165 | 40,228 |
| x² + 7y², 1024 bits | 2.23 M | 641,311 | 850,553 |

Known to be prime, as fast as PARI's for `x² + y²` at 128 and 256 bits, and 1.18 to 2 times faster otherwise; with the primality test of `Number::cornacchia`, up to 2.6 times slower.

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
- Soundness: no `unsafe` code of its own beyond the rings of residues, which use the powers' Montgomery kernels through their working memory plans; the operands are left as they were
