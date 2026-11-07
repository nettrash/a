[A](../../../README.md) › [Math](../../../README.md#math)

# Modular arithmetic

[Numbers](numbers.md) solve congruences and find orders, primitive roots and discrete logarithms: `Number::crt(&[(r, m), …])` joins congruences `x = r (mod m)` by the Chinese remainder theorem, `a.multiplicative_order(&m)` gives the order of `a` modulo `m`, `Number::primitive_root(&m)` the smallest primitive root, and `h.discrete_log(&g, &m)` the smallest `k` with `g^k = h (mod m)`. The module `dlog` has the logarithm's methods apart, for a modulus and an order the caller knows: `dlog::bsgs`, `dlog::rho` and `dlog::pohlig_hellman`.

- `crt` takes any moduli, coprime or not: it gives `Some((x, l))` with `l` the least common multiple of the moduli and `0 <= x < l`, or `None` when the congruences contradict each other; `Some((0, 1))` for none. The residues may have any sign and length. In the byte order of the first residue (little-endian for none).
- `multiplicative_order` gives the smallest `k >= 1` with `a^k = 1 (mod m)`, or `None` when `a` and `m` have a common factor; modulo 1 it is 1. In the byte order of `a`.
- `primitive_root` gives the smallest `g >= 0` whose powers give every residue prime to `m`, or `None` when there is none: there is one exactly for 1, 2, 4, `p^k` and `2p^k`, `p` an odd prime. Modulo 1 it is 0, modulo 2 it is 1 and modulo 4 it is 3, as PARI's `znprimroot`. In the byte order of `m`. PARI gives the smallest root modulo a prime, and modulo its powers, but not modulo `2p^k` (11 modulo 18, where this gives 5).
- `discrete_log` gives the smallest `k >= 0` with `g^k = h (mod m)` for any `m`, `g` and `h`, `g` prime to `m` or not, or `None` when there is none; modulo 1, and for `h = 1`, it is 0. In the byte order of `h`. PARI's `znlog` takes `g` prime to `m` only, and assumes that it generates a cyclic group: modulo numbers whose units are not cyclic its answer is not always a logarithm (in 3 of 1329 random cases of the fuzzing below), and `znlog(4, Mod(2, 12))` is `[]`, where `2^2 = 4` gives 2 here.
- `dlog::bsgs(h, g, m, order)` gives the smallest `k < order` with `g^k = h (mod m)`, whatever the order of `g`; `dlog::rho(h, g, m, order)` a `k < order` for `g` of the prime order `order`, `None` when `g^order` or `h^order` is not 1 or `h` is not a power of `g`; `dlog::pohlig_hellman(h, g, m, order)` the smallest `k`, for `order` a multiple of the order of `g` given by its factorisation, `None` when `g^order` is not 1. In the byte order of `h`.
- Every function panics for a zero or negative modulus (with `the modulus must be positive`), and the `dlog` methods for a zero or negative order (`the order must be positive`), at the caller's line. Each magnitude is read in its own byte order, zero bytes above its most significant non-zero byte ignored; the operands are left as they were. None is constant-time.

## The rings and the integers

Moduli below `2^64` take paths of their own on words: their factorisation in words (see [arithmetic functions](arith.md#the-other-functions)), residues in Montgomery's form in a register, and orders and logarithms counted in words, with no allocation. Longer moduli are factored by [`factor`](factor.md), and each prime power's residues kept in the ring that fits its length: Montgomery's form in registers up to seven limbs, by the [powers](pow.md)' arithmetic (one limb, two limbs, and the coarsely integrated operand scanning from three to seven), on limbs from eight, with the powers by `pow_mod`; wrapping products modulo powers of two up to `2^128`; divisions by a prepared [divisor](div.md) modulo the others. The algorithms are written once, over the ring and over the integers they count in, words or numbers.

## The Chinese remainder theorem

Two congruences `x = r (mod m)` and `x = s (mod n)` have a solution exactly when `g = gcd(m, n)` divides `s − r`, and it is `r + m·t` with `t = (s − r)/g · (m/g)^(−1) mod n/g`, modulo `m·n/g`. The inverse is tried first: it exists exactly when the moduli are coprime, the common case, which then needs no greatest common divisor.

- Moduli of a word are joined one at a time, as Garner's method does: `x` and `l` in registers while `l` fits in 128 bits, then on limbs, each step one pass over both for their remainders by the new modulus (by its reciprocal, the [division](div.md)'s `div_2by1`), an inverse in words (Euclid's algorithm, the quotients of 1, 2 and 3 by subtractions), and one product added in; in groups of 64, whose results are joined up a tree.
- Longer moduli are joined in pairs up a balanced tree, by the greatest common divisor and the inverse on numbers.
- From 8 congruences with a long modulus, and from 150 modulo words, they are first tried as pairwise coprime by the product and remainder trees, as PARI's `ZV_chinese` and FLINT's `fmpz_multi_CRT` take them: with `P` the product of the moduli, `c_i = (P/m_i) mod m_i = (P mod m_i²)/m_i` comes down the tree of the remainders of `P` by the squares of the products, `t_i = r_i·c_i^(−1) mod m_i` at each leaf (in words for the moduli of a word), and `x = Σ t_i·P/m_i` goes up the product tree as `x_l·M_r + x_r·M_l`, reduced by `P` at the top: `O(M(n) log n)` for `n` limbs in all, against the quadratic chain. A `c_i` without an inverse shows a common factor, and the pairwise joins take over.

## Multiplicative orders

`m` is factored, and the order is the least common multiple of the orders modulo its prime powers. Modulo `2^e` it is the number of squares that bring `a` to 1. Modulo `p^e`, `p` odd, it is the order modulo `p`, in the smaller ring, among the primes of `p − 1`, which is factored too, times the power of `p` that brings `a` to that order to 1 modulo `p^e`, at most `p^(e−1)`. The order among the primes `q_1 … q_k` of a multiple `n` of it is found after Sutherland (2007): the primes are split in two halves, of products `n_1` and `n_2`, and the orders of `a^(n_2)` and of `a^(n_1)` are found among those of `n_1` and of `n_2`, down to one prime `q`, whose exponent is the number of powers by `q` that bring the residue to 1. That takes about `log2(n)·log2(k)` products against `k·log2(n)` for one power by `n/q^e` for each prime.

## Primitive roots

The candidates `g = 2, 3, …` (the odd ones modulo `2p^k`) are tried in increasing order, the cheap tests first: the Jacobi symbol `(g/p)` must be −1, which is Euler's criterion for `q = 2` without a power, by reciprocity on `p` modulo `g`; then `g^((p−1)/q)` must not be 1 for each odd prime `q` of `p − 1`, the smallest `q` first, which most candidates fail first; and for `k >= 2`, `g^(p−1)` must not be 1 modulo `p²`, which makes a root modulo `p` one modulo every `p^k`. The smallest root is small in practice (below `p^(1/4 + ε)` by Burgess's bound).

## Discrete logarithms

`m` is factored, and each prime power `p^e` says what it can of `k`:

- When `p` does not divide `g`: `h` must not be divisible by `p`, and then `k` is a logarithm modulo the order of `g` there, by Pohlig and Hellman's reduction (1978) over `p^(e−1)(p − 1)`, factored. For each prime power `q^f` of it, `g_q = g^(n/q^f)` and `h_q = h^(n/q^f)`; the order of `g_q` is `q^f'` with `f' <= f`, found by powers by `q`, and the logarithm of `h_q` to `g_q` is found a digit in base `q` at a time, each a logarithm to `γ = g_q^(q^(f'−1))`, of order `q`: by stepping through its powers for `q < 16`; for `q = p` modulo `p^e`, `e >= 2`, by a division, since the elements of order `p` are `1 + p^(e−1)·c` and `(1 + p^(e−1)·c)^x = 1 + p^(e−1)·c·x`; by Shanks's baby steps and giant steps up to `2^23`, one table for all the digits, `sqrt(q·f')` baby steps; and by Pollard's rho beyond. The logarithms modulo each `q^f'` are joined by the Chinese remainder theorem and checked by one power.
- When `p` divides `g` by `p^v`: `g^k` has the valuation `k·v` until it reaches `e`, and is 0 from `t = ceil(e/v)`; so `h = 0 (mod p^e)` asks for `k >= t`, and otherwise `k = v_p(h)/v` exactly, or there is no `k`.
- The classes are joined by the Chinese remainder theorem, their moduli the orders, which need not be coprime; an exact `k` is checked by one power modulo `m`; and the answer is the least `k` of the class that is at least every bound.

The methods:

- `bsgs`: the baby steps `g^j`, `j < s = ceil(sqrt(order))`, at most `2^21`, go into an open-addressing table of words, each a tag of the residue and the step, half full, and stop early when they come round to 1, which gives the order of `g`; the giant steps `h·g^(−s·i)` are looked up, and the first found gives the smallest `k = s·i + j`, checked by one power when the tag of a residue of more than a word matched. A `g` that is not prime to `m` has no inverse: its powers are 0 modulo the part `m_1 = gcd(m, g^b)` of `m` made of its primes from `k = t <= log2(m)`, and prime to the rest `m_2`; the `k < t` are tried one by one, and the others searched modulo `m_2`.
- `rho`: Pollard's method for logarithms (1978) with Teske's walk (2001): steps `x → x·M_i` on elements `g^a·h^b`, the multiplier `M_i = g^(u_i)·h^(v_i)` one of 32 chosen by the element's key, until a walk comes to an element met before; `g^a·h^b = g^(a')·h^(b')` then gives `k = (a' − a)/(b − b') mod q`. The elements met are recognised by keeping the distinguished ones, about one in `2^(bits/4)`, in a table (van Oorschot and Wiener, 1999), which lets four walks run side by side and share it: each step waits for a product, and four independent ones keep the multipliers busy (an order of 40 bits took 17.7 million cycles with one walk and 5.7 to 11 million with four, as the walks' luck goes). The walk is the same for every run. A collision with `b = b'` tells nothing, and a walk that has one starts again elsewhere; every collision is such when `h` is not a power of `g`, so after four of them the answer is settled by `bsgs`.
- `pohlig_hellman`: the reduction above, on the modulus as it is.

Measured on an Apple M4 Max with nothing else running, single-threaded (interleaved runs, the median of five, in cycles per call by the harness's estimate of the clock), against FLINT 3.6 and PARI 2.19. The functions that factor their modulus, or the order of the group, take [`factor`](factor.md); the long moduli below are primes `c·q + 1` with `q` prime, or have `p − 1` of known factors, so that what is timed is mostly the method itself.

The Chinese remainder theorem, against FLINT's `fmpz_CRT` and `fmpz_multi_CRT` and PARI's `chinese` and `ZV_chinese`, all with the residues reduced:

| Congruences | a | FLINT | PARI |
| --- | --- | --- | --- |
| 2 primes of 64 bits | 517 | 707 | 379 |
| 2 primes of 256 bits | 2825 | 2210 | 2405 |
| 2 primes of 1024 bits | 8829 | 8687 | 10,029 |
| 2 primes of 4096 bits | 49,873 | 60,314 | 67,306 |
| 4 primes of 64 bits | 1257 | 4020 | 1947 |
| 8 primes of 64 bits | 2659 | 10,537 | 4255 |
| 16 primes of 64 bits | 5436 | 26,579 | 9447 |
| 32 primes of 64 bits | 13,254 | 66,615 | 23,024 |
| 64 primes of 64 bits | 37,328 | 181,025 | 57,150 |
| 100 primes of 64 bits | 78,643 | 351,218 | 107,200 |
| 200 primes of 64 bits | 356,893 | 1.06 M | 288,831 |
| 1000 primes of 64 bits | 3.40 M | 15.5 M | 3.27 M |
| 10,000 primes of 64 bits | 56.6 M | 848 M | 87.0 M |
| 8 primes of 1024 bits | 122,403 | 159,364 | 136,666 |
| 100 primes of 1024 bits | 5.08 M | 10.9 M | 6.40 M |

Faster than FLINT everywhere but for two primes of 256 bits (1.28 times slower) and of 1024 bits (2 %), whose time is mostly the inverse of one modulo the other; faster than PARI from 4 to 100 words, for 10,000, and for long moduli but two of 256 bits; PARI is 1.04 to 1.24 times faster for 200 to 1000 words, by its trees, 1.36 times for two words and 1.17 times for two primes of 256 bits.

Multiplicative orders, primitive roots and logarithms, against PARI's `znorder`, `znprimroot` and `znlog`, FLINT's `n_primitive_root_prime` and `n_discrete_log_bsgs` (which searches the whole group by baby steps and giant steps):

| Modulus | `multiplicative_order`: a | PARI |
| --- | --- | --- |
| primes of 32 bits | 2615 | 2752 |
| primes of 64 bits | 20,353 | 132,763 |
| random words | 17,099 | 65,545 |
| primes 2q + 1 of 128 bits | 26,982 | 148,834 |
| primes 30q + 1 of 256 bits | 150,610 | 688,330 |
| primes 2q + 1 of 1024 bits | 3.77 M | 7.92 M |

| Prime | `primitive_root`: a | FLINT | PARI |
| --- | --- | --- | --- |
| 32 bits | 2427 | 2467 | 2858 |
| 64 bits | 17,916 | 23,670 | 64,265 |
| 2q + 1 of 128 bits | 20,642 | – | 115,735 |
| 30q + 1 of 256 bits | 157,644 | – | 628,533 |
| 2q + 1 of 1024 bits | 2.43 M | – | 6.04 M |

| Group | `discrete_log`: a | `pohlig_hellman`, `p − 1` factored | PARI | FLINT |
| --- | --- | --- | --- | --- |
| random primes of 32 bits | 24,796 | – | 375,961 | 13.2 M |
| random primes of 48 bits | 802,787 | – | 7.19 M | – |
| random primes of 56 bits | 148.1 M | – | 38.3 M | – |
| 64 bits, `p − 1` of 20-bit primes | 66,249 | 48,514 | 639,526 | – |
| 128 bits, `p − 1` of 20-bit primes | 347,665 | 220,698 | 2.88 M | – |
| 128 bits, `p − 1` of 32-bit primes | 5.43 M | 4.87 M | 94.1 M | – |
| 256 bits, `p − 1` of 24-bit primes | 6.74 M | 3.20 M | 26.1 M | – |
| 256 bits, `p − 1` of 32-bit primes | 45.9 M | 29.0 M | 320.5 M | – |
| 512 bits, `p − 1` of 28-bit primes | 173.1 M | 75.6 M | 269.4 M | – |

- Orders 2.1 to 6.5 times faster than PARI from 64 bits, and 1.05 times at 32; primitive roots 1.2 to 5.6 times faster than PARI, as fast as FLINT at 32 bits and 1.32 times faster at 64, the factorisation of `p − 1` being most of the time on words.
- Logarithms: modulo random primes of 32 and 48 bits 9 to 15 times faster than PARI; modulo smooth `p − 1`, with its factorisation by `factor`, 1.6 to 17 times, and the method itself, `pohlig_hellman`, 3.6 to 19 times. Modulo random primes of 56 bits, whose `p − 1` has a prime factor of 40 to 55 bits, PARI is 3.9 times faster: it takes an index calculus there (`Fp_log_index`), subexponential, which the square-root methods here are not.

The methods alone, in the group of prime order `q` modulo `p = 2q + 1`:

| Order | `bsgs` | `rho` | `pohlig_hellman` |
| --- | --- | --- | --- |
| 16 bits | 3718 | 3855 | 5206 |
| 20 bits | 12,572 | 19,925 | 14,379 |
| 22 bits | 24,733 | 30,932 | 26,754 |
| 24 bits | 52,050 | 69,466 | 70,990 |
| 26 bits | 115,522 | 99,796 | 101,613 |
| 28 bits | 257,425 | 195,222 | 197,381 |
| 30 bits | 630,714 | 166,562 | 170,096 |
| 40 bits | 31.0 M | 6.89 M | 6.90 M |

The table of baby steps falls out of the first-level cache from about `2^24` (`2^12` steps of 8 bytes in slots half full), where rho, with its four walks, takes over between 24 and 26 bits; `pohlig_hellman` follows the same choice.

## Examples

```rust
use a::math::numbers::{ByteOrder, Factorization, Number, dlog};

let n = |x| Number::new_from_i64(x, ByteOrder::BigEndian);

// Sunzi: 2 modulo 3, 3 modulo 5, 2 modulo 7.
assert_eq!(Number::crt(&[(n(2), n(3)), (n(3), n(5)), (n(2), n(7))]), Some((n(23), n(105))));
// Moduli with a common factor, and a contradiction.
assert_eq!(Number::crt(&[(n(1), n(4)), (n(3), n(6))]), Some((n(9), n(12))));
assert_eq!(Number::crt(&[(n(1), n(4)), (n(2), n(6))]), None);

// Orders and primitive roots.
assert_eq!(n(2).multiplicative_order(&n(7)), Some(n(3)));
assert_eq!(n(10).multiplicative_order(&n(21)), Some(n(6)));
assert_eq!(Number::primitive_root(&n(40487)), Some(n(5)));
assert_eq!(Number::primitive_root(&n(40487 * 40487)), Some(n(10)));
assert_eq!(Number::primitive_root(&n(18)), Some(n(5)));
assert_eq!(Number::primitive_root(&n(8)), None);

// Logarithms, for any base and modulus.
assert_eq!(n(13).discrete_log(&n(3), &n(17)), Some(n(4)));
assert_eq!(n(4).discrete_log(&n(2), &n(12)), Some(n(2)));
assert_eq!(n(0).discrete_log(&n(2), &n(8)), Some(n(3)));
assert_eq!(n(3).discrete_log(&n(2), &n(7)), None);

// Beyond 2^64: modulo 2^100, the units are ±5^k, and 5 has order 2^98.
let m = n(1) << 100usize;
let k = (n(1) << 90usize) + 12345u32;
let h = n(5).pow_mod(&k, &m);
assert_eq!(h.discrete_log(&n(5), &m), Some(k.clone()));
assert_eq!((&m - 1u8).discrete_log(&n(5), &m), None);

// The methods, given the modulus and the order: 5 generates the units modulo the prime
// 3 * 2^30 + 1.
let q = n(3 * (1 << 30) + 1);
let order = Factorization::new(vec![(n(2), 30), (n(3), 1)], ByteOrder::BigEndian);
let h = n(5).pow_mod(&n(1_000_000_000), &q);
assert_eq!(dlog::pohlig_hellman(&h, &n(5), &q, &order), Some(n(1_000_000_000)));
assert_eq!(dlog::bsgs(&n(22), &n(3), &n(31), &n(30)), Some(n(17)));
```

## Efficiency

- Time complexity: the factorisations of `m` and of `p − 1` for its primes `p`; then `crt` `O(k²)` limb operations for `k` moduli of a word up to 150 of them and `O(M(n) log n)` beyond, `multiplicative_order` about `log2(p)·log2(k)` products modulo each `p^e`, `primitive_root` a few powers modulo `p` for each candidate, `discrete_log` about `sqrt(q)` products for the largest prime `q` of the order of `g`, and `log2(n)` powers for the reduction
- Space complexity: the factorisations; `bsgs` `min(sqrt(order), 2^21)` words of its table, `rho` about `sqrt(q) / 2^(bits/4)` distinguished elements, and `crt` the product and remainder trees, about `2 log2(k)` times the product's length
- Soundness: the residues on limbs use the [powers](pow.md)' Montgomery kernels through their working memory plans; the operands are left as they were
