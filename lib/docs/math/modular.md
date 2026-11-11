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

Moduli below `2^64` take paths of their own on words: their factorisation in words (see [arithmetic functions](arith.md#the-other-functions)), residues in Montgomery's form in a register, and orders and logarithms counted in words, with no allocation. Longer moduli are factored by [`factor`](factor.md), and each prime power's residues kept in the ring that fits its length: Montgomery's form in registers up to seven limbs, and eight for the discrete logarithms, whose steps and walks take millions of products, by the [powers](pow.md)' arithmetic (one limb, two limbs, and the coarsely integrated operand scanning from three, in inline assembly on aarch64), on limbs beyond, with the powers by `pow_mod` (on aarch64 up to sixteen limbs by the powers' kernels of residues in memory); wrapping products modulo powers of two up to `2^128`; divisions by a prepared [divisor](div.md) modulo the others. The algorithms are written once, over the ring and over the integers they count in, words or numbers.

## The Chinese remainder theorem

Two congruences `x = r (mod m)` and `x = s (mod n)` have a solution exactly when `g = gcd(m, n)` divides `s − r`, and it is `r + m·t` with `t = (s − r)/g · (m/g)^(−1) mod n/g`, modulo `m·n/g`. The inverse is tried first: it exists exactly when the moduli are coprime, the common case, which then needs no greatest common divisor.

- Up to 8 congruences modulo words: in registers while the least common multiple fits in 128 bits, each step a remainder of `x` and `l` by the new modulus and a product modulo it by its reciprocal (the [division](div.md)'s `div_2by1`) and an inverse in words (Euclid's algorithm, the quotients of 1, 2 and 3 by subtractions), with no vector; past 128 bits, on limbs from there, as below.
- Two congruences whose moduli have at most 16 limbs: on the stack, the residues reduced and the products and remainders in limbs by the schoolbook kernels, with the inverse of one modulus modulo the other by the [greatest common divisor](gcd.md)'s `mod_inverse`, the only number besides the two results. Moduli with a common factor take the joins below.
- More moduli of a word are joined one at a time, as Garner's method does: `x` and `l` in registers while `l` fits in 128 bits, then on limbs, each step one pass over both for their remainders by the new modulus and one product added in; in groups of 64, whose results are joined up a tree.
- Longer moduli are joined in pairs up a balanced tree, by the greatest common divisor and the inverse on numbers.
- From 8 congruences with a long modulus, and from 150 modulo words, they are first tried as pairwise coprime by the product and remainder trees, as PARI's `ZV_chinese` and FLINT's `fmpz_multi_CRT` take them: with `P` the product of the moduli, `c_i = (P/m_i) mod m_i = (P mod m_i²)/m_i` comes down the tree of the remainders of `P` by the squares of the products, `t_i = r_i·c_i^(−1) mod m_i` at each leaf, and `x = Σ t_i·P/m_i` goes up the product tree as `x_l·M_r + x_r·M_l`, reduced by `P` at the top: `O(M(n) log n)` for `n` limbs in all, against the quadratic chain. For moduli of a word the trees are in limbs, a buffer a level, each node on 16 bytes for the kernels, and the leaves in words: no number but the two results. A `c_i` without an inverse shows a common factor, and the pairwise joins take over.

## Multiplicative orders

The powers modulo a word go from the low bit of the exponent up, in pairs of bits: a product waits about ten cycles for the one before it, and the chain of squares is then the only one on the critical path, the products of the pairs' factors into the result off it (295 cycles for an exponent of 32 bits against 334 for a window from the top); longer residues in registers take a sliding window with its table on the stack, and pairs of powers to one exponent are taken side by side.

`m` is factored, and the order is the least common multiple of the orders modulo its prime powers. Modulo `2^e` it is the number of squares that bring `a` to 1. Modulo `p^e`, `p` odd, it is the order modulo `p`, in the smaller ring, among the primes of `p − 1`, which is factored too, times the power of `p` that brings `a` to that order to 1 modulo `p^e`, at most `p^(e−1)`. The order among the primes `q_1 … q_k` of a multiple `n` of it is found after Sutherland (2007): the primes are split in two halves, of products `n_1` and `n_2`, and the orders of `a^(n_2)` and of `a^(n_1)` are found among those of `n_1` and of `n_2`, down to one prime `q`, whose exponent is the number of powers by `q` that bring the residue to 1. That takes about `log2(n)·log2(k)` products against `k·log2(n)` for one power by `n/q^e` for each prime.

## Primitive roots

The candidates `g = 2, 3, …` (the odd ones modulo `2p^k`) are tried in increasing order, the cheap tests first: the Jacobi symbol `(g/p)` must be −1, which is Euler's criterion for `q = 2` without a power, by reciprocity on `p` modulo `g`; then `g^((p−1)/q)` must not be 1 for each odd prime `q` of `p − 1`, the smallest `q` first, which most candidates fail first; and for `k >= 2`, `g^(p−1)` must not be 1 modulo `p²`, which makes a root modulo `p` one modulo every `p^k`. The smallest root is small in practice (below `p^(1/4 + ε)` by Burgess's bound).

## Discrete logarithms

`m` is factored, and each prime power `p^e` says what it can of `k`:

- When `p` does not divide `g`: `h` must not be divisible by `p`, and then `k` is a logarithm modulo the order of `g` there. For an odd `p` the part of that order that divides `p − 1` is found modulo `p`, in the smaller ring, and the part that is a power of `p` modulo `p^e` on `g^(p−1)` and `h^(p−1)`, which are 1 modulo `p`: reduction modulo `p` maps the elements of order prime to `p` one to one, and the power to `p − 1` those of order a power of `p`, so the two logarithms are those of `g^k = h`, and they are joined. Each is Pohlig and Hellman's reduction (1978) over its order, factored: for each prime power `q^f` of it, `g_q = g^(n/q^f)` and `h_q = h^(n/q^f)`; the order of `g_q` is `q^f'` with `f' <= f`, found by powers by `q`, and the logarithm of `h_q` to `g_q` is found a digit in base `q` at a time, each a logarithm to `γ = g_q^(q^(f'−1))`, of order `q`: by stepping through its powers for `q < 16`; for `q = p` modulo `p^e`, `e >= 2`, by a division, since the elements of order `p` are `1 + p^(e−1)·c` and `(1 + p^(e−1)·c)^x = 1 + p^(e−1)·c·x`; by Shanks's baby steps and giant steps up to `2^21` modulo a word and `2^23` modulo longer numbers, one table for all the digits, `sqrt(q·f')` baby steps; by the index calculus below, modulo a prime `p` of at most 112 bits, for a `q` that divides `p − 1` once, from where it costs less than the walks, a power of two for each range of its parameters, whose cost steps with them: `2^33` for `p` up to 40 bits, `2^35` for 41 to 44 bits and one more for each four bits after, to `2^40` for 61 to 64; beyond a word, where a step of the walks costs about three times as much, `2^39`, `2^40`, `2^41`, `2^43`, `2^44`, `2^45`, `2^47`, `2^48`, `2^49`, `2^50`, `2^52` and `2^53` for 65 to 68 bits, 69 to 72 and so on to 109 to 112, one less without the kernels of the [powers](pow.md)' products in registers, whose walks there take about 1.35 times as long; and by Pollard's rho otherwise, once one power has shown that the element's order divides `q` (if not, `h` is not a power of `g`, which the walks would take about the root of a larger group's order to show). The logarithms modulo each `q^f'` are joined by the Chinese remainder theorem and checked by one power.
- When `p` divides `g` by `p^v`: `g^k` has the valuation `k·v` until it reaches `e`, and is 0 from `t = ceil(e/v)`; so `h = 0 (mod p^e)` asks for `k >= t`, and otherwise `k = v_p(h)/v` exactly, or there is no `k`.
- The classes are joined by the Chinese remainder theorem, their moduli the orders, which need not be coprime; an exact `k` is checked by one power modulo `m`; and the answer is the least `k` of the class that is at least every bound.

The index calculus modulo a prime `p`, for the logarithms modulo a large prime `q` of `p − 1`, is the linear sieve of Coppersmith, Odlyzko and Schroeppel (1986), which PARI's `Fp_log_index` takes too, with another elimination and linear algebra:

- With `H = ceil(sqrt(p))` and `J = H² − p`, `(H + c1)(H + c2) = J + (c1 + c2)·H + c1·c2 (mod p)`, a number of about `2C·sqrt(p)` for `|c1|, |c2| <= C`. When it factors over the primes up to `B`, `L(H + c1) + L(H + c2) = Σ e_l·L(l)` modulo `q`, for the logarithms `L` to any generator (`L(−1)` is 0 modulo the odd `q`). For each `c1` the numbers are a progression in `c2 >= c1`, sieved by each prime `l` from the root of `(H + c1)(H + c2) = p (mod l)`: a table of the first position by `c1 mod l` for each prime, the inverses modulo `l` by `1/x = −(l/x)/(l mod x)`; sixteen rows at a time, prime by prime, the logarithms in bytes, the primes below 8 left out; sixteen positions at once are compared with the size of the smallest number among them less 10 bits, and those that pass are divided by the primes of the base. `B` and `C` grow with `p`, from 192 and 100 at 40 bits to 768 and 600 at 64 and 8191 and 11,000 at 112; when the relations do not fix the logarithms, both are widened and the sieve runs again.
- The unknowns `L(H + c)` are eliminated through the graph of the `c`, with an edge for each relation: along a spanning forest by breadth, `L(H + v) = ±L(H + root) + F_v` for a combination `F_v` of the base, summed up the forest's path, and each edge off the forest gives a relation of the base alone, but the first that closes an odd cycle in a component, which fixes `L(H + root)`.
- These relations, the columns from the largest prime down, are reduced as they come against the rows so far, each zero before its pivot and 1 at it, modulo `q` in one limb or two: the multiplier of each pivot is the row's entry less the sum, not reduced, of the products by the multipliers before, so that each entry takes one Montgomery reduction. When one column is left free, the logarithms of the base are fixed up to a common factor, which a quotient of logarithms does not see.
- The logarithm of an element `z` comes from a multiple `z·w^e = u/v (mod p)` with `u, v < sqrt(p)`, by Euclid's algorithm stopped halfway, whose `u` and `v` both factor over the base but for at most one prime each a little beyond it (up to about four times the relations), whose logarithm comes from a pair of the sieve's range whose number it divides and is smooth after it, for a fixed `w`, a power of a prime of the base by an exponent that is not a multiple of `q`; that of `γ = g^c` from `c·L(g)`, `g` being mostly small and at once smooth. The quotient is checked by one power.

The methods:

- `bsgs`: the baby steps `g^j`, `j < s = ceil(sqrt(order))`, at most `2^21`, go into an open-addressing table of words, each a tag of the residue's key (as the walks of `rho` key it) and the step, half full, and stop early when they come round to 1, which gives the order of `g`; the giant steps `h·g^(−s·i)` are looked up, and the first found gives the smallest `k = s·i + j`, checked by one power when the tag of a residue of more than a word matched. Both kinds of steps are formed in four progressions at once, each waiting for its own products. A `g` that is not prime to `m` has no inverse: its powers are 0 modulo the part `m_1 = gcd(m, g^b)` of `m` made of its primes from `k = t <= log2(m)`, and prime to the rest `m_2`; the `k < t` are tried one by one, and the others searched modulo `m_2`.
- `rho`: Pollard's method for logarithms (1978) with Teske's walk (2001): steps `x → x·M_i` on elements `g^a·h^b`, the multiplier `M_i = g^(u_i)·h^(v_i)` one of 32 chosen by the low bits of the element's key (of the residue itself in Montgomery's form; of the residue mixed by a product modulo a power of two, whose low bits need not look random; and of all its limbs mixed modulo the other even numbers, whose elements of odd order all have the same low limb from `2^64` on), until a walk comes to an element met before; `g^a·h^b = g^(a')·h^(b')` then gives `k = (a' − a)/(b − b') mod q`. The elements met are recognised by keeping the distinguished ones, those whose key has `bits/4` zero bits above the multiplier's, in an open-addressing table (van Oorschot and Wiener, 1999), which lets eight walks run side by side, their elements in registers, each step written out for each walk so that the compiler keeps them there. The multipliers and the starts are products of the tables of `g^(i·16^j)` and `h^(i·16^j)`, independent products rather than chains of squares. The walk is the same for every run. A collision with `b = b'` tells nothing, and a walk that has one starts again elsewhere; every collision is such when `h` is not a power of `g`, so after four of them the answer is settled by `bsgs`.
- `pohlig_hellman`: the reduction above, on the modulus as it is, by the baby steps and giant steps and rho (not the index calculus, which needs a prime modulus).

Measured on an Apple M4 Max, single-threaded, in cycles per call by the harness's estimate of the clock, against FLINT 3.6 and PARI 2.19; the congruences, orders and primitive roots, and the logarithms, each with their conditions below. The functions that factor their modulus, or the order of the group, take [`factor`](factor.md); the long moduli below are primes `c·q + 1` with `q` prime, or have `p − 1` of known factors, so that what is timed is mostly the method itself.

The Chinese remainder theorem, against FLINT's `fmpz_CRT` and `fmpz_multi_CRT` and PARI's `chinese` and `ZV_chinese`, all with the residues reduced; and the orders and primitive roots, against PARI's `znorder` and `znprimroot` and FLINT's `n_primitive_root_prime` (which takes `p` as prime, where `primitive_root` tests it). These were measured on 2026-11-11 on the quiet machine (the load average at 3.4 to 5.2 of its 16 cores, from the system's own services; one thread, PARI's too, every process with `MallocNanoZone` unset, under the system's default allocator, the harness built with fat LTO and one codegen unit), interleaved with the previous version (0.2.32), in three passes of seven rounds, each cell the median of the 21, each round on fresh inputs of each contender's own, in an order that turns with the round (for many congruences a pool of systems, where the earlier measurements went over one again and again; the primes of 128 to 1024 bits of the orders and the primitive roots from a few drawn once a row, six of 1024 bits, no two contenders of a round on the same one):

| Congruences | a | 0.2.32 | FLINT | PARI |
| --- | --- | --- | --- | --- |
| 2 primes of 64 bits | 1007 | 1262 | 1252 | 1026 |
| 2 primes of 128 bits | 2481 | 2825 | 2850 | 2856 |
| 2 primes of 256 bits | 4523 | 5036 | 5129 | 5306 |
| 2 primes of 1024 bits | 16,538 | 17,694 | 20,538 | 22,530 |
| 2 primes of 4096 bits | 88,224 | 87,910 | 108,284 | 117,522 |
| 3 primes of 64 bits | 3020 | 3062 | 4731 | 3633 |
| 8 primes of 64 bits | 8072 | 8381 | 22,561 | 10,711 |
| 32 primes of 64 bits | 39,504 | 39,291 | 146,960 | 52,539 |
| 100 primes of 64 bits | 192,664 | 191,799 | 733,271 | 220,637 |
| 150 primes of 64 bits | 388,643 | 458,285 | 1.37 M | 373,794 |
| 200 primes of 64 bits | 550,723 | 634,672 | 2.10 M | 548,565 |
| 500 primes of 64 bits | 1.88 M | 2.12 M | 8.97 M | 1.96 M |
| 1000 primes of 64 bits | 4.84 M | 5.39 M | 26.9 M | 5.36 M |
| 10,000 primes of 64 bits | 87.3 M | 91.7 M | 1440 M | 143.2 M |
| 8 primes of 1024 bits | 226,210 | 228,734 | 343,379 | 275,503 |
| 100 primes of 1024 bits | 8.35 M | 8.34 M | 18.1 M | 10.5 M |

Faster than PARI and FLINT everywhere but for 150 primes of 64 bits, 1.04 times PARI's time, and 200, level with it: two words in registers 0.98 of PARI's time (1.23 times it before), two moduli of a few limbs on the stack 1.07 to 1.14 times faster than before (0.81 to 0.88 of FLINT's time), and from 150 to 1000 words the trees in limbs 0.90 to 1.04 of PARI's time (1.01 to 1.23 times it before), 0.61 at 10,000.

| Modulus | `multiplicative_order`: a | 0.2.32 | PARI |
| --- | --- | --- | --- |
| primes of 32 bits | 3861 | 5516 | 6278 |
| primes of 64 bits | 29,445 | 31,490 | 201,955 |
| random words | 38,378 | 39,617 | 154,082 |
| primes 2q + 1 of 128 bits | 46,922 | 52,258 | 262,414 |
| primes 30q + 1 of 256 bits | 212,250 | 286,828 | 1.19 M |
| primes 2q + 1 of 1024 bits | 4.34 M | 6.02 M | 13.2 M |

| Prime | `primitive_root`: a | 0.2.32 | FLINT | PARI |
| --- | --- | --- | --- | --- |
| 32 bits | 3313 | 5063 | 4981 | 5118 |
| 48 bits | 16,175 | 17,436 | 14,057 | 23,677 |
| 64 bits | 31,131 | 32,113 | 46,820 | 122,136 |
| 2q + 1 of 128 bits | 37,534 | 41,711 | – | 195,767 |
| 30q + 1 of 256 bits | 220,477 | 287,749 | – | 1.00 M |
| 2q + 1 of 1024 bits | 3.21 M | 4.34 M | – | 10.2 M |

- Orders 1.63 to 6.86 times faster than PARI, the 32-bit primes 1.43 times faster than before (they took 0.88 of PARI's time): the primality test and the factorisation of `p − 1` take most of the time on words, and the three strong tests' powers side by side and the powers from the low bit up took a third off them; the primes of 256 and 1024 bits 1.35 and 1.39 times faster than before, the others 1.03 to 1.11 times. Primitive roots 1.46 to 5.22 times faster than PARI (1.53 times faster than before at 32 bits, 1.31 and 1.35 at 256 and 1024, 1.03 to 1.11 for the others), and faster than FLINT but at 48 bits, where FLINT, which takes `p` as prime, is 1.15 times faster (1.07 to 1.24 from pass to pass): the Baillie–PSW test of `p` and the factorisation of `p − 1` are the time there.

Logarithms, against PARI's `znlog` and FLINT's `n_discrete_log_bsgs` (which searches the whole group by baby steps and giant steps). These, the safe primes and the methods below were measured on 2026-11-11 on the quiet machine (the load average at 4.1 to 5.5 of its 16 cores, from the system's own services; one thread, PARI's too, every process with `MallocNanoZone` unset, under the system's default allocator, the harness built with fat LTO and one codegen unit), interleaved with the previous version (0.2.32) where it is timed, in three passes of five rounds, each cell the median of the 15 (the safe primes of 104 and 112 bits in the first pass only, the median of its five), each round on fresh targets of each contender's own (`h` uniform below `p`, `g^e` for the methods; the moduli and the bases fixed for a row), in an order that turns with the round, where the earlier measurements went over one target a modulus again and again, so that it was learnt (the cheaper cells, `bsgs`'s most, take longer on fresh ones):

| Group | `discrete_log`: a | `pohlig_hellman`, `p − 1` factored | PARI | FLINT |
| --- | --- | --- | --- | --- |
| random primes of 32 bits | 26,256 | – | 635,128 | 23.9 M |
| random primes of 48 bits | 365,772 | – | 12.3 M | – |
| random primes of 56 bits | 1.95 M | – | 62.3 M | – |
| random primes of 64 bits | 3.62 M | – | 99.5 M | – |
| 64 bits, `p − 1` of 20-bit primes | 117,010 | 77,948 | 1.15 M | – |
| 128 bits, `p − 1` of 20-bit primes | 471,189 | 276,396 | 4.88 M | – |
| 128 bits, `p − 1` of 32-bit primes | 5.35 M | 4.58 M | 159.6 M | – |
| 256 bits, `p − 1` of 24-bit primes | 8.36 M | 2.92 M | 44.1 M | – |
| 256 bits, `p − 1` of 32-bit primes | 49.3 M | 30.4 M | 531.9 M | – |
| 512 bits, `p − 1` of 28-bit primes | 132.8 M | 48.7 M | 453.8 M | – |

Modulo safe primes `p = 2q + 1`, whose logarithms of order `q` the index calculus takes from 35 bits on (the factorisation of `p − 1` included, as in PARI's `znlog`, which takes its own index calculus there):

| Prime | `discrete_log`: a | PARI |
| --- | --- | --- |
| 40 bits | 390,162 | 13.0 M |
| 44 bits | 866,665 | 21.9 M |
| 48 bits | 1.43 M | 33.8 M |
| 56 bits | 2.88 M | 80.5 M |
| 64 bits | 7.19 M | 211.2 M |
| 68 bits | 16.5 M | 435.5 M |
| 72 bits | 21.0 M | 533.6 M |
| 80 bits | 60.6 M | 1183 M |
| 88 bits | 137.1 M | 2451 M |
| 96 bits | 316.4 M | 5124 M |
| 104 bits | 748.6 M | 10,338 M |
| 112 bits | 2401 M | 21,358 M |

- Logarithms: modulo random primes of 32 to 64 bits 24 to 34 times faster than PARI, and modulo safe primes from 40 to 112 bits 8.9 to 33 times (24 to 33 times up to 72 bits, 16 at 96, 8.9 at 112), both by an index calculus there; modulo smooth `p − 1`, with its factorisation by `factor`, 3.4 to 30 times, and the method itself, `pohlig_hellman`, 9.3 to 35 times (the factorisation of `p − 1` takes about two thirds of the time at 256 bits with primes of 24 bits and at 512 bits, 38 % at 256 bits with primes of 32 bits). Against 0.2.32 in the same runs: modulo random primes 1.9 times faster at 32 bits, 3.6 at 48 and about 65 at 56 (0.2.32's time there varies widely with the target, its passes' medians 108 M to 205 M cycles), and modulo safe primes 19 and 33 times at 40 and 44 bits (beyond 56 and 44 bits 0.2.32 takes seconds to hours a logarithm, and was not timed); modulo smooth `p − 1` `discrete_log` 1.26 to 2.09 times and `pohlig_hellman` 1.43 to 2.24 times.

The methods alone, in the group of prime order `q` modulo `p = 2q + 1` (eight groups a size, four of 40 bits, `g = 4`):

| Order | `bsgs` | `rho` | `pohlig_hellman` |
| --- | --- | --- | --- |
| 16 bits | 5356 | 5716 | 7217 |
| 20 bits | 18,980 | 17,376 | 20,927 |
| 22 bits | 38,823 | 24,744 | 26,649 |
| 24 bits | 73,947 | 35,364 | 36,813 |
| 26 bits | 173,759 | 59,941 | 64,039 |
| 28 bits | 457,180 | 99,469 | 106,434 |
| 30 bits | 1.10 M | 198,177 | 180,738 |
| 40 bits | 51.2 M | 5.73 M | 5.49 M |

Against 0.2.32, interleaved in the same runs, each on targets of its own: `bsgs` 1.3 to 1.6 times faster (16 to 40 bits), by its four chains; `rho` 2.1 to 2.4 times (20 to 40 bits, 1.5 at 16; about 5.1 cycles a step at 40 bits, against the `sqrt(π·q/2)` steps expected, the distinguished points' tail included), by its eight walks in registers and its tables of powers; `pohlig_hellman` 1.4 to 2.4 times. The table of baby steps falls out of the first-level cache from about `2^24` (`2^12` steps of 8 bytes in slots half full); inside `pohlig_hellman`, modulo a word, rho is the faster from orders of about `2^19.5` when each logarithm allocates a table of another size than the one before (on macOS a table of 16 to 128 KB then costs about 3.5 µs more, fresh pages) and from `2^23.3` when the same logarithm repeats, so that it is taken from `2^21` there (with `2^23` instead, `discrete_log` modulo primes of 64 bits whose `p − 1` has primes of 22 and 23 bits took 1.2 to 1.3 times as long, and repeated calls at most 6 % less); modulo longer numbers the two cross from about `2^21.5` to `2^25.5` by the length and the allocator's state, and rho is taken from `2^23`.

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

- Time complexity: the factorisations of `m` and of `p − 1` for its primes `p`; then `crt` `O(k²)` limb operations for `k` moduli of a word up to 150 of them and `O(M(n) log n)` beyond, `multiplicative_order` about `log2(p)·log2(k)` products modulo each `p^e`, `primitive_root` a few powers modulo `p` for each candidate, `discrete_log` about `sqrt(q)` products for the largest prime `q` of the order of `g`, or, modulo a prime `p` of at most 112 bits when it costs less, the index calculus, subexponential (about `exp(sqrt(ln p · ln ln p))`), and `log2(n)` powers for the reduction
- Space complexity: the factorisations; `bsgs` a table at most half full of its `min(sqrt(order), 2^21)` baby steps, two to four words a step (at most 32 MiB), `rho` about `sqrt(q) / 2^(bits/4)` distinguished elements, the index calculus its tables of roots, its rows and its relations, under a megabyte at 64 bits, about 6 MB at 96 and 26 MB at 112, and `crt` the product and remainder trees, about `2 log2(k)` times the product's length
- Soundness: the residues on limbs use the [powers](pow.md)' Montgomery kernels through their working memory plans; the Chinese remainder theorem's limbs call the multiplication's and the remainders' kernels with the working memory their plans ask for, on buffers aligned to 16 bytes; the operands are left as they were
