[A](../../../README.md) › [Math](../../../README.md#math)

# Quadratic sieve

`factor::siqs(&n)` finds a factor of a [number](numbers.md) by the self-initialising quadratic sieve, the method of choice for numbers of 24 to about 100 digits without small factors; [`Number::factor`](factor.md) calls it for such cofactors, from 24 to 110 digits.

- The result is `Some(f)` with `f` a factor of `|n|` other than 1 and `|n|`, positive, in the byte order of `n` and in the shortest form, or `None` when there is none to find: for 0, 1, every prime and every power of a prime (`2^k` included), as [`is_probable_prime`](prime.md) and an exact root tell. A negative `n` is factored as `|n|`.
- `f` is the smallest prime factor when `|n|` has one below 1000, the root of `|n|` when it is a perfect power of a composite number, and otherwise a factor that a congruence of squares gives: for a product of two primes, one of them; for more, any proper factor, not necessarily a prime.
- Numbers below `2^64` are factored by Pollard's rho method with Brent's cycle finding (and Lehman's method should it fail, which only makes sure that it terminates); numbers above 900 bits not at all (`None`), as the sieve would take years.
- It is deterministic: the same number gives the same factor in the same time, from run to run. The sieve's parameters and logarithms are rounded from the platform's floating-point `log2`, `ln` and `powf`, whose last bit Rust leaves to the platform (Apple's and glibc's differ on some), so that on another target a number may, rarely, give the other factor.
- The magnitude is read in its own byte order, and zero bytes above its most significant non-zero byte are ignored; an empty vector is zero, and a negative zero is zero. The number is left as it was.

## The method

The quadratic sieve (Pomerance 1982) collects many `Y` whose `Y² − kN` factors over a set of small primes, the factor base, and combines some of them by linear algebra over GF(2) into a congruence of squares `X² ≡ Y² (mod N)`; then `gcd(X − Y, N)` is a proper factor with probability about 1/2. Montgomery's multiple polynomials and Alford and Pomerance's self-initialisation (1993; Contini's thesis, 1997, has the details) keep the values small: polynomials `(Ax + B)² − kN = A·g(x)` over a short interval, `A` a product of `s` primes of the factor base, with `2^(s−1)` values of `B` for each `A` that cost almost nothing to move between. This implementation carries it out as follows.

### The factor base and the polynomials

- The multiplier: Knuth and Schroeppel's function scores each square-free `k` below 100 by what the small primes are expected to contribute to the values for `kN`, less `ln(k)/2`: `2 ln p/(p − 1)` for a prime of two roots, `ln p/(p − 1)` for one of `k`, and for 2 by `kN mod 8`, three bits when `kN ≡ 1 (mod 8)` as the polynomials are then half as large (below). The primes go up to eight times the bits of `N` (at least 300, at most 2000); `(kN/p)` is `(k/p)(N/p)`, the first from the symbols of the primes of `k`, each by quadratic reciprocity from a table of the squares modulo that prime, and the scores added without a branch.
- The factor base: 2 and the primes modulo which `kN` is a square, with a square root of `kN` modulo each (the power `(kN)^((p + 1)/4)` for `p ≡ 3 (mod 4)`, whose square tells at once whether there is one; Tonelli and Shanks otherwise, with Atkin's formula for `p ≡ 5 (mod 8)`) and the inverse of each modulo `2^64`. A prime that divides `N` is the factor found.
- The polynomials: when `kN ≡ 1 (mod 4)`, `(2Ax + B)² − kN = 4A·g(x)` with `B` odd and `g(x) = Ax² + Bx + C`, half the values of `(Ax + B)²`; otherwise `(Ax + B)² − kN = A·g(x)` with `g(x) = Ax² + 2Bx + C`. `A` is a product of `s` primes of 8 to 12 bits (by the size of `N`, from 64 up) close to `sqrt(2kN)/M` (half as much for the first form) for an interval `[−M, M)`: `s − 1` of them drawn by a fixed generator (SplitMix64) and the last one that brings the product within 7 % of the target, every set of primes once; `s` is the number nearest to the target over the bits asked for whose mean size lies half a bit within the primes allowed, so that the last one can always make up for the others (a 28-digit number had needed 19,530 draws for 5 values of `A` before). For `A = q_1⋯q_s`, `B_l = (A/q_l)·(t_l (A/q_l)^(−1) mod q_l)` with `t_l` the square root of `kN` modulo `q_l`, and polynomial `v` takes `±B_1 ± ⋯ ± B_(s−1) + B_s` with the signs of the Gray code `v ^ (v >> 1)`: from one polynomial to the next one `B_l` changes sign, and each root modulo `p` moves by `±2 B_l A^(−1) mod p`, a table made once for each `A` from the remainders of `A` and the `B_l` by each prime. The coefficients are numbers of a few limbs in two's complement, so that `C = (B² − kN)·A^(−1) mod 2^(64w)` takes no division, and `g(x)` and `Y` at a candidate are exact.

### The sieve

- The interval: blocks of 64 KiB, half the data cache of the cores measured on, 1 to 12 of them by the size of `N`, and for numbers below 50 digits a single shorter one (16 KiB to 36 digits, 28 KiB at 40, 32 KiB at 45). Each byte starts at `128 − threshold` and adds the scaled logarithms of the primes that divide the value there, so that a candidate is a byte with its top bit set, found sixty-four bytes at a time in vector registers; the scan sets the block back for the next one.
- The tiny primes, below 192 to 256 for numbers up to 36 digits, 384 to 768 from 40 to 60 digits and 1024 beyond, are not sieved, the small-prime variation (Pomerance 1985), as they hit most bytes for little. Two thresholds stand in for them: the sieve's own is lower by a margin (16 to 28 bits by the size), and for each byte that reaches it the logarithms of the tiny primes that divide the value there are added from their roots, eight primes at a time in vector registers; a candidate is a byte whose sum reaches the threshold proper, the logarithm of the largest value, `M·sqrt(kN/2)` (half for the first form), less 0.8 to 2.0 times the bits of the large-prime bound (more for larger numbers) and what 2 contributes on average. Against one threshold lowered by what the tiny primes contribute on average, this found more relations with fewer candidates: 17 % less time at 65 digits and 19 % at 40.
- The medium primes, up to 64 Ki: block by block from their roots, in runs of primes with the same number `⌊65536/p⌋` of hits a root for sure: those by a loop of a constant count, two hits a step, and the last one, if any, without a branch, a miss adding to one of 64 bytes of a guard after the block (so that those additions do not wait on each other). The exits of loops of varying counts, mispredicted, had cost more than the hits: the runs took 15 % less time at 60 digits and 18 % at 70. Each hit is three instructions, as the positions below their primes keep every byte within the block and its guard without a check of the bounds (5.9 % less time at 60 digits); the runs of one to three sure hits, whose bookkeeping outweighs their hits, go four primes at a time through a vector kernel that moves their positions in vector registers and adds through the lanes (4.8 % less at 60 digits). The primes and their positions are held in 16 bits, a third less beside the block in the data cache (4.5 % less time at 60 digits, 2.7 % at 70). The positions in each block are kept for the trial division.
- The large primes hit a block at most twice, most of them not at all. When a polynomial starts their roots are moved in vector registers, and their hits written to buckets, one for each block, slice of 2048 primes and lane of every fourth prime, with a prime's index and the position in a word: a hit for sure for each block the root covers and at most one more, a miss to a bucket of its lane after the blocks' that no block reads, so that hits and misses take the same steps (2.6 % less time at 70 digits than misses to buckets left uncounted, chosen by a select). The lanes make the counts of consecutive primes independent, which they were waiting on: 23 % less time for the buckets at 70 digits. On aarch64 the entries are written without checks of their bounds (2.3 % less at 70 digits) and four primes of a run at a time, their buckets and entries made in vector registers. Each block then adds the logarithm of each slice at the positions of its buckets (Wambach and Wettig 1995; yafu's arrangement).

### The relations

- The trial division of a candidate: `g(x)` from the coefficients; the primes of `A` by Hensel's test of exact division (`x·p^(−1) mod 2^64` limb by limb, the last carry zero exactly when `p` divides); every other prime below `2^16` eight at a time in vector registers, by the remainder of the position in the block (`⌊o·⌊2^16/p⌋/2^16⌋` is the quotient or one less, so the remainder modulo `2^16` less the root is 0 or `p` exactly when `p` divides, and rarely otherwise, which the division tells), the lanes that hit found from the comparisons narrowed to a byte each rather than tested one by one (with the vector scan of the block, 18 % less time at 26 and 28 digits); the large primes by the entries of the block's buckets at the candidate's position, sixteen compared at a time, unless what they added to the candidate's byte shows, within two bits, that the cofactor would stay too large to keep (a fifth fewer searches at 70 digits for the same relations). Each prime found divides `g(x)` exactly, as often as it divides it.
- Large primes: what is left is 1 (a full relation), a prime below the large-prime bound, 10 to 130 times the largest prime of the base by the size of `N` (a partial relation), or, from 68 digits on, a composite below that bound to the power 1.8 that is above the square of the largest prime of the base, fails the strong probable-prime test to base 2, and splits by Pollard's rho into two primes below the bound (the double large-prime variation, Lenstra and Manasse 1994). Each relation is kept once, by its value `|Y|` modulo `2^64`.
- Cycles: the relations with large primes are edges of a graph on the large primes and 1, whose independent cycles a union–find counts as they arrive; the sieve stops when the full relations and the cycles exceed the factor base by 12 when the matrix goes to Gaussian elimination (each vector of the null space gives a factor with probability about 1/2, and there are at least as many as the excess; at 26 to 28 digits 96 had cost a quarter more time), by 32 when it goes to block Lanczos. The tables of the relations are hashed by a product with an odd constant, the same every run. The cycles themselves come from a spanning forest by breadth-first search, an edge outside it closing each.

### The linear algebra and the square root

- Each full relation and each cycle is a column of the matrix of the exponents modulo 2 (with a row for −1); the columns with a row of weight one are removed until there are none, and the heaviest beyond 96 more than the rows.
- Gaussian elimination on the dense rows for matrices of at most 5000 columns, by the method of the four Russians (Arlazarov, Dinic, Kronrod and Faradžev 1970): the columns taken a few at a time, up to eight, as many as about half the rows left can repay, their pivots made to have one pivot bit each, and each row below reduced by one look-up of a table of the sums of the pivots; then the vectors, 64 at once in the bits of words, by substitution from the last pivot row up, each with one free column set. Against the elimination on the columns with an identity beside them, branched on each bit, the null space at 28 digits took 0.04 ms rather than 0.10; at 3800 columns it took 8 ms against 10 for Lanczos, at 7200 60 against 36.
- Montgomery's block Lanczos method (1995) beyond, on 64 vectors at once, the choice of each step's subspace by Montgomery's elimination, then the null vectors among the combinations of the solution and the last iterate that the matrix takes to zero (after msieve's arrangement); Gaussian elimination should it fail four times, up to 6000 columns.
- For each vector, `X` is the product of the relations' `Y` modulo `N` and the square root is the product of their primes to half their exponents (with a 2 for each relation of the first form), both in Montgomery's arithmetic modulo `N` (the crate's field arithmetic of the factoring methods); `gcd(X − Y, N)` is tried for both signs. Should every vector fail, a sixteenth more relations are collected.

The parameters by size (the primes of the base, the interval, the large-prime bound, the thresholds, the tiny primes and the size of the primes of `A`) started from msieve's and yafu's tables and were set by measurement, by coordinate descent over each, with 40 to 60 numbers of each size up to 44 digits and two to six beyond: below 45 digits factor bases a fifth to a third smaller than before paid, with the cheaper linear algebra and the smaller excess; from 75 digits on, factor bases two to four times as large as yafu's (msieve's are of the same order), as the lanes made the large primes cheaper; at 90 digits 80,000 primes took about 15 % less time than 45,000 and 40 % less than 25,000.

| Digits | primes in the base | interval | large primes up to | two large primes | tiny primes below |
| --- | --- | --- | --- | --- | --- |
| 26 | 117 | 16 KiB | 10 × the largest of the base | no | 256 |
| 30 | 180 | 16 KiB | 12 × | no | 201 |
| 40 | 638 | 28 KiB | 36 × | no | 393 |
| 50 | 1,544 | 64 KiB | 36 × | no | 750 |
| 60 | 3,800 | 64 KiB | 70 × | no | 768 |
| 70 | 10,500 | 192 KiB | 75 × | yes | 1024 |
| 80 | 30,000 | 384 KiB | 90 × | yes | 1024 |
| 90 | 80,000 | 640 KiB | 110 × | yes | 1024 |
| 100 | 150,000 | 768 KiB | 130 × | yes | 1024 |

## Measured

Measured on 2026-11-26 on an Apple M4 Max with nothing else running, by the harness in the repository's `bench/nt` (its method in `bench/README.md`), on one thread: `siqs` as a user builds it, by rustc 1.100 stable in cargo's release profile (no LTO, 16 codegen units, no `target-cpu`), against PARI 2.19.1's MPQS (`factorint(n, 14)`, which leaves out its other methods) and FLINT 3.6.0's `qsieve_factor` with one thread, as Homebrew builds them, called through shims compiled with `-O3 -mcpu=native`, and msieve 1.53 (`msieve -q -t 1`), a process a number. The numbers are balanced semiprimes, two random primes of half the digits each, a set of each size factored whole in every sample, its mean time a number given; each figure is the median of 15 samples (10 at 50 digits, then 5, 3, 3 and 2), `siqs` and the references alternating sample by sample, after a check that every factor found divides the number. A ratio is the time of `siqs` over the reference's: below 1, `siqs` is faster.

| Digits | numbers | `siqs` | PARI | ratio | FLINT | ratio | msieve | ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 26 | 16 | 215 µs | 431 µs | 0.499 | 2.27 ms | 0.0946 | 10.1 ms | 0.0213 |
| 28 | 16 | 330 µs | 562 µs | 0.587 | 2.16 ms | 0.152 | 13.9 ms | 0.0237 |
| 30 | 16 | 450 µs | 885 µs | 0.508 | 2.71 ms | 0.166 | 16.0 ms | 0.0282 |
| 32 | 16 | 692 µs | 1.30 ms | 0.534 | 3.66 ms | 0.189 | 20.9 ms | 0.0331 |
| 36 | 12 | 1.49 ms | 3.20 ms | 0.465 | 7.58 ms | 0.197 | 25.8 ms | 0.0577 |
| 40 | 8 | 3.28 ms | 8.96 ms | 0.366 | 14.9 ms | 0.221 | 39.8 ms | 0.0826 |
| 44 | 6 | 7.36 ms | 19.8 ms | 0.371 | 42.3 ms | 0.174 | 53.1 ms | 0.139 |
| 45 | 4 | 10.3 ms | 27.1 ms | 0.380 | 45.0 ms | 0.229 | 66.4 ms | 0.155 |
| 50 | 3 | 30.6 ms | 79.3 ms | 0.386 | 206 ms | 0.148 | 151 ms | 0.202 |
| 55 | 2 | 85.6 ms | 255 ms | 0.336 | 538 ms | 0.159 | 427 ms | 0.200 |
| 60 | 1 | 323 ms | 901 ms | 0.359 | 1.95 s | 0.166 | 1.11 s | 0.292 |
| 65 | 1 | 1.21 s | 3.87 s | 0.314 | 5.12 s | 0.237 | 6.61 s | 0.183 |
| 70 | 1 | 3.83 s | 13.6 s | 0.282 | 17.3 s | 0.221 | 15.8 s | 0.243 |

- `siqs` is the fastest at every size measured: 0.282 to 0.587 of the time of PARI's MPQS, the fastest of the references on these numbers, the most from 60 digits, where the double large-prime variation and the larger factor bases pay; and 0.0946 to 0.237 of FLINT's.
- msieve runs as a process for each number, and its start, 1.9 ms, is in its times, which it dominates below 40 digits: msieve does more there, so its column is shown and not counted. From 50 digits, where the start is a small part of its time, `siqs` takes 0.183 to 0.292 of it.
- Numbers of 75 to 100 digits, minutes to hours each, were not part of this run, nor yafu, which is not installed here.

## Examples

Rust usage: factors of semiprimes, a prime power and small numbers:

```rust
use a::math::numbers::{ByteOrder, Number, factor};

fn main() {
  // 1000000016000000063 = 1000000007 · 1000000009, below 2^64.
  let n = Number::new_from_u64(1_000_000_016_000_000_063, ByteOrder::BigEndian);
  let f = factor::siqs(&n).unwrap();
  assert!(f == 1_000_000_007u64 || f == 1_000_000_009u64);

  // A product of two primes of 20 digits, by the sieve.
  let p: Number = "28162596453413674991".parse().unwrap();
  let q: Number = "71875687879868738317".parse().unwrap();
  let n = &p * &q;
  let f = factor::siqs(&n).unwrap();
  assert!(f == p || f == q);
  assert_eq!(f.byte_order, ByteOrder::LittleEndian);

  // Primes and prime powers have no factor to find; a small factor comes first.
  assert_eq!(factor::siqs(&p), None);
  assert_eq!(factor::siqs(&p.pow(3)), None);
  let m = &n * &Number::new_from_u16(997, ByteOrder::LittleEndian);
  assert_eq!(factor::siqs(&m), Some(Number::new_from_u16(997, ByteOrder::LittleEndian)));
}
```

## Efficiency

- Time complexity: `L(N)^(1 + o(1))` with `L(N) = exp(sqrt(ln N · ln ln N))`, heuristically, as for every quadratic sieve; in practice the times above, about three to four times as long for every 5 digits more
- Space complexity: the factor base and the roots, some 40 bytes a prime; a block of 64 KiB; the buckets, 8 bytes a large prime and block, and 8 more for the steps past the interval; the relations, about 100 bytes each (their primes as indices into the factor base); and the matrix; in all, the most resident at once some 80 MB at 80 digits and 520 MB at 100
- Soundness: every factor returned divides `|n|`, as `gcd(X − Y, N)` does whatever the relations. The `unsafe` code is the medium sieve's addition to a byte of the block without a check of its bounds (each position below its prime, which Miri checks with the tests) and, on aarch64 only, the NEON kernels (the trial division's comparisons with the roots, the tiny primes' shares, the scan, the medium primes of one to three hits, the bucket entries) with their writes without checks; elsewhere and under Miri the same steps run in plain Rust, with the same results
