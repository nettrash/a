[A](../../../README.md) › [Math](../../../README.md#math)

# Quadratic sieve

`factor::siqs(&n)` finds a factor of a [number](numbers.md) by the self-initialising quadratic sieve, the method of choice for numbers of 40 to about 100 digits without small factors; [`Number::factor`](factor.md) calls it for such cofactors.

- The result is `Some(f)` with `f` a factor of `|n|` other than 1 and `|n|`, positive, in the byte order of `n` and in the shortest form, or `None` when there is none to find: for 0, 1, every prime and every power of a prime (`2^k` included), as [`is_probable_prime`](prime.md) and an exact root tell. A negative `n` is factored as `|n|`.
- `f` is the smallest prime factor when `|n|` has one below 1000, the root of `|n|` when it is a perfect power of a composite number, and otherwise a factor that a congruence of squares gives: for a product of two primes, one of them; for more, any proper factor, not necessarily a prime.
- Numbers below `2^64` are factored by Pollard's rho method with Brent's cycle finding (and Lehman's method should it fail, which only makes sure that it terminates); numbers above 900 bits not at all (`None`), as the sieve would take years.
- It is deterministic: the same number gives the same factor in the same time, every run.
- The magnitude is read in its own byte order, and zero bytes above its most significant non-zero byte are ignored; an empty vector is zero, and a negative zero is zero. The number is left as it was.

## The method

The quadratic sieve (Pomerance 1982) collects many `Y` whose `Y² − kN` factors over a set of small primes, the factor base, and combines some of them by linear algebra over GF(2) into a congruence of squares `X² ≡ Y² (mod N)`; then `gcd(X − Y, N)` is a proper factor with probability about 1/2. Montgomery's multiple polynomials and Alford and Pomerance's self-initialisation (1993; Contini's thesis, 1997, has the details) keep the values small: polynomials `(Ax + B)² − kN = A·g(x)` over a short interval, `A` a product of `s` primes of the factor base, with `2^(s−1)` values of `B` for each `A` that cost almost nothing to move between. This implementation carries it out as follows.

### The factor base and the polynomials

- The multiplier: Knuth and Schroeppel's function scores each square-free `k` below 100 by what the small primes are expected to contribute to the values for `kN`, less `ln(k)/2`: `2 ln p/(p − 1)` for a prime of two roots, `ln p/(p − 1)` for one of `k`, and for 2 by `kN mod 8`, three bits when `kN ≡ 1 (mod 8)` as the polynomials are then half as large (below). The primes go up to eight times the bits of `N` (at least 300, at most 2000); `(kN/p)` is `(k/p)(N/p)`, the first from the symbols of the primes of `k`.
- The factor base: 2 and the primes modulo which `kN` is a square, with a square root of `kN` modulo each (Tonelli and Shanks; Atkin's formula for `p ≡ 5 (mod 8)`), the inverse of each modulo `2^64`, and constants for remainders by it. A prime that divides `N` is the factor found.
- The polynomials: when `kN ≡ 1 (mod 4)`, `(2Ax + B)² − kN = 4A·g(x)` with `B` odd and `g(x) = Ax² + Bx + C`, half the values of `(Ax + B)²`; otherwise `(Ax + B)² − kN = A·g(x)` with `g(x) = Ax² + 2Bx + C`. `A` is a product of primes of 11 or 12 bits close to `sqrt(2kN)/M` (half as much for the first form) for an interval `[−M, M)`: `s − 1` of them drawn by a fixed generator (SplitMix64) and the last one that brings the product within 3.5 % of the target, every set of primes once. For `A = q_1⋯q_s`, `B_l = (A/q_l)·(t_l (A/q_l)^(−1) mod q_l)` with `t_l` the square root of `kN` modulo `q_l`, and polynomial `v` takes `±B_1 ± ⋯ ± B_(s−1) + B_s` with the signs of the Gray code `v ^ (v >> 1)`: from one polynomial to the next one `B_l` changes sign, and each root modulo `p` moves by `±2 B_l A^(−1) mod p`, a table made once for each `A` from the remainders of `A` and the `B_l` by each prime. The coefficients are numbers of a few limbs in two's complement, so that `C = (B² − kN)·A^(−1) mod 2^(64w)` takes no division, and `g(x)` and `Y` at a candidate are exact.

### The sieve

- The interval: blocks of 64 KiB, half the data cache of the cores measured on, 1 to 12 of them by the size of `N`, and for numbers below 167 bits a single shorter one (8 KiB below 91 bits, 16 KiB below 121, 32 KiB below 167). Each byte starts at `128 − threshold` and adds the scaled logarithms of the primes that divide the value there, so that a candidate is a byte with its top bit set, found eight bytes at a time; the scan sets the block back for the next one. The threshold is the logarithm of the largest value, `M·sqrt(kN/2)` (half for the first form), less 1.0 to 1.9 times the bits of the large-prime bound (more for larger numbers), less what the primes left out contribute on average.
- The small primes: those below 256 are not sieved, the small-prime variation (Pomerance 1985), as they hit most bytes for little.
- The medium primes, up to 64 Ki: block by block from their roots, in runs of primes with the same number `⌊65536/p⌋` of hits a root for sure: those by a loop of a constant count, two hits a step, and the last one, if any, without a branch, a miss adding to one of 64 bytes of a guard after the block (so that those additions do not wait on each other). The exits of loops of varying counts, mispredicted, had cost more than the hits: the runs took 15 % less time at 60 digits and 18 % at 70. The positions in each block are kept for the trial division.
- The large primes hit a block at most twice, most of them not at all. When a polynomial starts their roots are moved in vector registers, and their hits written to buckets, one for each block, slice of 2048 primes and lane of every fourth prime, with a prime's index and the position in a word: a hit for sure for each block the root covers and at most one more, the misses to eight buckets that are not counted. The lanes make the counts of consecutive primes independent, which they were waiting on: 23 % less time for the buckets at 70 digits. Each block then adds the logarithm of each slice at the positions of its buckets (Wambach and Wettig 1995; yafu's arrangement).

### The relations

- The trial division of a candidate: `g(x)` from the coefficients; the primes up to 256 and those of `A` by Hensel's test of exact division (`x·p^(−1) mod 2^64` limb by limb, the last carry zero exactly when `p` divides); the medium primes sixteen at a time in vector registers, by the remainder of the position in the block (`⌊o·⌊2^16/p⌋/2^16⌋` is the quotient or one less, so the remainder modulo `2^16` less the root is 0 or `p` exactly when `p` divides, and rarely otherwise, which the division tells); the large primes by the entries of the block's buckets at the candidate's position, sixteen compared at a time, unless what they added to the candidate's byte shows that the cofactor would stay too large to keep (which lost no relation and saved a quarter of that search at 80 digits). Each prime found divides `g(x)` exactly, as often as it divides it.
- Large primes: what is left is 1 (a full relation), a prime below the large-prime bound, 60 to 120 times the largest prime of the base (a partial relation), or, from 68 digits on, a composite below that bound to the power 1.8 that is above the square of the largest prime of the base, fails the strong probable-prime test to base 2, and splits by Pollard's rho into two primes below the bound (the double large-prime variation, Lenstra and Manasse 1994). Each relation is kept once, by its value `|Y|` modulo `2^64`.
- Cycles: the relations with large primes are edges of a graph on the large primes and 1, whose independent cycles a union–find counts as they arrive; the sieve stops when the full relations and the cycles exceed the factor base by 96. The cycles themselves come from a spanning forest by breadth-first search, an edge outside it closing each.

### The linear algebra and the square root

- Each full relation and each cycle is a column of the matrix of the exponents modulo 2 (with a row for −1); the columns with a row of weight one are removed until there are none, and the heaviest beyond 96 more than the rows.
- Montgomery's block Lanczos method (1995), on 64 vectors at once, the choice of each step's subspace by Montgomery's elimination, then the null vectors among the combinations of the solution and the last iterate that the matrix takes to zero (after msieve's arrangement); Gaussian elimination for matrices of at most 400 columns, and, up to 6000, should Lanczos fail four times.
- For each vector, `X` is the product of the relations' `Y` modulo `N` and the square root is the product of their primes to half their exponents (with a 2 for each relation of the first form); `gcd(X − Y, N)` is tried for both signs. Should every vector fail, more relations are collected.

The parameters by size (the primes of the base, the blocks, the large-prime bound, the slack of the threshold) started from msieve's and yafu's tables and were set by measurement (sweeps of each, a few numbers of each size): from 75 digits on, factor bases two to four times as large as yafu's paid (msieve's are of the same order), as the lanes made the large primes cheaper; at 90 digits 80,000 primes took about 15 % less time than 45,000 and 40 % less than 25,000.

| Digits | primes in the base | interval | large primes up to | two large primes |
| --- | --- | --- | --- | --- |
| 40 | 560 | 32 KiB | 36 × the largest of the base | no |
| 50 | 1,240 | 32 KiB | 49 × | no |
| 60 | 3,400 | 64 KiB | 59 × | no |
| 70 | 9,600 | 192 KiB | 60 × | yes |
| 80 | 29,000 | 384 KiB | 79 × | yes |
| 90 | 78,000 | 640 KiB | 99 × | yes |
| 100 | 148,000 | 768 KiB | 119 × | yes |

## Measured

Measured on an Apple M4 Max, single-threaded, in CPU seconds, on balanced semiprimes (two random primes of half the digits each, from PARI's `nextprime`), the median of two numbers of each size up to 85 digits (and of 3 to 5 runs of each below 75 digits), one number beyond (PARI and FLINT only to 90 digits: they took 27 and 46 minutes there), with the work of other processes on the machine (8 to 16 runnable on its 16 cores, which made some times up to a quarter longer from run to run; the contenders were measured in the same hours), but for the rows of 40, 50, 60, 70 and 80 digits, measured again with nothing else running: msieve 1.53 (`msieve -q -t 1`), PARI/GP 2.19's MPQS (`factorint(n, 14)`, which leaves out the other methods, `nbthreads` 1), FLINT 3.6's `qsieve_factor` (one thread), and yafu 3.1.9's published single-thread times on an AMD Ryzen 9 5900X with AVX2 (its own benchmark numbers, `qs_time`, from its wiki; another processor and other numbers, so only a guide). In parentheses, each contender's time over that of `siqs`:

| Digits | `siqs` | msieve 1.53 | PARI 2.19 | FLINT 3.6 | yafu 3.1.9 (Ryzen 9 5900X, published) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 40 | 6.9 ms | 30 ms (4.4×) | 8.5 ms (1.2×) | 16 ms (2.3×) | – |
| 45 | 14 ms | 73 ms (5.0×) | 22 ms (1.6×) | 33 ms (2.3×) | 0.05 s |
| 50 | 58 ms | 0.17 s (2.9×) | 83 ms (1.4×) | 0.18 s (3.2×) | 0.12 s |
| 55 | 0.17 s | 0.45 s (2.7×) | 0.28 s (1.7×) | 0.60 s (3.6×) | 0.25 s |
| 60 | 0.48 s | 1.52 s (3.2×) | 0.99 s (2.1×) | 2.26 s (4.7×) | 1.18 s |
| 65 | 1.65 s | 5.98 s (3.6×) | 3.59 s (2.2×) | 5.01 s (3.0×) | 3.84 s |
| 70 | 4.72 s | 13.7 s (2.9×) | 13.4 s (2.8×) | 16.2 s (3.4×) | 7.08 s |
| 75 | 13.1 s | 32.2 s (2.5×) | 36.8 s (2.8×) | 50.9 s (3.9×) | 23.78 s |
| 80 | 36.8 s | 96.0 s (2.6×) | 105 s (2.9×) | 140 s (3.8×) | 46.99 s |
| 85 | 115 s | 319 s (2.8×) | 344 s (3.0×) | 540 s (4.7×) | 139.87 s |
| 90 | 417 s | 909 s (2.2×) | 1,643 s (3.9×) | 2,786 s (6.7×) | – |
| 95 | 1,072 s | 2,014 s (1.9×) | – | – | – |
| 100 | 3,554 s | 7,430 s (2.1×) | – | – | – |

- `siqs` is the fastest of the four measured at every size: 1.2 to 2.2 times as fast as PARI up to 65 digits, whose MPQS is the strongest of the three on small numbers (at 40 digits all but the sieve, from the multiplier to the square root, is a fifth of the time), and from 70 digits 1.9 to 3.9 times as fast as msieve and PARI and 3.4 to 6.7 times as fast as FLINT, where the double large-prime variation and the large factor bases pay.
- Against yafu's published times on a Ryzen 9 5900X, about 1.2 to 3.6 times as fast; the M4 Max runs single-threaded code faster than a Ryzen 9 5900X, so this mostly says that the two are of a kind.
- Memory, the most resident at once: 13 MB at 60 digits, 75 MB at 80, 220 MB at 90, 320 MB at 95 and 415 MB at 100, most of it the relations.

Where the time goes, at 90 digits: moving the large primes' roots and filling their buckets 45 %, the medium primes 24 %, the buckets' logarithms 19 %, the trial division of 8.7 million candidates 12 %, the linear algebra (80,000 columns) about 1 %.

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

- Time complexity: `L(N)^(1 + o(1))` with `L(N) = exp(sqrt(ln N · ln ln N))`, heuristically, as for every quadratic sieve; in practice the times above, about three times as long for every 5 digits more
- Space complexity: the factor base and the roots, some 40 bytes a prime; a block of 64 KiB; the buckets, 8 bytes a large prime and block; the relations, about 100 bytes each (their primes as indices into the factor base); and the matrix: some 75 MB at 80 digits and 415 MB at 100
- Soundness: no `unsafe` code; every factor returned divides `|n|`, as `gcd(X − Y, N)` does whatever the relations
