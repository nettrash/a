[A](../../../README.md) › [Math](../../../README.md#math)

# Prime sieves and counting

`primes(range)` gives the primes in a range of `u64`s, in increasing order, as an iterator, `Primes`; `prime_pi(x)` is `π(x)`, the number of primes up to `x`; and `nth_prime(n)` is the `n`-th prime, `nth_prime(1) = 2`. They are in `a::math::numbers`, beside [`Number`](numbers.md), whose [primality test](prime.md) they share below `2^64`.

- `primes` takes any range of `u64`s, `a..b`, `a..=b`, `a..`, `..b` and the rest, up to `u64::MAX`: an empty or reversed range gives no primes. `Primes` is a fused iterator; it clones, and its `count`, `fold` (and so `sum`, `for_each`, `collect`) read the sieve directly, `count` without forming the primes. `count` panics if the count passes `usize::MAX`, which can only happen on 32-bit targets.
- `prime_pi` is exact for every `u64`: `π(2^64 − 1) = 425,656,284,035,217,743`.
- `nth_prime(n)` is exact for `n` from 1 to `π(2^64 − 1)`, whose prime is `2^64 − 59`; it panics for `n = 0` and above, as there is no such prime below `2^64`.
- All three are deterministic and single-threaded, and use the processor's caches as the measurements below describe; none is constant-time.

## The segmented sieve

The sieve is Kim Walisch's primesieve's ("primesieve: fast prime number generator", 2010–2026), a segmented sieve of Eratosthenes after Tomás Oliveira e Silva's ("Fast implementation of the segmented sieve of Eratosthenes", 2002) and Achim Flammenkamp's ("The Art of Prime Sieving", 1998). This implementation carries it out as follows:

- A byte holds the 30 numbers from a multiple of 30 in the eight prime to 30, as bits for `30j + 7, 11, 13, 17, 19, 23, 29, 31`, so that 1 never appears; a segment is up to 2 MiB, 63 million numbers (twice the square root of the range's end, in whole parts of 128 KiB), or the range's length when shorter.
- Pre-sieving: a segment starts as the AND of 16 tables of the bytes with the multiples of two or three of the primes from 7 to 163 crossed out (`7·23·37`, `11·19·31`, …, `97·101`, 123 KiB, computed at compile time), each repeating with the product of its primes, four tables a pass.
- Small primes, up to a fifth of 128 KiB, cross out their multiples in parts of 128 KiB, which stay in the first-level cache: a prime is held as `p/30`, its next multiple's byte and its wheel index (the class `p mod 30` and the position of the multiplier modulo 30), and a whole turn of the wheel, eight multiples, is crossed out at once at offsets fixed by the class, by a loop specialised for each of the 64 classes and positions.
- Medium primes, up to three times the segment's bytes, cross out a few multiples each: they are kept in 64 lists by wheel index, and each list runs a loop specialised for its class and position, which takes eight steps a turn with no table; the primes are filed in the lists of the next segment as they leave.
- Large primes, up to the square root of the range's end, have fewer multiples than there are segments: Oliveira e Silva's bucket sieve on the wheel of 210 keeps each in a bucket of the segment of its next multiple, 1024 primes a bucket, so that a segment touches only the primes that cross out one of its numbers. A prime with no multiple in the range is not kept at all, as in primesieve: the range `[2^64 − 10^7, 2^64)` keeps 33 MB, not the 1.6 GB of all the primes below `2^32`.
- The primes are formed from each 64-bit word of the sieve by its trailing zeros, four at a time into a batch of up to 1024: the iterator's `next` reads the batch. The counts are the words' population counts.
- The sieving primes come from a sieve of their own, by primes up to `2^16` from a plain sieve of Eratosthenes.

### Short ranges far up

The sieve's start costs a division and a place for each prime up to the square root of the range: about 4 ns for each of the 203 million primes below `2^32` near `2^64`. A range narrower than about `√a / 90` is cheaper to sieve with the primes up to a quarter of its width only and to test the numbers left by Baillie–PSW, which they leave about one in `ln(width)/0.56`, the primes among them costing 0.6 µs each and the others 70 ns: the two meet there. `Primes` therefore sieves a range in stretches: the first 4096 numbers, enough for a hundred primes even near `2^64`, then four times as many each time, each stretch narrower than `√a / 90` sieved by the smaller primes and tested, and the rest of the range at once by the full sieve. Below `2^40`, where a stretch's sieving primes take a third of a millisecond at most to start, the full sieve's stretches grow the same way while they are narrower than a segment (63 million numbers): the first segment of the whole rest is sized by its end, 2 MiB for an end near `2^64`, which took 1.7 ms for the first ten primes from 0 before this rule and takes 0.6 µs with it. `count` and `fold`, which take every prime, plan the rest of the range as one stretch. primesieve sieves every range in full, from a table below 720 and in chunks of about `√a` that grow fourfold.

## Prime counting

`prime_pi(x)` reads a table below `2^16`, counts the sieve's bits below `2·10^5`, and above uses Deléglise and Rivat's method ("Computing π(x): the Meissel, Lehmer, Lagarias, Miller, Odlyzko method", 1996), in the combinatorial form of Oliveira e Silva ("Computing π(x): the combinatorial method", 2006) and of Walisch's primecount: with `y = α·x^(1/3)` and `a = π(y)`,

`π(x) = φ(x, a) + a − 1 − P2(x, a)`,

where `φ(x, a)`, the numbers up to `x` with no prime factor up to `y`, is split by the leaves of Legendre's recursion into the ordinary leaves `S1` and the special ones `S2` (trivial, easy and hard), and `P2` counts the numbers up to `x` with exactly two prime factors above `y`. `α` is primecount's fit, a polynomial in `ln x`, from 2.6 at `10^9` to 17.8 at `10^16` and 38 near `2^64`: our measurements found no better.

- `φ(x, a)` for `a ≤ 8` is `(x / P)·φ(P) + φ(x mod P, a)`, `P = 2·3·5·7·11·13·17·19` taken as `510510` and a division by 19, from tables of the numbers prime to the first primes computed at compile time.
- The ordinary leaves `S1`: the square-free `n ≤ y` with prime factors above 19, by a recursion over the primes.
- The trivial leaves, whose `φ` is 1, are counted, by an arithmetic progression for the primes above `x^(1/3)`.
- The easy leaves `p_b·q` with `x/(p_b q) ≤ y`, whose `φ(x/(p_b q), b − 1)` is `π(x/(p_b q)) − b + 2`, from a table of `π` up to `y` (a bit for each number prime to 30 and a count for each 240). Those above `√(x/p_b)` come in runs that share one value, which primecount sums run by run, each run's end found from its value: a chain of a division and a look-up after another, which leaves the processor waiting (at `10^16`, 3.2·10^8 runs took most of the 3.5 s primecount's form of the method spends on these leaves). Here they are summed by swapping the order of the pairs of primes `(q, r)` with `q·r ≤ N = x/p_b`: `Σ_{Q<q≤T} π(N/q) = π(N/T)·(π(T) − π(Q)) + Σ_{N/T<r≤N/Q} (π(N/r) − π(Q))`, whose primes `r` run over the same primes as the other easy leaves, `q` from `z/p_b` to `√N`, so that one division and one look-up serve both sums, and no term waits for another. At `10^16` the easy leaves took 6 s summed run by run in one stream, 3.5 s in primecount's two streams, and take 0.3 s so.
- The hard leaves, `x/n > y`, need `φ(x/n, b − 1)` of numbers up to `z = x/y`: a segmented sieve of the numbers from 0 to `z` (a bit for each number prime to 30, segments of 64 KiB, which measured 2 % faster than 128 KiB) is crossed out by the primes in order, the leaves of each prime read before its multiples are crossed out, `φ` being the numbers left before the segment, kept for each prime, and those left in the segment up to the leaf. primecount counts the latter from the last leaf's point, by counters of blocks of the segment and the words after the last block; that count, carried from leaf to leaf, ends its loops at points the processor mispredicts. Here each leaf's count is formed afresh, with no branch: the bits before its block of 64 bytes, from the blocks' counters summed for each prime by four chains of additions, plus those of its block's words before its word (8 words, masked by a table: a mask formed from the index, a select, is turned back by the compiler into a loop to the index) and of its word up to it. So the leaves of a prime are independent of one another, and those of `p_b·m` are gathered first, from the table of least prime factors and Möbius values of the numbers prime to 2310 (primecount's), with no branch either; the primes below 400 cross out their multiples with no counts and the counters are formed again from the bits, the others take each bit off its block's counter.
- `P2(x, a) = Σ (π(x/p) − π(p) + 1)` over the primes `p` from `y` to `√x`: `π` at the first `x/p` by `prime_pi` itself, and the others from a sieve of the numbers from there to `x/y`, counted up to each `x/p` in turn, the primes `p` taken from the top down from sieves of stretches of `2^22`.
- The sums are of 64-bit numbers that wrap: each term is below `2^64` and the result is below `2^63`, so that it is exact although the parts on the way may not fit.

Lucy Hedgehog's method (2013), `O(x^(3/4))` with `O(√x)` memory, was measured too: never the fastest, 3.3 µs at `10^5` where the sieve takes 1.2 µs, 380 µs at `10^8` where Deléglise and Rivat's takes 29 µs, so `prime_pi` does not use it.

## The n-th prime

`nth_prime(n)` reads the table up to `π(2^16) = 6542`, and up to 18,000, about `π(2·10^5)`, sieves from `2^16` on. Above, it finds `x = R⁻¹(n)` by Newton's steps on Riemann's `R(x)`, from Cipolla's `n (ln n + ln ln n − 1 + (ln ln n − 2)/ln n)`, `R` by Gram's series `1 + Σ (ln x)^k / (k·k!·ζ(k + 1))`, whose terms are all positive: within a few times `√x / ln x` primes of the `n`-th. Then `π(x)` by `prime_pi`, and the sieve from `x` up or down to the prime, in windows of twice the expected distance (at least 4096 numbers), each read once upwards, as primecount does (it refines `R` by the zeros of `ζ` above `10^9`, which shortens a sieve that takes a small part of the time). Within `10^7` primes of the last, it sieves down from `2^64 − 1`, whose `π` is known, instead.

## Measured

Measured on an Apple M4 Max with other work running on the machine (the load average was 25 to 50 from other processes), single-threaded, interleaved with the references in the same process, the median of 5 runs (3 for the longest), in time per call; primesieve 12.16 and primecount 8.8 with one thread (`primesieve_set_num_threads(1)`, `primecount_set_num_threads(1)`), primesieve with its default sieve of 512 KiB (it reads the efficiency cores' caches here; 256 KiB to 2 MiB gave the same times within 2 % for `[0, 10^10]` and `[0, 10^11]`, 128 KiB 6 to 12 % more).

Counting and forming the primes of a range, `primes(a..=b).count()` against `primesieve_count_primes`, the primes formed in order by `next` and summed by `fold` (`sum`) against `primesieve_next_prime`, measured again with nothing else running:

| Range | count | primesieve | `next` | `fold` | primesieve iterator |
| --- | --- | --- | --- | --- | --- |
| `[0, 10^9]` | 38.4 ms | 42.9 ms | 89.2 ms | 76.7 ms | 85.0 ms |
| `[0, 10^10]` | 521 ms | 552 ms | 968 ms | 853 ms | 932 ms |
| `[10^12, 10^12 + 10^10]` | 912 ms | 1.05 s | 1.31 s | 1.22 s | 1.41 s |
| `[10^15, 10^15 + 10^10]` | 1.83 s | 2.58 s | | | |
| `[10^18, 10^18 + 10^10]` | 3.20 s | 3.87 s | | | |
| `[10^19, 10^19 + 10^9]` | 1.35 s | 1.53 s | | | |

- Counting is 6 to 29 % faster than primesieve, the algorithm being its own: the large primes' buckets and the medium primes' loops, specialised for each class and position by the compiler instead of jumped into by a `switch`, take less time per prime, most where they dominate, from `10^15` up. A segment of 2 MiB with parts of 128 KiB for the small primes (the first-level cache) measured 13 to 17 % faster than primesieve's 512 KiB with 64 KiB, from `10^12` up.
- `next` is within 5 % of primesieve's iterator, which keeps its batch the same way; `fold` reads a batch with no index kept in memory between primes, 9 to 14 % faster.

Short ranges far up and the first primes from far up, where the start of the full sieve dominates:

| Range | `primes(a..=b).count()` | primesieve |
| --- | --- | --- |
| `[10^15, 10^15 + 10^6]` | 8.3 ms | 8.4 ms |
| `[10^18, 10^18 + 10^6]` | 21.9 ms | 181 ms |
| `[10^18, 10^18 + 10^8]` | 298 ms | 350 ms |
| `[2^64 − 10^6, 2^64)` | 23.1 ms | 786 ms |
| `[2^64 − 10^7, 2^64)` | 210 ms | 809 ms |
| `[2^64 − 10^8, 2^64)` | 921 ms | 1.00 s |

| The first primes from `a` | `primes(a..).take(k)` | primesieve iterator |
| --- | --- | --- |
| 10 from `0` | 0.6 µs | 0.0 µs |
| 10 from `10^6` | 1.6 µs | 0.8 µs |
| 10 from `10^9` | 15.6 µs | 19.9 µs |
| 100 from `10^12` | 92 µs | 647 µs |
| 100 from `10^18` | 534 µs | 381 ms |
| 100 from `2^64 − 10^9` | 555 µs | 1.55 s |

`prime_pi(x)` against primecount's `primecount_pi` (Xavier Gourdon's 2001 refinement of Deléglise and Rivat's method, its default), the median of 7 runs to `10^16` (of loops of calls below `10^12`), 3 at `10^17` and 2 at `10^18`, one at `2^64 − 1`; primecount's own Deléglise–Rivat (`primecount x -d -t1`, from the shell) and PARI/GP 2.19's `primepi` (a sieve) for comparison:

| x | `prime_pi` | primecount | primecount, Deléglise–Rivat | PARI/GP |
| --- | --- | --- | --- | --- |
| `10^5` | 1.25 µs | 0.70 µs | | |
| `10^6` | 2.85 µs | 1.80 µs | | |
| `10^7` | 9.4 µs | 6.8 µs | | |
| `10^8` | 28.7 µs | 31.2 µs | | 16 ms |
| `10^10` | 405 µs | 411 µs | | 137 ms |
| `10^12` | 5.9 ms | 6.7 ms | 18 ms | 88 s |
| `10^14` | 88.6 ms | 97.5 ms | 337 ms | |
| `10^15` | 351 ms | 372 ms | | |
| `10^16` | 1.42 s | 1.45 s | 5.98 s | |
| `10^17` | 5.93 s | 5.72 s | | |
| `10^18` | 25.3 s | 22.6 s | | |
| `2^64 − 1` | 161 s, 83 MB | 129 s, 101 MB | | |

- From `10^8` to `10^16`, 2 to 12 % faster than primecount, whose Gourdon's method is about four times as fast as its Deléglise–Rivat here: the easy leaves by the swapped sum and the hard leaves' independent counts make up the difference. Above, primecount gains: at `10^18` our hard leaves took 16.2 s (primecount's corresponding `D` 12.4 s) and the easy ones 5.9 s (its `A + C` 5.8 s), and P2 2.6 s (its `B` 3.9 s): Gourdon's refinement leaves less work to its hard part at these sizes, which no tuning of ours made up (segments of 64 or 128 KiB, the bound of the primes crossed out with no counts, a loop with no setup for the primes with few multiples, `α` from 0.7 to 2 times primecount's: none faster by more than 2 %). Implementing Gourdon's method is the way to go further.
- Below `10^8`, the method's tables and sieves cost more than primecount's Meissel's and Legendre's formulas, which it takes there: 1.4 to 1.8 times as long, a few microseconds.
- FLINT 3.6's `n_prime_pi` sieves: 1.6 ms at `10^6` and 127 ms at `10^8`.

`nth_prime(n)` against primecount's `primecount_nth_prime`, measured again with nothing else running:

| n | `nth_prime` | primecount |
| --- | --- | --- |
| `10^4` | 1.35 µs | 1.83 µs |
| `10^6` | 12.5 µs | 12.0 µs |
| `10^9` | 829 µs | 862 µs |
| `10^12` | 44.4 ms | 44.8 ms |
| `10^14` | 716 ms | 732 ms |

Both are their `π` at the estimate and a short sieve. PARI/GP's `prime(n)` answers at once up to `n = 10^11` (it keeps checkpoints of `π` and sieves from the nearest), and took more than four minutes for `10^12`; FLINT's `n_nth_prime` sieves from 2: 2.9 s at `10^8`, 35 s at `10^9`.

## Examples

Rust usage: the primes of ranges, counts and the n-th prime:

```rust
use a::math::numbers::{nth_prime, prime_pi, primes};

fn main() {
  assert_eq!(primes(10..=30).collect::<Vec<_>>(), [11, 13, 17, 19, 23, 29]);
  assert_eq!(primes(..100).count(), 25);
  // The primes around 10^12, and the largest below 2^64.
  assert_eq!(primes(1_000_000_000_000..).next(), Some(1_000_000_000_039));
  assert_eq!(primes(u64::MAX - 100..).last(), Some(u64::MAX - 58));
  // pi(x) and the n-th prime.
  assert_eq!(prime_pi(1_000_000_000), 50_847_534);
  assert_eq!(prime_pi(10_000_000_000_000), 346_065_536_839);
  assert_eq!(nth_prime(1), 2);
  assert_eq!(nth_prime(1_000_000), 15_485_863);
  assert_eq!(nth_prime(425_656_284_035_217_743), u64::MAX - 58);
  // The sum of the primes below 10^6.
  assert_eq!(primes(..1_000_000).sum::<u64>(), 37_550_402_023);
}
```

## Efficiency

- Time complexity: the sieve `O(n log log n)` for `n` numbers, and the start `O(√b / log b)` for a range ending at `b` (a stretch narrower than `√a / 90` `O(w / log a)` tests of about `log a` products instead); `prime_pi` `O(x^(2/3) / log² x)`; `nth_prime` that of `prime_pi` at the prime, and a sieve of about `√p` numbers
- Space complexity: the sieve one segment of up to 2 MiB, a batch of 8 KiB, and 8 bytes for each sieving prime that has a multiple in the range; `prime_pi` `O(x^(1/3) log³ x)`: near `2^64`, 83 MB in all, most of it the table of factors (`2·480/2310` bytes a number up to `y ≈ 10^8`), the primes and the table of `π` up to `y`
- Soundness: the sieve's unrolled loops write without bounds checks within a stated bound, each with a `SAFETY` comment, and Miri runs the tests on a 64-bit, a 32-bit and a big-endian target; every other access is checked
