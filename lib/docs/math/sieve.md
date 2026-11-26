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
- The primes are formed from each 64-bit word of the sieve from both ends at once, by the trailing zeros of the word and of its bits reversed, the lowest and the highest left two at a time into their places in a batch of up to 1024: two chains of `bits &= bits − 1` side by side, each half as long as one (from 3.3 to about 2.4 cycles a prime), the first eight places from each end without a test, so that a word of up to 16 primes (about `240 / ln x` near `x`: 17.4 near `10^6`, 11.6 near `10^9`, 5.4 near `2^64`) does not pay the mispredicted end of the loop. The iterator's `next` reads the batch. The counts are the words' population counts.
- The sieving primes come from the table of the primes below `2^16` computed at compile time when they are below it (ranges up to `2^32`), and else from a sieve of their own, whose own primes come from that table.

### Short ranges far up

The sieve's start costs a division and a place for each prime up to the square root of the range: about 4 ns for each of the 203 million primes below `2^32` near `2^64`. A range narrower than about `√a / 90` is cheaper to sieve with the primes up to a quarter of its width only and to test the numbers left by Baillie–PSW, which they leave about one in `ln(width)/0.56`, the primes among them costing 0.6 µs each and the others 70 ns: the two meet there. `Primes` therefore sieves a range in stretches: the first 4096 numbers, enough for a hundred primes even near `2^64`, then four times as many each time, each stretch narrower than `√a / 90` sieved by the smaller primes and tested, and the rest of the range at once by the full sieve. Below `2^40`, where a stretch's sieving primes take a third of a millisecond at most to start, the full sieve's stretches grow the same way while they are narrower than a segment (63 million numbers): the first segment of the whole rest is sized by its end, 2 MiB for an end near `2^64`, which took 1.7 ms for the first ten primes from 0 before this rule and takes 167 ns with it (below). `count` and `fold`, which take every prime, plan the rest of the range as one stretch. primesieve sieves every range in full, from a table below 720 and in chunks of about `√a` that grow fourfold.

## Prime counting

`prime_pi(x)` reads a table below `2^16`, from `2^16` to `2^20` a second one, and above uses Deléglise and Rivat's method ("Computing π(x): the Meissel, Lehmer, Lagarias, Miller, Odlyzko method", 1996), in the combinatorial form of Oliveira e Silva ("Computing π(x): the combinatorial method", 2006) and of Walisch's primecount, with its special leaves sorted as Xavier Gourdon's ("Computation of π(x): improvements to the Meissel, Lehmer, Lagarias, Miller, Odlyzko, Deléglise and Rivat method", 2001): with `y = α·x^(1/3)` and `a = π(y)`,

`π(x) = φ(x, a) + a − 1 − P2(x, a)`,

where `φ(x, a)`, the numbers up to `x` with no prime factor up to `y`, is split by the leaves of Legendre's recursion into the ordinary leaves `S1` and the special ones `S2` (trivial, easy and hard), and `P2` counts the numbers up to `x` with exactly two prime factors above `y`. `α` is primecount's fit for the method, a polynomial in `ln x`, from 2.6 at `10^9` to 17.8 at `10^16` and 38 near `2^64`: our measurements, with the leaves sorted as below, found no better (at `10^17`, 0.85 to 1.45 times it: the hard leaves grow with `y`).

- Below `2^16`, a bit for each number prime to 30 and a count for each 240, and the primes below `2^16`, are tables computed at compile time (4.4 KB and 26 KB). From `2^16` to `2^20` (PARI's default `primelimit`, whose table of primes PARI reads for `primepi` and `prime`), the primes (82,027 of 4 bytes, 328 KB) and the same blocks from them (70 KB) are built from the sieve once 128 calls of `prime_pi` or `nth_prime` have asked there, and kept for the process: `π(x)` is then a look-up, 1.92 ns at `7·10^4` (below), and the `n`-th prime a read of the table; a program that calls them a few times never builds them. The method takes its primes up to `y` and its table of `π` up to `y` from them while `y` is below `2^16` (up to `x` of about `9.7·10^11`), and `P2`'s counts while `x/y` is (up to about `6.2·10^7`), so that small `x` allocate and sieve nothing but the hard leaves' segment: `π(x)` takes 3.50 µs at `10^7`, where primecount takes 7.70 µs.
- `φ(x, a)` for `a ≤ 8` is `(x / P)·φ(P) + φ(x mod P, a)`, `P = 2·3·5·7·11·13·17·19` taken as `510510` and a division by 19, from tables of the numbers prime to the first primes computed at compile time.
- The ordinary leaves `S1`: the square-free `n ≤ y` with prime factors above 19, by a recursion over the primes.
- The trivial leaves, whose `φ` is 1, are counted, by an arithmetic progression for the primes above `x^(1/3)`.
- A special leaf `p_b·m` whose `x/(p_b m)` is below `p_b^2` has `φ(x/(p_b m), b − 1) = π(x/(p_b m)) − b + 2`, as no number below `p_b^2` but 1 has all its prime factors from `p_b` up but the primes: it is easy, its `φ` a look-up in a table of `π`. Gourdon sorts the leaves so; Deléglise and Rivat's method takes as easy only those up to `y`, where its table of `π` ends, and sieves the others. So for the primes `p_b` above `√y`, whose leaves have prime `m`, only leaves at least `p_b^2` are hard here (the primes up to `√y` keep all their leaves in the sieve, those below `p_b^2`, Gourdon's `C1`, included), and as `m > p_b`, the primes with hard leaves are at most `x^(1/4)`: the primes above `x^(1/4)` (95 % of the sieve's primes at `10^18`) never cross out the sieve's numbers. The leaves `p_b·q`, `q` prime, between `y` and `p_b^2` are the easy leaves above `y`: at `10^18`, 0.65 of the 2.3·10^9 leaves Deléglise and Rivat's method sieves, at `2^64 − 1` 3.2 of 12.3·10^9.
- The easy leaves up to `y`, `p_b·q` with `x/(p_b q) ≤ y`, whose `φ(x/(p_b q), b − 1)` is `π(x/(p_b q)) − b + 2`, from the table of `π` up to `y` (a bit for each number prime to 30 and a count for each 240). Those above `√(x/p_b)` come in runs that share one value, which primecount sums run by run, each run's end found from its value: a chain of a division and a look-up after another, which leaves the processor waiting (at `10^16`, 3.2·10^8 runs took most of the 3.5 s primecount's form of the method spends on these leaves). Here they are summed by swapping the order of the pairs of primes `(q, r)` with `q·r ≤ N = x/p_b`: `Σ_{Q<q≤T} π(N/q) = π(N/T)·(π(T) − π(Q)) + Σ_{N/T<r≤N/Q} (π(N/r) − π(Q))`, whose primes `r` run over the same primes as the other easy leaves, `q` from `z/p_b` to `√N`, so that one division and one look-up serve both sums, and no term waits for another. The look-ups `π(N/q)`, runs of consecutive primes `q` for each `p_b`, are made by batches of 4096 primes, in increasing order of their values, a part of 64 KiB of the table at a time when the table is larger than two parts; two at a time, the divisor known not to be zero, the block's index kept to the table (no branch), and the two words' bits counted by one count of a 128-bit vector: 0.72 ns a look-up, 0.83 ns one at a time (`x = 10^18`, 7.4·10^9 look-ups; the divisions themselves take 0.51 ns each). Dividing by reciprocals of the primes kept in a table (`⌊(2^64 − 1)/q⌋`, a product and a correction) was 25 % faster alone, but 30 % slower in place, from the 7 MB of reciprocals the look-ups then read.
- The easy leaves above `y` need `π` from `y` to `√x`: from the table below `2^16` when `√x` is below it, and else from a sieve of the numbers from `y` up, in parts of 4096 words (983,040 numbers), with the count of the bits before each word, which stay in the first-level cache; the leaves of a prime, its primes `q` from the largest down, come in the order of their values, so that each part takes the next leaves of each prime, its bound found by one division and one look-up.
- The hard leaves, `x/n ≥ y` and, for `n = p_b·q`, `x/n ≥ p_b^2`, need `φ(x/n, b − 1)` of numbers up to `z = x/y`: a segmented sieve of the numbers from 0 to `z` (a bit for each number prime to 30, segments of 64 KiB, which measured 2 % faster than 128 KiB) is crossed out by the primes in order, the leaves of each prime read before its multiples are crossed out, `φ` being the numbers left before the segment, kept for each prime, and those left in the segment up to the leaf. primecount counts the latter from the last leaf's point, by counters of blocks of the segment and the words after the last block; that count, carried from leaf to leaf, ends its loops at points the processor mispredicts. Here each leaf's count is formed afresh, with no branch: the bits before its block of 64 bytes, from the blocks' counters summed for each prime, plus those of its block's words before its word and of its word up to it. On aarch64 with NEON these are vector kernels, with a portable version elsewhere and under Miri: the sums before the blocks eight a step in four chains (three shifted additions within the eight), the block's 64 bytes masked by a table and counted by bytes, one sum for all; and the leaves `p_b·m`, `m` square-free with its prime factors above `p_b`, are gathered from the table of least prime factors and Möbius values of the numbers prime to 2310 (primecount's) eight entries at a time, by one comparison and a table of the positions of each mask's set bits, as primecount's NEON code does. The sums before the blocks are prepared only once a prime has a leaf in the segment: near the end of the sieve a prime's range of `m` holds less than one number in most segments, and at `10^18` 10.3 of the 17 million preparations found no leaf and 3.9 million one to three; a prime with at most 8 leaves in one chunk of entries sums the blocks' counters for each leaf instead. The primes below 400 cross out their multiples with no counts and the counters are formed again from the bits; the others take each bit off its block's counter, the eight multiples of a turn of the wheel at once with one check of the segment's end. At `10^18` the hard leaves took 8.40 s with all this, against 15.1 s before (0.2.32, whose hard leaves included those now easy above `y`).
- `P2(x, a) = Σ (π(x/p) − π(p) + 1)` over the primes `p` from `y` to `√x`: from the table when `x/y` is below `2^16`; else `π` at the first `x/p` by `prime_pi` itself, and the others from a sieve of the numbers from there to `x/y`, counted up to each `x/p` in turn, the primes `p` taken from the top down from sieves of stretches of `2^22`.
- The sums are of 64-bit numbers that wrap: each term is below `2^64` and the result is below `2^63`, so that it is exact although the parts on the way may not fit.

Lucy Hedgehog's method (2013), `O(x^(3/4))` with `O(√x)` memory, was measured too: never the fastest, 3.3 µs at `10^5` and 380 µs at `10^8`, where `prime_pi` takes 1.91 ns (from the table) and 17.7 µs, so `prime_pi` does not use it.

## The n-th prime

`nth_prime(n)` reads the table up to `π(2^16) = 6542`, and the table of the primes below `2^20` up to `π(2^20) = 82,025` once it is there (before, up to 18,000, about `π(2·10^5)`, it sieves from `2^16` on). Above, it finds `x = R⁻¹(n)` by Newton's steps on Riemann's `R(x)`, from Cipolla's `n (ln n + ln ln n − 1 + (ln ln n − 2)/ln n)`, `R` by Gram's series `1 + Σ (ln x)^k / (k·k!·ζ(k + 1))`, whose terms are all positive (`ζ(s)` from a table to 20 and `1 + 2^−s` above): within a few times `√x / ln x` primes of the `n`-th. Then `π(x)` by `prime_pi`, and the sieve from `x` up or down to the prime, in windows of twice the expected distance (at least 4096 numbers), each read once, as primecount does (it refines `R` by the zeros of `ζ` above `10^9`, which shortens a sieve that takes a small part of the time); a window down is counted and the prime read from the bytes of its last segment, which still holds them, rather than sieved again. Within `10^7` primes of the last, it sieves down from `2^64 − 1`, whose `π` is known, instead.

## Measured

Measured on 2026-11-26 on an Apple M4 Max with nothing else running, by the harness in the repository's `bench/nt` (its method in `bench/README.md`), on one thread: this crate as a user builds it, by rustc 1.100 stable in cargo's release profile (no LTO, 16 codegen units, no `target-cpu`); the Rust references in the same profile, primal 0.3.3 and malachite 0.15.0; the C references as Homebrew builds them, primesieve 12.18 and primecount 9.0 set to one thread, FLINT 3.6.0 and PARI 2.19.1, called through shims compiled with `-O3 -mcpu=native`. Each figure is the median of 15 samples of at least a millisecond (fewer for the longest calls, 3 at the least), in time per call, this crate and every reference alternating sample by sample in one process, after a check that all give the same counts, sums and primes. A ratio is this crate's time over that of the fastest reference that does the same work: below 1, this crate is faster.

Counting the primes of a range, `primes(a..=b).count()`, against `primesieve_count_primes`, and forming them in order, summed by `next` and by `fold` (`sum`), against primesieve's iterator (`primesieve_next_prime`; `fold`'s ratio is against the iterator timed beside it, 70.2 ms, 811 ms and 1.28 s):

| Range | `count` | primesieve | ratio | `next` | primesieve iterator | ratio | `fold` | ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `[0, 10^6]` | 15.6 µs | 14.3 µs | 1.09 | | | | | |
| `[0, 10^8]` | 2.98 ms | 2.99 ms | 0.996 | | | | | |
| `[0, 10^9]` | 43.3 ms | 44.5 ms | 0.973 | 74.8 ms | 70.0 ms | 1.07 | 63.6 ms | 0.906 |
| `[0, 10^10]` | 582 ms | 575 ms | 1.01 | 867 ms | 813 ms | 1.07 | 768 ms | 0.947 |
| `[10^12, 10^12 + 10^10]` | 1.00 s | 1.09 s | 0.917 | 1.24 s | 1.28 s | 0.972 | 1.17 s | 0.915 |
| `[10^19, 10^19 + 10^9]` | 1.36 s | 1.58 s | 0.862 | | | | | |

- Counting takes 0.862 to 1.01 of primesieve's time from `[0, 10^8]` on: the sieve is primesieve's, its large primes' buckets and its medium primes' loops specialised for each class and position, a segment of 2 MiB with parts of 128 KiB for the small primes. `fold` takes 0.906 to 0.947 of the iterator's time, the primes formed from both ends of each word; `next` 0.972 to 1.07.
- The other sieves are slower: this crate takes 0.209 to 0.476 of the time of primal's `StreamingSieve::prime_pi` and 0.0411 to 0.109 of malachite's `primes_less_than_or_equal_to(b).count()`; forming the primes of `[0, 10^9]`, 0.441 of primal's (`Sieve::new(b).primes_from(0)`, its whole sieve kept, 33 MB) and 0.192 of malachite's.

Short ranges far up, where the start of the full sieve dominates and the shorter stretches are sieved by the small primes and tested (see above):

| Range | `primes(a..=b).count()` | primesieve | ratio |
| --- | --- | --- | --- |
| `[10^15, 10^15 + 10^6]` | 7.57 ms | 7.97 ms | 0.949 |
| `[10^18, 10^18 + 10^6]` | 18.5 ms | 171 ms | 0.108 |
| `[10^18, 10^18 + 10^8]` | 276 ms | 328 ms | 0.841 |
| `[2^64 − 10^6, 2^64)` | 19.5 ms | 739 ms | 0.0263 |
| `[2^64 − 10^7, 2^64)` | 177 ms | 760 ms | 0.233 |
| `[2^64 − 10^8, 2^64)` | 853 ms | 935 ms | 0.912 |

The first primes from a point, against a new primesieve iterator from it (`jump_to`) and, from the first three points, primal's `Primes::all()`, which sieves from 2:

| The first primes from `a` | `primes(a..).take(k)` | primesieve iterator | ratio | primal |
| --- | --- | --- | --- | --- |
| 10 from `0` | 167 ns | 322 ns | 0.520 | 81.5 µs |
| 10 from `10^6` | 1.07 µs | 1.09 µs | 0.982 | 219 µs |
| 10 from `10^9` | 10.7 µs | 12.2 µs | 0.880 | 158 ms |
| 100 from `10^12` | 72.4 µs | 258 µs | 0.280 | – |
| 100 from `10^18` | 436 µs | 138 ms | 0.0032 | – |
| 100 from `2^64 − 10^9` | 456 µs | 617 ms | 0.0007 | – |

The first primes below `2^16` come from the table of the primes computed at compile time, with no sieve to start. From `10^6` the first stretch's sieving primes are placed, a division each, and crossed out over 4096 numbers for ten primes, about as long as primesieve's iterator takes, which keeps its small sieving primes; far up, where primesieve sieves the whole stretch of `√a` numbers, the stretches tested by Baillie–PSW take a fraction of its time.

`prime_pi(x)` against primecount's `primecount_pi` and PARI's `primepi`, each sample over 64 arguments from `x` to 6 % above it up to `10^10` (at `10^6` some of them are above `2^20`), and over `x` itself beyond:

| x | `prime_pi` | primecount | PARI | ratio | fastest |
| --- | --- | --- | --- | --- | --- |
| `3·10^4` | 1.41 ns | 199 ns | 11.3 ns | 0.125 | PARI |
| `7·10^4` | 1.92 ns | 475 ns | 12.1 ns | 0.159 | PARI |
| `10^5` | 1.91 ns | 669 ns | 11.6 ns | 0.165 | PARI |
| `2·10^5` | 1.93 ns | 808 ns | 11.6 ns | 0.167 | PARI |
| `10^6` | 258 ns | 1.64 µs | 1.31 µs | 0.197 | PARI |
| `10^7` | 3.50 µs | 7.70 µs | 2.94 ms | 0.454 | primecount |
| `10^8` | 17.7 µs | 30.3 µs | 16.7 ms | 0.585 | primecount |
| `10^9` | 69.0 µs | 110 µs | 42.3 ms | 0.629 | primecount |
| `10^10` | 276 µs | 430 µs | 333 ms | 0.642 | primecount |
| `10^12` | 3.86 ms | 5.24 ms | – | 0.738 | primecount |
| `10^13` | 15.4 ms | 19.5 ms | – | 0.790 | primecount |
| `10^14` | 61.3 ms | 72.7 ms | – | 0.842 | primecount |
| `10^15` | 248 ms | 278 ms | – | 0.892 | primecount |
| `10^16` | 1.01 s | 1.08 s | – | 0.935 | primecount |

- Below `2^20` both this crate and PARI read tables: PARI's of the primes below its `primelimit`, made at its start, and here those of the counts above, built once 128 calls have asked there; 1.41 to 1.93 ns a call, 0.125 to 0.167 of PARI's time.
- From `10^7` the method, 0.454 to 0.935 of primecount's time, the margin narrowing as `x` grows: primecount 9.0 chooses a larger `y` than the `α` above gives, which moves work from `P2` to the leaves, and this crate's `α` was not fitted again for it.
- FLINT's `n_prime_pi` sieves, and keeps what it sieved: its first call (after `flint_cleanup`) took 48.4 µs at `3·10^4` and 110 ms at `10^8`; primal's `StreamingSieve::prime_pi` 1.64 µs at `3·10^4` and 1.30 s at `10^10`; PARI's `primepi` beyond its table 2.94 ms at `10^7` and 333 ms at `10^10`.
- `prime_pi` from `10^17` to `2^64 − 1`, and the counts of `[10^15, 10^15 + 10^10]` and `[10^18, 10^18 + 10^10]`, calls of seconds to minutes, were not part of this run.

`nth_prime(n)` against primecount's `primecount_nth_prime` and PARI's `prime`, each sample over arguments up to 6 % above `n`:

| n | `nth_prime` | primecount | PARI | ratio | fastest |
| --- | --- | --- | --- | --- | --- |
| `10^4` | 3.59 ns | 1.81 µs | 6.76 ns | 0.531 | PARI |
| `3·10^4` | 3.59 ns | 2.39 µs | 6.79 ns | 0.529 | PARI |
| `10^5` | 3.25 µs | 3.39 µs | 97.4 µs | 0.959 | primecount |
| `10^6` | 11.6 µs | 14.4 µs | 586 µs | 0.804 | primecount |
| `10^7` | 44.8 µs | 52.7 µs | 6.29 ms | 0.851 | primecount |
| `10^9` | 583 µs | 711 µs | – | 0.820 | primecount |
| `10^12` | 33.3 ms | 38.3 ms | – | 0.869 | primecount |
| `10^14` | 559 ms | 594 ms | – | 0.941 | primecount |

Up to `π(2^20)` a read of the table, as PARI's; above, both this crate and primecount take `π` at an estimate and a short sieve, 0.804 to 0.959 of primecount's time. PARI's `prime` beyond its table sieves from the last prime it found, 97.4 µs at `10^5` and 6.29 ms at `10^7`; FLINT's `n_nth_prime` sieves from 2, 1.40 ms at `10^5` and 229 ms at `10^7`; primal's `StreamingSieve::nth_prime` 4.69 µs at `10^4` and 12.2 ms at `10^7`.

Where the references are faster, this crate's time over the fastest reference's above 1.05:

- **Counting `[0, 10^6]`.** 1.09 times primesieve's time, 15.6 µs against 14.3: the start of a short range, the segment and the pre-sieve's tables set up, against primesieve's.
- **`next` over `[0, 10^9]` and `[0, 10^10]`.** 1.07 times primesieve's iterator. What `next` pays over `fold` (0.906 and 0.947), about a cycle a prime, is the iterator's own: each call loads the batch's index and length and stores the index, as the iterator's state escapes into the call that fills the batch, which is not inlined; a pair of pointers would save a load, but would put raw pointers into a public type, and was not done.

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
- Space complexity: the sieve one segment of up to 2 MiB, a batch of 8 KiB, and 8 bytes for each sieving prime that has a multiple in the range; `prime_pi` `O(x^(1/3) log³ x)`: near `2^64`, 83 MB in all (the largest resident set), most of it the table of factors (`2·480/2310` bytes a number up to `y ≈ 10^8`), the primes and the table of `π` up to `y`
- Soundness: the sieve's unrolled loops write without bounds checks within a stated bound, each with a `SAFETY` comment; the hard leaves' NEON kernels (the sums before the blocks, the gathering of the leaves' entries, the counts of a leaf) read within lengths asserted once a call, the leaf's word kept to the last by a minimum rather than checked, with portable versions under the same names elsewhere and under Miri; the easy leaves' look-ups in the table of `π` read their block unchecked, its index kept to the last block by a minimum; Miri runs the tests on a 64-bit, a 32-bit and a big-endian target, and the NEON kernels' tests run on Linux aarch64 as well as macOS; every other access is checked
