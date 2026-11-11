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
- The primes are formed from each 64-bit word of the sieve from both ends at once, by the trailing zeros of the word and of its bits reversed, the lowest and the highest left two at a time into their places in a batch of up to 1024: two chains of `bits &= bits − 1` side by side, each half as long as one (from 3.3 to about 2.4 cycles a prime). The iterator's `next` reads the batch. The counts are the words' population counts.
- The sieving primes come from the table of the primes below `2^16` computed at compile time when they are below it (ranges up to `2^32`), and else from a sieve of their own, whose own primes come from that table.

### Short ranges far up

The sieve's start costs a division and a place for each prime up to the square root of the range: about 4 ns for each of the 203 million primes below `2^32` near `2^64`. A range narrower than about `√a / 90` is cheaper to sieve with the primes up to a quarter of its width only and to test the numbers left by Baillie–PSW, which they leave about one in `ln(width)/0.56`, the primes among them costing 0.6 µs each and the others 70 ns: the two meet there. `Primes` therefore sieves a range in stretches: the first 4096 numbers, enough for a hundred primes even near `2^64`, then four times as many each time, each stretch narrower than `√a / 90` sieved by the smaller primes and tested, and the rest of the range at once by the full sieve. Below `2^40`, where a stretch's sieving primes take a third of a millisecond at most to start, the full sieve's stretches grow the same way while they are narrower than a segment (63 million numbers): the first segment of the whole rest is sized by its end, 2 MiB for an end near `2^64`, which took 1.7 ms for the first ten primes from 0 before this rule and takes 0.6 µs with it. `count` and `fold`, which take every prime, plan the rest of the range as one stretch. primesieve sieves every range in full, from a table below 720 and in chunks of about `√a` that grow fourfold.

## Prime counting

`prime_pi(x)` reads a table below `2^16` and above uses Deléglise and Rivat's method ("Computing π(x): the Meissel, Lehmer, Lagarias, Miller, Odlyzko method", 1996), in the combinatorial form of Oliveira e Silva ("Computing π(x): the combinatorial method", 2006) and of Walisch's primecount, with its special leaves sorted as Xavier Gourdon's ("Computation of π(x): improvements to the Meissel, Lehmer, Lagarias, Miller, Odlyzko, Deléglise and Rivat method", 2001): with `y = α·x^(1/3)` and `a = π(y)`,

`π(x) = φ(x, a) + a − 1 − P2(x, a)`,

where `φ(x, a)`, the numbers up to `x` with no prime factor up to `y`, is split by the leaves of Legendre's recursion into the ordinary leaves `S1` and the special ones `S2` (trivial, easy and hard), and `P2` counts the numbers up to `x` with exactly two prime factors above `y`. `α` is primecount's fit for the method, a polynomial in `ln x`, from 2.6 at `10^9` to 17.8 at `10^16` and 38 near `2^64`: our measurements, with the leaves sorted as below, found no better (at `10^17`, 0.85 to 1.45 times it: the hard leaves grow with `y`).

- Below `2^16`, a bit for each number prime to 30 and a count for each 240, and the primes below `2^16`, are tables computed at compile time (4.4 KB and 26 KB). The method takes its primes up to `y` and its table of `π` up to `y` from them while `y` is below `2^16` (up to `x` of about `9.7·10^11`), and `P2`'s counts while `x/y` is (up to about `6.2·10^7`), so that small `x` allocate and sieve nothing but the hard leaves' segment: `π(x)` takes 0.24 µs just above `2^16`, where the sieve's count it took before took 0.76 µs and primecount's Legendre's formula takes 0.46 µs, and 3.3 µs at `10^7`, where primecount's Meissel's formula takes 6.3 µs.
- `φ(x, a)` for `a ≤ 8` is `(x / P)·φ(P) + φ(x mod P, a)`, `P = 2·3·5·7·11·13·17·19` taken as `510510` and a division by 19, from tables of the numbers prime to the first primes computed at compile time.
- The ordinary leaves `S1`: the square-free `n ≤ y` with prime factors above 19, by a recursion over the primes.
- The trivial leaves, whose `φ` is 1, are counted, by an arithmetic progression for the primes above `x^(1/3)`.
- A special leaf `p_b·m` whose `x/(p_b m)` is below `p_b^2` has `φ(x/(p_b m), b − 1) = π(x/(p_b m)) − b + 2`, as no number below `p_b^2` but 1 has all its prime factors from `p_b` up but the primes: it is easy, its `φ` a look-up in a table of `π`. Gourdon sorts the leaves so; Deléglise and Rivat's method takes as easy only those up to `y`, where its table of `π` ends, and sieves the others. So for the primes `p_b` above `√y`, whose leaves have prime `m`, only leaves at least `p_b^2` are hard here (the primes up to `√y` keep all their leaves in the sieve, those below `p_b^2`, Gourdon's `C1`, included), and as `m > p_b`, the primes with hard leaves are at most `x^(1/4)`: the primes above `x^(1/4)` (95 % of the sieve's primes at `10^18`) never cross out the sieve's numbers. The leaves `p_b·q`, `q` prime, between `y` and `p_b^2` are the easy leaves above `y`: at `10^18`, 0.65 of the 2.3·10^9 leaves Deléglise and Rivat's method sieves, at `2^64 − 1` 3.2 of 12.3·10^9.
- The easy leaves up to `y`, `p_b·q` with `x/(p_b q) ≤ y`, whose `φ(x/(p_b q), b − 1)` is `π(x/(p_b q)) − b + 2`, from the table of `π` up to `y` (a bit for each number prime to 30 and a count for each 240). Those above `√(x/p_b)` come in runs that share one value, which primecount sums run by run, each run's end found from its value: a chain of a division and a look-up after another, which leaves the processor waiting (at `10^16`, 3.2·10^8 runs took most of the 3.5 s primecount's form of the method spends on these leaves). Here they are summed by swapping the order of the pairs of primes `(q, r)` with `q·r ≤ N = x/p_b`: `Σ_{Q<q≤T} π(N/q) = π(N/T)·(π(T) − π(Q)) + Σ_{N/T<r≤N/Q} (π(N/r) − π(Q))`, whose primes `r` run over the same primes as the other easy leaves, `q` from `z/p_b` to `√N`, so that one division and one look-up serve both sums, and no term waits for another. The look-ups `π(N/q)`, runs of consecutive primes `q` for each `p_b`, are made by batches of 4096 primes, in increasing order of their values, a part of 64 KiB of the table at a time when the table is larger than two parts; two at a time, the divisor known not to be zero, the block's index kept to the table (no branch), and the two words' bits counted by one count of a 128-bit vector: 0.72 ns a look-up, 0.83 ns one at a time (`x = 10^18`, 7.4·10^9 look-ups; the divisions themselves take 0.51 ns each). Dividing by reciprocals of the primes kept in a table (`⌊(2^64 − 1)/q⌋`, a product and a correction) was 25 % faster alone, but 30 % slower in place, from the 7 MB of reciprocals the look-ups then read.
- The easy leaves above `y` need `π` from `y` to `√x`: from the table below `2^16` when `√x` is below it, and else from a sieve of the numbers from `y` up, in parts of 4096 words (983,040 numbers), with the count of the bits before each word, which stay in the first-level cache; the leaves of a prime, its primes `q` from the largest down, come in the order of their values, so that each part takes the next leaves of each prime, its bound found by one division and one look-up.
- The hard leaves, `x/n ≥ y` and, for `n = p_b·q`, `x/n ≥ p_b^2`, need `φ(x/n, b − 1)` of numbers up to `z = x/y`: a segmented sieve of the numbers from 0 to `z` (a bit for each number prime to 30, segments of 64 KiB, which measured 2 % faster than 128 KiB) is crossed out by the primes in order, the leaves of each prime read before its multiples are crossed out, `φ` being the numbers left before the segment, kept for each prime, and those left in the segment up to the leaf. primecount counts the latter from the last leaf's point, by counters of blocks of the segment and the words after the last block; that count, carried from leaf to leaf, ends its loops at points the processor mispredicts. Here each leaf's count is formed afresh, with no branch: the bits before its block of 64 bytes, from the blocks' counters summed for each prime, plus those of its block's words before its word and of its word up to it. On aarch64 with NEON these are vector kernels, with a portable version elsewhere and under Miri: the sums before the blocks eight a step in four chains (three shifted additions within the eight), the block's 64 bytes masked by a table and counted by bytes, one sum for all; and the leaves `p_b·m`, `m` square-free with its prime factors above `p_b`, are gathered from the table of least prime factors and Möbius values of the numbers prime to 2310 (primecount's) eight entries at a time, by one comparison and a table of the positions of each mask's set bits, as primecount's NEON code does. The sums before the blocks are prepared only once a prime has a leaf in the segment: near the end of the sieve a prime's range of `m` holds less than one number in most segments, and at `10^18` 10.3 of the 17 million preparations found no leaf and 3.9 million one to three; a prime with at most 8 leaves in one chunk of entries sums the blocks' counters for each leaf instead. The primes below 400 cross out their multiples with no counts and the counters are formed again from the bits; the others take each bit off its block's counter, the eight multiples of a turn of the wheel at once with one check of the segment's end. At `10^18` the hard leaves took 8.40 s with all this, against 15.1 s before (0.2.32, whose hard leaves included those now easy above `y`).
- `P2(x, a) = Σ (π(x/p) − π(p) + 1)` over the primes `p` from `y` to `√x`: from the table when `x/y` is below `2^16`; else `π` at the first `x/p` by `prime_pi` itself, and the others from a sieve of the numbers from there to `x/y`, counted up to each `x/p` in turn, the primes `p` taken from the top down from sieves of stretches of `2^22`.
- The sums are of 64-bit numbers that wrap: each term is below `2^64` and the result is below `2^63`, so that it is exact although the parts on the way may not fit.

Lucy Hedgehog's method (2013), `O(x^(3/4))` with `O(√x)` memory, was measured too: never the fastest, 3.3 µs at `10^5`, where this method takes 0.28 µs, 380 µs at `10^8`, where it takes 16 µs, so `prime_pi` does not use it.

## The n-th prime

`nth_prime(n)` reads the table up to `π(2^16) = 6542`, and up to 18,000, about `π(2·10^5)`, sieves from `2^16` on. Above, it finds `x = R⁻¹(n)` by Newton's steps on Riemann's `R(x)`, from Cipolla's `n (ln n + ln ln n − 1 + (ln ln n − 2)/ln n)`, `R` by Gram's series `1 + Σ (ln x)^k / (k·k!·ζ(k + 1))`, whose terms are all positive (`ζ(s)` from a table to 20 and `1 + 2^−s` above): within a few times `√x / ln x` primes of the `n`-th. Then `π(x)` by `prime_pi`, and the sieve from `x` up or down to the prime, in windows of twice the expected distance (at least 4096 numbers), each read once, as primecount does (it refines `R` by the zeros of `ζ` above `10^9`, which shortens a sieve that takes a small part of the time); a window down is counted and the prime read from the bytes of its last segment, which still holds them, rather than sieved again. Within `10^7` primes of the last, it sieves down from `2^64 − 1`, whose `π` is known, instead.

## Measured

Measured on 2026-11-11 on an Apple M4 Max on a quiet machine (the load average at 1.3 to 1.9 of its 16 cores), single-threaded, every process with `MallocNanoZone` unset, the harnesses built with fat LTO and one codegen unit; interleaved in the same process with this crate before the work of 2026-11-08 (0.2.32, "before") and with the references, the median of 7 runs (5, 3 or 2 for the longest), in time per call; primesieve 12.16 and primecount 8.8 with one thread (`primesieve_set_num_threads(1)`, `primecount_set_num_threads(1)`).

Counting and forming the primes of a range, `primes(a..=b).count()` against `primesieve_count_primes`, the primes formed in order by `next` and summed by `fold` (`sum`) against `primesieve_next_prime`:

| Range | count | before | primesieve | `next` | before | `fold` | primesieve iterator |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `[0, 10^9]` | 38.3 ms | 37.9 ms | 43.1 ms | 81.0 ms | 88.3 ms | 69.3 ms | 84.6 ms |
| `[0, 10^10]` | 520 ms | 513 ms | 554 ms | 907 ms | 968 ms | 803 ms | 930 ms |
| `[10^12, 10^12 + 10^10]` | 909 ms | 911 ms | 1.05 s | 1.28 s | 1.31 s | 1.19 s | 1.41 s |
| `[10^15, 10^15 + 10^10]` | 1.84 s | 1.84 s | 2.59 s | | | | |
| `[10^18, 10^18 + 10^10]` | 3.24 s | 3.20 s | 3.88 s | | | | |
| `[10^19, 10^19 + 10^9]` | 1.36 s | 1.35 s | 1.54 s | | | | |

- Counting takes 0.71 to 0.94 of primesieve's time, as before (1.00 to 1.01 times its time before): the sieve is primesieve's, its large primes' buckets and medium primes' loops specialised for each class and position by the compiler (a loop with one check of the end a turn of the wheel, as the small primes', measured 8 to 15 % slower for the medium ones), a segment of 2 MiB with parts of 128 KiB for the small primes.
- `next` takes 0.91 to 0.97 of the time of primesieve's iterator (0.93 to 1.04 before), `fold` 0.82 to 0.86: the primes formed from both ends of each word, two chains side by side.

Short ranges far up, where the start of the full sieve dominates (the code is unchanged: 0.99 to 1.01 times its time before, in the same runs):

| Range | `primes(a..=b).count()` | primesieve |
| --- | --- | --- |
| `[10^15, 10^15 + 10^6]` | 7.87 ms | 7.81 ms |
| `[10^18, 10^18 + 10^6]` | 19.0 ms | 165 ms |
| `[10^18, 10^18 + 10^8]` | 282 ms | 316 ms |
| `[2^64 − 10^6, 2^64)` | 20.0 ms | 714 ms |
| `[2^64 − 10^7, 2^64)` | 182 ms | 738 ms |
| `[2^64 − 10^8, 2^64)` | 868 ms | 909 ms |

| The first primes from `a` | `primes(a..).take(k)` | before | primesieve iterator |
| --- | --- | --- | --- |
| 10 from `0` | 0.22 µs | 0.50 µs | 0.08 µs |
| 10 from `10^6` | 0.91 µs | 1.41 µs | 0.70 µs |
| 10 from `10^9` | 10.5 µs | 12.9 µs | 15.4 µs |
| 100 from `10^12` | 86.7 µs | 87.0 µs | 608 µs |
| 100 from `10^18` | 466 µs | 468 µs | 366 ms |
| 100 from `2^64 − 10^9` | 487 µs | 487 µs | 1.45 s |

The first primes below `2^16` come from the table of the primes computed at compile time, with no sieve to start; what is left from 0 is mostly the allocation of the batch, 8 KiB. From `10^6` the first stretch's sieving primes are placed, a division each, and crossed out over 4096 numbers, for ten primes: primesieve's iterator, which keeps its small sieving primes, is faster there.

`prime_pi(x)` against primecount's `primecount_pi` (Xavier Gourdon's method, its default, below `10^5` Legendre's formula and to `10^8` Meissel's), loops of calls with the argument changing up to `10^10`:

| x | `prime_pi` | before | primecount | to primecount |
| --- | --- | --- | --- | --- |
| `3·10^4` | 3.5 ns | 3.5 ns | 11.6 ns | 0.31 |
| `7·10^4` | 0.24 µs | 0.76 µs | 0.46 µs | 0.52 |
| `10^5` | 0.28 µs | 1.05 µs | 0.70 µs | 0.41 |
| `2·10^5` | 0.40 µs | 1.39 µs | 0.83 µs | 0.48 |
| `10^6` | 1.04 µs | 2.39 µs | 1.69 µs | 0.61 |
| `10^7` | 3.34 µs | 7.02 µs | 6.27 µs | 0.53 |
| `10^8` | 16.0 µs | 25.8 µs | 27.4 µs | 0.58 |
| `10^9` | 63.1 µs | 93.7 µs | 97.9 µs | 0.64 |
| `10^10` | 248 µs | 363 µs | 392 µs | 0.63 |
| `10^12` | 3.86 ms | 5.11 ms | 6.11 ms | 0.63 |
| `10^14` | 60.9 ms | 78.5 ms | 88.2 ms | 0.69 |
| `10^15` | 244 ms | 319 ms | 344 ms | 0.71 |
| `10^16` | 988 ms | 1.30 s | 1.35 s | 0.73 |
| `10^17` | 4.02 s | 5.42 s | 5.28 s | 0.76 |
| `10^18` | 16.4 s | 23.0 s | 20.8 s | 0.79 |
| `2^64 − 1` | 99.0 s, 83 MB | 161 s, 83 MB | 122 s, 104 MB | 0.81 |

- Below `10^8` the tables of the primes and of `π` below `2^16` leave the method with no table to build and no sieve but the hard leaves': 1.6 to 3.7 times faster than before, 0.41 to 0.61 of primecount's Legendre's and Meissel's formulas.
- Above, the hard leaves are fewer (Gourdon's sorting: at `10^18` 1.6 of the 2.3·10^9 that Deléglise and Rivat's method sieves, the primes above `x^(1/4)` not crossing out the sieve) and cheaper (the NEON kernels, the sums before the blocks prepared only for a prime with leaves in the segment), and the easy leaves' look-ups are two at a time: from `10^9` up, 0.63 to 0.81 of primecount's time, the share growing with `x` from `10^12`, and 0.67 to 0.78 of the time before up to `10^18`. At `10^18` (timed copies of both versions, one run each) the hard leaves took 8.40 s instead of 15.1 s, the easy ones 5.07 s instead of 5.51 s, `P2` 2.38 s (2.38 s before), the easy leaves above `y` 0.52 s; primecount's `D` (its hard leaves) took 11.6 s there, `A + C` 5.56 s and `B` 3.59 s.
- At `2^64 − 1` (one run each, not interleaved; "before" is the earlier measurement of 2026-11-07, with other work on the machine, the load average about 25, not measured again; primecount's time is its own total): from a timed copy, a second run, `P2` 14.7 s, the easy leaves 31.9 s, those above `y` 2.71 s, the hard leaves 49.6 s (64.9 s, in an earlier run with other work on the machine, before the preparations were made only for primes with leaves), 99.1 s in all; primecount's `D` 66.3 s, `A + C` 35.0 s, `B` 20.0 s, at 122 s in all.
- FLINT 3.6's `n_prime_pi` sieves: 4.15 ms at `10^6` and 124 ms at `10^8`, the first call (it keeps what it sieved: a call again takes 0.08 and 0.04 µs); PARI/GP's `primepi` took 14 ms at `10^8` and 18 ms at `10^9` (gp's clock counts whole milliseconds), and 88 s at `10^12` in an earlier measurement with other work on the machine.

`nth_prime(n)` against primecount's `primecount_nth_prime`:

| n | `nth_prime` | before | primecount | to primecount |
| --- | --- | --- | --- | --- |
| `10^4` | 1.05 µs | 1.35 µs | 1.82 µs | 0.58 |
| `3·10^4` | 1.28 µs | 4.06 µs | 2.37 µs | 0.54 |
| `10^5` | 2.25 µs | 6.07 µs | 3.16 µs | 0.71 |
| `10^6` | 6.45 µs | 12.4 µs | 11.7 µs | 0.55 |
| `10^7` | 29.1 µs | 52.9 µs | 45.0 µs | 0.65 |
| `10^9` | 557 µs | 811 µs | 785 µs | 0.71 |
| `10^12` | 32.8 ms | 44.5 ms | 44.9 ms | 0.73 |

Both are their `π` at the estimate and a short sieve; here the sieve's primes come from the table below `2^16`, its list of small primes is reserved at once (its growth took 45 % of placing the primes), `R` takes no powers, and a window down is sieved once. PARI/GP's `prime(n)` answers at once up to `n = 10^11` (it keeps checkpoints of `π` and sieves from the nearest; under a millisecond, gp's clock reading 0 ms, at `10^9` and `10^11`), and took more than four minutes for `10^12` in an earlier measurement with other work on the machine; FLINT's `n_nth_prime` sieves from 2: 2.30 s at `10^8`, 28.0 s at `10^9`.

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
