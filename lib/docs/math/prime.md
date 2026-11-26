[A](../../../README.md) › [Math](../../../README.md#math)

# Primes

[Numbers](numbers.md) tell whether they are prime with `a.is_probable_prime()`, by the Baillie–PSW test, give the next prime above them with `a.next_prime()`, and the previous one below them with `a.prev_prime()`. The primes of a range of `u64`s, their number and the n-th prime are on [their own page](sieve.md).

- `is_probable_prime` is `false` for every number below 2, negative numbers included (GMP's `mpz_probab_prime_p` takes the absolute value of a negative number; this does not), `true` for every prime, and `false` for every composite number it finds a factor of or that fails the test.
- It is exact below `2^64`: Feitsma and Galway's list of all the base-2 Fermat pseudoprimes below `2^64` (2009), which Gilchrist checked, has no number that passes both halves of the test. Above, no composite number is known that passes it, though such numbers are believed to exist: it is a probable-prime test, the one GMP 6.2 and later run for `mpz_probab_prime_p(n, 24)`.
- `next_prime` gives the smallest probable prime above `a`, in the byte order of `a`, in one new vector of exactly its length: 2 for everything below 2, negative numbers included. It is exact below `2^64`, as the test is.
- `prev_prime` gives the largest probable prime below `a`, in the byte order of `a`, in one new vector of exactly its length, or `None` when there is none: for everything up to 2, negative numbers included. It is exact below `2^64`, as the test is; GMP 6.3's `mpz_prevprime` returns 0 for the same inputs.
- Each magnitude is read in its own byte order, and zero bytes above its most significant non-zero byte are ignored; an empty vector is zero, and a magnitude of zero is zero whatever `negative` says. The number is left as it was.

## Baillie–PSW

The test is Baillie and Wagstaff's (1980) after Pomerance, Selfridge and Wagstaff: a strong probable-prime test to base 2 and a strong Lucas probable-prime test with Selfridge's parameters, whose pseudoprimes seem to have nothing in common. This implementation carries it out as follows:

- Trial division first: the long number is reduced modulo products of consecutive small primes that fit in a limb, by the division's two-limbs-by-one steps with a reciprocal, four products at a time in independent chains, then each prime of a product is tested on the remainder by one multiplication, as `x·p^(−1) mod 2^64 <= (2^64 − 1)/p` exactly when `p` divides `x`. The primes go up to four times the number's bits, from 256 to 8192, which balances the passes against the tests they save (measured: random odd numbers of 512 bits took 16 % less time than with the primes up to twice their bits); below `2^64`, up to 256, and a number below `257²` with no factor is prime.
- The strong test to base 2: `n − 1 = d·2^s` with `d` odd, `2^d mod n` from the top bit of `d` with a square for each bit and a doubling, an addition, for each one, as GMP's `mpn_2powm` does, then up to `s − 1` squares; `n` passes when `2^d = 1` or one of the squares is `−1`. Almost every composite stops here.
- Selfridge's parameters (method A): the first `D` of `5, −7, 9, −11, 13, …` whose Jacobi symbol `(D/n)` is `−1`, `P = 1` and `Q = (1 − D)/4`. The symbols come from one remainder of `n` by `9·5·7·11·…·47`, by quadratic reciprocity and the binary method on limbs (factors of two taken out, then the larger odd number less the smaller); a `D` beyond 47 takes a pass of its own. A square's symbols are never `−1`, so squares are told apart before the search: odd squares are 1 modulo 8 and squares modulo each prime to 47, which leaves the square root, by Newton's steps, for the few numbers that pass, about 1 in 65,000 of the others.
- The strong Lucas test: `n + 1 = d·2^s`, and `n` passes when `U_d = 0` or one of `V_(d·2^r)`, `r < s`, is 0. `U_d` and `U_(d−1)` come from the top bit of `d` with three squares a bit, as GMP's `mpz_lucas_mod` does after Hackman: `U_2k = U_(k+1)² − (U_(k+1) − U_k)²`, `U_(2k+1) = U_(k+1)² − Q·U_k²`, and for a one bit `U_(2k+2) = U_(2k+1) − Q·U_2k`, the products by `Q` by additions for `|Q| <= 4`; then `V_d = U_d − 2Q·U_(d−1)`, `Q^d = (V_d² − D·U_d²)/4`, and `V_2k = V_k² − 2Q^k`. For `D = 5`, half of all numbers, `Q = −1` and `Q^k = ±1`, so the ladder of `V_k` and `V_(k+1)` takes one square and one product a bit instead, as GMP's `mpn_strongfibo` does with the Fibonacci numbers: 9–13 % less time for primes of 256 to 1024 bits.
- Montgomery's form throughout, the [powers](pow.md)' arithmetic: one limb in registers; two to eight limbs in registers on aarch64, by kernels in inline assembly (the coarsely integrated operand scanning, and squares that take each product of two different limbs once), which made primes of 512 bits twice as fast, and two to seven limbs on the other targets, by rows in Rust, where eight limbs stay in memory (with the kernels turned off, the rows took 1.03 to 1.04 times as long as the residues in memory for numbers of 512 bits, at a clock of 3.9 GHz, and about as long at 4.4 GHz); nine to sixteen limbs by the kernels of residues in memory on aarch64, the window of the rows in registers (1024-bit primes 1.4 times as fast); beyond, and from eight limbs on the other targets, the multiplication's kernels with the REDC kernel (two rows a pass on aarch64), and REDC by products from 86 limbs; the residues always reduced below `n`, so that equal residues are equal limbs. GMP takes Montgomery's form for the strong test only, and reduces the Lucas sequence by divisions.

## Below 2^32

A number below `2^32` with no prime factor up to 256, and at least `257²`, takes the strong test to base 2 alone instead of Baillie–PSW, and a look-up in a table of the composites that pass it: of the 2314 strong pseudoprimes to base 2 below `2^32` (from Feitsma and Galway's list of the base-2 pseudoprimes, 2009), the 1971 with no prime factor up to 256, 7.9 KB, found by a binary search among those with the same top byte. Every prime passes the test and none is in the table, so the answer is exact, as Baillie–PSW's is below `2^64`; FLINT's `n_is_prime` takes the same way from `2^20` to `2^32`. The table was computed again for this crate (the strong tests to the bases 2 to 17 of every odd number below `2^32` without such a factor) and agrees with FLINT's, and all the odd numbers below `2^32` were checked against a sieve. The power `2^d` starts from the top six bits of `d` by a shift, then takes a square for each bit, reduced by Barrett's method with `⌊(2^64 − 1)/n⌋` (a product and one correction), and a doubling, an addition, for each one. Measured on 2026-11-11 on an Apple M4 Max on a quiet machine with the trial division before (the load average at 1.7, one thread, `MallocNanoZone` unset, fat LTO; interleaved runs in rotated order, the clock read around each, the median of 11): primes of 32 bits take 415 cycles, against 1287 by Baillie–PSW (0.2.32) and 639 by the strong tests to the bases 2, 7 and 61 side by side (measured earlier, with other work on the machine), which no composite below 4,759,123,141 passes (Jaeschke, "On strong pseudoprimes to several bases", 1993); the composites that pass the trial division 386, 494 and 643; random odd numbers 126, 218 and 164; `next_prime` and `prev_prime` from random numbers of 32 bits 1228 and 1262 cycles instead of 2223 and 2256 by Baillie–PSW (FLINT's `n_is_prime` and `n_nextprime` are measured below). The strong tests to 2, 7 and 61 were faster than Baillie–PSW at every size from 20 to 32 bits but for the composites that pass the trial division (Baillie–PSW took 0.71 to 0.78 of their time there), and the strong test to base 2 in Montgomery's form, as Baillie–PSW's, took 1.2 times as long as by Barrett's products.

## Next prime

The odd numbers above `a` are sieved in intervals by the small primes, and those left are tested in order, after GMP's `mpz_nextprime`: the start's remainders by the products of the primes give where each prime's odd multiples fall in the interval, those are crossed out, and the numbers left take the test without its trial division. The primes go up to GMP's estimate of where sieving more costs more than the tests it saves, `bits^2.5 / 124` (at least 500 and at most `2^24`; 270,000 for 1024 bits): the table's below 8192, whose products' remainders are taken only for the products with a prime below the bound (from 64 to about 250 bits the bound is below 8192), and beyond from the table of the primes below `2^16` that [the sieve](sieve.md) keeps, or from the sieve itself above (about 580 bits up), where they came from a sieve of Eratosthenes for each call; an interval holds five odd numbers a bit, about fourteen times the average gap between primes, so that a second interval is rare. The test is cheaper than GMP's, so at the small end the bound is lower than GMP's 1000: measured on an Apple M4 Max with the kernels of Montgomery's products and without, the next and the previous prime from random numbers took 5 to 7 % less time with 500 at 65 and 72 bits and 2 to 3 % at 80, where the best bounds were about 500, 500 and 650 (1000 from 112 bits); none of the divisors from 62 to 496 was faster than 124 by more than 1.2 % from 128 to 2048 bits, and 62 and 496 were up to 8 % slower. Below the last prime below `2^64`, the search steps through the numbers prime to 30 with the test of numbers below `2^64`.

## Previous prime

The same search, downwards, as GMP 6.3's `mpz_prevprime`: the odd numbers below `a` are sieved in intervals by the small primes, with the next prime's limits and lengths, the start's remainders by the products of the primes giving where each prime's odd multiples fall below it, and those left are tested from the largest down, without the test's trial division. A candidate can be one limb shorter than the start, never two, as a prime comes long before (at the latest `2^64 + 13`, the first prime above `2^64`), so the test's working memory is planned for both lengths. From `2^64 + 13` down, the search steps down through the numbers prime to 30 with the test of numbers below `2^64`.

## Measured

Measured on 2026-11-26 on an Apple M4 Max with nothing else running, by the harness in the repository's `bench/nt` (its method in `bench/README.md`), on one thread: this crate as a user builds it, by rustc 1.100 stable in cargo's release profile (no LTO, 16 codegen units, no `target-cpu`); the Rust references in the same profile, machine-prime 1.5.7 (with `wide`), primal 0.3.3 and malachite 0.15.0; the C references as Homebrew builds them, GMP 6.3.0, FLINT 3.6.0 and PARI 2.19.1, called through shims compiled with `-O3 -mcpu=native`. Each figure is the median of 15 samples of at least a millisecond (fewer for the calls of seconds), in time per call, this crate and every reference alternating sample by sample in one process, each sample on the same numbers for all, after a check that all give the same answers. A ratio is this crate's time over that of the fastest reference that does the same work: below 1, this crate is faster.

`is_probable_prime` against the Baillie–PSW tests of GMP (`mpz_probab_prime_p(n, 24)`), FLINT (`fmpz_is_probabprime_BPSW`) and PARI (`ispseudoprime(n, 0)`), and below `2^64` also against the exact tests of words, FLINT's `n_is_prime`, machine-prime's `is_prime`, primal's `is_prime` and malachite's `u64::is_prime`; random primes, random odd composites and random odd numbers of exactly the bits given, the fastest reference of each:

| Bits | primes | fastest | ratio | odd composites | fastest | ratio | random odd | fastest | ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 92.1 ns | FLINT `n_is_prime` 77.7 ns | 1.18 | 16.7 ns | FLINT `n_is_prime` 13.0 ns | 1.29 | 26.9 ns | FLINT `n_is_prime` 21.8 ns | 1.24 |
| 64 | 590 ns | machine-prime 386 ns | 1.53 | 44.3 ns | machine-prime 30.6 ns | 1.45 | 70.8 ns | machine-prime 50.1 ns | 1.41 |
| 128 | 2.97 µs | FLINT 6.77 µs | 0.439 | 166 ns | GMP 548 ns | 0.303 | 228 ns | GMP 784 ns | 0.291 |
| 256 | 11.4 µs | FLINT 25.8 µs | 0.443 | 488 ns | GMP 1.61 µs | 0.302 | 555 ns | GMP 1.79 µs | 0.310 |
| 512 | 54.6 µs | FLINT 90.0 µs | 0.606 | 2.10 µs | GMP 5.22 µs | 0.402 | 2.93 µs | GMP 7.89 µs | 0.372 |
| 1024 | 350 µs | FLINT 536 µs | 0.653 | 10.1 µs | GMP 21.1 µs | 0.477 | 14.9 µs | GMP 32.2 µs | 0.463 |
| 2048 | 2.94 ms | FLINT 3.50 ms | 0.839 | 104 µs | GMP 139 µs | 0.751 | 90.6 µs | GMP 144 µs | 0.629 |
| 4096 | 27.2 ms | FLINT 23.6 ms | 1.15 | 859 µs | GMP 1.10 ms | 0.779 | 559 µs | GMP 745 µs | 0.751 |

- From 128 to 2048 bits 0.291 to 0.839 of the fastest reference's time: primes, which take the whole test, 0.439 to 0.839 (FLINT), and composites and random odd numbers, which mostly stop at the trial division or the strong test to base 2, 0.291 to 0.751 (GMP). Against GMP 0.0637 to 0.876 at every size, against PARI 0.140 to 0.870, and against FLINT's Baillie–PSW from 0.0622 (odd composites of 1024 bits) to 1.15 (primes of 4096 bits).
- Below `2^64` the other tests of words are slower: this crate takes 0.0454 to 0.331 of primal's time and 0.197 to 0.798 of malachite's.
- At 128 bits machine-prime's `is_prime_128` is faster (this test takes 1.31 to 1.59 times its time), but it is a strong test to base 2 and a Lucas test on `V` alone, weaker than Baillie–PSW's strong Lucas test, so it is not the same test and is not counted.

`next_prime` and `prev_prime` from random numbers of the bits given, against FLINT's `fmpz_nextprime` and, at 32 and 64 bits, its `n_nextprime`, GMP's `mpz_prevprime`, and PARI's `nextprime` and `precprime`. GMP's `mpz_nextprime` tests the prime it finds once more, by a Miller–Rabin test to a random base: it does more, and is shown beside the ratios, not in them. Each sample takes the same starting numbers for all, and the times vary with the gaps to the primes:

| Bits | `next_prime` | fastest | ratio | GMP's `mpz_nextprime` (ratio) | `prev_prime` | fastest | ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 278 ns | FLINT `n_nextprime` 232 ns | 1.20 | 1.86 µs (0.150) | 286 ns | PARI 558 ns | 0.512 |
| 64 | 1.55 µs | FLINT `n_nextprime` 1.31 µs | 1.18 | 4.78 µs (0.324) | 1.51 µs | PARI 3.14 µs | 0.479 |
| 128 | 9.04 µs | FLINT 31.0 µs | 0.292 | 31.2 µs (0.290) | 9.44 µs | GMP 32.7 µs | 0.289 |
| 256 | 43.4 µs | FLINT 127 µs | 0.341 | 128 µs (0.338) | 45.3 µs | GMP 133 µs | 0.342 |
| 512 | 300 µs | FLINT 695 µs | 0.432 | 697 µs (0.431) | 322 µs | GMP 719 µs | 0.447 |
| 1024 | 1.65 ms | FLINT 3.08 ms | 0.537 | 3.16 ms (0.523) | 1.99 ms | GMP 3.80 ms | 0.523 |
| 2048 | 15.5 ms | FLINT 19.7 ms | 0.788 | 19.4 ms (0.798) | 26.6 ms | GMP 32.2 ms | 0.826 |
| 4096 | 1.53 s | FLINT 1.85 s | 0.826 | 1.86 s (0.824) | 1.21 s | GMP 1.45 s | 0.832 |

From 128 bits `next_prime` takes 0.292 to 0.826 of FLINT's time and `prev_prime` 0.289 to 0.832 of GMP's, from the test's kernels; PARI's take longer at every size (`next_prime` 0.142 to 0.500 of its time, `prev_prime` 0.146 to 0.512).

Where the references are faster, this crate's time over the fastest same-work reference's above 1.05:

- **`is_probable_prime` of words.** At 32 bits 1.18 to 1.29 times FLINT's `n_is_prime`, which takes fewer trial divisions before the strong test to base 2 than the primes to 256 here; both then take that test and the table of its pseudoprimes (above). At 64 bits 1.41 to 1.53 times machine-prime's `is_prime` (and 1.03 to 1.49 times FLINT's `n_is_prime`), by the algorithm: from `2^32` this is Baillie–PSW, a strong test and a strong Lucas test, about three strong tests' work, where both take one or two strong tests to bases chosen by a hash of the number (after Forišek and Jančina), exact below `2^64` by a table of bases (128 to 512 KB) that this crate does not carry.
- **`next_prime` of words.** 1.20 and 1.18 times FLINT's `n_nextprime` at 32 and 64 bits: each candidate left by the sieve takes the word test above, and the result its own vector.
- **Primes of 4096 bits.** 1.15 times FLINT's time (GMP's 0.876, PARI's 0.870): at 64 limbs the reduction is still the quadratic one of Montgomery's form (REDC by products from 86 limbs), where FLINT's Lucas test reduces by Barrett's method on its subquadratic products.

## Examples

Rust usage: primes, pseudoprimes, the next prime and the previous one:

```rust
use a::math::numbers::{ByteOrder, Number};

fn main() {
  // 2^127 - 1 is prime; 2^128 + 1 = 59649589127497217 * 5704689200685129054721 is not.
  let mut m127 = vec![0xFFu8; 16];
  m127[0] = 0x7F;
  assert!(Number::new(m127, ByteOrder::BigEndian).is_probable_prime());
  let mut f7 = vec![0u8; 17];
  f7[0] = 1;
  f7[16] = 1;
  assert!(!Number::new(f7, ByteOrder::BigEndian).is_probable_prime());

  // Strong pseudoprimes to base 2, a Carmichael number and a strong Lucas pseudoprime: composite.
  for n in [2047u64, 3215031751, 3825123056546413051, 561, 5459] {
    assert!(!Number::new_from_u64(n, ByteOrder::LittleEndian).is_probable_prime());
  }
  assert!(!Number::new_from_i32(-7, ByteOrder::BigEndian).is_probable_prime());

  // The first prime above 2^64 is 2^64 + 13.
  let p = Number::new_from_u64(u64::MAX, ByteOrder::BigEndian).next_prime();
  let mut expected = vec![0u8; 9];
  expected[0] = 1;
  expected[8] = 13;
  assert_eq!(p.value, expected);
  assert_eq!(Number::new_from_i32(-100, ByteOrder::BigEndian).next_prime().value, vec![2]);

  // The largest prime below 2^64 is 2^64 - 59, the one below 2^64 + 13 too; there is none below 2.
  let mut above = vec![0u8; 9];
  above[0] = 1;
  above[8] = 13;
  let q = Number::new(above, ByteOrder::BigEndian).prev_prime().expect("a prime below");
  assert_eq!(q.value, (u64::MAX - 58).to_be_bytes());
  assert!(Number::new_from_u8(2, ByteOrder::LittleEndian).prev_prime().is_none());
  assert!(Number::new_from_i32(-7, ByteOrder::BigEndian).prev_prime().is_none());
}
```

## Efficiency

- Time complexity: trial division in `O(n·t)` for `n` limbs and `t` products of primes; the test in about `b` squares for the strong test and `2b` to `3b` for the Lucas test, for `b` bits, each `O(n²)` below 86 limbs and about 2.5 products of `n` limbs beyond (2.7 from the transforms' threshold); the next and the previous prime about `g / ln(P)` tests and one remainder by each product of primes per interval, for a gap `g` to it and primes up to `P`, whatever the byte order
- Space complexity: the number's limbs and ten residues, on the stack up to 8 KiB, one temporary buffer on the heap beyond, and one more for each product of 360 limbs or more, which the [multiplication](mul.md) computes by transforms; the next and the previous prime the interval's flags and the primes beyond the table; tables of the 1027 odd primes below 8192 and of their products, computed at compile time
- Soundness: the bytes beyond a vector's length, which may be uninitialised, are never read, and every byte of a result is written before the vector's length is set; the number is left as it was
