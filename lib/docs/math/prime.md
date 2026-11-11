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
- Montgomery's form throughout, the [powers](pow.md)' arithmetic: one limb in registers; two to eight limbs in registers on aarch64, by kernels in inline assembly (the coarsely integrated operand scanning, and squares that take each product of two different limbs once), which made primes of 512 bits twice as fast, and two to seven limbs on the other targets, by rows in Rust, where eight limbs stay in memory (with the kernels turned off, the rows took 1.03 to 1.04 times as long as the residues in memory for numbers of 512 bits, at a clock of 3.9 GHz, and about as long at 4.4 GHz); nine to sixteen limbs by the kernels of residues in memory on aarch64, the window of the rows in registers (1024-bit primes 1.4 times as fast); beyond, and from eight limbs on the other targets, the multiplication's kernels with the REDC kernel (two rows a pass on aarch64), and REDC by products from 100 limbs; the residues always reduced below `n`, so that equal residues are equal limbs. GMP takes Montgomery's form for the strong test only, and reduces the Lucas sequence by divisions.

## Below 2^32

A number below `2^32` with no prime factor up to 256, and at least `257²`, takes the strong test to base 2 alone instead of Baillie–PSW, and a look-up in a table of the composites that pass it: of the 2314 strong pseudoprimes to base 2 below `2^32` (from Feitsma and Galway's list of the base-2 pseudoprimes, 2009), the 1971 with no prime factor up to 256, 7.9 KB, found by a binary search among those with the same top byte. Every prime passes the test and none is in the table, so the answer is exact, as Baillie–PSW's is below `2^64`; FLINT's `n_is_prime` takes the same way from `2^20` to `2^32`. The table was computed again for this crate (the strong tests to the bases 2 to 17 of every odd number below `2^32` without such a factor) and agrees with FLINT's, and all the odd numbers below `2^32` were checked against a sieve. The power `2^d` starts from the top six bits of `d` by a shift, then takes a square for each bit, reduced by Barrett's method with `⌊(2^64 − 1)/n⌋` (a product and one correction), and a doubling, an addition, for each one. Measured on 2026-11-11 on an Apple M4 Max on a quiet machine with the trial division before (the load average at 1.7, one thread, `MallocNanoZone` unset, fat LTO; interleaved runs in rotated order, the clock read around each, the median of 11): primes of 32 bits take 415 cycles, against 1287 by Baillie–PSW (0.2.32) and 639 by the strong tests to the bases 2, 7 and 61 side by side (measured earlier, with other work on the machine), which no composite below 4,759,123,141 passes (Jaeschke, "On strong pseudoprimes to several bases", 1993); the composites that pass the trial division 386, 494 and 643; random odd numbers 126, 218 and 164; FLINT's `n_is_prime` 356, 276 and 108; `next_prime` and `prev_prime` from random numbers of 32 bits 1228 and 1262 cycles instead of 2223 and 2256 by Baillie–PSW (FLINT's `n_nextprime` 1058). The strong tests to 2, 7 and 61 were faster than Baillie–PSW at every size from 20 to 32 bits but for the composites that pass the trial division (Baillie–PSW took 0.71 to 0.78 of their time there), and the strong test to base 2 in Montgomery's form, as Baillie–PSW's, took 1.2 times as long as by Barrett's products.

## Next prime

The odd numbers above `a` are sieved in intervals by the small primes, and those left are tested in order, after GMP's `mpz_nextprime`: the start's remainders by the products of the primes give where each prime's odd multiples fall in the interval, those are crossed out, and the numbers left take the test without its trial division. The primes go up to GMP's estimate of where sieving more costs more than the tests it saves, `bits^2.5 / 124` (at least 500 and at most `2^24`; 270,000 for 1024 bits): the table's below 8192, whose products' remainders are taken only for the products with a prime below the bound (from 64 to about 250 bits the bound is below 8192), and beyond from the table of the primes below `2^16` that [the sieve](sieve.md) keeps, or from the sieve itself above (about 580 bits up), where they came from a sieve of Eratosthenes for each call; an interval holds five odd numbers a bit, about fourteen times the average gap between primes, so that a second interval is rare. The test is cheaper than GMP's, so at the small end the bound is lower than GMP's 1000: measured on an Apple M4 Max with the kernels of Montgomery's products and without, the next and the previous prime from random numbers took 5 to 7 % less time with 500 at 65 and 72 bits and 2 to 3 % at 80, where the best bounds were about 500, 500 and 650 (1000 from 112 bits); none of the divisors from 62 to 496 was faster than 124 by more than 1.2 % from 128 to 2048 bits, and 62 and 496 were up to 8 % slower. Below the last prime below `2^64`, the search steps through the numbers prime to 30 with the test of numbers below `2^64`.

## Previous prime

The same search, downwards, as GMP 6.3's `mpz_prevprime`: the odd numbers below `a` are sieved in intervals by the small primes, with the next prime's limits and lengths, the start's remainders by the products of the primes giving where each prime's odd multiples fall below it, and those left are tested from the largest down, without the test's trial division. A candidate can be one limb shorter than the start, never two, as a prime comes long before (at the latest `2^64 + 13`, the first prime above `2^64`), so the test's working memory is planned for both lengths. From `2^64 + 13` down, the search steps down through the numbers prime to 30 with the test of numbers below `2^64`.

## Measured

Measured on 2026-11-11 on an Apple M4 Max on a quiet machine (the load average at 1.6 to 1.8 of its 16 cores, the clock at 4.2 to 4.5 GHz; one thread, every process with `MallocNanoZone` unset, the harnesses built with fat LTO and one codegen unit; interleaved runs, the median of seven, in cycles per call, rounded), `is_probable_prime` against GMP 6.3's `mpz_probab_prime_p(n, 24)`, for random primes, random odd composites and random odd numbers of exactly the bits given:

| Bits | primes: LE | BE | GMP | odd composites: LE | BE | GMP | random odd: LE | BE | GMP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 64 | 2538 | 2544 | 12,611 | 209 | 209 | 291 | 270 | 269 | 654 |
| 128 | 12,902 | 12,706 | 50,674 | 653 | 652 | 2270 | 827 | 829 | 3627 |
| 256 | 49,258 | 49,340 | 170,831 | 1911 | 1897 | 6262 | 2613 | 2612 | 8954 |
| 512 | 234,942 | 233,980 | 655,929 | 10,513 | 10,466 | 29,237 | 8334 | 8421 | 22,947 |
| 1024 | 1.60 M | 1.61 M | 3.55 M | 54,991 | 55,138 | 96,356 | 65,286 | 67,249 | 167,046 |
| 2048 | 16.1 M | 16.1 M | 21.6 M | 573,683 | 580,425 | 678,577 | 1.15 M | 1.15 M | 1.33 M |

And on primes alone against FLINT 3.6's `fmpz_is_probabprime_BPSW` and PARI/GP 2.19's `ispseudoprime` (the library's, called from C), and against this crate's previous version (0.2.32, before the kernels of Montgomery's products), in another run:

| Bits | LE | before | FLINT | PARI |
| --- | --- | --- | --- | --- |
| 64 | 2604 | 2646 | 5132 | 3903 |
| 128 | 11,955 | 14,453 | 28,439 | 49,532 |
| 256 | 45,776 | 69,094 | 112,503 | 183,421 |
| 512 | 207,942 | 428,325 | 376,375 | 739,912 |
| 1024 | 1.46 M | 2.08 M | 2.37 M | 3.90 M |
| 2048 | 14.3 M | 14.1 M | 15.6 M | 23.0 M |

- Primes, which take the whole test: from 5.0 to 1.3 times faster than GMP, 1.6 to 2.5 times faster than FLINT up to 1024 bits and 1.09 times at 2048, from the residues in registers and the kernels' products and squares, where GMP's Lucas test pays a division and an `mpz` call for each step.
- Odd composites stop at trial division or the strong test to base 2, which GMP takes in Montgomery's form too: from 1.2 to 3.5 times faster; random odd numbers, a mix, 1.16 to 4.4 times.

`next_prime` against this crate before the work of 2026-11-08 (0.2.32, "before"), GMP 6.3's `mpz_nextprime` into a result allocated before, and FLINT 3.6's `fmpz_nextprime`, from random numbers of the bits given, little-endian, a new set each of the 7 interleaved runs, in cycles per call (measured on 2026-11-11 on a quiet machine, the load average at 1.7, one thread, every process with `MallocNanoZone` unset, fat LTO; the cycles are the times at 4.51 GHz, the clock at the end of the run and at both ends of the next, as the clock read at its start, 2.09 GHz, was taken before the processor had raised its clock); GMP's tests the prime it finds once more, by a Miller–Rabin test to a random base:

| Bits | `next_prime` | before | GMP | FLINT |
| --- | --- | --- | --- | --- |
| 32 | 1233 | 2243 | 8326 | 1070 |
| 64 | 6731 | 6741 | 21,229 | 21,167 |
| 128 | 40,741 | 46,171 | 144,499 | 144,054 |
| 256 | 200,850 | 277,805 | 605,383 | 599,299 |
| 512 | 1.61 M | 2.98 M | 3.57 M | 3.58 M |
| 1024 | 15.5 M | 22.5 M | 28.2 M | 28.4 M |
| 2048 | 241 M | 245 M | 288 M | 288 M |

From 6.8 times faster than GMP at 32 bits to 1.2 times at 2048, and than FLINT from 64 bits up (3.5 to 1.2 times); from 128 to 1024 bits 0.54 to 0.88 of the time before, from the test's kernels (above), and about as long at 64 and 2048 bits. At 32 bits FLINT's word-sized search (`n_nextprime`: trial division by 32 primes with no branch, the strong test to base 2 and a table of its pseudoprimes) takes 0.87 of the time; the strong test to base 2 and the table, as FLINT's (above), took this from 2243 cycles with Baillie–PSW to 1233 (and a new vector for the result is some 150 of them). The times vary with the gaps to the next primes, which the operands share between the contenders.

`prev_prime` against this crate before, GMP's `mpz_prevprime` into a result allocated before, and PARI/GP 2.19's `precprime` (in `gp`, in the same campaign, the best of three loops of calls, its interpreter's cost included), in cycles per call at 4.51 GHz, the clock at both ends of the run and around PARI/GP's, from random numbers of the bits given:

| Bits | `prev_prime` | before | GMP | PARI/GP |
| --- | --- | --- | --- | --- |
| 32 | 1273 | 2264 | 8293 | 3000 |
| 64 | 6823 | 6851 | 21,080 | 14,800 |
| 128 | 41,074 | 46,819 | 145,051 | 240,000 |
| 256 | 206,438 | 284,367 | 613,061 | 1.45 M |
| 512 | 1.53 M | 2.90 M | 3.45 M | 10.7 M |
| 1024 | 15.0 M | 21.7 M | 27.3 M | 117 M |
| 2048 | 304 M | 310 M | 364 M | 1.89 G |

From 6.5 times faster than GMP at 32 bits to 1.2 times at 2048, as `next_prime` is: the test's speed, the sieve's remainders four products at a time, from 64 to 250 bits only those of the products of primes below the sieve's bound (4 % at 128 bits), and from 512 bits the primes beyond 8192 from the tables of the sieve rather than a sieve of Eratosthenes each call (2 to 3 %).

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

- Time complexity: trial division in `O(n·t)` for `n` limbs and `t` products of primes; the test in about `b` squares for the strong test and `2b` to `3b` for the Lucas test, for `b` bits, each `O(n²)` below 100 limbs and about 2.7 products of `n` limbs beyond; the next and the previous prime about `g / ln(P)` tests and one remainder by each product of primes per interval, for a gap `g` to it and primes up to `P`, whatever the byte order
- Space complexity: the number's limbs and ten residues, on the stack up to 8 KiB, one temporary buffer on the heap beyond, and one more for each product of 360 limbs or more, which the [multiplication](mul.md) computes by transforms; the next and the previous prime the interval's flags and the primes beyond the table; tables of the 1027 odd primes below 8192 and of their products, computed at compile time
- Soundness: the bytes beyond a vector's length, which may be uninitialised, are never read, and every byte of a result is written before the vector's length is set; the number is left as it was
