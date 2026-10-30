[A](../../../README.md) › [Math](../../../README.md#math)

# Primes

[Numbers](numbers.md) tell whether they are prime with `a.is_probable_prime()`, by the Baillie–PSW test, and give the next prime above them with `a.next_prime()`.

- `is_probable_prime` is `false` for every number below 2, negative numbers included (GMP's `mpz_probab_prime_p` takes the absolute value of a negative number; this does not), `true` for every prime, and `false` for every composite number it finds a factor of or that fails the test.
- It is exact below `2^64`: Feitsma and Galway's list of all the base-2 Fermat pseudoprimes below `2^64` (2009), which Gilchrist checked, has no number that passes both halves of the test. Above, no composite number is known that passes it, though such numbers are believed to exist: it is a probable-prime test, the one GMP 6.2 and later run for `mpz_probab_prime_p(n, 24)`.
- `next_prime` gives the smallest probable prime above `a`, in the byte order of `a`, in one new vector of exactly its length: 2 for everything below 2, negative numbers included. It is exact below `2^64`, as the test is.
- Each magnitude is read in its own byte order, and zero bytes above its most significant non-zero byte are ignored; an empty vector is zero, and a magnitude of zero is zero whatever `negative` says. The number is left as it was.

## Baillie–PSW

The test is Baillie and Wagstaff's (1980) after Pomerance, Selfridge and Wagstaff: a strong probable-prime test to base 2 and a strong Lucas probable-prime test with Selfridge's parameters, whose pseudoprimes seem to have nothing in common. This implementation carries it out as follows:

- Trial division first: the long number is reduced modulo products of consecutive small primes that fit in a limb, by the division's two-limbs-by-one steps with a reciprocal, four products at a time in independent chains, then each prime of a product is tested on the remainder by one multiplication, as `x·p^(−1) mod 2^64 <= (2^64 − 1)/p` exactly when `p` divides `x`. The primes go up to four times the number's bits, from 256 to 8192, which balances the passes against the tests they save (measured: random odd numbers of 512 bits took 16 % less time than with the primes up to twice their bits); below `2^64`, up to 256, and a number below `257²` with no factor is prime.
- The strong test to base 2: `n − 1 = d·2^s` with `d` odd, `2^d mod n` from the top bit of `d` with a square for each bit and a doubling, an addition, for each one, as GMP's `mpn_2powm` does, then up to `s − 1` squares; `n` passes when `2^d = 1` or one of the squares is `−1`. Almost every composite stops here.
- Selfridge's parameters (method A): the first `D` of `5, −7, 9, −11, 13, …` whose Jacobi symbol `(D/n)` is `−1`, `P = 1` and `Q = (1 − D)/4`. The symbols come from one remainder of `n` by `9·5·7·11·…·47`, by quadratic reciprocity and the binary method on limbs (factors of two taken out, then the larger odd number less the smaller); a `D` beyond 47 takes a pass of its own. A square's symbols are never `−1`, so squares are told apart before the search: odd squares are 1 modulo 8 and squares modulo each prime to 47, which leaves the square root, by Newton's steps, for the few numbers that pass, about 1 in 65,000 of the others.
- The strong Lucas test: `n + 1 = d·2^s`, and `n` passes when `U_d = 0` or one of `V_(d·2^r)`, `r < s`, is 0. `U_d` and `U_(d−1)` come from the top bit of `d` with three squares a bit, as GMP's `mpz_lucas_mod` does after Hackman: `U_2k = U_(k+1)² − (U_(k+1) − U_k)²`, `U_(2k+1) = U_(k+1)² − Q·U_k²`, and for a one bit `U_(2k+2) = U_(2k+1) − Q·U_2k`, the products by `Q` by additions for `|Q| <= 4`; then `V_d = U_d − 2Q·U_(d−1)`, `Q^d = (V_d² − D·U_d²)/4`, and `V_2k = V_k² − 2Q^k`. For `D = 5`, half of all numbers, `Q = −1` and `Q^k = ±1`, so the ladder of `V_k` and `V_(k+1)` takes one square and one product a bit instead, as GMP's `mpn_strongfibo` does with the Fibonacci numbers: 9–13 % less time for primes of 256 to 1024 bits.
- Montgomery's form throughout, the [powers](pow.md)' arithmetic: one limb and two in registers, three to seven limbs in registers by the coarsely integrated operand scanning, eight and more in limbs with the two-row REDC kernel and REDC by products from 100 limbs, the residues always reduced below `n` so that equal residues are equal limbs. GMP takes Montgomery's form for the strong test only, and reduces the Lucas sequence by divisions.

## Next prime

The odd numbers above `a` are sieved in intervals by the small primes, and those left are tested in order, after GMP's `mpz_nextprime`: the start's remainders by the products of the primes give where each prime's odd multiples fall in the interval, those are crossed out, and the numbers left take the test without its trial division. The primes go up to GMP's estimate of where sieving more costs more than the tests it saves, `bits^2.5 / 124` (at least 1000 and at most `2^24`; 270,000 for 1024 bits), the table's below 8192 and a sieve of Eratosthenes beyond; an interval holds five odd numbers a bit, about fourteen times the average gap between primes, so that a second interval is rare. Below the last prime below `2^64`, the search steps through the numbers prime to 30 with the test of numbers below `2^64`.

## Measured

Measured on an Apple M4 Max with other work running on the machine (interleaved runs, the median of five, in cycles per call, rounded; runs of the longest numbers that other work disturbed, by up to twice the time, were measured again), `is_probable_prime` against GMP 6.3's `mpz_probab_prime_p(n, 24)`, for random primes, random odd composites and random odd numbers of exactly the bits given:

| Bits | primes: LE | BE | GMP | odd composites: LE | BE | GMP | random odd: LE | BE | GMP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 64 | 2572 | 2566 | 12,524 | 217 | 213 | 301 | 273 | 273 | 663 |
| 128 | 14,280 | 14,124 | 48,503 | 732 | 733 | 1808 | 933 | 929 | 3587 |
| 256 | 68,457 | 68,222 | 159,951 | 2494 | 2479 | 6139 | 3474 | 3272 | 8770 |
| 512 | 415,072 | 432,582 | 631,798 | 18,985 | 21,158 | 28,642 | 16,139 | 16,054 | 22,026 |
| 1024 | 2.18 M | 2.20 M | 3.37 M | 76,539 | 75,229 | 88,241 | 96,297 | 93,930 | 154,771 |
| 2048 | 14.9 M | 15.2 M | 20.9 M | 554,920 | 539,392 | 646,475 | 1.08 M | 1.08 M | 1.25 M |

- Primes, which take the whole test: from 4.9 to 1.4 times faster than GMP, most below 256 bits, where the residues stay in registers and GMP's Lucas test pays a division and an `mpz` call for each step.
- Odd composites stop at trial division or the strong test to base 2, which GMP takes in Montgomery's form too: from 1.15 to 2.5 times faster, most below 512 bits, from the residues in registers and the trial division's four chains; random odd numbers, a mix, 1.16 to 3.8 times.

`next_prime` against GMP's `mpz_nextprime`, into a result allocated before and into a new `mpz_t`, from random numbers of the bits given; GMP's tests the prime it finds once more, by a Miller–Rabin test to a random base:

| Bits | LE | BE | GMP | GMP new |
| --- | --- | --- | --- | --- |
| 64 | 6826 | 6646 | 20,619 | 20,587 |
| 128 | 47,342 | 47,126 | 141,771 | 138,573 |
| 256 | 336,164 | 328,643 | 672,935 | 675,979 |
| 512 | 2.20 M | 1.98 M | 2.54 M | 2.57 M |
| 1024 | 29.5 M | 29.4 M | 36.8 M | 36.6 M |

From 3.0 to 1.15 times faster than GMP: the test's own speed, and the sieve's remainders four products at a time. The times vary with the gaps to the next primes, which the operands share between the contenders; at 512 bits the two byte orders differed by 10 % from run to run.

## Examples

Rust usage: primes, pseudoprimes and the next prime:

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
}
```

## Efficiency

- Time complexity: trial division in `O(n·t)` for `n` limbs and `t` products of primes; the test in about `b` squares for the strong test and `2b` to `3b` for the Lucas test, for `b` bits, each `O(n²)` below 100 limbs and about 2.7 products of `n` limbs beyond; the next prime about `g / ln(P)` tests and one remainder by each product of primes per interval, for a gap `g` to it and primes up to `P`, whatever the byte order
- Space complexity: the number's limbs and ten residues, on the stack up to 8 KiB, one temporary buffer on the heap beyond, and one more for each product of 360 limbs or more, which the [multiplication](mul.md) computes by transforms; the next prime the interval's flags and the primes beyond the table; tables of the 1027 odd primes below 8192 and of their products, computed at compile time
- Soundness: the bytes beyond a vector's length, which may be uninitialised, are never read, and every byte of a result is written before the vector's length is set; the number is left as it was
