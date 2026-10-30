[A](../../../README.md) › [Math](../../../README.md#math)

# Factorials and binomial coefficients

`Number::factorial(n, order)` is `n!`, and `Number::binomial(n, k, order)` the binomial coefficient `C(n, k) = n! / (k! (n − k)!)`, for `n` and `k` of type `u64`, as [numbers](numbers.md) in the byte order `order`.

- `0!` is 1. `C(n, k)` is 0 for `k > n`, 1 for `k = 0` and `k = n`, and `C(n, k) = C(n, n − k)`, as GMP's `mpz_fac_ui` and `mpz_bin_uiui` give them.
- The result is in the shortest form, in one new vector of exactly its length. Results of up to 16 bytes are computed in registers and allocate nothing else; longer ones take temporary buffers, freed before the result is returned.
- Neither function panics but when the result or its working memory cannot be allocated, as for any vector: `1 000 000!` takes 2.3 MB, and its sieve 62.5 KB.

`n!` is the odd part of `n!` shifted left by `n − popcount(n)` bits, the power of two in `n!`. The odd part is computed by Luschny's method of the swinging factorial ("Divide, Swing and Conquer the Factorial", 2008), as GMP's `mpz_oddfac_1` computes it; a binomial coefficient with a `k` small beside `n` is a product of `k` numbers divided exactly by `k!`, and otherwise the product of its prime factors, after Goetgheluck ("Computing Binomial Coefficients", 1987), as GMP's `mpz_bin_uiui` chooses between them. This implementation carries them out as follows:

- Tables: `n!` up to 34, the last below `2^128`; the odd parts of `n!` up to 25 and the products of the odd numbers up to 33, the last below `2^64`.
- The odd part of `m!` below 256 is the product of the odd numbers up to `m`, `m/2`, `m/4` and so on: each odd `j` once for each `j·2^i ≤ m`, from the table up to 33 and in pairs above it, packed into limbs on the stack and multiplied a limb at a time, in two interleaved products that the core computes side by side.
- From 256, the odd part of `n!` is the square of that of `⌊n/2⌋!` times the odd part of the swinging factorial `n! / ⌊n/2⌋!²`, whose prime factors are known: the primes above `n/2` once, those from `n/3` to `n/2` not at all, those from `√n` to `n/3` when `⌊n/p⌋` is odd, and each prime `p` below `√n` to the number of odd `⌊n/p^i⌋`. The remainders `n mod p` of increasing primes take a division only when the quotient changes.
- The primes come from a sieve of Eratosthenes on the odd numbers: the multiples of 3, 5 and 7 from a pattern of 105 words that repeats, and those of each other prime below 64 gathered a word at a time in a register. Measured on an Apple M4 Max, `C(10 000, 5000)` took 23,600 cycles so against 53,800 with a bit at a time.
- Products: as many factors as fit are multiplied into each limb first, and the limbs then by a balanced product tree, its halves in one buffer, so that the longest products take Karatsuba's and Toom-Cook's methods; 16 limbs and fewer are multiplied a limb at a time.
- `C(n, k)` with `k ≤ n − k`, below `2^128` (for every `k` up to 65 and `n` up to a tabulated bound, the largest `n` for which it fits): the product of the odd parts of `n − k + 1` to `n` modulo `2^128`, times the tabulated inverse of the odd part of `k!` modulo `2^128`, shifted left by the power of two in `C(n, k)`, which Kummer's theorem gives: `popcount(k) + popcount(n − k) − popcount(n)`.
- A small `k` beside `n` (below): the odd parts of `n − k + 1` to `n` packed into limbs and multiplied by a product tree, and the product divided exactly by the odd part of `k!`, by Hensel's method from the low end: by one limb up to `k = 25`, by the odd part of `k!` on the stack up to 255, and from there with the division of `/` when Hensel's method would take more than `2^18` limb products.
- A large `k`, above 100 and above `5√n`: the product of the prime factors of `C(n, k)`, the exponent of each odd prime `p ≤ n` being the number of carries when `k` and `n − k` are added in base `p` (Kummer's theorem): 1 for the primes above `n − k`, 0 from `n/2` to `n − k`, at most 1 above `√n`, when `n mod p < k mod p`. The division takes about the square of the numerator's limbs and the sieve the time of `n`: they met at `k = 5√n`, measured on an Apple M4 Max from `n = 1000` (`k = 158`) to `n = 10^6` (`k = 5000`), within a few percent. Below `k = 100` the sieve's fixed cost made the division faster whatever `n`. GMP takes the primes from `k > n/16`, which here would leave the division 4 to 8 times as slow from `n` of about `10^5`.

Measured on an Apple M4 Max, the results in the byte order given, against GMP 6.3's `mpz_fac_ui` and `mpz_bin_uiui` into an initialised result and, as num-bigint 0.4 has neither, its products `2·3·…·n` and `C(n, i) = C(n, i − 1)·(n − i + 1) / i` for `i` up to `k` (the median of 7 interleaved runs, in cycles per call):

| n | LE | BE | GMP | num-bigint |
| --- | --- | --- | --- | --- |
| 10 | 118 | 118 | 59 | 186 |
| 100 | 250 | 243 | 136 | 1857 |
| 1000 | 7620 | 7461 | 8331 | 118,987 |
| 10^4 | 360,770 | 360,706 | 416,394 | 17,627,290 |
| 10^5 | 15,436,890 | 15,284,677 | 13,986,789 | 2,294,276,893 |
| 10^6 | 520,104,855 | 511,704,774 | 237,275,082 | — |

| C(n, k) | LE | BE | GMP | num-bigint |
| --- | --- | --- | --- | --- |
| C(100, 50) | 205 | 204 | 141 | 4473 |
| C(1000, 10) | 125 | 127 | 66 | 809 |
| C(10^4, 5000) | 23,274 | 22,854 | 29,853 | 16,923,282 |
| C(10^5, 100) | 1378 | 1373 | 1103 | 33,458 |
| C(10^6, 5·10^5) | 15,709,590 | 15,596,373 | 17,226,816 | — |
| C(2^40, 3) | 124 | 125 | 69 | 296 |

- From `1000!` to `10^4!`, 9–13% less than GMP's time; `10^5!` takes 9–10% more, and `10^6!` 2.2 times as long: their last products, for results of about 24,000 and 290,000 limbs, are FFT products for GMP and Toom-4 ones here, as the multiplication of this crate has no FFT.
- Binomial coefficients from their primes take 9–23% less than GMP's; `C(10^5, 100)`, by a division, 25% more: GMP divides as it multiplies, a few limbs at a time.
- Up to a few hundred cycles, the result's allocation, about 50 cycles of `malloc` and `free`, is most of the difference: `10!`, `C(100, 50)`, `C(1000, 10)` and `C(2^40, 3)` come from a table or from `u128`s.
- Against num-bigint's products, from 1.6 to 730 times faster.

## Examples

```rust
use a::math::numbers::{ByteOrder, Number};

fn main() {
  assert_eq!(Number::factorial(0, ByteOrder::BigEndian).value, vec![1]);
  assert_eq!(Number::factorial(10, ByteOrder::BigEndian).value, vec![0x37, 0x5F, 0x00]); // 3,628,800

  // 30! = 2^26 times an odd number: its 26 low bits are zero, three zero bytes and two bits.
  let f = Number::factorial(30, ByteOrder::LittleEndian);
  assert_eq!(&f.value[..3], &[0, 0, 0]);
  assert_eq!(f.value[3] & 0b11, 0);
  assert_ne!(f.value[3] & 0b100, 0);

  // C(52, 5) = 2,598,960 hands of poker.
  assert_eq!(Number::binomial(52, 5, ByteOrder::BigEndian).value, vec![0x27, 0xA8, 0x30]);
  assert_eq!(Number::binomial(5, 52, ByteOrder::BigEndian).value, vec![0]);
}
```

Long results, checked against each other: `C(2n, n) = (2n)! / (n!)²`:

```rust
use a::math::numbers::{ByteOrder, Number};

fn main() {
  let n = 3000;
  let central = Number::binomial(2 * n, n, ByteOrder::LittleEndian);
  let f = Number::factorial(n, ByteOrder::LittleEndian);
  let quotient = Number::factorial(2 * n, ByteOrder::LittleEndian).div_exact(&(&f * &f));
  assert_eq!(central.value, quotient.value);
}
```

## Efficiency

- Time complexity: `n!` in O(M(N) log N) for its N = log2(n!) bits, the time of its last products, plus O(n) for the sieve; `C(n, k)` from its primes in O(M(N) log N) plus O(n), and by a division in O(k log n) packing plus the product and the division of about k·log2(n) bits
- Space complexity: the vector of the result; the sieve of n/16 bytes, and the products of the tree, about twice the result's limbs
- Soundness: every limb is written before it is read, and every byte of a result before the vector's length is set; the tables and the bounds of the method in registers are checked by the tests
