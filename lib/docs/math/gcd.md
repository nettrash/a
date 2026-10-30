[A](../../../README.md) › [Math](../../../README.md#math)

# Greatest common divisor

[Numbers](numbers.md) have four methods for their common divisors: `a.gcd(&b)` gives the greatest common divisor of `a` and `b`, `a.lcm(&b)` their least common multiple, `a.gcd_ext(&b)` the greatest common divisor `g` with cofactors `x` and `y` for which `g = a·x + b·y`, and `a.mod_inverse(&m)` the inverse of `a` modulo `|m|`, or `None` when there is none.

- `gcd` is never negative, and signs do not count: `gcd(-84, 36) = 12`. As for GMP's `mpz_gcd`, `gcd(x, 0) = |x|` and `gcd(0, 0) = 0`.
- `lcm` is never negative, and zero when either operand is: the smallest number that both divide, `|a·b| / gcd(a, b)`, as GMP's `mpz_lcm` gives it.
- `gcd_ext` gives GMP's `mpz_gcdext` cofactors, `|x| < |b| / 2g` and `|y| < |a| / 2g`, which define them uniquely, but for a few cases: when `|a| = |b|`, `x = 0` and `y = sgn(b)`; otherwise `x = sgn(a)` when `b = 0` or `|b| = 2g`, and `y = sgn(b)` when `a = 0` or `|a| = 2g`. So `x = 0` exactly when `g = |b|`, and `gcd_ext(0, 0) = (0, 0, 0)`.
- `mod_inverse` gives the `r` with `0 <= r < |m|` and `a·r = 1` modulo `|m|`, which exists exactly when `gcd(a, m) = 1`, and `Some(0)` for `|m| = 1`, as GMP 6.3's `mpz_invert` does. A modulus of zero panics, with the message of `%`: `attempt to calculate the remainder with a divisor of zero`, at the caller's line.
- Each magnitude is read in its own byte order, and zero bytes above its most significant non-zero byte are ignored; an empty vector is zero, and a magnitude of zero is zero whatever `negative` says.
- Every result is in the byte order of `self`, the left operand, and in the shortest form: zero is `[0]`, never empty and never negative. Each is built in one new vector of exactly its length; both operands are borrowed and left as they were.

The method is Euclid's algorithm, the larger number replaced by its remainder modulo the smaller one, carried out by Lehmer's method and, from a few hundred limbs, by Möller's subquadratic half gcd ("On Schönhage's algorithm and subquadratic integer gcd computation", Math. Comp. 77, 2008), as GMP's `mpn_gcd` and `mpn_gcdext` arrange them. This implementation carries it out as follows:

- Registers up to 16 bytes: the greatest common divisor by the binary method, the larger number replaced by the difference with its zero bits at the low end shifted out, after dividing once when one limb is 16 bits longer than the other, as GMP's `mpn_gcd_1` does; on two limbs a step is six instructions deep in assembly on aarch64, as GMP's `mpn_gcd_22`, where the compiler's code on `u128`s took 1.4 times as long. The cofactors by Euclid's algorithm on two limbs with the steps of hgcd2 (below) applied at once, then on one limb.
- Limbs for longer numbers: the magnitudes are read into native 64-bit limbs in working memory, on the stack up to 8 KiB; a much longer number is first divided by the shorter one, by [Newton's method](div.md) when the quotient has 768 limbs or more. The greatest common divisor alone shifts the zero bits at the low end out of both numbers first, and shifts the result back.
- Lehmer's method: the quotients of many of Euclid's steps are found from the top two limbs of the numbers, Möller's hgcd2, and their matrix applied to the whole numbers in one pass, two limbs read and two written a position, `u11·a − u01·b` and `u00·b − u10·a`, about 3.3 cycles a position. hgcd2 stops before a number would fall below `2^65`, which keeps every quotient one of the whole numbers; its steps take the larger number modulo the smaller one, the two swapping places at each step, and a quotient below 8, 83 % of Euclid's quotients, is three conditional subtractions in assembly on aarch64, a division otherwise: 560 cycles a call against 730 with a branch on whether the quotient is 1 and 820 for GMP 6.3's `mpn_hgcd2`, whose matrices are the same. A step that hgcd2 cannot take, when one number is much smaller than the other or the two are close, is one subtraction and one division.
- Cofactors: the extended algorithm keeps one row of the product of the matrices, the cofactor of the longer number, a row multiplied by each matrix in the same kind of pass, and finishes on one limb by Euclid's divisions; the other cofactor is `(g − a·x) / b`, an exact division. Every step is one of Euclid's own, the last perhaps not in full, so the cofactor found is Euclid's, which is GMP's: when the two numbers meet, the smaller of their two cofactors.
- The half gcd: from 800 limbs for the greatest common divisor and 300 for the cofactors, the top two thirds of the numbers (the top half first, for the cofactors) are reduced by about half their length by a matrix found recursively from their top half, by the same steps, and the matrix is applied to the rest: products of half the numbers' length, on the [multiplication](mul.md) with its Karatsuba and Toom-Cook methods, in O(M(n)·log n) for a product of n limbs in M(n). Below 250 limbs the half gcd takes Lehmer's steps one at a time. The cofactors finish with Lehmer's method on the reduced numbers with a row of their own, combined once with the half gcd's, as GMP's `mpn_gcdext` does.
- `lcm` divides the shorter operand exactly by the greatest common divisor and multiplies the quotient by the longer one; `mod_inverse` reduces `a` modulo `m` and runs the extended algorithm on `a` and `m`, keeping only the cofactor of `a`.

Measured on an Apple M4 Max, the median of 5 to 7 interleaved runs, in cycles per call, over pools of random operands of the lengths given in bytes each, little-endian (big-endian within a few percent), against GMP 6.3's `mpz_gcd`, `mpz_gcdext`, `mpz_lcm` and `mpz_invert` into variables with room for their results, and num-bigint 0.4's `gcd`, `extended_gcd` and `lcm` (num-integer) and `modinv`:

| Bytes | `gcd`: ours | GMP | num-bigint | `gcd_ext`: ours | GMP | num-bigint |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | 263 | 208 | 2590 | 531 | 594 | 8910 |
| 16 | 566 | 534 | 5028 | 1314 | 1741 | 41,458 |
| 32 | 2128 | 2774 | 12,107 | 3054 | 3872 | 110,855 |
| 64 | 5138 | 6853 | 26,954 | 6422 | 8118 | 262,919 |
| 128 | 11,182 | 14,787 | 54,690 | 13,120 | 17,001 | 609,685 |
| 256 | 23,905 | 32,040 | 125,327 | 28,316 | 37,051 | 1,453,849 |
| 1 KiB | 117,118 | 163,209 | 1,147,201 | 170,054 | 224,352 | 12,065,696 |
| 4 KiB | 795,848 | 1,189,009 | 16,800,785 | 1,396,811 | 1,675,992 | 322,268,454 |
| 16 KiB | 6,974,646 | 8,618,064 | 255,123,611 | 12,085,837 | 13,861,240 | 5,821,487,966 |
| 64 KiB | 55,502,917 | 64,418,953 | 4,012,716,606 | 99,230,127 | 105,858,722 | – |

| Bytes | `lcm`: ours | GMP | num-bigint | `mod_inverse`: ours | GMP | num-bigint |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | 280 | 214 | 2866 | 353 | 536 | 4874 |
| 16 | 579 | 616 | 5294 | 1090 | 1516 | 35,219 |
| 32 | 2427 | 2967 | 12,470 | 2726 | 3681 | 93,081 |
| 64 | 5602 | 6936 | 27,920 | 5929 | 7837 | 219,369 |
| 128 | 11,695 | 15,242 | 55,860 | 12,327 | 16,267 | 524,129 |
| 256 | 25,276 | 33,834 | 128,539 | 26,583 | 34,418 | 1,331,457 |
| 1 KiB | 128,216 | 174,244 | 1,166,706 | 148,228 | 198,790 | 14,276,885 |
| 4 KiB | 867,990 | 1,253,613 | 16,748,574 | 1,189,031 | 1,489,200 | 395,427,881 |
| 16 KiB | 7,539,412 | 9,350,967 | 260,608,813 | 10,282,808 | 12,390,802 | 7,332,492,698 |
| 64 KiB | 59,426,951 | 67,296,023 | – | 84,315,277 | 97,309,458 | – |

And for other shapes of operands: one much longer than the other, numbers with a long common factor (`a = g·u`, `b = g·v`, `g` half their length), and consecutive Fibonacci numbers, whose quotients are all 1:

| Operands | `gcd`: ours | GMP | `gcd_ext`: ours | GMP |
| --- | --- | --- | --- | --- |
| 64 KiB and 64 bytes | 316,320 | 391,193 | 552,083 | 578,953 |
| 4 KiB and 1 KiB | 176,301 | 229,959 | 314,457 | 366,055 |
| 1 KiB, a common factor of 512 bytes | 67,042 | 90,906 | 84,657 | 109,021 |
| 4 KiB, a common factor of 2 KiB | 510,809 | 749,315 | 718,649 | 887,575 |
| 16 KiB, a common factor of 8 KiB | 4,646,938 | 5,575,800 | 5,994,845 | 6,832,222 |
| 1 KiB, consecutive Fibonacci numbers | 88,320 | 104,034 | 131,940 | 122,813 |
| 4 KiB, consecutive Fibonacci numbers | 726,956 | 900,849 | 1,242,874 | 1,284,642 |

- From 32 bytes to 64 KiB, `gcd` takes 0.67 to 0.86 times GMP's time, `gcd_ext` 0.76 to 0.94, `lcm` 0.69 to 0.88 and `mod_inverse` 0.74 to 0.87: Lehmer's steps are cheaper (hgcd2 takes 560 cycles a call against 820, and its matrix is applied in one fused pass instead of four of GMP's kernels), which also moves the half gcd's thresholds up, to 800 and 300 limbs against GMP's 435 and 199 on an Apple M1. At 64 KiB, where GMP's half gcd applies its matrices by FFT-based products modulo `B^n − 1`, the lead narrows.
- At 8 and 16 bytes the time goes mostly to the result's allocation, which GMP's variables do not need: the crate's own `&a + &b` of 8 bytes takes 65 cycles against GMP's 20. So `gcd` of 8 bytes takes 1.27 times GMP's time and of 16 bytes 1.06, and `lcm` of 8 bytes 1.31; `gcd_ext` and `mod_inverse` take less than GMP's from 8 bytes.
- A much longer number's first division takes Newton's method from a quotient of 768 limbs: measured as above on one pair each, `gcd` of 1 MiB and 64 KiB took 65.9 million cycles against GMP's 133.6 million (0.49; 1.08 times by divide and conquer before), and of 64 KiB and 4 KiB 2.07 million against 3.33 million (0.62; 1.03 before). Balanced numbers divide only by Lehmer's steps and the half gcd, and are unchanged: 1 MiB and 1 MiB take 0.60 of GMP's time.
- On consecutive Fibonacci numbers, the extended algorithm of 1 KiB takes 1.07 times GMP's time: its steps are all subtractions, which GMP's branches on a quotient of 1 take best; Lehmer's method switches to such a branch after a call of hgcd2 with many steps, which brought this from 1.62 times.
- Against num-bigint, `gcd` and `lcm` take about a fifth of its time or less, and from 4 KiB a twentieth to a seventieth; `gcd_ext` and `mod_inverse`, whose cofactors num-bigint computes by divisions of whole numbers, 14 to 710 times less.
- No x86_64 timings: that code has been tested under emulation only.

## Examples

The greatest common divisor, the least common multiple and the cofactors:

```rust
use a::math::numbers::{ByteOrder, Number};

fn main() {
  let a = Number::new_from_i32(240, ByteOrder::BigEndian);
  let b = Number::new_from_i32(-46, ByteOrder::LittleEndian);

  let g = a.gcd(&b);
  assert_eq!(g.value, vec![2]);
  assert!(!g.is_negative());

  let l = a.lcm(&b);
  assert_eq!(l.value, vec![0x15, 0x90]); // 5520, big-endian as `a`

  // 2 = 240 · (-9) + (-46) · (-47)
  let (g, x, y) = a.gcd_ext(&b);
  assert_eq!(g.value, vec![2]);
  assert_eq!((x.is_negative(), x.value), (true, vec![9]));
  assert_eq!((y.is_negative(), y.value), (true, vec![47]));
}
```

An inverse modulo a prime, and none modulo a number with a common divisor:

```rust
use a::math::numbers::{ByteOrder, Number};

fn main() {
  // 2^61 - 1 is prime: 3 · 1537228672809129301 = 2 (2^61 - 1) + 1.
  let p = Number::new_from_u64((1 << 61) - 1, ByteOrder::BigEndian);
  let r = Number::new_from_u8(3, ByteOrder::BigEndian).mod_inverse(&p).unwrap();
  assert_eq!(r.value, 1537228672809129301u64.to_be_bytes().to_vec());

  let m = Number::new_from_u8(9, ByteOrder::BigEndian);
  assert!(Number::new_from_u8(6, ByteOrder::BigEndian).mod_inverse(&m).is_none());
}
```

Long numbers: `gcd(2^8000 − 1, 2^6000 − 1) = 2^2000 − 1`, as `gcd(2^i − 1, 2^j − 1) = 2^gcd(i, j) − 1`:

```rust
use a::math::numbers::{ByteOrder, Number};

fn main() {
  let a = Number::new(vec![0xFF; 1000], ByteOrder::LittleEndian);
  let b = Number::new(vec![0xFF; 750], ByteOrder::BigEndian);
  let g = a.gcd(&b);
  assert_eq!(g.value, vec![0xFF; 250]);
  let (g, x, y) = a.gcd_ext(&b);
  assert_eq!(g.value, vec![0xFF; 250]);
  // 2^8000 - 1 = (2^6000 - 1) 2^2000 + (2^2000 - 1): x = 1, y = -2^2000.
  assert_eq!(x.value, vec![1]);
  assert!(y.is_negative() && y.value.len() == 251 && y.value[250] == 1);
}
```

## Efficiency

- Time complexity: O(n²) for numbers of n limbs below 800 (300 for the cofactors), and from there O(M(n)·log n) for a product of n limbs in M(n), O(n^1.40·log n) with Toom-4's; a much longer number costs one division more
- Space complexity: the vectors of the results; working memory of a few times the longer number's limbs, on the stack up to 8 KiB, one temporary buffer on the heap beyond, and one more for each product of 360 limbs or more, which the [multiplication](mul.md) computes by transforms, and the buffers of Newton's method for a long first division; `lcm` allocates the greatest common divisor and the quotient
- Soundness: the bytes beyond a vector's length, which may be uninitialised, are never read, and every byte of a result is written before the vector's length is set; both operands are left as they were
