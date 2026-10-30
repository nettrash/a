[A](../../../README.md) › [Math](../../../README.md#math)

# Roots

[Numbers](numbers.md) have integer roots: `a.isqrt()` is the square root of `a` rounded down, `a.isqrt_rem()` gives it with the remainder, and `a.nth_root(n)` is the root of any degree `n`, truncated toward zero.

- `a.isqrt()` is `⌊√a⌋`, the largest integer whose square is not above `a`, as `u64::isqrt` gives it; `a.isqrt_rem()` is `(s, r)` with `s = ⌊√a⌋` and `r = a − s²`, which is at most `2s`, as GMP's `mpz_sqrtrem` gives them.
- `a.nth_root(n)` is the largest integer whose `n`-th power is not above `a` for an `a` that is not negative, and for a negative `a` and an odd `n` the root of `|a|` negated, truncated toward zero as GMP's `mpz_root` and num-integer's `Roots::nth_root` truncate it: the cube root of `-999 999` is `-99`. `a.nth_root(1)` is `a`, and `a.nth_root(2)` is `a.isqrt()`.
- The square root of a negative number panics with the message of `i64::isqrt`, `argument of integer square root cannot be negative`. A root of degree 0 panics with `root degree n must be at least 1`, and an even root of a negative number with `even roots of a negative are imaginary`, the messages of num-integer. A negative zero is zero.
- Each magnitude is read in its own byte order, and zero bytes above its most significant non-zero byte are ignored; an empty vector is zero.
- The results are in the byte order of `a`, in the shortest form: zero is `[0]`, never empty and never negative. Each is built in one new vector of exactly its length; `a` is borrowed and left as it was. Roots whose working memory fits in 8 KiB allocate nothing else; longer ones take one temporary buffer, freed before the result is returned.

The square root is Zimmermann's Karatsuba square root ("Karatsuba Square Root", INRIA research report 3805, 1999), as GMP's `mpn_dc_sqrtrem` arranges it, in the time of a product of its limbs up to a constant; the roots of other degrees are found by Newton's method at doubling precision, after Brent and Zimmermann ("Modern Computer Arithmetic", section 1.5.2), as GMP's `mpn_rootrem` finds them. This implementation carries them out as follows:

- Registers up to 32 bytes: a magnitude of at most 16 bytes is a `u128`. Its square root in double precision is right to about 52 bits, and one step of Newton's method with the error `a − s²` computed exactly as an integer takes it to within about `2^-30` of the root, whose floor it then is unless the root is that close to an integer: the remainder, also exact, tells, and a cold path corrects it. One limb is rooted in double precision directly. A root of another degree is the floor of an estimate of `a^(1/n)` in double precision, or one more or one less, which two exact powers computed side by side tell: the system's cube root for `n = 3`, and otherwise `2^(log2(a)/n)` from polynomials for `log2` and `2^x` (errors of `2·10^-6` and `10^-7`) for roots of up to 18 bits, a quarter of the latency of the system's `powf`, which takes the longer roots. From 17 to 32 bytes, one step of the Karatsuba square root on halves of one limb is unrolled in registers: the root of the top two limbs, a division by the precomputed reciprocal of that root, and the correction.
- The Karatsuba square root: a magnitude of `2n` limbs, shifted left by an even number of bits so that its top limb is at least `2^62`, is cut into a top half and two quarters; the root `s'` and remainder `r'` of the top half are computed the same way, `r'` and the next quarter are divided by `2s'`, the quotient is the root's low half, and the remainder less the square of that quotient is the remainder, once corrected when it is negative (taking one off the root). The divisions are those of `/`, by divide and conquer from 40 limbs of divisor, and the squares those of `*`, with Karatsuba's and Toom-Cook's methods; the step on halves of one limb is the unrolled one above.
- The root alone: the last square is computed only when the bits shifted out of the root at the end cannot absorb the correction. When fewer than two bits are shifted out, roots from 12 limbs are computed with one limb more, that of the magnitude with two zero limbs below it, so that the correction is almost never computed: measured on an Apple M4 Max, a root of 4 KiB took 34,800 cycles so against 40,700, and one of 64 KiB 2.41 million against 2.83 million, while at 8 limbs the limb more cost 4% more than it saved.
- Other degrees: the root of `U` is found from its top bits down. With `S` the exact root of the top bits of `U` whose root has `p` bits, the next `b` bits of the root are `(R 2^b + the next bits of U) / (k S^(k−1))` rounded down, where `R = T − S^k` is the remainder of those top bits `T`: never too small, and at most one too large when `S ≥ 2k·2^b`, so each step doubles the precision less `log2 k + 2` bits, and starts with one less when `S^k` is above `T`. The first 30 bits come from `2^(log2(T)/k)` in double precision (from the polynomials above up to 18 bits); a root of that few bits is finished there. The estimate is made exact by comparing its powers with `T`: first powers of its top 64 bits truncated at each step, which bound the exact power within a factor of `1 + k·2^-60`, and the exact power only when that bound cannot tell, which is rare but for perfect powers. The division takes only the top limbs of a divisor much longer than the quotient, and the dividend's plus one, which keeps the quotient an upper bound: for degree 100, the divisor is 99 times as long as the quotient. When `U` has more than `3k` limbs, the root is computed with 64 bits more, that of `U·2^(64k)`, and its top bits are the root of `U` unless the 64 extra bits are 0 or 1, as for perfect powers, where the bounded powers or an exact one decide.
- Working memory: square roots of up to 64 limbs of working memory in a buffer of the function's own frame, up to 8 KiB in one on the stack, and on the heap beyond; the `k`-th root's numbers in one buffer as well, with the scratch of the longest products and divisions in a vector of its own.

Measured on an Apple M4 Max with the magnitudes in the byte order given, over pools of random magnitudes of the lengths given in bytes, against GMP 6.3's `mpz_sqrt`, `mpz_sqrtrem` and `mpz_root` into an initialised result, and num-bigint 0.4's `sqrt` and `nth_root` (for the remainder, `sqrt` and `x − s²`) (the median of 9 interleaved runs, in cycles per call):

| Bytes | `isqrt`: LE | BE | GMP | num-bigint | `isqrt_rem`: LE | BE | GMP | num-bigint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 62 | 63 | 20 | 9 | 111 | 116 | 25 | 30 |
| 16 | 85 | 82 | 39 | 618 | 127 | 129 | 46 | 735 |
| 32 | 168 | 169 | 175 | 1411 | 270 | 280 | 175 | 1526 |
| 64 | 426 | 432 | 335 | 2798 | 504 | 511 | 345 | 3037 |
| 128 | 670 | 676 | 676 | 6297 | 755 | 769 | 637 | 6665 |
| 256 | 1192 | 1187 | 1181 | 13,538 | 1249 | 1259 | 1169 | 14,466 |
| 1 KiB | 4770 | 4797 | 4615 | 101,731 | 5298 | 5304 | 6019 | 110,753 |
| 4 KiB | 33,617 | 33,811 | 34,588 | 1,179,546 | 40,211 | 40,317 | 48,422 | 1,260,061 |
| 16 KiB | 293,306 | 291,415 | 305,430 | 13,482,799 | 341,839 | 339,185 | 418,033 | 14,049,311 |
| 64 KiB | 2,392,151 | 2,393,758 | 2,355,695 | 136,315,132 | 2,822,171 | 2,803,888 | 3,217,485 | 146,041,979 |

Roots of other degrees, little-endian (big-endian took the same within a few percent):

| Bytes | n = 3 | GMP | num-bigint | n = 5 | GMP | num-bigint | n = 17 | GMP | num-bigint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 89 | 258 | 22 | 151 | 196 | 36 | 154 | 78 | 79 |
| 16 | 131 | 382 | 714 | 184 | 414 | 871 | 215 | 105 | 960 |
| 32 | 691 | 634 | 1937 | 524 | 693 | 1816 | 273 | 585 | 1064 |
| 64 | 980 | 922 | 3586 | 886 | 979 | 3994 | 574 | 1086 | 1424 |
| 128 | 1307 | 1282 | 7249 | 1508 | 1483 | 7225 | 1258 | 1647 | 3183 |
| 256 | 1902 | 1741 | 15,710 | 2071 | 1928 | 16,553 | 1998 | 2517 | 10,539 |
| 1 KiB | 6392 | 6916 | 124,606 | 5826 | 6627 | 115,036 | 5129 | 6351 | 104,091 |
| 4 KiB | 42,024 | 42,581 | 1,446,230 | 36,226 | 39,345 | 1,373,667 | 27,790 | 33,536 | 1,210,100 |
| 16 KiB | 329,011 | 340,257 | 16,464,730 | 309,365 | 307,795 | 14,954,923 | 226,017 | 273,624 | 13,559,134 |
| 64 KiB | 2,571,513 | 2,688,167 | 163,159,878 | 2,279,789 | 2,463,173 | 149,766,418 | 1,731,867 | 2,122,774 | 144,538,849 |

| Bytes | n = 100 | GMP | num-bigint | n = bits − 3 | ours | GMP | num-bigint |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 64 | 19 | 8 | 61 | 149 | 88 | 46 |
| 16 | 198 | 96 | 3612 | 125 | 188 | 98 | 1086 |
| 32 | 274 | 104 | 4360 | 253 | 326 | 121 | 1363 |
| 64 | 310 | 153 | 3913 | 509 | 342 | 180 | 1572 |
| 128 | 298 | 1305 | 2084 | 1021 | 370 | 293 | 1885 |
| 256 | 298 | 3022 | 10,863 | 2045 | 397 | 569 | 7945 |
| 1 KiB | 7281 | 11,447 | 112,979 | 8189 | 570 | 4112 | 42,604 |
| 4 KiB | 33,209 | 39,729 | 1,689,449 | 32,765 | 1032 | 36,429 | 392,221 |
| 16 KiB | 213,138 | 264,223 | 16,401,656 | 131,069 | 1913 | 309,656 | 3,190,720 |
| 64 KiB | 1,683,956 | 2,106,647 | 159,115,945 | 524,285 | 5300 | 1,599,665 | 23,388,201 |

- From 128 bytes, square roots take within 4% of GMP's time either way; with their remainders, up to 18% more up to 256 bytes and 12–18% less from 1 KiB.
- Each result pays for its vector: a `malloc` and a `free` of a small block cost about 50 cycles here, which GMP's results, written into an initialised `mpz_t`, do not, nor num-bigint's roots of one limb, which it keeps inline; `isqrt_rem` allocates two. That is most of the difference up to 64 bytes: at 32 bytes, the root in registers with its allocation takes 168 cycles against GMP's 175.
- Roots of degree 3 and 5 take 0.3 to 0.8 of GMP's time up to 16 bytes, from 9% more to 24% less from 32 to 256 bytes, and from 1% more to 12% less from 1 KiB; of degree 17, 17–53% less from 32 bytes, and of degree 100, 16–36% less from 1 KiB, with the cut divisions.
- Roots of a few bits — of a degree near the magnitude's bits, of 100 up to 64 bytes, of 17 up to 16 bytes — take 45 to 205 cycles more than GMP's up to 64 bytes, for the allocation, the estimate and the two bounded powers; from 128 bytes, a fraction of GMP's time, as they need no exact power.
- Against num-bigint, from 2.5 to 4400 times faster above 8 bytes; at 8 bytes, num-bigint's roots of one limb take 8 to 80 cycles and allocate nothing.
- The x86_64 kernels that the divisions and squares use have been tested for correctness under emulation only; this code has no assembly of its own.

## Examples

Square roots and remainders:

```rust
use a::math::numbers::{ByteOrder, Number};

fn main() {
  // 1000 = 31² + 39.
  let a = Number::new_from_u16(1000, ByteOrder::BigEndian);
  assert_eq!(a.isqrt().value, vec![31]);
  let (s, r) = a.isqrt_rem();
  assert_eq!(s.value, vec![31]);
  assert_eq!(r.value, vec![39]);

  // (2^64 − 1)² = 2^128 − 2^65 + 1, in little-endian order.
  let mut bytes = vec![0x01];
  bytes.extend([0; 7]);
  bytes.push(0xFE);
  bytes.extend([0xFF; 7]);
  let square = Number::new(bytes, ByteOrder::LittleEndian);
  assert_eq!(square.isqrt().value, vec![0xFF; 8]);
}
```

Roots of other degrees, truncated toward zero:

```rust
use a::math::numbers::{ByteOrder, Number};

fn main() {
  let a = Number::new_from_i64(-1_000_000_000_000_000_000, ByteOrder::LittleEndian);
  let root = a.nth_root(3);
  assert_eq!(root.value, Number::new_from_u32(1_000_000, ByteOrder::LittleEndian).value);
  assert!(root.is_negative());

  // 2^1000 = (2^125)^8; its roots of degree 1000 and more are 2 and 1.
  let mut bytes = vec![0; 126];
  bytes[0] = 1;
  let two_to_1000 = Number::new(bytes, ByteOrder::BigEndian);
  let mut root = vec![0; 16];
  root[0] = 0x20;
  assert_eq!(two_to_1000.nth_root(8).value, root);
  assert_eq!(two_to_1000.nth_root(1000).value, vec![2]);
  assert_eq!(two_to_1000.nth_root(1001).value, vec![1]);
}
```

The square root of a negative number panics, as for `i64`:

```rust,should_panic
use a::math::numbers::{ByteOrder, Number};

fn main() {
  let _ = Number::new_from_i8(-4, ByteOrder::BigEndian).isqrt();
}
```

## Efficiency

- Time complexity: the square root of n limbs in that of a product of n limbs up to a constant, O(n^1.58) with Karatsuba's method and less with Toom-Cook's; a root of degree k in that of the powers and divisions of its last steps, about that of products of n limbs; a root of a few bits, of a large degree, in O(log k) products of 64-bit mantissas but when the magnitude is a perfect power or next to one
- Space complexity: the vector of each result; working memory of about 2.5 times the magnitude's limbs for a square root and about 8 times for other roots, on the stack up to 8 KiB, one temporary buffer on the heap beyond
- Soundness: the bytes beyond a vector's length, which may be uninitialised, are never read, and every byte of a result is written before the vector's length is set; the floating-point estimates only choose which candidates to check: every root is proved by exact powers or by bounds of them that hold whatever the estimates are
