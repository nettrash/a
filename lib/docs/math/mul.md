[A](../../../README.md) › [Math](../../../README.md#math)

# Multiplication

[Numbers](numbers.md) multiply with `*` and `*=`, each operand owned or borrowed: `a * b`, `a * &b`, `&a * b` and `&a * &b` give a new `Number`, and `a *= b` and `a *= &b` multiply `a`. The six forms give the same result for operands of any lengths, signs and byte orders; they differ only in the vector the product is built in.

- The product is exact. Each magnitude is read in its own byte order, and zero bytes above its most significant non-zero byte are ignored; an empty vector is zero, and a magnitude of zero is zero whatever `negative` says. Numbers built from the public fields, which need not be in the shortest form, multiply as the values they hold.
- The product is negative when exactly one operand is and neither is zero.
- The product is in the byte order of the left operand in every form, `&a * b` included, and `*=` keeps the order of the number it changes. The two operands' orders may differ.
- The product is in the shortest form: no zero bytes above the most significant one, and zero is `[0]`, never empty and never negative.
- The product is built where a sum would be: in `a`'s vector for `a *= &b`, `a *= b`, `a * &b` and `a * b` when its capacity is at least the length of the product (one byte for zero); in `b`'s for `&a * b`, and for `a * b` and `a *= b` when `a`'s is too small. Otherwise, and always for `&a * &b`, exactly one new vector is allocated. Products of up to 1 KiB allocate nothing else; longer ones may take one temporary buffer for the computation, and those computed by number-theoretic transforms one more for the transforms, both freed before the product is returned.
- A borrowed operand is left as it was. A number multiplied by itself through one borrow, `&a * &a`, is squared: each product of two different limbs is computed once.

The method is the schoolbook one, on limbs of 8 bytes: each limb of one operand times the other, the rows added up, in O(n·m) for operands of n and m limbs. From 24 limbs (192 bytes) in the shorter operand, Karatsuba's method takes over: each operand is split into halves, `x = x1·B^k + x0`, and the product is made of three products of halves instead of four, `x1·y1`, `x0·y0` and `(x0 − x1)·(y0 − y1)`, each computed the same way, in O(n^1.58). From 120 limbs (960 bytes), Toom-3 cuts each operand into three parts and makes the product of five products of thirds, in O(n^1.47), and from 224 limbs (1792 bytes) Toom-4 cuts them into four parts and takes seven products of quarters, in O(n^1.40). Toom-3 needs a shorter operand of more than two thirds of the longer one's limbs, and Toom-4 of more than three quarters; one of a quarter to two thirds is cut into two parts and the longer one into four (Toom-4×2, five products), and a still shorter one multiplies the longer one in pieces of about twice its own length, of its own length below Toom-3. A square takes Karatsuba's method from 56 limbs, Toom-3 from 168 and Toom-4 from 224. From 360 limbs (2880 bytes) in the shorter operand, from 180 when the longer one has at least twice as many, and from 380 limbs for a square, the product is computed by number-theoretic transforms, in O(n·log n) for operands of n limbs together: the operands are cut into coefficients of 48 to 104 bits, their cyclic convolution is computed modulo four to eight primes below 2^29 by transforms of a power of two points, and each coefficient of the product is found from its residues by the Chinese remainder theorem, exactly, as the primes are chosen so that their product exceeds twice the largest coefficient the operands can give. On the targets other than aarch64 the transforms run in plain Rust, about four times slower, and take over from 16,384 limbs. This implementation carries it out as follows:

- Registers up to 48 bytes: when both vectors hold at most 16 bytes, the magnitudes are read into two limbs each, as the addition reads them, and the product, at most 32 bytes, is computed and written from registers; up to 32 bytes each, the same with four limbs an operand and eight for the product, and up to 48 bytes with six and twelve.
- One limb, little-endian: a product of a longer little-endian number by one of at most eight bytes is computed in one pass, by the kernel below for one row on the number's bytes as they are, limb by limb, in place for `a *= &b` and into the new vector for `&a * &b`; the top bytes then take the last carry, which gives the product's exact length and so its place. Measured as `a *= &b` with `b` of one byte, a number of 1 KiB took 150 cycles against 247 through the limbs of the general path (GMP's `mpz_mul_ui` 121 to 135), and of 128 bytes 42 against 74. Big-endian, the limbs would have to move up by the bytes the product gains, a second pass, which took 370 cycles for 1 KiB: such products keep the general path. The primitive integers multiply the same way (see [Traits and conversions](traits.md)).
- Limbs for longer operands: the magnitudes are read into native 64-bit limbs, the least significant first, in either byte order, into working memory on the stack (in the operator's own frame for operands of up to 16 limbs together), the product is computed there, and its bytes are written in the byte order of the left operand into the vector chosen for it. The kernels load and store limbs in pairs, and the working memory is laid out for them: aligned to 16 bytes, an operand of an odd number of limbs 8 bytes past, so that its pairs after the odd limb, which the kernels take first, start on 16 bytes, and the product and the scratch on 16 bytes; and the buffer on the stack is 2 KiB longer than the 8 KiB it hands out, which keeps the working memory from lying a multiple of 4 KiB below the caller's frame, where loads whose addresses agree with those of unrelated stores in their low 12 bits wait for them.
- Two carry chains a row: adding a row `x · y` puts two terms into each position, the low half of `x_k · y` and the high half of `x_(k−1) · y`, and one carry flag cannot take both. So the kernels run two chains over a step of positions, one after the other, and fold the first chain's last carry into the high half that the second adds next, which can always take it; the second chain's carry stays in the flag for the next step, so nothing is saved or restored between steps. On aarch64 this is inline assembly with `mul` and `umulh`, in steps of 8 positions for one row, and of 4 for two rows at once (four chains), which saves half of the steps' overhead on operands of up to 16 limbs; on x86_64 with `mulx` (BMI2, looked up at run time; the portable code without it), whose multiplications leave the flags alone.
- Karatsuba in five passes: the differences are taken with their signs apart, and the three products are recombined in place with five additions and subtractions of n limbs, after GMP's `mpn_toom22_mul`, using the addition's multi-chain kernels.
- Toom-Cook: the operands are evaluated at 0, 1, −1, 2 and infinity (the product of the top parts) for Toom-3 and Toom-4×2, and at −2 and ½ as well for Toom-4, by additions and shifts of the parts; the values at negative points are multiplied as magnitudes with their signs kept apart, and each value has one small limb more than a part. The coefficients of the product are recovered by Marco Bodrato's interpolation sequences, which GMP 6.3 uses too: additions and subtractions by the addition's kernels, halvings, and exact divisions by 3, 9 (as two by 3) and 15. Such a division by a divisor d of 2^64 − 1 takes `q = q·B − a·(2^64 − 1)/d`: a multiplication by one limb and one borrow chain, a cycle per limb on aarch64. Toom-4's sequence goes through negative values, held in two's complement, and halves only values that are not. The coefficients are then added at their positions, each one's high part folded into the next, so that every limb above them is written once.
- Squares: the products of two different limbs are computed once, as a triangle of rows, then doubled and the squares of the limbs added in one carry chain (`extr` shifts in the bit from the limb below without touching the flags). Rows of up to 10 limbs are added two at a time, the one product of the pair that the other row lacks added first and carried into the kernel; longer rows one at a time.
- Number-theoretic transforms: the primes are the eight largest of the form `c·2^21 + 1` below 2^29, so that every transform of up to 2^21 points has its roots of unity and a residue fits a 32-bit lane with room for the sums of a few butterflies; a NEON vector holds four. A product by a twiddle `w` takes three instructions, `sqrdmulh`, `mul` and `mls`, with `round(w·2^31/q)` stored beside `w` (after Becker, Hwang, Kannwischer, Yang and Yang, "Neon NTT", 2022), written in assembly as the compiler emitted four; products of two transformed values are signed Montgomery products, five instructions. Values are reduced only once every two levels, which the bounds of these products allow: every value stays below 2.1 times the prime, the 32-bit lanes below 4.57 times. The levels run two at a time (four vectors loaded and stored once per two levels), with the twiddles in bit-reversed order, where every block of every level has one twiddle and one table of half the length serves all levels; the last four levels run in registers on blocks of 16, the vectors transposed for the two levels whose butterflies lie within a vector, and the transform is left in that order, which the pointwise products do not mind and the inverse starts from. The first one or two levels run while the residues of the coefficients are formed, from digits of up to 28 bits, and the last inverse ones while the residues are scaled for the Chinese remainder theorem; a square takes one forward transform. The width of the coefficients and the number of primes are chosen for each product among six pairs, from 48 bits with 4 primes to 104 bits with 8, by their measured costs, so that the product fits a transform with as little of it unused as can be. The Chinese remainder theorem is the explicit one, `c = Σ y_i·(P/q_i) − k·P`, on 32-bit digits by widening multiply-adds four coefficients at a time, with `k` from a fixed-point sum of the `y_i/q_i`; the limbs of eight coefficients are then added at their positions, each limb of the product taking the pieces that land on it in one sum. Measured on an Apple M4 Max, a pass of two levels takes 0.37 cycles a butterfly, and in cycles a butterfly per bit of prime, 0.0134 against 0.025 for FLINT's `fft_small` way (primes below 2^50 in `f64` lanes with fused multiply-adds, two lanes a vector) and 0.032 for three 62-bit primes with scalar Shoup products, the other candidates prototyped; Schönhage–Strassen's method modulo `2^N + 1`, which GMP 6.3 uses, is the GMP column of the tables below.
- Portable kernels elsewhere: the other targets and Miri multiply with `u128` products in plain Rust, and run the transforms on four `i32`s, which compute what the NEON instructions compute.

An Apple M4 core issues three multiplications a cycle but only about four integer operations in all, of which about 2.4 can write the flags; a product of two limbs takes at least four (`mul`, `umulh` and two additions), and the kernels run at about 1 cycle per limb product there. Measured on an Apple M4 Max, the median of 21 to 31 runs, the schoolbook products against GMP 6.3's `mpn_mul_basecase` and `mpn_sqr_basecase`, in cycles:

| Limbs | Product | GMP | Square | GMP |
| --- | --- | --- | --- | --- |
| 4 × 4 | 26 | 29 | 28 | 24 |
| 8 × 8 | 75 | 84 | 72 | 70 |
| 16 × 16 | 260 | 291 | 215 | 248 |
| 32 × 32 | 1008 | 1187 | 657 | 842 |
| 48 × 48 | 2367 | 2932 | 1332 | 1692 |
| 64 × 64 | 4328 | 5424 | 2304 | 2710 |

And on limbs, with the methods above (from 512 limbs the transforms), against Karatsuba's method alone and GMP's `mpn_mul` and `mpn_sqr`, which take Toom-Cook methods of three to eight parts from about 80 limbs and FFT-based ones (Schönhage–Strassen's) from about 2400, the median of 9 to 11 interleaved runs, in cycles:

| Limbs | Karatsuba | Product | GMP | Karatsuba | Square | GMP |
| --- | --- | --- | --- | --- | --- | --- |
| 64 × 64 | 2923 | 2915 | 3179 | 2131 | 2142 | 2705 |
| 128 × 128 | 9128 | 8792 | 10,756 | 6639 | 6630 | 8458 |
| 256 × 256 | 28,984 | 27,026 | 32,455 | 20,457 | 18,829 | 25,648 |
| 512 × 512 | 85,741 | 55,624 | 77,844 | 62,325 | 43,427 | 67,068 |
| 1024 × 1024 | 259,897 | 117,076 | 228,438 | 188,798 | 90,451 | 178,954 |
| 2048 × 2048 | 788,923 | 249,128 | 613,868 | 570,613 | 187,889 | 494,828 |
| 4096 × 4096 | 2,410,218 | 543,137 | 1,286,459 | 1,728,705 | 403,056 | 923,968 |
| 8192 × 8192 | 7,262,216 | 1,106,177 | 2,855,455 | 5,201,543 | 828,051 | 2,195,770 |

Unbalanced products go to the transforms directly, whatever their shapes, from a shorter operand of 180 limbs: 1000 × 500 limbs took 90,577 cycles against GMP's 143,114 (and 134,626 by Toom-4×2), 1000 × 260 84,047 against 111,598 (110,788), 4000 × 700 360,008 against 730,974 (641,706), 1000 × 200 81,913 against 94,827 (91,828 by pieces), and 100,000 × 200 8.16 million against 8.80 million (9.21 million by pieces).

The transforms from 8 KiB to 8 MiB a factor, against Toom-4 (this crate's products without them) and GMP's `mpn_mul` and `mpn_sqr`, the median of 9 interleaved runs, in cycles, and the transforms' time as a fraction of GMP's:

| Limbs | Toom-4 | Product | GMP | × GMP | Toom-4 | Square | GMP | × GMP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1024 (8 KiB) | 203,957 | 117,076 | 228,438 | 0.51 | 147,888 | 90,451 | 178,954 | 0.51 |
| 2048 (16 KiB) | 537,469 | 249,128 | 613,868 | 0.41 | 417,287 | 187,889 | 494,828 | 0.38 |
| 4096 (32 KiB) | 1,552,648 | 543,137 | 1,286,459 | 0.42 | 1,124,932 | 403,056 | 923,968 | 0.44 |
| 8192 (64 KiB) | 4,015,021 | 1,106,177 | 2,855,455 | 0.39 | 3,147,018 | 828,051 | 2,195,770 | 0.38 |
| 16,384 (128 KiB) | 11,309,956 | 2,324,743 | 6,069,801 | 0.38 | 8,194,382 | 1,705,624 | 4,685,241 | 0.36 |
| 32,768 (256 KiB) | 28,924,313 | 5,703,471 | 15,594,785 | 0.37 | 22,657,848 | 4,121,393 | 12,088,151 | 0.34 |
| 65,536 (512 KiB) | 80,372,379 | 12,058,424 | 35,822,740 | 0.34 | 58,693,758 | 8,763,891 | 25,570,986 | 0.34 |
| 131,072 (1 MiB) | 205,725,584 | 24,436,010 | 86,055,985 | 0.28 | 160,786,902 | 17,979,048 | 59,979,623 | 0.30 |
| 262,144 (2 MiB) | | 51,302,225 | 191,906,281 | 0.27 | | 38,183,398 | 130,132,332 | 0.29 |
| 524,288 (4 MiB) | | 106,488,016 | 600,054,992 | 0.18 | | 76,689,673 | 441,818,409 | 0.17 |
| 1,048,576 (8 MiB) | | 224,651,103 | 1,205,573,993 | 0.19 | | 163,636,132 | 864,875,388 | 0.19 |

The crossover was measured in steps of 20 limbs: a product of 340 limbs each took 40,830 cycles by the transforms against 39,596 by Toom-4, and one of 360 limbs 42,043 against 43,462; a square of 360 limbs 32,363 against 31,882, and one of 380 limbs 33,001 against 34,781. A shorter operand of 180 limbs times one of 1000 took 81,429 cycles against 83,237 by pieces, and times one of 20,000, 1.61 million against 1.64 million; of 160 limbs times 10,000, 773,329 against 763,891.

Whole multiplications were measured on the same machine as `a *= &b` with room in `a`, over 4 MiB of operand pairs of the length given in bytes, random magnitudes in the shortest form, against GMP 6.3's `mpz_mul(a, a, b)` and num-bigint 0.4's `a *= &b` (the median of 11 interleaved runs, in cycles per call):

| Bytes | Little-endian | Big-endian | GMP | num-bigint |
| --- | --- | --- | --- | --- |
| 16 | 19 | 20 | 33 | 79 |
| 17 | 33 | 35 | 38 | 93 |
| 32 | 31 | 36 | 52 | 110 |
| 33 | 57 | 64 | 65 | 134 |
| 48 | 54 | 58 | 81 | 156 |
| 49 | 104 | 113 | 97 | 192 |
| 64 | 109 | 118 | 107 | 217 |
| 128 | 329 | 339 | 387 | 750 |
| 256 | 1017 | 1026 | 1201 | 2551 |
| 1024 | 9545 | 9509 | 11,261 | 29,665 |
| 4096 | 56,917 | 57,160 | 79,701 | 285,619 |
| 8192 | 120,563 | 122,914 | 231,882 | 767,544 |
| 16,384 | 254,165 | 256,322 | 622,698 | 2,236,411 |
| 32,768 | 549,258 | 547,740 | 1,256,757 | 6,899,683 |
| 65,536 | 1,124,129 | 1,128,700 | 2,851,315 | 17,091,112 |

And longer ones, `a *= &b` for one pair of operands at a time with room in `a`, against this crate's products without the transforms (Toom-4), GMP's `mpz_mul(a, a, b)` and num-bigint's `a *= &b`, which takes seconds beyond 2 MiB (the median of 5 interleaved runs, in cycles), with the squares, `&a * &a`, and two unbalanced products:

| Bytes | Product | Toom-4 | GMP | num-bigint |
| --- | --- | --- | --- | --- |
| 8 KiB | 121,459 | 204,435 | 238,387 | 777,278 |
| 16 KiB | 255,419 | 539,590 | 617,056 | 2,277,217 |
| 32 KiB | 541,346 | 1,526,185 | 1,281,804 | 7,029,676 |
| 64 KiB | 1,109,869 | 3,977,473 | 2,926,772 | 16,762,434 |
| 128 KiB | 2,348,339 | 11,284,923 | 6,207,328 | 48,757,584 |
| 256 KiB | 5,628,332 | 28,964,328 | 15,584,867 | 166,804,628 |
| 512 KiB | 12,115,301 | 80,692,825 | 36,835,457 | 377,162,063 |
| 1 MiB | 24,493,410 | 204,037,768 | 86,277,968 | 1,058,184,163 |
| 2 MiB | 52,748,430 | 566,295,586 | 192,942,875 | 2,888,868,144 |
| 4 MiB | 105,941,060 | 1,447,052,377 | 605,140,368 | |
| 8 MiB | 228,265,334 | 3,960,644,832 | 1,184,410,178 | |
| 8 KiB squared | 91,895 | 153,795 | 192,697 | 755,050 |
| 64 KiB squared | 844,400 | 3,130,536 | 2,147,352 | 16,830,292 |
| 1 MiB squared | 18,223,471 | 162,869,594 | 58,603,484 | 1,041,280,396 |
| 8 MiB squared | 164,955,086 | 2,940,220,201 | 869,260,493 | |
| 1 MiB × 64 KiB | 12,281,043 | 63,032,752 | 34,002,304 | 273,112,810 |
| 256 KiB × 8 KiB | 2,783,341 | 5,959,623 | 6,497,004 | 24,642,922 |

- Up to 48 bytes, in registers, a call takes 19–57 cycles, 12–42% fewer than GMP's.
- From 49 to 64 bytes, where the operands go through limbs in memory but are too short for the kernels to make up for it, a call takes about as long as GMP's (up to 7% longer).
- From 128 bytes to 2 KiB a call takes 9–15% fewer cycles than GMP's, by Karatsuba's method and from 960 bytes by Toom-Cook's; by the transforms, from 2880 bytes, 0.71 times GMP's at 4 KiB, 0.52 at 8 KiB and 0.39 to 0.44 from 16 KiB to 64 KiB, where GMP multiplies by Schönhage–Strassen's method.
- Squares, `&a * &a`, which builds the square in a new vector, took 0.75 to 0.93 times GMP's `mpz_mul(r, a, a)` into a prepared one from 256 bytes to 2 KiB, and by the transforms 0.59 at 4 KiB, 0.50 at 8 KiB and 0.39 to 0.45 from 16 KiB to 64 KiB; below 256 bytes the allocation of the new vector, about 50 cycles, dominates.
- From 8 KiB to 8 MiB a product takes 0.51 to 0.18 times GMP's time, 0.28 at 1 MiB, squares 0.48 to 0.19, and unbalanced ones 0.36 (1 MiB × 64 KiB) and 0.43 (256 KiB × 8 KiB); against the crate without the transforms, 0.59 at 8 KiB, 0.28 at 64 KiB, 0.12 at 1 MiB and 0.06 at 8 MiB.
- Against num-bigint, a call takes a quarter to about half of its time up to 2 KiB, 1.8 to 4.2 times faster from 16 bytes, 0.20 to 0.07 of it from 4 KiB to 64 KiB, and 0.02 at 1 MiB.
- The x86_64 kernels have been tested for correctness under emulation only; their speed has not been measured, nor that of the portable transforms there, which take over from 16,384 limbs: measured on the Apple M4 Max, the same code took 4.67 million cycles against 4.02 million by Toom-4 for a product of 8192 limbs each, and 9.52 million against 11.3 million for 16,384.
- From 64 bytes to a few KiB, a call moves by a few percent between builds, and some sizes by much more, with where the code and the working memory land: loads that wait for stores to addresses that agree in their low 12 bits, and pairs of limbs that straddle 16 bytes. The layout of the working memory avoids the cases measured, which took up to 2.5 times as long.

## Examples

Rust usage: `1000 * -1500` is `-1500000`, the bytes `[0x16, 0xE3, 0x60]` with `negative` set:

```rust
use a::math::numbers::{ByteOrder, Number};

fn main() {
  let a = Number::new_from_i32(1000, ByteOrder::BigEndian);
  let b = Number::new_from_i32(-1500, ByteOrder::BigEndian);
  let product = a * b;
  assert_eq!(product.value, vec![0x16, 0xE3, 0x60]);
  assert!(product.is_negative());
}
```

`*=` grows a number in its own vector when the vector has room: in big-endian order, `0xFF_FF * 0xFF_FF` is `[0xFF, 0xFE, 0x00, 0x01]`:

```rust
use a::math::numbers::{ByteOrder, Number};

fn main() {
  let mut value = Vec::with_capacity(4);
  value.extend_from_slice(&[0xFF, 0xFF]);
  let mut n = Number::new(value, ByteOrder::BigEndian);
  let vector = n.value.as_ptr();

  n *= &Number::new_from_u16(0xFFFF, ByteOrder::BigEndian);
  assert_eq!(n.value, vec![0xFF, 0xFE, 0x00, 0x01]);
  assert_eq!(n.value.as_ptr(), vector);
}
```

Mixed byte orders: the product is in the order of the left operand:

```rust
use a::math::numbers::{ByteOrder, Number};

fn main() {
  let big = Number::new_from_u16(0x0102, ByteOrder::BigEndian);
  let little = Number::new_from_u16(0x0304, ByteOrder::LittleEndian);

  let product = &big * &little;
  assert!(product.byte_order() == ByteOrder::BigEndian);
  assert_eq!(product.value, vec![0x03, 0x0A, 0x08]);

  let product = &little * &big;
  assert!(product.byte_order() == ByteOrder::LittleEndian);
  assert_eq!(product.value, vec![0x08, 0x0A, 0x03]);
}
```

Squares and zero: `&a * &a` squares `a`, and a product with zero is `[0]` and not negative, whatever the sign of the other operand:

```rust
use a::math::numbers::{ByteOrder, Number};

fn main() {
  let a = Number::new_from_i64(-0x1_0000_0001, ByteOrder::LittleEndian);
  let square = &a * &a;
  // (2^32 + 1)^2 = 2^64 + 2^33 + 1.
  assert_eq!(square.value, vec![1, 0, 0, 0, 2, 0, 0, 0, 1]);
  assert!(!square.is_negative());

  let zero = Number::new_from_u8(0, ByteOrder::BigEndian);
  let product = &a * &zero;
  assert_eq!(product.value, vec![0]);
  assert!(!product.is_negative());
}
```

Long operands: a product of two numbers of 1000 bytes has 1999 or 2000 bytes, here computed by Toom-3:

```rust
use a::math::numbers::{ByteOrder, Number};

fn main() {
  let a = Number::new(vec![0xFF; 1000], ByteOrder::BigEndian);
  let b = Number::new(vec![0xFF; 1000], ByteOrder::BigEndian);
  // (256^1000 - 1)^2 = 256^2000 - 2 * 256^1000 + 1.
  let product = &a * &b;
  assert_eq!(product.value.len(), 2000);
  assert_eq!(&product.value[..999], &[0xFF; 999][..]);
  assert_eq!(product.value[999], 0xFE);
  assert_eq!(&product.value[1000..1999], &[0; 999][..]);
  assert_eq!(product.value[1999], 1);
}
```

## Efficiency

- Time complexity: O(n·m) for operands of n and m limbs below 24 limbs in the shorter one, O(n^1.58) from there, O(n^1.47) from 120 limbs and O(n^1.40) from 224 for balanced operands, and O(n·log n) for operands of n limbs together from 360 limbs in the shorter one (180 for a longer one of at least twice as many), up to operands of 11 MiB each, and Toom-4 over such products beyond; a shorter operand of m limbs below that, a quarter of the longer one's or less, costs n/m times a product of about m limbs each, whatever the signs and byte orders
- Space complexity: the vector of the result, unless the product is built in an operand's vector; working memory of about 3.7 times the operands' limbs together for balanced operands (the operands, the product and the scratch), on the stack for products of up to 1 KiB and for balanced operands of up to 147 limbs (1176 bytes) each, one temporary buffer on the heap beyond; from 360 limbs, the operands and the product (2 times their limbs), and a second buffer for the transforms of 4 to 10 times the product's limbs
- Soundness: the bytes beyond a vector's length, which may be uninitialised, are never read, and every byte of a product is written before the vector's length is set; a borrowed operand is left as it was
