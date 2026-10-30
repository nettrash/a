[A](../../../README.md) › [Math](../../../README.md#math)

# Multiplication

[Numbers](numbers.md) multiply with `*` and `*=`, each operand owned or borrowed: `a * b`, `a * &b`, `&a * b` and `&a * &b` give a new `Number`, and `a *= b` and `a *= &b` multiply `a`. The six forms give the same result for operands of any lengths, signs and byte orders; they differ only in the vector the product is built in.

- The product is exact. Each magnitude is read in its own byte order, and zero bytes above its most significant non-zero byte are ignored; an empty vector is zero, and a magnitude of zero is zero whatever `negative` says. Numbers built from the public fields, which need not be in the shortest form, multiply as the values they hold.
- The product is negative when exactly one operand is and neither is zero.
- The product is in the byte order of the left operand in every form, `&a * b` included, and `*=` keeps the order of the number it changes. The two operands' orders may differ.
- The product is in the shortest form: no zero bytes above the most significant one, and zero is `[0]`, never empty and never negative.
- The product is built where a sum would be: in `a`'s vector for `a *= &b`, `a *= b`, `a * &b` and `a * b` when its capacity is at least the length of the product (one byte for zero); in `b`'s for `&a * b`, and for `a * b` and `a *= b` when `a`'s is too small. Otherwise, and always for `&a * &b`, exactly one new vector is allocated. Products of up to 1 KiB allocate nothing else; longer ones may take one temporary buffer for the computation, freed before the product is returned.
- A borrowed operand is left as it was. A number multiplied by itself through one borrow, `&a * &a`, is squared: each product of two different limbs is computed once.

The method is the schoolbook one, on limbs of 8 bytes: each limb of one operand times the other, the rows added up, in O(n·m) for operands of n and m limbs. From 24 limbs (192 bytes) in the shorter operand, Karatsuba's method takes over: each operand is split into halves, `x = x1·B^k + x0`, and the product is made of three products of halves instead of four, `x1·y1`, `x0·y0` and `(x0 − x1)·(y0 − y1)`, each computed the same way, in O(n^1.58). From 120 limbs (960 bytes), Toom-3 cuts each operand into three parts and makes the product of five products of thirds, in O(n^1.47), and from 224 limbs (1792 bytes) Toom-4 cuts them into four parts and takes seven products of quarters, in O(n^1.40). Toom-3 needs a shorter operand of more than two thirds of the longer one's limbs, and Toom-4 of more than three quarters; one of a quarter to two thirds is cut into two parts and the longer one into four (Toom-4×2, five products), and a still shorter one multiplies the longer one in pieces of about twice its own length, of its own length below Toom-3. A square takes Karatsuba's method from 56 limbs, Toom-3 from 168 and Toom-4 from 224. This implementation carries it out as follows:

- Registers up to 48 bytes: when both vectors hold at most 16 bytes, the magnitudes are read into two limbs each, as the addition reads them, and the product, at most 32 bytes, is computed and written from registers; up to 32 bytes each, the same with four limbs an operand and eight for the product, and up to 48 bytes with six and twelve.
- One limb, little-endian: a product of a longer little-endian number by one of at most eight bytes is computed in one pass, by the kernel below for one row on the number's bytes as they are, limb by limb, in place for `a *= &b` and into the new vector for `&a * &b`; the top bytes then take the last carry, which gives the product's exact length and so its place. Measured as `a *= &b` with `b` of one byte, a number of 1 KiB took 150 cycles against 247 through the limbs of the general path (GMP's `mpz_mul_ui` 121 to 135), and of 128 bytes 42 against 74. Big-endian, the limbs would have to move up by the bytes the product gains, a second pass, which took 370 cycles for 1 KiB: such products keep the general path. The primitive integers multiply the same way (see [Traits and conversions](traits.md)).
- Limbs for longer operands: the magnitudes are read into native 64-bit limbs, the least significant first, in either byte order, into working memory on the stack (in the operator's own frame for operands of up to 16 limbs together), the product is computed there, and its bytes are written in the byte order of the left operand into the vector chosen for it. The kernels load and store limbs in pairs, and the working memory is laid out for them: aligned to 16 bytes, an operand of an odd number of limbs 8 bytes past, so that its pairs after the odd limb, which the kernels take first, start on 16 bytes, and the product and the scratch on 16 bytes; and the buffer on the stack is 2 KiB longer than the 8 KiB it hands out, which keeps the working memory from lying a multiple of 4 KiB below the caller's frame, where loads whose addresses agree with those of unrelated stores in their low 12 bits wait for them.
- Two carry chains a row: adding a row `x · y` puts two terms into each position, the low half of `x_k · y` and the high half of `x_(k−1) · y`, and one carry flag cannot take both. So the kernels run two chains over a step of positions, one after the other, and fold the first chain's last carry into the high half that the second adds next, which can always take it; the second chain's carry stays in the flag for the next step, so nothing is saved or restored between steps. On aarch64 this is inline assembly with `mul` and `umulh`, in steps of 8 positions for one row, and of 4 for two rows at once (four chains), which saves half of the steps' overhead on operands of up to 16 limbs; on x86_64 with `mulx` (BMI2, looked up at run time; the portable code without it), whose multiplications leave the flags alone.
- Karatsuba in five passes: the differences are taken with their signs apart, and the three products are recombined in place with five additions and subtractions of n limbs, after GMP's `mpn_toom22_mul`, using the addition's multi-chain kernels.
- Toom-Cook: the operands are evaluated at 0, 1, −1, 2 and infinity (the product of the top parts) for Toom-3 and Toom-4×2, and at −2 and ½ as well for Toom-4, by additions and shifts of the parts; the values at negative points are multiplied as magnitudes with their signs kept apart, and each value has one small limb more than a part. The coefficients of the product are recovered by Marco Bodrato's interpolation sequences, which GMP 6.3 uses too: additions and subtractions by the addition's kernels, halvings, and exact divisions by 3, 9 (as two by 3) and 15. Such a division by a divisor d of 2^64 − 1 takes `q = q·B − a·(2^64 − 1)/d`: a multiplication by one limb and one borrow chain, a cycle per limb on aarch64. Toom-4's sequence goes through negative values, held in two's complement, and halves only values that are not. The coefficients are then added at their positions, each one's high part folded into the next, so that every limb above them is written once.
- Squares: the products of two different limbs are computed once, as a triangle of rows, then doubled and the squares of the limbs added in one carry chain (`extr` shifts in the bit from the limb below without touching the flags). Rows of up to 10 limbs are added two at a time, the one product of the pair that the other row lacks added first and carried into the kernel; longer rows one at a time.
- Portable kernels elsewhere: the other targets and Miri multiply with `u128` products in plain Rust.

An Apple M4 core issues three multiplications a cycle but only about four integer operations in all, of which about 2.4 can write the flags; a product of two limbs takes at least four (`mul`, `umulh` and two additions), and the kernels run at about 1 cycle per limb product there. Measured on an Apple M4 Max, the median of 21 to 31 runs, the schoolbook products against GMP 6.3's `mpn_mul_basecase` and `mpn_sqr_basecase`, in cycles:

| Limbs | Product | GMP | Square | GMP |
| --- | --- | --- | --- | --- |
| 4 × 4 | 26 | 29 | 28 | 24 |
| 8 × 8 | 75 | 84 | 72 | 70 |
| 16 × 16 | 260 | 291 | 215 | 248 |
| 32 × 32 | 1008 | 1187 | 657 | 842 |
| 48 × 48 | 2367 | 2932 | 1332 | 1692 |
| 64 × 64 | 4328 | 5424 | 2304 | 2710 |

And on limbs, with the methods above, against Karatsuba's method alone and GMP's `mpn_mul` and `mpn_sqr`, which take Toom-Cook methods of three to eight parts from about 80 limbs and FFT-based ones from about 2400, the median of 11 interleaved runs, in cycles:

| Limbs | Karatsuba | Product | GMP | Karatsuba | Square | GMP |
| --- | --- | --- | --- | --- | --- | --- |
| 64 × 64 | 2923 | 2915 | 3179 | 2131 | 2142 | 2705 |
| 128 × 128 | 9128 | 8792 | 10,756 | 6639 | 6630 | 8458 |
| 256 × 256 | 28,984 | 27,026 | 32,455 | 20,457 | 18,829 | 25,648 |
| 512 × 512 | 85,741 | 72,312 | 78,455 | 62,325 | 54,517 | 66,253 |
| 1024 × 1024 | 259,897 | 200,615 | 227,879 | 188,798 | 150,589 | 187,691 |
| 2048 × 2048 | 788,923 | 548,517 | 619,961 | 570,613 | 417,768 | 493,405 |
| 4096 × 4096 | 2,410,218 | 1,517,055 | 1,259,563 | 1,728,705 | 1,120,140 | 915,413 |
| 8192 × 8192 | 7,262,216 | 3,994,358 | 2,834,901 | 5,201,543 | 3,149,916 | 2,180,805 |

Unbalanced, by Toom-4×2 and pieces: 1000 × 500 limbs in 130,817 cycles against GMP's 141,864, 1000 × 260 in 108,049 against 110,271, and 4000 × 700 in 638,182 against 729,252.

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
| 4096 | 73,011 | 72,527 | 80,262 | 284,601 |
| 8192 | 210,057 | 208,823 | 235,309 | 777,235 |
| 16,384 | 550,893 | 549,392 | 628,143 | 2,260,783 |
| 32,768 | 1,537,732 | 1,549,896 | 1,264,371 | 6,782,362 |
| 65,536 | 4,020,802 | 4,025,590 | 2,868,617 | 17,010,794 |

- Up to 48 bytes, in registers, a call takes 19–57 cycles, 12–42% fewer than GMP's.
- From 49 to 64 bytes, where the operands go through limbs in memory but are too short for the kernels to make up for it, a call takes about as long as GMP's (up to 7% longer).
- From 128 bytes to 16 KiB a call takes 9–15% fewer cycles than GMP's, by Karatsuba's method and from 960 bytes by Toom-Cook's; from 32 KiB, where GMP multiplies by FFT, 1.2 times as long as GMP's at 32 KiB and 1.4 times at 64 KiB.
- Squares, `&a * &a`, which builds the square in a new vector, took 0.81 to 0.93 times GMP's `mpz_mul(r, a, a)` into a prepared one from 256 bytes to 16 KiB, and 1.46 times at 64 KiB; below 256 bytes the allocation of the new vector, about 50 cycles, dominates.
- Against num-bigint, a call takes a quarter to about half of its time, 1.8 to 4.2 times faster from 16 bytes.
- From 64 bytes to a few KiB, a call moves by a few percent between builds, and some sizes by much more, with where the code and the working memory land: loads that wait for stores to addresses that agree in their low 12 bits, and pairs of limbs that straddle 16 bytes. The layout of the working memory avoids the cases measured, which took up to 2.5 times as long.
- The x86_64 kernels have been tested for correctness under emulation only; their speed has not been measured.

## Examples

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

- Time complexity: O(n·m) for operands of n and m limbs below 24 limbs in the shorter one, O(n^1.58) from there, O(n^1.47) from 120 limbs and O(n^1.40) from 224 for balanced operands; a shorter operand of m limbs, a quarter of the longer one's or less, costs n/m times a product of about m limbs each, whatever the signs and byte orders
- Space complexity: the vector of the result, unless the product is built in an operand's vector; working memory of about 3.7 times the operands' limbs together for balanced operands (the operands, the product and the scratch), on the stack for products of up to 1 KiB and for balanced operands of up to 147 limbs (1176 bytes) each, one temporary buffer on the heap beyond
- Soundness: the bytes beyond a vector's length, which may be uninitialised, are never read, and every byte of a product is written before the vector's length is set; a borrowed operand is left as it was
