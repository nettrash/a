[A](../../../README.md) › [Math](../../../README.md#math)

# Multiplication

[Numbers](numbers.md) multiply with `*` and `*=`, each operand owned or borrowed: `a * b`, `a * &b`, `&a * b` and `&a * &b` give a new `Number`, and `a *= b` and `a *= &b` multiply `a`. The six forms give the same result for operands of any lengths, signs and byte orders; they differ only in the vector the product is built in.

- The product is exact. Each magnitude is read in its own byte order, and zero bytes above its most significant non-zero byte are ignored; an empty vector is zero, and a magnitude of zero is zero whatever `negative` says. Numbers built from the public fields, which need not be in the shortest form, multiply as the values they hold.
- The product is negative when exactly one operand is and neither is zero.
- The product is in the byte order of the left operand in every form, `&a * b` included, and `*=` keeps the order of the number it changes. The two operands' orders may differ.
- The product is in the shortest form: no zero bytes above the most significant one, and zero is `[0]`, never empty and never negative.
- The product is built where a sum would be: in `a`'s vector for `a *= &b`, `a *= b`, `a * &b` and `a * b` when its capacity is at least the length of the product (one byte for zero); in `b`'s for `&a * b`, and for `a * b` and `a *= b` when `a`'s is too small. Otherwise, and always for `&a * &b`, exactly one new vector is allocated. Products of up to 1 KiB allocate nothing else; longer ones may take one temporary buffer for the computation, freed before the product is returned.
- A borrowed operand is left as it was. A number multiplied by itself through one borrow, `&a * &a`, is squared: each product of two different limbs is computed once.

The method is the schoolbook one, on limbs of 8 bytes: each limb of one operand times the other, the rows added up, in O(n·m) for operands of n and m limbs. From 24 limbs (192 bytes) in the shorter operand, Karatsuba's method takes over: each operand is split into halves, `x = x1·B^k + x0`, and the product is made of three products of halves instead of four, `x1·y1`, `x0·y0` and `(x0 − x1)·(y0 − y1)`, each computed the same way, in O(n^1.58). A shorter operand of at most half the longer one's limbs multiplies it in pieces of its own length. This implementation carries it out as follows:

- Registers up to 32 bytes: when both vectors hold at most 16 bytes, the magnitudes are read into two limbs each, as the addition reads them, and the product, at most 32 bytes, is computed and written from registers; up to 32 bytes each, the same with four limbs an operand and eight for the product.
- Limbs for longer operands: the magnitudes are read into native 64-bit limbs, the least significant first, in either byte order, into working memory on the stack (in the operator's own frame for operands of up to 16 limbs together), the product is computed there, and its bytes are written in the byte order of the left operand into the vector chosen for it.
- Two carry chains a row: adding a row `x · y` puts two terms into each position, the low half of `x_k · y` and the high half of `x_(k−1) · y`, and one carry flag cannot take both. So the kernels run two chains over a step of positions, one after the other, and fold the first chain's last carry into the high half that the second adds next, which can always take it; the second chain's carry stays in the flag for the next step, so nothing is saved or restored between steps. On aarch64 this is inline assembly with `mul` and `umulh`, in steps of 8 positions for one row, and of 4 for two rows at once (four chains), which saves half of the steps' overhead on operands of up to 16 limbs; on x86_64 with `mulx` (BMI2, looked up at run time; the portable code without it), whose multiplications leave the flags alone.
- Karatsuba in five passes: the differences are taken with their signs apart, and the three products are recombined in place with five additions and subtractions of n limbs, after GMP's `mpn_toom22_mul`, using the addition's multi-chain kernels.
- Squares: the products of two different limbs are computed once, as a triangle of rows, then doubled and the squares of the limbs added in one carry chain (`extr` shifts in the bit from the limb below without touching the flags); Karatsuba's method takes over from 64 limbs.
- Portable kernels elsewhere: the other targets and Miri multiply with `u128` products in plain Rust.

An Apple M4 core issues three multiplications a cycle but only about four integer operations in all, of which about 2.4 can write the flags; a product of two limbs takes at least four (`mul`, `umulh` and two additions), and the kernels run at about 1 cycle per limb product there. Measured on an Apple M4 Max, the median of 31 runs, the schoolbook products against GMP 6.3's `mpn_mul_basecase` and `mpn_sqr_basecase`, in cycles:

| Limbs | Product | GMP | Square | GMP |
| --- | --- | --- | --- | --- |
| 4 × 4 | 26 | 29 | 28 | 24 |
| 8 × 8 | 75 | 84 | 79 | 69 |
| 16 × 16 | 260 | 291 | 246 | 256 |
| 32 × 32 | 1008 | 1187 | 743 | 722 |
| 48 × 48 | 2367 | 2932 | 1433 | 1530 |
| 64 × 64 | 4328 | 5424 | 2401 | 2727 |

And on limbs, with Karatsuba's method, against GMP's `mpn_mul` and `mpn_sqr`, which take Toom-Cook methods of three and more parts from about a hundred limbs, and FFT-based ones for the longest operands:

| Limbs | Product | GMP | Square | GMP |
| --- | --- | --- | --- | --- |
| 24 × 24 | 529 | 601 | 468 | 533 |
| 64 × 64 | 2825 | 3171 | 2383 | 2738 |
| 128 × 128 | 8965 | 10,660 | 7360 | 8599 |
| 256 × 256 | 27,887 | 32,568 | 22,786 | 26,131 |
| 512 × 512 | 85,654 | 77,056 | 70,156 | 66,554 |
| 1024 × 1024 | 262,231 | 227,628 | 214,043 | 183,740 |

Whole multiplications were measured on the same machine as `a *= &b` with room in `a`, over 4 MiB of operand pairs of the length given in bytes, random magnitudes in the shortest form, against GMP 6.3's `mpz_mul(a, a, b)` and num-bigint 0.4's `a *= &b` (the median of 15 interleaved runs, in cycles per call):

| Bytes | Little-endian | Big-endian | GMP | num-bigint |
| --- | --- | --- | --- | --- |
| 16 | 16 | 18 | 38 | 96 |
| 17 | 38 | 35 | 42 | 106 |
| 32 | 34 | 36 | 57 | 127 |
| 48 | 82 | 94 | 93 | 168 |
| 64 | 103 | 111 | 126 | 227 |
| 128 | 319 | 332 | 408 | 773 |
| 256 | 1000 | 1019 | 1241 | 2556 |
| 1024 | 9683 | 9671 | 11,330 | 30,795 |
| 4096 | 88,426 | 88,289 | 80,396 | 283,984 |
| 8192 | 264,259 | 266,131 | 231,415 | 768,884 |

- Up to 32 bytes, in registers, a call takes 16–38 cycles, 10–58% fewer than GMP's.
- From 33 to 48 bytes, where the operands go through limbs in memory but are too short for the kernels to make up for it, a call takes about as long as GMP's (up to 25% longer at 33–40 bytes).
- From 64 bytes to 2 KiB a call takes 2–22% fewer cycles than GMP's; from 4 KiB, where GMP has switched to its Toom-Cook methods, 10–14% more.
- Against num-bigint, a call takes a third to a half of its time, 2.3–3.7× faster from 17 bytes.
- The x86_64 kernels have been tested for correctness under emulation only; their speed has not been measured.

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

Long operands: a product of two numbers of 1000 bytes has 1999 or 2000 bytes, here computed by Karatsuba's method:

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

- Time complexity: O(n·m) for operands of n and m limbs below 24 limbs in the shorter one, O(n^1.58) from there for balanced operands (O(n·m^0.58) for a shorter operand of m limbs), whatever the signs and byte orders
- Space complexity: the vector of the result, unless the product is built in an operand's vector; working memory of about six times the operands' limbs, on the stack for products of up to 1 KiB and of up to about 2.5 KiB for balanced operands, one temporary buffer on the heap beyond
- Soundness: the bytes beyond a vector's length, which may be uninitialised, are never read, and every byte of a product is written before the vector's length is set; a borrowed operand is left as it was
