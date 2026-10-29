[A](../../../README.md) › [Math](../../../README.md#math)

# Subtraction

[Numbers](numbers.md) subtract with `-` and `-=`, each operand owned or borrowed: `a - b`, `a - &b`, `&a - b` and `&a - &b` give a new `Number`, and `a -= b` and `a -= &b` subtract from `a`. The six forms give the same result for operands of any lengths, signs and byte orders; they differ only in the vector the difference is built in.

- The difference is exact. Each magnitude is read in its own byte order, and zero bytes above its most significant non-zero byte are ignored; an empty vector is zero, and a magnitude of zero is zero whatever `negative` says. Numbers built from the public fields, which need not be in the shortest form, subtract as the values they hold.
- The difference is in the byte order of the left operand in every form, `&a - b` included, and `-=` keeps the order of the number it changes. The two operands' orders may differ.
- The difference is in the shortest form: no zero bytes above the most significant one, and zero is `[0]`, never empty and never negative, so `a - a` is `[0]` whatever the sign of `a`.
- The difference is built where the sum would be: in `a`'s vector for `a -= &b`, `a -= b`, `a - &b` and `a - b` when its capacity is at least the length of the difference (one byte for zero), and then nothing at all is allocated; in `b`'s for `&a - b`, its bytes reversed first if the orders differ, and for `a - b` and `a -= b` when `a`'s is too small. When no vector the form may use is large enough, and always for `&a - &b`, exactly one new vector is allocated.
- A borrowed operand is left as it was, its sign included.

The method is `a - b = a + (-b)`: the difference is the [sum](add.md) of `a` and of `b` with its sign turned over, and so the schoolbook method on signs and magnitudes. When the signs of `a` and `b` agree, as in `5 - 3`, the smaller magnitude is subtracted from the larger, and the difference has the sign of `a` when `a` has the larger magnitude, the opposite sign otherwise; when they differ, as in `5 - (-3)`, the magnitudes are added and the difference has the sign of `a`. This implementation carries it out as follows:

- The addition's code, unchanged: the difference is computed by the code of `+` and `+=` as it is, the short path in registers up to 16 bytes, the length planned before any byte is written, eight bytes at a time, the borrow (and the carry) in the processor's carry flag through inline assembly on 64-bit little-endian aarch64 (`sbcs`, `adcs`, in several interleaved chains for long operands) and on x86_64 (`sbb`, `adc`), and portable kernels elsewhere. [Addition](add.md) describes each of these.
- The sign turned over where it costs least. Operands of at most 16 bytes take the addition's short path, inlined, with the sign of `b` turned over as it is read into a register. An owned `b` has its sign flag turned over in place, and is then added. `a -= &b` on longer operands cannot change `b`, so it turns over the sign of `a` instead and computes `-(-a + b)`: the sign of the sum is turned back without a branch (it is a coin toss for random operands), unless the sum is zero; only the sign of `a` passes through memory on its way to the addition, and there it decides branches only, which the core predicts. `&a - &b` on longer operands can change neither operand: it adds to `a` a copy of the fields of `b` with the opposite sign, which shares the vector of `b`, is only read and is never dropped. Reading the vector's address and length back from that copy costs a few cycles, which the allocation of the difference's vector hides.
- Long differences start at once: which magnitude is the larger is a coin toss for random operands of the same length, and so the addition picks the operands of a subtraction without a branch. But then the kernel cannot load a limb before the most significant limbs of both operands have been read and compared, which costs 12–23 cycles once the operands are out of the first-level cache. From 256 bytes little-endian and 512 big-endian, where the kernels run several carry chains, a difference built in an operand's vector picks them by branches instead: a predicted branch lets the kernel start while the comparison is still under way, and a mispredicted one costs about what the wait did. The sum of numbers of opposite signs takes the same path.

The figures below were measured on an Apple M4 Max as `a -= &b` with room in `a`, over 4 MiB of operand pairs of the length given, random magnitudes in the shortest form, against GMP 6.3's `mpz_sub(a, a, b)` on the same values (the median of 15 interleaved runs, in cycles per call). With the same signs the magnitudes are subtracted, with opposite signs they are added:

| Bytes | Same signs, little-endian | Same signs, big-endian | Opposite signs, little-endian | Opposite signs, big-endian | GMP, same signs | GMP, opposite signs |
| --- | --- | --- | --- | --- | --- | --- |
| 17 | 33 | 38 | 27 | 31 | 15 | 14 |
| 32 | 27 | 32 | 24 | 28 | 15 | 18 |
| 64 | 27 | 36 | 26 | 31 | 17 | 19 |
| 128 | 37 | 46 | 31 | 38 | 34 | 37 |
| 256 | 71 | 73 | 51 | 58 | 63 | 64 |
| 512 | 94 | 114 | 83 | 101 | 99 | 102 |
| 1024 | 157 | 194 | 140 | 182 | 161 | 167 |
| 4096 | 486 | 636 | 491 | 682 | 617 | 701 |

- A difference costs what the sum of `a` and `-b` does: measured call by call without link-time optimisation, within one cycle up to 16 bytes, and 0–4 cycles more from 17 to 128 bytes, where turning the sign over takes 6–25 instructions more a call; from 256 bytes, the same.
- Against GMP, a call takes 6–23 cycles more from 17 to 64 bytes, as for the sum, for the significant lengths and the plan GMP's normalised limb counts spare it; within 8 cycles from 128 to 1024 bytes little-endian, and a fifth fewer at 4096. Big-endian, with its byte reversals, it takes 3–20% more from 512 bytes.
- Long differences: with their operands picked without a branch, as for shorter ones, the differences of same-sign operands above took 105, 170 and 497 cycles at 512, 1024 and 4096 bytes little-endian, and 118, 204 and 646 big-endian. Over 512 pairs that live in the second-level cache, a random larger operand, which the branch mispredicts half of the time, saves 11–13 cycles from 512 bytes little-endian and 3–19 big-endian, and costs up to 10 more at 256 bytes; a predictable one saves 12 cycles at 256 bytes and 19–23 from 320 bytes little-endian, and 7–23 big-endian from 512.

## Examples

Rust usage: `1000 - 1500` is `-500`, the bytes `[0x01, 0xF4]` with `negative` set:

```rust
use a::math::numbers::{ByteOrder, Number};

fn main() {
  let a = Number::new_from_i32(1000, ByteOrder::BigEndian);
  let b = Number::new_from_i32(1500, ByteOrder::BigEndian);
  let difference = a - b;
  assert_eq!(difference.value, vec![0x01, 0xF4]);
  assert!(difference.is_negative());
}
```

`-=` shrinks a number in its own vector: in big-endian order, `0x01_00 - 1` is `[0xFF]`:

```rust
use a::math::numbers::{ByteOrder, Number};

fn main() {
  let mut n = Number::new(vec![0x01, 0x00], ByteOrder::BigEndian);
  let vector = n.value.as_ptr();

  let one = Number::new_from_u8(1, ByteOrder::BigEndian);
  n -= &one;
  assert_eq!(n.value, vec![0xFF]);
  assert_eq!(n.value.as_ptr(), vector);
}
```

Mixed byte orders: the difference is in the order of the left operand:

```rust
use a::math::numbers::{ByteOrder, Number};

fn main() {
  let big = Number::new_from_u16(0x0102, ByteOrder::BigEndian);
  let little = Number::new_from_u16(0x0304, ByteOrder::LittleEndian);

  let difference = &big - &little;
  assert!(difference.byte_order() == ByteOrder::BigEndian);
  assert_eq!(difference.value, vec![0x02, 0x02]);
  assert!(difference.is_negative());

  let difference = &little - &big;
  assert!(difference.byte_order() == ByteOrder::LittleEndian);
  assert_eq!(difference.value, vec![0x02, 0x02]);
  assert!(!difference.is_negative());
}
```

Signs: subtracting a negative number adds its magnitude, and a number minus itself is `[0]` and not negative, a negative one too:

```rust
use a::math::numbers::{ByteOrder, Number};

fn main() {
  let a = Number::new_from_i32(-1000, ByteOrder::LittleEndian);
  let b = Number::new_from_i32(-1500, ByteOrder::LittleEndian);
  // -1000 - -1500 = 500.
  let difference = &a - &b;
  assert_eq!(difference.value, vec![0xF4, 0x01]);
  assert!(!difference.is_negative());

  let zero = &a - &a;
  assert_eq!(zero.value, vec![0]);
  assert!(!zero.is_negative());
}
```

Borrowed and owned operands: `&a - &b` builds the difference in a new vector and leaves both operands as they were, signs included, and `a - &b` builds the same difference in `a`'s vector:

```rust
use a::math::numbers::{ByteOrder, Number};

fn main() {
  let a = Number::new_from_u16(0x0100, ByteOrder::LittleEndian);
  let b = Number::new_from_i16(-0x00FF, ByteOrder::LittleEndian);

  let difference = &a - &b;
  assert_eq!(difference.value, vec![0xFF, 0x01]);
  assert_eq!(a.value, vec![0x00, 0x01]);
  assert!(b.is_negative());

  let mut value = Vec::with_capacity(4);
  value.extend_from_slice(&[0x00, 0x01]);
  let a = Number::new(value, ByteOrder::LittleEndian);
  let vector = a.value.as_ptr();
  let difference = a - &b;
  assert_eq!(difference.value, vec![0xFF, 0x01]);
  assert_eq!(difference.value.as_ptr(), vector);
}
```

## Efficiency

- Time complexity: O(n) for n bytes of the longer operand, whatever the signs and byte orders, as for the [sum](add.md)
- Space complexity: nothing besides the vector of the result; no allocation at all when the difference is built in an operand's vector, otherwise exactly one, of a new vector (always for `&a - &b`, a difference of zero included)
- Soundness: the bytes beyond a vector's length, which may be uninitialised, are never read; a borrowed operand is left as it was, and the copy of a borrowed right operand's fields that `&a - &b` adds lives only while the operand is borrowed, is never written through and never frees the vector it shares
