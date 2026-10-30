[A](../../../README.md) › [Math](../../../README.md#math)

# Shifts

[Numbers](numbers.md) shift by a number of bits with `<<`, `>>`, `<<=` and `>>=`, the number owned or borrowed and the amount a `usize`: `a << k`, `&a << k`, `a >> k` and `&a >> k` give a new `Number`, and `a <<= k` and `a >>= k` shift `a`. The forms give the same result for a number of any length, sign and byte order; they differ only in the vector the result is built in.

- The rules are those of Rust's signed integers, without the bits that a fixed width loses at the top. `a << k` is `a · 2^k`, exactly, however long it gets. `a >> k` is `a / 2^k` rounded towards negative infinity, the arithmetic shift of `i32`: `7 >> 1` is `3`, `-7 >> 1` is `-4`, `-1 >> k` is `-1` for every `k`, and a number shifted right by its length in bits or more is `0`, or `-1` when it is negative. On a sign and a magnitude, a right shift of a negative number adds one to the magnitude of the quotient when a bit shifted out is not zero, which can carry into a new byte: `-0x1FF >> 1` is `-0x100`.
- The magnitude is read in its own byte order, and zero bytes above its most significant non-zero byte are ignored; an empty vector is zero, and a magnitude of zero is zero whatever `negative` says. Numbers built from the public fields, which need not be in the shortest form, shift as the values they hold.
- The result is in the byte order of the number shifted, in the shortest form: no zero bytes above the most significant one, and zero is `[0]`, never empty and never negative.
- The result is built in `a`'s vector for `a << k`, `a <<= k`, `a >> k` and `a >>= k` when its capacity is at least the length of the result (one byte for zero), and then nothing at all is allocated. A right shift is never longer than its operand, so `a >> k` and `a >>= k` always build it there, except for zero in a vector without capacity. Otherwise, and always for `&a << k` and `&a >> k`, exactly one new vector of exactly the result's length is allocated, and nothing else.
- Any amount works: a right shift by `usize::MAX` gives `0` or `-1`, and zero shifted left by any amount is zero. A left shift whose result does not fit in memory panics, as `Vec` does on a capacity overflow, rather than aborting or wrapping around.
- A borrowed operand is left as it was.

The method is the textbook one: a shift by `k = 8q + r` bits moves the bytes of the magnitude by `q` positions and their bits by `r`, each byte of the result made of the bits of two neighbouring bytes. This implementation carries it out as follows:

- Planned first: the significant length of the magnitude and its top byte (one read for a number in the shortest form), the length of the result, which is one byte longer for `<<` when the top `r` bits of the top byte are not zero and one byte shorter for `>>` when nothing is left of the top byte, and for a negative `>>` whether it is rounded down, from the bits shifted out, eight bytes at a time (one read for a random number). Only then is the vector chosen and are the bytes written, once each.
- Registers up to 16 bytes: a number whose vector holds at most 16 bytes, shifted left by fewer than 128 bits or right by any amount, is read into a `u128` as the [addition](add.md) reads it, shifted, and the result, at most 32 bytes, written from registers. `<<=` of 17 to 32 bytes by fewer than 64 bits does the same with four limbs, a result of up to 40 bytes.
- Whole bytes are moved: inline up to 64 bytes and with `memmove` beyond. Big-endian, the least significant byte is at the end of the vector, so in place `>>` by whole bytes moves nothing (it shortens the vector) and `<<` only writes the zeros after the number.
- Bits in byte lanes on aarch64: a byte of `x << r` is `x[i] << r | x[i - 1] >> (8 - r)`, and of `x >> r` `x[i] >> r | x[i + 1] << (8 - r)`, with no carry from one byte to the next. So 16 bytes are one NEON load, an `ext` that takes their neighbours from the block next to them, which the loop loads anyway, an `ushl` of the bytes shifted right and an `mla` that adds the others times `2^r`: in a byte, the product drops the bits shifted out as a shift does, and the two have no bit in common, so their sum is their `or`. It works in the order of memory, where the neighbour of a big-endian byte is the one after it, so both byte orders shift at the same speed without reversing any byte. From 64 bytes, the blocks are stored at addresses on 16 bytes, with one more block over an end: a shift by whole bytes and bits would otherwise put every store across 16 bytes, which costs more than a load. Elsewhere, and under Miri, portable kernels shift eight bytes at a time as `u64`s.
- In place: little-endian `<<` moves the bytes up and is written from the top down; the other shifts move them down, or big-endian `<<` one byte towards the end of the vector, and are written from the bottom up, each block of 16 bytes loaded before a store reaches it.
- No branch on coin tosses: for random numbers, whether `<<` grows by a byte, whether `>>` loses one, and whether a negative quotient is rounded are coin tosses for some amounts. The byte above the top is written without a branch, the kernels count their blocks from the input's length rather than the result's, and the rounding adds zero or one to the quotient's eight least significant bytes, computed from the input before the kernel writes over it, so they are never read back from stores still in flight.

Whole shifts were measured on an Apple M4 Max as `a <<= k` and `a >>= k` with room in `a`, over 1 MiB of random numbers of the length given in bytes, in the shortest form, against GMP 6.3's `mpz_mul_2exp(a, a, k)` and `mpz_fdiv_q_2exp(a, a, k)` (floor, as here) and num-bigint 0.4's `a <<= k` and `a >>= k`, by 13 bits and by 64 (the median of 9 interleaved runs, in cycles per call). Left shifts of positive numbers:

| Bytes | 13: LE | BE | GMP | num-bigint | 64: LE | BE | GMP | num-bigint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16 | 11 | 13 | 9 | 20 | 9 | 11 | 16 | 69 |
| 64 | 18 | 21 | 14 | 24 | 15 | 15 | 16 | 68 |
| 256 | 33 | 32 | 34 | 41 | 30 | 15 | 30 | 102 |
| 1024 | 113 | 118 | 137 | 104 | 110 | 15 | 87 | 182 |
| 4096 | 358 | 324 | 518 | 320 | 308 | 16 | 310 | 426 |
| 65,536 | 4294 | 4184 | 5909 | 4309 | 4244 | 57 | 4151 | 7767 |

Right shifts of negative numbers, rounded towards negative infinity:

| Bytes | 13: LE | BE | GMP | num-bigint | 64: LE | BE | GMP | num-bigint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16 | 10 | 11 | 12 | 19 | 10 | 11 | 12 | 21 |
| 64 | 18 | 18 | 16 | 23 | 16 | 13 | 13 | 30 |
| 256 | 32 | 32 | 32 | 37 | 30 | 13 | 28 | 39 |
| 1024 | 107 | 101 | 135 | 101 | 105 | 13 | 90 | 87 |
| 4096 | 324 | 309 | 439 | 386 | 315 | 16 | 313 | 264 |
| 65,536 | 4235 | 3914 | 5810 | 5254 | 4138 | 41 | 4082 | 4092 |

- Up to 64 bytes, in registers or in one to four blocks of the kernels, a shift takes 8.5 to 21 cycles: `>>` 0.8 to 1.2 times GMP's time; `<<` by bits 1.06 to 1.6 times (0.6 to 6 cycles more, big-endian up to 3 cycles more than little-endian), and by 64 bits as fast or up to twice as fast; against num-bigint, 1.2 to 8 times faster.
- A shift with bits takes about GMP's time at 256 bytes, and 63% to 86% of it from 1 KiB, about 0.065 cycles a byte at 64 KiB, where the bytes come from the level-2 cache. Shifts by whole bytes are moves, by `memmove`, which take up to 26% more than GMP's at 1 KiB and as long from 4 KiB.
- Big-endian shifts by whole bytes take constant time: a left shift writes the zeros after the number in place, a right shift shortens the vector, whatever its length (at 64 KiB, 41 to 57 cycles against GMP's 4100).
- Against num-bigint, with bits: `>>` is 16% to 26% faster from 4 KiB and up to 6% slower at 1 KiB; `<<` is as fast at 64 KiB and up to 14% slower at 1 to 4 KiB. By whole bytes, `<<` is 1.6 to 8 times faster, and `>>` faster up to 256 bytes and in big-endian; little-endian from 1 KiB, num-bigint's `>>` by 64 bits takes up to 20% less than the `memmove` here.
- The kernels on x86_64 are the portable ones, tested for correctness under emulation only; their speed has not been measured.

## Examples

Rust usage: `-7 >> 1` is `-4`, as for `i32`, and `5 << 100` is `5 · 2^100`:

```rust
use a::math::numbers::{ByteOrder, Number};

fn main() {
  let a = Number::new_from_i32(-7, ByteOrder::BigEndian);
  let quotient = &a >> 1;
  assert_eq!(quotient.value, vec![4]);
  assert!(quotient.is_negative());

  let b = Number::new_from_u8(5, ByteOrder::LittleEndian);
  let product = b << 100;
  // 5 · 2^100 = 0x50 followed by 12 zero bytes.
  assert_eq!(product.value.len(), 13);
  assert_eq!(product.value[12], 0x50);
  assert!(product.value[..12].iter().all(|&byte| byte == 0));
}
```

`<<=` grows a number in its own vector when the vector has room, and `>>=` always shrinks it in place:

```rust
use a::math::numbers::{ByteOrder, Number};

fn main() {
  let mut value = Vec::with_capacity(4);
  value.extend_from_slice(&[0x81, 0x02]);
  let mut n = Number::new(value, ByteOrder::BigEndian);
  let vector = n.value.as_ptr();

  n <<= 9;
  assert_eq!(n.value, vec![0x01, 0x02, 0x04, 0x00]);
  assert_eq!(n.value.as_ptr(), vector);

  n >>= 17;
  assert_eq!(n.value, vec![0x81]);
  assert_eq!(n.value.as_ptr(), vector);
}
```

Rounding towards negative infinity, which can carry into a new byte, and shifts by more than the length:

```rust
use a::math::numbers::{ByteOrder, Number};

fn main() {
  let a = Number::new_from_i32(-0x01FF, ByteOrder::LittleEndian);
  let quotient = &a >> 1;
  assert_eq!(quotient.value, vec![0x00, 0x01]);
  assert!(quotient.is_negative());

  let minus_one = &a >> 1000;
  assert_eq!(minus_one.value, vec![1]);
  assert!(minus_one.is_negative());

  let b = Number::new_from_i32(0x01FF, ByteOrder::LittleEndian);
  let zero = b >> usize::MAX;
  assert_eq!(zero.value, vec![0]);
  assert!(!zero.is_negative());
}
```

## Efficiency

- Time complexity: O(n + k/8) for `<<` and O(n) for `>>` on a number of n bytes shifted by k bits, one pass over the bytes; big-endian `>>` by whole bytes in place is O(1) besides finding the significant length, and big-endian `<<` by whole bytes in place O(k/8)
- Space complexity: the vector of the result, unless it is built in the operand's; nothing else
- Soundness: the bytes beyond a vector's length, which may be uninitialised, are never read, and every byte of a result is written before the vector's length is set; a borrowed operand is left as it was
