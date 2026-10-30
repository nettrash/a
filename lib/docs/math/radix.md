[A](../../../README.md) › [Math](../../../README.md#math)

# Text

[Numbers](numbers.md) are written as text with `to_str_radix(radix)` and the formatting traits `Display` (decimal), `LowerHex`, `UpperHex`, `Octal` and `Binary`, and read from text with `Number::from_str_radix(text, radix, byte_order)`, in any radix from 2 to 36.

- The text is the sign and the digits of the magnitude, the most significant first: `-` for a negative number, nothing otherwise, and no zeros in front; `0` for zero, of whatever form. Digits above 9 are the letters, lower-case when written, either case when read. A negative number in hexadecimal is `-ff`, its sign and its magnitude, not the two's complement that the formatting of a primitive integer gives.
- The formatting traits take the formatter's flags, width, fill and alignment as the primitive integers do: `{:+}`, `{:>20}`, `{:08}`, and `0x`, `0o` or `0b` in front with `{:#x}`, `{:#o}` and `{:#b}`.
- `from_str_radix` reads what `i128::from_str_radix` reads: an optional `+` or `-`, then at least one digit of the radix, nothing else, no spaces and no `_`; zeros in front count for nothing, and `-0` is zero. It builds the number in the byte order given, in the shortest form. Empty text is an error of kind `NumberErrorKind::Empty`, any other character, or a sign alone, of kind `NumberErrorKind::InvalidDigit`; the errors read as those of `ParseIntError`. `new_from_str` stays what it was: the bytes of a string as the magnitude, not its digits.
- A radix outside 2 to 36 panics, as for the primitive integers.

In a radix that is a power of two, each digit is a group of bits of the magnitude, and the text is written or read in one pass over the bytes, where they are, in either byte order. Written, hexadecimal is two digits a byte from a table, binary eight digits a byte spread over the bytes of a `u64` by one multiplication, and octal, base 4 and base 32 come from groups of 3, 1 and 5 bytes. Read, the digits are taken from the last, each byte written straight into its place in the byte order asked for, and the digits' values or-ed together, so that the radix is checked once. In another radix, the magnitude goes through the largest power `b` of the radix that fits in a limb, `10^19` for decimal, whose remainders are blocks of digits:

- Decimal blocks without division: the 19 digits of a block are cut into 3, 8 and 8, and the eight are made at once in the lanes of a `u64` by multiplications that divide every lane by 10^4, 100 and 10. Read, eight characters are checked to be digits and turned into their value at once in the same way. Up to 38 decimal digits are read into a `u128`, and numbers of up to two limbs are written from registers.
- Writing up to 80 limbs: the magnitude is divided by `b` again and again, limb by limb, with the reciprocal of `b` and no hardware division. Each step of a division waits for the remainder of the step above, about ten cycles, but the next division needs only the quotient's limbs, from the top: so up to 8 divisions run at once, each a limb behind the one before, which hands it its limbs in registers, and the core runs their chains side by side. Small numbers take smaller groups, which waste fewer divisions above the top.
- Writing more: divide and conquer, with a table of the powers `b^(2^i)`, each prepared once as a [`Divisor`](div.md#divisor): the magnitude is divided by the largest one that halves it, and the quotient and the remainder are written the same way, the remainder with as many digits as the power has places, zeros in front. So a long number takes O(M(n) log n) for a product M(n) of n limbs, instead of O(n²).
- Reading up to 220 blocks: the blocks multiplied in, `x b + v`, a pass over the limbs in place each; a pass waits for its carries, two additions a limb, so six passes run at once in waves as the divisions do, about 0.83 cycles a limb and block. Beyond, divide and conquer: the upper half of the digits times the power of `b` of the lower half's places, plus the lower half.
- The powers of divide and conquer drop their zero bytes at the bottom, a third of the bytes of the powers of `10^19 = 5^19 2^19`: the bytes of a number below them are set aside, and only those above are divided, or multiplied.

Measured on an Apple M4 Max on random magnitudes of the length given in bytes, writing with `to_str_radix` and reading with `from_str_radix`, against GMP 6.3's `mpz_get_str` into a buffer given to it and `mpz_set_str` into a number with room, and num-bigint 0.4's `to_str_radix` and `parse_bytes` (the median of 9 interleaved runs over pools of numbers, in cycles per call):

| Bytes | Radix | Write | GMP | num-bigint | Read | GMP | num-bigint |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 10 | 87 | 136 | 147 | 102 | 107 | 245 |
| 8 | 16 | 77 | 56 | 103 | 86 | 97 | 273 |
| 32 | 10 | 242 | 406 | 514 | 208 | 313 | 629 |
| 32 | 16 | 91 | 156 | 175 | 106 | 302 | 539 |
| 128 | 10 | 983 | 2453 | 4417 | 542 | 1281 | 1936 |
| 128 | 16 | 167 | 646 | 440 | 327 | 1196 | 2742 |
| 512 | 10 | 7266 | 12,749 | 43,641 | 2890 | 6481 | 10,135 |
| 512 | 16 | 459 | 2298 | 1567 | 1119 | 4617 | 10,942 |
| 2048 | 10 | 62,658 | 84,267 | 290,524 | 31,239 | 43,548 | 87,136 |
| 2048 | 16 | 1936 | 10,079 | 6120 | 4304 | 18,242 | 42,228 |
| 8192 | 10 | 504,112 | 607,962 | 2,328,377 | 270,570 | 305,921 | 1,165,632 |
| 8192 | 16 | 6477 | 38,301 | 23,267 | 16,858 | 72,317 | 171,393 |
| 32,768 | 10 | 4,111,333 | 4,794,894 | 17,743,217 | 1,534,735 | 2,238,511 | 17,465,915 |
| 32,768 | 16 | 25,213 | 137,452 | 89,067 | 67,035 | 280,094 | 687,631 |
| 65,536 | 10 | 10,888,381 | 12,792,689 | 50,260,714 | 3,688,716 | 5,836,601 | 69,649,316 |
| 65,536 | 16 | 52,778 | 274,488 | 179,159 | 134,256 | 551,025 | 1,390,697 |

- Decimal is written in 40% to 64% of GMP's time up to 512 bytes and in 74% to 86% from 2 to 64 KiB, and at 1 MiB in 99% (1.56 times before the multiplication had number-theoretic transforms). It is read in 42% to 95% of GMP's time up to 8 KiB, 69% at 32 KiB, 63% at 64 KiB and 52% at 1 MiB (1.57 times before). For the longest numbers, the time goes to the divisions and products of divide and conquer, whose products the transforms compute from a few KiB; writing, which divides, stays at about GMP's time, as the [division](div.md) does. The rows of 32 and 64 KiB in decimal were measured again with the transforms.
- Hexadecimal is written 3.9 to 5.9 times faster than GMP writes it from 128 bytes, and read 3.6 to 4.3 times faster from 64 bytes; binary is written 11 to 14 times faster from 128 bytes, and octal 2.8 to 3.5 times. In radix 3 (and 36), writing takes 67% to 80% (79% to 101%) of GMP's time from 128 bytes to 4 KiB, and reading 64% to 65% (75% to 95%). For 8 bytes in hexadecimal, octal and radices 3 and 36, writing takes 1.25 to 2 times GMP's time: the cost is the string's allocation, where GMP writes into the buffer it is given.
- Against num-bigint, decimal text is written 1.7 to 6 times faster and read 2.4 to 9 times faster; hexadecimal is written 1.3 to 3.6 times and read 3.2 to 10 times faster.

## Examples

Rust usage: decimal with `Display`, and the radices of the other traits:

```rust
use a::math::numbers::{ByteOrder, Number};

fn main() {
  let n = Number::new_from_i64(-1_234_567_890_123, ByteOrder::BigEndian);
  assert_eq!(n.to_string(), "-1234567890123");
  assert_eq!(format!("{n:>16}"), "  -1234567890123");
  assert_eq!(format!("{n:x}"), "-11f71fb04cb");
  assert_eq!(format!("{n:#X}"), "-0x11F71FB04CB");
  assert_eq!(n.to_str_radix(36), "-fr5hugnf");
}
```

Reading, in either byte order, and the errors:

```rust
use a::math::numbers::{ByteOrder, Number, NumberErrorKind};

fn main() {
  let n = Number::from_str_radix("-FF", 16, ByteOrder::BigEndian).unwrap();
  assert_eq!(n.value, vec![0xFF]);
  assert!(n.is_negative());

  let n = Number::from_str_radix("+00012345", 10, ByteOrder::LittleEndian).unwrap();
  assert_eq!(n.value, vec![0x39, 0x30]);

  let error = Number::from_str_radix("12_345", 10, ByteOrder::BigEndian).err().unwrap();
  assert_eq!(*error.kind(), NumberErrorKind::InvalidDigit);
  assert_eq!(error.to_string(), "invalid digit found in string");
}
```

A long number round trip: 1000 nines are `10^1000 - 1`:

```rust
use a::math::numbers::{ByteOrder, Number};

fn main() {
  let nines = "9".repeat(1000);
  let n = Number::from_str_radix(&nines, 10, ByteOrder::BigEndian).unwrap();
  assert_eq!(n.value.len(), 416);
  assert_eq!(n.to_string(), nines);
}
```

## Efficiency

- Time complexity: O(n) for a radix that is a power of two; O(n²) in another radix up to 80 limbs to write and 220 blocks of digits to read, and O(M(n) log n) beyond, for a product M(n) of n limbs, O(n^1.58) with Karatsuba's method
- Space complexity: the text, or the number read; on the stack, the limbs and blocks of up to 80 limbs written, 64 blocks of text read, and up to 160 digits formatted; for divide and conquer, the powers of the block and their divisors, and the parts of the number on the way
- Soundness: the bytes beyond a vector's length are never read; the text is always ASCII
