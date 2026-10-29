[A](../../../README.md) › [Math](../../../README.md#math)

# Numbers

`Number` (in `a::math::numbers`) is an integer of any size, stored as its sign and the bytes of its absolute value: a `Vec<u8>` holding the magnitude, a `negative` flag and the `ByteOrder` the bytes are written in, `BigEndian` (the most significant byte first) or `LittleEndian` (the least significant byte first). This is the sign-and-magnitude form most big-integer libraries use: `-2` is `[2]` with `negative` set, not the two's-complement byte `0xFE`.

- A number is created from the bytes of its absolute value (`new` for a non-negative number, `new_signed` with a sign, or `new_from_i8_vec` for bytes given as `i8`s), from a primitive integer (`new_from_u8` … `new_from_u128`, `new_from_usize`, `new_from_i8` … `new_from_i128`, `new_from_isize`), or from the UTF-8 bytes of a string (`new_from_str`).
- A signed integer gives its sign and its absolute value, exact for `MIN` too: `i32::MIN` is `[0x80, 0, 0, 0]` with `negative` set. A value gives the same number whatever its type: `-5i8` and `-5i128` are both `[5]`, negative, and `5i64` is `5u8`.
- `new_from_i8_vec` and `new_from_str` are never negative. The `i8`s are only how the bytes are given (`-1` is the byte `0xFF`), and the string is not parsed: `"123"` is the bytes `0x31 0x32 0x33`, the way textbook RSA turns a message into an integer, and `"-1"` is `0x2D 0x31`.
- Every constructor keeps the shortest form of the number: it drops the zero bytes above the most significant non-zero byte (the leading zeros in big-endian order, the trailing ones in little-endian order) and keeps one byte for zero, `[0]`. An empty vector or string stays empty, which is zero as well. Zero is never negative: there is one zero, not a positive and a negative one.
- `is_negative()` and `byte_order()` read the sign and the order. The fields are public; a `Number` built from them directly is kept exactly as given, zero bytes and the sign of zero included.
- Numbers add with `+` and `+=` and subtract with `-` and `-=`, whatever their lengths, signs and byte orders, and whether a constructor built them or not: see [Addition](add.md) and [Subtraction](sub.md).

## Examples

Rust usage with integers:

```rust
use a::math::numbers::{ByteOrder, Number};

fn main() {
  let big = Number::new_from_i32(-0x0001_0203, ByteOrder::BigEndian);
  assert_eq!(big.value, vec![0x01, 0x02, 0x03]);
  assert!(big.is_negative());

  let little = Number::new_from_u32(0x0001_0203, ByteOrder::LittleEndian);
  assert_eq!(little.value, vec![0x03, 0x02, 0x01]);
  assert!(!little.is_negative());
}
```

From bytes: only the zero bytes above the most significant non-zero byte are dropped, and zero has no sign:

```rust
use a::math::numbers::{ByteOrder, Number};

fn main() {
  let n = Number::new(vec![0, 0, 1, 0, 2], ByteOrder::BigEndian);
  assert_eq!(n.value, vec![1, 0, 2]);

  let n = Number::new_signed(vec![2, 0, 1, 0, 0], true, ByteOrder::LittleEndian);
  assert_eq!(n.value, vec![2, 0, 1]);
  assert!(n.is_negative());

  let zero = Number::new_signed(vec![0, 0, 0], true, ByteOrder::BigEndian);
  assert_eq!(zero.value, vec![0]);
  assert!(!zero.is_negative());
}
```

The smallest signed integers and strings:

```rust
use a::math::numbers::{ByteOrder, Number};

fn main() {
  let n = Number::new_from_i16(i16::MIN, ByteOrder::BigEndian);
  assert_eq!(n.value, vec![0x80, 0x00]);
  assert!(n.is_negative());

  let n = Number::new_from_str("-123", ByteOrder::BigEndian);
  assert_eq!(n.value, vec![0x2D, 0x31, 0x32, 0x33]);
  assert!(!n.is_negative());
}
```

## Efficiency

- Space: the significant bytes of the absolute value in a `Vec<u8>`, plus 32 bytes for the `Number` itself on a 64-bit target (the vector's pointer, capacity and length, the sign and the byte order). The vector may keep spare capacity: trimming does not shrink it.
- `new` and `new_signed`: O(n) time for n bytes, one scan for the most significant non-zero byte and, in big-endian order, one move of the bytes after it to the front. They allocate nothing and keep the vector they are given; whether the trimmed bytes are zero, which decides the sign, takes constant time.
- The other constructors allocate once (the width of the integer type, the bytes of the string or a copy of the `i8`s) and then trim as `new` does.
