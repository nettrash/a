[A](../../../README.md) › [Math](../../../README.md#math)

# Traits and conversions

[Numbers](numbers.md) compare, hash, copy, print and convert as Rust's primitive integers do, and take the primitive integers as operands of the arithmetic.

- Comparisons are by value: `==`, `!=`, `<`, `<=`, `>`, `>=`, and `cmp`, `max`, `min` and `clamp` of `Ord`. Two numbers are equal when they are the same integer, whatever their byte orders and however their magnitudes are stored: zero bytes above the most significant non-zero byte count for nothing, an empty vector is zero, and so is a zero magnitude with `negative` set. So a `Vec<Number>` sorts by value, and a number can be a key of a `BTreeMap`.
- A number compares with every primitive integer type on its right: `a == 5`, `a < -1i64`, `a >= u128::MAX`. Not on its left: `impl PartialEq<Number> for i32` would leave `i32` two types to compare with, and code elsewhere that leaves the other type to inference, such as `v == vec![]` for a `Vec<i32>`, would stop compiling.
- `Hash` agrees with `==`: equal numbers hash alike in either byte order, so a number can be a key of a `HashMap`.
- `Clone` copies the number exactly as it is stored; `clone_from` reuses the target's vector. `Default` is zero, little-endian, with an empty vector, which allocates nothing: `mem::take` of a number leaves it behind. `Debug` writes the number in decimal, as `Display` does.
- `-a` (`Neg`) turns the sign over, in the vector of an owned `a`; `abs` and `signum` (a number: -1, 0 or 1) are as for `i32`, and `is_zero` and `is_positive` complete `is_negative`.
- `From` every primitive integer type builds a number little-endian, the byte order of the arithmetic's limbs: `Number::from(-5i32)`. `TryFrom<&Number>` and `TryFrom<Number>` give each primitive integer type back, or a `TryFromNumberError` when the number is out of its range, as the conversions between the primitive types do.
- `to_f64` and `to_f32` give the nearest float, ties to even, as `as` does for the primitive integers, and infinity with the number's sign when it is too large. `to_f32` rounds the number itself, not its `to_f64`, which would round twice.
- `"…".parse::<Number>()` (`FromStr`) reads decimal text, little-endian, as `from_str_radix` with radix 10 does: see [Text](radix.md).
- `Sum` and `Product` take owned or borrowed numbers: `(1..=100u32).map(Number::from).product::<Number>()`. The result is in the byte order of the first number, one or zero little-endian for none.
- `+`, `-`, `*`, `/` and `%` and their assignments take a primitive integer of any type on either side: `a + 1`, `&a * 10u64`, `a %= 7`, `1i32 - &a`, `10u8 / a`. They give what the same operator gives with the integer made a number, results, byte orders and panics alike. With the number on the left, the result is in its byte order and built in its vector when it is owned and has room; with the integer on the left, in the number's byte order. An integer literal on the left needs its type, `1u8 - &a`, as `1 - &a` leaves the compiler every integer type to pick from; on the right it needs one too where the result goes straight into an operator or a method, as in `-(&a - 1u8)` or `(&a + 1u8).to_string()`, and elsewhere is an `i32`.
- Floor division, `div_floor` and `rem_floor`, is in [Division](div.md).

This implementation:

- Comparisons: two vectors of up to 16 bytes are read as `u128`s and compared; longer magnitudes by their signs, then their significant lengths, then their bytes from the top: in the same byte order as byte strings, which the standard library compares with vector instructions (little-endian after a look at the top four limbs, where random magnitudes differ), in different ones limb by limb. Equal magnitudes of 1 KiB took 107 cycles little-endian, against 199 limb by limb.
- Hashing: the sign, the significant length, then the significant bytes from the least significant one, in chunks of 256 bytes, which a big-endian number turns around in a buffer, so that both byte orders make the same calls to the hasher. With the standard library's SipHash a number of 1 KiB took 835 cycles, against 1,976 a limb at a time.
- Floats: a magnitude of up to 16 bytes converts as a `u128`; of a longer one, the top 64 bits convert and are scaled by a power of two, which is exact. The bits below the top 64 can only matter when those are exactly halfway between two floats, and are read only then, so a conversion reads at most 9 bytes of a random number, whatever its length.
- Integer operands: `+` and `-` change the number's bytes in place: up to 16 bytes in registers, otherwise the integer's 16 bytes added to the number's lowest 16 and the carry taken up only through bytes that are all ones, so that `a += 1` takes the same time for every length. `*` by an integer of up to 64 bits multiplies a number of up to 16 bytes in registers, and a longer little-endian one in place, in one pass, by the multiplication's kernel for one limb, as the multiplication of two numbers now does when one has a single limb (see [Multiplication](mul.md)); a longer big-endian one as two numbers multiply. `/` and `%` make the integer a number in a vector of 16 bytes that each thread keeps for the purpose, so they allocate nothing for it, and divide as two numbers do; a remainder by one limb then takes eight limbs a step (see [Division](div.md)), after computing the powers of `2^64` modulo the integer, about 70 cycles, which a `Divisor` made once from the integer keeps for every remainder.
- Products of iterators: the partial products wait on a stack, each multiplied with the one below it while that one is not longer, so that the factors of each product are about as long as each other, and long ones go to Karatsuba's and Toom-Cook's methods. The product of the numbers from 1 to 10,000 took 1.9 million cycles so, against 10.5 million multiplied in turn, and from 1 to 1,000 134,000 against 253,000.

Measured on an Apple M4 Max, over pools of random operands of the lengths given, with `a` little-endian, against GMP 6.3 and num-bigint 0.4 (the median of 9 interleaved runs, in cycles per call): `cmp` of two equal numbers, which reads them whole, against `mpz_cmp`; the standard library's SipHash of the number (GMP has no hash); `to_f64` against `mpz_get_d`, which truncates instead of rounding; and `a += 1`, `a *= 10` and `a %= 10` with integers, against `mpz_add_ui`, `mpz_mul_ui` and `mpz_tdiv_ui`, which gives the remainder as a limb, not a number:

| Bytes | `cmp` | GMP | num-bigint | `Hash` | num-bigint | `to_f64` | GMP | num-bigint | `a += 1` | GMP | num-bigint | `a *= 10` | GMP | num-bigint | `a %= 10` | GMP | num-bigint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 9 | 5 | 5 | 29 | 12 | 6 | 3 | 4 | 6 | 3 | 5 | 7 | 4 | 3 | 11 | 6 | 16 |
| 16 | 10 | 8 | 8 | 53 | 25 | 13 | 7 | 36 | 13 | 8 | 11 | 22 | 9 | 7 | 37 | 28 | 105 |
| 64 | 26 | 15 | 16 | 85 | 55 | 8 | 7 | 64 | 9 | 10 | 12 | 28 | 14 | 21 | 103 | 47 | 207 |
| 256 | 45 | 47 | 43 | 221 | 199 | 7 | 9 | 226 | 11 | 9 | 14 | 53 | 33 | 60 | 179 | 108 | 936 |
| 1024 | 145 | 274 | 183 | 881 | 810 | 7 | 9 | 842 | 12 | 10 | 13 | 158 | 147 | 310 | 297 | 254 | 3882 |
| 16384 | 1399 | 3375 | 2611 | 13,198 | 11,703 | 7 | 8 | 13,115 | 16 | 20 | 20 | 2253 | 2216 | 4497 | 2080 | 2929 | 56,810 |

- Comparisons take 1.7–1.8 times GMP's time up to 64 bytes, where the fixed costs of finding the significant bytes of both numbers dominate, and half of it or less from 1 KiB, where they are a `memcmp`.
- Hashing takes 1.1–2.4 times num-bigint's: SipHash takes three calls for the sign, the length and the bytes where num-bigint makes two, and its cost per byte is the same.
- `to_f64` takes about 7 cycles whatever the length, as GMP's does, where num-bigint's grows with it; ours rounds to the nearest float.
- `a += 1` takes 1.3–2 times GMP's time up to 16 bytes and about the same from 64; `a *= 10` 1.6–2 times up to 256 bytes and about the same from 1 KiB, twice as fast as num-bigint's there; `a %= 10`, which writes a number, 1.3–2.2 times up to 256 bytes, the same at 1 KiB and 0.7 times at 16 KiB, and 1.5–27 times as fast as num-bigint's.

## Examples

Comparing, sorting and hashing by value:

```rust
use a::math::numbers::{ByteOrder, Number};
use std::collections::HashSet;

fn main() {
  let big = Number::new_from_u32(1_000_000, ByteOrder::BigEndian);
  let little = Number::new_from_u32(1_000_000, ByteOrder::LittleEndian);
  assert_eq!(big, little);
  assert!(big > 999_999 && big < 1_000_001u64);

  let mut v = vec![Number::from(300), Number::from(-5), Number::from(7)];
  v.sort();
  assert_eq!(v, [-5, 7, 300]);

  let set: HashSet<Number> = [big].into_iter().collect();
  assert!(set.contains(&little));
}
```

Conversions:

```rust
use a::math::numbers::Number;

fn main() {
  let n = Number::from(-1_000_000i64);
  assert_eq!(i64::try_from(&n), Ok(-1_000_000));
  assert!(u64::try_from(&n).is_err());
  assert_eq!(n.to_f64(), -1e6);

  let n: Number = "123456789012345678901234567890".parse().unwrap();
  assert_eq!(n.to_f64(), 1.2345678901234568e29);
  assert_eq!((-n).signum(), -1);
}
```

Integer operands, sums and products:

```rust
use a::math::numbers::Number;

fn main() {
  let mut n = Number::from(u64::MAX);
  n += 1;
  n *= 1000u32;
  assert_eq!(n.to_string(), "18446744073709551616000");
  assert_eq!(&n % 7, 18446744073709551616000u128 % 7);
  assert_eq!(1u8 - &n, -(&n - 1u8));

  let sum: Number = (1..=100).map(Number::from).sum();
  assert_eq!(sum, 5050);
  let factorial: Number = (1..=20u64).map(Number::from).product();
  assert_eq!(factorial, 2432902008176640000u64);
}
```

## Efficiency

- Comparisons and hashing: O(n) time for n bytes at most, no allocation; random numbers of equal length differ in their top bytes, and compare in constant time.
- `to_f64` and `to_f32`: constant time for random numbers, O(n) at most; no allocation.
- `+`, `-` and `*` with an integer: O(n) time at most, in the vector of an owned number when it has room, otherwise one new vector; `+` and `-` in constant time but where carries run through whole runs of bytes.
- `Sum`: n − 1 additions in place. `Product`: n − 1 multiplications as a balanced tree.
