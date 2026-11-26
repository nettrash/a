[A](../../../README.md) › [Math](../../../README.md#math)

# Fixed-width integers

`Uint<N>` and `Int<N>` (in `a::math::uint`) are integers of `N` 64-bit limbs, `64·N` bits, on the stack and `Copy`: `Uint<N>` unsigned, from 0 to `2^(64N) − 1`, and `Int<N>` signed in two's complement, from `−2^(64N−1)` to `2^(64N−1) − 1`, as `i64` is to `u64`. They behave as a `u{64·N}` and an `i{64·N}` would, method for method where the method makes sense, and convert to and from [numbers](numbers.md) of any size. The limbs are kept the low one first: `Uint::<2>::from_limbs([1, 2])` is `1 + 2·2⁶⁴`. Nothing here is constant-time: the time of an operation depends on the values.

- [The integers](#the-integers): the operators and the methods of the primitive integers, with their `checked_*`, `wrapping_*`, `overflowing_*` and `saturating_*` forms and their behaviour on overflow; products, quotients by Knuth's algorithm D, powers, roots and logarithms; bits and bytes; text in any radix with the formatter's flags; and conversions to and from the primitive integers, floats and [numbers](numbers.md) of any size.
- [Montgomery multiplication and `pow_mod`](#montgomery-multiplication-and-pow_mod): `Montgomery<N>`, the arithmetic modulo an odd number in Montgomery form, and `pow_mod`, `mul_mod`, `add_mod`, `sub_mod` and `mod_inverse` on `Uint<N>` for any nonzero modulus, odd or even.

## Examples

```rust
use a::math::numbers::Number;
use a::math::uint::{Int, Uint};
use core::num::IntErrorKind;

type U256 = Uint<4>;
type I256 = Int<4>;

// The prime of secp256k1, read in a constant.
const P: U256 = match U256::from_str_radix(
    "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F",
    16,
) {
    Ok(p) => p,
    Err(_) => panic!("not hexadecimal"),
};
assert_eq!(P.bits(), 256);
assert_eq!(P.as_limbs()[0], 0xFFFF_FFFE_FFFF_FC2F);

// Arithmetic as u256 would have it, checked with Number.
let x = U256::from_u64(3).pow(150);
let (q, r) = x.div_rem(P >> 40u32);
assert_eq!(q * (P >> 40u32) + r, x);
assert_eq!(Number::from(q), Number::from(x) / Number::from(P >> 40u32));
assert_eq!(U256::MAX.checked_add(U256::ONE), None);
assert_eq!(U256::MAX.wrapping_add(U256::ONE), U256::ZERO);
assert_eq!(U256::MAX.overflowing_mul(U256::from_u64(2)), (U256::MAX - U256::ONE, true));
let (low, high) = U256::MAX.widening_mul(U256::MAX);
assert_eq!((low, high), (U256::ONE, U256::MAX - U256::ONE));
assert_eq!(x.isqrt(), U256::from_u64(3).pow(75));
assert_eq!(x.ilog10(), 71);

// Signed: truncated division, Euclid's, the edges of the range.
let m = I256::from_i64(-7);
assert_eq!(m / I256::from_i64(2), I256::from_i64(-3));
assert_eq!(m % I256::from_i64(2), I256::from_i64(-1));
assert_eq!(m.rem_euclid(I256::from_i64(2)), I256::ONE);
assert_eq!(I256::MIN.checked_div(I256::MINUS_ONE), None);
assert_eq!(I256::MIN.unsigned_abs(), U256::ONE << 255u32);
assert_eq!(I256::MIN >> 255u32, I256::MINUS_ONE);

// Text, with the formatter's flags, and the primitives' errors.
assert_eq!(format!("{:#066x}", U256::from_u64(0xbeef)), format!("0x{}beef", "0".repeat(60)));
assert_eq!(format!("{:+}", I256::from_i64(5)), "+5");
assert_eq!(format!("{:x}", I256::MINUS_ONE), "f".repeat(64));
assert_eq!(I256::MIN.to_string().parse::<I256>(), Ok(I256::MIN));
let e = "-1".parse::<U256>().unwrap_err();
assert_eq!(*e.kind(), IntErrorKind::InvalidDigit);

// Conversions.
assert_eq!(u128::try_from(U256::from_u128(u128::MAX)), Ok(u128::MAX));
assert!(u128::try_from(U256::ONE << 128u32).is_err());
assert_eq!(Uint::<2>::try_from(-1i32).is_err(), true);
assert_eq!(U256::try_from(&Number::from(12345u64)), Ok(U256::from_u64(12345)));
assert_eq!(x.resize::<8>().try_resize::<4>(), Some(x));
assert_eq!(U256::from_f64(1e20), U256::from_u128(100_000_000_000_000_000_000));
assert_eq!((U256::ONE << 200u32).to_f64(), 2f64.powi(200));
let bytes: [u8; 32] = P.to_be_bytes();
assert_eq!(U256::from_be_bytes(bytes), P);
```

The sections below give each part's functions, its methods and sources, an example, what it was verified against, and its limits. There are no timings here yet: they come with the release.

## The integers

| Item | What it is |
| --- | --- |
| `BITS`, `ZERO`, `ONE`, `MIN`, `MAX`; `Int::MINUS_ONE` | the width in bits and the constants, as the primitives have them |
| `from_limbs`, `as_limbs`, `as_limbs_mut`, `into_limbs` | the limbs, the low one first (two's complement for `Int`) |
| `from_u64`, `from_u128` / `from_i64`, `from_i128` | `const` constructors; they panic when the value does not fit (`N ≤ 1` for the 128-bit ones, `N = 0` for anything but 0) |
| `+ - * / %`, `& \| ^ !`, `<< >>`, unary `-` (`Int`) | the operators, owned or borrowed on either side, with `+=` and the other assignments; shifts by every primitive integer type |
| `Sum`, `Product` | of owned or borrowed items, overflowing as `+` and `*` do |
| `checked_*`, `wrapping_*`, `overflowing_*` | of `add`, `sub`, `mul`, `div`, `rem`, `div_euclid`, `rem_euclid`, `neg`, `shl`, `shr`, `pow`, and `abs` for `Int` |
| `saturating_*` | of `add`, `sub`, `mul`, `div`, `pow`, and `neg`, `abs` for `Int` |
| `*_add_signed` (`Uint`), `*_add_unsigned`, `*_sub_unsigned` (`Int`) | the four forms of the mixed sums, as `u64::checked_add_signed` and `i64::checked_add_unsigned` |
| `unbounded_shl`, `unbounded_shr` | the shifts that give 0 (or −1 for a negative `Int` shifted right) from `BITS` on |
| `div_euclid`, `rem_euclid`, `div_ceil` (`Uint`), `next_multiple_of`, `checked_next_multiple_of`, `is_multiple_of` (`Uint`) | the other quotients and multiples of the primitives |
| `abs`, `unsigned_abs`, `signum`, `is_positive`, `is_negative`, `abs_diff`, `midpoint` | as `i64`'s (`abs_diff` and `midpoint` for `Uint` too) |
| `pow`, `isqrt`, `checked_isqrt` (`Int`), `ilog`, `ilog2`, `ilog10`, `checked_ilog*` | powers, square roots and logarithms rounded down |
| `is_power_of_two`, `next_power_of_two`, `checked_next_power_of_two` | for `Uint` |
| `count_ones`, `count_zeros`, `leading_zeros`, `trailing_zeros`, `leading_ones`, `trailing_ones`, `bits`, `bit`, `is_zero` | the bits (of the two's complement for `Int`) |
| `rotate_left`, `rotate_right`, `swap_bytes`, `reverse_bits` | as the primitives' |
| `to_le_bytes`, `to_be_bytes`, `from_le_bytes`, `from_be_bytes` | the `8·N` bytes as an array, its length `B` a const parameter checked at compile time |
| `write_le_bytes`, `write_be_bytes`, `from_le_slice`, `from_be_slice` | the same through slices, for code generic over `N`; the readers take any length and return `None` when the value does not fit |
| `carrying_add`, `borrowing_sub`, `widening_mul`, `carrying_mul`, `widening_mul_u64`, `div_rem`, `div_rem_u64` | the limb-level helpers (`div_rem` for `Int` too) |
| `cast_signed`, `cast_unsigned`, `TryFrom` between `Uint<N>` and `Int<N>` | the same bits, or the same value when it fits |
| `resize::<M>()`, `try_resize::<M>()` | to another width: truncated or extended as `as` does, or `None` when the value does not fit |
| `to_f64`, `to_f32`, `from_f64`, `from_f32` | to the nearest float, ties to even; from a float as `as` converts it |
| `Display`, `LowerHex`, `UpperHex`, `Octal`, `Binary`, `Debug` | the text, with the formatter's flags (`Debug` as `{:#x}` takes them) |
| `to_str_radix`, `from_str_radix`, `FromStr` | text in any radix from 2 to 36; `from_str_radix` is `const` |
| `ParseIntError`, `TryFromIntError` | the errors, with the messages of current Rust (1.100) and the standard [`IntErrorKind`](https://doc.rust-lang.org/core/num/enum.IntErrorKind.html) from `kind()`: what was wrong with a text, and which way a value was out of range |

### Semantics

- **Overflow.** As the primitive integers under the default profiles: `+`, `-`, `*`, unary `-`, `pow`, `abs`, `next_power_of_two`, `next_multiple_of` and the shifts panic on overflow when `debug_assertions` are on, and wrap when they are off. What decides is `cfg!(debug_assertions)` of this crate as it is built, where the primitives follow `overflow-checks`; the two are the same unless a profile sets them apart. The messages are those of the primitives in current Rust (1.100): `attempt to add with overflow` and so on, and `attempt to exponentiate with overflow` for `pow`. Some of them changed between releases of Rust: 1.88 writes `attempt to multiply with overflow` for `pow`, and these types keep to the current wording whatever the toolchain. The `checked_*`, `wrapping_*`, `overflowing_*` and `saturating_*` forms never depend on the profile.
- **Division** rounds toward zero, and the remainder has the sign of the dividend, as for `i64`; `div_euclid` and `rem_euclid` give Euclid's division, whose remainder is never negative. Division and remainder by zero always panic (`attempt to divide by zero`, `attempt to calculate the remainder with a divisor of zero`), and so do `Int::MIN / −1` and `Int::MIN % −1`, whatever the profile, as for `i64`; `wrapping_div` gives `MIN` and `wrapping_rem` 0 there.
- **Shifts** take an amount of any primitive integer type, owned or borrowed. An amount outside `0..BITS`, negative ones included, panics with debug assertions; without them the shift is by the amount modulo `BITS`, Euclid's remainder, so that `−1` is `BITS − 1`. For a width that is a power of two (`N` = 1, 2, 4, 8, …) this is the primitives' masking of the low bits of the amount; for other widths it is the natural extension (bnum masks the amount with `BITS − 1` there instead, and gives other results from `BITS` on). The named methods take a `u32` and do the same: `overflowing_shl(k)` is the shift by `k mod BITS` and whether `k ≥ BITS`. `>>` on an `Int` is arithmetic.
- `0⁰` is 1. `ilog`, `ilog2` and `ilog10` panic for 0 and, for `Int`, for negative numbers (and `ilog` for a base below 2); `isqrt` of a negative `Int` panics. The `checked_` forms give `None` instead.
- As shifts are implemented for every primitive type of amount, an amount written as a bare literal gets its type only from the fallback to `i32` at the end: a method called on the result needs the literal's type, `(x << 3u32).count_ones()`, while `x << 3` alone, compared or assigned, is fine.
- Most methods are `const fn`, the arithmetic, the divisions, roots and logarithms, the bytes, the float conversions and `from_str_radix` among them: `const P: Uint<4> = match Uint::from_str_radix("…", 16) { Ok(p) => p, Err(_) => panic!() };`. The operators are not, as trait methods cannot be on stable Rust.

### Methods

- **Sums and differences** run along the limbs with the carry or the borrow, `O(N)`.
- **Products** are schoolbook, as in the other fixed-width libraries at these sizes: `widening_mul` and `carrying_mul` take the `N²` limb products of the full `2N`-limb product; `wrapping_mul` and the operator only the products that land below the width, at most `N(N + 1)/2`: where the significant limbs of the two operands fit in the width, the full product of them; for operands of the full width (and up to ten limbs, whatever they are) all of them in rows of fixed length, which unroll; otherwise rows over the significant limbs. The flag of `overflowing_mul` (and so `checked_mul` and `saturating_mul`) comes from the lengths of the operands in bits: no overflow where they sum to at most `64·N`, overflow above `64·N + 1`, and at `64·N + 1` the high half of the full product decides (for one and two limbs, `u64`'s and `u128`'s own test). Without debug assertions the operator `*` is `wrapping_mul`.
- **Quotients** are Knuth's algorithm D (*The Art of Computer Programming*, vol. 2, 3rd ed., 4.3.1): the divisor's `n` significant limbs shifted until its top bit is set and the dividend as far (D1); each quotient limb is the quotient of the top three limbs of the remainder by the top two of the divisor (D3), the true limb or one more, taken without a division by Möller and Granlund's 3-by-2 division with the divisor's reciprocal (*Improved division by invariant integers*, IEEE Transactions on Computers 60(2), 2011), which also gives the top two limbs of the new remainder, so only the divisor's other `n − 2` limbs times it are subtracted (D4); added back once when that went below zero (D6, on random operands with a probability of about `2/2⁶⁴`). Where the remainder's top two limbs equal the divisor's, the limb is `2⁶⁴ − 1` exactly. `O(n·m)` limb products for a quotient of `m` limbs. A divisor of one limb takes the loop of 2-by-1 divisions from the top, each two products by the divisor's reciprocal, which is also `div_rem_u64`; a dividend of one limb, the hardware's division. No limb is divided by a 128-bit division, which on aarch64 is a call into software. The reciprocal of a limb is Möller and Granlund's: an 11-bit start from a table, Newton steps in 64-bit products, and an exact correction. `Int` divides the absolute values and restores the signs.
- **Powers**: `wrapping_pow` (and without debug assertions `pow`) squares and multiplies from the top bit of the exponent, the squares with each cross product once and doubled, about half the products; a base of one limb is multiplied in as that limb, and its first steps are taken in a `u128` while the power fits; `Uint<2>` is `u128::wrapping_pow`; `overflowing_pow` (and `checked_pow`, `saturating_pow`, and `pow` with debug assertions) from the low bit, with the flag, which is exact, as every partial product of a base of at least 2 divides the power, a square wrapping exactly when the base has more than half the bits. `O(N² log e)`.
- **Square roots**: Zimmermann's Karatsuba square root (*Karatsuba Square Root*, INRIA RR-3805, 1999). The number is shifted up by an even amount, to an even number `2L` of limbs with one of its top two bits set, and written `H·b² + a₁·b + a₀` with `b = 2^(64·⌊L/2⌋)`; the root `s'` and the remainder `r'` of the top half `H` come from the same method recursively, then `q = ⌊(r'·b + a₁)/(2s')⌋` by algorithm D (on half the dividend, by `s'`, whose top bit is set), `s = s'·b + q` and the remainder `(r'·b + a₁ − 2s'·q)·b + a₀ − q²`. As the top half is normalised, `s'` is at least `b/2` and `s` is the root or one more; a negative remainder brings it down by one. Two limbs are one such step on 32-bit digits, the root of the top 64 bits by `u64::isqrt` and one 64-bit division. The root is shifted back down. About the cost of one division of the number by half its length; up to 64 bits, `u64::isqrt`.
- **Logarithms**: `ilog2` from the highest set bit; `ilog10` from `t = ⌊(bits − 1)·log₁₀2⌋` (by a product with `⌊2⁶⁴·log₁₀2⌋`), the logarithm being `t` or `t + 1`, and a comparison with `10^(t + 1)` formed by products by `10¹⁹`, `O(N²)` products and no division (`u64::ilog10` up to 64 bits, `u128::ilog10` up to 128); `ilog` to another base of one limb divides by the largest power of the base that fits in a limb, counting its digits at a time, and takes the logarithm of the last limb, `O(N²)` in all; a longer base is divided by itself, at most `BITS/64` times.
- **Text**: `Display` and the radix traits of a `Uint` of one and two limbs are those of `u64` and `u128`, which write the same digits with the same flags; so are those of an `Int`, `i64` and `i128`, but for `Display` without a width and without `+` (the only flags that change a decimal integer's text), which writes the sign and the digits at once from a buffer on the stack, as `Display` does at every other width without those flags. Otherwise, in a radix that is a power of two it is read off the bits, a limb at a time for radices 2, 4 and 16; in decimal the value is divided by `10¹⁹` (whose reciprocal is a constant), one chunk of 19 digits at a time, `O(N²)`, each chunk written two digits at a time from a table; in another radix it is divided by the largest power of the radix that fits in a limb and written a digit at a time; into a buffer on the stack up to 16 limbs. Text is read back by products with that power. With flags the digits go through `Formatter::pad_integral`, which applies `+`, `#`, `0`, the width, the fill and the alignment as for the primitives: `{:#066x}` of a `Uint<4>` is `0x` and 64 digits. An `Int` is written in decimal and by `to_str_radix` as its sign and the digits of its absolute value, and in hexadecimal, octal and binary as the bits of its two's complement (`{:x}` of −1 is all `f`), as `i64` is. `Debug` writes hexadecimal, `0x` and the significant digits, with a `-` and the absolute value for a negative `Int`: limbs are easier to see that way than in decimal, which is what `Display` gives. It takes the width, fill, alignment and the flags `+` and `0` as `{:#x}` takes them, the zeros after the `0x`: `{:>8?}` of 255 is `    0xff` and `{:08?}` is `0x0000ff`.
- **Reading** follows the primitives' `from_str_radix` exactly: an optional `+` (and `-` for `Int`), then at least one digit, letters in either case, nothing else; `-` before a `Uint` and a sign alone are invalid digits. Of an overflow and a bad character, the error is the one met first from the left, as the primitives report it: `"340282366920938463463374607431768211456x"` overflows a `Uint<2>`, `"34028236692093846346337460743176821145x6"` has an invalid digit. A radix outside 2 to 36 panics.
- **To a float**: the top 64 bits, with the lowest of them or'ed with every bit below them, converted by the hardware and scaled by a power of two, which is exact: one rounding, to nearest with ties to even, as `Number::to_f64` and `u128 as f64` round, infinity beyond the largest float. `to_f32` rounds once from the value, not through `f64`. **From a float**, as `as`: truncated toward zero, held to `[MIN, MAX]`, a NaN to 0.

### Conversions

- `From` where every width of one limb or more holds every value of the source, as the standard library implements `From` only for conversions that cannot fail: into `Uint<N>` from `bool`, `u8`, `u16`, `u32`, `u64` and `usize`; into `Int<N>` from `bool`, `i8` to `i64`, `isize`, and `u8` to `u32`.
- `TryFrom`, with `TryFromIntError`, for the rest: `u128` and every signed type into `Uint<N>` (negative values, or `u128` beyond 64 bits for `N = 1`, fail), `u64`, `usize`, `u128` and `i128` into `Int<N>`, both types into every primitive integer, and `Uint<N>` and `Int<N>` into each other. A conversion fails exactly when the primitive of the same width fails. The error says which way, as current Rust's does: `kind()` is `PosOverflow` and the text "number too large to fit in target type" above the range, `NegOverflow` and "number too small to fit in target type" below it (a negative value into an unsigned type included). Rust 1.88's error wrote "out of range integral type conversion attempted" for both.
- With `Number`: `From<Uint<N>>`, `From<&Uint<N>>` and the same for `Int` give a little-endian number; `TryFrom<&Number>` and `TryFrom<Number>` (with `TryFromNumberError`) read a number by value, in either byte order, whatever zero bytes it stores above its magnitude and whether a zero is marked negative.
- Between widths, `resize::<M>()` truncates or extends (with zeros, or the sign for `Int`), as `as` does between primitives, and `try_resize::<M>()` gives `None` when the value does not fit. (`TryFrom<Uint<M>> for Uint<N>` would overlap the standard library's `TryFrom<T> for T` when `M = N`.)

### Verified

The tests are in `lib/tests/math/uint/`, all but `mont.rs`, which is the Montgomery part's; they need no data files, as `Number` and the primitive integers are the references.

- **The primitives, method by method** (`prim.rs`). `Uint<1>` against `u64`, `Int<1>` against `i64`, `Uint<2>` against `u128` and `Int<2>` against `i128`, on every pair of an edge set (0 to 3, 10, `MAX`, `MAX − 1`, `MAX/2`, `MAX/2 + 1`, `2^k` and `2^k ± 1` at and around the limb boundaries, one nonzero high limb, all-ones limbs; for the signed types the same in two's complement and negated, with `MIN`, `MAX`, −1 and `MIN + 1`: 20, 31, 32 and 54 values) together with 48 seeded random values biased to the same edges: 68 to 102 values, every pair, about 50 binary methods per pair (each operator, each form of each arithmetic method, the comparisons, the mixed sums with the other sign), and for each value 42 (unsigned) or 53 (signed) unary methods (bits, roots, logarithms, negations, text in four radices, bytes, floats, `TryFrom` into all 12 primitive types) and 25 methods at 16 shift amounts and exponents from 0 to `u32::MAX`, with the shift amount in 9 of the 12 primitive types for `Uint` and 5 for `Int`, negative ones included (all 12, owned and borrowed, with the assignments, in `ops.rs`). Each comparison runs both sides and requires the same value or both to panic: about 1.4 million comparisons, all equal. Conversions from every primitive type at its edges, and floats into the integers against `as` on 4,051 values for each width (signed zeros, halves, infinities, NaN, `MAX`, subnormals, the powers `2^k` and their neighbours for `k` up to 129, random bit patterns), agree too.
- **Both profiles.** As the primitives panic on overflow under debug assertions and wrap without, the same tests run under `cargo test` and `cargo test --release` check both behaviours; the 48 tests pass in both. The `Number` model below checks each operator against the profile it runs in.
- **Every width against `Number`** (`model.rs`). For `N` from 1 to 9, 16 and 32, on 30 to 300 seeded pairs biased to the edges per width (2,410 unsigned and 2,410 signed in all), the true result of each operation is computed exactly with `Number`, and each form is checked against it: `overflowing` gives it modulo `2^(64N)` and whether it is out of range, `checked` and `wrapping` follow, `saturating` holds it to the range, and the operator panics or wraps as the profile says. This covers sums, differences, products, the limb helpers (`widening_mul`, `carrying_mul`, `carrying_add`, `borrowing_sub`, `widening_mul_u64`), quotients and remainders truncated and Euclid's (with `MIN / −1` and `MIN % −1` at every width), `div_ceil`, multiples, shifts and rotations at 11 amounts around the width, powers up to just past overflow, `isqrt`, every logarithm (against the powers), the bit counts, bytes through slices, `swap_bytes` and `reverse_bits` (against the reversed bytes and binary text), text in a random radix and the formatting flags against `Number`'s, the floats against `Number::to_f64` and `to_f32` (bit for bit), the widths and the `Number` round trip. An ignored test runs 59,200 pairs of each over 20 widths up to 64 limbs (`cargo test --release --test math -- --ignored uint::model`); it passes.
- **Knuth's algorithm D at its rare steps** (`division.rs`). The add-back step: with the divisor `v = V'·b^(n−2) + W`, its top two limbs `V'` normalised and `W ≥ 1`, and the dividend `Q·V'·b^(n−2)`, the estimate from the top three limbs is exactly `Q` while `Q·v` exceeds the dividend; built for every `n` from 3 to the full width `N`, `N` from 3 to 9 and 16, with lower limbs below it and shifted down so that the normalisation brings it back; at the full width, where the dividend has no room for the extra limb, the shift `s ≥ 1` and `Q ≤ 2^s` make the normalisation create it (1,260 cases, 30 for each pair of `N` and `n`, counted by the test; and the smallest ones, `2^192 / (2^191 + 2^64 − 1)` and at the full width of three limbs `2^191 / ((2^191 + 2^64 − 2)/2)`, by hand). Turning the add-back off only for three limbs, or only for divisors of the full width, makes the test fail: those cases reach it. The estimate of `2⁶⁴` or more: the remainder's top limb equal to the divisor's (1,200 cases), and its top two limbs equal to the divisor's with the lower ones below them, which only the test `q̂ ≥ 2⁶⁴` brings down (800 cases). Quotients built as `u = q·v + r` with `q`, `v` and `r < v` at the edges (all-ones limbs, `r = v − 1`, `q` the largest that fits) for `N` from 1 to 9, 16 and 32: 19,500 cases, each giving `q` and `r` back and agreeing with `Number`. Removing the add-back, weakening the third-limb test or dropping the test `q̂ ≥ 2⁶⁴` each makes these tests fail.
- **Text** (`text.rs`). 30,000 random texts for each of `N = 1` and `N = 2` (every radix, signs, zeros in front, digits beyond the radix, letters in both cases, bad characters at random places, lengths at and past the overflow edge), read by `Uint` and `Int` and by `u64`/`i64` or `u128`/`i128`: the same value or the same `IntErrorKind` on all 120,000 (each kind came up more than 50 times), and `str::parse` the same in decimal. The flags of 23 format strings (`{:>45}`, `{:*^46}`, `{:+045}`, `{:#066x}`, `{:#0130b}`, `{:.3}`, …) on 208 values, and of 20 on 5 signed values for each, give `u128`'s and `i128`'s text. `Debug` with 13 sets of flags (`{:>20?}`, `{:*^22?}`, `{:020?}`, `{:+025?}`, `{:x?}`, `{:#?}`, …) gives what `{:#x}` with the same flags gives for `u128` and for `Number` on 300 values of `Uint<2>`, and with 9 sets for four signed values each, against `Number`'s sign and magnitude: 15,900 comparisons. Every radix from 2 to 36 round-trips at every width from 1 to 9, 16 and 32, against `Number::to_str_radix`; `MAX + 1` and `MIN − 1` overflow; text is read in a constant.
- **Conversions** (`convert.rs`, `widths.rs`). `Number`s stored big-endian and little-endian, with zero bytes above the magnitude, empty, and a zero marked negative, convert by value; the primitives convert at every width that holds them, and fail where they must, every failure saying the right way (`PosOverflow` and "too large" above the range, `NegOverflow` and "too small" below it) for all 12 primitive types at the edges of `N = 1` and `N = 2`, with the same text as the standard library's error on the toolchain that runs the tests when it says which way (Rust 1.100 does); floats both ways at 1 to 32 limbs, including halfway cases that only a bit far below the top 64 decides, and rounding up past the largest `f64` to infinity; the constants at every width up to 64 limbs; a power, a root and a division at 4096 bits; methods evaluated in constants.
- **ruint 1.21.1, bnum 0.12.1 and crypto-bigint 0.6.1**, in a probe outside the crate, on 20,000 seeded random values biased to the edges per width and library (192, 256 and 512 bits; crypto-bigint at 256, 384 and 512): 4,190,175 comparisons of the operations they share (sums, products, `widening_mul`/`split_mul`, `div_rem`, powers, `isqrt`/`root(2)`/`sqrt_vartime`, logarithms, bit counts, rotations, shifts, `next_power_of_two`, text in every radix and with flags, floats both ways, and for bnum's `BInt` the signed sums, products, quotients, Euclid's, shifts, `abs`, text and floats). All agree except three groups, each explained: at 192 bits, a width that is not a power of two, bnum masks a shift amount with `BITS − 1` where these types take it modulo `BITS` (24,126 shifts and 11,318 signed shifts, all with the amount at least `BITS`), and its rotations at 192 bits mask the amount the same way, which is wrong even below `BITS`: its `rotate_right` differs on 14,897, of which the 2,113 with an amount below `BITS` all have it from 64 to 127 (checked separately: for every amount from 0 to 600, bnum's shift from 192 on equals the shift by the amount masked with 191 in 81,600 cases of 81,600; both its `rotate_left` and its `rotate_right` differ exactly for the 64 amounts from 64 to 127; ours equals `(x >> k) | (x << (192 − k))` for every `k`, and the rotation that the `Number` model computes); and bnum converts a float between 0.5 and 1 (or −1 and −0.5) to 1 (−1), where `as` gives 0, as these types do (51 cases).
- **Mutations.** Sixteen deliberate faults in the code (the signed product's boundary, the bound of a negative number read, the sticky bit of the float conversion, the zeros between digit groups, the overflow test of the truncated product, the start of the square root, the rotation, the byte order of a `Number`, the signed midpoint's rounding, a shift's carry, the saturating power's sign, the infinities of `from_f64`, a negative shift amount, Euclid's remainder by `MIN`, and two equivalent ones) were each caught by the tests, but the two that change nothing.
- On x86_64 (Ubuntu 26.04, Rust 1.100, in the CI container), these 48 tests and the module's doctests pass.

To run the tests: `cargo test --test math -- uint:: --skip uint::mont`, and with `--release` for the wrapping behaviour.

### Limits

- Not constant-time: the time of an operation depends on the values (the significant limbs of a product or a quotient, the add-back, the exponent's bits, the text's length). These types are not for secret data.
- Products are schoolbook (no Karatsuba) and quotients Knuth's algorithm D, as at these widths in the other fixed-width libraries. The rows of fixed length end at ten limbs, where they no longer unroll, for operands short of the full width: a crossover measured on an Apple M4 Max.
- `N = 0` is a type with the one value 0: sums, products, shifts, text, `Number` conversions and the bytes work on it, but the constants and methods that need a bit (`ONE`, `Int::MIN` and `Int::MAX`, `pow`, `Product`, …) fail to compile for it, and it is not tested beyond that.
- `Debug` is hexadecimal, and while it takes the width, fill, alignment, `+` and `0`, it cannot tell `{:x?}` and `{:X?}` from `{:?}` (stable Rust does not let a type ask), so `{:X?}` is lower-case too; the primitives' `Debug` is decimal.
- The byte arrays need their length named or inferred, `[u8; 32]` for `Uint<4>`: `[u8; 8·N]` cannot be written in code generic over `N` on stable Rust, which uses `write_le_bytes` and `from_le_slice` instead.
- With a shift amount of an untyped literal, a method on the result needs the literal's type (above).
- Overflow follows `debug_assertions` of this crate as it is built, not `overflow-checks`; a profile that sets them apart gives the primitives' behaviour of one and this crate's of the other.
- `div_ceil`, `next_multiple_of` and the other methods that are unstable for the primitives' signed types are not on `Int`.

## Montgomery multiplication and pow_mod

`Montgomery<N>` is the arithmetic modulo an odd `m > 1` of `N` limbs in Montgomery form, with `R = 2^(64·N)`: a number `x` is kept as its form `x·R mod m`, a `Uint<N>` below `m`, and products of forms are reduced by dividing by `R` instead of by `m`. It is computed once for a modulus (it is `Copy`) and then used for as many operations as needed. On top of it, `Uint<N>` has `pow_mod`, `mul_mod`, `add_mod`, `sub_mod` and `mod_inverse` for **any** nonzero modulus, odd or even. The issue's target is 256- to 512-bit moduli (`N` = 4 to 8); every `N ≥ 1` works, and the tests run every width from 1 to 17 limbs and 33 and 129.

Every argument may be any `Uint<N>` and stands for its residue modulo `m` (one at or above `m` is reduced first), and every result is below `m`. Exponents are a `Uint<M>` of any width `M`, the same as `N` or not, or for `pow_limbs` a slice of limbs, the low one first (`&[e]` for a `u64`). `x⁰ = 1` for every `x`, 0 included, and modulo 1 every result is 0, as for `Number::pow_mod`; every number has the inverse 0 modulo 1, as for `Number::mod_inverse` and GMP's `mpz_invert`. A modulus of zero panics with the message of `%`; `Montgomery::new` panics for an even modulus, for 1, and for `N = 0`.

| Function | Gives | Method | Cost |
| --- | --- | --- | --- |
| `Montgomery::new(&m)` | the arithmetic modulo the odd `m > 1` | `−m⁻¹ mod 2¹²⁸` by Newton's iteration; `R mod m` and `R² mod m` by Knuth's algorithm D | `O(N²)` |
| `modulus()`, `one()` | `m`; the form of 1, `R mod m` | | |
| `to_montgomery(&x)` | the form `x·R mod m` of any `x` | REDC of `x·(R² mod m)` | one product |
| `from_montgomery(&x)` | the number `x·R⁻¹ mod m` a form stands for | REDC of `x` | one product |
| `mul(&a, &b)`, `square(&a)` | `a·b·R⁻¹ mod m`, the form of the product | REDC interleaved with the product (CIOS); from 11 limbs the product or the square by `Number`'s schoolbook kernels and a separate REDC by its kernel | `2·N² + N` products of limbs; about `1.5·N²` for a square from 11 limbs |
| `add`, `sub`, `neg` | the forms of `a + b`, `a − b`, `−a` | one conditional subtraction or addition of `m` | `O(N)` |
| `pow(&x, &e)`, `pow_limbs(&x, &[u64])` | the form of `x^e` | sliding window of 1 to 6 bits, sized to the exponent; from 11 limbs `Number::pow_mod` of the number the form stands for | about `ℓ` squarings and `ℓ/(w + 1)` products for `ℓ` bits, and `2^(w−1)` for the table when `w ≥ 2` |
| `inverse(&x)` | the form of `x⁻¹`, or `None` when `gcd(x, m) ≠ 1` | binary GCD in batches of 31 steps (Pornin), then two products by `R² mod m`; from 11 limbs `Number::mod_inverse` | `O(N²)` |
| `x.pow_mod(&e, &m)` | `x^e mod m` for any nonzero `m` | odd `m`: `Montgomery`; even `m = 2ᵏ·q`: modulo `q` in Montgomery form, modulo `2ᵏ` by wrapping products, put together by the Chinese remainder theorem; from 11 limbs `Number::pow_mod` | `O(N²·ℓ)`; for an even `m` another `O(N²·log N)` for `q⁻¹ mod 2^(64·N)` |
| `x.mul_mod(&y, &m)` | `x·y mod m` for any nonzero `m` | the double-width product (from 11 limbs by `Number`'s schoolbook kernel), reduced by algorithm D | `O(N²)` |
| `x.add_mod(&y, &m)`, `x.sub_mod(&y, &m)` | `x ± y mod m` for any nonzero `m` | operands at or above `m` reduced by algorithm D, then one conditional correction | `O(N)` for reduced operands |
| `x.mod_inverse(&m)` | the `r < m` with `x·r ≡ 1`, or `None` | odd `m`: binary GCD in batches of 31 steps; even `m = 2ᵏ·q`: Newton's iteration modulo `2ᵏ`, the binary GCD modulo `q`, the Chinese remainder theorem; one limb, any `m`: the extended Euclidean algorithm; from 11 limbs `Number::mod_inverse` | `O(N²)`; `O(N²·log N)` for an even `m` |

### Methods

- **REDC, CIOS.** Montgomery's reduction (P. L. Montgomery, *Modular multiplication without trial division*, Math. Comp. 44, 1985) adds to `t` the multiple `q·m` with `q = t₀·(−m⁻¹) mod 2⁶⁴`, which clears the low limb, and shifts the limb out; after `N` limbs `t ≡ a·b·R⁻¹ (mod m)`. The product and the reduction are interleaved a limb of `b` at a time, the method CIOS of Koç, Acar and Kaliski (*Analyzing and comparing Montgomery multiplication algorithms*, IEEE Micro 16(3), 1996): a row `t += a·bᵢ`, a row `t = (t + q·m)/2⁶⁴`, `2·N² + N` products of limbs in all. Between the rows `t < a + m < 2·R`, so `t` is kept in `N` limbs and a carry; that carry is what makes a modulus near `R` (its top limb `u64::MAX`, up to `R − 1` itself) come out right. When `a·b < m·R`, which holds as soon as one factor is below `m`, the end result is below `2·m`, and one conditional subtraction leaves it below `m`. Below 11 limbs `square` is `mul` of a number by itself. From 11 limbs on, the product is `Number`'s schoolbook kernel (assembly on aarch64 and x86-64), the square its square kernel, with the `N(N − 1)/2` cross products once and doubled, and the reduction of the `2N` limbs a separate pass by `Number`'s REDC kernel, which clears two limbs at a time with `−m⁻¹ mod 2¹²⁸`, the crossover measured on an Apple M4 Max.
- **The constants.** `−m⁻¹ mod 2¹²⁸` by Newton's iteration `y ← y·(2 − m·y)` from `y = m`, which is right to 3 bits and doubles them each step. `R mod m` and `R² mod m` are the remainders of `2^(64·N)` and `2^(128·N)` by Knuth's algorithm D (*The Art of Computer Programming*, vol. 2, 4.3.1), normalised so that the divisor's top bit is set, the dividend shifted as far into a buffer on the stack, and reduced there from the top, with the loops unrolled for a divisor of `N` limbs: each quotient limb is the quotient of the top three limbs by the divisor's top two, by Möller and Granlund's 3-by-2 division with the divisor's reciprocal (a 2-by-1 division for a divisor of one limb), right or one too large (Theorem B), which the add-back of step D6 corrects; the 3-by-2 division gives the top two limbs of the remainder, so only the divisor's other limbs times the quotient limb are subtracted. The same division reduces the double-width products of `mul_mod` and the operands of `add_mod`, `sub_mod`, `mod_inverse`, and is the one of `Uint::div_rem`; where the dividend's top limbs are already below the divisor, as for a product of two residues, it starts a limb lower, `N` quotient limbs for `mul_mod`.
- **Powers.** The sliding-window method (Menezes, van Oorschot and Vanstone, *Handbook of Applied Cryptography*, 1996, Algorithm 14.85): from the top bit down a zero bit costs a squaring, and a window of up to `w` bits that starts and ends with a one costs its length in squarings and one product by an odd power from a table `x, x³, …, x^(2^w − 1)`. For an exponent of `ℓ` bits, `w` minimises `T(w) + ℓ/(w + 1)`, about `ℓ/(w + 1)` windows and the cost `T(w)` of the table: 0 for `w = 1`, which needs no table, and `2^(w−1)` for `w ≥ 2` (a squaring for `x²` and `2^(w−1) − 1` products), a tie going to the smaller `w`. That gives 1 up to 12 bits, 2 up to 24, 3 up to 80, 4 up to 240, 5 up to 672, 6 beyond (checked against the code for every length up to 20,000 bits). The table is on the stack: from 33 limbs on `w` is at most 4, from 129 limbs on at most 2, which keeps it within 8 KiB up to 512 limbs. Counted on 200 random exponents of each length, the products came to `ℓ/(w + 1) + 2^(w−1)` less 1.5 on average (57.2 for 256 bits, 99.7 for 512, 176.7 for 1024, where binary exponentiation takes 128, 256 and 512), and the squarings to `ℓ` less 1 to 4.
- **Even moduli.** `m = 2ᵏ·q` with `q` odd: `x^e mod q` in Montgomery form, `x^e mod 2^(64·N)` by wrapping products (the low halves) and its low `k` bits, put together in Garner's form `y = r₁ + q·((r₂ − r₁)·q⁻¹ mod 2ᵏ)`, which is below `m` and so needs no reduction. `q⁻¹ mod 2^(64·N)` is Newton's iteration again, from the inverse of the low limb, `⌈log₂ N⌉` steps of two wrapping products, `O(N²·log N)` in all, which is what an even modulus adds to `pow_mod` and `mod_inverse`. A power of two (`q = 1`) is the wrapping power alone. `mod_inverse` splits the same way: an inverse modulo `2ᵏ` exists exactly for odd numbers.
- **Inverses.** The binary GCD, which needs an odd modulus but not a prime one, in the form of Pornin (*Optimized binary GCD for modular inversion*, IACR ePrint 2020/972, Algorithm 2), not constant-time: it keeps `a ≡ u·x` and `b ≡ v·x (mod m)` from `a = x, b = m`, `b` odd; an even `a` is halved, an odd one, if below `b`, swapped with it, then `a ← (a − b)/2`, with `u` following modulo `m`; when `a` reaches 0, `b` is the GCD, and for `b = 1`, `v` the inverse. The steps run 31 at a time on 64-bit approximations of `a` and `b` — their low 31 bits, which decide every parity exactly, and their top 33 bits at a common position, which decide the comparisons — without branches, recording them as factors of at most `2³¹` that are then applied to the full `a` and `b` (negated where a wrong comparison made one negative) and to `u` and `v` modulo `m`, divided by `2³¹` there as in Montgomery's reduction. Below 65 bits the approximations are exact. Each batch shortens `a·b` by about 31 bits: about `2·64·N/31` batches of `O(N)` products. For one limb, odd modulus or even, the extended Euclidean algorithm with the hardware's 64-bit division is faster: the cofactors of `x` alternate in sign, so their absolute values, which grow to at most `m`, are kept without signs and the parity of the last step gives the sign. From 11 limbs on, `mod_inverse` and `Montgomery::inverse` call `Number::mod_inverse` (Lehmer's and subquadratic GCDs), and `pow_mod` and `Montgomery::pow` `Number::pow_mod` (products and reductions on assembly kernels), faster there even with the conversions (measured on an Apple M4 Max on random operands); below 11 limbs an inverse modulo an even number is not, as an even number has no inverse there, which this module sees at once. `Montgomery::inverse` inverts the form `x·R` itself and multiplies by `R² mod m` twice, which takes `(x·R)⁻¹` to `x⁻¹·R`.

### Example

```rust
use a::math::uint::{Montgomery, Uint};

// The prime of secp256k1, p = 2^256 − 2^32 − 977, the low limb first.
let p = Uint::<4>::from_limbs([0xFFFF_FFFE_FFFF_FC2F, u64::MAX, u64::MAX, u64::MAX]);
let f = Montgomery::new(&p);
let one = Uint::from_u64(1);
let (p1, _) = p.overflowing_sub(one);
let (p2, _) = p1.overflowing_sub(one);
let x = f.to_montgomery(&Uint::from_u64(7));
// Fermat: x^(p−1) = 1, and x^(p−2) is the inverse.
assert_eq!(f.pow(&x, &p1), f.one());
let inv = f.inverse(&x).unwrap();
assert_eq!(f.pow(&x, &p2), inv);
assert_eq!(f.from_montgomery(&f.mul(&x, &inv)), one);

// pow_mod for any modulus: 3^1000 modulo the even 5·2^200, against 1000 products.
let m = Uint::<4>::from_limbs([0, 0, 0, 5 << 8]);
let three = Uint::<4>::from_u64(3);
let mut y = Uint::from_u64(1);
for _ in 0..1000 {
    y = y.mul_mod(&three, &m);
}
assert_eq!(three.pow_mod(&Uint::<1>::from_u64(1000), &m), y);

// Modulo 2^40 the odd numbers have inverses and the even ones none.
let m = Uint::<2>::from_u64(1 << 40);
let i = Uint::<2>::from_u64(3).mod_inverse(&m).unwrap();
assert_eq!(i.mul_mod(&Uint::from_u64(3), &m), Uint::from_u64(1));
assert_eq!(Uint::<2>::from_u64(6).mod_inverse(&m), None);
```

### Verified

- **`Number`, at every width and every length of the modulus.** For each `N` from 1 to 9 and 16 (`lib/tests/math/uint/mont.rs`, seeded): an odd modulus of each bit length from 2 to `64·N`, 3,894 in all, against `Number` for `R mod m`, `to_montgomery`, `from_montgomery`, their round trip, `mul` (as `a·b·R⁻¹ mod m`, both orders), `square`, `add`, `sub`, `neg`, `inverse` (as `R²·a⁻¹ mod m` or `None`, by `Number::mod_inverse`), and `pow` or `pow_limbs` of the first operand by an exponent of up to 2 limbs (as `(a·R⁻¹)^e·R mod m`), on 9,696 pairs of operands: 5,166 random, half reduced and half anywhere below `R`, biased to 0, 1, `MAX`, all-ones and zero limbs and single bits, and 4,530 with a first operand from `m`, `m + 1`, `2m − 1` (where it fits), `m − 1` and `MAX`, which every operation must reduce or keep, taken in turn; `pow_mod` against `Number::pow_mod` for an odd and an even modulus of each length (7,788 powers), the even ones with 1 to `ℓ − 1` trailing zeros and powers of two among them, with exponents of 1 and 2 limbs and small ones; `mul_mod`, `add_mod`, `sub_mod` and `mod_inverse` (also of an odd operand modulo an even modulus) for a modulus of each length from 1 bit, odd and even (3,904).
- **Both sides of the crossovers**: at 9 and 10 limbs (the rows of CIOS, the powers' own windows) and at 11 and 12 (`Number`'s kernels for the products, powers and inverses by `Number`), the `Montgomery` check against `Number` for an odd modulus of every length, `pow_mod` for an odd and an even modulus of every length, and the moduli of special shape below (`lib/tests/math/uint/mont.rs`, `the_widths_on_both_sides_of_the_crossovers_agree_with_number`).
- **Moduli of special shape and edge operands.** For each of the ten widths, 304 moduli in all: `2^(64N) − 1`, `2^(64N−1) + 1`, `2^(64N−1)`, `2^(64N) − 2`, 2, 3, `2⁶⁴ − 1`, `2⁶³`, the top limb `u64::MAX` (odd and even), and `2ᵏ − 1`, `2ᵏ`, `2ᵏ + 1` for `k` = 2, 63, 64, 65, `32N`, `64N − 2`, `64N − 1`. For each, the bases 0, 1, `m − 1`, `m`, `m + 1`, `MAX` and a random one, and the exponents 0, 1, 2, `2⁶⁴ − 1`, `2^(64N) − 1`, `m − 1` and a random one: `pow_mod` and `Montgomery::pow` against `Number::pow_mod` (14,896 powers), `mul_mod`, `add_mod`, `sub_mod` for every pair of the bases, `mod_inverse` of each base, and the whole `Montgomery` check above on the odd moduli, with every one of the operands `m`, `m + 1`, `2m − 1`, `m − 1` and `MAX` that fits. Modulo `2^(64N−1) + 1` for `N ≥ 3`, the remainder of `R` goes through the add-back of algorithm D (estimated quotient 2, true quotient 1), checked by its own test.
- **Named primes.** The primes of P-256, secp256k1, BN254 and Curve25519 (4 limbs), P-384 and BLS12-381's base field (381 bits, 6 limbs), P-521 (9 limbs), and P-256 and BLS12-381 again in 5 and 8 limbs, where the top limbs are zero: Fermat's `x^(p−1) = 1` and `x^p = x`, `x^(p−2)` equal to `inverse`, `mod_inverse` equal to it, and `pow_mod` with full-length exponents against `Number`; and the whole `Montgomery` check, every edge operand included.
- **RSA-like moduli.** Products of two primes of 128, 256, 512 and 1024 bits (Python with SymPy's `nextprime`, seeded): the factors, their multiples, 0 and `m` have no inverse (both `inverse` and `mod_inverse`); for random `x`, `x^65537` against `Number`, the round trip with `d = 65537⁻¹ mod φ(m)`, and Euler's `x^φ(m) = 1` for units; and the whole `Montgomery` check, every edge operand included.
- **Exponents of other widths and every window.** `Uint<M>` exponents with `(N, M)` = (1, 12), (1, 3), (4, 1), (4, 11), (4, 16), (6, 2), (8, 4), (8, 12), (9, 1), (16, 8), of 1, 12, 13, 24, 25, 80, 81, 240, 241, 672, 673 and `64·M` bits (each side of every change of window), modulo odd and even moduli, against `Number`; `pow_limbs` equal to `pow`, also with zero limbs on top; exponents of no limbs and of zero limbs only. `N` = 33 and 129, where the table is smaller, against `Number` with exponents of 5 to 300 bits.
- **Inverses exactly for units.** Modulo 15015 = 3·5·7·11·13 every residue: `mod_inverse` and `Montgomery::inverse` exist exactly when the gcd is 1, and multiply to 1; modulo `2⁶⁴·15015` the numbers 1 to 2999 against `Number::mod_inverse`. Powers of two `2ᵏ` for `k` from 1 to 127 as moduli; modulo 1 every operation gives 0 and `mod_inverse` gives `Some(0)`; `0⁰ = 1`.
- **Python's `pow`.** 50 lines (`data/mont_powm.txt`, 18.9 KB, by `gen_mont_powm.py`): for each of the ten widths, odd and even moduli of full and random length, with many trailing zeros, and with the top limb all ones, bases below `R` and exponents of up to 1024 bits.
- **GMP 6.3** (`mpz_powm`, `mpz_mul` and `mpz_mod`, `mpz_add`, `mpz_sub`, `mpz_invert`), in a probe outside the crate, release build with overflow checks on: 2,312,500 random cases over the ten widths (1,145,673 odd and 1,166,827 even moduli of random length, bases and exponents of `N` limbs, a quarter of the exponents of up to 64 bits), each comparing `pow_mod`, `mul_mod`, `add_mod`, `sub_mod` and `mod_inverse`, and for the odd moduli `Montgomery::mul` against `a·b·R⁻¹ mod m`, `to_montgomery`, `pow` and `inverse` (4,538,004 operations): all agree.
- **ruint 1.21.1, crypto-bigint 0.6.1 and bnum 0.13.0**, in a probe: 185,000 random cases over the ten widths against ruint's `pow_mod`, `mul_mod`, `add_mod`, `inv_mod` and, for odd moduli, `mul_redc`; against bnum's products and remainders at twice the width, and a square-and-multiply on them; and, at the widths crypto-bigint supports (all but 9 limbs), against its `mul_mod_vartime` and `inv_mod` (165,000 cases, odd and even moduli) and, for the 82,690 odd moduli among them, its `MontyForm`: the Montgomery forms themselves (the same `R`), `mul`, `square`, `add`, `sub`, `neg`, `pow`, `inv` and `retrieve`, and its `add_mod` and `sub_mod`. All agree.
- **The tests catch faults.** Each of these deliberate faults made tests fail: the final subtraction ignoring the carry, the carry of the product row dropped, a missing reduction of an unreduced factor, the add-back of algorithm D removed, its quotient test removed, the shift back after the division dropped, the carry of a halving in the inverse dropped, `−m⁻¹ mod 2⁶⁴` one Newton step short, `q⁻¹ mod 2^(64N)` one step short, the Garner coefficient not masked to `k` bits, an off-by-one mask, a common divisor taken for an inverse, the carry of a modular sum ignored, a squaring missing in a window, the table built from the wrong power; and the two that an independent audit found the first version of these tests missed, `pow` not reducing an argument at or above `m`, and the shared reduction keeping an argument equal to `m` (`<` turned into `≤`).
- **Heavier**, in release (the ignored test `a_heavier_sweep_agrees_with_number`): 16 random moduli of each length, odd and even, at each of the ten widths, 62,304 powers with exponents of `N` limbs, and the `Montgomery` check on the odd ones.
- On x86_64 (Ubuntu 26.04, Rust 1.100, in the CI container) the same tests and the examples pass.

To run the tests: `cargo test --test math -- uint::mont`, and `cargo test --release --test math -- --ignored uint::mont` for the heavier sweep. The data file is written again by `python3 gen_mont_powm.py > mont_powm.txt`.

### Limits

- **Nothing is constant-time**, and none of it is meant for secret exponents or keys: the time of a power depends on the exponent's length and bits (a zero bit costs a squaring, a window a product); the final subtraction of a product is taken or not depending on the values; an operand at or above the modulus is reduced and one below it is not; the inverse takes as many steps as its values need; the division's corrections depend on the values.
- `Uint::pow_mod` and the other functions on `Uint` compute their constants on every call (for `pow_mod` the constants of `Montgomery`, for an even modulus also `q⁻¹ mod 2^(64·N)`); for many operations modulo the same odd number, a `Montgomery` kept and reused saves them.
- Below 11 limbs the products are the plain rows of CIOS (and `square` is a product), with no assembly and no SIMD. The crossovers (the kernels and `Number` from 11 limbs on) were measured on an Apple M4 Max only. One crossover serves powers and inverses alike, and the inverse modulo an even number holds it at 11 limbs: `Number`'s powers by long exponents, and its inverses modulo an odd number, are faster from 2 limbs on.
- The values and the table of a power are on the stack; for very wide `Uint`s (thousands of limbs) that is a lot of stack, beyond the window's limit of two powers from 129 limbs on.
- `N = 0` holds no odd number and no nonzero modulus: `Montgomery::new` panics, and so do `pow_mod`, `mul_mod`, `add_mod`, `sub_mod` and `mod_inverse` on `Uint<0>`, whose only modulus is 0.

## Measured

Measured on an Apple M4 Max (macOS 27.2) on 2026-11-26 between 07:55 and 07:57, with nothing else running: the areas of the release's final run were timed one after another, and every timed process ran alone on the machine. One thread; `a` as a user builds it, `cargo build --release` with the default profile (opt-level 3, no LTO, the default codegen units, no `target-cpu`), on Rust 1.100.0 stable, and the Rust references the same way, by the harness in the repository's [`bench/numtypes`](../../../bench/numtypes) (see [`bench/README.md`](../../../bench/README.md)). Each figure is the median time an operation of 21 samples of at least 1.5 ms, each a pass over a pool of fresh random inputs (64 bytes to 96 KiB), all the contenders of a row alternating sample by sample in one process; every reference's results were compared with `a`'s on the whole pool before the timing, exactly. Each row gives the fastest reference that does the same work, its time and `a`'s time over it: below 1, `a` is faster. Times below a nanosecond are a few cycles, and their spreads are wide. The references: ruint 1.21.1, crypto-bigint 0.6.1, bnum 0.13.0, malachite-nz 0.14.0, and GMP 6.3.0 and OpenSSL 3.5.9's `BIGNUM` (Homebrew's builds, called through FFI: GMP's `mpn` functions a library call an item, `mpz` functions into a preallocated `mpz_t`, OpenSSL's into `BIGNUM`s and a `BN_MONT_CTX` kept); the primitive `u64`, `u128`, `i64` and `i128` at 64 and 128 bits. Not counted, as they do other work: crypto-bigint's constant-time forms (`MontyForm::pow`, `inv_odd_mod`, `inv_mod`), which `a` does not offer (nothing here is constant-time), and its `add_mod` and `sub_mod`, which need reduced operands where `a`'s reduce them.

### `Uint` and `Int`

`Uint<N>` of `N` limbs, random full-width operands unless a row says otherwise; `div_rem` of a dividend of `N` limbs by a divisor of the limbs given, `div_rem_u64` by an odd 64-bit divisor, the shifts by random amounts below the width, `pow` of a base below `2^16` whose power fits, `wrapping_pow` of a random base by an exponent from 2 to 255, `checked_mul` of operands whose bit lengths sum to the width ± 32; `to_string` and `to_str_radix(16)` give a `String` (GMP's `mpz_get_str` writes into a caller's buffer):

| Operation | Bits | `a` | Fastest | Its time | `a`/fastest |
| --- | --- | ---: | --- | ---: | ---: |
| `wrapping_add` | 64 | 0.696 ns | bnum | 0.725 ns | 0.96 |
|  | 256 | 1.55 ns | ruint | 1.55 ns | 1.00 |
|  | 1,024 | 3.53 ns | ruint | 3.55 ns | 1.00 |
| `wrapping_mul` | 64 | 0.646 ns | bnum | 0.732 ns | 0.88 |
|  | 128 | 0.905 ns | ruint | 1.38 ns | 0.66 |
|  | 256 | 2.18 ns | ruint | 1.64 ns | 1.33 |
|  | 512 | 7.75 ns | ruint | 6.73 ns | 1.15 |
|  | 1,024 | 79.1 ns | GMP | 45.1 ns | 1.75 |
| `widening_mul` | 64 | 1.41 ns | `u64` | 1.35 ns | 1.04 |
|  | 256 | 3.85 ns | bnum | 3.77 ns | 1.02 |
|  | 512 | 20.1 ns | GMP | 27.7 ns | 0.73 |
|  | 1,024 | 144 ns | GMP | 92.0 ns | 1.57 |
| `overflowing_mul` | 128, operands of 64 bits | 1.81 ns | bnum | 0.815 ns | 2.22 |
|  | 256, operands of 128 bits | 4.07 ns | ruint | 2.62 ns | 1.56 |
|  | 1,024, operands of 512 bits | 41.2 ns | ruint | 44.0 ns | 0.94 |
| `checked_mul` | 128 | 2.02 ns | ruint | 1.60 ns | 1.26 |
|  | 256 | 4.99 ns | ruint | 2.61 ns | 1.91 |
|  | 1,024 | 53.6 ns | ruint | 45.5 ns | 1.18 |
| `div_rem` | 128, divisor of 1 limb | 7.56 ns | ruint | 5.85 ns | 1.29 |
|  | 256, divisor of 2 limbs | 23.9 ns | ruint | 15.9 ns | 1.50 |
|  | 1,024, divisor of 8 limbs | 127 ns | GMP | 99.7 ns | 1.27 |
|  | 128, divisor of 2 limbs | 5.66 ns | `u128` | 2.59 ns | 2.18 |
|  | 256, divisor of 4 limbs | 10.9 ns | ruint | 7.40 ns | 1.47 |
|  | 1,024, divisor of 16 limbs | 33.5 ns | GMP | 25.2 ns | 1.33 |
| `div_rem_u64` | 128 | 5.99 ns | GMP | 5.20 ns | 1.15 |
|  | 256 | 9.62 ns | GMP | 7.98 ns | 1.21 |
|  | 1,024 | 49.5 ns | GMP | 32.5 ns | 1.52 |
| `wrapping_shr` | 64 | 0.658 ns | ruint | 0.622 ns | 1.06 |
|  | 256 | 2.30 ns | ruint | 1.58 ns | 1.46 |
|  | 1,024 | 8.61 ns | bnum | 5.91 ns | 1.46 |
| `cmp` | 64 | 0.377 ns | bnum | 0.350 ns | 1.08 |
|  | 256 | 0.503 ns | crypto-bigint | 0.509 ns | 0.99 |
|  | 1,024 | 1.00 ns | crypto-bigint | 1.01 ns | 0.99 |
| `isqrt` | 64 | 4.19 ns | bnum | 1.74 ns | 2.41 |
|  | 128 | 8.19 ns | GMP | 9.06 ns | 0.91 |
|  | 256 | 45.2 ns | GMP | 42.1 ns | 1.08 |
|  | 1,024 | 149 ns | GMP | 169 ns | 0.88 |
| `pow` | 128 | 1.60 ns | `u128` | 1.61 ns | 0.99 |
|  | 256 | 5.37 ns | bnum | 8.57 ns | 0.63 |
|  | 1,024 | 88.2 ns | GMP | 57.9 ns | 1.52 |
| `wrapping_pow` | 128 | 5.60 ns | bnum | 4.80 ns | 1.17 |
|  | 256 | 14.8 ns | bnum | 12.4 ns | 1.19 |
|  | 1,024 | 554 ns | bnum | 643 ns | 0.86 |
| `ilog10` | 64 | 0.960 ns | `u64` | 1.04 ns | 0.93 |
|  | 128 | 2.50 ns | `u128` | 1.54 ns | 1.62 |
|  | 256 | 12.4 ns | ruint | 99.1 ns | 0.12 |
|  | 1,024 | 82.2 ns | ruint | 733 ns | 0.11 |
| `to_f64` | 128 | 1.45 ns | `u128` | 1.28 ns | 1.13 |
|  | 512 | 2.75 ns | ruint | 1.71 ns | 1.60 |
|  | 1,024 | 2.67 ns | ruint | 3.92 ns | 0.68 |
| `leading_zeros` | 256 | 0.699 ns | ruint | 0.678 ns | 1.03 |
|  | 384 | 1.50 ns | bnum | 1.15 ns | 1.31 |
|  | 1,024 | 2.85 ns | crypto-bigint | 2.90 ns | 0.98 |
| `to_string` | 64 | 26.7 ns | `u64` | 20.4 ns | 1.31 |
|  | 128 | 30.1 ns | `u128` | 24.6 ns | 1.23 |
|  | 256 | 75.3 ns | ruint | 96.5 ns | 0.78 |
|  | 1,024 | 496 ns | GMP | 599 ns | 0.83 |
| `from_str_radix(10)` | 64 | 14.6 ns | `u64` | 14.4 ns | 1.01 |
|  | 256 | 50.5 ns | bnum | 68.3 ns | 0.74 |
|  | 1,024 | 248 ns | malachite-nz | 259 ns | 0.96 |
| `to_str_radix(16)` | 64 | 27.9 ns | GMP | 14.1 ns | 1.98 |
|  | 256 | 53.5 ns | bnum | 38.8 ns | 1.38 |
|  | 1,024 | 150 ns | bnum | 114 ns | 1.32 |
| `from_str_radix(16)` | 64 | 8.91 ns | malachite-nz | 8.63 ns | 1.03 |
|  | 256 | 50.0 ns | malachite-nz | 65.9 ns | 0.76 |
|  | 1,024 | 254 ns | bnum | 222 ns | 1.15 |
| `Int` `wrapping_mul` | 64 | 0.596 ns | bnum | 0.711 ns | 0.84 |
|  | 256 | 2.16 ns | bnum | 1.63 ns | 1.33 |
|  | 1,024 | 81.6 ns | bnum | 62.7 ns | 1.30 |
| `Int` `wrapping_abs` | 64 | 0.739 ns | bnum | 0.640 ns | 1.15 |
|  | 256 | 1.55 ns | bnum | 2.28 ns | 0.68 |
|  | 1,024 | 2.73 ns | bnum | 11.7 ns | 0.23 |
| `Int` `div_rem` | 64, divisor of 1 limb | 1.40 ns | `i64` | 1.46 ns | 0.96 |
|  | 256, divisor of 2 limbs | 20.8 ns | crypto-bigint | 78.7 ns | 0.27 |
|  | 1,024, divisor of 8 limbs | 132 ns | crypto-bigint | 324 ns | 0.41 |
| `Int` `to_string` | 64 | 29.4 ns | `i64` | 19.8 ns | 1.48 |
|  | 128 | 42.4 ns | `i128` | 28.6 ns | 1.49 |
|  | 256 | 75.7 ns | bnum | 219 ns | 0.35 |
|  | 1,024 | 497 ns | bnum | 3.27 µs | 0.15 |
| `Int` `from_str_radix(10)` | 64 | 17.1 ns | `i64` | 14.5 ns | 1.18 |
|  | 256 | 53.5 ns | bnum | 70.8 ns | 0.76 |
|  | 1,024 | 256 ns | bnum | 346 ns | 0.74 |

- The sums, the comparisons and the bit counts take 0.74 to 1.08 of the fastest reference's time, above 1.05 only for `cmp` and `checked_add` of one limb, a fraction of a nanosecond; `widening_mul` to 256 bits 1.01 to 1.04; `widening_mul_u64` takes 0.44 to 1.04 of GMP's `mpn_mul_1`, and `widening_mul` at 512 bits 0.73 of GMP's `mpn_mul_n`.
- From 256 bits the text is faster than every reference's both ways in decimal (`to_string` 0.73 to 0.83, `from_str_radix(10)` 0.74 to 0.96), and so are `ilog10` (0.088 to 0.12) and the signed `div_rem`, `wrapping_abs`, arithmetic shifts and text (0.15 to 0.80).
- Behind: the quotients (`div_rem` 1.08 to 2.18 from 128 bits, `div_rem_u64` 1.15 to 1.52), the truncated and checked products from 128 to 384 bits, the shifts from 256 bits, hexadecimal text, and the text of one and two limbs against the primitive types (see [below](#where-the-references-are-faster)).

### Montgomery and modular arithmetic

`Montgomery` of an odd modulus of `N` limbs, against ruint (`mul_redc`, `square_redc`, `pow_mod`, `inv_mod`, `mul_mod`, `add_mod`), crypto-bigint (`MontyParams::new_vartime`, `MontyForm`'s `new`, `retrieve`, `mul`, `square` and `inv_vartime`), malachite-nz, GMP (`mpz_powm`, `mpz_invert`, `mpn_mul_n` and `mpn_tdiv_qr` for `mul_mod`) and OpenSSL (`BN_MONT_CTX_set`, `BN_mod_mul_montgomery`, `BN_mod_exp` and `BN_mod_exp_mont`), random odd moduli of the full width and, where marked, even ones; `pow_mod` and `pow` by an exponent of the modulus's width, `mul_mod` and `add_mod` of reduced operands:

| Operation | Bits | `a` | Fastest | Its time | `a`/fastest |
| --- | --- | ---: | --- | ---: | ---: |
| `Montgomery::new` | 64 | 6.10 ns | crypto-bigint | 77.7 ns | 0.079 |
|  | 256 | 47.7 ns | crypto-bigint | 376 ns | 0.13 |
|  | 1,024 | 284 ns | crypto-bigint | 1.98 µs | 0.14 |
|  | 4,096 | 3.67 µs | OpenSSL | 12.8 µs | 0.29 |
| `Montgomery::mul` | 64 | 1.00 ns | ruint | 1.03 ns | 0.97 |
|  | 256 | 11.7 ns | ruint | 10.9 ns | 1.08 |
|  | 512 | 49.9 ns | OpenSSL | 37.7 ns | 1.32 |
|  | 1,024 | 171 ns | OpenSSL | 130 ns | 1.32 |
|  | 2,048 | 545 ns | OpenSSL | 518 ns | 1.05 |
|  | 4,096 | 2.26 µs | OpenSSL | 2.03 µs | 1.11 |
| `Montgomery::square` | 256 | 11.8 ns | OpenSSL | 11.0 ns | 1.07 |
|  | 512 | 49.8 ns | OpenSSL | 27.4 ns | 1.82 |
|  | 1,024 | 160 ns | OpenSSL | 92.7 ns | 1.73 |
|  | 4,096 | 1.78 µs | OpenSSL | 1.30 µs | 1.37 |
| `to_montgomery` | 64 | 0.660 ns | crypto-bigint | 5.97 ns | 0.11 |
|  | 1,024 | 142 ns | crypto-bigint | 306 ns | 0.46 |
|  | 4,096 | 2.23 µs | crypto-bigint | 4.60 µs | 0.48 |
| `from_montgomery` | 64 | 0.555 ns | crypto-bigint | 2.24 ns | 0.25 |
|  | 1,024 | 143 ns | crypto-bigint | 172 ns | 0.83 |
|  | 4,096 | 2.24 µs | crypto-bigint | 3.07 µs | 0.73 |
| `Montgomery::inverse` | 64 | 96.2 ns | crypto-bigint | 256 ns | 0.38 |
|  | 256 | 879 ns | crypto-bigint | 1.11 µs | 0.79 |
|  | 1,024 | 2.88 µs | crypto-bigint | 7.97 µs | 0.36 |
|  | 4,096 | 20.0 µs | crypto-bigint | 129 µs | 0.16 |
| `Montgomery::pow` | 256 | 4.87 µs | OpenSSL | 4.56 µs | 1.07 |
|  | 512 | 30.4 µs | OpenSSL | 20.1 µs | 1.51 |
|  | 1,024 | 108 µs | OpenSSL | 123 µs | 0.88 |
|  | 2,048 | 1.09 ms | OpenSSL | 862 µs | 1.27 |
|  | 4,096 | 8.12 ms | OpenSSL | 6.63 ms | 1.22 |
| `Uint::pow_mod` | 64 | 281 ns | GMP | 250 ns | 1.13 |
|  | 256 | 4.97 µs | OpenSSL | 6.64 µs | 0.75 |
|  | 512 | 30.5 µs | OpenSSL | 22.2 µs | 1.37 |
|  | 1,024 | 108 µs | OpenSSL | 126 µs | 0.86 |
|  | 2,048 | 1.09 ms | OpenSSL | 870 µs | 1.25 |
|  | 4,096 | 8.07 ms | OpenSSL | 6.63 ms | 1.22 |
|  | 256, even modulus | 5.62 µs | GMP | 8.18 µs | 0.69 |
|  | 4,096, even modulus | 8.02 ms | GMP | 9.17 ms | 0.88 |
| `Uint::mod_inverse` | 64 | 106 ns | malachite-nz | 36.3 ns | 2.92 |
|  | 128 | 393 ns | ruint | 219 ns | 1.79 |
|  | 256 | 859 ns | ruint | 515 ns | 1.67 |
|  | 512 | 2.07 µs | ruint | 1.27 µs | 1.64 |
|  | 1,024 | 2.86 µs | malachite-nz | 2.90 µs | 0.98 |
|  | 4,096 | 15.0 µs | malachite-nz | 16.9 µs | 0.89 |
|  | 256, even modulus | 876 ns | ruint | 516 ns | 1.70 |
| `Uint::mul_mod` | 64 | 4.09 ns | ruint | 4.26 ns | 0.96 |
|  | 256 | 37.8 ns | ruint | 36.1 ns | 1.05 |
|  | 1,024 | 318 ns | GMP | 304 ns | 1.05 |
|  | 2,048 | 1.26 µs | GMP | 912 ns | 1.38 |
|  | 4,096 | 4.65 µs | GMP | 2.80 µs | 1.66 |
| `Uint::add_mod` | 64 | 0.749 ns | ruint | 3.26 ns | 0.23 |
|  | 256 | 3.08 ns | ruint | 4.28 ns | 0.72 |
|  | 1,024 | 9.46 ns | ruint | 20.6 ns | 0.46 |
|  | 4,096 | 80.5 ns | malachite-nz | 74.9 ns | 1.07 |

- Setting a modulus up and converting into and out of its form take 0.079 to 0.98 of the fastest reference's time, `Montgomery::inverse` 0.16 to 0.83 of crypto-bigint's `inv_vartime`.
- `Uint::pow_mod` takes 0.44 to 0.91 of the fastest reference's time at 128, 256, 384 and 1,024 bits, and is behind at 512, 2,048 and 4,096 bits for odd moduli (1.22 to 1.37, against OpenSSL), where the products are; for even moduli, against GMP, 0.53 to 1.11.
- `Uint::mod_inverse` is behind to 512 bits (1.64 to 2.92) and ahead from 1,024 (0.89 to 0.99); `mul_mod` at par to 1,024 bits (0.92 to 1.05) and behind from 2,048 (1.37 to 1.66).

### Where the references are faster

Every case above 1.05 times the fastest same-work reference's time, by cause: 105 of the 315 cases of the integers and the modular arithmetic that have one.

- **Quotients** (`div_rem` 1.08 to 2.18, `div_rem_u64` 1.15 to 1.52): Knuth's algorithm D with Möller and Granlund's reciprocals, 3-by-2 for the quotient digits and 2-by-1 for a divisor of one limb, in portable Rust, against ruint to 512 bits and GMP's `mpn_tdiv_qr` and `mpn_divrem_1` beyond (and `u128`'s division, compiler-rt's, at 128 bits by a full-width divisor, 2.18); the method is GMP's, and the remaining difference was not narrowed.
- **Products cut to the width, and the powers on them** (`wrapping_mul` 1.15 to 1.75, `overflowing_mul`, `checked_mul` and `saturating_mul` 1.07 to 2.22, `wrapping_pow` at 128 and 256 bits 1.17 to 1.19, `Int::wrapping_mul` 1.30 to 1.33): the low half is rows of fixed length up to ten limbs, unrolled, and the overflow is found from the bit lengths and the high half where they are ambiguous, against ruint's and bnum's; at 1,024 bits `wrapping_mul`, `widening_mul` and `pow` 1.52 to 1.75 are against GMP's `mpn_mullo_n`, `mpn_mul_n` and `mpz_pow_ui`, whose basecases are assembly. The cause of the rest was not established.
- **Small kernels** (shifts by a random amount 1.06 to 1.53, `leading_zeros` at 384 and 512 bits 1.18 to 1.31, `to_f64` 1.13 to 1.60, `isqrt` of one limb 2.41 against bnum and of 256 bits 1.08 against GMP, and one-limb `cmp`, `checked_add` and `Int::wrapping_abs` 1.08 to 1.15, a fraction of a nanosecond): generic loops over the limbs with branches on the amount or the length, where ruint and bnum specialise; not narrowed further.
- **Text of one and two limbs** (`to_string` 1.23 to 1.49 against `u64`, `u128`, `i64` and `i128`, `Int::from_str_radix(10)` of one limb 1.18, `ilog10` of two limbs 1.62), and **hexadecimal text** (`to_str_radix(16)` 1.20 to 1.98 against GMP into a buffer and bnum, `from_str_radix(16)` 1.15 to 1.17): at one and two limbs `Display` is `u64`'s and `u128`'s own, and the cause of the difference was not established; the hexadecimal digits are read off the bits a limb at a time into a `String`, against bnum and GMP's `mpz_get_str` into a buffer: not narrowed further.
- **Montgomery products from 512 bits and the powers on them** (`Montgomery::mul` 1.11 to 1.32, `square` 1.36 to 1.82, `Montgomery::pow` 1.22 to 1.51 and `Uint::pow_mod` of odd moduli 1.22 to 1.37, against OpenSSL): below 11 limbs the products are the plain rows of CIOS in portable Rust and `square` is a product, against OpenSSL's ARMv8 assembly (`bn_mul_mont`, and from 512 bits its 4-way and 8-way kernels and squaring); from 11 limbs `Number`'s schoolbook product and square kernels with a separate reduction. Matching OpenSSL there would take assembly of its quality. At 256 and 384 bits `mul`, `square` and `pow` take 1.07 to 1.08 of ruint's and OpenSSL's time, and `pow_mod` at 64 bits (1.13) and of even moduli at 512 bits (1.11) behind GMP's `mpz_powm`.
  **Modular inverses to 512 bits** (`Uint::mod_inverse` 1.64 to 2.92, against malachite-nz at 64 bits and ruint beyond): Pornin's binary GCD, 31 steps at a time on approximations, takes about `2·64·N/31` batches, where ruint's Lehmer steps and malachite-nz's extended Euclid on words take fewer; from 1,024 bits `a` is ahead. **`mul_mod` from 2,048 bits** (1.37 to 1.66 against GMP's `mpn_mul_n` and `mpn_tdiv_qr`): a schoolbook product and the division of algorithm D, where GMP multiplies by Karatsuba and Toom and divides by divide and conquer. **`add_mod` at 4,096 bits** (1.07 against malachite-nz): not established.

