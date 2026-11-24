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
- **Products** are schoolbook, as in the other fixed-width libraries at these sizes: `widening_mul` and `carrying_mul` take the `N²` limb products of the full `2N`-limb product; `wrapping_mul`, `overflowing_mul` and the operator only the products that land below the width, at most `N(N + 1)/2`, over the significant limbs, and the product overflows exactly when a pair of nonzero limbs or a row's carry lands at or above it.
- **Quotients** are Knuth's algorithm D (*The Art of Computer Programming*, vol. 2, 3rd ed., 4.3.1): the divisor's `n` significant limbs shifted until its top bit is set and the dividend as far (D1); each quotient limb estimated from the top two limbs of the remainder over the top limb of the divisor, then lowered while the second limb of the divisor shows it too large, or while it is `2⁶⁴` or more (D3), which leaves it at most one too large; the divisor times it subtracted (D4), and added back once when that went below zero (D6, on random operands with a probability of about `2/2⁶⁴`). `O(n·m)` limb products for a quotient of `m` limbs. A divisor of one limb takes the plain loop of two-limb by one-limb divisions from the top, which is also `div_rem_u64`. `Int` divides the absolute values and restores the signs.
- **Powers** square and multiply from the low bit of the exponent, `O(N² log e)`; the overflow flag is exact, as every partial product of a base of at least 2 divides the power.
- **Square roots** are Newton's iteration `x ↦ ⌊(x + ⌊a/x⌋)/2⌋` from `2^⌈b/2⌉`, above the root for `a` of `b` bits, which decreases to `⌊√a⌋` (Brent and Zimmermann, *Modern Computer Arithmetic*, 1.5.1): `O(log b)` divisions.
- **Logarithms**: `ilog2` from the highest set bit; `ilog10` and `ilog` to a base of one limb divide by the largest power of the base that fits in a limb, counting its digits at a time, and take the logarithm of the last limb, `O(N²)` in all; a longer base is divided by itself, at most `BITS/64` times.
- **Text** in a radix that is a power of two is read off the bits; in another radix the value is divided by the largest power of the radix that fits in a limb, one limb of digits at a time, `O(N²)`, and read back by products with that power. The digits go through `Formatter::pad_integral`, which applies the flags `+`, `#`, `0`, the width, the fill and the alignment as for the primitives: `{:#066x}` of a `Uint<4>` is `0x` and 64 digits. An `Int` is written in decimal and by `to_str_radix` as its sign and the digits of its absolute value, and in hexadecimal, octal and binary as the bits of its two's complement (`{:x}` of −1 is all `f`), as `i64` is. `Debug` writes hexadecimal, `0x` and the significant digits, with a `-` and the absolute value for a negative `Int`: limbs are easier to see that way than in decimal, which is what `Display` gives. It takes the width, fill, alignment and the flags `+` and `0` as `{:#x}` takes them, the zeros after the `0x`: `{:>8?}` of 255 is `    0xff` and `{:08?}` is `0x0000ff`.
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
- Nothing is tuned yet. Products are schoolbook and quotients Knuth's algorithm D, with no Karatsuba, no reciprocal for the quotient limbs, and the one-limb loop dividing with the two-limb hardware or library division.
- `N = 0` is a type with the one value 0: sums, products, shifts, text, `Number` conversions and the bytes work on it, but the constants and methods that need a bit (`ONE`, `Int::MIN` and `Int::MAX`, `pow`, `Product`, …) fail to compile for it, and it is not tested beyond that.
- `Debug` is hexadecimal, and while it takes the width, fill, alignment, `+` and `0`, it cannot tell `{:x?}` and `{:X?}` from `{:?}` (stable Rust does not let a type ask), so `{:X?}` is lower-case too; the primitives' `Debug` is decimal.
- The byte arrays need their length named or inferred, `[u8; 32]` for `Uint<4>`: `[u8; 8·N]` cannot be written in code generic over `N` on stable Rust, which uses `write_le_bytes` and `from_le_slice` instead.
- With a shift amount of an untyped literal, a method on the result needs the literal's type (above).
- Overflow follows `debug_assertions` of this crate as it is built, not `overflow-checks`; a profile that sets them apart gives the primitives' behaviour of one and this crate's of the other.
- `div_ceil`, `next_multiple_of` and the other methods that are unstable for the primitives' signed types are not on `Int`.

## Montgomery multiplication and pow_mod

`Montgomery<N>` is the arithmetic modulo an odd `m > 1` of `N` limbs in Montgomery form, with `R = 2^(64·N)`: a number `x` is kept as its form `x·R mod m`, a `Uint<N>` below `m`, and products of forms are reduced by dividing by `R` instead of by `m`. It is computed once for a modulus (it is `Copy`) and then used for as many operations as needed. On top of it, `Uint<N>` has `pow_mod`, `mul_mod`, `add_mod`, `sub_mod` and `mod_inverse` for **any** nonzero modulus, odd or even. The issue's target is 256- to 512-bit moduli (`N` = 4 to 8); every `N ≥ 1` works, and the tests run every width from 1 to 9 limbs and 16, 33 and 129.

Every argument may be any `Uint<N>` and stands for its residue modulo `m` (one at or above `m` is reduced first), and every result is below `m`. Exponents are a `Uint<M>` of any width `M`, the same as `N` or not, or for `pow_limbs` a slice of limbs, the low one first (`&[e]` for a `u64`). `x⁰ = 1` for every `x`, 0 included, and modulo 1 every result is 0, as for `Number::pow_mod`; every number has the inverse 0 modulo 1, as for `Number::mod_inverse` and GMP's `mpz_invert`. A modulus of zero panics with the message of `%`; `Montgomery::new` panics for an even modulus, for 1, and for `N = 0`.

| Function | Gives | Method | Cost |
| --- | --- | --- | --- |
| `Montgomery::new(&m)` | the arithmetic modulo the odd `m > 1` | `−m⁻¹ mod 2⁶⁴` by Newton's iteration; `R mod m` and `R² mod m` by Knuth's algorithm D | `O(N²)` |
| `modulus()`, `one()` | `m`; the form of 1, `R mod m` | | |
| `to_montgomery(&x)` | the form `x·R mod m` of any `x` | REDC of `x·(R² mod m)` | one product |
| `from_montgomery(&x)` | the number `x·R⁻¹ mod m` a form stands for | REDC of `x` | one product |
| `mul(&a, &b)`, `square(&a)` | `a·b·R⁻¹ mod m`, the form of the product | REDC interleaved with the product (CIOS) | `2·N² + N` products of limbs |
| `add`, `sub`, `neg` | the forms of `a + b`, `a − b`, `−a` | one conditional subtraction or addition of `m` | `O(N)` |
| `pow(&x, &e)`, `pow_limbs(&x, &[u64])` | the form of `x^e` | sliding window of 1 to 6 bits, sized to the exponent | about `ℓ` squarings and `ℓ/(w + 1)` products for `ℓ` bits, and `2^(w−1)` for the table when `w ≥ 2` |
| `inverse(&x)` | the form of `x⁻¹`, or `None` when `gcd(x, m) ≠ 1` | binary inversion, then two products by `R² mod m` | `O(N²)` |
| `x.pow_mod(&e, &m)` | `x^e mod m` for any nonzero `m` | odd `m`: `Montgomery`; even `m = 2ᵏ·q`: modulo `q` in Montgomery form, modulo `2ᵏ` by wrapping products, put together by the Chinese remainder theorem | `O(N²·ℓ)`; for an even `m` another `O(N²·log N)` for `q⁻¹ mod 2^(64·N)` |
| `x.mul_mod(&y, &m)` | `x·y mod m` for any nonzero `m` | the double-width product, reduced by algorithm D | `O(N²)` |
| `x.add_mod(&y, &m)`, `x.sub_mod(&y, &m)` | `x ± y mod m` for any nonzero `m` | operands at or above `m` reduced by algorithm D, then one conditional correction | `O(N)` for reduced operands |
| `x.mod_inverse(&m)` | the `r < m` with `x·r ≡ 1`, or `None` | odd `m`: binary inversion; even `m = 2ᵏ·q`: Newton's iteration modulo `2ᵏ`, binary inversion modulo `q`, the Chinese remainder theorem | `O(N²)`; `O(N²·log N)` for an even `m` |

### Methods

- **REDC, CIOS.** Montgomery's reduction (P. L. Montgomery, *Modular multiplication without trial division*, Math. Comp. 44, 1985) adds to `t` the multiple `q·m` with `q = t₀·(−m⁻¹) mod 2⁶⁴`, which clears the low limb, and shifts the limb out; after `N` limbs `t ≡ a·b·R⁻¹ (mod m)`. The product and the reduction are interleaved a limb of `b` at a time, the method CIOS of Koç, Acar and Kaliski (*Analyzing and comparing Montgomery multiplication algorithms*, IEEE Micro 16(3), 1996): a row `t += a·bᵢ`, a row `t = (t + q·m)/2⁶⁴`, `2·N² + N` products of limbs in all. Between the rows `t < a + m < 2·R`, so `t` is kept in `N` limbs and a carry; that carry is what makes a modulus near `R` (its top limb `u64::MAX`, up to `R − 1` itself) come out right. When `a·b < m·R`, which holds as soon as one factor is below `m`, the end result is below `2·m`, and one conditional subtraction leaves it below `m`. `square` is `mul` of a number by itself for now.
- **The constants.** `−m⁻¹ mod 2⁶⁴` by Newton's iteration `y ← y·(2 − m·y)` from `y = m`, which is right to 3 bits and doubles them each step: five steps. `R mod m` and `R² mod m` are the remainders of `2^(64·N)` and `2^(128·N)` by Knuth's algorithm D (*The Art of Computer Programming*, vol. 2, 4.3.1), normalised so that the divisor's top bit is set, and taken one quotient limb at a time from the top of the dividend, so that the dividend needs no room: the quotient limb is estimated from the top two limbs and the divisor's top limb, corrected by the next limbs of each (step D3), and is then right or one too large (Theorem B), which the add-back of step D6 corrects. The same division reduces the double-width products of `mul_mod` and the operands of `add_mod`, `sub_mod`, `mod_inverse`. These divisions are private to the module; they may later give way to the division of `Uint`.
- **Powers.** The sliding-window method (Menezes, van Oorschot and Vanstone, *Handbook of Applied Cryptography*, 1996, Algorithm 14.85): from the top bit down a zero bit costs a squaring, and a window of up to `w` bits that starts and ends with a one costs its length in squarings and one product by an odd power from a table `x, x³, …, x^(2^w − 1)`. For an exponent of `ℓ` bits, `w` minimises `T(w) + ℓ/(w + 1)`, about `ℓ/(w + 1)` windows and the cost `T(w)` of the table: 0 for `w = 1`, which needs no table, and `2^(w−1)` for `w ≥ 2` (a squaring for `x²` and `2^(w−1) − 1` products), a tie going to the smaller `w`. That gives 1 up to 12 bits, 2 up to 24, 3 up to 80, 4 up to 240, 5 up to 672, 6 beyond (checked against the code for every length up to 20,000 bits). The table is on the stack: from 33 limbs on `w` is at most 4, from 129 limbs on at most 2, which keeps it within 8 KiB up to 512 limbs. Counted on 200 random exponents of each length, the products came to `ℓ/(w + 1) + 2^(w−1)` less 1.5 on average (57.2 for 256 bits, 99.7 for 512, 176.7 for 1024, where binary exponentiation takes 128, 256 and 512), and the squarings to `ℓ` less 1 to 4.
- **Even moduli.** `m = 2ᵏ·q` with `q` odd: `x^e mod q` in Montgomery form, `x^e mod 2^(64·N)` by wrapping products (the low halves) and its low `k` bits, put together in Garner's form `y = r₁ + q·((r₂ − r₁)·q⁻¹ mod 2ᵏ)`, which is below `m` and so needs no reduction. `q⁻¹ mod 2^(64·N)` is Newton's iteration again, from the inverse of the low limb, `⌈log₂ N⌉` steps of two wrapping products, `O(N²·log N)` in all, which is what an even modulus adds to `pow_mod` and `mod_inverse`. A power of two (`q = 1`) is the wrapping power alone. `mod_inverse` splits the same way: an inverse modulo `2ᵏ` exists exactly for odd numbers.
- **Inverses.** The binary algorithm of Hankerson, Menezes and Vanstone (*Guide to Elliptic Curve Cryptography*, 2004, Algorithm 2.22), which needs an odd modulus but not a prime one: it keeps `x₁·x ≡ u` and `x₂·x ≡ v (mod m)` from `u = x, v = m`, halves the even one of `u` and `v` and its coefficient modulo `m`, and subtracts the smaller of the two odd ones from the larger, until one of them is 1, whose coefficient is the inverse, or they meet at a common divisor above 1, when there is none. Each step shortens `u·v` by a bit or more: at most `2·64·N` steps of `O(N)`. `Montgomery::inverse` inverts the form `x·R` itself and multiplies by `R² mod m` twice, which takes `(x·R)⁻¹` to `x⁻¹·R`.

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
- `square` is a product (no separate squaring kernel yet), the products are the plain rows of CIOS, with no assembly and no SIMD, and nothing is tuned: the release will time it and may reuse the register kernels of `a::math::ec`'s internal Montgomery arithmetic.
- The values and the table of a power are on the stack; for very wide `Uint`s (thousands of limbs) that is a lot of stack, beyond the window's limit of two powers from 129 limbs on.
- `N = 0` holds no odd number and no nonzero modulus: `Montgomery::new` panics, and so do `pow_mod`, `mul_mod`, `add_mod`, `sub_mod` and `mod_inverse` on `Uint<0>`, whose only modulus is 0.
