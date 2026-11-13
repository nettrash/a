[A](../../../README.md) › [Math](../../../README.md#math)

# Arbitrary-precision floats

`BigFloat` (in `a::math::float`) is a binary floating-point number of any precision: NaN, an infinity, a signed zero, or `±m·2^e` with an integer mantissa `m` of the precision `p` in bits, `2^(p−1) ≤ m < 2^p`, chosen for each value. Every operation is correctly rounded: its result is the exact result rounded once to the precision asked for, in one of the four modes of `Round`.

- `Round::Nearest` (ties to even), `Round::TowardZero`, `Round::Up` (toward +∞) and `Round::Down` (toward −∞).
- The methods whose names end in `_round` take the precision and the mode of their result and return it with the direction of the rounding, as MPFR's ternary value: `Ordering::Less` when the result is below the exact one, `Equal` when it is exact, `Greater` when it is above. `add_round`, `sub_round`, `mul_round`, `div_round`, `mul_add_round` (a fused multiply-add, rounded once) and `sqrt_round`.
- The operators `+`, `-`, `*` and `/`, each operand owned or borrowed, and their assigning forms, round to nearest at the larger precision of the operands; `sqrt` and `mul_add` at the precision of `self` (the largest of the three for `mul_add`). There is no global state: a `BigFloat` is `Send` and `Sync`.
- NaN, `±∞` and `±0` follow IEEE 754: `∞ − ∞`, `0·∞`, `0/0`, `∞/∞` and the square root of a number below zero are NaN, `x/0` is an infinity of the quotient's sign, and an exact zero sum is `+0`, or `−0` when rounding down, unless both terms are zeros of one sign. `==` and `<` compare by value, whatever the precisions: NaN is unordered and `−0 == +0`; `total_cmp` orders every value.
- The exponent `e` of the form `0.1…·2^e` lies in `BigFloat::EXP_MIN..=EXP_MAX`, `±(2^62 − 1)`; a result beyond is an infinity or the largest finite number, and one below zero or the smallest positive number, by the mode, as MPFR does with its exponent range. There are no subnormal numbers. Precisions are from 1 bit to `2^40`.
- `From` makes a float exactly from an `f64` (53 bits), an `f32` (24 bits), a primitive integer (the type's bits) or a `Number` (its bits); `from_number`, `from_parts` (`m·2^e`) and `from_rational` round to a precision; `Rational::try_from(&x)` gives the rational a finite float is, exactly. `mantissa_exponent` gives `m` and `e`, `exponent` the `e` of `0.1…·2^e`.
- `prec_round` and `set_prec` round to another precision; `with_prec` to nearest.
- `BigFloat::pi_round` and `BigFloat::ln2_round` give the constants, correctly rounded.

## Examples

```rust
use a::math::float::{BigFloat, Round};
use a::math::rational::Rational;
use core::cmp::Ordering;

// 1/3 to 10 bits, rounded down and up.
let one = BigFloat::from(1);
let three = BigFloat::from(3);
let (lo, o) = one.div_round(&three, 10, Round::Down);
assert_eq!(o, Ordering::Less);
let (hi, o) = one.div_round(&three, 10, Round::Up);
assert_eq!(o, Ordering::Greater);
let third = Rational::from(1) / 3u32;
assert!(Rational::try_from(&lo).unwrap() < third && third < Rational::try_from(&hi).unwrap());

// The operators round to nearest at the larger precision.
let x = BigFloat::from(2).with_prec(200);
let r = x.sqrt();
assert_eq!(r.prec(), 200);
assert_eq!((&r * &r).prec_round(53, Round::Nearest).0, BigFloat::from(2.0));

// IEEE 754's special values.
assert!((BigFloat::infinity(53) - BigFloat::infinity(53)).is_nan());
assert!((BigFloat::from(1) / BigFloat::neg_zero(53)).is_infinite());
assert_eq!(BigFloat::pi(53), BigFloat::from(core::f64::consts::PI));
```

The method:

- Every operation ends in one rounding of an exact magnitude `M·2^k`, or of one known only to lie strictly between `M·2^k` and `(M + 1)·2^k` (the sticky bit), `M` holding at least the bit below the last one kept: the bits dropped give the round bit and whether anything lies below it.
- A sum is exact, on at most the precision and the operands' lengths together, unless the smaller term lies below a quarter of the result's last place and below the larger term's last bit. Then it changes the result only as a sticky bit would: the sum lies strictly between the larger term and one unit more (or less) of a scale below both, and no number of the precision nor any halfway point lies there. A product is the exact product of the mantissas, the fused multiply-add that product added exactly, a quotient the integer quotient of the mantissas with two bits more than the precision and its remainder the sticky bit, and a square root the integer square root, with the exponent made even, likewise.
- The functions that can only be approximated use Ziv's strategy: the value is computed to a working precision with a bound of its error, both ends of the interval the bound gives are rounded, and the result is taken when they agree and it lies outside the interval, so that the direction is known as well; otherwise again at a working precision half as large again. π comes from the crate's table of its first 16,384 bits and Chudnovsky's series beyond, kept for later calls; `ln 2` from its series `Σ 1/(k·2^k)`.

## Measured

To come.
