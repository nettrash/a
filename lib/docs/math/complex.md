[A](../../../README.md) › [Math](../../../README.md#math)

# Complex numbers

`Complex<T>` (in `a::math::complex`) is the complex number `re + im·i`, its two parts public fields of the float type `T`, `f32` or `f64`. The trait `Float`, sealed, is what the two have in common, for code generic over both; [matrices](matrix.md) of complex numbers are built on it.

- `Complex::new(re, im)` builds one, `Complex::from(x)` a real one, and `Complex::from_polar(r, θ)` the one of absolute value `r` and argument `θ`. `Complex::ZERO`, `Complex::ONE` and `Complex::I` are constants.
- They add with `+` and `+=`, subtract with `-` and `-=`, multiply with `*` and `*=` and divide with `/` and `/=`, owned or borrowed, and take a float of their own type as an operand on either side: `z * 2.0`, `2.0 * z`, `1.0 / z`. A real operand multiplies both parts, so `2·∞` stays `∞ + 0i` rather than becoming a NaN through `0·∞`.
- `-z` negates, `conj` conjugates, `recip` is `1 / z`; `==` compares, `Default` is zero, `Sum` and `Product` take iterators of owned or borrowed numbers. `Display` writes `1.5-2i`, each part with the precision given, `{:.2}` making `1.50-2.00i`.
- `abs` is the absolute value, `norm_sqr` its square, cheaper but subject to overflow, `arg` the argument in `(−π, π]`, `to_polar` both.
- `exp`, `ln`, `sqrt`, `powi`, `powf` and `powc` are the principal values: `ln` has its imaginary part in `(−π, π]` and `sqrt` its real part not negative. Both have their branch cut along the negative real axis, where the sign of a zero imaginary part picks the side, as in C99's `<complex.h>`: `sqrt(−4 + 0i) = 2i` and `sqrt(−4 − 0i) = −2i`. `0⁰` is 1.
- `is_finite` and `is_nan` look at both parts.

## Examples

```rust
use a::math::complex::Complex;

let z = Complex::new(3.0, 4.0);
assert_eq!(z.abs(), 5.0);
assert_eq!(z * z.conj(), Complex::new(25.0, 0.0));
assert_eq!(z / Complex::new(0.0, 2.0), Complex::new(2.0, -1.5));
assert_eq!(z.sqrt(), Complex::new(2.0, 1.0));
assert_eq!(Complex::new(1.0, 1.0).powi(8), Complex::new(16.0, 0.0));
assert_eq!(format!("{}", z - 2.0 * Complex::I), "3+2i");

// The branch cut: the sign of zero picks the side.
assert_eq!(Complex::new(-4.0, 0.0).sqrt(), Complex::new(0.0, 2.0));
assert_eq!(Complex::new(-4.0, -0.0).sqrt(), Complex::new(0.0, -2.0));
assert_eq!(Complex::new(-1.0, 0.0).ln(), Complex::new(0.0, std::f64::consts::PI));

// Neither the squares of `abs` nor the quotient's intermediate values overflow.
let big = Complex::new(1e300_f64, 1e300);
assert_eq!(big / big, Complex::ONE);
assert!(big.abs().is_finite());
```

The method:

- A quotient is Smith's: the larger part `p` of the divisor divides the smaller `q`, `r = q/p`, and `(a + b·r)/(p + q·r)` and `(b − a·r)/(p + q·r)`, with the parts of the dividend in the order that the case asks, so that nothing overflows where the quotient doesn't. Which part is the larger is chosen by selects, not a branch, which random divisors would mispredict half the time, and so is `r = 0` for a zero divisor, which leaves the parts divided by zero, as floats are. The sums are at most twice the largest part, so all four parts are halved first when one of them is at least half the largest float, which leaves the quotient as it was: `(1e308 + 1e308·i) / (1e308 + 1e308·i)` is 1, not a NaN from `∞ / ∞`. With no branch at all, a quotient took 1.4 ns, against 1.6 with a branch for the zero divisor; the halving costs 0.3 ns of it.
- `abs` is `hypot`. `sqrt` is Kahan's: `t = √((|re| + |z|)/2)`, and the root `t + i·im/(2t)` for `re ≥ 0`, `|im|/(2t) ± i·t` otherwise, which cancels nothing, with the operand scaled by a power of 4 when its parts are near the subnormals or near overflow, and C99's values for infinities and NaNs.
- `ln` takes its real part as `ln(1 + (|z|² − 1))/2` near the unit circle, `(|z| − 1)(|z| + 1) + small²` with the larger part `|z|` of the two, so that `ln |1 + 10⁻¹⁰i| = 5·10⁻²¹` keeps its digits where `ln |z|` would round to 0; `ln |z|` elsewhere, but `ln(big) + ln(1 + (small/big)²)/2` where `|z|` overflows or is subnormal, so that `ln(MAX + MAX·i)` is finite.
- `exp` keeps a real number real, with the sign of its zero imaginary part; `powi` multiplies by repeated squaring, of the reciprocal for a negative power; `powf` and `powc` are `exp(w·ln z)`.

Measured on an Apple M4 Max for `Complex<f64>`, over 1,024 independent operations on random parts in `[−2, 2)`, each result stored, against num-complex 0.4.6 on the same values (the median of 21 interleaved runs, in nanoseconds per operation):

| Operation | `Complex` | num-complex |
| --- | --- | --- |
| `a * b` | 0.42 | 0.40 |
| `a / b` | 1.41 | 0.80 |
| `abs` | 1.18 | 1.21 |
| `sqrt` | 4.19 | 8.84 |
| `exp` | 5.00 | 5.27 |
| `ln` | 7.38 | 7.21 |
| `powc` | 14.93 | 14.94 |

- num-complex divides by `|b|²` as it is, so that its quotients overflow from `|b|` of about 10¹⁵⁴ and lose their digits below 10⁻¹⁵⁴, where Smith's method with the halving keeps them, at about 0.6 ns a quotient.
- `sqrt` takes half the time: a `hypot`, a square root and a division, where num-complex goes through the polar form, a `hypot`, an `atan2`, a square root, a sine and a cosine.
- `ln` takes 2 % longer for the digits it keeps near the unit circle, and `powc` as long.

## Efficiency

- Time complexity: O(1) for each operation
- Space complexity: O(1), two floats
