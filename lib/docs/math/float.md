[A](../../../README.md) › [Math](../../../README.md#math)

# Arbitrary-precision floats

`BigFloat` (in `a::math::float`) is a binary floating-point number of any precision: NaN, an infinity, a signed zero, or `±m·2^e` with an integer mantissa `m` of the precision `p` in bits, `2^(p−1) ≤ m < 2^p`, chosen for each value. Every operation is correctly rounded: its result is the exact result rounded once to the precision asked for, in one of the four modes of `Round`. `BigComplex` is a complex number of two such parts, as GNU MPC's `mpc_t`, each part of every result rounded once in a mode of its own.

- `Round::Nearest` (ties to even), `Round::TowardZero`, `Round::Up` (toward +∞) and `Round::Down` (toward −∞).
- The methods whose names end in `_round` take the precision and the mode of their result and return it with the direction of the rounding, as MPFR's ternary value: `Ordering::Less` when the result is below the exact one, `Equal` when it is exact, `Greater` when it is above. Most have a form without `_round` that rounds to nearest at the precision of `self`: `sqrt`, `exp`, `sin`, `cbrt` and the rest.
- NaN, `±∞` and `±0` follow IEEE 754: `∞ − ∞`, `0·∞`, `0/0`, `∞/∞` and the square root of a number below zero are NaN, `x/0` is an infinity of the quotient's sign, and an exact zero sum is `+0`, or `−0` when rounding down, unless both terms are zeros of one sign. `BigFloat::nan`, `infinity`, `neg_infinity`, `zero` and `neg_zero` make them at a precision; `is_nan`, `is_infinite`, `is_finite`, `is_zero`, `is_normal`, `is_sign_negative` and `is_sign_positive` tell them apart. `==` and `<` compare by value, whatever the precisions: NaN is unordered and `−0 == +0`; `total_cmp` orders every value, and `cmp_abs` compares magnitudes.
- The exponent `e` of the form `0.1…·2^e` lies in `BigFloat::EXP_MIN..=EXP_MAX`, `±(2^62 − 1)`; a result beyond is an infinity or the largest finite number, and one below zero or the smallest positive number, by the mode, as MPFR does with its exponent range. There are no subnormal numbers. Precisions are from `BigFloat::MIN_PREC`, 1 bit, to `MAX_PREC`, `2^40`. There is no global state: the precision and the mode are given to each operation, and a `BigFloat` is `Send` and `Sync`.
- `From` makes a float exactly from an `f64` (53 bits), an `f32` (24 bits), a primitive integer (the type's bits) or a `Number` (its bits); `from_number`, `from_parts` (`m·2^e`) and `from_rational` round to a precision. `prec` gives the precision, `mantissa_exponent` `m` and `e`, `exponent` the `e` of `0.1…·2^e`. `prec_round` and `set_prec` round to another precision, `with_prec` to nearest; `-x`, `abs` and `copysign` are exact.
- The arithmetic, roots and powers, conversions, text, the exponentials and logarithms, the hyperbolic and circular functions and the constants are below; then `BigComplex` and its functions, and `BigFloat` and `BigComplex` as elements of matrices. The methods are after the examples, and the places where this module and MPFR or MPC differ, by choice or by an error found in them, after the methods.

## Arithmetic

- `add_round`, `sub_round`, `mul_round`, `div_round`, `mul_add_round` (a fused multiply-add, rounded once) and `sqrt_round` return a new number. `set_add`, `set_sub`, `set_mul`, `set_div`, `set_mul_add` and `set_sqrt` write the result into an existing number, at its precision, in the vector of its mantissa, as MPFR's `mpfr_add (r, a, b, rnd)`, and return the direction; `add_assign_round`, `sub_assign_round`, `mul_assign_round`, `div_assign_round` and `sqrt_assign_round` change a number by an operand, as `mpfr_add (x, x, y, rnd)`.
- The operators `+`, `-`, `*` and `/`, each operand owned or borrowed, and their assigning forms, round to nearest at the larger precision of the operands; `sqrt` and `mul_add` at the precision of `self` (the largest of the three for `mul_add`). The assigning operators `+=`, `-=`, `*=` and `/=` write into the vector of the number they change.
- The operators take a primitive number (`i8` to `i128`, `u8` to `u128`, `isize`, `usize`, `f32`, `f64`) or a `Number` on either side, owned or borrowed, and their assigning forms take one on the right: `&x + 1`, `2.0 * &x`, `&x / &n`, `x -= 3u8`. The other operand is made a float exactly (an integer at the bits of its type, a float at its own precision, a `Number` at its bits), and the result is rounded to nearest at the precision of the `BigFloat`, once. A primitive operand is made in a float that each thread keeps for it, so that it allocates nothing.
- `BigFloat::sum_round(&terms, prec, round)` and `BigFloat::dot_round(&xs, &ys, prec, round)` give the exact sum of any number of terms and the exact dot product of two slices of one length, rounded once, as MPFR's `mpfr_sum` and `mpfr_dot`, with `mpfr_sum`'s zeros and special values.

## Roots and powers

- `rec_sqrt_round` is `1/√x`, as MPFR's `mpfr_rec_sqrt`: `+∞` for `±0`, `+0` for `+∞`, NaN below zero.
- `nth_root_round(n, prec, round)` is the `n`-th root for a `u32` `n`, as MPFR's `mpfr_rootn_ui` and IEEE 754's `rootn`: NaN for `n = 0`; an odd root of a negative number is negative, an even one NaN; the root of a zero is that zero for an odd `n` and `+0` for an even one. `cbrt_round` is the cube root.
- `powi_round(n, prec, round)` is `x^n` for an `i64` (MPFR's `mpfr_pow_si`) and `pow_number_round(&n, prec, round)` for a `Number` of any size (`mpfr_pow_z`): `x^0 = 1` for every `x`, NaN included; `(±0)^n` is `±0` for an odd `n > 0`, `+0` for an even one, `±∞` for an odd `n < 0` and `+∞` for an even one; `(±∞)^n` the other way round. A power beyond the exponent range overflows or underflows by the mode, found from `n·log2|x|` without computing the power, so `3^(2^62)` costs no more than `3^7`.
- `pow_rational_round(&r, prec, round)` is `x^(p/q)` for a `Rational` `p/q` in lowest terms: the `q`-th root as `nth_root_round` takes it, to the power `p` as `pow_number_round` takes it, rounded once. A negative number has a power for an odd `q` (negative for an odd `p`), NaN for an even one.
- The forms to nearest at the precision of `self` are `rec_sqrt`, `nth_root`, `cbrt`, `powi`, `pow_number` and `pow_rational`; `pow`, a float to a float's power, is with the exponentials [below](#exponentials-logarithms-and-powers).

## Conversions

- `to_f64_round(round)` and `to_f32_round(round)` give the primitive float nearest in the mode, and the direction, as `mpfr_get_d` and `mpfr_get_flt`: correctly rounded among the subnormals too, past the largest finite value an infinity (or the largest, when the mode rounds toward zero), NaN as NaN and the zeros with their signs; `to_f64()` and `to_f32()` round to nearest. `from_f64_round(d, prec, round)` and `from_f32_round` round a primitive float to a precision (`mpfr_set_d`), and `set_f64(d, round)` and `set_f32` write it into an existing number at its precision, in the vector of its mantissa, as `mpfr_set_d (r, d, rnd)` does; `From<f64>` and `From<f32>` stay exact.
- `floor`, `ceil`, `trunc`, `round` (halfway cases away from zero) and `round_ties_even` give the integer at the number's precision, which holds it, as `f64`'s methods; a zero result has the sign of the number, so `ceil(−0.5)` is `−0`. `floor_round(prec, round)`, `ceil_round`, `trunc_round`, `round_round` and `round_ties_even_round` round that integer once more, with the direction from the integer (`mpfr_rint_floor` and the others); `rint_round(prec, round)` is `mpfr_rint`, the integer of `prec` bits nearest in the mode's direction, rounded once (10.5 to 2 bits is 12). `set_rint(&x, round)` writes `x.rint_round(p, round)` into an existing number of the precision `p`, in the vector of its mantissa, and `set_floor`, `set_ceil` and `set_trunc` are it with `Round::Down`, `Round::Up` and `Round::TowardZero`, as MPFR's `mpfr_floor`, `mpfr_ceil` and `mpfr_trunc`; all return the direction from `x`. `fract` and `fract_round` give the fractional part of the number's sign, a zero of its sign for an integer and, as MPFR has it, for an infinity; `is_integer` says whether there is none.
- `to_number_round(round)` gives the integer that the number rounds to and the direction (`mpfr_get_z`), `None` for NaN and the infinities. `to_i64_round`, `to_u64_round`, `to_i128_round` and `to_u128_round` saturate as the `as` casts of the primitive floats do and as `mpfr_get_si` does: past the type's range, an infinity included, its largest or least value, with a direction against the number; `None` for NaN. `TryFrom<&BigFloat>` for `Number` and the primitive integers is exact: a `TryFromBigFloatError` for NaN, an infinity, a number that is not an integer or one out of range.
- `Rational::try_from(&x)` gives the rational a finite float is, exactly, and `BigFloat::from_rational(&r, prec, round)` rounds a rational once.
- A float compares exactly with the primitive integers and floats, `Number`s and `Rational`s, `x == 1`, `x < 0.5`, `x > n`, `x < &r`, and a `Number` or a `Rational` with a float, `r < x`: NaN is unordered and `−0 == +0`. The primitives have no comparisons of their own with a float (`1 < x`), which would leave the types of literals in code such as `v == vec![]` ambiguous.

## Text

- `Display` writes the shortest decimal that reads back to nearest at the number's precision as the number itself, the nearest to it of those, as `f64` does: `BigFloat::from(0.1)` is `0.1`, and the text of any `f64` is that of the `f64`. Where positional notation would add zeros past the digits, for a number of `10^21` or more or below `10^−5`, it is scientific, `1.5e300`. A formatter's precision, `{:.3}`, gives that many digits after the point, rounded to nearest from the exact value, as for `f64`; width, fill, alignment, `+` and `0` are as for numbers. `LowerExp` and `UpperExp` write `1.2345e3` and `1.2345E3`, the shortest digits or a precision's. `Debug` writes the precision and the value in hexadecimal, `BigFloat { prec: 53, value: 0x3p-1 }`.
- `to_digits(radix, digits, round)` gives the digits in a radix from 2 to 36 and the exponent `e` of `0.d₁d₂…·radix^e`, with the direction, as `mpfr_get_str`; with `digits` of `None`, `1 + ceil(p·log_radix 2)` of them (`p − 1` for a power of two), as many as any number of the precision needs to be read back (`mpfr_get_str_ndigits`). To nearest, a tie goes to the digits that make an even integer. `to_string_radix(radix, digits, round)` writes them in scientific notation, `-1.25e-3`, with `@` for the exponent above radix 10 (`f.fc@2`), and `nan`, `inf`, `-inf`, `0`, `-0` (`@nan@`, `@inf@` above radix 16); the text always reads back with `from_str_radix`.
- `BigFloat::from_str_radix(text, radix, prec, round)` reads a number and rounds it once to `prec` bits, with the direction, as `mpfr_strtofr`: a sign, digits with a point, an exponent `e` (radix 10 and below) or `@` (any radix) of the radix, or `p` of two in radix 2 and 16; the prefixes `0x` and `0b`, and radix 0 for the radix they give, 10 without one; `inf`, `infinity`, `nan` and `nan(…)` up to radix 16, `@inf@` and `@nan@` in any, in either case; no whitespace. Any number of digits; an exponent too large or too small for the exponent range overflows or underflows by the mode without the work growing with it (`1e999999999999999999999` is an infinity or the largest number at once).
- `FromStr` reads radix 0 to nearest: at 53 bits for up to 17 significant decimal digits (those of any `f64` written to be read back, so `"0.1"` is the `f64` 0.1), at `ceil(n·log2 10)` bits for `n` more, which keeps them (`"3.14159265358979323846264338327950288"` is read to 120 bits), and for hexadecimal or binary digits at their bits, at least 53, which holds them exactly. A `ParseBigFloatError` gives the kind of error, `BigFloatErrorKind::Empty` or `InvalidDigit`.

## Exponentials, logarithms and powers

- `exp_round`, `exp2_round`, `exp10_round` and `exp_m1_round` (`e^x − 1`, accurate near zero, as MPFR's `mpfr_expm1`), and `exp`, `exp2`, `exp10`, `exp_m1` to nearest. `e^0 = 1`, `2^n` for an integer `n` and `10^n` for an integer `n ≥ 0` whose `5^n` fits the precision are exact; `e^+∞ = +∞`, `e^−∞ = +0`, `e^−∞ − 1 = −1`, `±0` for `e^±0 − 1`.
- `ln_round`, `log2_round`, `log10_round` and `ln_1p_round` (`ln(1 + x)`, accurate near zero, as MPFR's `mpfr_log1p`), and `ln`, `log2`, `log10`, `ln_1p` to nearest. `ln 1 = +0` in every mode, `ln ±0 = −∞`, `ln +∞ = +∞`, NaN below zero; `log2(2^k) = k` and `log10(10^k) = k` exactly; `ln_1p(−1) = −∞`, `ln_1p(±0) = ±0`, NaN below `−1`.
- `pow_round(&y, prec, round)` and `pow(&y)` are `x^y` as MPFR's `mpfr_pow`, with the special values of C99's Annex F and IEEE 754: `x^±0 = 1` and `1^y = 1` for any `x` and `y`, NaN included; NaN for a finite `x < 0` and a finite `y` that is not an integer; `(−1)^±∞ = 1`; `x^±∞`, `±0^y` and `±∞^y` by the magnitude of `x` and the sign and parity of `y`. Exact results are found before any approximation, `4^0.5 = 2`, `2^−3`, `(−3)^3`, `9^1.5 = 27`, and a midpoint of the precision among them is rounded as an exact number.
- A result beyond the exponent range overflows or underflows by the mode, at once: `exp` of anything from `2^62` up, and `x^y` when `|y ln x| ≥ 2^63`, from the exponents alone.

## Hyperbolic functions

- `sinh_round`, `cosh_round`, `tanh_round`, `asinh_round`, `acosh_round` and `atanh_round`, `sinh_cosh_round` for both from one exponential, and the forms to nearest `sinh`, `cosh`, `tanh`, `asinh`, `acosh` and `atanh`.
- `±0` and `±∞` give themselves for the odd functions, `cosh ±0 = 1`, `cosh ±∞ = +∞`, `tanh ±∞ = ±1`; `acosh 1 = +0`, NaN below 1; `atanh ±1 = ±∞`, NaN beyond.
- `cosh` and `sinh` just below the overflow threshold are finite where `e^|x|` itself overflows.

## Circular functions

- `sin_round`, `cos_round`, `tan_round`, `sec_round` (`1/cos x`), `csc_round` (`1/sin x`), `cot_round` (`1/tan x`), `atan_round`, `asin_round` and `acos_round`, and `sin`, `cos`, `tan`, `sec`, `csc`, `cot`, `atan`, `asin`, `acos` to nearest. `sin_cos_round(prec, round)` gives both the sine and the cosine, each with its direction, from one reduction of the argument; `sin_cos` to nearest.
- `y.atan2_round(&x, prec, round)` is the angle of the point `(x, y)`, in `[−π, π]`; `y.atan2(&x)` to nearest at the precision of `y`.
- NaN and the infinities give NaN, but `atan(±∞) = ±π/2` rounded; at `±0` the sine, the tangent, the arctangent and the arcsine are `±0`, the cosine and the secant 1, the cosecant and the cotangent `±∞`, the arccosine `π/2` rounded. `asin` and `acos` are NaN beyond `[−1, 1]`, `asin(±1) = ±π/2` and `acos(−1) = π` rounded, `acos(1) = +0` exactly.
- `atan2` has the whole table of C99's Annex F, the sign of the result that of `y`: `atan2(±0, x)` is `±0` for `x = +0` or above and `±π` for `x = −0` or below; `atan2(y, ±0) = ±π/2` for `y` not zero; `atan2(±∞, +∞) = ±π/4`, `atan2(±∞, −∞) = ±3π/4`, `atan2(±∞, x) = ±π/2` for a finite `x`; `atan2(y, +∞) = ±0` and `atan2(y, −∞) = ±π` for a finite `y` not zero; NaN when either is NaN.
- A result beyond the exponent range overflows or underflows by the mode: `csc` of the smallest number is `+∞` to nearest and the largest number toward zero, `sin` of it the smallest number to nearest and `+0` toward zero.
- The argument may be as large as the exponent range allows: the reduction takes π to as many bits as the exponent of the argument and the precision need, so its time and memory grow with the exponent (π to `10^6` bits for an argument of the exponent `10^6`).

## Constants

- `BigFloat::pi_round(prec, round)`, `e_round`, `ln2_round`, `ln10_round`, `euler_gamma_round` (Euler's `γ = 0.5772…`) and `catalan_round` (Catalan's `G = 1 − 1/9 + 1/25 − … = 0.9159…`) give the constants correctly rounded to `prec` bits in the mode `round`, with the direction of the rounding, never `Equal`: they are irrational. `BigFloat::pi(prec)`, `e`, `ln2`, `ln10`, `euler_gamma` and `catalan` round to nearest.
- Up to 16,384 bits a constant is cut from a table of its bits; beyond, it is computed once and kept for the process, so that a later call at the same or a smaller precision costs a copy of its limbs. The constants are shared by the threads (a `Mutex` around each), and each keeps only its most precise value, up to `2^27` bits (16 MiB) after the point: 96 MiB for the six at most. A longer value is computed at each call; computing it needs several times as much memory for a while.
- The functions take their constants from here: the reduction of the circular functions and the arguments of complex numbers `π`, the exponential and the logarithm `ln 2`, `exp10` and `log10` `ln 10`.

## Complex numbers

`BigComplex` (in `a::math::float`) is a complex number `re + im·i` whose parts are `BigFloat`s, each of its own precision, as GNU MPC's `mpc_t`. Every operation rounds each part of its exact result once, correctly, to a precision and in a mode of its own: the real and the imaginary part are two real results of the operation, as MPC defines them.

- `BigComplex::new(re, im)` takes the parts as they are; `From` makes a number exactly from a `BigFloat` (the imaginary part `+0` of its precision), a pair `(re, im)`, a `Complex<f64>` (53 bits) or `Complex<f32>` (24 bits), and the primitive floats and integers; `zero`, `nan` and `i` make `+0 + 0i`, `NaN + NaN·i` (what a new MPC variable holds) and `i` at a precision. `re`, `im`, `re_mut`, `im_mut`, `into_parts` and `prec` give the parts and their precisions; `is_nan`, `is_infinite`, `is_finite` and `is_zero` test them; `prec_round`, `prec_round_parts` and `with_prec` round to other precisions.
- The methods whose names end in `_round` take one precision and one mode for both parts and return the result with the directions of the two roundings, `(re, im)`, MPC's `MPC_INEX_RE` and `MPC_INEX_IM`; those ending in `_round_parts` take a pair of precisions and a pair of modes, the real part's first, as MPC's `mpc_rnd_t`. `add_round`, `sub_round`, `mul_round`, `sqr_round`, `div_round` and `sqrt_round`; `add_real_round`, `sub_real_round`, `mul_real_round` and `div_real_round` with a `BigFloat` operand (its imaginary part an exact zero, as MPC's `_fr` functions: `(1 − 0i) + (+0)` is `1 − 0i`), `mul_int_round` with an `i64`, and `mul_i_round`. `norm_round` (`|z|²`) and `abs_round` (`|z|`) give a `BigFloat` and its direction.
- The forms into an existing number, as MPC's functions write into their first argument, round each part at the precision of that part of `self` and reuse its vector: `set_add`, `set_sub`, `set_mul`, `set_sqr`, `set_div` and `set_sqrt`; `set_add_real`, `set_sub_real`, `set_mul_real` and `set_div_real`; `set_neg` and `set_conj` (exact when the precisions are at least the operand's). Each takes one mode for both parts and has a `_parts` form with a pair of modes (`set_mul_parts`, …), and returns the directions of the two roundings. `add_assign_round`, `sub_assign_round`, `mul_assign_round`, `div_assign_round` and `sqr_assign_round` set `self` to `self op other` (`self²`) at its precisions, the parts computed before either is written. `BigFloat::set_complex_norm(&z, round)` and `BigFloat::set_complex_abs(&z, round)` write `|z|²` and `|z|` into a float, as `mpc_norm` and `mpc_abs`. Their results are those of the `_round_parts` forms at those precisions, special values and signs of zeros included.
- `conj`, `-z` and `mul_i(negative)` (`z·i`, or `z·(−i)`) are exact and keep the precisions. `sqr`, `sqrt`, `norm` and `abs` round to nearest at the precisions of the parts (the larger for the norm and the absolute value); the operators `+`, `-`, `*` and `/`, each operand owned or borrowed, and their assigning forms, round each part to nearest at the larger precision of that part of the operands; an assigning operator whose left operand has the larger precisions writes into it. `==` compares both parts by value. `Display` writes `(re im)` as MPC's `mpc_out_str` lays it out, each part as `BigFloat`'s `Display` writes it, the formatter's options applied to each part: `(1.5 -8)`; `Debug` writes both parts with `BigFloat`'s `Debug`.
- The special values are MPC's, after the C standard's Annex G. A number with an infinite part is an infinity whatever its other part. A product with an infinity is IEEE 754's products and sums, and when both parts come out NaN, the infinities are recovered as Annex G's `_Cmultd` does: `(∞ + 0i)·i` is `NaN + ∞i`, `(∞ + NaN·i)·2` is `∞ + NaN·i`. A quotient by zero is `∞·a + ∞·b·i`, the infinity of the sign of the divisor's real part; an infinity over a finite number an infinity (or NaN) in each part by the signs of `a·c + b·d` and `b·c − a·d` with the infinite parts boxed to `±1`; a finite number over an infinity zeros of those signs with the divisor boxed. `√(x ± ∞i) = +∞ ± ∞i`, `√(−∞ ± yi) = +0 ± ∞i`, `√(+∞ ± yi) = +∞ ± 0i`, and the branch cut along the negative real axis keeps the sign of the imaginary part's zero: `√(−4 + 0i) = 2i`, `√(−4 − 0i) = −2i`. `norm` and `abs` of an infinity are `+∞` even with a NaN part.
- An exact zero part follows IEEE 754's sums: two zeros of one sign give that zero, and zeros of both signs, or terms that cancel, `+0`, or `−0` when that part rounds down. At the ends of the exponent range every part is rounded by its mode, as everywhere in `BigFloat`.

## Functions of complex numbers

The elementary functions of `BigComplex` are MPC's: each part of the result is the exact value of that part, rounded once, correctly, to its own precision in its own mode, with the direction of each rounding; the branch cuts, the special values and the signs of zero parts are those of GNU MPC 1.4.1, after the C standard's Annex G.

- `exp_round`, `ln_round` and `log10_round`; `sin_round`, `cos_round`, `tan_round` and `sin_cos_round`; `sinh_round`, `cosh_round` and `tanh_round`; `asin_round`, `acos_round`, `atan_round`, `asinh_round`, `acosh_round` and `atanh_round`. Each takes one precision and one mode for both parts and returns the result with the directions of the two roundings, `(re, im)`; each has a `_round_parts` form with a pair of precisions and a pair of modes (`sin_cos_round_parts` one pair of each for the sine and one for the cosine), and a plain form, `exp()`, `ln()`, …, which rounds to nearest at the precisions of the parts of `self`.
- `pow_round(&w, prec, round)` is `z^w`, the principal value `e^(w ln z)`; `pow_real_round(&x, prec, round)` takes a real exponent (`x + 0i`, MPC's `mpc_pow_fr`) and `powi_round(n, prec, round)` an `i64` (`mpc_pow_si`); `pow`, `pow_real` and `powi` round to nearest. Exact powers are exact: `(3 + 4i)^(1/2) = 2 + i`, `(1 + 2i)^5 = 41 − 38i`, `(2 + 2i)^(−3) = −1/32 − i/32`, whatever the precision asked for, and real and imaginary powers (`(−1)^(n + ti)`, `(−x)^(n + 1/2)`, `(±i)^n`, `(t ± ti)^(2n)`) have their other part an exact zero.
- `arg_round` and `arg` give the argument `atan2(im, re)` in `[−π, π]` as a `BigFloat`; `proj_round` and `proj` the projection on the Riemann sphere (`cproj`): a number with an infinite part goes to `+∞ ± 0i`.
- The logarithms and the square root have their cut along the negative real axis, where the sign of a zero imaginary part picks the side: `ln(−1 + 0i) = πi`, `ln(−1 − 0i) = −πi`. `asin` and `acos` have theirs along the real axis beyond `±1` (`asin(2 ± 0i) = π/2 ± i acosh 2`), `atan` along the imaginary axis beyond `±i`, with `atan(±0 ± i) = ±0 ± ∞i`; `asinh z = −i asin(iz)`, `atanh z = −i atan(iz)` and `acosh z = ±i acos z` (the sign of `Im z`'s), as in MPC.
- No part of these functions is exact for a number that is neither real nor imaginary, by the Lindemann–Weierstrass theorem, except the powers above and `log10` of a number with `|z|²` a power of 10 (`log10(1 + 3i)` has the real part `1/2`); the exact cases of real and imaginary arguments are the real functions' (`e^0 = 1`, `ln 1 = 0`, `sin 0 = 0`, …).
- A part beyond the exponent range overflows or underflows by its mode: `e^(x + yi)` for `x` near `2^62 ln 2` is computed with its exponent unbounded and rounded once, and a part of a function of an argument of a huge exponent costs no more than its real functions do (the reduction of `sin x` modulo `π/2` needs `π` to the bits of `x`'s exponent).

## Matrices of floats

`BigFloat` and `BigComplex` are elements of a [`Matrix`](matrix.md): the arithmetic, transposes, row and column operations, traces and powers of every matrix, and, for `Matrix<BigFloat>` and `Matrix<BigComplex>`:

- Products whose every element is correctly rounded: the exact dot product of a row and a column, rounded once to nearest at the largest precision of the elements of the two matrices (each part, for complex numbers). With an infinity or NaN among complex elements the products and sums go one by one.
- `det()`, `solve(&b)` and `inverse()` by an LU factorisation with partial pivoting at the working precision, the largest precision of the elements. Each operation is correctly rounded; the results as a whole are not, and they are as accurate as the condition of the matrix allows at that precision. `solve` and `inverse` are `None` when a pivot is exactly zero, and the determinant is then zero.
- `rank()` counts the pivots of Gaussian elimination with complete pivoting above `max(m, n)·2^(1−w)·max|a_ij|`, the tolerance of NumPy's `matrix_rank` with pivots in place of singular values; `rank_with_tolerance(&tol)` takes another (`Matrix<BigFloat>` only). It is an estimate: exact ranks are `Matrix<Rational>`'s.
- `norm(Norm::One)`, `Norm::Infinity`, `Norm::Frobenius` and `Norm::Max` of a `Matrix<BigFloat>`, correctly rounded; the 2-norm and the nuclear norm take singular values and panic.

## Examples

The core: directions, the operators' precision and the special values.

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

Results into existing numbers, primitive operands, and sums rounded once:

```rust
use a::math::float::{BigFloat, Round};
use a::math::numbers::Number;
use core::cmp::Ordering;

// 1/3 into a number of 8 bits, rounded down; 10 · 0.5 in place, exactly.
let mut r = BigFloat::zero(8);
let o = r.set_div(&BigFloat::from(1), &BigFloat::from(3), Round::Down);
assert_eq!((r.prec(), o), (8, Ordering::Less));
let mut x = BigFloat::from(10).with_prec(100);
assert_eq!(x.mul_assign_round(&BigFloat::from(0.5), Round::Nearest), Ordering::Equal);
assert_eq!(x, 5);

// Primitives and numbers on either side of the operators.
let y = 2.0 * &x + 1;
assert_eq!(y, 11);
assert_eq!(&y / &Number::from(4), 2.75);
x -= 3u8;
assert_eq!(x, 2);

// 1e16 + 1 − 1e16 rounded once is 1; rounded at each step at 53 bits, 0.
let terms = [1e16, 1.0, -1e16].map(BigFloat::from);
assert_eq!(BigFloat::sum_round(&terms, 53, Round::Nearest).0, 1);
assert_eq!(&terms[0] + &terms[1] + &terms[2], 0);
let v = [3.0, 4.0].map(BigFloat::from);
assert_eq!(BigFloat::dot_round(&v, &v, 53, Round::Nearest), (BigFloat::from(25), Ordering::Equal));
```

Roots and powers, exact where they can be:

```rust
use a::math::float::{BigFloat, Round};
use a::math::numbers::Number;
use a::math::rational::Rational;
use core::cmp::Ordering;

let four = BigFloat::from(4);
assert_eq!(four.rec_sqrt(), 0.5);
assert_eq!(BigFloat::from(-27).cbrt(), -3);
assert!(BigFloat::from(-16).nth_root(4).is_nan());
assert_eq!(BigFloat::from(2).nth_root_round(10, 53, Round::Down).1, Ordering::Less);
assert_eq!(BigFloat::from(2).powi(-3), 0.125);

// 4^(3/2) = 8 exactly, in every mode; (−8)^(1/3) = −2.
let r = Rational::from(3) / 2u32;
assert_eq!(four.pow_rational_round(&r, 10, Round::Up), (BigFloat::from(8), Ordering::Equal));
assert_eq!(BigFloat::from(-8).pow_rational(&(Rational::from(1) / 3u32)), -2);

// 3^(2^62) overflows at once, from the exponents.
assert!(BigFloat::from(3).pow_number(&Number::from(1u64 << 62)).is_infinite());
```

Conversions and text:

```rust
use a::math::float::{BigFloat, Round};
use a::math::numbers::Number;
use a::math::rational::Rational;
use core::cmp::Ordering;

let x: BigFloat = "0.1".parse().unwrap();
assert_eq!(x, 0.1);
assert_eq!(x.to_string(), "0.1");
assert_eq!(format!("{x:.20}"), "0.10000000000000000555");
let (d, e, o) = x.to_digits(10, Some(3), Round::Up).unwrap();
assert_eq!((d.as_str(), e, o), ("101", 0, Ordering::Greater));
assert_eq!(BigFloat::from(-1000).to_string_radix(16, Some(2), Round::Up), "-3.e@2");
let (y, o) = BigFloat::from_str_radix("0.1", 10, 200, Round::Down).unwrap();
assert_eq!(o, Ordering::Less);
assert_eq!(y.to_f64_round(Round::Nearest), (0.1, Ordering::Greater));
assert_eq!(BigFloat::from_str_radix("0x1.8p1", 0, 53, Round::Nearest).unwrap().0, 3);

assert_eq!(BigFloat::from(-2.5).round_ties_even(), -2);
assert_eq!(BigFloat::from(10.5).rint_round(2, Round::Nearest).0, 12);
assert_eq!(BigFloat::from(7.5).to_number_round(Round::Down).unwrap().0, Number::from(7));
assert_eq!(BigFloat::from(1e300).to_i64_round(Round::Nearest), Some((i64::MAX, Ordering::Less)));
assert_eq!(i32::try_from(&BigFloat::from(42.0)).unwrap(), 42);
assert!(i32::try_from(&BigFloat::from(0.5)).is_err());

// Exact comparisons with rationals: 1/3 to 64 bits is not 1/3.
let third = Rational::from(1) / 3u32;
let t = BigFloat::from_rational(&third, 64, Round::Nearest).0;
assert!(t != third && t < 0.5);
```

Exponentials and logarithms:

```rust
use a::math::float::{BigFloat, Round};
use core::cmp::Ordering;

// e and ln 2 to 53 bits: the nearest doubles, both below the exact values.
assert_eq!(BigFloat::from(1).exp_round(53, Round::Nearest), (BigFloat::from(core::f64::consts::E), Ordering::Less));
assert_eq!(BigFloat::from(2).ln_round(53, Round::Nearest), (BigFloat::from(core::f64::consts::LN_2), Ordering::Less));
// Exact results, in every mode.
assert_eq!(BigFloat::from(4).pow_round(&BigFloat::from(0.5), 10, Round::Up), (BigFloat::from(2), Ordering::Equal));
assert_eq!(BigFloat::from(0.125).log2_round(10, Round::Down), (BigFloat::from(-3), Ordering::Equal));
assert_eq!(BigFloat::from(3).exp10_round(10, Round::Down), (BigFloat::from(1000), Ordering::Equal));
// Far beyond the exponent range at once: e^(2^100) overflows, to the largest number toward zero.
let huge = BigFloat::from(2).pow(&BigFloat::from(100));
assert!(huge.exp_round(53, Round::Nearest).0.is_infinite());
assert_eq!(huge.exp_round(53, Round::TowardZero).1, Ordering::Less);
// The hyperbolic functions, both from one exponential.
let x = BigFloat::from(0.75);
let (s, c) = x.sinh_cosh_round(80, Round::Up);
assert_eq!((s, c), (x.sinh_round(80, Round::Up), x.cosh_round(80, Round::Up)));
assert_eq!(BigFloat::from(1).atanh(), BigFloat::infinity(32));
```

Circular functions:

```rust
use a::math::float::{BigFloat, Round};
use a::math::numbers::Number;
use core::cmp::Ordering;

// sin(1) to 53 bits is the f64's; to 1000 bits rounded up, above the exact value.
let one = BigFloat::from(1.0);
assert_eq!(one.sin(), BigFloat::from(1f64.sin()));
let (s, o) = one.sin_round(1000, Round::Up);
assert_eq!((s.prec(), o), (1000, Ordering::Greater));

// The angle of (−1, −1) is −3π/4; tan(π/4) rounds to 1 at 24 bits.
let a = BigFloat::from(-1).atan2_round(&BigFloat::from(-1), 53, Round::Nearest).0;
assert_eq!(a, BigFloat::from(-3.0 * core::f64::consts::FRAC_PI_4));
let quarter = BigFloat::pi_round(64, Round::Nearest).0.mul_round(&BigFloat::from(0.25), 64, Round::Nearest).0;
assert_eq!(quarter.tan_round(24, Round::Nearest).0, BigFloat::from(1));

// sin(2^1000) needs π to more than 1000 bits; acos near 1 loses nothing to 1 − x².
let x = BigFloat::from_parts(&Number::from(1), 1000, 1, Round::Nearest).0;
let (s, _) = x.sin_round(53, Round::Nearest);
assert!(s.abs() <= BigFloat::from(1));
let near_one = BigFloat::from(1.0 - f64::EPSILON / 2.0);
assert!(near_one.acos() > BigFloat::zero(53));
```

Constants:

```rust
use a::math::float::{BigFloat, Round};
use core::cmp::Ordering;

let (pi, o) = BigFloat::pi_round(53, Round::Nearest);
assert_eq!(pi, BigFloat::from(core::f64::consts::PI));
assert_eq!(o, Ordering::Less);
assert_eq!(BigFloat::e(53), BigFloat::from(core::f64::consts::E));
assert_eq!(BigFloat::ln2(24), BigFloat::from(core::f32::consts::LN_2));
assert_eq!(BigFloat::ln10(53), BigFloat::from(core::f64::consts::LN_10));

// γ to 200 bits, rounded down and up: one unit of the last place apart.
let lo = BigFloat::euler_gamma_round(200, Round::Down).0;
let hi = BigFloat::euler_gamma_round(200, Round::Up).0;
assert!(lo < hi);
assert_eq!(BigFloat::catalan(53), BigFloat::from(0.915_965_594_177_219_f64));

// A long π, and a shorter one cut from it.
let long = BigFloat::pi_round(100_000, Round::Down).0;
let short = BigFloat::pi_round(50_000, Round::Down).0;
assert_eq!(long.prec_round(50_000, Round::Down).0, short);
```

Complex numbers and matrices of floats:

```rust
use a::math::float::{BigComplex, BigFloat, Round};
use a::math::matrix::Matrix;
use core::cmp::Ordering;

// (3 + 4i)/(1 + 2i) = 2.2 − 0.4i: each part rounded on its own, up and down.
let z = BigComplex::new(BigFloat::from(3), BigFloat::from(4));
let w = BigComplex::new(BigFloat::from(1), BigFloat::from(2));
let (q, o) = z.div_round_parts(&w, (10, 10), (Round::Up, Round::Down));
assert_eq!(o, (Ordering::Greater, Ordering::Less));
assert!(*q.re() > BigFloat::from(2.2) && *q.im() < BigFloat::from(-0.4));
assert_eq!(BigComplex::new(BigFloat::from(1.5), BigFloat::from(-8)).to_string(), "(1.5 -8)");

// The root of −4 on either side of the branch cut.
let up = BigComplex::new(BigFloat::from(-4), BigFloat::zero(53));
let down = BigComplex::new(BigFloat::from(-4), BigFloat::neg_zero(53));
assert_eq!(up.sqrt(), BigComplex::new(BigFloat::from(0), BigFloat::from(2)));
assert_eq!(down.sqrt(), BigComplex::new(BigFloat::from(0), BigFloat::from(-2)));
assert_eq!(z.abs(), BigFloat::from(5));

// A product of matrices, each element the exact dot product rounded once: 1e32 + 1 − 1e32,
// which products and sums one by one at 53 bits would make 0.
let a = Matrix::from_rows(&[[1e16, 1.0, -1e16]]).map(|&x| BigFloat::from(x));
let b = Matrix::from_rows(&[[1e16], [1.0], [1e16]]).map(|&x| BigFloat::from(x));
assert_eq!((&a * &b)[(0, 0)], BigFloat::from(1));
```

Functions of complex numbers:

```rust
use a::math::float::{BigComplex, BigFloat, Round};
use a::math::numbers::Number;
use core::cmp::Ordering;

// e^(iπ) = −1 + i·sin(π): the real part rounds to −1 from below, the imaginary part is tiny.
let z = BigComplex::new(BigFloat::zero(53), BigFloat::pi(53));
let (e, o) = z.exp_round(53, Round::Nearest);
assert_eq!(*e.re(), BigFloat::from(-1));
assert_eq!(o.0, Ordering::Less);
assert!(e.im().abs() < BigFloat::from(1e-15));

// ln on either side of the cut, and an exact power.
let up = BigComplex::new(BigFloat::from(-1.0), BigFloat::zero(53));
let down = BigComplex::new(BigFloat::from(-1.0), BigFloat::neg_zero(53));
assert_eq!(*up.ln().im(), BigFloat::pi(53));
assert_eq!(*down.ln().im(), -BigFloat::pi(53));
let w = BigComplex::new(BigFloat::from(3), BigFloat::from(4));
assert_eq!(w.pow_real(&BigFloat::from(0.5)), BigComplex::new(BigFloat::from(2), BigFloat::from(1)));

// A tiny argument: sin z = z − z³/6 + …, each part rounded beside the argument's at once.
let t = BigFloat::from_parts(&Number::from(3), -1_000_000, 53, Round::Nearest).0;
let s = BigComplex::new(t.clone(), t.clone()).sin_round(53, Round::Down);
assert_eq!((s.0.re().clone(), s.1.0), (t, Ordering::Less));
```

## The method

### Rounding and Ziv's strategy

- Every operation ends in one rounding of an exact magnitude `M·2^k`, or of one known only to lie strictly between `M·2^k` and `(M + 1)·2^k` (the sticky bit), `M` holding at least the bit below the last one kept: the bits dropped give the round bit and whether anything lies below it.
- The functions that can only be approximated use Ziv's strategy: the value is computed to a working precision with a bound of its error, both ends of the interval the bound gives are rounded, and the result is taken when they agree and it lies outside the interval, so that the direction is known as well; otherwise again at a working precision half as large again. A scan of the approximation's bits between the error and the rounding bit decides most cases at the cost of one rounding before the general test. The exact cases are found before the loop, which then ends.
- Most functions are compositions of operations on balls: a midpoint at the working precision and a radius rounded up (32 bits and an exponent), each operation adding its rounding to the radius, so that the bound is sound by construction. A result within less than a unit of the last place of a number it starts from (`e^x` near 1, `ln(1 + x)` near `x`, `sinh x` and `sin x` near `x`, `tanh x` near `±1`, `cos x` near 1) is rounded from that number and the side it lies on, as MPFR's `mpfr_round_near_x`.

### Arithmetic

- Numbers of one limb (a precision up to 64) and of two (up to 128) are added, multiplied and divided in a word of 64 or 128 bits, after MPFR's `mpfr_add1sp1`, `mpfr_sub1sp1`, `mpfr_mul_1`, `mpfr_div_1` and their two-limb forms, with the exponents as 64-bit integers. A sum shifts the smaller term right by the difference of the exponents, the bits shifted out the word below, and a sum that carries right by one; a difference is `x·2^W − y·2^(W − d)` exactly in two words for `d < W` (`W` the word's bits), and for `d ≥ W` the bits of `y` that reach the word below, one unit more off and a sticky bit when a fraction lies below them; it is then shifted left until its top bit is set, at most one bit unless the exponents are at most one apart, where it is exact. A product is the exact double word, a quotient one or two divisions of two limbs by one with the reciprocal of the divisor (two of three limbs by two), with one limb more of quotient at 64 and 128 bits. The word, the word below and a sticky bit are then rounded in registers. The forms into an existing number (`set_add` and the others) read the operands' fields and write the result's limbs straight-line, with the special values, an exact zero and a result past the exponent range left to the general path, out of line. Square roots of one and two limbs are in registers too, after MPFR's `mpfr_sqrt1`, `mpfr_sqrt1n` and `mpfr_sqrt2`. A square root of one limb is `√(x·2^64)` or `√(x·2^63)` from the square root of the top limb in double precision and one step of Newton's method on the exact remainder, within one of the integer root, which decides the rounding unless the root's bits below the round bit are within one of a boundary; only then is the root made exact, with its remainder. A root of two limbs takes one step of the Karatsuba square root on that one, its quotient first estimated by a product with the reciprocal of the top limb's root (Newton's step on the double-precision reciprocal), within a few units, and computed by the division only when those units straddle a boundary. For 64 and 128 bits, where the round bit lies below the root, the remainder above the root takes its place, so a precision of a whole limb costs no limb more.
- A sum is exact, on at most the precision and the operands' lengths together, unless the smaller term lies below a quarter of the result's last place and below the larger term's last bit. Then it changes the result only as a sticky bit would: the sum lies strictly between the larger term and one unit more (or less) of a scale below both, and no number of the precision nor any halfway point lies there.
- Operands of as many limbs as their result (three or more) are added as MPFR's `mpfr_add1sp` and `mpfr_sub1sp` add them, in the result's limbs: the smaller shifted right into them (with no copy for equal exponents), one pass of the crate's kernels with the larger, and the bits shifted out kept as the word below and a sticky bit. A difference is the larger term less the window, less one unit when the word below or the sticky bit is not zero, over the word below `−L − [f > 0]`; exponents two or more apart cancel at most one bit, which a shift by one refills from the word below, and exponents at most one apart leave no fraction, so the difference is exact and normalised by a shift.
- Operands of other lengths than the result are added in a word of their own: two limbs for one limb of result, four for two, and four for operands of at most four limbs whatever the result (a longer result takes the sum when it is exact in the word, and the general path otherwise). An operand longer than the word lies between its truncation `T` and `T` plus one unit of its last limb, strictly when its other limbs are not all zero, so the sum lies strictly between the sums of the lower and of the upper ends. Both are rounded; when they round to the same number `R` and `R` is not above the lower end (not below the upper one), `R` is the rounding of the sum, below (above) it. Otherwise, a chance of about `2^−64` for random operands and the rule for sums that cancel past the word, the general path. Products of operands of at most two limbs are exact in four limbs, those of at most four for results of at most four limbs in eight, a quotient of at most 62 bits from the top limbs of long operands is decided by the same rule from `X/(Y + 1)` and `(X + 1)/Y`, and a quotient of 3 to 20 limbs by a divisor of one limb from a dividend of at most two is computed limb by limb into the result's limbs (the general path folds the limbs in from 20). The new results (`add_round` and the others, the operators) take the same kernels into a vector of the result's length, made once the shapes are known.
- Longer numbers compute only the limbs their rounding needs, on a window of `n + 1` limbs for `n` limbs of result. A sum adds the window of the larger term and the bits of the smaller in it; what lies below the window, the tails, makes a sticky bit when one term has one, and when both have one the sum is known only within two units of the window's lowest limb: the rounding is then decided when that limb is not within those units of a limb boundary or of a halfway point, and computed exactly otherwise (a chance of about `2^−62` for random operands). A difference of exponents at most one apart, which can cancel any number of bits, is exact.
- A product of operands longer than `n + 2` limbs multiplies their top `n + 2` limbs: the product of the cut operands is below the exact one by less than `B^l + B^l + 1` units of its lowest limb (`B = 2^64`, `l` their length), so the exact product's top limbs are those of the cut one or up to two more, and the rounding is decided as for a sum. Otherwise it is the exact product, by the crate's multiplication: the schoolbook method, Karatsuba, Toom-3, Toom-4 and the number-theoretic transforms for the longest. A square takes the square's methods. A fused multiply-add takes the exact product as an operand of the sum, rounded once: for operands of at most two limbs and an addend of at most four the product is a word of 256 bits and the sum is one of the words' sums above, for operands of at most four and an addend of at most eight a word of 512 bits, in registers (a longer result takes the sum when it is exact in the word); longer ones on limbs.
- A quotient `F = 0.a·2^−c / 0.b` in `[1/2, 1)` (`c = 1` when `0.a ≥ 0.b`) comes from `n + 1` limbs of quotient: for a divisor longer than that, from the top `2(n + 1)` limbs of the dividend and the top `n + 1` of the divisor, whose quotient `Q'` lies within `(Q' − 2, Q' + 1)` of the exact `F·B^(n+1)` and decides as above. Otherwise, for a divisor of 3 to 480 limbs, the quotient alone, exactly, its last limbs from a divisor that drops a limb at each step (after GMP's `mpn_sbpi1_div_q`, about half the products): when its bits below the round bit are not all zero the remainder cannot change the rounding and is a sticky bit. Otherwise, or when that leaves it undecided, the exact division of the dividend's top limbs with the remainder for the sticky bit: the schoolbook method, Burnikel and Ziegler's divide and conquer, and Newton's method for the longest.
- A square root takes the top `2q` limbs of `0.a·2^−c` (`c` the parity of the exponent, `q` the limbs of `prec + 16` bits up to 64 limbs, one limb more than the precision's beyond) and their integer root by Zimmermann's Karatsuba square root: the floor of the root of the truncated radicand is that of the exact one, and its remainder is needed only when its bits below the round bit are all zero (a root halfway between two numbers is one of them). Up to 64 limbs, when fewer than 16 bits lie below the precision in its own `n` limbs, the root of `n` limbs is taken with its remainder, as MPFR does: for `64n` bits the round bit lies below the root and `√R ≥ S + 1/2` exactly when the remainder is above `S`, a tail leaving only a remainder of `S` itself to a limb more.
- `set_*` and `*_assign_round` round into the vector of the number they write, which has the room its precision needs; a primitive operand of the operators is written into a float that the thread keeps, so that neither allocates.
- A sum of many terms adds them exactly into one two's-complement accumulator from the lowest bit of any term to past the top (one product into a scratch buffer and one shifted addition each), and rounds once, when their bits span at most `4(l + p) + 2^16` bits, `l` the longest term; otherwise the terms are sorted by exponent and added exactly from the largest until the rest is below the last bit of the sum so far and below a quarter of the result's last place, where only the rest's sign counts, which the same accumulation at one bit gives as soon as it stops. A dot product takes the exact products as its terms.

### Roots and powers

- `1/√x` is `2^−j` for `x = 4^j` and otherwise irrational, and never halfway, so an approximation with a bound decides it once the bound keeps it from the boundaries. Of a mantissa of one or two limbs and to 127 bits it is the integer root of `Q = floor(2^E / M)`, of two limbs to 63 bits and four beyond, by the square roots in registers above: `floor(√Q) = floor(√(2^E / M))`. To 2,000 bits, from the square root: `s = floor(√X)` of the top `2q` limbs of the radicand, then `y = floor(2^K / s)`, `K = 128q − 1`, as `2^K / √X` lies in `(y − 2, y + 1)`. Beyond, by Newton's iteration on the reciprocal square root itself, `y ← y + y(1 − a·y²)/2`, on whole limbs from a double-precision start, the number of limbs after the point `m' ≤ 2m − 1` from one step to the next: with `y = y_k(1 + δ)` an exact step has the error `−(3/2)δ² − δ³/2`, and the cut of `a` to `m' + 1` limbs, that of `a·y²` and the floor of the correction add less than `(1 + 2^−60)` units of `2^(−64m')`, a bound carried from step to step. The rounding is taken when every integer within the bound rounds alike; otherwise (and for a power of two that is not one of 4) the integer root of a quotient of twice the limbs, as before.
- An `n`-th root of degree 3 to 16 to 127 bits starts from the radicand `R = floor(y·2^(64n))` (or `2^(128n)`) of `y = 0.M·2^r`, `x = y·2^(nq)`, `−n < r ≤ 0`: the root `s₀` of its top limbs in double precision, of any accuracy, and the exact remainder `E = R − s₀^n` give `|ρ − s₀| ≤ n|c|` for Newton's correction `c = E/(n·s₀^(n−1))` (from `(1 + u)^(1/n) ≤ 1 + u/n` and `(1 − u)^(1/n) ≥ 1 − u`), and the step `s₀ + c` overshoots `ρ` by at most `(n − 1)/2·Δ²/s₀·(1 + nΔ/s₀)`, below `2^−12` for `Δ ≤ 2^24`: its floor is within one of the integer root. Its bits below the round bit decide but within one of a boundary, where `s^n`, `(s ± 1)^n` and for 64 bits `(2F + 1)^n` against `2^n·R` make it exact. The root of 128 bits takes a second Newton's step from `S_hi·2^64`, the exact root of the top `n` limbs, which overshoots by less than `n − 1` units, its quotient by a division of two limbs by the top limb of `n·S_hi^(n−1)` within three; it decides but within `n + 2` units below and 2 above of a boundary, and the integer root takes the others.
- Otherwise an `n`-th root of degree up to 16, or of a radicand of up to `2^12` bits (`n·64(n + 1)`), is the integer root of the mantissa shifted to `prec + 16` bits of root (80 at least), by Newton's method on integers; the floor of the root of the truncated radicand is that of the exact one, and the power of the root, when its bits below its round bit are all zero, tells exact roots and halfway ones apart. A longer one is computed by Newton's method on floats with Ziv's strategy, each approximation `y` certified by `y_lo^n ≤ z ≤ y_hi^n` for `y(1 ∓ 2^(4 − w))` raised by squarings rounded outward, after the exact roots are found (`x = m^n·2^(kn)`, which for an odd `m ≥ 3` needs `n·log2 3` bits).
- `x^n` is exact for `x` a power of two and, for `n > 0`, computed exactly when the odd part `m` of `x` has `n(bits(m) − 1) + 1 ≤ prec + 1`, which covers the exact and the halfway results; otherwise it is neither, and the power by squarings from the top bit of `n`, every product cut to its top `W` limbs aligned at its top bit, is below the value by a relative error of less than `ε = 2^(1 − 64W)` each: `c ≤ v < c(1 + ε)^k`, `k ≤ (k_b + 2)·n` (`k_b` the base's own: 0 for an exact base, 1 for a cut one or the reciprocal `floor(2^K / M)` of a negative `n`, 2 for the reciprocal of a mantissa cut to `W + 1` limbs and rounded up), so the value lies in `[c, c + 4k)` units of `c`'s last bit and rounds as `c` with a sticky bit when no integer there carries into the round bit or has its bits below it all zero. `W` holds the precision, the bits of `4k` and 12 more, so that this rarely fails; then Ziv's strategy on half as many limbs again. To 64 bits and from a base of up to two limbs, the squarings are on 128 bits in registers; to `8·64` bits on arrays of 2 to 8 limbs with the products unrolled; beyond, on vectors with the crate's products.
- `x^(p/q)` is exact only when the `q`-th root of `x` is (from `ap + bq = 1`): an exact root is raised to the power and rounded once; otherwise Ziv's strategy on the root rounded toward zero to the working precision (away from zero for `p < 0`, then the reciprocal cut from below), and its power cut as above.

### Conversions

- A primitive float is rounded from the top limb of the mantissa, the lower limbs a sticky bit, in registers: a number `0.1…·2^e` has `min(53, e + 1074)` bits in an `f64`, fewer among the subnormals, and the float's encoding follows from the rounded mantissa alone, a carry taking a subnormal to the least normal number and the largest finite number to the infinity.
- An integer of a precision that holds it is the top limbs of the mantissa with the fraction cleared and one unit added when the direction rounds up, a carry out of the top making it the next power of two, written in one pass into the result's limbs: `rint_round` to a smaller precision takes only the limbs it keeps, and `set_rint` writes them into the number it changes. A number whose last bit is above the point is an integer already and is not expanded: the floor of `3·2^(2^61)` costs a copy. A primitive float into an existing number is its limb at the top of the mantissa and zeros below, or below 53 bits its limb rounded in a register.
- The comparisons with integers and `Number`s compare the signs, then the magnitudes `0.m·2^e` and `M·2^l` by their exponents, then by the mantissas aligned at their top bits, limb by limb, without making either a float; with a `Rational` `n/d`, `|x|·d` against `|n|`, exactly.

### Text

- Writing in radix `b = 2^a·c`, `c` odd, takes the integer `D` that `|x|·b^(n−1−E)` rounds to, `E` the exponent of the first digit: first `floor((e − 1)·log_b 2)` from a table of `log_b 2` to 128 bits, which is `E` or one less. When it is one less, the integer part `t` of `|x|·b^(n−1−E)` has `n + 1` digits, which its bits tell against bounds of `n·log2 b` (or `b^n` itself when they do not), and the digits come from the same work divided by `b`: with `t = b·t' + r` and `f` the fraction, the fraction of `t' + (r + f)/b` compares to one half as `2r + 2f` to `b`, above for `2r > b`, as `f` to one half for `2r = b − 1`, below for less. For a power of two the digits are bits. Otherwise `D` is exact (a product by `c^k`, or a quotient with its remainder) when `|x|·b^k` may be a half-integer, which needs `q + k·log2 c < log2(2y) + 1` for `k ≥ 0` (`q` the bits of the mantissa's odd part) and `c^j` to divide the odd part for `k = −j < 0`, or when the exact work is no larger than the operands and the result. Otherwise neither it nor its quotient by `b` is a half-integer and Ziv's strategy ends: the mantissa's top `w` bits and `c^k` to `w` bits, each product truncated, bound the value within `[A, A + 4K]` units for `K` truncations (`K < 2k/t`, `t` the digits of `c` in a word, as a square doubles the error of its operand), and a quotient within `[Q − 8K, Q + 9]`; the digits are taken when both ends have the same integer part on the same side of the half, otherwise at a working precision half as large again. The first working precision is the bits of the integer part and of `k` and 12 more, in whole limbs, which leaves about one attempt in `2^9` open. The power is squared and multiplied by `c^t` in whole limbs, each result shifted to its top bit and cut to its top limbs: up to six limbs in registers by unrolled column products (the products off the diagonal once and doubled), beyond by the crate's squares into two fixed buffers; up to 768 bits the rest of an attempt, the product or the crate's division and the integer parts of both ends, works on the stack too. In radix 10, up to 120 bits and 37 digits, `D` comes from the exact parts of `|x|·10^k` in registers: a product by `5^k` from a table to `5^82` in five limbs, or a quotient by `5^j`, `j ≤ 27`, in 128 bits.
- The shortest decimal of `Display` lies in the interval of the numbers that round to `x`, `x ∓ ulp/2` (`x − ulp/4` below a power of two), its ends included when the mantissa is even: at the first scale `10^k` where the interval holds an integer, its integers are `L..=H`, which a coarser scale makes fewer than ten. A multiple of ten among them is the shortest, its zeros dropped; otherwise all have as many digits and the nearest to `x·10^k` is taken. For `k ≥ 0` one power `5^k` gives the three products. Up to 120 bits the same integers are computed in registers, by the table of `5^k` to `5^82` and quotients by `5^j` to `5^27`, for the numbers whose lowest bit lies between about `2^−270` and `2^90`.
- Reading takes the digits from the first that is not zero to the last that is not zero, the others in the exponent, where they are in the text: the digits before the point and those after it are two slices, not a copy. Up to radix 10 the runs of digits are found eight bytes at a time in a word: a byte is no digit when its top bit is set, when adding `0x80 − (b'0' + radix)` sets it, or when subtracting `b'0'` from the byte with its top bit set clears it, and neither step carries across a digit, so the lowest flag is the first byte that is no digit. In a radix that is a power of two, the digits for two bits more than the precision and a sticky bit for the rest. Otherwise, up to 62 bits, the digits of a word, `D·c^k` as a product of two words or `D·2^s/c^j` as a quotient of a double word by a word, rounded in registers. Then, up to 1,024 bits, the digits of `prec + 40` bits at most (all of them when there are fewer), eight decimal digits at a time from a word of the text, into fixed buffers on the stack: for `k ≥ 0`, `P = D·c^k` exactly by products with the largest power of `c` in a word; for `k = −j`, `D·2^u` divided by `c^j` with the crate's division, `u` such that the quotient `Q` has `prec + 2` bits, the remainder `R` the sticky bit. When all the digits were read the rounding is exact. Otherwise the number lies strictly between `D·b^k` and `(D + 1)·b^k`, and the rounding is that of `P` with a sticky bit when the bits of `P` from `c^k`'s up to two below the precision are not all ones, and that of `Q` with a sticky bit when `R + 2^u ≤ c^j`: both fail for about one text in `2^37`, as `D` has `prec + 41` bits. Beyond, and when those fail, `D·b^k` is rounded once, exactly when it may be a number of the precision or the midpoint of two (`q + k·log2 c < p + 2` for `k ≥ 0`, `c^j` dividing the odd part of `D` for `k = −j`) or when the work is small, and by Ziv's strategy as for writing otherwise. The overflow and the underflow are decided first, from bounds of the bits of `D·b^k` (a table of `log2 b` to 58 bits), so an exponent of any size costs nothing more. A text of more digits than the precision needs is truncated to the digits of the precision and 8 bits more: the number lies strictly between `D·b^t` and `(D + 1)·b^t`, and when both round to the same number, which is not between them, that is its rounding with its direction; otherwise twice as many digits, up to all of them, which decide exactly.
- The digits of a long text are converted by the divide-and-conquer radix conversion of [numbers](radix.md), and those of a long `D` written by it; quotients of up to 40 limbs are Knuth's Algorithm D, longer ones the Newton division of numbers.

### Exponentials, logarithms and powers

- The exponential reduces `x = k ln 2 + r`, `0 ≤ r < 1` (`k` from the top 64 bits of `x` times `1/ln 2`, `ln 2` from the constants with `bits(k)` more bits), then `t = r/2^s`, `s ≈ 0.3·√W` for `W` bits, sums the Taylor series of `e^t` by rectangular splitting (Smith's method: the terms in blocks whose denominator `(a + 1)…b` fits a limb, the integers `(i + 1)…b` as coefficients of the powers `t^1…t^m`, one full product and one division by a limb a block) and squares the sum `s` times, in fixed point. From 11,264 bits (176 limbs) it sums the series by the bit burst instead: `t` cut into pieces of `2^j` bits, each piece's series by binary splitting, exactly, then one division, the pieces multiplied.
- The bound of the exponential: the reduction is within `|k|·δ_ln2 + u` (`u = 2^−W`), the sum within 8 units a block and one for the terms left out (`ceil(E/Q) + 8` with the block's denominator `Q`, from `c_i/Q ≤ 1/(i − a)!`, `Σ j/j! = e` and `t^j` within `j` units), the bit burst within 9 units a piece and one; `s` truncated squarings turn a relative `ε` into at most `2^s(ε + u)(1 + 2^−9)` while `2^s·ε ≤ 2^−20`; the reduction's error multiplies the result by at most `1 + 2(δ_r + 2^s·u)`.
- The logarithm writes `x = 2^e·u`, `2/3 ≤ u < 4/3`, and takes `ln x = e·ln 2 + ln u`. Up to 512 bits of working precision (eight limbs after the point) `ln u` comes from tables, as Arb's logarithm at short precisions: `d₁ = ⌊128u⌋` and `y₁ = ⌊128u/d₁⌋` in `[1, 1 + 1/d₁)`, `d₂ = ⌊2^14·y₁⌋` and `y₂ = ⌊2^14·y₁/d₂⌋ = 1 + t` with `t < 2^−14` (two divisions by one limb), `ln u = ln(d₁/128) + ln(d₂/2^14) + ln(1 + t)`, the two logarithms from tables of 96 and 208 entries at 576 bits and `ln(1 + t) = t·q₁`, `q_k = 1/k − t·q_(k+1)`, by Horner's rule with the coefficients `1/k` of a table. The tables are computed when the crate is compiled, `ln(c/2^k) = 2·atanh((c − 2^k)/(c + 2^k))` summed at 640 bits (within `2^−629` from below). The bound: a unit for each floor of `y₁`, `y₂` (`y_i ≥ 1`), a unit and `2^−574` for each entry, `2/(1 − t)` units for each `q_k` and `1.01` for `t·q₁`, half a unit of tail: `6u + 2^−573`. `e·ln 2` takes 640 bits of `ln 2` (`|e|·2^−629` and a unit); `log2 x = e + ln(u)·(1/ln 2)` and `log10 x = (e·ln 2 + ln u)·(1/ln 10)` multiply by constants of 640 bits computed the same way, so that they cost a product more than `ln`. For `e = 0` the fixed point takes as many more bits as `u − 1` has leading zeros, and when that is more than the working precision `ln(1 + d)` is `d` within `d²`.
- Beyond the tables, Newton's iteration on the exponential, `y ← y + L(u·e^−y − 1)` with `L(v) = v − v²/2 + v³/3`, from the `f64` logarithm, each step at four times the precision of the one before, so that the steps together cost about a third more than the last. Only the last step counts: `ln u = y + ln(1 + v)` exactly for the exact `v`, so with the `v` it finds, `|v| ≤ 1/4`, the bound is `2δ_v + 5u + |v|^4/2` (`δ_v` from the exponential's relative error). `e·ln 2` adds `|e|·δ_ln2 + u`.
- From 3,400 bits the logarithm is MPFR's AGM, `ln x = π·a/(2·AGM(a, 1)) − k·ln 2` with `a = s/4`, `s = x·2^k ≥ 2^(T/2 + 8)` (the AGM is homogeneous: `AGM(1, 4/s) = (4/s)·AGM(s/4, 1)`, which saves the division of `4/s`), the formula within `ln(s)·k'²/(1 − k'²)` for `k' = 4/s` (DLMF 19.12.1: every term past the first of the series of `K` lies in `(0, k'^(2j)·ln s]`). The steps are on numbers of `n` limbs and an exponent, each mean and each square root (the integer square root of the exact product, without its remainder) cut from below: as the AGM grows with each argument and is homogeneous, `K` steps cut by at most `η = 2^(2 − 64n)` each leave the pair's AGM within `(1 − η)^K` of `AGM(a, 1)`. The first step is `(a + 1)/2` and `√a` alone. The iteration stops when `δ = (a − b)/(a + b) ≤ 2^−(T/4 + 3)` and takes `AGM(a, b) = m − (a − b)²/(16m)`, `m = (a + b)/2`, within `δ⁴/3` relatively (`AGM(1 + δ, 1 − δ) = 1/(1 + S)`, `S = Σ_(j≥1) ((1/2)_j/j!)²·δ^(2j)`, `S − δ²/4 ≤ δ⁴/(4(1 − δ²))`), the correction computed to half the bits; that saves the last two steps. Then one product by π and one division: in all a relative error `(K + 8)·η` of `ln s`.
- At up to 256 bits of working precision (results to about 210 bits) the exponential's kernel runs in arrays on the stack, step for step the same, as the allocations of the limb vectors cost more than the arithmetic there (four limbs only for `|x| < 2^56`, whose `k·ln 2` fits the arrays).
- `exp2` splits `x = n + f` and takes `e^(f·ln 2)·2^n`; `exp10` takes `ln 10` from the constants' cache; `log2` and `log10` beyond the tables divide by `ln 2` and `ln 10`; `exp_m1` is `e^x − 1` with as many more bits as `x` has zeros after the point; `ln_1p` is the logarithm of `1 + x` exactly when it takes at most eight limbs (by the tables), otherwise of `1 + x` rounded to as many more bits as `x` has zeros after the point.
- `pow` is `e^(y·ln|x|)` with `ln|x|` to `bits(|y·ln x|)` more bits, bounds of `|y·ln x|` from the exponents alone deciding the range and the cases near 1 first. The exact powers: with `x = M·2^e`, `M` odd, and `y = c/2^t`, `c` odd, `x^y` is rational only when `x` is a `2^t`-th power (`M` a square `t` times over, `2^t` dividing `e`), and then it is exact when `M = 1` or `c > 0` and `(M^(1/2^t))^c` has at most `prec + 1` bits.

### Hyperbolic functions

- `sinh` and `cosh` are `(E ∓ 1/E)/2` and `tanh = (P − 1)/(P + 1)` in fixed point, from the raw exponential `E = e^|x|` (or `P = e^(2|x|)`) of the kernel within a relative `ε`: `1/E` as `⌊2^D/y⌋` for `E = y·2^l` and `D = 2·bits(y) + 16`, within `2^(−D−l)` of `1/Ẽ`, so that `sinh` and `cosh` are within `1.01·ε·(Ẽ + 1/Ẽ)/2` and a unit; `tanh` within `ε(1 + 2ε)/2` absolutely, as `2P/(P + 1)² ≤ 1/2`, and a unit of its quotient. The exponential takes as many more bits as the cancellation needs, `2 − e` for `sinh` (`coth |x| ≤ 2^(2−e)` below 1) and `3 − e` for `tanh`, `|x| < 2^e`. From `|x| ≥ W + 16`, `sinh` and `cosh` are `e^|x|/2` within a relative `2^−(W + 16)`, from the exponential's own exponent, so that a result near the overflow threshold is right where `e^|x|` itself overflows.
- `asinh x = ln(1 + |x| + x²/(1 + √(1 + x²)))` below 1, `ln(|x| + √(x² + 1))` above, `ln|x| + ln 2` within `2^−2e` far out; `acosh(1 + d) = ln(1 + d + √(d(d + 2)))` below 2, with `d` exact; `atanh x = ln(1 + 2|x|/(1 − |x|))/2`, `1 − |x|` exact where it cancels.

### Circular functions

- None of these functions has an exact result for an argument that is not zero but `acos(1) = 0`, so Ziv's loop ends; those values are treated before it. A small argument whose function lies within less than half an ulp of the argument (or of 1, or of `1/x` for a power of two) is rounded directly by the side it lies on: `0 < x − sin x < 2^(3e−2)`, `0 < tan x − x < 2^(3e−1)`, `0 < 1 − cos x < 2^(2e−1)`, `0 < sec x − 1 < 2^(2e)`, `0 < x − atan x < 2^(3e−1)`, `0 < asin x − x < 2^(3e−2)` for `|x| < 2^e ≤ 1/2`, and `0 < csc x − 1/x < 2^(e−2)`, `0 < 1/x − cot x < 2^(e−1)` for `x = 2^(e−1)`.
- The argument is reduced modulo `π/2`, `|x| = kπ/2 + r`, `|r| ≤ π/4`: with π to `W = w + e + 16` bits after the point (the working precision `w` and the exponent `e` of `x`), `N = ⌊|x|·2^(W+1)⌋` divided by `P = ⌊π·2^W⌋`, the quotient rounded to nearest. The error of `r` is at most `k·2^(ε_π − 1)` for π's error `2^ε_π`, plus a unit of `N` when it dropped bits. When `r` comes out smaller than the bits it must keep, near a multiple of `π/2`, the bits the cancellation took are read off `r` and the reduction is done again with that many more, not a fixed number. Below 1 the argument is its own `r`.
- To 512 bits after the point the sine and the cosine of `r` come from arrays, with no allocation: `r = j/64 + ρ`, `0 ≤ ρ < 1/64`, `sin(j/64)` and `cos(j/64)` from a table computed once at 576 bits (within 160 ulps there, one ulp at 512 bits), the series of `ρ` by Horner's rule with the coefficients `1/k!` of a table made when the crate is compiled (within 3.1 ulps: two each step, damped by `ρ² < 2^−12`), and the formulas of the sum: within 16 ulps. For `j = 0`, `sin r = r·Σ(−1)^n·ρ^(2n)/(2n + 1)!` with the whole `r`, relatively.
- Beyond, the argument is halved `K` times to `r' = r/2^K` of the exponent `−K`, `K ≈ (F/2)^(1/3)` for `F` bits, and `t₀ = 1 − cos r'` is its Taylor series summed by rectangular splitting (Smith): the powers `y, …, y^m` of `y = r'²` once, `m ≈ √N` even, the terms of each block of `m` combined with small integer factors exactly, and the blocks by Horner's rule with one division by a product of `2m` integers, a limb of them at a time. The bound: a power `y^i` within `i − 1` ulps, weighted `1/(2n + δ)!` in the sum, the roundings of the rule and the tail, below `3 + m/16` ulps. The doublings `t ← 4t − 2t²` keep the relative error of `t` (an error `η` becomes `η(4 + 4t + 2η) + 2`), so `F` is the precision plus `2K` bits; then `cos r = 1 − t` and `sin r = √(t(2 − t))`, within `η/√t_min + 1` ulps. An argument below `2^−K` takes the series of the sine and the cosine as it is.
- From about 770,000 bits, binary splitting: `r` is the sum of pieces `u_k/2^(2^k)` of `2^(k−1)` bits (Brent's bit-burst method, as MPFR's `mpfr_sincos_fast`), the first piece the bits of `r` to 32 after the point (one binary splitting of a numerator of 33 bits costs about half the six for the pieces of 1, 1, 2, 4, 8 and 16 bits it stands for), the sine of each piece its Taylor series summed exactly by binary splitting and divided once (within 1.25 ulps), its cosine `√(1 − s²)` (within 3), and the pieces put together by the formulas of the sum, each adding at most 6.3 ulps. (The threshold moved from 410,000 bits: the doublings, on the faster products of the merged numbers, are ahead to about 800,000 bits.)
- The tangent and the reciprocals are quotients of the sine and the cosine, `tan(kπ/2 + r)` being `tan r` for an even `k` and `−cot r` for an odd one, with the bound of a quotient `|a/b − ã/b̃| ≤ 2(|ã| + α)β/|b̃|² + α/|b̃|`.
- The arctangent of `z` in `(0, 1]` is Newton's iteration on `tan y = z`, `y ← y − cos y·(sin y − z·cos y)`, at doubling precisions from the `f64` arctangent, the last step's bound read off its own sine and cosine: for `y = θ + ε`, `sin ε = (sin y − z·cos y)·cos θ`, so `|ε| ≤ (π/2)·|sin y − z·cos y|`, and the exact step leaves at most `z·ε² + (2/3)|ε|³`. Below `2^−K`, its series `z·Σ(−1)^n·z^(2n)/(2n + 1)` by rectangular splitting, within `J(3m + 1) + 1` ulps over `J` blocks of `m`.
- From about 205,000 bits, the bit burst on the point `1 + iz`, whose angle is `atan z`: the angle is taken off in pieces `θ_k` of doubling lengths, `(re, im) ← (re, im)·(cos θ_k − i·sin θ_k)`, the sine of each piece by the binary splitting of its series and its cosine `√(1 − s²)`, as for the sine; the first piece is the `f64` arctangent cut to 32 bits (one binary splitting), and each next one is read off the point itself, `atan(im/re) = im/re` within `|im/re|³/3` (a quotient of the piece's bits and 96 more; the only long division is the last, of two thirds of the bits), so that the residual angle `φ` falls below `2^−2^k`. The pieces stop when `φ < 2^(1 − F/3)`, and the rest is `t = im/re` within `t³/3`: the last third of the pieces of the sine's bit burst is not needed. The bound: the modulus starts at `√(1 + z²) ≥ 1` and stays above `0.99`, a factor `ŵ` within `e_s + e_c` ulps of `e^(−iθ_k)` (1.25 and 3, as for the sine) turns the angle by at most that, the four products cut add an angle below `3/0.99` ulps; the pieces are exact dyadics, the last quotient within two ulps.
- `atan x = π/2 − atan(1/x)` above 1; `atan2(y, x)` is `atan(|y|/|x|)` or `π/2 ∓ atan(|x|/|y|)`, `π −` it for `x < 0`, the sign that of `y`; `asin x` is `atan(|x|/s)` or `π/2 − atan(s/|x|)` with `s = √(1 − x²)`, whichever ratio is at most 1, and `acos x` likewise (`atan(s/x)` near 1, `π − atan(s/|x|)` near −1), `1 − x²` computed exactly, so that nothing cancels near `±1`. No sum of a multiple of `π/2` and an arctangent cancels: each is above `π/4`.

### Constants

- Each constant `C` is computed as a fixed-point number `X ≈ C·2^w`, an integer within 2 of `C·2^w` (`floor(C·2^w)` itself from the tables), and rounded once. The bits of `X` decide the rounding when those below the precision, from the second bit above the error to the bit below the rounding bit, are not all equal: no number of the precision and no midpoint between two then lies between `X` and `C`, so `C` rounds as `X` does, and the result's limbs are copied from `X` (kept shifted so that its top bit is the top of a limb: a rounding copies limbs without shifting them). Otherwise, rarely, Ziv's loop computes `X` again at a precision half as large again.
- π comes from the Chudnovskys' series (1988), `1/π = 12·Σ (−1)^k (6k)! (13591409 + 545140134k) / ((3k)! (k!)³ 640320^(3k + 3/2))`, 47.11 bits a term; `e = Σ 1/k!`; `ln 2` from Zuniga's series of the Ramanujan type (2025, his equation 18), `ln 2 = (1/2160)·Σ (1497 + 1794k)·Π_(j ≤ k) j(2j − 1)/(216(6j + 1)(6j + 5))`, 11.92 bits a term; `G` from the Hessami Pilehroods' short series (2010), `G = (1/2)·Σ_(k ≥ 1) (580k² − 184k + 15)/(9(6k − 1)²(6k − 5)²)·Π_(j < k) 32j³(2j − 1)/(9(6j − 1)²(6j − 5)²)`, 7.51 bits a term. The series are those of y-cruncher's formula files and FLINT, in their form `(c_Q + c_P·Σ P(k)/Q(k)·Π_(j<k) R(j)/Q(j)) / c_D` for integer polynomials `P`, `Q`, `R`.
- The series are summed by binary splitting: over a range of terms, `T = T₁Q₂ + R₁T₂`, `Q = Q₁Q₂` and `R = R₁R₂`, `R` not computed on the right edge of the tree, whose leaves are blocks of 16 terms evaluated from the right with products by the polynomials' values (up to four limbs, by Horner's rule). The zero bits at the bottom of the leaves' numbers are kept as an exponent (`Q` of Chudnovsky's series holds 15 of them a term), so that the products above do not carry them. The series of positive terms (all but π's) cut their numbers to the working precision and 40 bits more once they outgrow it, with a bound of the relative error in units of the last bit kept: 2 for each cut, the sum of the factors' bounds and one more for a product, the larger of the terms' for a sum. Only the top bits of the large numbers near the root count for the quotient `T/Q`, and their products are made on those alone; π's tree is about twice the working precision at its root and is kept exact.
- The tail after `N` terms is bounded from a bound `r < 1` of the ratio of consecutive terms proved for each series (`2^−46` for π, `1/36` for `G`, `4·10^−4` for `ln 2`, `1/2` for `e`): it is at most `|t_(N+1)|/(1 − r)`, and `log2|t_(N+1)|` is bounded from the polynomials' values in floating point with the rounding errors of Horner's rule added. `N` is the first count whose tail is below a quarter of a unit.
- The final quotient `(c_Q·Q + c_P·T)/(c_D·Q)` cuts both operands to the quotient's bits and 4 more (within `1/4` of the quotient), then divides them by the integer division of [numbers](div.md) (Newton's method for long operands): `X` within 1.25 of the truncated sum's value, within `1/4` more from the cuts in the tree, within `1/4` more from the tail. π is `426880·√10005·Q/(13591409·Q + T)`: the quotient to `w + 9` bits times `floor(√10005·2^(w+9))` (the integer square root of numbers), within 0.44 of `π·2^w` before the floor.
- `ln 10 = 3·ln 2 + ln(5/4)`, `ln(5/4)` by Zuniga's series for `log(u/v)` (21.7 bits a term), each part to 3 bits more and within 2 of their units: the sum within 1 unit, its floor within 2.
- `γ` comes from the Brent–McMillan algorithm B3 (1980), `γ = S₀/I₀ − K₀/I₀² − log m`, with `I₀ = Σ (m^k/k!)²`, `S₀ = Σ (m^k/k!)²·H_k` and `K₀ = (1/(4m))·Σ_(k<2m) ((2k)!)³/((k!)⁴(16m)^(2k))`, within `24e^(−8m)` (Brent and Johansson, Math. Comp. 84, 2015, the bound MPFR uses). `S₀` and `I₀` come from one splitting over the dual numbers `a + bε`, `ε² = 0`, of `F(ε) = Σ m^(2k)/Π_(j≤k) (j + ε)²` (`I₀ = F(0)`, `S₀ = −F′(0)/2`), as FLINT's new `mp_real` does it: `D = Π (k + 1 + ε)` of half the size stands for `Q = D²`, and `m²`'s power of two is a shift. `m` is the cheapest number from `(w + 4.6)·ln 2/8` to half as much again whose primes are among 2, 3, 5 and 7, and `log m` is a combination of `ln 2` (cached) or of the logarithms of a few ratios near 1 (`9/8`, `256/243`, …, `2401/2400`, `4375/4374`, FLINT's sets), each by Zuniga's series for `log(u/v)`. `K₀/I₀²`, about `2^(−w/2)`, is computed to half the bits. `N` is at least `4.970626m + 1` and `2m` and enough for the tails of `S₀` and `I₀` (at most `2u_N·H_N` and `2u_N`, from Stirling's bounds) to move `S₀/I₀` by less than a unit; each part is within 2 units of `2^−(w+3)` and the sum's floor at `w` within `1 + 7/8`.
- The tables hold `floor(C·2^16384)`: π's and `ln 2`'s are those of the [partition numbers](sequences.md); `e`'s, `γ`'s and `G`'s were computed by these series at 16,448 bits and checked against MPFR 4.2; `ln 10`'s comes from MPFR's `mpfr_log` and the tests check it against the series. The partition numbers and the Bernoulli numbers take π beyond the table from here.

### Complex numbers

- Sums, differences and products by a real number are `BigFloat`'s own, part by part (into the parts of the result for the forms into a number). A product's parts `ac − bd` and `ad + bc` are each the exact sum of two exact products, rounded once. For parts of one limb the products are exact in 128 bits and each part is their sum by the two-limb addition in registers; for parts of at most two limbs the products are exact in 256 bits and each part is their sum in a window of 384 bits, `y` shifted right by the difference of the exponents, exact unless the shift passes 128 bits, when the bits shifted out are a sticky bit and a difference is taken one unit less (strictly below the value by less than a unit, its top bit kept within one, so that no halfway point of a precision to 128 bits lies between it and two units more), and then rounded once in registers. Longer parts take both products exactly in the thread's working memory and their sum by the crate's window addition, which reads of them only the limbs the rounding needs (MPFR's `mpfr_fmma`); a zero part takes the exact sum on an accumulator in two's complement, which gives the zero of IEEE 754's sums. From 1500 bits in every part, of exponents within 32 of each other, by three products, `ad`, `bc` and `(a + b)(c − d)`, so that `ac − bd = (a + b)(c − d) + ad − bc`, still exact. The square's parts are `a² − b²` and `2ab`, rounded once, the same way.
- A quotient's parts `(ac + bd)/(c² + d²)` and `(bc − ad)/(c² + d²)` are exact sums divided exactly and rounded once (the quotient with two bits more than the precision and its remainder the sticky bit), as long as those sums are at most `4p + 4096` bits long, `p` the longest part, and the result has fewer than 2000 bits. Longer sums (parts of very different exponents), and results of 2000 bits and more, where two quotients of cut sums cost less than two exact ones of `2p`-bit sums, go by Ziv's strategy: each sum cut to a working precision `w` is within `2^(2−w)` of it, relatively (a term below `2^(e − w − 4)` of the other, `e` that one's exponent, is left out), the quotient of the cut sums cut to `w` bits within `2^(1−w)` of theirs, so the result is within `5.1·2^(1−w)` relatively and within `2^(e+4−w)` for a quotient below `2^e`; when two attempts do not decide (an exact or halfway result), the exact quotient.
- The norm is the exact sum of the exact squares, rounded once, as the product's parts are. The absolute value is the square root of the exact sum, rounded once. To 127 bits the sum is exact on the accumulator (a vector the thread keeps) and its root is that of its top 256 bits in registers, the shift of even parity with the exponent of its lowest bit: `⌊√⌊x⌋⌋ = ⌊√x⌋` for a real `x ≥ 0`, so the root of 128 bits is the floor of the exact one, and the remainder and the bits cut off are its sticky bit, below the round bit. From 128 bits the squares are exact in the working memory, their sum is cut toward zero to `2p + 4` bits at least by the window addition (moved by an even power of two into the exponent range), and when the cut is not exact, the number halfway between it and the next stands for it: the squares of the numbers of `p` bits near the root and of their halfway points are multiples of `2^(2(e − p − 1))`, which no interval of the cut's last place holds, so the root of that number, by the crate's square root of limbs, rounds as the root of the sum does.
- The square root `t ± u·i` (or `u ± t·i` for `a < 0`), `t = √((|z| + |a|)/2)`, `u = |b|/(2t)`, is computed at a working precision `w`, every step cut toward zero to `w` bits, below its exact value by less than `η = 2^(1−w)` of it: `a²` and `b²`, their sum, its root `|z|`, `|z| + |a|`, `t` and `u`. All terms being above zero, `a² + b²` is within `2η`, `|z|` within `2η`, the sum with `|a|` within `3η`, `t` within `2.5η` and `u` within `2.6η`, so a part below `2^e` is within `2^(e+3−w)`, and `2^(e+4−w)` is the bound. For parts of at most two limbs and results to 113 bits each step is the real one- and two-limb arithmetic in registers at `w = min(p + guard, 127)`, 14 guard bits at least, and each part is decided in registers: both ends of its interval round alike and the result lies outside it. Otherwise (and when a part is not decided, about one time in 2^8 at 113 bits) the steps are the real operations into two numbers of the thread's pool at `w = p + 2 log2 p + 16`, for parts of exponents within `2^60` of zero, and a part is decided when the interval holds no multiple of half a unit of the last place of `p` bits: no number of the precision, no halfway point between two and no power of two. A part that is a dyadic number would hold the interval at every precision: when neither decides, the old way, Ziv's strategy on the exact squares and, after the first attempt, the root tried exactly, `a² + b² = s²` and `(s + |a|)/2 = t²` by integer square roots, then `u = |b|/(2t)` exactly, which is dyadic when `t` is. The check needs `|a|` and `|b|` within `2^(p+4)` of each other: a Pythagorean pair up to powers of two has its odd parts above `m + n`, which bounds the ratio of its legs by `2^(p+1)`.
- Parts whose exponents are far apart are decided before Ziv's strategy, which would need about `2(e_a − e_b)` bits (or `e_b − e_a`) when a part of the root is a number of the precision: with `p` the larger precision of the result and `p_a`, `p_b` the parts', for `e_a − e_b > P/2 + 4`, `P = 2p + p_a + 2p_b + 32`, `t` lies in `(√|a|, √|a|·(1 + ρ²/8))` and `u` in `(u₀·(1 − ρ²/8), u₀)`, `ρ = |b|/|a|` and `u₀ = |b|/(2√|a|)`; for `e_b − e_a > P + 7`, `P = 2p + p_b + 32`, `t` lies in `(s, s·(1 + ρ))` and `u` in `(s·(1 − ρ), s)`, `ρ = |a|/|b|` and `s = √(|b|/2)`. The square root of a rational near 1 whose denominator is below `2^n` lies further than `2^−(n + 2p + 8)` of every number of `p + 1` bits unless it is one, so `√|a|`, `u₀` and `s`, cut toward zero to `P` bits at exponents near zero, decide each part: a cut that is not exact rounds as a value strictly between it and one unit more, an exact one as a value just above it (`t`) or just below it (`u`).

### Functions of complex numbers

- Every function is MPC's formula on real functions at a working precision `w`, each value with a bound of its error in units of its last place and an exponent of 128 bits, so that an intermediate result may lie beyond the exponent range; each operation rounds toward zero to about `w + 16` bits and adds the unit it drops (a product `|ãb̃ − ab| ≤ |ã|β + |b̃|α + αβ`, a quotient within `(4/3)(ρ_a + ρ_b)` relatively for relative errors `ρ_b ≤ 1/4`, a square root within `α/√ã`, a sum exact at the cut of the larger term less `w + 24` bits, a term below `2^(e − w − 16)` going into the bound whole), and Ziv's strategy rounds each part as soon as its approximation decides it, the working precision growing by half each attempt. The real functions are the module's own: the exponential's kernel with its exponent unbounded (`e^x` within a few units of `2^−w` relatively for `|x| < 2^62`, beyond that over- or underflowing by the sign), the reduction modulo `π/2` and the sine and cosine kernels, the logarithm's kernel (with the bits of `x − 1` it needs), the arctangent's kernel and its quadrants for `atan2`, and `π`, `ln 2`, `ln 10` within `2^(1−w)`.
- `e^(x + yi) = e^x·cos y + i·e^x·sin y`; `ln z = ln|z| + i·atan2(y, x)`, `ln|z| = ln(x² + y²)/2`, and for a larger part of exponent 0 or 1 `ln(1 + d)/2` with `d = x² + y² − 1` exact (the larger square less 1, then the smaller square added exactly unless it lies more than `w + 16` bits below), so that nothing cancels; `log10 z = ln z / ln 10`. For parts that are finite and not zero the sign of the argument is `y`'s, without the `atan2` to 2 bits that the special points take. `sin z = sin x·cosh y + i·cos x·sinh y`, `cos z = cos x·cosh y − i·sin x·sinh y` (`sinh y` from `u = e^|y| − 1` as `(u + u/(u + 1))/2` below `1/2`); `tan z = (sin x·cos x + i·sinh y·cosh y)/(cos² x + sinh² y)`, the denominator a sum of squares. `sin_cos` shares the real sines, cosines and hyperbolic functions of each working precision between the two, lent rather than copied.
- `asin` and `acos` by Hull, Fairgrieve and Tang's formulas: `A = (|z + 1| + |z − 1|)/2`, `Re asin = atan2(x, √((A − |x|)(A + |x|)))`, `Im asin = ±ln(1 + (A − 1) + √((A − 1)(A + 1)))`, with `A − 1` and `A − |x|` sums of terms that are not negative (`|z + 1| − (|x| + 1) = y²/(|z + 1| + |x| + 1)` and its kind); `atan z = atan2(2x, 1 − x² − y²)/2 + (i/4)·ln(1 + 4|y|/(x² + (|y| − 1)²))`. `z^w = e^(w·ln z)`, `w·ln z` within the absolute error of its parts, so that a large one takes more bits.
- A value near a number of the precision would take Ziv's strategy as many bits as it lies near: for tiny arguments, `|x|, |y| < 2^−(p/2 + 4)`, each part is its first term (`x`, `y`, `1`, `−xy`) and a difference computed on its own (`(e^x − 1)·cos y − 2·sin²(y/2)` for `Re e^z`, `(sin x − x)·cosh y + 2x·sinh²(y/2)` for `Re sin z`, by their series), rounded beside the first term when the difference lies below its last place, and otherwise added to it; the inverse functions' parts by the series' first term, `Re f = x(1 + c(x² − 3y²) + θ)` with `|θ| ≤ 2|z|^4` for `asin` (`c = 1/6`) and `atan` (`c = −1/3`), as `|Re z^n| ≤ n|z|^(n−1)|x|` for an odd `n`. The tangent's imaginary part near `±1` is `±1` and `∓(cos 2x + e^(−2|y|))/(2(cos² x + sinh² y))`, of the sign of `−cos 2x` and below `2^(−2^62)` when `|y| ≥ 2^62`. The argument of a tiny dyadic ratio `y/x` (`x > 0`) is rounded beside it, `0 < q − atan q < q³/3`.
- The exact powers are MPC's `mpc_pow_exact`: for a real `w = m·2^e`, `m` odd, `z = (c + di)·2^k` takes `−e` square roots among the Gaussian integers (`c + di` is a square when `c² + d²` and then `(c + √(c² + d²))/2` are), `1/(c + di)` is dyadic when `c² + d²` is a power of 2, and `(c + di)^m` comes by squarings and products; as long as the Gaussian integers have at most twice the precision and 64 bits more (besides the spread of the parts' exponents), and otherwise again after each attempt of Ziv's strategy with twice as many, as MPC does. `powi` with `|n| ≥ 3` (no zero part) goes by binary powering, MPC's `mpc_pow_usi`: the correctly rounded squares and products into one number at the working precision and `bits(n) + 16` more (`sqr_assign_round` and `mul_assign_round`, the first square that of `z` itself), within `(2k − 1)·2^(2−w)·1.1` of `z^k` in the norm (by induction over the squarings and products), so within `2^(bits(n) + 5 − w)` relatively in all, the exact power after an attempt that does not decide.

### Matrices of floats

- A product's element is the exact dot product of a row and a column rounded once (`BigFloat::dot_round`'s): for elements of at most two limbs on an accumulator of 768 bits on the stack, each product of 256 bits (128 for one limb) added at its place in two's complement, and the top 256 bits of the sum and a sticky bit rounded in registers; otherwise, or when the terms span more, on limbs. For complex elements each part is the exact sum of the exact products of the parts, the same way.
- `det`, `solve` and `inverse` factor the matrix by Doolittle's method with partial pivoting: each element of `U`, and of `L` before its division by the pivot, is one exact dot product of the elements before it, rounded once at the working precision, and the substitutions likewise. For elements of at most two limbs (a working precision to 128 bits) the factorisation and the substitutions run on numbers in registers, every dot product on the stack accumulator and every quotient by the one- or two-limb division, with the same results and no vector for each element; a result beyond the exponent range or a sum too wide for the accumulator falls back to the factorisation on limbs, which works in place on a copy of the matrix at the working precision, its vectors kept by the thread for the next.
- `rank` eliminates with complete pivoting and counts the pivots above the tolerance; the norms are exact sums of absolute values, or of squares under one square root, each rounded once.

## Differences from MPFR and MPC

Deliberate, and documented in the functions and the tests:

- `Display` writes the shortest decimal that reads back, as `f64` does, rather than the fixed number of digits of `mpfr_get_str`, which `to_digits` and `to_string_radix` with `None` give. The exponent range is fixed at `±(2^62 − 1)`, MPFR's widest, and there are no flags: the direction of each rounding is returned with its result.
- A `BigFloat` NaN has no sign: `√(−∞ + NaN·i)` is `+0 + ∞i` here, where MPC gives the infinite part the sign bit of MPFR's NaN.
- A zero part of a complex result that rounds down follows IEEE 754's sums, `−0` for zeros of both signs or terms that cancel. MPC 1.4.1 differs only in the mode toward −∞: it makes `(+0) + (+0)` `−0` in its products with a zero part, and mixed or cancelling zeros `+0` in its quotients (which form the numerators toward zero) and in its long products, so that MPC itself differs between its two products; its `mpc_sqr`, and `mpc_pow_si` with 2 through it, make the cancelling real part of `(t + ti)²` `+0`.
- MPC's `mul_infinite` reads `Im(x)` where `Re(y)` is meant in one test, so that `(∞ + i)·(∞ − ∞i)` is `∞ − ∞i` in MPC, where Annex G and this module have `∞ + NaN·i`.
- MPC's `mpc_sqr` makes `NaN ± ∞i` and `±∞ + NaN·i` NaN in both parts; the square here is the product of the number by itself, an infinity.
- At the ends of the exponent range every part of a complex quotient is rounded by its mode; MPC's quotient uses a heuristic there.

Errors found in MPFR 4.2.2 and MPC 1.4.1, each checked against a computation at a higher precision (Python's `decimal`, mpmath, or the library itself); the tests check this module's values there, and the test vectors generated with the libraries leave those cases out or take them from another of the library's functions:

- MPFR's `mpfr_cosh` just below the overflow threshold (`|x|` a few ulps below `2^62·ln 2` at 90 bits) overflows, to an infinity or the largest number by the mode, because its `e^|x|` overflows, while `cosh x = sinh|x| + e^−|x|` is representable; MPFR's own `mpfr_sinh` gives the finite value.
- MPFR's `mpfr_asinh` for `|x| > 2^(2^60)`, where `x²` overflows, is a few ulps off: `asinh(2^(2^62 − 2))` at 64 bits is `0xb17217f7d1cf79a6p-2` in MPFR, and the exact value `(2^62 − 1)·ln 2` is `0xb17217f7d1cf79a9.04…p-2`, which rounds to `0x…79a9`.
- MPC's `mpc_log` takes `mpfr_abs` into a variable shorter than `x` as exact, so that `ln|z|` of a `|z|` near 1 with a long part is wrong: for `x = 0x42137fe0013ffffff·2^−239` and `y = −(1 + 2^−62)` of 67 bits, `ln|z|` is about `2.168·10^−19` (`2^−62`), and MPC gives `3.7·10^−105`. `mpc_log10` likewise.
- MPC's `mpc_tan` of `x + yi` for `|y|` beyond the range of its exponential (about `2^60.5`) takes `|Im tan z| < 1` always, where it is above 1 exactly when `cos 2x < 0`, and gives the real part's sign from the signs of overflowed parts. `mpc_tanh` likewise.
- MPC's `mpc_pow` beyond the exponent range rounds an underflowing part to zero in every mode, also away from zero; loses the exact zero part of real and imaginary powers there; and takes the sign of a part from `w·arg z` to only 64 bits more than its exponent, so that it can be wrong when `w·arg z` lies near a multiple of `π/2`: for `z = −0xffbf40fc001f9720000000052a25·2^−107 − 0x15100000000008e5dffff800001a3f61·2^369·i` and `n = 31312633701166011`, `cos(n·arg z) = +2.97·10^−131`, and the real part of `z^n` is `+∞` here and the largest negative number in MPC.
- Near the overflow threshold MPC's `mpc_exp`, and `mpc_sin` and `mpc_cos` through `cosh`, overflow in both parts as soon as `e^x` does: `e^(2^62·ln 2 − 2 + 0.5i)` is finite in both parts here.

## Measured

Measured on an Apple M4 Max on 2026-11-14 between 13:47 and 15:31 (load averages 1.2 to 2.2 at the start of each harness, 2.7 at the first one as the Miri runs ended; nothing else running), the text's tables again at 16:48, after the change to the scan of the digits (load averages 3.9 and falling, as the tests had just ended), and the constants on 2026-11-13 between 21:55 and 00:20 (load averages 1.2 to 1.9), none of their code having changed since, one thread, `env -u MallocNanoZone`, fat LTO: the median of three passes of rounds with fresh arguments each round and the contenders' order rotated, in ns, µs, ms or s a call. The clock read between 3.1 and 4.4 GHz from one table to the next, so the times compare within a table and the ratios across them. The references are MPFR 4.2.2 (Homebrew's static archives, called directly through FFI, with its exponent range set to that of `BigFloat`), Arb in FLINT 3.6 (its balls enclose the value but are not correctly rounded, so its column is for the time only), and GNU MPC 1.4.1 for the complex numbers; rug 1.30 on the same MPFR took the time of MPFR to within a few per cent everywhere, so it has no column. MPFR and MPC write into a variable that exists and `a` returns a new number, unless a table says otherwise. The arguments have the precision of the result and full mantissas, of exponents from −2 to 1 unless a table says otherwise, and every result is rounded to nearest; each harness checks its results against the reference's before it times them. The ratio is `a`'s time over the reference's: below 1, `a` is faster.

### Arithmetic

Into an existing number (`set_add` and the others against `mpfr_add (r, a, b, rnd)` and the others; `x·x` is `set_mul(&x, &x)` against `mpfr_sqr`). A precision `p (operands q)` is a result of `p` bits from operands of `q`; the sums take operands of both signs:

| Operation | Precision | `a` | MPFR | `a`/MPFR |
| --- | --- | ---: | ---: | ---: |
| `set_add` | 24 | 5 ns | 5.1 ns | 0.98 |
|  | 53 | 5 ns | 5.1 ns | 0.98 |
|  | 64 | 4.6 ns | 5.6 ns | 0.82 |
|  | 113 | 6.9 ns | 5.8 ns | 1.19 |
|  | 128 | 6.3 ns | 6.3 ns | 1.00 |
|  | 256 | 12.7 ns | 10.2 ns | 1.25 |
|  | 1,000 | 17.7 ns | 14.6 ns | 1.21 |
|  | 10,000 | 53.9 ns | 93.4 ns | 0.58 |
|  | 100,000 | 483 ns | 993 ns | 0.49 |
|  | 1,000,000 | 6.15 µs | 9.07 µs | 0.68 |
|  | 53 (operands 1,000) | 14 ns | 14.8 ns | 0.95 |
|  | 53 (operands 100,000) | 12.5 ns | 12.1 ns | 1.03 |
|  | 1,000 (operands 53) | 12.5 ns | 16.1 ns | 0.78 |
|  | 1,000 (operands 100,000) | 55.1 ns | 274 ns | 0.20 |
|  | 100,000 (operands 1,000,000) | 1.27 µs | 2.19 µs | 0.58 |
| `set_sub` | 24 | 4.9 ns | 5.5 ns | 0.89 |
|  | 53 | 4.8 ns | 5.7 ns | 0.84 |
|  | 64 | 4.4 ns | 6.1 ns | 0.72 |
|  | 113 | 6.7 ns | 6.4 ns | 1.05 |
|  | 128 | 6.1 ns | 6.8 ns | 0.90 |
|  | 256 | 12.3 ns | 10.6 ns | 1.16 |
|  | 1,000 | 18.2 ns | 15.4 ns | 1.18 |
|  | 10,000 | 53.9 ns | 87.2 ns | 0.62 |
|  | 100,000 | 375 ns | 875 ns | 0.43 |
|  | 1,000,000 | 6.27 µs | 10.3 µs | 0.61 |
|  | 53 (operands 1,000) | 14.6 ns | 15.4 ns | 0.95 |
|  | 53 (operands 100,000) | 13.6 ns | 269 ns | 0.05 |
|  | 1,000 (operands 53) | 12.2 ns | 15.9 ns | 0.77 |
|  | 1,000 (operands 100,000) | 53.7 ns | 142 ns | 0.38 |
|  | 100,000 (operands 1,000,000) | 1.28 µs | 2.11 µs | 0.61 |
| `set_mul` | 24 | 4 ns | 5.4 ns | 0.74 |
|  | 53 | 3.5 ns | 5.4 ns | 0.65 |
|  | 64 | 3.4 ns | 5.4 ns | 0.63 |
|  | 113 | 5.1 ns | 5.9 ns | 0.86 |
|  | 128 | 5 ns | 11 ns | 0.45 |
|  | 256 | 12 ns | 21.8 ns | 0.55 |
|  | 1,000 | 92.4 ns | 101 ns | 0.91 |
|  | 10,000 | 3.24 µs | 3.24 µs | 1.00 |
|  | 100,000 | 48.6 µs | 93.5 µs | 0.52 |
|  | 1,000,000 | 559 µs | 1.42 ms | 0.40 |
|  | 53 (operands 1,000) | 30.5 ns | 98.4 ns | 0.31 |
|  | 53 (operands 100,000) | 25.8 ns | 43.6 ns | 0.59 |
|  | 1,000 (operands 53) | 7.3 ns | 10.4 ns | 0.70 |
|  | 1,000 (operands 100,000) | 109 ns | 116 ns | 0.94 |
|  | 100,000 (operands 1,000,000) | 48.6 µs | 94.8 µs | 0.51 |
| `x·x` | 24 | 3.5 ns | 5.1 ns | 0.69 |
|  | 53 | 3.1 ns | 5.1 ns | 0.61 |
|  | 64 | 2.9 ns | 5.3 ns | 0.55 |
|  | 113 | 4.1 ns | 5.5 ns | 0.75 |
|  | 128 | 4.1 ns | 13.2 ns | 0.31 |
|  | 256 | 12.3 ns | 18.7 ns | 0.66 |
|  | 1,000 | 79 ns | 79.4 ns | 0.99 |
|  | 10,000 | 2.36 µs | 2.14 µs | 1.11 |
|  | 100,000 | 37.2 µs | 60.7 µs | 0.61 |
|  | 1,000,000 | 415 µs | 1.21 ms | 0.34 |
|  | 53 (operands 1,000) | 31.6 ns | 77.6 ns | 0.41 |
|  | 1,000 (operands 100,000) | 87.5 ns | 100 ns | 0.87 |
| `set_div` | 24 | 6 ns | 6.2 ns | 0.97 |
|  | 53 | 6 ns | 6.2 ns | 0.97 |
|  | 64 | 6.3 ns | 6.8 ns | 0.93 |
|  | 113 | 11.3 ns | 8.8 ns | 1.28 |
|  | 128 | 12.1 ns | 27 ns | 0.45 |
|  | 256 | 70.7 ns | 62.4 ns | 1.13 |
|  | 1,000 | 180 ns | 220 ns | 0.82 |
|  | 10,000 | 4.58 µs | 5.44 µs | 0.84 |
|  | 100,000 | 123 µs | 192 µs | 0.64 |
|  | 1,000,000 | 1.36 ms | 4.01 ms | 0.34 |
|  | 53 (operands 1,000) | 21.1 ns | 17.4 ns | 1.21 |
|  | 53 (operands 100,000) | 18.3 ns | 15 ns | 1.22 |
|  | 1,000 (operands 53) | 48.2 ns | 36.6 ns | 1.32 |
|  | 1,000 (operands 100,000) | 217 ns | 214 ns | 1.01 |
|  | 100,000 (operands 1,000,000) | 123 µs | 197 µs | 0.62 |
| `set_mul_add` | 24 | 8.9 ns | 11.6 ns | 0.77 |
|  | 53 | 8.8 ns | 12.2 ns | 0.72 |
|  | 64 | 8.8 ns | 21.6 ns | 0.41 |
|  | 113 | 16 ns | 22.8 ns | 0.70 |
|  | 128 | 15.7 ns | 23.9 ns | 0.66 |
|  | 256 | 29 ns | 30.2 ns | 0.96 |
|  | 1,000 | 137 ns | 113 ns | 1.21 |
|  | 10,000 | 3.51 µs | 3.64 µs | 0.96 |
|  | 100,000 | 50.3 µs | 109 µs | 0.46 |
|  | 1,000,000 | 573 µs | 1.43 ms | 0.40 |
|  | 53 (operands 1,000) | 98.8 ns | 110 ns | 0.89 |
|  | 53 (operands 100,000) | 48.7 µs | 109 µs | 0.44 |
|  | 1,000 (operands 53) | 13.4 ns | 23.2 ns | 0.58 |
|  | 1,000 (operands 100,000) | 48.6 µs | 109 µs | 0.44 |
|  | 100,000 (operands 1,000,000) | 562 µs | 1.42 ms | 0.40 |
| `set_sqrt` | 24 | 7.1 ns | 6.2 ns | 1.15 |
|  | 53 | 7.1 ns | 6.2 ns | 1.15 |
|  | 64 | 12.7 ns | 6.8 ns | 1.87 |
|  | 113 | 15.7 ns | 11.8 ns | 1.33 |
|  | 128 | 21 ns | 63 ns | 0.33 |
|  | 256 | 100 ns | 98.5 ns | 1.02 |
|  | 1,000 | 258 ns | 264 ns | 0.97 |
|  | 10,000 | 4.36 µs | 3.67 µs | 1.19 |
|  | 100,000 | 116 µs | 139 µs | 0.84 |
|  | 1,000,000 | 1.44 ms | 3.23 ms | 0.45 |
|  | 53 (operands 1,000) | 7.1 ns | 13.6 ns | 0.52 |
|  | 1,000 (operands 100,000) | 246 ns | 255 ns | 0.97 |

With a new result (`add_round` and the others against `mpfr_init2`, the operation and `mpfr_clear`):

| Operation | Precision | `a` | MPFR | `a`/MPFR |
| --- | --- | ---: | ---: | ---: |
| `add_round` | 24 | 13.5 ns | 17.1 ns | 0.79 |
|  | 53 | 13.6 ns | 17.3 ns | 0.79 |
|  | 64 | 12.9 ns | 17.2 ns | 0.75 |
|  | 113 | 15.9 ns | 18.1 ns | 0.88 |
|  | 128 | 15.2 ns | 18.4 ns | 0.83 |
|  | 256 | 20.6 ns | 22.6 ns | 0.91 |
|  | 1,000 | 27.6 ns | 25.9 ns | 1.07 |
|  | 10,000 | 79 ns | 109 ns | 0.73 |
|  | 100,000 | 618 ns | 1.01 µs | 0.61 |
|  | 1,000,000 | 7.83 µs | 9.14 µs | 0.86 |
|  | 53 (operands 1,000) | 22.9 ns | 27 ns | 0.85 |
|  | 53 (operands 100,000) | 21.4 ns | 23 ns | 0.93 |
|  | 1,000 (operands 53) | 22.8 ns | 29.3 ns | 0.78 |
|  | 1,000 (operands 100,000) | 62.2 ns | 286 ns | 0.22 |
|  | 100,000 (operands 1,000,000) | 1.29 µs | 2.2 µs | 0.58 |
| `sub_round` | 24 | 13.6 ns | 17 ns | 0.80 |
|  | 53 | 13.4 ns | 16.9 ns | 0.79 |
|  | 64 | 13 ns | 17.4 ns | 0.75 |
|  | 113 | 15.7 ns | 18.2 ns | 0.86 |
|  | 128 | 15.1 ns | 18.7 ns | 0.81 |
|  | 256 | 20.4 ns | 22.5 ns | 0.91 |
|  | 1,000 | 26.7 ns | 27.5 ns | 0.97 |
|  | 10,000 | 74.5 ns | 103 ns | 0.72 |
|  | 100,000 | 522 ns | 891 ns | 0.59 |
|  | 1,000,000 | 8.06 µs | 10.4 µs | 0.78 |
|  | 53 (operands 1,000) | 23.2 ns | 27.4 ns | 0.85 |
|  | 53 (operands 100,000) | 22.3 ns | 284 ns | 0.08 |
|  | 1,000 (operands 53) | 22.5 ns | 29.1 ns | 0.77 |
|  | 1,000 (operands 100,000) | 61.9 ns | 156 ns | 0.40 |
|  | 100,000 (operands 1,000,000) | 1.29 µs | 2.14 µs | 0.60 |
| `mul_round` | 24 | 13.2 ns | 18.2 ns | 0.73 |
|  | 53 | 12.8 ns | 18 ns | 0.71 |
|  | 64 | 12.9 ns | 18.1 ns | 0.71 |
|  | 113 | 15.3 ns | 19 ns | 0.81 |
|  | 128 | 15.1 ns | 25.5 ns | 0.59 |
|  | 256 | 22.4 ns | 34.7 ns | 0.65 |
|  | 1,000 | 102 ns | 112 ns | 0.92 |
|  | 10,000 | 3.25 µs | 3.26 µs | 1.00 |
|  | 100,000 | 48.7 µs | 93.5 µs | 0.52 |
|  | 1,000,000 | 561 µs | 1.41 ms | 0.40 |
|  | 53 (operands 1,000) | 37.5 ns | 109 ns | 0.34 |
|  | 53 (operands 100,000) | 34.1 ns | 57.4 ns | 0.59 |
|  | 1,000 (operands 53) | 16.8 ns | 22.9 ns | 0.73 |
|  | 1,000 (operands 100,000) | 121 ns | 127 ns | 0.96 |
|  | 100,000 (operands 1,000,000) | 48.7 µs | 95.2 µs | 0.51 |
| `&x * &x` | 24 | 13.7 ns | 17.5 ns | 0.78 |
|  | 53 | 12.6 ns | 17.7 ns | 0.71 |
|  | 64 | 13 ns | 18 ns | 0.72 |
|  | 113 | 14.9 ns | 19.3 ns | 0.77 |
|  | 128 | 14.9 ns | 27.3 ns | 0.55 |
|  | 256 | 22.2 ns | 31.7 ns | 0.70 |
|  | 1,000 | 90.8 ns | 91.3 ns | 0.99 |
|  | 10,000 | 2.38 µs | 2.17 µs | 1.09 |
|  | 100,000 | 37.2 µs | 60.9 µs | 0.61 |
|  | 1,000,000 | 415 µs | 1.21 ms | 0.34 |
|  | 53 (operands 1,000) | 38.9 ns | 88.8 ns | 0.44 |
|  | 1,000 (operands 100,000) | 102 ns | 111 ns | 0.92 |
| `div_round` | 24 | 14.2 ns | 20.3 ns | 0.70 |
|  | 53 | 14.6 ns | 20.5 ns | 0.71 |
|  | 64 | 14.3 ns | 20.8 ns | 0.69 |
|  | 113 | 18 ns | 19.9 ns | 0.90 |
|  | 128 | 18.7 ns | 39.8 ns | 0.47 |
|  | 256 | 77.9 ns | 77.6 ns | 1.00 |
|  | 1,000 | 188 ns | 227 ns | 0.83 |
|  | 10,000 | 4.59 µs | 5.46 µs | 0.84 |
|  | 100,000 | 123 µs | 193 µs | 0.64 |
|  | 1,000,000 | 1.36 ms | 4.01 ms | 0.34 |
|  | 53 (operands 1,000) | 31.1 ns | 30.4 ns | 1.02 |
|  | 53 (operands 100,000) | 26.4 ns | 25.9 ns | 1.02 |
|  | 1,000 (operands 53) | 57.9 ns | 56.7 ns | 1.02 |
|  | 1,000 (operands 100,000) | 226 ns | 224 ns | 1.01 |
|  | 100,000 (operands 1,000,000) | 124 µs | 197 µs | 0.63 |
| `mul_add_round` | 24 | 15.8 ns | 22.7 ns | 0.70 |
|  | 53 | 15.8 ns | 23.3 ns | 0.68 |
|  | 64 | 15.8 ns | 33.2 ns | 0.48 |
|  | 113 | 21.9 ns | 35 ns | 0.63 |
|  | 128 | 21.8 ns | 36.3 ns | 0.60 |
|  | 256 | 34.9 ns | 41.6 ns | 0.84 |
|  | 1,000 | 145 ns | 124 ns | 1.17 |
|  | 10,000 | 3.5 µs | 3.65 µs | 0.96 |
|  | 100,000 | 50.4 µs | 109 µs | 0.46 |
|  | 1,000,000 | 575 µs | 1.42 ms | 0.40 |
|  | 53 (operands 1,000) | 112 ns | 122 ns | 0.92 |
|  | 53 (operands 100,000) | 48.8 µs | 109 µs | 0.45 |
|  | 1,000 (operands 53) | 20.9 ns | 35.1 ns | 0.60 |
|  | 1,000 (operands 100,000) | 48.8 µs | 110 µs | 0.45 |
|  | 100,000 (operands 1,000,000) | 563 µs | 1.42 ms | 0.40 |
| `sqrt_round` | 24 | 23.7 ns | 19.4 ns | 1.22 |
|  | 53 | 22.4 ns | 19.2 ns | 1.17 |
|  | 64 | 25.7 ns | 17.9 ns | 1.44 |
|  | 113 | 32 ns | 20.7 ns | 1.55 |
|  | 128 | 39.1 ns | 82.7 ns | 0.47 |
|  | 256 | 112 ns | 116 ns | 0.96 |
|  | 1,000 | 267 ns | 280 ns | 0.95 |
|  | 10,000 | 4.35 µs | 3.7 µs | 1.18 |
|  | 100,000 | 116 µs | 139 µs | 0.84 |
|  | 1,000,000 | 1.44 ms | 3.23 ms | 0.45 |
|  | 53 (operands 1,000) | 20.1 ns | 29.7 ns | 0.68 |
|  | 1,000 (operands 100,000) | 256 ns | 265 ns | 0.97 |

### Roots and powers

With a new result, against `mpfr_rec_sqrt`, `mpfr_cbrt`, `mpfr_rootn_ui` and `mpfr_pow_si` into an existing variable:

| Operation | Precision | `a` | MPFR | `a`/MPFR |
| --- | --- | ---: | ---: | ---: |
| `rec_sqrt_round` | 24 | 29.3 ns | 55.1 ns | 0.53 |
|  | 53 | 29.5 ns | 55.1 ns | 0.54 |
|  | 64 | 39.1 ns | 76.4 ns | 0.51 |
|  | 113 | 38.9 ns | 96.6 ns | 0.40 |
|  | 128 | 114 ns | 118 ns | 0.96 |
|  | 256 | 160 ns | 172 ns | 0.93 |
|  | 1,000 | 439 ns | 428 ns | 1.02 |
|  | 10,000 | 8.49 µs | 9.33 µs | 0.91 |
|  | 100,000 | 167 µs | 268 µs | 0.62 |
| `cbrt_round` | 24 | 59.3 ns | 120 ns | 0.49 |
|  | 53 | 58.4 ns | 188 ns | 0.31 |
|  | 64 | 91.1 ns | 211 ns | 0.43 |
|  | 113 | 105 ns | 247 ns | 0.43 |
|  | 128 | 250 ns | 270 ns | 0.92 |
|  | 256 | 325 ns | 359 ns | 0.90 |
|  | 1,000 | 685 ns | 688 ns | 1.00 |
|  | 10,000 | 9.25 µs | 8.73 µs | 1.06 |
|  | 100,000 | 220 µs | 278 µs | 0.79 |
| `nth_root_round(5)` | 24 | 83.8 ns | 178 ns | 0.47 |
|  | 53 | 81.1 ns | 237 ns | 0.34 |
|  | 64 | 134 ns | 265 ns | 0.51 |
|  | 113 | 147 ns | 324 ns | 0.45 |
|  | 128 | 335 ns | 351 ns | 0.96 |
|  | 256 | 408 ns | 463 ns | 0.88 |
|  | 1,000 | 934 ns | 988 ns | 0.95 |
|  | 10,000 | 16.5 µs | 17.5 µs | 0.95 |
|  | 100,000 | 328 µs | 526 µs | 0.62 |
| `nth_root_round(100)` | 24 | 984 ns | 1.02 µs | 0.97 |
|  | 53 | 1.03 µs | 1.79 µs | 0.57 |
|  | 64 | 1.59 µs | 2.16 µs | 0.74 |
|  | 113 | 1.6 µs | 3.95 µs | 0.41 |
|  | 128 | 1.61 µs | 4.82 µs | 0.33 |
|  | 256 | 3.03 µs | 7.6 µs | 0.40 |
|  | 1,000 | 6.33 µs | 43.4 µs | 0.15 |
| `powi_round(7)` | 24 | 34.7 ns | 61.1 ns | 0.57 |
|  | 53 | 34.3 ns | 59.1 ns | 0.58 |
|  | 64 | 33.9 ns | 68.5 ns | 0.49 |
|  | 113 | 50.2 ns | 75.7 ns | 0.66 |
|  | 128 | 51.1 ns | 85.3 ns | 0.60 |
|  | 256 | 67 ns | 112 ns | 0.60 |
|  | 1,000 | 358 ns | 413 ns | 0.87 |
|  | 10,000 | 11.5 µs | 11.1 µs | 1.04 |
|  | 100,000 | 175 µs | 373 µs | 0.47 |
| `powi_round(-3)` | 24 | 41.3 ns | 89.7 ns | 0.46 |
|  | 53 | 40.9 ns | 92.7 ns | 0.44 |
|  | 64 | 41 ns | 100 ns | 0.41 |
|  | 113 | 57.7 ns | 106 ns | 0.54 |
|  | 128 | 59.4 ns | 112 ns | 0.53 |
|  | 256 | 155 ns | 159 ns | 0.98 |
|  | 1,000 | 398 ns | 470 ns | 0.85 |
|  | 10,000 | 12.3 µs | 13.1 µs | 0.94 |
|  | 100,000 | 211 µs | 409 µs | 0.51 |
| `powi_round(1_000_000_007)` | 24 | 125 ns | 372 ns | 0.34 |
|  | 53 | 125 ns | 386 ns | 0.32 |
|  | 64 | 126 ns | 418 ns | 0.30 |
|  | 113 | 196 ns | 547 ns | 0.36 |
|  | 128 | 196 ns | 867 ns | 0.23 |
|  | 256 | 361 ns | 1.19 µs | 0.30 |
|  | 1,000 | 3.39 µs | 4.31 µs | 0.79 |

### Conversions

Against `mpfr_get_d`, `mpfr_set_d`, `mpfr_floor`, `mpfr_rint` and `mpfr_get_si`, MPFR writing into an existing number: `set_f64`, `set_floor` and `set_rint` write into one too, and `from_f64_round`, `floor` and `rint_round` make a new one; `floor` of a number with half its bits after the point, `rint_round` and `set_rint` to half the precision:

| Operation | Precision | `a` | MPFR | `a`/MPFR |
| --- | --- | ---: | ---: | ---: |
| `to_f64_round` | 24 | 1.6 ns | 5.4 ns | 0.30 |
|  | 53 | 1.6 ns | 5.4 ns | 0.30 |
|  | 64 | 2.3 ns | 4.7 ns | 0.48 |
|  | 113 | 2.3 ns | 4.8 ns | 0.48 |
|  | 128 | 2.2 ns | 4.8 ns | 0.46 |
|  | 256 | 2.2 ns | 4.7 ns | 0.48 |
|  | 1,000 | 2.3 ns | 4.8 ns | 0.47 |
|  | 10,000 | 2.3 ns | 4.8 ns | 0.48 |
|  | 100,000 | 2.1 ns | 4.9 ns | 0.43 |
|  | 1,000,000 | 2.3 ns | 4.7 ns | 0.50 |
| `set_f64` | 24 | 2.5 ns | 8.1 ns | 0.31 |
|  | 53 | 2.8 ns | 5.6 ns | 0.50 |
|  | 64 | 2.8 ns | 7.5 ns | 0.37 |
|  | 113 | 3 ns | 7.7 ns | 0.39 |
|  | 256 | 3.5 ns | 8.1 ns | 0.43 |
|  | 1,000 | 4.5 ns | 7.5 ns | 0.60 |
|  | 10,000 | 16.1 ns | 17.9 ns | 0.90 |
|  | 100,000 | 113 ns | 115 ns | 0.99 |
|  | 1,000,000 | 782 ns | 575 ns | 1.36 |
| `from_f64_round` | 24 | 12.5 ns | 8.1 ns | 1.54 |
|  | 53 | 9.9 ns | 5.6 ns | 1.78 |
|  | 64 | 10.1 ns | 7.5 ns | 1.36 |
|  | 113 | 10 ns | 7.7 ns | 1.30 |
|  | 256 | 10.3 ns | 8.1 ns | 1.27 |
|  | 1,000 | 10.7 ns | 7.5 ns | 1.43 |
|  | 10,000 | 25.6 ns | 17.9 ns | 1.43 |
|  | 100,000 | 151 ns | 115 ns | 1.32 |
|  | 1,000,000 | 711 ns | 575 ns | 1.24 |
| `set_floor` | 24 | 7.3 ns | 8.5 ns | 0.86 |
|  | 53 | 7.2 ns | 8.2 ns | 0.87 |
|  | 64 | 7.2 ns | 8.4 ns | 0.85 |
|  | 113 | 8.7 ns | 8.7 ns | 1.00 |
|  | 128 | 8.2 ns | 8.5 ns | 0.97 |
|  | 256 | 8.7 ns | 9.1 ns | 0.96 |
|  | 1,000 | 8.2 ns | 8.7 ns | 0.95 |
|  | 10,000 | 15.5 ns | 16.3 ns | 0.95 |
|  | 100,000 | 123 ns | 112 ns | 1.10 |
|  | 1,000,000 | 786 ns | 839 ns | 0.94 |
| `floor` | 24 | 17 ns | 8.5 ns | 1.99 |
|  | 53 | 16.9 ns | 8.2 ns | 2.06 |
|  | 64 | 16.9 ns | 8.4 ns | 2.00 |
|  | 113 | 18 ns | 8.7 ns | 2.08 |
|  | 128 | 17.8 ns | 8.5 ns | 2.09 |
|  | 256 | 19.2 ns | 9.1 ns | 2.12 |
|  | 1,000 | 19.5 ns | 8.7 ns | 2.26 |
|  | 10,000 | 37 ns | 16.3 ns | 2.27 |
|  | 100,000 | 202 ns | 112 ns | 1.80 |
|  | 1,000,000 | 1.8 µs | 839 ns | 2.15 |
| `set_rint` | 24 | 8.2 ns | 8.6 ns | 0.96 |
|  | 53 | 8.3 ns | 8.5 ns | 0.97 |
|  | 64 | 8.4 ns | 8.6 ns | 0.98 |
|  | 113 | 7.6 ns | 8.5 ns | 0.89 |
|  | 128 | 7.9 ns | 8.4 ns | 0.94 |
|  | 256 | 8.2 ns | 8.6 ns | 0.95 |
|  | 1,000 | 9 ns | 8.3 ns | 1.08 |
|  | 10,000 | 14.9 ns | 13.3 ns | 1.12 |
|  | 100,000 | 71.3 ns | 71.1 ns | 1.00 |
|  | 1,000,000 | 525 ns | 527 ns | 0.99 |
| `rint_round(p/2)` | 24 | 19 ns | 8.6 ns | 2.22 |
|  | 53 | 19.1 ns | 8.5 ns | 2.24 |
|  | 64 | 19 ns | 8.6 ns | 2.21 |
|  | 113 | 18.5 ns | 8.5 ns | 2.16 |
|  | 128 | 18.7 ns | 8.4 ns | 2.24 |
|  | 256 | 18.8 ns | 8.6 ns | 2.19 |
|  | 1,000 | 18.8 ns | 8.3 ns | 2.26 |
|  | 10,000 | 34.4 ns | 13.3 ns | 2.59 |
|  | 100,000 | 103 ns | 71.1 ns | 1.44 |
|  | 1,000,000 | 630 ns | 527 ns | 1.20 |
| `to_i64_round` | 53 | 10.8 ns | 23.8 ns | 0.45 |
|  | 113 | 10.6 ns | 23.3 ns | 0.45 |
|  | 1,000 | 10.8 ns | 23.6 ns | 0.46 |

### Text

Writing against `mpfr_get_str`: `to_digits(10, None)` gives `1 + ceil(p·log10 2)` digits, as `mpfr_get_str` with `n = 0`; the exponents are those of the other tables, or up to `±2^60` where marked:

| Operation | Precision | `a` | MPFR | `a`/MPFR |
| --- | --- | ---: | ---: | ---: |
| `to_digits(10, None)` | 24 | 36.2 ns | 207 ns | 0.17 |
|  | 53 | 46.2 ns | 219 ns | 0.21 |
|  | 64 | 62.1 ns | 226 ns | 0.27 |
|  | 113 | 80.3 ns | 261 ns | 0.31 |
|  | 128 | 159 ns | 271 ns | 0.59 |
|  | 256 | 193 ns | 354 ns | 0.55 |
|  | 1,000 | 608 ns | 976 ns | 0.62 |
|  | 10,000 | 12.1 µs | 16.5 µs | 0.73 |
|  | 100,000 | 299 µs | 430 µs | 0.69 |
|  | 1,000,000 | 4.84 ms | 9.56 ms | 0.51 |
| `to_digits(10, None)`, exponents to `±2^60` | 53 | 299 ns | 1.33 µs | 0.23 |
|  | 113 | 428 ns | 1.6 µs | 0.27 |
|  | 1,000 | 4.92 µs | 6.61 µs | 0.75 |
|  | 10,000 | 129 µs | 151 µs | 0.86 |
| `to_digits(10, Some(20))` | 1,000 | 117 ns | 185 ns | 0.63 |
|  | 10,000 | 113 ns | 180 ns | 0.63 |
|  | 100,000 | 104 ns | 175 ns | 0.60 |
|  | 1,000,000 | 105 ns | 167 ns | 0.63 |

`to_string()`, the shortest digits that read back, against rug's `to_string()`, the digits of `mpfr_get_str`:

| Operation | Precision | `a` | rug | `a`/rug |
| --- | --- | ---: | ---: | ---: |
| `to_string()` | 24 | 97.8 ns | 298 ns | 0.33 |
|  | 53 | 133 ns | 310 ns | 0.43 |
|  | 64 | 150 ns | 326 ns | 0.46 |
|  | 113 | 173 ns | 355 ns | 0.49 |
|  | 128 | 363 ns | 377 ns | 0.96 |
|  | 256 | 404 ns | 448 ns | 0.90 |
|  | 1,000 | 885 ns | 1.09 µs | 0.81 |
|  | 10,000 | 13 µs | 16.8 µs | 0.77 |
|  | 100,000 | 309 µs | 426 µs | 0.73 |
|  | 1,000,000 | 4.91 ms | 9.6 ms | 0.51 |

Reading against `mpfr_strtofr`, texts of the digits given: at their worth of bits, `ceil(n·log2 10)`, at 53 bits, at their worth with decimal exponents up to `±10^17`, and short texts and other radices (1 to 19 decimal digits at 53 bits; 20 and 100 digits in radix 3, 7 and 36 at their worth of bits):

| Operation | Digits | `a` | MPFR | `a`/MPFR |
| --- | --- | ---: | ---: | ---: |
| `from_str_radix`, `ceil(n·log2 10)` bits | 20 | 58.1 ns | 121 ns | 0.48 |
|  | 50 | 71.3 ns | 183 ns | 0.39 |
|  | 100 | 128 ns | 290 ns | 0.44 |
|  | 1,000 | 2.18 µs | 3.09 µs | 0.71 |
|  | 10,000 | 66 µs | 89.1 µs | 0.74 |
|  | 100,000 | 1.05 ms | 2.11 ms | 0.50 |
|  | 1,000,000 | 14.1 ms | 33.7 ms | 0.42 |
| `from_str_radix`, 53 bits | 20 | 49.4 ns | 112 ns | 0.44 |
|  | 50 | 52.3 ns | 132 ns | 0.40 |
|  | 100 | 56.2 ns | 162 ns | 0.35 |
|  | 1,000 | 103 ns | 606 ns | 0.17 |
|  | 10,000 | 433 ns | 4.94 µs | 0.09 |
|  | 100,000 | 4.1 µs | 49.1 µs | 0.08 |
|  | 1,000,000 | 40.1 µs | 486 µs | 0.08 |
| `from_str_radix`, exponents to `±10^17` | 20 | 519 ns | 1.92 µs | 0.27 |
|  | 100 | 2.02 µs | 3.37 µs | 0.60 |
|  | 1,000 | 23 µs | 40.2 µs | 0.57 |
| `from_str_radix`, short and other radices | 1 in radix 10 | 25.7 ns | 77.8 ns | 0.33 |
|  | 5 in radix 10 | 31.3 ns | 85.8 ns | 0.36 |
|  | 10 in radix 10 | 30.6 ns | 93 ns | 0.33 |
|  | 15 in radix 10 | 38 ns | 102 ns | 0.37 |
|  | 17 in radix 10 | 38.1 ns | 107 ns | 0.35 |
|  | 19 in radix 10 | 40.3 ns | 108 ns | 0.37 |
|  | 20 in radix 3 | 50.8 ns | 107 ns | 0.47 |
|  | 100 in radix 3 | 153 ns | 250 ns | 0.61 |
|  | 20 in radix 7 | 64.1 ns | 152 ns | 0.42 |
|  | 100 in radix 7 | 181 ns | 275 ns | 0.66 |
|  | 20 in radix 36 | 80.8 ns | 120 ns | 0.67 |
|  | 100 in radix 36 | 276 ns | 355 ns | 0.78 |

### Exponentials, logarithms and powers

Against `mpfr_exp`, `mpfr_exp2`, `mpfr_exp10`, `mpfr_expm1`, `mpfr_log`, `mpfr_log2`, `mpfr_log10`, `mpfr_log1p` and `mpfr_pow`, and Arb's `arb_exp` and the others at the precision and the guard bits of Ziv's first attempt:

| Operation | Precision | `a` | MPFR | `a`/MPFR | Arb |
| --- | --- | ---: | ---: | ---: | ---: |
| `exp` | 24 | 331 ns | 354 ns | 0.93 | 67.4 ns |
|  | 53 | 334 ns | 508 ns | 0.66 | 84.4 ns |
|  | 64 | 339 ns | 545 ns | 0.62 | 89.4 ns |
|  | 113 | 451 ns | 752 ns | 0.60 | 119 ns |
|  | 128 | 447 ns | 789 ns | 0.57 | 120 ns |
|  | 256 | 828 ns | 1.21 µs | 0.68 | 223 ns |
|  | 1,000 | 3.5 µs | 5.18 µs | 0.68 | 1.86 µs |
|  | 10,000 | 245 µs | 278 µs | 0.88 | 63.4 µs |
|  | 100,000 | 5.52 ms | 8.32 ms | 0.66 | 3.49 ms |
|  | 1,000,000 | 94.9 ms | 187 ms | 0.51 | 75.3 ms |
| `exp2` | 24 | 391 ns | 494 ns | 0.79 | 128 ns |
|  | 53 | 397 ns | 647 ns | 0.61 | 157 ns |
|  | 64 | 402 ns | 705 ns | 0.57 | 155 ns |
|  | 113 | 521 ns | 881 ns | 0.59 | 202 ns |
|  | 128 | 530 ns | 944 ns | 0.56 | 207 ns |
|  | 256 | 886 ns | 1.36 µs | 0.65 | 312 ns |
|  | 1,000 | 3.46 µs | 5.36 µs | 0.65 | 2.03 µs |
|  | 10,000 | 248 µs | 278 µs | 0.89 | 67.4 µs |
|  | 100,000 | 5.59 ms | 8.55 ms | 0.65 | 3.58 ms |
| `exp10` | 24 | 436 ns | 842 ns | 0.52 | 146 ns |
|  | 53 | 441 ns | 1.03 µs | 0.43 | 179 ns |
|  | 64 | 442 ns | 1.19 µs | 0.37 | 185 ns |
|  | 113 | 562 ns | 1.81 µs | 0.31 | 223 ns |
|  | 128 | 560 ns | 2.11 µs | 0.27 | 227 ns |
|  | 256 | 973 ns | 3.33 µs | 0.29 | 337 ns |
|  | 1,000 | 3.57 µs | 11.8 µs | 0.30 | 2.07 µs |
|  | 10,000 | 249 µs | 448 µs | 0.56 | 67 µs |
|  | 100,000 | 5.57 ms | 15.1 ms | 0.37 | 3.59 ms |
| `exp_m1` | 24 | 420 ns | 467 ns | 0.90 | 81.9 ns |
|  | 53 | 424 ns | 610 ns | 0.69 | 108 ns |
|  | 64 | 427 ns | 652 ns | 0.66 | 109 ns |
|  | 113 | 548 ns | 828 ns | 0.66 | 144 ns |
|  | 128 | 679 ns | 886 ns | 0.77 | 149 ns |
|  | 256 | 1.07 µs | 1.32 µs | 0.81 | 252 ns |
|  | 1,000 | 3.53 µs | 5.27 µs | 0.67 | 1.89 µs |
|  | 10,000 | 245 µs | 278 µs | 0.88 | 63.2 µs |
|  | 100,000 | 5.55 ms | 8.39 ms | 0.66 | 3.49 ms |
| `ln` | 24 | 178 ns | 375 ns | 0.47 | 61.3 ns |
|  | 53 | 186 ns | 431 ns | 0.43 | 103 ns |
|  | 64 | 187 ns | 469 ns | 0.40 | 107 ns |
|  | 113 | 269 ns | 1.15 µs | 0.23 | 139 ns |
|  | 128 | 280 ns | 1.18 µs | 0.24 | 141 ns |
|  | 256 | 654 ns | 2.15 µs | 0.30 | 214 ns |
|  | 1,000 | 5.57 µs | 6.96 µs | 0.80 | 1.56 µs |
|  | 10,000 | 168 µs | 162 µs | 1.04 | 78.5 µs |
|  | 100,000 | 4.7 ms | 7.08 ms | 0.66 | 3.95 ms |
|  | 1,000,000 | 70.5 ms | 166 ms | 0.42 | 87 ms |
| `log2` | 24 | 190 ns | 458 ns | 0.41 | 101 ns |
|  | 53 | 206 ns | 581 ns | 0.35 | 156 ns |
|  | 64 | 208 ns | 586 ns | 0.36 | 161 ns |
|  | 113 | 296 ns | 1.32 µs | 0.22 | 210 ns |
|  | 128 | 305 ns | 1.3 µs | 0.24 | 212 ns |
|  | 256 | 681 ns | 2.28 µs | 0.30 | 311 ns |
|  | 1,000 | 5.79 µs | 7.14 µs | 0.81 | 1.79 µs |
|  | 10,000 | 173 µs | 167 µs | 1.04 | 84 µs |
|  | 100,000 | 4.83 ms | 6.83 ms | 0.71 | 4.18 ms |
| `log10` | 24 | 269 ns | 944 ns | 0.29 | 113 ns |
|  | 53 | 282 ns | 948 ns | 0.30 | 165 ns |
|  | 64 | 285 ns | 1.07 µs | 0.27 | 166 ns |
|  | 113 | 382 ns | 2.21 µs | 0.17 | 215 ns |
|  | 128 | 391 ns | 2.48 µs | 0.16 | 218 ns |
|  | 256 | 799 ns | 4.43 µs | 0.18 | 320 ns |
|  | 1,000 | 6.02 µs | 13.8 µs | 0.44 | 1.79 µs |
|  | 10,000 | 174 µs | 326 µs | 0.53 | 84 µs |
|  | 100,000 | 4.85 ms | 13.2 ms | 0.37 | 4.16 ms |
| `ln_1p` | 24 | 238 ns | 374 ns | 0.64 | 84.1 ns |
|  | 53 | 239 ns | 410 ns | 0.58 | 121 ns |
|  | 64 | 239 ns | 430 ns | 0.55 | 126 ns |
|  | 113 | 322 ns | 1.11 µs | 0.29 | 159 ns |
|  | 128 | 327 ns | 1.12 µs | 0.29 | 159 ns |
|  | 256 | 694 ns | 2.04 µs | 0.34 | 229 ns |
|  | 1,000 | 6.16 µs | 6.88 µs | 0.89 | 1.57 µs |
|  | 10,000 | 169 µs | 164 µs | 1.03 | 78.2 µs |
|  | 100,000 | 4.76 ms | 6.49 ms | 0.73 | 3.99 ms |
| `pow` | 24 | 624 ns | 918 ns | 0.68 | 177 ns |
|  | 53 | 658 ns | 1.1 µs | 0.60 | 266 ns |
|  | 64 | 672 ns | 1.28 µs | 0.52 | 276 ns |
|  | 113 | 870 ns | 2.08 µs | 0.42 | 339 ns |
|  | 128 | 887 ns | 2.42 µs | 0.37 | 344 ns |
|  | 256 | 1.66 µs | 3.68 µs | 0.45 | 525 ns |
|  | 1,000 | 9.06 µs | 12.8 µs | 0.71 | 3.59 µs |
|  | 10,000 | 415 µs | 443 µs | 0.94 | 146 µs |
|  | 100,000 | 10.3 ms | 15.3 ms | 0.68 | 7.61 ms |

### Hyperbolic functions

Against `mpfr_sinh` and the others, and Arb's:

| Operation | Precision | `a` | MPFR | `a`/MPFR | Arb |
| --- | --- | ---: | ---: | ---: | ---: |
| `sinh` | 24 | 484 ns | 530 ns | 0.91 | 150 ns |
|  | 53 | 486 ns | 604 ns | 0.80 | 191 ns |
|  | 64 | 613 ns | 727 ns | 0.84 | 190 ns |
|  | 113 | 616 ns | 843 ns | 0.73 | 251 ns |
|  | 128 | 744 ns | 918 ns | 0.81 | 259 ns |
|  | 256 | 1.12 µs | 1.36 µs | 0.82 | 383 ns |
|  | 1,000 | 3.78 µs | 5.54 µs | 0.68 | 2.17 µs |
|  | 10,000 | 252 µs | 282 µs | 0.89 | 69.5 µs |
|  | 100,000 | 5.63 ms | 8.58 ms | 0.66 | 3.68 ms |
| `cosh` | 24 | 488 ns | 544 ns | 0.90 | 136 ns |
|  | 53 | 490 ns | 653 ns | 0.75 | 169 ns |
|  | 64 | 617 ns | 711 ns | 0.87 | 171 ns |
|  | 113 | 620 ns | 874 ns | 0.71 | 230 ns |
|  | 128 | 744 ns | 902 ns | 0.82 | 235 ns |
|  | 256 | 1.12 µs | 1.36 µs | 0.82 | 357 ns |
|  | 1,000 | 3.75 µs | 5.48 µs | 0.68 | 2.13 µs |
|  | 10,000 | 252 µs | 286 µs | 0.88 | 69.6 µs |
|  | 100,000 | 5.64 ms | 8.59 ms | 0.66 | 3.67 ms |
| `tanh` | 24 | 500 ns | 544 ns | 0.92 | 162 ns |
|  | 53 | 496 ns | 694 ns | 0.71 | 200 ns |
|  | 64 | 624 ns | 694 ns | 0.90 | 199 ns |
|  | 113 | 634 ns | 885 ns | 0.72 | 259 ns |
|  | 128 | 752 ns | 954 ns | 0.79 | 263 ns |
|  | 256 | 1.14 µs | 1.42 µs | 0.80 | 407 ns |
|  | 1,000 | 3.98 µs | 5.6 µs | 0.71 | 2.23 µs |
|  | 10,000 | 252 µs | 283 µs | 0.89 | 69 µs |
|  | 100,000 | 5.66 ms | 8.63 ms | 0.66 | 3.73 ms |
| `asinh` | 24 | 392 ns | 896 ns | 0.44 | 193 ns |
|  | 53 | 423 ns | 746 ns | 0.57 | 278 ns |
|  | 64 | 422 ns | 1.05 µs | 0.40 | 278 ns |
|  | 113 | 674 ns | 1.38 µs | 0.49 | 345 ns |
|  | 128 | 674 ns | 1.53 µs | 0.44 | 348 ns |
|  | 256 | 1.29 µs | 2.64 µs | 0.49 | 465 ns |
|  | 1,000 | 6.95 µs | 7.33 µs | 0.95 | 2.16 µs |
|  | 10,000 | 180 µs | 168 µs | 1.07 | 85 µs |
|  | 100,000 | 4.98 ms | 6.72 ms | 0.74 | 4.2 ms |
| `acosh` | 24 | 359 ns | 587 ns | 0.61 | 148 ns |
|  | 53 | 394 ns | 673 ns | 0.59 | 235 ns |
|  | 64 | 396 ns | 745 ns | 0.53 | 240 ns |
|  | 113 | 587 ns | 1.59 µs | 0.37 | 308 ns |
|  | 128 | 617 ns | 1.53 µs | 0.40 | 310 ns |
|  | 256 | 1.13 µs | 2.53 µs | 0.45 | 427 ns |
|  | 1,000 | 6.35 µs | 8.18 µs | 0.78 | 2.11 µs |
|  | 10,000 | 175 µs | 168 µs | 1.04 | 84.7 µs |
|  | 100,000 | 4.91 ms | 6.72 ms | 0.73 | 4.18 ms |
| `atanh` | 24 | 353 ns | 1.11 µs | 0.32 | 159 ns |
|  | 53 | 362 ns | 1.03 µs | 0.35 | 197 ns |
|  | 64 | 364 ns | 1.14 µs | 0.32 | 207 ns |
|  | 113 | 520 ns | 1.57 µs | 0.33 | 246 ns |
|  | 128 | 535 ns | 1.71 µs | 0.31 | 259 ns |
|  | 256 | 1.03 µs | 2.61 µs | 0.39 | 370 ns |
|  | 1,000 | 6.34 µs | 7.54 µs | 0.84 | 1.98 µs |
|  | 10,000 | 174 µs | 169 µs | 1.03 | 84.5 µs |
|  | 100,000 | 4.83 ms | 6.71 ms | 0.72 | 4.21 ms |

### Circular functions

Against `mpfr_sin` and the others, and Arb's; arguments in `[1/4, 4)` for the sine, the cosine, the tangent and their reciprocals, `[1/16, 1)` for `asin` and `acos`, `[1/8, 4)` for `atan` and `atan2`:

| Operation | Precision | `a` | MPFR | `a`/MPFR | Arb |
| --- | --- | ---: | ---: | ---: | ---: |
| `sin` | 24 | 214 ns | 458 ns | 0.47 | 69.6 ns |
|  | 53 | 213 ns | 564 ns | 0.38 | 86.6 ns |
|  | 64 | 211 ns | 573 ns | 0.37 | 101 ns |
|  | 113 | 283 ns | 740 ns | 0.38 | 114 ns |
|  | 128 | 280 ns | 757 ns | 0.37 | 134 ns |
|  | 256 | 520 ns | 1.17 µs | 0.44 | 266 ns |
|  | 1,000 | 3.82 µs | 4.5 µs | 0.85 | 1.9 µs |
|  | 10,000 | 185 µs | 280 µs | 0.66 | 85.6 µs |
|  | 100,000 | 6.96 ms | 13.8 ms | 0.51 | 4.87 ms |
|  | 1,000,000 | 243 ms | 291 ms | 0.84 | 139 ms |
| `cos` | 24 | 213 ns | 318 ns | 0.67 | 68.5 ns |
|  | 53 | 213 ns | 372 ns | 0.57 | 84.9 ns |
|  | 64 | 212 ns | 387 ns | 0.55 | 97.9 ns |
|  | 113 | 285 ns | 494 ns | 0.58 | 112 ns |
|  | 128 | 283 ns | 528 ns | 0.54 | 132 ns |
|  | 256 | 526 ns | 874 ns | 0.60 | 258 ns |
|  | 1,000 | 3.68 µs | 3.99 µs | 0.92 | 1.89 µs |
|  | 10,000 | 181 µs | 276 µs | 0.66 | 84.9 µs |
|  | 100,000 | 7.03 ms | 13.8 ms | 0.51 | 4.86 ms |
| `tan` | 24 | 357 ns | 528 ns | 0.68 | 125 ns |
|  | 53 | 359 ns | 640 ns | 0.56 | 148 ns |
|  | 64 | 353 ns | 660 ns | 0.53 | 155 ns |
|  | 113 | 457 ns | 919 ns | 0.50 | 180 ns |
|  | 128 | 449 ns | 964 ns | 0.47 | 223 ns |
|  | 256 | 719 ns | 1.38 µs | 0.52 | 376 ns |
|  | 1,000 | 4.38 µs | 5.35 µs | 0.82 | 2.25 µs |
|  | 10,000 | 193 µs | 290 µs | 0.66 | 91.6 µs |
|  | 100,000 | 7.2 ms | 14.1 ms | 0.51 | 5.12 ms |
| `sin_cos` | 24 | 282 ns | 468 ns | 0.60 | 83 ns |
|  | 53 | 281 ns | 572 ns | 0.49 | 98.8 ns |
|  | 64 | 280 ns | 588 ns | 0.48 | 110 ns |
|  | 113 | 362 ns | 754 ns | 0.48 | 126 ns |
|  | 128 | 354 ns | 785 ns | 0.45 | 146 ns |
|  | 256 | 615 ns | 1.21 µs | 0.51 | 286 ns |
|  | 1,000 | 4.12 µs | 4.59 µs | 0.90 | 2.02 µs |
|  | 10,000 | 188 µs | 279 µs | 0.67 | 85.4 µs |
|  | 100,000 | 7.08 ms | 13.7 ms | 0.52 | 4.85 ms |
| `sec` | 53 | 318 ns | 466 ns | 0.68 | 140 ns |
|  | 1,000 | 4 µs | 4.29 µs | 0.93 | 2.11 µs |
| `csc` | 53 | 318 ns | 642 ns | 0.50 | 140 ns |
|  | 1,000 | 4.15 µs | 5.23 µs | 0.79 | 2.11 µs |
| `cot` | 53 | 361 ns | 732 ns | 0.49 | 148 ns |
|  | 1,000 | 4.33 µs | 5.65 µs | 0.77 | 2.23 µs |
| `atan` | 24 | 520 ns | 710 ns | 0.73 | 57.2 ns |
|  | 53 | 512 ns | 1.01 µs | 0.51 | 76.2 ns |
|  | 64 | 511 ns | 1.01 µs | 0.51 | 101 ns |
|  | 113 | 939 ns | 1.59 µs | 0.59 | 110 ns |
|  | 128 | 933 ns | 1.87 µs | 0.50 | 137 ns |
|  | 256 | 1.65 µs | 4.98 µs | 0.33 | 232 ns |
|  | 1,000 | 8.14 µs | 15.4 µs | 0.53 | 1.9 µs |
|  | 10,000 | 278 µs | 360 µs | 0.77 | 110 µs |
|  | 100,000 | 11 ms | 14 ms | 0.79 | 5.65 ms |
|  | 1,000,000 | 241 ms | 335 ms | 0.72 | 154 ms |
| `asin` | 24 | 701 ns | 795 ns | 0.88 | 224 ns |
|  | 53 | 733 ns | 1.15 µs | 0.64 | 248 ns |
|  | 64 | 737 ns | 1.14 µs | 0.64 | 305 ns |
|  | 113 | 1.18 µs | 1.98 µs | 0.60 | 332 ns |
|  | 128 | 1.18 µs | 2.09 µs | 0.57 | 390 ns |
|  | 256 | 1.98 µs | 5.19 µs | 0.38 | 564 ns |
|  | 1,000 | 8.7 µs | 18.3 µs | 0.48 | 2.58 µs |
|  | 10,000 | 290 µs | 373 µs | 0.78 | 124 µs |
|  | 100,000 | 11.2 ms | 14 ms | 0.80 | 6.14 ms |
| `acos` | 24 | 795 ns | 848 ns | 0.94 | 246 ns |
|  | 53 | 806 ns | 1.18 µs | 0.68 | 272 ns |
|  | 64 | 801 ns | 1.16 µs | 0.69 | 328 ns |
|  | 113 | 1.27 µs | 2.12 µs | 0.60 | 368 ns |
|  | 128 | 1.25 µs | 2.12 µs | 0.59 | 422 ns |
|  | 256 | 2.04 µs | 5.29 µs | 0.39 | 603 ns |
|  | 1,000 | 8.87 µs | 18.5 µs | 0.48 | 2.63 µs |
|  | 10,000 | 288 µs | 373 µs | 0.77 | 127 µs |
|  | 100,000 | 11.3 ms | 13.9 ms | 0.81 | 6.15 ms |
| `atan2` | 24 | 629 ns | 845 ns | 0.74 | 118 ns |
|  | 53 | 616 ns | 1.13 µs | 0.54 | 139 ns |
|  | 64 | 611 ns | 1.17 µs | 0.52 | 161 ns |
|  | 113 | 1.06 µs | 1.96 µs | 0.54 | 189 ns |
|  | 128 | 1.06 µs | 2 µs | 0.53 | 214 ns |
|  | 256 | 1.81 µs | 5.09 µs | 0.36 | 343 ns |
|  | 1,000 | 8.4 µs | 18.2 µs | 0.46 | 2.18 µs |
|  | 10,000 | 283 µs | 369 µs | 0.77 | 117 µs |
|  | 100,000 | 11.1 ms | 14.1 ms | 0.79 | 5.98 ms |

Arguments `m·2^k` of a large exponent, the reduction taking π to `k` bits and more. For `k = 10^6` Arb returns `[−1, 1]` at once, which is not a result:

| Operation | Precision | `a` | MPFR | `a`/MPFR | Arb |
| --- | --- | ---: | ---: | ---: | ---: |
| `sin`, `m·2^100` | 53 | 279 ns | 802 ns | 0.35 | 107 ns |
|  | 256 | 603 ns | 1.45 µs | 0.42 | 286 ns |
| `sin`, `m·2^10000` | 53 | 7.15 µs | 9.06 µs | 0.79 | 9.79 µs |
|  | 256 | 7.63 µs | 10.3 µs | 0.74 | 9.79 µs |
| `sin`, `m·2^1000000` | 53 | 1.39 ms | 3.88 ms | 0.36 | 9 ns |
|  | 256 | 1.38 ms | 3.95 ms | 0.35 | 9.2 ns |
| `cos`, `m·2^100` | 53 | 281 ns | 654 ns | 0.43 | 107 ns |
|  | 256 | 603 ns | 1.15 µs | 0.52 | 283 ns |
| `cos`, `m·2^10000` | 53 | 7.14 µs | 9.2 µs | 0.78 | 9.68 µs |
|  | 256 | 7.62 µs | 9.99 µs | 0.76 | 9.81 µs |
| `cos`, `m·2^1000000` | 53 | 1.38 ms | 4.05 ms | 0.34 | 9.2 ns |
|  | 256 | 1.38 ms | 3.94 ms | 0.35 | 8.8 ns |
| `tan`, `m·2^100` | 53 | 427 ns | 873 ns | 0.49 | 167 ns |
|  | 256 | 800 ns | 1.64 µs | 0.49 | 408 ns |
| `tan`, `m·2^10000` | 53 | 7.29 µs | 9.27 µs | 0.79 | 9.27 µs |
|  | 256 | 7.83 µs | 10.5 µs | 0.74 | 9.78 µs |
| `tan`, `m·2^1000000` | 53 | 1.38 ms | 3.94 ms | 0.35 | 12.5 ns |
|  | 256 | 1.38 ms | 3.95 ms | 0.35 | 12.5 ns |

### Constants

The first call in a fresh process, when no library has the constant yet: up to 16,384 bits `a` cuts it from its table, beyond it sums the series; MPFR computes from the first bit (`e` as `mpfr_exp` of 1 and `ln 10` as `mpfr_log` of 10, as it has neither constant), and Arb has tables of a few thousand bits for some constants. At 10⁸ bits one call each; MPFR's γ and `G` would take hours there:

| Operation | Precision | `a` | MPFR | `a`/MPFR | Arb |
| --- | --- | ---: | ---: | ---: | ---: |
| `pi` | 53 | 2.58 µs | 5.33 µs | 0.48 | 7.54 µs |
|  | 64 | 2.25 µs | 4 µs | 0.56 | 6.92 µs |
|  | 113 | 2.29 µs | 8.08 µs | 0.28 | 6.62 µs |
|  | 256 | 2.25 µs | 10 µs | 0.23 | 11.2 µs |
|  | 1,000 | 2.25 µs | 13.4 µs | 0.17 | 11.1 µs |
|  | 10,000 | 2.42 µs | 91.1 µs | 0.03 | 211 µs |
|  | 100,000 | 1.4 ms | 3.26 ms | 0.43 | 1.77 ms |
|  | 1,000,000 | 19.7 ms | 85.8 ms | 0.23 | 28.1 ms |
|  | 10,000,000 | 304 ms | 1.44 s | 0.21 | 449 ms |
|  | 100,000,000 | 4.44 s | 26.6 s | 0.17 | 6.48 s |
| `e` | 53 | 2.29 µs | 14.6 µs | 0.16 | 47 µs |
|  | 64 | 2.25 µs | 14.5 µs | 0.16 | 42.4 µs |
|  | 113 | 2.21 µs | 18.1 µs | 0.12 | 48.4 µs |
|  | 256 | 2.17 µs | 21.9 µs | 0.10 | 52.2 µs |
|  | 1,000 | 2.12 µs | 42.5 µs | 0.05 | 65.2 µs |
|  | 10,000 | 2.38 µs | 560 µs | < 0.01 | 133 µs |
|  | 100,000 | 805 µs | 2.38 ms | 0.34 | 1 ms |
|  | 1,000,000 | 7.98 ms | 47.4 ms | 0.17 | 12.5 ms |
|  | 10,000,000 | 101 ms | 674 ms | 0.15 | 164 ms |
|  | 100,000,000 | 1.19 s | 11.7 s | 0.10 | 2.02 s |
| `ln2` | 53 | 1.79 µs | 7.75 µs | 0.23 | 11.1 µs |
|  | 64 | 1.58 µs | 7.08 µs | 0.22 | 10.5 µs |
|  | 113 | 1.79 µs | 8.83 µs | 0.20 | 10.4 µs |
|  | 256 | 1.62 µs | 13.1 µs | 0.12 | 14.8 µs |
|  | 1,000 | 1.71 µs | 28.9 µs | 0.06 | 14.8 µs |
|  | 10,000 | 1.79 µs | 270 µs | 0.01 | 264 µs |
|  | 100,000 | 2.49 ms | 6.16 ms | 0.40 | 3.07 ms |
|  | 1,000,000 | 39 ms | 139 ms | 0.28 | 55.4 ms |
|  | 10,000,000 | 619 ms | 2.91 s | 0.21 | 946 ms |
|  | 100,000,000 | 9.41 s | 50.9 s | 0.18 | 13.8 s |
| `ln10` | 53 | 1.58 µs | 13.4 µs | 0.12 | 51.1 µs |
|  | 64 | 1.58 µs | 13 µs | 0.12 | 49.3 µs |
|  | 113 | 1.62 µs | 17.9 µs | 0.09 | 51 µs |
|  | 256 | 1.58 µs | 22.6 µs | 0.07 | 54.6 µs |
|  | 1,000 | 1.62 µs | 47.3 µs | 0.03 | 129 µs |
|  | 10,000 | 1.79 µs | 510 µs | < 0.01 | 456 µs |
|  | 100,000 | 3.87 ms | 16.4 ms | 0.24 | 6.87 ms |
|  | 1,000,000 | 61 ms | 391 ms | 0.16 | 123 ms |
|  | 10,000,000 | 961 ms | 7.08 s | 0.14 | 1.93 s |
|  | 100,000,000 | 14.5 s | 124 s | 0.12 | 28.4 s |
| `euler_gamma` | 53 | 1.75 µs | 28.3 µs | 0.06 | 6.54 µs |
|  | 64 | 1.67 µs | 35 µs | 0.05 | 6.17 µs |
|  | 113 | 1.62 µs | 42.9 µs | 0.04 | 6.33 µs |
|  | 256 | 1.62 µs | 72.3 µs | 0.02 | 11.2 µs |
|  | 1,000 | 1.62 µs | 238 µs | 0.01 | 11.3 µs |
|  | 10,000 | 1.79 µs | 4.09 ms | < 0.01 | 2.47 ms |
|  | 100,000 | 32.7 ms | 102 ms | 0.32 | 47.6 ms |
|  | 1,000,000 | 511 ms | 2.15 s | 0.24 | 950 ms |
|  | 10,000,000 | 7.62 s | 42.5 s | 0.18 | 16.4 s |
|  | 100,000,000 | 120 s | – | – | 244 s |
| `catalan` | 53 | 1.79 µs | 18.3 µs | 0.10 | 110 µs |
|  | 64 | 1.71 µs | 19.2 µs | 0.09 | 112 µs |
|  | 113 | 1.67 µs | 23.2 µs | 0.07 | 116 µs |
|  | 256 | 1.71 µs | 33.7 µs | 0.05 | 120 µs |
|  | 1,000 | 1.62 µs | 91.7 µs | 0.02 | 143 µs |
|  | 10,000 | 1.75 µs | 1.56 ms | < 0.01 | 512 µs |
|  | 100,000 | 6.47 ms | 43.3 ms | 0.15 | 8.76 ms |
|  | 1,000,000 | 113 ms | 979 ms | 0.12 | 180 ms |
|  | 10,000,000 | 1.82 s | 18.6 s | 0.10 | 3.05 s |
|  | 100,000,000 | 28.7 s | – | – | 48.3 s |

A later call, the constant already computed: `a` copies the limbs of its cached value, as MPFR and Arb copy theirs; MPFR computes `exp(1)` and `log(10)` again at every call:

| Operation | Precision | `a` | MPFR | `a`/MPFR | Arb |
| --- | --- | ---: | ---: | ---: | ---: |
| `pi`, cached | 53 | 14 ns | 19 ns | 0.74 | 7 ns |
|  | 64 | 14 ns | 19 ns | 0.74 | 8 ns |
|  | 113 | 14 ns | 19 ns | 0.74 | 7 ns |
|  | 256 | 15 ns | 21 ns | 0.71 | 16 ns |
|  | 1,000 | 16 ns | 19 ns | 0.84 | 22 ns |
|  | 10,000 | 32 ns | 36 ns | 0.89 | 70 ns |
|  | 100,000 | 143 ns | 157 ns | 0.91 | 415 ns |
|  | 1,000,000 | 1.24 µs | 1.18 µs | 1.05 | 3.98 µs |
|  | 10,000,000 | 14.7 µs | 14.7 µs | 1.00 | 38.4 µs |
| `e`, cached | 53 | 14 ns | 555 ns | 0.03 | 11 ns |
|  | 64 | 14 ns | 598 ns | 0.02 | 10 ns |
|  | 113 | 14 ns | 780 ns | 0.02 | 11 ns |
|  | 256 | 15 ns | 1.19 µs | 0.01 | 20 ns |
|  | 1,000 | 15 ns | 5.34 µs | < 0.01 | 25 ns |
|  | 10,000 | 35 ns | 287 µs | < 0.01 | 68 ns |
|  | 100,000 | 145 ns | 2.42 ms | < 0.01 | 415 ns |
|  | 1,000,000 | 1.26 µs | 47 ms | < 0.01 | 3.95 µs |
|  | 10,000,000 | 14.5 µs | 672 ms | < 0.01 | 37.7 µs |
| `ln2`, cached | 53 | 14 ns | 18 ns | 0.78 | 7 ns |
|  | 64 | 14 ns | 19 ns | 0.74 | 8 ns |
|  | 113 | 14 ns | 21 ns | 0.67 | 8 ns |
|  | 256 | 15 ns | 20 ns | 0.75 | 17 ns |
|  | 1,000 | 15 ns | 20 ns | 0.75 | 21 ns |
|  | 10,000 | 31 ns | 35 ns | 0.89 | 71 ns |
|  | 100,000 | 175 ns | 174 ns | 1.01 | 412 ns |
|  | 1,000,000 | 1.21 µs | 1.19 µs | 1.02 | 3.98 µs |
|  | 10,000,000 | 14.9 µs | 14.7 µs | 1.01 | 38.4 µs |
| `ln10`, cached | 53 | 14 ns | 364 ns | 0.04 | 11 ns |
|  | 64 | 14 ns | 390 ns | 0.04 | 10 ns |
|  | 113 | 14 ns | 1.13 µs | 0.01 | 11 ns |
|  | 256 | 15 ns | 1.87 µs | 0.01 | 20 ns |
|  | 1,000 | 16 ns | 6.49 µs | < 0.01 | 25 ns |
|  | 10,000 | 32 ns | 164 µs | < 0.01 | 70 ns |
|  | 100,000 | 172 ns | 6.72 ms | < 0.01 | 414 ns |
|  | 1,000,000 | 1.22 µs | 165 ms | < 0.01 | 3.96 µs |
|  | 10,000,000 | 14.4 µs | 2.74 s | < 0.01 | 37.8 µs |
| `euler_gamma`, cached | 53 | 14 ns | 19 ns | 0.74 | 8 ns |
|  | 64 | 14 ns | 18 ns | 0.78 | 8 ns |
|  | 113 | 14 ns | 20 ns | 0.70 | 8 ns |
|  | 256 | 15 ns | 20 ns | 0.75 | 17 ns |
|  | 1,000 | 16 ns | 19 ns | 0.84 | 21 ns |
|  | 10,000 | 32 ns | 34 ns | 0.94 | 70 ns |
|  | 100,000 | 186 ns | 148 ns | 1.26 | 415 ns |
|  | 1,000,000 | 1.21 µs | 1.17 µs | 1.03 | 3.94 µs |
|  | 10,000,000 | 14.7 µs | 14.5 µs | 1.01 | 37.5 µs |
| `catalan`, cached | 53 | 14 ns | 19 ns | 0.74 | 11 ns |
|  | 64 | 14 ns | 19 ns | 0.74 | 11 ns |
|  | 113 | 14 ns | 20 ns | 0.70 | 11 ns |
|  | 256 | 15 ns | 20 ns | 0.75 | 20 ns |
|  | 1,000 | 15 ns | 19 ns | 0.79 | 25 ns |
|  | 10,000 | 30 ns | 34 ns | 0.88 | 71 ns |
|  | 100,000 | 166 ns | 164 ns | 1.01 | 414 ns |
|  | 1,000,000 | 1.25 µs | 1.2 µs | 1.04 | 3.97 µs |
|  | 10,000,000 | 14.8 µs | 14.5 µs | 1.02 | 38 µs |

### Complex numbers

Against `mpc_add`, `mpc_mul`, `mpc_sqr`, `mpc_div`, `mpc_sqrt`, `mpc_abs` and `mpc_norm` into an existing variable; a precision is that of both parts of the operands and of the result. The column `a` is the form with a new result (`add_round` and the others), `a` into the form into an existing number (`set_add` and the others):

| Operation | Precision | `a` | `a` into | MPC | `a`/MPC | `a` into/MPC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `add_round` / `set_add` | 24 | 28.4 ns | 10.5 ns | 13.6 ns | 2.09 | 0.77 |
|  | 53 | 28.4 ns | 10.4 ns | 13.5 ns | 2.10 | 0.77 |
|  | 64 | 27.1 ns | 9.4 ns | 15.7 ns | 1.73 | 0.60 |
|  | 113 | 31.6 ns | 13.2 ns | 15.2 ns | 2.08 | 0.87 |
|  | 128 | 30.3 ns | 11.4 ns | 15.8 ns | 1.92 | 0.72 |
|  | 256 | 41.3 ns | 24.9 ns | 21.5 ns | 1.92 | 1.16 |
|  | 1,000 | 56 ns | 31.7 ns | 28.3 ns | 1.98 | 1.12 |
|  | 10,000 | 143 ns | 98.4 ns | 161 ns | 0.89 | 0.61 |
|  | 100,000 | 1.14 µs | 850 ns | 1.63 µs | 0.70 | 0.52 |
| `mul_round` / `set_mul` | 24 | 60.1 ns | 26.1 ns | 64.1 ns | 0.94 | 0.41 |
|  | 53 | 57.5 ns | 26.6 ns | 53.9 ns | 1.07 | 0.49 |
|  | 64 | 58 ns | 26.4 ns | 63 ns | 0.92 | 0.42 |
|  | 113 | 74.6 ns | 39.1 ns | 89.2 ns | 0.84 | 0.44 |
|  | 128 | 73.4 ns | 37.8 ns | 88.4 ns | 0.83 | 0.43 |
|  | 256 | 154 ns | 136 ns | 116 ns | 1.33 | 1.17 |
|  | 1,000 | 429 ns | 409 ns | 459 ns | 0.94 | 0.89 |
|  | 10,000 | 11.4 µs | 11.4 µs | 12.1 µs | 0.94 | 0.94 |
|  | 100,000 | 160 µs | 159 µs | 322 µs | 0.50 | 0.50 |
| `sqr_round` / `set_sqr` | 24 | 55.2 ns | 23.1 ns | 95.4 ns | 0.58 | 0.24 |
|  | 53 | 55.4 ns | 22.4 ns | 96.3 ns | 0.58 | 0.23 |
|  | 64 | 55 ns | 22.1 ns | 144 ns | 0.38 | 0.15 |
|  | 113 | 64.8 ns | 28.2 ns | 141 ns | 0.46 | 0.20 |
|  | 128 | 62.8 ns | 26.7 ns | 153 ns | 0.41 | 0.17 |
|  | 256 | 123 ns | 99.7 ns | 183 ns | 0.67 | 0.54 |
|  | 1,000 | 289 ns | 266 ns | 359 ns | 0.81 | 0.74 |
|  | 10,000 | 8.52 µs | 8.5 µs | 6.93 µs | 1.23 | 1.23 |
|  | 100,000 | 127 µs | 127 µs | 183 µs | 0.69 | 0.69 |
| `div_round` / `set_div` | 24 | 333 ns | 336 ns | 319 ns | 1.05 | 1.05 |
|  | 53 | 340 ns | 340 ns | 398 ns | 0.85 | 0.85 |
|  | 64 | 370 ns | 368 ns | 454 ns | 0.81 | 0.81 |
|  | 113 | 438 ns | 436 ns | 466 ns | 0.94 | 0.94 |
|  | 128 | 433 ns | 439 ns | 497 ns | 0.87 | 0.88 |
|  | 256 | 526 ns | 524 ns | 587 ns | 0.90 | 0.89 |
|  | 1,000 | 1.38 µs | 1.36 µs | 1.37 µs | 1.01 | 1.00 |
|  | 10,000 | 29.5 µs | 29.5 µs | 31.1 µs | 0.95 | 0.95 |
|  | 100,000 | 535 µs | 536 µs | 821 µs | 0.65 | 0.65 |
| `sqrt_round` / `set_sqrt` | 24 | 103 ns | 75.1 ns | 159 ns | 0.65 | 0.47 |
|  | 53 | 105 ns | 76.2 ns | 157 ns | 0.67 | 0.49 |
|  | 64 | 105 ns | 76.5 ns | 212 ns | 0.49 | 0.36 |
|  | 113 | 120 ns | 91.8 ns | 243 ns | 0.50 | 0.38 |
|  | 128 | 370 ns | 341 ns | 356 ns | 1.04 | 0.96 |
|  | 256 | 580 ns | 559 ns | 454 ns | 1.28 | 1.23 |
|  | 1,000 | 1.12 µs | 1.09 µs | 1.09 µs | 1.02 | 1.01 |
|  | 10,000 | 17.7 µs | 17.6 µs | 17.7 µs | 1.00 | 1.00 |
|  | 100,000 | 438 µs | 436 µs | 596 µs | 0.73 | 0.73 |
| `abs_round` / `BigFloat::set_complex_abs` | 24 | 102 ns | 87.3 ns | 99.8 ns | 1.02 | 0.87 |
|  | 53 | 100 ns | 84 ns | 104 ns | 0.97 | 0.81 |
|  | 64 | 99.6 ns | 84.7 ns | 117 ns | 0.85 | 0.72 |
|  | 113 | 116 ns | 98.4 ns | 127 ns | 0.91 | 0.78 |
|  | 128 | 157 ns | 141 ns | 172 ns | 0.91 | 0.82 |
|  | 256 | 210 ns | 195 ns | 231 ns | 0.91 | 0.84 |
|  | 1,000 | 439 ns | 427 ns | 518 ns | 0.85 | 0.82 |
|  | 10,000 | 9.04 µs | 9 µs | 9.74 µs | 0.93 | 0.92 |
|  | 100,000 | 193 µs | 193 µs | 306 µs | 0.63 | 0.63 |
| `norm_round` / `BigFloat::set_complex_norm` | 24 | 25 ns | 9.5 ns | 90.4 ns | 0.28 | 0.11 |
|  | 53 | 24.9 ns | 9.5 ns | 93.4 ns | 0.27 | 0.10 |
|  | 64 | 23.4 ns | 9.5 ns | 152 ns | 0.15 | 0.06 |
|  | 113 | 29.9 ns | 14.6 ns | 156 ns | 0.19 | 0.09 |
|  | 128 | 28.1 ns | 14 ns | 154 ns | 0.18 | 0.09 |
|  | 256 | 82.7 ns | 70.9 ns | 171 ns | 0.48 | 0.42 |
|  | 1,000 | 190 ns | 180 ns | 297 ns | 0.64 | 0.61 |
|  | 10,000 | 5.09 µs | 5.09 µs | 4.33 µs | 1.18 | 1.17 |
|  | 100,000 | 78.4 µs | 78.3 µs | 124 µs | 0.63 | 0.63 |

### Functions of complex numbers

Against `mpc_exp` and the others; arguments of parts of exponents −2 to 1, the powers' exponents `w = 4u + vi` with `u`, `v` in `[1/2, 1)`, and `n = 17` for `powi`:

| Operation | Precision | `a` | MPC | `a`/MPC |
| --- | --- | ---: | ---: | ---: |
| `exp` | 53 | 866 ns | 1.46 µs | 0.59 |
|  | 113 | 1.07 µs | 2.13 µs | 0.50 |
|  | 256 | 2.14 µs | 3.78 µs | 0.57 |
|  | 1,000 | 8.35 µs | 18.5 µs | 0.45 |
|  | 10,000 | 447 µs | 1.27 ms | 0.35 |
|  | 100,000 | 12.9 ms | 42.1 ms | 0.31 |
| `ln` | 53 | 1.27 µs | 1.69 µs | 0.75 |
|  | 113 | 1.95 µs | 3.4 µs | 0.57 |
|  | 256 | 3.16 µs | 8 µs | 0.40 |
|  | 1,000 | 15.9 µs | 26.6 µs | 0.60 |
|  | 10,000 | 467 µs | 543 µs | 0.86 |
|  | 100,000 | 16.3 ms | 21.3 ms | 0.76 |
| `log10` | 53 | 1.54 µs | 2.34 µs | 0.66 |
|  | 113 | 2.24 µs | 4.71 µs | 0.48 |
|  | 256 | 4.36 µs | 10.5 µs | 0.42 |
|  | 1,000 | 17.8 µs | 35.1 µs | 0.51 |
|  | 10,000 | 482 µs | 723 µs | 0.67 |
|  | 100,000 | 16.5 ms | 28.4 ms | 0.58 |
| `arg` | 53 | 642 ns | 1.07 µs | 0.60 |
|  | 113 | 1.12 µs | 1.95 µs | 0.57 |
|  | 256 | 1.87 µs | 5.32 µs | 0.35 |
|  | 1,000 | 8.64 µs | 18.7 µs | 0.46 |
|  | 10,000 | 287 µs | 366 µs | 0.78 |
|  | 100,000 | 11.4 ms | 14.3 ms | 0.80 |
| `sin` | 53 | 1.5 µs | 1.77 µs | 0.85 |
|  | 113 | 1.73 µs | 2.21 µs | 0.78 |
|  | 256 | 2.84 µs | 3.12 µs | 0.91 |
|  | 1,000 | 9.47 µs | 11.5 µs | 0.83 |
|  | 10,000 | 457 µs | 575 µs | 0.79 |
|  | 100,000 | 13.3 ms | 23.2 ms | 0.57 |
| `cos` | 53 | 1.49 µs | 1.7 µs | 0.87 |
|  | 113 | 1.71 µs | 2.23 µs | 0.77 |
|  | 256 | 2.84 µs | 3.16 µs | 0.90 |
|  | 1,000 | 9.5 µs | 11.6 µs | 0.82 |
|  | 10,000 | 470 µs | 585 µs | 0.80 |
|  | 100,000 | 13.2 ms | 23.1 ms | 0.57 |
| `tan` | 53 | 1.99 µs | 4.22 µs | 0.47 |
|  | 113 | 2.28 µs | 4.72 µs | 0.48 |
|  | 256 | 3.4 µs | 6.01 µs | 0.57 |
|  | 1,000 | 10.7 µs | 16.8 µs | 0.63 |
|  | 10,000 | 480 µs | 634 µs | 0.76 |
|  | 100,000 | 13.7 ms | 24.5 ms | 0.56 |
| `sin_cos` | 53 | 1.8 µs | 1.86 µs | 0.97 |
|  | 113 | 2.02 µs | 2.29 µs | 0.88 |
|  | 256 | 3.18 µs | 3.24 µs | 0.98 |
|  | 1,000 | 9.99 µs | 11.8 µs | 0.84 |
|  | 10,000 | 466 µs | 586 µs | 0.80 |
|  | 100,000 | 13.4 ms | 23.5 ms | 0.57 |
| `sinh` | 53 | 1.52 µs | 1.78 µs | 0.86 |
|  | 113 | 1.73 µs | 2.23 µs | 0.78 |
|  | 256 | 2.84 µs | 3.16 µs | 0.90 |
|  | 1,000 | 9.57 µs | 11.5 µs | 0.83 |
|  | 10,000 | 464 µs | 587 µs | 0.79 |
|  | 100,000 | 13.2 ms | 23.1 ms | 0.57 |
| `cosh` | 53 | 1.49 µs | 1.75 µs | 0.85 |
|  | 113 | 1.73 µs | 2.25 µs | 0.77 |
|  | 256 | 2.83 µs | 3.11 µs | 0.91 |
|  | 1,000 | 9.6 µs | 11.5 µs | 0.84 |
|  | 10,000 | 461 µs | 591 µs | 0.78 |
|  | 100,000 | 13.2 ms | 23 ms | 0.57 |
| `tanh` | 53 | 2 µs | 4.11 µs | 0.49 |
|  | 113 | 2.28 µs | 4.42 µs | 0.51 |
|  | 256 | 3.4 µs | 5.92 µs | 0.57 |
|  | 1,000 | 10.5 µs | 15.6 µs | 0.68 |
|  | 10,000 | 477 µs | 623 µs | 0.77 |
|  | 100,000 | 13.4 ms | 24.1 ms | 0.56 |
| `asin` | 53 | 3.42 µs | 6.3 µs | 0.54 |
|  | 113 | 4.39 µs | 11.5 µs | 0.38 |
|  | 256 | 7.02 µs | 25 µs | 0.28 |
|  | 1,000 | 19.9 µs | 158 µs | 0.13 |
|  | 10,000 | 509 µs | 583 µs | 0.87 |
|  | 100,000 | 17.1 ms | 22.5 ms | 0.76 |
| `acos` | 53 | 3.41 µs | 7.35 µs | 0.46 |
|  | 113 | 4.38 µs | 14.4 µs | 0.30 |
|  | 256 | 7.03 µs | 24.3 µs | 0.29 |
|  | 1,000 | 19.6 µs | 141 µs | 0.14 |
|  | 10,000 | 504 µs | 577 µs | 0.87 |
|  | 100,000 | 17.1 ms | 22.4 ms | 0.76 |
| `atan` | 53 | 1.71 µs | 4.45 µs | 0.38 |
|  | 113 | 2.38 µs | 8.19 µs | 0.29 |
|  | 256 | 4.83 µs | 17.2 µs | 0.28 |
|  | 1,000 | 17.7 µs | 53.5 µs | 0.33 |
|  | 10,000 | 480 µs | 1.08 ms | 0.44 |
|  | 100,000 | 16.5 ms | 42.3 ms | 0.39 |
| `asinh` | 53 | 3.42 µs | 6.36 µs | 0.54 |
|  | 113 | 4.27 µs | 11.8 µs | 0.36 |
|  | 256 | 6.94 µs | 26.2 µs | 0.27 |
|  | 1,000 | 19.4 µs | 168 µs | 0.12 |
|  | 10,000 | 506 µs | 577 µs | 0.88 |
|  | 100,000 | 16.9 ms | 22.5 ms | 0.75 |
| `acosh` | 53 | 3.4 µs | 7.37 µs | 0.46 |
|  | 113 | 4.31 µs | 14.5 µs | 0.30 |
|  | 256 | 7.07 µs | 24.6 µs | 0.29 |
|  | 1,000 | 19.8 µs | 141 µs | 0.14 |
|  | 10,000 | 505 µs | 580 µs | 0.87 |
|  | 100,000 | 17 ms | 22.5 ms | 0.75 |
| `atanh` | 53 | 1.73 µs | 4.41 µs | 0.39 |
|  | 113 | 2.4 µs | 7.7 µs | 0.31 |
|  | 256 | 4.82 µs | 17.3 µs | 0.28 |
|  | 1,000 | 17.8 µs | 54.1 µs | 0.33 |
|  | 10,000 | 479 µs | 1.09 ms | 0.44 |
|  | 100,000 | 16.3 ms | 41.6 ms | 0.39 |
| `pow` | 53 | 3.45 µs | 6.57 µs | 0.52 |
|  | 113 | 4.28 µs | 9.4 µs | 0.46 |
|  | 256 | 7.04 µs | 17.8 µs | 0.40 |
|  | 1,000 | 26.4 µs | 62.8 µs | 0.42 |
|  | 10,000 | 929 µs | 2.73 ms | 0.34 |
|  | 100,000 | 29.2 ms | 65.9 ms | 0.44 |
| `pow_real` | 53 | 3.26 µs | 6.14 µs | 0.53 |
|  | 113 | 4.08 µs | 8.63 µs | 0.47 |
|  | 256 | 6.87 µs | 17.3 µs | 0.40 |
|  | 1,000 | 26.4 µs | 59.3 µs | 0.45 |
|  | 10,000 | 923 µs | 1.84 ms | 0.50 |
|  | 100,000 | 29.2 ms | 66.8 ms | 0.44 |
| `powi`, `n = 17` | 53 | 315 ns | 791 ns | 0.40 |
|  | 113 | 830 ns | 810 ns | 1.03 |
|  | 256 | 943 ns | 992 ns | 0.95 |
|  | 1,000 | 1.97 µs | 2.1 µs | 0.94 |
|  | 10,000 | 46 µs | 44.7 µs | 1.03 |
|  | 100,000 | 679 µs | 1.07 ms | 0.63 |

### Matrices of floats

`&a * &b` of `n × n` matrices of `BigFloat`, every element correctly rounded, against a loop of `mpfr_fma` (rounded at each step) and against `mpfr_dot` (correctly rounded); `a.solve(&b)` with one column against Gaussian elimination by `mpfr_fms`:

| Operation | Precision | `a` | MPFR, `mpfr_fma` loop | `a`/loop | `mpfr_dot` |
| --- | --- | ---: | ---: | ---: | ---: |
| `&a * &b`, 4 × 4 | 53 | 938 ns | 640 ns | 1.47 | 2.49 µs |
|  | 113 | 1.05 µs | 1.26 µs | 0.83 | 2.64 µs |
|  | 256 | 1.94 µs | 1.86 µs | 1.05 | 3.25 µs |
|  | 1,000 | 6.87 µs | 7.24 µs | 0.95 | 9.15 µs |
| `&a * &b`, 8 × 8 | 53 | 5.51 µs | 5.73 µs | 0.96 | 18.7 µs |
|  | 113 | 6.43 µs | 12.3 µs | 0.52 | 19.3 µs |
|  | 256 | 13.2 µs | 18.1 µs | 0.72 | 23.1 µs |
|  | 1,000 | 51.1 µs | 61.8 µs | 0.83 | 69.4 µs |
| `&a * &b`, 16 × 16 | 53 | 39.5 µs | 53.7 µs | 0.74 | 145 µs |
|  | 113 | 44.5 µs | 123 µs | 0.36 | 144 µs |
|  | 256 | 103 µs | 163 µs | 0.63 | 173 µs |
|  | 1,000 | 395 µs | 507 µs | 0.78 | 548 µs |
| `a.solve(&b)`, 4 × 4 | 53 | 921 ns | 476 ns | 1.94 | – |
|  | 113 | 1.04 µs | 800 ns | 1.30 | – |
|  | 256 | 3.15 µs | 1.57 µs | 2.01 | – |
|  | 1,000 | 6.66 µs | 5.35 µs | 1.25 | – |
| `a.solve(&b)`, 8 × 8 | 53 | 3.71 µs | 3.11 µs | 1.19 | – |
|  | 113 | 4.37 µs | 5.9 µs | 0.74 | – |
|  | 256 | 13.5 µs | 9.85 µs | 1.37 | – |
|  | 1,000 | 33.7 µs | 32.1 µs | 1.05 | – |
| `a.solve(&b)`, 16 × 16 | 53 | 18.7 µs | 23 µs | 0.81 | – |
|  | 113 | 22.8 µs | 48.4 µs | 0.47 | – |
|  | 256 | 69.7 µs | 67.6 µs | 1.03 | – |
|  | 1,000 | 198 µs | 211 µs | 0.94 | – |

### Where the references are faster

- **Arb** at every size to 10⁵ bits, and at 10⁶ bits for `exp`, `sin` and `atan` (1.3 to 6.7 times at 53 bits for the elementary functions; `ln` at 10⁶ bits is faster than Arb's, 70.5 against 87 ms): tables of the functions at many points, fixed point on GMP's `mpn` level and no Ziv loop, because its balls only enclose the value. It returns something else, so it is listed for its time.
- **Sums of two limbs and of three to sixteen into an existing number** (`set_add` and `set_sub` at 113 bits 1.05 to 1.19 times, at 256 bits 1.16 to 1.25, at 1,000 bits 1.18 to 1.21; with a new result 0.86 to 1.07): MPFR's `mpfr_add1sp2` and `mpfr_add1sp` are specialised for operands of the result's precision, whose low bits are zero, where `a`'s kernels take operands of any bits and carry a whole word below the result; past one limb that costs 0.3 to 3.1 ns. At one limb and at 128 bits `a` is at par or faster (0.72 to 1.00), and from 10⁴ bits 0.43 to 0.68 times MPFR's time.
- **Quotients of two limbs and of four** (`set_div` at 113 bits 1.28 times, at 256 bits 1.13): `a` takes the exact reciprocal of the divisor's top two limbs and two exact divisions of three limbs by two, MPFR's `mpfr_div_2` an approximate quotient from an approximate reciprocal of the top limb, with an error below 21 units and a test of the bits it needs. With a new result `a` is at par or faster (0.90 to 1.00).
- **Quotients from long operands to 53 bits and of 1,000 bits from short ones** (`53 (operands 1,000)` 1.21 times, `53 (operands 100,000)` 1.22, `1,000 (operands 53)` 1.32; with a new result 1.02): two divisions of a limb, one for each end of the interval of the truncated operands, against MPFR's one; and the 17 limbs of a quotient by one limb, each about 10 cycles waiting for the remainder before it.
- **`mul_add_round` at 1,000 bits** (1.17 to 1.21 times): operands longer than four limbs take the general path, the exact product of 32 limbs in working memory and the window sum; up to 256 bits `a` is 0.41 to 0.96 times MPFR's time, and `1,000 (operands 53)` 0.58.
- **`x·x` and `norm` at 10⁴ bits** (`x·x` 1.11 times, `BigComplex`'s `sqr_round` 1.23 and `norm_round` 1.18): MPFR's `mpfr_sqr` and MPC's `a² ± b²` round with short products (`mpfr_mulhigh`), which the crate's integers have not; `a` multiplies exactly. The full product of 157 limbs is itself at par with GMP's (`set_mul` 1.00).
- **`set_f64` at 10⁶ bits** (1.36 times, 782 against 575 ns): the zeros of the 15,625 limbs below the top one; from 24 to 10⁵ bits `set_f64` takes 2.5 to 113 ns, 0.31 to 0.99 times MPFR's `mpfr_set_d`.
- **`set_floor` at 10⁵ bits and `set_rint` at 1,000 and 10⁴ bits** (1.10, 1.08 and 1.12 times, 0.7 to 11 ns): the same copy of the limbs and scan of the fraction as MPFR's; at the other sizes `set_floor` is 0.85 to 1.00 times `mpfr_floor` and `set_rint` 0.89 to 1.00 times `mpfr_rint`. The forms with a new result (`from_f64_round`, `floor`, `rint_round`) allocate and stay 1.2 to 2.6 times MPFR's into an existing number (and faster than rug's, which allocate too). Division at 1,000 and 10⁴ bits is no longer behind (0.82 and 0.84 times).
- **Square root into an existing number at 24 to 113 bits** (1.15 to 1.87 times, 0.9 to 5.9 ns of 6.2 to 11.8; the 64-bit figure, 12.7 ns, is 9.9 ns in another build of the same machine code at other addresses), **and at 10⁴ bits** (1.19 times; at par from 256 to 1,000 bits, 0.97 to 1.02): the roots of one and two limbs take about 4 and 10 ns in registers, and the rest is the fixed cost of `set_sqrt` over the `Kind` and the vector of the mantissa; with a new result both allocate, and the gap is 1.17 to 1.55 times, the allocation and the release of a vector costing about 1.5 to 2 times MPFR's `mpfr_init2` and `mpfr_clear`. From 256 bits both take the integer root of Zimmermann's Karatsuba square root, GMP's `mpn_sqrtrem` faster than the crate's at 10⁴ bits. At 128 bits `a` is 3 times faster, MPFR having no form for it.
- **`rec_sqrt_round` at 1,000 bits** (1.02 times): the square root of the crate's integers and a division, where MPFR's Newton's iteration takes short products, which the crate's integers have not; at 24 to 113 bits `a` is 1.9 to 2.5 times faster, at 128 and 256 bits 1.04 to 1.08 times, from 10⁴ bits 1.1 to 1.6 times.
- **`cbrt_round` at 10⁴ bits** (1.06 times): the integer root of the crate's numbers against GMP's `mpz_root`, both Newton's method on integers; to 113 bits `a` is 2 to 3.2 times faster, at 128 and 256 bits 1.09 to 1.11 times, at 1,000 bits at par, at 10⁵ bits 1.27 times.
- **`powi_round(7)` at 10⁴ bits** (1.04 times): both are the cost of three products of 157 limbs; at every other size to 10⁵ bits `powi_round` is faster than MPFR's `mpfr_pow_si`.
- **The logarithms and the inverse hyperbolic functions at 10⁴ bits** (`ln`, `log2`, `ln_1p`, `asinh`, `acosh` and `atanh` 1.03 to 1.07 times; `log10` 0.53, MPFR's being slower): the AGM on limbs takes 21 steps of a product and a square root of 159 limbs, about 7.4 µs a step here (a product 3.4 µs and a root 4.0 µs, measured alone), where MPFR takes 23 steps of GMP's, about 6.2 µs a step; the first step's root alone, the correction that stands for the last two and the division of `4/s` saved do not make up the difference of the kernels. Newton's iteration on the exponential would cost more than one exponential of 10⁴ bits (0.88 times MPFR's, against MPFR's logarithm at 0.58 of its exponential). From 3,400 bits the AGM is ahead of Newton's iteration; from 24 to 256 bits the tables make the logarithms 0.16 to 0.64 times MPFR's.
- **Complex numbers**: `set_add` at 256 and 1,000 bits (1.16 and 1.12 times) and `add_round` to 1,000 bits (1.73 to 2.10): two `BigFloat` additions of three to sixteen limbs (see above), and the allocating form two new numbers against MPC's into variables. `set_mul` at 256 bits (1.17 times, `mul_round` 1.33): the window addition's fixed cost, twice. `set_sqrt` at 256 bits (1.23 times, `sqrt_round` 1.28; at par from 128 to 10⁴ bits, 0.96 to 1.01): the real square roots and quotients of four limbs at the working precision (see the square root above). Among the functions only `powi_round` is behind, at 113 and 10⁴ bits (1.03 times both): its products at the working precision, which with the guard bits is past two limbs at 113 bits, take the exact window path; `ln_round`, `log10_round`, `sin_cos_round` and the inverse functions are faster than MPC's at every size (0.12 to 0.98). From 10⁵ bits `a` is faster everywhere (0.39 to 0.80).
- **Matrices**: the solves of 4 × 4 matrices at every size (1.25 to 2.01 times), of 8 × 8 at 53, 256 and 1,000 bits (1.19, 1.37 and 1.05) and of 16 × 16 at 256 bits (1.03), and the 4 × 4 products at 53 and 256 bits (1.47 and 1.05): every element of a new matrix is a new vector (about 20 ns each, `n²` of them for a product and `n` a column for a solve), where the MPFR loops write into variables that exist; and every element is the exact dot product rounded once, against `mpfr_fma` and `mpfr_fms` rounded at each step: elements of one and two limbs take their products in registers (`short_dot`), longer ones the general exact path, which is why 256 bits are further behind than 113 (the solve of 4 × 4 at 113 bits went from 2.75 to 1.30 times in this pass, at 256 bits it did not move). From 8 × 8 the products are faster at every size, 0.36 to 0.96 times the `mpfr_fma` loop.
- **Cached constants below 256 bits**: Arb copies 7 to 11 ns, `a` 14 ns (an allocation). From 10⁵ bits `a` and MPFR copy at the same speed (0.9 to 1.26, the noise of a copy).
