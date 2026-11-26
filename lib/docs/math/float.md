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

Measured on an Apple M4 Max (macOS 27.2) on 2026-11-26 between 06:07 and 06:22, with nothing else running: the areas of the release's final run were timed one after another, and every timed process ran alone on the machine. One thread; `a` as a user builds it, `cargo build --release` with the default profile (opt-level 3, no LTO, the default codegen units, no `target-cpu`), on Rust 1.100.0 stable, by the harness in the repository's [`bench/float`](../../../bench/float) (see [`bench/README.md`](../../../bench/README.md)). Each figure is the median of 15 samples (fewer where a call is long), each a batch of at least 1 ms over a pool of distinct arguments, all the contenders of a row alternating sample by sample in one process; every reference's results were compared with `a`'s before the timing, and agreed, but where a reference does less by design (Arb's balls, dashu-float's sums kept to a bit more than the precision, its decimal text). The references: MPFR 4.2.2 and GNU MPC 1.4.1 on GMP 6.3.0 (Homebrew's static archives, called through FFI, MPFR's exponent range set to that of `BigFloat`), rug 1.30.0 on the same libraries, malachite-float 0.15.0, astro-float 0.9.6 (which rounds every precision up to whole words, so its rows are against `a` at that precision), dashu-float 0.6.2, and Arb of FLINT 3.6.0. Each row gives the fastest reference that does the same work, the same arguments to the same correctly rounded result, its time and `a`'s time over it: below 1, `a` is faster. Arb's balls enclose the value but are not correctly rounded, so Arb does less: its column is its time alone, at the precision and the guard bits of Ziv's first attempt. MPFR and MPC write into a variable that exists and `a` returns a new number, unless a table says otherwise; the arguments have the precision of the result and full mantissas, of magnitudes in `[1/4, 4)` and both signs (positive for the roots and the logarithms) unless a table says otherwise, and every result is rounded to nearest.

### Arithmetic

Into an existing number, at its precision: `set_add` and the others against `mpfr_add (r, a, b, rnd)` and rug's `assign_round`, `x·x` being `set_mul(&x, &x)`. A precision `p (operands q)` is a result of `p` bits from operands of `q`:

| Operation | Precision | `a` | Fastest | Its time | `a`/fastest |
| --- | --- | ---: | --- | ---: | ---: |
| `set_add` | 24 | 3.90 ns | MPFR | 5.68 ns | 0.69 |
|  | 53 | 3.44 ns | MPFR | 4.57 ns | 0.75 |
|  | 113 | 5.50 ns | MPFR | 5.71 ns | 0.96 |
|  | 256 | 12.5 ns | MPFR | 9.49 ns | 1.32 |
|  | 1,000 | 18.6 ns | rug | 13.9 ns | 1.34 |
|  | 10,000 | 63.4 ns | MPFR | 101 ns | 0.63 |
|  | 100,000 | 540 ns | rug | 1.19 µs | 0.45 |
|  | 1,000,000 | 4.91 µs | MPFR | 8.40 µs | 0.58 |
|  | 53 (operands 100,000) | 16.4 ns | rug | 214 ns | 0.077 |
|  | 1,000 (operands 53) | 15.7 ns | MPFR | 16.3 ns | 0.97 |
| `set_sub` | 24 | 3.28 ns | MPFR | 5.24 ns | 0.63 |
|  | 53 | 3.04 ns | rug | 5.25 ns | 0.58 |
|  | 113 | 5.43 ns | MPFR | 6.23 ns | 0.87 |
|  | 256 | 12.7 ns | MPFR | 10.4 ns | 1.22 |
|  | 1,000 | 19.1 ns | MPFR | 14.8 ns | 1.29 |
|  | 10,000 | 61.6 ns | rug | 97.9 ns | 0.63 |
|  | 100,000 | 528 ns | rug | 1.09 µs | 0.48 |
|  | 1,000,000 | 6.03 µs | rug | 9.84 µs | 0.61 |
|  | 53 (operands 100,000) | 16.2 ns | MPFR | 207 ns | 0.078 |
|  | 1,000 (operands 53) | 15.5 ns | MPFR | 16.2 ns | 0.95 |
| `set_mul` | 24 | 2.84 ns | MPFR | 5.01 ns | 0.57 |
|  | 53 | 2.44 ns | MPFR | 5.24 ns | 0.47 |
|  | 113 | 3.97 ns | MPFR | 5.38 ns | 0.74 |
|  | 256 | 13.0 ns | rug | 23.2 ns | 0.56 |
|  | 1,000 | 97.7 ns | rug | 102 ns | 0.96 |
|  | 10,000 | 2.70 µs | rug | 3.39 µs | 0.80 |
|  | 100,000 | 50.5 µs | rug | 90.4 µs | 0.56 |
|  | 1,000,000 | 567 µs | MPFR | 1.41 ms | 0.40 |
|  | 53 (operands 100,000) | 41.0 ns | MPFR | 56.1 ns | 0.73 |
|  | 1,000 (operands 53) | 7.55 ns | rug | 10.4 ns | 0.72 |
| `x·x` | 24 | 2.43 ns | MPFR | 4.86 ns | 0.50 |
|  | 53 | 1.93 ns | MPFR | 4.65 ns | 0.41 |
|  | 113 | 3.37 ns | rug | 5.09 ns | 0.66 |
|  | 256 | 11.8 ns | rug | 19.6 ns | 0.60 |
|  | 1,000 | 81.9 ns | MPFR | 80.3 ns | 1.02 |
|  | 10,000 | 1.68 µs | rug | 1.99 µs | 0.85 |
|  | 100,000 | 38.3 µs | MPFR | 63.4 µs | 0.60 |
|  | 1,000,000 | 416 µs | MPFR | 1.22 ms | 0.34 |
|  | 53 (operands 1,000) | 35.1 ns | MPFR | 80.2 ns | 0.44 |
| `set_div` | 24 | 5.11 ns | rug | 5.56 ns | 0.92 |
|  | 53 | 5.11 ns | rug | 5.57 ns | 0.92 |
|  | 113 | 10.5 ns | rug | 8.04 ns | 1.30 |
|  | 256 | 54.3 ns | MPFR | 69.0 ns | 0.79 |
|  | 1,000 | 191 ns | rug | 226 ns | 0.85 |
|  | 10,000 | 4.74 µs | MPFR | 5.84 µs | 0.81 |
|  | 100,000 | 122 µs | MPFR | 200 µs | 0.61 |
|  | 1,000,000 | 1.34 ms | MPFR | 3.99 ms | 0.34 |
|  | 53 (operands 1,000) | 25.1 ns | rug | 19.0 ns | 1.32 |
|  | 53 (operands 100,000) | 28.2 ns | MPFR | 21.1 ns | 1.34 |
|  | 1,000 (operands 53) | 53.6 ns | MPFR | 39.3 ns | 1.37 |
| `set_mul_add` | 24 | 10.3 ns | MPFR | 10.8 ns | 0.95 |
|  | 53 | 10.4 ns | MPFR | 11.6 ns | 0.90 |
|  | 113 | 15.7 ns | rug | 23.2 ns | 0.67 |
|  | 256 | 30.6 ns | MPFR | 32.5 ns | 0.94 |
|  | 1,000 | 139 ns | MPFR | 118 ns | 1.18 |
|  | 10,000 | 2.94 µs | MPFR | 3.89 µs | 0.76 |
|  | 100,000 | 52.3 µs | MPFR | 109 µs | 0.48 |
|  | 1,000,000 | 581 µs | rug | 1.43 ms | 0.41 |
|  | 53 (operands 100,000) | 50.3 µs | MPFR | 108 µs | 0.47 |
|  | 1,000 (operands 53) | 13.2 ns | MPFR | 24.2 ns | 0.55 |
| `set_sqrt` | 24 | 4.46 ns | rug | 5.53 ns | 0.81 |
|  | 53 | 4.48 ns | rug | 5.61 ns | 0.80 |
|  | 113 | 11.7 ns | rug | 11.3 ns | 1.04 |
|  | 256 | 118 ns | MPFR | 106 ns | 1.12 |
|  | 1,000 | 272 ns | rug | 279 ns | 0.97 |
|  | 10,000 | 3.98 µs | MPFR | 3.84 µs | 1.04 |
|  | 100,000 | 121 µs | MPFR | 141 µs | 0.86 |
|  | 1,000,000 | 1.45 ms | MPFR | 3.29 ms | 0.44 |
|  | 53 (operands 1,000) | 7.58 ns | rug | 13.8 ns | 0.55 |

With a new result: `add_round` and the others against `mpfr_init2`, the operation and `mpfr_clear`, rug's `with_val` and the Rust crates' allocating forms (malachite-float's `_prec_round_ref`):

| Operation | Precision | `a` | Fastest | Its time | `a`/fastest |
| --- | --- | ---: | --- | ---: | ---: |
| `add_round` | 24 | 5.28 ns | malachite-float | 8.56 ns | 0.62 |
|  | 53 | 5.54 ns | malachite-float | 8.54 ns | 0.65 |
|  | 113 | 7.20 ns | MPFR | 17.1 ns | 0.42 |
|  | 256 | 23.6 ns | MPFR | 22.5 ns | 1.05 |
|  | 1,000 | 30.9 ns | MPFR | 27.6 ns | 1.12 |
|  | 10,000 | 95.7 ns | MPFR | 115 ns | 0.83 |
|  | 1,000,000 | 8.09 µs | MPFR | 8.50 µs | 0.95 |
| `mul_round` | 24 | 4.86 ns | malachite-float | 9.01 ns | 0.54 |
|  | 53 | 4.30 ns | malachite-float | 8.80 ns | 0.49 |
|  | 113 | 5.07 ns | malachite-float | 17.9 ns | 0.28 |
|  | 256 | 26.6 ns | rug | 34.5 ns | 0.77 |
|  | 1,000 | 107 ns | rug | 117 ns | 0.91 |
|  | 10,000 | 2.71 µs | rug | 3.48 µs | 0.78 |
|  | 1,000,000 | 559 µs | malachite-float | 940 µs | 0.59 |
| `x·x` | 24 | 4.49 ns | malachite-float | 6.17 ns | 0.73 |
|  | 53 | 3.85 ns | malachite-float | 5.76 ns | 0.67 |
|  | 113 | 4.69 ns | malachite-float | 15.6 ns | 0.30 |
|  | 256 | 26.3 ns | MPFR | 31.2 ns | 0.84 |
|  | 1,000 | 91.9 ns | MPFR | 89.8 ns | 1.02 |
|  | 10,000 | 1.70 µs | rug | 2.11 µs | 0.81 |
|  | 1,000,000 | 416 µs | malachite-float | 948 µs | 0.44 |
| `div_round` | 24 | 6.19 ns | malachite-float | 16.1 ns | 0.39 |
|  | 53 | 6.21 ns | malachite-float | 15.3 ns | 0.40 |
|  | 113 | 11.5 ns | MPFR | 20.3 ns | 0.57 |
|  | 256 | 68.1 ns | MPFR | 80.9 ns | 0.84 |
|  | 1,000 | 201 ns | MPFR | 237 ns | 0.85 |
|  | 10,000 | 4.66 µs | rug | 5.71 µs | 0.82 |
|  | 1,000,000 | 1.34 ms | rug | 4.04 ms | 0.33 |
| `mul_add_round` | 24 | 11.0 ns | MPFR | 23.9 ns | 0.46 |
|  | 53 | 10.4 ns | rug | 23.9 ns | 0.43 |
|  | 113 | 15.8 ns | rug | 36.0 ns | 0.44 |
|  | 256 | 42.3 ns | rug | 44.4 ns | 0.95 |
|  | 1,000 | 155 ns | MPFR | 130 ns | 1.19 |
|  | 10,000 | 2.95 µs | rug | 4.57 µs | 0.65 |
|  | 1,000,000 | 571 µs | malachite-float | 968 µs | 0.59 |
| `sqrt_round` | 24 | 4.95 ns | malachite-float | 10.3 ns | 0.48 |
|  | 53 | 4.92 ns | malachite-float | 9.35 ns | 0.53 |
|  | 113 | 12.6 ns | malachite-float | 20.0 ns | 0.63 |
|  | 256 | 125 ns | MPFR | 124 ns | 1.01 |
|  | 1,000 | 283 ns | rug | 295 ns | 0.96 |
|  | 10,000 | 3.97 µs | rug | 3.95 µs | 1.01 |
|  | 1,000,000 | 1.45 ms | MPFR | 3.30 ms | 0.44 |

Assigning, `x op= y` rounded at the precision of `x` (`add_assign_round` and the others) against `mpfr_add (x, x, y, rnd)`, rug's `op_assign_round` and malachite-float's `_prec_round_assign_ref`; and the operators, rounding to nearest at the larger precision, with primitive and `Number` operands against MPFR's `mpfr_add_si`, `mpfr_mul_d` and `mpfr_div_z` with a new variable:

| Operation | Precision | `a` | Fastest | Its time | `a`/fastest |
| --- | --- | ---: | --- | ---: | ---: |
| `add_assign_round` | 24 | 5.61 ns | MPFR | 4.58 ns | 1.22 |
|  | 53 | 4.75 ns | MPFR | 4.75 ns | 1.00 |
|  | 113 | 6.84 ns | MPFR | 5.22 ns | 1.31 |
|  | 256 | 14.1 ns | MPFR | 9.86 ns | 1.43 |
|  | 1,000 | 18.6 ns | MPFR | 15.2 ns | 1.22 |
|  | 10,000 | 74.2 ns | MPFR | 126 ns | 0.59 |
|  | 1,000,000 | 6.05 µs | MPFR | 11.7 µs | 0.52 |
| `mul_assign_round` | 24 | 9.83 ns | rug | 8.50 ns | 1.16 |
|  | 53 | 9.60 ns | rug | 8.49 ns | 1.13 |
|  | 113 | 11.4 ns | MPFR | 8.81 ns | 1.30 |
|  | 256 | 22.2 ns | MPFR | 25.4 ns | 0.87 |
|  | 1,000 | 104 ns | MPFR | 106 ns | 0.98 |
|  | 10,000 | 3.17 µs | rug | 3.43 µs | 0.93 |
|  | 1,000,000 | 563 µs | malachite-float | 981 µs | 0.57 |
| `div_assign_round` | 24 | 15.4 ns | MPFR | 12.9 ns | 1.19 |
|  | 53 | 15.4 ns | MPFR | 13.0 ns | 1.19 |
|  | 113 | 21.9 ns | MPFR | 21.6 ns | 1.01 |
|  | 256 | 63.4 ns | MPFR | 70.5 ns | 0.90 |
|  | 1,000 | 197 ns | MPFR | 224 ns | 0.88 |
|  | 10,000 | 4.70 µs | rug | 5.73 µs | 0.82 |
|  | 1,000,000 | 1.34 ms | rug | 4.03 ms | 0.33 |

| Operation | Precision | `a` | Fastest | Its time | `a`/fastest |
| --- | --- | ---: | --- | ---: | ---: |
| `&a + &b` | 53 | 7.16 ns | malachite-float | 8.33 ns | 0.86 |
|  | 1,000 | 32.9 ns | MPFR | 27.1 ns | 1.21 |
|  | 10,000 | 92.8 ns | rug | 126 ns | 0.74 |
| `&a * &b` | 53 | 5.88 ns | malachite-float | 8.81 ns | 0.67 |
|  | 1,000 | 106 ns | rug | 113 ns | 0.94 |
|  | 10,000 | 2.72 µs | rug | 3.40 µs | 0.80 |
| `&a / &b` | 53 | 8.45 ns | malachite-float | 14.0 ns | 0.60 |
|  | 1,000 | 194 ns | rug | 234 ns | 0.83 |
|  | 10,000 | 4.65 µs | MPFR | 5.74 µs | 0.81 |
| `&a + i64` | 53 | 5.21 ns | rug | 28.1 ns | 0.18 |
|  | 1,000 | 36.4 ns | rug | 32.7 ns | 1.11 |
|  | 10,000 | 110 ns | MPFR | 72.7 ns | 1.52 |
| `&a * f64` | 53 | 4.46 ns | MPFR | 26.3 ns | 0.17 |
|  | 1,000 | 46.9 ns | rug | 41.8 ns | 1.12 |
|  | 10,000 | 90.2 ns | rug | 111 ns | 0.81 |
| `&a / &Number` | 53 | 29.3 ns | rug | 40.5 ns | 0.72 |
|  | 1,000 | 118 ns | MPFR | 81.5 ns | 1.44 |
|  | 10,000 | 412 ns | rug | 452 ns | 0.91 |

Sums and dot products rounded once, `sum_round` against `mpfr_sum` and `dot_round` against `mpfr_dot`, both into an existing variable, of `n` terms of exponents from −20 to 20:

| Operation | Precision | `a` | Fastest | Its time | `a`/fastest |
| --- | --- | ---: | --- | ---: | ---: |
| `sum_round` | 53, n = 4 | 30.0 ns | MPFR | 36.6 ns | 0.82 |
|  | 53, n = 100 | 421 ns | MPFR | 392 ns | 1.07 |
|  | 256, n = 4 | 49.6 ns | MPFR | 37.4 ns | 1.33 |
|  | 1,000, n = 4 | 82.5 ns | MPFR | 56.1 ns | 1.47 |
|  | 1,000, n = 100 | 1.10 µs | MPFR | 1.02 µs | 1.07 |
|  | 10,000, n = 4 | 337 ns | MPFR | 317 ns | 1.06 |
|  | 10,000, n = 1,000 | 48.1 µs | MPFR | 65.2 µs | 0.74 |
| `dot_round` | 53, n = 4 | 107 ns | MPFR | 140 ns | 0.77 |
|  | 53, n = 100 | 1.36 µs | MPFR | 2.82 µs | 0.48 |
|  | 256, n = 4 | 204 ns | MPFR | 188 ns | 1.09 |
|  | 1,000, n = 4 | 590 ns | MPFR | 564 ns | 1.05 |
|  | 1,000, n = 100 | 9.83 µs | MPFR | 13.5 µs | 0.73 |
|  | 10,000, n = 4 | 14.1 µs | MPFR | 16.0 µs | 0.88 |
|  | 10,000, n = 1,000 | 3.40 ms | MPFR | 4.09 ms | 0.83 |

- One and two limbs, up to 128 bits, are added, multiplied, divided and rooted in registers, the mantissa held in the number itself with no vector: into an existing number `a` takes 0.21 to 1.10 of the fastest reference's time but for the quotient at 113 bits (1.30), `set_mul` at 128 bits 0.33 and `x·x` 0.21, where MPFR's two-limb kernels stop at 127 bits; with a new result 0.15 to 1.00, malachite-float being the fastest reference there at most sizes. From `10^4` bits the products, quotients and roots are those of the crate's integers, with short products for the squares and the fused products: 0.34 to 1.04 into an existing number, 0.33 to 1.01 with a new result.
- Between three and sixteen limbs (256 and 1,000 bits) the sums and differences are behind MPFR's (`set_add` 1.32 and 1.34, `set_sub` 1.22 and 1.29), and so are the assigning forms at 13 of their 28 sizes up to 1,000 bits (up to 1.43); the products and quotients there are at par or ahead. See [below](#where-the-references-are-faster).
- `dot_round` takes 0.48 to 0.84 of `mpfr_dot`'s time from 16 terms, and with four 0.77 to 1.09; `sum_round` 0.56 to 1.47 of `mpfr_sum`'s, behind for 4 to 100 terms of 256 and 1,000 bits (1.04 to 1.47), for 100 terms of 53 bits (1.07) and for four of `10^4` bits (1.06).

### Roots and powers

With a new result, against `mpfr_rec_sqrt`, `mpfr_cbrt`, `mpfr_rootn_ui`, `mpfr_pow_si` and `mpfr_pow_z` into an existing variable; `pow_rational_round` against malachite-float's `pow_rational`, as MPFR has no `x^(p/q)` rounded once:

| Operation | Precision | `a` | Fastest | Its time | `a`/fastest |
| --- | --- | ---: | --- | ---: | ---: |
| `rec_sqrt_round` | 53 | 16.2 ns | MPFR | 58.8 ns | 0.28 |
|  | 113 | 29.4 ns | MPFR | 101 ns | 0.29 |
|  | 1,000 | 490 ns | MPFR | 425 ns | 1.15 |
|  | 10,000 | 7.73 µs | MPFR | 9.18 µs | 0.84 |
|  | 100,000 | 166 µs | MPFR | 270 µs | 0.61 |
| `cbrt_round` | 53 | 53.8 ns | MPFR | 188 ns | 0.28 |
|  | 113 | 96.2 ns | MPFR | 240 ns | 0.40 |
|  | 1,000 | 708 ns | MPFR | 698 ns | 1.02 |
|  | 10,000 | 9.18 µs | MPFR | 9.04 µs | 1.02 |
|  | 100,000 | 229 µs | MPFR | 286 µs | 0.80 |
| `nth_root_round(5)` | 53 | 73.9 ns | MPFR | 234 ns | 0.32 |
|  | 113 | 135 ns | MPFR | 314 ns | 0.43 |
|  | 1,000 | 1.01 µs | MPFR | 1.05 µs | 0.96 |
|  | 10,000 | 16.6 µs | MPFR | 17.1 µs | 0.97 |
|  | 100,000 | 349 µs | MPFR | 626 µs | 0.56 |
| `nth_root_round(100)` | 53 | 625 ns | MPFR | 1.81 µs | 0.34 |
|  | 113 | 1.54 µs | MPFR | 4.14 µs | 0.37 |
|  | 1,000 | 6.38 µs | MPFR | 43.9 µs | 0.14 |
|  | 10,000 | 73.7 µs | MPFR | 1.18 ms | 0.063 |
| `powi_round(7)` | 53 | 24.9 ns | MPFR | 62.0 ns | 0.40 |
|  | 113 | 41.8 ns | MPFR | 80.6 ns | 0.52 |
|  | 1,000 | 360 ns | MPFR | 417 ns | 0.86 |
|  | 10,000 | 11.5 µs | MPFR | 11.4 µs | 1.01 |
|  | 100,000 | 178 µs | malachite-float | 293 µs | 0.61 |
| `powi_round(-3)` | 53 | 28.7 ns | MPFR | 101 ns | 0.28 |
|  | 113 | 51.8 ns | MPFR | 110 ns | 0.47 |
|  | 1,000 | 410 ns | MPFR | 482 ns | 0.85 |
|  | 10,000 | 12.4 µs | MPFR | 13.6 µs | 0.91 |
|  | 100,000 | 212 µs | MPFR | 417 µs | 0.51 |
| `powi_round(1_000_000_007)` | 53 | 120 ns | MPFR | 429 ns | 0.28 |
|  | 113 | 178 ns | MPFR | 595 ns | 0.30 |
|  | 1,000 | 3.36 µs | MPFR | 4.37 µs | 0.77 |
| `pow_number_round(2^64 + 13)` | 53 | 224 ns | MPFR | 462 ns | 0.48 |
|  | 113 | 411 ns | MPFR | 969 ns | 0.42 |
|  | 1,000 | 4.84 µs | MPFR | 5.95 µs | 0.81 |
|  | 10,000 | 164 µs | MPFR | 140 µs | 1.17 |
| `pow_rational_round(5/3)` | 53 | 580 ns | malachite-float | 11.9 µs | 0.049 |
|  | 113 | 696 ns | malachite-float | 16.3 µs | 0.043 |
|  | 1,000 | 1.50 µs | malachite-float | 64.4 µs | 0.023 |
|  | 10,000 | 19.2 µs | malachite-float | 2.36 ms | 0.0081 |

The roots and powers take 0.0081 to 1.17 of the fastest reference's time; behind are `rec_sqrt_round` at 1,000 bits (1.15) and `pow_number_round` at `10^4` bits (1.17).

### Conversions

Against `mpfr_get_d`, `mpfr_set_d`, `mpfr_floor`, `mpfr_rint`, `mpfr_set`, `mpfr_cmp`, `mpfr_get_si`, `mpfr_get_z`, `mpfr_set_z`, `mpfr_set_q` and `mpfr_get_q`, and the Rust crates' conversions. `set_f64`, `set_floor` and `set_rint` write into an existing number, as MPFR does; `from_f64_round`, `floor`, `rint_round`, `prec_round` and `from_number` make a new one, against MPFR's `mpfr_init2`, the operation and `mpfr_clear`. `floor` is of a number with half its bits after the point, `rint_round`, `set_rint` and `prec_round` to half the precision, `from_number` of an integer of twice the precision's bits, `from_rational` of a fraction of parts of the precision's bits, `to_i64_round` of numbers below `2^62`:

| Operation | Precision | `a` | Fastest | Its time | `a`/fastest |
| --- | --- | ---: | --- | ---: | ---: |
| `to_f64_round` | 53 | 3.20 ns | MPFR | 5.36 ns | 0.60 |
|  | 1,000 | 4.38 ns | MPFR | 4.60 ns | 0.95 |
|  | 100,000 | 4.43 ns | MPFR | 4.57 ns | 0.97 |
| `set_f64` | 53 | 2.98 ns | rug | 5.69 ns | 0.52 |
|  | 1,000 | 3.97 ns | MPFR | 7.68 ns | 0.52 |
|  | 100,000 | 209 ns | MPFR | 132 ns | 1.58 |
|  | 1,000,000 | 447 ns | MPFR | 574 ns | 0.78 |
| `from_f64_round` | 53 | 4.48 ns | malachite-float | 14.1 ns | 0.32 |
|  | 1,000 | 12.6 ns | dashu-float | 16.1 ns | 0.79 |
|  | 100,000 | 170 ns | dashu-float | 15.3 ns | 11.1 |
|  | 1,000,000 | 734 ns | dashu-float | 14.6 ns | 50.3 |
| `set_floor` | 53 | 9.18 ns | MPFR | 8.41 ns | 1.09 |
|  | 1,000 | 10.9 ns | MPFR | 8.67 ns | 1.26 |
|  | 100,000 | 138 ns | MPFR | 142 ns | 0.97 |
| `floor` | 53 | 10.7 ns | MPFR | 18.0 ns | 0.60 |
|  | 1,000 | 24.5 ns | MPFR | 19.4 ns | 1.26 |
|  | 100,000 | 235 ns | MPFR | 159 ns | 1.48 |
|  | 1,000,000 | 1.80 µs | MPFR | 1.14 µs | 1.58 |
| `set_rint` | 53 | 11.5 ns | MPFR | 10.0 ns | 1.15 |
|  | 1,000 | 12.2 ns | MPFR | 9.37 ns | 1.30 |
|  | 100,000 | 129 ns | MPFR | 101 ns | 1.28 |
| `rint_round(p/2)` | 53 | 11.6 ns | MPFR | 21.6 ns | 0.54 |
|  | 1,000 | 23.3 ns | MPFR | 22.8 ns | 1.02 |
|  | 100,000 | 132 ns | MPFR | 121 ns | 1.08 |
| `prec_round(p/2)` | 53 | 10.5 ns | malachite-float | 8.23 ns | 1.28 |
|  | 1,000 | 21.7 ns | MPFR | 18.8 ns | 1.15 |
|  | 100,000 | 123 ns | MPFR | 183 ns | 0.67 |
| `a < b` | 53 | 2.42 ns | MPFR | 2.06 ns | 1.17 |
|  | 1,000 | 2.43 ns | MPFR | 2.04 ns | 1.19 |
|  | 100,000 | 2.64 ns | MPFR | 2.14 ns | 1.23 |
| `to_i64_round` | 53 | 5.87 ns | malachite-float | 6.58 ns | 0.89 |
|  | 1,000 | 5.52 ns | malachite-float | 21.2 ns | 0.26 |
| `to_number_round` | 53 | 17.7 ns | malachite-float | 5.20 ns | 3.41 |
|  | 1,000 | 40.5 ns | malachite-float | 26.4 ns | 1.53 |
|  | 10,000 | 98.0 ns | malachite-float | 57.3 ns | 1.71 |
|  | 100,000 | 533 ns | malachite-float | 198 ns | 2.69 |
| `from_number` | 53 | 11.6 ns | MPFR | 19.7 ns | 0.59 |
|  | 1,000 | 41.3 ns | MPFR | 22.1 ns | 1.87 |
|  | 10,000 | 94.0 ns | MPFR | 55.8 ns | 1.68 |
|  | 100,000 | 546 ns | MPFR | 204 ns | 2.68 |
| `from_rational` | 53 | 41.0 ns | MPFR | 56.4 ns | 0.73 |
|  | 1,000 | 267 ns | MPFR | 274 ns | 0.98 |
|  | 10,000 | 4.93 µs | MPFR | 5.89 µs | 0.84 |
| `Rational::try_from` | 53 | 13.9 ns | malachite-float | 7.25 ns | 1.92 |
|  | 1,000 | 71.2 ns | MPFR | 25.3 ns | 2.82 |
|  | 10,000 | 189 ns | MPFR | 59.7 ns | 3.16 |
|  | 100,000 | 1.32 µs | MPFR | 502 ns | 2.63 |

The conversions are where `a` is furthest behind: `to_f64_round`, `set_f64`, `floor` and `rint_round` to 113 bits, `prec_round` from 113 bits and `to_i64_round` from 64 are ahead or at par, but comparisons (1.15 to 1.35), the rounding to integers in place at 14 of its 20 sizes (up to 1.30) and the conversions through `Number` and `Rational` from 1,000 bits (1.53 to 3.16) are behind the fastest reference, and `from_f64_round` from `10^4` bits 1.83 to 50.3 of dashu-float's (see [below](#where-the-references-are-faster)).

### Text

Writing against `mpfr_get_str` (and freeing its text): `to_digits(10, None)` gives `1 + ceil(p·log10 2)` digits, as `mpfr_get_str` with `n = 0`, and `to_digits(10, Some(20))` twenty; the exponents are those of the other tables, or up to `±2^60` where marked. Reading against `mpfr_strtofr` into an existing variable, texts of the digits given, at their worth of bits, `ceil(n·log2 10)`, or at 53 bits:

| Operation | Size | `a` | Fastest | Its time | `a`/fastest |
| --- | --- | ---: | --- | ---: | ---: |
| `to_digits(10, None)` | 53 | 47.0 ns | MPFR | 207 ns | 0.23 |
|  | 1,000 | 519 ns | MPFR | 965 ns | 0.54 |
|  | 10,000 | 11.9 µs | MPFR | 16.1 µs | 0.74 |
|  | 1,000,000 | 4.76 ms | MPFR | 9.86 ms | 0.48 |
|  | 53, exponents to `±2^60` | 286 ns | MPFR | 1.27 µs | 0.23 |
| `to_digits(10, Some(20))` | 1,000 | 106 ns | MPFR | 177 ns | 0.60 |
|  | 1,000,000 | 114 ns | MPFR | 183 ns | 0.62 |
| `from_str_radix(10)`, `ceil(n·log2 10)` bits | 20 digits | 49.2 ns | MPFR | 119 ns | 0.41 |
|  | 1,000 digits | 2.20 µs | MPFR | 3.16 µs | 0.70 |
|  | 1,000,000 digits | 13.7 ms | malachite-float | 30.8 ms | 0.45 |
| `from_str_radix(10)`, 53 bits | 20 digits | 44.5 ns | MPFR | 121 ns | 0.37 |
|  | 1,000 digits | 101 ns | MPFR | 615 ns | 0.17 |
|  | 1,000,000 digits | 36.9 µs | MPFR | 504 µs | 0.073 |
| `from_str_radix` | 1 to 19 decimal digits, 53 bits | 31.7 ns | MPFR | 98.6 ns | 0.32 |
|  | 100 hexadecimal digits, 400 bits | 193 ns | MPFR | 193 ns | 1.00 |

`to_string()`, the shortest digits that read back, against rug's `to_string()`, which writes all the digits of `mpfr_get_str`: not the same work (rug's text is longer and has no search for the shortest), so not counted:

| Operation | Precision | `a` | rug | `a`/rug |
| --- | --- | ---: | ---: | ---: |
| `to_string()` | 53 | 124 ns | 317 ns | 0.39 |
|  | 128 | 396 ns | 366 ns | 1.08 |
|  | 1,000 | 882 ns | 1.10 µs | 0.80 |
|  | 1,000,000 | 4.91 ms | 9.83 ms | 0.50 |

Writing takes 0.18 to 0.74 of the fastest reference's time and reading 0.073 to 1.00, at par only for 100 hexadecimal digits (1.00).

### Exponentials, logarithms and powers

Against `mpfr_exp`, `mpfr_exp2`, `mpfr_exp10`, `mpfr_expm1`, `mpfr_log`, `mpfr_log2`, `mpfr_log10`, `mpfr_log1p` and `mpfr_pow` into an existing variable, malachite-float, astro-float and dashu-float; the logarithms of arguments in `[1/4, 4)`, the powers `x^y` of `x` in `[1/4, 4)` and `y` of magnitudes in `[1/4, 4)`:

| Operation | Precision | `a` | Fastest | Its time | `a`/fastest | Arb (does less) |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| `exp` | 53 | 352 ns | MPFR | 550 ns | 0.64 | 94.1 ns |
|  | 113 | 453 ns | MPFR | 780 ns | 0.58 | 125 ns |
|  | 1,000 | 3.41 µs | MPFR | 5.38 µs | 0.63 | 1.96 µs |
|  | 10,000 | 252 µs | MPFR | 297 µs | 0.85 | 65.4 µs |
|  | 100,000 | 5.65 ms | MPFR | 8.67 ms | 0.65 | 3.67 ms |
|  | 1,000,000 | 95.9 ms | malachite-float | 167 ms | 0.57 | 79.1 ms |
| `exp2` | 53 | 393 ns | MPFR | 690 ns | 0.57 | 183 ns |
|  | 113 | 556 ns | MPFR | 954 ns | 0.58 | 233 ns |
|  | 1,000 | 3.57 µs | MPFR | 5.66 µs | 0.63 | 2.14 µs |
|  | 10,000 | 253 µs | MPFR | 293 µs | 0.86 | 70.1 µs |
|  | 100,000 | 5.76 ms | MPFR | 8.85 ms | 0.65 | 3.90 ms |
| `exp10` | 53 | 441 ns | MPFR | 1.08 µs | 0.41 | 190 ns |
|  | 113 | 578 ns | MPFR | 1.88 µs | 0.31 | 237 ns |
|  | 1,000 | 3.71 µs | MPFR | 13.2 µs | 0.28 | 2.14 µs |
|  | 10,000 | 253 µs | MPFR | 458 µs | 0.55 | 70.7 µs |
|  | 100,000 | 5.68 ms | MPFR | 15.5 ms | 0.37 | 3.88 ms |
| `exp_m1` | 53 | 447 ns | MPFR | 653 ns | 0.69 | 116 ns |
|  | 113 | 603 ns | MPFR | 890 ns | 0.68 | 153 ns |
|  | 1,000 | 3.62 µs | MPFR | 5.54 µs | 0.65 | 1.96 µs |
|  | 10,000 | 252 µs | MPFR | 291 µs | 0.87 | 66.2 µs |
|  | 100,000 | 5.65 ms | MPFR | 8.74 ms | 0.65 | 3.83 ms |
| `ln` | 53 | 207 ns | MPFR | 390 ns | 0.53 | 114 ns |
|  | 113 | 311 ns | MPFR | 1.15 µs | 0.27 | 149 ns |
|  | 1,000 | 5.94 µs | MPFR | 7.35 µs | 0.81 | 1.69 µs |
|  | 10,000 | 160 µs | MPFR | 168 µs | 0.95 | 83.8 µs |
|  | 100,000 | 4.91 ms | MPFR | 6.75 ms | 0.73 | 4.22 ms |
|  | 1,000,000 | 72.5 ms | MPFR | 170 ms | 0.43 | 94.3 ms |
| `ln_1p` | 53 | 272 ns | MPFR | 439 ns | 0.62 | 135 ns |
|  | 113 | 378 ns | MPFR | 1.21 µs | 0.31 | 175 ns |
|  | 1,000 | 6.12 µs | MPFR | 7.15 µs | 0.86 | 1.69 µs |
|  | 10,000 | 160 µs | MPFR | 167 µs | 0.96 | 81.6 µs |
|  | 100,000 | 4.92 ms | MPFR | 6.83 ms | 0.72 | 4.31 ms |
| `log2` | 53 | 225 ns | MPFR | 480 ns | 0.47 | 170 ns |
|  | 113 | 328 ns | MPFR | 1.21 µs | 0.27 | 221 ns |
|  | 1,000 | 6.38 µs | MPFR | 7.48 µs | 0.85 | 1.94 µs |
|  | 10,000 | 172 µs | MPFR | 175 µs | 0.98 | 91.0 µs |
|  | 100,000 | 5.03 ms | MPFR | 7.03 ms | 0.72 | 4.41 ms |
| `log10` | 53 | 278 ns | MPFR | 787 ns | 0.35 | 174 ns |
|  | 113 | 383 ns | MPFR | 2.16 µs | 0.18 | 223 ns |
|  | 1,000 | 6.32 µs | MPFR | 14.1 µs | 0.45 | 1.92 µs |
|  | 10,000 | 168 µs | MPFR | 339 µs | 0.50 | 89.2 µs |
|  | 100,000 | 5.03 ms | MPFR | 13.5 ms | 0.37 | 4.54 ms |
| `pow` | 53 | 693 ns | MPFR | 1.06 µs | 0.65 | 281 ns |
|  | 113 | 948 ns | MPFR | 2.03 µs | 0.47 | 347 ns |
|  | 1,000 | 9.74 µs | MPFR | 12.9 µs | 0.76 | 3.73 µs |
|  | 10,000 | 414 µs | MPFR | 473 µs | 0.88 | 150 µs |
|  | 100,000 | 10.6 ms | MPFR | 15.5 ms | 0.69 | 7.95 ms |

### Hyperbolic functions

Against `mpfr_sinh` and the others; `acosh` of arguments in `[1, 8)` and `atanh` of magnitudes in `[1/16, 1)`:

| Operation | Precision | `a` | Fastest | Its time | `a`/fastest | Arb (does less) |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| `sinh` | 53 | 529 ns | MPFR | 749 ns | 0.71 | 193 ns |
|  | 113 | 651 ns | MPFR | 926 ns | 0.70 | 252 ns |
|  | 1,000 | 3.81 µs | MPFR | 5.70 µs | 0.67 | 2.23 µs |
|  | 10,000 | 254 µs | MPFR | 300 µs | 0.85 | 72.4 µs |
|  | 100,000 | 5.71 ms | MPFR | 9.00 ms | 0.64 | 3.88 ms |
| `cosh` | 53 | 532 ns | MPFR | 624 ns | 0.85 | 183 ns |
|  | 113 | 649 ns | MPFR | 902 ns | 0.72 | 243 ns |
|  | 1,000 | 3.82 µs | MPFR | 5.68 µs | 0.67 | 2.21 µs |
|  | 10,000 | 254 µs | MPFR | 302 µs | 0.84 | 75.1 µs |
|  | 100,000 | 5.72 ms | MPFR | 8.81 ms | 0.65 | 3.84 ms |
| `tanh` | 53 | 531 ns | MPFR | 676 ns | 0.79 | 209 ns |
|  | 113 | 662 ns | MPFR | 882 ns | 0.75 | 272 ns |
|  | 1,000 | 3.87 µs | MPFR | 5.75 µs | 0.67 | 2.28 µs |
|  | 10,000 | 254 µs | MPFR | 297 µs | 0.85 | 71.7 µs |
|  | 100,000 | 5.69 ms | MPFR | 9.02 ms | 0.63 | 3.83 ms |
| `sinh_cosh` | 53 | 589 ns | MPFR | 872 ns | 0.68 | 221 ns |
|  | 113 | 944 ns | MPFR | 947 ns | 1.00 | 284 ns |
|  | 1,000 | 4.21 µs | MPFR | 5.61 µs | 0.75 | 2.27 µs |
|  | 10,000 | 254 µs | MPFR | 286 µs | 0.89 | 71.2 µs |
|  | 100,000 | 5.89 ms | MPFR | 9.16 ms | 0.64 | 3.85 ms |
| `asinh` | 53 | 379 ns | MPFR | 654 ns | 0.58 | 285 ns |
|  | 113 | 699 ns | MPFR | 1.46 µs | 0.48 | 351 ns |
|  | 1,000 | 6.74 µs | MPFR | 7.47 µs | 0.90 | 2.20 µs |
|  | 10,000 | 171 µs | MPFR | 170 µs | 1.00 | 86.6 µs |
|  | 100,000 | 5.15 ms | MPFR | 6.84 ms | 0.75 | 4.45 ms |
| `acosh` | 53 | 363 ns | MPFR | 549 ns | 0.66 | 242 ns |
|  | 113 | 643 ns | MPFR | 1.32 µs | 0.49 | 316 ns |
|  | 1,000 | 6.80 µs | MPFR | 7.50 µs | 0.91 | 2.17 µs |
|  | 10,000 | 171 µs | MPFR | 171 µs | 1.00 | 88.1 µs |
|  | 100,000 | 5.02 ms | MPFR | 6.86 ms | 0.73 | 4.45 ms |
| `atanh` | 53 | 332 ns | MPFR | 984 ns | 0.34 | 200 ns |
|  | 113 | 548 ns | MPFR | 1.49 µs | 0.37 | 256 ns |
|  | 1,000 | 6.56 µs | MPFR | 7.72 µs | 0.85 | 1.97 µs |
|  | 10,000 | 169 µs | MPFR | 171 µs | 0.99 | 87.5 µs |
|  | 100,000 | 4.95 ms | MPFR | 6.84 ms | 0.72 | 4.37 ms |

### Circular functions

Against `mpfr_sin` and the others; arguments of magnitudes in `[1/4, 4)` for the sine, the cosine, the tangent and their reciprocals, `[1/8, 4)` for `atan` and `atan2`, `[1/16, 1)` for `asin` and `acos`:

| Operation | Precision | `a` | Fastest | Its time | `a`/fastest | Arb (does less) |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| `sin` | 53 | 213 ns | MPFR | 571 ns | 0.37 | 110 ns |
|  | 113 | 278 ns | MPFR | 716 ns | 0.39 | 140 ns |
|  | 1,000 | 3.84 µs | MPFR | 4.73 µs | 0.81 | 2.16 µs |
|  | 10,000 | 184 µs | MPFR | 282 µs | 0.65 | 91.2 µs |
|  | 100,000 | 7.27 ms | MPFR | 13.9 ms | 0.52 | 5.05 ms |
|  | 1,000,000 | 243 ms | malachite-float | 248 ms | 0.98 | 142 ms |
| `cos` | 53 | 209 ns | MPFR | 367 ns | 0.57 | 106 ns |
|  | 113 | 264 ns | MPFR | 496 ns | 0.53 | 134 ns |
|  | 1,000 | 3.69 µs | MPFR | 4.02 µs | 0.92 | 2.14 µs |
|  | 10,000 | 184 µs | MPFR | 276 µs | 0.67 | 88.3 µs |
|  | 100,000 | 7.19 ms | MPFR | 13.7 ms | 0.53 | 5.11 ms |
|  | 1,000,000 | 243 ms | malachite-float | 251 ms | 0.97 | 142 ms |
| `tan` | 53 | 354 ns | MPFR | 652 ns | 0.54 | 167 ns |
|  | 113 | 450 ns | MPFR | 909 ns | 0.49 | 226 ns |
|  | 1,000 | 4.43 µs | MPFR | 5.38 µs | 0.82 | 2.51 µs |
|  | 10,000 | 196 µs | MPFR | 285 µs | 0.69 | 97.2 µs |
|  | 100,000 | 7.29 ms | MPFR | 14.1 ms | 0.52 | 5.20 ms |
| `sin_cos` | 53 | 275 ns | MPFR | 558 ns | 0.49 | 124 ns |
|  | 113 | 336 ns | MPFR | 718 ns | 0.47 | 153 ns |
|  | 1,000 | 4.14 µs | MPFR | 4.58 µs | 0.90 | 2.26 µs |
|  | 10,000 | 187 µs | MPFR | 295 µs | 0.63 | 89.6 µs |
|  | 100,000 | 7.30 ms | MPFR | 13.7 ms | 0.53 | 5.20 ms |
| `atan` | 53 | 531 ns | MPFR | 1.05 µs | 0.51 | 103 ns |
|  | 113 | 954 ns | MPFR | 1.57 µs | 0.61 | 144 ns |
|  | 1,000 | 8.17 µs | MPFR | 16.2 µs | 0.51 | 1.95 µs |
|  | 10,000 | 288 µs | MPFR | 383 µs | 0.75 | 120 µs |
|  | 100,000 | 11.1 ms | MPFR | 14.1 ms | 0.79 | 5.87 ms |
|  | 1,000,000 | 244 ms | MPFR | 346 ms | 0.71 | 153 ms |
| `atan2` | 53 | 640 ns | MPFR | 1.11 µs | 0.58 | 174 ns |
|  | 113 | 1.09 µs | MPFR | 1.94 µs | 0.56 | 229 ns |
|  | 1,000 | 8.45 µs | MPFR | 18.7 µs | 0.45 | 2.30 µs |
|  | 10,000 | 294 µs | MPFR | 402 µs | 0.73 | 123 µs |
|  | 100,000 | 11.3 ms | MPFR | 14.4 ms | 0.78 | 6.16 ms |
| `asin` | 53 | 765 ns | MPFR | 1.12 µs | 0.68 | 335 ns |
|  | 113 | 1.19 µs | MPFR | 1.93 µs | 0.62 | 412 ns |
|  | 1,000 | 8.83 µs | MPFR | 18.4 µs | 0.48 | 2.74 µs |
|  | 10,000 | 296 µs | MPFR | 397 µs | 0.74 | 137 µs |
|  | 100,000 | 11.3 ms | MPFR | 14.0 ms | 0.81 | 5.96 ms |
| `acos` | 53 | 834 ns | MPFR | 1.15 µs | 0.72 | 365 ns |
|  | 113 | 1.29 µs | MPFR | 2.04 µs | 0.63 | 448 ns |
|  | 1,000 | 8.99 µs | MPFR | 18.7 µs | 0.48 | 2.78 µs |
|  | 10,000 | 295 µs | MPFR | 399 µs | 0.74 | 137 µs |
|  | 100,000 | 11.3 ms | MPFR | 14.0 ms | 0.80 | 5.94 ms |

| Operation | Precision | `a` | Fastest | Its time | `a`/fastest | Arb (does less) |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| `sec` | 24 | 330 ns | malachite-float | 300 ns | 1.10 | 125 ns |
|  | 53 | 324 ns | MPFR | 412 ns | 0.79 | 156 ns |
|  | 1,000 | 4.03 µs | MPFR | 4.30 µs | 0.94 | 2.35 µs |
| `csc` | 24 | 324 ns | MPFR | 535 ns | 0.60 | 122 ns |
|  | 53 | 317 ns | MPFR | 688 ns | 0.46 | 156 ns |
|  | 1,000 | 4.28 µs | MPFR | 5.32 µs | 0.81 | 2.35 µs |
| `cot` | 24 | 378 ns | malachite-float | 601 ns | 0.63 | 131 ns |
|  | 53 | 361 ns | MPFR | 769 ns | 0.47 | 167 ns |
|  | 1,000 | 4.48 µs | MPFR | 5.74 µs | 0.78 | 2.54 µs |

Arguments `m·2^k` of a large exponent, at 53 bits, the reduction taking π to `k` bits and more. For `k = 10^6` Arb returns `[−1, 1]` at once, which is not a result:

| Operation | Argument | `a` | Fastest | Its time | `a`/fastest | Arb (does less) |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| `sin` | `m·2^100` | 287 ns | MPFR | 795 ns | 0.36 | 131 ns |
|  | `m·2^10000` | 7.11 µs | MPFR | 10.0 µs | 0.71 | 9.17 µs |
|  | `m·2^1000000` | 1.37 ms | MPFR | 3.91 ms | 0.35 | 8.09 ns |
| `cos` | `m·2^100` | 283 ns | MPFR | 605 ns | 0.47 | 133 ns |
|  | `m·2^10000` | 7.08 µs | MPFR | 9.25 µs | 0.77 | 9.13 µs |
|  | `m·2^1000000` | 1.37 ms | MPFR | 3.91 ms | 0.35 | 7.81 ns |
| `tan` | `m·2^100` | 441 ns | MPFR | 850 ns | 0.52 | 189 ns |
|  | `m·2^10000` | 7.25 µs | MPFR | 8.92 µs | 0.81 | 9.28 µs |
|  | `m·2^1000000` | 1.38 ms | MPFR | 3.94 ms | 0.35 | 11.3 ns |

The elementary functions take 0.15 to 1.10 of the fastest reference's time, MPFR's at almost every size: `sec` at 24 bits is the one behind (1.10, against malachite-float), and `log2`, `asinh`, `acosh` and `atanh` at `10^4` bits are at par (0.98 to 1.00). Arb, which does less, takes less time at every size but for `ln` at `10^6` bits and the reduction of `m·2^(10^4)`.

### Constants

The first call in a fresh process, each sample a process of its own, when no library has the constant yet: up to 16,384 bits `a` cuts it from its table, beyond it sums the series; MPFR computes from the first bit (`e` as `mpfr_exp` of 1 and `ln 10` as `mpfr_log` of 10, as it has neither constant):

| Operation | Precision | `a` | Fastest | Its time | `a`/fastest | Arb (does less) |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| `pi` | 53 | 1.75 µs | MPFR | 5.54 µs | 0.32 | 6.08 µs |
|  | 1,000 | 1.83 µs | MPFR | 14.5 µs | 0.13 | 10.2 µs |
|  | 10,000 | 1.83 µs | MPFR | 93.0 µs | 0.020 | 191 µs |
|  | 100,000 | 1.36 ms | MPFR | 3.14 ms | 0.43 | 1.67 ms |
|  | 1,000,000 | 19.3 ms | MPFR | 84.9 ms | 0.23 | 27.8 ms |
| `e` | 53 | 1.88 µs | dashu-float | 12.8 µs | 0.15 | 37.2 µs |
|  | 1,000 | 1.79 µs | dashu-float | 21.7 µs | 0.083 | 49.5 µs |
|  | 10,000 | 1.83 µs | dashu-float | 84.8 µs | 0.022 | 124 µs |
|  | 100,000 | 776 µs | dashu-float | 1.38 ms | 0.56 | 941 µs |
|  | 1,000,000 | 7.92 ms | malachite-float | 38.2 ms | 0.21 | 12.6 ms |
| `ln2` | 53 | 2.42 µs | MPFR | 7.83 µs | 0.31 | 9.75 µs |
|  | 1,000 | 2.38 µs | MPFR | 29.4 µs | 0.081 | 13.4 µs |
|  | 10,000 | 2.42 µs | MPFR | 261 µs | 0.0093 | 253 µs |
|  | 100,000 | 2.44 ms | MPFR | 5.94 ms | 0.41 | 3.03 ms |
|  | 1,000,000 | 38.1 ms | malachite-float | 120 ms | 0.32 | 54.5 ms |
| `ln10` | 53 | 1.75 µs | MPFR | 14.1 µs | 0.12 | 43.0 µs |
|  | 1,000 | 1.83 µs | MPFR | 48.2 µs | 0.038 | 124 µs |
|  | 10,000 | 2.25 µs | MPFR | 567 µs | 0.0040 | 509 µs |
|  | 100,000 | 3.85 ms | MPFR | 15.9 ms | 0.24 | 6.95 ms |
|  | 1,000,000 | 59.2 ms | MPFR | 383 ms | 0.15 | 122 ms |
| `euler_gamma` | 53 | 1.75 µs | MPFR | 26.8 µs | 0.065 | 5.50 µs |
|  | 1,000 | 1.75 µs | MPFR | 237 µs | 0.0074 | 9.46 µs |
|  | 10,000 | 2.38 µs | MPFR | 3.93 ms | 0.00060 | 2.40 ms |
|  | 100,000 | 32.4 ms | malachite-float | 81.6 ms | 0.40 | 47.3 ms |
|  | 1,000,000 | 507 ms | malachite-float | 1.97 s | 0.26 | 938 ms |
| `catalan` | 53 | 2.04 µs | MPFR | 19.1 µs | 0.11 | 101 µs |
|  | 1,000 | 1.75 µs | MPFR | 92.9 µs | 0.019 | 127 µs |
|  | 10,000 | 1.88 µs | MPFR | 1.49 ms | 0.0013 | 509 µs |
|  | 100,000 | 6.32 ms | MPFR | 42.4 ms | 0.15 | 8.88 ms |
|  | 1,000,000 | 111 ms | MPFR | 959 ms | 0.12 | 179 ms |

A later call, the constant already computed: `a` returns a new number, cut from its table or copied from the value it keeps, against MPFR's cached constants into an existing variable (`mpfr_const_pi (x, rnd)`; MPFR computes `exp(1)` and `log(10)` again at every call) and the Rust crates' caches:

| Operation | Precision | `a` | Fastest | Its time | `a`/fastest | Arb (does less) |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| `pi` | 53 | 8.77 ns | MPFR | 7.51 ns | 1.17 | 6.34 ns |
|  | 256 | 17.2 ns | MPFR | 8.23 ns | 2.10 | 6.83 ns |
|  | 10,000 | 31.5 ns | MPFR | 17.0 ns | 1.85 | 51.8 ns |
|  | 1,000,000 | 1.19 µs | MPFR | 1.03 µs | 1.16 | 3.73 µs |
| `e` | 53 | 8.82 ns | astro-float | 31.2 ns | 0.28 | 10.1 ns |
|  | 256 | 17.8 ns | astro-float | 38.8 ns | 0.46 | 10.6 ns |
|  | 10,000 | 34.2 ns | astro-float | 133 ns | 0.26 | 51.7 ns |
|  | 1,000,000 | 1.20 µs | astro-float | 18.7 µs | 0.064 | 3.71 µs |
| `ln2` | 53 | 8.57 ns | MPFR | 7.34 ns | 1.17 | 5.66 ns |
|  | 256 | 17.0 ns | MPFR | 6.57 ns | 2.59 | 6.80 ns |
|  | 10,000 | 31.2 ns | MPFR | 16.9 ns | 1.85 | 52.4 ns |
|  | 1,000,000 | 1.23 µs | MPFR | 1.03 µs | 1.20 | 3.71 µs |
| `ln10` | 53 | 8.49 ns | astro-float | 37.1 ns | 0.23 | 9.04 ns |
|  | 256 | 17.1 ns | astro-float | 98.0 ns | 0.17 | 10.1 ns |
|  | 10,000 | 32.8 ns | astro-float | 6.23 µs | 0.0053 | 51.7 ns |
|  | 1,000,000 | 1.21 µs | astro-float | 497 µs | 0.0024 | 3.73 µs |
| `euler_gamma` | 53 | 8.74 ns | MPFR | 7.72 ns | 1.13 | 6.29 ns |
|  | 256 | 18.3 ns | MPFR | 8.46 ns | 2.16 | 6.59 ns |
|  | 10,000 | 31.5 ns | MPFR | 18.0 ns | 1.75 | 51.7 ns |
|  | 1,000,000 | 1.19 µs | MPFR | 1.04 µs | 1.14 | 3.81 µs |
| `catalan` | 53 | 8.30 ns | MPFR | 7.26 ns | 1.14 | 9.30 ns |
|  | 256 | 16.6 ns | MPFR | 8.17 ns | 2.03 | 10.2 ns |
|  | 10,000 | 31.4 ns | MPFR | 17.7 ns | 1.78 | 51.2 ns |
|  | 1,000,000 | 1.19 µs | MPFR | 1.03 µs | 1.16 | 3.74 µs |

The first call takes 0.00060 to 0.56 of the fastest reference's time at every size. A cached π, `ln 2`, γ or `G` takes 1.10 to 2.62 of MPFR's time into an existing variable (see [below](#where-the-references-are-faster)); `e` and `ln 10`, which MPFR does not keep, 0.0024 to 0.46 of the fastest reference's.

### Complex numbers

Into an existing number, at the precisions of its parts: `set_add` and the others against `mpc_add (r, a, b, rnd)` and the others and rug's `assign_round`, a precision being that of both parts of the operands and of the result; `set_add_real`, `set_mul_real` and `set_div_real` against `mpc_add_fr` and the others, and `BigFloat::set_complex_norm` and `set_complex_abs` against `mpc_norm` and `mpc_abs`:

| Operation | Precision | `a` | Fastest | Its time | `a`/fastest |
| --- | --- | ---: | --- | ---: | ---: |
| `set_add` | 53 | 6.86 ns | MPC | 9.65 ns | 0.71 |
|  | 256 | 24.2 ns | MPC | 20.3 ns | 1.20 |
|  | 1,000 | 34.9 ns | MPC | 29.4 ns | 1.19 |
|  | 100,000 | 1.06 µs | rug | 2.18 µs | 0.48 |
| `set_mul` | 53 | 21.6 ns | MPC | 48.9 ns | 0.44 |
|  | 256 | 142 ns | rug | 117 ns | 1.22 |
|  | 1,000 | 411 ns | MPC | 456 ns | 0.90 |
|  | 100,000 | 159 µs | rug | 316 µs | 0.50 |
| `set_sqr` | 53 | 15.4 ns | MPC | 92.9 ns | 0.17 |
|  | 256 | 112 ns | MPC | 180 ns | 0.62 |
|  | 1,000 | 271 ns | rug | 356 ns | 0.76 |
|  | 100,000 | 129 µs | MPC | 180 µs | 0.71 |
| `set_div` | 53 | 342 ns | MPC | 379 ns | 0.90 |
|  | 256 | 591 ns | rug | 621 ns | 0.95 |
|  | 1,000 | 1.45 µs | MPC | 1.37 µs | 1.06 |
|  | 100,000 | 534 µs | MPC | 817 µs | 0.65 |
| `set_sqrt` | 53 | 76.9 ns | MPC | 168 ns | 0.46 |
|  | 256 | 578 ns | MPC | 479 ns | 1.21 |
|  | 1,000 | 1.12 µs | MPC | 1.12 µs | 1.00 |
|  | 100,000 | 435 µs | rug | 596 µs | 0.73 |
| `set_complex_norm` | 53 | 9.06 ns | MPC | 95.8 ns | 0.095 |
|  | 256 | 76.6 ns | MPC | 172 ns | 0.45 |
|  | 1,000 | 184 ns | MPC | 307 ns | 0.60 |
|  | 100,000 | 85.0 µs | MPC | 132 µs | 0.64 |
| `set_complex_abs` | 53 | 76.8 ns | MPC | 95.5 ns | 0.80 |
|  | 256 | 211 ns | MPC | 230 ns | 0.92 |
|  | 1,000 | 425 ns | MPC | 517 ns | 0.82 |
|  | 100,000 | 193 µs | MPC | 303 µs | 0.64 |
| `set_add_real` | 53 | 6.09 ns | MPC | 7.98 ns | 0.76 |
|  | 256 | 17.2 ns | MPC | 14.4 ns | 1.20 |
|  | 1,000 | 20.6 ns | MPC | 18.6 ns | 1.11 |
|  | 100,000 | 709 ns | MPC | 1.36 µs | 0.52 |
| `set_mul_real` | 53 | 4.52 ns | MPC | 12.7 ns | 0.35 |
|  | 256 | 24.7 ns | MPC | 47.7 ns | 0.52 |
|  | 1,000 | 191 ns | MPC | 210 ns | 0.91 |
|  | 100,000 | 99.1 µs | MPC | 179 µs | 0.55 |
| `set_div_real` | 53 | 9.23 ns | MPC | 34.0 ns | 0.27 |
|  | 256 | 105 ns | MPC | 148 ns | 0.71 |
|  | 1,000 | 372 ns | MPC | 451 ns | 0.82 |
|  | 100,000 | 238 µs | MPC | 384 µs | 0.62 |
| `set_conj` | 53 | 4.11 ns | MPC | 6.05 ns | 0.68 |
|  | 256 | 8.34 ns | MPC | 7.66 ns | 1.09 |
|  | 1,000 | 6.64 ns | MPC | 6.75 ns | 0.98 |
|  | 100,000 | 333 ns | MPC | 325 ns | 1.02 |

With a new result, against `mpc_init2`, the operation and `mpc_clear`, and rug's `with_val`:

| Operation | Precision | `a` | Fastest | Its time | `a`/fastest |
| --- | --- | ---: | --- | ---: | ---: |
| `add_round` | 53 | 11.6 ns | rug | 32.9 ns | 0.35 |
|  | 256 | 49.8 ns | MPC | 44.4 ns | 1.12 |
|  | 1,000 | 65.6 ns | MPC | 55.8 ns | 1.18 |
|  | 100,000 | 1.41 µs | MPC | 2.30 µs | 0.61 |
| `mul_round` | 53 | 25.8 ns | MPC | 81.8 ns | 0.32 |
|  | 256 | 175 ns | rug | 147 ns | 1.19 |
|  | 1,000 | 450 ns | MPC | 498 ns | 0.90 |
|  | 100,000 | 161 µs | MPC | 316 µs | 0.51 |
| `sqr_round` | 53 | 19.4 ns | MPC | 113 ns | 0.17 |
|  | 256 | 140 ns | MPC | 203 ns | 0.69 |
|  | 1,000 | 306 ns | MPC | 373 ns | 0.82 |
|  | 100,000 | 129 µs | MPC | 181 µs | 0.71 |
| `div_round` | 53 | 346 ns | rug | 427 ns | 0.81 |
|  | 256 | 579 ns | MPC | 609 ns | 0.95 |
|  | 1,000 | 1.41 µs | rug | 1.40 µs | 1.00 |
|  | 100,000 | 532 µs | MPC | 864 µs | 0.62 |
| `sqrt_round` | 53 | 82.2 ns | MPC | 182 ns | 0.45 |
|  | 256 | 595 ns | rug | 486 ns | 1.23 |
|  | 1,000 | 1.14 µs | MPC | 1.13 µs | 1.01 |
|  | 100,000 | 432 µs | MPC | 592 µs | 0.73 |
| `norm_round` | 53 | 11.4 ns | MPC | 104 ns | 0.11 |
|  | 256 | 92.7 ns | MPC | 178 ns | 0.52 |
|  | 1,000 | 202 ns | rug | 306 ns | 0.66 |
|  | 100,000 | 78.4 µs | MPC | 123 µs | 0.64 |
| `abs_round` | 53 | 86.0 ns | MPC | 106 ns | 0.81 |
|  | 256 | 222 ns | rug | 233 ns | 0.95 |
|  | 1,000 | 444 ns | MPC | 518 ns | 0.86 |
|  | 100,000 | 193 µs | MPC | 301 µs | 0.64 |
| `add_real_round` | 53 | 9.46 ns | MPC | 34.5 ns | 0.28 |
|  | 256 | 41.8 ns | MPC | 39.9 ns | 1.05 |
|  | 1,000 | 46.7 ns | MPC | 44.3 ns | 1.05 |
|  | 100,000 | 894 ns | MPC | 1.53 µs | 0.59 |
| `mul_int_round` | 53 | 13.9 ns | MPC | 44.2 ns | 0.31 |
|  | 256 | 58.3 ns | MPC | 49.1 ns | 1.19 |
|  | 1,000 | 93.0 ns | MPC | 59.4 ns | 1.57 |
|  | 100,000 | 1.13 µs | MPC | 1.75 µs | 0.65 |
| `mul_i_round` | 53 | 12.0 ns | MPC | 27.6 ns | 0.43 |
|  | 256 | 28.0 ns | MPC | 30.6 ns | 0.91 |
|  | 1,000 | 30.3 ns | MPC | 29.5 ns | 1.03 |
|  | 100,000 | 360 ns | MPC | 368 ns | 0.98 |

Assigning, `z op= w` (`add_assign_round` and the others), against `mpc_add (z, z, w, rnd)` and rug's `op_assign_round`:

| Operation | Precision | `a` | Fastest | Its time | `a`/fastest |
| --- | --- | ---: | --- | ---: | ---: |
| `add_assign_round` | 24 | 12.2 ns | MPC | 9.45 ns | 1.29 |
|  | 113 | 14.0 ns | MPC | 11.0 ns | 1.28 |
|  | 256 | 27.6 ns | MPC | 19.1 ns | 1.44 |
|  | 1,000 | 37.9 ns | MPC | 33.5 ns | 1.13 |
|  | 100,000 | 1.12 µs | rug | 2.29 µs | 0.49 |
| `sub_assign_round` | 24 | 11.3 ns | MPC | 10.4 ns | 1.08 |
|  | 113 | 13.9 ns | MPC | 12.5 ns | 1.12 |
|  | 256 | 29.0 ns | MPC | 20.9 ns | 1.39 |
|  | 1,000 | 36.8 ns | rug | 32.7 ns | 1.13 |
|  | 100,000 | 1.13 µs | rug | 2.30 µs | 0.49 |
| `mul_assign_round` | 24 | 38.9 ns | MPC | 106 ns | 0.37 |
|  | 113 | 61.7 ns | rug | 137 ns | 0.45 |
|  | 256 | 180 ns | MPC | 168 ns | 1.07 |
|  | 1,000 | 445 ns | MPC | 522 ns | 0.85 |
|  | 100,000 | 163 µs | MPC | 319 µs | 0.51 |
| `div_assign_round` | 24 | 361 ns | MPC | 336 ns | 1.08 |
|  | 113 | 469 ns | MPC | 493 ns | 0.95 |
|  | 256 | 591 ns | rug | 606 ns | 0.98 |
|  | 1,000 | 1.42 µs | MPC | 1.40 µs | 1.02 |
|  | 100,000 | 534 µs | rug | 830 µs | 0.64 |

Products, squares, norms and the operations with a real operand of one and two limbs a part take 0.059 to 0.59 of the fastest reference's time; from `10^5` bits everything is ahead (0.47 to 0.73, the exact copies `set_neg`, `set_conj` and `mul_i_round` at par). Sums at 256 and 1,000 bits and the assigning sums at most sizes to 1,000 bits are behind, as for `BigFloat`.

### Functions of complex numbers

Against `mpc_exp` and the others into an existing variable; arguments of parts of magnitudes in `[1/8, 2)`, the powers' exponents `w` of a real part in `[4, 8)` and an imaginary part in `[1/2, 1)`, and `n = 17` for `powi`:

| Operation | Precision | `a` | Fastest | Its time | `a`/fastest |
| --- | --- | ---: | --- | ---: | ---: |
| `exp` | 53 | 801 ns | MPC | 1.43 µs | 0.56 |
|  | 1,000 | 8.51 µs | MPC | 18.4 µs | 0.46 |
|  | 100,000 | 12.8 ms | MPC | 42.0 ms | 0.30 |
| `ln` | 53 | 1.22 µs | MPC | 1.57 µs | 0.78 |
|  | 1,000 | 15.9 µs | MPC | 27.0 µs | 0.59 |
|  | 100,000 | 16.3 ms | MPC | 22.3 ms | 0.73 |
| `log10` | 53 | 1.51 µs | MPC | 2.38 µs | 0.64 |
|  | 1,000 | 17.6 µs | MPC | 34.9 µs | 0.50 |
|  | 100,000 | 16.5 ms | MPC | 28.5 ms | 0.58 |
| `sin` | 53 | 1.41 µs | MPC | 1.76 µs | 0.80 |
|  | 1,000 | 9.16 µs | MPC | 11.2 µs | 0.82 |
|  | 100,000 | 12.9 ms | MPC | 21.7 ms | 0.59 |
| `cos` | 53 | 1.40 µs | MPC | 1.53 µs | 0.92 |
|  | 1,000 | 9.20 µs | MPC | 11.2 µs | 0.82 |
|  | 100,000 | 12.9 ms | MPC | 21.7 ms | 0.59 |
| `tan` | 53 | 1.84 µs | MPC | 3.63 µs | 0.51 |
|  | 1,000 | 10.2 µs | MPC | 18.0 µs | 0.57 |
|  | 100,000 | 13.2 ms | MPC | 23.0 ms | 0.57 |
| `sin_cos` | 53 | 1.76 µs | MPC | 1.83 µs | 0.97 |
|  | 1,000 | 9.79 µs | MPC | 11.4 µs | 0.86 |
|  | 100,000 | 13.1 ms | MPC | 22.9 ms | 0.57 |
| `sinh` | 53 | 1.48 µs | MPC | 1.65 µs | 0.90 |
|  | 1,000 | 9.24 µs | MPC | 12.3 µs | 0.75 |
|  | 100,000 | 12.9 ms | MPC | 21.8 ms | 0.59 |
| `cosh` | 53 | 1.46 µs | MPC | 1.65 µs | 0.89 |
|  | 1,000 | 9.20 µs | MPC | 12.3 µs | 0.75 |
|  | 100,000 | 13.0 ms | MPC | 21.8 ms | 0.59 |
| `tanh` | 53 | 1.93 µs | MPC | 4.48 µs | 0.43 |
|  | 1,000 | 10.4 µs | MPC | 16.7 µs | 0.62 |
|  | 100,000 | 13.3 ms | MPC | 23.0 ms | 0.58 |
| `asin` | 53 | 3.44 µs | MPC | 6.91 µs | 0.50 |
|  | 1,000 | 19.8 µs | MPC | 141 µs | 0.14 |
|  | 100,000 | 17.1 ms | MPC | 21.5 ms | 0.79 |
| `acos` | 53 | 3.54 µs | MPC | 9.34 µs | 0.38 |
|  | 1,000 | 19.9 µs | MPC | 147 µs | 0.14 |
|  | 100,000 | 17.1 ms | MPC | 42.8 ms | 0.40 |
| `atan` | 53 | 1.76 µs | MPC | 4.50 µs | 0.39 |
|  | 1,000 | 17.5 µs | MPC | 52.1 µs | 0.34 |
|  | 100,000 | 16.5 ms | MPC | 42.4 ms | 0.39 |
| `asinh` | 53 | 3.47 µs | MPC | 8.99 µs | 0.39 |
|  | 1,000 | 19.6 µs | MPC | 121 µs | 0.16 |
|  | 100,000 | 16.9 ms | MPC | 21.8 ms | 0.78 |
| `acosh` | 53 | 3.46 µs | MPC | 10.9 µs | 0.32 |
|  | 1,000 | 20.0 µs | MPC | 102 µs | 0.20 |
|  | 100,000 | 17.1 ms | MPC | 21.9 ms | 0.78 |
| `atanh` | 53 | 1.71 µs | MPC | 5.00 µs | 0.34 |
|  | 1,000 | 17.6 µs | MPC | 52.7 µs | 0.33 |
|  | 100,000 | 16.4 ms | MPC | 42.4 ms | 0.39 |
| `arg` | 53 | 611 ns | MPC | 1.16 µs | 0.53 |
|  | 1,000 | 8.60 µs | MPC | 19.3 µs | 0.45 |
|  | 100,000 | 11.3 ms | MPC | 14.6 ms | 0.78 |
| `pow` | 53 | 3.33 µs | MPC | 6.40 µs | 0.52 |
|  | 1,000 | 26.6 µs | MPC | 68.2 µs | 0.39 |
|  | 100,000 | 29.6 ms | MPC | 130 ms | 0.23 |
| `pow_real` | 53 | 3.21 µs | MPC | 6.36 µs | 0.51 |
|  | 1,000 | 26.5 µs | MPC | 64.1 µs | 0.41 |
|  | 100,000 | 29.4 ms | MPC | 66.0 ms | 0.45 |
| `powi(17)` | 53 | 247 ns | MPC | 769 ns | 0.32 |
|  | 1,000 | 2.00 µs | MPC | 2.09 µs | 0.96 |
|  | 100,000 | 670 µs | MPC | 1.04 ms | 0.64 |
| `proj` | 53 | 11.1 ns | MPC | 6.08 ns | 1.83 |
|  | 1,000 | 29.8 ns | MPC | 6.59 ns | 4.52 |
|  | 100,000 | 264 ns | MPC | 238 ns | 1.11 |

The functions take 0.0085 to 1.03 of MPC's time (`asin` and `acos` at `10^4` bits 0.0085 and 0.0087); `powi(17)` from 113 to `10^4` bits and `sin_cos` at 256 bits are at par (0.93 to 1.03), and `proj`, a copy of both parts into a new number, is behind (1.11 to 4.52).

### Matrices of floats

`&a * &b` of `n × n` matrices, every element correctly rounded, against `mpfr_dot` (or `mpc_dot`) for each element into an existing variable, which does the same work; the loop of `mpfr_fma` rounded at each step and Arb's `arb_mat_mul` do less, and their columns are their times alone:

| Operation | Precision | `a` | Fastest | Its time | `a`/fastest | `mpfr_fma` loop (does less) | Arb (does less) |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| `&a * &b`, 4 × 4 | 53 | 882 ns | MPFR | 2.62 µs | 0.34 | 642 ns | 466 ns |
|  | 113 | 957 ns | MPFR | 2.79 µs | 0.34 | 1.30 µs | 626 ns |
|  | 256 | 2.22 µs | MPFR | 3.32 µs | 0.67 | 1.91 µs | 1.29 µs |
|  | 1,000 | 7.22 µs | MPFR | 9.01 µs | 0.80 | 7.46 µs | 5.35 µs |
| `&a * &b`, 8 × 8 | 53 | 5.32 µs | MPFR | 18.2 µs | 0.29 | 5.87 µs | 2.71 µs |
|  | 113 | 6.04 µs | MPFR | 18.5 µs | 0.33 | 13.1 µs | 3.42 µs |
|  | 256 | 15.0 µs | MPFR | 23.3 µs | 0.64 | 19.0 µs | 9.07 µs |
|  | 1,000 | 52.4 µs | MPFR | 70.1 µs | 0.75 | 62.8 µs | 41.5 µs |
| `&a * &b`, 16 × 16 | 53 | 40.2 µs | MPFR | 147 µs | 0.27 | 74.3 µs | 20.9 µs |
|  | 113 | 45.9 µs | MPFR | 145 µs | 0.32 | 130 µs | 25.5 µs |
|  | 256 | 111 µs | MPFR | 178 µs | 0.62 | 169 µs | 64.6 µs |
|  | 1,000 | 404 µs | MPFR | 554 µs | 0.73 | 509 µs | 325 µs |
| `BigComplex`, 4 × 4 | 53 | 2.96 µs | MPC | 7.12 µs | 0.41 | – | – |
|  | 256 | 7.70 µs | MPC | 10.0 µs | 0.77 | – | – |
| `BigComplex`, 8 × 8 | 53 | 20.4 µs | MPC | 60.9 µs | 0.34 | – | – |
|  | 256 | 53.0 µs | MPC | 74.5 µs | 0.71 | – | – |

`a.solve(&b)` with one column has no reference that does the same work: MPFR's is Gaussian elimination with partial pivoting by `mpfr_fma`, rounded at each step, where `a` rounds each element's exact dot product once. Its times, for what they are worth (the copies of `A` and `b` included on both sides), not counted:

| Operation | Precision | `a` | MPFR elimination | `a`/MPFR elimination |
| --- | --- | ---: | ---: | ---: |
| `a.solve(&b)`, 4 × 4 | 53 | 928 ns | 465 ns | 1.99 |
|  | 113 | 1.04 µs | 844 ns | 1.23 |
|  | 256 | 2.53 µs | 1.59 µs | 1.59 |
|  | 1,000 | 6.12 µs | 5.32 µs | 1.15 |
| `a.solve(&b)`, 8 × 8 | 53 | 3.41 µs | 2.90 µs | 1.18 |
|  | 113 | 4.04 µs | 5.80 µs | 0.70 |
|  | 256 | 10.6 µs | 9.76 µs | 1.08 |
|  | 1,000 | 31.5 µs | 32.3 µs | 0.98 |
| `a.solve(&b)`, 16 × 16 | 53 | 16.1 µs | 20.8 µs | 0.77 |
|  | 113 | 19.9 µs | 47.9 µs | 0.41 |
|  | 256 | 64.6 µs | 72.0 µs | 0.90 |
|  | 1,000 | 189 µs | 217 µs | 0.87 |

The products take 0.27 to 0.80 of the per-element `mpfr_dot`'s time, and less than the `mpfr_fma` loop's from 8 × 8.

### Where the references are faster

Every case above 1.05 times the fastest same-work reference's time, by cause: 180 of the 1305 cases of the scoreboard that have one. The causes are those found in the speed pass, or say that none was established.

- **Sums and differences of four and sixteen limbs** (256 and 1,000 bits), in every form: `set_add` and `set_sub` 1.22 to 1.34, `add_round`, `sub_round` and `&a + &b` at 1,000 bits 1.12 to 1.21, the assigning forms 1.22 to 1.43, and the complex sums built on them 1.09 to 1.44. Four limbs take `small::add_short` and sixteen `same::add`, which writes into the result's limbs in place after MPFR's `mpfr_add1sp`: 12.5 ns against 9.49 ns at 256 bits and 18.6 ns against 13.9 ns at 1,000. The cause of the rest was not established.
- **`x op= y` of one and two limbs** (1.08 to 1.31: `add_assign_round` at 24 and 113 bits, `mul_assign_round` at 24, 53, 64 and 113, `div_assign_round` at 24 and 53, and the complex `+=`, `-=` and `/=` there): the same register kernels as `set_add`, after the match on both operands' forms and a copy of the first one, against MPFR's `mpfr_add (x, x, y, rnd)`, its plain call; `set_add` at 24 bits takes 3.90 ns and `add_assign_round` 5.61 ns.
- **Quotients of two limbs and of operands of other lengths** (1.15 to 1.37): `set_div` at 113 bits (1.30) divides exactly where MPFR's `mpfr_div_2` takes an approximate quotient and a test of the bits it needs; a result of 53 bits from operands of 1,000 or `10^5` (1.32 and 1.34) takes two divisions of a limb, one for each end of the interval of the truncated operands, against MPFR's one; and 1,000 bits from operands of 53 (1.37, with a new result 1.15 against malachite-float) a quotient of 17 limbs by one, each limb waiting for the remainder before it.
- **`mul_add_round` and `set_mul_add` at 1,000 bits** (1.18 to 1.19): operands past four limbs take the general exact path, the whole product in working memory and then the sum; the cause of the difference was not established.
- **Square roots at 64 and 256 bits** (`set_sqrt` 1.10 and 1.12), and the complex roots at 128 and 256 bits that run on them (1.06 to 1.23): from 256 bits both take the integer root by Zimmermann's method (GMP's `mpn_sqrtrem` for MPFR); the cause of the difference at four limbs was not established, nor at 64 bits, where the root is of one full limb.
- **Exact sums of few terms** (`sum_round` 1.06 to 1.47, `dot_round` of four terms at 256 and 1,000 bits 1.09): the window sum, after Lefèvre's algorithm as `mpfr_sum` has it, has fixed costs (its two buffers made zero, every term shifted into them); that they are the difference was not measured.
- **Primitive and `Number` operands from 1,000 bits** (`&a + i64` 1.11 and 1.52, `&a * f64` 1.12, `&a / &Number` 1.44): the operand is made a float in the one each thread keeps and the general path follows, where `mpfr_add_si` and `mpfr_mul_d` take the word directly; `mpfr_div_z` reads the integer's limbs, `a` converts the `Number`'s bytes first.
- **`rec_sqrt_round` at 1,000 bits** (1.15): below 2,000 bits `a` takes a square root and a division, which it found faster there than its Newton's iteration, where MPFR runs Newton's iteration; **`pow_number_round(2^64 + 13)` at `10^4` bits** (1.17): the 65 squarings of about 160 limbs at the working precision; the cause of the difference was not established.
- **Comparisons**, `a < b` at every size (1.15 to 1.35, 2.42 ns against 2.06 ns at 53 bits): the tests of both numbers' kinds and of the mantissa's form before the exponents; not narrowed further.
- **Rounding to an integer or to another precision** (`set_floor` and `set_rint` 1.08 to 1.30 from 24 bits to `10^5`, `floor` with a new result from 256 bits 1.15 to 1.58, `rint_round` at `10^4` and `10^5` bits 1.08, `prec_round` at 24 to 64 bits against malachite-float and at 1,000 bits 1.15 to 1.28): the copy of the limbs, the scan of the fraction and the rounding, as MPFR's; the cause of the difference was not established.
- **Conversions through `Number` and `Rational`** (`to_number_round` 1.53 to 3.41, `from_number` 1.21 to 2.68, `from_rational` 1.24 to 1.45, `Rational::try_from` 1.33 to 3.16): a `Number` is a vector of bytes, so each conversion makes a vector of limbs from them, or writes the limbs back as bytes, besides the result's own vector; malachite-float's `Integer` keeps a word in place (17.7 ns against 5.20 ns at 53 bits), and MPFR's `mpfr_get_z`, `mpfr_set_z` and `mpfr_get_q` read and write GMP's limbs directly (`mpfr_get_z` and `mpfr_get_q` into variables that exist).
- **`from_f64_round` from `10^4` bits against dashu-float** (1.83 to 50.3, 734 ns against 14.6 ns at `10^6` bits): dashu-float keeps the 53 significant bits of a value whose precision is a bound, so nothing grows with the precision, where `a` writes a mantissa of all `p` bits, as MPFR does; it would take mantissas trimmed of their trailing zeros. `set_f64` at `10^5` bits (1.58) makes the limbs below the top one zero, as `mpfr_set_d` does, and is ahead at `10^6` (0.78): not explained; `to_f64_round` at `10^4` bits (1.07): not established.
- **Cached constants** (π, `ln 2`, γ and `G`, 1.10 to 2.62 against MPFR into an existing variable; 17.8 ns against 7.56 ns for π at 1,000 bits): each call makes a new number (a vector allocated from three limbs; one or two limbs are held in the number) and rounds the bits of the table, or from 16,384 bits copies the kept value from behind the cache's `Mutex`, where MPFR copies its cached value into a variable that exists. The forms into an existing number, `set_pi` and the others, have no rows in this run.
- **Complex numbers**, besides the sums and roots above: `set_mul` and `mul_round` at 256 bits (1.22 and 1.19, `*=` 1.07), four products and two sums of four limbs; `set_div` at 24 and 1,000 bits (1.09 and 1.06, `/=` at 24 bits 1.08); `set_conj` at 256 bits (1.09), an exact copy; `mul_int_round` at 256 and 1,000 bits (1.19 and 1.57); and `proj` (1.11 to 4.52), for a finite number a copy of both parts into a new number against `mpc_proj` into a variable, 29.8 ns against 6.59 ns at 1,000 bits. For all but `proj` the cause was not established. **`sec` at 24 bits** (1.10 against malachite-float): not established.

Not counted, as the references do less:

- **Arb**, at almost every size of the elementary functions (`a` takes 0.77 to 7.29 times its time, below 1 only for `ln` at `10^6` bits): tables of the functions at many points, fixed point on GMP's `mpn` level and no Ziv loop, its balls only enclosing the value. It is slower than `a` for the constants' first call and for the reduction of `m·2^(10^4)`.
- **MPFR's Gaussian elimination**, for the solves of 4 × 4 at every size and of 8 × 8 at 53 and 256 bits (1.08 to 1.99): it rounds at each step, where every element of `a`'s factorisation is an exact dot product rounded once, each a new vector.
- **rug's `to_string`** at 128 bits (1.08), which writes all the digits of `mpfr_get_str` with no search for the shortest.

