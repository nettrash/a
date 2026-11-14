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

- `to_f64_round(round)` and `to_f32_round(round)` give the primitive float nearest in the mode, and the direction, as `mpfr_get_d` and `mpfr_get_flt`: correctly rounded among the subnormals too, past the largest finite value an infinity (or the largest, when the mode rounds toward zero), NaN as NaN and the zeros with their signs; `to_f64()` and `to_f32()` round to nearest. `from_f64_round(d, prec, round)` and `from_f32_round` round a primitive float to a precision (`mpfr_set_d`); `From<f64>` and `From<f32>` stay exact.
- `floor`, `ceil`, `trunc`, `round` (halfway cases away from zero) and `round_ties_even` give the integer at the number's precision, which holds it, as `f64`'s methods and MPFR's `mpfr_floor` and the others; a zero result has the sign of the number, so `ceil(−0.5)` is `−0`. `floor_round(prec, round)`, `ceil_round`, `trunc_round`, `round_round` and `round_ties_even_round` round that integer once more, with the direction from the integer (`mpfr_rint_floor` and the others); `rint_round(prec, round)` is `mpfr_rint`, the integer of `prec` bits nearest in the mode's direction, rounded once (10.5 to 2 bits is 12). `fract` and `fract_round` give the fractional part of the number's sign, a zero of its sign for an integer and, as MPFR has it, for an infinity; `is_integer` says whether there is none.
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
- `conj`, `-z` and `mul_i(negative)` (`z·i`, or `z·(−i)`) are exact and keep the precisions. `sqr`, `sqrt`, `norm` and `abs` round to nearest at the precisions of the parts (the larger for the norm and the absolute value); the operators `+`, `-`, `*` and `/`, each operand owned or borrowed, and their assigning forms, round each part to nearest at the larger precision of that part of the operands. `==` compares both parts by value. `Display` writes `(re im)` as MPC's `mpc_out_str` lays it out, each part as `BigFloat`'s `Display` writes it, the formatter's options applied to each part: `(1.5 -8)`; `Debug` writes both parts with `BigFloat`'s `Debug`.
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

- Numbers of one limb (a precision up to 64) and of two (up to 128) are added, multiplied, divided and rooted in registers, after MPFR's `mpfr_add1sp1`, `mpfr_mul_1`, `mpfr_div_1`, `mpfr_sqrt1` and their two-limb forms: the exact result in 128 or 256 bits, or the result truncated with a sticky bit for the bits shifted out, at least one bit longer than the precision, then rounded in registers. A quotient takes one or two divisions of two limbs by one with the reciprocal of the divisor (two or three of three limbs by two), and a root the integer square root of 128 or 256 bits; MPFR's special forms for 63 and 64 bits (127 and 128) are a limb more here.
- A sum is exact, on at most the precision and the operands' lengths together, unless the smaller term lies below a quarter of the result's last place and below the larger term's last bit. Then it changes the result only as a sticky bit would: the sum lies strictly between the larger term and one unit more (or less) of a scale below both, and no number of the precision nor any halfway point lies there.
- Operands of as many limbs as their result (three or more) are added as MPFR's `mpfr_add1sp` and `mpfr_sub1sp` add them: the smaller shifted right into place, one pass of the crate's kernels into the result's vector, and the bits shifted out kept as two bits and a sticky bit. A difference takes one unit off and the complement of those bits, `3 − G` with a sticky bit or `4 − G` exactly; exponents two or more apart cancel at most one bit, which the second bit refills, and exponents at most one apart shift out at most one bit, so the difference is exact and normalised by a shift.
- Longer numbers compute only the limbs their rounding needs, on a window of `n + 1` limbs for `n` limbs of result. A sum adds the window of the larger term and the bits of the smaller in it; what lies below the window, the tails, makes a sticky bit when one term has one, and when both have one the sum is known only within two units of the window's lowest limb: the rounding is then decided when that limb is not within those units of a limb boundary or of a halfway point, and computed exactly otherwise (a chance of about `2^−62` for random operands). A difference of exponents at most one apart, which can cancel any number of bits, is exact.
- A product of operands longer than `n + 2` limbs multiplies their top `n + 2` limbs: the product of the cut operands is below the exact one by less than `B^l + B^l + 1` units of its lowest limb (`B = 2^64`, `l` their length), so the exact product's top limbs are those of the cut one or up to two more, and the rounding is decided as for a sum. Otherwise it is the exact product, by the crate's multiplication: the schoolbook method, Karatsuba, Toom-3, Toom-4 and the number-theoretic transforms for the longest. A square takes the square's methods. A fused multiply-add takes the exact product as an operand of the sum, rounded once.
- A quotient `F = 0.a·2^−c / 0.b` in `[1/2, 1)` (`c = 1` when `0.a ≥ 0.b`) comes from `n + 1` limbs of quotient: for a divisor longer than that, from the top `2(n + 1)` limbs of the dividend and the top `n + 1` of the divisor, whose quotient `Q'` lies within `(Q' − 2, Q' + 1)` of the exact `F·B^(n+1)` and decides as above; otherwise, or when that leaves it undecided, by the exact division of the dividend's top limbs with the remainder for the sticky bit: the schoolbook method, Burnikel and Ziegler's divide and conquer, and Newton's method for the longest.
- A square root takes the top `2(n + 1)` limbs of `0.a·2^−c` (`c` the parity of the exponent) and their integer root by Zimmermann's Karatsuba square root: the floor of the root of the truncated radicand is that of the exact one, and a root is never halfway, so its remainder is needed only when its bits below the round bit are all zero.
- `set_*` and `*_assign_round` round into the vector of the number they write, which has the room its precision needs; a primitive operand of the operators is written into a float that the thread keeps, so that neither allocates.
- A sum of many terms adds them exactly into one two's-complement accumulator from the lowest bit of any term to past the top (one product into a scratch buffer and one shifted addition each), and rounds once, when their bits span at most `4(l + p) + 2^16` bits, `l` the longest term; otherwise the terms are sorted by exponent and added exactly from the largest until the rest is below the last bit of the sum so far and below a quarter of the result's last place, where only the rest's sign counts, which the same accumulation at one bit gives as soon as it stops. A dot product takes the exact products as its terms.

### Roots and powers

- `1/√x` is `2^−j` for `x = 4^j` and otherwise irrational: the floor of `√(floor(2^E / M))` for the mantissa `M` and an exponent `E` of the parity of `x`'s that gives `64(n + 1)` bits of root, which is the floor of `√(2^E / M)`, rounded with a sticky bit. To 63 bits, of a mantissa of one limb, it is `floor(√(floor(2^E / m)))` for `E` of 190 or 191, in registers.
- An `n`-th root of degree up to 16, or of a radicand of up to `2^12` bits (`n·64(n + 1)`), is the integer root of the mantissa shifted to that many bits, by Newton's method on integers; the floor of the root of the truncated radicand is that of the exact one, and the remainder, when the root's bits below its round bit are all zero, tells exact roots and halfway ones apart. A longer one is computed by Newton's method on floats with Ziv's strategy, each approximation `y` certified by `y_lo^n ≤ z ≤ y_hi^n` for `y(1 ∓ 2^(4 − w))` raised by squarings rounded outward, after the exact roots are found (`x = m^n·2^(kn)`, which for an odd `m ≥ 3` needs `n·log2 3` bits).
- `x^n` is exact for `x` a power of two and, for `n > 0`, computed exactly when the odd part `m` of `x` has `n(bits(m) − 1) + 1 ≤ prec + 1`, which covers the exact and the halfway results; otherwise it is neither, and Ziv's strategy ends: the power by squarings from the top bit of `n`, every product cut to its top `wl` limbs, from below, a relative error of `ε = 2^(64 − 64wl)` each, for a total error below `2(k_b + 2)·n·ε` of the result (`k_b` the base's own, one for the cut base or the reciprocal `floor(2^K / M)` of a negative `n`).
- To 64 bits and from a base of up to two limbs, the power is computed in registers first: squarings on 128 bits, each product cut to its top 128 bits (`ε = 2^−127`), the reciprocal `floor(2^191 / m)` for a negative `n`, and the result taken when the roundings of the cut power `c` and of `c + 4k` units (`k ≤ (k_b + 2)·n`) agree, as the rounding is monotone; otherwise as above.
- `x^(p/q)` is exact only when the `q`-th root of `x` is (from `ap + bq = 1`): an exact root is raised to the power and rounded once; otherwise Ziv's strategy on the root rounded toward zero to the working precision (away from zero for `p < 0`, then the reciprocal cut from below), and its power cut as above.

### Conversions

- A primitive float is rounded from the top limb of the mantissa, the lower limbs a sticky bit, in registers: a number `0.1…·2^e` has `min(53, e + 1074)` bits in an `f64`, fewer among the subnormals, and the float's encoding follows from the rounded mantissa alone, a carry taking a subnormal to the least normal number and the largest finite number to the infinity.
- An integer of the number's precision is its mantissa with the fraction cleared and one unit added when the direction rounds up, a carry out of the top making it the next power of two. A number whose last bit is above the point is an integer already and is not expanded: the floor of `3·2^(2^61)` costs a copy.
- The comparisons with integers and `Number`s compare the signs, then the magnitudes `0.m·2^e` and `M·2^l` by their exponents, then by the mantissas aligned at their top bits, limb by limb, without making either a float; with a `Rational` `n/d`, `|x|·d` against `|n|`, exactly.

### Text

- Writing in radix `b = 2^a·c`, `c` odd, takes the integer `D` that `|x|·b^(n−1−E)` rounds to, `E` the exponent of the first digit: first `floor((e − 1)·log_b 2)` from a table of `log_b 2` to 128 bits, which is `E` or one less, and one more when `D` has `n + 1` digits. For a power of two the digits are bits. Otherwise `D` is exact (a product by `c^k`, or a quotient with its remainder) when `|x|·b^k` may be a half-integer, which needs `q + k·log2 c < log2(2y) + 1` for `k ≥ 0` (`q` the bits of the mantissa's odd part) and `c^j` to divide the odd part for `k = −j < 0`, or when the exact work is no larger than the operands and the result. Otherwise it is not a half-integer and Ziv's strategy ends: the mantissa's top `w` bits and `c^k` to `w` bits, each product truncated, bound the value within `[A, A + 4K]` units for `K` truncations (`K ≤ 2k + 3`, as a square doubles the error of its operand), and a quotient within `[Q − 8K, Q + 9]`; the digits are taken when both ends have the same integer part on the same side of the half, otherwise at a working precision half as large again.
- The shortest decimal of `Display` lies in the interval of the numbers that round to `x`, `x ∓ ulp/2` (`x − ulp/4` below a power of two), its ends included when the mantissa is even: at the first scale `10^k` where the interval holds an integer, its integers are `L..=H`, which a coarser scale makes fewer than ten. A multiple of ten among them is the shortest, its zeros dropped; otherwise all have as many digits and the nearest to `x·10^k` is taken. For `k ≥ 0` one power `5^k` gives the three products.
- Reading takes the digits from the first that is not zero to the last that is not zero, the others in the exponent. In a radix that is a power of two, the digits for two bits more than the precision and a sticky bit for the rest. Otherwise, up to 62 bits, the digits of a word, `D·c^k` as a product of two words or `D·2^s/c^j` as a quotient of a double word by a word, rounded in registers; beyond, `D·b^k` rounded once, exactly when it may be a number of the precision or the midpoint of two (`q + k·log2 c < p + 2` for `k ≥ 0`, `c^j` dividing the odd part of `D` for `k = −j`) or when the work is small, and by Ziv's strategy as for writing otherwise. The overflow and the underflow are decided first, from bounds of the bits of `D·b^k` (a table of `log2 b` to 58 bits), so an exponent of any size costs nothing more. A text of more digits than the precision needs is truncated to the digits of the precision and 8 bits more: the number lies strictly between `D·b^t` and `(D + 1)·b^t`, and when both round to the same number, which is not between them, that is its rounding with its direction; otherwise twice as many digits, up to all of them, which decide exactly.
- The digits of a long text are converted by the divide-and-conquer radix conversion of [numbers](radix.md), and those of a long `D` written by it; quotients of up to 40 limbs are Knuth's Algorithm D, longer ones the Newton division of numbers.

### Exponentials, logarithms and powers

- The exponential reduces `x = k ln 2 + r`, `0 ≤ r < 1` (`k` from the top 64 bits of `x` times `1/ln 2`, `ln 2` from the constants with `bits(k)` more bits), then `t = r/2^s`, `s ≈ 0.3·√W` for `W` bits, sums the Taylor series of `e^t` by rectangular splitting (Smith's method: the terms in blocks whose denominator `(a + 1)…b` fits a limb, the integers `(i + 1)…b` as coefficients of the powers `t^1…t^m`, one full product and one division by a limb a block) and squares the sum `s` times, in fixed point. From 11,264 bits (176 limbs) it sums the series by the bit burst instead: `t` cut into pieces of `2^j` bits, each piece's series by binary splitting, exactly, then one division, the pieces multiplied.
- The bound of the exponential: the reduction is within `|k|·δ_ln2 + u` (`u = 2^−W`), the sum within 8 units a block and one for the terms left out (`ceil(E/Q) + 8` with the block's denominator `Q`, from `c_i/Q ≤ 1/(i − a)!`, `Σ j/j! = e` and `t^j` within `j` units), the bit burst within 9 units a piece and one; `s` truncated squarings turn a relative `ε` into at most `2^s(ε + u)(1 + 2^−9)` while `2^s·ε ≤ 2^−20`; the reduction's error multiplies the result by at most `1 + 2(δ_r + 2^s·u)`.
- The logarithm writes `x = 2^e·u`, `2/3 ≤ u < 4/3`, and takes `ln u` by Newton's iteration on the exponential, `y ← y + L(u·e^−y − 1)` with `L(v) = v − v²/2 + v³/3`, from the `f64` logarithm, each step at four times the precision of the one before, so that the steps together cost about a third more than the last. Only the last step counts: `ln u = y + ln(1 + v)` exactly for the exact `v`, so with the `v` it finds, `|v| ≤ 1/4`, the bound is `2δ_v + 5u + |v|^4/2` (`δ_v` from the exponential's relative error). `e·ln 2` adds `|e|·δ_ln2 + u`. For `e = 0` the fixed point takes as many more bits as `u − 1` has leading zeros, and when that is more than the working precision `ln(1 + d)` is `d` within `d²`.
- From 5,000 bits the logarithm is MPFR's AGM, `ln x = π/(2·AGM(1, 4/s)) − m·ln 2` for `s = x·2^m ≥ 2^(W/2 + 8)`, the formula within `ln(s)·k'²/(1 − k'²)` for `k' = 4/s` (DLMF 19.12.1: every term past the first of the series of `K` lies in `(0, k'^(2j)·ln s]`), the AGM enclosed by its last pair.
- At up to 192 bits of working precision (results to about 150 bits) both kernels run in arrays on the stack, step for step the same, as the allocations of the limb vectors cost more than the arithmetic there.
- `exp2` splits `x = n + f` and takes `e^(f·ln 2)·2^n`; `exp10` and `log10` take `ln 10` from the constants' cache, `log2` divides by `ln 2`; `exp_m1` is `e^x − 1` with as many more bits as `x` has zeros after the point, `ln_1p` the logarithm of `1 + x` rounded to as many more.
- `pow` is `e^(y·ln|x|)` with `ln|x|` to `bits(|y·ln x|)` more bits, bounds of `|y·ln x|` from the exponents alone deciding the range and the cases near 1 first. The exact powers: with `x = M·2^e`, `M` odd, and `y = c/2^t`, `c` odd, `x^y` is rational only when `x` is a `2^t`-th power (`M` a square `t` times over, `2^t` dividing `e`), and then it is exact when `M = 1` or `c > 0` and `(M^(1/2^t))^c` has at most `prec + 1` bits.

### Hyperbolic functions

- `sinh x = (u + u/(u + 1))/2` and `tanh x = u/(u + 2)` for `u = e^|x| − 1` or `e^(2|x|) − 1`, sums of terms of one sign; `cosh x = (e^|x| + e^−|x|)/2`. From `|x| ≥ W + 16`, `sinh` and `cosh` are `e^|x|/2` within a relative `2^−(W + 16)`, from the exponential's own exponent, so that a result near the overflow threshold is right where `e^|x|` itself overflows.
- `asinh x = ln(1 + |x| + x²/(1 + √(1 + x²)))` below 1, `ln(|x| + √(x² + 1))` above, `ln|x| + ln 2` within `2^−2e` far out; `acosh(1 + d) = ln(1 + d + √(d(d + 2)))` below 2, with `d` exact; `atanh x = ln(1 + 2|x|/(1 − |x|))/2`, `1 − |x|` exact where it cancels.

### Circular functions

- None of these functions has an exact result for an argument that is not zero but `acos(1) = 0`, so Ziv's loop ends; those values are treated before it. A small argument whose function lies within less than half an ulp of the argument (or of 1, or of `1/x` for a power of two) is rounded directly by the side it lies on: `0 < x − sin x < 2^(3e−2)`, `0 < tan x − x < 2^(3e−1)`, `0 < 1 − cos x < 2^(2e−1)`, `0 < sec x − 1 < 2^(2e)`, `0 < x − atan x < 2^(3e−1)`, `0 < asin x − x < 2^(3e−2)` for `|x| < 2^e ≤ 1/2`, and `0 < csc x − 1/x < 2^(e−2)`, `0 < 1/x − cot x < 2^(e−1)` for `x = 2^(e−1)`.
- The argument is reduced modulo `π/2`, `|x| = kπ/2 + r`, `|r| ≤ π/4`: with π to `W = w + e + 16` bits after the point (the working precision `w` and the exponent `e` of `x`), `N = ⌊|x|·2^(W+1)⌋` divided by `P = ⌊π·2^W⌋`, the quotient rounded to nearest. The error of `r` is at most `k·2^(ε_π − 1)` for π's error `2^ε_π`, plus a unit of `N` when it dropped bits. When `r` comes out smaller than the bits it must keep, near a multiple of `π/2`, the bits the cancellation took are read off `r` and the reduction is done again with that many more, not a fixed number. Below 1 the argument is its own `r`.
- To 512 bits after the point the sine and the cosine of `r` come from arrays, with no allocation: `r = j/64 + ρ`, `0 ≤ ρ < 1/64`, `sin(j/64)` and `cos(j/64)` from a table computed once at 576 bits (within 160 ulps there, one ulp at 512 bits), the series of `ρ` by Horner's rule with the coefficients `1/k!` of a table made when the crate is compiled (within 3.1 ulps: two each step, damped by `ρ² < 2^−12`), and the formulas of the sum: within 16 ulps. For `j = 0`, `sin r = r·Σ(−1)^n·ρ^(2n)/(2n + 1)!` with the whole `r`, relatively.
- Beyond, the argument is halved `K` times to `r' = r/2^K` of the exponent `−K`, `K ≈ (F/2)^(1/3)` for `F` bits, and `t₀ = 1 − cos r'` is its Taylor series summed by rectangular splitting (Smith): the powers `y, …, y^m` of `y = r'²` once, `m ≈ √N` even, the terms of each block of `m` combined with small integer factors exactly, and the blocks by Horner's rule with one division by a product of `2m` integers, a limb of them at a time. The bound: a power `y^i` within `i − 1` ulps, weighted `1/(2n + δ)!` in the sum, the roundings of the rule and the tail, below `3 + m/16` ulps. The doublings `t ← 4t − 2t²` keep the relative error of `t` (an error `η` becomes `η(4 + 4t + 2η) + 2`), so `F` is the precision plus `2K` bits; then `cos r = 1 − t` and `sin r = √(t(2 − t))`, within `η/√t_min + 1` ulps. An argument below `2^−K` takes the series of the sine and the cosine as it is.
- From about 410,000 bits, binary splitting: `r` is the sum of pieces `u_k/2^(2^k)` of `2^(k−1)` bits (Brent's bit-burst method, as MPFR's `mpfr_sincos_fast`), the sine of each piece its Taylor series summed exactly by binary splitting and divided once (within 1.25 ulps), its cosine `√(1 − s²)` (within 3), and the pieces put together by the formulas of the sum, each adding at most 6.3 ulps.
- The tangent and the reciprocals are quotients of the sine and the cosine, `tan(kπ/2 + r)` being `tan r` for an even `k` and `−cot r` for an odd one, with the bound of a quotient `|a/b − ã/b̃| ≤ 2(|ã| + α)β/|b̃|² + α/|b̃|`.
- The arctangent of `z` in `(0, 1]` is Newton's iteration on `tan y = z`, `y ← y − cos y·(sin y − z·cos y)`, at doubling precisions from the `f64` arctangent, the last step's bound read off its own sine and cosine: for `y = θ + ε`, `sin ε = (sin y − z·cos y)·cos θ`, so `|ε| ≤ (π/2)·|sin y − z·cos y|`, and the exact step leaves at most `z·ε² + (2/3)|ε|³`. Below `2^−K`, its series `z·Σ(−1)^n·z^(2n)/(2n + 1)` by rectangular splitting, within `J(3m + 1) + 1` ulps over `J` blocks of `m`.
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

- Sums, differences and products by a real number are `BigFloat`'s own, part by part. A product's parts `ac − bd` and `ad + bc` are each the exact sum of two exact products, rounded once with the sticky shortcut of a sum (the products in registers when every part's significant bits fit one limb); from 1500 bits in every part, of exponents within 32 of each other, by three products, `ad`, `bc` and `(a + b)(c − d)`, so that `ac − bd = (a + b)(c − d) + ad − bc`, still exact. The square's parts are `a² − b²` and `2ab`, rounded once.
- A quotient's parts `(ac + bd)/(c² + d²)` and `(bc − ad)/(c² + d²)` are exact sums divided exactly and rounded once (the quotient with two bits more than the precision and its remainder the sticky bit), as long as those sums are at most `4p + 4096` bits long, `p` the longest part, and the result has fewer than 2000 bits. Longer sums (parts of very different exponents), and results of 2000 bits and more, where two quotients of cut sums cost less than two exact ones of `2p`-bit sums, go by Ziv's strategy: each sum cut to a working precision `w` is within `2^(2−w)` of it, relatively (a term below `2^(e − w − 4)` of the other, `e` that one's exponent, is left out), the quotient of the cut sums cut to `w` bits within `2^(1−w)` of theirs, so the result is within `5.1·2^(1−w)` relatively and within `2^(e+4−w)` for a quotient below `2^e`; when two attempts do not decide (an exact or halfway result), the exact quotient.
- The norm is the exact sum of the exact squares, rounded once. The absolute value is the square root of the exact sum, rounded once: when one square is far below the other the sum is the larger one with a sticky bit below `2p + 4` bits, which the squares of the numbers of `p` bits near the root and of their halfway points (multiples of `2^(2(e − p − 1))`) cannot fall between, so the root of the sum with a half unit in place of the sticky bit rounds as the root does.
- The square root `t ± u·i` (or `u ± t·i` for `a < 0`), `t = √((|z| + |a|)/2)`, `u = |b|/(2t)`, is computed at a working precision `w` from the exact squares, every step cut toward zero to `w` bits: with `η = 2^(1−w)`, `a² + b²` is within `2η`, its root `2.02η`, the sum with `|a|` `4.03η`, `t` `3.03η` and `u` `4.05η`, so a part below `2^e` is within `2^(e+4−w)`; Ziv's strategy rounds each part. A part that is a dyadic number would hold the interval at every precision: after the first attempt the root is tried exactly, `a² + b² = s²` and `(s + |a|)/2 = t²` by integer square roots, then `u = |b|/(2t)` exactly, which is dyadic when `t` is. The check needs `|a|` and `|b|` within `2^(p+4)` of each other: a Pythagorean pair up to powers of two has its odd parts above `m + n`, which bounds the ratio of its legs by `2^(p+1)`.

### Functions of complex numbers

- Every function is MPC's formula on real functions at a working precision `w`, each value with a bound of its error in units of its last place and an exponent of 128 bits, so that an intermediate result may lie beyond the exponent range; each operation rounds toward zero to about `w + 16` bits and adds the unit it drops (a product `|ãb̃ − ab| ≤ |ã|β + |b̃|α + αβ`, a quotient within `(4/3)(ρ_a + ρ_b)` relatively for relative errors `ρ_b ≤ 1/4`, a square root within `α/√ã`, a sum exact at the cut of the larger term less `w + 24` bits, a term below `2^(e − w − 16)` going into the bound whole), and Ziv's strategy rounds each part as soon as its approximation decides it, the working precision growing by half each attempt. The real functions are the module's own: the exponential's kernel with its exponent unbounded (`e^x` within a few units of `2^−w` relatively for `|x| < 2^62`, beyond that over- or underflowing by the sign), the reduction modulo `π/2` and the sine and cosine kernels, the logarithm's kernel (with the bits of `x − 1` it needs), the arctangent's kernel and its quadrants for `atan2`, and `π`, `ln 2`, `ln 10` within `2^(1−w)`.
- `e^(x + yi) = e^x·cos y + i·e^x·sin y`; `ln z = ln|z| + i·atan2(y, x)`, `ln|z| = ln(x² + y²)/2`, and for a larger part of exponent 0 or 1 `ln(1 + d)/2` with `d = x² + y² − 1` exact (the larger square less 1, then the smaller square added exactly unless it lies more than `w + 16` bits below), so that nothing cancels; `log10 z = ln z / ln 10`. `sin z = sin x·cosh y + i·cos x·sinh y`, `cos z = cos x·cosh y − i·sin x·sinh y` (`sinh y` from `u = e^|y| − 1` as `(u + u/(u + 1))/2` below `1/2`); `tan z = (sin x·cos x + i·sinh y·cosh y)/(cos² x + sinh² y)`, the denominator a sum of squares.
- `asin` and `acos` by Hull, Fairgrieve and Tang's formulas: `A = (|z + 1| + |z − 1|)/2`, `Re asin = atan2(x, √((A − |x|)(A + |x|)))`, `Im asin = ±ln(1 + (A − 1) + √((A − 1)(A + 1)))`, with `A − 1` and `A − |x|` sums of terms that are not negative (`|z + 1| − (|x| + 1) = y²/(|z + 1| + |x| + 1)` and its kind); `atan z = atan2(2x, 1 − x² − y²)/2 + (i/4)·ln(1 + 4|y|/(x² + (|y| − 1)²))`. `z^w = e^(w·ln z)`, `w·ln z` within the absolute error of its parts, so that a large one takes more bits.
- A value near a number of the precision would take Ziv's strategy as many bits as it lies near: for tiny arguments, `|x|, |y| < 2^−(p/2 + 4)`, each part is its first term (`x`, `y`, `1`, `−xy`) and a difference computed on its own (`(e^x − 1)·cos y − 2·sin²(y/2)` for `Re e^z`, `(sin x − x)·cosh y + 2x·sinh²(y/2)` for `Re sin z`, by their series), rounded beside the first term when the difference lies below its last place, and otherwise added to it; the inverse functions' parts by the series' first term, `Re f = x(1 + c(x² − 3y²) + θ)` with `|θ| ≤ 2|z|^4` for `asin` (`c = 1/6`) and `atan` (`c = −1/3`), as `|Re z^n| ≤ n|z|^(n−1)|x|` for an odd `n`. The tangent's imaginary part near `±1` is `±1` and `∓(cos 2x + e^(−2|y|))/(2(cos² x + sinh² y))`, of the sign of `−cos 2x` and below `2^(−2^62)` when `|y| ≥ 2^62`. The argument of a tiny dyadic ratio `y/x` (`x > 0`) is rounded beside it, `0 < q − atan q < q³/3`.
- The exact powers are MPC's `mpc_pow_exact`: for a real `w = m·2^e`, `m` odd, `z = (c + di)·2^k` takes `−e` square roots among the Gaussian integers (`c + di` is a square when `c² + d²` and then `(c + √(c² + d²))/2` are), `1/(c + di)` is dyadic when `c² + d²` is a power of 2, and `(c + di)^m` comes by squarings and products; as long as the Gaussian integers have at most twice the precision and 64 bits more (besides the spread of the parts' exponents), and otherwise again after each attempt of Ziv's strategy with twice as many, as MPC does. `powi` with `|n| ≥ 3` (no zero part) goes by binary powering, MPC's `mpc_pow_usi`: the correctly rounded products and squares at the working precision and `bits(n) + 16` more, within `(2k − 1)·2^(2−w)·1.1` of `z^k` in the norm (by induction over the squarings and products), so within `2^(bits(n) + 5 − w)` relatively in all, the exact power after an attempt that does not decide.

### Matrices of floats

- A product's element is `BigFloat::dot_round` of a row and a column, rounded once (for complex elements, each part the exact sum of the exact products of the parts).
- `det`, `solve` and `inverse` factor the matrix by Doolittle's method with partial pivoting: each element of `U`, and of `L` before its division by the pivot, is one exact dot product of the elements before it, rounded once at the working precision, and the substitutions likewise.
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

Measured on an Apple M4 Max on 2026-11-13 and 14 between 21:55 and 00:20 (load averages 1.2 to 1.9, nothing else running), one thread, `env -u MallocNanoZone`, fat LTO: the median of three passes of rounds with fresh arguments each round and the contenders' order rotated, in ns, µs, ms or s a call. The clock read between 3.4 and 4.5 GHz from one table to the next, so the times compare within a table and the ratios across them. The references are MPFR 4.2.2 (Homebrew's static archives, called directly through FFI, with its exponent range set to that of `BigFloat`), Arb in FLINT 3.6 (its balls enclose the value but are not correctly rounded, so its column is for the time only), and GNU MPC 1.4.1 for the complex numbers; rug 1.30 on the same MPFR took the time of MPFR to within a few per cent everywhere, so it has no column. MPFR and MPC write into a variable that exists and `a` returns a new number, unless a table says otherwise. The arguments have the precision of the result and full mantissas, of exponents from −2 to 1 unless a table says otherwise, and every result is rounded to nearest; each harness checks its results against the reference's before it times them. The ratio is `a`'s time over the reference's: below 1, `a` is faster.

### Arithmetic

Into an existing number (`set_add` and the others against `mpfr_add (r, a, b, rnd)` and the others; `x·x` is `set_mul(&x, &x)` against `mpfr_sqr`). A precision `p (operands q)` is a result of `p` bits from operands of `q`; the sums take operands of both signs:

| Operation | Precision | `a` | MPFR | `a`/MPFR |
| --- | --- | ---: | ---: | ---: |
| `set_add` | 24 | 7.5 ns | 5.2 ns | 1.44 |
|  | 53 | 7.4 ns | 5 ns | 1.48 |
|  | 64 | 7.5 ns | 5.6 ns | 1.34 |
|  | 113 | 8.8 ns | 5.8 ns | 1.52 |
|  | 128 | 8.5 ns | 6.3 ns | 1.35 |
|  | 256 | 22.4 ns | 10.2 ns | 2.20 |
|  | 1,000 | 30.6 ns | 14.6 ns | 2.10 |
|  | 10,000 | 116 ns | 92.7 ns | 1.25 |
|  | 100,000 | 716 ns | 988 ns | 0.72 |
|  | 1,000,000 | 9.03 µs | 8.98 µs | 1.01 |
|  | 53 (operands 1,000) | 34.9 ns | 15.2 ns | 2.30 |
|  | 53 (operands 100,000) | 34.5 ns | 12.1 ns | 2.85 |
|  | 1,000 (operands 53) | 38.1 ns | 16.4 ns | 2.32 |
|  | 1,000 (operands 100,000) | 53 ns | 268 ns | 0.20 |
|  | 100,000 (operands 1,000,000) | 1.17 µs | 2.25 µs | 0.52 |
| `set_sub` | 24 | 7.3 ns | 5.5 ns | 1.33 |
|  | 53 | 7.3 ns | 5.6 ns | 1.30 |
|  | 64 | 7.2 ns | 6 ns | 1.20 |
|  | 113 | 8.6 ns | 6.2 ns | 1.39 |
|  | 128 | 8.3 ns | 6.8 ns | 1.22 |
|  | 256 | 22.1 ns | 10.5 ns | 2.10 |
|  | 1,000 | 29.7 ns | 15.3 ns | 1.94 |
|  | 10,000 | 116 ns | 86.2 ns | 1.35 |
|  | 100,000 | 615 ns | 874 ns | 0.70 |
|  | 1,000,000 | 11.8 µs | 10.3 µs | 1.15 |
|  | 53 (operands 1,000) | 35 ns | 15.8 ns | 2.22 |
|  | 53 (operands 100,000) | 34.6 ns | 263 ns | 0.13 |
|  | 1,000 (operands 53) | 38 ns | 16.2 ns | 2.35 |
|  | 1,000 (operands 100,000) | 51.6 ns | 140 ns | 0.37 |
|  | 100,000 (operands 1,000,000) | 1.15 µs | 2.14 µs | 0.54 |
| `set_mul` | 24 | 8.7 ns | 5.4 ns | 1.61 |
|  | 53 | 8.7 ns | 5.4 ns | 1.61 |
|  | 64 | 8.4 ns | 5.4 ns | 1.56 |
|  | 113 | 9.5 ns | 5.9 ns | 1.61 |
|  | 128 | 9.5 ns | 10.8 ns | 0.88 |
|  | 256 | 31 ns | 21.2 ns | 1.46 |
|  | 1,000 | 89.3 ns | 100 ns | 0.89 |
|  | 10,000 | 3.24 µs | 3.36 µs | 0.96 |
|  | 100,000 | 48.6 µs | 92.3 µs | 0.53 |
|  | 1,000,000 | 560 µs | 1.41 ms | 0.40 |
|  | 53 (operands 1,000) | 27.1 ns | 98.6 ns | 0.27 |
|  | 53 (operands 100,000) | 23.9 ns | 43.8 ns | 0.55 |
|  | 1,000 (operands 53) | 22.8 ns | 10.4 ns | 2.19 |
|  | 1,000 (operands 100,000) | 107 ns | 115 ns | 0.93 |
|  | 100,000 (operands 1,000,000) | 48.5 µs | 92.3 µs | 0.53 |
| `x·x` | 24 | 9 ns | 5 ns | 1.80 |
|  | 53 | 8.4 ns | 5.2 ns | 1.62 |
|  | 64 | 8.2 ns | 5.3 ns | 1.55 |
|  | 113 | 9.8 ns | 5.5 ns | 1.78 |
|  | 128 | 9.1 ns | 13.5 ns | 0.67 |
|  | 256 | 30.7 ns | 18 ns | 1.71 |
|  | 1,000 | 76.6 ns | 79.9 ns | 0.96 |
|  | 10,000 | 2.37 µs | 2.04 µs | 1.16 |
|  | 100,000 | 37.2 µs | 60.9 µs | 0.61 |
|  | 1,000,000 | 415 µs | 1.22 ms | 0.34 |
|  | 53 (operands 1,000) | 29.5 ns | 77.8 ns | 0.38 |
|  | 1,000 (operands 100,000) | 85.7 ns | 99.7 ns | 0.86 |
| `set_div` | 24 | 8.4 ns | 6.2 ns | 1.35 |
|  | 53 | 8.4 ns | 6.2 ns | 1.35 |
|  | 64 | 10.1 ns | 6.8 ns | 1.49 |
|  | 113 | 14.8 ns | 8.8 ns | 1.68 |
|  | 128 | 15.3 ns | 26.8 ns | 0.57 |
|  | 256 | 72.8 ns | 61.9 ns | 1.18 |
|  | 1,000 | 250 ns | 220 ns | 1.14 |
|  | 10,000 | 7.03 µs | 5.48 µs | 1.28 |
|  | 100,000 | 127 µs | 195 µs | 0.65 |
|  | 1,000,000 | 1.4 ms | 3.99 ms | 0.35 |
|  | 53 (operands 1,000) | 36.8 ns | 17.4 ns | 2.11 |
|  | 53 (operands 100,000) | 33.3 ns | 15.2 ns | 2.19 |
|  | 1,000 (operands 53) | 83 ns | 36.6 ns | 2.27 |
|  | 1,000 (operands 100,000) | 249 ns | 216 ns | 1.15 |
|  | 100,000 (operands 1,000,000) | 126 µs | 193 µs | 0.65 |
| `set_mul_add` | 24 | 12.3 ns | 11.7 ns | 1.05 |
|  | 53 | 11.9 ns | 12.4 ns | 0.96 |
|  | 64 | 12.2 ns | 22 ns | 0.55 |
|  | 113 | 54.1 ns | 22.6 ns | 2.39 |
|  | 128 | 53.7 ns | 23.5 ns | 2.29 |
|  | 256 | 60.9 ns | 29.4 ns | 2.07 |
|  | 1,000 | 134 ns | 111 ns | 1.21 |
|  | 10,000 | 3.42 µs | 3.66 µs | 0.93 |
|  | 100,000 | 50.2 µs | 106 µs | 0.47 |
|  | 1,000,000 | 573 µs | 1.42 ms | 0.40 |
|  | 53 (operands 1,000) | 118 ns | 110 ns | 1.07 |
|  | 53 (operands 100,000) | 48.6 µs | 107 µs | 0.45 |
|  | 1,000 (operands 53) | 54.9 ns | 23.4 ns | 2.35 |
|  | 1,000 (operands 100,000) | 48.6 µs | 106 µs | 0.46 |
|  | 100,000 (operands 1,000,000) | 563 µs | 1.42 ms | 0.40 |
| `set_sqrt` | 24 | 10.1 ns | 6.2 ns | 1.63 |
|  | 53 | 10.1 ns | 6.3 ns | 1.60 |
|  | 64 | 22.5 ns | 6.8 ns | 3.31 |
|  | 113 | 26.2 ns | 11.9 ns | 2.20 |
|  | 128 | 84.9 ns | 62.8 ns | 1.35 |
|  | 256 | 121 ns | 99.5 ns | 1.22 |
|  | 1,000 | 301 ns | 265 ns | 1.14 |
|  | 10,000 | 4.27 µs | 3.68 µs | 1.16 |
|  | 100,000 | 120 µs | 139 µs | 0.86 |
|  | 1,000,000 | 1.47 ms | 3.23 ms | 0.46 |
|  | 53 (operands 1,000) | 8.9 ns | 13.7 ns | 0.65 |
|  | 1,000 (operands 100,000) | 260 ns | 253 ns | 1.03 |

With a new result (`add_round` and the others against `mpfr_init2`, the operation and `mpfr_clear`):

| Operation | Precision | `a` | MPFR | `a`/MPFR |
| --- | --- | ---: | ---: | ---: |
| `add_round` | 24 | 17.3 ns | 17 ns | 1.02 |
|  | 53 | 17.4 ns | 16.7 ns | 1.04 |
|  | 64 | 17.3 ns | 16.5 ns | 1.05 |
|  | 113 | 19.6 ns | 17.2 ns | 1.14 |
|  | 128 | 19.3 ns | 17.5 ns | 1.10 |
|  | 256 | 31.8 ns | 21.4 ns | 1.49 |
|  | 1,000 | 40 ns | 25.4 ns | 1.57 |
|  | 10,000 | 127 ns | 108 ns | 1.18 |
|  | 100,000 | 725 ns | 1.01 µs | 0.72 |
|  | 1,000,000 | 8.9 µs | 9.09 µs | 0.98 |
|  | 53 (operands 1,000) | 42.3 ns | 26.3 ns | 1.61 |
|  | 53 (operands 100,000) | 36.4 ns | 23.7 ns | 1.54 |
|  | 1,000 (operands 53) | 47.2 ns | 28.6 ns | 1.65 |
|  | 1,000 (operands 100,000) | 57.8 ns | 280 ns | 0.21 |
|  | 100,000 (operands 1,000,000) | 1.2 µs | 2.29 µs | 0.53 |
| `sub_round` | 24 | 17.2 ns | 16.4 ns | 1.05 |
|  | 53 | 17.2 ns | 16.2 ns | 1.06 |
|  | 64 | 17 ns | 16.6 ns | 1.02 |
|  | 113 | 19.4 ns | 17.3 ns | 1.12 |
|  | 128 | 19 ns | 18 ns | 1.06 |
|  | 256 | 31.4 ns | 21.8 ns | 1.44 |
|  | 1,000 | 39.4 ns | 26.4 ns | 1.49 |
|  | 10,000 | 127 ns | 100 ns | 1.27 |
|  | 100,000 | 644 ns | 892 ns | 0.72 |
|  | 1,000,000 | 12 µs | 10.3 µs | 1.17 |
|  | 53 (operands 1,000) | 42.3 ns | 27 ns | 1.57 |
|  | 53 (operands 100,000) | 37.9 ns | 299 ns | 0.13 |
|  | 1,000 (operands 53) | 47.7 ns | 28.3 ns | 1.69 |
|  | 1,000 (operands 100,000) | 58.2 ns | 155 ns | 0.38 |
|  | 100,000 (operands 1,000,000) | 1.16 µs | 2.23 µs | 0.52 |
| `mul_round` | 24 | 19.9 ns | 17.8 ns | 1.12 |
|  | 53 | 19.5 ns | 17.8 ns | 1.10 |
|  | 64 | 19.5 ns | 17.5 ns | 1.11 |
|  | 113 | 22.1 ns | 18.6 ns | 1.19 |
|  | 128 | 21.2 ns | 25.9 ns | 0.82 |
|  | 256 | 37.8 ns | 34.1 ns | 1.11 |
|  | 1,000 | 99 ns | 112 ns | 0.88 |
|  | 10,000 | 3.25 µs | 3.36 µs | 0.97 |
|  | 100,000 | 48.7 µs | 92.6 µs | 0.53 |
|  | 1,000,000 | 561 µs | 1.41 ms | 0.40 |
|  | 53 (operands 1,000) | 34.8 ns | 110 ns | 0.32 |
|  | 53 (operands 100,000) | 31.9 ns | 57.2 ns | 0.56 |
|  | 1,000 (operands 53) | 33.3 ns | 22.2 ns | 1.50 |
|  | 1,000 (operands 100,000) | 119 ns | 125 ns | 0.95 |
|  | 100,000 (operands 1,000,000) | 48.7 µs | 92.5 µs | 0.53 |
| `&x * &x` | 24 | 20.6 ns | 17.2 ns | 1.20 |
|  | 53 | 19.6 ns | 17.3 ns | 1.13 |
|  | 64 | 20.3 ns | 17.1 ns | 1.19 |
|  | 113 | 22.7 ns | 18.8 ns | 1.21 |
|  | 128 | 21.1 ns | 27.2 ns | 0.78 |
|  | 256 | 38.1 ns | 30.7 ns | 1.24 |
|  | 1,000 | 88.2 ns | 91.1 ns | 0.97 |
|  | 10,000 | 2.37 µs | 2.07 µs | 1.15 |
|  | 100,000 | 37.2 µs | 61 µs | 0.61 |
|  | 1,000,000 | 415 µs | 1.22 ms | 0.34 |
|  | 53 (operands 1,000) | 36.2 ns | 89.2 ns | 0.41 |
|  | 1,000 (operands 100,000) | 101 ns | 111 ns | 0.91 |
| `div_round` | 24 | 21.6 ns | 20.6 ns | 1.05 |
|  | 53 | 20.7 ns | 20 ns | 1.03 |
|  | 64 | 22 ns | 20.7 ns | 1.06 |
|  | 113 | 24 ns | 19.5 ns | 1.23 |
|  | 128 | 25.2 ns | 39.7 ns | 0.63 |
|  | 256 | 83.3 ns | 77.1 ns | 1.08 |
|  | 1,000 | 260 ns | 230 ns | 1.13 |
|  | 10,000 | 7.01 µs | 5.51 µs | 1.27 |
|  | 100,000 | 127 µs | 195 µs | 0.65 |
|  | 1,000,000 | 1.4 ms | 3.99 ms | 0.35 |
|  | 53 (operands 1,000) | 46.6 ns | 30.3 ns | 1.54 |
|  | 53 (operands 100,000) | 40.9 ns | 25.3 ns | 1.62 |
|  | 1,000 (operands 53) | 93 ns | 56.8 ns | 1.64 |
|  | 1,000 (operands 100,000) | 260 ns | 225 ns | 1.16 |
|  | 100,000 (operands 1,000,000) | 127 µs | 194 µs | 0.65 |
| `mul_add_round` | 24 | 21.9 ns | 22.4 ns | 0.98 |
|  | 53 | 21.1 ns | 22.9 ns | 0.92 |
|  | 64 | 21.5 ns | 33.1 ns | 0.65 |
|  | 113 | 63.2 ns | 34.5 ns | 1.83 |
|  | 128 | 62.8 ns | 35.2 ns | 1.78 |
|  | 256 | 67.9 ns | 40.7 ns | 1.67 |
|  | 1,000 | 142 ns | 122 ns | 1.16 |
|  | 10,000 | 3.43 µs | 3.69 µs | 0.93 |
|  | 100,000 | 50.3 µs | 106 µs | 0.47 |
|  | 1,000,000 | 574 µs | 1.42 ms | 0.40 |
|  | 53 (operands 1,000) | 124 ns | 120 ns | 1.03 |
|  | 53 (operands 100,000) | 48.7 µs | 107 µs | 0.46 |
|  | 1,000 (operands 53) | 66.6 ns | 35.1 ns | 1.90 |
|  | 1,000 (operands 100,000) | 48.7 µs | 107 µs | 0.46 |
|  | 100,000 (operands 1,000,000) | 562 µs | 1.41 ms | 0.40 |
| `sqrt_round` | 24 | 26.7 ns | 20.5 ns | 1.30 |
|  | 53 | 26.1 ns | 19.1 ns | 1.37 |
|  | 64 | 39.2 ns | 17.8 ns | 2.20 |
|  | 113 | 43.2 ns | 21 ns | 2.06 |
|  | 128 | 95 ns | 82.3 ns | 1.15 |
|  | 256 | 130 ns | 117 ns | 1.11 |
|  | 1,000 | 311 ns | 278 ns | 1.12 |
|  | 10,000 | 4.3 µs | 3.7 µs | 1.16 |
|  | 100,000 | 120 µs | 139 µs | 0.86 |
|  | 1,000,000 | 1.47 ms | 3.22 ms | 0.46 |
|  | 53 (operands 1,000) | 21.8 ns | 30.4 ns | 0.72 |
|  | 1,000 (operands 100,000) | 267 ns | 265 ns | 1.01 |

### Roots and powers

Into an existing number, against `mpfr_rec_sqrt`, `mpfr_cbrt`, `mpfr_rootn_ui` and `mpfr_pow_si` into an existing variable:

| Operation | Precision | `a` | MPFR | `a`/MPFR |
| --- | --- | ---: | ---: | ---: |
| `rec_sqrt_round` | 24 | 57.4 ns | 55.1 ns | 1.04 |
|  | 53 | 56.4 ns | 55 ns | 1.03 |
|  | 64 | 139 ns | 78 ns | 1.78 |
|  | 113 | 159 ns | 95.8 ns | 1.66 |
|  | 128 | 160 ns | 118 ns | 1.36 |
|  | 256 | 253 ns | 174 ns | 1.45 |
|  | 1,000 | 699 ns | 427 ns | 1.64 |
|  | 10,000 | 17.4 µs | 9.46 µs | 1.84 |
|  | 100,000 | 309 µs | 268 µs | 1.15 |
| `cbrt_round` | 24 | 302 ns | 119 ns | 2.54 |
|  | 53 | 299 ns | 187 ns | 1.60 |
|  | 64 | 298 ns | 211 ns | 1.41 |
|  | 113 | 354 ns | 243 ns | 1.46 |
|  | 128 | 353 ns | 266 ns | 1.33 |
|  | 256 | 402 ns | 350 ns | 1.15 |
|  | 1,000 | 823 ns | 688 ns | 1.20 |
|  | 10,000 | 9.51 µs | 8.68 µs | 1.10 |
|  | 100,000 | 221 µs | 279 µs | 0.79 |
| `nth_root_round(5)` | 24 | 383 ns | 177 ns | 2.16 |
|  | 53 | 382 ns | 238 ns | 1.61 |
|  | 64 | 380 ns | 265 ns | 1.43 |
|  | 113 | 449 ns | 324 ns | 1.39 |
|  | 128 | 447 ns | 346 ns | 1.29 |
|  | 256 | 486 ns | 461 ns | 1.05 |
|  | 1,000 | 1.12 µs | 982 ns | 1.14 |
|  | 10,000 | 17.2 µs | 16.9 µs | 1.02 |
|  | 100,000 | 330 µs | 528 µs | 0.62 |
| `nth_root_round(100)` | 24 | 1.3 µs | 1.02 µs | 1.27 |
|  | 53 | 1.35 µs | 1.79 µs | 0.75 |
|  | 64 | 2.33 µs | 2.16 µs | 1.08 |
|  | 113 | 2.35 µs | 3.94 µs | 0.60 |
|  | 128 | 2.47 µs | 4.81 µs | 0.51 |
|  | 256 | 3.44 µs | 7.6 µs | 0.45 |
|  | 1,000 | 6.74 µs | 43.6 µs | 0.15 |
| `powi_round(7)` | 24 | 55.6 ns | 61.1 ns | 0.91 |
|  | 53 | 55.6 ns | 58.3 ns | 0.95 |
|  | 64 | 54.8 ns | 68.5 ns | 0.80 |
|  | 113 | 264 ns | 75.9 ns | 3.48 |
|  | 128 | 264 ns | 86.2 ns | 3.06 |
|  | 256 | 292 ns | 112 ns | 2.61 |
|  | 1,000 | 578 ns | 409 ns | 1.41 |
|  | 10,000 | 12.1 µs | 11.2 µs | 1.08 |
|  | 100,000 | 178 µs | 371 µs | 0.48 |
| `powi_round(-3)` | 24 | 56.1 ns | 91 ns | 0.62 |
|  | 53 | 56.1 ns | 91.8 ns | 0.61 |
|  | 64 | 55.6 ns | 99.7 ns | 0.56 |
|  | 113 | 263 ns | 106 ns | 2.48 |
|  | 128 | 262 ns | 113 ns | 2.32 |
|  | 256 | 314 ns | 160 ns | 1.96 |
|  | 1,000 | 638 ns | 471 ns | 1.35 |
|  | 10,000 | 13 µs | 13.4 µs | 0.97 |
|  | 100,000 | 212 µs | 416 µs | 0.51 |
| `powi_round(1_000_000_007)` | 24 | 143 ns | 372 ns | 0.38 |
|  | 53 | 142 ns | 385 ns | 0.37 |
|  | 64 | 143 ns | 418 ns | 0.34 |
|  | 113 | 873 ns | 556 ns | 1.57 |
|  | 128 | 931 ns | 873 ns | 1.07 |
|  | 256 | 1.21 µs | 1.19 µs | 1.01 |
|  | 1,000 | 3.79 µs | 4.29 µs | 0.88 |

### Conversions

Against `mpfr_get_d`, `mpfr_set_d`, `mpfr_floor`, `mpfr_rint` and `mpfr_get_si`; `from_f64_round`, `floor` and `rint_round` make a new number and MPFR writes into an existing one; `floor` of a number with half its bits after the point, `rint_round` to half the precision:

| Operation | Precision | `a` | MPFR | `a`/MPFR |
| --- | --- | ---: | ---: | ---: |
| `to_f64_round` | 24 | 1.7 ns | 5.4 ns | 0.31 |
|  | 53 | 1.6 ns | 5.4 ns | 0.30 |
|  | 64 | 2.4 ns | 4.5 ns | 0.53 |
|  | 113 | 2.4 ns | 4.6 ns | 0.52 |
|  | 128 | 2.3 ns | 4.6 ns | 0.50 |
|  | 256 | 2.3 ns | 4.5 ns | 0.51 |
|  | 1,000 | 2.3 ns | 4.6 ns | 0.50 |
|  | 10,000 | 2.3 ns | 4.5 ns | 0.51 |
|  | 100,000 | 2.2 ns | 4.8 ns | 0.46 |
|  | 1,000,000 | 2.2 ns | 4.6 ns | 0.48 |
| `from_f64_round` | 24 | 19.8 ns | 8.3 ns | 2.39 |
|  | 53 | 10.7 ns | 5.6 ns | 1.91 |
|  | 64 | 12.4 ns | 7.5 ns | 1.65 |
|  | 113 | 10.5 ns | 7.7 ns | 1.36 |
|  | 256 | 9.9 ns | 8.2 ns | 1.21 |
|  | 1,000 | 10.4 ns | 7.5 ns | 1.39 |
|  | 10,000 | 24.4 ns | 17.8 ns | 1.37 |
|  | 100,000 | 191 ns | 131 ns | 1.46 |
|  | 1,000,000 | 1.14 µs | 586 ns | 1.95 |
| `floor` | 24 | 14.9 ns | 8.6 ns | 1.73 |
|  | 53 | 14.7 ns | 8.5 ns | 1.73 |
|  | 64 | 14.8 ns | 8.5 ns | 1.74 |
|  | 113 | 15.3 ns | 8.6 ns | 1.78 |
|  | 128 | 15.1 ns | 8.6 ns | 1.76 |
|  | 256 | 16.2 ns | 9.1 ns | 1.78 |
|  | 1,000 | 16.7 ns | 8.7 ns | 1.92 |
|  | 10,000 | 34.3 ns | 16.6 ns | 2.07 |
|  | 100,000 | 186 ns | 110 ns | 1.69 |
|  | 1,000,000 | 1.78 µs | 829 ns | 2.15 |
| `rint_round(p/2)` | 24 | 24.1 ns | 9.1 ns | 2.65 |
|  | 53 | 23.9 ns | 8.7 ns | 2.75 |
|  | 64 | 23.9 ns | 8.8 ns | 2.72 |
|  | 113 | 24.6 ns | 8.8 ns | 2.80 |
|  | 128 | 25 ns | 8.7 ns | 2.87 |
|  | 256 | 25 ns | 9 ns | 2.78 |
|  | 1,000 | 26 ns | 8.5 ns | 3.06 |
|  | 10,000 | 58.2 ns | 13.7 ns | 4.25 |
|  | 100,000 | 188 ns | 72.1 ns | 2.61 |
|  | 1,000,000 | 622 ns | 522 ns | 1.19 |
| `to_i64_round` | 53 | 10.7 ns | 23.5 ns | 0.46 |
|  | 113 | 10.4 ns | 23.4 ns | 0.44 |
|  | 1,000 | 10.2 ns | 23.5 ns | 0.43 |

### Text

Writing against `mpfr_get_str`: `to_digits(10, None)` gives `1 + ceil(p·log10 2)` digits, as `mpfr_get_str` with `n = 0`; the exponents are those of the other tables, or up to `±2^60` where marked:

| Operation | Precision | `a` | MPFR | `a`/MPFR |
| --- | --- | ---: | ---: | ---: |
| `to_digits(10, None)` | 24 | 111 ns | 205 ns | 0.54 |
|  | 53 | 123 ns | 217 ns | 0.57 |
|  | 64 | 145 ns | 223 ns | 0.65 |
|  | 113 | 182 ns | 258 ns | 0.71 |
|  | 128 | 170 ns | 269 ns | 0.63 |
|  | 256 | 198 ns | 357 ns | 0.56 |
|  | 1,000 | 620 ns | 991 ns | 0.63 |
|  | 10,000 | 13.5 µs | 16.7 µs | 0.81 |
|  | 100,000 | 300 µs | 433 µs | 0.69 |
|  | 1,000,000 | 4.79 ms | 9.61 ms | 0.50 |
| `to_digits(10, None)`, exponents to `±2^60` | 53 | 1.7 µs | 1.34 µs | 1.27 |
|  | 113 | 1.93 µs | 1.61 µs | 1.20 |
|  | 1,000 | 6.85 µs | 6.71 µs | 1.02 |
|  | 10,000 | 151 µs | 153 µs | 0.99 |
| `to_digits(10, Some(20))` | 1,000 | 192 ns | 186 ns | 1.03 |
|  | 10,000 | 187 ns | 180 ns | 1.04 |
|  | 100,000 | 165 ns | 175 ns | 0.94 |
|  | 1,000,000 | 164 ns | 169 ns | 0.97 |

`to_string()`, the shortest digits that read back, against rug's `to_string()`, the digits of `mpfr_get_str`:

| Operation | Precision | `a` | rug | `a`/rug |
| --- | --- | ---: | ---: | ---: |
| `to_string()` | 24 | 279 ns | 299 ns | 0.93 |
|  | 53 | 309 ns | 314 ns | 0.99 |
|  | 64 | 332 ns | 329 ns | 1.01 |
|  | 113 | 370 ns | 360 ns | 1.03 |
|  | 128 | 374 ns | 381 ns | 0.98 |
|  | 256 | 406 ns | 451 ns | 0.90 |
|  | 1,000 | 890 ns | 1.1 µs | 0.81 |
|  | 10,000 | 13.2 µs | 16.1 µs | 0.82 |
|  | 100,000 | 311 µs | 433 µs | 0.72 |
|  | 1,000,000 | 4.9 ms | 9.83 ms | 0.50 |

Reading against `mpfr_strtofr`, texts of the digits given: at their worth of bits, `ceil(n·log2 10)`, at 53 bits, and at their worth with decimal exponents up to `±10^17`:

| Operation | Digits | `a` | MPFR | `a`/MPFR |
| --- | --- | ---: | ---: | ---: |
| `from_str_radix`, `ceil(n·log2 10)` bits | 20 | 144 ns | 120 ns | 1.20 |
|  | 50 | 271 ns | 184 ns | 1.47 |
|  | 100 | 406 ns | 295 ns | 1.38 |
|  | 1,000 | 3.19 µs | 3.16 µs | 1.01 |
|  | 10,000 | 73.6 µs | 90 µs | 0.82 |
|  | 100,000 | 1.08 ms | 2.13 ms | 0.51 |
|  | 1,000,000 | 14.4 ms | 34.1 ms | 0.42 |
| `from_str_radix`, 53 bits | 20 | 130 ns | 112 ns | 1.16 |
|  | 50 | 150 ns | 132 ns | 1.13 |
|  | 100 | 176 ns | 163 ns | 1.08 |
|  | 1,000 | 410 ns | 619 ns | 0.66 |
|  | 10,000 | 2.76 µs | 5.02 µs | 0.55 |
|  | 100,000 | 25.4 µs | 49.9 µs | 0.51 |
|  | 1,000,000 | 258 µs | 495 µs | 0.52 |
| `from_str_radix`, exponents to `±10^17` | 20 | 1.42 µs | 1.95 µs | 0.73 |
|  | 100 | 2.34 µs | 3.39 µs | 0.69 |
|  | 1,000 | 25.8 µs | 39.7 µs | 0.65 |

### Exponentials, logarithms and powers

Against `mpfr_exp`, `mpfr_exp2`, `mpfr_exp10`, `mpfr_expm1`, `mpfr_log`, `mpfr_log2`, `mpfr_log10`, `mpfr_log1p` and `mpfr_pow`, and Arb's `arb_exp` and the others at the precision and the guard bits of Ziv's first attempt. The rows of 10⁴ bits of the logarithms, `pow` and the inverse hyperbolic functions are from a later run of the same harness, after the logarithm's AGM threshold moved from 16,000 to 5,000 bits:

| Operation | Precision | `a` | MPFR | `a`/MPFR | Arb |
| --- | --- | ---: | ---: | ---: | ---: |
| `exp` | 24 | 330 ns | 362 ns | 0.91 | 69 ns |
|  | 53 | 329 ns | 517 ns | 0.64 | 88 ns |
|  | 64 | 335 ns | 558 ns | 0.60 | 90 ns |
|  | 113 | 443 ns | 765 ns | 0.58 | 120 ns |
|  | 128 | 441 ns | 796 ns | 0.55 | 124 ns |
|  | 256 | 846 ns | 1.22 µs | 0.69 | 228 ns |
|  | 1,000 | 3.42 µs | 5.27 µs | 0.65 | 1.89 µs |
|  | 10,000 | 248 µs | 288 µs | 0.86 | 65.8 µs |
|  | 100,000 | 5.62 ms | 8.57 ms | 0.66 | 3.56 ms |
|  | 1,000,000 | 96.8 ms | 189 ms | 0.51 | 76.4 ms |
| `exp2` | 24 | 407 ns | 492 ns | 0.83 | 127 ns |
|  | 53 | 407 ns | 668 ns | 0.61 | 160 ns |
|  | 64 | 410 ns | 706 ns | 0.58 | 158 ns |
|  | 113 | 542 ns | 886 ns | 0.61 | 206 ns |
|  | 128 | 547 ns | 944 ns | 0.58 | 208 ns |
|  | 256 | 902 ns | 1.36 µs | 0.66 | 313 ns |
|  | 1,000 | 3.48 µs | 5.44 µs | 0.64 | 2.04 µs |
|  | 10,000 | 252 µs | 291 µs | 0.87 | 68.5 µs |
|  | 100,000 | 5.67 ms | 8.59 ms | 0.66 | 3.63 ms |
| `exp10` | 24 | 449 ns | 854 ns | 0.53 | 147 ns |
|  | 53 | 453 ns | 1.04 µs | 0.44 | 180 ns |
|  | 64 | 452 ns | 1.2 µs | 0.38 | 183 ns |
|  | 113 | 572 ns | 1.85 µs | 0.31 | 227 ns |
|  | 128 | 569 ns | 2.16 µs | 0.26 | 229 ns |
|  | 256 | 988 ns | 3.41 µs | 0.29 | 339 ns |
|  | 1,000 | 3.68 µs | 12 µs | 0.31 | 2.12 µs |
|  | 10,000 | 250 µs | 448 µs | 0.56 | 67.7 µs |
|  | 100,000 | 5.67 ms | 15.5 ms | 0.37 | 3.66 ms |
| `exp_m1` | 24 | 454 ns | 476 ns | 0.95 | 82 ns |
|  | 53 | 452 ns | 617 ns | 0.73 | 108 ns |
|  | 64 | 457 ns | 659 ns | 0.69 | 110 ns |
|  | 113 | 586 ns | 851 ns | 0.69 | 145 ns |
|  | 128 | 811 ns | 901 ns | 0.90 | 148 ns |
|  | 256 | 1.11 µs | 1.35 µs | 0.82 | 253 ns |
|  | 1,000 | 3.6 µs | 5.4 µs | 0.67 | 1.92 µs |
|  | 10,000 | 249 µs | 286 µs | 0.87 | 65.3 µs |
|  | 100,000 | 5.62 ms | 8.46 ms | 0.66 | 3.54 ms |
| `ln` | 24 | 400 ns | 381 ns | 1.05 | 62 ns |
|  | 53 | 449 ns | 436 ns | 1.03 | 104 ns |
|  | 64 | 458 ns | 475 ns | 0.96 | 109 ns |
|  | 113 | 561 ns | 1.18 µs | 0.48 | 143 ns |
|  | 128 | 599 ns | 1.2 µs | 0.50 | 145 ns |
|  | 256 | 1.77 µs | 2.18 µs | 0.81 | 218 ns |
|  | 1,000 | 5.45 µs | 6.97 µs | 0.78 | 1.58 µs |
|  | 10,000 | 199 µs | 165 µs | 1.21 | 80.3 µs |
|  | 100,000 | 5.23 ms | 6.64 ms | 0.79 | 4.03 ms |
|  | 1,000,000 | 77 ms | 168 ms | 0.46 | 87.1 ms |
| `log2` | 24 | 542 ns | 467 ns | 1.16 | 104 ns |
|  | 53 | 606 ns | 593 ns | 1.02 | 157 ns |
|  | 64 | 614 ns | 594 ns | 1.03 | 162 ns |
|  | 113 | 768 ns | 1.34 µs | 0.57 | 212 ns |
|  | 128 | 811 ns | 1.32 µs | 0.62 | 216 ns |
|  | 256 | 2 µs | 2.33 µs | 0.86 | 316 ns |
|  | 1,000 | 6.03 µs | 7.32 µs | 0.82 | 1.8 µs |
|  | 10,000 | 206 µs | 169 µs | 1.22 | 85.8 µs |
|  | 100,000 | 5.35 ms | 6.83 ms | 0.78 | 4.23 ms |
| `log10` | 24 | 626 ns | 966 ns | 0.65 | 114 ns |
|  | 53 | 697 ns | 958 ns | 0.73 | 167 ns |
|  | 64 | 698 ns | 1.1 µs | 0.64 | 169 ns |
|  | 113 | 848 ns | 2.24 µs | 0.38 | 218 ns |
|  | 128 | 883 ns | 2.51 µs | 0.35 | 222 ns |
|  | 256 | 2.1 µs | 4.49 µs | 0.47 | 326 ns |
|  | 1,000 | 6.15 µs | 13.9 µs | 0.44 | 1.83 µs |
|  | 10,000 | 209 µs | 335 µs | 0.62 | 85.7 µs |
|  | 100,000 | 5.37 ms | 14.2 ms | 0.38 | 4.24 ms |
| `ln_1p` | 24 | 565 ns | 375 ns | 1.51 | 86 ns |
|  | 53 | 562 ns | 413 ns | 1.36 | 124 ns |
|  | 64 | 561 ns | 440 ns | 1.27 | 129 ns |
|  | 113 | 720 ns | 1.12 µs | 0.64 | 162 ns |
|  | 128 | 721 ns | 1.14 µs | 0.63 | 163 ns |
|  | 256 | 2.07 µs | 2.08 µs | 0.99 | 233 ns |
|  | 1,000 | 6.22 µs | 7 µs | 0.89 | 1.58 µs |
|  | 10,000 | 199 µs | 164 µs | 1.21 | 79.9 µs |
|  | 100,000 | 5.23 ms | 6.9 ms | 0.76 | 4.04 ms |
| `pow` | 24 | 817 ns | 941 ns | 0.87 | 179 ns |
|  | 53 | 906 ns | 1.11 µs | 0.82 | 271 ns |
|  | 64 | 934 ns | 1.29 µs | 0.72 | 282 ns |
|  | 113 | 1.17 µs | 2.11 µs | 0.56 | 342 ns |
|  | 128 | 1.26 µs | 2.41 µs | 0.52 | 348 ns |
|  | 256 | 2.77 µs | 3.76 µs | 0.74 | 536 ns |
|  | 1,000 | 9.39 µs | 13.1 µs | 0.72 | 3.6 µs |
|  | 10,000 | 452 µs | 455 µs | 0.99 | 148 µs |
|  | 100,000 | 10.9 ms | 15.5 ms | 0.70 | 7.71 ms |

### Hyperbolic functions

Against `mpfr_sinh` and the others, and Arb's:

| Operation | Precision | `a` | MPFR | `a`/MPFR | Arb |
| --- | --- | ---: | ---: | ---: | ---: |
| `sinh` | 24 | 551 ns | 550 ns | 1.00 | 152 ns |
|  | 53 | 564 ns | 615 ns | 0.92 | 193 ns |
|  | 64 | 691 ns | 749 ns | 0.92 | 195 ns |
|  | 113 | 799 ns | 872 ns | 0.92 | 257 ns |
|  | 128 | 1 µs | 934 ns | 1.07 | 263 ns |
|  | 256 | 1.32 µs | 1.4 µs | 0.95 | 390 ns |
|  | 1,000 | 4 µs | 5.64 µs | 0.71 | 2.2 µs |
|  | 10,000 | 257 µs | 292 µs | 0.88 | 70.7 µs |
|  | 100,000 | 5.75 ms | 8.68 ms | 0.66 | 3.73 ms |
| `cosh` | 24 | 506 ns | 556 ns | 0.91 | 137 ns |
|  | 53 | 517 ns | 655 ns | 0.79 | 170 ns |
|  | 64 | 519 ns | 720 ns | 0.72 | 175 ns |
|  | 113 | 660 ns | 905 ns | 0.73 | 231 ns |
|  | 128 | 882 ns | 918 ns | 0.96 | 239 ns |
|  | 256 | 1.17 µs | 1.4 µs | 0.84 | 364 ns |
|  | 1,000 | 3.88 µs | 5.61 µs | 0.69 | 2.15 µs |
|  | 10,000 | 255 µs | 291 µs | 0.88 | 70.8 µs |
|  | 100,000 | 5.75 ms | 8.67 ms | 0.66 | 3.71 ms |
| `tanh` | 24 | 537 ns | 553 ns | 0.97 | 164 ns |
|  | 53 | 546 ns | 705 ns | 0.77 | 203 ns |
|  | 64 | 674 ns | 706 ns | 0.95 | 203 ns |
|  | 113 | 760 ns | 889 ns | 0.85 | 262 ns |
|  | 128 | 977 ns | 964 ns | 1.01 | 267 ns |
|  | 256 | 1.3 µs | 1.44 µs | 0.90 | 412 ns |
|  | 1,000 | 4.02 µs | 5.66 µs | 0.71 | 2.28 µs |
|  | 10,000 | 257 µs | 293 µs | 0.88 | 70.6 µs |
|  | 100,000 | 5.75 ms | 8.7 ms | 0.66 | 3.77 ms |
| `asinh` | 24 | 739 ns | 933 ns | 0.79 | 194 ns |
|  | 53 | 793 ns | 755 ns | 1.05 | 280 ns |
|  | 64 | 785 ns | 1.06 µs | 0.74 | 282 ns |
|  | 113 | 1.18 µs | 1.41 µs | 0.84 | 350 ns |
|  | 128 | 1.18 µs | 1.55 µs | 0.76 | 353 ns |
|  | 256 | 2.57 µs | 2.68 µs | 0.96 | 472 ns |
|  | 1,000 | 7.13 µs | 7.44 µs | 0.96 | 2.2 µs |
|  | 10,000 | 213 µs | 171 µs | 1.25 | 85.4 µs |
|  | 100,000 | 5.52 ms | 7.2 ms | 0.77 | 4.26 ms |
| `acosh` | 24 | 648 ns | 606 ns | 1.07 | 151 ns |
|  | 53 | 704 ns | 679 ns | 1.04 | 238 ns |
|  | 64 | 704 ns | 752 ns | 0.94 | 244 ns |
|  | 113 | 992 ns | 1.52 µs | 0.65 | 312 ns |
|  | 128 | 1.09 µs | 1.55 µs | 0.71 | 314 ns |
|  | 256 | 2.31 µs | 2.56 µs | 0.90 | 434 ns |
|  | 1,000 | 6.44 µs | 8.24 µs | 0.78 | 2.15 µs |
|  | 10,000 | 208 µs | 171 µs | 1.22 | 85.8 µs |
|  | 100,000 | 5.39 ms | 7.2 ms | 0.75 | 4.25 ms |
| `atanh` | 24 | 662 ns | 1.13 µs | 0.59 | 160 ns |
|  | 53 | 672 ns | 1.04 µs | 0.65 | 200 ns |
|  | 64 | 673 ns | 1.15 µs | 0.59 | 209 ns |
|  | 113 | 896 ns | 1.6 µs | 0.56 | 250 ns |
|  | 128 | 920 ns | 1.73 µs | 0.53 | 262 ns |
|  | 256 | 2.27 µs | 2.67 µs | 0.85 | 375 ns |
|  | 1,000 | 6.51 µs | 7.65 µs | 0.85 | 2.01 µs |
|  | 10,000 | 208 µs | 169 µs | 1.23 | 85.9 µs |
|  | 100,000 | 5.35 ms | 7.67 ms | 0.70 | 4.27 ms |

### Circular functions

Against `mpfr_sin` and the others, and Arb's; arguments in `[1/4, 4)` for the sine, the cosine, the tangent and their reciprocals, `[1/16, 1)` for `asin` and `acos`, `[1/8, 4)` for `atan` and `atan2`:

| Operation | Precision | `a` | MPFR | `a`/MPFR | Arb |
| --- | --- | ---: | ---: | ---: | ---: |
| `sin` | 24 | 217 ns | 460 ns | 0.47 | 72 ns |
|  | 53 | 216 ns | 579 ns | 0.37 | 88 ns |
|  | 64 | 213 ns | 576 ns | 0.37 | 102 ns |
|  | 113 | 289 ns | 754 ns | 0.38 | 115 ns |
|  | 128 | 286 ns | 771 ns | 0.37 | 136 ns |
|  | 256 | 534 ns | 1.2 µs | 0.44 | 265 ns |
|  | 1,000 | 3.87 µs | 4.6 µs | 0.84 | 1.92 µs |
|  | 10,000 | 188 µs | 286 µs | 0.66 | 86.4 µs |
|  | 100,000 | 7.11 ms | 14 ms | 0.51 | 4.98 ms |
|  | 1,000,000 | 283 ms | 294 ms | 0.96 | 141 ms |
| `cos` | 24 | 216 ns | 326 ns | 0.66 | 69 ns |
|  | 53 | 214 ns | 379 ns | 0.56 | 85 ns |
|  | 64 | 217 ns | 397 ns | 0.55 | 99 ns |
|  | 113 | 288 ns | 506 ns | 0.57 | 114 ns |
|  | 128 | 288 ns | 541 ns | 0.53 | 135 ns |
|  | 256 | 533 ns | 895 ns | 0.60 | 265 ns |
|  | 1,000 | 3.74 µs | 4.05 µs | 0.92 | 1.93 µs |
|  | 10,000 | 184 µs | 275 µs | 0.67 | 86.7 µs |
|  | 100,000 | 7.14 ms | 14 ms | 0.51 | 5 ms |
| `tan` | 24 | 363 ns | 528 ns | 0.69 | 126 ns |
|  | 53 | 361 ns | 646 ns | 0.56 | 148 ns |
|  | 64 | 362 ns | 669 ns | 0.54 | 157 ns |
|  | 113 | 459 ns | 931 ns | 0.49 | 182 ns |
|  | 128 | 458 ns | 978 ns | 0.47 | 225 ns |
|  | 256 | 733 ns | 1.4 µs | 0.53 | 382 ns |
|  | 1,000 | 4.46 µs | 5.4 µs | 0.83 | 2.3 µs |
|  | 10,000 | 197 µs | 287 µs | 0.69 | 93.7 µs |
|  | 100,000 | 7.36 ms | 14.2 ms | 0.52 | 5.18 ms |
| `sin_cos` | 24 | 286 ns | 472 ns | 0.61 | 84 ns |
|  | 53 | 287 ns | 579 ns | 0.50 | 100 ns |
|  | 64 | 285 ns | 592 ns | 0.48 | 112 ns |
|  | 113 | 368 ns | 767 ns | 0.48 | 128 ns |
|  | 128 | 362 ns | 806 ns | 0.45 | 148 ns |
|  | 256 | 626 ns | 1.25 µs | 0.50 | 282 ns |
|  | 1,000 | 4.17 µs | 4.66 µs | 0.89 | 2.06 µs |
|  | 10,000 | 190 µs | 285 µs | 0.67 | 86.2 µs |
|  | 100,000 | 7.21 ms | 14 ms | 0.52 | 4.99 ms |
| `sec` | 53 | 320 ns | 468 ns | 0.68 | 142 ns |
|  | 1,000 | 4.07 µs | 4.34 µs | 0.94 | 2.14 µs |
| `csc` | 53 | 321 ns | 651 ns | 0.49 | 142 ns |
|  | 1,000 | 4.21 µs | 5.35 µs | 0.79 | 2.15 µs |
| `cot` | 53 | 367 ns | 726 ns | 0.51 | 150 ns |
|  | 1,000 | 4.41 µs | 5.74 µs | 0.77 | 2.29 µs |
| `atan` | 24 | 535 ns | 714 ns | 0.75 | 58 ns |
|  | 53 | 519 ns | 1.01 µs | 0.51 | 77 ns |
|  | 64 | 520 ns | 1.03 µs | 0.51 | 102 ns |
|  | 113 | 955 ns | 1.6 µs | 0.60 | 112 ns |
|  | 128 | 956 ns | 1.9 µs | 0.50 | 141 ns |
|  | 256 | 1.71 µs | 5.08 µs | 0.34 | 238 ns |
|  | 1,000 | 8.31 µs | 15.7 µs | 0.53 | 1.93 µs |
|  | 10,000 | 282 µs | 367 µs | 0.77 | 109 µs |
|  | 100,000 | 11.2 ms | 14.1 ms | 0.79 | 5.76 ms |
|  | 1,000,000 | 447 ms | 338 ms | 1.32 | 156 ms |
| `asin` | 24 | 723 ns | 804 ns | 0.90 | 227 ns |
|  | 53 | 754 ns | 1.17 µs | 0.64 | 252 ns |
|  | 64 | 756 ns | 1.16 µs | 0.65 | 312 ns |
|  | 113 | 1.21 µs | 2 µs | 0.60 | 339 ns |
|  | 128 | 1.21 µs | 2.12 µs | 0.57 | 393 ns |
|  | 256 | 2.03 µs | 5.24 µs | 0.39 | 568 ns |
|  | 1,000 | 8.88 µs | 18.6 µs | 0.48 | 2.61 µs |
|  | 10,000 | 292 µs | 376 µs | 0.78 | 127 µs |
|  | 100,000 | 11.4 ms | 14.1 ms | 0.81 | 6.24 ms |
| `acos` | 24 | 807 ns | 858 ns | 0.94 | 250 ns |
|  | 53 | 815 ns | 1.19 µs | 0.69 | 275 ns |
|  | 64 | 816 ns | 1.17 µs | 0.70 | 333 ns |
|  | 113 | 1.28 µs | 2.13 µs | 0.60 | 372 ns |
|  | 128 | 1.27 µs | 2.14 µs | 0.59 | 427 ns |
|  | 256 | 2.09 µs | 5.32 µs | 0.39 | 613 ns |
|  | 1,000 | 9.02 µs | 18.7 µs | 0.48 | 2.67 µs |
|  | 10,000 | 293 µs | 379 µs | 0.77 | 127 µs |
|  | 100,000 | 11.5 ms | 14.1 ms | 0.82 | 6.26 ms |
| `atan2` | 24 | 645 ns | 844 ns | 0.76 | 120 ns |
|  | 53 | 627 ns | 1.14 µs | 0.55 | 141 ns |
|  | 64 | 624 ns | 1.16 µs | 0.54 | 162 ns |
|  | 113 | 1.08 µs | 2.01 µs | 0.54 | 190 ns |
|  | 128 | 1.08 µs | 2.03 µs | 0.53 | 218 ns |
|  | 256 | 1.85 µs | 5.13 µs | 0.36 | 348 ns |
|  | 1,000 | 8.54 µs | 18.3 µs | 0.47 | 2.2 µs |
|  | 10,000 | 287 µs | 375 µs | 0.77 | 119 µs |
|  | 100,000 | 11.4 ms | 14.3 ms | 0.80 | 6.08 ms |

Arguments `m·2^k` of a large exponent, the reduction taking π to `k` bits and more. For `k = 10^6` Arb returns `[−1, 1]` at once, which is not a result:

| Operation | Precision | `a` | MPFR | `a`/MPFR | Arb |
| --- | --- | ---: | ---: | ---: | ---: |
| `sin`, `m·2^100` | 53 | 282 ns | 814 ns | 0.35 | 108 ns |
|  | 256 | 614 ns | 1.49 µs | 0.41 | 289 ns |
| `sin`, `m·2^10000` | 53 | 7.24 µs | 9.11 µs | 0.79 | 9.33 µs |
|  | 256 | 7.72 µs | 10.4 µs | 0.74 | 9.53 µs |
| `sin`, `m·2^1000000` | 53 | 1.39 ms | 3.93 ms | 0.35 | 9 ns |
|  | 256 | 1.4 ms | 3.99 ms | 0.35 | 9 ns |
| `cos`, `m·2^100` | 53 | 282 ns | 657 ns | 0.43 | 109 ns |
|  | 256 | 611 ns | 1.17 µs | 0.52 | 289 ns |
| `cos`, `m·2^10000` | 53 | 7.23 µs | 9.38 µs | 0.77 | 9.35 µs |
|  | 256 | 7.72 µs | 10.2 µs | 0.76 | 9.62 µs |
| `cos`, `m·2^1000000` | 53 | 1.4 ms | 4.12 ms | 0.34 | 9 ns |
|  | 256 | 1.4 ms | 3.97 ms | 0.35 | 9 ns |
| `tan`, `m·2^100` | 53 | 431 ns | 882 ns | 0.49 | 168 ns |
|  | 256 | 815 ns | 1.65 µs | 0.49 | 410 ns |
| `tan`, `m·2^10000` | 53 | 7.39 µs | 9.41 µs | 0.79 | 9.39 µs |
|  | 256 | 7.98 µs | 10.7 µs | 0.75 | 10.2 µs |
| `tan`, `m·2^1000000` | 53 | 1.39 ms | 4 ms | 0.35 | 13 ns |
|  | 256 | 1.39 ms | 4 ms | 0.35 | 12 ns |

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

Against `mpc_add`, `mpc_mul`, `mpc_sqr`, `mpc_div`, `mpc_sqrt`, `mpc_abs` and `mpc_norm`; a precision is that of both parts of the operands and of the result. `BigComplex` has no forms into an existing number yet, so `a` makes a new one where MPC writes into one: for `add_round` that is most of the difference (two new `BigFloat`s, about 17 ns each, against two `mpfr_add` into a variable, about 5 ns each):

| Operation | Precision | `a` | MPC | `a`/MPC |
| --- | --- | ---: | ---: | ---: |
| `add_round` | 24 | 36.5 ns | 13.5 ns | 2.70 |
|  | 53 | 35.7 ns | 13.5 ns | 2.64 |
|  | 64 | 35.7 ns | 15.2 ns | 2.35 |
|  | 113 | 40.7 ns | 15 ns | 2.71 |
|  | 128 | 39 ns | 16 ns | 2.44 |
|  | 256 | 69.2 ns | 21.4 ns | 3.23 |
|  | 1,000 | 89.3 ns | 28.1 ns | 3.18 |
|  | 10,000 | 276 ns | 160 ns | 1.73 |
|  | 100,000 | 2.13 µs | 1.62 µs | 1.32 |
| `mul_round` | 24 | 70.2 ns | 65.9 ns | 1.07 |
|  | 53 | 69.7 ns | 54 ns | 1.29 |
|  | 64 | 67.8 ns | 63.7 ns | 1.06 |
|  | 113 | 206 ns | 89.9 ns | 2.29 |
|  | 128 | 204 ns | 90.2 ns | 2.26 |
|  | 256 | 236 ns | 115 ns | 2.05 |
|  | 1,000 | 520 ns | 453 ns | 1.15 |
|  | 10,000 | 11.3 µs | 11.9 µs | 0.95 |
|  | 100,000 | 158 µs | 320 µs | 0.49 |
| `sqr_round` | 24 | 116 ns | 96.3 ns | 1.20 |
|  | 53 | 114 ns | 97.3 ns | 1.17 |
|  | 64 | 114 ns | 142 ns | 0.80 |
|  | 113 | 154 ns | 141 ns | 1.09 |
|  | 128 | 154 ns | 151 ns | 1.02 |
|  | 256 | 168 ns | 181 ns | 0.93 |
|  | 1,000 | 344 ns | 353 ns | 0.97 |
|  | 10,000 | 8.43 µs | 6.94 µs | 1.21 |
|  | 100,000 | 126 µs | 182 µs | 0.69 |
| `div_round` | 24 | 374 ns | 326 ns | 1.15 |
|  | 53 | 374 ns | 388 ns | 0.96 |
|  | 64 | 376 ns | 459 ns | 0.82 |
|  | 113 | 446 ns | 464 ns | 0.96 |
|  | 128 | 450 ns | 501 ns | 0.90 |
|  | 256 | 554 ns | 584 ns | 0.95 |
|  | 1,000 | 1.49 µs | 1.35 µs | 1.10 |
|  | 10,000 | 34.1 µs | 30.6 µs | 1.11 |
|  | 100,000 | 538 µs | 813 µs | 0.66 |
| `sqrt_round` | 24 | 567 ns | 159 ns | 3.57 |
|  | 53 | 667 ns | 157 ns | 4.25 |
|  | 64 | 661 ns | 211 ns | 3.13 |
|  | 113 | 739 ns | 246 ns | 3.00 |
|  | 128 | 738 ns | 349 ns | 2.11 |
|  | 256 | 888 ns | 448 ns | 1.98 |
|  | 1,000 | 1.62 µs | 1.08 µs | 1.50 |
|  | 10,000 | 22.2 µs | 17.5 µs | 1.27 |
|  | 100,000 | 452 µs | 592 µs | 0.76 |
| `abs_round` | 24 | 144 ns | 101 ns | 1.43 |
|  | 53 | 143 ns | 104 ns | 1.38 |
|  | 64 | 142 ns | 117 ns | 1.21 |
|  | 113 | 197 ns | 126 ns | 1.56 |
|  | 128 | 195 ns | 170 ns | 1.15 |
|  | 256 | 283 ns | 229 ns | 1.24 |
|  | 1,000 | 667 ns | 511 ns | 1.31 |
|  | 10,000 | 9.96 µs | 9.69 µs | 1.03 |
|  | 100,000 | 201 µs | 304 µs | 0.66 |
| `norm_round` | 24 | 72.1 ns | 89.6 ns | 0.80 |
|  | 53 | 70.8 ns | 92.4 ns | 0.77 |
|  | 64 | 70.6 ns | 150 ns | 0.47 |
|  | 113 | 108 ns | 156 ns | 0.69 |
|  | 128 | 107 ns | 157 ns | 0.68 |
|  | 256 | 117 ns | 169 ns | 0.69 |
|  | 1,000 | 237 ns | 295 ns | 0.80 |
|  | 10,000 | 5.04 µs | 4.31 µs | 1.17 |
|  | 100,000 | 77.8 µs | 123 µs | 0.63 |

### Functions of complex numbers

Against `mpc_exp` and the others; arguments of parts of exponents −2 to 1, the powers' exponents `w = 4u + vi` with `u`, `v` in `[1/2, 1)`, and `n = 17` for `powi`:

| Operation | Precision | `a` | MPC | `a`/MPC |
| --- | --- | ---: | ---: | ---: |
| `exp` | 53 | 859 ns | 1.45 µs | 0.59 |
|  | 113 | 1.05 µs | 2.11 µs | 0.50 |
|  | 256 | 2.11 µs | 3.71 µs | 0.57 |
|  | 1,000 | 8.29 µs | 18.4 µs | 0.45 |
|  | 10,000 | 444 µs | 1.27 ms | 0.35 |
|  | 100,000 | 12.9 ms | 42.1 ms | 0.31 |
| `ln` | 53 | 1.57 µs | 1.68 µs | 0.93 |
|  | 113 | 2.33 µs | 3.34 µs | 0.70 |
|  | 256 | 4.41 µs | 8 µs | 0.55 |
|  | 1,000 | 15.5 µs | 26.6 µs | 0.58 |
|  | 10,000 | 570 µs | 543 µs | 1.05 |
|  | 100,000 | 16.7 ms | 21.4 ms | 0.78 |
| `log10` | 53 | 2.42 µs | 2.29 µs | 1.05 |
|  | 113 | 3.26 µs | 4.68 µs | 0.70 |
|  | 256 | 6.27 µs | 10.5 µs | 0.60 |
|  | 1,000 | 18.2 µs | 34.5 µs | 0.53 |
|  | 10,000 | 588 µs | 718 µs | 0.82 |
|  | 100,000 | 16.9 ms | 28.9 ms | 0.58 |
| `arg` | 53 | 638 ns | 1.07 µs | 0.60 |
|  | 113 | 1.11 µs | 1.93 µs | 0.57 |
|  | 256 | 1.86 µs | 5.32 µs | 0.35 |
|  | 1,000 | 8.68 µs | 18.7 µs | 0.46 |
|  | 10,000 | 285 µs | 368 µs | 0.77 |
|  | 100,000 | 11.5 ms | 14.3 ms | 0.80 |
| `sin` | 53 | 1.53 µs | 1.76 µs | 0.87 |
|  | 113 | 1.8 µs | 2.18 µs | 0.83 |
|  | 256 | 2.86 µs | 3.06 µs | 0.93 |
|  | 1,000 | 9.36 µs | 11.4 µs | 0.82 |
|  | 10,000 | 456 µs | 584 µs | 0.78 |
|  | 100,000 | 13.1 ms | 23 ms | 0.57 |
| `cos` | 53 | 1.53 µs | 1.67 µs | 0.92 |
|  | 113 | 1.75 µs | 2.19 µs | 0.80 |
|  | 256 | 2.82 µs | 3.1 µs | 0.91 |
|  | 1,000 | 9.61 µs | 11.3 µs | 0.85 |
|  | 10,000 | 456 µs | 581 µs | 0.78 |
|  | 100,000 | 13.1 ms | 23 ms | 0.57 |
| `tan` | 53 | 1.96 µs | 4.27 µs | 0.46 |
|  | 113 | 2.26 µs | 4.63 µs | 0.49 |
|  | 256 | 3.4 µs | 6.01 µs | 0.57 |
|  | 1,000 | 10.5 µs | 16.7 µs | 0.63 |
|  | 10,000 | 475 µs | 619 µs | 0.77 |
|  | 100,000 | 13.4 ms | 24.2 ms | 0.55 |
| `sin_cos` | 53 | 1.86 µs | 1.82 µs | 1.02 |
|  | 113 | 2.1 µs | 2.21 µs | 0.95 |
|  | 256 | 3.23 µs | 3.15 µs | 1.03 |
|  | 1,000 | 9.95 µs | 11.7 µs | 0.85 |
|  | 10,000 | 464 µs | 580 µs | 0.80 |
|  | 100,000 | 13.2 ms | 23.1 ms | 0.57 |
| `sinh` | 53 | 1.52 µs | 1.75 µs | 0.87 |
|  | 113 | 1.76 µs | 2.23 µs | 0.79 |
|  | 256 | 2.85 µs | 3.08 µs | 0.92 |
|  | 1,000 | 9.44 µs | 11.3 µs | 0.84 |
|  | 10,000 | 457 µs | 580 µs | 0.79 |
|  | 100,000 | 13.1 ms | 22.9 ms | 0.57 |
| `cosh` | 53 | 1.52 µs | 1.75 µs | 0.87 |
|  | 113 | 1.75 µs | 2.21 µs | 0.79 |
|  | 256 | 2.84 µs | 3.08 µs | 0.92 |
|  | 1,000 | 9.48 µs | 11.3 µs | 0.84 |
|  | 10,000 | 456 µs | 592 µs | 0.77 |
|  | 100,000 | 13.1 ms | 22.8 ms | 0.57 |
| `tanh` | 53 | 1.98 µs | 4.04 µs | 0.49 |
|  | 113 | 2.21 µs | 4.36 µs | 0.51 |
|  | 256 | 3.4 µs | 5.84 µs | 0.58 |
|  | 1,000 | 10.5 µs | 15.2 µs | 0.69 |
|  | 10,000 | 475 µs | 639 µs | 0.74 |
|  | 100,000 | 13.4 ms | 24.1 ms | 0.56 |
| `asin` | 53 | 3.67 µs | 6.19 µs | 0.59 |
|  | 113 | 4.56 µs | 11.9 µs | 0.38 |
|  | 256 | 8.12 µs | 24.9 µs | 0.33 |
|  | 1,000 | 19.6 µs | 158 µs | 0.12 |
|  | 10,000 | 612 µs | 574 µs | 1.07 |
|  | 100,000 | 17.5 ms | 22.6 ms | 0.77 |
| `acos` | 53 | 3.7 µs | 7.25 µs | 0.51 |
|  | 113 | 4.57 µs | 14.5 µs | 0.32 |
|  | 256 | 8.16 µs | 24.7 µs | 0.33 |
|  | 1,000 | 19.6 µs | 142 µs | 0.14 |
|  | 10,000 | 611 µs | 577 µs | 1.06 |
|  | 100,000 | 17.5 ms | 22.5 ms | 0.78 |
| `atan` | 53 | 2.01 µs | 4.4 µs | 0.46 |
|  | 113 | 2.69 µs | 8.16 µs | 0.33 |
|  | 256 | 5.94 µs | 17 µs | 0.35 |
|  | 1,000 | 17.6 µs | 53.5 µs | 0.33 |
|  | 10,000 | 586 µs | 1.07 ms | 0.55 |
|  | 100,000 | 16.9 ms | 41.9 ms | 0.40 |
| `asinh` | 53 | 3.67 µs | 6.38 µs | 0.57 |
|  | 113 | 4.55 µs | 11.9 µs | 0.38 |
|  | 256 | 8.09 µs | 25.7 µs | 0.31 |
|  | 1,000 | 19.7 µs | 173 µs | 0.11 |
|  | 10,000 | 613 µs | 573 µs | 1.07 |
|  | 100,000 | 17.5 ms | 22.4 ms | 0.78 |
| `acosh` | 53 | 3.72 µs | 7.27 µs | 0.51 |
|  | 113 | 4.59 µs | 14.5 µs | 0.32 |
|  | 256 | 8.21 µs | 24.9 µs | 0.33 |
|  | 1,000 | 19.7 µs | 142 µs | 0.14 |
|  | 10,000 | 612 µs | 574 µs | 1.07 |
|  | 100,000 | 17.5 ms | 22.4 ms | 0.78 |
| `atanh` | 53 | 2.02 µs | 4.45 µs | 0.45 |
|  | 113 | 2.71 µs | 7.74 µs | 0.35 |
|  | 256 | 5.98 µs | 17.3 µs | 0.35 |
|  | 1,000 | 17.6 µs | 52.9 µs | 0.33 |
|  | 10,000 | 585 µs | 1.08 ms | 0.54 |
|  | 100,000 | 16.9 ms | 41.7 ms | 0.41 |
| `pow` | 53 | 3.75 µs | 6.66 µs | 0.56 |
|  | 113 | 5.08 µs | 9.41 µs | 0.54 |
|  | 256 | 8.36 µs | 18.1 µs | 0.46 |
|  | 1,000 | 26.9 µs | 62 µs | 0.43 |
|  | 10,000 | 1.04 ms | 2.74 ms | 0.38 |
|  | 100,000 | 29.7 ms | 65.5 ms | 0.45 |
| `pow_real` | 53 | 3.58 µs | 6.27 µs | 0.57 |
|  | 113 | 4.67 µs | 8.6 µs | 0.54 |
|  | 256 | 8 µs | 17.3 µs | 0.46 |
|  | 1,000 | 26.8 µs | 60 µs | 0.45 |
|  | 10,000 | 1.04 ms | 1.85 ms | 0.56 |
|  | 100,000 | 29.7 ms | 66.3 ms | 0.45 |
| `powi`, `n = 17` | 53 | 999 ns | 800 ns | 1.25 |
|  | 113 | 1.09 µs | 822 ns | 1.33 |
|  | 256 | 1.21 µs | 997 ns | 1.21 |
|  | 1,000 | 2.43 µs | 2.12 µs | 1.14 |
|  | 10,000 | 45.8 µs | 41.7 µs | 1.10 |
|  | 100,000 | 673 µs | 1.04 ms | 0.65 |

### Matrices of floats

`&a * &b` of `n × n` matrices of `BigFloat`, every element correctly rounded, against a loop of `mpfr_fma` (rounded at each step) and against `mpfr_dot` (correctly rounded); `a.solve(&b)` with one column against Gaussian elimination by `mpfr_fms`:

| Operation | Precision | `a` | MPFR, `mpfr_fma` loop | `a`/loop | `mpfr_dot` |
| --- | --- | ---: | ---: | ---: | ---: |
| `&a * &b`, 4 × 4 | 53 | 1.32 µs | 647 ns | 2.04 | 2.49 µs |
|  | 113 | 1.42 µs | 1.25 µs | 1.14 | 2.64 µs |
|  | 256 | 1.87 µs | 1.81 µs | 1.03 | 3.24 µs |
|  | 1,000 | 6.93 µs | 7.22 µs | 0.96 | 9.09 µs |
| `&a * &b`, 8 × 8 | 53 | 8.83 µs | 5.66 µs | 1.56 | 18.9 µs |
|  | 113 | 9.19 µs | 12.4 µs | 0.74 | 18.8 µs |
|  | 256 | 12.4 µs | 17.6 µs | 0.70 | 23 µs |
|  | 1,000 | 51.1 µs | 61.7 µs | 0.83 | 68.7 µs |
| `&a * &b`, 16 × 16 | 53 | 64.1 µs | 53.9 µs | 1.19 | 144 µs |
|  | 113 | 70.4 µs | 124 µs | 0.57 | 142 µs |
|  | 256 | 98.8 µs | 162 µs | 0.61 | 170 µs |
|  | 1,000 | 396 µs | 503 µs | 0.79 | 543 µs |
| `a.solve(&b)`, 4 × 4 | 53 | 1.95 µs | 474 ns | 4.11 | – |
|  | 113 | 2.22 µs | 807 ns | 2.75 | – |
|  | 256 | 2.98 µs | 1.55 µs | 1.92 | – |
|  | 1,000 | 7.15 µs | 5.31 µs | 1.35 | – |
| `a.solve(&b)`, 8 × 8 | 53 | 8.69 µs | 3.1 µs | 2.80 | – |
|  | 113 | 9.43 µs | 5.86 µs | 1.61 | – |
|  | 256 | 12.7 µs | 9.66 µs | 1.31 | – |
|  | 1,000 | 35.7 µs | 31.7 µs | 1.13 | – |
| `a.solve(&b)`, 16 × 16 | 53 | 44.2 µs | 22.5 µs | 1.96 | – |
|  | 113 | 48 µs | 48.3 µs | 0.99 | – |
|  | 256 | 66.6 µs | 66.8 µs | 1.00 | – |
|  | 1,000 | 205 µs | 210 µs | 0.98 | – |

### Where the references are faster

- **Arb** at every size up to about 10⁵ bits (3 to 6 times at 53 bits for the elementary functions): tables of the functions at many points, fixed point on GMP's `mpn` level and no Ziv loop, because its balls only enclose the value. It returns something else, so it is listed for its time.
- **One and two limbs into an existing number** (`set_*`, 1.2 to 1.8 times, 2 to 4 ns of 5 to 9): MPFR's `mpfr_add1sp1` and the others are straight-line functions on the fields of an `mpfr_t`; `a` matches the special values of the `Kind`, makes a plan and writes through the `Vec` of the mantissa. With a new result both allocate, and the gap is 0 to 20 per cent.
- **Three to sixteen limbs and mixed lengths with a short result** (`set_add` 256 to 1,000 bits, `53 (operands 1,000)` and the like, about 2 times): the same fixed cost, about 25 ns, in the general window code, where MPFR reads the top limbs with code of its own. Where MPFR scans long operands (`1,000 (operands 100,000)`), `a` is 3 to 10 times faster.
- **Square root at 64 to 113 bits, `rec_sqrt_round`, `cbrt_round` and `nth_root_round(5)` below 1,000 bits** (1.1 to 3.3 times): `a` takes the integer square root or `n`-th root of the crate's numbers, where MPFR's `sqrt1n` and `sqrt2` use a table and Newton's steps in registers and `mpfr_rec_sqrt` iterates on the reciprocal square root itself.
- **`mul_add_round` and `powi_round` at 113 to 256 bits** (2 to 3.5 times): two-limb operands go through the general path (working memory, a four-limb product, the window sum), and powers beyond 64 bits through the general Ziv loop; at 64 bits and less both have register paths and are at par or faster.
- **Division at 1,000 and 10⁴ bits** (1.13 and 1.28 times): `a` divides `2n + 1` by `n` limbs with the remainder, MPFR computes the short quotient alone (`mpfr_divhigh_n`). From 10⁵ bits Newton's division makes `a` 1.5 to 3 times faster.
- **`from_f64_round`, `floor` and `rint_round`** (1.2 to 4 times): `a` makes a new number where MPFR writes into one, and `rint_round` to a smaller precision copies the mantissa through the general rounding. rug, which allocates too, is slower than `a` there.
- **Reading 20 to 100 digits** (1.1 to 1.5 times): the general exact path allocates the digits, a power and the quotient; from 10⁴ digits `a` is 1.2 to 2.4 times faster.
- **The digits of floats of exponents near `2^60` at 53 to 113 bits** (1.2 to 1.3 times): Ziv's loop squares the power of 10 about 60 times at some 200 bits, each into a new buffer.
- **The logarithms and the inverse hyperbolic functions at 10⁴ bits** (1.2 times) and at 24 to 64 bits (1.0 to 1.5 times): from 5,000 bits both take the AGM, whose steps here are balls over the general square root and division; at small sizes one Newton step on the exponential costs more than MPFR's small-precision logarithm. `sinh` and `tanh` at 128 bits are 1.01 to 1.07 times.
- **`atan` at 10⁶ bits** (1.32 times): Newton's iteration costs about one and a half `sin_cos` at the full precision, and the bit burst is only a little ahead of the halvings there.
- **Complex numbers at small sizes**: `BigComplex` has no forms into an existing number, so `add_round` is two new `BigFloat`s against two `mpfr_add` into variables (2.4 to 3.2 times); `mul_round` at 113 to 256 bits (2.1 to 2.3 times), `sqrt_round` (up to 4.3 times), `abs_round` and the products and solves of matrices at 53 bits (up to 4.1 times for a 4 × 4 solve) are built on the allocating real operations. Among the functions, `powi_round` (1.1 to 1.3 times to 10⁴ bits: each product of the binary powering is a new number), `sin_cos_round` at 53 and 256 bits, `log10_round` at 53 bits and `ln_round` and the inverse functions at 10⁴ bits (the real logarithm's AGM, above) are 1.02 to 1.07 times. From 10⁵ bits `a`'s exact products make it faster everywhere but `add_round`.
- **Cached constants below 256 bits**: Arb copies 7 to 11 ns, `a` 14 ns (an allocation). From 10⁵ bits `a` and MPFR copy at the same speed (0.9 to 1.26, the noise of a copy).
