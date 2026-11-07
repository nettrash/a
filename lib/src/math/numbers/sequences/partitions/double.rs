//! The terms of Rademacher's series in double precision, with bounds of their errors.
//!
//! The functions here do not rely on the platform's `exp` and `cos`, whose accuracy Rust does not
//! specify: each is a polynomial of a reduced argument, evaluated by Horner's rule, whose error
//! follows from IEEE 754's correctly rounded operations. `u = 2^-53` is the unit roundoff.

use super::expsum::Sum;

/// The unit roundoff of `f64`.
pub(super) const U: f64 = 1.0 / 9_007_199_254_740_992.0;

/// `ln 2` in two parts, fdlibm's: the high part has 21 zero bits at the bottom, so that `q ln2_hi` is
/// exact for `|q| < 2^21`; together they are `ln 2` within `2^-100`.
const LN2_HI: f64 = f64::from_bits(0x3FE6_2E42_FEE0_0000);
const LN2_LO: f64 = f64::from_bits(0x3DEA_39EF_3579_3C76);

/// `1 / k!` for `k` up to 16, each the nearest double.
const INVERSE_FACTORIALS: [f64; 17] = [
    1.0,
    1.0,
    0.5,
    0.166_666_666_666_666_66,
    0.041_666_666_666_666_664,
    0.008_333_333_333_333_333,
    0.001_388_888_888_888_889,
    0.000_198_412_698_412_698_4,
    2.480_158_730_158_73e-5,
    2.755_731_922_398_589_3e-6,
    2.755_731_922_398_589e-7,
    2.505_210_838_544_172e-8,
    2.087_675_698_786_81e-9,
    1.605_904_383_682_161_3e-10,
    1.147_074_559_772_972_5e-11,
    7.647_163_731_819_816e-13,
    4.779_477_332_387_385e-14,
];

/// `e^x` for `|x| <= 700`, with a relative error below `16 u`.
///
/// `x = q ln 2 + r` with `q` the nearest integer to `x / ln 2` and `|r| <= 0.35` (Cody and Waite's
/// reduction, `r` within `2u |r| + 2^-90` of its value), then `e^r` by its Taylor polynomial of
/// degree 13, whose remainder is below `0.35^14 / 14! < 2^-57`, evaluated by Horner's rule: every
/// partial value is below `e^0.35 < 1.42` and the step's factor below 0.35, so the rounding errors
/// add up to less than `1.42 u / (1 - 0.35) < 2.2 u` absolutely, `3.2 u` relatively as `e^r > 0.7`;
/// the coefficients' own half-ulp errors add `0.8 u`. The error of `r` adds `2.1 u`; the scaling
/// by `2^q` is exact.
pub(super) fn exp(x: f64) -> f64 {
    debug_assert!(x.abs() <= 700.0);
    let q = (x * core::f64::consts::LOG2_E).round();
    let r = (x - q * LN2_HI) - q * LN2_LO;
    let mut p = INVERSE_FACTORIALS[13];
    for i in (0..13).rev() {
        p = p * r + INVERSE_FACTORIALS[i];
    }
    // `2^q` as two factors, so that neither leaves the range of normal numbers.
    let (a, b) = (q as i32 / 2, q as i32 - q as i32 / 2);
    p * pow2(a) * pow2(b)
}

/// `2^e`, exactly, for `|e| <= 1022`.
fn pow2(e: i32) -> f64 {
    f64::from_bits(((1023 + e) as u64) << 52)
}

/// `cos(π a / b)` for `0 <= a < 2b`, `b >= 1`, with a relative error below `12 u`, unless it is
/// zero, which it gives exactly.
///
/// The angle is brought to `π t`, `0 <= t <= 1/2`, by `cos(π (2 - t)) = cos(π t)` and
/// `cos(π (1 - t)) = -cos(π t)`, in integers; then `cos(y)` for `y = π t <= π/4` and `sin(y)` for
/// `y = π (1/2 - t) < π/4` otherwise, by their Taylor polynomials of degrees 16 and 17, whose
/// remainders are below `0.79^18 / 18! < 2^-57`. `y` is `π` times a quotient, within `2.5 u y`; the
/// error of `y` moves `cos(y)` by `tan(y) 2.5 u y < 2.5 u` and `sin(y)` by `y cot(y) 2.5 u < 2.5 u`
/// relatively; Horner's rule on `y^2` adds less than `4 u`, and the coefficients and the last
/// product less than `3 u`.
pub(super) fn cos_pi(a: u64, b: u64) -> f64 {
    debug_assert!(a < 2 * b);
    let (a, b) = (u128::from(a), u128::from(b));
    // `t = a / b` in `[0, 2)`: `2 - t` for `t > 1`.
    let a = if a > b { 2 * b - a } else { a };
    // `t` in `[0, 1]`: `1 - t` and a minus for `t > 1/2`.
    let (a, negative) = if 2 * a > b { (b - a, true) } else { (a, false) };
    // `t = a / b` in `[0, 1/2]`.
    let v = if 4 * a <= b {
        let y = core::f64::consts::PI * (a as f64 / b as f64);
        cos_poly(y)
    } else if 2 * a == b {
        0.0
    } else {
        // `cos(π t) = sin(π (1/2 - t))`, `1/2 - t = (b - 2a) / (2b)`.
        let y = core::f64::consts::PI * ((b - 2 * a) as f64 / (2 * b) as f64);
        sin_poly(y)
    };
    if negative { -v } else { v }
}

/// `cos(y)` for `0 <= y <= π/4`, its Taylor polynomial of degree 16: `Σ (-1)^j z^j / (2j)!` for
/// `z = y^2`, by Horner's rule from `j = 8`.
fn cos_poly(y: f64) -> f64 {
    let z = y * y;
    let mut p = INVERSE_FACTORIALS[16];
    for j in (0..8).rev() {
        let c = INVERSE_FACTORIALS[2 * j];
        p = if j & 1 == 1 { -c } else { c } + z * p;
    }
    p
}

/// `sin(y)` for `0 <= y <= π/4`, its Taylor polynomial of degree 17.
fn sin_poly(y: f64) -> f64 {
    let z = y * y;
    // `Σ (-1)^j z^j / (2j + 1)!`, by Horner's rule from `j = 8`.
    let mut p = INVERSE_FACTORIALS[16] / 17.0;
    for j in (0..8).rev() {
        let c = INVERSE_FACTORIALS[2 * j + 1];
        p = c - z * p;
    }
    y * p
}

/// `cosh(x) - sinh(x) / x` for `x > 0`, and a bound of its relative error in units of `u`, for
/// an `x` within `dx` (relative) of the true argument.
///
/// From 2 up, `(e^x (x - 1) + e^-x (x + 1)) / (2x)`, a sum of positive parts: the error of `e^x`
/// (`16 u`, and the argument's `x dx` relatively) and of `e^-x` as its inverse, and six roundings.
/// Below 2, the Taylor series `Σ_(j>=1) 2j x^(2j) / (2j+1)!`, whose terms are positive, to `j = 14`,
/// where the remainder is below `2^-60` of the sum.
pub(super) fn u_of(x: f64, dx: f64) -> (f64, f64) {
    if x >= 2.0 {
        let e = exp(x);
        let v = (e * (x - 1.0) + (x + 1.0) / e) / (2.0 * x);
        // `x - 1` and `x + 1` carry the argument's error times `x / (x - 1) <= 2`.
        (
            v,
            16.0 + 1.0 + 8.0 + (x * dx) / U * 1.000_001 + 4.0 * dx / U,
        )
    } else {
        let z = x * x;
        let mut term = z / 3.0;
        let mut v = term;
        for j in 2..=14u32 {
            let j = f64::from(j);
            // `t_j / t_(j-1) = z j / ((j - 1) (2j) (2j + 1))`.
            term = term * z * j / ((j - 1.0) * (2.0 * j) * (2.0 * j + 1.0));
            v += term;
        }
        // Each term within `6 j u`, the sum within `14 u`; the argument's error times `2j <= 28`.
        (v, 14.0 * 6.0 + 14.0 + 28.0 * dx / U * 1.000_001)
    }
}

/// The term `√(3/k) (4 / (24n - 1)) A_k(n) U(C/k)` in double precision for `c = C`, within `c_err`
/// relatively, and a bound of its absolute error.
pub(super) fn term(sum: &Sum, k: u64, n24: f64, n24_err: f64, c: f64, c_err: f64) -> (f64, f64) {
    // `x = C / k`, within `c_err + u`.
    let x = c / k as f64;
    let dx = c_err + U;
    let (uv, mut err) = u_of(x, dx);
    // `√(3 sp / (k sq))`: the quotient within `u` (and the products' roundings when they pass
    // `2^53`), the square root within `u` and half the quotient's.
    let num = 3 * u128::from(sum.sp);
    let den = u128::from(k) * u128::from(sum.sq);
    let mut v = (num as f64 / den as f64).sqrt();
    err += 3.0;
    v *= sum.prefactor as f64 * 4.0;
    err += 1.0;
    v /= n24;
    err += 1.0 + n24_err / U;
    for &(a, b) in &sum.cos[..sum.len] {
        let c = cos_pi(a, b);
        if c == 0.0 {
            return (0.0, 0.0);
        }
        v *= c;
        err += 12.0 + 1.0;
    }
    v *= uv;
    err += 1.0;
    // The relative bound, made absolute, with room for the roundings of the bound itself.
    (v, v.abs() * err * U * 1.0001)
}
