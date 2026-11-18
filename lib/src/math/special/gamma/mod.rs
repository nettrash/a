//! The gamma function and its family: logarithms, polygammas, beta functions, the regularised
//! incomplete gamma and beta functions and their inverses, and factorials of floats.

use super::{Real, via};

mod beta;
mod cplx;
mod ddm;
mod ibeta;
mod igamma;
mod igamma_tab;
mod psi;
mod real;
mod tab;

pub(crate) use cplx::{gamma_c64, lgamma_c64};
// `sin_pi` is for the other areas.
#[allow(unused_imports)]
pub(crate) use real::{gamma_f64, lgamma_f64, sin_pi};

/// `ln |Γ(x)|` and the sign of `Γ(x)` (`1` or `−1`; `1` at the poles, NaN for NaN).
pub fn lgamma_sign<T: Real>(x: T) -> (T, T) {
    let v = x.to_f64();
    let sign = if v.is_nan() {
        f64::NAN
    } else if v > 0.0 || v == v.floor() {
        if v == 0.0 && v.is_sign_negative() {
            -1.0
        } else {
            1.0
        }
    } else if (v.floor() as i64) % 2 == 0 {
        1.0
    } else {
        -1.0
    };
    (T::from_f64(lgamma_f64(v)), T::from_f64(sign))
}

/// The digamma function `ψ(x) = Γ′(x)/Γ(x)` (`∓∞` at `±0`, NaN at the negative integers and
/// `−∞`).
pub fn digamma<T: Real>(x: T) -> T {
    via(x, psi::digamma_f64)
}

/// The polygamma function `ψ⁽ⁿ⁾(x)`, the `n`-th derivative of [`digamma`] (`n = 0` is
/// `digamma`; for `n ≥ 1` an infinity at `±0`, NaN at the negative integers, and `±∞` or `±0`
/// where the value leaves the range).
pub fn polygamma<T: Real>(n: u32, x: T) -> T {
    via(x, |x| psi::polygamma_f64(n, x))
}

/// The beta function `B(a, b) = Γ(a) Γ(b) / Γ(a + b)`, of any real `a` and `b` (accurate to the
/// stated bound for `a, b > 0`): `±∞` at `a = ±0` (or `b`), zero where only `a + b` is a pole of
/// `Γ`, NaN at the other poles.
pub fn beta<T: Real>(a: T, b: T) -> T {
    T::from_f64(beta::beta_f64(a.to_f64(), b.to_f64()))
}

/// `ln |B(a, b)|`: `+∞` at the poles of `Γ(a)` or `Γ(b)`, `−∞` where only `a + b` is a pole,
/// NaN where both are.
pub fn lbeta<T: Real>(a: T, b: T) -> T {
    T::from_f64(beta::lbeta_f64(a.to_f64(), b.to_f64()))
}

/// The regularised lower incomplete gamma function `P(a, x) = γ(a, x) / Γ(a)`, for `a > 0`,
/// `x ≥ 0`.
pub fn gamma_p<T: Real>(a: T, x: T) -> T {
    T::from_f64(igamma::gamma_p_f64(a.to_f64(), x.to_f64()))
}

/// The regularised upper incomplete gamma function `Q(a, x) = Γ(a, x) / Γ(a) = 1 − P(a, x)`.
pub fn gamma_q<T: Real>(a: T, x: T) -> T {
    T::from_f64(igamma::gamma_q_f64(a.to_f64(), x.to_f64()))
}

/// The `x ≥ 0` with `P(a, x) = p`, for `a > 0` and `0 ≤ p ≤ 1`.
pub fn gamma_p_inv<T: Real>(a: T, p: T) -> T {
    T::from_f64(igamma::gamma_p_inv_f64(a.to_f64(), p.to_f64()))
}

/// The `x ≥ 0` with `Q(a, x) = q`, for `a > 0` and `0 ≤ q ≤ 1`.
pub fn gamma_q_inv<T: Real>(a: T, q: T) -> T {
    T::from_f64(igamma::gamma_q_inv_f64(a.to_f64(), q.to_f64()))
}

/// The regularised incomplete beta function `I_x(a, b) = B(x; a, b) / B(a, b)`, for `a, b > 0`
/// and `0 ≤ x ≤ 1`.
pub fn beta_inc<T: Real>(a: T, b: T, x: T) -> T {
    T::from_f64(ibeta::beta_inc_f64(a.to_f64(), b.to_f64(), x.to_f64()))
}

/// `1 − I_x(a, b) = I_(1−x)(b, a)`, without the cancellation.
pub fn beta_inc_c<T: Real>(a: T, b: T, x: T) -> T {
    T::from_f64(ibeta::beta_inc_c_f64(a.to_f64(), b.to_f64(), x.to_f64()))
}

/// The `x` in `[0, 1]` with `I_x(a, b) = p`.
pub fn beta_inc_inv<T: Real>(a: T, b: T, p: T) -> T {
    T::from_f64(ibeta::beta_inc_inv_f64(a.to_f64(), b.to_f64(), p.to_f64()))
}

/// The `x` in `[0, 1]` with `1 − I_x(a, b) = q`.
pub fn beta_inc_c_inv<T: Real>(a: T, b: T, q: T) -> T {
    T::from_f64(ibeta::beta_inc_c_inv_f64(
        a.to_f64(),
        b.to_f64(),
        q.to_f64(),
    ))
}

/// `x! = Γ(x + 1)`, without rounding `x + 1` (NaN at the negative integers).
pub fn factorial<T: Real>(x: T) -> T {
    via(x, beta::factorial_f64)
}

/// The binomial coefficient `C(n, k) = Γ(n + 1) / (Γ(k + 1) Γ(n − k + 1))` of floats: zero for a
/// negative integer `k` and for integers `0 ≤ n < k`, `(−1)^k C(k − n − 1, k)` for a negative
/// integer `n`, and NaN where only the numerator has a pole.
pub fn binomial<T: Real>(n: T, k: T) -> T {
    T::from_f64(beta::binomial_f64(n.to_f64(), k.to_f64()))
}

/// The rising factorial (Pochhammer's symbol) `(x)ₙ = Γ(x + n) / Γ(x)` of floats: the product
/// `x(x + 1)…(x + n − 1)` for an integer `n` (its reciprocal for a negative one), zero where
/// only `x` is a pole, NaN where only `x + n` is.
pub fn pochhammer<T: Real>(x: T, n: T) -> T {
    T::from_f64(beta::pochhammer_f64(x.to_f64(), n.to_f64()))
}
