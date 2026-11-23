//! The Bessel functions of any real order, scaled and spherical.

mod general;
mod ik01;
mod int01;
pub(crate) mod kernels;
mod sph;
mod tables01;
mod tables_ik;
mod tables_olver;

use super::Real;
use super::dd;
use dd::Dd;
use general::sin_cos_pi_dd;
pub(crate) use int01::{about_zero, dd2, horner, poly_dd3};

/// Whether `v` is an integer, and whether it is odd.
#[inline]
fn int_parity(v: f64) -> (bool, bool) {
    let is = v == v.floor();
    let h = 0.5 * v;
    (is, is && h != h.floor())
}

/// `J_ν(x)` of `f64`s.
pub(crate) fn j_f64(nu: f64, x: f64) -> f64 {
    // The orders 0 and 1 first: their own functions handle every x.
    if nu == 0.0 {
        return int01::j0(x);
    }
    if nu == 1.0 {
        return int01::j1(x);
    }
    if nu.is_nan() || x.is_nan() || nu.is_infinite() {
        return f64::NAN;
    }
    let (is_int, odd) = int_parity(nu);
    if x < 0.0 {
        if !is_int {
            return f64::NAN;
        }
        let v = j_f64(nu, -x);
        return if odd { -v } else { v };
    }
    if nu < 0.0 {
        if is_int {
            let v = j_f64(-nu, x);
            return if odd { -v } else { v };
        }
        // J_(−a) = cos(aπ)·J_a − sin(aπ)·Y_a, the terms in double-double, Y_a's scale apart (it
        // may overflow where sin(aπ)·Y_a does not).
        let a = -nu;
        let (s, c) = sin_cos_pi_dd(a);
        if x == 0.0 {
            return f64::INFINITY.copysign(s.hi);
        }
        if x.is_infinite() {
            return 0.0;
        }
        if x < 1.0e-150 && a >= 0.5 {
            // −sin(aπ)·Y_a dominates; formed before the rounding, as Y_a may overflow alone.
            let (m, e) = general::y_tiny(a, x);
            return kernels::ldexp(m.mul(s).neg().value(), e);
        }
        let (j, ym, ye) = general::jy_parts(a, x, c.hi != 0.0, true);
        if !ym.hi.is_finite() {
            return -s.hi * ym.hi;
        }
        let jt = if c.hi != 0.0 {
            c.mul_f64(j)
        } else {
            Dd::new(0.0)
        };
        return general::add_scaled(jt, s.mul(ym).neg(), ye);
    }
    if x == 0.0 {
        return if nu == 0.0 { 1.0 } else { 0.0 };
    }
    if x.is_infinite() {
        return 0.0;
    }
    general::jy(nu, x, true, false).0
}

/// `Y_ν(x)` of `f64`s.
pub(crate) fn y_f64(nu: f64, x: f64) -> f64 {
    if nu == 0.0 {
        return int01::y0(x);
    }
    if nu == 1.0 {
        return int01::y1(x);
    }
    if nu.is_nan() || x.is_nan() || nu.is_infinite() || x < 0.0 {
        return f64::NAN;
    }
    let (is_int, odd) = int_parity(nu);
    if nu < 0.0 {
        if is_int {
            let v = y_f64(-nu, x);
            return if odd { -v } else { v };
        }
        // Y_(−a) = sin(aπ)·J_a + cos(aπ)·Y_a, as J_(−a).
        let a = -nu;
        let (s, c) = sin_cos_pi_dd(a);
        if x == 0.0 {
            return if c.hi == 0.0 {
                0.0
            } else {
                f64::NEG_INFINITY * c.hi.signum()
            };
        }
        if x.is_infinite() {
            return 0.0;
        }
        if x < 1.0e-150 && a >= 0.5 && c.hi != 0.0 {
            let (m, e) = general::y_tiny(a, x);
            return kernels::ldexp(m.mul(c).value(), e);
        }
        let (j, ym, ye) = general::jy_parts(a, x, true, c.hi != 0.0);
        if c.hi == 0.0 {
            return s.mul_f64(j).value();
        }
        if !ym.hi.is_finite() {
            return c.hi * ym.hi;
        }
        return general::add_scaled(s.mul_f64(j), c.mul(ym), ye);
    }
    if x == 0.0 {
        return f64::NEG_INFINITY;
    }
    if x.is_infinite() {
        return 0.0;
    }
    general::jy(nu, x, false, true).1
}

/// `I_ν(x)` of `f64`s, times `e^(−|x|)` when `scaled`.
pub(crate) fn i_f64(nu: f64, x: f64, scaled: bool) -> f64 {
    if nu == 0.0 {
        return ik01::i0(x, scaled);
    }
    if nu == 1.0 {
        return ik01::i1(x, scaled);
    }
    if nu.is_nan() || x.is_nan() || nu.is_infinite() {
        return f64::NAN;
    }
    let (is_int, odd) = int_parity(nu);
    if x < 0.0 {
        if !is_int {
            return f64::NAN;
        }
        let v = i_f64(nu, -x, scaled);
        return if odd { -v } else { v };
    }
    if nu < 0.0 {
        if is_int {
            return i_f64(-nu, x, scaled);
        }
        // I_(−a) = I_a + (2/π)·sin(aπ)·K_a, the terms in double-double, K_a's scale apart (it may
        // overflow where the sum does not; scaled, e^(−x)·K_a = (e^x·K_a)·e^(−2x) with the scales
        // apart, see k_exp_neg_parts, as either factor alone may be out of range).
        let a = -nu;
        let (s, _) = sin_cos_pi_dd(a);
        if x == 0.0 {
            return f64::INFINITY.copysign(s.hi);
        }
        if x.is_infinite() {
            return if scaled { 0.0 } else { f64::INFINITY };
        }
        let s2 = s.mul(int01::TWO_OVER_PI);
        if x < 1.0e-150 && a >= 0.5 {
            // (2/π)·sin(aπ)·K_a dominates (e^(∓x) = 1 here).
            let (m, e) = general::k_tiny(a, x);
            return kernels::ldexp(m.mul(s2).value(), e);
        }
        let iv = general::i(a, x, scaled);
        let (km, ke) = if scaled {
            general::k_exp_neg_parts(a, x)
        } else {
            general::k_parts(a, x, false)
        };
        if !km.hi.is_finite() {
            return s2.hi * km.hi;
        }
        let t = s2.mul(km);
        let v = general::add_scaled(Dd::new(iv), t, ke);
        // Near the zero of I_(−a) (one, for a in (2m − 1, 2m)) the two terms cancel, by more than a
        // bit here: there its power series in double-double, whose terms cancel far less.
        let tv = kernels::ldexp(t.hi, ke);
        if v.abs() < 0.5 * iv.abs().max(tv.abs()) && a < 100.0 && x <= 2.0 * a + 30.0 {
            return general::i_neg_series(a, x, scaled);
        }
        return v;
    }
    if x == 0.0 {
        return if nu == 0.0 { 1.0 } else { 0.0 };
    }
    if x.is_infinite() {
        return if scaled { 0.0 } else { f64::INFINITY };
    }
    general::i(nu, x, scaled)
}

/// `K_ν(x)` of `f64`s, times `e^x` when `scaled`.
pub(crate) fn k_f64(nu: f64, x: f64, scaled: bool) -> f64 {
    if nu == 0.0 {
        return ik01::k0(x, scaled);
    }
    if nu.abs() == 1.0 {
        return ik01::k1(x, scaled);
    }
    if nu.is_nan() || x.is_nan() || nu.is_infinite() || x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return f64::INFINITY;
    }
    if x.is_infinite() {
        return 0.0;
    }
    general::k(nu.abs(), x, scaled)
}

/// The Bessel function of the first kind `J_ν(x)` (for `x < 0`, defined for integer `ν` only).
pub fn bessel_j<T: Real>(nu: T, x: T) -> T {
    T::from_f64(j_f64(nu.to_f64(), x.to_f64()))
}

/// The Bessel function of the second kind `Y_ν(x)`, for `x ≥ 0` (`−∞` at 0).
pub fn bessel_y<T: Real>(nu: T, x: T) -> T {
    T::from_f64(y_f64(nu.to_f64(), x.to_f64()))
}

/// The modified Bessel function of the first kind `I_ν(x)` (for `x < 0`, integer `ν` only).
pub fn bessel_i<T: Real>(nu: T, x: T) -> T {
    T::from_f64(i_f64(nu.to_f64(), x.to_f64(), false))
}

/// The modified Bessel function of the second kind `K_ν(x)`, for `x ≥ 0` (`∞` at 0).
pub fn bessel_k<T: Real>(nu: T, x: T) -> T {
    T::from_f64(k_f64(nu.to_f64(), x.to_f64(), false))
}

/// `e^(−|x|) I_ν(x)`, which does not overflow.
pub fn bessel_i_scaled<T: Real>(nu: T, x: T) -> T {
    T::from_f64(i_f64(nu.to_f64(), x.to_f64(), true))
}

/// `e^x K_ν(x)`, which does not underflow.
pub fn bessel_k_scaled<T: Real>(nu: T, x: T) -> T {
    T::from_f64(k_f64(nu.to_f64(), x.to_f64(), true))
}

/// The spherical Bessel function of the first kind `jₙ(x) = √(π/(2x)) J_(n+1/2)(x)`.
pub fn sph_bessel_j<T: Real>(n: u32, x: T) -> T {
    T::from_f64(sph::sph_j(n, x.to_f64()))
}

/// The spherical Bessel function of the second kind `yₙ(x) = √(π/(2x)) Y_(n+1/2)(x)`.
pub fn sph_bessel_y<T: Real>(n: u32, x: T) -> T {
    T::from_f64(sph::sph_y(n, x.to_f64()))
}

/// The modified spherical Bessel function of the first kind `iₙ(x) = √(π/(2x)) I_(n+1/2)(x)`.
pub fn sph_bessel_i<T: Real>(n: u32, x: T) -> T {
    T::from_f64(sph::sph_i(n, x.to_f64()))
}

/// The modified spherical Bessel function of the second kind `kₙ(x) = √(π/(2x)) K_(n+1/2)(x)`
/// (SciPy's `spherical_kn`).
pub fn sph_bessel_k<T: Real>(n: u32, x: T) -> T {
    T::from_f64(sph::sph_k(n, x.to_f64()))
}
