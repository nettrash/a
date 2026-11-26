//! The Bessel functions of any real order, scaled and spherical.

mod general;
mod ik01;
mod int01;
pub(crate) mod kernels;
mod sph;
mod tables01;
mod tables_f32;
mod tables_ik;
mod tables_olver;

use super::Real;
use super::dd;
use dd::Dd;
use general::sin_cos_pi_dd;
pub(crate) use int01::{about_zero, dd2, horner, poly_dd3};
use tables_f32::{J0_F32, J1_F32, Y0_F32, Y0_F32_LOG, Y1_F32, Y1_F32_LOG};

/// Whether `v` is an integer, and whether it is odd.
#[inline]
fn int_parity(v: f64) -> (bool, bool) {
    let is = v == v.floor();
    let h = 0.5 * v;
    (is, is && h != h.floor())
}

/// `J_ν(x)` of `f64`s (the dispatch to the orders 0 and 1 inlined into the caller).
#[inline]
pub(crate) fn j_f64(nu: f64, x: f64) -> f64 {
    if nu == 0.0 {
        return int01::j0(x);
    }
    if nu == 1.0 {
        return int01::j1(x);
    }
    j_rest(nu, x)
}

/// `J₀` or `J₁` (`rows` `J0_F32` or `J1_F32`) at `0 ≤ a < 25` for the `f32` form: the piece's
/// polynomial (its first 15 entries), times `a − z` where the piece holds a zero `z`.
#[inline(always)]
fn j01_f32_piece(a: f64, rows: &[[f64; 18]; 25]) -> f64 {
    let i = (a as usize).min(24);
    let row = &rows[i];
    let (c, z) = row.split_at(15);
    let c: &[f64; 15] = c.try_into().unwrap_or(&[0.0; 15]);
    let p = kernels::estrin(a - (i as f64 + 0.5), c);
    let f = if z[0] != 0.0 { (a - z[1]) - z[2] } else { 1.0 };
    p * f
}

/// `J_ν(x)` for the `f32` form: `J₀` and `J₁` on `|x| < 25` from polynomials of `f32` grade
/// (`2⁻⁴⁵` relative, in `f64`, then rounded once to the `f32`), integer orders `2 ≤ n ≤ 400` by
/// `jy_int_f32` for `x > n` and `j_series_f32` for `x² ≤ 4(n + 1)`, elsewhere `j_f64`.
fn j_f32(nu: f64, x: f64) -> f64 {
    let a = x.abs();
    if a < 25.0 {
        if nu == 0.0 {
            return j01_f32_piece(a, &J0_F32);
        }
        if nu == 1.0 {
            let v = j01_f32_piece(a, &J1_F32);
            // (+0 at −0, as the `f64` form: `x < 0`, not the sign bit.)
            return if x < 0.0 { -v } else { v };
        }
    }
    if f32_int_order(nu) && x > 0.0 && x < f64::INFINITY {
        if x > nu {
            return jy_int_f32(nu as u32, x, true);
        }
        if x * x <= 4.0 * (nu + 1.0) {
            return j_series_f32(nu as u32, x);
        }
    }
    j_f64(nu, x)
}

/// `j_f64` for the orders other than 0 and 1.
#[inline(never)]
fn j_rest(nu: f64, x: f64) -> f64 {
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

/// `Y_ν(x)` of `f64`s (the dispatch to the orders 0 and 1 inlined into the caller).
#[inline]
pub(crate) fn y_f64(nu: f64, x: f64) -> f64 {
    if nu == 0.0 {
        return int01::y0(x);
    }
    if nu == 1.0 {
        return int01::y1(x);
    }
    y_rest(nu, x)
}

/// The piece of `1/2 ≤ a < 25` in `Y0_F32` and `Y1_F32`: of width 1/4 on `[1/2, 1)`, 1/2 on
/// `[1, 2)` and 1 from 2 on.
#[inline(always)]
fn y01_f32_index(a: f64) -> usize {
    if a < 2.0 {
        if a < 1.0 {
            (4.0 * a) as usize - 2
        } else {
            (2.0 * a) as usize
        }
    } else {
        (a as usize).min(24) + 2
    }
}

/// `Y₀` or `Y₁` (`rows` `Y0_F32` or `Y1_F32`) at `1/2 ≤ a < 25` for the `f32` form: the piece's
/// polynomial in `a − c` (its first 15 entries, `c` the last), times `a − z` where the piece holds
/// a zero `z`.
#[inline(always)]
fn y01_f32_piece(a: f64, rows: &[[f64; 19]; 27]) -> f64 {
    let row = &rows[y01_f32_index(a)];
    let (c, z) = row.split_at(15);
    let c: &[f64; 15] = c.try_into().unwrap_or(&[0.0; 15]);
    let p = kernels::estrin(a - z[3], c);
    let f = if z[0] != 0.0 { (a - z[1]) - z[2] } else { 1.0 };
    p * f
}

/// `Y_ν(x)` for the `f32` form: `Y₀` and `Y₁` on `0 < x < 25` from polynomials of `f32` grade
/// (`2⁻⁴⁵` relative, in `f64`, then rounded once to the `f32`): on `[1/2, 25)` the pieces of
/// `Y0_F32`/`Y1_F32`, below `(2/π) ln(x) J(x) + P(x²)` (over `x` for `Y₁`) with `J` from the
/// pieces of `J0_F32`/`J1_F32`; integer orders `2 ≤ n ≤ 400` by `jy_int_f32`; elsewhere `y_f64`.
fn y_f32(nu: f64, x: f64) -> f64 {
    if x > 0.0 && x < 25.0 && (nu == 0.0 || nu == 1.0) {
        return y01_f32(x, nu == 1.0);
    }
    if f32_int_order(nu) && x > 0.0 && x < f64::INFINITY {
        return jy_int_f32(nu as u32, x, false);
    }
    y_f64(nu, x)
}

/// `Y₀(x)` or `Y₁(x)` (`one`) of `f32` grade for `0 < x < 25` (see `y_f32`).
#[inline(always)]
fn y01_f32(x: f64, one: bool) -> f64 {
    if x >= 0.5 {
        return y01_f32_piece(x, if one { &Y1_F32 } else { &Y0_F32 });
    }
    let l = core::f64::consts::FRAC_2_PI * kernels::ln_dd(x).hi;
    let t = x * x - 0.125;
    if one {
        l * j01_f32_piece(x, &J1_F32) + kernels::estrin(t, &Y1_F32_LOG) / x
    } else {
        l * j01_f32_piece(x, &J0_F32) + kernels::estrin(t, &Y0_F32_LOG)
    }
}

/// `J_n(x)` for the `f32` form, `2 ≤ n ≤ 400`, `0 < x ≤ 2√(n + 1)`: the power series
/// `(x/2)ⁿ/n! · Σ (−x²/4)ᵏ/(k! (n + 1)ₖ)` in plain `f64`, whose terms decrease from the first
/// (their sum is at least about `e⁻¹`, so that it cancels by less than a factor of 10); the
/// prefactor by products of four factors `x/2k` at a time, so that it neither overflows nor
/// divides once a factor. Its error is a few `n` ulps of the value in `f64`.
fn j_series_f32(n: u32, x: f64) -> f64 {
    let h = 0.5 * x;
    let h2 = h * h;
    let h4 = h2 * h2;
    let mut pre = 1.0;
    let mut k = 1u32;
    while k + 3 <= n {
        let d = (k as f64 * (k + 1) as f64) * ((k + 2) as f64 * (k + 3) as f64);
        pre *= h4 / d;
        k += 4;
    }
    while k <= n {
        pre *= h / k as f64;
        k += 1;
    }
    let (mut t, mut sum) = (1.0f64, 1.0f64);
    let nf = n as f64;
    for k in 1..40 {
        let kf = k as f64;
        t *= -h2 / (kf * (nf + kf));
        sum += t;
        if t.abs() <= 1e-17 * sum {
            break;
        }
    }
    pre * sum
}

/// Whether `ν` is an integer order `2 ≤ ν ≤ 400` for the plain recurrence of the `f32` forms.
#[inline(always)]
fn f32_int_order(nu: f64) -> bool {
    (2.0..=400.0).contains(&nu) && nu == nu.round()
}

/// `J_n(x)` (`want_j`) or `Y_n(x)` for the `f32` forms, `2 ≤ n ≤ 400`, `0 < x < ∞` (`x > n` for
/// `J`, where the forward recurrence is stable in units of the modulus `M`): the plain forward
/// recurrence `c_(k+1) = (2k/x) c_k − c_(k−1)` in `f64` from the `f32`-grade `J₀`, `J₁` or `Y₀`,
/// `Y₁` below 25 and the modulus-phase forms from 25 on. Its error is a few `n` ulps of `M` in
/// `f64`, far below the `f32`'s rounding.
fn jy_int_f32(n: u32, x: f64, want_j: bool) -> f64 {
    let (a, b) = if x < 25.0 {
        if want_j {
            (j01_f32_piece(x, &J0_F32), j01_f32_piece(x, &J1_F32))
        } else {
            (y01_f32(x, false), y01_f32(x, true))
        }
    } else {
        let (j0, j1, y0, y1) = int01::jy01_large_both(x);
        if want_j { (j0, j1) } else { (y0, y1) }
    };
    let r = 2.0 / x;
    let (mut p, mut q) = (b, a);
    for k in 1..n {
        let c = kernels::mla(k as f64 * r, p, -q);
        q = p;
        p = c;
        if c.abs() >= 1e300 {
            // Y past the range (or ∞; the finite p and q make no NaN): it only grows from here
            // (k > x).
            return c;
        }
    }
    p
}

/// `y_f64` for the orders other than 0 and 1.
#[inline(never)]
fn y_rest(nu: f64, x: f64) -> f64 {
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

/// `I_ν(x)` of `f64`s, times `e^(−|x|)` when `scaled` (the dispatch to the orders 0 and 1
/// inlined into the caller).
#[inline]
pub(crate) fn i_f64(nu: f64, x: f64, scaled: bool) -> f64 {
    if nu == 0.0 {
        return ik01::i0(x, scaled);
    }
    if nu == 1.0 {
        return ik01::i1(x, scaled);
    }
    i_rest(nu, x, scaled)
}

/// `i_f64` for the orders other than 0 and 1.
#[inline(never)]
fn i_rest(nu: f64, x: f64, scaled: bool) -> f64 {
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

/// `K_ν(x)` of `f64`s, times `e^x` when `scaled` (the dispatch to the orders 0 and ±1 inlined into
/// the caller).
#[inline]
pub(crate) fn k_f64(nu: f64, x: f64, scaled: bool) -> f64 {
    if nu == 0.0 {
        return ik01::k0(x, scaled);
    }
    if nu.abs() == 1.0 {
        return ik01::k1(x, scaled);
    }
    k_rest(nu, x, scaled)
}

/// `k_f64` for the orders other than 0 and ±1.
#[inline(never)]
fn k_rest(nu: f64, x: f64, scaled: bool) -> f64 {
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
    let (nu, x) = (nu.to_f64(), x.to_f64());
    T::from_f64(if <T as super::sealed::Real>::F32 {
        j_f32(nu, x)
    } else {
        j_f64(nu, x)
    })
}

/// The Bessel function of the second kind `Y_ν(x)`, for `x ≥ 0` (`−∞` at 0).
pub fn bessel_y<T: Real>(nu: T, x: T) -> T {
    let (nu, x) = (nu.to_f64(), x.to_f64());
    T::from_f64(if <T as super::sealed::Real>::F32 {
        y_f32(nu, x)
    } else {
        y_f64(nu, x)
    })
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
