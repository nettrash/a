//! The elliptic integrals, Carlson's symmetric forms and the Jacobi elliptic functions, all of the
//! parameter `m = k²`.
//!
//! * `K(m)` and `E(m)`: fitted polynomials in `m` on intervals of width 0.05 for `−1 ≤ m < 0.8`,
//!   the logarithmic forms `K = −ln(m₁) K(m₁)/π + B(m₁)` and `E = 1 + m₁(−ln(m₁) S₁(m₁) + S₂(m₁))`
//!   (DLMF 19.12.1–2, `m₁ = 1 − m`) for `m ≥ 0.8`, and the imaginary-modulus transformation
//!   (DLMF 19.7.2) for `m < −1`.
//! * Carlson's integrals by the duplication algorithm (`carlson`), and the incomplete integrals
//!   through them, `φ` reduced by the periodicity, with forms whose terms are all positive where
//!   the textbook forms cancel.
//! * The Jacobi functions by the descending Landen transformation of their values (`jacobi`).

mod carlson;
pub(in crate::math::special) mod eft;
mod jacobi;
mod tables;

use super::Real;
use super::dd::Dd;
use core::f64::consts::{FRAC_PI_2, PI};
use eft::{fast_two_sum, fma, mla, two_prod, two_sum};
use tables::{E_HI, E_LO, E_S1, E_S2, K_B, K_HI, K_LO, K_P};

pub(crate) use jacobi::jacobi_f64;

/// A fitted polynomial `[centre, c0 hi, c0 lo, c1, …]` at `t = x − centre`, as a double-double.
#[inline(always)]
fn poly_dd(r: &[f64], t: f64) -> Dd {
    let n = r.len();
    let mut p = r[n - 1];
    let mut k = n - 2;
    while k >= 3 {
        p = mla(p, t, r[k]);
        k -= 1;
    }
    let v = mla(t, p, r[2]);
    let (hi, lo) = fast_two_sum(r[1], v);
    Dd { hi, lo }
}

/// The table row of `m` in `[−1, 0.8)`.
#[inline(always)]
fn row<'a, const A: usize, const B: usize>(
    lo: &'a [[f64; A]; 30],
    hi: &'a [[f64; B]; 6],
    m: f64,
) -> Dd {
    let i = ((m + 1.0) * 20.0) as usize;
    if i < 30 {
        let r = &lo[i];
        poly_dd(r, m - r[0])
    } else {
        let r = &hi[(i - 30).min(5)];
        poly_dd(r, m - r[0])
    }
}

/// `K(m)` for `m₁ = 1 − m` in `(0, 0.2]`, with `l = −ln m₁`: `l·K(m₁)/π + B(m₁)`.
#[inline(always)]
fn k_log(m1: f64, l: f64) -> Dd {
    let kp = poly_dd(&K_P, m1 - K_P[0]);
    let b = poly_dd(&K_B, m1 - K_B[0]);
    let (p, pe) = two_prod(l, kp.hi);
    let (hi, lo) = two_sum(p, b.hi);
    let (hi, lo) = fast_two_sum(hi, lo + (mla(l, kp.lo, pe) + b.lo));
    Dd { hi, lo }
}

/// `E(m)` for `m₁ = 1 − m` in `[0, 0.2]`, with `l = −ln m₁`.
#[inline(always)]
fn e_log(m1: f64, l: f64) -> Dd {
    let s1 = poly_dd(&E_S1, m1 - E_S1[0]).value();
    let s2 = poly_dd(&E_S2, m1 - E_S2[0]).value();
    let inner = mla(l, s1, s2);
    let (p, pe) = two_prod(m1, inner);
    let (hi, lo) = fast_two_sum(1.0, p);
    Dd { hi, lo: lo + pe }
}

/// `K(m)` of a finite `m < 1`, as a double-double (about 0.6 ulp).
pub(super) fn k_dd(m: f64) -> Dd {
    if m >= 0.8 {
        let m1 = 1.0 - m;
        return k_log(m1, -m1.ln());
    }
    if m >= -1.0 {
        return row(&K_LO, &K_HI, m);
    }
    // K(m) = K(m')/√(1 − m), m' = −m/(1 − m), 1 − m' = 1/(1 − m).
    let (dh, dl) = two_sum(1.0, -m);
    let d = Dd { hi: dh, lo: dl };
    let kp = if m > -4.0 {
        row(&K_LO, &K_HI, -m / dh)
    } else {
        k_log(1.0 / dh, (-m).ln_1p())
    };
    kp.div(d.sqrt())
}

/// `E(m)` of a finite `m ≤ 1`, as a double-double.
fn e_dd(m: f64) -> Dd {
    if m >= 0.8 {
        let m1 = 1.0 - m;
        if m1 == 0.0 {
            return Dd::new(1.0);
        }
        return e_log(m1, -m1.ln());
    }
    if m >= -1.0 {
        return row(&E_LO, &E_HI, m);
    }
    // E(m) = √(1 − m) E(m').
    let (dh, dl) = two_sum(1.0, -m);
    let d = Dd { hi: dh, lo: dl };
    let ep = if m > -4.0 {
        row(&E_LO, &E_HI, -m / dh)
    } else {
        e_log(1.0 / dh, (-m).ln_1p())
    };
    ep.mul(d.sqrt())
}

/// `K(m)` of an `f64`.
pub(crate) fn ellip_k_f64(m: f64) -> f64 {
    if m.is_nan() || m > 1.0 {
        return f64::NAN;
    }
    if m == 1.0 {
        return f64::INFINITY;
    }
    if m == f64::NEG_INFINITY {
        return 0.0;
    }
    k_dd(m).value()
}

/// `E(m)` of an `f64`.
pub(crate) fn ellip_e_f64(m: f64) -> f64 {
    if m.is_nan() || m > 1.0 {
        return f64::NAN;
    }
    if m == f64::NEG_INFINITY {
        return f64::INFINITY;
    }
    e_dd(m).value()
}

// ---------------------------------------------------------------------------------------------
// The incomplete integrals.

/// `1 − m s²` for `s, c = sin φ, cos φ`, without cancellation where its terms have one sign.
#[inline(always)]
fn delta2(s: f64, c: f64, m: f64) -> f64 {
    if (0.0..=1.0).contains(&m) {
        // c² + (1 − m) s².
        mla(1.0 - m, s * s, c * c)
    } else {
        let (s2, e2) = two_prod(s, s);
        fma(-m, s2, 1.0) - m * e2
    }
}

/// `φ = jπ + r` with `|r| ≤ π/2`: `(j, sin r, cos r)`, from the sine and cosine of `φ` itself.
#[inline(always)]
fn reduce(phi: f64) -> (f64, f64, f64) {
    let (s, c) = phi.sin_cos();
    if phi.abs() <= FRAC_PI_2 {
        return (0.0, s, c);
    }
    let mut j = (phi / PI).round();
    let odd = j % 2.0 != 0.0;
    let (mut sr, mut cr) = if odd { (-s, -c) } else { (s, c) };
    if cr < 0.0 {
        j += sr.signum();
        sr = -sr;
        cr = -cr;
    }
    (j, sr, cr)
}

/// `F` of `s = sin r ≥ 0`, `c = cos r ≥ 0`.
#[inline(always)]
fn f_kernel(s: f64, c: f64, m: f64) -> f64 {
    s * carlson::rf(c * c, delta2(s, c, m), 1.0)
}

/// `E` of `s = sin r ≥ 0`, `c = cos r ≥ 0`, `m < 1` or `c > 0`.
#[inline(always)]
fn e_kernel(s: f64, c: f64, m: f64) -> f64 {
    let d2 = delta2(s, c, m);
    if m > 0.0 && m < 1.0 && m * s * s > 0.5 {
        // DLMF 19.25.10: three positive terms, where the usual form below would cancel.
        let m1 = 1.0 - m;
        let c2 = c * c;
        let t1 = m1 * s * carlson::rf(c2, d2, 1.0);
        let t2 = (m * m1 / 3.0) * (s * s * s) * carlson::rd(c2, 1.0, d2);
        let t3 = m * s * c / d2.sqrt();
        t1 + t2 + t3
    } else {
        let c2 = c * c;
        s * carlson::rf(c2, d2, 1.0) - (m / 3.0) * (s * s * s) * carlson::rd(c2, d2, 1.0)
    }
}

/// The complementary part `∫₀^ρ dθ/√(1 − m cos²θ)` of `F` with `sin ρ = c`, `cos ρ = s`.
#[inline(always)]
fn f_comp(s: f64, c: f64, m: f64) -> f64 {
    let m1 = 1.0 - m;
    c * carlson::rf(m1 * s * s, delta2(s, c, m), m1)
}

/// The complementary part `∫₀^ρ √(1 − m cos²θ) dθ` of `E` with `sin ρ = c`, `cos ρ = s`, `0 ≤ m < 1`.
#[inline(always)]
fn e_comp(s: f64, c: f64, m: f64) -> f64 {
    let m1 = 1.0 - m;
    let x = m1 * s * s;
    let y = delta2(s, c, m);
    c * m1 * carlson::rf(x, y, m1) + (m1 * m / 3.0) * (c * c * c) * carlson::rd(x, y, m1)
}

/// `F(φ, m)` of `f64`s.
pub(crate) fn ellip_f_f64(phi: f64, m: f64) -> f64 {
    if phi.is_nan() || m.is_nan() {
        return f64::NAN;
    }
    if m > 1.0 {
        if phi.is_infinite() || phi.abs() > FRAC_PI_2 {
            return f64::NAN;
        }
        let (s, c) = phi.sin_cos();
        let d2 = delta2(s, c, m);
        if d2 < 0.0 {
            return f64::NAN;
        }
        return s.signum() * f_kernel(s.abs(), c, m);
    }
    if phi.is_infinite() {
        return if m == f64::NEG_INFINITY {
            f64::NAN
        } else {
            phi
        };
    }
    if m == f64::NEG_INFINITY {
        return 0.0 * phi;
    }
    let (j, sr, cr) = reduce(phi);
    let sg = if sr.is_sign_negative() { -1.0 } else { 1.0 };
    let a = sr.abs();
    if m == 1.0 {
        if j != 0.0 || cr == 0.0 {
            return f64::INFINITY.copysign(phi);
        }
        return atanh_f(sr);
    }
    if j == 0.0 {
        return sg * f_kernel(a, cr, m);
    }
    let k = k_dd(m).value();
    let sj = j.signum();
    if sg != sj && a * a * (1.0 - m).sqrt() > cr * cr {
        // φ = (2j − σ)π/2 + σρ: (2j − σ)K + σ G(ρ), no cancellation.
        return fma(2.0 * j - sj, k, sj * f_comp(a, cr, m));
    }
    fma(2.0 * j, k, sg * f_kernel(a, cr, m))
}

/// `atanh(x)` for `|x| < 1`, sign kept.
#[inline(always)]
fn atanh_f(x: f64) -> f64 {
    let a = x.abs();
    (0.5 * (2.0 * a / (1.0 - a)).ln_1p()).copysign(x)
}

/// `E(φ, m)` of `f64`s.
pub(crate) fn ellip_e_inc_f64(phi: f64, m: f64) -> f64 {
    if phi.is_nan() || m.is_nan() {
        return f64::NAN;
    }
    if m > 1.0 {
        if phi.is_infinite() || phi.abs() > FRAC_PI_2 {
            return f64::NAN;
        }
        let (s, c) = phi.sin_cos();
        if delta2(s, c, m) < 0.0 {
            return f64::NAN;
        }
        return s.signum() * e_kernel(s.abs(), c, m);
    }
    if phi.is_infinite() || m == f64::NEG_INFINITY {
        return if phi == 0.0 {
            phi
        } else {
            f64::INFINITY.copysign(phi)
        };
    }
    let (j, sr, cr) = reduce(phi);
    if m == 1.0 {
        return fma(2.0, j, sr);
    }
    let sg = if sr.is_sign_negative() { -1.0 } else { 1.0 };
    let a = sr.abs();
    if j == 0.0 {
        return sg * e_kernel(a, cr, m);
    }
    let e = e_dd(m).value();
    let sj = j.signum();
    if m >= 0.0 && sg != sj && a > cr {
        return fma(2.0 * j - sj, e, sj * e_comp(a, cr, m));
    }
    fma(2.0 * j, e, sg * e_kernel(a, cr, m))
}

/// `Π(n; r, m)` of `s = sin r ≥ 0`, `c = cos r ≥ 0` and `m ≤ 1` (`c > 0` if `m = 1`): the principal
/// value past a pole.
fn pi_kernel(n: f64, s: f64, c: f64, m: f64) -> f64 {
    if s == 0.0 {
        return s;
    }
    let c2 = c * c;
    let d2 = delta2(s, c, m);
    let s3 = s * s * s;
    if n == 0.0 {
        return s * carlson::rf(c2, d2, 1.0);
    }
    if n > 1.0 {
        // A&S 17.7.7: Π(n) = F − Π(m/n) + ln|(Δ + p₁ tan φ)/(Δ − p₁ tan φ)| / (2p₁).
        let nn = m / n;
        let p1 = ((n - 1.0) * (1.0 - nn)).sqrt();
        let num = p1 * s;
        let den = c * d2.sqrt();
        let lt = if num == den {
            f64::INFINITY
        } else if num < den {
            atanh_pos(num / den)
        } else {
            atanh_pos(den / num)
        };
        // 1 − N s² = c² + (1 − N) s², 1 − N = (n − m)/n.
        let pn = mla((n - m) / n, s * s, c2);
        let j = if nn == 0.0 {
            0.0
        } else {
            (nn / 3.0) * s3 * carlson::rj(c2, d2, 1.0, pn)
        };
        return lt / p1 - j;
    }
    // 1 − n s² = c² + (1 − n) s²: no cancellation for 0 < n ≤ 1, two positive terms for n < 0.
    let p = mla(1.0 - n, s * s, c2);
    if p == 0.0 {
        return f64::INFINITY;
    }
    if n < 0.0 && n < m {
        // Boost's form of A&S 17.7.15 (by R. Romeo): with N = (m − n)/(1 − n),
        // Π(n) = F/(1 − n) + A (N/3) s³ R_J(…, 1 − N s²) + C atan(√(−nN) s c/Δ), all positive.
        let n1 = 1.0 - n;
        let nn = (m - n) / n1;
        let a = (-n / n1) * ((1.0 - m) / (m - n));
        let cc = (-n / ((m - n) * n1)).sqrt();
        let p2 = (-n * nn).sqrt();
        let pn = mla((1.0 - m) / n1, s * s, c2); // 1 − N s², 1 − N = (1 − m)/(1 − n)
        let rf = carlson::rf(c2, d2, 1.0);
        let rj = carlson::rj(c2, d2, 1.0, pn);
        let at = (p2 * s * c / d2.sqrt()).atan();
        return (s * rf / n1 + (a * nn / 3.0) * s3 * rj) + cc * at;
    }
    s * carlson::rf(c2, d2, 1.0) + (n / 3.0) * s3 * carlson::rj(c2, d2, 1.0, p)
}

/// `atanh(v)` for `0 ≤ v ≤ 1`.
#[inline(always)]
fn atanh_pos(v: f64) -> f64 {
    0.5 * (2.0 * v / (1.0 - v)).ln_1p()
}

/// `Π(n, m)` of `f64`s (the principal value for `n > 1`).
pub(crate) fn ellip_pi_f64(n: f64, m: f64) -> f64 {
    if n.is_nan() || m.is_nan() || m > 1.0 {
        return f64::NAN;
    }
    if n == 1.0 {
        return f64::INFINITY;
    }
    if m == 1.0 {
        return if n < 1.0 {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        };
    }
    if n.is_infinite() || m == f64::NEG_INFINITY {
        return 0.0;
    }
    let m1 = 1.0 - m;
    if n == 0.0 {
        return k_dd(m).value();
    }
    if n > 1.0 {
        // DLMF 19.6.5: Π(n, m) = K − Π(m/n, m) = −(N/3) R_J(0, m₁, 1, 1 − N), N = m/n.
        let nn = m / n;
        if nn == 0.0 {
            return 0.0;
        }
        return -(nn / 3.0) * carlson::rj(0.0, m1, 1.0, (n - m) / n);
    }
    let k = k_dd(m).value();
    if n < 0.0 && n < m {
        let n1 = 1.0 - n;
        let nn = (m - n) / n1;
        let a = (-n / n1) * (m1 / (m - n));
        return k / n1 + (a * nn / 3.0) * carlson::rj(0.0, m1, 1.0, m1 / n1);
    }
    k + (n / 3.0) * carlson::rj(0.0, m1, 1.0, 1.0 - n)
}

/// `Π(n; φ, m)` of `f64`s.
pub(crate) fn ellip_pi_inc_f64(n: f64, phi: f64, m: f64) -> f64 {
    if n.is_nan() || phi.is_nan() || m.is_nan() {
        return f64::NAN;
    }
    if m > 1.0 {
        // DLMF 19.7.4: Π(n; φ, m) = Π(n/m; β, 1/m)/√m with sin β = √m sin φ, cos β = Δ.
        if phi.is_infinite() || phi.abs() > FRAC_PI_2 {
            return f64::NAN;
        }
        let (s, c) = phi.sin_cos();
        let d2 = delta2(s, c, m);
        if d2 < 0.0 {
            return f64::NAN;
        }
        let r = m.sqrt();
        let sb = (r * s.abs()).min(1.0);
        let v = pi_kernel(n / m, sb, d2.sqrt(), 1.0 / m) / r;
        return if phi.is_sign_negative() { -v } else { v };
    }
    if phi.is_infinite() || m == f64::NEG_INFINITY {
        if m == f64::NEG_INFINITY {
            return 0.0 * phi;
        }
        let p = ellip_pi_f64(n, m);
        return if p == 0.0 { f64::NAN } else { p * phi };
    }
    let (j, sr, cr) = reduce(phi);
    let sg = if sr.is_sign_negative() { -1.0 } else { 1.0 };
    let a = sr.abs();
    if m == 1.0 && (j != 0.0 || cr == 0.0) {
        return f64::INFINITY.copysign(phi);
    }
    let v = sg * pi_kernel(n, a, cr, m);
    if j == 0.0 {
        return v;
    }
    fma(2.0 * j, ellip_pi_f64(n, m), v)
}

// ---------------------------------------------------------------------------------------------
// Carlson's integrals of `f64`s.

/// `R_F(x, y, z)` of `f64`s.
pub(crate) fn carlson_rf_f64(x: f64, y: f64, z: f64) -> f64 {
    if x.is_nan() || y.is_nan() || z.is_nan() || x < 0.0 || y < 0.0 || z < 0.0 {
        return f64::NAN;
    }
    if x.is_infinite() || y.is_infinite() || z.is_infinite() {
        return 0.0;
    }
    let zeros = (x == 0.0) as u8 + (y == 0.0) as u8 + (z == 0.0) as u8;
    if zeros >= 2 {
        return f64::INFINITY;
    }
    carlson::rf(x, y, z)
}

/// `R_D(x, y, z)` of `f64`s.
pub(crate) fn carlson_rd_f64(x: f64, y: f64, z: f64) -> f64 {
    if x.is_nan() || y.is_nan() || z.is_nan() || x < 0.0 || y < 0.0 || z < 0.0 {
        return f64::NAN;
    }
    if z == 0.0 || (x == 0.0 && y == 0.0) {
        return f64::INFINITY;
    }
    if x.is_infinite() || y.is_infinite() || z.is_infinite() {
        return 0.0;
    }
    carlson::rd(x, y, z)
}

/// `R_J(x, y, z, p)` of `f64`s.
pub(crate) fn carlson_rj_f64(x: f64, y: f64, z: f64, p: f64) -> f64 {
    if x.is_nan() || y.is_nan() || z.is_nan() || p.is_nan() || x < 0.0 || y < 0.0 || z < 0.0 {
        return f64::NAN;
    }
    let zeros = (x == 0.0) as u8 + (y == 0.0) as u8 + (z == 0.0) as u8;
    if p == 0.0 || zeros >= 2 {
        return f64::INFINITY;
    }
    if x.is_infinite() || y.is_infinite() || z.is_infinite() || p.is_infinite() {
        return 0.0;
    }
    if p < 0.0 {
        return carlson::rj_pv(x, y, z, p);
    }
    carlson::rj(x, y, z, p)
}

/// `R_C(x, y)` of `f64`s.
pub(crate) fn carlson_rc_f64(x: f64, y: f64) -> f64 {
    if x.is_nan() || y.is_nan() || x < 0.0 {
        return f64::NAN;
    }
    if y == 0.0 {
        return f64::INFINITY;
    }
    if x.is_infinite() || y.is_infinite() {
        return 0.0;
    }
    carlson::rc(x, y)
}

// ---------------------------------------------------------------------------------------------
// The public functions.

#[inline(always)]
fn via2<T: Real>(a: T, b: T, f: impl FnOnce(f64, f64) -> f64) -> T {
    T::from_f64(f(a.to_f64(), b.to_f64()))
}

#[inline(always)]
fn via3<T: Real>(a: T, b: T, c: T, f: impl FnOnce(f64, f64, f64) -> f64) -> T {
    T::from_f64(f(a.to_f64(), b.to_f64(), c.to_f64()))
}

/// The complete elliptic integral of the first kind `K(m) = ∫₀^(π/2) dθ / √(1 − m sin²θ)`, for
/// `m < 1` (`∞` at 1).
pub fn ellip_k<T: Real>(m: T) -> T {
    super::via(m, ellip_k_f64)
}

/// The complete elliptic integral of the second kind `E(m) = ∫₀^(π/2) √(1 − m sin²θ) dθ`, for
/// `m ≤ 1`.
pub fn ellip_e<T: Real>(m: T) -> T {
    super::via(m, ellip_e_f64)
}

/// The complete elliptic integral of the third kind
/// `Π(n, m) = ∫₀^(π/2) dθ / ((1 − n sin²θ) √(1 − m sin²θ))` (its Cauchy principal value for
/// `n > 1`).
pub fn ellip_pi<T: Real>(n: T, m: T) -> T {
    via2(n, m, ellip_pi_f64)
}

/// The incomplete elliptic integral of the first kind `F(φ, m) = ∫₀^φ dθ / √(1 − m sin²θ)`.
pub fn ellip_f<T: Real>(phi: T, m: T) -> T {
    via2(phi, m, ellip_f_f64)
}

/// The incomplete elliptic integral of the second kind `E(φ, m) = ∫₀^φ √(1 − m sin²θ) dθ`.
pub fn ellip_e_inc<T: Real>(phi: T, m: T) -> T {
    via2(phi, m, ellip_e_inc_f64)
}

/// The incomplete elliptic integral of the third kind
/// `Π(n; φ, m) = ∫₀^φ dθ / ((1 − n sin²θ) √(1 − m sin²θ))`.
pub fn ellip_pi_inc<T: Real>(n: T, phi: T, m: T) -> T {
    via3(n, phi, m, ellip_pi_inc_f64)
}

/// Carlson's symmetric integral `R_F(x, y, z) = ½ ∫₀^∞ dt / √((t + x)(t + y)(t + z))`.
pub fn carlson_rf<T: Real>(x: T, y: T, z: T) -> T {
    via3(x, y, z, carlson_rf_f64)
}

/// Carlson's symmetric integral `R_D(x, y, z) = (3/2) ∫₀^∞ dt / ((t + z) √((t + x)(t + y)(t + z)))`.
pub fn carlson_rd<T: Real>(x: T, y: T, z: T) -> T {
    via3(x, y, z, carlson_rd_f64)
}

/// Carlson's symmetric integral
/// `R_J(x, y, z, p) = (3/2) ∫₀^∞ dt / ((t + p) √((t + x)(t + y)(t + z)))` (its Cauchy principal
/// value for `p < 0`).
pub fn carlson_rj<T: Real>(x: T, y: T, z: T, p: T) -> T {
    T::from_f64(carlson_rj_f64(
        x.to_f64(),
        y.to_f64(),
        z.to_f64(),
        p.to_f64(),
    ))
}

/// Carlson's degenerate integral `R_C(x, y) = R_F(x, y, y)` (its Cauchy principal value for
/// `y < 0`).
pub fn carlson_rc<T: Real>(x: T, y: T) -> T {
    via2(x, y, carlson_rc_f64)
}

/// The Jacobi elliptic functions `(sn(u, m), cn(u, m), dn(u, m))`.
pub fn jacobi_elliptic<T: Real>(u: T, m: T) -> (T, T, T) {
    let (s, c, d) = jacobi_f64(u.to_f64(), m.to_f64());
    (T::from_f64(s), T::from_f64(c), T::from_f64(d))
}
