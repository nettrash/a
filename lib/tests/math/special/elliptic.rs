//! The elliptic area of issue #20: its data in `data/elliptic_*.txt`, made by `data/gen_elliptic.py`.
//!
//! The elliptic integrals and Carlson's integrals are checked in ulps of the result. The Jacobi
//! functions and the orthogonal polynomials are checked in ulps of `max(|f|, S)`, `S` the scale of
//! the function (1 for `sn` and `cn`; for the polynomials the largest `|P|` on the interval of
//! orthogonality, or the weighted largest value inside the oscillatory region of Hermite and
//! Laguerre, carried by the data): relative error away from the zeros, an absolute error relative
//! to the function's size near them, where relative error is unbounded.

use super::common::*;
use a::math::special::{
    assoc_laguerre_l, assoc_legendre_p, carlson_rc, carlson_rd, carlson_rf, carlson_rj,
    chebyshev_t, chebyshev_u, ellip_e, ellip_e_inc, ellip_f, ellip_k, ellip_pi, ellip_pi_inc,
    gegenbauer_c, hermite_h, hermite_he, jacobi_elliptic, jacobi_p, laguerre_l, legendre_p,
};

/// Under Miri every 37th case of a file.
fn thin(text: &str) -> String {
    if !cfg!(miri) {
        return text.to_string();
    }
    let mut out = String::new();
    let mut k = 0;
    for l in text.lines() {
        if l.starts_with('#') || l.trim().is_empty() {
            continue;
        }
        if k % 37 == 0 {
            out.push_str(l);
            out.push('\n');
        }
        k += 1;
    }
    out
}

/// The bound under Miri, which perturbs the results of the elementary functions (`sin`, `ln`, …) by
/// a few ulps on purpose.
fn miri(bound: f64) -> f64 {
    if cfg!(miri) {
        4.0 * bound + 16.0
    } else {
        bound
    }
}

#[track_caller]
fn real(name: &str, text: &str, p: Prec, bound: f64, f: impl Fn(&[f64]) -> f64) {
    let bound = miri(bound);
    let e = check_real(name, &thin(text), p, bound, f);
    eprintln!("{name}: {e:.3} ulps (bound {bound})");
}

/// The error of `y` in ulps of `max(|ref|, scale)`.
fn scaled_ulps(y: f64, r: (f64, f64), scale: f64, p: Prec) -> f64 {
    if !r.0.is_finite() || !y.is_finite() || scale == 0.0 || r.0.abs() >= scale {
        return ulps(y, r, p);
    }
    ((y - r.0) - r.1).abs() / ulp(scale, p)
}

/// Checks a function whose data carry a scale as their second reference.
#[track_caller]
fn scaled(name: &str, text: &str, p: Prec, bound: f64, f: impl Fn(&[f64]) -> f64) {
    let bound = miri(bound);
    let cases = cases(&thin(text));
    assert!(!cases.is_empty(), "{name}: no cases");
    let w = worst(
        &cases,
        |a| vec![f(a)],
        |g, r| scaled_ulps(g[0], r[0], r[1].0, p),
    );
    assert!(
        w.max <= bound,
        "{name}: {:.3} scaled ulps > {bound} at {:?}: got {:e}, want {:e} (scale {:e})",
        w.max,
        w.args,
        w.got[0],
        w.want[0].0,
        w.want[1].0
    );
    eprintln!("{name}: {:.3} scaled ulps (bound {bound})", w.max);
}

const K_F64: &str = include_str!("data/elliptic_k_f64.txt");
const K_F32: &str = include_str!("data/elliptic_k_f32.txt");
const E_F64: &str = include_str!("data/elliptic_e_f64.txt");
const E_F32: &str = include_str!("data/elliptic_e_f32.txt");
const PI_F64: &str = include_str!("data/elliptic_pi_f64.txt");
const PI_F32: &str = include_str!("data/elliptic_pi_f32.txt");
const F_F64: &str = include_str!("data/elliptic_f_f64.txt");
const F_F32: &str = include_str!("data/elliptic_f_f32.txt");
const EINC_F64: &str = include_str!("data/elliptic_einc_f64.txt");
const EINC_F32: &str = include_str!("data/elliptic_einc_f32.txt");
const PIINC_F64: &str = include_str!("data/elliptic_piinc_f64.txt");
const PIINC_F32: &str = include_str!("data/elliptic_piinc_f32.txt");
const RF_F64: &str = include_str!("data/elliptic_rf_f64.txt");
const RF_F32: &str = include_str!("data/elliptic_rf_f32.txt");
const RD_F64: &str = include_str!("data/elliptic_rd_f64.txt");
const RD_F32: &str = include_str!("data/elliptic_rd_f32.txt");
const RJ_F64: &str = include_str!("data/elliptic_rj_f64.txt");
const RJ_F32: &str = include_str!("data/elliptic_rj_f32.txt");
const RC_F64: &str = include_str!("data/elliptic_rc_f64.txt");
const RC_F32: &str = include_str!("data/elliptic_rc_f32.txt");
const JAC_F64: &str = include_str!("data/elliptic_jacobi_f64.txt");
const JAC_F32: &str = include_str!("data/elliptic_jacobi_f32.txt");

const B_K: f64 = 1.0;
const B_E: f64 = 0.75;
const B_PI: f64 = 9.0; // 3 but where m ≤ n < 0, m < −1 (the direct form cancels)
const B_F: f64 = 2.5;
const B_EINC: f64 = 4.0;
const B_PIINC: f64 = 4.0;
const B_RF: f64 = 0.75;
const B_RD: f64 = 0.75;
const B_RJ: f64 = 2.0;
const B_RC: f64 = 2.5;
const B_RJ_PV: f64 = 4.0;

const B_F32: f64 = 0.501;

/// `Π(n; φ, m)` in ulps divided by `1 + κ`, `κ = |φ ∂Π/∂φ / Π| = |φ| / (|1 − n sin²φ| Δ |Π|)` its
/// condition number in `φ`, which is large near the poles `n sin²φ = 1` and the zeros of the
/// principal value.
fn pi_inc_conditioned(text: &str) {
    let cases = cases(&thin(text));
    let mut w = (0.0f64, vec![]);
    for c in &cases {
        let (n, phi, m) = (c.args[0], c.args[1], c.args[2]);
        let got = ellip_pi_inc(n, phi, m);
        let e = ulps(got, c.refs[0], F64);
        let s2 = phi.sin().powi(2);
        let kappa =
            phi.abs() / ((1.0 - n * s2).abs() * (1.0 - m * s2).abs().sqrt() * c.refs[0].0.abs());
        let r = e / (1.0 + kappa);
        if r > w.0 || r.is_nan() {
            w = (r, c.args.clone());
        }
    }
    let bound = miri(B_PIINC);
    assert!(
        w.0 <= bound,
        "ellip_pi_inc f64: {:.3} ulps/(1 + κ) > {bound} at {:?}",
        w.0,
        w.1
    );
    eprintln!("ellip_pi_inc f64: {:.3} ulps/(1 + κ) (bound {bound})", w.0);
}

/// `R_J` with `p > 0` in ulps; its principal value (`p < 0`), which changes sign, in ulps of
/// `max(|R_J|, R_J(x, y, z, |p|))`.
fn rj_split(text: &str) {
    let cases = cases(&thin(text));
    let (mut pos, mut pv) = (0.0f64, 0.0f64);
    for c in &cases {
        let a = &c.args;
        let got = carlson_rj(a[0], a[1], a[2], a[3]);
        if a[3] > 0.0 {
            pos = pos.max(ulps(got, c.refs[0], F64));
        } else {
            let s = carlson_rj(a[0], a[1], a[2], -a[3]);
            pv = pv.max(scaled_ulps(got, c.refs[0], s, F64));
        }
    }
    let (b1, b2) = (miri(B_RJ), miri(B_RJ_PV));
    assert!(pos <= b1, "carlson_rj f64 (p > 0): {pos:.3} ulps > {b1}");
    assert!(
        pv <= b2,
        "carlson_rj f64 (p < 0): {pv:.3} scaled ulps > {b2}"
    );
    eprintln!("carlson_rj f64: p > 0 {pos:.3} ulps, p < 0 {pv:.3} scaled ulps");
}

#[test]
fn complete_integrals() {
    real("ellip_k f64", K_F64, F64, B_K, |a| ellip_k(a[0]));
    real("ellip_e f64", E_F64, F64, B_E, |a| ellip_e(a[0]));
    real("ellip_pi f64", PI_F64, F64, B_PI, |a| ellip_pi(a[0], a[1]));
    real("ellip_k f32", K_F32, F32, B_F32, |a| {
        ellip_k(a[0] as f32) as f64
    });
    real("ellip_e f32", E_F32, F32, B_F32, |a| {
        ellip_e(a[0] as f32) as f64
    });
    real("ellip_pi f32", PI_F32, F32, B_F32, |a| {
        ellip_pi(a[0] as f32, a[1] as f32) as f64
    });
}

#[test]
fn incomplete_integrals() {
    real("ellip_f f64", F_F64, F64, B_F, |a| ellip_f(a[0], a[1]));
    real("ellip_e_inc f64", EINC_F64, F64, B_EINC, |a| {
        ellip_e_inc(a[0], a[1])
    });
    pi_inc_conditioned(PIINC_F64);
    real("ellip_f f32", F_F32, F32, B_F32, |a| {
        ellip_f(a[0] as f32, a[1] as f32) as f64
    });
    real("ellip_e_inc f32", EINC_F32, F32, B_F32, |a| {
        ellip_e_inc(a[0] as f32, a[1] as f32) as f64
    });
    real("ellip_pi_inc f32", PIINC_F32, F32, B_F32, |a| {
        ellip_pi_inc(a[0] as f32, a[1] as f32, a[2] as f32) as f64
    });
}

#[test]
fn carlson_integrals() {
    real("carlson_rf f64", RF_F64, F64, B_RF, |a| {
        carlson_rf(a[0], a[1], a[2])
    });
    real("carlson_rd f64", RD_F64, F64, B_RD, |a| {
        carlson_rd(a[0], a[1], a[2])
    });
    rj_split(RJ_F64);
    real("carlson_rc f64", RC_F64, F64, B_RC, |a| {
        carlson_rc(a[0], a[1])
    });
    let s = |x: f64| x as f32;
    real("carlson_rf f32", RF_F32, F32, B_F32, |a| {
        carlson_rf(s(a[0]), s(a[1]), s(a[2])) as f64
    });
    real("carlson_rd f32", RD_F32, F32, B_F32, |a| {
        carlson_rd(s(a[0]), s(a[1]), s(a[2])) as f64
    });
    real("carlson_rj f32", RJ_F32, F32, B_F32, |a| {
        carlson_rj(s(a[0]), s(a[1]), s(a[2]), s(a[3])) as f64
    });
    real("carlson_rc f32", RC_F32, F32, B_F32, |a| {
        carlson_rc(s(a[0]), s(a[1])) as f64
    });
}

/// The scales of `sn`, `cn` and `dn` of the parameter `m`: 1 where the function has zeros
/// (`1/√m` for `sn` when `m > 1`), 0 (relative error) where it has none.
fn jacobi_scales(m: f64) -> [f64; 3] {
    if m > 1.0 {
        [1.0 / m.sqrt(), 0.0, 1.0]
    } else {
        [1.0, 1.0, 0.0]
    }
}

/// The bounds of the Jacobi functions by region of `m` (`f64`): they grow as `m → 1`, where `cn`
/// and `dn` are both about `m₁^(1/4)` near `u = K/2 (mod K)` and the condition number there is
/// about `K/2`.
fn jacobi_bound(m: f64) -> f64 {
    if m < 0.0 {
        12.0 // |m| ≤ 10⁶; the parameter μ = −m/(1 − m) of the transformation near 1
    } else if m < 0.9 {
        3.5
    } else if m < 1.0 - 1e-6 {
        20.0
    } else if m < 1.0 {
        80.0
    } else {
        4.0
    }
}

fn check_jacobi(
    name: &str,
    text: &str,
    p: Prec,
    bound: impl Fn(f64) -> f64,
    f: impl Fn(f64, f64) -> [f64; 3],
) {
    let cases = cases(&thin(text));
    assert!(!cases.is_empty(), "{name}: no cases");
    let mut maxes = [0.0f64; 3];
    let mut at = [[0.0f64; 2]; 3];
    for c in &cases {
        let g = f(c.args[0], c.args[1]);
        let sc = jacobi_scales(c.args[1]);
        let b = miri(bound(c.args[1]));
        for k in 0..3 {
            let e = scaled_ulps(g[k], c.refs[k], sc[k], p);
            assert!(
                e <= b,
                "{name} {}: {e:.3} scaled ulps > {b} at {:?}",
                ["sn", "cn", "dn"][k],
                c.args
            );
            if e > maxes[k] {
                maxes[k] = e;
                at[k] = [c.args[0], c.args[1]];
            }
        }
    }
    eprintln!(
        "{name}: sn {:.3}, cn {:.3}, dn {:.3} scaled ulps (largest at {:?})",
        maxes[0], maxes[1], maxes[2], at
    );
}

#[test]
fn jacobi_functions() {
    check_jacobi("jacobi_elliptic f64", JAC_F64, F64, jacobi_bound, |u, m| {
        let (s, c, d) = jacobi_elliptic(u, m);
        [s, c, d]
    });
    check_jacobi(
        "jacobi_elliptic f32",
        JAC_F32,
        F32,
        |_| B_F32,
        |u, m| {
            let (s, c, d) = jacobi_elliptic(u as f32, m as f32);
            [s as f64, c as f64, d as f64]
        },
    );
}

const LEG_F64: &str = include_str!("data/elliptic_legendre_f64.txt");
const LEG_F32: &str = include_str!("data/elliptic_legendre_f32.txt");
const ALEG_F64: &str = include_str!("data/elliptic_assoc_legendre_f64.txt");
const ALEG_F32: &str = include_str!("data/elliptic_assoc_legendre_f32.txt");
const HH_F64: &str = include_str!("data/elliptic_hermite_h_f64.txt");
const HH_F32: &str = include_str!("data/elliptic_hermite_h_f32.txt");
const HE_F64: &str = include_str!("data/elliptic_hermite_he_f64.txt");
const HE_F32: &str = include_str!("data/elliptic_hermite_he_f32.txt");
const LAG_F64: &str = include_str!("data/elliptic_laguerre_f64.txt");
const LAG_F32: &str = include_str!("data/elliptic_laguerre_f32.txt");
const ALAG_F64: &str = include_str!("data/elliptic_assoc_laguerre_f64.txt");
const ALAG_F32: &str = include_str!("data/elliptic_assoc_laguerre_f32.txt");
const CT_F64: &str = include_str!("data/elliptic_chebyshev_t_f64.txt");
const CT_F32: &str = include_str!("data/elliptic_chebyshev_t_f32.txt");
const CU_F64: &str = include_str!("data/elliptic_chebyshev_u_f64.txt");
const CU_F32: &str = include_str!("data/elliptic_chebyshev_u_f32.txt");
const JP_F64: &str = include_str!("data/elliptic_jacobi_p_f64.txt");
const JP_F32: &str = include_str!("data/elliptic_jacobi_p_f32.txt");
const GG_F64: &str = include_str!("data/elliptic_gegenbauer_f64.txt");
const GG_F32: &str = include_str!("data/elliptic_gegenbauer_f32.txt");

const B_POLY: f64 = 0.55;

#[test]
fn orthogonal_polynomials() {
    let n = |a: &[f64]| a[0] as u32;
    let s = |x: f64| x as f32;
    scaled("legendre_p f64", LEG_F64, F64, B_POLY, |a| {
        legendre_p(n(a), a[1])
    });
    scaled("assoc_legendre_p f64", ALEG_F64, F64, B_POLY, |a| {
        assoc_legendre_p(n(a), a[1] as u32, a[2])
    });
    scaled("hermite_h f64", HH_F64, F64, B_POLY, |a| {
        hermite_h(n(a), a[1])
    });
    scaled("hermite_he f64", HE_F64, F64, B_POLY, |a| {
        hermite_he(n(a), a[1])
    });
    scaled("laguerre_l f64", LAG_F64, F64, B_POLY, |a| {
        laguerre_l(n(a), a[1])
    });
    scaled("assoc_laguerre_l f64", ALAG_F64, F64, B_POLY, |a| {
        assoc_laguerre_l(n(a), a[1], a[2])
    });
    scaled("chebyshev_t f64", CT_F64, F64, B_POLY, |a| {
        chebyshev_t(n(a), a[1])
    });
    scaled("chebyshev_u f64", CU_F64, F64, B_POLY, |a| {
        chebyshev_u(n(a), a[1])
    });
    scaled("jacobi_p f64", JP_F64, F64, B_POLY, |a| {
        jacobi_p(n(a), a[1], a[2], a[3])
    });
    scaled("gegenbauer_c f64", GG_F64, F64, B_POLY, |a| {
        gegenbauer_c(n(a), a[1], a[2])
    });
    scaled("legendre_p f32", LEG_F32, F32, B_F32, |a| {
        legendre_p(n(a), s(a[1])) as f64
    });
    scaled("assoc_legendre_p f32", ALEG_F32, F32, B_F32, |a| {
        assoc_legendre_p(n(a), a[1] as u32, s(a[2])) as f64
    });
    scaled("hermite_h f32", HH_F32, F32, B_F32, |a| {
        hermite_h(n(a), s(a[1])) as f64
    });
    scaled("hermite_he f32", HE_F32, F32, B_F32, |a| {
        hermite_he(n(a), s(a[1])) as f64
    });
    scaled("laguerre_l f32", LAG_F32, F32, B_F32, |a| {
        laguerre_l(n(a), s(a[1])) as f64
    });
    scaled("assoc_laguerre_l f32", ALAG_F32, F32, B_F32, |a| {
        assoc_laguerre_l(n(a), s(a[1]), s(a[2])) as f64
    });
    scaled("chebyshev_t f32", CT_F32, F32, B_F32, |a| {
        chebyshev_t(n(a), s(a[1])) as f64
    });
    scaled("chebyshev_u f32", CU_F32, F32, B_F32, |a| {
        chebyshev_u(n(a), s(a[1])) as f64
    });
    scaled("jacobi_p f32", JP_F32, F32, B_F32, |a| {
        jacobi_p(n(a), s(a[1]), s(a[2]), s(a[3])) as f64
    });
    scaled("gegenbauer_c f32", GG_F32, F32, B_F32, |a| {
        gegenbauer_c(n(a), s(a[1]), s(a[2])) as f64
    });
}

// ---------------------------------------------------------------------------------------------
// Special values, identities and the edges of the domains.

/// `a == b`, or nearly under Miri (whose elementary functions are not reproducible).
fn eq(a: f64, b: f64) -> bool {
    if cfg!(miri) {
        close(a, b, 1e-14)
    } else {
        a == b
    }
}

/// An absolute tolerance, wider under Miri.
fn tol(t: f64) -> f64 {
    if cfg!(miri) { 64.0 * t } else { t }
}

fn close(a: f64, b: f64, tol: f64) -> bool {
    let tol = if cfg!(miri) { 64.0 * tol } else { tol };
    a == b || (a - b).abs() <= tol * b.abs().max(f64::MIN_POSITIVE)
}

#[test]
fn complete_special_values() {
    use core::f64::consts::{FRAC_PI_2, PI};
    assert_eq!(ellip_k(0.0f64), FRAC_PI_2);
    assert_eq!(ellip_e(0.0f64), FRAC_PI_2);
    assert_eq!(ellip_k(1.0f64), f64::INFINITY);
    assert_eq!(ellip_e(1.0f64), 1.0);
    assert!(ellip_k(1.5f64).is_nan() && ellip_e(1.5f64).is_nan());
    assert!(ellip_k(f64::NAN).is_nan() && ellip_e(f64::NAN).is_nan());
    assert_eq!(ellip_k(f64::NEG_INFINITY), 0.0);
    assert_eq!(ellip_e(f64::NEG_INFINITY), f64::INFINITY);
    assert_eq!(ellip_k(0.0f32), core::f32::consts::FRAC_PI_2);
    // Legendre's relation E K' + E' K − K K' = π/2.
    for &m in &[1e-12, 1e-9, 0.1, 0.3, 0.5, 0.7, 0.9, 0.999_999] {
        let (k, e, k1, e1) = (ellip_k(m), ellip_e(m), ellip_k(1.0 - m), ellip_e(1.0 - m));
        let l = e * k1 + e1 * k - k * k1;
        assert!(close(l, FRAC_PI_2, 1e-14), "m = {m}: {l}");
    }
    // Π(0, m) = K(m), Π(n, 0) = π/(2√(1 − n)), Π(m, m) = E(m)/(1 − m), Π(1, m) = ∞.
    for &m in &[-3.0, 0.0, 0.25, 0.9] {
        assert!(close(ellip_pi(0.0, m), ellip_k(m), 4e-16));
        assert!(close(ellip_pi(m, m), ellip_e(m) / (1.0 - m), 1e-15));
    }
    for &n in &[-5.0, -0.5, 0.5, 0.99] {
        assert!(close(
            ellip_pi(n, 0.0f64),
            PI / (2.0 * (1.0 - n).sqrt()),
            1e-15
        ));
    }
    assert_eq!(ellip_pi(1.0f64, 0.5), f64::INFINITY);
    assert_eq!(ellip_pi(2.0f64, 0.0), 0.0); // the principal value
    assert!(ellip_pi(0.5f64, 1.5).is_nan() && ellip_pi(f64::NAN, 0.5).is_nan());
}

#[test]
fn incomplete_special_values() {
    use core::f64::consts::{FRAC_PI_2, PI};
    for &phi in &[0.0, 1e-300, 0.3, 1.0, 1.5, 3.0, -7.0, 100.0] {
        assert!(close(ellip_f(phi, 0.0f64), phi, 4e-16), "{phi}");
        assert!(close(ellip_e_inc(phi, 0.0f64), phi, 4e-16), "{phi}");
    }
    assert!(ellip_f(-0.0f64, 0.5).is_sign_negative());
    for &m in &[-10.0, 0.0, 0.3, 0.99] {
        let (k, e) = (ellip_k(m), ellip_e(m));
        assert!(close(ellip_f(FRAC_PI_2, m), k, 1e-15));
        assert!(close(ellip_e_inc(FRAC_PI_2, m), e, 1e-15));
        for &phi in &[0.4, 1.2] {
            let (f, ei) = (ellip_f(phi, m), ellip_e_inc(phi, m));
            assert!(eq(ellip_f(-phi, m), -f));
            assert!(close(ellip_f(phi + PI, m), f + 2.0 * k, 1e-15));
            assert!(close(ellip_e_inc(phi - 3.0 * PI, m), ei - 6.0 * e, 1e-14));
            assert!(close(ellip_pi_inc(0.0, phi, m), f, 1e-15));
        }
    }
    // m = 1: F = atanh(sin φ), E = sin φ.
    assert!(close(ellip_f(1.0f64, 1.0), 1.0f64.sin().atanh(), 1e-15));
    assert_eq!(ellip_f(FRAC_PI_2 + 0.1, 1.0f64), f64::INFINITY);
    assert!(close(ellip_e_inc(1.0f64, 1.0), 1.0f64.sin(), 1e-16));
    // m > 1: real only for |sin φ| ≤ 1/√m.
    assert!(ellip_f(1.0f64, 4.0).is_nan());
    assert!(ellip_f(0.5f64, 4.0).is_finite());
    assert!(ellip_e_inc(0.6f64, 4.0).is_nan());
    // Π(n; φ, 0) = atan(√(1 − n) tan φ)/√(1 − n) for n < 1.
    for &n in &[-4.0f64, 0.5] {
        let r = (1.0 - n).sqrt();
        assert!(close(
            ellip_pi_inc(n, 1.0, 0.0),
            (r * 1.0f64.tan()).atan() / r,
            1e-15
        ));
    }
    // Next to the pole n sin²φ = 1 (FRAC_PI_4 is just below π/4): large, the principal value.
    assert!(ellip_pi_inc(2.0f64, core::f64::consts::FRAC_PI_4, 0.3) > 15.0);
    assert!(ellip_pi_inc(1.0f64, FRAC_PI_2, 0.3) > 1e15); // cos(FRAC_PI_2) ≈ 6e−17
    assert_eq!(ellip_pi_inc(1.0f64, FRAC_PI_2 + 3.0, 0.3), f64::INFINITY);
    assert!(ellip_f(f64::NAN, 0.5).is_nan() && ellip_pi_inc(0.5, 1.0, f64::NAN).is_nan());
    assert_eq!(ellip_f(f64::INFINITY, 0.5), f64::INFINITY);
}

#[test]
fn carlson_special_values() {
    use core::f64::consts::FRAC_PI_2;
    for &x in &[1e-300f64, 0.25, 3.0, 1e300] {
        let r = 1.0 / x.sqrt();
        assert!(close(carlson_rf(x, x, x), r, 3e-16));
        assert!(close(carlson_rc(x, x), r, 3e-16));
        assert!(close(carlson_rd(x, x, x), r / x, 3e-16));
        assert!(close(carlson_rj(x, x, x, x), r / x, 3e-16));
    }
    assert!(close(carlson_rf(0.0, 1.0, 1.0), FRAC_PI_2, 3e-16));
    assert!(close(carlson_rc(0.0, 1.0), FRAC_PI_2, 3e-16));
    let (x, y, z) = (0.3, 2.0, 7.5);
    let f = carlson_rf(x, y, z);
    for (a, b, c) in [(y, x, z), (z, y, x), (x, z, y)] {
        assert!(close(carlson_rf(a, b, c), f, 3e-16));
    }
    assert!(close(carlson_rj(x, y, z, z), carlson_rd(x, y, z), 5e-16));
    assert!(close(carlson_rd(y, x, z), carlson_rd(x, y, z), 3e-16));
    // Homogeneity.
    assert!(close(carlson_rf(4.0 * x, 4.0 * y, 4.0 * z), f / 2.0, 3e-16));
    // R_C(x, y) = R_F(x, y, y).
    assert!(close(
        carlson_rc(0.3, 2.0),
        carlson_rf(0.3, 2.0, 2.0),
        5e-16
    ));
    assert!(carlson_rf(-1.0f64, 1.0, 1.0).is_nan() && carlson_rc(-1.0f64, 1.0).is_nan());
    assert_eq!(carlson_rf(0.0, 0.0, 1.0), f64::INFINITY);
    assert_eq!(carlson_rd(1.0, 1.0, 0.0), f64::INFINITY);
    assert_eq!(carlson_rj(1.0, 1.0, 1.0, 0.0), f64::INFINITY);
    assert_eq!(carlson_rc(1.0, 0.0), f64::INFINITY);
    assert_eq!(carlson_rf(1.0, 1.0, f64::INFINITY), 0.0);
    assert!(carlson_rj(1.0f64, 2.0, 3.0, -0.5).is_finite());
    assert!(carlson_rf(f64::NAN, 1.0, 1.0).is_nan());
    assert!((carlson_rf(1.0f32, 2.0, 3.0) as f64 - carlson_rf(1.0, 2.0, 3.0)).abs() < 1e-7);
}

#[test]
fn jacobi_identities() {
    assert_eq!(jacobi_elliptic(0.0f64, 0.5), (0.0, 1.0, 1.0));
    let (s, c, d) = jacobi_elliptic(0.7f64, 0.0);
    assert!(eq(s, 0.7f64.sin()) && eq(c, 0.7f64.cos()) && d == 1.0);
    let (s, c, d) = jacobi_elliptic(0.7f64, 1.0);
    assert!(close(s, 0.7f64.tanh(), 2e-16) && close(c, 1.0 / 0.7f64.cosh(), 2e-16) && c == d);
    for &m in &[-20.0f64, -0.5, 1e-12, 0.3, 0.8, 0.999_999, 3.0, 50.0] {
        for &u in &[-9.0f64, -0.3, 0.01, 0.6, 2.5, 40.0, 1e5] {
            let (s, c, d) = jacobi_elliptic(u, m);
            assert!((s * s + c * c - 1.0).abs() < tol(1e-14), "u {u} m {m}");
            assert!(
                (d * d + m * s * s - 1.0).abs() < tol(1e-13) * m.abs().max(1.0),
                "u {u} m {m}"
            );
            let (s2, c2, d2) = jacobi_elliptic(-u, m);
            assert!(eq(s2, -s) && eq(c2, c) && eq(d2, d));
        }
        if (0.0..1.0).contains(&m) {
            let k = ellip_k(m);
            let (s, c, d) = jacobi_elliptic(k, m);
            assert!(
                close(s, 1.0, 1e-15) && c.abs() < tol(1e-15) && close(d, (1.0 - m).sqrt(), 1e-12)
            );
            let (s0, c0, d0) = jacobi_elliptic(0.4, m);
            let (s1, c1, d1) = jacobi_elliptic(0.4 + 4.0 * k, m);
            let t = tol(1e-14);
            assert!((s1 - s0).abs() < t && (c1 - c0).abs() < t && (d1 - d0).abs() < t);
        }
    }
    let (s, c, d) = jacobi_elliptic(f64::NAN, 0.5);
    assert!(s.is_nan() && c.is_nan() && d.is_nan());
    assert!(jacobi_elliptic(f64::INFINITY, 0.5).0.is_nan());
    let (s, _, _) = jacobi_elliptic(0.5f32, 0.3);
    assert!((s as f64 - jacobi_elliptic(0.5f64, 0.3).0).abs() < 1e-7);
}

#[test]
fn polynomial_identities() {
    for n in 0..40u32 {
        assert_eq!(legendre_p(n, 1.0f64), 1.0);
        assert_eq!(legendre_p(n, -1.0f64), if n % 2 == 0 { 1.0 } else { -1.0 });
        assert_eq!(chebyshev_t(n, 1.0f64), 1.0);
        assert_eq!(chebyshev_u(n, 1.0f64), (n + 1) as f64);
        assert_eq!(laguerre_l(n, 0.0f64), 1.0);
        let x = 0.3f64;
        let th = x.acos();
        assert!((chebyshev_t(n, x) - (n as f64 * th).cos()).abs() < tol(1e-14));
        assert!((jacobi_p(n, 0.0, 0.0, x) - legendre_p(n, x)).abs() < 1e-15);
        assert!((gegenbauer_c(n, 0.5, x) - legendre_p(n, x)).abs() < 1e-15);
        assert!((gegenbauer_c(n, 1.0, x) - chebyshev_u(n, x)).abs() < 1e-14 * (n + 1) as f64);
        assert_eq!(assoc_legendre_p(n, n + 1, x), 0.0);
        assert!((assoc_legendre_p(n, 0, x) - legendre_p(n, x)).abs() < 1e-16);
        // He_n(x) = 2^(−n/2) H_n(x/√2).
        let h = hermite_h(n, x / 2f64.sqrt()) * 2f64.powf(-(n as f64) / 2.0);
        assert!((hermite_he(n, x) - h).abs() <= 1e-13 * h.abs().max(1.0));
    }
    // L_n^α(0) = C(n + α, n).
    assert!(close(
        assoc_laguerre_l(5, 2.5f64, 0.0),
        7.5 * 6.5 * 5.5 * 4.5 * 3.5 / 120.0,
        1e-15
    ));
    assert!(close(assoc_legendre_p(1, 1, 0.6f64), -0.8, 1e-15));
    assert!(close(
        assoc_legendre_p(2, 1, 0.6f64),
        -3.0 * 0.6 * 0.8,
        1e-15
    ));
    assert!(assoc_legendre_p(3, 1, 1.5f64).is_nan());
    assert_eq!(hermite_h(3, f64::NEG_INFINITY), f64::NEG_INFINITY);
    assert_eq!(laguerre_l(3, f64::INFINITY), f64::NEG_INFINITY);
    assert_eq!(chebyshev_t(4, f64::NEG_INFINITY), f64::INFINITY);
    assert_eq!(gegenbauer_c(5, 0.0f64, 0.3), 0.0);
    assert!(legendre_p(3, f64::NAN).is_nan() && jacobi_p(2, f64::NAN, 1.0, 0.5).is_nan());
    // No spurious overflow: large degrees stay finite where the values are.
    let p = legendre_p(100_000, 0.3f64);
    assert!(p.is_finite() && p.abs() < 0.01);
    assert!(chebyshev_t(1_000_000_007, 0.5f64).abs() <= 1.0);
    assert_eq!(hermite_h(2000, 0.5f64).abs(), f64::INFINITY);
    // P_150^100(0.2) ≈ 6.80468782103575e212, past (2·100 − 1)!! ≈ 10^187.
    assert!(close(
        assoc_legendre_p(150, 100, 0.2f64),
        6.804_687_821_035_75e212,
        1e-13
    ));
    assert_eq!(assoc_legendre_p(300, 150, 0.2f64), f64::INFINITY);
    // Degenerate Jacobi parameters (a recurrence coefficient vanishes): the explicit sum.
    // P_3^(−1,−1)(x) = Σ_s C(2, 3 − s) C(2, s) ((x − 1)/2)^s ((x + 1)/2)^(3 − s): −0.168 at 0.4.
    assert!(close(jacobi_p(3, -1.0f64, -1.0, 0.4), -0.168, 1e-15));
}
