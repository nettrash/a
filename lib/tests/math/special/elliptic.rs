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

/// Checks a real function case by case against the bound of the region of its arguments,
/// `region(args) = (name, bound)`, in ulps or (where `kappa` gives a condition number) in ulps
/// divided by `1 + κ`; prints the largest error of each region.
#[track_caller]
fn by_region(
    name: &str,
    text: &str,
    p: Prec,
    region: impl Fn(&[f64]) -> (&'static str, f64),
    kappa: impl Fn(&[f64], f64) -> f64,
    f: impl Fn(&[f64]) -> f64,
) {
    let cases = cases(&thin(text));
    assert!(!cases.is_empty(), "{name}: no cases");
    let mut worst: Vec<(&'static str, f64, f64, Vec<f64>)> = Vec::new();
    for c in &cases {
        let (reg, bound) = region(&c.args);
        let bound = miri(bound);
        let got = f(&c.args);
        let e = ulps(got, c.refs[0], p) / (1.0 + kappa(&c.args, c.refs[0].0));
        assert!(
            e <= bound,
            "{name} [{reg}]: {e:.3} ulps > {bound} at {:?} ({:?}): got {got:e}, want {:e} + {:e}",
            c.args,
            c.args.iter().map(|a| a.to_bits()).collect::<Vec<_>>(),
            c.refs[0].0,
            c.refs[0].1
        );
        match worst.iter_mut().find(|w| w.0 == reg) {
            Some(w) if e > w.2 => (w.2, w.3) = (e, c.args.clone()),
            Some(_) => {}
            None => worst.push((reg, bound, e, c.args.clone())),
        }
    }
    for (reg, bound, e, at) in worst {
        eprintln!("{name} [{reg}]: {e:.3} (bound {bound}) at {at:?}");
    }
}

/// No condition number: plain ulps.
fn plain(_: &[f64], _: f64) -> f64 {
    0.0
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

// The bounds of `docs/math/special.md`, by region.
const B_K: f64 = 0.6;
const B_E: f64 = 0.6;
const B_PI: f64 = 3.5;
const B_F: f64 = 2.5; // m ≤ 1; m > 1 in units of 1 + κ
const B_EINC: f64 = 3.5; // 0 ≤ m ≤ 1
const B_EINC_NEG: f64 = 4.0; // m < 0
const B_EINC_GT1: f64 = 4.0; // m > 1
const B_PIINC_PLAIN: f64 = 3.0; // n < 1, 0 ≤ m < 1, |φ| ≤ π/2
const B_PIINC: f64 = 4.0; // everywhere, in units of 1 + κ
const B_RF: f64 = 0.75;
const B_RD: f64 = 0.75;
const B_RJ: f64 = 2.5;
const B_RC: f64 = 2.5;
const B_RJ_PV: f64 = 4.5;

const B_F32: f64 = 0.501;

/// `κ = |φ/(Δ F)|`, the condition number of `F(φ, m)` in `φ` (unbounded at the edge of the real
/// domain of `m > 1`, `Δ → 0`), for `m > 1`; 0 elsewhere.
fn f_kappa(a: &[f64], f: f64) -> f64 {
    let (phi, m) = (a[0], a[1]);
    if m <= 1.0 {
        return 0.0;
    }
    let d = (1.0 - m * phi.sin().powi(2)).abs().sqrt();
    (phi / (d * f)).abs()
}

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
    let (mut pos, mut pv) = ((0.0f64, vec![]), (0.0f64, vec![]));
    for c in &cases {
        let a = &c.args;
        let got = carlson_rj(a[0], a[1], a[2], a[3]);
        let (w, e) = if a[3] > 0.0 {
            (&mut pos, ulps(got, c.refs[0], F64))
        } else {
            let s = carlson_rj(a[0], a[1], a[2], -a[3]);
            (&mut pv, scaled_ulps(got, c.refs[0], s, F64))
        };
        if e > w.0 || e.is_nan() {
            *w = (e, a.clone());
        }
    }
    let (b1, b2) = (miri(B_RJ), miri(B_RJ_PV));
    assert!(
        pos.0 <= b1,
        "carlson_rj f64 (p > 0): {:.3} ulps > {b1} at {:?}",
        pos.0,
        pos.1
    );
    assert!(
        pv.0 <= b2,
        "carlson_rj f64 (p < 0): {:.3} scaled ulps > {b2} at {:?}",
        pv.0,
        pv.1
    );
    eprintln!(
        "carlson_rj f64: p > 0 {:.3} ulps, p < 0 {:.3} scaled ulps",
        pos.0, pv.0
    );
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
    let f_region = |a: &[f64]| {
        if a[1] <= 1.0 {
            ("m <= 1", B_F)
        } else {
            ("m > 1, ulps/(1 + κ)", B_F)
        }
    };
    by_region("ellip_f f64", F_F64, F64, f_region, f_kappa, |a| {
        ellip_f(a[0], a[1])
    });
    let e_region = |a: &[f64]| {
        let m = a[1];
        if (0.0..=1.0).contains(&m) {
            ("0 <= m <= 1", B_EINC)
        } else if m < 0.0 {
            ("m < 0", B_EINC_NEG)
        } else {
            ("m > 1", B_EINC_GT1)
        }
    };
    by_region("ellip_e_inc f64", EINC_F64, F64, e_region, plain, |a| {
        ellip_e_inc(a[0], a[1])
    });
    pi_inc_conditioned(PIINC_F64);
    // The region of the plain bound: n < 1, 0 ≤ m < 1, |φ| ≤ π/2.
    let plain_cases: String = PIINC_F64
        .lines()
        .filter(|l| {
            let c = cases(l);
            c.len() == 1 && {
                let a = &c[0].args;
                a[0] < 1.0
                    && (0.0..1.0).contains(&a[2])
                    && a[1].abs() <= core::f64::consts::FRAC_PI_2
            }
        })
        .map(|l| format!("{l}\n"))
        .collect();
    real(
        "ellip_pi_inc f64 (plain region)",
        &plain_cases,
        F64,
        B_PIINC_PLAIN,
        |a| ellip_pi_inc(a[0], a[1], a[2]),
    );
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

/// The bounds of the Jacobi functions (`f64`) by the band of the parameter `μ` they are computed
/// at: `m` itself in `[0, 1)`, `μ = −m/(1 − m)` for `m < 0` and `1/m` for `m > 1` (A&S 16.10–11).
/// They grow as `μ → 1`, where `cn` and `dn` of `μ` are both about `μ₁^(1/4)` near `u ≡ K/2
/// (mod K)` and the condition number there is about `K/2`. They are the core's for `0 ≤ m < 1`,
/// for every form: the Landen ascent carries its values as double-doubles, rounded once. Rounded
/// at each step, `μ < 0.9` reached 5.19 ulps for `m < 0`, 4.19 for `0 ≤ m < 1` (at `(74.3, 0.890)`)
/// and 3.74 for `m > 1`, and `0.9 ≤ μ < 1 − 10⁻⁶` 34.7 (at `(−0.274, −552330)`); 2.05, 1.55, 2.49
/// and 10.8 now, on 60,000 points of each.
fn jacobi_bound(m: f64) -> f64 {
    let mu = if m < 0.0 {
        -m / (1.0 - m)
    } else if m > 1.0 {
        1.0 / m
    } else {
        m
    };
    if mu < 0.9 {
        B_JAC
    } else if mu < 1.0 - 1e-6 {
        B_JAC_09
    } else {
        B_JAC_1
    }
}

const B_JAC: f64 = 3.5; // μ < 0.9
const B_JAC_09: f64 = 20.0; // 0.9 ≤ μ < 1 − 10⁻⁶
const B_JAC_1: f64 = 80.0; // 1 − 10⁻⁶ ≤ μ < 1

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
    // (0.030236235303514835 is the audit's: s R_F(c², 1, 1) of the rounded s, c is 2 ulps off.)
    for &phi in &[
        0.0,
        1e-300,
        0.3,
        0.030_236_235_303_514_835,
        1.0,
        1.5,
        3.0,
        -7.0,
        100.0,
        1e300,
    ] {
        assert_eq!(ellip_f(phi, 0.0f64), phi);
        assert_eq!(ellip_e_inc(phi, 0.0f64), phi);
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
    // F(φ, 1) = asinh(tan φ) is finite up to the f64 next to π/2, below it.
    assert!(close(
        ellip_f(FRAC_PI_2, 1.0f64),
        38.025_003_373_828_87,
        1e-15
    ));
    assert_eq!(ellip_f(f32::from_bits(0x3fc9_0fda), 1.0), 17.092_308);
    // Π(n → −∞) → 0, odd in φ, as for n → +∞ and the complete integral.
    let p = ellip_pi_inc(f64::NEG_INFINITY, 1.0, 0.5);
    assert!(p == 0.0 && p.is_sign_positive());
    let p = ellip_pi_inc(f64::NEG_INFINITY, -1.0, 0.5);
    assert!(p == 0.0 && p.is_sign_negative());
    assert_eq!(ellip_pi_inc(f64::NEG_INFINITY, 0.1, 20.0), 0.0);
    // m = 1 past ±π/2: 2jΠ(n, 1) diverges, to −∞ for n > 1 (as Π(n, 1) and Π(n; φ → π/2, 1) do).
    assert_eq!(ellip_pi_inc(2.0f64, 2.0, 1.0), f64::NEG_INFINITY);
    assert_eq!(ellip_pi_inc(2.0f64, -2.0, 1.0), f64::INFINITY);
    assert_eq!(ellip_pi_inc(0.5f64, 2.0, 1.0), f64::INFINITY);
    assert!(ellip_pi_inc(2.0f64, FRAC_PI_2 - 1e-4, 1.0) < -8.0);
    // m > 1 within the rounding of sin φ of the edge of the real domain: the value at the edge
    // (1 − m sin²φ = 5.0e−17 there, where the computed one is negative).
    let f = ellip_f(0.119_272_229_864_690_45f64, 70.628_780_048_130_31);
    assert!(close(f, 0.187_575_396_222_624_12, 1e-8), "{f}");
    assert!(ellip_e_inc(0.119_272_229_864_690_45f64, 70.628_780_048_130_31).is_finite());
    assert!(ellip_f(0.2f64, 70.6).is_nan());
    // m = 1 + 2^−52: FRAC_PI_2 is past the edge (sin φ = 1/√m at π/2 − 1.5e−8) within the rounding
    // of its sine, 1: the values at the edge, K(1/m)/√m, √m E(1/m) − ((m − 1)/√m) K(1/m) and
    // Π(n/m, 1/m)/√m (mpmath), not those of the point's own cosine (F was 38.7 there).
    let m = 1.0 + f64::EPSILON;
    assert!(close(ellip_f(FRAC_PI_2, m), 19.408_121_055_678_468, 1e-15));
    assert!(close(
        ellip_e_inc(FRAC_PI_2, m),
        0.999_999_999_999_997_9,
        1e-15
    ));
    assert!(close(
        ellip_pi_inc(0.5, FRAC_PI_2, m),
        37.569_791_631_076_47,
        1e-15
    ));
    assert!(close(
        ellip_f(-FRAC_PI_2, m),
        -19.408_121_055_678_468,
        1e-15
    ));
    // |φ| > 2^52, the reduction without j: a result beyond f64::MAX is ±∞ (not NaN), one next to
    // it finite, and Π(1; φ, m) past π/2 diverges with the sign of φ.
    assert_eq!(ellip_f(1.7e308f64, 0.9), f64::INFINITY);
    assert_eq!(ellip_f(-1.7e308f64, 0.5), f64::NEG_INFINITY);
    assert_eq!(ellip_e_inc(-3.2e239f64, -5.35e219), f64::NEG_INFINITY);
    assert_eq!(ellip_e_inc(1.7e308f64, -1.0), f64::INFINITY);
    assert_eq!(ellip_pi_inc(0.9f64, 1.7e308, 0.5), f64::INFINITY);
    assert!(close(
        ellip_f(1.5e308f64, -0.1),
        1.464_473_267_685_807_2e308,
        1e-15
    ));
    assert!(ellip_pi_inc(-0.5f64, -1.7e308, -2.0).is_finite());
    assert_eq!(ellip_pi_inc(1.0f64, 1e16, 0.5), f64::INFINITY);
    assert_eq!(ellip_pi_inc(1.0f64, -1e300, -3.0), f64::NEG_INFINITY);
    assert_eq!(ellip_pi_inc(1.0f32, 1e20, 0.5), f32::INFINITY);
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
    // Results beyond f64::MAX are ∞, and none below is lost to the scaling of the arguments.
    assert_eq!(carlson_rd(5e-324, 1.0, 5e-324), f64::INFINITY);
    assert_eq!(carlson_rj(1e-310, 1e-310, 1e-310, 1e-310), f64::INFINITY);
    let m = f64::MAX;
    assert!(close(carlson_rf(m, m, m), 1.0 / m.sqrt(), 3e-16));
    assert!(close(carlson_rc(m, m), 1.0 / m.sqrt(), 3e-16));
    assert!(close(
        carlson_rf(5e-324, 5e-324, 5e-324),
        4.498_913_794_543_196_4e161,
        3e-16
    ));
    // R_J(x, x, x, p) = 3(1/√x − R_C(x, p))/(p − x) ≈ 3/(p √x) for p ≫ x.
    assert!(close(
        carlson_rj(1e-300, 1e-300, 1e-300, 1e300),
        3e-150,
        1e-15
    ));
    // Principal values with x, y far below max(y, −p): finite (ρ = p'q/(xy) and √z/(√x √y) beyond
    // the range), the limit −0 of one below the range, and one whose (z − p) R_J is beyond it.
    assert!(close(
        carlson_rj(
            2.556_170_558e-314,
            2.557_064_187_5e-314,
            0.003_301_674_300_142_264_4,
            -0.003_301_674_300_142_264_4
        ),
        -5_665_077.279_996_699,
        1e-15
    ));
    let v = carlson_rj(
        2.494_427_078_406_156e-278f64,
        1.004_119_275_926_421_1e-135,
        5.105_809_125_734_171e228,
        -5.105_809_111_706_431e228,
    );
    assert!(v == 0.0 && v.is_sign_negative(), "{v}");
    assert!(close(
        carlson_rj(
            2.164_225_721_491_554e-273,
            1.706_943_946_995_611_7e-271,
            1.524_684_731_269_045_9e212,
            -6.883_237_039_464_134e-294
        ),
        3.057_157_391_908_898_3e167,
        1e-15
    ));
    // Spans beyond 2^1600 with z the largest: finite (R_D by its expansion for z ≫ x, y; R_J's
    // duplication with its terms' factors apart), where they were 1.3e−317 (for 1.8e−315), +∞ or
    // NaN (references mpmath, rounded to f64; the first is subnormal, to a unit of its last place).
    assert!(close(
        carlson_rd(1e-316f64, 1e-316, 1e212),
        1.822_726_837e-315,
        3e-9
    ));
    assert!(close(
        carlson_rj(0.0f64, 1e-310, 1e200, 1e-310),
        3.000_000_000_000_009e210,
        1e-15
    ));
    let v = carlson_rj(0.0f64, 1e-310, 1e200, -1e300);
    assert!(v == 0.0 && v.is_sign_negative(), "{v}");
    // A documented limit (an open defect): two arguments above about 2^1022 and a third below
    // 2^−1000, a spread no scaling brings into the range, give NaN for a finite integral.
    let h = f64::MAX / 2.0;
    assert!(carlson_rd(h, h, 5e-324).is_nan());
    assert!(carlson_rj(h, h, 1.0, 5e-324).is_nan());
    // A documented limit: arguments spanning beyond 2^1600 and a result just above 2^−1022, where
    // the duplication runs on subnormal arguments: 14.8 ulps (of 2^−1074) here, up to about 24
    // measured, not a value outside that loss (reference by Arb).
    let v = carlson_rj(
        1.520_179_05e-315f64,
        5.11e-321,
        6.644_927_698_423_634e262,
        2.538_944_011_659_919_7e179,
    );
    let r = 2.611_312_042_229_738_4e-308;
    assert!((v - r).abs() <= 32.0 * f64::from_bits(1), "{v:e}");
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
    // Huge arguments, reduced with K to their bits (sn² + cn² = 1 holds to rounding anyway; the
    // values are checked against mpmath in the data): no collapse to (±1, 0, √m₁).
    // (Under Miri, whose many-word arithmetic of the reduction is slow, two of them.)
    let us: &[f64] = if cfg!(miri) {
        &[1e20, -2.5e307]
    } else {
        &[1e16, 1e20, 1e100, 1e300, -2.5e307]
    };
    for &m in &[0.5f64, 1e-300, -3.0, 4.0] {
        for &u in us {
            let (s, c, d) = jacobi_elliptic(u, m);
            assert!((s * s + c * c - 1.0).abs() < tol(1e-14), "u {u} m {m}");
            assert!(
                s.abs() < 0.999_999 || c.abs() > 1e-6,
                "u {u} m {m}: collapsed"
            );
            assert!(d.is_finite());
        }
    }
    // Next to 21K (m = ½): cn is 1.7e−15 just below, of the sign of cos, not ∓0.
    let (s, c, _) = jacobi_elliptic(f64::from_bits(0x4043_77c0_b315_2fe1), 0.5);
    assert!(s == 1.0 && c > 1.7e-15 && c < 1.8e-15, "{c}");
    // m = 1e−300: cn(u | m) = cos u to 1e−300.
    for b in [0x4073_bbae_5529_8768u64, 0x40a8_8e53_afb5_1c8f] {
        let u = f64::from_bits(b);
        let c = jacobi_elliptic(u, 1e-300).1;
        assert!((c - u.cos()).abs() < tol(1e-28), "{u}: {c:e}");
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
    // P_n^(−1,−1)(x) = ((x² − 1)/4) P_(n−2)^(1,1)(x), for all n ≥ 2.
    for n in [2u32, 5, 20, 50, 300] {
        for &x in &[-0.9f64, -0.3, 0.0, 0.5, 0.99, 1.7] {
            let p = jacobi_p(n, -1.0, -1.0, x);
            let q = (x * x - 1.0) / 4.0 * jacobi_p(n - 2, 1.0, 1.0, x);
            assert!(
                (p - q).abs() <= 1e-15 * q.abs().max(0.1),
                "n {n} x {x}: {p} {q}"
            );
        }
    }
    // Gegenbauer of α = −l vanishes identically above the degree 2l; Laguerre of α = −l is
    // (−x)^l (n − l)!/n! L_(n−l)^(l)(x).
    for (n, a) in [(7u32, -3.0f64), (300, -3.0), (9, -4.0), (128, -3.0)] {
        for &x in &[-0.7f64, 0.0, 0.3, 1.0, 1.36] {
            assert_eq!(gegenbauer_c(n, a, x), 0.0, "C_{n}^({a})({x})");
        }
    }
    let l = assoc_laguerre_l(200, -2.0f64, 1e-9);
    assert!(close(l, 4.999_999_670_000_009e-19, 1e-15), "{l}");
    // No NaN from the low words where the plain recurrence meets an exact zero.
    assert!(gegenbauer_c(200, -1.5f64, 0.999_999_999).is_finite());
    assert!(gegenbauer_c(400, -2.5f64, 1.0).is_finite());
    // |x| > 2^400: the leading term, finite where it is (P₂(1e154) = 1.5e308), and ±∞ past MAX.
    assert!(close(legendre_p(2, 1e154f64), 1.5e308, 1e-15));
    assert_eq!(legendre_p(3, -1e200f64), f64::NEG_INFINITY);
    assert_eq!(chebyshev_t(1, 1e300f64), 1e300);
    assert_eq!(chebyshev_u(1, 1.5e158f64), 3e158);
    assert_eq!(hermite_h(2, 1e200f64), f64::INFINITY);
    assert_eq!(laguerre_l(3, 1e200f64), f64::NEG_INFINITY);
}
