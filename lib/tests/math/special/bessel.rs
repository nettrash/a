//! The bessel area of issue #20: its data in `data/bessel_*.txt`, made by `data/gen_bessel.py`.
//!
//! Each data set is checked by regions, each against its stated bound: in ulps of the value, or
//! (for `J`, `Y` and the spherical `j`, `y` of the general orders, whose zeros no method of
//! this kind resolves) in ulps of the modulus `M = √(J² + Y²)`, the second reference of a case.
//! Below their turning points (`|x| ≤ ν`), where they have no zeros and `M` is the size of `Y`,
//! `J_ν` and `jₙ` are measured in ulps of their value.

use super::common::*;
use a::math::special::{
    airy_ai, airy_ai_prime, airy_bi, airy_bi_prime, bessel_i, bessel_i_scaled, bessel_j, bessel_k,
    bessel_k_scaled, bessel_y, sph_bessel_i, sph_bessel_j, sph_bessel_k, sph_bessel_y,
};

const J_F64: &str = include_str!("data/bessel_j_f64.txt");
const J_F32: &str = include_str!("data/bessel_j_f32.txt");
const Y_F64: &str = include_str!("data/bessel_y_f64.txt");
const Y_F32: &str = include_str!("data/bessel_y_f32.txt");
const I_F64: &str = include_str!("data/bessel_i_f64.txt");
const I_F32: &str = include_str!("data/bessel_i_f32.txt");
const K_F64: &str = include_str!("data/bessel_k_f64.txt");
const K_F32: &str = include_str!("data/bessel_k_f32.txt");
const IS_F64: &str = include_str!("data/bessel_is_f64.txt");
const IS_F32: &str = include_str!("data/bessel_is_f32.txt");
const KS_F64: &str = include_str!("data/bessel_ks_f64.txt");
const KS_F32: &str = include_str!("data/bessel_ks_f32.txt");
const SJ_F64: &str = include_str!("data/bessel_sj_f64.txt");
const SJ_F32: &str = include_str!("data/bessel_sj_f32.txt");
const SY_F64: &str = include_str!("data/bessel_sy_f64.txt");
const SY_F32: &str = include_str!("data/bessel_sy_f32.txt");
const SI_F64: &str = include_str!("data/bessel_si_f64.txt");
const SI_F32: &str = include_str!("data/bessel_si_f32.txt");
const SK_F64: &str = include_str!("data/bessel_sk_f64.txt");
const SK_F32: &str = include_str!("data/bessel_sk_f32.txt");
const AI_F64: &str = include_str!("data/bessel_ai_f64.txt");
const AI_F32: &str = include_str!("data/bessel_ai_f32.txt");
const AIP_F64: &str = include_str!("data/bessel_aip_f64.txt");
const AIP_F32: &str = include_str!("data/bessel_aip_f32.txt");
const BI_F64: &str = include_str!("data/bessel_bi_f64.txt");
const BI_F32: &str = include_str!("data/bessel_bi_f32.txt");
const BIP_F64: &str = include_str!("data/bessel_bip_f64.txt");
const BIP_F32: &str = include_str!("data/bessel_bip_f32.txt");

/// A region of a data set: its name, its bound, and whether the error is in ulps of the
/// modulus (the second reference) rather than of the value.
type Region = (&'static str, f64, bool);

/// The worst case of a region: name, bound, error, count, arguments, result.
type Worst = (&'static str, f64, f64, usize, Vec<f64>, f64);

/// The cases of a data set, every 37th under Miri.
fn data(text: &str) -> Vec<Case> {
    let c = cases(text);
    assert!(!c.is_empty(), "no cases");
    if cfg!(miri) {
        c.into_iter().step_by(37).collect()
    } else {
        c
    }
}

/// Checks `f` over the cases of `text`, each region (from `region(args)`) against its bound.
#[track_caller]
fn check(
    label: &str,
    text: &str,
    p: Prec,
    f: impl Fn(&[f64]) -> f64,
    region: impl Fn(&[f64]) -> Region,
) {
    let mut worst: Vec<Worst> = Vec::new();
    for c in data(text) {
        let (name, bound, modulus) = region(&c.args);
        let got = f(&c.args);
        let e = if modulus && c.refs.len() > 1 && c.refs[0].0.is_finite() && got.is_finite() {
            let (hi, lo) = c.refs[0];
            let scale = hi.abs().max(c.refs[1].0.abs());
            ((got - hi) - lo).abs() / ulp(scale, p)
        } else {
            ulps(got, c.refs[0], p)
        };
        match worst.iter_mut().find(|w| w.0 == name) {
            Some(w) => {
                w.3 += 1;
                if e > w.2 || e.is_nan() {
                    w.2 = e;
                    w.4 = c.args.clone();
                    w.5 = got;
                }
            }
            None => worst.push((name, bound, e, 1, c.args.clone(), got)),
        }
    }
    let mut failed = Vec::new();
    for w in &worst {
        eprintln!(
            "{label} {}: {} cases, max {:.3} (bound {})",
            w.0, w.3, w.2, w.1
        );
        if w.2.is_nan() || w.2 > w.1 {
            failed.push(format!(
                "{} {:.3} > {} at {:?}: got {:e}",
                w.0, w.2, w.1, w.4, w.5
            ));
        }
    }
    assert!(failed.is_empty(), "{label}: {failed:?}");
}

// The regions of the cylinder functions by order and argument: for `J` and `Y` of the orders
// 0 and 1, relative below 24 (the expansions about the zeros), in ulps of the modulus beyond.
fn order_region(nu: f64, x: f64, jy: bool, int01: Region, general: Region) -> Region {
    if nu == 0.0 || nu == 1.0 {
        if jy && x.abs() >= 24.0 {
            ("ν = 0, 1, x ≥ 24, ulps of M", 1.0, true)
        } else {
            int01
        }
    } else {
        general
    }
}

/// The regions of `I`, `K` and their scaled forms: Debye's expansions from `ν = 50` on (1 ulp);
/// at the negative non-integer orders of `I` the reflection `I_a + (2/π)·sin(aπ)·K_a`, measured in
/// ulps of `I_a + (2/π)|sin aπ|·K_a` (the second reference), which its zero does not resolve.
fn ik_region(nu: f64, x: f64, plain_i: bool) -> Region {
    let neg = plain_i && nu < 0.0 && nu != nu.floor();
    if nu == 0.0 || nu == 1.0 || (!plain_i && nu == -1.0) {
        ("ν = 0, 1", 1.0, false)
    } else if neg && nu <= -50.0 {
        (
            "ν ≤ −50 non-integer, ulps of the reflection's terms",
            1.0,
            true,
        )
    } else if neg {
        (
            "−50 < ν < 0 non-integer, ulps of the reflection's terms",
            6.0,
            true,
        )
    } else if nu.abs() >= 50.0 && x > 0.0 {
        ("|ν| ≥ 50 (Debye's)", 1.0, false)
    } else {
        ("general ν", 6.0, false)
    }
}

/// Whether `J_ν(x)` is below its turning point, `|x| ≤ ν`, at a positive order other than 0 and 1
/// up to Olver's region: it has no zeros there, and `M` is the size of `Y`, so the error is
/// measured in ulps of the value.
fn below_turning_point(nu: f64, x: f64) -> bool {
    nu > 0.0 && nu != 1.0 && nu <= 1.0e7 && x.abs() <= nu
}

#[test]
fn bessel_j_f64() {
    check(
        "J f64",
        J_F64,
        F64,
        |a| bessel_j(a[0], a[1]),
        |a| {
            if below_turning_point(a[0], a[1]) {
                return ("general ν > 0, |x| ≤ ν, ulps of the value", 5.0, false);
            }
            order_region(
                a[0],
                a[1],
                true,
                ("ν = 0, 1, x < 24", 1.5, false),
                ("general ν, ulps of M", 5.0, true),
            )
        },
    );
}

#[test]
fn bessel_y_f64() {
    check(
        "Y f64",
        Y_F64,
        F64,
        |a| bessel_y(a[0], a[1]),
        |a| {
            order_region(
                a[0],
                a[1],
                true,
                ("ν = 0, 1, x < 24", 1.5, false),
                ("general ν, ulps of M", 6.0, true),
            )
        },
    );
}

#[test]
fn bessel_i_f64() {
    check(
        "I f64",
        I_F64,
        F64,
        |a| bessel_i(a[0], a[1]),
        |a| ik_region(a[0], a[1], true),
    );
}

#[test]
fn bessel_k_f64() {
    check(
        "K f64",
        K_F64,
        F64,
        |a| bessel_k(a[0], a[1]),
        |a| ik_region(a[0], a[1], false),
    );
}

#[test]
fn bessel_i_scaled_f64() {
    check(
        "I scaled f64",
        IS_F64,
        F64,
        |a| bessel_i_scaled(a[0], a[1]),
        |a| ik_region(a[0], a[1], true),
    );
}

#[test]
fn bessel_k_scaled_f64() {
    check(
        "K scaled f64",
        KS_F64,
        F64,
        |a| bessel_k_scaled(a[0], a[1]),
        |a| ik_region(a[0], a[1], false),
    );
}

/// The cases of `text` about the zero of `I_ν` at a non-integer `ν` in `(−50, 0)` (the value below
/// an eighth of the reflection's terms, the second reference) checked in ulps of the value.
#[track_caller]
fn check_about_the_zero(label: &str, text: &str, f: fn(f64, f64) -> f64) {
    let mut n = 0;
    for c in data(text) {
        let (nu, x) = (c.args[0], c.args[1]);
        let near_zero = nu > -50.0
            && nu < 0.0
            && nu != nu.floor()
            && c.refs.len() > 1
            && c.refs[0].0.abs() < 0.125 * c.refs[1].0;
        if !near_zero {
            continue;
        }
        n += 1;
        let e = ulps(f(nu, x), c.refs[0], F64);
        assert!(e <= 6.0, "{label}({nu}, {x}): {e:.3} ulps of the value");
    }
    assert!(cfg!(miri) || n >= 30, "{label}: {n} cases");
}

#[test]
fn i_keeps_its_error_relative_about_its_zero_at_negative_orders() {
    // About the zero of I_ν for a non-integer ν in (−50, 0) the reflection's terms cancel, so
    // `ik_region` measures in their scale; there the power series of I_ν keeps the error relative
    // (within an ulp of the value down to about 10⁻¹⁰ of the terms' size for |ν| up to 75; the
    // cases of the data are 3·10⁻⁶ of the zero or further from it).
    check_about_the_zero("I", I_F64, bessel_i);
    check_about_the_zero("I scaled", IS_F64, bessel_i_scaled);
}

/// `f32` results: correctly rounded but within a hair, and the general `J`, `Y` near their zeros
/// in ulps of the modulus (`I` at the negative non-integer orders in ulps of the reflection's
/// terms, see `ik_region`).
fn f32_region(nu: f64, jy: bool) -> Region {
    if jy && nu != 0.0 && nu != 1.0 {
        ("f32, ulps of M", 0.51, true)
    } else {
        ("f32", 0.51, false)
    }
}

fn f32_i_region(nu: f64) -> Region {
    if nu < 0.0 && nu != nu.floor() {
        (
            "f32, ν < 0 non-integer, ulps of the reflection's terms",
            0.51,
            true,
        )
    } else {
        ("f32", 0.51, false)
    }
}

#[test]
fn bessel_f32() {
    check(
        "J f32",
        J_F32,
        F32,
        |a| bessel_j(a[0] as f32, a[1] as f32) as f64,
        |a| {
            if below_turning_point(a[0], a[1]) {
                ("f32, ν > 0, |x| ≤ ν", 0.51, false)
            } else {
                f32_region(a[0], true)
            }
        },
    );
    check(
        "Y f32",
        Y_F32,
        F32,
        |a| bessel_y(a[0] as f32, a[1] as f32) as f64,
        |a| f32_region(a[0], true),
    );
    check(
        "I f32",
        I_F32,
        F32,
        |a| bessel_i(a[0] as f32, a[1] as f32) as f64,
        |a| f32_i_region(a[0]),
    );
    check(
        "K f32",
        K_F32,
        F32,
        |a| bessel_k(a[0] as f32, a[1] as f32) as f64,
        |a| f32_region(a[0], false),
    );
    check(
        "I scaled f32",
        IS_F32,
        F32,
        |a| bessel_i_scaled(a[0] as f32, a[1] as f32) as f64,
        |a| f32_i_region(a[0]),
    );
    check(
        "K scaled f32",
        KS_F32,
        F32,
        |a| bessel_k_scaled(a[0] as f32, a[1] as f32) as f64,
        |a| f32_region(a[0], false),
    );
}

#[test]
fn spherical_f64() {
    check(
        "j f64",
        SJ_F64,
        F64,
        |a| sph_bessel_j(a[0] as u32, a[1]),
        |a| {
            if a[0] == 0.0 {
                ("n = 0", 1.0, false)
            } else if a[0] <= 1.0e7 && a[1].abs() <= a[0] {
                // Below the turning point n + 1/2 (M is the size of yₙ there, far above jₙ).
                ("n ≥ 1, |x| ≤ n, ulps of the value", 2.0, false)
            } else {
                ("n ≥ 1, ulps of M", 2.0, true)
            }
        },
    );
    check(
        "y f64",
        SY_F64,
        F64,
        |a| sph_bessel_y(a[0] as u32, a[1]),
        |a| {
            if a[0] == 0.0 {
                ("n = 0", 1.0, false)
            } else {
                ("n ≥ 1, ulps of M", 2.0, true)
            }
        },
    );
    check(
        "i f64",
        SI_F64,
        F64,
        |a| sph_bessel_i(a[0] as u32, a[1]),
        |_| ("all", 3.0, false),
    );
    check(
        "k f64",
        SK_F64,
        F64,
        |a| sph_bessel_k(a[0] as u32, a[1]),
        |_| ("all", 1.0, false),
    );
}

#[test]
fn spherical_f32() {
    let sph = |a: &[f64]| {
        if a[0] == 0.0 {
            ("f32", 0.51, false)
        } else {
            ("f32, ulps of M", 0.51, true)
        }
    };
    check(
        "j f32",
        SJ_F32,
        F32,
        |a| sph_bessel_j(a[0] as u32, a[1] as f32) as f64,
        |a| {
            if a[0] >= 1.0 && a[1].abs() <= a[0] {
                ("f32, n ≥ 1, |x| ≤ n", 0.51, false)
            } else {
                sph(a)
            }
        },
    );
    check(
        "y f32",
        SY_F32,
        F32,
        |a| sph_bessel_y(a[0] as u32, a[1] as f32) as f64,
        sph,
    );
    check(
        "i f32",
        SI_F32,
        F32,
        |a| sph_bessel_i(a[0] as u32, a[1] as f32) as f64,
        |_| ("f32", 0.51, false),
    );
    check(
        "k f32",
        SK_F32,
        F32,
        |a| sph_bessel_k(a[0] as u32, a[1] as f32) as f64,
        |_| ("f32", 0.51, false),
    );
}

/// The regions of the Airy functions: relative from the expansions about the zeros (down to
/// about −8.5) up; below, in ulps of the modulus, which far out grow with the phase
/// `ζ = (2/3)|x|^(3/2)`, known to `2⁻¹⁰⁴·ζ` only (beyond `x = −10⁹`, see `airy_far`).
fn airy_region(x: f64) -> Region {
    if x >= -8.4 {
        ("x ≥ −8.4", 1.5, false)
    } else if x >= -1.0e5 {
        ("−10⁵ ≤ x < −8.4, ulps of M", 1.0, true)
    } else if x >= -1.0e9 {
        ("−10⁹ ≤ x < −10⁵, ulps of M", 1.0, true)
    } else {
        ("x < −10⁹, see airy_far", f64::INFINITY, true)
    }
}

/// Beyond `x = −10⁹` the error of an Airy function is its bound `base` in ulps of the modulus
/// `M` (the second reference) and the phase error, `2⁻¹⁰⁴·ζ` of `M` (counted four times over), and
/// the value never leaves `[−M, M]` (where the phase is lost entirely, it is all that is left).
#[track_caller]
fn airy_far(label: &str, text: &str, p: Prec, base: f64, f: impl Fn(f64) -> f64) {
    let mut n = 0;
    for c in data(text) {
        let x = c.args[0];
        if x >= -1.0e9 {
            continue;
        }
        n += 1;
        let got = f(x);
        let ((hi, lo), m) = (c.refs[0], c.refs[1].0);
        assert!(
            got.is_finite() && got.abs() <= m * (1.0 + 4.0 * 2f64.powi(1 - p.bits)),
            "{label}({x:e}) = {got:e} is beyond the modulus {m:e}"
        );
        let zeta = (2.0 / 3.0) * (-x).powf(1.5);
        let bound = base + 4.0 * zeta * 2f64.powi(p.bits - 105);
        let err = ((got - hi) - lo).abs() / ulp(m, p);
        assert!(
            err <= bound,
            "{label}({x:e}): {err:.3} ulps of M > {bound:.3e}"
        );
    }
    // Under Miri every 37th case is kept, which may leave none this far out.
    assert!(cfg!(miri) || n > 0, "{label}: no case beyond −10⁹");
}

#[test]
fn airy_f64() {
    check(
        "Ai f64",
        AI_F64,
        F64,
        |a| airy_ai(a[0]),
        |a| airy_region(a[0]),
    );
    check(
        "Ai' f64",
        AIP_F64,
        F64,
        |a| airy_ai_prime(a[0]),
        |a| airy_region(a[0]),
    );
    check(
        "Bi f64",
        BI_F64,
        F64,
        |a| airy_bi(a[0]),
        |a| airy_region(a[0]),
    );
    check(
        "Bi' f64",
        BIP_F64,
        F64,
        |a| airy_bi_prime(a[0]),
        |a| airy_region(a[0]),
    );
    airy_far("Ai f64", AI_F64, F64, 1.0, airy_ai);
    airy_far("Ai' f64", AIP_F64, F64, 1.0, airy_ai_prime);
    airy_far("Bi f64", BI_F64, F64, 1.0, airy_bi);
    airy_far("Bi' f64", BIP_F64, F64, 1.0, airy_bi_prime);
}

#[test]
fn airy_f32() {
    let r = |a: &[f64]| {
        if a[0] >= -8.4 {
            ("f32", 0.51, false)
        } else if a[0] >= -1.0e9 {
            ("f32, x < −8.4, ulps of M", 0.51, true)
        } else {
            ("f32, x < −10⁹, see airy_far", f64::INFINITY, true)
        }
    };
    check("Ai f32", AI_F32, F32, |a| airy_ai(a[0] as f32) as f64, r);
    check(
        "Ai' f32",
        AIP_F32,
        F32,
        |a| airy_ai_prime(a[0] as f32) as f64,
        r,
    );
    check("Bi f32", BI_F32, F32, |a| airy_bi(a[0] as f32) as f64, r);
    check(
        "Bi' f32",
        BIP_F32,
        F32,
        |a| airy_bi_prime(a[0] as f32) as f64,
        r,
    );
    let f32_of = |f: fn(f32) -> f32| move |x: f64| f(x as f32) as f64;
    airy_far("Ai f32", AI_F32, F32, 0.51, f32_of(airy_ai));
    airy_far("Ai' f32", AIP_F32, F32, 0.51, f32_of(airy_ai_prime));
    airy_far("Bi f32", BI_F32, F32, 0.51, f32_of(airy_bi));
    airy_far("Bi' f32", BIP_F32, F32, 0.51, f32_of(airy_bi_prime));
}

#[test]
fn special_values() {
    let inf = f64::INFINITY;
    assert_eq!(bessel_j(0.0, 0.0), 1.0);
    assert_eq!(bessel_j(2.5, 0.0), 0.0);
    assert_eq!(bessel_j(-0.5, 0.0), inf);
    assert_eq!(bessel_j(-1.5, 0.0), -inf);
    assert_eq!(bessel_y(0.0, 0.0), -inf);
    assert_eq!(bessel_y(3.3, 0.0), -inf);
    assert_eq!(bessel_y(-0.5, 0.0), 0.0);
    assert_eq!(bessel_i(0.0, 0.0), 1.0);
    assert_eq!(bessel_i(1.0, 0.0), 0.0);
    assert_eq!(bessel_k(0.7, 0.0), inf);
    assert_eq!(bessel_k_scaled(0.7, 0.0), inf);
    assert_eq!(bessel_j(0.3, inf), 0.0);
    assert_eq!(bessel_y(0.3, inf), 0.0);
    assert_eq!(bessel_i(0.3, inf), inf);
    assert_eq!(bessel_i_scaled(0.3, inf), 0.0);
    assert_eq!(bessel_k(0.3, inf), 0.0);
    assert_eq!(bessel_k(2.0, 800.0), 0.0);
    assert_eq!(bessel_i(2.0, 800.0), inf);
    // Outside the domains, and NaN in.
    assert!(bessel_j(0.5f64, -1.0).is_nan() && bessel_i(0.5f64, -1.0).is_nan());
    assert!(bessel_y(1.0f64, -1.0).is_nan() && bessel_k(1.0f64, -1.0).is_nan());
    assert!(bessel_k_scaled(1.0f64, -1.0).is_nan() && sph_bessel_k(1, -1.0f64).is_nan());
    let fs: [fn(f64, f64) -> f64; 6] = [
        bessel_j,
        bessel_y,
        bessel_i,
        bessel_k,
        bessel_i_scaled,
        bessel_k_scaled,
    ];
    for f in fs {
        assert!(f(f64::NAN, 1.0).is_nan() && f(1.0, f64::NAN).is_nan() && f(inf, 1.0).is_nan());
    }
    let fs: [fn(f64) -> f64; 4] = [airy_ai, airy_ai_prime, airy_bi, airy_bi_prime];
    for f in fs {
        assert!(f(f64::NAN).is_nan());
    }
    assert_eq!(airy_ai(inf), 0.0);
    assert_eq!(airy_bi(inf), inf);
    assert_eq!(airy_bi_prime(inf), inf);
    assert_eq!(airy_ai(-inf), 0.0);
    assert_eq!(airy_ai(0.0), 0.355_028_053_887_817_2);
    assert_eq!(airy_bi(0.0), 0.614_926_627_446_000_7);
    assert_eq!(sph_bessel_j(0, 0.0), 1.0);
    assert_eq!(sph_bessel_j(3, 0.0), 0.0);
    assert_eq!(sph_bessel_y(2, 0.0), -inf);
    assert_eq!(sph_bessel_i(0, 0.0), 1.0);
    assert_eq!(sph_bessel_k(4, 0.0), inf);
    assert_eq!(sph_bessel_k(4, 1.0e7), 0.0);
}

#[test]
fn symmetries_and_reflections() {
    for &x in &[0.3, 1.7, 9.5, 31.0, 250.0] {
        for n in 0..6 {
            let nf = n as f64;
            let s = if n % 2 == 1 { -1.0 } else { 1.0 };
            assert_eq!(bessel_j(nf, -x), s * bessel_j(nf, x));
            assert_eq!(bessel_i(nf, -x), s * bessel_i(nf, x));
            assert_eq!(bessel_j(-nf, x), s * bessel_j(nf, x));
            assert_eq!(bessel_y(-nf, x), s * bessel_y(nf, x));
            assert_eq!(bessel_i(-nf, x), bessel_i(nf, x));
            assert_eq!(bessel_k(-nf, x), bessel_k(nf, x));
            assert_eq!(sph_bessel_j(n, -x), s * sph_bessel_j(n, x));
            assert_eq!(sph_bessel_y(n, -x), -s * sph_bessel_y(n, x));
            assert_eq!(sph_bessel_i(n, -x), s * sph_bessel_i(n, x));
        }
        // J_(−1/2) = Y_(1/2) = −√(2/(πx))·cos x... and Y_(−1/2) = J_(1/2).
        let a = bessel_y(-0.5, x);
        let b = bessel_j(0.5, x);
        assert!((a - b).abs() <= 4e-16 * b.abs().max(0.1), "{x}: {a} {b}");
    }
}

#[test]
fn identities() {
    use core::f64::consts::PI;
    // Wronskians: J_ν Y_(ν+1) − J_(ν+1) Y_ν = −2/(πx); I_ν K_(ν+1) + I_(ν+1) K_ν = 1/x;
    // Ai Bi′ − Ai′ Bi = 1/π.
    for &nu in &[0.0f64, 0.25, 1.0, 2.7, 10.5] {
        for &x in &[0.1, 1.3, 4.0, 17.0, 60.0] {
            let w =
                bessel_j(nu, x) * bessel_y(nu + 1.0, x) - bessel_j(nu + 1.0, x) * bessel_y(nu, x);
            let t = -2.0 / (PI * x);
            let scale = (bessel_j(nu, x) * bessel_y(nu + 1.0, x)).abs().max(t.abs());
            assert!((w - t).abs() <= 1e-14 * scale, "JY {nu} {x}: {w} {t}");
            let w = bessel_i_scaled(nu, x) * bessel_k_scaled(nu + 1.0, x)
                + bessel_i_scaled(nu + 1.0, x) * bessel_k_scaled(nu, x);
            assert!((w * x - 1.0).abs() <= 1e-14, "IK {nu} {x}: {}", w * x);
        }
    }
    for &x in &[-30.0f64, -5.5, -1.0, 0.0, 0.7, 3.0, 9.0] {
        let w = airy_ai(x) * airy_bi_prime(x) - airy_ai_prime(x) * airy_bi(x);
        let scale = (airy_ai(x) * airy_bi_prime(x)).abs().max(1.0 / PI);
        assert!((w - 1.0 / PI).abs() <= 1e-15 * scale, "Airy {x}: {w}");
    }
    // Scaled and plain agree.
    for &nu in &[0.0f64, 1.0, 0.6, 3.0, 25.5] {
        for &x in &[0.4f64, 3.0, 40.0] {
            let i = bessel_i(nu, x) * (-x).exp();
            let k = bessel_k(nu, x) * x.exp();
            assert!((bessel_i_scaled(nu, x) - i).abs() <= 1e-14 * i, "{nu} {x}");
            assert!((bessel_k_scaled(nu, x) - k).abs() <= 1e-14 * k, "{nu} {x}");
        }
    }
    // jₙ, yₙ against J, Y of the half-integer orders.
    for n in 0..8u32 {
        for &x in &[0.5, 3.0, 12.0, 70.0] {
            let f = (PI / (2.0 * x)).sqrt();
            let m = f * bessel_j(n as f64 + 0.5, x).hypot(bessel_y(n as f64 + 0.5, x));
            let dj = (sph_bessel_j(n, x) - f * bessel_j(n as f64 + 0.5, x)).abs();
            let dy = (sph_bessel_y(n, x) - f * bessel_y(n as f64 + 0.5, x)).abs();
            assert!(dj <= 1e-14 * m && dy <= 1e-14 * m, "{n} {x}");
        }
    }
}

#[test]
fn monotone_in_the_order() {
    // K_ν(x) grows with ν, I_ν(x) falls, for x > 0.
    for &x in &[0.2, 2.0, 20.0] {
        let mut k = 0.0;
        let mut i = f64::INFINITY;
        for j in 0..40 {
            let nu = 0.25 * j as f64;
            let (kk, ii) = (bessel_k(nu, x), bessel_i(nu, x));
            assert!(kk >= k && ii <= i, "{nu} {x}");
            k = kk;
            i = ii;
        }
    }
}

#[test]
fn very_large_orders() {
    use core::f64::consts::PI;
    // Olver's expansion above 10⁷: J_ν(ν) ≈ 2^(1/3)/(3^(2/3)·Γ(2/3)·ν^(1/3)), Y_ν(ν) = −√3 times
    // that, to O(ν^(−2/3)).
    for &nu in &[2.0e7f64, 1.0e9, 3.0e12] {
        let c = 0.447_307_318_396_472_3 / nu.cbrt();
        let (j, y) = (bessel_j(nu, nu), bessel_y(nu, nu));
        assert!((j / c - 1.0).abs() < 2e-4, "{nu}: {j} {c}");
        assert!((y / (-3f64.sqrt() * c) - 1.0).abs() < 2e-4, "{nu}: {y}");
        // The Wronskian, to the stated bound (a few max(ν, x)·ε of the modulus).
        for &z in &[1.0f64, 1.000_01, 1.3, 40.0] {
            let x = nu * z;
            let w =
                bessel_j(nu, x) * bessel_y(nu + 1.0, x) - bessel_j(nu + 1.0, x) * bessel_y(nu, x);
            let t = -2.0 / (PI * x);
            assert!(
                (w / t - 1.0).abs() < 1e-14 * x.max(1.0),
                "{nu} {z}: {w} {t}"
            );
        }
        // Well below the order J underflows and Y overflows.
        assert_eq!(bessel_j(nu, 0.5 * nu), 0.0);
        assert_eq!(bessel_y(nu, 0.5 * nu), f64::NEG_INFINITY);
    }
    // Debye's expansions of I and K (from ν = 50): the Wronskian
    // I_ν K_(ν+1) + I_(ν+1) K_ν = 1/x, of the scaled functions, which stay in range here.
    for &nu in &[60.0f64, 300.5, 1000.0] {
        for &z in &[1.0f64, 3.0, 10.0] {
            let x = nu * z;
            let w = bessel_i_scaled(nu, x) * bessel_k_scaled(nu + 1.0, x)
                + bessel_i_scaled(nu + 1.0, x) * bessel_k_scaled(nu, x);
            assert!((w * x - 1.0).abs() < 1e-14, "{nu} {z}: {}", w * x);
        }
    }
}

#[test]
fn the_spherical_functions_overflow_to_infinity_at_tiny_arguments() {
    // yₙ(x) ~ −(2n − 1)!!/x^(n+1) and kₙ(x) ~ (π/2)·(2n − 1)!!/x^(n+1): ∓∞ where that overflows,
    // never NaN; finite just above.
    let tiny = [
        (0u32, 1.922_924_643_533_27e-310f64),
        (1, 7.0e-155),
        (2, 2.0e-103),
        (12, 3.751_632_369_503_705_4e-187),
        (57, 3.487_034_011_196_528_6e-285),
        (300, 1.0e-100),
        (5, 5.0e-324),
    ];
    for (n, x) in tiny {
        assert_eq!(sph_bessel_y(n, x), f64::NEG_INFINITY, "y_{n}({x:e})");
        assert_eq!(sph_bessel_k(n, x), f64::INFINITY, "k_{n}({x:e})");
        assert_eq!(
            sph_bessel_y(n, -x),
            sph_bessel_y(n, x) * if n % 2 == 0 { -1.0 } else { 1.0 }
        );
    }
    // y₁(x) = −1/x² − 1/2 + O(x²), k₁(x) = (π/2)·(1/x² + 1/x)·e^(−x).
    let x = 7.5e-155f64;
    let y = sph_bessel_y(1, x);
    assert!(y.is_finite() && (y * x * x + 1.0).abs() < 1e-15, "{y:e}");
    let x = 1.0e-154f64;
    let k = sph_bessel_k(1, x);
    let want = core::f64::consts::FRAC_PI_2 / (x * x);
    assert!((k / want - 1.0).abs() < 1e-15, "{k:e} {want:e}");
    assert_eq!(sph_bessel_y(0, 1.0e-300), -1.0 / 1.0e-300);
}

#[test]
fn huge_orders_give_no_nan_and_stay_within_the_envelope() {
    use core::f64::consts::PI;
    // Olver's expansion above ν = 10⁷, through the Airy functions far out: for x > ν the modulus
    // obeys J² + Y² ≤ 2/(π√(x² − ν²)) (Watson 13.74), where the phase is lost included.
    for (nu, x) in [
        (1.0e18f64, 1.0e33f64),
        (1.0e20, 3.0e34),
        (1.0e20, 1.0e35),
        (1.0e100, 3.0e100),
        (1.0e200, 2.0e200),
        (1.0e300, 1.7e308),
        (2.0e7, 1.0e300),
    ] {
        let env = (2.0 / PI).sqrt() / ((x - nu).sqrt() * (x + nu).sqrt()).sqrt();
        let (j, y) = (bessel_j(nu, x), bessel_y(nu, x));
        assert!(
            j.abs() <= 1.01 * env && y.abs() <= 1.01 * env,
            "{nu:e} {x:e}: {j:e} {y:e} > {env:e}"
        );
    }
    // Debye's exponent at the top of the range: decided, or in range, never NaN.
    for (nu, x) in [
        (1.0e300f64, 1.0e300f64),
        (1.0e300, 6.627_434_193_491_816e299),
        (1.7e308, 1.0),
        (60.0, 1.7e308),
        (1.0e150, 1.7e308),
    ] {
        for v in [
            bessel_i(nu, x),
            bessel_k(nu, x),
            bessel_i_scaled(nu, x),
            bessel_k_scaled(nu, x),
            bessel_j(nu, x),
            bessel_y(nu, x),
        ] {
            assert!(!v.is_nan(), "{nu:e} {x:e}");
        }
    }
    // Far below the order the power series underflows: decided from its prefactor, before the sum
    // (whose k·(ν + k) overflow near the top of the range) and before e^(−x) of a huge x (each
    // gave NaN times 0).
    for (nu, x) in [
        (f64::MAX, 9.937_755_552_368_654e153),
        (f64::MAX, -1.657_623_191_617_270_8e152),
        (-f64::MAX, -1.621_819_681_460_619_8e218),
    ] {
        assert_eq!(bessel_j(nu, x), 0.0, "J({nu:e}, {x:e})");
        assert_eq!(bessel_i(nu, x), 0.0, "I({nu:e}, {x:e})");
        assert_eq!(bessel_i_scaled(nu, x), 0.0, "I scaled({nu:e}, {x:e})");
    }
    for (nu, x) in [
        (
            1.357_285_216_772_864_7e198f64,
            1.587_311_227_697_203_5e62f64,
        ),
        (8.751_566_830_855_348e301, 6.652_930_593_830_105e108),
        (-7.970_463_773_781_016e229, 1.693_231_194_202_729_8e70),
    ] {
        assert_eq!(bessel_i_scaled(nu, x), 0.0, "I scaled({nu:e}, {x:e})");
    }
    // The scaled reflection of I at large negative orders: e^(−x)·K_a beyond the range below
    // x = 0.448a (the sign of sin(aπ)), below it from there, and e^(−x)·I_a alone in range.
    assert_eq!(
        bessel_i_scaled(-251.5f64, 6.727_266_568_798_35e-149),
        f64::NEG_INFINITY
    );
    assert_eq!(bessel_i_scaled(-251.5f32, 1.0e-30), f32::NEG_INFINITY);
    assert_eq!(bessel_i_scaled(-10_000_000.5f64, 2.0e6), f64::INFINITY);
    assert_eq!(bessel_i_scaled(-3_000_000.25f64, 1.2e6), f64::INFINITY);
    assert_eq!(bessel_i_scaled(-100_000.5f64, 5.5e4), 0.0);
    let (a, x) = (1000.5f64, 2.0e6);
    let want = bessel_i_scaled(a, x);
    assert!(want > 0.0 && bessel_i_scaled(-a, x) == want, "{want:e}");
    assert_eq!(bessel_j(1.7e308f64, 1.0), 0.0);
    assert_eq!(bessel_y(1.7e308f64, 1.0), f64::NEG_INFINITY);
    assert_eq!(bessel_i(1.7e308f64, 1.0), 0.0);
    assert_eq!(bessel_k(1.7e308f64, 1.0), f64::INFINITY);
    assert_eq!(bessel_y(1.0e40f64, 1.0e-200), f64::NEG_INFINITY);
    assert_eq!(bessel_k(1.0e40f64, 1.0e-200), f64::INFINITY);
    // The spherical iₙ of orders above 10⁷ follow Debye's I in constant time.
    let (n, x) = (20_000_000u32, 1.3e7f64);
    let i = sph_bessel_i(n, x);
    let k = sph_bessel_k(n, x);
    assert!(i == 0.0 && k == f64::INFINITY, "{i:e} {k:e}");
    // Where Debye's exponent puts iₙ beyond the range by far, ±∞ (it was ∞ times a double-double).
    for (n, x) in [
        (65_754u32, 1.0e7f64),
        (8_260_425, 1.0e7),
        (71_703, 1_646_031.058_576_568_5),
    ] {
        let s = if n % 2 == 0 { 1.0 } else { -1.0 };
        assert_eq!(sph_bessel_i(n, x), f64::INFINITY, "i_{n}({x:e})");
        assert_eq!(sph_bessel_i(n, -x), s * f64::INFINITY, "i_{n}(−{x:e})");
        assert_eq!(sph_bessel_i(n, x as f32), f32::INFINITY, "i_{n}({x:e}) f32");
    }
}

#[test]
fn integer_orders_of_i_and_k_stay_right_where_e_to_the_x_leaves_the_range() {
    // From x ≈ 705.5 K₀ and K₁ are subnormal (0 from 745): the recurrence of K_n starts from the
    // scaled ones with e^(−x) apart, and I_n (orders 35 to 49 take the Wronskian with K_n up to
    // x = 0.6n²) is right below its overflow and +∞ beyond it (references: mpmath, Arb). They
    // were 59,179 ulps off, 14.7 subnormal ulps off and NaN.
    let i = bessel_i(39.0f64, 715.0);
    let e = ulps(
        i,
        (1.707_055_217_967_512_4e308, -6.712_344_071_948_752e291),
        F64,
    );
    assert!(e <= 6.0, "I_39(715) = {i:e}: {e:.2} ulps");
    // K_49(740) = 19.7466 times the least subnormal.
    let k = bessel_k(49.0f64, 740.0) / 5e-324;
    assert!(
        (k - 19.746_623_881_211_89).abs() <= 1.0,
        "K_49(740) = {k} subnormal ulps"
    );
    for (n, x) in [
        (39.0f64, 826.403_758_284_107_3f64),
        (46.0, 1059.405),
        (49.0, 1440.0),
    ] {
        assert_eq!(bessel_i(n, x), f64::INFINITY, "I_{n}({x})");
        assert_eq!(bessel_i(-n, x), f64::INFINITY, "I_−{n}({x})");
        assert_eq!(
            bessel_i(n as f32, x as f32),
            f32::INFINITY,
            "I_{n}({x}) f32"
        );
        assert_eq!(bessel_k(n, x), 0.0, "K_{n}({x})");
    }
}

#[test]
fn the_scaled_debye_forms_keep_their_bound_at_the_top_of_the_range() {
    // x near f64::MAX and ν/x below 2⁻⁵¹⁰, where (ν/x)² is subnormal: the exponent's term
    // xν²/(r(r + x)) from factors that stay normal (references: Debye's and Hankel's series at
    // 2000 bits). They were 3.41, 2.44 and 2.24 ulps off.
    let cases = [
        (
            9.022_384_834_968_027e149f64,
            1.648_196_539_696_38e308f64,
            (3.107_460_134_034_769_3e-155, -1.677_463_588_001_679_3e-171),
            false,
        ),
        (
            1.110_923_287_225_908_1e153,
            1.168_570_240_878_125_8e308,
            (3.671_041_710_310_453_6e-155, 1.825_456_624_454_554_2e-171),
            false,
        ),
        (
            5.653_783_006_465_378e153,
            9.867_269_418_738_659e307,
            (1.483_565_853_257_099_9e-154, -3.931_059_450_792_981_3e-171),
            true,
        ),
    ];
    for (nu, x, want, k) in cases {
        let got = if k {
            bessel_k_scaled(nu, x)
        } else {
            bessel_i_scaled(nu, x)
        };
        let e = ulps(got, want, F64);
        assert!(e <= 1.0, "{nu:e} {x:e}: {got:e}, {e:.2} ulps");
    }
}

#[test]
fn beyond_minus_1e200_ai_and_bi_are_0_and_their_derivatives_nan() {
    // The documented end of the negative axis: ζ = (2/3)|x|^(3/2) nears the overflow and no
    // phase is left; Ai and Bi are within their modulus |x|^(−1/4)/√π there, the derivatives'
    // modulus grows without bound. Up to −10²⁰⁰ the values are finite and within the modulus.
    let fs: [fn(f64) -> f64; 4] = [airy_ai, airy_bi, airy_ai_prime, airy_bi_prime];
    for x in [
        -1.000_000_000_000_001e200f64,
        -1.0e250,
        -f64::MAX,
        f64::NEG_INFINITY,
    ] {
        assert_eq!(airy_ai(x), 0.0, "Ai({x:e})");
        assert_eq!(airy_bi(x), 0.0, "Bi({x:e})");
        assert!(airy_ai_prime(x).is_nan(), "Ai′({x:e})");
        assert!(airy_bi_prime(x).is_nan(), "Bi′({x:e})");
    }
    let x = -1.0e200f64;
    for (k, f) in fs.iter().enumerate() {
        let v = f(x);
        // √(Ai² + Bi²) and √(Ai′² + Bi′²) are |x|^(∓1/4)/√π to within 10⁻²⁰⁰ here.
        let m = if k < 2 { 1.0e-50 } else { 1.0e50 } / core::f64::consts::PI.sqrt();
        assert!(v.is_finite() && v.abs() <= m * (1.0 + 1e-15), "{k}: {v:e}");
    }
    assert_eq!(airy_ai(f32::NEG_INFINITY), 0.0);
    assert_eq!(airy_bi(f32::NEG_INFINITY), 0.0);
    assert!(airy_ai_prime(f32::NEG_INFINITY).is_nan() && airy_bi_prime(f32::NEG_INFINITY).is_nan());
}

#[test]
fn the_reflection_keeps_its_terms_apart_where_one_overflows() {
    // J_(−a) = cos(aπ)·J_a − sin(aπ)·Y_a with Y_a beyond the range but the product within it.
    let j = bessel_j(-277.939_742_268_752_87f64, 15.690_829_707_042_946);
    assert!(
        j.is_finite() && (j / -9.230_466_156_031_65e307 - 1.0).abs() < 1e-15,
        "{j:e}"
    );
    let i = bessel_i_scaled(-279.084_485_589_302_46f64, 15.487_150_338_544_55);
    assert!(
        i.is_finite() && (i / -3.553_686_446_514_410_6e304 - 1.0).abs() < 1e-15,
        "{i:e}"
    );
}
