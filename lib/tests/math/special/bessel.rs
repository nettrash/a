//! The bessel area of issue #20: its data in `data/bessel_*.txt`, made by `data/gen_bessel.py`.
//!
//! Each data set is checked by regions, each against its stated bound: in ulps of the value, or
//! (for `J`, `Y` and the spherical `j`, `y` of the general orders, whose zeros no method of
//! this kind resolves) in ulps of the modulus `M = √(J² + Y²)`, the second reference of a case.

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

#[test]
fn bessel_j_f64() {
    check(
        "J f64",
        J_F64,
        F64,
        |a| bessel_j(a[0], a[1]),
        |a| {
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
        |a| {
            order_region(
                a[0],
                a[1],
                false,
                ("ν = 0, 1", 1.0, false),
                ("general ν", 6.0, false),
            )
        },
    );
}

#[test]
fn bessel_k_f64() {
    check(
        "K f64",
        K_F64,
        F64,
        |a| bessel_k(a[0], a[1]),
        |a| {
            order_region(
                a[0],
                a[1],
                false,
                ("ν = 0, 1", 1.0, false),
                ("general ν", 6.0, false),
            )
        },
    );
}

#[test]
fn bessel_i_scaled_f64() {
    check(
        "I scaled f64",
        IS_F64,
        F64,
        |a| bessel_i_scaled(a[0], a[1]),
        |a| {
            order_region(
                a[0],
                a[1],
                false,
                ("ν = 0, 1", 1.0, false),
                ("general ν", 6.0, false),
            )
        },
    );
}

#[test]
fn bessel_k_scaled_f64() {
    check(
        "K scaled f64",
        KS_F64,
        F64,
        |a| bessel_k_scaled(a[0], a[1]),
        |a| {
            order_region(
                a[0],
                a[1],
                false,
                ("ν = 0, 1", 1.0, false),
                ("general ν", 6.0, false),
            )
        },
    );
}

/// `f32` results: correctly rounded but within a hair, and the general `J`, `Y` near their zeros
/// in ulps of the modulus.
fn f32_region(nu: f64, jy: bool) -> Region {
    if jy && nu != 0.0 && nu != 1.0 {
        ("f32, ulps of M", 0.51, true)
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
        |a| f32_region(a[0], true),
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
        |a| f32_region(a[0], false),
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
        |a| f32_region(a[0], false),
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
        sph,
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
/// `ζ = (2/3)|x|^(3/2)`, known to `2⁻¹⁰⁴·ζ` only.
fn airy_region(x: f64) -> Region {
    if x >= -8.4 {
        ("x ≥ −8.4", 1.5, false)
    } else if x >= -1.0e5 {
        ("−10⁵ ≤ x < −8.4, ulps of M", 1.0, true)
    } else {
        ("−10⁹ ≤ x < −10⁵, ulps of M", 1.0, true)
    }
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
}

#[test]
fn airy_f32() {
    let r = |a: &[f64]| {
        if a[0] >= -8.4 {
            ("f32", 0.51, false)
        } else {
            ("f32, x < −8.4, ulps of M", 0.51, true)
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
