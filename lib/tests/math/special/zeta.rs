//! The zeta area of issue #20: its data in `data/zeta_*.txt`, made by `data/gen_zeta.py`.
//!
//! Each data test checks the stated bound of a function on each region of its domain (the
//! regions as in `docs/math/special.md`). The bounds of the real zeta function left of −1/2, of
//! the complex one's reflection and of the polylogarithm past −2 and of negative orders rest on
//! the gamma area's `gamma_f64` and `gamma_c64`.

use super::common::*;
use a::math::complex::Complex;
use a::math::special::{
    dilog, expint_e1, expint_ei, expint_en, hurwitz_zeta, lambert_w0, lambert_wm1, polylog, zeta,
};

macro_rules! file {
    ($name:literal) => {
        include_str!(concat!("data/zeta_", $name, ".txt"))
    };
}

/// The cases of a data file whose arguments satisfy `keep`, every 37th of them under Miri.
fn data(text: &str, keep: impl Fn(&[f64]) -> bool) -> String {
    let word = |w: &str| f64::from_bits(u64::from_str_radix(w, 16).unwrap());
    let lines = text
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty());
    let kept: Vec<&str> = lines
        .filter(|l| {
            let args: Vec<f64> = l
                .split(':')
                .next()
                .unwrap()
                .split_whitespace()
                .map(word)
                .collect();
            keep(&args)
        })
        .collect();
    let step = if cfg!(miri) { 37 } else { 1 };
    kept.into_iter()
        .step_by(step)
        .collect::<Vec<_>>()
        .join("\n")
}

/// The bound `b`, widened under Miri, which perturbs the results of `exp`, `ln`, `powf` and the
/// like by a few units at random (its runs check for undefined behaviour, not for accuracy).
fn ub(b: f64) -> f64 {
    if cfg!(miri) { 64.0 * b.max(1.0) } else { b }
}

/// `k` units of `f64::EPSILON`, widened under Miri as [`ub`].
fn eps(k: f64) -> f64 {
    if cfg!(miri) {
        64.0 * k * f64::EPSILON
    } else {
        k * f64::EPSILON
    }
}

/// `a == b`, or within the noise of Miri's transcendental functions under it (the two sides
/// compute their `exp`s and `ln`s apart).
#[track_caller]
fn same(a: f64, b: f64) {
    if cfg!(miri) {
        assert!((a - b).abs() <= 1e-12 * b.abs(), "{a} != {b}");
    } else {
        assert_eq!(a, b);
    }
}

/// All the cases (a subset under Miri).
fn all(text: &str) -> String {
    data(text, |_| true)
}

#[test]
fn lambert_w0_data() {
    let f = file!("lambertw0_f64");
    let e = check_real("W0 f64", &all(f), F64, ub(1.0), |a| lambert_w0(a[0]));
    let e32 = check_real(
        "W0 f32",
        &all(file!("lambertw0_f32")),
        F32,
        ub(0.501),
        |a| lambert_w0(a[0] as f32) as f64,
    );
    eprintln!("W0: f64 {e:.3} ulps, f32 {e32:.3}");
}

#[test]
fn lambert_wm1_data() {
    let e = check_real(
        "W-1 f64",
        &all(file!("lambertwm1_f64")),
        F64,
        ub(1.1),
        |a| lambert_wm1(a[0]),
    );
    let e32 = check_real(
        "W-1 f32",
        &all(file!("lambertwm1_f32")),
        F32,
        ub(0.501),
        |a| lambert_wm1(a[0] as f32) as f64,
    );
    eprintln!("W-1: f64 {e:.3} ulps, f32 {e32:.3}");
}

#[test]
fn lambert_special_values() {
    let inv_e = -1.0 / core::f64::consts::E;
    assert_eq!(lambert_w0(0.0f64), 0.0);
    assert!(lambert_w0(-0.0f64).is_sign_negative());
    assert_eq!(lambert_w0(inv_e), -1.0);
    assert_eq!(lambert_wm1(inv_e), -1.0);
    assert_eq!(lambert_w0(-1.0f32 / core::f32::consts::E), -1.0);
    assert!(lambert_w0(-0.37f64).is_nan() && lambert_wm1(-0.37f64).is_nan());
    assert!(lambert_wm1(0.1f64).is_nan() && lambert_w0(f64::NAN).is_nan());
    assert_eq!(lambert_wm1(0.0f64), f64::NEG_INFINITY);
    assert_eq!(lambert_w0(f64::INFINITY), f64::INFINITY);
    assert!((lambert_w0(core::f64::consts::E) - 1.0).abs() <= eps(1.0));
    // w eʷ = x on both branches
    for &x in &[-0.3f64, -0.1, -1e-5, -1e-300] {
        for w in [lambert_w0(x), lambert_wm1(x)] {
            let tol = eps(4.0) * x.abs() * (1.0 + w.abs());
            assert!((w * w.exp() - x).abs() <= tol, "{x} {w}");
        }
    }
    // monotone: W0 increasing, W-1 decreasing
    let xs: Vec<f64> = (1..200).map(|i| -0.3678 + i as f64 * 0.0018).collect();
    for p in xs.windows(2) {
        assert!(lambert_w0(p[0]) < lambert_w0(p[1]));
        if p[1] < 0.0 {
            assert!(lambert_wm1(p[0]) > lambert_wm1(p[1]));
        }
    }
}

#[test]
fn zeta_real_data() {
    let f = file!("zeta_f64");
    let e = check_real(
        "zeta f64, s >= 0",
        &data(f, |a| a[0] >= 0.0 || a[0].is_nan()),
        F64,
        ub(0.7),
        |a| zeta(a[0]),
    );
    let em = check_real(
        "zeta f64, -1/2 <= s < 0",
        &data(f, |a| (-0.5..0.0).contains(&a[0])),
        F64,
        ub(1.05),
        |a| zeta(a[0]),
    );
    // the reflection, through Γ(1 − s)
    let er = check_real(
        "zeta f64, s < -1/2",
        &data(f, |a| a[0] < -0.5),
        F64,
        ub(3.5),
        |a| zeta(a[0]),
    );
    let e32 = check_real("zeta f32", &all(file!("zeta_f32")), F32, ub(0.501), |a| {
        zeta(a[0] as f32) as f64
    });
    eprintln!("zeta: f64 {e:.3} ulps ([-1/2, 0): {em:.3}; s < -1/2: {er:.3}), f32 {e32:.3}");
}

#[test]
fn zeta_special_values() {
    let pi = core::f64::consts::PI;
    assert_eq!(zeta(1.0f64), f64::INFINITY);
    assert_eq!(zeta(0.0f64), -0.5);
    assert_eq!(zeta(2.0f64), pi * pi / 6.0);
    assert_eq!(zeta(f64::INFINITY), 1.0);
    assert_eq!(zeta(60.0f64), 1.0);
    assert!(zeta(f64::NEG_INFINITY).is_nan() && zeta(f64::NAN).is_nan());
    for k in 1..200 {
        assert_eq!(zeta(-2.0 * k as f64), 0.0, "trivial zero -{}", 2 * k);
    }
    assert_eq!(zeta(-1000.5f64).abs(), f64::INFINITY);
    assert!((zeta(4.0f64) - pi.powi(4) / 90.0).abs() <= eps(1.0));
    // decreasing on (1, ∞), increasing on (−∞, 1) left of the first trivial zero's neighbourhood
    let mut last = f64::INFINITY;
    for i in 1..300 {
        let v = zeta(1.0 + i as f64 * 0.1);
        assert!(v <= last);
        last = v;
    }
    // the real and complex forms agree on the real axis
    let z = zeta(Complex::new(3.5f64, 0.0));
    same(z.re, zeta(3.5f64));
    assert_eq!(z.im, 0.0);
}

/// Whether the complex zeta function takes the Euler–Maclaurin formula at `s` (as the source
/// chooses; the reflection's points rest on `Γ`).
fn by_em(sigma: f64, t: f64) -> bool {
    let t = t.abs();
    t == 0.0 || sigma >= 0.5 || t > 200.0 || (sigma >= -3.0 && t >= 5.0 - 30.0 * sigma.min(0.0))
}

#[test]
fn zeta_complex_data() {
    let f = file!("zetac_c64");
    let zc = |re: f64, im: f64| {
        let z = zeta(Complex::new(re, im));
        (z.re, z.im)
    };
    // relative to |ζ| (≥ 10⁻³ in the data) right of the critical strip; left of it the
    // Euler–Maclaurin terms cancel as 2.2^(−σ)
    let right = |a: &[f64]| a[0] > 1.0;
    let e = check_complex("zeta c64, σ > 1", &data(f, right), F64, ub(2.0), zc);
    let left = |a: &[f64]| by_em(a[0], a[1]) && a[0] < 0.0;
    let el = check_complex("zeta c64, EM, σ < 0", &data(f, left), F64, ub(8.0), zc);
    // in the strip, relative to max(|ζ|, 1): absolute where |ζ| < 1
    let strip = |a: &[f64]| by_em(a[0], a[1]) && (0.0..=1.0).contains(&a[0]);
    let mut es = 0.0f64;
    for c in &cases(&data(f, strip)) {
        let (re, im) = zc(c.args[0], c.args[1]);
        let d = (re - c.refs[0].0 - c.refs[0].1).hypot(im - c.refs[1].0 - c.refs[1].1);
        let n = c.refs[0].0.hypot(c.refs[1].0).max(1.0);
        es = es.max(d / (f64::EPSILON * n));
    }
    assert!(
        es <= ub(3.5),
        "zeta c64 in the strip: {es} ε of max(|ζ|, 1)"
    );
    let er = check_complex(
        "zeta c64, reflection",
        &data(f, |a| !by_em(a[0], a[1])),
        F64,
        ub(6.5),
        zc,
    );
    let e32 = check_complex(
        "zeta c32",
        &all(file!("zetac_c32")),
        F32,
        ub(0.6),
        |re, im| {
            let z = zeta(Complex::new(re as f32, im as f32));
            (z.re as f64, z.im as f64)
        },
    );
    // near the zeros on the critical line: an absolute bound, in ε
    let cases = cases(&all(file!("zetaczeros_c64")));
    let mut worst = 0.0f64;
    for c in &cases {
        let (re, im) = zc(c.args[0], c.args[1]);
        let d = (re - c.refs[0].0 - c.refs[0].1).hypot(im - c.refs[1].0 - c.refs[1].1);
        worst = worst.max(d / f64::EPSILON);
    }
    assert!(
        worst <= ub(2.5),
        "zeta c64 near the zeros: {worst} ε absolute"
    );
    eprintln!(
        "zeta c64: {e:.3} ε for σ > 1, {es:.3} in the strip, {el:.3} left of it, near zeros \
         {worst:.3} ε absolute, reflection {er:.3}; c32 {e32:.3}"
    );
}

#[test]
fn zeta_complex_symmetry_and_specials() {
    let s = Complex::new(0.3f64, 17.25);
    let (a, b) = (zeta(s), zeta(s.conj()));
    same(a.re, b.re);
    same(a.im, -b.im);
    let z = zeta(Complex::new(f64::NAN, 1.0));
    assert!(z.re.is_nan() && z.im.is_nan());
    // the first zero
    let z = zeta(Complex::new(0.5f64, 14.134_725_141_734_694));
    assert!(z.abs() < 100.0 * eps(1.0), "{z:?}");
    // ζ(s) ~ 1/(s − 1) at the pole
    let e = 1e-9f64;
    let z = zeta(Complex::new(1.0, e));
    assert!((z.im + 1.0 / e).abs() < 1e-6 * (1.0 / e));
}

#[test]
fn hurwitz_data() {
    let e = check_real(
        "hurwitz f64",
        &all(file!("hurwitz_f64")),
        F64,
        ub(1.35),
        |a| hurwitz_zeta(a[0], a[1]),
    );
    let e32 = check_real(
        "hurwitz f32",
        &all(file!("hurwitz_f32")),
        F32,
        ub(0.501),
        |a| hurwitz_zeta(a[0] as f32, a[1] as f32) as f64,
    );
    eprintln!("hurwitz: f64 {e:.3} ulps, f32 {e32:.3}");
}

#[test]
fn hurwitz_identities() {
    // ζ(s, 1) = ζ(s); ζ(s, 1/2) = (2ˢ − 1) ζ(s); ζ(s, a) = a⁻ˢ + ζ(s, a + 1)
    for &s in &[1.5f64, 2.0, 3.7, 10.0, 40.0] {
        same(hurwitz_zeta(s, 1.0), zeta(s));
        let h = hurwitz_zeta(s, 0.5);
        assert!(
            (h / ((s.exp2() - 1.0) * zeta(s)) - 1.0).abs() < eps(4.0),
            "{s}"
        );
        for &a in &[0.3f64, 2.5, 17.0] {
            let l = hurwitz_zeta(s, a);
            let r = a.powf(-s) + hurwitz_zeta(s, a + 1.0);
            assert!((l / r - 1.0).abs() < eps(4.0), "{s} {a}");
        }
    }
    assert!(hurwitz_zeta(0.5f64, 1.0).is_nan() && hurwitz_zeta(2.0f64, 0.0).is_nan());
    assert!(hurwitz_zeta(2.0f64, -1.0).is_nan() && hurwitz_zeta(f64::NAN, 1.0).is_nan());
    assert_eq!(hurwitz_zeta(1.0f64, 3.0), f64::INFINITY);
    assert_eq!(hurwitz_zeta(2.0f64, f64::INFINITY), 0.0);
}

#[test]
fn dilog_data() {
    let e = check_real("dilog f64", &all(file!("dilog_f64")), F64, ub(1.3), |a| {
        dilog(a[0])
    });
    let e32 = check_real("dilog f32", &all(file!("dilog_f32")), F32, ub(0.501), |a| {
        dilog(a[0] as f32) as f64
    });
    eprintln!("dilog: f64 {e:.3} ulps, f32 {e32:.3}");
}

#[test]
fn dilog_identities() {
    let pi2 = core::f64::consts::PI * core::f64::consts::PI;
    assert_eq!(dilog(0.0f64), 0.0);
    assert_eq!(dilog(1.0f64), pi2 / 6.0);
    assert!((dilog(-1.0f64) + pi2 / 12.0).abs() <= eps(1.0));
    let l2 = core::f64::consts::LN_2;
    assert!((dilog(0.5f64) - (pi2 / 12.0 - l2 * l2 / 2.0)).abs() <= eps(1.0));
    assert_eq!(dilog(f64::NEG_INFINITY), f64::NEG_INFINITY);
    assert_eq!(dilog(f64::INFINITY), f64::NEG_INFINITY);
    assert!(dilog(f64::NAN).is_nan());
    // Li₂(x) + Li₂(−x) = Li₂(x²)/2
    for &x in &[0.1f64, 0.3, 0.7, 0.95] {
        let l = dilog(x) + dilog(-x);
        assert!((l - dilog(x * x) / 2.0).abs() < eps(4.0), "{x}");
    }
    // the real part is continuous at 1
    assert!((dilog(1.0 + 1e-12f64) - pi2 / 6.0).abs() < 1e-9);
}

/// The regions of the polylogarithm's stated bounds, in order (a case belongs to the first that
/// holds it): a name, the test of `(s, x)`, the bound, and whether it is in units of the
/// condition number `κ = (|x ∂ₓLi| + |s ∂ₛLi|)/|Li|` (the data's second reference, for `s < 0`,
/// where `Li_s` has real zeros on the negative axis).
type Region = (&'static str, fn(f64, f64) -> bool, f64, bool);
const POLYLOG_REGIONS: [Region; 8] = [
    // the series, the expansions about 1 and −1
    ("s >= 1, x >= -2", |s, x| s >= 1.0 && x >= -2.0, 3.0, false),
    // the expansion about −1 cancels a little for small orders
    (
        "0 <= s < 1, x >= -2",
        |s, x| (0.0..1.0).contains(&s) && x >= -2.0,
        8.5,
        false,
    ),
    // the inversion of the integer orders
    (
        "integers s >= 1, x < -2",
        |s, x| s >= 1.0 && s == s.round() && x < -2.0,
        2.0,
        false,
    ),
    // the integral of Fermi and Dirac over Γ(s)
    ("s > 0, x < -2", |s, x| s > 0.0 && x < -2.0, 3.0, false),
    // the Eulerian polynomials
    (
        "integers -64 <= s < 0, x >= 0",
        |s, x| s == s.round() && (-64.0..0.0).contains(&s) && x >= 0.0,
        2.5,
        false,
    ),
    (
        "integers -64 <= s < 0, x < 0",
        |s, _| s == s.round() && (-64.0..0.0).contains(&s),
        2.5,
        true,
    ),
    // the Hurwitz zeta function of the complex a, times Γ(1 − s), in units of κ
    ("s < 0, x < 0", |s, x| s < 0.0 && x < 0.0, 8.5, true),
    // Γ(1 − s) and the reflection of ζ(s − k)
    ("s < 0, x > 0", |_, _| true, 16.0, false),
];

#[test]
fn polylog_data() {
    let text = all(file!("polylog_f64"));
    let cs = cases(&text);
    let mut seen = vec![false; cs.len()];
    for (name, inside, bound, by_cond) in POLYLOG_REGIONS {
        let mut worst = 0.0f64;
        let mut at = (0.0, 0.0);
        for (c, done) in cs.iter().zip(seen.iter_mut()) {
            if *done || !inside(c.args[0], c.args[1]) {
                continue;
            }
            *done = true;
            let mut e = ulps(polylog(c.args[0], c.args[1]), c.refs[0], F64);
            if by_cond {
                e /= c.refs[1].0.max(1.0);
            }
            if e > worst || e.is_nan() {
                worst = e;
                at = (c.args[0], c.args[1]);
            }
        }
        assert!(
            worst <= ub(bound),
            "polylog, {name}: {worst} > {bound} at {at:?}"
        );
        let unit = if by_cond { "κ" } else { "ulps" };
        eprintln!("polylog, {name}: {worst:.3} {unit} at {at:?}");
    }
    assert!(seen.iter().all(|&d| d));
    let e32 = check_real(
        "polylog f32",
        &data(file!("polylog_f32"), |a| a[0] >= 0.0),
        F32,
        ub(0.6),
        |a| polylog(a[0] as f32, a[1] as f32) as f64,
    );
    eprintln!("polylog f32 (s ≥ 0): {e32:.3}");
}

#[test]
fn polylog_identities() {
    let x = 0.37f64;
    same(polylog(1.0, x), -(-x).ln_1p());
    assert_eq!(polylog(0.0, x), x / (1.0 - x));
    same(polylog(2.0, x), dilog(x));
    assert_eq!(polylog(-1.0, x), x / ((1.0 - x) * (1.0 - x)));
    same(polylog(3.0f64, 1.0), zeta(3.0f64));
    assert_eq!(polylog(0.5f64, 1.0), f64::INFINITY);
    assert!(polylog(2.5f64, 1.5).is_nan() && polylog(f64::NAN, 0.5).is_nan());
    assert_eq!(polylog(2.5f64, 0.0), 0.0);
    // Li_s(x) + Li_s(−x) = 2^(1−s) Li_s(x²)
    for &s in &[0.5f64, 1.7, 3.0, 4.2] {
        for &x in &[0.3f64, 0.6, 0.9] {
            let l = polylog(s, x) + polylog(s, -x);
            let r = (1.0 - s).exp2() * polylog(s, x * x);
            assert!((l - r).abs() < eps(8.0) * r.abs().max(1.0), "{s} {x}");
        }
    }
    // continuity across the integer orders (the pole pair in closed form)
    for &n in &[1.0f64, 2.0, 3.0, 5.0] {
        for &x in &[0.6f64, 0.9, 0.99] {
            let (a, b) = (polylog(n - 1e-12, x), polylog(n + 1e-12, x));
            assert!((a - b).abs() < 1e-10 * a.abs().max(1.0), "{n} {x}: {a} {b}");
        }
    }
}

#[test]
fn expint_data() {
    let e1 = check_real("E1 f64", &all(file!("e1_f64")), F64, ub(1.1), |a| {
        expint_e1(a[0])
    });
    let ei = check_real("Ei f64", &all(file!("ei_f64")), F64, ub(1.8), |a| {
        expint_ei(a[0])
    });
    let en = check_real("En f64", &all(file!("en_f64")), F64, ub(1.05), |a| {
        expint_en(a[0] as u32, a[1])
    });
    let e132 = check_real("E1 f32", &all(file!("e1_f32")), F32, ub(0.501), |a| {
        expint_e1(a[0] as f32) as f64
    });
    let ei32 = check_real("Ei f32", &all(file!("ei_f32")), F32, ub(0.501), |a| {
        expint_ei(a[0] as f32) as f64
    });
    let en32 = check_real("En f32", &all(file!("en_f32")), F32, ub(0.501), |a| {
        expint_en(a[0] as u32, a[1] as f32) as f64
    });
    eprintln!("E1 {e1:.3}, Ei {ei:.3}, En {en:.3} ulps; f32 {e132:.3}, {ei32:.3}, {en32:.3}");
}

#[test]
fn expint_special_values() {
    assert_eq!(expint_e1(0.0f64), f64::INFINITY);
    assert!(expint_e1(-1.0f64).is_nan() && expint_e1(f64::NAN).is_nan());
    assert_eq!(expint_e1(800.0f64), 0.0);
    assert_eq!(expint_ei(0.0f64), f64::NEG_INFINITY);
    assert_eq!(expint_ei(800.0f64), f64::INFINITY);
    assert_eq!(expint_ei(-800.0f64), 0.0);
    same(expint_ei(-2.5f64), -expint_e1(2.5f64));
    same(expint_en(0, 2.0f64), (-2.0f64).exp() / 2.0);
    same(expint_en(1, 2.0f64), expint_e1(2.0f64));
    assert_eq!(expint_en(5, 0.0f64), 0.25);
    assert_eq!(expint_en(0, 0.0f64), f64::INFINITY);
    assert_eq!(expint_en(1, 0.0f64), f64::INFINITY);
    assert!(expint_en(3, -1.0f64).is_nan());
    assert_eq!(expint_en(3, f64::INFINITY), 0.0);
    // n Eₙ₊₁(x) = e⁻ˣ − x Eₙ(x)
    for &x in &[0.5f64, 3.0, 12.0] {
        for n in 1..8u32 {
            let l = n as f64 * expint_en(n + 1, x);
            let r = (-x).exp() - x * expint_en(n, x);
            assert!((l - r).abs() < eps(64.0) * l.abs(), "{n} {x}");
        }
    }
    // the zero of Ei
    let x0 = 0.372_507_410_781_366_6f64;
    assert!(expint_ei(x0).abs() < eps(1.0));
}
