//! The zeta area of issue #20: its data in `data/zeta_*.txt`, made by `data/gen_zeta.py`.
//!
//! Each data test checks the stated bound of a function on each region of its domain (the
//! regions as in `docs/math/special.md`). The bounds of the real zeta function left of −1/2, of
//! the complex one's reflection with `Γ(1 − s)` (`−12 < σ < 1/2`, `t < 13.5`) and of the
//! polylogarithm past −2 and of negative orders rest on the gamma area's `gamma_f64` and
//! `gamma_c64`. Every data file ends with the regression set of the audit of 2026-11-21, and
//! some with that of the first review of the fixes (see `data/gen_zeta.py`).

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

/// The error of a complex result in ε of `norm` (the reference's absolute value if `None`); a
/// reference with an infinite part (past the range) needs that part exactly, and the other within
/// the bound of its own size.
fn cerr(g: (f64, f64), re: (f64, f64), im: (f64, f64), eps: f64, norm: Option<f64>) -> f64 {
    if re.0.is_finite() && im.0.is_finite() {
        if g.0.is_nan() || g.1.is_nan() {
            return f64::INFINITY;
        }
        let d = ((g.0 - re.0) - re.1).hypot((g.1 - im.0) - im.1);
        let n = norm.unwrap_or(re.0.hypot(im.0));
        return if n == 0.0 {
            if d == 0.0 { 0.0 } else { f64::INFINITY }
        } else {
            d / n / eps
        };
    }
    let part = |y: f64, r: (f64, f64)| -> f64 {
        if !r.0.is_finite() || r.0 == 0.0 {
            if y == r.0 { 0.0 } else { f64::INFINITY }
        } else {
            ((y - r.0) - r.1).abs() / r.0.abs() / eps
        }
    };
    part(g.0, re).max(part(g.1, im))
}

/// The largest error of the complex zeta function over the cases of `text` that `keep` holds,
/// checked against `bound`; in the strip relative to `max(|ζ|, 1)`.
#[track_caller]
fn check_zeta_c(
    name: &str,
    text: &str,
    single: bool,
    bound: f64,
    keep: impl Fn(&[f64]) -> bool,
) -> f64 {
    let eps = if single {
        f32::EPSILON as f64
    } else {
        f64::EPSILON
    };
    let mut worst = 0.0f64;
    let mut at = (0.0, 0.0);
    let cs = cases(&data(text, keep));
    assert!(!cs.is_empty(), "{name}: no cases");
    for c in &cs {
        let (sg, t) = (c.args[0], c.args[1]);
        let g = if single {
            let z = zeta(Complex::new(sg as f32, t as f32));
            (z.re as f64, z.im as f64)
        } else {
            let z = zeta(Complex::new(sg, t));
            (z.re, z.im)
        };
        let strip = (0.0..=1.0).contains(&sg);
        let norm = if strip {
            Some(c.refs[0].0.hypot(c.refs[1].0).max(1.0))
        } else {
            None
        };
        // for f32, a reference past its range is that infinity
        let fit = |r: (f64, f64)| -> (f64, f64) {
            if single && r.0.abs() > f32::MAX as f64 {
                (f64::INFINITY.copysign(r.0), 0.0)
            } else {
                r
            }
        };
        let e = cerr(g, fit(c.refs[0]), fit(c.refs[1]), eps, norm);
        if e > worst || e.is_nan() {
            worst = e;
            at = (sg, t);
        }
    }
    assert!(worst <= bound, "{name}: {worst} ε > {bound} at {at:?}");
    worst
}

#[test]
fn zeta_complex_data() {
    let f = file!("zetac_c64");
    let t = |a: &[f64]| a[1].abs();
    // relative to |ζ| off the strip: the Euler–Maclaurin formula, the Dirichlet series past
    // σ = 10, the Riemann–Siegel formula past t = 10⁴, 1 + 2⁻ˢ past σ = 56
    let e = check_zeta_c("zeta c64, σ > 1", f, false, ub(2.0), |a| a[0] > 1.0);
    // in the strip relative to max(|ζ|, 1), by the methods: below t = 13.5 the series about the
    // pole and the reflection with Γ(1 − s); to t = 10⁴ the Euler–Maclaurin formula (σ ≥ 1/2)
    // and its reflection (σ < 1/2), whose χ(s) ≈ (t/2π)^(1/2 − σ) multiplies the errors of its
    // terms (double-doubles in the strip); past it the Riemann–Siegel formula, at σ itself and
    // its terms double-doubles (about √(t/2π) of them, whose roundings in double reached 7 ε at
    // t = 10¹⁰, and 22 ε with the reflection's rounded 1 − σ)
    let strip = |a: &[f64]| (0.0..=1.0).contains(&a[0]);
    let es0 = check_zeta_c(
        "zeta c64, 0 ≤ σ ≤ 1, t < 13.5",
        f,
        false,
        ub(3.0),
        |a| strip(a) && t(a) < 13.5,
    );
    let es1 = check_zeta_c(
        "zeta c64, 0 ≤ σ ≤ 1, 13.5 ≤ t ≤ 1e4",
        f,
        false,
        ub(3.0),
        |a| strip(a) && (13.5..=1e4).contains(&t(a)),
    );
    let es2 = check_zeta_c(
        "zeta c64, 0 ≤ σ ≤ 1, t > 1e4",
        f,
        false,
        ub(3.0),
        |a| strip(a) && t(a) > 1e4,
    );
    // left of the strip the reflection: with Γ(1 − s) for −12 < σ < 0 and t < 13.5, its
    // factors and products double-doubles, else in its logarithmic form
    let el0 = check_zeta_c("zeta c64, σ < 0, t < 13.5", f, false, ub(2.5), |a| {
        a[0] < 0.0 && t(a) < 13.5
    });
    let el1 = check_zeta_c("zeta c64, σ < 0, t ≥ 13.5", f, false, ub(2.5), |a| {
        a[0] < 0.0 && t(a) >= 13.5
    });
    let e32 = check_zeta_c("zeta c32", file!("zetac_c32"), true, ub(0.6), |_| true);
    // near the zeros on the critical line: an absolute bound, in ε
    let cases = cases(&all(file!("zetaczeros_c64")));
    let mut worst = 0.0f64;
    for c in &cases {
        let z = zeta(Complex::new(c.args[0], c.args[1]));
        let d = (z.re - c.refs[0].0 - c.refs[0].1).hypot(z.im - c.refs[1].0 - c.refs[1].1);
        worst = worst.max(d / f64::EPSILON);
    }
    assert!(
        worst <= ub(2.5),
        "zeta c64 near the zeros: {worst} ε absolute"
    );
    eprintln!(
        "zeta c64: {e:.3} ε for σ > 1; in the strip {es0:.3} (t < 13.5), {es1:.3} (to 1e4), \
         {es2:.3} (past it); left of it {el0:.3} (t < 13.5), {el1:.3}; near zeros {worst:.3} ε \
         absolute; c32 {e32:.3}"
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
    // and on σ = 1 itself with a tiny t: ζ = γ − i/t − iγ₁t + …, both parts correctly rounded,
    // −i∞ where 1/t leaves the range (the series about the pole squared t, which went subnormal
    // below t = 1.5·10⁻¹⁵⁴: the real part was NaN, and (∞, NaN) below t = 2⁻⁶⁰⁰)
    let below = |t: f64| f64::from_bits(t.to_bits() - 1);
    for t in [
        1e-154f64,
        1e-155,
        1e-160,
        1e-170,
        1e-200,
        2f64.powi(-500),
        below(2f64.powi(-500)),
        1e-300,
        1e-308,
        1e-310,
        5e-324,
    ] {
        for t in [t, -t] {
            let z = zeta(Complex::new(1.0f64, t));
            assert_eq!((z.re, z.im), (0.577_215_664_901_532_9, -1.0 / t), "{t:e}");
        }
    }
    assert_eq!(zeta(Complex::new(1.0f64, 1e-310)).im, f64::NEG_INFINITY);
    // infinite and huge arguments return at once (they looped for ever before): no limit as
    // |t| → ∞, 1 as σ → ∞, NaN past the supported |t| = 2⁴⁰ (σ < 56)
    let nan = |z: Complex<f64>| z.re.is_nan() && z.im.is_nan();
    assert!(nan(zeta(Complex::new(0.5, f64::INFINITY))));
    assert!(nan(zeta(Complex::new(0.5, f64::NEG_INFINITY))));
    assert!(nan(zeta(Complex::new(f64::NEG_INFINITY, 1.0))));
    assert!(nan(zeta(Complex::new(-1e300, 1.0))));
    assert!(nan(zeta(Complex::new(0.5, 1e17))));
    assert_eq!(
        zeta(Complex::new(f64::INFINITY, 1.0)),
        Complex::new(1.0, 0.0)
    );
    assert_eq!(zeta(Complex::new(1e300, 1.0)).re, 1.0);
    let z = zeta(Complex::new(60.0f64, 1e300));
    assert!((z.re - 1.0).abs() <= f64::EPSILON && z.im.abs() < 1e-17);
    // past the range: infinities in the directions of the parts
    let z = zeta(Complex::new(-300.0, 1.0));
    assert_eq!((z.re, z.im), (f64::NEG_INFINITY, f64::NEG_INFINITY));
    // the trivial zeros with a tiny imaginary part: ζ(−2n + it) ≈ it ζ′(−2n), not 0, and not
    // an infinity where t′ ζ′(−2n) at the scaled t′ = m·2⁻⁶⁰⁰ would overflow
    let z = zeta(Complex::new(-62.0f64, 9.498_931_719_573_825e-280));
    assert!(
        (z.im / -4.868_434_214_548_976e-244 - 1.0).abs() < 1e-15,
        "{z:?}"
    );
    let z = zeta(Complex::new(-400.0f64, 1e-300));
    assert!(
        (z.im / 1.711_736_213_951_613_4e249 - 1.0).abs() < 1e-15,
        "{z:?}"
    );
    // each part on its own at a trivial zero with a tiny t (references by mpmath at 1400 bits):
    // from σ = −12 both to a few ulps of themselves; at −2, …, −10, through Γ(1 − s), the real
    // part, of order t², only to about 60 ulps (the phases of (2π)^(s−1), Γ(1 − s) and ζ(1 − s)
    // cancel in it to a sixtieth: ln 2π − ψ(7) − ζ′(7)/ζ(7) ≈ −0.029), as special.md states
    let part = |y: f64, r: f64| ulps(y, (r, 0.0), F64);
    let z = zeta(Complex::new(-52.0f64, 6.101_253_513_670_254e-42));
    assert!(part(z.re, 9.955_048_922_487_834e-57) <= 3.0, "{z:?}");
    assert!(part(z.im, 7.685_715_649_066_524e-16) <= 3.0, "{z:?}");
    let z = zeta(Complex::new(-6.0f64, 1.554_214_523_839_487e-34));
    assert!(part(z.re, -4.122_021_079_408_08e-72) <= 60.0, "{z:?}");
    assert!(part(z.im, -9.169_491_348_007_283e-37) <= 3.0, "{z:?}");
    // the real part at a trivial zero, of order t², with its sign where both parts leave the
    // range of f32 (it was a rounding residue of either sign)
    let z = zeta(Complex::new(-256.0f32, 6.335_842e-36));
    assert_eq!((z.re, z.im), (f32::INFINITY, f32::INFINITY));
    let z = zeta(Complex::new(-218.0f32, -1.260_001_5e-31));
    assert_eq!((z.re, z.im), (f32::NEG_INFINITY, f32::INFINITY));
}

#[test]
fn hurwitz_data() {
    // normal results; a subnormal result (its terms rounded to the subnormals apart) within 3.5
    // units of the least subnormal
    let f = file!("hurwitz_f64");
    let normal = |r: f64| !(r != 0.0 && r.abs() < f64::MIN_POSITIVE);
    let mut sub = 0.0f64;
    let mut text = String::new();
    for c in cases(&all(f)) {
        if normal(c.refs[0].0) {
            continue;
        }
        let g = hurwitz_zeta(c.args[0], c.args[1]);
        sub = sub.max(((g - c.refs[0].0) - c.refs[0].1).abs() / f64::from_bits(1));
    }
    assert!(
        sub <= ub(3.5),
        "hurwitz f64, subnormal results: {sub} units of 2⁻¹⁰⁷⁴"
    );
    for l in all(f).lines() {
        let c = &cases(l)[0];
        if normal(c.refs[0].0) {
            text.push_str(l);
            text.push('\n');
        }
    }
    let e = check_real("hurwitz f64", &text, F64, ub(1.35), |a| {
        hurwitz_zeta(a[0], a[1])
    });
    let e32 = check_real(
        "hurwitz f32",
        &all(file!("hurwitz_f32")),
        F32,
        ub(0.501),
        |a| hurwitz_zeta(a[0] as f32, a[1] as f32) as f64,
    );
    eprintln!("hurwitz: f64 {e:.3} ulps (subnormal results {sub:.2} units), f32 {e32:.3}");
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
    // past the range of A^(−s) alone, and of A/(s − 1): ζ(s, a) ≈ a^(1−s)/(s − 1)
    assert!((hurwitz_zeta(2.0f64, 1e170) / 1e-170 - 1.0).abs() < eps(2.0));
    assert!((hurwitz_zeta(1.5f64, f64::MAX) / 1.491_668_146_240_041_3e-154 - 1.0).abs() < eps(2.0));
    assert_eq!(hurwitz_zeta(1e300f64, 1e300), 0.0);
    assert_eq!(hurwitz_zeta(500.0f64, 0.001), f64::INFINITY);
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
/// condition number (the data's second reference, for `s < 0`, where `Li_s` has real zeros on the
/// negative axis: `κ = |x ∂ₓLi|/|Li|` for an integer order, `(|x ∂ₓLi| + |s ∂ₛLi|)/|Li|` else).
type Region = (&'static str, fn(f64, f64) -> bool, f64, bool);
const POLYLOG_REGIONS: [Region; 8] = [
    // the series, the expansions about 1 and −1
    ("s >= 1, x >= -2", |s, x| s >= 1.0 && x >= -2.0, 3.0, false),
    // the expansion about −1 cancels a little for small orders
    (
        "0 <= s < 1, x >= -2",
        |s, x| (0.0..1.0).contains(&s) && x >= -2.0,
        4.5,
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
    ("s > 0, x < -2", |s, x| s > 0.0 && x < -2.0, 4.0, false),
    // the Eulerian polynomials, the series, the expansion about 1
    (
        "integers s < 0, x >= 0",
        |s, x| s == s.round() && s < 0.0 && x >= 0.0,
        2.5,
        false,
    ),
    // the Hurwitz zeta function of the complex a, in units of κ
    (
        "integers s < 0, x < 0",
        |s, _| s == s.round() && s < 0.0,
        2.5,
        true,
    ),
    ("s < 0, x < 0", |s, x| s < 0.0 && x < 0.0, 4.0, true),
    // Γ(1 − s) and the reflection of ζ(s − k); the series
    ("s < 0, x > 0", |_, _| true, 3.0, false),
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
        &all(file!("polylog_f32")),
        F32,
        ub(0.501),
        |a| polylog(a[0] as f32, a[1] as f32) as f64,
    );
    eprintln!("polylog f32: {e32:.3}");
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
    // huge orders: x (1 + x 2⁻ˢ + …) rounds to x; negative ones past the range: +∞ for x > 0
    assert_eq!(polylog(2024.0f64, 0.5), 0.5);
    assert_eq!(polylog(1e300f64, 0.5), 0.5);
    assert_eq!(polylog(2.0e5f64, -3.0), -3.0);
    assert_eq!(polylog(1.0e6f64, -1e10), -1e10);
    assert_eq!(polylog(-1000.0f64, 0.5), f64::INFINITY);
    assert_eq!(polylog(-20.5f64, 1.0 - f64::EPSILON / 2.0), f64::INFINITY);
    // and from the orders where the expansion about 1 keeps its sum (above −20) and the Eulerian
    // form (to −18) leave the range near x = 1 (both were NaN: ∞ − ∞): Li₋₁₈(1 − 2⁻⁵³) ≈ 8.8·10³¹⁸
    let below_one = 1.0 - f64::EPSILON / 2.0;
    for s in [-18.0f64, -17.9, -18.5, -19.5, -19.9] {
        assert_eq!(polylog(s, below_one), f64::INFINITY, "{s}");
    }
    assert_eq!(
        polylog(-19.263_766_399_854_106f64, 0.999_999_999_999_998_6),
        f64::INFINITY
    );
    assert!(polylog(-17.0f64, below_one).is_finite());
    assert_eq!(polylog(-1.0e15f64, -0.5), f64::NEG_INFINITY);
    assert!(polylog(-1e300f64, -2.0).is_nan());
    // the zeros Li₋₂ₖ(−1) = −η(−2k) = 0 are exact, and an order near 0 is Li₀ = x/(1 − x)
    for k in 1..33 {
        assert_eq!(polylog(-2.0 * k as f64, -1.0f64), 0.0, "{k}");
    }
    assert!((polylog(-5e-324f64, -0.6) - (-0.375)).abs() <= eps(1.0));
    assert_eq!(polylog(0.0f64, f64::NEG_INFINITY), -1.0);
    // and a positive subnormal order too, at every x (the integral of Fermi and Dirac over
    // Γ(s) ≈ 1/s made it NaN for −2.4·10¹⁶ ≤ x < −2); Li_s(−∞) = −∞ for every s > 0
    for s in [5e-324f64, 1e-310, f64::MIN_POSITIVE / 2.0] {
        for x in [
            -3.0f64,
            -100.0,
            -2.3e16,
            -1e300,
            -0.9,
            0.5,
            1.0 - f64::EPSILON,
        ] {
            assert_eq!(polylog(s, x), x / (1.0 - x), "{s:e} {x:e}");
        }
        assert_eq!(polylog(s, f64::NEG_INFINITY), f64::NEG_INFINITY);
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
    // E₁ to 2.5 by this crate's fits; past it Boost.Math's rational, whose roundings in double
    // reach 1.4 ulps of x eˣ E₁ at its left end
    let f = file!("e1_f64");
    let e1 = check_real(
        "E1 f64, x <= 2.5",
        &data(f, |a| a[0] <= 2.5),
        F64,
        ub(1.1),
        |a| expint_e1(a[0]),
    );
    let e1b = check_real(
        "E1 f64, x > 2.5",
        &data(f, |a| a[0] > 2.5),
        F64,
        ub(1.7),
        |a| expint_e1(a[0]),
    );
    // Ei past 0 (1.8 where the target has no fused multiply-add), and −E₁(−x) below it
    let fma = cfg!(any(target_arch = "aarch64", target_feature = "fma"));
    let f = file!("ei_f64");
    let ei = check_real(
        "Ei f64, x > 0",
        &data(f, |a| a[0] > 0.0),
        F64,
        ub(if fma { 1.45 } else { 1.8 }),
        |a| expint_ei(a[0]),
    );
    let eim = check_real(
        "Ei f64, -2.5 <= x <= 0",
        &data(f, |a| (-2.5..=0.0).contains(&a[0])),
        F64,
        ub(1.1),
        |a| expint_ei(a[0]),
    );
    let eib = check_real(
        "Ei f64, x < -2.5",
        &data(f, |a| a[0] < -2.5),
        F64,
        ub(1.7),
        |a| expint_ei(a[0]),
    );
    let f = file!("en_f64");
    let en = check_real(
        "En f64, n >= 2",
        &data(f, |a| a[0] >= 2.0),
        F64,
        ub(1.05),
        |a| expint_en(a[0] as u32, a[1]),
    );
    let en0 = check_real(
        "En f64, n = 0",
        &data(f, |a| a[0] == 0.0),
        F64,
        ub(0.6),
        |a| expint_en(a[0] as u32, a[1]),
    );
    // n = 1 is E₁
    let en1 = check_real(
        "En f64, n = 1",
        &data(f, |a| a[0] == 1.0),
        F64,
        ub(1.7),
        |a| expint_en(a[0] as u32, a[1]),
    );
    let e132 = check_real("E1 f32", &all(file!("e1_f32")), F32, ub(0.501), |a| {
        expint_e1(a[0] as f32) as f64
    });
    let ei32 = check_real("Ei f32", &all(file!("ei_f32")), F32, ub(0.501), |a| {
        expint_ei(a[0] as f32) as f64
    });
    let en32 = check_real("En f32", &all(file!("en_f32")), F32, ub(0.501), |a| {
        expint_en(a[0] as u32, a[1] as f32) as f64
    });
    eprintln!(
        "E1 {e1:.3} ({e1b:.3} past 2.5), Ei {ei:.3} ({eim:.3} for x < 0, {eib:.3} past −2.5), \
         En {en:.3} ({en0:.3} for n = 0, {en1:.3} for n = 1) ulps; f32 {e132:.3}, {ei32:.3}, \
         {en32:.3}"
    );
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
    // e⁻ˣ/x past the range (NaN before)
    assert_eq!(expint_en(0, 4e-309f64), f64::INFINITY);
    assert_eq!(expint_en(0, f64::from_bits(1)), f64::INFINITY);
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

#[test]
fn lambert_w0_without_a_fused_multiply_add() {
    // The last step `x·c₀ + x·hl` of the pieces of `x e^(−W₀)` rounded twice without a fused
    // multiply-add (x86_64 by default): 1.59 and 1.11 ulps at these points (mpmath at 60 digits);
    // with the product exact, within the bound of 1 ulp on every target.
    for (x, hi, lo) in [
        (
            0.7939862750941973,
            0.4875904216912718,
            -2.2708827691094167e-17,
        ),
        (
            0.7851795518000724,
            0.48394374643921784,
            6.042651393545385e-18,
        ),
    ] {
        let e = ulps(lambert_w0(x), (hi, lo), F64);
        assert!(e <= 1.0, "lambert_w0({x}): {e:.3} ulps");
    }
}
