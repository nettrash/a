//! The gamma area of issue #20: its data in `data/gamma_*.txt`, made by `data/gen_gamma.py`.
//!
//! Each data file is checked to the area's stated bound (in ulps of the result's type, or in ε
//! relative to `|w|` for a complex result); hand-written tests cover the special values,
//! identities and symmetries.

use super::common::*;
use a::math::complex::Complex;
use a::math::special::*;

/// The data lines of `text` whose case satisfies `keep`, or under Miri every 37th of them.
fn select(text: &str, keep: impl Fn(&Case) -> bool) -> String {
    let lines: Vec<&str> = text
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.starts_with('#'))
        .filter(|l| cases(l).iter().all(&keep))
        .collect();
    let step = if cfg!(miri) { 37 } else { 1 };
    let last = lines.len().saturating_sub(1);
    lines
        .iter()
        .enumerate()
        .filter(|(i, _)| i % step == 0 || *i == last)
        .map(|(_, l)| *l)
        .collect::<Vec<_>>()
        .join("\n")
}

/// Every case.
fn all(_: &Case) -> bool {
    true
}

/// Checks a real function to `bound` ulps on the cases that satisfy `keep`.
#[track_caller]
fn real(
    name: &str,
    text: &str,
    keep: impl Fn(&Case) -> bool,
    p: Prec,
    bound: f64,
    f: impl Fn(&[f64]) -> f64,
) -> f64 {
    check_real(name, &select(text, keep), p, bound, f)
}

/// Checks a function of several reference values (`f` returns them in order) to `bound` ulps.
#[track_caller]
fn multi(
    name: &str,
    text: &str,
    keep: impl Fn(&Case) -> bool,
    p: Prec,
    bound: f64,
    f: impl Fn(&[f64]) -> Vec<f64>,
) -> f64 {
    let cs = cases(&select(text, keep));
    assert!(!cs.is_empty(), "{name}: no cases");
    let w = worst(&cs, f, |g, r| {
        g.iter()
            .zip(r)
            .map(|(&y, &r)| ulps(y, r, p))
            .fold(0.0, f64::max)
    });
    assert!(
        w.max <= bound,
        "{name}: {:.3} ulps > {bound} at {:?}: got {:?}, want {:?}",
        w.max,
        w.args,
        w.got,
        w.want
    );
    w.max
}

/// Checks a real function's absolute error to `bound` ε (where its value nears zero).
#[track_caller]
fn absolute(name: &str, text: &str, bound: f64, f: impl Fn(&[f64]) -> f64) -> f64 {
    let cs = cases(&select(text, all));
    assert!(!cs.is_empty(), "{name}: no cases");
    let w = worst(
        &cs,
        |a| vec![f(a)],
        |g, r| ((g[0] - r[0].0) - r[0].1).abs() / f64::EPSILON,
    );
    assert!(
        w.max <= bound,
        "{name}: absolute error {:.3} ε > {bound} at {:?}: got {:?}, want {:?}",
        w.max,
        w.args,
        w.got,
        w.want
    );
    w.max
}

/// Checks a complex function to `bound` ε (relative to `|w|`) on the cases that satisfy `keep`.
#[track_caller]
fn complex(
    name: &str,
    text: &str,
    keep: impl Fn(&Case) -> bool,
    bound: f64,
    f: impl Fn(f64, f64) -> (f64, f64),
) -> f64 {
    check_complex(name, &select(text, keep), F64, bound, f)
}

/// The reference value of a case is a subnormal `f64` (or zero).
fn subnormal(c: &Case) -> bool {
    c.refs[0].0.abs() < f64::MIN_POSITIVE
}

// ------------------------------------------------------------------------------ Γ and ln |Γ|

const GAMMA_F64: &str = include_str!("data/gamma_gamma_f64.txt");
const GAMMA_F32: &str = include_str!("data/gamma_gamma_f32.txt");
const LGAMMA_F64: &str = include_str!("data/gamma_lgamma_f64.txt");
const LGAMMA_F32: &str = include_str!("data/gamma_lgamma_f32.txt");
const LGAMMA_NEGZERO: &str = include_str!("data/gamma_lgamma_negzero_f64.txt");

#[test]
fn gamma_of_reals() {
    real(
        "gamma f64",
        GAMMA_F64,
        |c| !subnormal(c),
        F64,
        0.6,
        |a| gamma(a[0]),
    );
    // A subnormal result is rounded once too (from the double-double). (Its reference, a
    // double-double too, is itself resolved only to 2⁻¹⁰⁷⁴, so 0.6 here is in effect a check of
    // the correct rounding: an error of 0.7 ulps reads as 1.)
    real("gamma f64 subnormal", GAMMA_F64, subnormal, F64, 0.6, |a| {
        gamma(a[0])
    });
    real("gamma f32", GAMMA_F32, all, F32, 0.501, |a| {
        gamma(a[0] as f32) as f64
    });
}

/// Checks `lgamma` of `f64`s: to 0.6 ulps (for `x > 0` also beside the zeros 1 and 2), but for
/// `x < 0` where `|ln |Γ(x)|| < 1/4` (beside the zeros on the negative axis) to an absolute
/// 0.1 ε.
#[track_caller]
fn lgamma_f64_check(name: &str, text: &str) {
    let negative = |c: &Case| c.args[0] < 0.0;
    let rest = select(text, |c| !negative(c));
    if !rest.is_empty() {
        check_real(name, &rest, F64, 0.6, |a| lgamma(a[0]));
    }
    let cs = cases(&select(text, negative));
    assert!(!cs.is_empty(), "{name}: no cases with x < 0");
    let w = worst(
        &cs,
        |a| vec![lgamma(a[0])],
        |g, r| {
            let (h, l) = r[0];
            if h.abs() < 0.25 {
                // In units of 0.1 ε / 0.6 ulps, so that both bounds read as 0.6.
                ((g[0] - h) - l).abs() / f64::EPSILON * 6.0
            } else {
                ulps(g[0], (h, l), F64)
            }
        },
    );
    assert!(
        w.max <= 0.6,
        "{name}, x < 0: {:.3} (ulps, or 0.1 ε as 0.6) at {:?}: got {:?}, want {:?}",
        w.max,
        w.args,
        w.got,
        w.want
    );
}

#[test]
fn lgamma_of_reals() {
    lgamma_f64_check("lgamma f64", LGAMMA_F64);
    real("lgamma f32", LGAMMA_F32, all, F32, 0.501, |a| {
        lgamma(a[0] as f32) as f64
    });
    // Beside the zeros of ln |Γ| on the negative axis the error is absolute (where the value is
    // below 1/4).
    lgamma_f64_check("lgamma near its negative zeros", LGAMMA_NEGZERO);
}

#[test]
fn gamma_special_values() {
    assert_eq!(gamma(0.0f64), f64::INFINITY);
    assert_eq!(gamma(-0.0f64), f64::NEG_INFINITY);
    assert_eq!(gamma(f64::INFINITY), f64::INFINITY);
    assert!(gamma(f64::NEG_INFINITY).is_nan() && gamma(f64::NAN).is_nan());
    for n in [-1.0f64, -2.0, -170.0, -1e10, -1e300] {
        assert!(gamma(n).is_nan(), "gamma({n})");
        assert_eq!(lgamma(n), f64::INFINITY, "lgamma({n})");
    }
    assert!(gamma(171.624f64).is_finite());
    assert_eq!(gamma(171.625f64), f64::INFINITY);
    // Underflow keeps the sign: Γ < 0 on (−201, −200), > 0 on (−202, −201).
    assert_eq!(gamma(-200.5f64).to_bits(), (-0.0f64).to_bits());
    assert_eq!(gamma(-201.5f64).to_bits(), 0.0f64.to_bits());
    // Factorials are exact while they fit.
    let mut f = 1.0f64;
    for n in 1..=22 {
        assert_eq!(gamma(n as f64), f, "{n}!");
        f *= n as f64;
    }
    // √π, correctly rounded (the square root of the rounded π is one ulp below).
    assert_eq!(gamma(0.5f64), 1.772453850905516);
    assert_eq!(gamma(1e-310f64), f64::INFINITY);
    assert_eq!(gamma(5e-324f64), f64::INFINITY);
    assert_eq!(lgamma(0.0f64), f64::INFINITY);
    assert_eq!(lgamma(-0.0f64), f64::INFINITY);
    assert_eq!(lgamma(f64::NEG_INFINITY), f64::INFINITY);
    assert_eq!(lgamma(f64::INFINITY), f64::INFINITY);
    assert!(lgamma(f64::NAN).is_nan());
    assert_eq!(lgamma(1.0f64), 0.0);
    assert_eq!(lgamma(2.0f64), 0.0);
    assert_eq!(lgamma(f64::MAX), f64::INFINITY);
    assert_eq!(gamma(1.0f32), 1.0);
    assert_eq!(gamma(0.0f32), f32::INFINITY);
    assert_eq!(gamma(-0.0f32), f32::NEG_INFINITY);
    assert_eq!(gamma(36.0f32), f32::INFINITY);
    assert_eq!(lgamma(f32::MAX), f32::INFINITY);
}

#[test]
fn lgamma_sign_of_reals() {
    let cases: &[(f64, f64)] = &[
        (2.5, 1.0),
        (0.5, 1.0),
        (-0.5, -1.0),
        (-1.5, 1.0),
        (-2.5, -1.0),
        (-100.5, -1.0),
        (-101.5, 1.0),
        (0.0, 1.0),
        (-0.0, -1.0),
        (-3.0, 1.0),
        (f64::INFINITY, 1.0),
    ];
    for &(x, s) in cases {
        let (l, sg) = lgamma_sign(x);
        assert_eq!(sg, s, "sign of gamma({x})");
        assert_eq!(l.to_bits(), lgamma(x).to_bits(), "lgamma_sign({x}).0");
        let g = gamma(x);
        if g.is_finite() && g != 0.0 {
            assert_eq!(g.signum(), s, "gamma({x}) has the sign");
        }
    }
    assert!(lgamma_sign(f64::NAN).1.is_nan());
    assert_eq!(lgamma_sign(-0.5f32).1, -1.0f32);
}

#[test]
fn gamma_identities() {
    // Γ(x + 1) = x Γ(x) and the reflection formula, to a few ulps.
    let mut x = -9.875f64;
    while x < 20.0 {
        if x != x.floor() {
            let r = gamma(x + 1.0) / (x * gamma(x));
            assert!((r - 1.0).abs() < 4e-16, "recurrence at {x}: {r}");
            // sin(πx) with the argument reduced first: sin(π·(n + r)) = (−1)ⁿ sin(πr), and r is
            // exact here. Forming π·x first lets the rounding of double π shift the argument by
            // |x|·1e-17·π, which dominates the product's error for large x (and Miri's software
            // `sin` differs by a few ulps between targets).
            let n = x.round();
            let sx = (core::f64::consts::PI * (x - n)).sin()
                * if n as i64 % 2 == 0 { 1.0 } else { -1.0 };
            let s = gamma(x) * gamma(1.0 - x) * sx / core::f64::consts::PI;
            assert!((s - 1.0).abs() < 1e-14, "reflection at {x}: {s}");
            let l = lgamma(x) - gamma(x).abs().ln();
            assert!(l.abs() < 1e-14 * lgamma(x).abs().max(1.0), "lgamma at {x}");
        }
        x += 0.375;
    }
}

// ---------------------------------------------------------------------- the complex Γ, ln Γ

const GAMMA_C64: &str = include_str!("data/gamma_gamma_c64.txt");
const LGAMMA_C64: &str = include_str!("data/gamma_lgamma_c64.txt");

/// `|z|` of a complex case.
fn modulus(c: &Case) -> f64 {
    c.args[0].hypot(c.args[1])
}

#[test]
fn gamma_of_complex() {
    complex(
        "gamma c64",
        GAMMA_C64,
        |c| modulus(c) < 1e14,
        1.25,
        |re, im| {
            let z = gamma(Complex::new(re, im));
            (z.re, z.im)
        },
    );
    // The documented limit: from |z| = 10¹⁴ on (where the condition number |z ψ(z)| of Γ passes
    // 1/ε) the relative error is that of ln Γ(z) in double-double, whose main terms are of the size
    // |z| ln |z|: an allowance of |z| ln |z| · 2⁻¹⁰⁴.
    let far = cases(&select(GAMMA_C64, |c| modulus(c) >= 1e14));
    assert!(!far.is_empty(), "gamma c64: no cases beyond 1e14");
    for c in &far {
        let m = modulus(c);
        let allow = (m * m.ln() * f64::EPSILON).max(1.25);
        let z = gamma(Complex::new(c.args[0], c.args[1]));
        let e = rel_eps((z.re, z.im), c.refs[0], c.refs[1], F64);
        assert!(
            e <= allow,
            "gamma c64 at {:?}: {e:.3} ε > {allow:.3}: got {z:?}",
            c.args
        );
    }
    complex("lgamma c64", LGAMMA_C64, all, 1.25, |re, im| {
        let z = lgamma(Complex::new(re, im));
        (z.re, z.im)
    });
    let z = gamma(Complex::new(2.5f32, 1.0));
    let w = gamma(Complex::new(2.5f64, 1.0));
    assert_eq!((z.re, z.im), (w.re as f32, w.im as f32));
}

#[test]
fn complex_gamma_branches_and_special_values() {
    use core::f64::consts::PI;
    // The cut: the side of the sign of the zero, −π⌈−x⌉ from above (mpmath's and SciPy's).
    let l = lgamma(Complex::new(-2.5f64, 0.0));
    assert_eq!(l.re, lgamma(-2.5f64));
    assert_eq!(l.im, -3.0 * PI);
    let l = lgamma(Complex::new(-2.5f64, -0.0));
    assert_eq!(l.im, 3.0 * PI);
    // Continuity across the real axis away from the cut, and onto it from above.
    let a = lgamma(Complex::new(-2.5f64, 1e-300));
    assert!((a.im + 3.0 * PI).abs() < 1e-15 && (a.re - lgamma(-2.5f64)).abs() < 1e-15);
    let b = lgamma(Complex::new(3.0f64, -1e-300));
    assert_eq!(b.re, core::f64::consts::LN_2);
    // ln Γ(conj z) = conj ln Γ(z), Γ likewise.
    for &(x, y) in &[
        (0.3, 2.0),
        (-7.25, 0.5),
        (25.0, -40.0),
        (1e-3, 1e-3),
        (-100.5, 3.0),
    ] {
        // Bitwise on a target; under Miri, whose atan2 is perturbed at random (it is only the
        // first guess of the argument, corrected in double-double), to within an ulp or so.
        let tol = if cfg!(miri) { 4e-16 } else { 0.0 };
        let near = |p: Complex<f64>, q: Complex<f64>| {
            (p.re - q.re).abs() <= tol * p.abs() && (p.im + q.im).abs() <= tol * p.abs()
        };
        let p = lgamma(Complex::new(x, y));
        let q = lgamma(Complex::new(x, -y));
        assert!(near(p, q), "lgamma conj at {x} {y}: {p:?} {q:?}");
        let p = gamma(Complex::new(x, y));
        let q = gamma(Complex::new(x, -y));
        assert!(near(p, q), "gamma conj at {x} {y}: {p:?} {q:?}");
    }
    // Where both parts underflow, the zeros keep the conjugate's signs (they were both +0 for z
    // and its conjugate: the second review's cases). Not under Miri: there the phase of such a
    // z, about |z| ln |z| in size, depends on its perturbed atan2.
    if !cfg!(miri) {
        for &(x, y) in &[
            (-1.1816180553724866e236f64, -2.1391109902859666e235f64),
            (-9.858351493080197e34, -1.5722588066541558e35),
            (-3.7296713787713606e155, 5.085252663361583e155),
            (-5.314226948048298e116, -1.130186739063547e115),
        ] {
            let p = gamma(Complex::new(x, y));
            let q = gamma(Complex::new(x, -y));
            assert!(
                p.re == 0.0 && p.im == 0.0,
                "gamma at {x} {y} should underflow: {p:?}"
            );
            assert!(
                p.re.to_bits() == q.re.to_bits() && p.im.to_bits() == (-q.im).to_bits(),
                "gamma conj (signed zeros) at {x} {y}: {p:?} {q:?}"
            );
        }
    }
    // The recurrence ln Γ(z + 1) = ln Γ(z) + ln z holds on the principal branch where Im z > 0
    // and z is in the right half-plane.
    let z = Complex::new(0.75f64, 3.0);
    let d = lgamma(z + Complex::new(1.0, 0.0)) - lgamma(z) - z.ln();
    assert!(d.abs() < 1e-14, "{d:?}");
    // Poles, infinities and NaN.
    let p = lgamma(Complex::new(-3.0f64, 0.0));
    assert!(p.re == f64::INFINITY && p.im.is_nan());
    let g = gamma(Complex::new(-3.0f64, 0.0));
    assert!(g.re.is_nan());
    let g = gamma(Complex::new(0.0f64, 0.0));
    assert_eq!(g.re, f64::INFINITY);
    let g = gamma(Complex::new(f64::NAN, 1.0));
    assert!(g.re.is_nan() && g.im.is_nan());
    let g = gamma(Complex::new(1.0f64, f64::INFINITY));
    assert_eq!((g.re, g.im), (0.0, 0.0));
    let l = lgamma(Complex::new(1.0f64, f64::INFINITY));
    assert_eq!((l.re, l.im), (f64::NEG_INFINITY, f64::INFINITY));
    // Overflow of one part only.
    let g = gamma(Complex::new(173.08934693102756f64, -50.501779676731175));
    assert!(g.re.is_infinite() && g.im.is_finite());
}

// ---------------------------------------------------------------------------- ψ and ψ⁽ⁿ⁾

const DIGAMMA_F64: &str = include_str!("data/gamma_digamma_f64.txt");
const DIGAMMA_F32: &str = include_str!("data/gamma_digamma_f32.txt");
const DIGAMMA_NEGZERO: &str = include_str!("data/gamma_digamma_negzero_f64.txt");
const POLYGAMMA_F64: &str = include_str!("data/gamma_polygamma_f64.txt");
const POLYGAMMA_NEG_F64: &str = include_str!("data/gamma_polygamma_neg_f64.txt");

#[test]
fn digamma_of_reals() {
    real("digamma f64", DIGAMMA_F64, all, F64, 0.75, |a| {
        digamma(a[0])
    });
    real("digamma f32", DIGAMMA_F32, all, F32, 0.501, |a| {
        digamma(a[0] as f32) as f64
    });
    absolute(
        "digamma near its negative zeros",
        DIGAMMA_NEGZERO,
        0.25,
        |a| digamma(a[0]),
    );
}

#[test]
fn polygamma_of_reals() {
    real("polygamma f64", POLYGAMMA_F64, all, F64, 0.75, |a| {
        polygamma(a[0] as u32, a[1])
    });
    // For x < 0 the error is relative to n!(r^(−n−1) + p^(−n−1)), the size of the largest terms
    // (r, p the distances of x to the integers beside it).
    let cs = cases(&select(POLYGAMMA_NEG_F64, all));
    let w = worst(
        &cs,
        |a| vec![polygamma(a[0] as u32, a[1])],
        |g, r| {
            if !r[0].0.is_finite() {
                return if g[0] == r[0].0 { 0.0 } else { f64::INFINITY };
            }
            ((g[0] - r[0].0) - r[0].1).abs() / (r[1].0 * f64::EPSILON)
        },
    );
    assert!(
        w.max <= 0.75,
        "polygamma x < 0: {:.3} at {:?}",
        w.max,
        w.args
    );
}

#[test]
fn digamma_and_polygamma_special_values() {
    assert_eq!(digamma(0.0f64), f64::NEG_INFINITY);
    assert_eq!(digamma(-0.0f64), f64::INFINITY);
    assert_eq!(digamma(f64::INFINITY), f64::INFINITY);
    assert!(digamma(f64::NEG_INFINITY).is_nan() && digamma(-2.0f64).is_nan());
    assert!(digamma(f64::NAN).is_nan());
    // ψ(1) = −γ, ψ(1/2) = −γ − 2 ln 2, correctly rounded.
    assert_eq!(digamma(1.0f64), -0.577_215_664_901_532_9);
    assert_eq!(digamma(0.5f64), -1.963_510_026_021_423_5);
    assert_eq!(polygamma(0, 3.5f64), digamma(3.5f64));
    // ψ′(1) = π²/6.
    assert_eq!(polygamma(1, 1.0f64), 1.644_934_066_848_226_4);
    assert_eq!(polygamma(1, 0.0f64), f64::INFINITY);
    assert_eq!(polygamma(2, 0.0f64), f64::NEG_INFINITY);
    assert_eq!(polygamma(2, -0.0f64), f64::INFINITY);
    assert_eq!(polygamma(3, f64::INFINITY), 0.0);
    assert!(polygamma(2, f64::INFINITY).is_sign_negative());
    assert!(polygamma(1, -3.0f64).is_nan() && polygamma(1, f64::NAN).is_nan());
    assert_eq!(polygamma(5, 1e-70f64), f64::INFINITY);
    assert_eq!(polygamma(200, 0.5f64), f64::NEG_INFINITY);
    assert!(polygamma(1, 1e300f64) > 0.0);
    assert_eq!(polygamma(4, 1e200f64), 0.0);
    // The recurrence ψ⁽ⁿ⁾(x + 1) = ψ⁽ⁿ⁾(x) + (−1)ⁿ n!/x^(n+1), and the reflection of ψ.
    for n in 1..6u32 {
        for &x in &[0.3f64, 1.7, 4.2, 11.5, 37.0] {
            let f: f64 = (1..=n).map(|k| k as f64).product();
            let s = if n % 2 == 0 { 1.0 } else { -1.0 };
            let lhs = polygamma(n, x + 1.0);
            let (p, t) = (polygamma(n, x), s * f / x.powi(n as i32 + 1));
            let rhs = p + t;
            assert!(
                (lhs - rhs).abs() <= 1e-14 * (p.abs() + t.abs()),
                "n {n} x {x}"
            );
        }
    }
    for &x in &[0.25f64, 0.7, 2.3, 9.9] {
        // tan has period 1 in x, so the reduced argument is exact and double π is harmless.
        let d = digamma(1.0 - x)
            - digamma(x)
            - core::f64::consts::PI / (core::f64::consts::PI * (x - x.round())).tan();
        assert!(d.abs() < 1e-13, "reflection at {x}: {d}");
    }
    assert_eq!(digamma(2.0f32), 0.422_784_34f32);
}

#[test]
fn psi_at_the_edges_of_the_range() {
    // −1/x − γ is beyond the range from 2⁻¹⁰²⁴ down (1/x overflows): ∓∞, not NaN.
    let edge = f64::from_bits(1 << 50);
    for x in [f64::from_bits(1), 1e-320, 1e-310, edge] {
        assert_eq!(digamma(x), f64::NEG_INFINITY, "digamma({x:e})");
        assert_eq!(digamma(-x), f64::INFINITY, "digamma(-{x:e})");
    }
    let above = f64::from_bits((1 << 50) + 1);
    assert!(digamma(above).is_finite() && digamma(-above).is_finite());
    // ψ′(x) = 1/x² overflows from 2⁻⁵¹² down.
    let x = f64::from_bits(511 << 52);
    assert_eq!(polygamma(1, x), f64::INFINITY);
    assert!(polygamma(1, f64::from_bits(x.to_bits() + 1)).is_finite());
}

// ---------------------------------------------------------------- B, ln B, x!, C(n, k), (x)ₙ

const BETA_F64: &str = include_str!("data/gamma_beta_f64.txt");
const BETA_NEG_F64: &str = include_str!("data/gamma_beta_neg_f64.txt");
const BETA_F32: &str = include_str!("data/gamma_beta_f32.txt");
const FACTORIAL_F64: &str = include_str!("data/gamma_factorial_f64.txt");
const BINOMIAL_F64: &str = include_str!("data/gamma_binomial_f64.txt");
const POCHHAMMER_F64: &str = include_str!("data/gamma_pochhammer_f64.txt");

#[test]
fn beta_and_lbeta() {
    real("beta f64", BETA_F64, all, F64, 1.0, |a| beta(a[0], a[1]));
    let cs = cases(&select(BETA_F64, all));
    let w = worst(
        &cs,
        |a| vec![lbeta(a[0], a[1])],
        |g, r| {
            let (h, l) = r[1];
            if h.abs() >= 0.5 {
                ulps(g[0], (h, l), F64)
            } else {
                ((g[0] - h) - l).abs() / (0.5 * f64::EPSILON)
            }
        },
    );
    assert!(w.max <= 1.0, "lbeta f64: {:.3} at {:?}", w.max, w.args);
    real("beta f32", BETA_F32, all, F32, 0.501, |a| {
        beta(a[0] as f32, a[1] as f32) as f64
    });
    multi(
        "beta, lbeta f64, negative arguments",
        BETA_NEG_F64,
        all,
        F64,
        1.0,
        |a| vec![beta(a[0], a[1]), lbeta(a[0], a[1])],
    );
}

#[test]
fn factorials_binomials_pochhammer() {
    real("factorial f64", FACTORIAL_F64, all, F64, 0.6, |a| {
        factorial(a[0])
    });
    real("binomial f64", BINOMIAL_F64, all, F64, 0.6, |a| {
        binomial(a[0], a[1])
    });
    real("pochhammer f64", POCHHAMMER_F64, all, F64, 0.6, |a| {
        pochhammer(a[0], a[1])
    });
}

#[test]
fn beta_family_special_values() {
    assert_eq!(beta(1.0f64, 1.0), 1.0);
    assert_eq!(beta(2.0f64, 3.0), 1.0 / 12.0);
    assert_eq!(beta(0.5f64, 0.5), core::f64::consts::PI);
    assert_eq!(lbeta(1.0f64, 1.0), 0.0);
    assert_eq!(beta(0.0f64, 2.0), f64::INFINITY);
    assert_eq!(beta(-0.0f64, 2.0), f64::NEG_INFINITY);
    assert!(beta(-1.0f64, 0.5).is_nan() && beta(f64::NAN, 1.0).is_nan());
    assert_eq!(beta(-0.5f64, 0.5), 0.0);
    assert_eq!(lbeta(-0.5f64, 0.5), f64::NEG_INFINITY);
    assert_eq!(lbeta(-1.0f64, 0.5), f64::INFINITY);
    assert_eq!(beta(f64::INFINITY, 2.0), 0.0);
    assert_eq!(lbeta(2.0f64, f64::INFINITY), f64::NEG_INFINITY);
    assert_eq!(beta(1e-310f64, 1.0), f64::INFINITY);
    assert_eq!(beta(3.0f64, 4.0), beta(4.0f64, 3.0));
    let mut f = 1.0f64;
    for n in 0..=22 {
        assert_eq!(factorial(n as f64), f, "{n}!");
        f *= (n + 1) as f64;
    }
    assert_eq!(factorial(-0.5f64), 1.772453850905516);
    assert!(factorial(-1.0f64).is_nan() && factorial(-7.0f64).is_nan());
    assert_eq!(factorial(171.0f64), f64::INFINITY);
    assert_eq!(factorial(170.0f64), gamma(171.0f64));
    // Correctly rounded for n ≤ 60 (the table; `u128 as f64` rounds to nearest), exact below 2⁵³
    // beyond it.
    for n in 0..=100u32 {
        let mut c = 1u128;
        for k in 0..=n {
            if n <= 60 || c < 1 << 53 {
                assert_eq!(binomial(n as f64, k as f64), c as f64, "C({n}, {k})");
            }
            c = c * (n - k) as u128 / (k + 1) as u128;
        }
    }
    assert_eq!(binomial(5.0f64, 7.0), 0.0);
    assert_eq!(binomial(5.0f64, -1.0), 0.0);
    assert_eq!(binomial(-3.0f64, 2.0), 6.0);
    assert_eq!(binomial(-3.0f64, 3.0), -10.0);
    assert_eq!(binomial(0.5f64, 2.0), -0.125);
    assert_eq!(binomial(1e15f64, 1.0), 1e15);
    assert!(binomial(f64::NAN, 1.0).is_nan());
    assert_eq!(pochhammer(3.0f64, 4.0), 360.0);
    assert_eq!(pochhammer(-3.0f64, 3.0), -6.0);
    assert_eq!(pochhammer(-3.0f64, 4.0), 0.0);
    assert_eq!(pochhammer(5.0f64, -2.0), 1.0 / 12.0);
    assert!(pochhammer(2.0f64, -3.0).is_nan());
    assert_eq!(pochhammer(7.5f64, 0.0), 1.0);
    assert_eq!(pochhammer(1.0f64, 10.0), 3_628_800.0);
    assert_eq!(pochhammer(1.0f32, 5.0), 120.0f32);
    assert_eq!(binomial(10.0f32, 3.0), 120.0f32);
}

/// `f32` values spread over the arguments of the area (log-uniform from 2⁻²⁰ to 2²⁰ of either
/// sign, and a few integers).
fn f32_points() -> Vec<f32> {
    let mut v = Vec::new();
    let mut s = 0x9e37_79b9_7f4a_7c15u64;
    for _ in 0..if cfg!(miri) { 40 } else { 600 } {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let u = (s >> 11) as f64 / (1u64 << 53) as f64;
        let x = (2.0f64).powf(40.0 * u - 20.0) as f32;
        v.push(if s & 1 == 0 { x } else { -x });
    }
    v.extend([1.0f32, 2.0, 3.0, -1.0, 0.5, -0.5, 10.0, 170.0]);
    v
}

/// The same bits, NaN for NaN.
fn same(a: f32, b: f32) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

#[test]
fn f32_forms_are_the_f64_results_rounded_once() {
    // So an f32 result is within 0.5 + 2⁻²⁹·(the f64 error) f32 ulps: 0.501 wherever the f64
    // bound is relative.
    let xs = f32_points();
    for (i, &x) in xs.iter().enumerate() {
        let y = xs[(i * 7 + 3) % xs.len()];
        let (xd, yd) = (x as f64, y as f64);
        assert!(same(lbeta(x, y), lbeta(xd, yd) as f32), "lbeta({x}, {y})");
        assert!(same(factorial(x), factorial(xd) as f32), "factorial({x})");
        assert!(
            same(binomial(x, y), binomial(xd, yd) as f32),
            "binomial({x}, {y})"
        );
        assert!(
            same(pochhammer(x, y), pochhammer(xd, yd) as f32),
            "pochhammer({x}, {y})"
        );
        let (l, sg) = lgamma_sign(x);
        let (ld, sd) = lgamma_sign(xd);
        assert!(
            same(l, ld as f32) && same(sg, sd as f32),
            "lgamma_sign({x})"
        );
        for n in [1u32, 2, 5, 30] {
            assert!(
                same(polygamma(n, x), polygamma(n, xd) as f32),
                "polygamma({n}, {x})"
            );
        }
        let z = gamma(Complex::new(x, y));
        let w = gamma(Complex::new(xd, yd));
        assert!(
            same(z.re, w.re as f32) && same(z.im, w.im as f32),
            "gamma({x} + {y}i)"
        );
        let z = lgamma(Complex::new(x, y));
        let w = lgamma(Complex::new(xd, yd));
        assert!(
            same(z.re, w.re as f32) && same(z.im, w.im as f32),
            "lgamma({x} + {y}i)"
        );
    }
}

#[test]
fn beta_family_beside_the_poles_of_the_sum() {
    // a + b rounds to −1 but is not −1 exactly: B is finite (its exact value, by mpmath), not NaN.
    let b = beta(-0.3f64, -0.7);
    assert!(
        (b / -1.026_486_323_837_799_5e-15 - 1.0).abs() < 4e-16,
        "{b:e}"
    );
    assert!((lbeta(-0.3f64, -0.7) + 34.512_634_760_622_64).abs() < 1e-14);
    let p = pochhammer(-0.3f64, -0.7);
    assert!((p / 4_163_396_903_752_755.5 - 1.0).abs() < 4e-16, "{p:e}");
    let c = binomial(0.1f64, 19.1);
    assert!(
        (c / -5.264_875_749_448_132e-17 - 1.0).abs() < 4e-16,
        "{c:e}"
    );
    // a + b exactly −1: a pole of Γ(a + b) alone, so B = 0.
    assert_eq!(beta(-0.6f64, -0.4), 0.0);
    // Both arguments beyond 1.3·10³⁰⁸: B underflows, ln B overflows.
    assert_eq!(beta(1.3e308f64, 1.3e308), 0.0);
    assert_eq!(lbeta(f64::MAX, f64::MAX), f64::NEG_INFINITY);
    assert!(lbeta(10.0f64, f64::MAX).is_finite());
}

// ===== igamma (P, Q and inverses) =====

const PQ_F64: &str = include_str!("data/gamma_gamma_pq_f64.txt");
const PQ_F32: &str = include_str!("data/gamma_gamma_pq_f32.txt");
const P_INV_F64: &str = include_str!("data/gamma_gamma_p_inv_f64.txt");
const Q_INV_F64: &str = include_str!("data/gamma_gamma_q_inv_f64.txt");
const P_INV_F32: &str = include_str!("data/gamma_gamma_p_inv_f32.txt");
const Q_INV_F32: &str = include_str!("data/gamma_gamma_q_inv_f32.txt");

#[test]
fn incomplete_gamma_of_reals() {
    // 0.66 ulp but in the region of Temme's expansion (a ≥ 20, x within −42 % … +58 % of a), 1.2.
    multi("gamma_p, gamma_q f64", PQ_F64, all, F64, 1.2, |a| {
        vec![gamma_p(a[0], a[1]), gamma_q(a[0], a[1])]
    });
    multi("gamma_p, gamma_q f32", PQ_F32, all, F32, 0.501, |a| {
        let (s, x) = (a[0] as f32, a[1] as f32);
        vec![gamma_p(s, x) as f64, gamma_q(s, x) as f64]
    });
}

#[test]
fn incomplete_gamma_inverses() {
    real("gamma_p_inv f64", P_INV_F64, all, F64, 0.6, |a| {
        gamma_p_inv(a[0], a[1])
    });
    real("gamma_q_inv f64", Q_INV_F64, all, F64, 0.6, |a| {
        gamma_q_inv(a[0], a[1])
    });
    real("gamma_p_inv f32", P_INV_F32, all, F32, 0.501, |a| {
        gamma_p_inv(a[0] as f32, a[1] as f32) as f64
    });
    real("gamma_q_inv f32", Q_INV_F32, all, F32, 0.501, |a| {
        gamma_q_inv(a[0] as f32, a[1] as f32) as f64
    });
}

#[test]
fn incomplete_gamma_special_values() {
    let nan = f64::NAN;
    let inf = f64::INFINITY;
    for (a, x) in [
        (nan, 1.0),
        (1.0, nan),
        (0.0, 1.0),
        (-1.0, 1.0),
        (1.0, -1e-300),
        (inf, 1.0),
    ] {
        assert!(
            gamma_p(a, x).is_nan() && gamma_q(a, x).is_nan(),
            "P, Q({a}, {x})"
        );
    }
    assert_eq!((gamma_p(2.5f64, 0.0), gamma_q(2.5f64, 0.0)), (0.0, 1.0));
    assert_eq!((gamma_p(2.5f64, inf), gamma_q(2.5f64, inf)), (1.0, 0.0));
    assert_eq!(
        (gamma_p(1e-300f64, inf), gamma_q(1e300f64, 0.0)),
        (1.0, 1.0)
    );
    // Far tails underflow to zero; the other side is then 1.
    assert_eq!((gamma_p(1e3f64, 1.0), gamma_q(1e3f64, 1.0)), (0.0, 1.0));
    assert_eq!((gamma_q(2.0f64, 1e3), gamma_p(2.0f64, 1e3)), (0.0, 1.0));
    assert_eq!(gamma_q(0.5f64, 1e300), 0.0);
    assert!(gamma_p(1e300f64, 1e300).is_finite());
    // P(1, x) = 1 − e^(−x), Q(1, x) = e^(−x) (std's exp: Miri perturbs it by a few ulps).
    let tol = if cfg!(miri) { 1e-13 } else { 4e-16 };
    for x in [1e-10f64, 0.25, 1.0, 3.5, 20.0, 700.0] {
        let q = gamma_q(1.0, x);
        assert!((q - (-x).exp()).abs() <= tol * q, "Q(1, {x}) = {q}");
        let p = gamma_p(1.0, x);
        assert!((p + (-x).exp_m1()).abs() <= tol * p, "P(1, {x}) = {p}");
    }
    // P(1/2, x) = erf(√x), Q(1/2, x) = erfc(√x): erf(1) and erfc(2), correctly rounded.
    assert_eq!(gamma_p(0.5f64, 1.0), 0.842_700_792_949_714_9);
    assert_eq!(gamma_q(0.5f64, 4.0), 0.004677734981047266);
    // Inverses: the ends, outside [0, 1] and the round trip.
    for (a, p) in [
        (nan, 0.5),
        (1.0, nan),
        (0.0, 0.5),
        (-2.0, 0.5),
        (1.0, -0.1),
        (1.0, 1.1),
        (inf, 0.5),
    ] {
        assert!(
            gamma_p_inv(a, p).is_nan() && gamma_q_inv(a, p).is_nan(),
            "inverses at ({a}, {p})"
        );
    }
    assert_eq!(
        (gamma_p_inv(3.0f64, 0.0), gamma_p_inv(3.0f64, 1.0)),
        (0.0, inf)
    );
    assert_eq!(
        (gamma_q_inv(3.0f64, 1.0), gamma_q_inv(3.0f64, 0.0)),
        (0.0, inf)
    );
    assert_eq!(gamma_p_inv(0.5f32, 0.0), 0.0f32);
    for &a in &[1e-8f64, 0.3, 1.0, 4.5, 30.0, 1e3, 1e6] {
        for &p in &[1e-300f64, 1e-20, 0.01, 0.5, 0.9, 1.0 - 1e-12] {
            let x = gamma_p_inv(a, p);
            if x > 0.0 && x < f64::MAX {
                let back = gamma_p(a, x);
                assert!(
                    (back - p).abs() <= 1e-10 * p.min(1.0 - p).max(p * 1e-6) + 1e-300,
                    "P({a}, P⁻¹({p})) = {back}"
                );
            }
            let y = gamma_q_inv(a, p);
            if y > 0.0 && y < f64::MAX {
                let back = gamma_q(a, y);
                assert!(
                    (back - p).abs() <= 1e-10 * p.min(1.0 - p).max(p * 1e-6) + 1e-300,
                    "Q({a}, Q⁻¹({p})) = {back}"
                );
            }
        }
    }
}

#[test]
fn incomplete_gamma_identities() {
    // P + Q = 1, P increasing in x, decreasing in a; Q(a + 1, x) = Q(a, x) + x^a e^(−x)/Γ(a + 1).
    let mut a = 0.0625f64;
    while a < 300.0 {
        let mut prev = 0.0;
        let mut x = a / 16.0;
        while x < 8.0 * a + 10.0 {
            let (p, q) = (gamma_p(a, x), gamma_q(a, x));
            assert!((p + q - 1.0).abs() <= 2.3e-16, "P + Q at ({a}, {x})");
            assert!(p >= prev, "P increasing at ({a}, {x})");
            assert!(
                gamma_p(a * 1.0625, x) <= p,
                "P decreasing in a at ({a}, {x})"
            );
            if a < 100.0 && x < 100.0 {
                // (std's ln and exp, which Miri perturbs: a looser bound there.)
                let d = (a * x.ln() - x - lgamma(a + 1.0)).exp();
                let r = gamma_q(a + 1.0, x) - (q + d);
                let tol = if cfg!(miri) { 1e-11 } else { 1e-14 };
                assert!(
                    r.abs() <= tol * gamma_q(a + 1.0, x),
                    "recurrence at ({a}, {x})"
                );
            }
            prev = p;
            x *= 1.375;
        }
        a *= 2.5;
    }
}

// ===== ibeta (I, its complement and inverses) =====

const BETA_INC_F64: &str = include_str!("data/gamma_beta_inc_f64.txt");
const BETA_INC_F32: &str = include_str!("data/gamma_beta_inc_f32.txt");
const BETA_INC_INV_F64: &str = include_str!("data/gamma_beta_inc_inv_f64.txt");
const BETA_INC_C_INV_F64: &str = include_str!("data/gamma_beta_inc_c_inv_f64.txt");
const BETA_INC_INV_F32: &str = include_str!("data/gamma_beta_inc_inv_f32.txt");

#[test]
fn beta_inc_of_reals() {
    // I and 1 − I together, each to its own ulps.
    multi("beta_inc f64", BETA_INC_F64, all, F64, 1.5, |a| {
        vec![beta_inc(a[0], a[1], a[2]), beta_inc_c(a[0], a[1], a[2])]
    });
    multi("beta_inc f32", BETA_INC_F32, all, F32, 0.501, |a| {
        let (p, q, x) = (a[0] as f32, a[1] as f32, a[2] as f32);
        vec![beta_inc(p, q, x) as f64, beta_inc_c(p, q, x) as f64]
    });
}

/// Checks an inverse to `bound` ulps times its condition number `κ = |p/(x I′(x))|` (the second
/// reference of the data; at least 1): an inverse can be no better than its function's error
/// carried through `κ`.
#[track_caller]
fn inverse(name: &str, text: &str, p: Prec, bound: f64, f: impl Fn(&[f64]) -> f64) {
    let cs = cases(&select(text, all));
    assert!(!cs.is_empty(), "{name}: no cases");
    let w = worst(
        &cs,
        |a| vec![f(a)],
        |g, r| ulps(g[0], r[0], p) / r[1].0.max(1.0),
    );
    assert!(
        w.max <= bound,
        "{name}: {:.3} ulps (over max(1, κ)) > {bound} at {:?}: got {:?}, want {:?}",
        w.max,
        w.args,
        w.got,
        w.want
    );
}

#[test]
fn beta_inc_inverses() {
    inverse("beta_inc_inv f64", BETA_INC_INV_F64, F64, 1.5, |a| {
        beta_inc_inv(a[0], a[1], a[2])
    });
    inverse("beta_inc_c_inv f64", BETA_INC_C_INV_F64, F64, 1.5, |a| {
        beta_inc_c_inv(a[0], a[1], a[2])
    });
    inverse("beta_inc_inv f32", BETA_INC_INV_F32, F32, 0.6, |a| {
        beta_inc_inv(a[0] as f32, a[1] as f32, a[2] as f32) as f64
    });
}

#[test]
fn beta_inc_special_values() {
    let nan = f64::NAN;
    for (a, b, x) in [
        (nan, 1.0, 0.5),
        (1.0, nan, 0.5),
        (1.0, 1.0, nan),
        (0.0, 1.0, 0.5),
        (1.0, 0.0, 0.5),
        (-1.0, 1.0, 0.5),
        (f64::INFINITY, 1.0, 0.5),
        (1.0, f64::INFINITY, 0.5),
        (1.0, 1.0, -0.1),
        (1.0, 1.0, 1.1),
    ] {
        assert!(beta_inc(a, b, x).is_nan(), "beta_inc({a}, {b}, {x})");
        assert!(beta_inc_c(a, b, x).is_nan(), "beta_inc_c({a}, {b}, {x})");
        assert!(
            beta_inc_inv(a, b, x).is_nan(),
            "beta_inc_inv({a}, {b}, {x})"
        );
        assert!(
            beta_inc_c_inv(a, b, x).is_nan(),
            "beta_inc_c_inv({a}, {b}, {x})"
        );
    }
    for (a, b) in [(0.5, 0.5), (2.0, 3.0), (1e-3, 1e3), (1e6, 1e6)] {
        assert_eq!(beta_inc(a, b, 0.0), 0.0);
        assert_eq!(beta_inc(a, b, 1.0), 1.0);
        assert_eq!(beta_inc_c(a, b, 0.0), 1.0);
        assert_eq!(beta_inc_c(a, b, 1.0), 0.0);
        assert_eq!(beta_inc_inv(a, b, 0.0), 0.0);
        assert_eq!(beta_inc_inv(a, b, 1.0), 1.0);
        assert_eq!(beta_inc_c_inv(a, b, 0.0), 1.0);
        assert_eq!(beta_inc_c_inv(a, b, 1.0), 0.0);
    }
    // I_x(1, 1) = x, I_x(a, 1) = x^a, I_x(1, b) = 1 − (1 − x)^b.
    for x in [1e-100f64, 1e-10, 0.25, 0.5, 0.75, 1.0 - 1e-16] {
        assert_eq!(beta_inc(1.0, 1.0, x), x);
        assert_eq!(beta_inc_c(1.0, 1.0, 0.25), 0.75);
        let r = beta_inc(3.0, 1.0, x) / (x * x * x);
        assert!((r - 1.0).abs() < 4e-16, "x^3 at {x}: {r}");
        let c = beta_inc_c(1.0, 4.0, x) / (1.0 - x).powi(4);
        assert!((c - 1.0).abs() < 1e-15, "(1 − x)^4 at {x}: {c}");
    }
    assert_eq!(beta_inc(0.5f32, 0.5, 0.5), 0.5);
    assert!(beta_inc(0.5f32, f32::NAN, 0.5).is_nan());
}

#[test]
fn beta_inc_identities() {
    // The symmetry I_x(a, b) = 1 − I_(1−x)(b, a), monotonicity in x, and the round trips of the
    // inverses.
    let params = [
        (0.3, 0.7),
        (2.5, 4.0),
        (40.0, 60.0),
        (0.05, 300.0),
        (2e3, 3e3),
        (1e6, 2e6),
    ];
    for &(a, b) in &params {
        let mut last = 0.0;
        for k in 1..64 {
            let x = k as f64 / 64.0;
            let i = beta_inc(a, b, x);
            let j = beta_inc_c(b, a, 1.0 - x);
            assert!(
                (i - j).abs() <= 4e-16 * i.max(f64::MIN_POSITIVE),
                "symmetry at {a}, {b}, {x}: {i} {j}"
            );
            assert!(i >= last, "monotone at {a}, {b}, {x}");
            last = i;
            if i > 1e-300 && i < 1.0 {
                let y = beta_inc_inv(a, b, i);
                let back = beta_inc(a, b, y);
                assert!(
                    (back - i).abs() <= 1e-14 * i,
                    "round trip at {a}, {b}, {x}: {back} {i}"
                );
            }
            let q = beta_inc_c(a, b, x);
            if q > 1e-300 && q < 1.0 {
                let y = beta_inc_c_inv(a, b, q);
                let back = beta_inc_c(a, b, y);
                assert!(
                    (back - q).abs() <= 1e-14 * q,
                    "c round trip at {a}, {b}, {x}: {back} {q}"
                );
            }
        }
    }
}

#[test]
fn beta_inc_extremes() {
    // Parameters whose sum overflows: a point mass at the mean.
    assert_eq!(beta_inc(1e308, 1e308, 0.5), 0.5);
    assert_eq!(beta_inc(1e308, 1e308, 0.6), 1.0);
    assert_eq!(beta_inc_c(1e308, 1e308, 0.4), 1.0);
    assert_eq!(beta_inc_inv(1e308, 1e308, 0.3), 0.5);
    // Subnormal parameters: I_x(a, b) → b/(a + b), correctly rounded here.
    assert_eq!(beta_inc(1e-320, 1e-320, 0.3), 0.5);
    assert_eq!(beta_inc(2e-308, 3e-308, 0.7), 0.600_000_000_000_000_1);
    assert_eq!(beta_inc_c(1e-200, 1e-100, 0.5), 1e-100);
    // Results in the subnormals, and inverses of subnormal probabilities.
    let q = beta_inc_c(1e-310f64, 2.0, 0.3);
    assert!((q - 5.039728043259e-311).abs() <= 1e-323, "{q:e}");
    let x = beta_inc_inv(3.0, 4.0, 5e-324);
    assert!(x > 0.0 && beta_inc(3.0, 4.0, x) <= 1e-323, "{x:e}");
    assert_eq!(beta_inc_inv(0.01, 0.01, 1e-300), 0.0);
    assert_eq!(beta_inc_c_inv(5.0, 1e-320, 0.5), 1.0);
    let p = 3.076_371_787_25e-313f64;
    let x = beta_inc_inv(1.046_788_842_541_722_7, 29.153_617_815_665_974, p);
    assert!(
        (x / 9.999_999_999_927_277e-301 - 1.0).abs() < 1e-15,
        "{x:e}"
    );
    // Roots in the subnormals (by the leading terms, to 1e-300): the nearest subnormal, not the
    // least one or zero.
    let x = beta_inc_inv(
        0.693_109_705_948_962_4f64,
        1_283.903_673_112_845_8,
        7.138_512_485_316_517e-212,
    );
    assert!((x - 1.564_140_930_735_766_5e-308).abs() <= 5e-324, "{x:e}");
    let x = gamma_p_inv(0.156_312_500_449_204_2f64, 3.993_821_593_293_502_6e-51);
    assert_eq!(x, 5.0 * 5e-324);
    // Beside the least subnormal (a subnormal x beside b ≥ 8, review round 1): I at 5e-324 is p
    // (0.35 ulps; it was 1 % off), and a target two ulps below p has its root just below 2⁻¹⁰⁷⁴
    // (so 0, the documented rule), two ulps above it just above (so 5e-324, where 0 was given).
    let (a, b) = (0.715_728_087_772_465_8f64, 11.269_352_670_023_864);
    let p = 2.453_475_550_700_287_7e-231;
    assert!((beta_inc(a, b, 5e-324) / p - 1.0).abs() < 3e-16);
    assert_eq!(beta_inc_inv(a, b, f64::from_bits(p.to_bits() - 2)), 0.0);
    assert_eq!(beta_inc_inv(a, b, f64::from_bits(p.to_bits() + 2)), 5e-324);
    // A small a beside 8 <= b < 15 at a tiny x: the series (once NaN).
    let (a, b, x) = (
        0.001_805_913_590_426_878f64,
        13.552_467_641_142_972,
        7.838_550_122_480_618e-50,
    );
    assert!((beta_inc(a, b, x) - 0.819_947_165_055_452_1).abs() < 2e-16);
    assert!((beta_inc_c(a, b, x) - 0.180_052_834_944_547_9).abs() < 6e-17);
    // A parameter beyond 2⁹⁹⁶: I_c(1, b, x) = (1 − x)^b, the product b·x exact.
    let q = beta_inc_c(1.0f64, 1e300, 7.000_000_000_000_001e-298);
    assert!(
        (q / 9.859_676_543_758_132e-305 - 1.0).abs() < 4e-16,
        "{q:e}"
    );
}

#[test]
fn gamma_q_inv_in_the_tail_of_q() {
    // `Q(a, x) = q` for 1 ≤ a ≤ 50 and q from 10⁻³⁰⁰ to 10⁻²⁰, where −ln q > 3a takes the
    // start from the fixed point of the tail of `Q` (the speed pass): mpmath at 60 digits, the
    // root of `ln Q(a, x) − ln q` by `findroot` (to 10⁻⁵⁰), as `hi + lo`. The bound of the page
    // (0.6 ulp of the root) is asserted, so a root one float off fails.
    const CASES: [(f64, f64, f64, f64); 112] = [
        (1.0, 1e-300, 690.7755278982137, 2.3670096176709832e-14),
        (1.0, 1e-250, 575.6462732485114, 7.441568995976443e-16),
        (1.0, 1e-200, 460.51701859880916, -2.2080942657241066e-14),
        (1.0, 1e-150, 345.38776394910684, 1.1841282276040347e-14),
        (1.0, 1e-100, 230.25850929940458, -1.1069413097223595e-14),
        (1.0, 1e-60, 138.15510557964274, 4.7685975897958734e-15),
        (1.0, 1e-40, 92.10340371976183, -1.506890415327876e-15),
        (1.0, 1e-20, 46.051701859880914, -7.339520394182075e-16),
        (1.25, 1e-300, 692.5092406811294, 1.2243068882468331e-14),
        (1.25, 1e-250, 577.3345829463083, 4.509443131528603e-14),
        (1.25, 1e-200, 462.1498026366691, 2.520253137508011e-14),
        (1.25, 1e-150, 346.94904902830143, 2.1696739203338404e-14),
        (1.25, 1e-100, 231.71923759788407, 1.0454687394437498e-14),
        (1.25, 1e-60, 139.48965617089786, -1.3011966749711345e-14),
        (1.25, 1e-40, 93.3383871937701, 3.575476374653447e-15),
        (1.25, 1e-20, 47.11835002876929, 2.199722241955026e-15),
        (2.0, 1e-300, 697.3242113793526, -3.405499076585198e-14),
        (2.0, 1e-250, 582.0144852806608, -4.496818119169132e-14),
        (2.0, 1e-200, 466.6647703353572, -1.855605711861702e-14),
        (2.0, 1e-150, 351.25211109303456, 9.182347110469166e-15),
        (2.0, 1e-100, 235.72541016192412, 4.890479341464336e-15),
        (2.0, 1e-60, 143.12579205382013, -9.373416733912112e-15),
        (2.0, 1e-40, 96.68515330518728, 1.2771925087796459e-15),
        (2.0, 1e-20, 49.98319798709075, -2.6786910441852104e-15),
        (3.5, 1e-300, 705.9770535723017, 4.3018529167566807e-14),
        (3.5, 1e-250, 590.4015430514652, 2.027384201953012e-14),
        (3.5, 1e-200, 474.7281698353882, -1.61136839344013e-14),
        (3.5, 1e-150, 358.90137995896856, -1.8526553312479316e-14),
        (3.5, 1e-100, 242.79842164284838, 3.931146910377954e-15),
        (3.5, 1e-60, 149.48893891402344, 6.9467093638657345e-15),
        (3.5, 1e-40, 102.50157271025061, 6.628639448170493e-15),
        (3.5, 1e-20, 54.91072183699409, 3.11712504673297e-15),
        (5.0, 1e-300, 713.8859780649443, -4.9499232759576984e-15),
        (5.0, 1e-250, 598.0496138649248, -1.8483234992697832e-14),
        (5.0, 1e-200, 482.05955030654906, -1.2219442182663254e-14),
        (5.0, 1e-150, 365.8293417935189, -1.0333584372072013e-14),
        (5.0, 1e-100, 249.16910020808962, -1.2751722365790936e-14),
        (5.0, 1e-60, 155.1813716934729, -2.2027142802501047e-15),
        (5.0, 1e-40, 107.67947589349083, 2.024661707911115e-15),
        (5.0, 1e-20, 59.27072129552329, 2.898187526474696e-15),
        (7.75, 1e-300, 727.2368338909743, 3.2757684053266368e-15),
        (7.75, 1e-250, 610.9330716312724, -5.27640692932288e-14),
        (7.75, 1e-200, 494.3775511492638, 2.190754070708609e-14),
        (7.75, 1e-150, 377.43066407204833, 6.3879126458502875e-15),
        (7.75, 1e-100, 259.788377251782, 7.415621258308876e-15),
        (7.75, 1e-60, 164.62085101124612, -6.090818095978444e-15),
        (7.75, 1e-40, 116.23777356578442, 4.6262083640865245e-15),
        (7.75, 1e-20, 66.45841244810269, 1.1081065394127647e-15),
        (10.0, 1e-300, 737.4143124556944, -3.9284239940787625e-14),
        (10.0, 1e-250, 620.7371970160995, 4.066462116149705e-14),
        (10.0, 1e-200, 503.73157451671995, 1.2827380315137418e-14),
        (10.0, 1e-150, 386.21704793369236, 2.352517167773229e-14),
        (10.0, 1e-100, 267.80299764163215, -2.028726951232554e-14),
        (10.0, 1e-60, 171.71954011541675, -1.1773058561794734e-14),
        (10.0, 1e-40, 122.66198878161858, -6.6977426065965805e-15),
        (10.0, 1e-20, 71.85311626769968, -2.0193873726174365e-15),
        (12.5, 1e-300, 748.1592317396277, -4.8980777879230086e-14),
        (12.5, 1e-250, 631.0756834834955, 2.149978440194731e-14),
        (12.5, 1e-200, 513.5814717073567, 4.152805036288458e-14),
        (12.5, 1e-150, 395.45322449962197, 6.189751995786783e-15),
        (12.5, 1e-100, 276.2098770483148, 1.0219477452208048e-14),
        (12.5, 1e-60, 179.15135515783018, 1.0625241753919586e-14),
        (12.5, 1e-40, 129.38328932509825, 1.2932468202170703e-14),
        (12.5, 1e-20, 77.50424678810471, -5.885860590749002e-15),
        (15.0, 1e-300, 758.4406160027348, -3.750834188652557e-14),
        (15.0, 1e-250, 640.9586092820043, -6.2022379616965605e-15),
        (15.0, 1e-200, 522.9866611734905, 2.6056680393799954e-14),
        (15.0, 1e-150, 404.26052998584856, 8.844077023848164e-15),
        (15.0, 1e-100, 284.21378522456, 1.8079487926265803e-14),
        (15.0, 1e-60, 186.21846217475124, 1.3110131553852617e-14),
        (15.0, 1e-40, 135.77396087752535, -1.2156349602521295e-14),
        (15.0, 1e-20, 82.88740560569713, 6.0357726337040666e-15),
        (17.25, 1e-300, 767.3705395436773, 2.2406298333451662e-14),
        (17.25, 1e-250, 649.5361471017613, 2.7715018237535925e-15),
        (17.25, 1e-200, 531.1426553212079, -3.671047639139674e-15),
        (17.25, 1e-150, 411.89060401105974, -1.6061259078291504e-14),
        (17.25, 1e-100, 291.14045099383134, -6.38824924129423e-15),
        (17.25, 1e-60, 192.33048622338083, -9.879968547121611e-15),
        (17.25, 1e-40, 141.30205282691645, -5.4108725143737634e-15),
        (17.25, 1e-20, 87.55332861081001, 4.6509273142962695e-15),
        (20.0, 1e-300, 777.9365929966503, 4.7040461554594554e-14),
        (20.0, 1e-250, 659.6788779213207, -3.6003147956932694e-14),
        (20.0, 1e-200, 540.7800722275616, -4.7455057514286666e-14),
        (20.0, 1e-150, 420.89943973247273, -2.3690777757781007e-14),
        (20.0, 1e-100, 299.3122263553858, -1.6457399649710516e-14),
        (20.0, 1e-60, 199.53888173464816, 1.1187473137516106e-14),
        (20.0, 1e-40, 147.82445334257247, 8.163520483695553e-15),
        (20.0, 1e-20, 93.07062456222539, 2.6698167419941243e-15),
        (30.0, 1e-300, 813.9083065028428, 5.4648038554736315e-14),
        (30.0, 1e-250, 694.1706604661815, -2.951128812327334e-14),
        (30.0, 1e-200, 573.513430224711, -5.834245899271105e-15),
        (30.0, 1e-150, 451.4590082578617, 1.5565828807736735e-14),
        (30.0, 1e-100, 327.00313493485527, 1.6030539884906484e-14),
        (30.0, 1e-60, 223.96980549965636, 2.3410359027449814e-15),
        (30.0, 1e-40, 169.96403081758785, -6.528075465296139e-15),
        (30.0, 1e-20, 111.90149889835537, -4.784962729967733e-15),
        (40.0, 1e-300, 847.1228217031288, 4.8058064161100783e-14),
        (40.0, 1e-250, 725.9831954904735, -4.3396847619978814e-14),
        (40.0, 1e-200, 603.6700028252254, 2.8861475580292187e-14),
        (40.0, 1e-150, 479.5845061905084, 1.3033286982118352e-14),
        (40.0, 1e-100, 352.4781733553545, 8.305468673217568e-15),
        (40.0, 1e-60, 246.47839667585242, 4.6830303857458355e-15),
        (40.0, 1e-40, 190.41899599206909, 7.145177688104776e-15),
        (40.0, 1e-20, 129.43905776551796, 1.1999294278447301e-14),
        (50.0, 1e-300, 878.3937665919384, 1.3597571521833334e-14),
        (50.0, 1e-250, 755.9160349066102, 3.368128434543432e-14),
        (50.0, 1e-200, 632.0297132807783, -4.007769940334807e-14),
        (50.0, 1e-150, 506.02649685924524, -9.458709412932282e-15),
        (50.0, 1e-100, 376.4387823636859, -1.656212033073376e-14),
        (50.0, 1e-60, 267.69265764300155, 2.4835357581249746e-14),
        (50.0, 1e-40, 209.7522949699067, 1.0407431701625648e-14),
        (50.0, 1e-20, 146.1295669357228, 5.8604538965065935e-15),
    ];
    let mut worst = 0.0f64;
    for (a, q, hi, lo) in CASES {
        let x = gamma_q_inv(a, q);
        let e = ulps(x, (hi, lo), F64);
        assert!(
            e <= 0.6,
            "gamma_q_inv({a}, {q:e}) = {x:e}: {e:.3} ulps (want {hi:e})"
        );
        worst = worst.max(e);
    }
    eprintln!("gamma_q_inv tail: {worst:.3} ulps");
}
