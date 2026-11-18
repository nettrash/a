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
    // A subnormal result is rounded twice.
    real("gamma f64 subnormal", GAMMA_F64, subnormal, F64, 1.0, |a| {
        gamma(a[0])
    });
    real("gamma f32", GAMMA_F32, all, F32, 0.501, |a| {
        gamma(a[0] as f32) as f64
    });
}

#[test]
fn lgamma_of_reals() {
    real("lgamma f64", LGAMMA_F64, all, F64, 0.6, |a| lgamma(a[0]));
    real("lgamma f32", LGAMMA_F32, all, F32, 0.501, |a| {
        lgamma(a[0] as f32) as f64
    });
    // Beside the zeros of ln |Γ| on the negative axis the error is absolute.
    absolute("lgamma near its negative zeros", LGAMMA_NEGZERO, 0.1, |a| {
        lgamma(a[0])
    });
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

#[test]
fn gamma_of_complex() {
    complex("gamma c64", GAMMA_C64, all, 1.25, |re, im| {
        let z = gamma(Complex::new(re, im));
        (z.re, z.im)
    });
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
    for n in 0..=60u32 {
        let mut c = 1u128;
        for k in 0..=n {
            if c < 1 << 53 {
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
}
