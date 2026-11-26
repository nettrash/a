//! The numerical derivatives of [`a::math::numerics::diff`], by Richardson extrapolation of
//! central differences, against derivatives the test knows in closed form: at points that are
//! not binary fractions, at huge and tiny arguments, next to singularities, for periodic
//! functions far from the origin and narrow peaks, in `f64` and `f32`; and an honest `None` at
//! kinks, jumps and poles, and when the table cannot settle.

use a::math::numerics::{
    DiffOptions, derivative, derivative_with, second_derivative, second_derivative_with,
};

/// A function of the tests.
type Function = fn(f64) -> f64;

/// A point, the derivative there, and the function.
type Point = (f64, f64, Function);

/// The relative error of `got` against `want`, the absolute one when `want` is zero.
fn rel(got: f64, want: f64) -> f64 {
    if want == 0.0 {
        got.abs()
    } else {
        ((got - want) / want).abs()
    }
}

/// A uniform number in [0, 1) from a seeded linear congruential generator.
fn uniform(seed: &mut u64) -> f64 {
    *seed = seed
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*seed >> 11) as f64 / (1u64 << 53) as f64
}

#[test]
fn the_first_derivative_of_a_smooth_function_is_its_own() {
    // name, the point, the function and its derivative.
    type Case<'a> = (&'a str, f64, fn(f64) -> f64, fn(f64) -> f64);
    let cases: &[Case] = &[
        ("sin", 1.0, |x| x.sin(), |x| x.cos()),
        ("exp", 0.5, |x| x.exp(), |x| x.exp()),
        ("x³", 2.0, |x| x.powi(3), |x| 3.0 * x * x),
        ("1/x", 4.0, |x| 1.0 / x, |x| -1.0 / (x * x)),
        ("atan", 1.0, |x| x.atan(), |x| 1.0 / (1.0 + x * x)),
        ("ln", 0.37, |x| x.ln(), |x| 1.0 / x),
        ("tan", 1.5, |x| x.tan(), |x| 1.0 / (x.cos() * x.cos())),
        (
            "runge",
            0.2,
            |x| 1.0 / (1.0 + 25.0 * x * x),
            |x| -50.0 * x / (1.0 + 25.0 * x * x).powi(2),
        ),
    ];
    for (name, x, f, d) in cases {
        let got = derivative(f, *x).unwrap();
        let want = d(*x);
        assert!(
            rel(got, want) <= 1e-12,
            "{name}′ at {x}: {got} against {want}"
        );
    }
    // The example of the doc page: fourteen digits of the slope of the sine at 1.
    assert!((derivative(|x: f64| x.sin(), 1.0).unwrap() - 1.0f64.cos()).abs() < 1e-14);
}

#[test]
fn the_second_derivative_is_the_derivative_of_the_derivative() {
    type Case<'a> = (&'a str, f64, f64, fn(f64) -> f64);
    let cases: &[Case] = &[
        ("sin", 1.0, -1.0_f64.sin(), |x| x.sin()),
        ("exp", 0.25, 0.25f64.exp(), |x| x.exp()),
        ("x⁶", 1.5, 30.0 * 1.5 * 1.5 * 1.5 * 1.5, |x| x.powi(6)),
        ("cos", 0.0, -1.0, |x| x.cos()),
        ("ln", 0.37, -1.0 / (0.37 * 0.37), |x| x.ln()),
    ];
    for (name, x, want, f) in cases {
        let got = second_derivative(f, *x).unwrap();
        assert!(
            rel(got, *want) <= 1e-9,
            "{name}″ at {x}: {got} against {want}"
        );
    }
}

#[test]
fn polynomials_of_low_degree_at_points_that_are_not_binary_fractions() {
    // Audit n_interp-1: a quotient with no h² term (a line, a parabola; for the second
    // derivative also a cubic) never showed the 4 : 1 gaps the old step scan waited for, which
    // ran to steps the float grid could not resolve, where every numerator was 0 and Some(0.0)
    // came back. These are the audit's cases and their closed forms.
    let first: [Point; 4] = [
        (0.3, -3.3, |x| 7.1 - 3.3 * x),
        (1.3, 2.6, |x| x * x),
        (0.7, 4.4, |x| x * x + 3.0 * x + 1.0),
        (3.0, -2.0 * (-3.0f64).exp(), |x| x * (-x).exp()),
    ];
    for (x, want, f) in first {
        let got = derivative(f, x).unwrap();
        assert!(rel(got, want) < 1e-12, "at {x}: {got} against {want}");
    }
    let second: [Point; 3] = [
        (1.3, 7.8, |x| x * x * x - 2.0 * x),
        (1.0, -0.5, |x| x.atan()),
        (1.3, 2.0, |x| x * x),
    ];
    for (x, want, f) in second {
        let got = second_derivative(f, x).unwrap();
        assert!(rel(got, want) < 1e-9, "at {x}: {got} against {want}");
    }
    // Every polynomial of degree 1 to 4 with these coefficients, at eight points that are not
    // binary fractions (the audit found 31 of 56 first and 39 of 56 second derivatives wrong).
    let polys: [&[f64]; 6] = [
        &[7.1, -3.3],
        &[0.4, 1.7, -2.9],
        &[1.0, 3.0, 1.0],
        &[-0.3, -2.0, 0.0, 1.0],
        &[0.0, 0.0, 0.0, 2.7],
        &[1.1, -0.6, 0.35, 0.8, -0.45],
    ];
    let eval = |c: &[f64], x: f64| c.iter().rev().fold(0.0, |s, &a| s * x + a);
    let deriv = |c: &[f64]| -> Vec<f64> {
        c.iter()
            .enumerate()
            .skip(1)
            .map(|(k, &a)| a * k as f64)
            .collect()
    };
    for c in polys {
        let (d1, d2) = (deriv(c), deriv(&deriv(c)));
        for x in [0.3, -0.77, 1.3, 2.9, -4.1, 13.7, 0.123, 1e3 + 0.1] {
            let (w1, w2) = (eval(&d1, x), eval(&d2, x));
            let g1 = derivative(|t| eval(c, t), x).unwrap();
            let scale1 = w1.abs().max(eval(&deriv(&d1), x).abs()).max(1e-300);
            assert!(
                (g1 - w1).abs() <= 1e-9 * scale1,
                "{c:?}′ at {x}: {g1} against {w1}"
            );
            if d2.is_empty() {
                continue;
            }
            let g2 = second_derivative(|t| eval(c, t), x).unwrap();
            let scale2 = w2.abs().max(1.0);
            assert!(
                (g2 - w2).abs() <= 1e-5 * scale2,
                "{c:?}″ at {x}: {g2} against {w2}"
            );
        }
    }
}

#[test]
fn f32_derivatives_reach_f32_precision() {
    let d = derivative(|x: f32| x * x, 3.0f32).unwrap();
    assert!((f64::from(d) - 6.0).abs() < 1e-5);
    let d = derivative(|x: f32| x.sin(), 1.0f32).unwrap();
    assert!((f64::from(d) - 1.0_f64.cos()).abs() < 1e-5);
    // The audit's f32 cases: a line and a parabola at points that are not binary fractions, and
    // the curvature of the sine at 1, which came back as Some(0.0).
    let d = derivative(|x: f32| 7.1 - 3.3 * x, 0.3f32).unwrap();
    assert!((f64::from(d) + 3.3).abs() < 1e-5, "{d}");
    let d = derivative(|x: f32| x * x, 1.3f32).unwrap();
    assert!((f64::from(d) - 2.6).abs() < 1e-5, "{d}");
    let d = second_derivative(|x: f32| x.sin(), 1.0f32).unwrap();
    assert!((f64::from(d) + 1.0f64.sin()).abs() < 1e-3, "{d}");
    let d = second_derivative(|x: f32| x.sqrt(), 1.0f32).unwrap();
    assert!((f64::from(d) + 0.25).abs() < 1e-3, "{d}");
    let d = derivative(|x: f32| x.ln(), 1e-20f32).unwrap();
    assert!(rel(f64::from(d), 1e20) < 1e-4, "{d}");
    // A kink in f32 too.
    assert_eq!(derivative(|x: f32| (x - 0.3).abs(), 0.3f32), None);
}

#[test]
fn kinks_jumps_poles_and_infinite_slopes_have_no_derivative() {
    // Audit n_interp-1: away from 0 these came back as Some(0.0). The symmetric quotient of a
    // kink is the mean of the two slopes, not a derivative: the one-sided quotients must agree.
    let none: [(&str, Function, f64); 11] = [
        ("|x|", |x| x.abs(), 0.0),
        ("|x − 0.3|", |x| (x - 0.3).abs(), 0.3),
        ("|x − 1|", |x| (x - 1.0).abs(), 1.0),
        ("max(x, 2x − 1)", |x| x.max(2.0 * x - 1.0), 1.0),
        ("floor", |x| x.floor(), 2.0),
        ("a step", |x| if x >= 2.5 { 1.0 } else { 0.0 }, 2.5),
        ("cbrt(x − 1)", |x| (x - 1.0).cbrt(), 1.0),
        ("√(x − 1)", |x| (x - 1.0).sqrt(), 1.0),
        ("1/x", |x| 1.0 / x, 0.0),
        ("1/(x − 2)", |x| 1.0 / (x - 2.0), 2.0),
        ("sin(1/(x − 1))", |x| (1.0 / (x - 1.0)).sin(), 1.0),
    ];
    for (name, f, x) in none {
        assert_eq!(derivative(f, x), None, "{name} at {x}");
        assert_eq!(second_derivative(f, x), None, "{name}″ at {x}");
    }
    // A kink of the first derivative is one of the second.
    assert_eq!(second_derivative(|x: f64| x * x.abs(), 0.0), None);
    assert_eq!(
        second_derivative(|x: f64| (x - 1.3) * (x - 1.3).abs(), 1.3),
        None
    );
    // x·|x| is differentiable at 0, but its quotient is h itself, not a series in h², so the
    // table cannot settle and says so rather than guess.
    assert_eq!(derivative(|x: f64| x * x.abs(), 0.0), None);
    // tan at the float nearest π/2, 6e-17 from its pole: steep but smooth there, and either
    // right or nothing.
    let x = core::f64::consts::FRAC_PI_2;
    if let Some(d) = derivative(|t: f64| t.tan(), x) {
        assert!(rel(d, 1.0 / (x.cos() * x.cos())) < 1e-6, "{d}");
    }
}

#[test]
fn periodic_functions_far_from_the_origin_are_not_aliased() {
    // Audit n_interp-2: the old scan started at max(|x|, 1), dozens of periods wide, and on
    // its halving steps a sinusoid can look like a slow one. The ladder still starts at a
    // fraction of |x|, but an answer must also predict a quotient off the ladder.
    let x = 399.7335879946275f64;
    let d = derivative(|t: f64| t.sin(), x).unwrap();
    assert!((d - x.cos()).abs() < 1e-11, "{d} against {}", x.cos());
    let x = 1234567.891f64;
    let d = derivative(|t: f64| t.sin(), x).unwrap();
    assert!((d - x.cos()).abs() < 1e-8, "{d} against {}", x.cos());
    // sin(812.598x + 2.739) at 0.1203, which fooled SciPy too.
    let d = derivative(|t: f64| (812.598 * t + 2.739).sin(), 0.1203).unwrap();
    let want = 812.598 * (812.598f64 * 0.1203 + 2.739).cos();
    assert!(rel(d, want) < 1e-9, "{d} against {want}");
    // sin(10¹²x) at 0: the slope 10¹² or nothing, never the alias.
    if let Some(d) = derivative(|t: f64| (1e12 * t).sin(), 0.0) {
        assert!(rel(d, 1e12) < 1e-8, "{d}");
    }
    // 2000 random points of [1e2, 1e4] (136 of them wrong before), and as many of [1e4, 1e6],
    // both derivatives: whatever is returned is right.
    let n = if cfg!(miri) { 20 } else { 2000 };
    let mut seed = 21;
    for (lo, hi) in [(1e2f64, 1e4f64), (1e4, 1e6)] {
        let mut answered = 0;
        for _ in 0..n {
            let x = lo * (hi / lo).powf(uniform(&mut seed));
            if let Some(d) = derivative(|t: f64| t.sin(), x) {
                answered += 1;
                assert!((d - x.cos()).abs() < 1e-8 * x.max(1.0), "sin′ at {x}: {d}");
            }
            if let Some(d) = second_derivative(|t: f64| t.sin(), x) {
                assert!((d + x.sin()).abs() < 1e-4, "sin″ at {x}: {d}");
            }
        }
        assert!(
            answered * 10 >= n * 9,
            "{answered} of {n} answered on [{lo}, {hi}]"
        );
    }
    // sin(ωx + φ) for random ω in [1e2, 1e4] and x in [-2, 2].
    for _ in 0..n {
        let w = 1e2 * 100f64.powf(uniform(&mut seed));
        let p = 6.3 * uniform(&mut seed);
        let x = 4.0 * uniform(&mut seed) - 2.0;
        if let Some(d) = derivative(|t: f64| (w * t + p).sin(), x) {
            let want = w * (w * x + p).cos();
            assert!(
                (d - want).abs() < 1e-7 * w,
                "ω = {w} at {x}: {d} against {want}"
            );
        }
    }
}

/// A number of `[lo, hi]` whose logarithm is uniform.
fn log_uniform(seed: &mut u64, lo: f64, hi: f64) -> f64 {
    lo * (hi / lo).powf(uniform(seed))
}

#[test]
fn loose_tolerances_let_no_alias_through() {
    // Review round 1, B3: a periodic function sampled at steps of many periods gives quotients of
    // the size of the reciprocal step, among which a loose tolerance is met by chance, and on the
    // halvings of a step of 2^j periods the values of a slower sinusoid. One quotient off the
    // ladder, compared at the tolerance, let 53 of 20000 such answers through at tol 1e-2, up to
    // the whole slope wrong. Now three quotients off the ladder must agree, the ladder goes on
    // below an answer while a finer step could do better, a finer step that shows the function
    // varying faster than the answer's steps resolved leaves no answer, and so do values at the
    // floats next to x that move further than the answer allows.
    let loose = DiffOptions::new().with_tol(1e-2);
    let x = 563.0725f64;
    let d = derivative_with(|t: f64| t.sin(), x, &loose).unwrap();
    assert!((d - x.cos()).abs() < 1e-2, "{d} against {}", x.cos());
    // The sine at 20000 points of [1e2, 1e4]: 37 answers more than 0.1 off before.
    let n = if cfg!(miri) { 20 } else { 20000 };
    for k in 0..n {
        let x = 100.0 + 9900.0 * (k as f64 + 0.5) / n as f64;
        if let Some(d) = derivative_with(|t: f64| t.sin(), x, &loose) {
            assert!((d - x.cos()).abs() <= 2e-2, "sin′ at {x}: {d}");
        }
    }
    // sin(ω(t − x) + φ), computed accurately, at x in ±[1, 1e6] and ω in [0.1, 1e4]: every answer
    // within twice the tolerance of the slope scale ω (ω² for the second derivative).
    let trials = if cfg!(miri) { 4 } else { 3000 };
    let mut seed = 31;
    for tol in [1e-3, 3e-3, 1e-2, 3e-2, 1e-1] {
        let opts = DiffOptions::new().with_tol(tol);
        for _ in 0..trials {
            let w = log_uniform(&mut seed, 0.1, 1e4);
            let p = 2.0 * core::f64::consts::PI * uniform(&mut seed);
            let sign = if uniform(&mut seed) < 0.5 { -1.0 } else { 1.0 };
            let x = sign * log_uniform(&mut seed, 1.0, 1e6);
            let f = |t: f64| (w * (t - x) + p).sin();
            if let Some(d) = derivative_with(f, x, &opts) {
                let want = w * p.cos();
                assert!(
                    (d - want).abs() <= 2.0 * tol * w,
                    "tol {tol}, ω = {w}, x = {x}, φ = {p}: {d} against {want}"
                );
            }
            if let Some(d) = second_derivative_with(f, x, &opts) {
                let want = -w * w * p.sin();
                assert!(
                    (d - want).abs() <= 2.0 * tol * w * w,
                    "tol {tol}, ω = {w}, x = {x}, φ = {p}: {d} against {want}″"
                );
            }
        }
    }
}

#[test]
fn each_check_against_aliases_is_needed() {
    // Review round 1, B3: each case below passes all but one of the checks of the module's
    // overview, and what it gave without that one was wrong (or, for the comparison of answers,
    // missing); they were found by taking the checks out one at a time.
    // Going on below the first answer that passes: from steps of thousands of periods it was
    // −1.3e-5, for the slope −0.133 of this sinusoid.
    let (w, x, p) = (
        1.7151738683266868f64,
        111937.39816205553f64,
        1.648408632431408f64,
    );
    let opts = DiffOptions::new().with_tol(3e-2);
    let d = derivative_with(|t: f64| (w * (t - x) + p).sin(), x, &opts).unwrap();
    assert!((d - w * p.cos()).abs() <= 3e-2 * w, "{d}");
    // Comparing answers by their error relative to their size: an alias of size 5e-6 had the
    // smaller error in absolute terms beside the curvature 7.9e6.
    let (w, x, p) = (
        2937.1072830940175f64,
        -4180.865110224713f64,
        5.119722922738454f64,
    );
    let opts = DiffOptions::new().with_tol(0.1);
    let d = second_derivative_with(|t: f64| (w * (t - x) + p).sin(), x, &opts).unwrap();
    assert!(rel(d, -w * w * p.sin()) < 1e-6, "{d}");
    // No answer when a finer step shows the function varying faster than the steps of the answer
    // resolved: in f32 it was −0.0134, for −435.7.
    let (w, x, p) = (962.84937f32, -606.1065f32, 2.0403302f32);
    let (ww, pp) = (f64::from(w), f64::from(p));
    let opts = DiffOptions::new().with_tol(0.1f32);
    let d = derivative_with(|t: f32| (w * (t - x) + p).sin(), x, &opts);
    assert!(
        d.is_none_or(|d| (f64::from(d) - ww * pp.cos()).abs() <= 0.2 * ww),
        "{d:?}"
    );
    // Three quotients off the ladder, not one: a curvature at a loose tolerance in f32.
    let (w, x, p) = (279.66592f32, 58.68149f32, 0.40944713f32);
    let (ww, pp) = (f64::from(w), f64::from(p));
    let d = second_derivative_with(|t: f32| (w * (t - x) + p).sin(), x, &opts);
    assert!(
        d.is_none_or(|d| (f64::from(d) + ww * ww * pp.sin()).abs() <= 0.2 * ww * ww),
        "{d:?}"
    );
    // The values at the floats next to x: a period of two thirds of their spacing here, which
    // no step of the ladder resolves, gave −1.47 for the slope 778 in f32 at the defaults.
    let (w, x, p) = (9632.745f32, -8336.077f32, 1.4899014f32);
    let (ww, pp) = (f64::from(w), f64::from(p));
    let d = derivative(|t: f32| (w * (t - x) + p).sin(), x);
    let t1 = f64::from(f32::EPSILON.sqrt());
    assert!(
        d.is_none_or(|d| (f64::from(d) - ww * pp.cos()).abs() <= 2.0 * t1 * ww),
        "{d:?}"
    );
}

#[test]
fn f32_derivatives_are_not_aliased_at_the_defaults_or_at_a_loose_tolerance() {
    // Review round 1, B3: the default tolerance of the f32 second derivative, ε^(1/3) = 4.9e-3,
    // let aliases through: −319 for the curvature −369922 of this sinusoid.
    let (w, x, p) = (628.9612f32, 229.22186f32, 1.2085681f32);
    let d = second_derivative(|t: f32| (w * (t - x) + p).sin(), x).unwrap();
    let want = -f64::from(w) * f64::from(w) * f64::from(p).sin();
    assert!(rel(f64::from(d), want) < 1e-3, "{d} against {want}");
    let (w, x, p) = (52.31078f32, 380.14752f32, 1.6157597f32);
    let d = second_derivative(|t: f32| (w * (t - x) + p).sin(), x).unwrap();
    let want = -f64::from(w) * f64::from(w) * f64::from(p).sin();
    assert!(rel(f64::from(d), want) < 1e-3, "{d} against {want}");
    // sin(ω(t − x) + φ) in f32 at x in ±[1, 1e3] and ω in [0.1, 1e3], at the default tolerances
    // and at 1e-2: every answer within twice the tolerance of the slope scale.
    let trials = if cfg!(miri) { 4 } else { 3000 };
    let mut seed = 77;
    let (t1, t2) = (f32::EPSILON.sqrt(), f32::EPSILON.cbrt());
    for loose in [false, true] {
        let (tol1, tol2) = if loose { (1e-2, 1e-2) } else { (t1, t2) };
        let (o1, o2) = (
            DiffOptions::new().with_tol(tol1),
            DiffOptions::new().with_tol(tol2),
        );
        for _ in 0..trials {
            let w = log_uniform(&mut seed, 0.1, 1e3) as f32;
            let p = (2.0 * core::f64::consts::PI * uniform(&mut seed)) as f32;
            let sign = if uniform(&mut seed) < 0.5 { -1.0 } else { 1.0 };
            let x = (sign * log_uniform(&mut seed, 1.0, 1e3)) as f32;
            let f = |t: f32| (w * (t - x) + p).sin();
            let (ww, pp) = (f64::from(w), f64::from(p));
            if let Some(d) = derivative_with(f, x, &o1) {
                let want = ww * pp.cos();
                assert!(
                    (f64::from(d) - want).abs() <= 2.0 * f64::from(tol1) * ww,
                    "tol {tol1}, ω = {w}, x = {x}, φ = {p}: {d} against {want}"
                );
            }
            if let Some(d) = second_derivative_with(f, x, &o2) {
                let want = -ww * ww * pp.sin();
                assert!(
                    (f64::from(d) - want).abs() <= 2.0 * f64::from(tol2) * ww * ww,
                    "tol {tol2}, ω = {w}, x = {x}, φ = {p}: {d} against {want}″"
                );
            }
        }
    }
}

#[test]
fn small_derivatives_and_narrow_peaks_keep_their_relative_accuracy() {
    // Audit n_interp-3: both acceptance tests were absolute below 1, so a Gaussian of width
    // 1e-3 gave Some(0.0), exp′(−700) came back with the wrong sign and ln′(1e10) 0.4 % off.
    let s = 1e-3f64;
    let d = derivative(|t: f64| (-0.5 * (t / s) * (t / s)).exp(), s).unwrap();
    assert!(rel(d, -(-0.5f64).exp() / s) < 1e-10, "{d}");
    let d = derivative(|t: f64| t.exp(), -700.0).unwrap();
    assert!(rel(d, (-700.0f64).exp()) < 1e-12, "{d}");
    let d = derivative(|t: f64| t.ln(), 1e10).unwrap();
    assert!(rel(d, 1e-10) < 1e-12, "{d}");
    for c in [1e-6, 1e-9, 1e-12, 1e-100, 1e-300] {
        let d = derivative(|t: f64| c * t.sin(), 1.0).unwrap();
        assert!(rel(d, c * 1f64.cos()) < 1e-12, "c = {c}: {d}");
    }
    let d = derivative(|t: f64| 1e-18 * (100.0 * t).sin(), 0.0).unwrap();
    assert!(rel(d, 1e-16) < 1e-12, "{d}");
    // Gaussians of widths 1e-6 to 1e-2 within two widths of their centres (50.9 % to 100 %
    // Some(0.0) before): an answer, when there is one, is the derivative.
    let n = if cfg!(miri) { 20 } else { 2000 };
    let mut seed = 5;
    let mut answered = 0;
    for _ in 0..n {
        let width = 1e-6 * 1e4f64.powf(uniform(&mut seed));
        let centre = 2.0 * uniform(&mut seed) - 1.0;
        let x = centre + width * (4.0 * uniform(&mut seed) - 2.0);
        let g = |t: f64| (-0.5 * ((t - centre) / width).powi(2)).exp();
        let want = -(x - centre) / (width * width) * g(x);
        if let Some(d) = derivative(g, x) {
            answered += 1;
            // Relative to the slopes around the point, 1/width in size, since the derivative
            // itself vanishes at the centre.
            assert!(
                (d - want).abs() < 1e-7 * want.abs().max(1.0 / width),
                "width {width} at {x}: {d} against {want}"
            );
        }
    }
    assert!(answered * 10 >= n * 9, "{answered} of {n} answered");
}

#[test]
fn tiny_arguments_of_functions_singular_at_zero_are_differentiated() {
    // Audit n_interp-12: the old scan's steps went no lower than max(|x|, 1)·2⁻⁵⁹, so ln, √ and
    // 1/x below 1e-17 were evaluated at negative arguments at every step and gave None. A second
    // ladder starts at the scale of x itself.
    let cases: [(Function, f64, f64); 5] = [
        (|t| t.ln(), 1e-100, 1e100),
        (|t| t.ln(), 1e-300, 1e300),
        (|t| t.sqrt(), 1e-30, 5e14),
        (|t| 1.0 / t, 1e-20, -1e40),
        (|t| t.sqrt(), 1e-300, 5e149),
    ];
    for (f, x, want) in cases {
        let d = derivative(f, x).unwrap();
        assert!(rel(d, want) < 1e-10, "at {x}: {d} against {want}");
    }
    let d = second_derivative(|t: f64| t.ln(), 1e-100).unwrap();
    assert!(rel(d, -1e200) < 1e-8, "{d}");
}

#[test]
fn the_function_is_sampled_within_a_fraction_of_the_point() {
    // The first step is 0.37·|x| (0.37 below 1), so f is never asked for a value as far from x
    // as x is from 0 (the old scan sampled sin at 1000.5 on [0, 2001]).
    for x in [1000.5f64, -42.0, 1e15, 3.0e-7] {
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        let reach = 0.37 * x.abs().max(1.0) * (1.0 + 1e-15);
        let _ = derivative(
            |t: f64| {
                lo = lo.min(t);
                hi = hi.max(t);
                t.sin()
            },
            x,
        );
        assert!(
            x - lo <= reach && hi - x <= reach,
            "{x}: sampled [{lo}, {hi}]"
        );
    }
}

#[test]
fn no_step_is_finer_than_sixteen_units_in_the_last_place() {
    // Below that the step spans a few floats, on which any function looks smooth (review round 1:
    // removing the floor passed the tests). The sine at 1e15, where the floats are 0.125 apart,
    // runs the ladder to the bottom; the others stop above it. The two values at the floats next
    // to x, which an answer is checked against last, are the only samples nearer.
    for x in [1e15f64, 1e10, 12345.678, -3.0e8] {
        let ulp = f64::EPSILON * x.abs();
        let (mut between, mut next_to) = (0, 0);
        let _ = derivative(
            |t: f64| {
                let d = (t - x).abs();
                if d > 0.0 && d <= 2.0 * ulp {
                    next_to += 1;
                } else if d > 0.0 && d < 15.9 * ulp {
                    between += 1;
                }
                t.sin()
            },
            x,
        );
        assert_eq!(between, 0, "{x}: steps between 2 and 16 floats");
        assert!(next_to <= 2, "{x}: {next_to} samples next to x");
    }
}

#[test]
fn any_number_of_columns_can_be_asked_for() {
    // Audit n_interp-13: the table reserved max_levels rows up front, and usize::MAX panicked
    // with a capacity overflow.
    for levels in [usize::MAX, usize::MAX / 2, 1 << 40, 64] {
        let opts = DiffOptions::new().with_max_levels(levels);
        let d = derivative_with(|x: f64| x.sin(), 1.0, &opts).unwrap();
        assert!((d - 1f64.cos()).abs() < 1e-14);
        let d = second_derivative_with(|x: f64| x.sin(), 1.0, &opts.with_tol(1e-6)).unwrap();
        assert!((d + 1f64.sin()).abs() < 1e-9);
    }
}

#[test]
fn a_stationary_point_is_found_as_one() {
    // At the bottom of a parabola the derivative is exactly zero, and an extrapolation that is
    // off by a constant would show up here.
    assert!(derivative(|x: f64| x * x, 0.0).unwrap().abs() < 1e-12);
    assert!(derivative(|x: f64| x * x - 4.0 * x, 2.0).unwrap().abs() < 1e-10);
    assert!(derivative(|x: f64| x.cos(), 0.0).unwrap().abs() < 1e-12);
    assert!((second_derivative(|x: f64| x * x, 0.0).unwrap() - 2.0).abs() < 1e-6);
    // Near a stationary point the error is relative to the slopes around it.
    let d = derivative(|x: f64| (x - 0.31).powi(2), 0.31).unwrap();
    assert!(d.abs() < 1e-12, "{d}");
}

#[test]
fn a_function_that_does_not_change_has_derivative_zero() {
    for x in [0.0, 1.0, -3.7, 1e10] {
        assert_eq!(derivative(|_: f64| 5.0, x), Some(0.0));
        assert_eq!(second_derivative(|_: f64| 5.0, x), Some(0.0));
    }
    // The second derivative of a line, and a function that is constant to the last bit.
    assert_eq!(second_derivative(|x: f64| 7.1 - 3.3 * x, 0.3), Some(0.0));
    assert_eq!(derivative(|x: f64| 1.0 + 1e-20 * x, 0.5), Some(0.0));
}

#[test]
fn a_linear_function_differentiates_to_its_slope_everywhere() {
    for x in [-1e4, -1.0, 0.0, 0.3, 1.0, 3.0, 1e4, 1e9, 1e15] {
        let d = derivative(|t: f64| 7.0 - 3.0 * t, x).unwrap();
        assert!((d + 3.0).abs() <= 1e-12 * 3.0, "the slope at {x} is {d}");
    }
}

#[test]
fn the_options_choose_the_first_step_the_tolerance_and_the_depth_of_the_table() {
    let opts = DiffOptions::new().with_step(1e-3);
    let d = derivative_with(|x: f64| x.sin(), 1.0, &opts).unwrap();
    assert!((d - 1.0_f64.cos()).abs() < 1e-12);
    // Two columns, one extrapolation: still within the tolerance asked for.
    let shallow = DiffOptions::new().with_max_levels(2).with_tol(1e-6);
    let rough = derivative_with(|x: f64| x.sin(), 1.0, &shallow).unwrap();
    assert!((rough - 1.0_f64.cos()).abs() < 1e-6);
    // A tolerance below what the arithmetic allows gives nothing.
    let tight = DiffOptions::new().with_tol(1e-20);
    assert_eq!(derivative_with(|x: f64| x.sin(), 1.0, &tight), None);
    // A step far below the round-off floor of the second difference: nothing settles.
    let fine = DiffOptions::new().with_step(f64::EPSILON.sqrt());
    assert_eq!(second_derivative_with(|x: f64| x.exp(), 0.0, &fine), None);
}

#[test]
fn invalid_arguments_give_none() {
    let sin = |x: f64| x.sin();
    for opts in [
        DiffOptions::new().with_tol(0.0),
        DiffOptions::new().with_tol(-1e-8),
        DiffOptions::new().with_tol(f64::NAN),
        DiffOptions::new().with_tol(f64::INFINITY),
        DiffOptions::new().with_step(-1e-3),
        DiffOptions::new().with_step(f64::NAN),
        DiffOptions::new().with_step(f64::INFINITY),
        DiffOptions::new().with_max_levels(0),
        DiffOptions::new().with_max_levels(1),
    ] {
        assert_eq!(derivative_with(sin, 1.0, &opts), None, "{opts:?}");
        assert_eq!(second_derivative_with(sin, 1.0, &opts), None, "{opts:?}");
    }
    assert_eq!(derivative(sin, f64::NAN), None);
    assert_eq!(derivative(sin, f64::INFINITY), None);
    // A function not finite at the point itself.
    assert_eq!(derivative(|x: f64| x.sin() / x, 0.0), None);
}

#[test]
fn a_function_computed_too_roughly_gives_none() {
    // Noise of a millionth on the values of the sine: no table settles to 1.5e-8.
    let noisy = |x: f64| {
        let bits = x.to_bits().wrapping_mul(0x9e37_79b9_7f4a_7c15);
        x.sin() * (1.0 + 1e-6 * ((bits >> 11) as f64 / (1u64 << 53) as f64 - 0.5))
    };
    assert_eq!(derivative(noisy, 1.0), None);
    // With a tolerance it can meet, the answer is within it.
    let loose = DiffOptions::new().with_tol(1e-2);
    if let Some(d) = derivative_with(noisy, 1.0, &loose) {
        assert!((d - 1f64.cos()).abs() < 1e-2, "{d}");
    }
}

#[test]
fn a_second_derivative_with_options_agrees_with_the_plain_one() {
    let opts = DiffOptions::new().with_step(1e-2).with_tol(1e-8);
    let a = second_derivative_with(|x: f64| x.exp(), 0.0, &opts).unwrap();
    let b = second_derivative(|x: f64| x.exp(), 0.0).unwrap();
    assert!((a - 1.0).abs() < 1e-8);
    assert!((a - b).abs() < 1e-8);
}

#[test]
fn a_steep_function_keeps_its_relative_accuracy() {
    // d/dx sin(1000x) at 0 is 1000: the scale has to come from the function, not from the point.
    let d = derivative(|x: f64| (1000.0 * x).sin(), 0.0).unwrap();
    assert!((d - 1000.0).abs() < 1e-9, "the slope is {d}");
    // And d/dx sin(x/1000) at 0 is 10⁻³.
    let d = derivative(|x: f64| (x / 1000.0).sin(), 0.0).unwrap();
    assert!((d - 1e-3).abs() < 1e-15, "the shallow slope is {d}");
    // Next to a pole: 1/(x − 1.001) at 1, a thousandth away.
    let d = derivative(|x: f64| 1.0 / (x - 1.001), 1.0).unwrap();
    assert!(rel(d, -1e6) < 1e-10, "{d}");
    let d = second_derivative(|x: f64| 1.0 / (x - 1.001), 1.0).unwrap();
    assert!(rel(d, -2e9) < 1e-8, "{d}");
}

/// A function, its first and second derivatives, and the points to try.
type Family = (&'static str, Function, Function, Function, &'static [f64]);

/// The battery of the audit and more: 28 functions at 133 points, from 1e-300 to 1e300, next to
/// poles, at stationary points, periodic far from the origin and steep at it.
const BATTERY: &[Family] = &[
    (
        "sin",
        |x| x.sin(),
        |x| x.cos(),
        |x| -x.sin(),
        &[
            0.0,
            1e-8,
            0.5,
            1.0,
            2.0,
            10.0,
            100.0,
            399.7335879946275,
            1000.5,
            12345.678,
            1e5 + 0.1,
            1e6,
            1234567.891,
            1e8,
            1e10,
            1e15,
        ],
    ),
    (
        "exp",
        |x| x.exp(),
        |x| x.exp(),
        |x| x.exp(),
        &[-700.0, -50.0, -10.0, -1.0, 0.0, 1.0, 10.0, 50.0, 700.0],
    ),
    (
        "ln",
        |x| x.ln(),
        |x| 1.0 / x,
        |x| -1.0 / (x * x),
        &[
            1e-300, 1e-100, 1e-10, 1e-3, 0.5, 1.0, 2.0, 100.0, 1e10, 1e100, 1e300,
        ],
    ),
    (
        "sqrt",
        |x| x.sqrt(),
        |x| 0.5 / x.sqrt(),
        |x| -0.25 / (x * x.sqrt()),
        &[
            1e-300, 1e-30, 1e-12, 1e-3, 0.5, 1.0, 2.0, 100.0, 1e10, 1e100,
        ],
    ),
    (
        "1/x",
        |x| 1.0 / x,
        |x| -1.0 / (x * x),
        |x| 2.0 / (x * x * x),
        &[-2.0, 1e-5, 0.1, 1.0, 4.0, 1e5],
    ),
    (
        "tan",
        |x| x.tan(),
        |x| 1.0 / x.cos().powi(2),
        |x| 2.0 * x.tan() / x.cos().powi(2),
        &[0.0, 1.0, 1.5, 1.57],
    ),
    (
        "atan",
        |x| x.atan(),
        |x| 1.0 / (1.0 + x * x),
        |x| -2.0 * x / (1.0 + x * x).powi(2),
        &[0.0, 1.0, 1.00001, 1e3, 1e8],
    ),
    (
        "x³",
        |x| x * x * x,
        |x| 3.0 * x * x,
        |x| 6.0 * x,
        &[0.0, 1e-5, 1.0, 2.0, 1e3, 1e100, -1e100],
    ),
    (
        "x³ − 2x",
        |x| x * x * x - 2.0 * x,
        |x| 3.0 * x * x - 2.0,
        |x| 6.0 * x,
        &[1.3, -0.7, 0.123, 9.87],
    ),
    (
        "x²",
        |x| x * x,
        |x| 2.0 * x,
        |_| 2.0,
        &[0.0, 1.3, -0.77, 3.0, 123.456],
    ),
    (
        "7.1 − 3.3x",
        |x| 7.1 - 3.3 * x,
        |_| -3.3,
        |_| 0.0,
        &[0.3, -1e4, 0.0, 1.0, 1e4, 1e9, 1e15],
    ),
    (
        "x² + 3x + 1",
        |x| x * x + 3.0 * x + 1.0,
        |x| 2.0 * x + 3.0,
        |_| 2.0,
        &[0.7, -2.3],
    ),
    (
        "x·e⁻ˣ",
        |x| x * (-x).exp(),
        |x| (1.0 - x) * (-x).exp(),
        |x| (x - 2.0) * (-x).exp(),
        &[3.0, 1.0, 2.0],
    ),
    (
        "x⁷ − 3x² + 1",
        |x| x.powi(7) - 3.0 * x * x + 1.0,
        |x| 7.0 * x.powi(6) - 6.0 * x,
        |x| 42.0 * x.powi(5) - 6.0,
        &[0.0, 0.5, 2.0, 10.0],
    ),
    (
        "e^(−x²)",
        |x| (-x * x).exp(),
        |x| -2.0 * x * (-x * x).exp(),
        |x| (4.0 * x * x - 2.0) * (-x * x).exp(),
        &[0.0, 0.5, 2.0, 5.0],
    ),
    (
        "a Gaussian of width 1e-3",
        |x| (-0.5 * (x / 1e-3) * (x / 1e-3)).exp(),
        |x| -x / 1e-6 * (-0.5 * (x / 1e-3) * (x / 1e-3)).exp(),
        |x| (x * x / 1e-12 - 1e6) * (-0.5 * (x / 1e-3) * (x / 1e-3)).exp(),
        &[1e-3, 0.0, 2.5e-3, 1.3e-3],
    ),
    (
        "Runge's",
        |x| 1.0 / (1.0 + 25.0 * x * x),
        |x| -50.0 * x / (1.0 + 25.0 * x * x).powi(2),
        |x| (3750.0 * x * x - 50.0) / (1.0 + 25.0 * x * x).powi(3),
        &[0.0, 0.1, 0.2, 1.0],
    ),
    (
        "1/(x − 1.001)",
        |x| 1.0 / (x - 1.001),
        |x| -1.0 / (x - 1.001).powi(2),
        |x| 2.0 / (x - 1.001).powi(3),
        &[1.0, 0.0, 1.002],
    ),
    (
        "1/(x − 1.0000001)",
        |x| 1.0 / (x - 1.0000001),
        |x| -1.0 / (x - 1.0000001).powi(2),
        |x| 2.0 / (x - 1.0000001).powi(3),
        &[1.0],
    ),
    (
        "sin 1000x",
        |x| (1000.0 * x).sin(),
        |x| 1000.0 * (1000.0 * x).cos(),
        |x| -1e6 * (1000.0 * x).sin(),
        &[0.0, 1.0, 3.3],
    ),
    (
        "sin 1e6x",
        |x| (1e6 * x).sin(),
        |x| 1e6 * (1e6 * x).cos(),
        |x| -1e12 * (1e6 * x).sin(),
        &[0.0, 1.0],
    ),
    (
        "sin 1e12x",
        |x| (1e12 * x).sin(),
        |x| 1e12 * (1e12 * x).cos(),
        |x| -1e24 * (1e12 * x).sin(),
        &[0.0],
    ),
    (
        "sin(812.598x + 2.739)",
        |x| (812.598 * x + 2.739).sin(),
        |x| 812.598 * (812.598 * x + 2.739).cos(),
        |x| -812.598f64.powi(2) * (812.598 * x + 2.739).sin(),
        &[0.1203, 0.12],
    ),
    (
        "sin(x/1e6)",
        |x| (x / 1e6).sin(),
        |x| (x / 1e6).cos() / 1e6,
        |x| -(x / 1e6).sin() / 1e12,
        &[0.0, 1.0, 1e6],
    ),
    (
        "1e-12 sin",
        |x| 1e-12 * x.sin(),
        |x| 1e-12 * x.cos(),
        |x| -1e-12 * x.sin(),
        &[1.0],
    ),
    (
        "cos",
        |x| x.cos(),
        |x| -x.sin(),
        |x| -x.cos(),
        &[0.0, 1e-8, 1e-3, 1.0, 3.0],
    ),
    (
        "x² − 4x",
        |x| x * x - 4.0 * x,
        |x| 2.0 * x - 4.0,
        |_| 2.0,
        &[2.0, 0.0, 1.0, 2.0000001],
    ),
    ("5", |_| 5.0, |_| 0.0, |_| 0.0, &[0.0, 1.0, 1e10]),
];

#[test]
fn a_battery_of_functions_and_points_has_no_wrong_answer() {
    // Every answer of either derivative is within ten times its tolerance of the closed form
    // (relatively; absolutely where the derivative is 0). The answers that are None are the ones
    // no table can give to the tolerance: the sine at 1e15, where the floats are 0.125 apart;
    // atan′ at 1e8, a derivative of 1e-16 against round-off of 1e-23 in a step of 4e7; and
    // second derivatives that overflow (ln and √ at 1e-300) or are buried in round-off. The
    // counts and the largest errors are printed (`--nocapture`), for the doc page.
    let (tol1, tol2) = (f64::EPSILON.sqrt(), f64::EPSILON.cbrt());
    let (mut cases, mut none1, mut none2, mut evals1, mut evals2) = (0, 0, 0, 0usize, 0usize);
    let (mut worst1, mut worst2) = (0.0f64, 0.0f64);
    let mut errors: Vec<(f64, &str, f64, u8)> = Vec::new();
    for &(name, f, d1, d2, points) in BATTERY {
        for &x in points {
            cases += 1;
            let (mut c1, mut c2) = (0usize, 0usize);
            let r1 = derivative(
                |t| {
                    c1 += 1;
                    f(t)
                },
                x,
            );
            let r2 = second_derivative(
                |t| {
                    c2 += 1;
                    f(t)
                },
                x,
            );
            evals1 += c1;
            evals2 += c2;
            match r1 {
                None => none1 += 1,
                Some(v) => {
                    let e = rel(v, d1(x));
                    worst1 = worst1.max(e);
                    errors.push((e, name, x, 1));
                    assert!(e <= 10.0 * tol1, "{name}′ at {x}: {v} against {}", d1(x));
                }
            }
            match r2 {
                None => none2 += 1,
                Some(v) => {
                    let e = rel(v, d2(x));
                    worst2 = worst2.max(e);
                    errors.push((e, name, x, 2));
                    assert!(e <= 10.0 * tol2, "{name}″ at {x}: {v} against {}", d2(x));
                }
            }
        }
    }
    println!(
        "{cases} cases; first derivative: {none1} None, worst {worst1:.1e}, {:.1} evaluations; second: {none2} None, worst {worst2:.1e}, {:.1} evaluations",
        evals1 as f64 / cases as f64,
        evals2 as f64 / cases as f64
    );
    errors.sort_by(|a, b| b.0.total_cmp(&a.0));
    for (e, name, x, order) in errors.iter().take(6) {
        println!("    derivative {order} of {name} at {x}: {e:.1e}");
    }
    let median = |order: u8| {
        let mut v: Vec<f64> = errors
            .iter()
            .filter(|e| e.3 == order)
            .map(|e| e.0)
            .collect();
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    println!(
        "    median relative errors {:.1e} and {:.1e}",
        median(1),
        median(2)
    );
    assert!(none1 <= 4 && none2 <= 8, "{none1} and {none2} None");
}

#[test]
fn quotients_and_round_off_that_underflow_give_no_answer() {
    // At steps so large beside the values that the second difference over h² and its
    // round-off both fall below the smallest subnormal number, the relative tests read 0 ≤ 0:
    // these were Some(0.0). The sine's floats at 1e200 are 1e184 apart, so nothing can be
    // known of its curvature there; the others' derivatives are not 0 either.
    assert_eq!(second_derivative(|t: f64| t.sin(), 1e200), None); // −sin 1e200 = 0.644
    assert_eq!(second_derivative(|t: f64| t.cos(), 1e300), None); // −cos 1e300 = 0.818
    assert_eq!(derivative(|t: f64| 1e-200 * t.sin(), 1e200), None); // 7.65e−201
    let o = DiffOptions::new().with_step(1e200).with_tol(6e-6);
    assert_eq!(second_derivative_with(|t: f64| t.sin(), 1.0, &o), None); // −0.841
    // A step whose square overflows while the answer is a normal number: the round-off is
    // formed as a/h/h, where a/h² was 0, and the answer is kept (it was None).
    for x in [1e200, 1e300] {
        let d = second_derivative(|t: f64| 1e-300 * t * t, x).expect("a normal curvature");
        assert!(rel(d, 2e-300) < 1e-12, "at {x:e}: {d:e}");
    }
    let o = DiffOptions::new().with_step(1e200).with_tol(1e-6);
    let d = second_derivative_with(|t: f64| 1e100 * (t / 1e200).exp(), 1e200, &o).unwrap();
    assert!(rel(d, 1e100 * 1f64.exp() / 1e200 / 1e200) < 1e-10, "{d:e}");
    let d = second_derivative(|t: f32| 1e-30 * t * t, 1e30f32).expect("a normal f32 curvature");
    assert!((d / 2e-30 - 1.0).abs() < 1e-4, "{d:e}");
}
