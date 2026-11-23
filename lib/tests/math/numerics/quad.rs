//! The quadrature of `a::math::numerics`: the adaptive Kronrod rules (QAGS, QAGI), Clenshaw and
//! Curtis, the double exponential rules, the cubature and the Monte Carlo rules.
//!
//! Every rule is held to values known independently of it — closed forms and mpmath, through the
//! data of `quad/data` — and its error estimate is held to the truth: a converged result whose
//! true error exceeds its estimate is a failure, however accurate it is. The contract the rules
//! share is tested here; each rule has a file of its own.

mod battery;
mod clenshaw;
mod cubature;
mod kronrod;
mod monte_carlo;
mod tanh_sinh;

use a::math::numerics::{
    QuadOptions, QuadResult, clenshaw_curtis, cubature, monte_carlo, quad, quad_g7k15, quad_g10k21,
    quad_half_infinite, quad_infinite, quasi_monte_carlo, tanh_sinh,
};

/// An integrand of the tables of cases.
type F = fn(f64) -> f64;

/// Pairs of narrow Gaussian peaks of width `w` on `[0, 1]`, as the confirmation of review round 2
/// drew them: 400 draws of a fixed linear congruential sequence, the first centre in
/// `[0.05, 0.45]`, the second there too or, if `split`, in `[0.55, 0.95]`, and the second
/// peak's height in `[0.05, 1]` (the first's is 1); pairs closer than 20 widths are left out.
/// Each pair is `(c₁, c₂, h₂)`, its integral `w√π(1 + h₂)`, the peaks lying 50 widths and more
/// inside the interval.
fn peak_pairs(w: f64, split: bool) -> Vec<(f64, f64, f64)> {
    let mut s = 987_654_321u64;
    let mut next = move || {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut pairs = Vec::new();
    for _ in 0..400 {
        let c1 = 0.05 + 0.4 * next();
        let c2 = if split { 0.55 } else { 0.05 } + 0.4 * next();
        let h2 = 0.05 + 0.95 * next();
        if (c1 - c2).abs() >= 20.0 * w {
            pairs.push((c1, c2, h2));
        }
    }
    pairs
}

/// A rule of one variable on `[0, 1]`, for [`pair_sweep`].
type OnUnit<'a> = &'a dyn Fn(&dyn Fn(f64) -> f64, &QuadOptions<f64>) -> QuadResult<f64>;

/// `rule` on the pairs of peaks of [`peak_pairs`] at the tolerance `tol`: how many runs saw both
/// peaks at more than `10⁻³` of their heights at some point, and those of them that converged
/// beyond their error estimates, described.
fn pair_sweep(rule: OnUnit<'_>, w: f64, split: bool, tol: f64) -> (usize, Vec<String>) {
    let sqrt_pi = core::f64::consts::PI.sqrt();
    let opts = QuadOptions::new().with_epsabs(tol).with_epsrel(tol);
    let (mut seen, mut beyond) = (0, Vec::new());
    for (c1, c2, h2) in peak_pairs(w, split) {
        let (s1, s2) = (std::cell::Cell::new(0.0f64), std::cell::Cell::new(0.0f64));
        let f = |x: f64| {
            let (t, u) = ((x - c1) / w, (x - c2) / w);
            let (v1, v2) = ((-t * t).exp(), h2 * (-u * u).exp());
            s1.set(s1.get().max(v1));
            s2.set(s2.get().max(v2 / h2));
            v1 + v2
        };
        let r = rule(&f, &opts);
        if s1.get() > 1e-3 && s2.get() > 1e-3 {
            seen += 1;
            let exact = w * sqrt_pi * (1.0 + h2);
            if r.converged && (r.value - exact).abs() > r.error + 4.0 * f64::EPSILON * exact {
                beyond.push(format!(
                    "w {w:e} split {split} tol {tol:e}: c1 {c1} c2 {c2} h2 {h2}: {r:?} for {exact:e}, \
                     the peaks seen at {:.3e} and {:.3e} of their heights",
                    s1.get(),
                    s2.get()
                ));
            }
        }
    }
    (seen, beyond)
}

/// The four rules of one variable, on the same integrand and interval.
fn rules(
    f: fn(f64) -> f64,
    a: f64,
    b: f64,
    opts: &QuadOptions<f64>,
) -> Vec<(&'static str, Option<a::math::numerics::QuadResult<f64>>)> {
    vec![
        ("quad_g7k15", quad_g7k15(f, a, b, opts)),
        ("quad_g10k21", quad_g10k21(f, a, b, opts)),
        ("clenshaw_curtis", clenshaw_curtis(f, a, b, opts)),
        ("tanh_sinh", tanh_sinh(f, a, b, opts)),
    ]
}

#[test]
fn an_empty_interval_is_the_integral_zero_and_a_reversed_one_its_negation() {
    // Audit n_quad-16: both were None. ∫₁¹ = 0 with no evaluation, ∫₁⁰ eˣ dx = 1 − e.
    let opts = QuadOptions::new();
    let e = core::f64::consts::E;
    for (name, r) in rules(|x| x.exp(), 1.0, 1.0, &opts) {
        let r = r.unwrap_or_else(|| panic!("{name} refused an empty interval"));
        assert_eq!((r.value, r.error, r.evaluations), (0.0, 0.0, 0), "{name}");
        assert!(r.converged, "{name}");
    }
    for (name, r) in rules(|x| x.exp(), 1.0, 0.0, &opts) {
        let r = r.unwrap();
        assert!((r.value - (1.0 - e)).abs() < 1e-14, "{name}: {}", r.value);
        assert!(r.converged, "{name}");
    }
    // Infinite limits the other way round: ∫_∞^0 e⁻ˣ dx = −1.
    let r = quad(|x: f64| (-x).exp(), f64::INFINITY, 0.0).unwrap();
    assert!((r.value + 1.0).abs() < 1e-14 && r.converged);
    let r = tanh_sinh(|x: f64| (-x).exp(), f64::INFINITY, 0.0, &opts).unwrap();
    assert!((r.value + 1.0).abs() < 1e-14 && r.converged);
}

#[test]
fn a_limit_that_is_not_a_number_answers_nothing() {
    let opts = QuadOptions::new();
    for (a, b) in [(f64::NAN, 1.0), (0.0, f64::NAN), (f64::NAN, f64::NAN)] {
        for (name, r) in rules(|x| x, a, b, &opts) {
            assert!(r.is_none(), "{name} answered {r:?}");
        }
        assert!(quad(|x: f64| x, a, b).is_none());
    }
    // An infinite limit is for the rules that map one: Clenshaw–Curtis refuses it.
    assert!(clenshaw_curtis(|x: f64| (-x).exp(), 0.0, f64::INFINITY, &opts).is_none());
}

#[test]
fn options_that_ask_for_nothing_answer_nothing() {
    // Audit n_quad-16: at tolerances 0 the old rule reported an error of 0, converged.
    for opts in [
        QuadOptions::new().with_epsabs(0.0).with_epsrel(0.0),
        QuadOptions::new().with_epsabs(-1e-10),
        QuadOptions::new().with_epsrel(f64::NAN),
        QuadOptions::new().with_max_subintervals(0),
        QuadOptions::new().with_max_evaluations(0),
    ] {
        for (name, r) in rules(|x| (30.0 * x).sin(), 0.0, 1.0, &opts) {
            assert!(r.is_none(), "{name} with {opts:?} answered {r:?}");
        }
    }
    // One tolerance zero is a pure absolute or relative test, and fine.
    let r = quad_g10k21(
        |x: f64| x.exp(),
        0.0,
        1.0,
        &QuadOptions::new().with_epsabs(0.0),
    )
    .unwrap();
    assert!(r.converged);
}

#[test]
fn the_default_tolerances_suit_the_type() {
    // Audit n_quad-11: QuadOptions::<f32>::new() asked for 1e-12, which no f32 sum can meet, and
    // quad::<f32>(cos 200x) spent 8379 evaluations to fail.
    assert_eq!(QuadOptions::<f64>::new().epsabs, 1e-12);
    assert_eq!(QuadOptions::<f32>::new().epsabs, 1e-5);
    let r = quad(|x: f32| (200.0 * x).cos(), 0.0, 1.0).unwrap();
    let want = 200f64.sin() / 200.0;
    assert!(r.converged, "cos 200x in f32 did not converge");
    assert!((f64::from(r.value) - want).abs() <= f64::from(r.error));
    assert!(r.evaluations < 1000, "{} evaluations", r.evaluations);
    for (name, r) in [
        (
            "tanh_sinh",
            tanh_sinh(|x: f32| x.exp(), 0.0, 1.0, &QuadOptions::new()),
        ),
        (
            "clenshaw_curtis",
            clenshaw_curtis(|x: f32| x.exp(), 0.0, 1.0, &QuadOptions::new()),
        ),
    ] {
        let r = r.unwrap();
        assert!(r.converged && r.evaluations < 200, "{name}: {r:?}");
        let e = core::f64::consts::E - 1.0;
        assert!(
            (f64::from(r.value) - e).abs() <= f64::from(r.error),
            "{name}"
        );
    }
}

#[test]
fn the_budget_bounds_the_work_of_every_rule() {
    // Audit n_quad-16: tanh_sinh ignored max_subintervals. Every rule now stops at
    // max_evaluations, and the adaptive ones at max_subintervals.
    let hard = |x: f64| (x - 0.3).abs().sqrt() * (1.0 / (1e-3 + x * x)).sin();
    for limit in [40usize, 333, 1000] {
        let opts = QuadOptions::new().with_max_evaluations(limit);
        for (name, r) in rules(hard, 0.0, 1.0, &opts) {
            if let Some(r) = r {
                assert!(
                    r.evaluations <= limit,
                    "{name}: {} > {limit}",
                    r.evaluations
                );
            }
        }
    }
    for m in [1usize, 2, 7] {
        let opts = QuadOptions::new().with_max_subintervals(m);
        let r = quad_g7k15(hard, 0.0, 1.0, &opts).unwrap();
        assert!(r.evaluations <= 15 * (2 * m - 1), "{m}: {}", r.evaluations);
        assert!(!r.converged);
    }
    // Every budget, not only round ones (review round 1): the round-1 tanh_sinh spent 74
    // evaluations of a budget of 73 on ∫₀¹ 1 dx, the first side's nodes past its reach eating
    // what the level had set aside for the other side's.
    let limits = if cfg!(miri) { 30..=40 } else { 1..=400 };
    for limit in limits {
        let opts = QuadOptions::new().with_max_evaluations(limit);
        for f in [hard, |_: f64| 1.0, |x: f64| (-x).exp() / x.sqrt()] {
            for (name, r) in rules(f, 0.0, 1.0, &opts) {
                if let Some(r) = r {
                    assert!(
                        r.evaluations <= limit,
                        "{name}: {} > {limit}",
                        r.evaluations
                    );
                }
            }
            let r = tanh_sinh(f, 0.0, f64::INFINITY, &opts).unwrap();
            assert!(
                r.evaluations <= limit,
                "tanh_sinh on [0, ∞): {r:?} > {limit}"
            );
        }
    }
}

#[test]
fn a_rule_with_no_room_to_subdivide_says_so() {
    // One subinterval cannot integrate a boundary layer; the result must be refused, not passed.
    let opts = QuadOptions::new()
        .with_epsabs(1e-14)
        .with_epsrel(1e-14)
        .with_max_subintervals(1);
    let f = |x: f64| (-100.0 * (x - 0.5) * (x - 0.5)).exp();
    let r = quad_g10k21(f, 0.0, 1.0, &opts).unwrap();
    assert!(!r.converged, "one panel claimed convergence at {}", r.value);
    assert_eq!(r.evaluations, 21);
    // The same integral with room to work converges: √π/10 · erf(5).
    let r = quad(f, 0.0, 1.0).unwrap();
    let want = 0.177_245_385_090_279_1;
    assert!(r.converged && (r.value - want).abs() <= r.error);
}

#[test]
fn an_integrand_that_is_not_a_number_ends_the_integration_unconverged() {
    let opts = QuadOptions::new();
    let cases: [(&str, F); 3] = [
        ("NaN on half", |x| if x > 0.5 { f64::NAN } else { 1.0 }),
        ("infinite everywhere", |_| f64::INFINITY),
        ("NaN everywhere", |_| f64::NAN),
    ];
    for (what, f) in cases {
        for (name, r) in rules(f, 0.0, 1.0, &opts) {
            let r = r.unwrap();
            assert!(!r.converged, "{name} on {what} converged to {}", r.value);
            assert!(
                !r.value.is_finite() || !r.error.is_finite(),
                "{name} on {what}"
            );
        }
    }
}

/// Whether a result whose value or error is not a finite number says so: not converged, and an
/// infinite error beside a value that is not finite.
fn refused<T: a::math::numerics::Real>(r: &QuadResult<T>) -> bool {
    let finite = r.value.is_finite() && r.error.is_finite();
    !finite && !r.converged && (r.value.is_finite() || r.error.is_infinite())
}

#[test]
fn a_result_beyond_the_range_of_its_type_is_never_converged() {
    // Review round 1, B2. Clenshaw–Curtis in f64 accepted ∞ ± ∞ (∞ ≤ epsrel·∞), and every rule
    // in f32 works in f64 and rounded a value above f32::MAX to ∞ after its test: ∫₀¹⁵⁰⁰ x¹²
    // = 1.5·10⁴⁰ came back ∞, error 8.9·10³⁴, converged, though every value of the integrand
    // fits an f32.
    let opts = QuadOptions::new();
    for (what, r) in [
        (
            "1e308 on [0, 10]",
            clenshaw_curtis(|_: f64| 1e308, 0.0, 10.0, &opts),
        ),
        (
            "2 on [0, 1.7e308]",
            clenshaw_curtis(|_: f64| 2.0, 0.0, 1.7e308, &opts),
        ),
    ] {
        let r = r.unwrap();
        assert!(refused(&r), "clenshaw_curtis on {what}: {r:?}");
    }
    let opts = QuadOptions::<f32>::new();
    let power = |x: f32| x.powi(12);
    for (name, r) in [
        ("quad", quad(power, 0.0, 1500.0)),
        ("quad_g7k15", quad_g7k15(power, 0.0, 1500.0, &opts)),
        (
            "clenshaw_curtis",
            clenshaw_curtis(power, 0.0, 1500.0, &opts),
        ),
        ("tanh_sinh", tanh_sinh(power, 0.0, 1500.0, &opts)),
        ("quad, x on [0, 1e20]", quad(|x: f32| x, 0.0, 1e20)),
        (
            "quad_half_infinite",
            quad_half_infinite(|x: f32| 1e38 * (-x / 100.0).exp(), 0.0, &opts),
        ),
        (
            "cubature",
            cubature(|_: &[f32]| 1e38, &[0.0; 2], &[10.0; 2], &opts),
        ),
        (
            "monte_carlo",
            monte_carlo(|_: &[f32]| 1e38, &[0.0], &[10.0], 1000),
        ),
        (
            "quasi_monte_carlo",
            quasi_monte_carlo(|_: &[f32]| 1e38, &[0.0], &[10.0], 1024),
        ),
    ] {
        let r = r.unwrap();
        assert!(refused(&r), "{name} in f32: {r:?}");
    }
    // An integral just inside the range still converges: ∫₀¹⁰⁰⁰ x¹² dx = 7.7·10³⁷.
    let want = 1e39 / 13.0;
    for (name, r) in [
        ("quad", quad(power, 0.0, 1000.0)),
        ("quad_g7k15", quad_g7k15(power, 0.0, 1000.0, &opts)),
        (
            "clenshaw_curtis",
            clenshaw_curtis(power, 0.0, 1000.0, &opts),
        ),
        ("tanh_sinh", tanh_sinh(power, 0.0, 1000.0, &opts)),
    ] {
        let r = r.unwrap();
        let err = (f64::from(r.value) - want).abs();
        assert!(r.converged, "{name}: {r:?}");
        assert!(err <= f64::from(r.error) + 1e-6 * want, "{name}: {r:?}");
    }
}

#[test]
fn a_feature_the_nodes_only_grazed_is_not_passed_off_as_converged() {
    // Review round 1 (the limits of tanh_sinh): a result whose error estimate exceeds the result
    // itself is refused by tanh_sinh and clenshaw_curtis as by QAGS, unless the error is at the
    // round-off of the sums (since review round 2, one whose error is a tenth of it or more). The round-1 code accepted ∫₀^∞ e^(−(x−40)²) dx = √π from tanh_sinh
    // as 3.1e-58 ± 6.1e-58 after 38 evaluations (centre 35: 1.7e-18, 45: 1.1e-119, and the
    // bump of width 3 about 52 as 2.7e-26), and a peak of width 0.0035 about 2/3 from
    // clenshaw_curtis as 1.8e-18 ± 1.2e-12 after 35, at every tolerance; now each is found.
    let sqrt_pi = core::f64::consts::PI.sqrt();
    let honest = |r: &QuadResult<f64>, exact: f64| {
        r.converged && (r.value - exact).abs() <= r.error + 4.0 * f64::EPSILON * exact
    };
    for c in [35.0, 40.0, 45.0] {
        let bump = move |x: f64| (-(x - c) * (x - c)).exp();
        for tol in [1e-12, 1e-8, 1e-4] {
            let opts = QuadOptions::new().with_epsabs(tol).with_epsrel(tol);
            let r = tanh_sinh(bump, 0.0, f64::INFINITY, &opts).unwrap();
            assert!(
                honest(&r, sqrt_pi),
                "tanh_sinh, centre {c} at {tol:e}: {r:?}"
            );
        }
    }
    let wide = |x: f64| (-((x - 52.0) / 3.0) * ((x - 52.0) / 3.0)).exp();
    let r = tanh_sinh(wide, 0.0, f64::INFINITY, &QuadOptions::new()).unwrap();
    assert!(honest(&r, 3.0 * sqrt_pi), "tanh_sinh, width 3: {r:?}");
    for (c, w, tols) in [
        (2.0 / 3.0, 0.0035, &[1e-9, 1e-6, 1e-3][..]),
        (0.4, 0.003, &[1e-3][..]),
        (0.31, 0.0031, &[1e-6, 1e-3][..]),
    ] {
        let peak = move |x: f64| (-((x - c) / w) * ((x - c) / w)).exp();
        for &tol in tols {
            let opts = QuadOptions::new().with_epsabs(tol).with_epsrel(tol);
            let r = clenshaw_curtis(peak, 0.0, 1.0, &opts).unwrap();
            assert!(
                honest(&r, w * sqrt_pi),
                "clenshaw_curtis, peak at {c} at {tol:e}: {r:?}"
            );
        }
    }
}

#[test]
fn integrals_add_and_scale_as_they_should() {
    let f = |x: f64| x * x * (1.0 - x);
    let g = |x: f64| x.exp();
    let i = quad(f, 0.0, 1.0).unwrap().value;
    let j = quad(g, 0.0, 1.0).unwrap().value;
    let both = quad(|x: f64| f(x) + g(x), 0.0, 1.0).unwrap().value;
    assert!((both - (i + j)).abs() < 1e-15);
    let scaled = quad(|x: f64| 3.0 * f(x), 0.0, 1.0).unwrap().value;
    assert!((scaled - 3.0 * i).abs() < 1e-15);
    // Additivity over the halves, for all four rules.
    let opts = QuadOptions::new();
    for (name, r) in rules(f, 0.0, 1.0, &opts) {
        let one = r.unwrap().value;
        let left = quad(f, 0.0, 0.5).unwrap().value;
        let right = quad(f, 0.5, 1.0).unwrap().value;
        assert!(
            (one - (left + right)).abs() < 1e-15,
            "{name} is not additive"
        );
    }
}

#[test]
fn the_shorthands_for_infinite_intervals_are_quad_with_infinite_limits() {
    let opts = QuadOptions::new();
    let f = |x: f64| (-x * x).exp();
    assert_eq!(
        quad_half_infinite(f, 0.0, &opts),
        quad_g10k21(f, 0.0, f64::INFINITY, &opts)
    );
    assert_eq!(
        quad_infinite(f, &opts),
        quad_g10k21(f, f64::NEG_INFINITY, f64::INFINITY, &opts)
    );
    // Audit n_quad-16: quad_half_infinite(eˣ, −∞) was 0, converged; it is the whole line now,
    // and eˣ over the line diverges.
    let r = quad_half_infinite(|x: f64| x.exp(), f64::NEG_INFINITY, &opts).unwrap();
    assert!(!r.converged, "{r:?}");
}

#[test]
fn cubature_contract_matches_the_one_dimensional_rules() {
    let opts = QuadOptions::new();
    // A degenerate box is the integral 0, a reversed side negates it.
    let r = cubature(|_: &[f64]| 1.0, &[0.0, 1.0], &[1.0, 1.0], &opts).unwrap();
    assert_eq!((r.value, r.evaluations), (0.0, 0));
    let r = cubature(|v: &[f64]| v[0] * v[1], &[1.0, 0.0], &[0.0, 1.0], &opts).unwrap();
    assert!((r.value + 0.25).abs() < 1e-15 && r.converged);
    assert!(cubature(|_: &[f64]| 1.0, &[f64::NAN, 0.0], &[1.0, 1.0], &opts).is_none());
    assert!(cubature(|_: &[f64]| 1.0, &[0.0, 0.0], &[1.0, f64::INFINITY], &opts).is_none());
}

#[test]
#[should_panic(expected = "different dimensions")]
fn a_cubature_box_with_mismatched_corners_panics() {
    let _ = cubature(|_: &[f64]| 1.0, &[0.0], &[1.0, 1.0], &QuadOptions::new());
}
