//! The double exponential rules: tanh-sinh on a finite interval, exp-sinh and sinh-sinh on the
//! infinite ones. Singular ends, nodes that never touch an end, integrands small where the
//! coarse levels look, the reuse of the nodes from level to level, and the regressions of the
//! audit (n_quad-2, -7, -8 and -9).

use a::math::numerics::{QuadOptions, QuadResult, tanh_sinh};
use std::cell::RefCell;

/// An integrand of the tables of cases.
type F = fn(f64) -> f64;

thread_local! {
    /// The points an integrand was evaluated at.
    static SEEN: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
}

/// Whether a result keeps its promise.
fn honest(r: &QuadResult<f64>, exact: f64) -> bool {
    !r.converged || (r.value - exact).abs() <= r.error + 4.0 * f64::EPSILON * exact.abs()
}

fn at(tol: f64) -> QuadOptions<f64> {
    QuadOptions::new().with_epsabs(tol).with_epsrel(tol)
}

#[test]
fn singular_ends_are_integrated_to_the_precision_of_the_arithmetic() {
    // Audit n_quad-7: the old nodes came no closer to an end than its ulp, and ∫₀¹ x^(−½) was
    // capped 1.5e-8 off while it claimed convergence at 1e-10. Formed from their distances to
    // the ends, the nodes reach 0 down to the smallest normal number.
    let cases: [(&str, F, f64); 5] = [
        ("x^(−½)", |x| 1.0 / x.sqrt(), 2.0),
        ("ln x", |x| x.ln(), -1.0),
        ("x^(−0.9)", |x| x.powf(-0.9), 10.0),
        (
            "ln x ln(1 − x)",
            |x| x.ln() * (1.0 - x).ln(),
            2.0 - core::f64::consts::PI.powi(2) / 6.0,
        ),
        (
            "cos x / √x",
            |x| x.cos() / x.sqrt(),
            1.809_048_475_800_544_2,
        ),
    ];
    for (what, f, exact) in cases {
        for tol in [1e-12, 1e-10, 1e-6] {
            let r = tanh_sinh(f, 0.0, 1.0, &at(tol)).unwrap();
            assert!(r.converged && honest(&r, exact), "{what} at {tol:e}: {r:?}");
        }
        let r = tanh_sinh(f, 0.0, 1.0, &QuadOptions::new()).unwrap();
        assert!(
            (r.value - exact).abs() <= 1e-13 * exact.abs(),
            "{what}: {r:?}"
        );
    }
    // A singularity at a nonzero end can be approached only to within an ulp of it: what lies
    // beyond is in the error estimate, and the integrand written in the distance to that end is
    // integrated in full.
    let r = tanh_sinh(|x: f64| (1.0 - x).powf(-0.9), 0.0, 1.0, &QuadOptions::new()).unwrap();
    assert!(
        !r.converged && honest(&r, 10.0) && (r.value - 10.0).abs() <= r.error,
        "{r:?}"
    );
    let r = tanh_sinh(|s: f64| s.powf(-0.9), 0.0, 1.0, &QuadOptions::new()).unwrap();
    assert!(r.converged && (r.value - 10.0).abs() < 1e-12, "{r:?}");
    // Both ends at once: the arcsine density, singular at 0 and at 1. At a loose tolerance the
    // ulp at 1 is no obstacle; at the default one the estimate says what it costs.
    let arcsine = |x: f64| 1.0 / (x * (1.0 - x)).sqrt();
    let pi = core::f64::consts::PI;
    let r = tanh_sinh(arcsine, 0.0, 1.0, &at(1e-6)).unwrap();
    assert!(r.converged && honest(&r, pi), "{r:?}");
    let r = tanh_sinh(arcsine, 0.0, 1.0, &QuadOptions::new()).unwrap();
    assert!(honest(&r, pi) && (r.value - pi).abs() <= r.error, "{r:?}");
    // On [−1, 1], with the singularities at both ends 1/√(1 − x²).
    let r = tanh_sinh(|x: f64| 1.0 / (1.0 - x * x).sqrt(), -1.0, 1.0, &at(1e-6)).unwrap();
    assert!(r.converged && honest(&r, pi), "{r:?}");
    // In f32, where the old rule claimed convergence 3.4e-4 off.
    let r = tanh_sinh(|x: f32| 1.0 / x.sqrt(), 0.0, 1.0, &QuadOptions::new()).unwrap();
    assert!(
        r.converged && (f64::from(r.value) - 2.0).abs() <= f64::from(r.error),
        "{r:?}"
    );
    // x^(−0.99) has 0.083 of its integral 100 below the smallest normal number, where no
    // point of an f64 can sample it: an honest rule says so.
    let r = tanh_sinh(|x: f64| x.powf(-0.99), 0.0, 1.0, &QuadOptions::new()).unwrap();
    assert!(!r.converged && (r.value - 100.0).abs() <= r.error, "{r:?}");
    // A divergent one is not converged at any tolerance: the old rule gave 37.0 for ∫₀¹ dx/x
    // at 1e-3.
    for tol in [1e-12, 1e-6, 1e-3] {
        let r = tanh_sinh(|x: f64| 1.0 / x, 0.0, 1.0, &at(tol)).unwrap();
        assert!(!r.converged, "∫₀¹ dx/x at {tol:e}: {r:?}");
    }
}

#[test]
fn no_node_falls_on_an_end_and_an_integrand_undefined_there_is_welcome() {
    // An integrand that is NaN exactly at the ends, and the points recorded: none is an end, in
    // f64 on intervals at 0 and away from it, and in f32. Far from 0 against its width, an
    // interval's nodes come no closer to its ends than an ulp of them — 10⁻¹⁰ of the width of
    // [10⁶, 10⁶ + 1] — and the error estimate says what that costs.
    for (a, b) in [
        (0.0, 1.0),
        (1.0, 2.0),
        (-1.0, 1.0),
        (-3.0, -2.5),
        (1e6, 1e6 + 1.0),
    ] {
        SEEN.with(|s| s.borrow_mut().clear());
        let f = move |x: f64| {
            SEEN.with(|s| s.borrow_mut().push(x));
            if x == a || x == b { f64::NAN } else { 1.0 }
        };
        let r = tanh_sinh(f, a, b, &QuadOptions::new()).unwrap();
        assert!(honest(&r, b - a), "[{a}, {b}]: {r:?}");
        if a < 1e6 {
            assert!(r.converged, "[{a}, {b}]: {r:?}");
        } else {
            assert!(!r.converged && r.error < 1e-8, "[{a}, {b}]: {r:?}");
            let r = tanh_sinh(f, a, b, &at(1e-8)).unwrap();
            assert!(
                r.converged && honest(&r, b - a),
                "[{a}, {b}] at 1e-8: {r:?}"
            );
        }
        let seen = SEEN.with(|s| s.borrow().clone());
        assert!(
            seen.iter().all(|&x| x > a && x < b),
            "[{a}, {b}]: an end was evaluated"
        );
    }
    let r = tanh_sinh(
        |x: f32| {
            if x <= 0.0 || x >= 1.0 {
                f32::NAN
            } else {
                x.ln()
            }
        },
        0.0,
        1.0,
        &QuadOptions::new(),
    )
    .unwrap();
    assert!(
        r.converged && (f64::from(r.value) + 1.0).abs() <= f64::from(r.error),
        "{r:?}"
    );
}

#[test]
fn an_integrand_small_where_the_coarse_levels_look_is_found() {
    // Audit n_quad-2: the old sums stopped at the first pair of small terms from the centre, and
    // x³⁰ on [−1, 1] came out 5e-41, converged; so did e^(−x²) on [0, 20] (4e-36) and the step
    // at 0.7 (0). Audit n_quad-9: the old rule accepted its first level of three points.
    let sqrt_pi = core::f64::consts::PI.sqrt();
    let cases: [(&str, F, f64, f64, f64); 9] = [
        ("x³⁰", |x| x.powi(30), -1.0, 1.0, 2.0 / 31.0),
        ("x⁶⁰", |x| x.powi(60), -1.0, 1.0, 2.0 / 61.0),
        ("x²⁰⁰", |x| x.powi(200), -1.0, 1.0, 2.0 / 201.0),
        (
            "e^(−x²) on [0, 20]",
            |x| (-x * x).exp(),
            0.0,
            20.0,
            sqrt_pi / 2.0,
        ),
        (
            "x e^(−x²) on [0, 100]",
            |x| x * (-x * x).exp(),
            0.0,
            100.0,
            0.5,
        ),
        (
            "e^(−x)/√x on [0, 200]",
            |x| (-x).exp() / x.sqrt(),
            0.0,
            200.0,
            sqrt_pi,
        ),
        (
            "e^(−(x−13)²) on [0, 40]",
            |x| (-(x - 13.0) * (x - 13.0)).exp(),
            0.0,
            40.0,
            sqrt_pi,
        ),
        (
            "the step at 0.7",
            |x| if x > 0.7 { 1.0 } else { 0.0 },
            0.0,
            1.0,
            0.3,
        ),
        // mpmath.quad at 30 digits.
        (
            "x⁴⁰ eˣ on [−1, 1]",
            |x| x.powi(40) * x.exp(),
            -1.0,
            1.0,
            0.073_948_278_460_741_26,
        ),
    ];
    for (what, f, a, b, exact) in cases {
        for tol in [1e-12, 1e-8, 1e-4] {
            let r = tanh_sinh(f, a, b, &at(tol)).unwrap();
            assert!(honest(&r, exact), "{what} at {tol:e}: {r:?}, exact {exact}");
            assert!(r.evaluations > 20, "{what}: {} evaluations", r.evaluations);
            // All but the step, which the rule is not for, converge.
            if !what.starts_with("the step") {
                assert!(r.converged, "{what} at {tol:e}: {r:?}");
            }
        }
    }
    // f32 at its default tolerance, where the old rule failed already on x¹⁴.
    let r = tanh_sinh(|x: f32| x.powi(14), -1.0, 1.0, &QuadOptions::new()).unwrap();
    let err = (f64::from(r.value) - 2.0 / 15.0).abs();
    assert!(r.converged && err <= f64::from(r.error) + 1e-7, "{r:?}");
}

#[test]
fn an_integrand_that_is_not_a_number_inside_ends_the_integration() {
    // Audit n_quad-8: the old rule threw every non-finite value away and integrated f = +∞ to
    // 0, converged.
    let cases: [(&str, F); 4] = [
        ("+∞", |_| f64::INFINITY),
        ("NaN", |_| f64::NAN),
        ("NaN on [0, 0.05)", |x| {
            if x < 0.05 { f64::NAN } else { x.exp() }
        }),
        ("1/(x − 0.5)", |x| 1.0 / (x - 0.5)),
    ];
    for (what, f) in cases {
        for tol in [1e-12, 1e-3] {
            let r = tanh_sinh(f, 0.0, 1.0, &at(tol)).unwrap();
            assert!(!r.converged, "{what} at {tol:e}: {r:?}");
        }
    }
}

#[test]
fn each_level_reuses_the_nodes_of_the_levels_before() {
    // The step halves and each level evaluates only the new nodes: no point is evaluated
    // twice, and an analytic integrand takes a few dozen points.
    for (a, b) in [
        (0.0, 1.0),
        (0.0, f64::INFINITY),
        (f64::NEG_INFINITY, f64::INFINITY),
    ] {
        SEEN.with(|s| s.borrow_mut().clear());
        let f = |x: f64| {
            SEEN.with(|s| s.borrow_mut().push(x));
            (-x * x).exp() * (1.0 + x * x).recip()
        };
        let r = tanh_sinh(f, a, b, &QuadOptions::new()).unwrap();
        assert!(r.converged, "[{a}, {b}]: {r:?}");
        let mut seen = SEEN.with(|s| s.borrow().clone());
        assert_eq!(seen.len(), r.evaluations);
        seen.sort_by(f64::total_cmp);
        seen.dedup();
        assert_eq!(
            seen.len(),
            r.evaluations,
            "[{a}, {b}]: a point was evaluated twice"
        );
        assert!(
            r.evaluations < 400,
            "[{a}, {b}]: {} evaluations",
            r.evaluations
        );
    }
}

#[test]
fn exp_sinh_and_sinh_sinh_integrate_over_infinite_intervals() {
    let pi = core::f64::consts::PI;
    let inf = f64::INFINITY;
    let cases: [(&str, F, f64, f64, f64); 8] = [
        (
            "e^(−x)/√x on [0, ∞)",
            |x| (-x).exp() / x.sqrt(),
            0.0,
            inf,
            pi.sqrt(),
        ),
        ("x² e^(−x) on [0, ∞)", |x| x * x * (-x).exp(), 0.0, inf, 2.0),
        (
            "e^(−x) cos x on [0, ∞)",
            |x| (-x).exp() * x.cos(),
            0.0,
            inf,
            0.5,
        ),
        (
            "1/(1 + x)² on [0, ∞)",
            |x| 1.0 / ((1.0 + x) * (1.0 + x)),
            0.0,
            inf,
            1.0,
        ),
        ("eˣ on (−∞, 0]", |x| x.exp(), -inf, 0.0, 1.0),
        (
            "e^(−x²) on the line",
            |x| (-x * x).exp(),
            -inf,
            inf,
            pi.sqrt(),
        ),
        (
            "1/(1 + x²) on the line",
            |x| 1.0 / (1.0 + x * x),
            -inf,
            inf,
            pi,
        ),
        (
            "e^(−(x−30)²) on [0, ∞)",
            |x| (-(x - 30.0) * (x - 30.0)).exp(),
            0.0,
            inf,
            pi.sqrt(),
        ),
    ];
    for (what, f, a, b, exact) in cases {
        for tol in [1e-12, 1e-6] {
            let r = tanh_sinh(f, a, b, &at(tol)).unwrap();
            assert!(r.converged && honest(&r, exact), "{what} at {tol:e}: {r:?}");
        }
    }
    // Too slow a decay for the rule, or none at all: not converged, and within the budget.
    let cases: [(&str, F, f64, f64); 3] = [
        ("x^(−1.1) on [1, ∞)", |x| x.powf(-1.1), 1.0, 10.0),
        ("1/x on [1, ∞)", |x| 1.0 / x, 1.0, f64::NAN),
        ("1 on [0, ∞)", |_| 1.0, 0.0, f64::NAN),
    ];
    for (what, f, a, exact) in cases {
        let r = tanh_sinh(f, a, inf, &QuadOptions::new()).unwrap();
        assert!(!r.converged || honest(&r, exact), "{what}: {r:?}");
        if exact.is_nan() {
            assert!(!r.converged, "{what}: {r:?}");
        }
        assert!(r.evaluations <= 100_000);
    }
}

#[test]
fn rounding_that_swamps_the_differences_of_the_levels_is_in_the_error_estimate() {
    // Review round 1 (mutation M15): where the integrand is large beside its integral, the
    // rounding of the sums, about ε ∫|f|, is larger than the difference of the last two levels,
    // which see the same rounding. Without the round-off term in its estimate the rule reported
    // ∫₀¹ (10⁴ sin 14πx + 1) dx = 1 as 1 + 5.3e-12 with the error 7.9e-13: 9 of these 54
    // results were beyond their estimate, none is with it.
    let two_pi = 2.0 * core::f64::consts::PI;
    let mut converged = 0;
    for scale in [1e4, 1e8, 1e12] {
        for k in [1.0, 3.0, 7.0] {
            let wave = move |x: f64| scale * (two_pi * k * x).sin() + 1.0;
            let shifted = move |x: f64| scale * (core::f64::consts::PI * k * x).sin() + x;
            for tol in [1e-2, 1e-6, 1e-10] {
                let r = tanh_sinh(wave, 0.0, 1.0, &at(tol)).unwrap();
                assert!(honest(&r, 1.0), "{scale} sin 2π{k}x + 1 at {tol:e}: {r:?}");
                converged += usize::from(r.converged);
                let r = tanh_sinh(shifted, 0.0, 2.0, &at(tol)).unwrap();
                assert!(honest(&r, 2.0), "{scale} sin π{k}x + x at {tol:e}: {r:?}");
                converged += usize::from(r.converged);
            }
        }
    }
    // Not vacuously: the tolerances the rounding allows are met (measured 36 of 54).
    assert!(converged >= 30, "{converged} of 54 converged");
}

#[test]
fn no_result_is_accepted_before_the_third_level() {
    // Review round 1 (mutation M3). The error estimate of the second level rests on a single
    // difference of levels, and where the two agree to the round-off of their sums it is that
    // round-off: two levels that both miss a feature agree exactly. A peak of width 0.005 centred
    // on a node of the third level, x(t = 1/4) = ½ + ½ tanh((π/2) sinh ¼) = 0.6886, rounds to 0
    // at every node of the first two: accepting the second level gives 0, converged.
    let centre = 0.5 + 0.5 * (core::f64::consts::FRAC_PI_2 * 0.25f64.sinh()).tanh();
    let peak = move |x: f64| (-((x - centre) / 0.005) * ((x - centre) / 0.005)).exp();
    let exact = 0.005 * core::f64::consts::PI.sqrt();
    for tol in [1e-3, 1e-8, 1e-12] {
        let r = tanh_sinh(peak, 0.0, 1.0, &at(tol)).unwrap();
        assert!(r.converged && honest(&r, exact), "at {tol:e}: {r:?}");
    }
    // On [0, 1] the first level has 10 nodes (its sides reach t = 3 towards 1 and t = 6 towards
    // 0 before a node would round onto an end) and the second adds 9: a budget that does not
    // allow the third level's 18 nodes ends unconverged, whatever the tolerance.
    let short = at(1e-1).with_max_evaluations(36);
    let r = tanh_sinh(|_: f64| 1.0, 0.0, 1.0, &short).unwrap();
    assert!(!r.converged && r.evaluations == 19, "{r:?}");
}

/// A Gaussian peak of width `w` about `c`.
fn gauss(c: f64, w: f64) -> impl Fn(f64) -> f64 {
    move |x| {
        let t = (x - c) / w;
        (-t * t).exp()
    }
}

#[test]
fn levels_that_disagree_in_their_first_digit_are_not_taken_for_converged() {
    // Review round 2, R2-B2. Where the nodes have seen only the tail of a peak, a coarser level
    // holds less of it than a finer one, anything from nothing to most of it, and their
    // difference is a large part of the finer level's sum, yet below it: a test of the error
    // against the whole estimate let through 1.1·10⁻⁸ ± 1.1·10⁻⁸ for ∫ e^(−((x+0.04)/0.01)²) =
    // 0.0177 at 10⁻⁶. A result is now accepted only with an error below a tenth of it. Peaks
    // near the centre of [−1, 1], where the old rule had 36 (width 0.01) and 110 (0.003) of 405
    // runs beyond their estimates, up to 2.5·10⁶ times.
    let sqrt_pi = core::f64::consts::PI.sqrt();
    let mut failures = Vec::new();
    for (c, w) in [(-0.04, 0.01), (-0.012, 0.003)] {
        let r = tanh_sinh(gauss(c, w), -1.0, 1.0, &at(1e-6)).unwrap();
        if !(r.converged && honest(&r, w * sqrt_pi)) {
            failures.push(format!("the review's c = {c}, w = {w}: {r:?}"));
        }
    }
    let step = if cfg!(miri) { 20 } else { 1 };
    for w in [0.01, 0.003] {
        for tol in [1e-6, 1e-8] {
            for k in (-40..=40).step_by(step) {
                let c = f64::from(k) * w / 10.0;
                let r = tanh_sinh(gauss(c, w), -1.0, 1.0, &at(tol)).unwrap();
                if !honest(&r, w * sqrt_pi) {
                    failures.push(format!("c = {c:e}, w = {w}, tol {tol:e}: {r:?}"));
                }
            }
        }
    }
    // A Lorentzian of width 1.16·10⁻⁴ about 2.959 on [1, ∞) (exact 3.6467·10⁻⁴, the review's
    // bumps.csv id 626): level 9 puts a node near its centre, the levels after it halve that
    // node's share, and the old rule accepted 4.29·10⁻⁴ ± 3.87·10⁻⁴ at 10⁻³ — within its
    // estimate only because the estimate was nine tenths of the value. Refused that, it must not
    // take the falls 3.9·10⁻⁴, 1.7·10⁻⁴ of a sum of 2.6·10⁻⁴ for double exponential
    // convergence either, which they are not: the run ends unconverged, or right.
    let (c, w) = (2.9593111465700366, 0.00011608163360711831);
    let lorentz = move |x: f64| 1.0 / (1.0 + ((x - c) / w) * ((x - c) / w));
    // data/gen_review2.py
    let exact = 0.000_364_674_329_967_458_4;
    let r = tanh_sinh(lorentz, 1.0, f64::INFINITY, &at(1e-3)).unwrap();
    if !(honest(&r, exact) && (!r.converged || r.error < 0.1 * r.value.abs())) {
        failures.push(format!("the Lorentzian on [1, ∞): {r:?}"));
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn the_rounding_of_the_points_is_in_the_error_estimate() {
    // Review round 2. A node is formed from the end it approaches, x = b − d, and rounded to the
    // integrand's type: it lies up to about ε(|x| + d) from where the rule has it, and every
    // level shares the error that makes, which no difference of levels sees. The old estimates:
    // 4.3·10⁻¹⁶ for a true error of 1.5·10⁻¹⁵ on a Gaussian of width 0.061 about 21.4 on
    // [10, ∞) (the review's bumps.csv id 31; exact 0.10801050469439561, data/gen_review2.py), and 5.5·10⁻¹⁸
    // for 1.8·10⁻¹⁷ on a peak of width 0.001 about −0.0008 on [−1, 1]. Both now hold.
    let exact = 0.108_010_504_694_395_6;
    for tol in [1e-12, 1e-6, 1e-3] {
        let r = tanh_sinh(
            gauss(21.415596906922598, 0.06093840166231403),
            10.0,
            f64::INFINITY,
            &at(tol),
        )
        .unwrap();
        assert!(r.converged && honest(&r, exact), "at {tol:e}: {r:?}");
    }
    let sqrt_pi = core::f64::consts::PI.sqrt();
    let step = if cfg!(miri) { 20 } else { 1 };
    for w in [1e-3, 3e-3] {
        for k in (-40..=40).step_by(step) {
            let c = f64::from(k) * w / 10.0;
            let r = tanh_sinh(gauss(c, w), -1.0, 1.0, &at(1e-12)).unwrap();
            assert!(honest(&r, w * sqrt_pi), "c = {c:e}, w = {w}: {r:?}");
        }
    }
}
