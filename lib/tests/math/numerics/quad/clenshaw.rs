//! Clenshaw–Curtis quadrature: its nodes and weights against the interpolatory rule computed by
//! mpmath (`data/gen_clenshaw.py`), the nesting of its levels, the honesty of its error estimate
//! where the old rule was fooled (audit n_quad-3), and its adaptivity in degree and panels.

use a::math::numerics::{QuadOptions, QuadResult, clenshaw_curtis};
use std::cell::{Cell, RefCell};

/// An integrand of the tables of cases.
type F = fn(f64) -> f64;

const REFERENCE: &str = include_str!("data/clenshaw_ref.txt");

thread_local! {
    /// The points an integrand was evaluated at, in order.
    static SEEN: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
    /// The node an indicator integrand is 1 at.
    static NODE: RefCell<f64> = const { RefCell::new(f64::NAN) };
}

/// The reference rule of `N` points on `[-1, 1]`: nodes (`k = 0` at 1) and weights.
fn reference(n: usize) -> (Vec<f64>, Vec<f64>) {
    REFERENCE
        .lines()
        .filter(|l| !l.starts_with('#'))
        .map(|l| l.split_whitespace().collect::<Vec<_>>())
        .filter(|f| f[0].parse::<usize>().unwrap() == n)
        .map(|f| (f[2].parse::<f64>().unwrap(), f[3].parse::<f64>().unwrap()))
        .unzip()
}

/// The distance from `got` to `want` in units in the last place of `want`.
fn ulps(got: f64, want: f64) -> f64 {
    if got == want {
        return 0.0;
    }
    let unit = 2f64.powi(want.abs().log2().floor() as i32 - 52);
    (got - want).abs() / unit
}

/// The two points off the grid at which the rule checks a panel on `[-1, 1]` against its
/// interpolant.
const OFF_GRID: [f64; 2] = [0.37, -0.61];

/// The rule on `[-1, 1]` with room for exactly the first panel of `N + 1` points.
fn first_panel(f: fn(f64) -> f64, n: usize) -> QuadResult<f64> {
    // The N + 1 points and the two off the grid.
    let opts = QuadOptions::new().with_max_evaluations(n + 3);
    clenshaw_curtis(f, -1.0, 1.0, &opts).unwrap()
}

#[test]
fn the_nodes_and_weights_are_those_of_the_interpolatory_rule() {
    // With room for the first N + 1 points, the rule is the interpolatory rule on cos(kπ/N),
    // N = 32, 64, 128, 256; its weights are read off as the estimates of integrands that are 1
    // at one node and 0 at the others. Each level evaluates only the points the level before
    // lacks.
    let mut before: Vec<f64> = Vec::new();
    let (mut worst_node, mut worst_weight) = (0.0f64, 0.0f64);
    for n in [32usize, 64, 128, 256] {
        let (nodes, weights) = reference(n);
        let largest = weights.iter().fold(0.0f64, |a, &w| a.max(w));
        SEEN.with(|s| s.borrow_mut().clear());
        let r = first_panel(
            |x| {
                SEEN.with(|s| s.borrow_mut().push(x));
                // Beyond every level up to 256 points, so that each is evaluated.
                (400.0 * x).cos()
            },
            n,
        );
        let seen = SEEN.with(|s| s.borrow().clone());
        assert_eq!(r.evaluations, n + 3, "N = {n}");
        assert_eq!(seen.len(), n + 3);
        // Nested: the points of the level before come first, and are not evaluated again.
        assert_eq!(&seen[..before.len()], &before[..], "N = {n}: not nested");
        assert_eq!(&seen[33..35], &OFF_GRID, "the points off the grid");
        let mut sorted: Vec<f64> = seen
            .iter()
            .copied()
            .filter(|x| !OFF_GRID.contains(x))
            .collect();
        sorted.sort_by(|a, b| b.total_cmp(a));
        sorted.dedup();
        assert_eq!(sorted.len(), n + 1, "N = {n}: a point evaluated twice");
        for (k, (&x, &want)) in sorted.iter().zip(&nodes).enumerate() {
            if want == 0.0 {
                assert_eq!(x, 0.0, "N = {n}: the middle node");
            } else {
                let e = ulps(x, want);
                assert!(e <= 2.0, "N = {n}, node {k}: {x} is {e} ulps from {want}");
                worst_node = worst_node.max(e);
            }
        }
        // Linearity: an integrand no level resolves takes the rule through every level, and the
        // indicator of a node added to it adds the node's weight to the estimate (the small
        // amplitude keeps the rounding of the two sums out of the difference).
        let base = first_panel(|x| 1e-6 * (400.0 * x).cos(), n).value;
        for (k, &want) in weights.iter().enumerate() {
            NODE.with(|c| *c.borrow_mut() = sorted[k]);
            let r = first_panel(
                |x| {
                    let one = if NODE.with(|c| x == *c.borrow()) {
                        1.0
                    } else {
                        0.0
                    };
                    one + 1e-6 * (400.0 * x).cos()
                },
                n,
            );
            let got = r.value - base;
            // In units in the last place of the largest weight, the accuracy a sum of the
            // rule's terms feels (the weights are within 4, the difference of the two sums adds
            // its own rounding).
            let e = ulps(got, want) * want / largest;
            assert!(
                e <= 8.0,
                "N = {n}, weight {k}: {got} is {e} ulps of the largest from {want}"
            );
            worst_weight = worst_weight.max(e);
        }
        before = seen;
    }
    // Measured when the test was written: the nodes within an ulp, the weights within 4 ulps
    // of the largest weight (the smallest weights, next to the ends, lose relative accuracy to
    // the cancellation in their cosine series: 1.6e-13 relatively at N = 256).
    assert!(
        worst_node <= 2.0 && worst_weight <= 8.0,
        "{worst_node} {worst_weight}"
    );
}

/// Whether a result keeps its promise.
fn honest(r: &QuadResult<f64>, exact: f64) -> bool {
    !r.converged || (r.value - exact).abs() <= r.error + 4.0 * f64::EPSILON * exact.abs()
}

/// The Chebyshev polynomial `T_n(x) = cos(n arccos x)`.
fn chebyshev(n: u32, x: f64) -> f64 {
    (f64::from(n) * x.clamp(-1.0, 1.0).acos()).cos()
}

#[test]
fn an_integrand_that_aliases_on_the_coarse_points_is_not_taken_for_converged() {
    // Audit n_quad-3: the old rule compared its 5-point rule with its 3-point rule and took
    // T₈, which is 1 at all five points, for 2 with an error of 4e-16, converged. ∫₋₁¹ Tₙ =
    // 2/(1 − n²) for even n; T₁₀₀ is T₂₈ on the 33 points and the 65, and T₄₀₀ and T₅₁₀ are
    // beyond one panel altogether.
    for n in [8u32, 30, 34, 64, 100, 250, 400, 510] {
        let exact = 2.0 / (1.0 - f64::from(n) * f64::from(n));
        let opts = QuadOptions::new();
        let r = clenshaw_curtis(move |x: f64| chebyshev(n, x), -1.0, 1.0, &opts).unwrap();
        assert!(honest(&r, exact), "T_{n}: {r:?}, exact {exact}");
        assert!(r.converged, "T_{n}: {r:?}");
        assert!(r.evaluations >= 33);
    }
}

#[test]
fn the_rule_is_honest_where_the_old_one_was_fooled() {
    // Audit n_quad-3: a Gaussian peak at 13 on [0, 40] came out 9e-21 from five points,
    // converged; x^0.1 was 2.1e-9 off at the tolerance 1e-8, converged; √frac(128x) 3.1e-5 off
    // at 1e-6 with the error 7.2e-7, converged (the old errors were the largest panel's, not
    // their sum).
    let sqrt_pi = core::f64::consts::PI.sqrt();
    let cases: [(&str, F, f64, f64, f64, f64); 6] = [
        (
            "e^(−(x−13)²) on [0, 40]",
            |x| (-(x - 13.0) * (x - 13.0)).exp(),
            0.0,
            40.0,
            sqrt_pi,
            1e-12,
        ),
        ("x^0.1 at 1e-8", |x| x.powf(0.1), 0.0, 1.0, 1.0 / 1.1, 1e-8),
        (
            "x^0.1 at 1e-10",
            |x| x.powf(0.1),
            0.0,
            1.0,
            1.0 / 1.1,
            1e-10,
        ),
        (
            "√frac(128x)",
            |x| (128.0 * x).fract().sqrt(),
            0.0,
            1.0,
            2.0 / 3.0,
            1e-6,
        ),
        (
            "|sin 64πx|",
            |x| (64.0 * core::f64::consts::PI * x).sin().abs(),
            0.0,
            1.0,
            2.0 / core::f64::consts::PI,
            1e-8,
        ),
        ("√x", |x| x.sqrt(), 0.0, 1.0, 2.0 / 3.0, 1e-12),
    ];
    for (what, f, a, b, exact, tol) in cases {
        let opts = QuadOptions::new().with_epsabs(tol).with_epsrel(tol);
        let r = clenshaw_curtis(f, a, b, &opts).unwrap();
        assert!(honest(&r, exact), "{what}: {r:?}, exact {exact}");
        // The peak and √x converge; the rest at least come within their estimate.
        if what.starts_with('e') || what == "√x" {
            assert!(r.converged, "{what}: {r:?}");
        }
        assert!((r.value - exact).abs() <= r.error.max(tol), "{what}: {r:?}");
    }
}

#[test]
fn a_smooth_integrand_takes_one_panel_and_no_value_twice() {
    // The rule doubles the degree of a panel before it bisects it, and reuses every value: eˣ
    // in one panel of 33 points (and the two off its grid); a hard integrand that bisects many
    // times still never evaluates a point twice.
    let r = clenshaw_curtis(|x: f64| x.exp(), 0.0, 1.0, &QuadOptions::new()).unwrap();
    assert!(r.converged && [35, 67].contains(&r.evaluations), "{r:?}");
    let runge = |x: f64| 1.0 / (1.0 + 25.0 * x * x);
    let r = clenshaw_curtis(runge, -1.0, 1.0, &QuadOptions::new()).unwrap();
    assert!(r.converged && r.evaluations < 1000, "{r:?}");
    assert!((r.value - 2.0 * 5.0f64.atan() / 5.0).abs() <= r.error);
    SEEN.with(|s| s.borrow_mut().clear());
    let r = clenshaw_curtis(
        |x: f64| {
            SEEN.with(|s| s.borrow_mut().push(x));
            (128.0 * x).fract().sqrt()
        },
        0.0,
        1.0,
        &QuadOptions::new().with_epsabs(1e-6).with_epsrel(1e-6),
    )
    .unwrap();
    let mut seen = SEEN.with(|s| s.borrow().clone());
    assert_eq!(seen.len(), r.evaluations);
    seen.sort_by(f64::total_cmp);
    seen.dedup();
    assert_eq!(seen.len(), r.evaluations, "a point was evaluated twice");
    assert!(r.evaluations > 1000, "the rule did not bisect: {r:?}");
}

#[test]
fn an_integrand_infinite_at_an_end_is_beyond_the_rule_and_says_so() {
    // The points include the ends: ∫₀¹ x^(−½) evaluates 1/0 (the last of the first 33 points)
    // and stops, not converged.
    let r = clenshaw_curtis(|x: f64| 1.0 / x.sqrt(), 0.0, 1.0, &QuadOptions::new()).unwrap();
    assert!(
        !r.converged && !(r.value.is_finite() && r.error.is_finite()),
        "{r:?}"
    );
    assert_eq!(r.evaluations, 33);
}

/// A Gaussian peak of width `w` about `c`, recording whether any value was other than 0.
fn peak(c: f64, w: f64, seen: &Cell<bool>) -> impl Fn(f64) -> f64 + '_ {
    move |x| {
        let t = (x - c) / w;
        let v = (-t * t).exp();
        if v != 0.0 {
            seen.set(true);
        }
        v
    }
}

#[test]
fn a_peak_a_panel_saw_is_never_lost_by_the_halves_it_is_bisected_into() {
    // Review round 2, R2-B1. A panel of 257 points that sees a narrow peak at a point or two is
    // bisected into halves of 33 points, which can miss it entirely, see nothing, and agree on
    // that to the last digit: their cover replaced the panel, and ∫₀¹ e^(−((x−0.286)/0.001)²) dx
    // came out 9.9·10⁻⁶⁰ within its estimate after a point had seen 0.95. A sample of the panel
    // that a half's points do not account for now stays in the half's error until a point beside
    // it sees a tenth of it. On this grid of peaks of widths 10⁻³ and 3·10⁻⁴ on [0, 1] the old rule had 13
    // to 50 runs per width and tolerance beyond their estimates after a point had seen the peak;
    // now every run is within its estimate, or saw nothing at all — every value exactly 0, the
    // peak between the 35 points of the first panel, which no rule that samples can see.
    let step = if cfg!(miri) { 97 } else { 3 };
    let sqrt_pi = core::f64::consts::PI.sqrt();
    let (mut failures, mut seen_runs) = (Vec::new(), 0);
    for w in [1e-3, 3e-4] {
        for tol in [1e-12, 1e-8, 1e-4] {
            let opts = QuadOptions::new().with_epsabs(tol).with_epsrel(tol);
            for ci in (1..1000).step_by(step) {
                let c = f64::from(ci) / 1000.0;
                if c < 50.0 * w || c > 1.0 - 50.0 * w {
                    continue;
                }
                let seen = Cell::new(false);
                let r = clenshaw_curtis(peak(c, w, &seen), 0.0, 1.0, &opts).unwrap();
                let fine = if seen.get() {
                    seen_runs += 1;
                    honest(&r, w * sqrt_pi)
                } else {
                    r.value == 0.0 && r.error == 0.0
                };
                if !fine {
                    failures.push(format!("w {w:e} c {c} tol {tol:e}: {r:?}"));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    assert!(
        seen_runs > 500 || cfg!(miri),
        "only {seen_runs} runs saw the peak"
    );
    // The review's examples.
    for c in [0.286, 0.714, 0.237, 0.143] {
        let seen = Cell::new(false);
        let r = clenshaw_curtis(peak(c, 1e-3, &seen), 0.0, 1.0, &QuadOptions::new()).unwrap();
        assert!(r.converged && honest(&r, 1e-3 * sqrt_pi), "c = {c}: {r:?}");
    }
}

#[test]
fn every_peak_a_panel_saw_is_kept_not_only_the_heaviest() {
    // The confirmation of review round 2, its case CC1. A half kept one witness, the heaviest:
    // of two peaks of width 3·10⁻⁴ about 0.160 and 0.303 on [0, 1], which the panel of 257
    // points saw at 0.9997 and 0.98 of their heights, the half [0, 0.5] kept the evidence of the
    // first, its points then found that one, and the second was lost: 5.3·10⁻⁴ ± 2.8·10⁻¹⁰,
    // converged, at 10⁻⁸, for 8.5·10⁻⁴. A half keeps the heaviest four now.
    let (c1, c2, h2, w) = (
        0.159_637_226_506_726_2,
        0.302_965_945_384_920_83,
        0.596_902_082_585_257_3,
        3e-4,
    );
    let f = move |x: f64| {
        let (t, u) = ((x - c1) / w, (x - c2) / w);
        (-t * t).exp() + h2 * (-u * u).exp()
    };
    let exact = w * core::f64::consts::PI.sqrt() * (1.0 + h2);
    for tol in [1e-12, 1e-8] {
        let opts = QuadOptions::new().with_epsabs(tol).with_epsrel(tol);
        let r = clenshaw_curtis(f, 0.0, 1.0, &opts).unwrap();
        assert!(r.converged && honest(&r, exact), "at {tol:e}: {r:?}");
    }
    // The confirmation's sweep of pairs of peaks on [0, 1], both in one half or one in each: at
    // 10⁻¹² and 10⁻⁸ no converged run whose points saw both peaks above 10⁻³ of their heights is
    // beyond its estimate (the code of round 2: CC1). At 10⁻⁴ some are, at most as many as the
    // documentation states (its Limits: evidence that shrinks below the tolerance before a point
    // finds the peak; 154 of 1009 converged runs that saw both when this was written, 173 on
    // the code of round 2).
    let on_unit =
        |f: &dyn Fn(f64) -> f64, o: &QuadOptions<f64>| clenshaw_curtis(f, 0.0, 1.0, o).unwrap();
    let (mut seen, mut failures, mut loose) = (0, Vec::new(), 0);
    for split in [false, true] {
        for w in [1e-3, 3e-4] {
            for tol in [1e-12, 1e-8] {
                let (s, beyond) = super::pair_sweep(&on_unit, w, split, tol);
                seen += s;
                failures.extend(beyond);
            }
            loose += super::pair_sweep(&on_unit, w, split, 1e-4).1.len();
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    assert!(seen > 1000, "only {seen} runs saw both peaks");
    assert!(
        loose <= 154,
        "{loose} runs beyond their estimates at 1e-4, the documentation says 154"
    );
}

#[test]
fn the_rounding_of_the_points_is_in_the_error_estimate() {
    // Review round 2. A point of a panel is formed as mid + half·cos(kπ/N) and rounded, and lies
    // up to about ε·max(|a|, |b|) from where the rule has it; every level shares the error that
    // makes, and no difference of levels sees it. Peaks of the review's sweep away from the
    // origin, at 10⁻¹²: the old estimates were 1.6 to 5.6 times below the true errors (about
    // 10⁻¹⁴ relatively), the estimate now holds the rounding of the points.
    let cases: [(f64, f64, f64, f64, f64); 4] = [
        // a, b, centre, width, ∫ (mpmath, data/gen_review2.py)
        (
            100.0,
            167.66011795304195,
            101.01364004955065,
            0.05986618496275109,
            0.106_110_050_076_250_06,
        ),
        (
            100.0,
            124.35747911845812,
            118.9773779740595,
            0.06948934569848714,
            0.123_166_658_380_188_18,
        ),
        (
            2.0,
            10.123824066840644,
            8.480353527792523,
            0.00839085192224972,
            0.014_872_397_801_969_469,
        ),
        (
            0.0,
            1.4653368690885704,
            1.1560060288518472,
            0.0006237717139641789,
            0.001_105_606_576_501_743,
        ),
    ];
    for (a, b, c, w, exact) in cases {
        let seen = Cell::new(false);
        let r = clenshaw_curtis(peak(c, w, &seen), a, b, &QuadOptions::new()).unwrap();
        assert!(
            r.converged && honest(&r, exact),
            "peak at {c}: {r:?}, exact {exact}"
        );
    }
}
