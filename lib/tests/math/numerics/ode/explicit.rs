//! The explicit methods against Hairer's codes, against SciPy and against the exact solutions:
//! the steps they take, the accuracy they reach, their orders with fixed steps, the proportion of
//! their global errors to the tolerance, what they cost, and the same in `f32`.

use super::problems::{Problem, col, error, fixed, log_slope, miri_or, tol};
use super::reference::{DOP853, DOPRI5, EXACT, Run, SCIPY_DOP853, SCIPY_RK45};
use a::math::matrix::{Matrix, Scalar};
use a::math::numerics::{
    DormandPrince5, DormandPrince853, OdeMethod, OdeOptions, OdeResult, OdeSolution, Real, Rk4,
    integrate, integrate_with, solve_ivp_with, solve_with,
};

fn exact(p: Problem) -> &'static [f64] {
    EXACT.iter().find(|e| e.0 == p).unwrap().1
}

fn run<T: Real + Scalar, M: OdeMethod<T>>(p: Problem, tolerance: f64, method: M) -> OdeResult<T> {
    let t1 = T::from_f64(p.t1());
    integrate_with(
        p.f::<T>(),
        &col(&p.y0()),
        T::ZERO,
        t1,
        method,
        &tol(tolerance),
    )
    .unwrap_or_else(|| panic!("{p:?} at {tolerance:e} failed"))
}

/// The runs a test can afford: all of them, and under Miri the two shortest, at the loosest
/// tolerance.
fn affordable(runs: &[Run]) -> Vec<&Run> {
    runs.iter()
        .filter(|r| {
            !cfg!(miri)
                || (r.tol >= 1e-4 && matches!(r.problem, Problem::Decay | Problem::Brusselator))
        })
        .collect()
}

/// `a` within 1% of `b`.
fn close(a: usize, b: usize) -> bool {
    a.abs_diff(b) <= (b / 100).max(1)
}

/// The integrations of `runs` by `method` take the reference's steps, rejections and evaluations,
/// and end where it ends to a tenth of their common error. The counts were equal in every run
/// when the data were made (aarch64, macOS); one percent is allowed for a platform whose `pow`
/// rounds a borderline step the other way, but nine runs in ten must still agree exactly.
fn same_steps<M: OdeMethod<f64>>(runs: &[Run], method: impl Fn() -> M) {
    let runs = affordable(runs);
    let mut exact_matches = 0;
    for r in &runs {
        let ours = run::<f64, _>(r.problem, r.tol, method());
        let label = format!("{:?} at {:e}", r.problem, r.tol);
        assert!(
            close(ours.steps, r.steps),
            "{label}: {} steps, the code {}",
            ours.steps,
            r.steps
        );
        assert!(
            ours.rejected.abs_diff(r.rejected) <= (r.rejected / 50).max(2),
            "{label}: {} rejected, the code {}",
            ours.rejected,
            r.rejected
        );
        assert!(
            close(ours.nfev, r.nfev),
            "{label}: {} evaluations, the code {}",
            ours.nfev,
            r.nfev
        );
        if (ours.steps, ours.rejected, ours.nfev) == (r.steps, r.rejected, r.nfev) {
            exact_matches += 1;
        }
        let apart = error(ours.y.as_slice(), r.y);
        let wrong = error(r.y, exact(r.problem));
        assert!(
            apart <= 0.1 * wrong + 1e-14,
            "{label}: {apart:e} from the code, whose error is {wrong:e}"
        );
    }
    assert!(
        exact_matches * 10 >= runs.len() * 9,
        "{exact_matches} of {} runs",
        runs.len()
    );
}

#[test]
fn dormand_prince5_takes_the_steps_of_hairers_dopri5() {
    // 18 runs: six problems at rtol = atol = 1e-4, 1e-7, 1e-10.
    assert_eq!(DOPRI5.len(), 18);
    same_steps(DOPRI5, DormandPrince5::new);
}

#[test]
fn dormand_prince853_takes_the_steps_of_the_published_dop853() {
    // 24 runs: six problems at rtol = atol = 1e-4, 1e-7, 1e-10, 1e-13.
    assert_eq!(DOP853.len(), 24);
    same_steps(DOP853, DormandPrince853::new);
}

/// Against `solve_ivp`, whose controllers are simpler (no PI control, a different first step):
/// the error within three times SciPy's or the tolerance, and the steps and evaluations within a
/// quarter of SciPy's.
fn same_class<M: OdeMethod<f64>>(runs: &[Run], method: impl Fn() -> M) {
    for r in affordable(runs) {
        let ours = run::<f64, _>(r.problem, r.tol, method());
        let label = format!("{:?} at {:e}", r.problem, r.tol);
        let (mine, theirs) = (
            error(ours.y.as_slice(), exact(r.problem)),
            error(r.y, exact(r.problem)),
        );
        assert!(
            mine <= 3.0 * theirs.max(r.tol),
            "{label}: error {mine:e}, SciPy {theirs:e}"
        );
        let steps = ours.steps as f64 / r.steps as f64;
        let nfev = ours.nfev as f64 / r.nfev as f64;
        assert!(
            (0.8..=1.25).contains(&steps),
            "{label}: {} steps, SciPy {}",
            ours.steps,
            r.steps
        );
        assert!(
            (0.8..=1.25).contains(&nfev),
            "{label}: {} evaluations, SciPy {}",
            ours.nfev,
            r.nfev
        );
    }
}

#[test]
fn the_explicit_methods_match_scipys_accuracy_and_work() {
    same_class(SCIPY_RK45, DormandPrince5::new);
    same_class(SCIPY_DOP853, DormandPrince853::new);
}

/// How far the energy `(y₀² + y₁²)/2` of the oscillator has drifted from its initial ½.
fn drift(y: &[f64]) -> f64 {
    (0.5 * (y[0] * y[0] + y[1] * y[1]) - 0.5).abs()
}

#[test]
fn the_harmonic_oscillator_drifts_in_energy_no_more_than_with_scipy() {
    // Over ten periods neither method conserves the energy — the drift grows with the time, at
    // some two tolerances a period for the 5(4) pair and less for the 8(5,3) method — and the
    // drift is not larger than that of SciPy's methods at the same tolerance.
    for (runs, five) in [(SCIPY_RK45, true), (SCIPY_DOP853, false)] {
        let afford = |r: &&Run| !cfg!(miri) || r.tol >= 1e-4;
        for r in runs
            .iter()
            .filter(|r| r.problem == Problem::Oscillator)
            .filter(afford)
        {
            let ours = if five {
                run::<f64, _>(r.problem, r.tol, DormandPrince5::new()).y
            } else {
                run::<f64, _>(r.problem, r.tol, DormandPrince853::new()).y
            };
            let (mine, theirs) = (drift(ours.as_slice()), drift(r.y));
            assert!(
                mine <= 1.5 * theirs,
                "{five} at {:e}: {mine:e}, SciPy {theirs:e}",
                r.tol
            );
            assert!(mine < 25.0 * r.tol, "{five} at {:e}: {mine:e}", r.tol);
        }
    }
}

/// The logistic equation `y′ = y(1 − y)` from ½ on `[0, 10]`, whose solution is `1/(1 + e⁻ᵗ)`.
fn logistic(_t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>) {
    dy[(0, 0)] = y[(0, 0)] * (1.0 - y[(0, 0)]);
}

/// The errors at `t = 10` of `n` equal steps, for each `n`.
fn fixed_errors(ns: &[usize], solve: impl Fn(usize) -> Matrix<f64>) -> Vec<(f64, f64)> {
    let exact = 1.0 / (1.0 + (-10.0f64).exp());
    ns.iter()
        .map(|&n| (10.0 / n as f64, (solve(n)[(0, 0)] - exact).abs()))
        .collect()
}

#[test]
fn each_method_has_its_order_with_fixed_steps() {
    if cfg!(miri) {
        // Hundreds of steps for a statistic; the lighter tests run the same code under Miri.
        return;
    }
    let y0 = col::<f64>(&[0.5]);
    let opts = OdeOptions::new();
    // The classical method in its own fixed form: 16 times smaller a halving.
    let rk4 = fixed_errors(&[20, 40, 80, 160], |n| {
        solve_with(logistic, &y0, 0.0, 10.0, Rk4::fixed(n), &opts).unwrap()
    });
    let p = log_slope(&rk4);
    assert!((3.8..4.4).contains(&p), "Rk4: order {p}, errors {rk4:?}");
    // The pairs held to a fixed step by tolerances no step can fail.
    let five = fixed_errors(&[20, 40, 80, 160], |n| {
        let r = integrate_with(
            logistic,
            &y0,
            0.0,
            10.0,
            DormandPrince5::new(),
            &fixed(10.0 / n as f64),
        );
        let r = r.unwrap();
        assert_eq!((r.steps, r.rejected), (n, 0));
        r.y
    });
    let p = log_slope(&five);
    assert!(
        (4.8..5.6).contains(&p),
        "DormandPrince5: order {p}, errors {five:?}"
    );
    let eight = fixed_errors(&[5, 10, 20], |n| {
        let r = integrate_with(
            logistic,
            &y0,
            0.0,
            10.0,
            DormandPrince853::new(),
            &fixed(10.0 / n as f64),
        );
        let r = r.unwrap();
        assert_eq!((r.steps, r.rejected), (n, 0));
        r.y
    });
    let p = log_slope(&eight);
    assert!(
        (7.6..9.0).contains(&p),
        "DormandPrince853: order {p}, errors {eight:?}"
    );
}

/// The errors against `reference` of `solve` at each of the tolerances.
fn tolerance_errors(
    tols: &[f64],
    reference: &Matrix<f64>,
    solve: impl Fn(f64) -> Matrix<f64>,
) -> Vec<(f64, f64)> {
    tols.iter()
        .map(|&t| (t, error(solve(t).as_slice(), reference.as_slice())))
        .collect()
}

#[test]
fn the_global_error_follows_the_tolerance() {
    // Over five decades the error tracks the tolerance: in proportion for the pairs, whose
    // estimates are of the error of the solution they propagate, and as its 4/5th power for the
    // classical method, whose estimate is of the error of a step (h⁵) while the solution is of
    // the fourth order.
    if cfg!(miri) {
        // Thousands of steps for a statistic; the lighter tests run the same code under Miri.
        return;
    }
    let decades = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9];
    for p in [Problem::Lotka, Problem::Brusselator] {
        let reference = run::<f64, _>(p, 1e-14, DormandPrince853::new()).y;
        let five = tolerance_errors(&decades, &reference, |t| {
            run::<f64, _>(p, t, DormandPrince5::new()).y
        });
        let eight = tolerance_errors(&[1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11], &reference, |t| {
            run::<f64, _>(p, t, DormandPrince853::new()).y
        });
        let four = tolerance_errors(&decades, &reference, |t| run::<f64, _>(p, t, Rk4::new()).y);
        for (name, points, range) in [
            ("DormandPrince5", &five, 0.85..1.2),
            ("DormandPrince853", &eight, 0.9..1.3),
            ("Rk4", &four, 0.7..1.0),
        ] {
            let slope = log_slope(points);
            assert!(
                range.contains(&slope),
                "{p:?}, {name}: slope {slope}, {points:?}"
            );
            // And never far from its line: the error over the tolerance to that power moves by
            // less than 30 across the five decades.
            let ratios: Vec<f64> = points.iter().map(|(t, e)| e / t.powf(slope)).collect();
            let (lo, hi) = ratios
                .iter()
                .fold((f64::MAX, 0.0f64), |(lo, hi), &r| (lo.min(r), hi.max(r)));
            assert!(hi / lo < 30.0, "{p:?}, {name}: {points:?}");
        }
    }
}

#[test]
fn the_evaluations_are_counted_as_the_codes_count_them() {
    let (p, tolerance) = miri_or((Problem::Brusselator, 1e-4), (Problem::Arenstorf, 1e-8));
    // DOPRI5: one evaluation at the start, one for the first step, six for every step tried.
    let r = run::<f64, _>(p, tolerance, DormandPrince5::new());
    assert!(r.rejected > 0);
    assert_eq!(r.nfev, 2 + 6 * (r.steps + r.rejected));
    assert_eq!((r.njev, r.nlu), (0, 0));
    // DOP853: eleven for every step tried and one more for every one accepted.
    let r = run::<f64, _>(p, tolerance, DormandPrince853::new());
    assert!(r.rejected > 0);
    assert_eq!(r.nfev, 2 + 11 * (r.steps + r.rejected) + r.steps);
    // With the dense output kept, three more for every accepted step, and only for those.
    let opts = tol::<f64>(tolerance);
    let y0 = col(&p.y0());
    let s = solve_ivp_with(
        p.f::<f64>(),
        &y0,
        0.0,
        p.t1(),
        DormandPrince853::new(),
        &mut [],
        &opts,
    );
    assert_eq!(s.nfev, 2 + 11 * (s.steps + s.rejected) + 4 * s.steps);
    if !cfg!(miri) {
        // The same steps as without the dense output (under Miri, whose powf is perturbed from
        // run to run, the two runs need not agree to the last bit).
        assert_eq!((s.steps, s.rejected), (r.steps, r.rejected));
        assert_eq!(s.y, r.y);
    }
    // The classical method: eleven for a pair of steps and the whole step it is checked by.
    let r = run::<f64, _>(p, tolerance, Rk4::new());
    assert_eq!(r.steps % 2, 0);
    assert_eq!(r.nfev, 2 + 11 * (r.steps / 2 + r.rejected));
    // And in its fixed form four a step, nothing else.
    let n = miri_or(50, 1000);
    let r = integrate(p.f::<f64>(), &y0, 0.0, p.t1(), Rk4::fixed(n)).unwrap();
    assert_eq!((r.steps, r.rejected, r.nfev), (n, 0, 4 * n));
    // Every evaluation is counted: the count is that of the calls of f.
    let calls = core::cell::Cell::new(0usize);
    let counted = |t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>| {
        calls.set(calls.get() + 1);
        p.rhs(t, y, dy);
    };
    let r = integrate_with(counted, &y0, 0.0, p.t1(), DormandPrince853::new(), &opts).unwrap();
    assert_eq!(r.nfev, calls.get());
}

#[test]
fn the_classical_method_reports_its_half_steps() {
    // The pair of half-steps that passed the doubling test is what is kept, so the steps come
    // in pairs of equal length.
    let p = Problem::Lotka;
    let opts = tol::<f64>(miri_or(1e-3, 1e-7));
    let s = solve_ivp_with(
        p.f::<f64>(),
        &col(&p.y0()),
        0.0,
        p.t1(),
        Rk4::new(),
        &mut [],
        &opts,
    );
    let steps = s.dense.as_ref().unwrap().steps();
    assert_eq!(steps.len() % 2, 0);
    for pair in steps.chunks(2) {
        let (a, b) = (pair[0].t1() - pair[0].t0(), pair[1].t1() - pair[1].t0());
        assert!((a - b).abs() <= 1e-12 * a, "{a} and {b}");
        assert_eq!(pair[0].t1(), pair[1].t0());
    }
}

#[test]
fn the_explicit_methods_work_in_f32() {
    // At tolerances an f32 can meet, the f32 integration is as accurate as the f64 one at the
    // same tolerance, give or take the rounding of the f32 state.
    let problems = miri_or(
        &[Problem::Decay][..],
        &[Problem::Decay, Problem::Lotka, Problem::Brusselator][..],
    );
    for &p in problems {
        let reference = run::<f64, _>(p, 1e-13, DormandPrince853::new()).y;
        for tolerance in [1e-4, 1e-5] {
            let check = |name: &str, low: Matrix<f32>, high: Matrix<f64>| {
                let (e32, e64) = (
                    error(low.as_slice(), reference.as_slice()),
                    error(high.as_slice(), reference.as_slice()),
                );
                assert!(
                    e32 <= 12.0 * e64.max(tolerance) && e32 < 50.0 * tolerance,
                    "{p:?} at {tolerance:e}, {name}: f32 {e32:e}, f64 {e64:e}"
                );
            };
            check(
                "DormandPrince5",
                run::<f32, _>(p, tolerance, DormandPrince5::new()).y,
                run::<f64, _>(p, tolerance, DormandPrince5::new()).y,
            );
            check(
                "DormandPrince853",
                run::<f32, _>(p, tolerance, DormandPrince853::new()).y,
                run::<f64, _>(p, tolerance, DormandPrince853::new()).y,
            );
            check(
                "Rk4",
                run::<f32, _>(p, tolerance, Rk4::new()).y,
                run::<f64, _>(p, tolerance, Rk4::new()).y,
            );
        }
    }
    // The fixed form too, until the rounding of the f32 state takes over.
    let y0 = col::<f32>(&[0.5]);
    let f =
        |_t: f32, y: &Matrix<f32>, dy: &mut Matrix<f32>| dy[(0, 0)] = y[(0, 0)] * (1.0 - y[(0, 0)]);
    let exact = 1.0 / (1.0 + (-10.0f64).exp());
    let opts = OdeOptions::new();
    let at = |n: usize| {
        let y = solve_with(f, &y0, 0.0, 10.0, Rk4::fixed(n), &opts).unwrap();
        (f64::from(y[(0, 0)]) - exact).abs()
    };
    // f64 gives 7.1e-6 with ten steps and 1.4e-8 with forty, where the f32 state's spacing of
    // 6e-8 near 1 takes over.
    assert!(at(10) < 1e-5, "{:e}", at(10));
    assert!(at(40) < 5e-8, "{:e}", at(40));
}

#[test]
fn the_step_doubling_estimate_is_a_fifteenth_of_the_difference() {
    // Rk4::new() on y′ = −y from 1, the first pair of steps 0.5 long: the whole step makes R(−0.5)
    // and the two half-steps R(−0.25)², R(z) = 1 + z + z²/2 + z³/6 + z⁴/24 the method's stability
    // polynomial. Richardson's estimate of the error of the half-steps is a fifteenth of their
    // difference from the whole step, and sets the next pair to
    // 0.5·min(5, max(0.2, 0.9·err^(−1/5))), computed here; the half-steps are what is kept.
    let r = |z: f64| 1.0 + z * (1.0 + z * (0.5 + z * (1.0 / 6.0 + z / 24.0)));
    let (h, tolerance) = (0.5, 1e-4);
    let (whole, halves) = (r(-h), r(-0.5 * h) * r(-0.5 * h));
    let err = ((halves - whole) / 15.0).abs() / (tolerance + tolerance * halves.abs().max(1.0));
    assert!(err < 1.0, "{err}");
    let next = h * (0.9 * err.powf(-0.2)).clamp(0.2, 5.0);
    let f = |_t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>| dy[(0, 0)] = -y[(0, 0)];
    let opts = tol::<f64>(tolerance).with_h0(h);
    let s = solve_ivp_with(f, &col(&[1.0]), 0.0, 5.0, Rk4::new(), &mut [], &opts);
    let dense = s.dense.as_ref().unwrap();
    let steps = dense.steps();
    assert_eq!(s.rejected, 0);
    assert!((steps[1].t1() - h).abs() < 1e-15, "{}", steps[1].t1());
    assert!((dense.eval(h).unwrap()[(0, 0)] - halves).abs() < 1e-15);
    let half = steps[2].t1() - steps[2].t0();
    assert!(
        (half - 0.5 * next).abs() < 1e-12 * next,
        "{half} against {}",
        0.5 * next
    );
}

/// The largest error of `method` against `reference` in units of the tolerance (the scale
/// `tol·(1 + |y|)` of `rtol = atol = tol`), at the end and at 199 points between the steps.
fn in_tolerances<M: OdeMethod<f64>>(
    p: Problem,
    tolerance: f64,
    method: M,
    reference: &OdeSolution<f64>,
) -> (f64, f64) {
    let opts = tol(tolerance);
    let s = solve_ivp_with(
        p.f::<f64>(),
        &col(&p.y0()),
        0.0,
        p.t1(),
        method,
        &mut [],
        &opts,
    );
    let (got, want) = (s.dense.as_ref().unwrap(), reference.dense.as_ref().unwrap());
    let end = error(s.y.as_slice(), reference.y.as_slice()) / tolerance;
    let between = (1..200)
        .map(|k| p.t1() * f64::from(k) / 200.0)
        .map(|t| {
            error(
                got.eval(t).unwrap().as_slice(),
                want.eval(t).unwrap().as_slice(),
            )
        })
        .fold(0.0, f64::max)
        / tolerance;
    (end, between)
}

#[test]
fn the_classical_method_errs_by_many_tolerances_at_stringent_ones() {
    // Rk4::new()'s tolerance bounds Richardson's estimate of the error of each step, of the fifth
    // order, while the solution is of the fourth: at 1e-9 its global error is many tolerances,
    // where DormandPrince5's stays in the class of the tolerance, and between its steps the cubic
    // Hermite polynomial is further off still. These are the figures the documentation of Rk4
    // quotes (measured: Lotka–Volterra 96 at the end and 330 between the steps, the Brusselator
    // 3.6 and 120; DormandPrince5 6.2 and 1.1 at the end); the bounds leave a factor of two.
    if cfg!(miri) {
        // Thousands of steps; the lighter tests run the same code under Miri.
        return;
    }
    let tolerance = 1e-9;
    for (p, rk4_end, rk4_between, dp5_end) in [
        (Problem::Lotka, (50.0, 200.0), (150.0, 700.0), 12.0),
        (Problem::Brusselator, (1.8, 8.0), (60.0, 250.0), 2.2),
    ] {
        let opts = tol(1e-14);
        let y0 = col(&p.y0());
        let reference = solve_ivp_with(
            p.f::<f64>(),
            &y0,
            0.0,
            p.t1(),
            DormandPrince853::new(),
            &mut [],
            &opts,
        );
        let (end, between) = in_tolerances(p, tolerance, Rk4::new(), &reference);
        let (five, _) = in_tolerances(p, tolerance, DormandPrince5::new(), &reference);
        println!("{p:?} at 1e-9: Rk4 {end:.2} at the end, {between:.1} between; DP5 {five:.2}");
        assert!(end > rk4_end.0 && end < rk4_end.1, "{p:?}: {end}");
        assert!(
            between > rk4_between.0 && between < rk4_between.1,
            "{p:?}: {between}"
        );
        assert!(five < dp5_end && five < end, "{p:?}: {five}");
    }
}
