//! The two stiff methods, `RadauIia` and `Bdf`, against SciPy's `Radau` and `BDF` at the same
//! tolerances, on the stiff problems (Robertson to 10¹¹, Van der Pol at μ = 1000, HIRES, OREGO, a
//! linear system of dimension 50) and on the non-stiff ones of the explicit methods; their global
//! error against the tolerance; their dense output and events on stiff problems.

use super::problems::{Problem, col, error, log_slope, miri_or, tol};
use super::reference::EXACT;
use super::stiff_problems::{Stiff, linear_exact, linear_matrix, stiff_error};
use super::stiff_reference::{
    REFERENCE, ROBERTSON_40, ROBERTSON_HALF, Run, SCIPY_BDF, SCIPY_BDF_NONSTIFF, SCIPY_RADAU,
    SCIPY_RADAU_NONSTIFF, VAN_DER_POL_10, VDP_ZEROS,
};
use a::math::matrix::Matrix;
use a::math::numerics::{
    Bdf, Event, EventDirection, OdeMethod, OdeOptions, OdeResult, OdeSolution, OdeStatus, RadauIia,
    integrate_with, solve_ivp_with,
};

/// The stiff problem `p` integrated by `method` under `opts`.
fn integrate_stiff<M: OdeMethod<f64>>(
    p: Stiff,
    a: &Matrix<f64>,
    opts: &OdeOptions<f64>,
    method: M,
) -> Option<OdeResult<f64>> {
    let f = |_t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>| p.rhs(a, y, dy);
    integrate_with(f, &col(&p.y0()), 0.0, p.t1(), method, opts)
}

/// The integration of the stiff problem of `run` at its tolerances by Radau (if `radau`) or BDF,
/// the Jacobian differenced but for the linear system, which SciPy was given the matrix of.
fn mine(run: &Run<Stiff>, a: &Matrix<f64>, radau: bool) -> OdeResult<f64> {
    let p = run.problem;
    let opts = OdeOptions::new().with_tolerances(run.rtol, run.atol);
    let jac = |_t: f64, y: &Matrix<f64>, out: &mut Matrix<f64>| p.jacobian(a, y, out);
    let r = match (radau, p == Stiff::Linear) {
        (true, false) => integrate_stiff(p, a, &opts, RadauIia::new()),
        (true, true) => integrate_stiff(p, a, &opts, RadauIia::new().with_jacobian(jac)),
        (false, false) => integrate_stiff(p, a, &opts, Bdf::new()),
        (false, true) => integrate_stiff(p, a, &opts, Bdf::new().with_jacobian(jac)),
    };
    r.unwrap_or_else(|| panic!("{:?} at {:e} did not finish", p, run.rtol))
}

fn reference(p: Stiff) -> &'static [f64] {
    REFERENCE.iter().find(|r| r.0 == p).unwrap().1
}

/// SciPy's count of evaluations with those of its differenced Jacobians, which its `nfev` leaves
/// out: `n` for each, as this module counts every one.
fn scipy_nfev<P>(run: &Run<P>, n: usize, differenced: bool) -> usize {
    run.nfev + if differenced { n * run.njev } else { 0 }
}

#[test]
fn the_stiff_methods_match_scipy_on_the_stiff_problems() {
    // Robertson to 10¹¹, Van der Pol at μ = 1000, HIRES, OREGO and the linear system of
    // dimension 50 at rtol = 1e-4, 1e-6, 1e-8: the error of each final state against the
    // reference is of the class of SciPy's at the same tolerances, and so are the numbers of steps
    // and evaluations. (Under Miri, HIRES at 1e-4 alone.)
    let a = linear_matrix();
    for (name, runs, radau) in [("RadauIia", SCIPY_RADAU, true), ("Bdf", SCIPY_BDF, false)] {
        let mut ratios = Vec::new();
        for run in runs {
            if cfg!(miri) && (run.problem != Stiff::Hires || run.rtol != 1e-4) {
                continue;
            }
            let p = run.problem;
            let r = mine(run, &a, radau);
            let exact = reference(p);
            let (e, e_scipy) = (
                stiff_error(r.y.as_slice(), exact, run.atol),
                stiff_error(run.y, exact, run.atol),
            );
            let n = exact.len();
            let nfev = scipy_nfev(run, n, p != Stiff::Linear);
            let steps = r.steps as f64 / run.steps as f64;
            let evaluations = r.nfev as f64 / nfev as f64;
            eprintln!(
                "{name} {p:?} {:e}: {} steps ({} rejected), {} evaluations, {} Jacobians, {} LU, \
                 error {e:.2e}; SciPy {} steps, {nfev} evaluations, {} Jacobians, {} LU, error \
                 {e_scipy:.2e}",
                run.rtol, r.steps, r.rejected, r.nfev, r.njev, r.nlu, run.steps, run.njev, run.nlu
            );
            // Each error within ten times SciPy's (or the tolerance), and on the whole within
            // twice. The scale of the module's norm, `atol + rtol·max(|yₙ|, |yₙ₊₁|)`, is looser
            // than SciPy's `atol + rtol·|yₙ₊₁|` where a component falls fast: OREGO's at 1e-4
            // makes the BDFs six times less accurate than SciPy's there, and as accurate with
            // SciPy's scale.
            let ratio = e / e_scipy.max(run.rtol);
            ratios.push(ratio);
            assert!(
                ratio <= 10.0,
                "{name} {p:?} {:e}: error {e:e} against SciPy's {e_scipy:e}",
                run.rtol
            );
            assert!(
                (0.5..=1.6).contains(&steps),
                "{name} {p:?} {:e}: {} steps against {}",
                run.rtol,
                r.steps,
                run.steps
            );
            assert!(
                (0.5..=2.0).contains(&evaluations),
                "{name} {p:?} {:e}: {} evaluations against {nfev}",
                run.rtol,
                r.nfev
            );
        }
        let mean = (ratios.iter().map(|r| r.ln()).sum::<f64>() / ratios.len() as f64).exp();
        eprintln!("{name}: geometric mean of the error ratios {mean:.3}");
        assert!(mean <= 2.0, "{name}: {mean}");
    }
}

#[test]
fn the_stiff_methods_match_scipy_on_the_nonstiff_problems() {
    // An implicit method is no worse than SciPy's on a problem that is not stiff: the six
    // problems of the explicit methods at rtol = atol = 1e-4 and 1e-7.
    if cfg!(miri) {
        return;
    }
    for (name, runs, radau) in [
        ("RadauIia", SCIPY_RADAU_NONSTIFF, true),
        ("Bdf", SCIPY_BDF_NONSTIFF, false),
    ] {
        for run in runs {
            let p: Problem = run.problem;
            let opts = tol(run.rtol);
            let y0 = col(&p.y0());
            let r = if radau {
                integrate_with(p.f(), &y0, 0.0, p.t1(), RadauIia::new(), &opts)
            } else {
                integrate_with(p.f(), &y0, 0.0, p.t1(), Bdf::new(), &opts)
            }
            .unwrap();
            let exact = EXACT.iter().find(|e| e.0 == p).unwrap().1;
            let (e, e_scipy) = (error(r.y.as_slice(), exact), error(run.y, exact));
            let nfev = scipy_nfev(run, exact.len(), true);
            eprintln!(
                "{name} {p:?} {:e}: {} steps, {} evaluations, error {e:.2e}; SciPy {} steps, \
                 {nfev} evaluations, error {e_scipy:.2e}",
                run.rtol, r.steps, r.nfev, run.steps
            );
            assert!(
                e <= 3.0 * e_scipy.max(run.rtol),
                "{name} {p:?} {:e}: error {e:e} against SciPy's {e_scipy:e}",
                run.rtol
            );
            let steps = r.steps as f64 / run.steps as f64;
            let evaluations = r.nfev as f64 / nfev as f64;
            assert!(
                (0.5..=1.6).contains(&steps) && (0.4..=2.0).contains(&evaluations),
                "{name} {p:?} {:e}: {} steps and {} evaluations against {} and {nfev}",
                run.rtol,
                r.steps,
                r.nfev,
                run.steps
            );
        }
    }
}

#[test]
fn both_stiff_methods_reach_robertsons_problem_at_1e11() {
    // The stiffness benchmark over its whole range, with the Jacobian differenced and analytic:
    // both methods get there, keep the mass y₁ + y₂ + y₃ = 1, and agree with the Test Set's values
    // (which the reference matches to 4·10⁻¹³).
    let t1 = miri_or(0.01, 1e11);
    let a = Matrix::zeros(0, 0);
    let p = Stiff::Robertson;
    let opts = OdeOptions::new().with_tolerances(1e-6, 1e-14);
    let jac = |_t: f64, y: &Matrix<f64>, out: &mut Matrix<f64>| p.jacobian(&a, y, out);
    let f = |_t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>| p.rhs(&a, y, dy);
    let y0 = col(&p.y0());
    for (name, r) in [
        (
            "RadauIia",
            integrate_with(f, &y0, 0.0, t1, RadauIia::new(), &opts),
        ),
        (
            "RadauIia, analytic",
            integrate_with(f, &y0, 0.0, t1, RadauIia::new().with_jacobian(jac), &opts),
        ),
        ("Bdf", integrate_with(f, &y0, 0.0, t1, Bdf::new(), &opts)),
        (
            "Bdf, analytic",
            integrate_with(f, &y0, 0.0, t1, Bdf::new().with_jacobian(jac), &opts),
        ),
    ] {
        let r = r.unwrap_or_else(|| panic!("{name} did not reach the end"));
        let sum: f64 = r.y.as_slice().iter().sum();
        assert!((sum - 1.0).abs() < 1e-12, "{name}: {sum}");
        if t1 == 1e11 {
            let e = stiff_error(r.y.as_slice(), reference(p), 1e-14);
            assert!(e < 1e-4, "{name}: {e:e}, {:?}", r.y);
            assert!(r.steps < 1200, "{name}: {} steps", r.steps);
        }
    }
}

/// The least-squares slope of the global error against the tolerance over five decades,
/// `rtol = tol`, `atol = scale·tol`.
fn tolerance_slope<M: OdeMethod<f64>>(
    f: impl FnMut(f64, &Matrix<f64>, &mut Matrix<f64>) + Copy,
    y0: &[f64],
    t1: f64,
    exact: &[f64],
    scale: f64,
    method: impl Fn() -> M,
) -> f64 {
    let mut points = Vec::new();
    for k in 4..=9 {
        let tolerance = 10f64.powi(-k);
        let opts = OdeOptions::new().with_tolerances(tolerance, tolerance * scale);
        let r = integrate_with(f, &col(y0), 0.0, t1, method(), &opts).unwrap();
        points.push((tolerance, stiff_error(r.y.as_slice(), exact, scale)));
    }
    log_slope(&points)
}

#[test]
fn the_global_error_of_the_stiff_methods_follows_the_tolerance() {
    // Over five decades of the tolerance (1e-4 to 1e-9) the global error falls about as the
    // tolerance does, on a problem that is not stiff (the Brusselator), one that is mildly so
    // (Van der Pol at μ = 10) and one that is very (Robertson to 40).
    if cfg!(miri) {
        return;
    }
    let bruss = Problem::Brusselator;
    let bruss_exact = EXACT.iter().find(|e| e.0 == bruss).unwrap().1;
    let vdp = |_t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>| {
        dy[(0, 0)] = y[(1, 0)];
        dy[(1, 0)] = 10.0 * (1.0 - y[(0, 0)] * y[(0, 0)]) * y[(1, 0)] - y[(0, 0)];
    };
    let a = Matrix::zeros(0, 0);
    let rob = |_t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>| Stiff::Robertson.rhs(&a, y, dy);
    let cases: [(&str, f64, f64); 6] = [
        (
            "RadauIia, Brusselator",
            tolerance_slope(
                bruss.f(),
                &bruss.y0(),
                bruss.t1(),
                bruss_exact,
                1.0,
                RadauIia::new,
            ),
            0.0,
        ),
        (
            "Bdf, Brusselator",
            tolerance_slope(
                bruss.f(),
                &bruss.y0(),
                bruss.t1(),
                bruss_exact,
                1.0,
                Bdf::new,
            ),
            0.0,
        ),
        (
            "RadauIia, Van der Pol",
            tolerance_slope(vdp, &[2.0, 0.0], 20.0, VAN_DER_POL_10, 1.0, RadauIia::new),
            0.0,
        ),
        (
            "Bdf, Van der Pol",
            tolerance_slope(vdp, &[2.0, 0.0], 20.0, VAN_DER_POL_10, 1.0, Bdf::new),
            0.0,
        ),
        (
            "RadauIia, Robertson",
            tolerance_slope(
                rob,
                &[1.0, 0.0, 0.0],
                40.0,
                ROBERTSON_40,
                1e-6,
                RadauIia::new,
            ),
            0.0,
        ),
        (
            "Bdf, Robertson",
            tolerance_slope(rob, &[1.0, 0.0, 0.0], 40.0, ROBERTSON_40, 1e-6, Bdf::new),
            0.0,
        ),
    ];
    for (name, slope, _) in cases {
        eprintln!("{name}: slope {slope:.3}");
        assert!(slope > 0.65 && slope < 1.45, "{name}: slope {slope}");
    }
}

/// The largest error of the dense output against `exact` at the ends of the steps and at nine
/// points inside each, over the steps that start after `from` (a transient before it is left
/// out: its error is that of the steps, not of the interpolation), and the longest step.
fn dense_errors(
    s: &OdeSolution<f64>,
    exact: impl Fn(f64) -> Vec<f64>,
    from: f64,
) -> (f64, f64, f64) {
    let dense = s.dense.as_ref().unwrap();
    let (mut at_ends, mut inside, mut longest) = (0.0f64, 0.0f64, 0.0f64);
    for step in dense.steps().iter().filter(|step| step.t0() >= from) {
        longest = longest.max(step.t1() - step.t0());
        let e = error(step.eval(step.t1()).as_slice(), &exact(step.t1()));
        at_ends = at_ends.max(e);
        for k in 1..10 {
            let t = step.t0() + (step.t1() - step.t0()) * f64::from(k) / 10.0;
            let e = error(step.eval(t).as_slice(), &exact(t));
            inside = inside.max(e);
        }
    }
    (at_ends, inside, longest)
}

#[test]
fn the_dense_output_of_the_stiff_methods_holds_between_the_steps() {
    // y′ = −1000·(y − cos t) − sin t from 2, whose solution is cos t + e⁻¹⁰⁰⁰ᵗ, and the linear
    // system of dimension 50, whose solution is known everywhere. The interpolant of the BDFs
    // through the last k + 1 points is within a small factor of the error at the steps. Radau's
    // collocation polynomial is of the stage order 3 between the steps against 5 at their ends,
    // and on a stiff problem the method takes long steps (up to 0.8 here at 1e-6): its error
    // inside is larger, by 44 and 14 at 1e-6 and 1e-8 here, as with SciPy's Radau (38 and 13).
    let f = |t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>| {
        dy[(0, 0)] = -1000.0 * (y[(0, 0)] - t.cos()) - t.sin();
    };
    let exact = |t: f64| vec![t.cos() + (-1000.0 * t).exp()];
    let t1 = miri_or(0.5, 10.0);
    for tolerance in [1e-6, 1e-8] {
        let opts = tol::<f64>(tolerance);
        for (name, s, factor) in [
            (
                "RadauIia",
                solve_ivp_with(f, &col(&[2.0]), 0.0, t1, RadauIia::new(), &mut [], &opts),
                100.0,
            ),
            (
                "Bdf",
                solve_ivp_with(f, &col(&[2.0]), 0.0, t1, Bdf::new(), &mut [], &opts),
                30.0,
            ),
        ] {
            let (at_ends, inside, longest) = dense_errors(&s, exact, 0.02);
            eprintln!(
                "{name} {tolerance:e}: at the ends {at_ends:.2e}, inside {inside:.2e}, longest \
                 step {longest:.3}"
            );
            assert!(
                inside <= factor * at_ends.max(tolerance),
                "{name} {tolerance:e}: {inside:e} against {at_ends:e}"
            );
        }
    }
    if cfg!(miri) {
        return;
    }
    let a = linear_matrix();
    let p = Stiff::Linear;
    let f = |_t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>| p.rhs(&a, y, dy);
    let jac = |_t: f64, y: &Matrix<f64>, out: &mut Matrix<f64>| p.jacobian(&a, y, out);
    let opts = OdeOptions::new().with_tolerances(1e-7, 1e-12);
    let y0 = col(&p.y0());
    for (name, s) in [
        (
            "RadauIia",
            solve_ivp_with(
                f,
                &y0,
                0.0,
                10.0,
                RadauIia::new().with_jacobian(jac),
                &mut [],
                &opts,
            ),
        ),
        (
            "Bdf",
            solve_ivp_with(
                f,
                &y0,
                0.0,
                10.0,
                Bdf::new().with_jacobian(jac),
                &mut [],
                &opts,
            ),
        ),
    ] {
        let (at_ends, inside, longest) = dense_errors(&s, linear_exact, 1e-3);
        eprintln!(
            "{name}, linear: at the ends {at_ends:.2e}, inside {inside:.2e}, longest step \
             {longest:.3}"
        );
        assert!(
            inside <= 100.0 * at_ends.max(1e-7),
            "{name}: {inside:e} against {at_ends:e}"
        );
        // And continuous across the steps.
        let steps = s.dense.as_ref().unwrap().steps();
        for pair in steps.windows(2) {
            let (a, b) = (pair[0].eval(pair[0].t1()), pair[1].eval(pair[1].t0()));
            for (x, y) in a.as_slice().iter().zip(b.as_slice()) {
                assert!((x - y).abs() <= 1e-13 * (1.0 + x.abs()), "{name}: {x} {y}");
            }
        }
    }
}

#[test]
fn the_stiff_methods_locate_the_events_of_a_relaxation_oscillation() {
    // The zeros of y₀ of Van der Pol's oscillator at μ = 1000 on [0, 3000]: three, each in a
    // jump of the relaxation oscillation, against SciPy's events at 1e-12.
    if cfg!(miri) {
        return;
    }
    let a = Matrix::zeros(0, 0);
    let p = Stiff::VanDerPol;
    let f = |_t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>| p.rhs(&a, y, dy);
    let opts = OdeOptions::new()
        .with_tolerances(1e-8, 1e-12)
        .with_dense_output(false);
    let zero = || Event::new(|_t: f64, y: &Matrix<f64>| y[(0, 0)]);
    let y0 = col(&p.y0());
    for (name, s) in [
        (
            "RadauIia",
            solve_ivp_with(f, &y0, 0.0, 3000.0, RadauIia::new(), &mut [zero()], &opts),
        ),
        (
            "Bdf",
            solve_ivp_with(f, &y0, 0.0, 3000.0, Bdf::new(), &mut [zero()], &opts),
        ),
    ] {
        assert_eq!(s.status, OdeStatus::Finished, "{name}");
        let times: Vec<f64> = s.events.iter().map(|e| e.t).collect();
        assert_eq!(times.len(), VDP_ZEROS.len(), "{name}: {times:?}");
        for (k, (got, want)) in times.iter().zip(VDP_ZEROS).enumerate() {
            eprintln!("{name}: zero {k} at {got}, SciPy {want}");
            assert!(
                (got - want).abs() < 1e-5 * want,
                "{name}: {got} against {want}"
            );
            // Falling first, then rising, then falling: the jumps alternate.
            assert_eq!(s.events[k].rising, k % 2 == 1, "{name}");
        }
    }
    // A terminal event on Robertson's problem: the time the third component reaches one half.
    let p = Stiff::Robertson;
    let f = |_t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>| p.rhs(&a, y, dy);
    let half = || {
        Event::new(|_t: f64, y: &Matrix<f64>| y[(2, 0)] - 0.5)
            .with_direction(EventDirection::Rising)
            .with_terminal(true)
    };
    let opts = OdeOptions::new().with_tolerances(1e-8, 1e-16);
    let y0 = col(&p.y0());
    for (name, s) in [
        (
            "RadauIia",
            solve_ivp_with(f, &y0, 0.0, 1e11, RadauIia::new(), &mut [half()], &opts),
        ),
        (
            "Bdf",
            solve_ivp_with(f, &y0, 0.0, 1e11, Bdf::new(), &mut [half()], &opts),
        ),
    ] {
        assert_eq!(s.status, OdeStatus::Terminated, "{name}");
        assert!(
            (s.t - ROBERTSON_HALF).abs() < 1e-6 * ROBERTSON_HALF,
            "{name}: {} against {ROBERTSON_HALF}",
            s.t
        );
        assert!((s.y[(2, 0)] - 0.5).abs() < 1e-12, "{name}: {}", s.y[(2, 0)]);
        assert_eq!(s.dense.as_ref().unwrap().t1(), s.t);
    }
}

#[test]
fn the_events_of_the_explicit_methods_work_with_the_stiff_ones() {
    // The scenes of events.rs, with RadauIia and Bdf: a body dropped from 10 m stops on the
    // ground at √(20/9.81); the zeros of sin t over ten periods are found in each direction; an
    // event that never happens is never reported.
    let falling = |_t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>| {
        dy[(0, 0)] = y[(1, 0)];
        dy[(1, 0)] = -9.81;
    };
    let ground = || {
        Event::new(|_t: f64, y: &Matrix<f64>| y[(0, 0)])
            .with_direction(EventDirection::Falling)
            .with_terminal(true)
    };
    let impact = (20.0f64 / 9.81).sqrt();
    let opts = tol::<f64>(1e-8);
    let y0 = col(&[10.0, 0.0]);
    for (name, s) in [
        (
            "RadauIia",
            solve_ivp_with(
                falling,
                &y0,
                0.0,
                100.0,
                RadauIia::new(),
                &mut [ground()],
                &opts,
            ),
        ),
        (
            "Bdf",
            solve_ivp_with(falling, &y0, 0.0, 100.0, Bdf::new(), &mut [ground()], &opts),
        ),
    ] {
        assert_eq!(s.status, OdeStatus::Terminated, "{name}");
        assert!((s.t - impact).abs() < 1e-7, "{name}: {}", s.t);
        assert_eq!(s.events.len(), 1, "{name}");
    }
    let oscillator = |_t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>| {
        dy[(0, 0)] = y[(1, 0)];
        dy[(1, 0)] = -y[(0, 0)];
    };
    let periods = miri_or(1, 10);
    let end = (2.0 * f64::from(periods) + 0.5) * core::f64::consts::PI;
    let sine = || Event::new(|_t: f64, y: &Matrix<f64>| y[(0, 0)]);
    let never = || Event::new(|_t: f64, y: &Matrix<f64>| y[(0, 0)] * y[(0, 0)] + 1.0);
    let opts = tol::<f64>(1e-9).with_dense_output(false);
    let y0 = col(&[0.0, 1.0]);
    for (name, s) in [
        (
            "RadauIia",
            solve_ivp_with(
                oscillator,
                &y0,
                0.0,
                end,
                RadauIia::new(),
                &mut [sine(), never()],
                &opts,
            ),
        ),
        (
            "Bdf",
            solve_ivp_with(
                oscillator,
                &y0,
                0.0,
                end,
                Bdf::new(),
                &mut [sine(), never()],
                &opts,
            ),
        ),
    ] {
        // The zero at the start is not an event; those at π, 2π, …, 2·periods·π are, falling
        // and rising in turn.
        assert_eq!(s.events.len(), 2 * periods as usize, "{name}");
        for (k, e) in s.events.iter().enumerate() {
            let exact = core::f64::consts::PI * (k + 1) as f64;
            assert_eq!((e.index, e.rising), (0, k % 2 == 1), "{name}");
            assert!(
                (e.t - exact).abs() < 1e-6,
                "{name}: {} against {exact}",
                e.t
            );
        }
    }
}

#[test]
fn the_stiff_methods_keep_the_energy_of_the_oscillator_as_scipys_do() {
    // The harmonic oscillator through ten periods: Radau IIA loses next to nothing; the BDFs,
    // which damp every oscillation, lose more — as SciPy's codes do, at the same tolerance.
    if cfg!(miri) {
        return;
    }
    let p = Problem::Oscillator;
    let energy = |y: &[f64]| 0.5 * (y[0] * y[0] + y[1] * y[1]);
    for (name, runs, radau) in [
        ("RadauIia", SCIPY_RADAU_NONSTIFF, true),
        ("Bdf", SCIPY_BDF_NONSTIFF, false),
    ] {
        for run in runs.iter().filter(|r| r.problem == p) {
            let y0 = col(&p.y0());
            let opts = tol(run.rtol);
            let r = if radau {
                integrate_with(p.f(), &y0, 0.0, p.t1(), RadauIia::new(), &opts)
            } else {
                integrate_with(p.f(), &y0, 0.0, p.t1(), Bdf::new(), &opts)
            }
            .unwrap();
            let drift = (energy(r.y.as_slice()) - 0.5).abs();
            let scipy = (energy(run.y) - 0.5).abs();
            eprintln!(
                "{name} {:e}: energy drift {drift:.2e}, SciPy {scipy:.2e}",
                run.rtol
            );
            assert!(
                drift <= 3.0 * scipy.max(run.rtol),
                "{name} {:e}: {drift:e} against {scipy:e}",
                run.rtol
            );
        }
    }
}

#[test]
fn an_empty_system_is_integrated_by_the_stiff_methods() {
    let f = |_t: f64, _y: &Matrix<f64>, _dy: &mut Matrix<f64>| {};
    let y0 = Matrix::<f64>::zeros(0, 1);
    let mut events = [Event::new(|t: f64, _y: &Matrix<f64>| t - 0.25).with_terminal(true)];
    let s = solve_ivp_with(
        f,
        &y0,
        0.0,
        1.0,
        RadauIia::new(),
        &mut events,
        &OdeOptions::new(),
    );
    assert_eq!(s.status, OdeStatus::Terminated);
    assert!((s.t - 0.25).abs() < 1e-15);
    let mut events = [Event::new(|t: f64, _y: &Matrix<f64>| t - 0.25).with_terminal(true)];
    let s = solve_ivp_with(
        f,
        &y0,
        0.0,
        1.0,
        Bdf::new(),
        &mut events,
        &OdeOptions::new(),
    );
    assert_eq!(s.status, OdeStatus::Terminated);
    assert!((s.t - 0.25).abs() < 1e-15);
    assert!(s.y.is_empty());
}
