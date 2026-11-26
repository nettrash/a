//! The driver and the step-size control every method shares: the options and their checks, the
//! limits on the step, the end of the interval, the direction of time, and the failures, which
//! are reported rather than passed off as the solution.

use super::problems::{col, miri_or, tol};
use a::math::matrix::Matrix;
use a::math::numerics::{
    Bdf, DenseStep, DormandPrince5, DormandPrince853, Event, OdeFailure, OdeMethod, OdeOptions,
    OdeSolution, OdeStatus, RadauIia, Rk4, integrate, integrate_with, solve, solve_at, solve_ivp,
    solve_ivp_with, solve_with,
};
use std::cell::Cell;

fn decay(_t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>) {
    dy[(0, 0)] = -y[(0, 0)];
}

fn growth(_t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>) {
    dy[(0, 0)] = y[(0, 0)];
}

/// `y′ = y²` from 1, whose solution `1/(1 − t)` leaves the reals at `t = 1`.
fn blow_up(_t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>) {
    dy[(0, 0)] = y[(0, 0)] * y[(0, 0)];
}

/// Runs `body` with each of the methods of the module, by name.
fn each_method(
    mut body: impl FnMut(&str, &dyn Fn(&OdeOptions<f64>, f64, f64) -> OdeSolution<f64>),
) {
    let y0 = col(&[1.0]);
    let five = |o: &OdeOptions<f64>, t0: f64, t1: f64| {
        solve_ivp_with(growth, &y0, t0, t1, DormandPrince5::new(), &mut [], o)
    };
    let eight = |o: &OdeOptions<f64>, t0: f64, t1: f64| {
        solve_ivp_with(growth, &y0, t0, t1, DormandPrince853::new(), &mut [], o)
    };
    let four = |o: &OdeOptions<f64>, t0: f64, t1: f64| {
        solve_ivp_with(growth, &y0, t0, t1, Rk4::new(), &mut [], o)
    };
    let radau = |o: &OdeOptions<f64>, t0: f64, t1: f64| {
        solve_ivp_with(growth, &y0, t0, t1, RadauIia::new(), &mut [], o)
    };
    let bdf = |o: &OdeOptions<f64>, t0: f64, t1: f64| {
        solve_ivp_with(growth, &y0, t0, t1, Bdf::new(), &mut [], o)
    };
    body("DormandPrince5", &five);
    body("DormandPrince853", &eight);
    body("Rk4", &four);
    body("RadauIia", &radau);
    body("Bdf", &bdf);
}

#[test]
fn the_options_are_built_by_their_setters() {
    let opts = OdeOptions::<f64>::new()
        .with_rtol(1e-7)
        .with_atol(1e-11)
        .with_h0(0.01)
        .with_hmax(0.5)
        .with_hmin(1e-9)
        .with_max_steps(77)
        .with_dense_output(false);
    assert_eq!(
        (opts.rtol, opts.atol, opts.h0, opts.hmax, opts.hmin),
        (1e-7, 1e-11, 0.01, 0.5, 1e-9)
    );
    assert_eq!((opts.max_steps, opts.dense_output), (77, false));
    let both = OdeOptions::<f32>::default().with_tolerances(1e-3, 1e-5);
    assert_eq!((both.rtol, both.atol), (1e-3, 1e-5));
    assert_eq!(OdeOptions::<f64>::default(), OdeOptions::new());
}

#[test]
fn options_that_cannot_be_met_are_refused() {
    let y0 = col(&[1.0]);
    let bad = [
        OdeOptions::new().with_rtol(-1e-6),
        OdeOptions::new().with_atol(f64::NAN),
        OdeOptions::new().with_rtol(f64::INFINITY),
        OdeOptions::new().with_tolerances(0.0, 0.0),
        OdeOptions::new().with_h0(-0.1),
        OdeOptions::new().with_h0(f64::NAN),
        OdeOptions::new().with_hmax(0.0),
        OdeOptions::new().with_hmax(f64::NAN),
        OdeOptions::new().with_hmin(-1.0),
        OdeOptions::new().with_hmin(1.0).with_hmax(0.5),
        OdeOptions::new().with_hmin(0.1).with_h0(0.01),
    ];
    for opts in &bad {
        assert!(
            solve_with(decay, &y0, 0.0, 1.0, DormandPrince5::new(), opts).is_none(),
            "{opts:?}"
        );
        let s = solve_ivp_with(decay, &y0, 0.0, 1.0, DormandPrince5::new(), &mut [], opts);
        assert_eq!(
            s.status,
            OdeStatus::Failed(OdeFailure::InvalidInput),
            "{opts:?}"
        );
        // Nothing was evaluated, and what is reported is the initial point.
        assert_eq!((s.nfev, s.steps, s.t), (0, 0, 0.0));
        assert_eq!(s.y, y0);
    }
    // Either tolerance alone may be zero.
    assert!(
        solve_with(
            decay,
            &y0,
            0.0,
            1.0,
            DormandPrince5::new(),
            &OdeOptions::new().with_rtol(0.0)
        )
        .is_some()
    );
    assert!(
        solve_with(
            decay,
            &y0,
            0.0,
            1.0,
            DormandPrince5::new(),
            &OdeOptions::new().with_atol(0.0)
        )
        .is_some()
    );
    // Times and states that are not finite.
    for (t0, t1) in [
        (f64::NAN, 1.0),
        (0.0, f64::INFINITY),
        (f64::NEG_INFINITY, 0.0),
    ] {
        assert!(solve(decay, &y0, t0, t1, DormandPrince5::new()).is_none());
    }
    assert!(solve(decay, &col(&[f64::NAN]), 0.0, 1.0, DormandPrince5::new()).is_none());
    // The fixed form needs a step.
    let s = solve_ivp(decay, &y0, 0.0, 1.0, Rk4::fixed(0), &mut []);
    assert_eq!(s.status, OdeStatus::Failed(OdeFailure::InvalidInput));
    assert_eq!(solve(decay, &y0, 0.0, 0.0, Rk4::fixed(0)), Some(y0.clone()));
}

#[test]
#[should_panic(expected = "the initial state must be a column")]
fn a_state_that_is_not_a_column_panics() {
    let y0 = Matrix::from_rows(&[[1.0, 2.0]]);
    let _ = solve(decay, &y0, 0.0, 1.0, DormandPrince5::new());
}

#[test]
fn an_interval_of_no_length_returns_the_initial_state() {
    let y0 = col(&[3.0, -1.0]);
    let r = integrate(decay2, &y0, 2.0, 2.0, DormandPrince853::new()).unwrap();
    assert_eq!((r.y, r.t, r.steps, r.nfev), (y0.clone(), 2.0, 0, 0));
    let s = solve_ivp(decay2, &y0, 2.0, 2.0, DormandPrince5::new(), &mut []);
    assert_eq!(s.status, OdeStatus::Finished);
    assert_eq!((s.t, s.y.clone()), (2.0, y0));
}

fn decay2(_t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>) {
    dy[(0, 0)] = -y[(0, 0)];
    dy[(1, 0)] = -2.0 * y[(1, 0)];
}

#[test]
fn every_method_lands_on_the_end_exactly_and_runs_backwards() {
    let (tolerance, bound) = miri_or((1e-5, 1e-3), (1e-9, 1e-7));
    each_method(|name, solve| {
        for (t0, t1) in [(0.0, 1.0), (0.0, -1.0), (1.0, 0.3), (-2.0, 7.0)] {
            let s = solve(&tol(tolerance), t0, t1);
            assert_eq!(s.status, OdeStatus::Finished, "{name} {t0} → {t1}");
            assert_eq!(s.t, t1, "{name}");
            assert_eq!(s.dense.as_ref().unwrap().t1(), t1, "{name}");
            let want = (t1 - t0).exp();
            assert!(
                (s.y[(0, 0)] - want).abs() < bound * want,
                "{name} {t0} → {t1}: {}",
                s.y[(0, 0)]
            );
        }
        // Far from zero the clock's rounding does not leave the end unreached or overshot.
        let s = solve(&tol(tolerance), 1e6, 1e6 + 3.0);
        assert_eq!((s.status, s.t), (OdeStatus::Finished, 1e6 + 3.0), "{name}");
        let want = 3.0f64.exp();
        assert!(
            (s.y[(0, 0)] - want).abs() < bound * want,
            "{name}: {}",
            s.y[(0, 0)]
        );
    });
}

#[test]
fn no_step_is_longer_than_hmax_and_the_first_is_h0() {
    each_method(|name, solve| {
        let opts = tol::<f64>(1e-6).with_hmax(0.0937).with_h0(0.001);
        let s = solve(&opts, 0.0, 3.0);
        let steps = s.dense.as_ref().unwrap().steps();
        // The step the clock makes, t + h rounded, may exceed hmax by that rounding.
        for step in steps {
            let h = step.t1() - step.t0();
            assert!(h <= 0.0937 + 4.0 * f64::EPSILON * 3.0, "{name}: {h}");
        }
        // The first step is h0, unless it was rejected — and from y′ = y at 1e-6 it is not. The
        // classical method's first step is the first half of a pair of size h0.
        let first = steps[0].t1() - steps[0].t0();
        let want = if name == "Rk4" { 0.0005 } else { 0.001 };
        assert!((first - want).abs() < 1e-15, "{name}: {first}");
        assert!(s.steps >= 32, "{name}");
    });
    // The last step is not stretched past hmax to reach the end: what is left after 0.5 + 0.5
    // + 0.5 is 0.502, which becomes two steps of 0.251.
    let opts = OdeOptions::new()
        .with_tolerances(1e10, 1e10)
        .with_h0(0.5)
        .with_hmax(0.5);
    let s = solve_ivp_with(
        decay,
        &col(&[1.0]),
        0.0,
        2.002,
        DormandPrince5::new(),
        &mut [],
        &opts,
    );
    let lengths: Vec<f64> = s
        .dense
        .as_ref()
        .unwrap()
        .steps()
        .iter()
        .map(|s| s.t1() - s.t0())
        .collect();
    assert_eq!(lengths.len(), 5, "{lengths:?}");
    assert!(lengths.iter().all(|&h| h <= 0.5));
    assert!(
        (lengths[3] - 0.251).abs() < 1e-12 && (lengths[4] - 0.251).abs() < 1e-12,
        "{lengths:?}"
    );
}

#[test]
fn a_solution_that_leaves_the_reals_is_not_passed_off_as_the_answer() {
    // y′ = y² from 1 cannot be followed past t = 1.
    let y0 = col(&[1.0]);
    let mut runs = vec![(
        "DormandPrince5",
        solve_ivp(blow_up, &y0, 0.0, 2.0, DormandPrince5::new(), &mut []),
    )];
    // Under Miri one method is all the time allows.
    if !cfg!(miri) {
        runs.push((
            "DormandPrince853",
            solve_ivp(blow_up, &y0, 0.0, 2.0, DormandPrince853::new(), &mut []),
        ));
        runs.push((
            "Rk4",
            solve_ivp(blow_up, &y0, 0.0, 2.0, Rk4::new(), &mut []),
        ));
        runs.push((
            "RadauIia",
            solve_ivp(blow_up, &y0, 0.0, 2.0, RadauIia::new(), &mut []),
        ));
        runs.push((
            "Bdf",
            solve_ivp(blow_up, &y0, 0.0, 2.0, Bdf::new(), &mut []),
        ));
    }
    for (name, s) in runs {
        assert!(
            matches!(
                s.status,
                OdeStatus::Failed(OdeFailure::StepTooSmall | OdeFailure::ToleranceTooSmall)
            ),
            "{name}: {:?}",
            s.status
        );
        // What it reached is where its numerical solution ends — the pole moved by the error of
        // the integration, which for the BDFs at this tolerance is some ten times larger — and it
        // is far up the pole.
        let moved = if name == "Bdf" { 1e-4 } else { 1e-5 };
        assert!((s.t - 1.0).abs() < moved, "{name}: {}", s.t);
        assert!(s.y[(0, 0)] > 1e4, "{name}: {}", s.y[(0, 0)]);
        let dense = s.dense.as_ref().unwrap();
        let half = dense.eval(0.5).unwrap()[(0, 0)];
        assert!((half - 2.0).abs() < 2.0 * moved, "{name}: {half}");
    }
    if !cfg!(miri) {
        assert!(solve(blow_up, &y0, 0.0, 2.0, DormandPrince5::new()).is_none());
        assert!(integrate(blow_up, &y0, 0.0, 2.0, DormandPrince853::new()).is_none());
        assert!(solve_at(blow_up, &y0, 0.0, &[0.5, 2.0], DormandPrince5::new()).is_none());
    }
    // A floor on the step ends it sooner.
    let opts = OdeOptions::new().with_hmin(1e-3);
    let s = solve_ivp_with(
        blow_up,
        &y0,
        0.0,
        2.0,
        DormandPrince5::new(),
        &mut [],
        &opts,
    );
    assert_eq!(s.status, OdeStatus::Failed(OdeFailure::StepTooSmall));
    assert!(s.t < 1.0 && s.t > 0.9, "{}", s.t);
}

#[test]
fn the_step_limit_is_reported() {
    let y0 = col(&[1.0]);
    let opts = tol::<f64>(1e-10).with_max_steps(10);
    let s = solve_ivp_with(decay, &y0, 0.0, 10.0, DormandPrince5::new(), &mut [], &opts);
    assert_eq!(s.status, OdeStatus::Failed(OdeFailure::MaxSteps));
    assert_eq!(s.steps, 10);
    assert!(s.t > 0.0 && s.t < 10.0);
    assert!((s.y[(0, 0)] - (-s.t).exp()).abs() < 1e-9);
    assert!(integrate_with(decay, &y0, 0.0, 10.0, DormandPrince5::new(), &opts).is_none());
    // The fixed form needs as many steps as it has.
    let s = solve_ivp_with(decay, &y0, 0.0, 1.0, Rk4::fixed(11), &mut [], &opts);
    assert_eq!(s.status, OdeStatus::Failed(OdeFailure::MaxSteps));
    assert!(solve_with(decay, &y0, 0.0, 1.0, Rk4::fixed(10), &opts).is_some());
}

#[test]
fn a_tolerance_below_the_rounding_of_the_state_is_reported() {
    let y0 = col(&[1.0]);
    for opts in [
        OdeOptions::new().with_tolerances(1e-18, 1e-20),
        OdeOptions::new().with_tolerances(0.0, 1e-25),
    ] {
        for s in [
            solve_ivp_with(decay, &y0, 0.0, 1.0, DormandPrince5::new(), &mut [], &opts),
            solve_ivp_with(
                decay,
                &y0,
                0.0,
                1.0,
                DormandPrince853::new(),
                &mut [],
                &opts,
            ),
            solve_ivp_with(decay, &y0, 0.0, 1.0, Rk4::new(), &mut [], &opts),
            solve_ivp_with(decay, &y0, 0.0, 1.0, RadauIia::new(), &mut [], &opts),
            solve_ivp_with(decay, &y0, 0.0, 1.0, Bdf::new(), &mut [], &opts),
        ] {
            assert_eq!(s.status, OdeStatus::Failed(OdeFailure::ToleranceTooSmall));
        }
    }
    // While one a few times the rounding is met.
    let s = solve_ivp_with(
        decay,
        &y0,
        0.0,
        1.0,
        DormandPrince853::new(),
        &mut [],
        &tol(1e-15),
    );
    assert_eq!(s.status, OdeStatus::Finished);
    assert!((s.y[(0, 0)] - (-1.0f64).exp()).abs() < 1e-14);
}

#[test]
fn a_right_hand_side_that_is_not_finite_rejects_the_step() {
    // y′ = −y, but f is not a number below zero, which a first step far too long probes: the
    // step is rejected and retried shorter, and the integration goes on.
    let f = |_t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>| {
        dy[(0, 0)] = if y[(0, 0)] < 0.0 {
            f64::NAN
        } else {
            -y[(0, 0)]
        };
    };
    let y0 = col(&[1.0]);
    let tolerance = miri_or(1e-6, 1e-8);
    let opts = tol::<f64>(tolerance).with_h0(8.0);
    each_method_on(f, &y0, 0.0, 10.0, &opts, |name, s| {
        assert_eq!(s.status, OdeStatus::Finished, "{name}");
        assert!(s.rejected > 0, "{name}");
        assert!((s.y[(0, 0)] - (-10.0f64).exp()).abs() < tolerance, "{name}");
    });
    // Not finite where the integration starts: nothing can be done.
    let nan = |_t: f64, _y: &Matrix<f64>, dy: &mut Matrix<f64>| dy[(0, 0)] = f64::NAN;
    each_method_on(nan, &y0, 0.0, 1.0, &OdeOptions::new(), |name, s| {
        assert_eq!(s.status, OdeStatus::Failed(OdeFailure::NonFinite), "{name}");
        assert_eq!(s.t, 0.0);
    });
    // Not finite past t = ½ whatever the state: no step gets past it.
    let wall = |t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>| {
        dy[(0, 0)] = if t > 0.5 { f64::INFINITY } else { -y[(0, 0)] };
    };
    each_method_on(wall, &y0, 0.0, 1.0, &OdeOptions::new(), |name, s| {
        assert_eq!(
            s.status,
            OdeStatus::Failed(OdeFailure::StepTooSmall),
            "{name}"
        );
        assert!(s.t <= 0.5 && s.t > 0.49, "{name}: {}", s.t);
    });
    // In the fixed form a step cannot be shrunk, so the first one that is not finite fails: the
    // step that ends on ½ evaluates f there and no further, the next one fails.
    let s = solve_ivp(wall, &y0, 0.0, 1.0, Rk4::fixed(10), &mut []);
    assert_eq!(s.status, OdeStatus::Failed(OdeFailure::NonFinite));
    assert_eq!(s.t, 0.5);
}

fn each_method_on(
    f: impl Fn(f64, &Matrix<f64>, &mut Matrix<f64>) + Copy,
    y0: &Matrix<f64>,
    t0: f64,
    t1: f64,
    opts: &OdeOptions<f64>,
    mut check: impl FnMut(&str, OdeSolution<f64>),
) {
    check(
        "DormandPrince5",
        solve_ivp_with(f, y0, t0, t1, DormandPrince5::new(), &mut [], opts),
    );
    check(
        "DormandPrince853",
        solve_ivp_with(f, y0, t0, t1, DormandPrince853::new(), &mut [], opts),
    );
    check(
        "Rk4",
        solve_ivp_with(f, y0, t0, t1, Rk4::new(), &mut [], opts),
    );
    check(
        "RadauIia",
        solve_ivp_with(f, y0, t0, t1, RadauIia::new(), &mut [], opts),
    );
    check(
        "Bdf",
        solve_ivp_with(f, y0, t0, t1, Bdf::new(), &mut [], opts),
    );
}

#[test]
fn a_method_driven_by_hand_follows_the_trait() {
    let mut f = decay;
    let y0 = col(&[1.0]);
    let mut method = DormandPrince853::new();
    // Not started: nothing to step.
    assert_eq!(method.step(&mut f), Err(OdeFailure::InvalidInput));
    for _ in 0..2 {
        // A method can be started again for another integration.
        method
            .start(&mut f, 0.0, &y0, 2.0, &OdeOptions::new())
            .unwrap();
        assert_eq!((method.t(), method.t_old()), (0.0, 0.0));
        let mut step = DenseStep::new();
        let mut steps = 0;
        while method.t() != 2.0 {
            let before = method.t();
            method.step(&mut f).unwrap();
            assert_eq!(method.t_old(), before);
            assert!(method.t() > before && method.t() <= 2.0);
            method.dense(&mut f, &mut step).unwrap();
            assert_eq!((step.t0(), step.t1()), (method.t_old(), method.t()));
            steps += 1;
        }
        assert!(steps > 3);
        assert!((method.y()[(0, 0)] - (-2.0f64).exp()).abs() < 1e-6);
        // At the end a step does nothing.
        method.step(&mut f).unwrap();
        assert_eq!(method.t(), 2.0);
    }
    assert_eq!(method.order(), 8);
    assert_eq!((method.jacobians(), method.factorisations()), (0, 0));
}

#[test]
fn an_interval_shorter_than_the_smallest_step_is_one_last_step() {
    // [1, 1 + 4ε], a microsecond at t = 1.7·10⁹ (four units in the last place there), and
    // intervals shorter than hmin, forwards and backwards: what is left of the interval is the
    // last step, taken whatever its size, so each method ends on t₁ at once, with the solution
    // there. (Such intervals failed with StepTooSmall while the step was tested against the
    // floor before being cut to what is left.) Rk4::new() takes its pair as two steps. The
    // intervals shorter than hmin are given tolerances one step of 0.05 can meet — the BDFs start
    // at order 1 — since a step that fails them cannot be shortened below hmin.
    let y0 = col(&[1.0]);
    let loose = OdeOptions::new().with_tolerances(1e-2, 1e-2).with_hmin(0.1);
    for (t0, t1, opts, bound) in [
        (1.0, 1.0 + 4.0 * f64::EPSILON, OdeOptions::new(), 1e-15),
        (1.7e9, 1.7e9 + 1e-6, OdeOptions::new(), 1e-12),
        (0.0, 0.05, loose, 1e-2),
        (0.0, -0.05, loose, 1e-2),
    ] {
        each_method_on(decay, &y0, t0, t1, &opts, |name, s| {
            assert_eq!(s.status, OdeStatus::Finished, "{name} on [{t0}, {t1}]");
            assert_eq!(s.t, t1, "{name} on [{t0}, {t1}]");
            let steps = if name == "Rk4" { 2 } else { 1 };
            assert_eq!(s.steps, steps, "{name} on [{t0}, {t1}]");
            let want = (t0 - t1).exp();
            assert!(
                (s.y[(0, 0)] - want).abs() < bound,
                "{name} on [{t0}, {t1}]: {} against {want}",
                s.y[(0, 0)]
            );
        });
    }
    // A method that cannot take what is left is still stopped: once rejected, its steps shrink
    // and no longer reach t₁, and the floor applies to them. Here f is not a number past the
    // middle of an interval of four units in the last place.
    let wall = |t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>| {
        dy[(0, 0)] = if t > 1.0 + 2.0 * f64::EPSILON {
            f64::NAN
        } else {
            -y[(0, 0)]
        };
    };
    let (t0, t1) = (1.0, 1.0 + 4.0 * f64::EPSILON);
    each_method_on(wall, &y0, t0, t1, &OdeOptions::new(), |name, s| {
        assert_eq!(
            s.status,
            OdeStatus::Failed(OdeFailure::StepTooSmall),
            "{name}"
        );
        assert_eq!((s.t, s.steps), (t0, 0), "{name}");
    });
}

#[test]
fn no_step_is_retried_for_ever_when_what_is_left_is_just_above_hmin() {
    // [0, 0.1005] with hmin = 0.1: what is left lies within the 1% the last step is stretched
    // by, so a first attempt is the whole interval. When it is rejected, hmin itself is tried —
    // a shorter step, not stretched back to the end — and when that fails too the integration
    // ends with StepTooSmall. (Each rejection used to shrink the step to hmin and the stretch
    // made it the whole interval again, for ever: the counter below stopped that loop for four
    // methods of five on tolerances no step of 0.1 meets, and for all five on the wall.)
    let calls = Cell::new(0u32);
    let count = |t: f64| {
        calls.set(calls.get() + 1);
        assert!(
            calls.get() < 100_000,
            "still evaluating at t = {t} after 100000 evaluations"
        );
    };
    let decay = |t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>| {
        count(t);
        dy[(0, 0)] = -y[(0, 0)];
    };
    let wall = |t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>| {
        count(t);
        dy[(0, 0)] = if t > 0.05 { f64::NAN } else { -y[(0, 0)] };
    };
    let y0 = col(&[1.0]);
    let t1 = 0.1005;
    let tight = OdeOptions::new()
        .with_tolerances(1e-12, 1e-14)
        .with_hmin(0.1);
    each_method_on(decay, &y0, 0.0, t1, &tight, |name, s| {
        if name == "DormandPrince853" {
            // Its one step of 0.1005 meets even these tolerances.
            assert_eq!((s.status, s.t, s.steps), (OdeStatus::Finished, t1, 1));
            assert!((s.y[(0, 0)] - (-t1).exp()).abs() < 1e-13, "{}", s.y[(0, 0)]);
        } else {
            let failed = OdeStatus::Failed(OdeFailure::StepTooSmall);
            // The whole interval and hmin, each tried once.
            assert_eq!(
                (s.status, s.t, s.steps, s.rejected),
                (failed, 0.0, 0, 2),
                "{name}"
            );
        }
        calls.set(0);
    });
    let opts = OdeOptions::new().with_hmin(0.1);
    each_method_on(wall, &y0, 0.0, t1, &opts, |name, s| {
        let failed = OdeStatus::Failed(OdeFailure::StepTooSmall);
        assert_eq!(
            (s.status, s.t, s.steps, s.rejected),
            (failed, 0.0, 0, 2),
            "{name}"
        );
        calls.set(0);
    });
    // And when hmin succeeds where the whole interval failed — here f is not finite only the
    // first time it is asked past t = 0.05 — the integration goes on: a step of hmin, then what
    // is left, 0.0005, as the last step.
    let spent = Cell::new(false);
    let once = |t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>| {
        count(t);
        dy[(0, 0)] = if t > 0.05 && !spent.replace(true) {
            f64::NAN
        } else {
            -y[(0, 0)]
        };
    };
    let loose = OdeOptions::new().with_tolerances(1e-2, 1e-2).with_hmin(0.1);
    each_method_on(once, &y0, 0.0, t1, &loose, |name, s| {
        assert_eq!(
            (s.status, s.t, s.rejected),
            (OdeStatus::Finished, t1, 1),
            "{name}"
        );
        // The classical method takes each of the two as a pair of half-steps.
        let lengths: Vec<f64> = s
            .dense
            .as_ref()
            .unwrap()
            .steps()
            .iter()
            .map(|p| p.t1() - p.t0())
            .collect();
        let want: &[f64] = if name == "Rk4" {
            &[0.05, 0.05, 0.00025, 0.00025]
        } else {
            &[0.1, 0.0005]
        };
        assert_eq!(lengths.len(), want.len(), "{name}: {lengths:?}");
        for (h, w) in lengths.iter().zip(want) {
            assert!((h - w).abs() < 1e-15, "{name}: {lengths:?}");
        }
        assert!((s.y[(0, 0)] - (-t1).exp()).abs() < 1e-2, "{name}");
        calls.set(0);
        spent.set(false);
    });
}

#[test]
fn an_interval_shorter_than_the_floor_is_tried_as_one_step() {
    // At t = 10¹⁵ the floor 10·ε·|t| is 2.2 (a unit in the last place is 0.125 there), so
    // [10¹⁵, 10¹⁵ + 0.25] is shorter than the smallest step that may be proposed there, but for
    // the last. Hairer's first step, about 0.025, is raised to
    // that floor, which reaches t₁: the interval is one step, the last, taken whatever its size.
    // (It was refused against the floor before it could reach the end, by every method.) The
    // explicit methods meet the tolerances with it; the stiff ones, whose estimates are larger
    // for so long a step, reject it once and then fail, since no shorter step may be tried.
    let y0 = col(&[1.0]);
    let (t0, t1) = (1e15, 1e15 + 0.25);
    each_method_on(decay, &y0, t0, t1, &OdeOptions::new(), |name, s| {
        if name == "RadauIia" || name == "Bdf" {
            let failed = OdeStatus::Failed(OdeFailure::StepTooSmall);
            assert_eq!(
                (s.status, s.t, s.steps, s.rejected),
                (failed, t0, 0, 1),
                "{name}"
            );
        } else {
            assert_eq!(
                (s.status, s.t, s.rejected),
                (OdeStatus::Finished, t1, 0),
                "{name}"
            );
            let steps = if name == "Rk4" { 2 } else { 1 };
            assert_eq!(s.steps, steps, "{name}");
            let want = (-0.25f64).exp();
            assert!((s.y[(0, 0)] - want).abs() < 1e-6, "{name}: {}", s.y[(0, 0)]);
        }
    });
    // The classical method's estimate needs a pair of half-steps, and so a midpoint: an interval
    // of one unit in the last place of t has none, and its one step is rejected and the
    // integration refused. Two units are enough, and the fixed form needs no midpoint.
    let one = 1.0 + f64::EPSILON;
    let s = solve_ivp(decay, &y0, 1.0, one, Rk4::new(), &mut []);
    let failed = OdeStatus::Failed(OdeFailure::StepTooSmall);
    assert_eq!((s.status, s.t, s.steps, s.rejected), (failed, 1.0, 0, 1));
    let two = 1.0 + 2.0 * f64::EPSILON;
    let s = solve_ivp(decay, &y0, 1.0, two, Rk4::new(), &mut []);
    assert_eq!((s.status, s.t, s.steps), (OdeStatus::Finished, two, 2));
    let s = solve_ivp(decay, &y0, 1.0, one, Rk4::fixed(1), &mut []);
    assert_eq!((s.status, s.t, s.steps), (OdeStatus::Finished, one, 1));
}

#[test]
fn a_proposal_below_hmin_is_raised_to_it() {
    // y′ = −50·(y − cos t) from 0 with hmin = 10⁻³: after its first step RadauIia's controller
    // asks for less than hmin. The step is raised to hmin, as LSODE does, and accepted under the
    // error test as any other, so the integration goes on (it ended with StepTooSmall at t = 10⁻³
    // while such a proposal was refused). Every step but the last is at least hmin.
    let forced = |t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>| {
        dy[(0, 0)] = -50.0 * (y[(0, 0)] - t.cos());
    };
    let exact = |t: f64| (2500.0 * t.cos() + 50.0 * t.sin() - 2500.0 * (-50.0 * t).exp()) / 2501.0;
    let y0 = col(&[0.0]);
    let hmin = 1e-3;
    let opts = OdeOptions::new().with_hmin(hmin);
    let s = solve_ivp_with(forced, &y0, 0.0, 1.001, RadauIia::new(), &mut [], &opts);
    assert_eq!((s.status, s.t), (OdeStatus::Finished, 1.001));
    let steps = s.dense.as_ref().unwrap().steps();
    for step in &steps[..steps.len() - 1] {
        let h = step.t1() - step.t0();
        assert!(h >= hmin * (1.0 - 1e-12), "{h}");
    }
    assert!((s.y[(0, 0)] - exact(1.001)).abs() < 1e-5, "{}", s.y[(0, 0)]);
    // A step of hmin that fails still ends the integration: the BDFs start at order 1, whose
    // step of 10⁻³ errs by about 10⁻³ against a tolerance of 10⁻⁹ here. It is tried once.
    let s = solve_ivp_with(forced, &y0, 0.0, 1.001, Bdf::new(), &mut [], &opts);
    let failed = OdeStatus::Failed(OdeFailure::StepTooSmall);
    assert_eq!((s.status, s.t, s.steps, s.rejected), (failed, 0.0, 0, 1));
    // When what is left is too long to stretch the last step over but half of it is below hmin
    // (hmin = hmax = 0.01, [0, 0.0101]), it is covered by a step of hmax and what remains, not
    // by two halves below hmin. (Tolerances a step of 0.01 meets at order 1.)
    let opts = OdeOptions::new()
        .with_tolerances(1e-3, 1e-3)
        .with_hmin(0.01)
        .with_hmax(0.01);
    each_method_on(decay, &col(&[1.0]), 0.0, 0.0101, &opts, |name, s| {
        assert_eq!((s.status, s.t), (OdeStatus::Finished, 0.0101), "{name}");
        let lengths: Vec<f64> = s
            .dense
            .as_ref()
            .unwrap()
            .steps()
            .iter()
            .map(|p| p.t1() - p.t0())
            .collect();
        // The classical method's steps are the halves of its pairs, which hmin and hmax bound.
        let want: &[f64] = if name == "Rk4" {
            &[0.005, 0.005, 5e-5, 5e-5]
        } else {
            &[0.01, 1e-4]
        };
        assert_eq!(lengths.len(), want.len(), "{name}: {lengths:?}");
        for (h, w) in lengths.iter().zip(want) {
            assert!((h - w).abs() < 1e-15, "{name}: {lengths:?}");
        }
    });
}

#[test]
fn a_zero_absolute_tolerance_leaves_the_stiff_methods_nothing_to_measure_a_zero_against() {
    // With atol = 0 a component that is exactly zero has a scale of nought in the norm of the
    // Newton iteration — at the start of the step for RadauIia, in the prediction for Bdf — and
    // no correction to it can meet that: the documented limit, reported as a failure, never as
    // a result. The explicit methods measure their error against max(|yₙ|, |yₙ₊₁|) and finish.
    let opts = OdeOptions::new().with_tolerances(1e-6, 0.0);
    let oscillator = |_t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>| {
        dy[(0, 0)] = y[(1, 0)];
        dy[(1, 0)] = -y[(0, 0)];
    };
    let ramp = |t: f64, _y: &Matrix<f64>, dy: &mut Matrix<f64>| dy[(0, 0)] = t;
    let (t1, capped) = (miri_or(1.0, 10.0), opts.with_max_steps(miri_or(20, 1000)));
    each_method_on(oscillator, &col(&[1.0, 0.0]), 0.0, t1, &opts, |name, s| {
        if name == "RadauIia" {
            let failed = OdeStatus::Failed(OdeFailure::StepTooSmall);
            assert_eq!((s.status, s.t, s.steps), (failed, 0.0, 0), "{name}");
        } else {
            // Bdf's prediction of the second component, −h, is not zero.
            assert_eq!(s.status, OdeStatus::Finished, "{name}");
            assert!(
                (s.y[(0, 0)] - t1.cos()).abs() < 1e-4,
                "{name}: {}",
                s.y[(0, 0)]
            );
        }
    });
    each_method_on(ramp, &col(&[0.0]), 0.0, 1.0, &capped, |name, s| {
        if name == "RadauIia" || name == "Bdf" {
            // Steps shrink until the corrections underflow to zero, and the state with them.
            let failed = OdeStatus::Failed(OdeFailure::MaxSteps);
            assert_eq!((s.status, s.y[(0, 0)]), (failed, 0.0), "{name}");
            assert!(s.t < 1e-100, "{name}: {}", s.t);
        } else {
            assert_eq!(s.status, OdeStatus::Finished, "{name}");
            assert!((s.y[(0, 0)] - 0.5).abs() < 1e-6, "{name}: {}", s.y[(0, 0)]);
        }
    });
}

#[test]
fn no_method_writes_a_continuous_extension_before_it_is_started() {
    // The trait asks for dense() only after a step. A call before the method was ever started is
    // refused as step() is, rather than reading the stages of a step that was never taken
    // (DormandPrince853 indexed its empty stages and panicked).
    let mut f = decay;
    let mut out = DenseStep::new();
    let refused = Err(OdeFailure::InvalidInput);
    assert_eq!(Rk4::<f64>::new().dense(&mut f, &mut out), refused);
    assert_eq!(Rk4::<f64>::fixed(4).dense(&mut f, &mut out), refused);
    assert_eq!(
        DormandPrince5::<f64>::new().dense(&mut f, &mut out),
        refused
    );
    assert_eq!(
        DormandPrince853::<f64>::new().dense(&mut f, &mut out),
        refused
    );
    assert_eq!(RadauIia::<f64>::new().dense(&mut f, &mut out), refused);
    assert_eq!(Bdf::<f64>::new().dense(&mut f, &mut out), refused);
}

#[test]
fn an_empty_system_has_its_events_all_the_same() {
    // No unknowns, and an event of the time alone.
    let f = |_t: f64, _y: &Matrix<f64>, _dy: &mut Matrix<f64>| {};
    let y0 = Matrix::<f64>::zeros(0, 1);
    let mut events = [Event::new(|t: f64, _y: &Matrix<f64>| t - 0.25).with_terminal(true)];
    let s = solve_ivp(f, &y0, 0.0, 1.0, DormandPrince5::new(), &mut events);
    assert_eq!(s.status, OdeStatus::Terminated);
    assert!((s.t - 0.25).abs() < 1e-15);
    assert!(s.y.is_empty());
    assert_eq!(solve(f, &y0, 0.0, 1.0, DormandPrince853::new()), Some(y0));
}

#[test]
fn a_zero_right_hand_side_keeps_the_state_in_a_few_steps() {
    // Nothing moves: every error estimate is exactly zero, so the steps grow by the most each
    // controller allows from Hairer's first guess of 1e-6 — ten-fold, six-fold, five-fold — and
    // the state is never touched. The classical method's pairs, five-fold, need eleven (22
    // steps) to reach 10.
    let f = |_t: f64, _y: &Matrix<f64>, dy: &mut Matrix<f64>| {
        dy[(0, 0)] = 0.0;
        dy[(1, 0)] = 0.0;
    };
    let y0 = col(&[3.0, -0.5]);
    each_method_on(f, &y0, 0.0, 10.0, &OdeOptions::new(), |name, s| {
        assert_eq!(s.status, OdeStatus::Finished, "{name}");
        assert_eq!(s.y, y0, "{name}");
        assert!(
            s.steps <= 22 && s.rejected == 0,
            "{name}: {} steps",
            s.steps
        );
        assert_eq!(s.dense.as_ref().unwrap().eval(4.2).unwrap(), y0, "{name}");
    });
}
