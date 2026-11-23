//! `Bdf`, the variable-order backward differentiation formulas as in `ode15s`: the order of each
//! formula, the numerical differentiation formulas against the backward differentiation formulas
//! themselves, the choice of the order, the cap on it, and the reuse of the Jacobian. The
//! comparisons with SciPy, the dense output and the events of both stiff methods are in
//! `stiff.rs`.

use super::problems::{Problem, col, error, log_slope, miri_or, tol};
use super::reference::EXACT;
use super::stiff_problems::stiff_error;
use super::stiff_reference::ROBERTSON_40;
use a::math::matrix::Matrix;
use a::math::numerics::{
    Bdf, OdeFailure, OdeMethod, OdeOptions, OdeStatus, integrate_with, solve_ivp_with,
};

/// Robertson's chemical kinetics.
fn robertson(_t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>) {
    let (a, b, c) = (y[(0, 0)], y[(1, 0)], y[(2, 0)]);
    dy[(0, 0)] = -0.04 * a + 1e4 * b * c;
    dy[(1, 0)] = 0.04 * a - 1e4 * b * c - 3e7 * b * b;
    dy[(2, 0)] = 3e7 * b * b;
}

/// The error and the number of steps of `method` on Robertson's problem to 40 at `rtol = tol`,
/// `atol = 10⁻⁶·tol`, relative to each component of the reference.
fn on_robertson<M: OdeMethod<f64>>(tolerance: f64, method: M) -> (usize, f64) {
    let opts = OdeOptions::new().with_tolerances(tolerance, tolerance * 1e-6);
    let r = integrate_with(robertson, &col(&[1.0, 0.0, 0.0]), 0.0, 40.0, method, &opts).unwrap();
    (r.steps, stiff_error(r.y.as_slice(), ROBERTSON_40, 0.0))
}

/// The same on the Brusselator to 20 at `rtol = atol = tol`, against mpmath's solution.
fn on_brusselator<M: OdeMethod<f64>>(tolerance: f64, method: M) -> (usize, f64) {
    let p = Problem::Brusselator;
    let r = integrate_with(p.f(), &col(&p.y0()), 0.0, p.t1(), method, &tol(tolerance)).unwrap();
    let exact = EXACT.iter().find(|e| e.0 == p).unwrap().1;
    (r.steps, error(r.y.as_slice(), exact))
}

#[test]
fn each_formula_has_its_order() {
    // With the order held to k, the global error falls as the k-th power of the number of steps,
    // over tolerances at which that order is the one in use: the formulas of every order, the
    // NDFs and the BDFs, are of the order they claim. (The adaptive steps make it a least-squares
    // slope through a cloud, and the startup at the lower orders steepens it a little.)
    if cfg!(miri) {
        return;
    }
    let ranges: [&[f64]; 5] = [
        &[1e-3, 1e-4, 1e-5],
        &[1e-4, 1e-5, 1e-6, 1e-7],
        &[1e-5, 1e-6, 1e-7, 1e-8],
        &[1e-6, 1e-7, 1e-8, 1e-9, 1e-10],
        &[1e-7, 1e-8, 1e-9, 1e-10, 1e-11],
    ];
    for (k, range) in (1..=5).zip(ranges) {
        for ndf in [true, false] {
            let method = || Bdf::new().with_ndf(ndf).with_max_order(k);
            for (name, points) in [
                (
                    "Robertson",
                    range
                        .iter()
                        .map(|&t| on_robertson(t, method()))
                        .collect::<Vec<_>>(),
                ),
                (
                    "Brusselator",
                    range.iter().map(|&t| on_brusselator(t, method())).collect(),
                ),
            ] {
                let points: Vec<(f64, f64)> = points.iter().map(|&(n, e)| (n as f64, e)).collect();
                let slope = -log_slope(&points);
                let kf = k as f64;
                assert!(
                    slope > kf - 0.35 && slope < kf + 0.9,
                    "order {k}, ndf {ndf}, {name}: slope {slope}, {points:?}"
                );
            }
        }
    }
}

#[test]
fn the_numerical_differentiation_formulas_take_fewer_steps() {
    // Shampine and Reichelt chose the κ of the NDFs to allow a step some 26% longer at the same
    // error at orders 1 to 3, 12% at order 4. At the same tolerance the NDFs take fewer steps,
    // and are no less accurate.
    if cfg!(miri) {
        return;
    }
    for (k, tolerance) in [(1, 1e-5), (2, 1e-6), (3, 1e-6), (4, 1e-6)] {
        let (ndf_steps, ndf_err) = on_robertson(tolerance, Bdf::new().with_max_order(k));
        let (bdf_steps, bdf_err) =
            on_robertson(tolerance, Bdf::new().with_ndf(false).with_max_order(k));
        assert!(
            ndf_steps < bdf_steps && ndf_err < 1.5 * bdf_err,
            "order {k}: NDF {ndf_steps} steps, {ndf_err:e}; BDF {bdf_steps}, {bdf_err:e}"
        );
    }
}

/// The orders `method` used, step by step, on `f` from `y0` to `t1`.
fn orders(
    mut f: impl FnMut(f64, &Matrix<f64>, &mut Matrix<f64>),
    y0: &[f64],
    t1: f64,
    opts: &OdeOptions<f64>,
    mut method: impl OdeMethod<f64>,
) -> Vec<usize> {
    method.start(&mut f, 0.0, &col(y0), t1, opts).unwrap();
    let mut used = Vec::new();
    while method.t() != t1 {
        method.step(&mut f).unwrap();
        used.push(method.order());
    }
    used
}

#[test]
fn the_order_rises_from_one_and_falls_where_the_solution_turns() {
    // On a smooth solution at a stringent tolerance the order climbs to 5; through the fast
    // jumps of Van der Pol's oscillator at μ = 1000 it comes down again and climbs once more.
    let used = orders(
        robertson,
        &[1.0, 0.0, 0.0],
        miri_or(0.01, 40.0),
        &OdeOptions::new().with_tolerances(1e-9, 1e-15),
        Bdf::new(),
    );
    // The first steps are of order 1, and the order changes only after k + 1 steps of order k.
    assert_eq!(used[..2], [1, 1]);
    assert!(used.windows(2).all(|w| w[0].abs_diff(w[1]) <= 1));
    if !cfg!(miri) {
        assert_eq!(*used.iter().max().unwrap(), 5);
        let vdp = |_t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>| {
            dy[(0, 0)] = y[(1, 0)];
            dy[(1, 0)] = 1000.0 * (1.0 - y[(0, 0)] * y[(0, 0)]) * y[(1, 0)] - y[(0, 0)];
        };
        let opts = OdeOptions::new().with_tolerances(1e-6, 1e-10);
        let used = orders(vdp, &[2.0, 0.0], 3000.0, &opts, Bdf::new());
        let falls = used.windows(2).filter(|w| w[1] < w[0]).count();
        let rises = used.windows(2).filter(|w| w[1] > w[0]).count();
        assert!(falls >= 3 && rises >= 6, "{falls} falls, {rises} rises");
    }
}

#[test]
fn the_order_is_held_to_its_cap() {
    for cap in 1..=5 {
        let used = orders(
            robertson,
            &[1.0, 0.0, 0.0],
            miri_or(0.01, 40.0),
            &OdeOptions::new().with_tolerances(1e-8, 1e-14),
            Bdf::new().with_max_order(cap),
        );
        assert!(used.iter().all(|&k| k <= cap), "{cap}");
        if !cfg!(miri) {
            assert_eq!(*used.iter().max().unwrap(), cap);
        }
    }
}

#[test]
#[should_panic(expected = "the order of a backward differentiation formula is from 1 to 5")]
fn an_order_cap_of_nought_panics() {
    let _ = Bdf::<f64>::new().with_max_order(0);
}

#[test]
#[should_panic(expected = "the order of a backward differentiation formula is from 1 to 5")]
fn an_order_cap_above_five_panics() {
    let _ = Bdf::<f64>::new().with_max_order(6);
}

#[test]
fn the_jacobian_is_formed_again_only_when_the_iteration_fails() {
    // A linear problem with its exact Jacobian: the first is the only one.
    let f = |t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>| {
        dy[(0, 0)] = -50.0 * (y[(0, 0)] - t.cos()) - y[(1, 0)];
        dy[(1, 0)] = y[(0, 0)] - 0.5 * y[(1, 0)];
    };
    let jac = |_t: f64, _y: &Matrix<f64>, j: &mut Matrix<f64>| {
        j[(0, 0)] = -50.0;
        j[(0, 1)] = -1.0;
        j[(1, 0)] = 1.0;
        j[(1, 1)] = -0.5;
    };
    let t1 = miri_or(1.0, 20.0);
    let r = integrate_with(
        f,
        &col(&[0.0, 0.0]),
        0.0,
        t1,
        Bdf::new().with_jacobian(jac),
        &tol(1e-7),
    )
    .unwrap();
    assert_eq!(r.njev, 1);
    // The decomposition is formed again when the step or the order changes, which on the
    // quasi-constant steps of the method is far less often than every step.
    assert!(
        r.nlu < r.steps / 2,
        "{} decompositions, {} steps",
        r.nlu,
        r.steps
    );
    // Robertson's problem to 10¹¹, the Jacobian differenced: kept for tens of steps at a time.
    if !cfg!(miri) {
        let opts = OdeOptions::new().with_tolerances(1e-6, 1e-14);
        let r = integrate_with(
            robertson,
            &col(&[1.0, 0.0, 0.0]),
            0.0,
            1e11,
            Bdf::new(),
            &opts,
        )
        .unwrap();
        assert!(
            r.njev * 20 < r.steps,
            "{} Jacobians, {} steps",
            r.njev,
            r.steps
        );
    }
}

#[test]
fn the_first_step_is_the_implicit_euler_step() {
    // y′ = −y from 1 with h₀ = 0.01: the formula of order 1 with κ = 0 is the implicit Euler
    // method, and the integration starts at order 1.
    let mut f = |_t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>| dy[(0, 0)] = -y[(0, 0)];
    let mut method = Bdf::new().with_ndf(false);
    // A tolerance the step's error estimate, ½·|y₁ − (y₀ − h·y₀)| = 5·10⁻⁵, is within.
    let opts = OdeOptions::new().with_tolerances(1e-3, 1e-3).with_h0(0.01);
    method.start(&mut f, 0.0, &col(&[1.0]), 1.0, &opts).unwrap();
    method.step(&mut f).unwrap();
    assert_eq!(method.order(), 1);
    // The implicit Euler step y₁ = y₀/(1 + h), to the tolerance of the Newton iteration.
    let want = 1.0 / 1.01;
    assert!(
        (method.y()[(0, 0)] - want).abs() < 1e-9,
        "{}",
        method.y()[(0, 0)]
    );
    assert_eq!(method.t(), 0.01);
}

#[test]
fn a_step_whose_iteration_fails_is_retried_shorter() {
    // Van der Pol's oscillator at μ = 1000 from a first step of 100: the iteration from the
    // prediction of a step that long does not converge, nothing it produced is kept, and the
    // integration recovers with shorter steps.
    let f = |_t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>| {
        dy[(0, 0)] = y[(1, 0)];
        dy[(1, 0)] = 1000.0 * (1.0 - y[(0, 0)] * y[(0, 0)]) * y[(1, 0)] - y[(0, 0)];
    };
    let t1 = miri_or(1.0, 100.0);
    let opts = OdeOptions::new()
        .with_tolerances(1e-6, 1e-10)
        .with_h0(100.0);
    let s = solve_ivp_with(f, &col(&[2.0, 0.0]), 0.0, t1, Bdf::new(), &mut [], &opts);
    assert_eq!(s.status, OdeStatus::Finished);
    assert!(s.rejected >= 3, "{}", s.rejected);
    let steps = s.dense.as_ref().unwrap().steps();
    assert!(steps[0].t1() - steps[0].t0() < 1.0);
    // Against the Radau method at a stringent tolerance.
    let reference = solve_ivp_with(
        f,
        &col(&[2.0, 0.0]),
        0.0,
        t1,
        a::math::numerics::RadauIia::new(),
        &mut [],
        &OdeOptions::new().with_tolerances(1e-10, 1e-14),
    );
    let e = stiff_error(s.y.as_slice(), reference.y.as_slice(), 1e-10);
    assert!(e < 1e-3, "{e}: {:?} against {:?}", s.y, reference.y);
}

#[test]
fn a_jacobian_that_is_not_finite_at_the_start_ends_the_integration() {
    let f = |_t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>| dy[(0, 0)] = -y[(0, 0)];
    let jac = |_t: f64, _y: &Matrix<f64>, j: &mut Matrix<f64>| j[(0, 0)] = f64::NAN;
    let s = solve_ivp_with(
        f,
        &col(&[1.0]),
        0.0,
        1.0,
        Bdf::new().with_jacobian(jac),
        &mut [],
        &OdeOptions::new(),
    );
    assert_eq!(s.status, OdeStatus::Failed(OdeFailure::NonFinite));
    assert_eq!((s.t, s.steps), (0.0, 0));
    // Not finite only beyond t = 1/2, where it is never asked for: the Jacobian of the start
    // serves the linear problem throughout, and nothing else is formed.
    let jac = |t: f64, _y: &Matrix<f64>, j: &mut Matrix<f64>| {
        j[(0, 0)] = if t > 0.5 { f64::NAN } else { -1.0 };
    };
    let s = solve_ivp_with(
        f,
        &col(&[1.0]),
        0.0,
        1.0,
        Bdf::new().with_jacobian(jac),
        &mut [],
        &OdeOptions::new(),
    );
    assert_eq!((s.status, s.njev), (OdeStatus::Finished, 1));
    assert!((s.y[(0, 0)] - (-1.0f64).exp()).abs() < 1e-5);
}

#[test]
fn the_formulas_work_in_f32() {
    let f = |_t: f32, y: &Matrix<f32>, dy: &mut Matrix<f32>| {
        let (a, b, c) = (y[(0, 0)], y[(1, 0)], y[(2, 0)]);
        dy[(0, 0)] = -0.04 * a + 1e4 * b * c;
        dy[(1, 0)] = 0.04 * a - 1e4 * b * c - 3e7 * b * b;
        dy[(2, 0)] = 3e7 * b * b;
    };
    let t1 = miri_or(1.0, 40.0);
    let opts = OdeOptions::<f32>::new().with_tolerances(1e-4, 1e-10);
    let s = solve_ivp_with(
        f,
        &col::<f32>(&[1.0, 0.0, 0.0]),
        0.0,
        t1,
        Bdf::new(),
        &mut [],
        &opts,
    );
    assert_eq!(s.status, OdeStatus::Finished);
    let sum: f32 = s.y.as_slice().iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "{sum}");
    if t1 == 40.0 {
        let e = stiff_error(s.y.as_slice(), ROBERTSON_40, 0.0);
        assert!(e < 1e-2, "{e}: {:?}", s.y);
    }
    assert_eq!(Bdf::<f32>::new().order(), 1);
}

/// The first step of `Bdf` on `y′ = λy` from 1 with `h₀ = h` and the exact Jacobian: the formula
/// of order 1 from the differences the integration starts with, `∇⁰y₀ = 1` and `∇¹y₀ = h·f₀`.
/// The prediction is `1 + z`, `z = h·λ`; the formula `(1 − κ)·(y₁ − y₁⁽⁰⁾) = h·f(y₁) − ∇¹y₀`
/// gives `y₁ = (1 − κ − κ·z)/(1 − κ − z)`, and the error estimate is `(κ + 1/2)·(y₁ − y₁⁽⁰⁾)`.
fn first_bdf_step(kappa: f64, z: f64) -> (f64, f64) {
    let y1 = (1.0 - kappa - kappa * z) / (1.0 - kappa - z);
    (y1, (kappa + 0.5) * (y1 - (1.0 + z)))
}

#[test]
fn the_first_step_is_the_formula_of_order_one_with_its_error_constant() {
    // y′ = −2y from 1 with h₀ = 0.1 and a purely absolute tolerance: the end of the first step is
    // that of the NDF of order 1, κ₁ = −0.1850 (Shampine and Reichelt's Table 1), or of the
    // implicit Euler method, κ = 0; and its error constant κ₁ + 1/2 — 0.315 and 0.5 — decides
    // the step, accepted where the tolerance makes the closed-form estimate 0.7 and rejected where
    // it makes it 1.4.
    let (lambda, h) = (-2.0, 0.1);
    let f = move |_t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>| dy[(0, 0)] = lambda * y[(0, 0)];
    let jac = move |_t: f64, _y: &Matrix<f64>, j: &mut Matrix<f64>| j[(0, 0)] = lambda;
    for (ndf, kappa) in [(true, -0.1850), (false, 0.0)] {
        let (y1, e1) = first_bdf_step(kappa, h * lambda);
        for (estimate, accepted) in [(0.7, true), (1.4, false)] {
            let opts = OdeOptions::new()
                .with_tolerances(0.0, e1.abs() / estimate)
                .with_h0(h);
            let method = Bdf::new().with_jacobian(jac).with_ndf(ndf);
            let s = solve_ivp_with(f, &col(&[1.0]), 0.0, h, method, &mut [], &opts);
            assert_eq!(s.status, OdeStatus::Finished, "κ = {kappa}");
            if accepted {
                assert_eq!((s.steps, s.rejected), (1, 0), "κ = {kappa}");
                assert!(
                    (s.y[(0, 0)] - y1).abs() < 1e-15,
                    "κ = {kappa}: {} against {y1}",
                    s.y[(0, 0)]
                );
            } else {
                assert!(s.rejected >= 1, "κ = {kappa}: the step was accepted");
            }
        }
    }
}
