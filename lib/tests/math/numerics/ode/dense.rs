//! Dense output: the continuous extensions of the methods against the exact solution between
//! the steps, their orders, their continuity, the sampling at given times, and the edges of the
//! interval.

use super::problems::{col, fixed, log_slope, miri_or, same_state, tol};
use a::math::matrix::Matrix;
use a::math::numerics::{
    DenseOutput, DenseStep, DormandPrince5, DormandPrince853, OdeMethod, OdeOptions, OdeSolution,
    RadauIia, Rk4, solve_at, solve_at_with, solve_ivp, solve_ivp_with,
};

/// The harmonic oscillator from `(1, 0)`, whose solution is `(cos t, −sin t)`.
fn oscillator(_t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>) {
    dy[(0, 0)] = y[(1, 0)];
    dy[(1, 0)] = -y[(0, 0)];
}

fn exact(t: f64) -> [f64; 2] {
    [t.cos(), -t.sin()]
}

fn err_at(y: &Matrix<f64>, t: f64) -> f64 {
    let e = exact(t);
    (y[(0, 0)] - e[0]).abs().max((y[(1, 0)] - e[1]).abs())
}

/// The largest error at the ends of the steps, and the largest at nine points inside each.
fn errors(solution: &OdeSolution<f64>) -> (f64, f64) {
    let dense = solution.dense.as_ref().unwrap();
    let (mut at_steps, mut between) = (0.0f64, 0.0f64);
    for step in dense.steps() {
        at_steps = at_steps.max(err_at(&step.eval(step.t1()), step.t1()));
        for k in 1..10 {
            let t = step.t0() + (step.t1() - step.t0()) * f64::from(k) / 10.0;
            between = between.max(err_at(&dense.eval(t).unwrap(), t));
        }
    }
    (at_steps, between)
}

fn integrate<M: OdeMethod<f64>>(method: M, opts: &OdeOptions<f64>) -> OdeSolution<f64> {
    let y0 = col(&[1.0, 0.0]);
    solve_ivp_with(oscillator, &y0, 0.0, 20.0, method, &mut [], opts)
}

#[test]
fn the_dense_output_is_as_accurate_between_the_steps_as_at_them() {
    // Adaptive steps at three tolerances: the worst error inside the steps is of the size of the
    // worst at their ends.
    for &t in miri_or(&[1e-5][..], &[1e-5, 1e-8, 1e-11][..]) {
        let opts = tol::<f64>(t);
        for (name, solution, factor) in [
            (
                "DormandPrince5",
                integrate(DormandPrince5::new(), &opts),
                3.0,
            ),
            (
                "DormandPrince853",
                integrate(DormandPrince853::new(), &opts),
                3.0,
            ),
            ("Rk4", integrate(Rk4::new(), &opts), 3.0),
        ] {
            let (at_steps, between) = errors(&solution);
            assert!(
                between <= factor * at_steps.max(t),
                "{name} at {t:e}: {between:e} between, {at_steps:e} at the steps"
            );
        }
    }
}

/// The largest error at the midpoints of `n` equal steps over `[0, 20]`, and at their ends.
fn midpoint_errors<M: OdeMethod<f64>>(method: M, n: usize) -> (f64, f64) {
    let h = 20.0 / n as f64;
    let solution = integrate(method, &fixed::<f64>(h));
    assert_eq!(solution.steps, n);
    let dense = solution.dense.as_ref().unwrap();
    let (mut ends, mut middles) = (0.0f64, 0.0f64);
    for step in dense.steps() {
        let m = 0.5 * (step.t0() + step.t1());
        middles = middles.max(err_at(&dense.eval(m).unwrap(), m));
        ends = ends.max(err_at(&step.eval(step.t1()), step.t1()));
    }
    (middles, ends)
}

#[test]
fn the_dense_output_converges_at_the_order_of_its_method() {
    if cfg!(miri) {
        // Hundreds of steps for a statistic; the lighter tests run the same code under Miri.
        return;
    }
    // With fixed steps the error at the midpoints falls as fast as the error at the steps: as
    // h⁴ for the Hermite cubic of the classical method, h⁵ for the continuous extension of order
    // 4 of the 5(4) pair, h⁸ for that of order 7 of the 8(5,3) method — the continuous
    // extension of order q adds a local error of order h^(q+1), no more than the method's own.
    let cases: [(&str, Vec<usize>, f64, f64); 2] = [
        ("DormandPrince5", vec![40, 80, 160, 320], 4.7, 5.5),
        ("DormandPrince853", vec![20, 40, 80], 7.5, 8.7),
    ];
    for (name, ns, lo, hi) in cases {
        let points: Vec<(f64, f64)> = ns
            .iter()
            .map(|&n| {
                let (middle, _) = if name == "DormandPrince5" {
                    midpoint_errors(DormandPrince5::new(), n)
                } else {
                    midpoint_errors(DormandPrince853::new(), n)
                };
                (20.0 / n as f64, middle)
            })
            .collect();
        let p = log_slope(&points);
        assert!(
            (lo..hi).contains(&p),
            "{name}: order {p} at the midpoints, {points:?}"
        );
    }
    // The classical method in its fixed form.
    let points: Vec<(f64, f64)> = [40, 80, 160, 320]
        .iter()
        .map(|&n| {
            let s = solve_ivp(
                oscillator,
                &col(&[1.0, 0.0]),
                0.0,
                20.0,
                Rk4::fixed(n),
                &mut [],
            );
            let dense = s.dense.as_ref().unwrap();
            let worst = dense
                .steps()
                .iter()
                .map(|step| {
                    let m = 0.5 * (step.t0() + step.t1());
                    err_at(&step.eval(m), m)
                })
                .fold(0.0, f64::max);
            (20.0 / n as f64, worst)
        })
        .collect();
    let p = log_slope(&points);
    assert!(
        (3.8..4.4).contains(&p),
        "Rk4: order {p} at the midpoints, {points:?}"
    );
}

#[test]
fn the_dense_output_is_continuous_across_the_steps() {
    let opts = tol::<f64>(miri_or(1e-4, 1e-6));
    for solution in [
        integrate(DormandPrince5::new(), &opts),
        integrate(DormandPrince853::new(), &opts),
        integrate(Rk4::new(), &opts),
        integrate(RadauIia::new(), &opts),
    ] {
        let steps = solution.dense.as_ref().unwrap().steps();
        assert!(steps.len() > 10);
        for pair in steps.windows(2) {
            assert_eq!(pair[0].t1(), pair[1].t0());
            let (a, b) = (pair[0].eval(pair[0].t1()), pair[1].eval(pair[1].t0()));
            for (x, y) in a.as_slice().iter().zip(b.as_slice()) {
                assert!((x - y).abs() <= 4.0 * f64::EPSILON, "{x} and {y}");
            }
        }
        // The first step starts on the initial state exactly, the last ends on the final one.
        assert_eq!(steps[0].eval(0.0).as_slice(), &[1.0, 0.0]);
        let last = steps.last().unwrap();
        let end = last.eval(last.t1());
        for (x, y) in end.as_slice().iter().zip(solution.y.as_slice()) {
            assert!((x - y).abs() <= 4.0 * f64::EPSILON);
        }
    }
}

#[test]
fn sampling_at_given_times_reads_the_same_polynomials() {
    let y0 = col(&[1.0, 0.0]);
    let ts: Vec<f64> = (0..=40).map(|k| 0.5 * f64::from(k)).collect();
    let tolerance = miri_or(1e-6, 1e-9);
    let opts = tol::<f64>(tolerance);
    let samples = solve_at_with(oscillator, &y0, 0.0, &ts, DormandPrince853::new(), &opts).unwrap();
    let solution = solve_ivp_with(
        oscillator,
        &y0,
        0.0,
        20.0,
        DormandPrince853::new(),
        &mut [],
        &opts,
    );
    assert_eq!(samples.len(), ts.len());
    for (t, y) in ts.iter().zip(&samples) {
        assert!(
            same_state(y.as_slice(), solution.eval(*t).unwrap().as_slice()),
            "at {t}"
        );
        assert!(err_at(y, *t) < 10.0 * tolerance);
    }
    // The first is the initial state and the last the end of the integration, exactly.
    assert_eq!(samples[0], y0);
    assert!(same_state(samples[40].as_slice(), solution.y.as_slice()));
    // Without the extra stages of the dense output except in the steps a time falls in.
    let sparse = solve_at_with(
        oscillator,
        &y0,
        0.0,
        &[20.0],
        DormandPrince853::new(),
        &opts,
    );
    assert!(same_state(
        sparse.unwrap()[0].as_slice(),
        solution.y.as_slice()
    ));
}

#[test]
fn sampling_takes_repeated_times_and_refuses_disordered_ones() {
    let y0 = col(&[1.0, 0.0]);
    let ys = solve_at(
        oscillator,
        &y0,
        0.0,
        &[0.0, 0.0, 1.0, 1.0, 3.0],
        DormandPrince5::new(),
    );
    let ys = ys.unwrap();
    assert_eq!(ys.len(), 5);
    assert_eq!((ys[0].clone(), ys[1].clone()), (y0.clone(), y0.clone()));
    assert_eq!(ys[2], ys[3]);
    assert!(
        solve_at(
            oscillator,
            &y0,
            0.0,
            &[1.0, 0.5, 2.0],
            DormandPrince5::new()
        )
        .is_none()
    );
    assert!(
        solve_at(
            oscillator,
            &y0,
            0.0,
            &[1.0, f64::NAN],
            DormandPrince5::new()
        )
        .is_none()
    );
    // A time behind the start in the direction the last one gives is out of order too.
    assert!(solve_at(oscillator, &y0, 0.0, &[-1.0, 2.0], DormandPrince5::new()).is_none());
    // Nothing asked, nothing integrated.
    assert_eq!(
        solve_at(oscillator, &y0, 0.0, &[], DormandPrince5::new()),
        Some(Vec::new())
    );
    // Backwards in time.
    let ts = [-0.5, -1.0, -2.0, -2.0];
    let ys = solve_at_with(
        oscillator,
        &y0,
        0.0,
        &ts,
        DormandPrince853::new(),
        &tol(1e-10),
    );
    for (t, y) in ts.iter().zip(&ys.unwrap()) {
        assert!(err_at(y, *t) < 1e-9, "{t}");
    }
}

#[test]
fn the_dense_output_knows_its_interval() {
    let y0 = col(&[1.0, 0.0]);
    let solution = solve_ivp(oscillator, &y0, 0.0, 5.0, DormandPrince5::new(), &mut []);
    let dense: &DenseOutput<f64> = solution.dense.as_ref().unwrap();
    assert_eq!((dense.t0(), dense.t1()), (0.0, 5.0));
    assert_eq!(dense.eval(0.0).unwrap(), y0);
    assert!(dense.eval(-1e-9).is_none());
    assert!(dense.eval(5.0 + 1e-9).is_none());
    assert!(dense.eval(f64::NAN).is_none());
    assert_eq!(solution.eval(5.0).unwrap(), solution.y);
    let mut out = [0.0; 2];
    assert!(dense.eval_into(2.5, &mut out));
    assert_eq!(out, dense.eval(2.5).unwrap().as_slice());
    // An integration of no length is its initial point.
    let none = solve_ivp(oscillator, &y0, 3.0, 3.0, DormandPrince5::new(), &mut []);
    let dense = none.dense.as_ref().unwrap();
    assert!(dense.steps().is_empty());
    assert_eq!(dense.eval(3.0).unwrap(), y0);
    assert!(dense.eval(3.1).is_none());
    // Backwards: the interval is [t1, t0].
    let back = solve_ivp(oscillator, &y0, 0.0, -5.0, DormandPrince853::new(), &mut []);
    let dense = back.dense.as_ref().unwrap();
    assert_eq!((dense.t0(), dense.t1()), (0.0, -5.0));
    assert!(dense.eval(-2.5).is_some() && dense.eval(2.5).is_none());
    assert!(err_at(&dense.eval(-2.5).unwrap(), -2.5) < 1e-5);
}

#[test]
#[should_panic(expected = "the output does not have the size of the state")]
fn the_dense_output_panics_on_an_output_of_the_wrong_size() {
    let y0 = col(&[1.0, 0.0]);
    let solution = solve_ivp(oscillator, &y0, 0.0, 1.0, DormandPrince5::new(), &mut []);
    let mut out = [0.0; 3];
    let _ = solution.dense.unwrap().eval_into(0.5, &mut out);
}

#[test]
fn a_dense_step_is_a_newton_form() {
    // The cubic p(t) = t³ − 2t + 1 over [1, 3], as q(θ) = p(1 + 2θ): its divided differences
    // on the nodes 0, 0, 1, 1 are q(0), q′(0), q[0, 0, 1] and q[0, 0, 1, 1], and the Newton form
    // on the first three nodes with them is q itself.
    let p = |t: f64| t * t * t - 2.0 * t + 1.0;
    let q = |theta: f64| p(1.0 + 2.0 * theta);
    let dq = |theta: f64| 2.0 * (3.0 * (1.0 + 2.0 * theta).powi(2) - 2.0);
    let (q0, q1, d0, d1) = (q(0.0), q(1.0), dq(0.0), dq(1.0));
    let c2 = q1 - q0 - d0;
    let c3 = d1 - (q1 - q0) - c2;
    let mut step = DenseStep::<f64>::new();
    step.reset(1.0, 3.0, 1, &[0.0, 0.0, 1.0]);
    step.coefficient_mut(0)[0] = q0;
    step.coefficient_mut(1)[0] = d0;
    step.coefficient_mut(2)[0] = c2;
    step.coefficient_mut(3)[0] = c3;
    assert_eq!(
        (step.t0(), step.t1(), step.dimension(), step.degree()),
        (1.0, 3.0, 1, 3)
    );
    assert_eq!(step.nodes(), &[0.0, 0.0, 1.0]);
    for k in 0..=20 {
        let t = 0.5 + 0.15 * f64::from(k);
        assert!((step.eval(t)[(0, 0)] - p(t)).abs() < 1e-12, "{t}");
    }
}
