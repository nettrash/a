//! `RadauIia`, the three-stage Radau IIA method as in RADAU5: one step against the method's
//! stability function and its collocation polynomial computed here from the closed forms of its
//! coefficients, the embedded error estimate with its two filters and the step-size controller
//! against RADAU5's formulas, the reuse of Jacobians and decompositions, and Robertson's problem.
//! The comparisons with SciPy, the dense output and the events of both stiff methods are in
//! `stiff.rs`.

use super::problems::{Problem, col, error, fixed, log_slope, miri_or, tol};
use super::reference::EXACT;
use super::stiff_problems::{Stiff, stiff_error};
use super::stiff_reference::{REFERENCE, ROBERTSON_40};
use a::math::complex::Complex;
use a::math::matrix::Matrix;
use a::math::numerics::{
    DormandPrince5, OdeFailure, OdeMethod, OdeOptions, OdeSolution, OdeStatus, RadauIia,
    integrate_with, solve_ivp_with,
};

/// Robertson's chemical kinetics, the stiffness benchmark.
fn robertson(_t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>) {
    let (a, b, c) = (y[(0, 0)], y[(1, 0)], y[(2, 0)]);
    dy[(0, 0)] = -0.04 * a + 1e4 * b * c;
    dy[(1, 0)] = 0.04 * a - 1e4 * b * c - 3e7 * b * b;
    dy[(2, 0)] = 3e7 * b * b;
}

fn robertson_jacobian(_t: f64, y: &Matrix<f64>, j: &mut Matrix<f64>) {
    let (b, c) = (y[(1, 0)], y[(2, 0)]);
    j[(0, 0)] = -0.04;
    j[(0, 1)] = 1e4 * c;
    j[(0, 2)] = 1e4 * b;
    j[(1, 0)] = 0.04;
    j[(1, 1)] = -1e4 * c - 6e7 * b;
    j[(1, 2)] = -1e4 * b;
    j[(2, 1)] = 6e7 * b;
}

/// The coefficients of the three-stage Radau IIA method, from their closed forms: the zeros
/// `(4 ∓ √6)/10` of `10x² − 8x + 1` and 1, and `a[i][j]` the integral from 0 to `cᵢ` of the
/// Lagrange polynomial of `cⱼ` (as `data/gen_radau.py` derives them).
fn radau_iia() -> ([f64; 3], [[f64; 3]; 3]) {
    let s6 = 6f64.sqrt();
    let c = [(4.0 - s6) / 10.0, (4.0 + s6) / 10.0, 1.0];
    let a = [
        [
            11.0 / 45.0 - 7.0 * s6 / 360.0,
            37.0 / 225.0 - 169.0 * s6 / 1800.0,
            -2.0 / 225.0 + s6 / 75.0,
        ],
        [
            37.0 / 225.0 + 169.0 * s6 / 1800.0,
            11.0 / 45.0 + 7.0 * s6 / 360.0,
            -2.0 / 225.0 - s6 / 75.0,
        ],
        [4.0 / 9.0 - s6 / 36.0, 4.0 / 9.0 + s6 / 36.0, 1.0 / 9.0],
    ];
    (c, a)
}

/// The stages `Y = (I − z·A)⁻¹·𝟙` of one step of the method on `y′ = λy` from `y = 1`, `z = h·λ`,
/// by Gaussian elimination in complex arithmetic.
fn stages(z: Complex<f64>) -> [Complex<f64>; 3] {
    let (_, a) = radau_iia();
    let one = Complex::new(1.0, 0.0);
    let mut m = [[Complex::new(0.0, 0.0); 4]; 3];
    for i in 0..3 {
        for j in 0..3 {
            m[i][j] = -(z * a[i][j]);
            if i == j {
                m[i][j] += one;
            }
        }
        m[i][3] = one;
    }
    for k in 0..3 {
        let pivot = m[k][k];
        for v in &mut m[k][k..] {
            *v /= pivot;
        }
        let row = m[k];
        for (i, other) in m.iter_mut().enumerate() {
            if i != k {
                let factor = other[k];
                for (x, &v) in other[k..].iter_mut().zip(&row[k..]) {
                    *x -= factor * v;
                }
            }
        }
    }
    [m[0][3], m[1][3], m[2][3]]
}

/// The stability function of the three-stage Radau IIA method, the (2, 3) Padé approximation of
/// `eᶻ`: `(1 + 2z/5 + z²/20) / (1 − 3z/5 + 3z²/20 − z³/60)`.
fn stability(z: Complex<f64>) -> Complex<f64> {
    let one = Complex::new(1.0, 0.0);
    let num = one + z * 0.4 + z * z * 0.05;
    let den = one - z * 0.6 + z * z * 0.15 - z * z * z * (1.0 / 60.0);
    num / den
}

/// One step of size `h` on `y′ = [[a, −b], [b, a]]·y` from `(1, 0)`, which is `w′ = (a + ib)·w`
/// for `w = y₀ + i·y₁`, with the exact Jacobian and tolerances no step can fail.
fn one_step(a: f64, b: f64, h: f64) -> OdeSolution<f64> {
    let f = move |_t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>| {
        dy[(0, 0)] = a * y[(0, 0)] - b * y[(1, 0)];
        dy[(1, 0)] = b * y[(0, 0)] + a * y[(1, 0)];
    };
    let jac = move |_t: f64, _y: &Matrix<f64>, j: &mut Matrix<f64>| {
        j[(0, 0)] = a;
        j[(0, 1)] = -b;
        j[(1, 0)] = b;
        j[(1, 1)] = a;
    };
    let opts = OdeOptions::new().with_tolerances(1e10, 1e10).with_h0(h);
    let method = RadauIia::new().with_jacobian(jac);
    solve_ivp_with(f, &col(&[1.0, 0.0]), 0.0, h, method, &mut [], &opts)
}

#[test]
fn one_step_is_the_stability_function_of_radau_iia() {
    // A linear problem with its exact Jacobian is solved by the first Newton iterate, so one
    // step is R(z)·y₀ to rounding: real z, through the real decomposition and the complex one
    // together, and complex z, which the complex one carries.
    let h = 0.5;
    let zs = [
        Complex::new(0.3, 0.0),
        Complex::new(-0.5, 0.0),
        Complex::new(-2.0, 0.0),
        Complex::new(-10.0, 0.0),
        Complex::new(-1e3, 0.0),
        Complex::new(-1e6, 0.0),
        Complex::new(-1.0, 2.0),
        Complex::new(-0.1, 5.0),
        Complex::new(-100.0, 100.0),
        Complex::new(0.0, 0.5),
        Complex::new(-3.0, -40.0),
    ];
    for z in zs {
        let s = one_step(z.re / h, z.im / h, h);
        assert_eq!((s.status, s.steps, s.rejected), (OdeStatus::Finished, 1, 0));
        let r = stability(z);
        let got = Complex::new(s.y[(0, 0)], s.y[(1, 0)]);
        // The end of the step is y₀ + Z₃, which carries the rounding of y₀ = 1 when R(z) is small.
        assert!(
            (got - r).abs() <= 1e-13 * r.abs() + 2e-15,
            "z = {z}: {got} against {r}"
        );
    }
    // L-stability: the stiffest components are all but wiped out, as R(z) ~ −3/z.
    let s = one_step(-1e12, 0.0, 1.0);
    assert!(s.y[(0, 0)].abs() < 4e-12, "{}", s.y[(0, 0)]);
}

#[test]
fn the_dense_output_is_the_collocation_polynomial() {
    // The cubic through (0, 1) and (cᵢ, Yᵢ), the stages computed here, against the dense output
    // of the step.
    let (c, _) = radau_iia();
    for (a, b, h) in [(-3.0, 0.0, 0.4), (-1.0, 4.0, 0.3), (-500.0, 50.0, 0.1)] {
        let z = Complex::new(a * h, b * h);
        let y = stages(z);
        let nodes = [0.0, c[0], c[1], c[2]];
        let values = [Complex::new(1.0, 0.0), y[0], y[1], y[2]];
        let cubic = |theta: f64| {
            let mut sum = Complex::new(0.0, 0.0);
            for i in 0..4 {
                let mut l = 1.0;
                for j in 0..4 {
                    if i != j {
                        l *= (theta - nodes[j]) / (nodes[i] - nodes[j]);
                    }
                }
                sum += values[i] * l;
            }
            sum
        };
        let s = one_step(a, b, h);
        let dense = s.dense.as_ref().unwrap();
        for theta in [0.05, c[0], 0.3, 0.5, c[1], 0.8, 0.95, 1.0] {
            let got = dense.eval(theta * h).unwrap();
            let got = Complex::new(got[(0, 0)], got[(1, 0)]);
            let want = cubic(theta);
            assert!(
                (got - want).abs() < 1e-13,
                "z = {z}, θ = {theta}: {got} against {want}"
            );
        }
    }
    // A cubic solution is the collocation polynomial itself, in every step.
    let f = |t: f64, _y: &Matrix<f64>, dy: &mut Matrix<f64>| dy[(0, 0)] = 3.0 * t * t - 1.0;
    let opts = tol::<f64>(1e-9).with_hmax(0.3);
    let s = solve_ivp_with(f, &col(&[0.0]), 0.0, 2.0, RadauIia::new(), &mut [], &opts);
    let dense = s.dense.as_ref().unwrap();
    assert!(dense.steps().len() >= 7);
    for k in 0..=200 {
        let t = 0.01 * f64::from(k);
        let y = dense.eval(t).unwrap()[(0, 0)];
        assert!((y - (t * t * t - t)).abs() < 1e-13, "{t}: {y}");
    }
}

/// The error estimate of RADAU5 for the first step of size `h` on `y′ = λy` from 1, computed
/// here from the stages: `(γ₀·h·λ + γ₀·Σ ddⱼ·Zⱼ)/(1 − h·γ₀·λ)`, and filtered a second time,
/// `(γ₀·h·λ·(1 + err) + γ₀·Σ ddⱼ·Zⱼ)/(1 − h·γ₀·λ)`; and the end of the step.
fn first_estimate(lambda: f64, h: f64) -> (f64, f64, f64) {
    let s6 = 6f64.sqrt();
    let gamma = 3.0 + 3f64.powf(2.0 / 3.0) - 3f64.powf(1.0 / 3.0);
    let g0 = 1.0 / gamma;
    let dd = [
        -(13.0 + 7.0 * s6) / 3.0,
        (-13.0 + 7.0 * s6) / 3.0,
        -1.0 / 3.0,
    ];
    let z = h * lambda;
    let y = stages(Complex::new(z, 0.0));
    let ddz: f64 = (0..3).map(|j| dd[j] * (y[j].re - 1.0)).sum();
    let first = (g0 * z + g0 * ddz) / (1.0 - g0 * z);
    let second = (g0 * z * (1.0 + first) + g0 * ddz) / (1.0 - g0 * z);
    (first, second, y[2].re)
}

/// The step RADAU5's controller takes after a first step of error `err` and size `h`, whose
/// Newton iteration took two iterates (the first from the zero start, the second to see it
/// converge): `h · 0.84375/err^(1/4)` held to `[h/5, 8h]`, and `h` itself within `[1, 1.2]·h`.
fn after_first(h: f64, err: f64) -> f64 {
    let fac = 0.9f64.min(0.9 * 15.0 / (2.0 + 14.0));
    let quot = (err.max(1e-10).powf(0.25) / fac).clamp(0.125, 5.0);
    let next = h / quot;
    if (1.0..=1.2).contains(&(next / h)) {
        h
    } else {
        next
    }
}

fn lengths(s: &OdeSolution<f64>) -> Vec<f64> {
    s.dense
        .as_ref()
        .unwrap()
        .steps()
        .iter()
        .map(|step| step.t1() - step.t0())
        .collect()
}

#[test]
fn the_error_estimate_and_the_step_after_it_are_radau5s() {
    // y′ = λy from 1 with h₀ given: the second step is what RADAU5's controller makes of the
    // first step's embedded estimate, computed here from the closed forms.
    for (lambda, h, tolerance) in [(-2.0, 0.2, 1e-3), (-2.0, 0.2, 1e-4), (-30.0, 0.05, 0.05)] {
        let f =
            move |_t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>| dy[(0, 0)] = lambda * y[(0, 0)];
        let jac = move |_t: f64, _y: &Matrix<f64>, j: &mut Matrix<f64>| j[(0, 0)] = lambda;
        let opts = tol::<f64>(tolerance).with_h0(h);
        let method = RadauIia::new().with_jacobian(jac);
        let s = solve_ivp_with(f, &col(&[1.0]), 0.0, 10.0, method, &mut [], &opts);
        let (e, _, y1) = first_estimate(lambda, h);
        let err = e.abs() / (tolerance + tolerance * y1.abs().max(1.0));
        assert!(err < 1.0, "λ = {lambda}: the first step is to be accepted");
        let steps = lengths(&s);
        assert_eq!(s.rejected, 0, "λ = {lambda}");
        assert!((steps[0] - h).abs() < 1e-16);
        let want = after_first(h, err);
        assert!(
            (steps[1] - want).abs() < 1e-12 * want,
            "λ = {lambda}, tol = {tolerance}: {} against {want} (err {err})",
            steps[1]
        );
    }
    // 1e-4 is a tolerance at which the controller asks for 1.05 h, which RADAU5 does not take:
    // it keeps h, and with it the decompositions of the first step.
    let f = |_t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>| dy[(0, 0)] = -2.0 * y[(0, 0)];
    let jac = |_t: f64, _y: &Matrix<f64>, j: &mut Matrix<f64>| j[(0, 0)] = -2.0;
    let opts = tol::<f64>(1e-4).with_h0(0.2);
    let s = solve_ivp_with(
        f,
        &col(&[1.0]),
        0.0,
        0.4,
        RadauIia::new().with_jacobian(jac),
        &mut [],
        &opts,
    );
    assert_eq!(lengths(&s).len(), 2);
    assert_eq!((s.njev, s.nlu), (1, 2));
}

#[test]
fn the_predictive_controller_shortens_a_step_whose_error_grows_too_fast() {
    // y′ = y from 1 under a purely absolute tolerance: the error of a step grows with the
    // solution as well as with the step, faster than h⁴, and Gustafsson's predictive term
    // `(h₁/h₂)·(err₂²/err₁)^(1/4)/0.9` asks for a shorter third step than the classical
    // `err₂^(1/4)/fac` — by 3% here, `fac` being 0.9·15/16 after the two iterates of the second
    // step (by 10% against the 0.9 of a step of one) — and is what RADAU5 takes. Computed from
    // the closed forms as in the test above; the predictive term does not depend on the iterates
    // of the second step, so neither does the third step.
    let (lambda, h0, atol) = (1.0, 0.2, 1e-3);
    let (e1, _, y1) = first_estimate(lambda, h0);
    let err1 = e1.abs() / atol;
    let h1 = after_first(h0, err1);
    let (e2, _, _) = first_estimate(lambda, h1);
    let err2 = (e2 * y1).abs() / atol;
    assert!(err1 < 1.0 && err2 < 1.0, "{err1} {err2}");
    let classical = (err2.powf(0.25) / (0.9 * 15.0 / 16.0)).clamp(0.125, 5.0);
    let predictive =
        ((h0 / h1) * (err2 * err2 / err1.max(1e-2)).powf(0.25) / 0.9).clamp(0.125, 5.0);
    assert!(predictive > 1.02 * classical, "{predictive} {classical}");
    let want = h1 / predictive;
    let f = move |_t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>| dy[(0, 0)] = lambda * y[(0, 0)];
    let jac = move |_t: f64, _y: &Matrix<f64>, j: &mut Matrix<f64>| j[(0, 0)] = lambda;
    let opts = OdeOptions::new().with_tolerances(0.0, atol).with_h0(h0);
    let method = RadauIia::new().with_jacobian(jac);
    let s = solve_ivp_with(f, &col(&[1.0]), 0.0, 5.0, method, &mut [], &opts);
    let steps = lengths(&s);
    assert_eq!(s.rejected, 0);
    assert!(
        (steps[1] - h1).abs() < 1e-10 * h1,
        "{} against {h1}",
        steps[1]
    );
    assert!(
        (steps[2] - want).abs() < 1e-10 * want,
        "{} against {want} (the classical step would be {})",
        steps[2],
        h1 / classical
    );
}

#[test]
fn a_first_step_through_a_stiff_transient_is_judged_by_the_second_filter() {
    // y′ = −2·10⁴·y with h₀ = 0.1: the first filter of the estimate, (I − h·γ₀·J)⁻¹, leaves an
    // error of about 50, which would reject the step; the second, through f(y + err), divides it
    // by 1 + 2000·γ₀ ≈ 550 and accepts it, as RADAU5 does for a first step.
    let lambda = -2e4;
    let (h, tolerance) = (0.1, 0.01);
    let (first, second, y1) = first_estimate(lambda, h);
    let scale = tolerance + tolerance * y1.abs().max(1.0);
    let (err, err2) = (first.abs() / scale, second.abs() / scale);
    assert!(err > 10.0 && err2 < 0.2, "{err} {err2}");
    let f = move |_t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>| dy[(0, 0)] = lambda * y[(0, 0)];
    let jac = move |_t: f64, _y: &Matrix<f64>, j: &mut Matrix<f64>| j[(0, 0)] = lambda;
    let opts = tol::<f64>(tolerance).with_h0(h);
    let method = RadauIia::new().with_jacobian(jac);
    let s = solve_ivp_with(f, &col(&[1.0]), 0.0, 10.0, method, &mut [], &opts);
    let steps = lengths(&s);
    assert!((steps[0] - h).abs() < 1e-16, "the first step was rejected");
    let want = after_first(h, err2);
    assert!(
        (steps[1] - want).abs() < 1e-12 * want,
        "{} against {want}",
        steps[1]
    );
    assert_eq!(s.status, OdeStatus::Finished);
    assert!(s.y[(0, 0)].abs() < 1e-10);
}

#[test]
fn with_fixed_steps_the_method_is_of_order_five_and_its_dense_output_of_order_three() {
    // The harmonic oscillator over [0, 10] with n equal steps. It is linear, so the Newton
    // iteration with the exact Jacobian is exact whatever the tolerance, and tolerances no step
    // can fail hold the step at h: the end of the integration converges at the order of the
    // method, 5, and the collocation polynomial at the middle of the steps at 4, that of an
    // interpolant of the stage order 3. (A nonlinear problem cannot be run with fixed steps:
    // RADAU5's iteration is held to a fraction of the tolerance, so the tolerances that fix the
    // step would also leave the stages unconverged. Its order there is in the next test.)
    if cfg!(miri) {
        return;
    }
    let f = |_t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>| {
        dy[(0, 0)] = y[(1, 0)];
        dy[(1, 0)] = -y[(0, 0)];
    };
    let jac = |_t: f64, _y: &Matrix<f64>, j: &mut Matrix<f64>| {
        j[(0, 1)] = 1.0;
        j[(1, 0)] = -1.0;
    };
    let (mut ends, mut middles) = (Vec::new(), Vec::new());
    // From h = 0.25 down, where the midpoint error is in its asymptotic regime (its slopes
    // between neighbours are 4.31, 4.19, 4.1).
    for n in [40, 80, 160, 320] {
        let h = 10.0 / f64::from(n);
        let method = RadauIia::new().with_jacobian(jac);
        let s = solve_ivp_with(f, &col(&[1.0, 0.0]), 0.0, 10.0, method, &mut [], &fixed(h));
        assert_eq!((s.steps, s.rejected), (n as usize, 0));
        let e = error(s.y.as_slice(), &[10f64.cos(), -10f64.sin()]);
        let dense = s.dense.as_ref().unwrap();
        let middle = dense
            .steps()
            .iter()
            .map(|step| {
                let t = 0.5 * (step.t0() + step.t1());
                error(step.eval(t).as_slice(), &[t.cos(), -t.sin()])
            })
            .fold(0.0, f64::max);
        ends.push((h, e));
        middles.push((h, middle));
    }
    let (order, dense_order) = (log_slope(&ends), log_slope(&middles));
    assert!(order > 4.7 && order < 5.4, "{order}: {ends:?}");
    assert!(
        dense_order > 3.7 && dense_order < 4.4,
        "{dense_order}: {middles:?}"
    );
}

#[test]
fn the_global_error_falls_as_the_fifth_power_of_the_steps() {
    // The Brusselator, which is not stiff, at tolerances from 1e-4 to 1e-10: the error against
    // mpmath's solution falls as the number of steps to the power −5.
    if cfg!(miri) {
        return;
    }
    let p = Problem::Brusselator;
    let exact = EXACT.iter().find(|e| e.0 == p).unwrap().1;
    let points: Vec<(f64, f64)> = (4..=10)
        .map(|k| {
            let opts = tol::<f64>(10f64.powi(-k));
            let r =
                integrate_with(p.f(), &col(&p.y0()), 0.0, p.t1(), RadauIia::new(), &opts).unwrap();
            (r.steps as f64, error(r.y.as_slice(), exact))
        })
        .collect();
    let slope = -log_slope(&points);
    assert!(slope > 4.6 && slope < 6.0, "{slope}: {points:?}");
}

#[test]
fn the_stiff_method_solves_robertsons_problem_to_forty() {
    // Under Miri only the first tenth of a time unit, and the invariant y₁ + y₂ + y₃ = 1.
    let t1 = miri_or(0.1, 40.0);
    let y0 = col(&[1.0, 0.0, 0.0]);
    let opts = OdeOptions::new().with_tolerances(1e-8, 1e-13);
    for (name, r) in [
        (
            "differenced",
            integrate_with(robertson, &y0, 0.0, t1, RadauIia::new(), &opts),
        ),
        (
            "analytic",
            integrate_with(
                robertson,
                &y0,
                0.0,
                t1,
                RadauIia::new().with_jacobian(robertson_jacobian),
                &opts,
            ),
        ),
    ] {
        let r = r.unwrap();
        let sum: f64 = r.y.as_slice().iter().sum();
        assert!((sum - 1.0).abs() < 1e-12, "{name}: {sum}");
        if t1 == 40.0 {
            let e = stiff_error(r.y.as_slice(), ROBERTSON_40, 1e-13);
            assert!(e < 1e-8, "{name}: error {e}, {:?}", r.y);
            assert!(
                r.steps < 400 && r.rejected < 20,
                "{name}: {} steps",
                r.steps
            );
        }
        // The Jacobian is formed again only when the iteration converged slowly.
        assert!(r.njev < r.steps, "{name}: {} Jacobians", r.njev);
    }
    // The explicit pair cannot get there in 20,000 steps: once the transient has passed, the
    // stiff eigenvalue of some −3·10³ holds it to steps of about a thousandth for stability
    // alone, which forty time units need forty thousand of. (Not under Miri: far too long.)
    if !cfg!(miri) {
        let budget = opts.with_max_steps(20_000);
        let explicit = integrate_with(robertson, &y0, 0.0, 40.0, DormandPrince5::new(), &budget);
        assert!(explicit.is_none());
    }
}

#[test]
fn an_analytic_jacobian_saves_the_evaluations_of_the_differenced_one() {
    let y0 = col(&[1.0, 0.0, 0.0]);
    let opts = OdeOptions::new().with_tolerances(1e-6, 1e-10);
    let t1 = miri_or(0.05, 1.0);
    let differenced = integrate_with(robertson, &y0, 0.0, t1, RadauIia::new(), &opts).unwrap();
    let analytic = integrate_with(
        robertson,
        &y0,
        0.0,
        t1,
        RadauIia::new().with_jacobian(robertson_jacobian),
        &opts,
    )
    .unwrap();
    // The same steps to within a few, and at least n evaluations less for every Jacobian.
    assert!(differenced.steps.abs_diff(analytic.steps) <= differenced.steps / 10 + 2);
    assert!(
        analytic.nfev + 3 * analytic.njev <= differenced.nfev,
        "{} {}",
        analytic.nfev,
        differenced.nfev
    );
    for (a, b) in analytic.y.as_slice().iter().zip(differenced.y.as_slice()) {
        assert!((a - b).abs() < 1e-6 * a.abs().max(1e-6), "{a} {b}");
    }
}

#[test]
fn jacobians_and_decompositions_are_kept_while_they_serve() {
    // A linear problem with its exact Jacobian: in every step the first iterate solves the
    // stages and the second shows a rate at the rounding level, so the first Jacobian is the only
    // one; and the decompositions are formed only when the step changes, which it does not within
    // [1, 1.2] of the last.
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
    let opts = tol::<f64>(1e-7);
    let t1 = miri_or(1.0, 20.0);
    let r = integrate_with(
        f,
        &col(&[0.0, 0.0]),
        0.0,
        t1,
        RadauIia::new().with_jacobian(jac),
        &opts,
    )
    .unwrap();
    assert_eq!(r.njev, 1);
    assert_eq!(r.nlu % 2, 0, "a real and a complex decomposition each time");
    assert!(
        r.nlu < 2 * (r.steps + r.rejected),
        "{} decompositions for {} steps",
        r.nlu,
        r.steps
    );
    // With the Jacobian differenced, a linear problem still needs only the first: the
    // differences are exact to rounding, and the rate stays far below 0.001.
    let r = integrate_with(f, &col(&[0.0, 0.0]), 0.0, t1, RadauIia::new(), &opts).unwrap();
    assert!(r.njev <= 2, "{}", r.njev);
}

/// Kaps's problem `y₁′ = −(1/ε + 2)·y₁ + y₂²/ε`, `y₂′ = y₁ − y₂ − y₂²` from `(1, 1)`, whose
/// solution is `(e^(−2t), e^(−t))` whatever `ε`: the stiff component follows the slow one through
/// the square, `y₁ ≈ y₂²`.
fn kaps(eps: f64) -> impl Fn(f64, &Matrix<f64>, &mut Matrix<f64>) + Copy {
    move |_t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>| {
        dy[(0, 0)] = -(1.0 / eps + 2.0) * y[(0, 0)] + y[(1, 0)] * y[(1, 0)] / eps;
        dy[(1, 0)] = y[(0, 0)] - y[(1, 0)] - y[(1, 0)] * y[(1, 0)];
    }
}

/// The error of `y` against `exact` in the norm the step is controlled in,
/// `max |yᵢ − exactᵢ| / (atol + rtol·|exactᵢ|)`: below one is within the tolerance.
fn in_tolerances(y: &Matrix<f64>, exact: &[f64], rtol: f64, atol: f64) -> f64 {
    y.as_slice()
        .iter()
        .zip(exact)
        .map(|(v, e)| (v - e).abs() / (atol + rtol * e.abs()))
        .fold(0.0, f64::max)
}

#[test]
fn kaps_problem_ends_within_its_tolerance_at_every_end_time() {
    // RADAU5 accepts the first Newton iterate of a step on the rate of the step before, raised
    // to 0.8. After a step that converged fast that lets a first correction of hundreds of
    // tolerances pass while the Jacobian, kept from earlier steps, no longer fits — and here its
    // entry 2y₂/ε carries the stiff component after the slow one. The first version of this
    // code, which did that, ended Kaps's problem at ε = 10⁻⁹, rtol = 10⁻³ and t = 1.1784 at
    // y₁ = 0.104213 for 0.094723, 99 tolerances out, the error estimate of every step below one;
    // over end times in [0.5, 5] it was more than ten tolerances out at 30 of 200. With a rate
    // measured in every step, as SciPy's Radau requires (whose largest error over the end times
    // below is 0.06: data/check_stiff_coupling.py), every end is within the tolerance.
    let (rtol, atol) = (1e-3, 1e-6);
    let opts = OdeOptions::new().with_tolerances(rtol, atol);
    let y0 = col(&[1.0, 1.0]);
    let t1 = 1.1784;
    let r = integrate_with(kaps(1e-9), &y0, 0.0, t1, RadauIia::new(), &opts).unwrap();
    let e = in_tolerances(&r.y, &[(-2.0 * t1).exp(), (-t1).exp()], rtol, atol);
    assert!(e < 1.0, "t = {t1}: {e} tolerances, {:?}", r.y);
    if cfg!(miri) {
        return;
    }
    let jacobian = |eps: f64| {
        move |_t: f64, y: &Matrix<f64>, j: &mut Matrix<f64>| {
            j[(0, 0)] = -(1.0 / eps + 2.0);
            j[(0, 1)] = 2.0 * y[(1, 0)] / eps;
            j[(1, 0)] = 1.0;
            j[(1, 1)] = -1.0 - 2.0 * y[(1, 0)];
        }
    };
    let mut worst = 0.0f64;
    for eps in [1e-6, 1e-9] {
        for (rtol, atol) in [(1e-3, 1e-6), (1e-4, 1e-7)] {
            let opts = OdeOptions::new().with_tolerances(rtol, atol);
            for k in 0..25 {
                let t1 = 0.5 + 4.5 * f64::from(k) / 24.0;
                let exact = [(-2.0 * t1).exp(), (-t1).exp()];
                for (name, r) in [
                    (
                        "differenced",
                        integrate_with(kaps(eps), &y0, 0.0, t1, RadauIia::new(), &opts),
                    ),
                    (
                        "analytic",
                        integrate_with(
                            kaps(eps),
                            &y0,
                            0.0,
                            t1,
                            RadauIia::new().with_jacobian(jacobian(eps)),
                            &opts,
                        ),
                    ),
                ] {
                    let r = r.unwrap_or_else(|| panic!("{name}, ε = {eps:e}, t = {t1}: failed"));
                    let e = in_tolerances(&r.y, &exact, rtol, atol);
                    worst = worst.max(e);
                    assert!(
                        e < 1.0,
                        "{name}, ε = {eps:e}, rtol = {rtol:e}, t = {t1}: {e} tolerances"
                    );
                }
            }
        }
    }
    eprintln!("Kaps: the largest error over the end times, {worst:.3} tolerances");
}

/// The problem `y₁′ = (y₂³ − y₁)/ε`, `y₂′ = −y₂ + y₃`, `y₃′ = −2y₃ + cos(t)/2` from `(0, 1.5, 1)`.
fn cube(eps: f64) -> impl Fn(f64, &Matrix<f64>, &mut Matrix<f64>) + Copy {
    move |t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>| {
        dy[(0, 0)] = (y[(1, 0)].powi(3) - y[(0, 0)]) / eps;
        dy[(1, 0)] = -y[(1, 0)] + y[(2, 0)];
        dy[(2, 0)] = -2.0 * y[(2, 0)] + 0.5 * t.cos();
    }
}

/// The solution of [`cube`] past its initial layer: `y₃` and `y₂` exactly, and `y₁` by the outer
/// expansion `u − ε·u′ + ε²·u″` of `ε·y₁′ = u − y₁`, `u = y₂³`, whose error is `O(ε³)` (it agrees
/// with SciPy's Radau at rtol = 1e-13 to 2·10⁻¹⁴: data/check_stiff_coupling.py).
fn cube_solution(eps: f64, t: f64) -> [f64; 3] {
    let (c, s, e1, e2) = (t.cos(), t.sin(), (-t).exp(), (-2.0 * t).exp());
    let y3 = 0.2 * c + 0.1 * s + 0.8 * e2;
    let y2 = 0.05 * c + 0.15 * s - 0.8 * e2 + 2.25 * e1;
    let d2 = -0.05 * s + 0.15 * c + 1.6 * e2 - 2.25 * e1;
    let dd2 = -0.05 * c - 0.15 * s - 3.2 * e2 + 2.25 * e1;
    let u = y2 * y2 * y2;
    let du = 3.0 * y2 * y2 * d2;
    let ddu = 6.0 * y2 * d2 * d2 + 3.0 * y2 * y2 * dd2;
    [u - eps * du + eps * eps * ddu, y2, y3]
}

#[test]
fn a_stiff_component_that_follows_a_cube_ends_within_its_tolerance() {
    // The coupling of Kaps's problem through a cube, the slow part forced: the first version of
    // this code, which accepted a first iterate on the rate of the step before, ended up to 37
    // tolerances out at rtol = 1e-3 and 8 at 1e-5 over end times in [0.5, 10]; SciPy's Radau
    // stays below 0.25 and 0.04 over the end times below (data/check_stiff_coupling.py).
    if cfg!(miri) {
        return;
    }
    let y0 = col(&[0.0, 1.5, 1.0]);
    let mut worst = 0.0f64;
    for eps in [1e-5, 1e-8] {
        for (rtol, atol) in [(1e-3, 1e-6), (1e-5, 1e-8)] {
            let opts = OdeOptions::new().with_tolerances(rtol, atol);
            for k in 0..20 {
                let t1 = 0.5 + 9.5 * f64::from(k) / 19.0;
                let r = integrate_with(cube(eps), &y0, 0.0, t1, RadauIia::new(), &opts)
                    .unwrap_or_else(|| panic!("ε = {eps:e}, t = {t1}: failed"));
                let e = in_tolerances(&r.y, &cube_solution(eps, t1), rtol, atol);
                worst = worst.max(e);
                assert!(
                    e < 1.0,
                    "ε = {eps:e}, rtol = {rtol:e}, t = {t1}: {e} tolerances, {:?}",
                    r.y
                );
            }
        }
    }
    eprintln!("cube: the largest error over the end times, {worst:.3} tolerances");
}

#[test]
fn starting_values_that_are_already_the_solution_keep_the_jacobian() {
    // y′ = λ·(y − p) + p′ with p cubic, from y = p(0), and a second component coupled to the
    // first: the solution is p (and 2p), which the collocation polynomial of degree 3 reproduces,
    // so from the second step on the starting values are the stages to rounding and every Newton
    // correction is rounding. Such a correction ends the iteration: a rate computed from it would
    // be a ratio of rounding errors, which re-forms the Jacobian at almost every step (2 to 4
    // Jacobians here) and can abandon a converged iteration.
    let p = |t: f64| 0.5 + t * (1.0 + t * (-0.3 + 0.05 * t));
    let dp = |t: f64| 1.0 + t * (-0.6 + 0.15 * t);
    let t1 = miri_or(2.0, 20.0);
    for lambda in [-1.0, -1e3, -1e6] {
        let f = move |t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>| {
            dy[(0, 0)] = lambda * (y[(0, 0)] - p(t)) + dp(t);
            dy[(1, 0)] =
                0.5 * lambda * (y[(1, 0)] - 2.0 * p(t)) + 2.0 * dp(t) + 0.1 * (y[(0, 0)] - p(t));
        };
        let jac = move |_t: f64, _y: &Matrix<f64>, j: &mut Matrix<f64>| {
            j[(0, 0)] = lambda;
            j[(1, 0)] = 0.1;
            j[(1, 1)] = 0.5 * lambda;
        };
        let opts = OdeOptions::new().with_tolerances(1e-6, 1e-9);
        let y0 = col(&[p(0.0), 2.0 * p(0.0)]);
        for (name, r) in [
            (
                "differenced",
                integrate_with(f, &y0, 0.0, t1, RadauIia::new(), &opts),
            ),
            (
                "analytic",
                integrate_with(f, &y0, 0.0, t1, RadauIia::new().with_jacobian(jac), &opts),
            ),
        ] {
            let r = r.unwrap();
            let want = [p(t1), 2.0 * p(t1)];
            for (y, w) in r.y.as_slice().iter().zip(want) {
                assert!(
                    (y - w).abs() <= 1e-12 * w.abs(),
                    "λ = {lambda}, {name}: {y} {w}"
                );
            }
            assert_eq!(r.rejected, 0, "λ = {lambda}, {name}");
            assert!(r.njev <= 2, "λ = {lambda}, {name}: {} Jacobians", r.njev);
        }
    }
}

#[test]
fn a_first_newton_iterate_is_not_accepted_without_a_rate_of_the_step() {
    // As above with a quartic p of a small quartic term δ = 10⁻⁸, the exact Jacobian and steps
    // held at 0.1: the collocation polynomial extrapolated from the step before misses the new
    // stages by some 5·δ·h⁴, about 5·10⁻⁶ tolerances — above the rounding of the state (2·10⁻⁹
    // here) and below the Newton tolerance κ = 10⁻³. The first correction is that, the second,
    // the problem being linear and the Jacobian exact, is rounding: every step takes two iterates
    // (six evaluations) and f at its end, seven in all. A first iterate accepted because its
    // correction is below κ — with no rate measured in the step to say how much error is left —
    // would make it four.
    let delta = 1e-8;
    let p = move |t: f64| 0.5 + t * (1.0 + t * (-0.3 + t * (0.05 + delta * t)));
    let dp = move |t: f64| 1.0 + t * (-0.6 + t * (0.15 + 4.0 * delta * t));
    let lambda = -1e3;
    let f = move |t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>| {
        dy[(0, 0)] = lambda * (y[(0, 0)] - p(t)) + dp(t);
    };
    let jac = move |_t: f64, _y: &Matrix<f64>, j: &mut Matrix<f64>| j[(0, 0)] = lambda;
    let t1 = miri_or(1.0, 5.0);
    let opts = OdeOptions::new()
        .with_tolerances(1e-6, 1e-9)
        .with_h0(0.1)
        .with_hmax(0.1);
    let method = RadauIia::new().with_jacobian(jac);
    let r = integrate_with(f, &col(&[p(0.0)]), 0.0, t1, method, &opts).unwrap();
    assert!((r.y[(0, 0)] - p(t1)).abs() < 1e-12, "{}", r.y[(0, 0)]);
    // One Jacobian, kept: no rate says otherwise. (The last 0.1, a hair over hmax in the rounding
    // of the time, is two steps of 0.05.)
    assert_eq!((r.rejected, r.njev), (0, 1));
    let later = r.nfev - first_step_evaluations(f, &col(&[p(0.0)]), &opts);
    assert_eq!(
        later,
        7 * (r.steps - 1),
        "{} evaluations, {} steps",
        r.nfev,
        r.steps
    );
}

/// The evaluations of `RadauIia` with the exact Jacobian up to the end of its first step: at the
/// start, then the iterates and the end of the step.
fn first_step_evaluations(
    f: impl Fn(f64, &Matrix<f64>, &mut Matrix<f64>) + Copy,
    y0: &Matrix<f64>,
    opts: &OdeOptions<f64>,
) -> usize {
    let jac = |_t: f64, _y: &Matrix<f64>, j: &mut Matrix<f64>| j[(0, 0)] = -1e3;
    let method = RadauIia::new().with_jacobian(jac);
    let r = integrate_with(f, y0, 0.0, opts.h0, method, opts).unwrap();
    assert_eq!(r.steps, 1);
    r.nfev
}

#[test]
fn a_newton_iteration_that_diverges_is_abandoned_and_the_step_retried() {
    // Van der Pol's oscillator at μ = 1000 with a first step of 100: the iteration from the
    // zero start on the strongly nonlinear system does not converge, and nothing it produced is
    // kept; the integration recovers and reaches the solution at t = 3000.
    let f = |_t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>| {
        dy[(0, 0)] = y[(1, 0)];
        dy[(1, 0)] = 1000.0 * (1.0 - y[(0, 0)] * y[(0, 0)]) * y[(1, 0)] - y[(0, 0)];
    };
    let t1 = miri_or(1.0, 3000.0);
    let opts = OdeOptions::new()
        .with_tolerances(1e-6, 1e-10)
        .with_h0(100.0);
    let s = solve_ivp_with(
        f,
        &col(&[2.0, 0.0]),
        0.0,
        t1,
        RadauIia::new(),
        &mut [],
        &opts,
    );
    assert_eq!(s.status, OdeStatus::Finished);
    assert!(s.rejected >= 3, "{}", s.rejected);
    let first = lengths(&s)[0];
    assert!(first < 1.0, "{first}");
    if t1 == 3000.0 {
        let want = REFERENCE
            .iter()
            .find(|r| r.0 == Stiff::VanDerPol)
            .unwrap()
            .1;
        let e = stiff_error(s.y.as_slice(), want, 1e-10);
        assert!(e < 1e-5, "{e}: {:?}", s.y);
    }
}

#[test]
fn a_jacobian_that_is_not_finite_ends_the_integration() {
    let f = |_t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>| dy[(0, 0)] = -y[(0, 0)];
    let jac = |_t: f64, _y: &Matrix<f64>, j: &mut Matrix<f64>| j[(0, 0)] = f64::NAN;
    let s = solve_ivp_with(
        f,
        &col(&[1.0]),
        0.0,
        1.0,
        RadauIia::new().with_jacobian(jac),
        &mut [],
        &OdeOptions::new(),
    );
    assert_eq!(s.status, OdeStatus::Failed(OdeFailure::NonFinite));
    assert_eq!((s.t, s.steps), (0.0, 0));
}

#[test]
fn the_stiff_method_works_in_f32() {
    let f = |t: f32, y: &Matrix<f32>, dy: &mut Matrix<f32>| {
        dy[(0, 0)] = -1000.0 * (y[(0, 0)] - t.cos())
    };
    for tolerance in [1e-3, 1e-5] {
        let opts = OdeOptions::<f32>::new().with_tolerances(tolerance, tolerance * 1e-2);
        let r = integrate_with(f, &col::<f32>(&[0.0]), 0.0, 2.0, RadauIia::new(), &opts).unwrap();
        let want = (1e6 * 2.0f64.cos() + 1000.0 * 2.0f64.sin()) / (1e6 + 1.0);
        let e = (f64::from(r.y[(0, 0)]) - want).abs();
        assert!(e < f64::from(tolerance), "{tolerance}: {e}");
        assert!(r.steps < 60, "{}", r.steps);
    }
    assert_eq!(RadauIia::<f32>::new().order(), 5);
}
