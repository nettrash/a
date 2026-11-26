//! The solvers of `a::math::numerics` with exact derivatives from dual numbers: Newton's and
//! Halley's methods with `derivative` and `second_derivative`; BFGS, the conjugate gradient and
//! L-BFGS-B with `gradient`; Newton's method for systems and Broyden's with `jacobian`; Gauss–Newton
//! and Levenberg–Marquardt with the `jacobian` of the residual; the stiff methods for ODEs with the
//! `jacobian` of the right-hand side; and Newton's method for a minimum with `hessian`. Each function is written once, generic over `DualNum`.

use a::math::dual::{DualNum, derivative, gradient, hessian, jacobian, second_derivative};
use a::math::matrix::Matrix;
use a::math::numerics::{
    Bdf, LbfgsbOptions, LbfgsbStop, OdeOptions, OptOptions, RadauIia, RootOptions, SystemOptions,
    bfgs, bfgs_with, broyden_with, broyden_with_jacobian, conjugate_gradient,
    conjugate_gradient_with, gauss_newton, halley, integrate_with, lbfgsb, lbfgsb_with,
    levenberg_marquardt, newton, newton_system, newton_with,
};

fn column(v: &[f64]) -> Matrix<f64> {
    Matrix::from_column_major(v.len(), 1, v.to_vec())
}

fn rosen<D: DualNum<Float = f64>>(x: &[D]) -> D {
    let a = D::constant(1.0) - x[0];
    let b = x[1] - x[0] * x[0];
    a * a + b * b * 100.0
}

/// Rosenbrock's function in `n` dimensions, `Σ 100(xᵢ₊₁ − xᵢ²)² + (1 − xᵢ)²`.
fn extended_rosen<D: DualNum<Float = f64>>(x: &[D]) -> D {
    x.windows(2)
        .map(|w| {
            let b = w[1] - w[0] * w[0];
            let a = D::constant(1.0) - w[0];
            b * b * 100.0 + a * a
        })
        .sum()
}

/// Beale's function, whose minimum is 0 at (3, 1/2).
fn beale<D: DualNum<Float = f64>>(v: &[D]) -> D {
    let (x, y) = (v[0], v[1]);
    let t1 = D::constant(1.5) - x + x * y;
    let t2 = D::constant(2.25) - x + x * y * y;
    let t3 = D::constant(2.625) - x + x * y * y * y;
    t1 * t1 + t2 * t2 + t3 * t3
}

#[test]
fn newton_and_halley_with_exact_derivatives() {
    fn f<D: DualNum<Float = f64>>(x: D) -> D {
        x.exp() - x * x * 3.0
    }
    let by_dual = newton(1.0, f, |x| derivative(f, x)).unwrap();
    let by_hand = newton(1.0, f, |x: f64| x.exp() - 6.0 * x).unwrap();
    assert!((by_dual - 0.910_007_572_488_709).abs() < 1e-15);
    assert!((by_dual - by_hand).abs() <= 2.0 * f64::EPSILON);
    // Halley's cubic convergence needs f″ too: cos x − x from 0.5, and √2 in at most four steps.
    fn g<D: DualNum<Float = f64>>(x: D) -> D {
        x.cos() - x
    }
    let x = halley(0.5, g, |x| derivative(g, x), |x| second_derivative(g, x)).unwrap();
    assert!((x - 0.739_085_133_215_160_7).abs() < 1e-15);
    fn s<D: DualNum<Float = f64>>(x: D) -> D {
        x * x - 2.0
    }
    let opts = RootOptions::new().with_max_iter(4);
    let r = a::math::numerics::halley_with(
        1.0,
        s,
        |x| derivative(s, x),
        |x| second_derivative(s, x),
        &opts,
    );
    assert!((r.unwrap() - core::f64::consts::SQRT_2).abs() <= 4.0 * f64::EPSILON);
    // A double root, where Newton is linear, with a tolerance on the value.
    fn d<D: DualNum<Float = f64>>(x: D) -> D {
        (x - 1.0).powi(2)
    }
    let r = newton_with(
        3.0,
        d,
        |x| derivative(d, x),
        &RootOptions::new().with_ftol(1e-20),
    )
    .unwrap();
    assert!((r - 1.0).powi(2) <= 1e-20);
    // In f32.
    fn h<D: DualNum<Float = f32>>(x: D) -> D {
        x * x * x - 2.0
    }
    let c = newton(1.0_f32, h, |x| derivative(h, x)).unwrap();
    assert!((c - 2f32.cbrt()).abs() <= 2.0 * f32::EPSILON);
}

#[test]
fn bfgs_with_the_gradient() {
    let x0 = column(&[-1.2, 1.0]);
    let r = bfgs(|x| rosen(x.as_slice()), |x| gradient(rosen, x), &x0).unwrap();
    assert!(r.converged);
    assert!((r.x[(0, 0)] - 1.0).abs() < 1e-7 && (r.x[(1, 0)] - 1.0).abs() < 1e-7);
    let r = bfgs(
        |x| beale(x.as_slice()),
        |x| gradient(beale, x),
        &column(&[1.0, 1.0]),
    )
    .unwrap();
    assert!(r.converged);
    assert!((r.x[(0, 0)] - 3.0).abs() < 1e-6 && (r.x[(1, 0)] - 0.5).abs() < 1e-6);
    // With the gradient test tight, ‖∇f‖∞ at the answer is as the method documents.
    let opts = OptOptions::new().with_gtol(1e-10);
    let x0 = column(&[-1.2, 1.0, -1.2, 1.0]);
    let r = bfgs_with(
        |x| extended_rosen(x.as_slice()),
        |x| gradient(extended_rosen, x),
        &x0,
        &opts,
    )
    .unwrap();
    assert!(r.converged);
    assert!(
        gradient(extended_rosen, &r.x)
            .as_slice()
            .iter()
            .all(|g| g.abs() <= 1e-10)
    );
    assert!(r.x.as_slice().iter().all(|v| (v - 1.0).abs() < 1e-8));
}

#[test]
fn conjugate_gradient_with_the_gradient() {
    let x0 = column(&[-1.2, 1.0]);
    let r = conjugate_gradient(|x| rosen(x.as_slice()), |x| gradient(rosen, x), &x0).unwrap();
    assert!(r.converged);
    assert!((r.x[(0, 0)] - 1.0).abs() < 1e-7 && (r.x[(1, 0)] - 1.0).abs() < 1e-7);
    let opts = OptOptions::new().with_gtol(1e-9);
    let r = conjugate_gradient_with(
        |x| beale(x.as_slice()),
        |x| gradient(beale, x),
        &column(&[1.0, 1.0]),
        &opts,
    )
    .unwrap();
    assert!(r.converged);
    assert!(
        gradient(beale, &r.x)
            .as_slice()
            .iter()
            .all(|g| g.abs() <= 1e-9)
    );
    assert!((r.x[(0, 0)] - 3.0).abs() < 1e-7 && (r.x[(1, 0)] - 0.5).abs() < 1e-7);
}

#[test]
fn lbfgsb_with_the_gradient() {
    // Rosenbrock held to x₀ ≤ 0.5: the minimum on the bound is (0.5, 0.25).
    let lower = column(&[-2.0, -2.0]);
    let upper = column(&[0.5, 2.0]);
    let opts = LbfgsbOptions::new().with_pgtol(1e-10).with_ftol(0.0);
    let r = lbfgsb_with(
        |x| rosen(x.as_slice()),
        |x| gradient(rosen, x),
        &column(&[-1.2, 1.0]),
        &lower,
        &upper,
        &opts,
    )
    .unwrap();
    assert_eq!(r.stop, LbfgsbStop::ProjectedGradient);
    assert_eq!(r.x[(0, 0)], 0.5);
    assert!((r.x[(1, 0)] - 0.25).abs() < 1e-9);
    // Ten variables, unbounded.
    let n = 10;
    let inf = Matrix::from_column_major(n, 1, vec![f64::INFINITY; n]);
    let x0 = Matrix::from_fn(n, 1, |i, _| if i % 2 == 0 { -1.2 } else { 1.0 });
    let r = lbfgsb(
        |x| extended_rosen(x.as_slice()),
        |x| gradient(extended_rosen, x),
        &x0,
        &(-&inf),
        &inf,
    )
    .unwrap();
    assert!(r.converged);
    assert!(
        r.x.as_slice().iter().all(|v| (v - 1.0).abs() < 1e-4),
        "{:?}",
        r.x.as_slice()
    );
}

/// x + y + z = 6, x² + y² + z² = 14, xyz = 6: roots at the permutations of (1, 2, 3).
fn system<D: DualNum<Float = f64>>(v: &[D]) -> [D; 3] {
    let (x, y, z) = (v[0], v[1], v[2]);
    [
        x + y + z - 6.0,
        x * x + y * y + z * z - 14.0,
        x * y * z - 6.0,
    ]
}

fn residual(x: &Matrix<f64>) -> Matrix<f64> {
    column(&system(x.as_slice()))
}

#[test]
fn newton_system_and_broyden_with_the_jacobian() {
    let x0 = column(&[1.2, 1.7, 3.1]);
    let r = newton_system(&x0, residual, |x| jacobian(system, x)).unwrap();
    assert!(r.converged);
    for (got, want) in r.x.as_slice().iter().zip([1.0, 2.0, 3.0]) {
        assert!((got - want).abs() < 1e-14, "{:?}", r.x.as_slice());
    }
    // Broyden's method from the exact first Jacobian, as from one by differences.
    let opts = SystemOptions::new();
    let exact = broyden_with_jacobian(&x0, residual, &jacobian(system, &x0), &opts).unwrap();
    let differences = broyden_with(&x0, residual, &opts).unwrap();
    assert!(exact.converged && differences.converged);
    for (got, want) in exact.x.as_slice().iter().zip([1.0, 2.0, 3.0]) {
        assert!((got - want).abs() < 1e-12);
    }
    // The exact Jacobian at the root is the one of the system, [[1, 1, 1], [2x, 2y, 2z], [yz, xz, xy]].
    let j = jacobian(system, &column(&[1.0, 2.0, 3.0]));
    assert_eq!(
        j,
        Matrix::from_rows(&[[1.0, 1.0, 1.0], [2.0, 4.0, 6.0], [6.0, 3.0, 2.0]])
    );
}

#[test]
fn least_squares_with_the_jacobian_of_the_residual() {
    // y = a·e^(−λt) through points made from a = 2, λ = 1.5.
    let t: Vec<f64> = (0..8).map(|i| 0.25 * f64::from(i)).collect();
    let y: Vec<f64> = t.iter().map(|t| 2.0 * (-1.5 * t).exp()).collect();
    let model = |p: &[a::math::dual::Dual<f64>]| -> Vec<a::math::dual::Dual<f64>> {
        t.iter()
            .zip(&y)
            .map(|(&t, &y)| p[0] * (p[1] * -t).exp() - y)
            .collect()
    };
    let r = |p: &Matrix<f64>| {
        Matrix::from_column_major(
            8,
            1,
            t.iter()
                .zip(&y)
                .map(|(&t, &y)| p[(0, 0)] * (-p[(1, 0)] * t).exp() - y)
                .collect(),
        )
    };
    let fit = levenberg_marquardt(r, |p| jacobian(model, p), &column(&[1.0, 1.0])).unwrap();
    assert!(fit.converged);
    assert!((fit.x[(0, 0)] - 2.0).abs() < 1e-10 && (fit.x[(1, 0)] - 1.5).abs() < 1e-10);
    let fit = gauss_newton(r, |p| jacobian(model, p), &column(&[1.8, 1.4])).unwrap();
    assert!(fit.converged);
    assert!((fit.x[(0, 0)] - 2.0).abs() < 1e-10 && (fit.x[(1, 0)] - 1.5).abs() < 1e-10);
    // Rosenbrock as a sum of squares, (10(x₁ − x₀²), 1 − x₀), from the usual start.
    fn rr<D: DualNum<Float = f64>>(x: &[D]) -> [D; 2] {
        [(x[1] - x[0] * x[0]) * 10.0, D::constant(1.0) - x[0]]
    }
    let fit = levenberg_marquardt(
        |x| column(&rr(x.as_slice())),
        |x| jacobian(rr, x),
        &column(&[-1.2, 1.0]),
    )
    .unwrap();
    assert!(fit.converged);
    assert!((fit.x[(0, 0)] - 1.0).abs() < 1e-10 && (fit.x[(1, 0)] - 1.0).abs() < 1e-10);
}

#[test]
fn newton_for_a_minimum_with_the_hessian_converges_quadratically() {
    // x ← x − H⁻¹∇f on Rosenbrock from (−1.2, 1), the steps taken in full.
    let mut x = column(&[-1.2, 1.0]);
    let mut norms = Vec::new();
    for _ in 0..12 {
        let g = gradient(rosen, &x);
        let norm = g.as_slice().iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        norms.push(norm);
        if norm < 1e-13 {
            break;
        }
        let h = hessian(rosen, &x);
        x = &x - &h.solve(&g).unwrap();
    }
    assert!(
        (x[(0, 0)] - 1.0).abs() < 1e-14 && (x[(1, 0)] - 1.0).abs() < 1e-14,
        "{norms:?}"
    );
    // At the end the gradient squares a step: each norm below 1e−2 is at most 10³ times the
    // square of the one before.
    for w in norms.windows(2) {
        if w[0] < 1e-2 && w[1] > 1e-13 {
            assert!(w[1] <= 1e3 * w[0] * w[0], "{norms:?}");
        }
    }
    assert!(norms.len() <= 10, "{norms:?}");
}

#[test]
fn stiff_ode_methods_with_the_jacobian_of_the_right_hand_side() {
    // Robertson's reactions to t = 40 by Radau IIA and by the BDFs, each with the exact Jacobian
    // from dual numbers in place of differences. SciPy's Radau at rtol 1e−12 gives
    // y₁ = 0.7158270687194167 and y₃ = 0.2841637457458203 there.
    fn rhs<D: DualNum<Float = f64>>(y: &[D]) -> [D; 3] {
        let a = y[0] * -0.04 + y[1] * y[2] * 1e4;
        let c = y[1] * y[1] * 3e7;
        [a, -a - c, c]
    }
    let f = |_t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>| {
        for (k, v) in rhs(y.as_slice()).iter().enumerate() {
            dy[(k, 0)] = *v;
        }
    };
    let jac = |_t: f64, y: &Matrix<f64>, out: &mut Matrix<f64>| *out = jacobian(rhs, y);
    let y0 = column(&[1.0, 0.0, 0.0]);
    let opts = OdeOptions::new().with_tolerances(1e-8, 1e-12);
    let r = integrate_with(f, &y0, 0.0, 40.0, RadauIia::new().with_jacobian(jac), &opts).unwrap();
    let b = integrate_with(f, &y0, 0.0, 40.0, Bdf::new().with_jacobian(jac), &opts).unwrap();
    for s in [&r, &b] {
        assert!(
            (s.y[(0, 0)] - 0.715_827_068_719_416_7).abs() < 1e-6,
            "{:?}",
            s.y.as_slice()
        );
        assert!((s.y[(2, 0)] - 0.284_163_745_745_820_3).abs() < 1e-6);
        assert!((s.y.as_slice().iter().sum::<f64>() - 1.0).abs() < 1e-10);
        assert!(s.njev > 0);
    }
    // The Jacobian is the one of the reactions, exactly.
    let j = jacobian(rhs, &column(&[0.5, 2e-5, 0.3]));
    let want = Matrix::from_rows(&[
        [-0.04, 1e4 * 0.3, 1e4 * 2e-5],
        [0.04, -1e4 * 0.3 - 6e7 * 2e-5, -1e4 * 2e-5],
        [0.0, 6e7 * 2e-5, 0.0],
    ]);
    for (g, w) in j.as_slice().iter().zip(want.as_slice()) {
        assert!((g - w).abs() <= 4.0 * f64::EPSILON * w.abs(), "{g} {w}");
    }
}
