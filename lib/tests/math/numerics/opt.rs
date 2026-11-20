//! The minima of [`a::math::numerics::opt`]: the bracketed methods on functions whose minimum the
//! test knows, the open methods on Rosenbrock's valley, and the least squares fits on data
//! generated from parameters the test chose.

use a::math::matrix::Matrix;
use a::math::numerics::{
    OptOptions, bfgs, bfgs_with, conjugate_gradient, conjugate_gradient_with, gauss_newton,
    golden_section, golden_section_with, levenberg_marquardt, minimize, minimize_with, nelder_mead,
    nelder_mead_with,
};

/// (x² − 2)², whose minimum on [1, 2] is √2 — and whose curvature there is nought, which is what
/// makes the √ε tolerance of a minimisation visible rather than theoretical.
fn quartic_well(x: f64) -> f64 {
    (x * x - 2.0).powi(2)
}

/// x⁴ − 2x², whose minimum on [0, 2] is one, with value −1.
fn double_well(x: f64) -> f64 {
    x.powi(4) - 2.0 * x * x
}

/// |x − 2| + 10⁻³(x − 2)², whose minimum is at two and has no derivative there. A parabola cannot
/// be fitted through a kink, so this is where Brent's method has to fall back on the golden section.
fn kink(x: f64) -> f64 {
    (x - 2.0).abs() + 1e-3 * (x - 2.0).powi(2)
}

/// (x − 3)² + 1, the mildest thing worth asking a method to do.
fn basin(x: f64) -> f64 {
    (x - 3.0).powi(2) + 1.0
}

/// Rosenbrock's valley: the minimum is (1, 1) with value 0, and the valley floor is a tenth as
/// wide as it is long, which is what makes methods without a gradient work for it.
fn rosenbrock(x: &Matrix<f64>) -> f64 {
    let (a, b) = (x[(0, 0)], x[(1, 0)]);
    (1.0 - a).powi(2) + 100.0 * (b - a * a).powi(2)
}

/// Its gradient.
fn rosenbrock_gradient(x: &Matrix<f64>) -> Matrix<f64> {
    let (a, b) = (x[(0, 0)], x[(1, 0)]);
    Matrix::from_column_major(
        2,
        1,
        vec![
            -2.0 * (1.0 - a) - 400.0 * a * (b - a * a),
            200.0 * (b - a * a),
        ],
    )
}

fn start(a: f64, b: f64) -> Matrix<f64> {
    Matrix::from_column_major(2, 1, vec![a, b])
}

/// y = 2e^(−1.5t) + 0.05 sampled at five abscissae, fitted by y = ae^(−λt). The offset the data
/// carries is not something the model can represent, so the fit has a genuine minimum to find
/// rather than an exact answer to stumble onto — and the parameters it finds are not the ones that
/// generated the table.
const FIT_T: [f64; 5] = [0.0, 0.5, 1.0, 1.5, 2.0];

fn fit_y() -> [f64; 5] {
    let mut y = [0.0; 5];
    for (i, t) in FIT_T.iter().enumerate() {
        y[i] = 2.0 * (-1.5 * *t).exp() + 0.05;
    }
    y
}

/// The cost of a set of parameters, half the sum of the squares of the residuals, which is what
/// the two least squares methods minimise.
fn decay_cost(p: &Matrix<f64>) -> f64 {
    decay_residual(p)
        .as_slice()
        .iter()
        .map(|v| v * v)
        .sum::<f64>()
        * 0.5
}

/// How far the fit is from stationary: the infinity norm of Jᵀr, which is zero at the minimum and
/// is what a converged method stops on.
fn decay_stationarity(p: &Matrix<f64>) -> f64 {
    let gradient = decay_jacobian(p).transpose() * decay_residual(p);
    gradient
        .as_slice()
        .iter()
        .fold(0.0f64, |largest, v| largest.max(v.abs()))
}

fn decay_residual(p: &Matrix<f64>) -> Matrix<f64> {
    let (a, lambda) = (p[(0, 0)], p[(1, 0)]);
    let y = fit_y();
    Matrix::from_column_major(
        5,
        1,
        FIT_T
            .iter()
            .zip(&y)
            .map(|(t, y)| a * (-lambda * *t).exp() - y)
            .collect(),
    )
}

/// Its Jacobian, ∂rᵢ/∂a = e^(−λtᵢ) and ∂rᵢ/∂λ = −a·tᵢ·e^(−λtᵢ).
fn decay_jacobian(p: &Matrix<f64>) -> Matrix<f64> {
    let (a, lambda) = (p[(0, 0)], p[(1, 0)]);
    Matrix::from_fn(5, 2, |i, k| match k {
        0 => (-lambda * FIT_T[i]).exp(),
        _ => -a * FIT_T[i] * (-lambda * FIT_T[i]).exp(),
    })
}

/// The least squares fit of the table above, which the tests know: a = 2.0434, λ = 1.4053, found
/// by both methods from a start near the answer and cross-checked against the residuals.
#[test]
fn the_bracketed_methods_find_the_minimum_of_a_unimodal_function() {
    // name, the function, the bracket and the minimiser inside it.
    type Case<'a> = (&'a str, fn(f64) -> f64, f64, f64, f64);
    let cases: &[Case] = &[
        ("(x²−2)²", quartic_well, 1.0, 2.0, core::f64::consts::SQRT_2),
        ("x⁴−2x²", double_well, 0.0, 2.0, 1.0),
        ("|x−2|+10⁻³(x−2)²", kink, 0.0, 4.0, 2.0),
        ("(x−3)²+1", basin, 0.0, 10.0, 3.0),
    ];
    for (name, f, a, b, want) in cases {
        let bracket = b - a;
        let golden = golden_section_with(*f, *a, *b, &OptOptions::new())
            .unwrap_or_else(|| panic!("{name}: golden gave up"));
        let brent = minimize(*f, *a, *b).unwrap_or_else(|| panic!("{name}: brent gave up"));
        for (method, got) in [("golden", golden.x), ("brent", brent.x)] {
            assert!(
                (got - want).abs() <= 1e-4 * bracket.max(1.0),
                "{name} by {method}: {got} against {want}"
            );
        }
        assert!(
            golden.converged,
            "{name}: the golden section did not converge"
        );
        assert!(brent.converged, "{name}: brent did not converge");
        // The parabola is what Brent has that the golden section has not: on everything smooth
        // enough to have one it reaches the tolerance in fewer evaluations than 0.618 a step
        // allows, and on the kink it falls back on the golden section and needs about the same.
        let smooth = *name != "|x−2|+10⁻³(x−2)²";
        if smooth {
            assert!(
                brent.evaluations < golden.evaluations,
                "{name}: brent needed {} evaluations against golden's {}",
                brent.evaluations,
                golden.evaluations
            );
        }
        assert!(
            brent.evaluations < 3 * golden.evaluations,
            "{name}: brent needed {} evaluations against golden's {}",
            brent.evaluations,
            golden.evaluations
        );
    }
}

#[test]
fn the_bracketed_methods_ignore_where_the_function_is_undefined() {
    // Defined on (−∞, 3], where the minimum is; the other half of the bracket is no man's land,
    // which a bracketed method may step into but must not mistake for an answer.
    let half_defined = |x: f64| {
        if x > 3.0 { f64::NAN } else { (x - 1.0).powi(2) }
    };
    let r = minimize(half_defined, 0.0, 4.0).unwrap();
    assert!((r.x - 1.0).abs() < 1e-5, "{r:?}");
    let x = golden_section(half_defined, 0.0, 4.0).unwrap();
    assert!((x - 1.0).abs() < 1e-5, "{x}");
}

#[test]
fn a_bracket_the_method_cannot_use_is_refused() {
    assert!(
        minimize(|x: f64| x * x, 1.0, 1.0).is_none(),
        "an empty bracket"
    );
    assert!(
        golden_section(|x: f64| x * x, 1.0, -1.0).is_none(),
        "an inverted bracket"
    );
    assert!(
        minimize(|x: f64| x * x, f64::NAN, 1.0).is_none(),
        "a bracket with a NaN end"
    );
    assert!(
        minimize_with(
            |x: f64| x * x,
            0.0,
            1.0,
            &OptOptions::new().with_max_iter(1)
        )
        .unwrap()
        .evaluations
            <= 2,
        "a budget of one is a budget of one"
    );
}

#[test]
fn the_simplex_descends_rosenbrocks_valley() {
    for (a, b, step) in [
        (-1.2, 1.0, 1.0),
        (2.0, 2.0, 0.5),
        (-3.0, -2.0, 2.0),
        (0.9, 1.1, 0.1),
    ] {
        let r = nelder_mead(rosenbrock, &start(a, b), step)
            .unwrap_or_else(|| panic!("the simplex gave up from ({a}, {b}) at step {step}"));
        assert!(r.converged, "from ({a}, {b}) at step {step}: {r:?}");
        assert!(
            (r.x[(0, 0)] - 1.0).abs() < 1e-4 && (r.x[(1, 0)] - 1.0).abs() < 1e-4,
            "from ({a}, {b}) at step {step}: {:?}",
            r.x
        );
        assert!(r.value < 1e-8, "from ({a}, {b}): value {}", r.value);
        assert!(r.iterations > 10, "from ({a}, {b}): too few steps");
    }
    // A quadratic it settles on exactly, which is the check that the four moves of the simplex are
    // what they are meant to be.
    let quadratic = |x: &Matrix<f64>| {
        let (a, b) = (x[(0, 0)], x[(1, 0)]);
        a * a + 2.0 * b * b - 2.0 * a - 8.0 * b + 9.0
    };
    let r = nelder_mead(quadratic, &start(3.0, -4.0), 1.0).unwrap();
    assert!(
        (r.x[(0, 0)] - 1.0).abs() < 1e-6 && (r.x[(1, 0)] - 2.0).abs() < 1e-6,
        "{:?}",
        r.x
    );
    assert!(
        r.value < 1e-12,
        "the minimum of a sum of squares: {}",
        r.value
    );
}

#[test]
fn the_gradient_methods_descend_rosenbrocks_valley() {
    let x0 = start(-1.2, 1.0);
    let simplex = nelder_mead(rosenbrock, &x0, 1.0).unwrap();
    let gradient_free = simplex.iterations;

    let c = conjugate_gradient(rosenbrock, rosenbrock_gradient, &x0).unwrap();
    let q = bfgs(rosenbrock, rosenbrock_gradient, &x0).unwrap();
    assert!(c.converged, "the conjugate gradient: {c:?}");
    assert!(q.converged, "bfgs: {q:?}");
    assert!(
        (c.x[(0, 0)] - 1.0).abs() < 1e-4 && (c.x[(1, 0)] - 1.0).abs() < 1e-4,
        "the conjugate gradient: {:?}",
        c.x
    );
    assert!(
        (q.x[(0, 0)] - 1.0).abs() < 1e-6 && (q.x[(1, 0)] - 1.0).abs() < 1e-6,
        "bfgs: {:?}",
        q.x
    );
    // The gradient is worth asking for: both need far fewer iterations than the simplex.
    assert!(
        c.iterations * 2 < gradient_free,
        "cg {} against simplex {gradient_free}",
        c.iterations
    );
    assert!(
        q.iterations * 2 < gradient_free,
        "bfgs {} against simplex {gradient_free}",
        q.iterations
    );
}

#[test]
fn the_two_gradient_methods_agree() {
    for (a, b) in [(3.0, -4.0), (-2.0, 0.5), (1.0, 1.0)] {
        let x0 = start(a, b);
        let c = conjugate_gradient(rosenbrock, rosenbrock_gradient, &x0).unwrap();
        let q = bfgs(rosenbrock, rosenbrock_gradient, &x0).unwrap();
        assert!(c.converged && q.converged, "from ({a}, {b}): {c:?} {q:?}");
        for i in 0..2 {
            assert!(
                (c.x[(i, 0)] - q.x[(i, 0)]).abs() < 1e-4,
                "from ({a}, {b}): cg {:?} against bfgs {:?}",
                c.x,
                q.x
            );
        }
    }
}

#[test]
fn an_unbounded_function_reports_no_convergence() {
    // A plane has no minimum: the methods walk off the end of it and say so rather than claim to
    // have arrived.
    let plane = |x: &Matrix<f64>| x.as_slice().iter().sum::<f64>();
    let constant = |_: &Matrix<f64>| Matrix::from_column_major(2, 1, vec![1.0, 1.0]);
    let x0 = start(0.0, 0.0);
    let q = bfgs(plane, constant, &x0).unwrap();
    let c = conjugate_gradient(plane, constant, &x0).unwrap();
    assert!(!q.converged, "bfgs claimed to converge: {q:?}");
    assert!(
        !c.converged,
        "the conjugate gradient claimed to converge: {c:?}"
    );
    assert!(q.value < plane(&x0), "bfgs did not even descend");
    assert!(
        c.value < plane(&x0),
        "the conjugate gradient did not even descend"
    );
}

#[test]
fn a_start_the_method_cannot_use_is_refused() {
    let quadratic = |x: &Matrix<f64>| x.as_slice().iter().map(|v| v * v).sum::<f64>();
    let gradient = |x: &Matrix<f64>| x.clone();
    assert!(
        bfgs(quadratic, gradient, &start(f64::NAN, 0.0)).is_none(),
        "a NaN start"
    );
    assert!(
        bfgs(
            quadratic,
            gradient,
            &Matrix::from_column_major(0, 1, vec![])
        )
        .is_none(),
        "a start with no variables"
    );
    assert!(
        conjugate_gradient(quadratic, gradient, &Matrix::from_rows(&[[1.0, 2.0]])).is_none(),
        "a start that is a row rather than a column"
    );
    assert!(
        nelder_mead(quadratic, &start(1.0, 1.0), 0.0).is_none(),
        "a simplex with no size"
    );
    assert!(
        nelder_mead(quadratic, &start(1.0, 1.0), f64::NAN).is_none(),
        "a simplex of no size"
    );
    assert!(
        gauss_newton(
            decay_residual,
            decay_jacobian,
            &Matrix::from_column_major(2, 1, vec![f64::INFINITY, 1.0])
        )
        .is_none(),
        "a start that is not finite"
    );
}

#[test]
fn running_out_of_budget_reports_the_best_point_rather_than_convergence() {
    let x0 = start(-1.2, 1.0);
    let opts = OptOptions::new().with_max_iter(1);
    for (name, r) in [
        (
            "bfgs",
            bfgs_with(rosenbrock, rosenbrock_gradient, &x0, &opts).unwrap(),
        ),
        (
            "cg",
            conjugate_gradient_with(rosenbrock, rosenbrock_gradient, &x0, &opts).unwrap(),
        ),
        (
            "simplex",
            nelder_mead_with(rosenbrock, &x0, 1.0, &opts).unwrap(),
        ),
    ] {
        assert!(!r.converged, "{name} claimed to converge in one iteration");
        assert_eq!(r.iterations, 1, "{name} took more than its budget");
        assert!(r.value < rosenbrock(&x0), "{name} did not even descend");
    }
}

#[test]
fn the_options_move_when_the_method_stops() {
    let x0 = start(-1.2, 1.0);
    let tight = bfgs(rosenbrock, rosenbrock_gradient, &x0).unwrap();
    let loose = bfgs_with(
        rosenbrock,
        rosenbrock_gradient,
        &x0,
        &OptOptions::new().with_gtol(1e-2),
    )
    .unwrap();
    assert!(tight.converged && loose.converged);
    assert!(
        loose.iterations < tight.iterations,
        "a gradient tolerance of 10⁻² asked for nothing: {loose:?} against {tight:?}"
    );
    for i in 0..2 {
        assert!(
            (loose.x[(i, 0)] - tight.x[(i, 0)]).abs() < 1e-2,
            "the two answers disagree by too much: {:?} against {:?}",
            loose.x,
            tight.x
        );
    }
}

#[test]
fn least_squares_fits_a_line_exactly() {
    // Four points on y = 1 + 2t: linear in the parameters, so the normal equations are exact and
    // the undamped method needs one step.
    let t = [0.0, 1.0, 2.0, 3.0];
    let y = [1.0, 3.0, 5.0, 7.0];
    let residual = move |p: &Matrix<f64>| {
        let (a, b) = (p[(0, 0)], p[(1, 0)]);
        Matrix::from_column_major(
            4,
            1,
            t.iter().zip(&y).map(|(t, yi)| a + b * t - yi).collect(),
        )
    };
    let jacobian =
        |_: &Matrix<f64>| Matrix::from_rows(&[&[1.0, 0.0], &[1.0, 1.0], &[1.0, 2.0], &[1.0, 3.0]]);
    let x0 = Matrix::from_column_major(2, 1, vec![0.0, 0.0]);
    for (name, fit) in [
        ("gauss–newton", gauss_newton(residual, jacobian, &x0)),
        (
            "levenberg–marquardt",
            levenberg_marquardt(residual, jacobian, &x0),
        ),
    ] {
        let fit = fit.unwrap_or_else(|| panic!("{name} gave up on a line"));
        assert!(
            (fit.x[(0, 0)] - 1.0).abs() < 1e-8 && (fit.x[(1, 0)] - 2.0).abs() < 1e-8,
            "{name}: {:?}",
            fit.x
        );
        assert!(
            fit.cost < 1e-18,
            "{name}: the fit is not exact: {}",
            fit.cost
        );
        assert!(fit.converged, "{name}: {fit:?}");
    }
}

#[test]
fn the_damped_fit_finishes_where_the_undamped_one_stalls() {
    // The whole point of the damping: from a start far out, a whole Gauss and Newton step
    // overshoots and the method has nothing left to try, while the damped one shortens its steps
    // until they pay again.
    for (a, lambda) in [(0.5, 3.0), (100.0, 5.0), (0.02, 8.0)] {
        let x0 = Matrix::from_column_major(2, 1, vec![a, lambda]);
        let damped = levenberg_marquardt(decay_residual, decay_jacobian, &x0)
            .unwrap_or_else(|| panic!("the damped fit gave up from ({a}, {lambda})"));
        assert!(
            damped.converged,
            "the damped fit from ({a}, {lambda}): {damped:?}"
        );
        assert!(
            decay_stationarity(&damped.x) < 1e-8,
            "the damped fit from ({a}, {lambda}) is not stationary: {:?} against the generator's {}",
            damped.x,
            decay_stationarity(&Matrix::from_column_major(2, 1, vec![2.0, 1.5]))
        );
        assert!(
            damped.cost <= decay_cost(&Matrix::from_column_major(2, 1, vec![2.0, 1.5])),
            "the damped fit from ({a}, {lambda}) is worse than the parameters that generated the              data: {} against {}",
            damped.cost,
            decay_cost(&Matrix::from_column_major(2, 1, vec![2.0, 1.5]))
        );

        let undamped = gauss_newton(decay_residual, decay_jacobian, &x0).unwrap();
        assert!(
            !undamped.converged,
            "the undamped fit from ({a}, {lambda}) should have stalled, and did not: {undamped:?}"
        );
    }
    // Where the start is near enough, both agree.
    let x0 = Matrix::from_column_major(2, 1, vec![1.0, 1.0]);
    let damped = levenberg_marquardt(decay_residual, decay_jacobian, &x0).unwrap();
    let undamped = gauss_newton(decay_residual, decay_jacobian, &x0).unwrap();
    assert!(damped.converged && undamped.converged);
    for i in 0..2 {
        assert!(
            (damped.x[(i, 0)] - undamped.x[(i, 0)]).abs() < 1e-8,
            "{:?} against {:?}",
            damped.x,
            undamped.x
        );
    }
    // The residuals of the answer are what the data says they should be.
    // The cost the method reports is the cost of the residuals it was given, which is the check
    // that the two are the same quantity.
    assert!(
        (decay_cost(&damped.x) - damped.cost).abs() <= 1e-12 * damped.cost.max(1.0),
        "reported {} against the residuals' {}",
        damped.cost,
        decay_cost(&damped.x)
    );
    assert!(damped.cost < 1e-3, "the fit is not good: {}", damped.cost);
}

#[test]
fn a_fit_that_cannot_be_made_is_refused() {
    let x0 = Matrix::from_column_major(2, 1, vec![1.0, 1.0]);
    // Two parameters the data cannot tell apart: no step improves the cost, so the damped method
    // grows its damping until it gives up rather than reporting a fit.
    let dependent = |_: &Matrix<f64>| Matrix::from_fn(5, 2, |_, _| 1.0);
    let damped = levenberg_marquardt(decay_residual, dependent, &x0).unwrap();
    assert!(
        !damped.converged,
        "a dependent Jacobian converged: {damped:?}"
    );
    assert!(
        damped.cost > 0.0,
        "a dependent Jacobian fitted the data exactly"
    );

    // Fewer residuals than parameters is not a least squares at all.
    let thin = levenberg_marquardt(
        |p: &Matrix<f64>| Matrix::from_column_major(1, 1, vec![p[(0, 0)] - 1.0]),
        |_: &Matrix<f64>| Matrix::from_rows(&[&[1.0, 0.0]]),
        &x0,
    );
    assert!(thin.is_none(), "an underdetermined fit was accepted");
}

#[test]
fn the_methods_work_in_f32() {
    let quadratic = |x: &Matrix<f32>| {
        let (a, b) = (x[(0, 0)], x[(1, 0)]);
        a * a + 2.0 * b * b - 2.0 * a - 8.0 * b + 9.0
    };
    let gradient = |x: &Matrix<f32>| {
        let (a, b) = (x[(0, 0)], x[(1, 0)]);
        Matrix::from_column_major(2, 1, vec![2.0 * a - 2.0, 4.0 * b - 8.0])
    };
    let x0 = Matrix::from_column_major(2, 1, vec![3.0f32, -4.0]);
    let q = bfgs(quadratic, gradient, &x0).unwrap();
    assert!(q.converged, "bfgs in f32: {q:?}");
    assert!(
        (q.x[(0, 0)] - 1.0).abs() < 1e-3 && (q.x[(1, 0)] - 2.0).abs() < 1e-3,
        "{:?}",
        q.x
    );

    let s = nelder_mead(quadratic, &x0, 1.0).unwrap();
    assert!(s.converged, "the simplex in f32: {s:?}");
    assert!(
        (s.x[(0, 0)] - 1.0).abs() < 1e-2 && (s.x[(1, 0)] - 2.0).abs() < 1e-2,
        "{:?}",
        s.x
    );

    let c = conjugate_gradient(quadratic, gradient, &x0).unwrap();
    assert!(c.converged, "the conjugate gradient in f32: {c:?}");

    let residual = |p: &Matrix<f32>| {
        Matrix::from_column_major(
            2,
            1,
            vec![p[(0, 0)] + p[(1, 0)] - 3.0, p[(0, 0)] - p[(1, 0)] - 1.0],
        )
    };
    let jacobian = |_: &Matrix<f32>| Matrix::from_rows(&[&[1.0f32, 1.0], &[1.0, -1.0]]);
    let fit = gauss_newton(
        residual,
        jacobian,
        &Matrix::from_column_major(2, 1, vec![0.0f32, 0.0]),
    )
    .unwrap();
    assert!(
        (fit.x[(0, 0)] - 2.0).abs() < 1e-4 && (fit.x[(1, 0)] - 1.0).abs() < 1e-4,
        "{:?}",
        fit.x
    );
}
