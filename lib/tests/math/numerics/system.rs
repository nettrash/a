//! The systems of [`a::math::numerics::system`]: Newton's method on an analytic Jacobian and
//! Broyden's method on the function alone.

use a::math::matrix::Matrix;
use a::math::numerics::{SystemOptions, broyden, broyden_with, newton_system, newton_system_with};

/// The circle of radius 2 about the origin and the line y = x − 1, whose intersection nearest
/// (1.5, 0.5) is ((1 + √7)/2, (√7 − 1)/2).
fn circle_and_line(x: &Matrix<f64>) -> Matrix<f64> {
    Matrix::from_column_major(
        2,
        1,
        vec![
            x[(0, 0)] * x[(0, 0)] + x[(1, 0)] * x[(1, 0)] - 4.0,
            x[(0, 0)] - x[(1, 0)] - 1.0,
        ],
    )
}

/// Its Jacobian.
fn circle_and_line_jacobian(x: &Matrix<f64>) -> Matrix<f64> {
    Matrix::from_rows(&[&[2.0 * x[(0, 0)], 2.0 * x[(1, 0)]], &[1.0, -1.0]])
}

const ROOT: [f64; 2] = [1.822_875_655_532_295_4, 0.822_875_655_532_295_2];

#[test]
fn newton_and_broyden_reach_the_same_root() {
    let x0 = Matrix::from_column_major(2, 1, vec![1.5, 0.5]);
    let n = newton_system(&x0, circle_and_line, circle_and_line_jacobian).unwrap();
    let b = broyden(&x0, circle_and_line).unwrap();
    for (got, want) in n.as_slice().iter().zip(&ROOT) {
        assert!((got - want).abs() < 1e-12, "newton: {got} against {want}");
    }
    for (got, want) in b.as_slice().iter().zip(&ROOT) {
        assert!((got - want).abs() < 1e-8, "broyden: {got} against {want}");
    }
}

#[test]
fn newton_solves_a_linear_system_from_any_start_in_one_step() {
    // A linear f is exactly linearised by its own Jacobian, so one Newton step lands on the
    // solution wherever the start is.
    let a = Matrix::from_rows(&[&[4.0, -1.0, 0.0], &[-1.0, 4.0, -1.0], &[0.0, -1.0, 3.0]]);
    let b = Matrix::from_column_major(3, 1, vec![1.0, 2.0, 3.0]);
    for start in [
        Matrix::from_column_major(3, 1, vec![0.0, 0.0, 0.0]),
        Matrix::from_column_major(3, 1, vec![10.0, -7.0, 3.5]),
    ] {
        let (af, aj, bf) = (a.clone(), a.clone(), b.clone());
        let x = newton_system(
            &start,
            move |x: &Matrix<f64>| &af * x - &bf,
            move |_: &Matrix<f64>| aj.clone(),
        )
        .unwrap();
        let residual = &a * &x - &b;
        assert!(
            residual.as_slice().iter().all(|&v| v.abs() < 1e-12),
            "the residual did not vanish: {:?}",
            residual.as_slice()
        );
    }
}

#[test]
fn a_system_with_a_known_answer_is_solved_to_its_tolerance() {
    // e^x + y = 3 and x² + y² = 4 have a root near (0.8346, 1.8291); both methods are started
    // there and must agree with each other far past what either claims on its own.
    let f = |x: &Matrix<f64>| {
        Matrix::from_column_major(
            2,
            1,
            vec![
                x[(0, 0)].exp() + x[(1, 0)] - 3.0,
                x[(0, 0)] * x[(0, 0)] + x[(1, 0)] * x[(1, 0)] - 4.0,
            ],
        )
    };
    let j = |x: &Matrix<f64>| {
        Matrix::from_rows(&[&[x[(0, 0)].exp(), 1.0], &[2.0 * x[(0, 0)], 2.0 * x[(1, 0)]]])
    };
    let x0 = Matrix::from_column_major(2, 1, vec![1.0, 1.5]);
    let n = newton_system(&x0, f, j).unwrap();
    let b = broyden(&x0, f).unwrap();
    for k in 0..2 {
        assert!(
            (n[(k, 0)] - b[(k, 0)]).abs() < 1e-7,
            "the methods disagree at {k}"
        );
    }
    let r = f(&n);
    assert!(
        r.as_slice().iter().all(|&v| v.abs() < 1e-12),
        "residual {:?}",
        r.as_slice()
    );
}

#[test]
fn a_singular_jacobian_answers_nothing() {
    // Both equations are the same plane: the Jacobian is singular everywhere and no answer can be
    // given, so the method must refuse rather than wander.
    let f = |x: &Matrix<f64>| {
        Matrix::from_column_major(
            2,
            1,
            vec![
                x[(0, 0)] + x[(1, 0)] - 1.0,
                2.0 * x[(0, 0)] + 2.0 * x[(1, 0)] - 2.0,
            ],
        )
    };
    let j = |_: &Matrix<f64>| Matrix::from_rows(&[[1.0, 1.0], [2.0, 2.0]]);
    let x0 = Matrix::from_column_major(2, 1, vec![0.0, 0.0]);
    assert!(newton_system(&x0, f, j).is_none());
    assert!(broyden(&x0, f).is_none());
}

#[test]
fn shapes_that_are_not_a_square_system_answer_nothing() {
    let f = |x: &Matrix<f64>| Matrix::from_column_major(2, 1, vec![x[(0, 0)], x[(1, 0)]]);
    let j = |_: &Matrix<f64>| Matrix::identity(2);
    // A row vector is not a state.
    let row = Matrix::from_rows(&[[0.0, 0.0]]);
    assert!(newton_system(&row, f, j).is_none());
    assert!(broyden(&row, f).is_none());
    // Nor is an empty one.
    let empty = Matrix::from_column_major(0, 1, vec![]);
    assert!(newton_system(&empty, f, j).is_none());
    // A function of the wrong dimension is refused as well.
    let wrong = |_: &Matrix<f64>| Matrix::from_column_major(3, 1, vec![0.0, 0.0, 0.0]);
    let x0 = Matrix::from_column_major(2, 1, vec![0.0, 0.0]);
    assert!(newton_system(&x0, wrong, j).is_none());
    // And so is a Jacobian of the wrong shape. The start must not already be a root, or the
    // method accepts it before it ever asks for a Jacobian.
    let bad_j = |_: &Matrix<f64>| Matrix::identity(3);
    let off = Matrix::from_column_major(2, 1, vec![1.0, 2.0]);
    assert!(newton_system(&off, f, bad_j).is_none());
}

#[test]
fn the_options_bound_the_work_and_the_tolerances() {
    let opts = SystemOptions::new().with_max_iter(1);
    let x0 = Matrix::from_column_major(2, 1, vec![1.5, 0.5]);
    // One step is not enough to converge on the circle, but it is enough to get near.
    let one = newton_system_with(&x0, circle_and_line, circle_and_line_jacobian, &opts);
    assert!(one.is_none());
    let many = SystemOptions::new().with_max_iter(50);
    let x = newton_system_with(&x0, circle_and_line, circle_and_line_jacobian, &many).unwrap();
    assert!((x[(0, 0)] - ROOT[0]).abs() < 1e-12);
    // A loose residual tolerance accepts a point that is not the root.
    let loose = SystemOptions::new().with_ftol(1e-2).with_max_iter(1);
    let quick = newton_system_with(&x0, circle_and_line, circle_and_line_jacobian, &loose);
    if let Some(q) = quick {
        assert!(
            circle_and_line(&q)
                .as_slice()
                .iter()
                .all(|&v| v.abs() < 1e-2)
        );
    }
    // Broyden's finite differences can be coarsened without losing the answer.
    let coarse = SystemOptions::new()
        .with_finite_difference_step(1e-3)
        .with_max_iter(200);
    let b = broyden_with(&x0, circle_and_line, &coarse).unwrap();
    assert!((b[(0, 0)] - ROOT[0]).abs() < 1e-5);
}

#[test]
fn f32_systems_solve_to_f32_precision() {
    let f = |x: &Matrix<f32>| {
        Matrix::from_column_major(
            2,
            1,
            vec![
                x[(0, 0)] * x[(0, 0)] + x[(1, 0)] * x[(1, 0)] - 4.0,
                x[(0, 0)] - x[(1, 0)] - 1.0,
            ],
        )
    };
    let j =
        |x: &Matrix<f32>| Matrix::from_rows(&[&[2.0 * x[(0, 0)], 2.0 * x[(1, 0)]], &[1.0, -1.0]]);
    let x0 = Matrix::from_column_major(2, 1, vec![1.5f32, 0.5]);
    let n = newton_system(&x0, f, j).unwrap();
    assert!((f64::from(n[(0, 0)]) - ROOT[0]).abs() < 1e-5);
    assert!((f64::from(n[(1, 0)]) - ROOT[1]).abs() < 1e-5);
}
