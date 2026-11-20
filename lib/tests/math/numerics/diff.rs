//! The numerical derivatives of [`a::math::numerics::diff`], by Richardson extrapolation of
//! central differences, against derivatives the test knows in closed form.

use a::math::numerics::{
    DiffOptions, derivative, derivative_with, second_derivative, second_derivative_with,
};

#[test]
fn the_first_derivative_of_a_smooth_function_is_its_own() {
    // name, the point, the derivative there, the function and its derivative.
    type Case<'a> = (&'a str, f64, f64, fn(f64) -> f64, fn(f64) -> f64);
    let cases: &[Case] = &[
        ("sin", 1.0, 1.0_f64.cos(), |x| x.sin(), |x| x.cos()),
        ("exp", 0.5, 0.5f64.exp(), |x| x.exp(), |x| x.exp()),
        ("x³", 2.0, 12.0, |x| x.powi(3), |x| 3.0 * x * x),
        ("1/x", 4.0, -1.0 / 16.0, |x| 1.0 / x, |x| -1.0 / (x * x)),
        ("atan", 1.0, 0.5, |x| x.atan(), |x| 1.0 / (1.0 + x * x)),
    ];
    for (name, x, want, f, d) in cases {
        // The closed form in the table is the same one the test compares against, so a typo in
        // either is caught here rather than passing unnoticed.
        assert!(
            (d(*x) - want).abs() <= 1e-15 * want.abs().max(1.0),
            "{name}″ is wrong in the table at {x}"
        );
        let got = derivative(f, *x).unwrap();
        assert!(
            (got - want).abs() <= 1e-8 * want.abs().max(1.0),
            "{name}′ at {x}: {got} against {want}"
        );
    }
}

#[test]
fn the_second_derivative_is_the_derivative_of_the_derivative() {
    // name, the point, the second derivative there, the function.
    type Case<'a> = (&'a str, f64, f64, fn(f64) -> f64);
    let cases: &[Case] = &[
        ("sin", 1.0, -1.0_f64.sin(), |x| x.sin()),
        ("exp", 0.25, 0.25f64.exp(), |x| x.exp()),
        ("x⁶", 1.5, 30.0 * 1.5 * 1.5 * 1.5 * 1.5, |x| x.powi(6)),
        ("cos", 0.0, -1.0, |x| x.cos()),
    ];
    for (name, x, want, f) in cases {
        let got = second_derivative(f, *x).unwrap();
        assert!(
            (got - want).abs() <= 1e-6 * want.abs().max(1.0),
            "{name}″ at {x}: {got} against {want}"
        );
    }
}

#[test]
fn a_stationary_point_is_found_as_one() {
    // At the bottom of a parabola the derivative is exactly zero, and an extrapolation that is
    // off by a constant would show up here.
    assert!(derivative(|x: f64| x * x, 0.0).unwrap().abs() < 1e-12);
    assert!(derivative(|x: f64| x * x - 4.0 * x, 2.0).unwrap().abs() < 1e-10);
    assert!((second_derivative(|x: f64| x * x, 0.0).unwrap() - 2.0).abs() < 1e-6);
}

#[test]
fn a_linear_function_differentiates_to_its_slope_everywhere() {
    for x in [-1e4, -1.0, 0.0, 1.0, 3.0, 1e4] {
        let d = derivative(|t: f64| 7.0 - 3.0 * t, x).unwrap();
        assert!((d + 3.0).abs() <= 1e-9 * 3.0, "the slope at {x} is {d}");
    }
}

#[test]
fn the_options_choose_the_first_step_and_the_depth_of_the_tableau() {
    let opts = DiffOptions::new().with_step(1e-3);
    let d = derivative_with(|x: f64| x.sin(), 1.0, &opts).unwrap();
    assert!((d - 1.0_f64.cos()).abs() < 1e-9);
    // Two columns, accepted as soon as they agree to a thousandth: the right answer, but visibly
    // coarser than the full table's.
    let shallow = DiffOptions::new()
        .with_step(1e-2)
        .with_max_levels(2)
        .with_tol(1e-3);
    let rough = derivative_with(|x: f64| x.sin(), 1.0, &shallow).unwrap();
    let full = derivative(|x: f64| x.sin(), 1.0).unwrap();
    assert!((rough - 1.0_f64.cos()).abs() > (full - 1.0_f64.cos()).abs());
    // A single column has no other to be compared against, so nothing is claimed for it.
    let alone = DiffOptions::new().with_max_levels(1);
    assert!(derivative_with(|x: f64| x.sin(), 1.0, &alone).is_none());
    // A tolerance no table can satisfy simply stops.
    let impossible = DiffOptions::new().with_tol(0.0);
    assert!(derivative_with(|x: f64| x.sin(), 1.0, &impossible).is_none());
}

#[test]
fn a_second_derivative_with_options_agrees_with_the_plain_one() {
    let opts = DiffOptions::new().with_step(1e-2);
    let a = second_derivative_with(|x: f64| x.exp(), 0.0, &opts).unwrap();
    let b = second_derivative(|x: f64| x.exp(), 0.0).unwrap();
    assert!((a - 1.0).abs() < 1e-6);
    assert!((a - b).abs() < 1e-6);
    // A step far below the round-off floor of the second difference, where the numerator of the
    // quotient is two rounding of `1` apart, gives a table that never settles.
    let fine = DiffOptions::new().with_step(f64::EPSILON.sqrt());
    assert!(second_derivative_with(|x: f64| x.exp(), 0.0, &fine).is_none());
}

#[test]
fn a_steep_function_keeps_its_relative_accuracy() {
    // d/dx sin(1000x) at 0 is 1000: the scale has to come from the function, not from the point.
    let d = derivative(|x: f64| (1000.0 * x).sin(), 0.0).unwrap();
    assert!((d - 1000.0).abs() < 1e-6, "the slope is {d}");
    // And d/dx sin(x/1000) at 0 is 10⁻³.
    let d = derivative(|x: f64| (x / 1000.0).sin(), 0.0).unwrap();
    assert!((d - 1e-3).abs() < 1e-12, "the shallow slope is {d}");
}

#[test]
fn f32_derivatives_reach_f32_precision() {
    let d = derivative(|x: f32| x * x, 3.0f32).unwrap();
    assert!((f64::from(d) - 6.0).abs() < 1e-4);
    let d = derivative(|x: f32| x.sin(), 1.0f32).unwrap();
    assert!((f64::from(d) - 1.0_f64.cos()).abs() < 1e-4);
}
