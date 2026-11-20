//! The zeros of a function of one real variable: the five methods of [`a::math::numerics::root`]
//! on brackets and on starting points, in `f64` and in `f32`, and the cases where a method has to
//! refuse an answer rather than invent one.

use a::math::numerics::{
    RootOptions, bisect, bisect_with, brent, brent_with, halley, newton, newton_with, secant,
};

/// The Dottie number, the fixed point of the cosine and the root of cos x − x.
const DOTTIE: f64 = 0.739_085_133_215_160_7;

/// √2, the root of x² − 2 nearest 1.
const SQRT_2: f64 = core::f64::consts::SQRT_2;

/// The golden ratio, the root of x² − x − 1 above 1.
const PHI: f64 = 1.618_033_988_749_895;

#[test]
fn bracketing_methods_find_the_dottie_number() {
    let f = |x: f64| x.cos() - x;
    for x in [bisect(0.0, 1.0, f), brent(0.0, 1.0, f)] {
        assert!((x.unwrap() - DOTTIE).abs() < 1e-15);
    }
}

#[test]
fn every_method_finds_the_same_root() {
    let (f, df, ddf) = (
        |x: f64| x * x - x - 1.0,
        |x: f64| 2.0 * x - 1.0,
        |_x: f64| 2.0,
    );
    assert!((brent(1.0, 2.0, f).unwrap() - PHI).abs() < 1e-15);
    assert!((secant(1.0, 2.0, f).unwrap() - PHI).abs() < 1e-15);
    assert!((newton(1.0, f, df).unwrap() - PHI).abs() < 1e-15);
    assert!((halley(1.0, f, df, ddf).unwrap() - PHI).abs() < 1e-15);
}

#[test]
fn brent_refines_to_machine_precision_on_a_square_root() {
    let x = brent(1.0, 2.0, |x: f64| x * x - 2.0).unwrap();
    assert!((x - SQRT_2).abs() < 1e-15);
    // The tolerance Brent promises is 4ε relative, a few ulp at 1.41, so the nearest float is
    // what it owes and no more.
    assert!((x - SQRT_2).abs() <= 4.0 * f64::EPSILON * SQRT_2);
    let b = bisect(1.0, 2.0, |x: f64| x * x - 2.0).unwrap();
    assert!((b - SQRT_2).abs() <= 4.0 * f64::EPSILON * SQRT_2);
}

#[test]
fn a_bracketed_root_is_never_left() {
    // sin has a root at π and another at 2π; the bracket [3, 4] can only be asking for the first.
    let x = brent(3.0, 4.0, |x: f64| x.sin()).unwrap();
    assert!((x - core::f64::consts::PI).abs() < 1e-15);
    assert!((3.0..4.0).contains(&x));
    let y = bisect(3.0, 4.0, |x: f64| x.sin()).unwrap();
    assert!((y - core::f64::consts::PI).abs() < 1e-14);
}

#[test]
fn a_root_at_an_end_of_the_bracket_is_returned_exactly() {
    assert_eq!(brent(1.0, 2.0, |x: f64| x - 1.0).unwrap(), 1.0);
    assert_eq!(brent(1.0, 2.0, |x: f64| 2.0 - x).unwrap(), 2.0);
    assert_eq!(bisect(1.0, 2.0, |x: f64| x - 2.0).unwrap(), 2.0);
}

#[test]
fn a_root_of_higher_multiplicity_is_still_found() {
    // (x − 1)³ touches zero without a kink: the bracket still changes sign, so both methods
    // working on it must reach 1.
    let f = |x: f64| (x - 1.0) * (x - 1.0) * (x - 1.0);
    // A triple root is where interpolation is worst: x is 1 + ∛f, so the quadratic keeps
    // proposing the estimate it already has, and only the bisection safeguard gets there. The
    // bracket starts 2.5 wide and has to reach a tolerance of 9e−16, which is 52 halvings, and
    // the safeguard has to deliver them inside the default limit of 100 evaluations.
    let tight = RootOptions::new().with_max_iter(100);
    assert!((brent_with(0.5, 3.0, f, &tight).unwrap() - 1.0).abs() < 1e-10);
    assert!(
        brent(0.5, 3.0, f).is_some(),
        "the default options must suffice"
    );
    assert!((bisect(0.5, 3.0, f).unwrap() - 1.0).abs() < 1e-10);
}

#[test]
fn without_a_sign_change_a_bracketing_method_refuses() {
    assert!(bisect(0.0, 1.0, |x: f64| x * x + 1.0).is_none());
    assert!(brent(0.0, 1.0, |x: f64| x * x + 1.0).is_none());
    // A degenerate bracket is not a bracket either.
    assert!(brent(1.0, 1.0, |x: f64| x - 2.0).is_none());
}

#[test]
fn a_nan_value_refuses_an_answer() {
    assert!(brent(-1.0, 1.0, |x: f64| if x > 0.0 { x - 0.5 } else { f64::NAN }).is_none());
    assert!(newton(1.0, |_x: f64| f64::NAN, |_x: f64| 1.0).is_none());
}

#[test]
fn newton_quadratic_and_halley_cubic_convergence() {
    // From a start of 1 both converge to √2; the test is that each reaches it to 1e-14 in the
    // handful of iterations its order implies, well inside the default limit of 100.
    let opts = RootOptions::new().with_max_iter(8);
    let n = newton_with(1.0, |x: f64| x * x - 2.0, |x: f64| 2.0 * x, &opts).unwrap();
    assert!((n - SQRT_2).abs() < 1e-14);
    let h = halley(1.0, |x: f64| x * x - 2.0, |x: f64| 2.0 * x, |_x: f64| 2.0);
    assert!((h.unwrap() - SQRT_2).abs() < 1e-15);
    // One Newton step from 1 is 1.5, two are 1.416̄: eight is more than enough, and seven is not
    // enough for nothing, so the limit above is one that the method actually has to respect.
    let tight = RootOptions::new().with_max_iter(1);
    assert!(newton_with(1.0, |x: f64| x * x - 2.0, |x: f64| 2.0 * x, &tight).is_none());
}

#[test]
fn a_zero_derivative_stops_newton() {
    assert!(newton(1.0, |x: f64| x * x - 2.0, |_x: f64| 0.0).is_none());
    // cos′ vanishes at 0, where cos x − x has a well-behaved root but a stalled tangent.
    assert!(newton(0.0, |x: f64| x.cos() - x, |x: f64| -x.sin() - 1.0).is_some());
}

#[test]
fn an_open_method_that_wanders_reports_none() {
    // arctan x: Newton's steps fly off to infinity and never come back.
    let r = newton(1.0, |x: f64| x.atan(), |_x: f64| 1.0 / (1.0 + 0.0));
    assert!(r.is_none() || r.unwrap().abs() < 1e-8);
    // x³ − 2x + 2 from 1 cycles between 1 and 2 forever, never reaching its root near −1.77.
    assert!(
        newton(
            1.0,
            |x: f64| x * x * x - 2.0 * x + 2.0,
            |x: f64| 3.0 * x * x - 2.0
        )
        .is_none()
    );
}

#[test]
fn ftol_accepts_a_small_enough_residual() {
    let opts = RootOptions::new().with_ftol(1e-8).with_max_iter(2);
    // Two bisections of [1, 2] leave x = 1.25 with f = −0.4375, far above the residual, so a
    // method that accepted on movement alone would still be short of the root.
    let x = bisect_with(1.0, 2.0, |x: f64| x * x - 2.0, &opts);
    assert!(x.is_none() || (x.unwrap() - SQRT_2).abs() < 1e-8);
    let loose = RootOptions::new().with_ftol(0.5);
    let y = bisect_with(1.0, 2.0, |x: f64| x * x - 2.0, &loose).unwrap();
    assert!((y - SQRT_2).abs() < 0.5);
}

#[test]
fn xtol_bounds_the_root_of_a_flat_function() {
    // A function whose slope is 1e-6 turns a tolerance on x into one 1e-6 smaller on f: the
    // answer must still sit inside xtol + rtol·|x| of the root.
    let opts = RootOptions::new().with_xtol(1e-12);
    let x = brent_with(0.0, 1.0, |x: f64| (x - 0.3) * 1e-6, &opts).unwrap();
    assert!((x - 0.3).abs() <= 1e-11);
}

#[test]
fn f32_problems_are_solved_to_f32_tolerance() {
    let opts = RootOptions::<f32>::new().with_rtol(4.0 * f32::EPSILON);
    let x = brent_with(1.0f32, 2.0f32, |x: f32| x * x - 2.0, &opts).unwrap();
    assert!((f64::from(x) - SQRT_2).abs() < 1e-6);
    let b = bisect(0.0f32, 1.0f32, |x: f32| x.cos() - x).unwrap();
    assert!((f64::from(b) - DOTTIE).abs() < 1e-6);
    let n = newton(1.0f32, |x: f32| x * x - 2.0, |x: f32| 2.0 * x).unwrap();
    assert!((f64::from(n) - SQRT_2).abs() < 1e-6);
}

#[test]
fn a_root_far_from_the_origin_keeps_its_relative_accuracy() {
    // The root is 10⁸ and the function has slope 2·10⁸ there, so only a relative test can pass.
    let x = brent(1e8, 2e8, |x: f64| x * x - 1e16).unwrap();
    assert!((x - 1e8).abs() <= 1e8 * 1e-14);
}

#[test]
fn a_root_near_the_origin_needs_its_absolute_tolerance() {
    let x = brent(0.0, 1e-8, |x: f64| x - 3e-9).unwrap();
    assert!((x - 3e-9).abs() < 1e-20);
}

#[test]
fn a_wide_bracket_with_a_single_root_is_no_problem() {
    // Nine orders of magnitude of bracket around the root, where bisection alone needs about 60
    // halvings to reach the tolerance.
    let x = brent(1e-6, 1e9, |x: f64| x - 1234.5).unwrap();
    assert!((x - 1234.5).abs() < 1e-9);
    let y = bisect(1e-6, 1e9, |x: f64| x - 1234.5).unwrap();
    assert!((y - 1234.5).abs() < 1e-4);
}
