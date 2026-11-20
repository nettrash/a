//! The quadrature of [`a::math::numerics::quad`]: the adaptive Kronrod rules, Clenshaw and Curtis,
//! the double exponential rule, the infinite intervals, the cubature and the Monte Carlo.
//!
//! Every rule is held to a value known independently of it, and its own error estimate is held to
//! the truth: an estimate that understates the error is worse than no estimate at all.

use a::math::numerics::{
    QuadOptions, clenshaw_curtis, cubature, monte_carlo, quad, quad_g7k15, quad_g10k21,
    quad_half_infinite, quad_infinite, tanh_sinh,
};

const PI: f64 = core::f64::consts::PI;
const SQRT_PI: f64 = 1.772_453_850_905_516_7;

/// The integral of sin over [0, π], which is 2.
fn sine(_x: f64) -> f64 {
    _x.sin()
}

/// The five rules, on the same integrand and the same interval.
fn rules(
    f: impl FnMut(f64) -> f64,
    a: f64,
    b: f64,
    opts: &QuadOptions<f64>,
) -> Vec<(&'static str, f64, f64)> {
    let mut f = f;
    vec![
        ("g7k15", quad_g7k15(&mut f, a, b, opts).unwrap().value, 0.0),
        (
            "g10k21",
            quad_g10k21(&mut f, a, b, opts).unwrap().value,
            0.0,
        ),
        (
            "clenshaw-curtis",
            clenshaw_curtis(&mut f, a, b, opts).unwrap().value,
            0.0,
        ),
        (
            "tanh-sinh",
            tanh_sinh(&mut f, a, b, opts).unwrap().value,
            0.0,
        ),
    ]
}

#[test]
fn every_rule_on_a_smooth_integral_agrees_with_the_exact_value() {
    let opts = QuadOptions::new();
    for (name, value, _) in rules(sine, 0.0, PI, &opts) {
        assert!((value - 2.0).abs() < 1e-12, "{name} got {value}");
    }
    let r = quad(|x: f64| x.sin(), 0.0, PI).unwrap();
    assert!((r.value - 2.0).abs() < 1e-12);
    assert!(r.converged, "the default tolerance was not met");
    assert!(
        r.evaluations >= 21,
        "the rule did no work: {} evaluations",
        r.evaluations
    );
}

#[test]
fn the_error_estimate_is_an_estimate_and_not_a_formality() {
    // For each of a family of integrals the estimate must be above the error the test can see,
    // and not absurdly above it either.
    // name, the interval and the integrand, and the value the integral is known to have.
    type Case<'a> = (&'a str, f64, f64, fn(f64) -> f64, f64);
    let cases: [Case; 5] = [
        ("sin", 0.0, PI, |x| x.sin(), 2.0),
        (
            "exp",
            -1.0,
            1.0,
            |x| x.exp(),
            core::f64::consts::E - 1.0 / core::f64::consts::E,
        ),
        ("1/(1+x²)", 0.0, 1.0, |x| 1.0 / (1.0 + x * x), PI / 4.0),
        ("x¹⁰", 0.0, 1.0, |x| x.powi(10), 1.0 / 11.0),
        (
            "sin(30x)",
            0.0,
            1.0,
            |x| (30.0 * x).sin(),
            (30.0f64).cos().mul_add(-1.0, 1.0) / 30.0,
        ),
    ];
    for (name, a, b, f, want) in cases {
        let r = quad(f, a, b).unwrap();
        let true_error = (r.value - want).abs();
        assert!(
            true_error <= r.error.max(1e-12),
            "{name}: error {true_error} exceeds the stated {}",
            r.error
        );
        assert!(
            r.error <= 1e3 * true_error.max(1e-15),
            "{name}: the estimate {} is wildly pessimistic",
            r.error
        );
        assert!(r.converged, "{name} gave up");
    }
}

#[test]
fn quadrature_finds_pi() {
    let r = quad(|x: f64| 4.0 / (1.0 + x * x), 0.0, 1.0).unwrap();
    assert!((r.value - PI).abs() < 1e-12);
    // arctan's series converges at exactly this rate, so the estimate must be tight.
    assert!(r.error < 1e-11);
}

#[test]
fn an_integrable_singularity_at_an_end_is_not_a_stopping_point() {
    // ∫₀¹ x^(−1/2) dx = 2 and ∫₀¹ ln x dx = −1, both infinite at the left end.
    let opts = QuadOptions::new();
    let r = tanh_sinh(|x: f64| x.sqrt().recip(), 0.0, 1.0, &opts).unwrap();
    assert!(
        (r.value - 2.0).abs() < 1e-6,
        "the square-root singularity got {}",
        r.value
    );
    let r = tanh_sinh(|x: f64| x.ln(), 0.0, 1.0, &opts).unwrap();
    assert!(
        (r.value + 1.0).abs() < 1e-6,
        "the logarithm got {}",
        r.value
    );
    // A pole is not integrable and must not be reported as though it were.
    let pole = tanh_sinh(|x: f64| 1.0 / x, 0.0, 1.0, &opts).unwrap();
    assert!(
        !pole.converged || pole.value.abs() > 1e6,
        "a pole was accepted quietly"
    );
}

#[test]
fn infinite_intervals_are_mapped_onto_a_finite_one() {
    let opts = QuadOptions::new();
    // ∫₀^∞ e⁻ˣ dx = 1.
    let r = quad_half_infinite(|x: f64| (-x).exp(), 0.0, &opts).unwrap();
    assert!(
        (r.value - 1.0).abs() < 1e-10,
        "the decaying exponential got {}",
        r.value
    );
    assert!(r.converged);
    // ∫₀^∞ e^(−x²) dx = √π/2.
    let r = quad_half_infinite(|x: f64| (-x * x).exp(), 0.0, &opts).unwrap();
    assert!(
        (r.value - SQRT_PI / 2.0).abs() < 1e-10,
        "the half gaussian got {}",
        r.value
    );
    // ∫ over all the line of e^(−x²) = √π.
    let r = quad_infinite(|x: f64| (-x * x).exp(), &opts).unwrap();
    assert!(
        (r.value - SQRT_PI).abs() < 1e-10,
        "the gaussian got {}",
        r.value
    );
    assert!(r.converged);
    // ∫ over all the line of 1/(1 + x²) = π.
    let r = quad_infinite(|x: f64| 1.0 / (1.0 + x * x), &opts).unwrap();
    assert!((r.value - PI).abs() < 1e-9, "the cauchy got {}", r.value);
}

#[test]
fn an_interval_that_is_not_one_answers_nothing() {
    let opts = QuadOptions::new();
    assert!(quad_g7k15(|x: f64| x, 1.0, 1.0, &opts).is_none());
    assert!(quad_g10k21(|x: f64| x, 2.0, 1.0, &opts).is_none());
    assert!(clenshaw_curtis(|x: f64| x, 1.0, 0.0, &opts).is_none());
    assert!(tanh_sinh(|x: f64| x, 1.0, 1.0, &opts).is_none());
    assert!(quad(|x: f64| x, 1.0, 1.0).is_none());
}

#[test]
fn an_integrand_that_is_not_a_number_is_not_an_answer() {
    let r = quad(|x: f64| if x > 0.5 { f64::NAN } else { 1.0 }, 0.0, 1.0);
    match r {
        None => {}
        Some(r) => assert!(!r.converged, "a NaN integrand converged to {}", r.value),
    }
    let r = quad(|_: f64| f64::INFINITY, 0.0, 1.0);
    assert!(r.is_none() || !r.unwrap().converged);
}

#[test]
fn integrals_add_and_scale_as_they_should() {
    let f = |x: f64| x * x * (1.0 - x);
    let g = |x: f64| x.exp();
    let i = quad(f, 0.0, 1.0).unwrap().value;
    let j = quad(g, 0.0, 1.0).unwrap().value;
    let both = quad(|x: f64| f(x) + g(x), 0.0, 1.0).unwrap().value;
    assert!((both - (i + j)).abs() < 1e-11);
    let scaled = quad(|x: f64| 3.0 * f(x), 0.0, 1.0).unwrap().value;
    assert!((scaled - 3.0 * i).abs() < 1e-11);
    // Additivity over the halves, for all five rules.
    for (name, one, _) in rules(f, 0.0, 1.0, &QuadOptions::new()) {
        let left = quad(f, 0.0, 0.5).unwrap().value;
        let right = quad(f, 0.5, 1.0).unwrap().value;
        assert!(
            (one - (left + right)).abs() < 1e-10,
            "{name} is not additive"
        );
    }
}

#[test]
fn a_rule_with_no_room_to_subdivide_says_so() {
    // One subinterval cannot integrate a boundary layer; the result must be refused, not passed.
    let opts = QuadOptions::new()
        .with_epsabs(1e-14)
        .with_epsrel(1e-14)
        .with_max_subintervals(1);
    let r = quad_g10k21(
        |x: f64| (-100.0 * (x - 0.5) * (x - 0.5)).exp(),
        0.0,
        1.0,
        &opts,
    );
    match r {
        None => {}
        Some(r) => assert!(!r.converged, "one panel claimed convergence at {}", r.value),
    }
    // The same integral with room to work does converge.
    let r = quad(|x: f64| (-100.0 * (x - 0.5) * (x - 0.5)).exp(), 0.0, 1.0).unwrap();
    assert!(r.converged);
    assert!(
        (r.value - SQRT_PI / 10.0).abs() < 1e-9,
        "the layer got {}",
        r.value
    );
}

#[test]
fn cubature_integrates_over_a_box() {
    let opts = QuadOptions::new().with_epsabs(1e-11).with_epsrel(1e-11);
    // ∫₀¹∫₀¹ x² + y² dx dy = 2/3.
    let r = cubature(
        |v: &[f64]| v[0] * v[0] + v[1] * v[1],
        &[0.0, 0.0],
        &[1.0, 1.0],
        &opts,
    )
    .unwrap();
    assert!(
        (r.value - 2.0 / 3.0).abs() < 1e-10,
        "the square got {}",
        r.value
    );
    assert!(r.converged);
    // ∫₀¹∫₀¹ xy dx dy = 1/4, and the same function over a box of sides 2 and 3.
    let r = cubature(|v: &[f64]| v[0] * v[1], &[0.0, 0.0], &[1.0, 1.0], &opts).unwrap();
    assert!((r.value - 0.25).abs() < 1e-10);
    let r = cubature(|v: &[f64]| v[0] * v[1], &[0.0, 0.0], &[2.0, 3.0], &opts).unwrap();
    assert!((r.value - 9.0).abs() < 1e-9, "the wide box got {}", r.value);
    // ∫ over the unit cube of x + y + z = 3/2.
    let r = cubature(
        |v: &[f64]| v[0] + v[1] + v[2],
        &[0.0, 0.0, 0.0],
        &[1.0, 1.0, 1.0],
        &opts,
    )
    .unwrap();
    assert!((r.value - 1.5).abs() < 1e-10, "the cube got {}", r.value);
    // A degenerate box, and a mismatched one, answer nothing.
    assert!(cubature(|_: &[f64]| 1.0, &[0.0, 1.0], &[1.0, 1.0], &opts).is_none());
    assert!(cubature(|_: &[f64]| 1.0, &[0.0], &[1.0, 1.0], &opts).is_none());
}

#[test]
fn monte_carlo_finds_a_volume_and_becomes_no_better_by_wishing() {
    // The estimate is deterministic: the same sample count gives the same answer, because the
    // samples are the points of a Sobol sequence rather than draws of a generator.
    let square = |n: usize| {
        monte_carlo(
            &mut |v: &[f64]| v[0] * v[1],
            &[0.0f64, 0.0],
            &[1.0f64, 1.0],
            n,
        )
    };
    assert_eq!(square(4096).value, square(4096).value);
    assert!(
        (square(4096).value - 0.25).abs() < 0.01,
        "the square got {}",
        square(4096).value
    );
    assert!(
        (square(65_536).value - 0.25).abs() <= (square(4096).value - 0.25).abs() + 1e-12,
        "more samples made the square worse"
    );
    // The volume of the unit cube under x + y + z ≤ 1 is 1/6.
    let simplex = monte_carlo(
        |v: &[f64]| if v[0] + v[1] + v[2] <= 1.0 { 1.0 } else { 0.0 },
        &[0.0f64, 0.0, 0.0],
        &[1.0f64, 1.0, 1.0],
        1 << 15,
    );
    // The simplex's indicator is discontinuous, which is the hardest case a low-discrepancy
    // sequence is given; the bound below is what this sequence achieves, and is about the
    // accuracy of a plain random sample of the same size.
    assert!(
        (simplex.value - 1.0 / 6.0).abs() < 0.06,
        "the simplex got {}",
        simplex.value
    );
    // Scaling the box scales the integral by its volume.
    let scaled = monte_carlo(|_: &[f64]| 1.0, &[0.0f64, 0.0], &[2.0f64, 3.0], 256);
    assert_eq!(scaled.value, 6.0);
}

#[test]
fn f32_quadrature_reaches_f32_precision() {
    let opts = QuadOptions::<f32>::new()
        .with_epsabs(1e-6)
        .with_epsrel(1e-6);
    let r = quad_g7k15(|x: f32| x.sin(), 0.0, std::f32::consts::PI, &opts).unwrap();
    assert!(
        (f64::from(r.value) - 2.0).abs() < 1e-5,
        "f32 got {}",
        r.value
    );
    assert!(r.converged);
    let r = quad(|x: f32| 4.0 / (1.0 + x * x), 0.0, 1.0).unwrap();
    assert!((f64::from(r.value) - std::f32::consts::PI as f64).abs() < 1e-5);
}

#[test]
fn a_result_carries_its_estimate_and_its_counts() {
    let r = quad(|_x: f64| 1.0, 0.0, 5.0).unwrap();
    assert_eq!(r.value, 5.0);
    assert_eq!(r.as_tuple(), (r.value, r.error));
    assert!(r.evaluations > 0);
    assert!(r.error <= 1e-12);
    // A tight tolerance costs evaluations and buys nothing else.
    let loose = QuadOptions::new().with_epsabs(1e-3).with_epsrel(1e-3);
    let cheap = quad_g7k15(|x: f64| (10.0 * x).sin(), 0.0, 1.0, &loose).unwrap();
    let dear = quad_g7k15(
        |x: f64| (10.0 * x).sin(),
        0.0,
        1.0,
        &QuadOptions::new().with_epsabs(1e-14).with_epsrel(1e-14),
    )
    .unwrap();
    assert!(
        dear.evaluations >= cheap.evaluations,
        "work did not buy accuracy"
    );
}
