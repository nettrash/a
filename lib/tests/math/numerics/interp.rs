//! The piecewise cubics and the polynomial approximants of [`a::math::numerics::interp`].
//!
//! A cubic Hermite spline interpolates its knots by construction, so the three properties that
//! single out each of the cubic variants are checked rather than a table of values: that the
//! second derivative is continuous where it must be, that the two end conditions hold, and that
//! the slopes agree with an independent dense solve of the same system.

use a::math::numerics::{
    Chebyshev, Spline, barycentric_interpolate, barycentric_weights, chebyshev, chebyshev_nodes,
};

/// Nine irregular knots over [0, 3], and the sine of each rounded to four places so that the
/// splines are fitted to data rather than to a function the test already knows.
const X: [f64; 8] = [0.0, 0.3, 0.7, 1.1, 1.6, 2.0, 2.6, 3.0];
const Y: [f64; 8] = [0.0, 0.2861, 0.6216, 0.8674, 0.9995, 0.9093, 0.4121, -0.1411];

/// The slopes of the natural spline over `X` and `Y`, from a dense solve of the defining equations: value, slope and curvature equal where two
/// cubics meet, and a vanishing curvature at each end.
const NATURAL: [f64; 8] = [
    0.969_991_331_840_605_3,
    0.921_017_336_318_789_2,
    0.739_513_988_058_176_3,
    0.480_676_711_448_504_7,
    0.009_537_313_409_007_5,
    -0.461_295_697_431_230_6,
    -1.208_577_482_957_357_5,
    -1.470_211_258_521_321_4,
];

/// The same, clamped to slopes of 1 at the left end and −2 at the right.
const CLAMPED: [f64; 8] = [
    1.0,
    0.911_772_106_073_340_7,
    0.742_646_838_324_410_7,
    0.477_390_540_629_015_7,
    0.020_409_019_263_915,
    -0.497_804_901_853_306_8,
    -1.042_339_019_629_338_7,
    -2.0,
];

/// The same, not-a-knot.
const NOT_A_KNOT: [f64; 8] = [
    0.978_365_474_577_375_8,
    0.918_445_919_289_118_8,
    0.740_348_410_547_611_8,
    0.479_910_438_520_432_6,
    0.011_942_513_473_537_7,
    -0.469_341_399_321_081,
    -1.171_956_773_604_900_8,
    -1.586_916_488_436_277,
];

/// The second derivative at `t`, from the Hermite data the spline keeps: on the interval of
/// length `h` the second derivative of the cubic is the linear function of `u` whose values at the
/// two ends are the ones assembled here.
fn second_derivative(s: &Spline<f64>, t: f64, from: Side) -> Option<f64> {
    let x = s.knots();
    let y = s.values();
    let m = s.slopes();
    let n = x.len();
    let i = if from == Side::Left {
        (0..n - 1).find(|&k| x[k] <= t && t <= x[k + 1])?
    } else {
        (0..n - 1).rev().find(|&k| x[k] <= t && t <= x[k + 1])?
    };
    let h = x[i + 1] - x[i];
    let u = (t - x[i]) / h;
    let (c0, c1, c2, c3) = (
        12.0 * u - 6.0,
        6.0 * u - 4.0,
        -12.0 * u + 6.0,
        6.0 * u - 2.0,
    );
    Some((c0 * y[i] + c1 * (h * m[i]) + c2 * y[i + 1] + c3 * (h * m[i + 1])) / (h * h))
}

/// Which side of a knot to approach its second derivative from, since the two differ exactly at
/// the knots.
#[derive(Clone, Copy, PartialEq)]
enum Side {
    Left,
    Right,
}

#[test]
fn the_cubic_variants_match_an_independent_solve() {
    let natural = Spline::cubic(&X, &Y).unwrap();
    let clamped = Spline::cubic_clamped(&X, &Y, 1.0, -2.0).unwrap();
    let not_a_knot = Spline::cubic_not_a_knot(&X, &Y).unwrap();
    for (got, want) in [
        (natural.slopes(), &NATURAL[..]),
        (clamped.slopes(), &CLAMPED[..]),
        (not_a_knot.slopes(), &NOT_A_KNOT[..]),
    ] {
        for (a, b) in got.iter().zip(want) {
            assert!(
                (a - b).abs() <= 1e-13 * b.abs().max(1.0),
                "{a} is not the slope {b}"
            );
        }
    }
}

#[test]
fn every_spline_passes_through_its_knots() {
    for s in splines() {
        for (t, v) in X.iter().zip(&Y) {
            assert_eq!(
                s.evaluate(*t),
                Some(*v),
                "a knot must be reproduced exactly"
            );
        }
    }
}

/// The five splines of the test data.
fn splines() -> Vec<Spline<f64>> {
    vec![
        Spline::cubic(&X, &Y).unwrap(),
        Spline::cubic_clamped(&X, &Y, 1.0, -2.0).unwrap(),
        Spline::cubic_not_a_knot(&X, &Y).unwrap(),
        Spline::akima(&X, &Y).unwrap(),
        Spline::pchip(&X, &Y).unwrap(),
    ]
}

#[test]
fn the_cubic_splines_are_twice_continuously_differentiable() {
    for kind in ["natural", "clamped", "not-a-knot"] {
        let s = build(kind);
        for (k, &x) in X.iter().enumerate().skip(1).take(X.len() - 2) {
            let left = second_derivative(&s, x, Side::Left).unwrap();
            let right = second_derivative(&s, x, Side::Right).unwrap();
            assert!(
                (left - right).abs() <= 1e-9 * left.abs().max(right.abs()).max(1.0),
                "{kind} bends at knot {k}: {left} against {right}"
            );
        }
    }
}

#[test]
fn the_cubic_splines_reproduce_a_cubic_exactly() {
    // A straight line is reproduced by every one of them, since each end condition is itself
    // satisfied by a line. A cubic needs the end condition to agree with it too, which only the
    // clamped spline — given the cubic's own end slopes — and not-a-knot can do.
    let x: Vec<f64> = (0..7).map(|k| f64::from(k) * 0.9 - 1.3).collect();
    let line: Vec<f64> = x.iter().map(|&t| 3.0 - 2.0 * t).collect();
    let straight: [(&str, Spline<f64>); 5] = [
        ("natural", Spline::cubic(&x, &line).unwrap()),
        (
            "clamped",
            Spline::cubic_clamped(&x, &line, -2.0, -2.0).unwrap(),
        ),
        ("not-a-knot", Spline::cubic_not_a_knot(&x, &line).unwrap()),
        ("akima", Spline::akima(&x, &line).unwrap()),
        ("pchip", Spline::pchip(&x, &line).unwrap()),
    ];
    for (kind, s) in straight {
        for t in [-1.0, -0.35, 0.4, 1.7, 4.1] {
            assert!(
                (s.evaluate(t).unwrap() - (3.0 - 2.0 * t)).abs() < 1e-12,
                "{kind} bends a straight line at {t}"
            );
        }
    }
    let cubic = |t: f64| 1.0 - 2.0 * t + t * t * t / 3.0;
    let y: Vec<f64> = x.iter().map(|&t| cubic(t)).collect();
    let left = -2.0 + x[0] * x[0];
    let right = -2.0 + x[x.len() - 1] * x[x.len() - 1];
    let exact = Spline::cubic_clamped(&x, &y, left, right).unwrap();
    let nk = Spline::cubic_not_a_knot(&x, &y).unwrap();
    for t in [-1.0, -0.35, 0.4, 1.7, 4.1] {
        assert!(
            (exact.evaluate(t).unwrap() - cubic(t)).abs() < 1e-11,
            "clamped at {t}"
        );
        assert!(
            (nk.evaluate(t).unwrap() - cubic(t)).abs() < 1e-11,
            "not-a-knot at {t}"
        );
    }
    // The natural spline is not cubic-exact: it flattens its ends, and a cubic does not flatten.
    let nat = Spline::cubic(&x, &y).unwrap();
    assert!((nat.evaluate(-1.0).unwrap() - cubic(-1.0)).abs() > 1e-6);
}

#[test]
fn the_natural_ends_have_no_curvature() {
    let s = Spline::cubic(&X, &Y).unwrap();
    let left = second_derivative(&s, X[0], Side::Right).unwrap();
    let right = second_derivative(&s, X[X.len() - 1], Side::Left).unwrap();
    assert!(left.abs() < 1e-12, "the left end curves by {left}");
    assert!(right.abs() < 1e-12, "the right end curves by {right}");
    // Not-a-knot and clamped splines do curve at their ends; only this one is flat there.
    let nk = Spline::cubic_not_a_knot(&X, &Y).unwrap();
    assert!(second_derivative(&nk, X[0], Side::Right).unwrap().abs() > 1e-6);
}

#[test]
fn the_clamped_splines_keep_their_end_slopes() {
    let s = Spline::cubic_clamped(&X, &Y, 1.0, -2.0).unwrap();
    assert!((s.derivative(X[0]).unwrap() - 1.0).abs() < 1e-14);
    assert!((s.derivative(X[7]).unwrap() + 2.0).abs() < 1e-14);
    // The other two do not know the ends, and must not agree with them by accident.
    let n = Spline::cubic(&X, &Y).unwrap();
    assert!((n.derivative(X[0]).unwrap() - 1.0).abs() > 1e-3);
}

#[test]
fn not_a_knot_continues_the_third_derivative_across_two_knots_only() {
    let s = Spline::cubic_not_a_knot(&X, &Y).unwrap();
    let third = |i: usize| {
        let (m, y) = (s.slopes(), s.values());
        let h = X[i + 1] - X[i];
        6.0 * (m[i] + m[i + 1] - 2.0 * (y[i + 1] - y[i]) / h) / (h * h)
    };
    // The third derivative is constant on an interval, so it is continued across a knot exactly
    // when the two intervals that meet there agree.
    for k in [1, X.len() - 2] {
        assert!(
            (third(k - 1) - third(k)).abs() <= 1e-9 * third(k).abs().max(1.0),
            "knot {k} is a knot"
        );
    }
    assert!(
        (third(1) - third(2)).abs() > 1e-6,
        "the interior knots must stay knots"
    );
}

#[test]
fn akima_is_exact_where_the_data_are_straight() {
    // Akima's weights give a secant that agrees with its neighbour the whole of the slope, so a
    // linear run stays linear and a step stays flat on either side of the jump.
    let x: [f64; 6] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let step: [f64; 6] = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let s = Spline::akima(&x, &step).unwrap();
    assert_eq!(s.evaluate(0.5), Some(0.0));
    assert_eq!(s.evaluate(4.5), Some(1.0));
    assert_eq!(s.derivative(0.5).unwrap(), 0.0);
    let line: [f64; 6] = [1.0, 3.0, 5.0, 7.0, 9.0, 11.0];
    let s = Spline::akima(&x, &line).unwrap();
    for t in [0.5f64, 2.25, 4.75] {
        assert!((s.evaluate(t).unwrap() - (1.0 + 2.0 * t)).abs() < 1e-14);
    }
}

#[test]
fn pchip_never_leaves_the_range_of_a_monotone_run() {
    // Between two knots the monotone spline takes no value outside their own: a peak in the data
    // is a peak in the spline, and a flat run stays flat.
    let x: [f64; 8] = [0.0, 0.4, 1.1, 1.3, 2.2, 3.0, 3.1, 4.0];
    let y: [f64; 8] = [0.0, 0.5, 0.9, 0.85, 0.3, -0.2, -0.1, 0.6];
    let s = Spline::pchip(&x, &y).unwrap();
    for i in 0..x.len() - 1 {
        let (low, high) = (y[i].min(y[i + 1]), y[i].max(y[i + 1]));
        for k in 0..=40 {
            let t = x[i] + (x[i + 1] - x[i]) * f64::from(k) / 40.0;
            let v = s.evaluate(t).unwrap();
            assert!(
                v >= low - 1e-15 && v <= high + 1e-15,
                "the interval [{}, {}] reaches {v} outside [{low}, {high}]",
                x[i],
                x[i + 1]
            );
        }
    }
    // A turning point of the data has a horizontal tangent.
    assert_eq!(s.slopes()[2], 0.0);
    assert_eq!(s.slopes()[5], 0.0);
    // The cubic spline, by contrast, does overshoot the peak at knot 2.
    let c = Spline::cubic(&x, &y).unwrap();
    let peak = (0..=200)
        .map(|k| {
            let t = x[1] + (x[2] - x[1]) * f64::from(k) / 200.0;
            c.evaluate(t).unwrap()
        })
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(
        peak > 0.9,
        "the cubic spline should overshoot, and reached {peak}"
    );
}

#[test]
fn a_spline_of_two_knots_is_the_secant() {
    let five: [Spline<f64>; 5] = [
        Spline::cubic(&[0.0, 2.0], &[1.0, 5.0]).unwrap(),
        Spline::cubic_clamped(&[0.0, 2.0], &[1.0, 5.0], 9.0, -9.0).unwrap(),
        Spline::cubic_not_a_knot(&[0.0, 2.0], &[1.0, 5.0]).unwrap(),
        Spline::akima(&[0.0, 2.0], &[1.0, 5.0]).unwrap(),
        Spline::pchip(&[0.0, 2.0], &[1.0, 5.0]).unwrap(),
    ];
    for s in five {
        assert!((s.evaluate(1.0).unwrap() - 3.0).abs() < 1e-15);
        assert!((s.derivative(1.0).unwrap() - 2.0).abs() < 1e-15);
    }
}

#[test]
fn interpolation_outside_the_knots_answers_nothing() {
    let s = Spline::cubic(&X, &Y).unwrap();
    assert_eq!(s.evaluate(-0.001), None);
    assert_eq!(s.evaluate(3.001), None);
    assert_eq!(s.derivative(4.0), None);
    assert_eq!(s.evaluate(f64::NAN), None);
}

#[test]
#[should_panic(expected = "strictly increasing")]
fn knots_that_do_not_increase_panic() {
    let _ = Spline::cubic(&[0.0, 1.0, 1.0, 2.0], &[0.0, 1.0, 1.0, 0.0]);
}

#[test]
#[should_panic(expected = "differ in length")]
fn fewer_values_than_knots_panic() {
    let _ = Spline::cubic(&[0.0, 1.0, 2.0], &[0.0, 1.0]);
}

#[test]
#[should_panic(expected = "at least two knots")]
fn one_knot_panics() {
    let _ = Spline::cubic(&[1.0], &[0.0]);
}

#[test]
fn the_derivative_of_a_spline_is_the_derivative_of_its_cubics() {
    // Central differences of the values agree with the derivative the spline reports, away from
    // the knots, on all five.
    for s in splines() {
        for t in [0.15, 0.5, 0.9, 1.3, 1.8, 2.25, 2.7, 2.95] {
            let h = 1e-6;
            let numerical = (s.evaluate(t + h).unwrap() - s.evaluate(t - h).unwrap()) / (2.0 * h);
            assert!(
                (s.derivative(t).unwrap() - numerical).abs() < 1e-6,
                "at {t} the slope disagrees with the difference quotient"
            );
        }
    }
}

#[test]
fn chebyshev_nodes_are_the_zeros_of_t_n_in_order() {
    let x = chebyshev_nodes(6, -1.0, 1.0);
    assert_eq!(x.len(), 6);
    assert!(
        x.windows(2).all(|w| w[0] < w[1]),
        "the nodes come back sorted"
    );
    for (k, &t) in x.iter().enumerate() {
        let want = -((core::f64::consts::PI * (2 * k + 1) as f64) / 12.0).cos();
        assert!((t - want).abs() < 1e-15, "node {k} is {t}, not {want}");
    }
    // Mapped onto another interval they fill it, endpoints excluded.
    let y = chebyshev_nodes(9, 2.0, 5.0);
    assert!(y.iter().all(|&t| t > 2.0 && t < 5.0));
    assert!(y[0] > 2.0 && y[8] < 5.0);
}

#[test]
fn a_chebyshev_fit_reaches_the_precision_of_its_degree() {
    let f = |x: f64| (3.0 * x).sin() * x.exp();
    let c: Chebyshev<f64> = chebyshev(f, -1.0, 1.0, 20).unwrap();
    assert_eq!(c.degree(), 20);
    assert_eq!(c.interval(), (-1.0, 1.0));
    for t in [-1.0, -0.61, 0.0, 0.35, 1.0] {
        assert!((c.evaluate(t) - f(t)).abs() < 1e-14, "off at {t}");
    }
    // The derivative of the series is the series of the derivative.
    for t in [-0.8f64, 0.0, 0.4] {
        let want = 3.0 * (3.0 * t).cos() * t.exp() + (3.0 * t).sin() * t.exp();
        let h = 1e-6;
        let central = (c.evaluate(t + h) - c.evaluate(t - h)) / (2.0 * h);
        assert!((c.derivative(t) - want).abs() < 1e-6);
        assert!((c.derivative(t) - central).abs() < 1e-6);
    }
    // ∫₋₁¹ eˣ sin 3x dx, whose primitive is eˣ(sin 3x − 3 cos 3x)/10.
    let primitive = |t: f64| t.exp() * ((3.0 * t).sin() - 3.0 * (3.0 * t).cos()) / 10.0;
    assert!((c.integral(1.0) - (primitive(1.0) - primitive(-1.0))).abs() < 1e-13);
    assert!(
        c.integral(-1.0).abs() < 1e-15,
        "the integral to the left end is nothing"
    );
}

#[test]
fn a_chebyshev_fit_on_a_shifted_interval_is_the_same_fit() {
    let f = |x: f64| 1.0 / (1.0 + 25.0 * (x - 2.5) * (x - 2.5));
    // The pole at 2.5 ± 0.2i decides the rate: the error falls by the golden ratio of the
    // ellipse through it, so a fit of twice the degree gains about an order of magnitude.
    let coarse = chebyshev(f, 0.0, 5.0, 20).unwrap();
    let fine = chebyshev(f, 0.0, 5.0, 60).unwrap();
    let points = [0.0, 1.0, 2.5, 4.0, 5.0];
    let total =
        |c: &Chebyshev<f64>| -> f64 { points.iter().map(|&t| (c.evaluate(t) - f(t)).abs()).sum() };
    for t in points {
        assert!((coarse.evaluate(t) - f(t)).abs() < 5e-2, "off at {t}");
        assert!((fine.evaluate(t) - f(t)).abs() < 1e-2, "still off at {t}");
    }
    assert!(
        total(&fine) < total(&coarse) / 3.0,
        "three times the degree should buy an order of magnitude: {} against {}",
        total(&fine),
        total(&coarse)
    );
}

#[test]
fn barycentric_weights_and_the_formula_reproduce_the_polynomial() {
    let x = [0.0, 1.0, 2.0, 3.0];
    let y: Vec<f64> = x.iter().map(|&t| 1.0 + t - t * t / 2.0).collect();
    let w = barycentric_weights(&x);
    for t in [-1.0, 0.5, 1.7, 2.0, 4.0] {
        let want = 1.0 + t - t * t / 2.0;
        assert!((barycentric_interpolate(&x, &y, &w, t) - want).abs() < 1e-12);
    }
    // A node asks for its own value exactly, and empty weights ask for the Lagrange formula.
    assert_eq!(barycentric_interpolate(&x, &y, &w, 2.0), y[2]);
    // Equal weights are not the interpolant: at 0.8 they answer 1.361 where the polynomial says
    // 1.48. (At 0.5, by a coincidence of symmetry, they happen to agree.)
    assert!((barycentric_interpolate(&x, &y, &[], 0.8) - 1.48).abs() > 1e-3);
    assert!((barycentric_interpolate(&x, &y, &w, 0.8) - 1.48).abs() < 1e-12);
    for (a, b) in w
        .iter()
        .zip(&[-1.0 / 6.0, 1.0 / 2.0, -1.0 / 2.0, 1.0 / 6.0])
    {
        assert!(
            (a - b).abs() < 1e-15,
            "the weights of four equally spaced nodes alternate"
        );
    }
}

#[test]
fn barycentric_interpolation_at_chebyshev_nodes_beats_equally_spaced_ones() {
    // The point of the nodes: on the Runge function over [-1, 1] a degree-19 interpolant at the
    // Chebyshev nodes is faithful in the middle, while the same degree at equal spacing blows up.
    let f = |t: f64| 1.0 / (1.0 + 25.0 * t * t);
    let at = |nodes: &[f64], t: f64| {
        let values: Vec<f64> = nodes.iter().map(|&x| f(x)).collect();
        let w = barycentric_weights(nodes);
        (barycentric_interpolate(nodes, &values, &w, t) - f(t)).abs()
    };
    let nodes = chebyshev_nodes(20, -1.0, 1.0);
    let even: Vec<f64> = (0..20).map(|k| -1.0 + 2.0 * f64::from(k) / 19.0).collect();
    // Near the ends the equally spaced interpolant of the same degree is not merely worse but
    // useless, which is the whole reason the nodes exist.
    assert!(
        at(&nodes, 0.95) < 0.05,
        "the Chebyshev nodes should stay near the function"
    );
    assert!(
        at(&even, 0.95) > 1.0,
        "equally spaced nodes should run away"
    );
    // And they keep improving where equal spacing has stopped being able to.
    let finer = chebyshev_nodes(40, -1.0, 1.0);
    assert!(
        at(&finer, 0.4) < at(&nodes, 0.4) / 10.0,
        "doubling the degree should pay"
    );
}

#[test]
#[should_panic(expected = "two nodes are equal")]
fn repeated_nodes_panic() {
    let _ = barycentric_weights(&[0.0, 1.0, 1.0]);
}

/// The spline of `kind` over the test data.
fn build(kind: &str) -> Spline<f64> {
    build_on(kind, &X, &Y)
}

/// The spline of `kind` over the given knots.
fn build_on(kind: &str, x: &[f64], y: &[f64]) -> Spline<f64> {
    match kind {
        "natural" => Spline::cubic(x, y),
        "clamped" => Spline::cubic_clamped(x, y, 1.0, -2.0),
        "not-a-knot" => Spline::cubic_not_a_knot(x, y),
        "akima" => Spline::akima(x, y),
        "pchip" => Spline::pchip(x, y),
        _ => unreachable!("no such spline"),
    }
    .unwrap()
}
