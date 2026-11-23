//! The piecewise cubics and the polynomial approximants of [`a::math::numerics::interp`].
//!
//! The reference slopes, values and derivatives of the splines are exact (rational arithmetic on
//! the data as stored) and SciPy 1.18 agrees with them to 35 eps or better, so the rules followed
//! are SciPy's: `CubicSpline` for the three `C²` splines, `Akima1DInterpolator` (method `akima`,
//! Akima's own end rule) and `PchipInterpolator`. They, the not-a-knot sweep and the Chebyshev
//! references are in `data/interp_ref.rs`, made by `data/gen_interp.py`.

use a::math::numerics::{
    Chebyshev, Spline, barycentric_interpolate, barycentric_weights, chebyshev, chebyshev_nodes,
};

/// One spline data set of `data/interp_ref.rs`: the knots and values, the clamped end slopes,
/// the exact slopes of the five splines (natural, clamped, not-a-knot, Akima, PCHIP), and the
/// exact value and first derivative of each at the points `at`.
struct Case {
    x: &'static [f64],
    y: &'static [f64],
    left: f64,
    right: f64,
    slopes: [&'static [f64]; 5],
    at: &'static [f64],
    value: [&'static [f64]; 5],
    derivative: [&'static [f64]; 5],
}

/// A Chebyshev interpolant of `data/interp_ref.rs`, exact.
struct ChebCase {
    name: &'static str,
    a: f64,
    b: f64,
    coefficients: &'static [f64],
    at: &'static [f64],
    value: &'static [f64],
    derivative: &'static [f64],
    integral: &'static [f64],
}

/// A not-a-knot spline of `data/interp_ref.rs` over knots with a short interval among long ones,
/// or intervals of very different lengths: the data (rounded to `f32` when `f32` is set), the
/// exact slopes, and how far one unit in the last place of the values moves them, relative to the
/// largest slope, in eps of the type.
struct NakCase {
    f32: bool,
    x: &'static [f64],
    y: &'static [f64],
    slopes: &'static [f64],
    sensitivity: f64,
}

// The generated values include e and other constants to the last digit, which are data here.
#[allow(clippy::approx_constant)]
mod reference {
    use super::{Case, ChebCase, NakCase};
    include!("data/interp_ref.rs");
}

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

/// The five splines of `kind` 0 to 4 (natural, clamped, not-a-knot, Akima, PCHIP) over `x` and
/// `y`, the clamped one with the end slopes `left` and `right`.
fn spline<T: a::math::numerics::Real>(
    kind: usize,
    x: &[T],
    y: &[T],
    left: T,
    right: T,
) -> Option<Spline<T>> {
    match kind {
        0 => Spline::cubic(x, y),
        1 => Spline::cubic_clamped(x, y, left, right),
        2 => Spline::cubic_not_a_knot(x, y),
        3 => Spline::akima(x, y),
        _ => Spline::pchip(x, y),
    }
}

const NAMES: [&str; 5] = ["natural", "clamped", "not-a-knot", "akima", "pchip"];

/// The five splines of the test data.
fn splines() -> Vec<Spline<f64>> {
    (0..5)
        .map(|k| spline(k, &X, &Y, 1.0, -2.0).unwrap())
        .collect()
}

/// The largest of `|a − b|` over two slices, relative to `scale`, in units of `eps`.
fn worst(got: &[f64], want: &[f64], scale: f64, eps: f64) -> f64 {
    got.iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs() / scale / eps)
        .fold(0.0, f64::max)
}

fn largest(v: &[f64]) -> f64 {
    v.iter()
        .fold(0.0f64, |m, x| m.max(x.abs()))
        .max(f64::MIN_POSITIVE)
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
fn the_five_splines_agree_with_the_exact_ones_and_with_scipy_in_f64() {
    // Slopes, values and first derivatives against the exact splines of 2, 3, 4, 5, 8 and 13
    // random non-uniform knots; SciPy is within 35 eps of the same references.
    for case in reference::SPLINES_F64 {
        for (kind, name) in NAMES.iter().enumerate() {
            let s = spline(kind, case.x, case.y, case.left, case.right).unwrap();
            let m = largest(case.slopes[kind]);
            let slopes = worst(s.slopes(), case.slopes[kind], m, f64::EPSILON);
            let values: Vec<f64> = case.at.iter().map(|&t| s.evaluate(t).unwrap()).collect();
            let derivs: Vec<f64> = case.at.iter().map(|&t| s.derivative(t).unwrap()).collect();
            let size = largest(case.y).max(largest(case.value[kind]));
            let v = worst(&values, case.value[kind], size, f64::EPSILON);
            let d = worst(&derivs, case.derivative[kind], m, f64::EPSILON);
            let n = case.x.len();
            assert!(
                slopes <= 64.0 && v <= 16.0 && d <= 64.0,
                "{} through {n} knots: slopes {slopes:.1} eps, values {v:.1}, derivatives {d:.1}",
                name
            );
        }
    }
}

#[test]
fn the_five_splines_agree_with_the_exact_ones_in_f32() {
    for case in reference::SPLINES_F32 {
        let to32 = |v: &[f64]| v.iter().map(|&t| t as f32).collect::<Vec<f32>>();
        let (x, y) = (to32(case.x), to32(case.y));
        for (kind, name) in NAMES.iter().enumerate() {
            let s = spline(kind, &x, &y, case.left as f32, case.right as f32).unwrap();
            let eps = f64::from(f32::EPSILON);
            let m = largest(case.slopes[kind]);
            let back = |v: &[f32]| v.iter().map(|&t| f64::from(t)).collect::<Vec<f64>>();
            let slopes = worst(&back(s.slopes()), case.slopes[kind], m, eps);
            let values: Vec<f32> = case
                .at
                .iter()
                .map(|&t| s.evaluate(t as f32).unwrap())
                .collect();
            let derivs: Vec<f32> = case
                .at
                .iter()
                .map(|&t| s.derivative(t as f32).unwrap())
                .collect();
            let size = largest(case.y).max(largest(case.value[kind]));
            let v = worst(&back(&values), case.value[kind], size, eps);
            let d = worst(&back(&derivs), case.derivative[kind], m, eps);
            assert!(
                slopes <= 64.0 && v <= 16.0 && d <= 64.0,
                "{} through {} knots in f32: slopes {slopes:.1} eps, values {v:.1}, derivatives {d:.1}",
                name,
                x.len()
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

#[test]
fn the_cubic_splines_are_twice_continuously_differentiable() {
    for (kind, name) in NAMES.iter().enumerate().take(3) {
        let s = spline(kind, &X, &Y, 1.0, -2.0).unwrap();
        for (k, &x) in X.iter().enumerate().skip(1).take(X.len() - 2) {
            // The second derivative of the interval to the left, at its right end, from a point
            // just inside it, against the value of the interval to the right at the knot.
            let left = s.second_derivative(x - 1e-9).unwrap();
            let right = s.second_derivative(x).unwrap();
            assert!(
                (left - right).abs() <= 1e-7 * left.abs().max(right.abs()).max(1.0),
                "{} bends at knot {k}: {left} against {right}",
                name
            );
        }
    }
    // Akima's and PCHIP's splines are only once differentiable: their curvature jumps.
    let a = Spline::akima(&X, &Y).unwrap();
    let jump = a.second_derivative(X[3] - 1e-9).unwrap() - a.second_derivative(X[3]).unwrap();
    assert!(jump.abs() > 1e-3, "Akima's curvature jumps, by {jump}");
}

#[test]
fn the_second_derivative_is_the_derivative_of_the_derivative() {
    for (kind, s) in splines().iter().enumerate() {
        for t in [0.15, 0.5, 0.9, 1.3, 1.8, 2.25, 2.7, 2.95] {
            let h = 1e-6;
            let numerical =
                (s.derivative(t + h).unwrap() - s.derivative(t - h).unwrap()) / (2.0 * h);
            let got = s.second_derivative(t).unwrap();
            assert!(
                (got - numerical).abs() < 1e-6,
                "{} at {t}: {got} against {numerical}",
                NAMES[kind]
            );
        }
    }
}

#[test]
fn the_integral_of_a_spline_is_simpsons_rule_on_each_cubic() {
    // Simpson's rule is exact for a cubic, so it is an independent check of the integral of
    // every piece, and of how the pieces are added up between any two points.
    for (kind, s) in splines().iter().enumerate() {
        let simpson = |a: f64, b: f64| {
            (b - a) / 6.0
                * (s.evaluate(a).unwrap()
                    + 4.0 * s.evaluate((a + b) / 2.0).unwrap()
                    + s.evaluate(b).unwrap())
        };
        // Split at every knot between the two limits.
        let exact = |a: f64, b: f64| {
            let mut cuts = vec![a];
            cuts.extend(X.iter().copied().filter(|&k| k > a && k < b));
            cuts.push(b);
            cuts.windows(2).map(|w| simpson(w[0], w[1])).sum::<f64>()
        };
        for (a, b) in [(0.0, 3.0), (0.1, 0.2), (0.25, 2.75), (1.1, 1.6), (2.9, 3.0)] {
            let got = s.integral(a, b).unwrap();
            assert!(
                (got - exact(a, b)).abs() <= 1e-14,
                "{} from {a} to {b}: {got} against {}",
                NAMES[kind],
                exact(a, b)
            );
            assert_eq!(s.integral(b, a), Some(-got));
        }
        assert_eq!(s.integral(1.3, 1.3), Some(0.0));
        assert_eq!(s.integral(-0.1, 1.0), None);
        assert_eq!(s.integral(1.0, 3.1), None);
    }
}

#[test]
fn the_cubic_splines_reproduce_a_cubic_exactly() {
    // A straight line is reproduced by every one of them, since each end condition is itself
    // satisfied by a line. A cubic needs the end condition to agree with it too, which only the
    // clamped spline — given the cubic's own end slopes — and not-a-knot can do.
    let x: Vec<f64> = (0..7).map(|k| f64::from(k) * 0.9 - 1.3).collect();
    let line: Vec<f64> = x.iter().map(|&t| 3.0 - 2.0 * t).collect();
    for (kind, name) in NAMES.iter().enumerate() {
        let s = spline(kind, &x, &line, -2.0, -2.0).unwrap();
        for t in [-1.0, -0.35, 0.4, 1.7, 4.1] {
            assert!(
                (s.evaluate(t).unwrap() - (3.0 - 2.0 * t)).abs() < 1e-12,
                "{} bends a straight line at {t}",
                name
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
    let left = s.second_derivative(X[0]).unwrap();
    let right = s.second_derivative(X[X.len() - 1]).unwrap();
    assert!(left.abs() < 1e-12, "the left end curves by {left}");
    assert!(right.abs() < 1e-12, "the right end curves by {right}");
    // Not-a-knot and clamped splines do curve at their ends; only this one is flat there.
    let nk = Spline::cubic_not_a_knot(&X, &Y).unwrap();
    assert!(nk.second_derivative(X[0]).unwrap().abs() > 1e-6);
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
fn not_a_knot_through_three_knots_is_the_parabola() {
    // Audit n_interp-4: the two conditions coincide, and the spline is the parabola through the
    // three points (SciPy and MATLAB); before, both end eliminations were applied to the single
    // interior row and the result was not even C².
    let s = Spline::cubic_not_a_knot(&[0.0, 1.0, 2.0], &[0.0, 1.0, 4.0]).unwrap();
    assert_eq!(s.slopes(), &[0.0, 2.0, 4.0]);
    assert_eq!(s.evaluate(0.5), Some(0.25));
    assert_eq!(s.evaluate(1.5), Some(2.25));
    assert_eq!(s.second_derivative(0.999_999), Some(2.0));
    assert_eq!(s.second_derivative(1.0), Some(2.0));
    // A straight line through unequal intervals stays straight.
    let line = Spline::cubic_not_a_knot(&[0.0, 1.0, 3.0], &[1.0, 3.0, 7.0]).unwrap();
    assert_eq!(line.slopes(), &[2.0, 2.0, 2.0]);
    assert_eq!(line.evaluate(2.0), Some(5.0));
    // And four knots give the cubic through them.
    let cubic = |t: f64| t * t * t - t;
    let x = [-1.0, 0.25, 0.5, 2.0];
    let y: Vec<f64> = x.iter().map(|&t| cubic(t)).collect();
    let s = Spline::cubic_not_a_knot(&x, &y).unwrap();
    for t in [-0.9, 0.0, 0.4, 1.3, 1.99] {
        assert!((s.evaluate(t).unwrap() - cubic(t)).abs() < 1e-13, "at {t}");
    }
}

#[test]
fn a_short_interval_costs_not_a_knot_no_more_than_the_data_allow() {
    // Review round 1, B1: with the conditions combined into the first and last rows of the system
    // of all the slopes (de Boor's rows, SciPy's), a short second or penultimate interval lost
    // about R eps beside its neighbours R times longer, and through four knots R² eps: 7 % of the
    // first slope of the sine at 0, 1, 1.001, 2 in f32. Solved as the spline of fewer knots, every
    // slope is within a few times the change one unit in the last place of the values makes, which
    // is what the data allow; plain partial pivoting of that system lost 280 times as much over the
    // knots whose intervals range from 4e-3 to 6e3.
    for c in reference::NAK_SHORT {
        let (got, eps): (Vec<f64>, f64) = if c.f32 {
            let x: Vec<f32> = c.x.iter().map(|&v| v as f32).collect();
            let y: Vec<f32> = c.y.iter().map(|&v| v as f32).collect();
            let s = Spline::cubic_not_a_knot(&x, &y).unwrap();
            let slopes = s.slopes().iter().map(|&v| f64::from(v)).collect();
            (slopes, f64::from(f32::EPSILON))
        } else {
            let s = Spline::cubic_not_a_knot(c.x, c.y).unwrap();
            (s.slopes().to_vec(), f64::EPSILON)
        };
        let err = worst(&got, c.slopes, largest(c.slopes), eps);
        assert!(
            err <= 4.0 * c.sensitivity.max(1.0),
            "{} at {:?}: {err:.1} eps, where one ulp of the values moves the slopes {:.1}",
            if c.f32 { "f32" } else { "f64" },
            c.x,
            c.sensitivity
        );
    }
    // The review's example: the sine at 0, 1, 1.0001 and 2 in f32 had the slopes [6.53, 0.540,
    // 0.541, −5.96] for [1.056, 0.541, 0.540, −0.491], and the value 1.169 at 0.5, where the cubic
    // through the four points is within 0.005 of sin 0.5.
    let x = [0.0f32, 1.0, 1.0001, 2.0];
    let y: Vec<f32> = x.iter().map(|t| t.sin()).collect();
    let s = Spline::cubic_not_a_knot(&x, &y).unwrap();
    assert!(
        (s.slopes()[0] - 1.056_464_4).abs() < 1e-3,
        "{:?}",
        s.slopes()
    );
    assert!((s.evaluate(0.5).unwrap() - 0.5f32.sin()).abs() < 0.01);
}

#[test]
fn a_long_end_interval_costs_not_a_knot_no_accuracy() {
    // Audit n_interp-10: the first interval R times longer than the next. Recovering the end
    // slope from the eliminated relation lost about R eps (1.8e4 eps at R = 1000 in f64, 28 % at
    // R = 1e5 in f32); solved as the spline of fewer knots the error stays at the 30 to 100 eps
    // SciPy has too.
    let ys = reference::NAK_VALUES;
    for &(r, want64, want32) in reference::NAK_RATIO {
        let x = [0.0, r, r + 1.0, r + 2.1, r + 3.3, r + 4.2];
        let got = Spline::cubic_not_a_knot(&x, &ys).unwrap().slopes()[0];
        let err = ((got - want64) / want64).abs() / f64::EPSILON;
        assert!(err <= 256.0, "R = {r}: {err:.0} eps in f64");
        let x32: Vec<f32> = x.iter().map(|&v| v as f32).collect();
        let y32: Vec<f32> = ys.iter().map(|&v| v as f32).collect();
        let got = f64::from(Spline::cubic_not_a_knot(&x32, &y32).unwrap().slopes()[0]);
        let err = ((got - want32) / want32).abs() / f64::from(f32::EPSILON);
        assert!(err <= 256.0, "R = {r}: {err:.0} eps in f32");
    }
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
fn akima_continues_the_secants_at_the_right_end_too() {
    // Audit n_interp-5: the right-hand continuation read a secant that was never written, so
    // the last two slopes were wrong for any data not straight or flat there. For a parabola
    // Akima's continuation is exact at both ends (SciPy: the same slopes).
    let x: Vec<f64> = (0..7).map(f64::from).collect();
    let y: Vec<f64> = x.iter().map(|t| t * t).collect();
    let s = Spline::akima(&x, &y).unwrap();
    assert_eq!(s.slopes(), &[0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);
    assert_eq!(s.evaluate(5.5), Some(30.25));
    assert_eq!(s.derivative(6.0), Some(12.0));
}

#[test]
fn every_spline_of_mirrored_data_is_the_mirrored_spline() {
    // x → −x with the knots reversed: the slopes change sign and order. Akima and PCHIP form
    // their slopes symmetrically and mirror to the last bit; the C² splines solve a system in
    // one direction and mirror to rounding.
    for case in reference::SPLINES_F64 {
        let n = case.x.len();
        let mx: Vec<f64> = case.x.iter().rev().map(|v| -v).collect();
        let my: Vec<f64> = case.y.iter().rev().copied().collect();
        for (kind, name) in NAMES.iter().enumerate() {
            let s = spline(kind, case.x, case.y, case.left, case.right).unwrap();
            let m = spline(kind, &mx, &my, -case.right, -case.left).unwrap();
            let back: Vec<f64> = m.slopes().iter().rev().map(|v| -v).collect();
            if kind >= 3 {
                assert_eq!(back, s.slopes(), "{} through {n} knots", name);
            } else {
                let err = worst(&back, s.slopes(), largest(s.slopes()), f64::EPSILON);
                assert!(err <= 64.0, "{} through {n} knots: {err} eps", name);
            }
        }
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
fn pchip_takes_the_three_point_slope_at_an_end_where_the_data_turn() {
    // Audit n_interp-6: the end slope was set to zero whenever the two end secants differed in
    // sign. PCHIP (SLATEC's PCHIM, SciPy) keeps the one-sided three-point slope there, limited
    // to three times the end secant; these are SciPy's slopes and values.
    let s = Spline::pchip(&[0.0, 1.0, 2.0], &[0.0, 1.0, 0.0]).unwrap();
    assert_eq!(s.slopes(), &[2.0, 0.0, -2.0]);
    assert_eq!(s.evaluate(0.5), Some(0.75));
    let s = Spline::pchip(&[0.0, 1.0, 2.0, 3.0], &[1.0, 0.0, 0.0, 0.0]).unwrap();
    assert_eq!(s.slopes(), &[-1.5, 0.0, 0.0, 0.0]);
    assert_eq!(s.evaluate(0.5), Some(0.3125));
    // The limit: the data turn at the second knot and the three-point slope, 10, would
    // overshoot; 3·d₀ = 6 is taken.
    let s = Spline::pchip(&[0.0, 1.0, 1.5], &[0.0, 2.0, -3.0]).unwrap();
    assert_eq!(s.slopes()[0], 6.0);
    // Monotone data stay monotone, the ends included.
    let x = [0.0, 0.2, 0.9, 1.0, 2.5, 2.6];
    let y = [0.0, 0.1, 0.1, 0.8, 0.9, 3.0];
    let s = Spline::pchip(&x, &y).unwrap();
    let mut last = f64::NEG_INFINITY;
    for k in 0..=520 {
        let v = s.evaluate(2.6 * f64::from(k) / 520.0).unwrap();
        assert!(v >= last, "PCHIP of monotone data falls at step {k}");
        last = v;
    }
}

#[test]
fn akima_and_pchip_handle_f32_data_of_any_size() {
    // Audit n_interp-8: both formed products of two secants, which overflowed for data near
    // 1e20 and underflowed near 1e-23 in f32. Scaling the data by a power of two must scale every
    // slope of every spline by the same power, to the last bit, over the whole range.
    let x = [0.0f32, 1.0, 2.0, 3.0, 4.0];
    let y = [1.0f32, 2.0, 4.0, 7.0, 9.0];
    for (kind, name) in NAMES.iter().enumerate() {
        let base = spline(kind, &x, &y, 0.5, 1.5).unwrap();
        for k in (-120..=120).step_by(4) {
            let f = 2f32.powi(k);
            let scaled: Vec<f32> = y.iter().map(|v| v * f).collect();
            let s = spline(kind, &x, &scaled, 0.5 * f, 1.5 * f).unwrap();
            let want: Vec<f32> = base.slopes().iter().map(|m| m * f).collect();
            assert_eq!(s.slopes(), &want[..], "{} scaled by 2^{k}", name);
        }
    }
    // The audit's data, against SciPy's PCHIP slopes of the data divided by 1e20.
    let big = [1.0e20f32, 2.0e20, 4.0e20, 7.0e20, 9.0e20];
    let s = Spline::pchip(&x, &big).unwrap();
    for (got, want) in s.slopes().iter().zip([0.5, 4.0 / 3.0, 2.4, 2.4, 1.5]) {
        assert!((f64::from(*got) / 1e20 - want).abs() < 1e-6 * want, "{got}");
    }
    assert!(
        Spline::akima(&x, &big)
            .unwrap()
            .evaluate(1.5)
            .unwrap()
            .is_finite()
    );
    let small = [1.0e-23f32, 2.0e-23, 4.0e-23, 7.0e-23, 9.0e-23];
    let s = Spline::pchip(&x, &small).unwrap();
    for (got, want) in s.slopes().iter().zip([0.5, 4.0 / 3.0, 2.4, 2.4, 1.5]) {
        assert!(
            (f64::from(*got) / 1e-23 - want).abs() < 1e-5 * want,
            "{got}"
        );
    }
    assert!(
        Spline::akima(&x, &small)
            .unwrap()
            .slopes()
            .iter()
            .all(|&m| m > 0.0)
    );
}

#[test]
fn the_derivative_does_not_cancel_a_large_offset() {
    // Audit n_interp-11: the two values entered separately, which cost eps·|y|/|Δy| (1.1e-9 for
    // this line); through two knots the derivative is the slope the spline holds, exactly.
    let s = Spline::cubic(&[0.0f64, 1.0], &[101_325.0, 101_325.01]).unwrap();
    let m = s.slopes()[0];
    assert_eq!(s.derivative(0.5), Some(m));
    let s = Spline::cubic(&[0.0f32, 1.0], &[1000.0, 1000.5]).unwrap();
    assert_eq!(s.derivative(0.5), Some(0.5));
    // A cubic spline of data with a large common offset has the derivative of the data without it.
    let y: Vec<f64> = Y.iter().map(|v| v + 1e6).collect();
    let offset = Spline::cubic(&X, &y).unwrap();
    let plain = Spline::cubic(&X, &Y).unwrap();
    for t in [0.1, 0.8, 1.9, 2.8] {
        let (a, b) = (offset.derivative(t).unwrap(), plain.derivative(t).unwrap());
        assert!((a - b).abs() < 1e-8, "at {t}: {a} against {b}");
    }
}

#[test]
fn a_spline_of_two_knots_is_the_secant_or_the_hermite_cubic() {
    let x = [0.0f64, 2.0];
    let y = [1.0f64, 5.0];
    for kind in [0, 2, 3, 4] {
        let s = spline(kind, &x, &y, 0.0, 0.0).unwrap();
        assert_eq!(s.slopes(), &[2.0, 2.0], "{}", NAMES[kind]);
        assert!((s.evaluate(1.0).unwrap() - 3.0).abs() < 1e-15);
        assert!((s.derivative(1.0).unwrap() - 2.0).abs() < 1e-15);
    }
    // Audit n_interp-7: the clamped spline ignored its slopes here. Through two knots it is the
    // cubic with the given slopes (SciPy: 7.5 at the midpoint).
    let s = Spline::cubic_clamped(&x, &y, 9.0, -9.0).unwrap();
    assert_eq!(s.slopes(), &[9.0, -9.0]);
    assert_eq!(s.derivative(0.0), Some(9.0));
    assert_eq!(s.derivative(2.0), Some(-9.0));
    assert_eq!(s.evaluate(1.0), Some(7.5));
}

#[test]
fn interpolation_outside_the_knots_answers_nothing() {
    let s = Spline::cubic(&X, &Y).unwrap();
    assert_eq!(s.evaluate(-0.001), None);
    assert_eq!(s.evaluate(3.001), None);
    assert_eq!(s.derivative(4.0), None);
    assert_eq!(s.second_derivative(-1.0), None);
    assert_eq!(s.evaluate(f64::NAN), None);
    assert_eq!(s.evaluate(f64::INFINITY), None);
    // The floats next to the ends: outside, nothing; inside, the end interval.
    assert_eq!(s.evaluate(X[0].next_down()), None);
    assert_eq!(s.evaluate(X[7].next_up()), None);
    assert!(s.evaluate(X[0].next_up()).is_some());
    assert!(s.evaluate(X[7].next_down()).is_some());
}

#[test]
fn the_interval_of_a_point_is_found_by_bisection() {
    // A thousand knots: every knot and the floats on either side of it land in the right
    // interval, which shows in a value continuous across the knots.
    let n = if cfg!(miri) { 50 } else { 1000 };
    let x: Vec<f64> = (0..n)
        .map(|k| f64::from(k) + 0.25 * (f64::from(k) * 0.7).sin())
        .collect();
    let y: Vec<f64> = x.iter().map(|t| (t * 0.1).cos()).collect();
    let s = Spline::cubic(&x, &y).unwrap();
    for (k, &t) in x.iter().enumerate() {
        assert_eq!(s.evaluate(t), Some(y[k]));
        if k > 0 {
            assert!((s.evaluate(t.next_down()).unwrap() - y[k]).abs() < 1e-13);
        }
        if k + 1 < x.len() {
            assert!((s.evaluate(t.next_up()).unwrap() - y[k]).abs() < 1e-13);
        }
    }
}

#[test]
fn knots_or_values_that_are_not_finite_give_none() {
    // Audit n_interp-15: an infinite end knot or a non-finite value gave a spline of NaN slopes.
    let x = [0.0, 1.0, 2.0, 3.0];
    let y = [0.0, 1.0, 1.0, 0.0];
    for (kind, name) in NAMES.iter().enumerate() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut bx = x;
            bx[3] = bad;
            assert_eq!(spline(kind, &bx, &y, 0.0, 0.0), None, "{} knot {bad}", name);
            let mut bx = x;
            bx[0] = bad;
            assert_eq!(spline(kind, &bx, &y, 0.0, 0.0), None, "{} knot {bad}", name);
            let mut by = y;
            by[1] = bad;
            assert_eq!(
                spline(kind, &x, &by, 0.0, 0.0),
                None,
                "{} value {bad}",
                name
            );
        }
    }
    assert_eq!(Spline::cubic_clamped(&x, &y, f64::NAN, 0.0), None);
    assert_eq!(Spline::cubic_clamped(&x, &y, 0.0, f64::INFINITY), None);
    // Secants that overflow are not a spline either.
    assert_eq!(
        Spline::pchip(&[0.0, 1e-300, 1.0], &[-1e300, 1e300, 0.0]),
        None
    );
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
#[should_panic(expected = "finite and increasing")]
fn chebyshev_nodes_of_a_reversed_interval_panic() {
    let _ = chebyshev_nodes(3, 1.0f64, 0.0);
}

#[test]
#[should_panic(expected = "degree + 1 nodes must be countable")]
fn a_chebyshev_fit_of_the_largest_degree_panics() {
    // degree + 1 overflowed: a panic on overflow in debug builds, and in release a wrap to zero
    // nodes, which panicked in chebyshev_nodes with another message.
    let _ = chebyshev(|x: f64| x, 0.0, 1.0, usize::MAX);
}

/// The Chebyshev coefficients of the interpolant of the values `v` at the first-kind nodes, as
/// [`chebyshev`] defines them, `c_k = (2/n)·Σ_j v_j·(−1)^k·cos(πk(2j + 1)/(2n))` (half that for
/// `k = 0`), to an ulp or so of `Σ|v_j|/n`: each product is split exactly by `mul_add`, the sum
/// carries its rounding in a second part, and each cosine is taken of an angle folded into
/// `[0, π/4]`, so that it is off by about an ulp of a number at most 1. Measured against mpmath's
/// transform of the same values (cos(20t), degree 100): 0.73 eps·max|c| on aarch64, 0.52 on
/// x86_64.
fn transform(v: &[f64]) -> Vec<f64> {
    let n = v.len();
    let angle = |r: usize| core::f64::consts::FRAC_PI_2 * r as f64 / n as f64;
    // cos(πt/(2n)) = cos(qπ/2 + πr/(2n)) for t = qn + r.
    let cosine = |t: usize| {
        let (q, r) = ((t / n) % 4, t % n);
        let (c, s) = if 2 * r <= n {
            let a = angle(r);
            (a.cos(), a.sin())
        } else {
            let b = angle(n - r);
            (b.sin(), b.cos())
        };
        match q {
            0 => c,
            1 => -s,
            2 => -c,
            _ => s,
        }
    };
    (0..n)
        .map(|k| {
            let (mut sum, mut low) = (0.0f64, 0.0f64);
            for (j, &value) in v.iter().enumerate() {
                let t = cosine(k * (2 * j + 1));
                let t = if k % 2 == 1 { -t } else { t };
                let p = value * t;
                let e = value.mul_add(t, -p);
                let s = sum + p;
                let b = s - sum;
                low += (sum - (s - b)) + (p - b) + e;
                sum = s;
            }
            let total = sum + low;
            if k == 0 {
                total / n as f64
            } else {
                2.0 * total / n as f64
            }
        })
        .collect()
}

#[test]
fn chebyshev_fits_agree_with_the_exact_interpolants() {
    // Coefficients, values, derivatives and integrals against the exact interpolant at the
    // first-kind points (numpy's Chebyshev.interpolate is within 1e-14 of the same), degrees 0
    // to 100. The last point of each is outside the interval, where the polynomial is continued:
    // there each coefficient's rounding, about eps·max|c|, is multiplied by |T_k(u)| > 1, so the
    // error is measured against eps·max|c|·Σ|T_k(u)|.
    //
    // The coefficients are checked twice. Against the transform of the fit's own samples,
    // computed to an ulp or so (`transform`): that is the rounding of the transform itself,
    // 2.8 eps·max|c| at most on aarch64 and 3.4 on x86_64, where cosines of angles not reduced
    // first (up to 100π) give 63. And against the exact interpolant, which adds the errors of
    // the samples: within 64 eps·max|c|, except for cos(20t) on [-3, 7], which is conditioned
    // like |20t·sin 20t|, up to 140 — the nodes, within 6.3 eps of the exact ones, and 20t,
    // rounded to an ulp of 140, move its values by up to 145 eps — and whose coefficients the
    // samples alone put 61 eps·max|c| (aarch64) and 72 (x86_64, whose libm rounds the cosines of
    // the nodes differently) from the exact interpolant's (mpmath on the samples as computed):
    // there the bound is 128.
    for c in reference::CHEBYSHEV {
        let f: fn(f64) -> f64 = match c.name {
            "exp" => |t| t.exp(),
            "runge" => |t| 1.0 / (1.0 + 25.0 * t * t),
            "sin3exp" => |t| (3.0 * t).sin() * t.exp(),
            _ => |t| (20.0 * t).cos(),
        };
        let degree = c.coefficients.len() - 1;
        let fit = chebyshev(f, c.a, c.b, degree).unwrap();
        assert_eq!(fit.degree(), degree);
        let samples: Vec<f64> = chebyshev_nodes(degree + 1, c.a, c.b)
            .into_iter()
            .map(f)
            .collect();
        let own = worst(
            fit.coefficients(),
            &transform(&samples),
            largest(c.coefficients),
            f64::EPSILON,
        );
        assert!(
            own <= 16.0,
            "{} of degree {degree}: coefficients {own:.1} eps from the transform of the samples",
            c.name
        );
        let coef = worst(
            fit.coefficients(),
            c.coefficients,
            largest(c.coefficients),
            f64::EPSILON,
        );
        let bound = if c.name == "cos20" { 128.0 } else { 64.0 };
        assert!(
            coef <= bound,
            "{} of degree {degree}: coefficients {coef:.1} eps",
            c.name
        );
        let inside = c.at.len() - 1;
        let values: Vec<f64> = c.at[..inside].iter().map(|&t| fit.evaluate(t)).collect();
        let derivs: Vec<f64> = c.at[..inside].iter().map(|&t| fit.derivative(t)).collect();
        let integrals: Vec<f64> = c.at[..inside].iter().map(|&t| fit.integral(t)).collect();
        let v = worst(&values, &c.value[..inside], largest(c.value), f64::EPSILON);
        let d = worst(
            &derivs,
            &c.derivative[..inside],
            largest(c.derivative),
            f64::EPSILON,
        );
        let i = worst(
            &integrals,
            &c.integral[..inside],
            largest(c.integral),
            f64::EPSILON,
        );
        // The derivative is conditioned like the square of the degree.
        let dtol = 64.0 * (degree as f64 + 1.0).powi(2);
        assert!(
            v <= 128.0 && d <= dtol && i <= 64.0,
            "{} of degree {degree}: values {v:.1} eps, derivatives {d:.1}, integrals {i:.1}",
            c.name
        );
        let out = c.at[inside];
        let u = (2.0 * out - c.a - c.b) / (c.b - c.a);
        let (mut t0, mut t1, mut sum) = (1.0f64, u, 0.0f64);
        for k in 0..=degree {
            sum += if k == 0 { t0 } else { t1 }.abs();
            if k > 0 {
                (t0, t1) = (t1, 2.0 * u * t1 - t0);
            }
        }
        let cond = largest(c.coefficients) * sum;
        let err = (fit.evaluate(out) - c.value[inside]).abs() / (cond * f64::EPSILON);
        assert!(
            err <= 16.0,
            "{} of degree {degree} at {out}: {err:.1} times eps·max|c|·Σ|T|",
            c.name
        );
    }
}

#[test]
fn chebyshev_fits_in_f32_reach_f32_precision() {
    for c in reference::CHEBYSHEV {
        if c.name != "exp" && c.name != "sin3exp" {
            continue;
        }
        let f = |t: f32| -> f32 {
            if c.name == "exp" {
                t.exp()
            } else {
                (3.0 * t).sin() * t.exp()
            }
        };
        let degree = c.coefficients.len() - 1;
        let fit = chebyshev(f, c.a as f32, c.b as f32, degree).unwrap();
        let eps = f64::from(f32::EPSILON);
        let coef: Vec<f64> = fit.coefficients().iter().map(|&v| f64::from(v)).collect();
        let err = worst(&coef, c.coefficients, largest(c.coefficients), eps);
        assert!(err <= 64.0, "{} of degree {degree}: {err:.1} eps", c.name);
        for (k, &t) in c.at[..c.at.len() - 1].iter().enumerate() {
            let v = f64::from(fit.evaluate(t as f32));
            let e = (v - c.value[k]).abs() / (eps * largest(c.value));
            assert!(
                e <= 64.0,
                "{} of degree {degree} at {t}: {e:.1} eps",
                c.name
            );
        }
    }
}

#[test]
fn a_chebyshev_fit_of_degree_zero_is_the_value_at_the_middle() {
    let c = chebyshev(|x: f64| x.exp(), 0.0, 2.0, 0).unwrap();
    assert_eq!(c.coefficients(), &[1f64.exp()]);
    assert_eq!(c.evaluate(0.3), 1f64.exp());
    assert_eq!(c.derivative(0.3), 0.0);
    assert!((c.integral(2.0) - 2.0 * 1f64.exp()).abs() < 1e-15);
}

#[test]
fn a_chebyshev_fit_needs_a_finite_increasing_interval_and_finite_values() {
    assert!(chebyshev(|x: f64| x, 1.0, 0.0, 3).is_none());
    assert!(chebyshev(|x: f64| x, 0.0, 0.0, 3).is_none());
    assert!(chebyshev(|x: f64| x, f64::NAN, 1.0, 3).is_none());
    assert!(chebyshev(|x: f64| x, 0.0, f64::INFINITY, 3).is_none());
    // A value that is not finite at one of the nodes.
    assert!(chebyshev(|x: f64| if x > 0.5 { f64::NAN } else { x }, -1.0, 1.0, 2).is_none());
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
        assert!((c.derivative(t) - want).abs() < 1e-12);
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
fn a_chebyshev_fit_is_continued_as_a_polynomial_outside_its_interval() {
    // The fit is the polynomial; outside [a, b] it is evaluated as such, which follows the
    // function a little way and then leaves it.
    let c = chebyshev(|x: f64| x.exp(), 0.0, 1.0, 12).unwrap();
    assert!((c.evaluate(1.1) - 1.1f64.exp()).abs() < 1e-10);
    assert!((c.evaluate(-0.1) - (-0.1f64).exp()).abs() < 1e-10);
    assert!((c.evaluate(10.0) - 10f64.exp()).abs() > 1.0);
    // A polynomial of degree 2 is its own fit everywhere.
    let p = |x: f64| 3.0 * x * x - x + 0.5;
    let c = chebyshev(p, -1.0, 2.0, 2).unwrap();
    for t in [-30.0, -2.0, 5.0, 40.0] {
        assert!((c.evaluate(t) - p(t)).abs() < 1e-12 * p(t).abs());
        assert!((c.derivative(t) - (6.0 * t - 1.0)).abs() < 1e-12 * t.abs().max(1.0));
    }
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
    // A node asks for its own value exactly.
    assert_eq!(barycentric_interpolate(&x, &y, &w, 2.0), y[2]);
    // The weights of four equally spaced nodes are −1/6, 1/2, −1/2, 1/6 up to a common factor.
    for (a, b) in w
        .iter()
        .zip(&[-1.0 / 6.0, 1.0 / 2.0, -1.0 / 2.0, 1.0 / 6.0])
    {
        assert!(
            (a / w[0] - b / (-1.0 / 6.0)).abs() < 1e-15,
            "the weights of four equally spaced nodes alternate"
        );
    }
}

#[test]
fn barycentric_weights_stay_in_range_for_many_nodes_on_any_interval() {
    // Audit n_interp-9: the plain products left the range of the type (NaN from 129 Chebyshev
    // nodes on [0, 1000], 100 on [0, 1e-3], 1000 on [0, 1]; in f32 from 30 on [0, 100]), with a
    // few weights zero and the answer silently wrong just before. SciPy is at about 1e-15.
    let sizes: &[usize] = if cfg!(miri) {
        &[130]
    } else {
        &[200, 400, 1000]
    };
    for &n in sizes {
        for (a, b) in [(0.0, 1e-3), (0.0, 1.0), (0.0, 1000.0), (-1e5, 3e5)] {
            let len = b - a;
            let f = |t: f64| (3.0 * (t - a) / len).cos();
            let x = chebyshev_nodes(n, a, b);
            let y: Vec<f64> = x.iter().map(|&t| f(t)).collect();
            let w = barycentric_weights(&x);
            assert!(w.iter().all(|v| v.is_finite() && *v != 0.0));
            assert!(w.iter().all(|v| v.abs() <= 1.0));
            let mut err = 0.0f64;
            for k in 0..50 {
                let t = a + len * (f64::from(k) + 0.37) / 50.0;
                err = err.max((barycentric_interpolate(&x, &y, &w, t) - f(t)).abs());
            }
            assert!(err < 1e-14, "{n} nodes on [{a}, {b}]: {err:e}");
        }
    }
    let sizes32: &[usize] = if cfg!(miri) {
        &[70]
    } else {
        &[30, 64, 128, 300]
    };
    for &n in sizes32 {
        for (a, b) in [(0.0f32, 1.0f32), (-1.0, 1.0), (0.0, 10.0), (0.0, 100.0)] {
            let len = b - a;
            let f = |t: f32| (3.0 * (t - a) / len).cos();
            let x = chebyshev_nodes(n, a, b);
            let y: Vec<f32> = x.iter().map(|&t| f(t)).collect();
            let w = barycentric_weights(&x);
            let mut err = 0.0f32;
            for k in 0..50 {
                let t = a + len * (k as f32 + 0.37) / 50.0;
                err = err.max((barycentric_interpolate(&x, &y, &w, t) - f(t)).abs());
            }
            assert!(err < 1e-5, "{n} f32 nodes on [{a}, {b}]: {err:e}");
        }
    }
}

#[test]
fn barycentric_weights_of_nodes_near_the_ends_of_the_range() {
    // ±1.5e308 are 3e308 apart, beyond the largest float: that difference is taken of the halves
    // and counted twice as large (review round 1: dropping that power of two passed the tests).
    // Three equally spaced nodes have the weights 1, −2, 1, and the polynomial through data on a
    // line is the line; with differences near 1e308 its terms w/(t − x) were subnormal and the
    // value 2.333 for 2.5, before they were measured in units of the span.
    let x = [-1.5e308f64, 0.0, 1.5e308];
    let w = barycentric_weights(&x);
    assert!(
        (w[1] / w[0] + 2.0).abs() < 1e-15 && (w[2] / w[0] - 1.0).abs() < 1e-15,
        "{w:?}"
    );
    let y = [1.0f64, 2.0, 3.0];
    let p = barycentric_interpolate(&x, &y, &w, 7.5e307);
    assert!((p - 2.5).abs() < 1e-15, "{p}");
}

#[test]
fn barycentric_interpolation_next_to_a_node_is_its_datum() {
    // Audit n_interp-14 (b): one subnormal away from a node the term overflowed and the answer
    // was inf/inf = NaN.
    let x = [0.0f64, 0.5, 1.0];
    let w = barycentric_weights(&x);
    assert_eq!(
        barycentric_interpolate(&x, &[1.0, 2.0, 3.0], &w, 5e-324),
        1.0
    );
    assert_eq!(
        barycentric_interpolate(&x, &[1.0, 2.0, 3.0], &w, -5e-324),
        1.0
    );
    // The float after a node is a point like any other, of the line 1 + 2t there.
    let next = 0.5f64.next_up();
    assert!(
        (barycentric_interpolate(&x, &[1.0, 2.0, 3.0], &w, next) - (1.0 + 2.0 * next)).abs()
            <= 4.5e-16
    );
}

#[test]
#[should_panic(expected = "weights differ in length")]
fn barycentric_interpolation_without_weights_panics() {
    // Audit n_interp-14 (a): an empty `w` used to weight every node equally, a rational function
    // with a pole between every two nodes (5 at 0.5 for data on y = x). The weights are needed.
    let _ = barycentric_interpolate(&[0.0f64, 1.0, 2.0], &[0.0, 1.0, 2.0], &[], 0.5);
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

#[test]
#[should_panic(expected = "nodes must be finite")]
fn infinite_nodes_panic() {
    let _ = barycentric_weights(&[0.0, 1.0, f64::INFINITY]);
}
