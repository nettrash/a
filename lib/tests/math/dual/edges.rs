//! The derivatives at the edges: zeros of either sign, singular points and kinks, ties,
//! infinities and NaN, and a tangent that does not move.

use a::math::dual::{Dual, DualNum, HyperDual, gradient};
use a::math::matrix::Matrix;

const INF: f64 = f64::INFINITY;
const SECH2_30: f64 = 3.502_604_305_078_608e-26;

/// `f(x + ε₁ + ε₂)` as (value, f′, f″).
fn second(f: impl Fn(HyperDual<f64>) -> HyperDual<f64>, x: f64) -> (f64, f64, f64) {
    let h = f(HyperDual::new(x, 1.0, 1.0, 0.0));
    assert_eq!(
        h.eps1.to_bits(),
        h.eps2.to_bits(),
        "the two directions differ at {x}"
    );
    (h.re, h.eps1, h.eps1eps2)
}

/// `f(x + ε)` as (value, f′).
fn first(f: impl Fn(Dual<f64>) -> Dual<f64>, x: f64) -> (f64, f64) {
    let d = f(Dual::new(x, 1.0));
    (d.re, d.eps)
}

#[test]
fn roots_have_infinite_slopes_at_zero_of_either_sign() {
    assert_eq!(first(|x| x.sqrt(), 0.0), (0.0, INF));
    let (v, d) = first(|x| x.sqrt(), -0.0);
    assert!(v == 0.0 && v.is_sign_negative() && d == INF);
    assert_eq!(second(|x| x.sqrt(), 0.0), (0.0, INF, -INF));
    assert_eq!(second(|x| x.sqrt(), -0.0).1, INF);
    assert_eq!(second(|x| x.sqrt(), -0.0).2, -INF);
    let (v, d) = first(|x| x.sqrt(), -1.0);
    assert!(v.is_nan() && d.is_nan());
    assert_eq!(first(|x| x.sqrt(), INF), (INF, 0.0));
    // The cube root: +∞ at either zero; the second derivative takes the side of the zero's sign.
    assert_eq!(second(|x| x.cbrt(), 0.0), (0.0, INF, -INF));
    assert_eq!(second(|x| x.cbrt(), -0.0).1, INF);
    assert_eq!(second(|x| x.cbrt(), -0.0).2, INF);
    assert_eq!(first(|x| x.cbrt(), -8.0), (-2.0, 1.0 / 12.0));
}

#[test]
fn logarithms_have_their_slope_from_the_right_at_zero() {
    for f in [
        (|x: HyperDual<f64>| x.ln()) as fn(HyperDual<f64>) -> HyperDual<f64>,
        |x| x.log2(),
        |x| x.log10(),
        |x| x.log(3.0),
    ] {
        for z in [0.0, -0.0] {
            let (v, d1, d2) = second(f, z);
            assert!(v == -INF && d1 == INF && d2 == -INF, "{v} {d1} {d2} at {z}");
        }
    }
    // ln_1p at −1, its zero.
    assert_eq!(second(|x| x.ln_1p(), -1.0), (-INF, INF, -INF));
    // Outside the domain the value is NaN; the derivative is what the formula gives, 1/x.
    let (v, d) = first(|x| x.ln(), -2.0);
    assert!(v.is_nan() && d == -0.5);
    // Near zero the derivatives are those of the function, not of 1 + x and eˣ − 1 rounded.
    assert_eq!(second(|x| x.ln_1p(), 1e-20), (1e-20, 1.0, -1.0));
    assert_eq!(second(|x| x.exp_m1(), 1e-20), (1e-20, 1.0, 1.0));
    assert_eq!(second(|x| x.exp_m1(), -INF), (-1.0, 0.0, 0.0));
}

#[test]
fn powers_at_zero_and_below() {
    // x^e at 0: 0 for e = 0 (also its derivatives), 1 for e = 1, 0 for e > 1, ∞ for e < 1.
    assert_eq!(second(|x| x.powf(0.0), 0.0), (1.0, 0.0, 0.0));
    assert_eq!(second(|x| x.powf(1.0), 0.0), (0.0, 1.0, 0.0));
    assert_eq!(second(|x| x.powf(2.0), 0.0), (0.0, 0.0, 2.0));
    assert_eq!(second(|x| x.powf(0.5), 0.0), (0.0, INF, -INF));
    assert_eq!(second(|x| x.powf(-1.0), 0.0), (INF, -INF, INF));
    // Below 0, where e is an integer, the derivatives are finite.
    assert_eq!(second(|x| x.powf(3.0), -2.0), (-8.0, 12.0, -12.0));
    assert_eq!(second(|x| x.powf(-2.0), -2.0), (0.25, 0.25, 0.375));
    let (v, d) = first(|x| x.powf(0.5), -2.0);
    assert!(v.is_nan() && d.is_nan());
    // powi the same, and at the ends of i32.
    assert_eq!(second(|x| x.powi(0), 0.0), (1.0, 0.0, 0.0));
    assert_eq!(second(|x| x.powi(1), 0.0), (0.0, 1.0, 0.0));
    assert_eq!(second(|x| x.powi(2), 0.0), (0.0, 0.0, 2.0));
    assert_eq!(second(|x| x.powi(-1), 0.0), (INF, -INF, INF));
    let min = f64::from(i32::MIN);
    assert_eq!(
        second(|x| x.powi(i32::MIN), 1.0),
        (1.0, min, min * (min - 1.0))
    );
    assert_eq!(
        second(|x| x.powi(i32::MIN), -1.0),
        (1.0, -min, min * (min - 1.0))
    );
    let m1 = f64::from(i32::MIN + 1);
    assert_eq!(
        second(|x| x.powi(i32::MIN + 1), -1.0),
        (-1.0, m1, -m1 * (m1 - 1.0))
    );
    assert_eq!(second(|x| x.powi(i32::MAX), 1.0).1, f64::from(i32::MAX));
}

#[test]
fn a_power_with_a_varying_exponent() {
    // x^y at (2, 3): ∂x = y·x^(y−1) = 12, ∂y = x^y·ln x = 8 ln 2.
    let p = Dual::new(2.0, 1.0).pow(Dual::new(3.0, 1.0));
    assert_eq!(p.re, 8.0);
    assert!((p.eps - (12.0 + 8.0 * 2f64.ln())).abs() < 1e-14);
    // A negative base with a constant exponent: the derivative of x³, not NaN from ln x.
    let q = Dual::new(-2.0, 1.0).pow(Dual::new(3.0, 0.0));
    assert_eq!((q.re, q.eps), (-8.0, 12.0));
    let hq = HyperDual::new(-2.0, 1.0, 1.0, 0.0).pow(HyperDual::new(3.0, 0.0, 0.0, 0.0));
    assert_eq!((hq.re, hq.eps1, hq.eps1eps2), (-8.0, 12.0, -12.0));
    // With the exponent varying there is no derivative.
    assert!(
        Dual::new(-2.0_f64, 0.0)
            .pow(Dual::new(3.0, 1.0))
            .eps
            .is_nan()
    );
    // At x = 0 the derivative in the exponent is its limit 0, for y > 0.
    let z = Dual::new(0.0, 0.0).pow(Dual::new(2.0, 1.0));
    assert_eq!((z.re, z.eps), (0.0, 0.0));
    let hz = HyperDual::new(0.0, 0.0, 0.0, 0.0).pow(HyperDual::new(2.0, 1.0, 1.0, 0.0));
    assert_eq!((hz.re, hz.eps1, hz.eps1eps2), (0.0, 0.0, 0.0));
    // An exponent of zero: x⁰ = 1 for every x, with no slope in x.
    let o = Dual::new(5.0, 1.0).pow(Dual::new(0.0, 0.0));
    assert_eq!((o.re, o.eps), (1.0, 0.0));
    // For floats, pow is powf.
    assert_eq!(DualNum::pow(2.0_f64, 0.5), 2f64.sqrt());
}

#[test]
fn inverse_functions_at_their_branch_points() {
    for x in [1.0, -1.0] {
        assert_eq!(first(|v| v.asin(), x).1, INF);
        assert_eq!(first(|v| v.acos(), x).1, -INF);
        assert_eq!(first(|v| v.atanh(), x).1, INF);
        assert!(first(|v| v.asin(), 1.5 * x).1.is_nan());
    }
    assert_eq!(second(|v| v.asin(), 1.0).2, INF);
    assert_eq!(second(|v| v.asin(), -1.0).2, -INF);
    assert_eq!(second(|v| v.acos(), 1.0).2, -INF);
    assert_eq!(first(|v| v.acosh(), 1.0), (0.0, INF));
    assert!(first(|v| v.acosh(), 0.5).1.is_nan());
    // Just inside, the derivative keeps its digits: 1/√((1 − x)(1 + x)) with 1 − x exact.
    let x = 1.0 - 2f64.powi(-53);
    let want = 1.0 / (2f64.powi(-53) * (2.0 - 2f64.powi(-53))).sqrt();
    assert!((first(|v| v.acos(), x).1 + want).abs() <= 2e-16 * want);
    // atanh near −1 is the negative of atanh near 1, as the function is odd.
    let y = -1.0 + 2f64.powi(-53);
    assert_eq!(DualNum::atanh(y), -DualNum::atanh(-y));
    assert_eq!(first(|v| v.atanh(), y).0, -(1.0 - 2f64.powi(-53)).atanh());
}

#[test]
fn tanh_keeps_its_slope_far_out() {
    // sech²(30), where 1 − tanh² would be 0 (mpmath).
    let (t, d) = first(|v| v.tanh(), 30.0);
    assert_eq!(t, 1.0);
    assert!((d - SECH2_30).abs() <= 4e-16 * d);
    for x in [-1000.0, 1000.0, INF, -INF] {
        let (v, d1, d2) = second(|v| v.tanh(), x);
        assert!(
            v.abs() == 1.0 && d1 == 0.0 && d2 == 0.0,
            "{v} {d1} {d2} at {x}"
        );
    }
    // And in f32, where cosh overflows from 89.
    let h = Dual::new(Dual::new(100.0_f32, 1.0), Dual::new(1.0, 0.0)).tanh();
    assert!(h.eps.eps == 0.0 && h.eps.re == 0.0);
}

#[test]
fn atan2_and_hypot_at_the_origin_and_on_the_axes() {
    // On the negative x axis: π and −π by the sign of zero, with the same derivatives.
    let up = Dual::new(0.0, 1.0).atan2(Dual::new(-1.0, 0.0));
    let down = Dual::new(-0.0, 1.0).atan2(Dual::new(-1.0, 0.0));
    assert_eq!((up.re, up.eps), (core::f64::consts::PI, -1.0));
    assert_eq!((down.re, down.eps), (-core::f64::consts::PI, -1.0));
    let across = Dual::new(0.0, 0.0).atan2(Dual::new(-1.0, 1.0));
    assert_eq!(across.eps, 0.0);
    // At the origin the angle is not continuous: NaN.
    let o = Dual::new(0.0_f64, 1.0).atan2(Dual::new(0.0, 0.0));
    assert!(o.re == 0.0 && o.eps.is_nan());
    // hypot at the origin: 0 with zero derivatives (the slope of least size of the cone).
    let h = Dual::new(0.0, 1.0).hypot(Dual::new(-0.0, 1.0));
    assert_eq!((h.re, h.eps), (0.0, 0.0));
    let hh = HyperDual::new(0.0, 1.0, 1.0, 0.0).hypot(HyperDual::new(0.0, 0.0, 0.0, 0.0));
    assert_eq!((hh.re, hh.eps1, hh.eps1eps2), (0.0, 0.0, 0.0));
    // On an axis hypot is |x|, and does not overflow beyond the squares.
    assert_eq!(
        second(|v| v.hypot(HyperDual::constant(0.0)), -3.0),
        (3.0, -1.0, 0.0)
    );
    let big = Dual::new(1e300, 1.0).hypot(Dual::new(1e300, 0.0));
    assert!((big.eps - core::f64::consts::FRAC_1_SQRT_2).abs() <= 2.3e-16);
}

#[test]
fn kinks_and_ties_take_the_stated_derivative() {
    // abs: the sign, and 0 at either zero.
    assert_eq!(second(|x| x.abs(), -2.0), (2.0, -1.0, 0.0));
    assert_eq!(second(|x| x.abs(), 2.0), (2.0, 1.0, 0.0));
    assert_eq!(second(|x| x.abs(), 0.0), (0.0, 0.0, 0.0));
    assert_eq!(second(|x| x.abs(), -0.0), (0.0, 0.0, 0.0));
    let n = first(|x| x.abs(), f64::NAN);
    assert!(n.0.is_nan());
    // signum: ±1, with no slope, at the jump too.
    assert_eq!(second(|x| x.signum(), 0.0), (1.0, 0.0, 0.0));
    assert_eq!(second(|x| x.signum(), -0.0), (-1.0, 0.0, 0.0));
    assert_eq!(second(|x| x.signum(), -3.0), (-1.0, 0.0, 0.0));
    // min and max: the operand chosen, the receiver on a tie, the other one beside a NaN.
    let (a, b) = (Dual::new(1.0, 10.0), Dual::new(1.0, 20.0));
    assert_eq!(a.min(b).eps, 10.0);
    assert_eq!(b.min(a).eps, 20.0);
    assert_eq!(a.max(b).eps, 10.0);
    assert_eq!(Dual::new(0.0, 1.0).min(Dual::new(2.0, 5.0)).eps, 1.0);
    assert_eq!(Dual::new(0.0, 1.0).max(Dual::new(2.0, 5.0)).eps, 5.0);
    let nan = Dual::new(f64::NAN, 3.0);
    assert_eq!(nan.min(b).eps, 20.0);
    assert_eq!(b.max(nan).eps, 20.0);
    let h = HyperDual::new(2.0, 1.0, 1.0, 7.0).max(HyperDual::new(1.0, 0.0, 0.0, 0.0));
    assert_eq!(h.eps1eps2, 7.0);
}

#[test]
fn a_tangent_that_does_not_move_contributes_nothing() {
    // ∂/∂x of x + √y at y = 0, where √′ is infinite: 1, not NaN from ∞·0.
    let f = |v: &[Dual<f64>]| v[0] + v[1].sqrt() + v[1].powf(0.5);
    let g = gradient(f, &Matrix::from_column_major(2, 1, vec![1.0, 0.0]));
    assert_eq!(g[(0, 0)], 1.0);
    assert_eq!(g[(1, 0)], INF);
    let c = Dual::new(0.0, 0.0);
    for v in [
        c.sqrt(),
        c.cbrt(),
        c.ln(),
        c.recip(),
        c.powf(0.5),
        c.asin() * 0.0 + Dual::new(1.0, 0.0).acos(),
    ] {
        assert_eq!(v.eps, 0.0);
    }
    let h = HyperDual::new(0.0, 0.0, 0.0, 0.0).sqrt();
    assert_eq!((h.eps1, h.eps2, h.eps1eps2), (0.0, 0.0, 0.0));
    let k = HyperDual::new(1.0, 0.0, 0.0, 0.0).acos();
    assert_eq!((k.eps1, k.eps2, k.eps1eps2), (0.0, 0.0, 0.0));
}

#[test]
fn infinities_and_nan_pass_through() {
    assert_eq!(second(|x| x.exp(), INF), (INF, INF, INF));
    assert_eq!(second(|x| x.exp(), -INF), (0.0, 0.0, 0.0));
    assert_eq!(first(|x| x.recip(), INF), (0.0, -0.0));
    assert_eq!(first(|x| x.atan(), INF).1, 0.0);
    assert_eq!(first(|x| x.atan(), -1e300).1, 0.0);
    assert_eq!(first(|x| x.ln(), INF), (INF, 0.0));
    for f in [
        (|x: HyperDual<f64>| x.sin()) as fn(HyperDual<f64>) -> HyperDual<f64>,
        |x| x.exp(),
        |x| x.sqrt(),
        |x| x.tanh(),
        |x| x.atan(),
    ] {
        let (v, d1, d2) = second(f, f64::NAN);
        assert!(v.is_nan() && d1.is_nan() && d2.is_nan());
    }
}

#[test]
fn an_overflowed_value_turns_nested_derivatives_nan() {
    // A limit, not a goal: x⁵ at 10¹⁰ in f32 overflows, and Dual<Dual<f32>> carries x⁴ = ∞ as a
    // value whose product with a zero tangent part is NaN, where the hyper-dual number, which
    // forms f″ = 20x³ directly, has it.
    let x = 1e10_f32;
    let nested = Dual::new(Dual::new(x, 1.0), Dual::new(1.0, 0.0)).powi(5);
    assert!(nested.re.re.is_infinite() && nested.eps.eps.is_nan());
    let hyper = HyperDual::new(x, 1.0, 1.0, 0.0).powi(5);
    assert!((hyper.eps1eps2 - 2e31).abs() <= 2e31 * 4.0 * f32::EPSILON);
}

#[test]
fn mul_add_is_the_product_rule() {
    let (x, a, b) = (
        Dual::new(2.0, 1.0),
        Dual::new(3.0, -1.0),
        Dual::new(0.5, 4.0),
    );
    let r = x.mul_add(a, b);
    assert_eq!((r.re, r.eps), (6.5, 3.0 - 2.0 + 4.0));
    let (hx, ha, hb) = (
        HyperDual::new(2.0, 1.0, 0.0, 0.0),
        HyperDual::new(3.0, 0.0, 1.0, 0.0),
        HyperDual::new(0.5, 0.0, 0.0, 1.0),
    );
    let h = hx.mul_add(ha, hb);
    let p = hx * ha + hb;
    assert_eq!(
        (h.re, h.eps1, h.eps2, h.eps1eps2),
        (p.re, p.eps1, p.eps2, p.eps1eps2)
    );
}
