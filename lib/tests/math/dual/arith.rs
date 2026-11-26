//! The arithmetic of dual and hyper-dual numbers: the operators in every form, the rules of the
//! algebra on exact cases, comparisons by value, `Sum`, `Product`, constants and `Display`.

#![allow(clippy::op_ref)]

use a::math::dual::{Dual, DualNum, HyperDual};

fn parts(d: Dual<f64>) -> (f64, f64) {
    (d.re, d.eps)
}

fn hparts(h: HyperDual<f64>) -> (f64, f64, f64, f64) {
    (h.re, h.eps1, h.eps2, h.eps1eps2)
}

#[test]
fn the_product_rule_holds() {
    // (x² + 1)(3x) at x = 2: value 30, derivative 2x·3x + (x² + 1)·3 = 24 + 15 = 39.
    let x = Dual::new(2.0, 1.0);
    let f = (x * x + Dual::new(1.0, 0.0)) * (Dual::new(3.0, 0.0) * x);
    assert_eq!((f.re, f.eps), (30.0, 39.0));
}

#[test]
fn dual_operators_take_owned_and_borrowed_operands() {
    let (a, b) = (Dual::new(3.0, 2.0), Dual::new(0.5, -1.0));
    // (a + a′ε) ∘ (b + b′ε) for each operator, by its rule.
    let sum = (3.5, 1.0);
    let difference = (2.5, 3.0);
    let product = (1.5, -3.0 + 2.0 * 0.5);
    let quotient = (6.0, (2.0 * 0.5 + 3.0) / 0.25);
    for (got, want) in [
        (a + b, sum),
        (a + &b, sum),
        (&a + b, sum),
        (&a + &b, sum),
        (a - b, difference),
        (a - &b, difference),
        (&a - b, difference),
        (&a - &b, difference),
        (a * b, product),
        (a * &b, product),
        (&a * b, product),
        (&a * &b, product),
        (a / b, quotient),
        (a / &b, quotient),
        (&a / b, quotient),
        (&a / &b, quotient),
    ] {
        assert_eq!(parts(got), want);
    }
    assert_eq!(parts(-a), (-3.0, -2.0));
    assert_eq!(parts(-&a), (-3.0, -2.0));
    let mut c = a;
    c += b;
    assert_eq!(parts(c), sum);
    c -= &b;
    assert_eq!(parts(c), parts(a));
    c *= b;
    assert_eq!(parts(c), product);
    c /= &b;
    assert_eq!(parts(c), parts(a));
    c *= &b;
    c /= b;
    c += &b;
    c -= b;
    assert_eq!(parts(c), parts(a));
}

#[test]
fn dual_operators_take_a_float_on_either_side() {
    let a = Dual::new(3.0_f64, 2.0);
    for (got, want) in [
        (a + 1.5, (4.5, 2.0)),
        (a + &1.5, (4.5, 2.0)),
        (&a + 1.5, (4.5, 2.0)),
        (&a + &1.5, (4.5, 2.0)),
        (1.5 + a, (4.5, 2.0)),
        (&1.5 + a, (4.5, 2.0)),
        (1.5 + &a, (4.5, 2.0)),
        (&1.5 + &a, (4.5, 2.0)),
        (a - 1.0, (2.0, 2.0)),
        (1.0 - a, (-2.0, -2.0)),
        (&1.0 - &a, (-2.0, -2.0)),
        (a * 2.0, (6.0, 4.0)),
        (2.0 * a, (6.0, 4.0)),
        (&2.0 * &a, (6.0, 4.0)),
        (a / 2.0, (1.5, 1.0)),
        (&a / &2.0, (1.5, 1.0)),
        // 6/x at 3: 2, with the derivative −6/x²·2 = −4/3.
        (6.0 / a, (2.0, -6.0 / 9.0 * 2.0)),
        (&6.0 / &a, (2.0, -6.0 / 9.0 * 2.0)),
    ] {
        assert_eq!(parts(got), want);
    }
    let mut c = a;
    c += 1.0;
    c -= &1.0;
    c *= 4.0;
    c /= &2.0;
    assert_eq!(parts(c), (6.0, 4.0));
    c /= 2.0;
    c *= &1.0;
    c += &0.0;
    c -= 0.0;
    assert_eq!(parts(c), (3.0, 2.0));
    // f32 too.
    let b = Dual::new(1.5_f32, -1.0);
    let d = 2.0 * b - 1.0;
    assert_eq!((d.re, d.eps), (2.0, -2.0));
    let q = 3.0_f32 / b;
    assert_eq!((q.re, q.eps), (2.0, 3.0 / 2.25));
}

#[test]
fn hyper_dual_operators_follow_their_rules() {
    let a = HyperDual::new(2.0, 1.0, 0.5, 0.25);
    let b = HyperDual::new(4.0, -1.0, 2.0, 1.0);
    // The ε₁ε₂ part of a product: a₀b₁₂ + a₁b₂ + a₂b₁ + a₁₂b₀.
    let product = (
        8.0,
        -2.0 + 1.0 * 4.0,
        2.0 * 2.0 + 0.5 * 4.0,
        2.0 + 2.0 - 0.5 + 1.0,
    );
    for (got, want) in [
        (a + b, (6.0, 0.0, 2.5, 1.25)),
        (&a + &b, (6.0, 0.0, 2.5, 1.25)),
        (a - b, (-2.0, 2.0, -1.5, -0.75)),
        (a - &b, (-2.0, 2.0, -1.5, -0.75)),
        (a * b, product),
        (&a * b, product),
        (a + 1.0, (3.0, 1.0, 0.5, 0.25)),
        (1.0 + a, (3.0, 1.0, 0.5, 0.25)),
        (1.0 - a, (-1.0, -1.0, -0.5, -0.25)),
        (a * 2.0, (4.0, 2.0, 1.0, 0.5)),
        (&2.0 * &a, (4.0, 2.0, 1.0, 0.5)),
        (a / 2.0, (1.0, 0.5, 0.25, 0.125)),
        (-a, (-2.0, -1.0, -0.5, -0.25)),
        (-&a, (-2.0, -1.0, -0.5, -0.25)),
    ] {
        assert_eq!(hparts(got), want);
    }
    // The quotient undoes the product, and 1/b is b's reciprocal.
    let q = (a * b) / b;
    assert_eq!(hparts(q), hparts(a));
    let r = 1.0 / b;
    assert_eq!(hparts(r * b), (1.0, 0.0, 0.0, 0.0));
    assert_eq!(hparts(b.recip()), hparts(r));
    let mut c = a;
    c *= b;
    c /= &b;
    c += 1.0;
    c -= &1.0;
    c *= &2.0;
    c /= 2.0;
    c += b;
    c -= b;
    assert_eq!(hparts(c), hparts(a));
}

#[test]
fn a_quotient_does_not_square_the_divisor() {
    // (1e300 + 1e300ε)/(1e300 + 1ε) = 1 + (1e300 − 1)/1e300·ε, where b₀² would overflow.
    let q = Dual::new(1e300, 1e300) / Dual::new(1e300, 1.0);
    assert_eq!(parts(q), (1.0, 1.0));
    let h = HyperDual::new(1e300, 1e300, 1e300, 0.0) / HyperDual::new(1e300, 0.0, 0.0, 0.0);
    assert_eq!(hparts(h), (1.0, 1.0, 1.0, 0.0));
    // And a tiny divisor whose square would underflow.
    let t = Dual::new(1e-300, 1.0) / Dual::new(1e-300, 0.0);
    assert!(t.re == 1.0 && (t.eps - 1e300).abs() <= 4e-16 * 1e300);
}

#[test]
fn sum_and_product_take_owned_and_borrowed_numbers() {
    let v = [
        Dual::new(1.0, 1.0),
        Dual::new(2.0, 0.0),
        Dual::new(3.0, -1.0),
    ];
    assert_eq!(parts(v.iter().sum()), (6.0, 0.0));
    assert_eq!(parts(v.into_iter().sum()), (6.0, 0.0));
    // (x)(2)(3 − ...) by the product rule: 6, with 1·2·3 + 1·2·(−1) = 4.
    assert_eq!(parts(v.iter().product()), (6.0, 4.0));
    assert_eq!(parts(v.into_iter().product()), (6.0, 4.0));
    let none: [Dual<f64>; 0] = [];
    assert_eq!(parts(none.iter().sum()), (0.0, 0.0));
    assert_eq!(parts(none.iter().product()), (1.0, 0.0));
    let h = [
        HyperDual::new(2.0, 1.0, 0.0, 0.0),
        HyperDual::new(3.0, 0.0, 1.0, 0.0),
    ];
    assert_eq!(hparts(h.iter().product()), (6.0, 3.0, 2.0, 1.0));
    assert_eq!(hparts(h.into_iter().sum()), (5.0, 1.0, 1.0, 0.0));
}

#[test]
fn comparisons_look_at_the_values_alone() {
    let (a, b) = (Dual::new(1.0, 2.0), Dual::new(1.0, -5.0));
    assert!(a == b);
    assert_eq!(a.partial_cmp(&b), Some(core::cmp::Ordering::Equal));
    assert!(Dual::new(0.5, 9.0) < a && a <= b && a >= b);
    assert!(a == 1.0 && a < 1.5 && a > 0.5 && a <= 1.0);
    let nan = Dual::new(f64::NAN, 0.0);
    assert!(nan != nan && nan.partial_cmp(&a).is_none());
    let h = HyperDual::new(2.0, 1.0, 1.0, 1.0);
    assert!(
        h == HyperDual::new(2.0, 0.0, 0.0, 0.0)
            && h > 1.0
            && h < HyperDual::new(3.0, 0.0, 0.0, 0.0)
    );
    // Nested: the innermost value.
    let n = Dual::new(Dual::new(1.0, 7.0), Dual::new(3.0, 1.0));
    assert!(n == 1.0 && n < 2.0);
    // A branch on a comparison takes the path of the value, so the derivative is the branch's.
    fn relu<D: DualNum<Float = f64>>(x: D) -> D {
        if x > 0.0 { x } else { D::ZERO }
    }
    assert_eq!(relu(Dual::new(2.0, 1.0)).eps, 1.0);
    assert_eq!(relu(Dual::new(-2.0, 1.0)).eps, 0.0);
}

#[test]
fn constants_and_the_value_through_every_nesting() {
    assert_eq!(parts(Dual::<f64>::ZERO), (0.0, 0.0));
    assert_eq!(parts(Dual::<f64>::ONE), (1.0, 0.0));
    assert_eq!(parts(Dual::<f64>::constant(2.5)), (2.5, 0.0));
    assert_eq!(parts(Dual::<f64>::from_f64(0.25)), (0.25, 0.0));
    assert_eq!(parts(Dual::<f64>::default()), (0.0, 0.0));
    assert_eq!(hparts(HyperDual::<f64>::ONE), (1.0, 0.0, 0.0, 0.0));
    assert_eq!(
        hparts(HyperDual::<f64>::constant(-3.0)),
        (-3.0, 0.0, 0.0, 0.0)
    );
    assert_eq!(hparts(HyperDual::default()), (0.0, 0.0, 0.0, 0.0));
    let c = Dual::<HyperDual<f64>>::constant(4.0);
    assert_eq!(hparts(c.re), (4.0, 0.0, 0.0, 0.0));
    assert_eq!(hparts(c.eps), (0.0, 0.0, 0.0, 0.0));
    assert_eq!(c.re(), 4.0);
    assert_eq!(Dual::<f32>::from_f64(0.1).re, 0.1_f32);
    assert_eq!(f32::from_f64(1e40), f32::INFINITY);
    assert_eq!(DualNum::re(2.0_f64), 2.0);
    let deep = HyperDual::new(
        Dual::new(1.5, 1.0),
        Dual::new(1.0, 0.0),
        Dual::new(0.0, 0.0),
        Dual::new(0.0, 0.0),
    );
    assert_eq!(deep.re(), 1.5);
}

#[test]
fn nested_numbers_take_floats_on_either_side() {
    let x = Dual::new(Dual::new(2.0_f64, 1.0), Dual::new(1.0, 0.0));
    // 3x² − 1 at 2: 11, with 6x = 12 and 6 in both directions and the second derivative.
    let f = 3.0 * x * x - 1.0;
    assert_eq!(
        (f.re.re, f.re.eps, f.eps.re, f.eps.eps),
        (11.0, 12.0, 12.0, 6.0)
    );
    let g = 1.0 / x;
    assert_eq!(
        (g.re.re, g.re.eps, g.eps.re, g.eps.eps),
        (0.5, -0.25, -0.25, 0.25)
    );
    let h = Dual::new(
        HyperDual::new(2.0_f32, 1.0, 1.0, 0.0),
        HyperDual::new(1.0, 0.0, 0.0, 0.0),
    );
    let cube = h * h * h * 0.5;
    // ½x³: the third derivative is 3.
    assert_eq!(cube.eps.eps1eps2, 3.0);
}

#[test]
fn display_writes_each_part_with_its_unit() {
    assert_eq!(format!("{}", Dual::new(1.5, -2.0)), "1.5-2ε");
    assert_eq!(format!("{}", Dual::new(1.5, 2.0)), "1.5+2ε");
    assert_eq!(format!("{:.2}", Dual::new(1.5, -2.0)), "1.50-2.00ε");
    assert_eq!(format!("{}", Dual::new(1.0, -0.0)), "1-0ε");
    assert_eq!(
        format!("{}", Dual::new(f64::NAN, f64::INFINITY)),
        "NaN+infε"
    );
    assert_eq!(
        format!("{}", Dual::new(-1.0_f32, f32::NEG_INFINITY)),
        "-1-infε"
    );
    assert_eq!(
        format!("{}", HyperDual::new(1.0, -2.0, 0.5, 3.0)),
        "1-2ε₁+0.5ε₂+3ε₁ε₂"
    );
    assert_eq!(
        format!("{:.1}", HyperDual::new(1.0, 2.0, -0.5, 0.0)),
        "1.0+2.0ε₁-0.5ε₂+0.0ε₁ε₂"
    );
    let n = Dual::new(Dual::new(1.0, 1.0), Dual::new(-1.0, 0.0));
    assert_eq!(format!("{n}"), "(1+1ε)+(-1+0ε)ε");
    assert_eq!(format!("{n:.1}"), "(1.0+1.0ε)+(-1.0+0.0ε)ε");
}

#[test]
fn powers_by_multiplication_agree_with_powi() {
    // The arithmetic and the elementary functions agree where both are exact.
    let x = HyperDual::new(1.5, 1.0, 1.0, 0.0);
    let by_products = x * x * x * x;
    assert_eq!(hparts(x.powi(4)), hparts(by_products));
    assert_eq!(hparts(x.powf(4.0)), hparts(by_products));
    let d = Dual::new(1.5, 1.0);
    let close = |a: Dual<f64>, b: Dual<f64>| {
        (a.re - b.re).abs() <= 4e-16 * b.re.abs() && (a.eps - b.eps).abs() <= 4e-16 * b.eps.abs()
    };
    assert!(close(d.powi(-2), 1.0 / (d * d)));
    assert!(close(d.sqrt() * d.sqrt(), d));
    assert!(close(d.exp().ln(), d));
}
