//! The product, quotient and chain rules to second order on random pairs of functions, random
//! compositions three deep against Richardson extrapolation (`numerics::derivative` and
//! `numerics::second_derivative`) to its tolerance, and their third derivatives by nesting.

use a::math::dual::{Dual, DualNum, HyperDual, derivative, second_derivative};
use a::math::numerics;

/// xorshift64*: reproducible draws without a dependency.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// Smooth functions of every real number, of moderate growth, so that compositions of them stay
/// in range: the pieces of the random compositions.
const PIECES: usize = 14;

fn piece<D: DualNum>(k: usize, x: D) -> D {
    let c = D::from_f64;
    match k {
        0 => x.sin(),
        1 => x.cos(),
        2 => x.atan(),
        3 => x.tanh(),
        4 => x.asinh(),
        5 => (x * x + c(1.0)).ln(),
        6 => (x * x + c(1.0)).sqrt(),
        7 => (x * c(0.5)).exp(),
        8 => (x * x + c(1.0)).recip(),
        9 => (x * x).ln_1p() * x,
        10 => (x * c(0.3)).sinh() + (x * c(0.2)).cosh(),
        11 => (x * x + c(2.0)).powf(D::from_f64(1.5).re()),
        12 => (x * c(0.4)).exp_m1().atan2(x.cos() + c(2.0)),
        _ => (x * x + c(1.0)).log2() + x.hypot(c(0.5)),
    }
}

/// A composition `f₃(f₂(f₁(x)))` of pieces, for any type.
#[derive(Clone, Copy, Debug)]
struct Composition([usize; 3]);

impl Composition {
    fn eval<D: DualNum>(self, x: D) -> D {
        self.0.iter().fold(x, |v, &k| piece(k, v))
    }
}

fn hyper(f: impl Fn(HyperDual<f64>) -> HyperDual<f64>, x: f64) -> [f64; 3] {
    let h = f(HyperDual::new(x, 1.0, 1.0, 0.0));
    [h.re, h.eps1, h.eps1eps2]
}

/// Whether `got` is `want` to `tol` times the size `size`.
fn near(got: f64, want: f64, size: f64, tol: f64) -> bool {
    (got - want).abs() <= tol * size.max(f64::MIN_POSITIVE)
}

#[test]
fn product_quotient_and_chain_rules_hold_to_second_order() {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let cases = if cfg!(miri) { 20 } else { 2_000 };
    let tol = 64.0 * f64::EPSILON;
    for _ in 0..cases {
        let (i, j) = (rng.below(PIECES), rng.below(PIECES));
        let x = rng.uniform(-2.0, 2.0);
        let [f, f1, f2] = hyper(|v| piece(i, v), x);
        let [g, g1, g2] = hyper(|v| piece(j, v), x);
        // (fg)″ = f″g + 2f′g′ + fg″.
        let [p, p1, p2] = hyper(|v| piece(i, v) * piece(j, v), x);
        let s = (f2 * g).abs() + 2.0 * (f1 * g1).abs() + (f * g2).abs();
        assert!(near(p, f * g, (f * g).abs(), tol));
        assert!(near(
            p1,
            f1 * g + f * g1,
            (f1 * g).abs() + (f * g1).abs(),
            tol
        ));
        assert!(
            near(p2, f2 * g + 2.0 * f1 * g1 + f * g2, s, tol),
            "({i}, {j}) at {x}"
        );
        // (f/h)′ and (f/h)″ with h = g + 3, which stays away from 0.
        let h = g + 3.0;
        let [q, q1, q2] = hyper(|v| piece(i, v) / (piece(j, v) + 3.0), x);
        let want1 = (f1 * h - f * g1) / (h * h);
        let want2 =
            f2 / h - 2.0 * f1 * g1 / (h * h) - f * g2 / (h * h) + 2.0 * f * g1 * g1 / (h * h * h);
        let s2 = (f2 / h).abs()
            + (2.0 * f1 * g1 / (h * h)).abs()
            + (f * g2 / (h * h)).abs()
            + (2.0 * f * g1 * g1 / (h * h * h)).abs();
        assert!(near(q, f / h, (f / h).abs(), tol));
        assert!(near(
            q1,
            want1,
            (f1 / h).abs() + (f * g1 / (h * h)).abs(),
            tol
        ));
        assert!(near(q2, want2, s2, tol), "({i}, {j}) at {x}");
        // (f∘g)′ = f′(g)·g′ and (f∘g)″ = f″(g)·g′² + f′(g)·g″.
        let [_, o1, o2] = hyper(|v| piece(i, v), g);
        let [c, c1, c2] = hyper(|v| piece(i, piece(j, v)), x);
        assert!(near(c, piece(i, g), piece(i, g).abs(), tol));
        assert!(near(c1, o1 * g1, (o1 * g1).abs(), tol));
        assert!(near(
            c2,
            o2 * g1 * g1 + o1 * g2,
            (o2 * g1 * g1).abs() + (o1 * g2).abs(),
            tol
        ));
    }
}

#[test]
fn random_compositions_agree_with_richardson_extrapolation() {
    let mut rng = Rng(0x2545_f491_4f6c_dd1d);
    let cases = if cfg!(miri) { 5 } else { 300 };
    let (mut first, mut second, mut answered) = (0.0_f64, 0.0_f64, 0);
    for _ in 0..cases {
        let comp = Composition([rng.below(PIECES), rng.below(PIECES), rng.below(PIECES)]);
        let x = rng.uniform(-2.0, 2.0);
        let d = derivative(|v: Dual<f64>| comp.eval(v), x);
        let dd = second_derivative(|v: HyperDual<f64>| comp.eval(v), x);
        // Richardson's answers are verified to its tolerance (√ε for the first derivative,
        // ε^(1/3) for the second) relative to the size of the derivative or of the slopes about
        // it; the dual numbers' must lie within a few times that.
        if let Some(r) = numerics::derivative(|v: f64| comp.eval(v), x) {
            let e = (d - r).abs() / d.abs().max(1.0);
            assert!(e <= 1e-7, "{comp:?} at {x}: dual {d}, Richardson {r}");
            first = first.max(e);
            answered += 1;
        }
        if let Some(r) = numerics::second_derivative(|v: f64| comp.eval(v), x) {
            let e = (dd - r).abs() / dd.abs().max(1.0);
            assert!(e <= 1e-4, "{comp:?} at {x}: dual {dd}, Richardson {r}");
            second = second.max(e);
        }
    }
    assert!(
        answered >= cases * 9 / 10,
        "Richardson answered {answered} of {cases}"
    );
    println!("largest difference from Richardson: first {first:.1e}, second {second:.1e}");
}

#[test]
fn third_derivatives_by_nesting_agree_with_each_other_and_with_richardson() {
    let mut rng = Rng(0x1234_5678_9abc_def1);
    let cases = if cfg!(miri) { 5 } else { 300 };
    for _ in 0..cases {
        let comp = Composition([rng.below(PIECES), rng.below(PIECES), rng.below(PIECES)]);
        let x = rng.uniform(-2.0, 2.0);
        // f‴ two ways: the ε part of f″ by a dual pass over hyper-dual numbers, and the ε₁ε₂ part
        // of f′ by a hyper-dual pass over dual numbers.
        let dh = comp.eval(Dual::new(
            HyperDual::new(x, 1.0, 1.0, 0.0),
            HyperDual::new(1.0, 0.0, 0.0, 0.0),
        ));
        let hd = comp.eval(HyperDual::new(
            Dual::new(x, 1.0),
            Dual::new(1.0, 0.0),
            Dual::new(1.0, 0.0),
            Dual::new(0.0, 0.0),
        ));
        let (a, b) = (dh.eps.eps1eps2, hd.eps1eps2.eps);
        assert!(
            (a - b).abs() <= 1e-12 * a.abs().max(1.0),
            "{comp:?} at {x}: {a} and {b}"
        );
        // And a fourth way: derivative(second_derivative) through the free functions, nested.
        let c = derivative(
            |v: Dual<f64>| second_derivative(|w: HyperDual<Dual<f64>>| comp.eval(w), v),
            x,
        );
        assert!(
            (a - c).abs() <= 1e-12 * a.abs().max(1.0),
            "{comp:?} at {x}: {a} and {c}"
        );
        // Richardson on the exact second derivative.
        if let Some(r) = numerics::derivative(
            |v: f64| second_derivative(|w: HyperDual<f64>| comp.eval(w), v),
            x,
        ) {
            assert!(
                (a - r).abs() <= 1e-6 * a.abs().max(1.0),
                "{comp:?} at {x}: {a} and {r}"
            );
        }
    }
}

#[test]
fn the_same_function_gives_the_same_value_in_every_type() {
    let mut rng = Rng(7);
    for _ in 0..if cfg!(miri) { 5 } else { 200 } {
        let comp = Composition([rng.below(PIECES), rng.below(PIECES), rng.below(PIECES)]);
        let x = rng.uniform(-2.0, 2.0);
        let v = comp.eval(x);
        assert_eq!(comp.eval(Dual::new(x, 1.0)).re, v);
        assert_eq!(comp.eval(HyperDual::new(x, 1.0, 1.0, 0.0)).re, v);
        assert_eq!(
            comp.eval(Dual::new(Dual::new(x, 1.0), Dual::new(1.0, 0.0)))
                .re
                .re,
            v
        );
        let v32 = comp.eval(x as f32);
        assert_eq!(comp.eval(Dual::new(x as f32, 1.0)).re, v32);
        // f32 and f64 derivatives agree to f32's precision.
        let d64 = derivative(|t: Dual<f64>| comp.eval(t), x);
        let d32 = derivative(|t: Dual<f32>| comp.eval(t), x as f32);
        let x32 = f64::from(x as f32);
        let d64_at_x32 = derivative(|t: Dual<f64>| comp.eval(t), x32);
        assert!(
            (f64::from(d32) - d64_at_x32).abs() <= 1e-3 * d64_at_x32.abs().max(1.0),
            "{comp:?} at {x}: {d32} and {d64}"
        );
    }
}
