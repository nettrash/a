//! Complex numbers: exact cases, branch cuts, extreme magnitudes and identities.

use a::math::complex::{Complex, Float};
use core::f64::consts::{FRAC_PI_2, PI};

fn c(re: f64, im: f64) -> Complex<f64> {
    Complex::new(re, im)
}

/// Asserts `|a - b| ≤ tol·|b|`, or `≤ tol` for `b = 0`.
#[track_caller]
fn near(a: Complex<f64>, b: Complex<f64>, tol: f64) {
    let scale = b.abs().max(1.0);
    assert!((a - b).abs() <= tol * scale, "{a} is not {b}");
}

#[test]
fn arithmetic() {
    let (a, b) = (c(1.0, 2.0), c(3.0, -4.0));
    assert_eq!(a + b, c(4.0, -2.0));
    assert_eq!(a - b, c(-2.0, 6.0));
    assert_eq!(a * b, c(11.0, 2.0));
    assert_eq!(a / b, c(-0.2, 0.4));
    assert_eq!(-a, c(-1.0, -2.0));
    assert_eq!(a.conj(), c(1.0, -2.0));
    assert_eq!(b.abs(), 5.0);
    assert_eq!(b.norm_sqr(), 25.0);
    assert_eq!(c(0.0, 2.0).recip(), c(0.0, -0.5));
    // Owned and borrowed, floats on either side, and the assignments.
    #[allow(clippy::op_ref)]
    let sum = &a + &b;
    assert_eq!(sum, a + b);
    assert_eq!(a * 2.0, c(2.0, 4.0));
    assert_eq!(2.0 * a, c(2.0, 4.0));
    assert_eq!(1.0 - a, c(0.0, -2.0));
    assert_eq!(10.0 / c(3.0, 4.0), c(1.2, -1.6));
    assert_eq!(a / 2.0, c(0.5, 1.0));
    assert_eq!(a + 1.0, c(2.0, 2.0));
    let mut d = a;
    d += b;
    d -= &b;
    d *= 3.0;
    d /= c(3.0, 0.0);
    assert_eq!(d, a);
    // `0·∞` stays out of a real factor's product.
    let inf = c(f64::INFINITY, 0.0);
    assert_eq!(2.0 * inf, c(f64::INFINITY, 0.0));
    assert_eq!(
        [a, b, c(0.0, 1.0)].iter().sum::<Complex<f64>>(),
        c(4.0, -1.0)
    );
    assert_eq!([a, b].into_iter().product::<Complex<f64>>(), a * b);
    assert_eq!(Complex::<f64>::I * Complex::I, -Complex::ONE);
    assert_eq!(Complex::from(2.5), c(2.5, 0.0));
    assert_eq!(Complex::<f64>::default(), Complex::ZERO);
}

#[test]
fn extreme_magnitudes() {
    // The squares would overflow: `abs` and Smith's division don't.
    let big = c(1e300, 1e300);
    assert_eq!(big / big, c(1.0, 0.0));
    assert!((big.abs() - 1e300 * 2f64.sqrt()).abs() <= 1e285);
    let tiny = c(1e-300, -1e-300);
    near(tiny / c(1e-300, 0.0), c(1.0, -1.0), 1e-15);
    near(big.sqrt() * big.sqrt(), big, 1e-15);
    let sub = c(5e-324, 5e-324);
    let r = sub.sqrt();
    assert!(r.re > 0.0 && r.im > 0.0);
    // mpmath at 40 digits.
    near(
        c(f64::MAX, f64::MAX).sqrt(),
        c(1.473_094_556_905_565_2e154, 6.101_757_441_282_702_4e153),
        1e-15,
    );
    assert_eq!(c(1.0, 1.0) / c(0.0, 0.0), c(f64::INFINITY, f64::INFINITY));
}

#[test]
fn the_top_and_the_bottom_of_the_range() {
    // Quotients whose sums would overflow, of either operand, and both.
    let top = c(1e308, 1e308);
    assert_eq!(top / top, Complex::ONE);
    assert_eq!(top / c(1.0, 1.0), c(1e308, 0.0));
    assert_eq!(c(1.0, 1.0) / top, c(1e-308, 0.0));
    let max = c(f64::MAX, -f64::MAX);
    assert_eq!(max / max, Complex::ONE);
    // Its reciprocal is subnormal, a few bits short.
    near(top.recip() * top, Complex::ONE, 1e-14);
    // A zero divisor and non-finite operands as before.
    assert!((top / Complex::ZERO).re.is_infinite());
    assert!((Complex::ZERO / Complex::<f64>::ZERO).re.is_nan());
    assert_eq!(c(1.0, 0.0) / c(f64::INFINITY, 0.0), Complex::ZERO);
    // Logarithms whose magnitude overflows, or is subnormal: mpmath at 40 digits.
    let ln = c(f64::MAX, f64::MAX).ln();
    assert!((ln.re - 710.129_286_483_663_9).abs() <= 1e-13, "{ln}");
    assert!((ln.im - core::f64::consts::FRAC_PI_4).abs() <= 1e-15);
    let ln = c(3e-320, 4e-320).ln();
    assert!((ln.re + 735.217_802_978_539_8).abs() <= 1e-12, "{ln}");
    let ln32 = Complex::new(f32::MAX, f32::MAX).ln();
    assert!((ln32.re - 89.069_41).abs() <= 1e-5, "{ln32}");
    // Zero, infinities and NaNs as before.
    assert_eq!(Complex::<f64>::ZERO.ln().re, f64::NEG_INFINITY);
    assert_eq!(c(f64::INFINITY, f64::NAN).ln().re, f64::INFINITY);
    assert_eq!(c(f64::NAN, f64::INFINITY).ln().re, f64::INFINITY);
    assert!(c(f64::NAN, 1.0).ln().re.is_nan());
    // An exponential whose `e^re` overflows but whose parts don't: mpmath at 40 digits.
    let e = c(710.0, core::f64::consts::FRAC_PI_4).exp();
    near(
        e,
        c(1.579_672_848_288_201_5e308, 1.579_672_848_288_201_5e308),
        1e-15,
    );
    assert!(c(710.0, 0.0).exp().re.is_infinite());
    // Quotients whose intermediate products underflow: of subnormal operands, and where
    // `q/p` underflows.
    let u = f64::from_bits(1);
    near(c(u, u) / c(u, 2.0 * u), c(0.6, -0.2), 1e-15);
    let q = c(1e300, 0.0) / c(1e100, 1e-250);
    assert!(
        (q.re - 1e200).abs() <= 1e185 && (q.im + 1e-150).abs() <= 1e-164,
        "{q:?}"
    );
    near(c(3e-310, 4e-310) / c(1e-310, 0.0), c(3.0, 4.0), 1e-12);
}

#[test]
fn square_roots() {
    assert_eq!(c(3.0, 4.0).sqrt(), c(2.0, 1.0));
    assert_eq!(c(-4.0, 0.0).sqrt(), c(0.0, 2.0));
    // The sign of a zero imaginary part picks the side of the cut.
    assert_eq!(c(-4.0, -0.0).sqrt(), c(0.0, -2.0));
    assert_eq!(c(0.0, 0.0).sqrt(), c(0.0, 0.0));
    assert!(c(0.0, -0.0).sqrt().im.is_sign_negative());
    assert_eq!(c(0.0, 2.0).sqrt(), c(1.0, 1.0));
    // C99's special values.
    assert_eq!(
        c(1.0, f64::INFINITY).sqrt(),
        c(f64::INFINITY, f64::INFINITY)
    );
    assert_eq!(c(f64::NEG_INFINITY, 1.0).sqrt(), c(0.0, f64::INFINITY));
    assert_eq!(c(f64::INFINITY, -1.0).sqrt(), c(f64::INFINITY, -0.0));
    assert!(c(f64::NAN, 1.0).sqrt().is_nan());
    for (re, im) in [(0.3, -7.0), (-2.5, 1e-8), (1e10, 3.0), (-1e-10, -1e-10)] {
        let z = c(re, im);
        let r = z.sqrt();
        assert!(r.re >= 0.0);
        near(r * r, z, 4e-16);
    }
}

#[test]
fn exponentials_and_logarithms() {
    near(c(0.0, PI).exp(), c(-1.0, 0.0), 1e-15);
    assert_eq!(c(2.0, 0.0).exp(), c(2f64.exp(), 0.0));
    assert!(c(1.0, -0.0).exp().im.is_sign_negative());
    assert_eq!(c(-1.0, 0.0).ln(), c(0.0, PI));
    assert_eq!(c(-1.0, -0.0).ln(), c(0.0, -PI));
    assert_eq!(c(0.0, 1.0).ln(), c(0.0, FRAC_PI_2));
    assert_eq!(c(0.0, 0.0).ln().re, f64::NEG_INFINITY);
    // Near the unit circle the real part keeps its digits: ln|1 + 1e-10·i| = 5e-21.
    let l = c(1.0, 1e-10).ln();
    assert!((l.re - 5e-21).abs() <= 1e-35, "{}", l.re);
    for z in [c(0.5, -3.0), c(-7.0, 0.25), c(1e-5, 1e5), c(2.0, 0.0)] {
        near(z.ln().exp(), z, 4e-16);
        near(z.exp().ln(), z.exp().ln().exp().ln(), 4e-16);
    }
    assert_eq!(c(1.0, 0.0).arg(), 0.0);
    assert_eq!(c(-1.0, -0.0).arg(), -PI);
    let (r, theta) = c(-3.0, 4.0).to_polar();
    near(Complex::from_polar(r, theta), c(-3.0, 4.0), 1e-15);
}

#[test]
fn powers() {
    assert_eq!(c(1.0, 1.0).powi(8), c(16.0, 0.0));
    assert_eq!(c(0.0, 2.0).powi(-2), c(-0.25, 0.0));
    assert_eq!(c(5.0, 3.0).powi(0), Complex::ONE);
    assert_eq!(c(0.0, 0.0).powi(0), Complex::ONE);
    near(c(-8.0, 0.0).powf(1.0 / 3.0), c(1.0, 3f64.sqrt()), 1e-15);
    assert_eq!(c(0.0, 0.0).powf(2.0), Complex::ZERO);
    assert_eq!(c(0.0, 0.0).powf(0.0), Complex::ONE);
    // `i^i = e^(-π/2)`.
    near(
        Complex::I.powc(Complex::I),
        c((-FRAC_PI_2).exp(), 0.0),
        1e-15,
    );
    assert_eq!(c(0.0, 0.0).powc(c(1.0, 1.0)), Complex::ZERO);
    assert_eq!(c(0.0, 0.0).powc(Complex::ZERO), Complex::ONE);
}

#[test]
fn single_precision() {
    let z = Complex::new(3.0f32, 4.0);
    assert_eq!(z.abs(), 5.0);
    assert_eq!(z.sqrt(), Complex::new(2.0, 1.0));
    assert_eq!(2.0f32 * z, Complex::new(6.0, 8.0));
    let w = Complex::new(0.5f32, -1.5).ln().exp();
    assert!((w - Complex::new(0.5, -1.5)).abs() < 1e-6);
    assert_eq!(<f32 as Float>::EPSILON, f32::EPSILON);
}

#[test]
fn formatting() {
    assert_eq!(format!("{}", c(1.5, -2.0)), "1.5-2i");
    assert_eq!(format!("{}", c(0.0, 0.0)), "0+0i");
    assert_eq!(format!("{}", c(0.0, -0.0)), "0-0i");
    assert_eq!(format!("{:.2}", c(1.0, 1.0 / 3.0)), "1.00+0.33i");
    assert_eq!(format!("{:?}", c(1.0, 2.0)), "Complex { re: 1.0, im: 2.0 }");
}
