//! The dispatch of the core: every type of `Real` and `Scalar` reaches its function, and the
//! harness reads the data. The bounds are the core's Lanczos baseline of the gamma function; the
//! gamma area's own tests are the accurate ones.

use super::common::*;
use a::math::complex::Complex;
use a::math::special::{gamma, lgamma, lgamma_sign};

const GAMMA_F64: &str = include_str!("data/core_gamma_f64.txt");
const GAMMA_F32: &str = include_str!("data/core_gamma_f32.txt");
const LGAMMA_F64: &str = include_str!("data/core_lgamma_f64.txt");
const GAMMA_C64: &str = include_str!("data/core_gamma_c64.txt");

#[test]
fn real_types_reach_their_functions() {
    let e64 = check_real("gamma f64", GAMMA_F64, F64, 1e3, |a| gamma(a[0]));
    let e32 = check_real("gamma f32", GAMMA_F32, F32, 1.0, |a| {
        gamma(a[0] as f32) as f64
    });
    let el = check_real("lgamma f64", LGAMMA_F64, F64, 16.0, |a| lgamma(a[0]));
    eprintln!("baseline: gamma f64 {e64:.2} ulps, f32 {e32:.2}, lgamma f64 {el:.2}");
}

#[test]
fn complex_types_reach_their_functions() {
    let e = check_complex("gamma c64", GAMMA_C64, F64, 1e3, |re, im| {
        let z = gamma(Complex::new(re, im));
        (z.re, z.im)
    });
    let z = gamma(Complex::new(2.5f32, 1.0));
    let w = gamma(Complex::new(2.5f64, 1.0));
    assert!((z.re as f64 - w.re).abs() < 1e-6 && (z.im as f64 - w.im).abs() < 1e-6);
    eprintln!("baseline: gamma c64 {e:.2} eps");
}

#[test]
fn poles_signs_and_nan() {
    assert_eq!(gamma(0.0f64), f64::INFINITY);
    assert_eq!(gamma(-0.0f64), f64::NEG_INFINITY);
    assert!(gamma(-3.0f64).is_nan() && gamma(f64::NAN).is_nan());
    assert_eq!(gamma(200.0f64), f64::INFINITY);
    assert_eq!(lgamma(-3.0f64), f64::INFINITY);
    assert_eq!(lgamma_sign(-0.5f64).1, -1.0);
    assert_eq!(lgamma_sign(-1.5f64).1, 1.0);
    assert_eq!(lgamma_sign(2.5f32).1, 1.0);
    assert!(lgamma_sign(f64::NAN).1.is_nan());
}

#[test]
fn the_harness_measures_ulps() {
    assert_eq!(ulp(1.0, F64), f64::EPSILON);
    assert_eq!(ulp(1.5, F32), f32::EPSILON as f64);
    assert_eq!(ulp(0.0, F64), f64::from_bits(1));
    assert_eq!(ulp(f64::from_bits(5), F64), f64::from_bits(1));
    assert_eq!(ulps(1.0 + f64::EPSILON, (1.0, 0.0), F64), 1.0);
    assert_eq!(ulps(1.0, (1.0, f64::EPSILON / 4.0), F64), 0.25);
    assert_eq!(ulps(f64::INFINITY, (1e300 * 1e10, 0.0), F64), 0.0);
    assert_eq!(ulps(f32::INFINITY as f64, (1e39, 0.0), F32), 0.0);
    assert_eq!(ulps(f64::NAN, (f64::NAN, 0.0), F64), 0.0);
    assert!(ulps(1.0, (f64::NAN, 0.0), F64).is_infinite());
}
