//! The exponential, the logarithm, the square root and the powers: against mpmath, against
//! 160-bit arithmetic on random quaternions of every size, against `Complex` on the complex
//! numbers, and at the special values.

use super::common::{Rng, identical, lines, pow2, quat, rel, to32, to64};
use super::reference;
use a::math::complex::Complex;
use a::math::quaternion::Quaternion;
use core::f64::consts::PI;

type Q = Quaternion<f64>;
type Q32 = Quaternion<f32>;

fn q(w: f64, x: f64, y: f64, z: f64) -> Q {
    Quaternion::new(w, x, y, z)
}

const EPS: f64 = f64::EPSILON;
const INF: f64 = f64::INFINITY;
const NAN: f64 = f64::NAN;

const MPMATH: &str = include_str!("data/mpmath_functions.txt");

/// The normwise error of `got` against `want` in units of `ε·|want|`, divided by the condition
/// `kappa` of the computation; `None` where the condition leaves fewer than ten bits determined
/// (a vector part of length near `1/ε` turns the result by an angle that the precision does not
/// fix), or where `want` has a part that is not finite (where `got` must then agree in it).
fn err(got: Q, want: Q, kappa: f64) -> Option<f64> {
    if kappa * EPS > 1e-3 {
        return None;
    }
    if !want.is_finite() {
        for (g, w) in [
            (got.w, want.w),
            (got.x, want.x),
            (got.y, want.y),
            (got.z, want.z),
        ] {
            if w.is_infinite() {
                assert_eq!(g, w, "{got} is not {want}");
            }
        }
        return None;
    }
    let n = want.norm();
    if n == 0.0 {
        return Some(got.norm() / f64::MIN_POSITIVE);
    }
    Some((got - want).norm() / (n * EPS) / kappa)
}

#[test]
fn the_functions_agree_with_mpmath_at_60_digits() {
    let p = q(0.5, 0.25, -0.125, 1.0);
    let mut worst = [0.0f64; 6];
    let mut at = [Q::ZERO; 6];
    let names = ["exp", "ln", "sqrt", "q^(1/3)", "q^(-2.5)", "q^p"];
    for l in lines(MPMATH, "f") {
        let x = quat(&l[0..4]);
        let want: Vec<Q> = (0..6).map(|i| quat(&l[4 + 4 * i..8 + 4 * i])).collect();
        let got = [
            x.exp(),
            x.ln(),
            x.sqrt(),
            x.powf(1.0 / 3.0),
            x.powf(-2.5),
            x.powq(p),
        ];
        // The condition of exp(t·ln q) in its argument: errors of ε in ln q, of size up to |t·ln q|,
        // become errors of up to |t·ln q|·ε in the power.
        let ln = reference::ln(x);
        let kappa = [
            1.0 + x.norm(),
            1.0,
            1.0,
            1.0 + ln.norm() / 3.0,
            1.0 + 2.5 * ln.norm(),
            1.0 + 2.0 * ln.norm(),
        ];
        for i in 0..6 {
            if let Some(e) = err(got[i], want[i], kappa[i]) {
                assert!(
                    e <= 4.0,
                    "{} of {x}: {} is not {} ({e} ε)",
                    names[i],
                    got[i],
                    want[i]
                );
                if e > worst[i] {
                    (worst[i], at[i]) = (e, x);
                }
            }
        }
    }
    eprintln!(
        "mpmath: largest error in ε, scaled by the condition, of {names:?}: {worst:?}, at {at:?}"
    );
}

#[test]
fn the_functions_agree_with_160_bit_arithmetic_on_random_quaternions() {
    let mut rng = Rng::new(20);
    let mut worst = [0.0f64; 5];
    let n = if cfg!(miri) { 10 } else { 1500 };
    for i in 0..n {
        // Parts of moderate size, then vector parts of any size beside a real part of any size.
        let x = if i % 2 == 0 {
            rng.quaternion() * 4.0
        } else {
            let s = rng.log_uniform(-60.0, 60.0).abs();
            let w = rng.log_uniform(-60.0, 60.0);
            Q::from_parts(w, rng.quaternion().vector().map(|t| t * s))
        };
        let ln = reference::ln(x);
        let t = 3.0 * rng.signed();
        let kappa = [1.0 + x.norm(), 1.0, 1.0, 1.0 + t.abs() * ln.norm(), 1.0];
        let pairs = [
            (x.exp(), reference::exp(x)),
            (x.ln(), ln),
            (x.sqrt(), reference::sqrt(x)),
            (x.powf(t), reference::powf(x, t)),
            (
                (x * (1.0 / x.norm())).ln(),
                reference::ln(x * (1.0 / x.norm())),
            ),
        ];
        for (k, (got, want)) in pairs.into_iter().enumerate() {
            if let Some(e) = err(got, want, kappa[k]) {
                worst[k] = worst[k].max(e);
            }
        }
    }
    eprintln!(
        "160 bits over {n}: largest error in ε of exp, ln, sqrt, powf, ln of unit: {worst:?}"
    );
    for (w, bound) in worst.iter().zip([4.0, 3.0, 3.0, 4.0, 3.0]) {
        assert!(*w <= bound, "{worst:?}");
    }
}

#[test]
fn exp_and_ln_are_inverse_on_the_principal_branch() {
    let mut rng = Rng::new(21);
    let mut worst = [0.0f64; 3];
    for _ in 0..if cfg!(miri) { 20 } else { 2000 } {
        let x = rng.quaternion() * 3.0;
        worst[0] = worst[0].max(rel(x.ln().exp(), x) / EPS);
        // ln(exp(q)) = q when the vector part is shorter than π.
        let v = x.vector();
        let r = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        if r < PI - 0.01 {
            let back = x.exp().ln();
            worst[1] = worst[1].max((back - x).norm() / (EPS * (1.0 + x.norm())));
        }
        // Past π the logarithm takes the vector part to length 2π − r, the other way round.
        if r > PI + 0.01 && r < 2.0 * PI - 0.01 {
            let back = x.exp().ln();
            let s = (2.0 * PI - r) / r;
            let want = q(x.w, -v[0] * s, -v[1] * s, -v[2] * s);
            worst[2] = worst[2].max((back - want).norm() / (EPS * (1.0 + x.norm())));
        }
    }
    eprintln!("exp(ln q) = q, ln(exp q) = q, and past π: largest error in ε: {worst:?}");
    assert!(
        worst[0] <= 6.0 && worst[1] <= 10.0 && worst[2] <= 10.0,
        "{worst:?}"
    );
}

#[test]
fn the_powers_agree_with_products_and_with_each_other() {
    let mut rng = Rng::new(22);
    for _ in 0..if cfg!(miri) { 10 } else { 500 } {
        let x = rng.quaternion();
        let mut p = Q::ONE;
        for n in 0..8 {
            let bound = 4.0 * (n as f64 + 1.0) * EPS;
            assert!(rel(x.powi(n), p) <= bound, "{x}^{n}");
            assert!(rel(x.powi(-n), p.recip()) <= 2.0 * bound, "{x}^-{n}");
            p *= x;
        }
        // powf at integers is powi, to the condition of exp(t ln q).
        let k = 1.0 + 3.0 * x.ln().norm();
        assert!(rel(x.powf(3.0), x.powi(3)) <= 8.0 * k * EPS);
        assert!(rel(x.powf(-2.0), x.powi(-2)) <= 8.0 * k * EPS);
        // The square root is the power one half, and squares back.
        let s = x.sqrt();
        assert!(s.w >= 0.0);
        assert!(rel(s * s, x) <= 4.0 * EPS);
        assert!(rel(s, x.powf(0.5)) <= 8.0 * (1.0 + 0.5 * x.ln().norm()) * EPS);
        // A real quaternion power is the real power; the logarithm on the right.
        let t = rng.signed() * 2.0;
        assert!(rel(x.powq(Q::from(t)), x.powf(t)) <= 4.0 * EPS * (1.0 + t.abs() * x.ln().norm()));
    }
    // The order matters for a power that does not commute with the base.
    let (x, p) = (q(1.0, 2.0, 0.0, 0.0), q(0.0, 0.0, 1.0, 0.0));
    let (right, left) = (x.powq(p), (p * x.ln()).exp());
    assert!((right - left).norm() > 0.1);
    assert!(rel(right, reference::powq(x, p)) <= 4.0 * EPS);
    // q^0 = 1 for every q; 0^t.
    assert_eq!(q(NAN, 1.0, 2.0, 3.0).powf(0.0), Q::ONE);
    assert_eq!(Q::ZERO.powi(0), Q::ONE);
    assert_eq!(Q::ZERO.powq(Q::ZERO), Q::ONE);
    assert_eq!(Q::ZERO.powf(2.5), Q::ZERO);
    assert_eq!(Q::ZERO.powf(-2.5), q(INF, 0.0, 0.0, 0.0));
    assert_eq!(Q::ZERO.powq(q(0.5, 1.0, 0.0, 0.0)), Q::ZERO);
    assert_eq!(Q::ZERO.powq(q(-0.5, 0.0, 0.0, 0.0)), q(INF, 0.0, 0.0, 0.0));
    assert!(Q::ZERO.powq(q(-0.5, 1.0, 0.0, 0.0)).w.is_nan());
    assert!(q(1.0, 2.0, 3.0, 4.0).powf(NAN).x.is_nan());
    assert!(q(1.0, NAN, 3.0, 4.0).powq(Q::ONE).w.is_nan());
    // (−1)^½ = i, as for the complex number.
    assert!(rel(q(-1.0, 0.0, 0.0, 0.0).powf(0.5), Q::I) <= EPS);
    // powi is exact while the products are.
    assert_eq!(q(1.0, 1.0, 1.0, 1.0).powi(3), q(-8.0, 0.0, 0.0, 0.0));
    assert_eq!(q(0.0, 2.0, 0.0, 0.0).powi(-2), q(-0.25, 0.0, 0.0, 0.0));
    assert_eq!(Q::I.powi(i32::MIN), Q::ONE);
}

#[test]
fn the_functions_are_those_of_complex_on_the_complex_numbers() {
    let mut rng = Rng::new(23);
    let close = |a: Q, b: Complex<f64>, tol: f64| {
        let b = Q::from(b);
        let n = b.norm().max(1e-300);
        assert!((a - b).norm() <= tol * n, "{a} is not {b}");
        assert!(a.y == 0.0 && a.z == 0.0, "{a}");
    };
    for i in 0..if cfg!(miri) { 20 } else { 1000 } {
        let z = if i % 4 == 0 {
            Complex::new(-rng.uniform() * 4.0, 0.0)
        } else {
            Complex::new(rng.signed() * 4.0, rng.signed() * 4.0)
        };
        let x = Q::from(z);
        close(x.exp(), z.exp(), 4.0 * EPS);
        close(x.ln(), z.ln(), 4.0 * EPS);
        close(x.sqrt(), z.sqrt(), 4.0 * EPS);
        close(x.powi(5), z.powi(5), 16.0 * EPS);
        let t = rng.signed() * 3.0;
        close(
            x.powf(t),
            z.powf(t),
            16.0 * EPS * (1.0 + t.abs() * z.ln().abs()),
        );
        close(x.recip(), z.recip(), 4.0 * EPS);
    }
    // The branch cut on the negative real axis: the sign of the zero x picks the side.
    let m = |x: f64| q(-4.0, x, 0.0, 0.0);
    assert!(identical(m(0.0).sqrt(), q(0.0, 2.0, 0.0, 0.0)));
    assert!(identical(m(-0.0).sqrt(), q(0.0, -2.0, 0.0, 0.0)));
    assert_eq!(m(0.0).ln(), q(4f64.ln(), PI, 0.0, 0.0));
    assert_eq!(m(-0.0).ln(), q(4f64.ln(), -PI, 0.0, 0.0));
    assert_eq!(
        Q::from(Complex::new(-4.0, 0.0)).ln(),
        Q::from(Complex::new(-4.0, 0.0).ln())
    );
    // The zeros of j and k are kept.
    assert!(identical(
        q(-1.0, 0.0, -0.0, -0.0).ln(),
        q(0.0, PI, -0.0, -0.0)
    ));
    assert!(identical(
        q(2.0, -0.0, 0.0, -0.0).exp(),
        q(2f64.exp(), -0.0, 0.0, -0.0)
    ));
}

#[test]
fn tiny_vector_parts_and_the_identity_keep_their_digits() {
    // exp: sin(r)/r is 1 for r below the sine's argument.
    for s in [1e-9, 1e-20, 1e-200, 1e-310, pow2(-1074)] {
        let x = q(0.0, s, -2.0 * s, 0.5 * s);
        let e = x.exp();
        assert_eq!(e.w, 1.0);
        assert_eq!((e.x, e.y, e.z), (x.x, x.y, x.z), "{s}");
        // ln: the vector part of ln(1 + v) is v for tiny v.
        let l = (Q::ONE + x).ln();
        assert_eq!((l.x, l.y, l.z), (x.x, x.y, x.z), "{s}");
        assert!(l.w.abs() <= 3.0 * s * s + f64::MIN_POSITIVE * EPS);
    }
    // Near the identity: ln|q| from (big − 1)(big + 1).
    let x = q(1.0, 1e-10, 0.0, 0.0);
    assert!((x.ln().w - 5e-21).abs() <= 1e-36);
    let x = q(1.0 + 2f64.powi(-40), 0.0, 0.0, 0.0);
    assert!((x.ln().w - 2f64.powi(-40)).abs() <= 2f64.powi(-80));
    // A vector part tiny beside a negative real part: the angle is π.
    let l = q(-1.0, 1e-300, 0.0, 0.0).ln();
    assert!((l.x - PI).abs() <= 4.0 * EPS && l.y == 0.0);
    let s = q(-4.0, 1e-300, 0.0, 0.0).sqrt();
    assert!((s.w - 2.5e-301).abs() <= 2.5e-301 * 4.0 * EPS && s.x == 2.0);
    let s = q(-4.0, 3.0 * pow2(-1074), 4.0 * pow2(-1074), 0.0).sqrt();
    assert_eq!((s.x, s.y), (1.2, 1.6));
    // A real part far larger than the vector part: ln's vector part is v/w.
    let l = q(1e300, 1e10, 0.0, 0.0).ln();
    assert_eq!(l.x, 1e10 / 1e300);
    let l = q(1.0, 3e-320, 0.0, 0.0).ln();
    assert_eq!(l.x, 3e-320);
    // Either side of the switch to v/w at r/w = √ε/2: within a unit of the last place.
    for f in [0.4, 0.49, 0.51, 0.6, 2.0, 10.0] {
        let x = q(3.0, 3.0 * f * EPS.sqrt(), 0.0, 0.0);
        let (got, want) = (x.ln(), reference::ln(x));
        assert!(
            (got.x - want.x).abs() <= want.x * EPS,
            "{f}: {} {}",
            got.x,
            want.x
        );
    }
}

#[test]
fn huge_and_tiny_norms_neither_overflow_nor_lose_their_digits() {
    let check = |x: Q| {
        for (name, got, want) in [
            ("ln", x.ln(), reference::ln(x)),
            ("sqrt", x.sqrt(), reference::sqrt(x)),
        ] {
            let e = err(got, want, 1.0).unwrap();
            assert!(e <= 3.0, "{name} of {x}: {got} is not {want} ({e} ε)");
        }
    };
    for x in [
        q(f64::MAX, f64::MAX, f64::MAX, f64::MAX),
        q(f64::MAX, 0.0, 0.0, 0.0),
        q(-f64::MAX, 0.0, 1.0, 0.0),
        q(1e300, -1e300, 1e300, 2e300),
        q(1e-300, -1e-300, 1e-300, 2e-300),
        q(3e-320, 0.0, -4e-320, 0.0),
        q(-pow2(-1074), 0.0, 0.0, pow2(-1074)),
        q(0.0, f64::MAX, 0.0, 0.0),
    ] {
        check(x);
    }
    // exp where e^w overflows and the parts do not.
    let x = q(710.0, 1.0, 0.0, 0.0);
    let (got, want) = (x.exp(), reference::exp(x));
    assert!((got.w - want.w).abs() <= 2.0 * EPS * want.w.abs());
    assert_eq!(got.x, INF);
    assert_eq!((got.y, got.z), (0.0, 0.0));
    let x = q(709.9, 3.0, -4.0, 0.0);
    assert!(err(x.exp(), reference::exp(x), 1.0).unwrap() <= 4.0);
    assert_eq!(q(800.0, 1e-300, 0.0, 0.0).exp().y, 0.0);
    // exp of a large negative real part: zero, keeping the zeros' signs.
    assert!(identical(
        q(-800.0, 1.0, -0.0, 0.0).exp(),
        q(0.0, 0.0, -0.0, 0.0)
    ));
}

#[test]
fn the_special_values_are_those_documented() {
    // ln of zero, of infinities and of NaN.
    assert!(identical(Q::ZERO.ln(), q(-INF, 0.0, 0.0, 0.0)));
    assert!(identical(
        q(-0.0, 0.0, 0.0, 0.0).ln(),
        q(-INF, PI, 0.0, 0.0)
    ));
    assert!(identical(q(INF, 1.0, 0.0, 0.0).ln(), q(INF, 0.0, 0.0, 0.0)));
    assert!(identical(q(-INF, 0.0, 0.0, 0.0).ln(), q(INF, PI, 0.0, 0.0)));
    let l = q(1.0, INF, 0.0, 0.0).ln();
    assert!(l.w == INF && (l.x - PI / 2.0).abs() <= EPS && l.y == 0.0);
    let l = q(INF, INF, 0.0, 0.0).ln();
    assert!(l.w == INF && (l.x - PI / 4.0).abs() <= EPS);
    let l = q(INF, INF, INF, INF).ln();
    let a = 3f64.sqrt().atan2(1.0) / 3f64.sqrt();
    assert!(l.w == INF && (l.x - a).abs() <= EPS && (l.z - a).abs() <= EPS);
    assert!(q(NAN, 0.0, 0.0, 0.0).ln().x.is_nan());
    assert!(q(INF, NAN, 0.0, 0.0).ln().w.is_nan());
    // exp of infinities and NaN.
    assert!(identical(
        q(INF, 0.0, 0.0, 0.0).exp(),
        q(INF, 0.0, 0.0, 0.0)
    ));
    assert!(identical(
        q(-INF, 0.0, 0.0, 0.0).exp(),
        q(0.0, 0.0, 0.0, 0.0)
    ));
    let e = q(INF, 1.0, 0.0, 0.0).exp();
    assert!(e.w == INF && e.x == INF && e.y == 0.0 && e.z == 0.0);
    let e = q(-INF, 1.0, 2.0, 3.0).exp();
    assert_eq!(e, Q::ZERO);
    assert!(q(0.0, INF, 0.0, 0.0).exp().w.is_nan());
    assert!(q(NAN, 0.0, 0.0, 0.0).exp().w.is_nan());
    assert!(identical(
        q(NAN, 0.0, -0.0, 0.0).exp(),
        q(NAN, 0.0, -0.0, 0.0)
    ));
    // sqrt of zeros, infinities and NaN.
    assert!(identical(
        q(-0.0, -0.0, 0.0, -0.0).sqrt(),
        q(0.0, -0.0, 0.0, -0.0)
    ));
    assert!(identical(
        q(INF, 1.0, -2.0, 0.0).sqrt(),
        q(INF, 0.0, -0.0, 0.0)
    ));
    assert!(identical(
        q(1.0, -INF, 2.0, 0.0).sqrt(),
        q(INF, -INF, 0.0, 0.0)
    ));
    assert!(identical(
        q(-INF, 0.0, -0.0, 0.0).sqrt(),
        q(0.0, INF, -0.0, 0.0)
    ));
    assert!(identical(
        q(-INF, -0.0, 0.0, 0.0).sqrt(),
        q(0.0, -INF, 0.0, 0.0)
    ));
    assert!(identical(
        q(-INF, 1.0, 0.0, -2.0).sqrt(),
        q(0.0, INF, 0.0, -INF)
    ));
    assert!(q(1.0, NAN, 0.0, 0.0).sqrt().w.is_nan());
    assert!(q(-INF, NAN, 0.0, 0.0).sqrt().y.is_nan());
}

#[test]
fn the_functions_of_f32_agree_with_those_of_f64() {
    // On f32 inputs, the f64 result (within a few units of 2⁻⁵³) is a reference for the f32 one.
    let mut rng = Rng::new(24);
    let mut worst = [0.0f64; 5];
    for i in 0..if cfg!(miri) { 20 } else { 2000 } {
        let x = if i % 2 == 0 {
            to32(rng.quaternion() * 3.0)
        } else {
            let s = rng.log_uniform(-60.0, 60.0).abs();
            to32(Q::from_parts(
                rng.signed(),
                rng.quaternion().vector().map(|t| t * s),
            ))
        };
        let x64 = to64(x);
        let t = rng.signed() * 2.0;
        let ln = x64.ln().norm();
        let kappa = [1.0 + x64.norm(), 1.0, 1.0, 1.0 + t.abs() * ln, 1.0];
        let pairs = [
            (to64(x.exp()), x64.exp()),
            (to64(x.ln()), x64.ln()),
            (to64(x.sqrt()), x64.sqrt()),
            (to64(x.powf(t as f32)), x64.powf((t as f32) as f64)),
            (to64(x.recip()), x64.recip()),
        ];
        for (k, (got, want)) in pairs.into_iter().enumerate() {
            if want.is_finite() && want.norm() > 1e-30 && want.norm() < 1e30 {
                let e = (got - want).norm() / (want.norm() * f32::EPSILON as f64);
                worst[k] = worst[k].max(e / kappa[k]);
            }
        }
    }
    eprintln!("f32 against f64, largest error in ε(f32) of exp, ln, sqrt, powf, recip: {worst:?}");
    for w in worst {
        assert!(w <= 3.0, "{worst:?}");
    }
    // The ends of the f32 range.
    let m = Q32::new(f32::MAX, f32::MAX, 0.0, 0.0);
    assert!((m.ln().w as f64 - (f32::MAX as f64 * 2f64.sqrt()).ln()).abs() < 2e-5);
    let t = Q32::new(0.0, 3e-45, 4e-45, 0.0);
    assert!(t.sqrt().is_finite() && t.sqrt().w > 0.0);
    assert_eq!(
        Q32::new(-4.0, 0.0, 0.0, 0.0).sqrt(),
        Q32::new(0.0, 2.0, 0.0, 0.0)
    );
}

#[test]
fn powi_starts_from_the_base_and_overflows_to_infinities() {
    // q¹ is q and q² is q·q, infinities included, and where the square overflows.
    for x in [
        q(INF, 0.0, 0.0, 0.0),
        q(1.0, 1e300, 0.0, 0.0),
        q(1e200, 1e200, 1e200, 1e200),
        q(-0.0, NAN, 1.0, 0.0),
        q(2.938103389672808e-5, -f64::MAX, -f64::MAX, f64::MAX),
        q(pow2(1023), pow2(-60), 0.0, 0.0),
    ] {
        assert!(identical(x.powi(1), x), "{x}");
        assert!(identical(x.powi(2), x * x), "{x}");
    }
    // Real, complex and pure-vector bases keep their zero parts.
    assert_eq!(q(2.0, 0.0, 0.0, 0.0).powi(1100), q(INF, 0.0, 0.0, 0.0));
    assert_eq!(q(-2.0, 0.0, 0.0, 0.0).powi(1101), q(-INF, 0.0, 0.0, 0.0));
    assert_eq!(q(2.0, 0.0, 0.0, 0.0).powi(-1100), Q::ZERO);
    assert_eq!(q(0.0, 0.0, 2.0, 0.0).powi(1100), q(INF, 0.0, 0.0, 0.0));
    assert_eq!(q(0.0, 0.0, 2.0, 0.0).powi(1101), q(0.0, 0.0, INF, 0.0));
    assert_eq!(q(0.0, 0.0, 2.0, 0.0).powi(1102), q(-INF, 0.0, 0.0, 0.0));
    assert_eq!(
        q(pow2(-1074), 0.0, 0.0, 0.0).powi(-1),
        q(INF, 0.0, 0.0, 0.0)
    );
    assert_eq!(q(pow2(-600), 0.0, 0.0, 0.0).powi(-2), q(INF, 0.0, 0.0, 0.0));
    let z = Q::from(Complex::new(1.0, 1.0)).powi(2100);
    assert!(z.y == 0.0 && z.z == 0.0 && !z.is_nan(), "{z}");
    // A part 2⁻¹⁰⁷³ of the base's 2 grows to 3000·2⁻¹⁰⁷³·2²⁹⁹⁹, past the largest float though the
    // scaled power holds it as a subnormal: the scaling back takes the three steps it needs.
    let t = q(2.0, pow2(-1073), 0.0, 0.0).powi(3000);
    assert_eq!(t, q(INF, INF, 0.0, 0.0));
    assert!(identical(
        q(pow2(1023), pow2(-60), 0.0, 0.0).powi(2),
        q(INF, pow2(964), 0.0, 0.0)
    ));
    // A part far below the norm of an overflowing power is only as accurate as the norm allows:
    // here x is 10⁻³³ of the norm and lost to cancellation in the square, as in any product.
    let c = q(
        2.8862847467051496e86,
        -1.2230225720273404e77,
        1.7490544209238627e90,
        -3.361504409565855e98,
    );
    let p4 = c.powi(4);
    assert!(!p4.is_nan() && p4.z.is_infinite() && p4.y.is_infinite());
    assert!(reference::check_power(c, 4, p4, 8.0 * EPS).1, "{p4}");
    // Random bases and powers that overflow, against 160-bit powers.
    let mut rng = Rng::new(71);
    let (mut overflowing, n) = (0, if cfg!(miri) { 4 } else { 300 });
    for i in 0..n {
        let s = 1.5 + 1.5 * rng.uniform();
        let x = match i % 3 {
            0 => rng.quaternion(),
            1 => q(rng.signed(), rng.signed(), 0.0, 0.0),
            _ => q(0.0, 0.0, rng.signed(), 0.0),
        }
        .normalize()
            * s;
        let k = 600 + (rng.next_u64() % 2400) as u32;
        let got = x.powi(k as i32);
        let (over, ok) = reference::check_power(x, k, got, 2.0 * k as f64 * EPS);
        assert!(ok, "{x}^{k} = {got}");
        overflowing += usize::from(over);
    }
    eprintln!("overflowing powers: {overflowing} of {n}, each within 2nε of |qⁿ| and no NaN");
    assert!(cfg!(miri) || overflowing > n / 2);
    // Mixed part sizes: norms 1.5 to 3 with parts down to 2⁻⁴⁰ of it and n to 3,000; huge bases
    // with small n; tiny bases with negative n, whose reciprocals overflow; one dominant part.
    let mut rng = Rng::new(75);
    let (mut overflowing, n) = (0, if cfg!(miri) { 8 } else { 800 });
    for i in 0..n {
        let mixed = |rng: &mut super::common::Rng, lo: f64| {
            Q::new(
                rng.log_uniform(lo, 0.0),
                rng.log_uniform(lo, 0.0),
                rng.log_uniform(lo, 0.0),
                rng.log_uniform(lo, 0.0),
            )
            .normalize()
        };
        let (x, k): (Q, i32) = match i % 4 {
            0 => (
                mixed(&mut rng, -40.0) * (1.5 + 1.5 * rng.uniform()),
                600 + (rng.next_u64() % 2400) as i32,
            ),
            1 => (
                mixed(&mut rng, -80.0) * rng.log_uniform(300.0, 1020.0).abs(),
                2 + (rng.next_u64() % 6) as i32,
            ),
            2 => (
                mixed(&mut rng, -80.0) * rng.log_uniform(-1070.0, -300.0).abs(),
                -1 - (rng.next_u64() % 7) as i32,
            ),
            _ => (
                q(
                    rng.log_uniform(1.0, 600.0),
                    rng.log_uniform(-200.0, 0.0),
                    rng.log_uniform(-200.0, 0.0),
                    rng.log_uniform(-200.0, 0.0),
                ),
                2 + (rng.next_u64() % 1200) as i32,
            ),
        };
        let got = x.powi(k);
        let (base, m) = if k < 0 {
            (reference::recip(x), k.unsigned_abs())
        } else {
            (x, k as u32)
        };
        if !base.is_finite() {
            // The reciprocal itself overflows: the power from the scaled base, checked whole.
            assert!(!got.is_nan(), "{x}^{k}");
            continue;
        }
        let (over, ok) = reference::check_power(base, m, got, 2.0 * f64::from(m + 1) * EPS);
        assert!(ok, "{x}^{k} = {got}");
        overflowing += usize::from(over);
    }
    eprintln!(
        "mixed powers: {overflowing} of {n} overflow, each within 2(n+1)ε of |qⁿ| and no NaN"
    );
}

#[test]
fn ln_at_minus_infinity_keeps_the_direction_of_the_vector_part() {
    let l = q(-INF, 0.0, 1.0, 0.0).ln();
    assert!(identical(l, q(INF, 0.0, PI, 0.0)), "{l}");
    let l = q(-INF, 1.0, 1.0, 0.0).ln();
    let h = PI / 2f64.sqrt();
    assert!(
        l.w == INF && (l.x - h).abs() <= 2.0 * EPS && (l.y - h).abs() <= 2.0 * EPS && l.z == 0.0
    );
    let l = q(-INF, 0.0, 0.0, -1e-300).ln();
    assert!(identical(l, q(INF, 0.0, 0.0, -PI)), "{l}");
    // The limit of its finite neighbours.
    let near = q(-f64::MAX, 0.0, 1.0, 0.0).ln();
    assert!((near.y - PI).abs() <= 2.0 * EPS && near.x == 0.0);
    // A zero vector part takes the complex value; +∞ gives zeros of the vector part's signs.
    assert!(identical(
        q(-INF, -0.0, 0.0, 0.0).ln(),
        q(INF, -PI, 0.0, 0.0)
    ));
    assert!(identical(
        q(INF, 0.0, -1.0, 2.0).ln(),
        q(INF, 0.0, -0.0, 0.0)
    ));
    // sqrt agrees on the direction.
    assert!(identical(
        q(-INF, 0.0, 1.0, 0.0).sqrt(),
        q(0.0, 0.0, INF, 0.0)
    ));
}

#[test]
fn the_shortcut_of_ln_switches_at_half_the_root_of_epsilon() {
    // Below r/w = √ε/2 the vector part is v/w exactly; from about 1.2·√ε, where atan(t)/t is
    // below 1 by more than a unit, it is the general formula's, a unit or more below v/w.
    let s = EPS.sqrt();
    for f in [0.1, 0.3, 0.45, 0.49] {
        let t = f * s;
        assert_eq!(q(1.0, t, 0.0, 0.0).ln().x, t, "{f}");
        assert_eq!(q(4.0, 0.0, -4.0 * t, 0.0).ln().y, -t, "{f}");
    }
    for f in [1.3, 1.6, 1.9, 3.0] {
        let x = q(1.0, f * s, 0.0, 0.0);
        let (got, want) = (x.ln().x, reference::ln(x).x);
        assert!(got < f * s, "{f}");
        assert!((got - want).abs() <= want * EPS, "{f}: {got} {want}");
    }
}

#[test]
fn powq_agrees_with_160_bit_arithmetic_for_random_powers() {
    let mut rng = Rng::new(72);
    let mut worst = 0.0f64;
    for _ in 0..if cfg!(miri) { 10 } else { 1000 } {
        let (x, p) = (rng.quaternion() * 3.0, rng.quaternion() * 2.0);
        let want = reference::powq(x, p);
        let kappa = 1.0 + (reference::ln(x) * p).norm();
        if let Some(e) = err(x.powq(p), want, kappa) {
            worst = worst.max(e);
        }
    }
    eprintln!("powq against 160 bits, largest error in ε scaled by the condition: {worst}");
    assert!(worst <= 3.0);
    // i^i = e^(−π/2), as for complex numbers; j^i = −k on the right, where e^(i·ln j) = k.
    let (i, j, k) = (Q::I, Q::J, Q::K);
    assert!((i.powq(i) - Q::from((-PI / 2.0).exp())).norm() <= EPS);
    assert!((j.powq(i) + k).norm() <= 2.0 * EPS);
    assert!(((i * j.ln()).exp() - k).norm() <= 2.0 * EPS);
}
