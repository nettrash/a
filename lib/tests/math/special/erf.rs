//! The erf area of issue #20: its data in `data/erf_*.txt`, made by `data/gen_erf.py`.
//!
//! Every bound is the measured largest error of its data (see `docs/math/special.md`), rounded
//! up a little. Below about 2⁻⁹⁷⁰ the low word of a reference is itself rounded to the least
//! subnormal, so a result there can measure up to 1 ulp while it is within 0.5 ulp + ε.

use super::common::*;
use a::math::complex::Complex;
use a::math::special::{dawson, erf, erf_inv, erfc, erfc_inv, erfcx, faddeeva, fresnel};

const ERF_F64: &str = include_str!("data/erf_erf_f64.txt");
const ERFC_F64: &str = include_str!("data/erf_erfc_f64.txt");
const ERFCX_F64: &str = include_str!("data/erf_erfcx_f64.txt");
const ERF_INV_F64: &str = include_str!("data/erf_erf_inv_f64.txt");
const ERFC_INV_F64: &str = include_str!("data/erf_erfc_inv_f64.txt");
const DAWSON_F64: &str = include_str!("data/erf_dawson_f64.txt");
const FRESNEL_F64: &str = include_str!("data/erf_fresnel_f64.txt");
const ERF_F32: &str = include_str!("data/erf_erf_f32.txt");
const ERFC_F32: &str = include_str!("data/erf_erfc_f32.txt");
const ERFCX_F32: &str = include_str!("data/erf_erfcx_f32.txt");
const ERF_INV_F32: &str = include_str!("data/erf_erf_inv_f32.txt");
const ERFC_INV_F32: &str = include_str!("data/erf_erfc_inv_f32.txt");
const DAWSON_F32: &str = include_str!("data/erf_dawson_f32.txt");
const FRESNEL_F32: &str = include_str!("data/erf_fresnel_f32.txt");
const FADDEEVA_C64: &str = include_str!("data/erf_faddeeva_c64.txt");
const FADDEEVA_LHP_C64: &str = include_str!("data/erf_faddeeva_lhp_c64.txt");
const ERF_C64: &str = include_str!("data/erf_erf_c64.txt");
const ERFC_C64: &str = include_str!("data/erf_erfc_c64.txt");
const ERFCX_C64: &str = include_str!("data/erf_erfcx_c64.txt");

/// The data, or under Miri every 37th case of it.
fn data(text: &str) -> String {
    if !cfg!(miri) {
        return text.to_string();
    }
    text.lines()
        .filter(|l| !l.starts_with('#'))
        .step_by(37)
        .collect::<Vec<_>>()
        .join("\n")
}

/// A real function of `f64` and of `f32` to their bounds.
fn real(
    name: &str,
    (t64, b64): (&str, f64),
    (t32, b32): (&str, f64),
    f64f: impl Fn(f64) -> f64,
    f32f: impl Fn(f32) -> f32,
) {
    let e64 = check_real(name, &data(t64), F64, b64, |a| f64f(a[0]));
    let e32 = check_real(name, &data(t32), F32, b32, |a| f32f(a[0] as f32) as f64);
    eprintln!("{name}: f64 {e64:.3} ulps, f32 {e32:.3} ulps");
}

#[test]
fn erf_data() {
    real("erf", (ERF_F64, 0.8), (ERF_F32, 0.501), erf, erf);
}

#[test]
fn erfc_data() {
    real("erfc", (ERFC_F64, 1.0), (ERFC_F32, 0.501), erfc, erfc);
}

#[test]
fn erfcx_data() {
    real("erfcx", (ERFCX_F64, 0.7), (ERFCX_F32, 0.501), erfcx, erfcx);
}

#[test]
fn erf_inv_data() {
    real(
        "erf_inv",
        (ERF_INV_F64, 0.8),
        (ERF_INV_F32, 0.501),
        erf_inv,
        erf_inv,
    );
}

#[test]
fn erfc_inv_data() {
    real(
        "erfc_inv",
        (ERFC_INV_F64, 0.8),
        (ERFC_INV_F32, 0.501),
        erfc_inv,
        erfc_inv,
    );
}

#[test]
fn dawson_data() {
    real(
        "dawson",
        (DAWSON_F64, 0.8),
        (DAWSON_F32, 0.501),
        dawson,
        dawson,
    );
}

#[test]
fn fresnel_data() {
    for (k, name) in ["fresnel S", "fresnel C"].into_iter().enumerate() {
        let part = |text: &str| -> String {
            // The case with the k-th reference only.
            text.lines()
                .map(|l| match l.split_once(':') {
                    Some((a, r)) if !l.starts_with('#') => {
                        let w: Vec<&str> = r.split_whitespace().collect();
                        format!("{a}: {} {}", w[2 * k], w[2 * k + 1])
                    }
                    _ => l.to_string(),
                })
                .collect::<Vec<_>>()
                .join("\n")
        };
        let pick = |p: (f64, f64)| if k == 0 { p.0 } else { p.1 };
        real(
            name,
            (&part(FRESNEL_F64), 1.0),
            (&part(FRESNEL_F32), 0.501),
            |x| pick(fresnel(x)),
            |x| {
                let (s, c) = fresnel(x);
                if k == 0 { s } else { c }
            },
        );
    }
}

fn cplx(name: &str, text: &str, bound: f64, f: impl Fn(Complex<f64>) -> Complex<f64>) {
    let e = check_complex(name, &data(text), F64, bound, |re, im| {
        let w = f(Complex::new(re, im));
        (w.re, w.im)
    });
    eprintln!("{name}: {e:.3} ε");
}

#[test]
fn faddeeva_data() {
    cplx("faddeeva", FADDEEVA_C64, 3.5, faddeeva);
    // The lower half-plane: 2e^(−z²) − w(−z) cancels near the zeros of w.
    cplx("faddeeva lower", FADDEEVA_LHP_C64, 9.0, faddeeva);
}

#[test]
fn complex_erf_data() {
    // Near the zeros of erf, erfc and erfcx (|z| > 2.3, off the real axis) the error relative
    // to |w| grows as |w| falls: the bounds are those regions'.
    cplx("erf c64", ERF_C64, 12.5, erf);
    cplx("erfc c64", ERFC_C64, 12.5, erfc);
    cplx("erfcx c64", ERFCX_C64, 21.0, erfcx);
}

#[test]
fn complex_f32_rounds_the_f64_values() {
    for &(x, y) in &[
        (0.3f32, 0.4f32),
        (-2.5, 1.25),
        (3.0, -2.0),
        (0.0, 1.5),
        (1.5, 0.0),
    ] {
        let z32 = Complex::new(x, y);
        let z64 = Complex::new(x as f64, y as f64);
        for (a, b) in [
            (erf(z32), erf(z64)),
            (erfc(z32), erfc(z64)),
            (erfcx(z32), erfcx(z64)),
            (faddeeva(z32), faddeeva(z64)),
        ] {
            assert_eq!((a.re, a.im), (b.re as f32, b.im as f32), "at {x} {y}");
        }
    }
}

#[test]
fn special_values() {
    assert_eq!(erf(0.0f64).to_bits(), 0.0f64.to_bits());
    assert_eq!(erf(-0.0f64).to_bits(), (-0.0f64).to_bits());
    assert_eq!(erf(f64::INFINITY), 1.0);
    assert_eq!(erf(f64::NEG_INFINITY), -1.0);
    assert_eq!(erfc(f64::INFINITY), 0.0);
    assert_eq!(erfc(f64::NEG_INFINITY), 2.0);
    assert_eq!(erfc(0.0f64), 1.0);
    assert_eq!(erfcx(0.0f64), 1.0);
    assert_eq!(erfcx(f64::INFINITY), 0.0);
    assert_eq!(erfcx(f64::NEG_INFINITY), f64::INFINITY);
    assert_eq!(erfcx(-30.0f64), f64::INFINITY);
    let big = erfcx(f64::MAX) / (0.564_189_583_547_756_3 / f64::MAX);
    assert!((big - 1.0).abs() < 1e-15, "{big}");
    assert_eq!(erf_inv(1.0f64), f64::INFINITY);
    assert_eq!(erf_inv(-1.0f64), f64::NEG_INFINITY);
    assert_eq!(erf_inv(0.0f64).to_bits(), 0.0f64.to_bits());
    assert_eq!(erf_inv(-0.0f64).to_bits(), (-0.0f64).to_bits());
    assert_eq!(erfc_inv(0.0f64), f64::INFINITY);
    assert_eq!(erfc_inv(2.0f64), f64::NEG_INFINITY);
    assert_eq!(erfc_inv(1.0f64), 0.0);
    assert_eq!(dawson(0.0f64), 0.0);
    assert_eq!(dawson(f64::INFINITY), 0.0);
    assert_eq!(dawson(f64::NEG_INFINITY).to_bits(), (-0.0f64).to_bits());
    assert_eq!(fresnel(0.0f64), (0.0, 0.0));
    assert_eq!(fresnel(f64::INFINITY), (0.5, 0.5));
    assert_eq!(fresnel(f64::NEG_INFINITY), (-0.5, -0.5));
    assert_eq!(fresnel(1e300f64), (0.5, 0.5));
    for x in [f64::NAN, -f64::NAN] {
        assert!(erf(x).is_nan() && erfc(x).is_nan() && erfcx(x).is_nan());
        assert!(erf_inv(x).is_nan() && erfc_inv(x).is_nan() && dawson(x).is_nan());
        assert!(fresnel(x).0.is_nan() && fresnel(x).1.is_nan());
    }
    for x in [1.0000001f64, -1.5, f64::INFINITY] {
        assert!(erf_inv(x).is_nan(), "{x}");
    }
    for x in [-1e-300f64, 2.0000001, f64::NEG_INFINITY] {
        assert!(erfc_inv(x).is_nan(), "{x}");
    }
    assert!(erf_inv(f32::NAN).is_nan() && erfc_inv(3.0f32).is_nan());
    // Underflow and overflow at the right places.
    assert!(erfc(27.2f64) > 0.0 && erfc(27.3f64) == 0.0);
    assert!(erfcx(-26.62f64).is_finite() && erfcx(-26.63f64) == f64::INFINITY);
    assert!(erfc(10.0f32) > 0.0 && erfc(10.1f32) == 0.0);
}

#[test]
fn complex_special_values() {
    let c = Complex::new;
    let w = faddeeva(c(0.0f64, 0.0));
    assert_eq!((w.re, w.im), (1.0, 0.0));
    let w = faddeeva(c(f64::INFINITY, 1.0));
    assert_eq!((w.re, w.im), (0.0, 0.0));
    assert!(faddeeva(c(f64::NAN, 1.0f64)).re.is_nan());
    assert!(faddeeva(c(1.0, f64::NEG_INFINITY)).re.is_nan());
    let e = erf(c(f64::INFINITY, 2.0));
    assert_eq!((e.re, e.im), (1.0, 0.0));
    let e = erf(c(f64::NEG_INFINITY, -2.0));
    assert_eq!((e.re, e.im), (-1.0, -0.0));
    let e = erfc(c(f64::NEG_INFINITY, 2.0));
    assert_eq!(e.re, 2.0);
    let e = erfcx(c(f64::INFINITY, 3.0));
    assert_eq!((e.re, e.im), (0.0, 0.0));
    assert!(erf(c(f64::NAN, 0.5f64)).re.is_nan() && erfc(c(0.5, f64::NAN)).re.is_nan());
    // On the axes the real functions.
    for x in [-3.0f64, -0.3, 0.7, 2.0, 9.0] {
        assert_eq!(erf(c(x, 0.0)).re, erf(x));
        assert_eq!(erfc(c(x, 0.0)).re, erfc(x));
        assert_eq!(erfcx(c(x, 0.0)).re, erfcx(x));
        let w = faddeeva(c(0.0, x.abs()));
        assert_eq!((w.re, w.im), (erfcx(x.abs()), 0.0));
        let w = faddeeva(c(x, 0.0));
        assert!((w.im - core::f64::consts::FRAC_2_SQRT_PI * dawson(x)).abs() <= 1e-16 * w.im.abs());
    }
}

#[test]
fn identities() {
    for i in -60..60 {
        let x = i as f64 * 0.1 + 0.03;
        // erf + erfc = 1 and the symmetries.
        assert!((erf(x) + erfc(x) - 1.0).abs() < 3e-16, "{x}");
        assert_eq!(erf(-x), -erf(x));
        assert!((erfc(-x) - (2.0 - erfc(x))).abs() <= 4.5e-16, "{x}");
        assert_eq!(dawson(-x), -dawson(x));
        let (s, c) = fresnel(x);
        let (s2, c2) = fresnel(-x);
        assert_eq!((s2, c2), (-s, -c));
        // The inverses invert, to the error that half an ulp of erf(x) makes (|d erf_inv/dy| =
        // √π/2·e^(x²)) and their own.
        let eps = f64::EPSILON;
        if x.abs() < 2.0 {
            let y = erf(x);
            let tol = 2.0 * eps * (x.abs() + (x * x).exp() * y.abs());
            assert!((erf_inv(y) - x).abs() <= tol, "{x}: {}", erf_inv(y));
        }
        if x > -1.0 && x < 5.5 {
            let tol = 4.0 * eps * (x.abs() + erfcx(x));
            assert!(
                (erfc_inv(erfc(x)) - x).abs() <= tol,
                "{x}: {}",
                erfc_inv(erfc(x))
            );
        }
    }
    // erfcx = e^(x²)·erfc where neither under- nor overflows.
    for x in [-3.0f64, -0.7, 0.2, 1.3, 5.5, 20.0] {
        let want = (x * x).exp() * erfc(x);
        assert!((erfcx(x) - want).abs() <= 1e-14 * want, "{x}");
    }
    // Monotonic across the seams of the pieces.
    for &edge in &[0.5f64, 1.0, 2.0, 4.0, 6.0, 8.0] {
        let mut x = edge;
        for _ in 0..8 {
            x = f64::from_bits(x.to_bits() - 1);
        }
        let mut prev = (erf(x), erfc(x), erfcx(x));
        for _ in 0..16 {
            x = f64::from_bits(x.to_bits() + 1);
            let now = (erf(x), erfc(x), erfcx(x));
            assert!(
                now.0 >= prev.0 && now.1 <= prev.1 && now.2 <= prev.2,
                "at {x}"
            );
            prev = now;
        }
    }
}

#[test]
fn complex_symmetries() {
    let c = Complex::new;
    for &(x, y) in &[
        (0.3f64, 0.4f64),
        (-2.5, 1.25),
        (3.0, -2.0),
        (0.01, 5.0),
        (7.0, 0.5),
        (12.0, 3.0),
    ] {
        let z = c(x, y);
        // erf(conj z) = conj erf(z), erf(−z) = −erf(z).
        let (a, b) = (erf(c(x, -y)), erf(z));
        assert_eq!((a.re, a.im), (b.re, -b.im));
        let (a, b) = (erf(c(-x, -y)), erf(z));
        assert_eq!((a.re, a.im), (-b.re, -b.im));
        // erf + erfc = 1 (where erf is not large).
        let (e, f) = (erf(z), erfc(z));
        let n = e.re.hypot(e.im).max(1.0);
        assert!(
            (e.re + f.re - 1.0).abs() < 1e-14 * n && (e.im + f.im).abs() < 1e-14 * n,
            "{x} {y}"
        );
        // erfcx(z) = w(iz).
        let (a, b) = (erfcx(z), faddeeva(c(-y, x)));
        assert_eq!((a.re, a.im), (b.re, b.im));
        // w(−conj z) = conj w(z) in the upper half-plane.
        if y > 0.0 {
            let (a, b) = (faddeeva(c(-x, y)), faddeeva(z));
            assert_eq!((a.re, a.im), (b.re, -b.im));
        }
    }
}
