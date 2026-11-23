//! The erf area of issue #20: its data in `data/erf_*.txt`, made by `data/gen_erf.py`.
//!
//! The bounds are those of `docs/math/special.md` (by region), which also gives the largest
//! errors found by a search of millions of arguments; the data hold the worst arguments found.
//! Below about 2⁻¹⁰⁰⁰ the low word of a reference is itself rounded to the least subnormal, up to
//! 2⁻¹⁰⁷⁵: a result there is held to its bound plus that much (half an ulp for a subnormal one,
//! whose reference is then just the correctly rounded value). A complex case carries the scale of
//! its bound as a third reference: the larger of the two terms the formula subtracts, or `|v|`
//! where nothing cancels; one beyond the largest double is stored negated and divided by 2¹⁰²⁴.

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
const FADDEEVA_AXIS_C64: &str = include_str!("data/erf_faddeeva_axis_c64.txt");
const ERF_AXIS_C64: &str = include_str!("data/erf_erf_axis_c64.txt");
const ERFC_AXIS_C64: &str = include_str!("data/erf_erfc_axis_c64.txt");
const FADDEEVA_IMAG_C64: &str = include_str!("data/erf_faddeeva_imag_c64.txt");
const FADDEEVA_C32: &str = include_str!("data/erf_faddeeva_c32.txt");
const ERF_C32: &str = include_str!("data/erf_erf_c32.txt");
const ERFC_C32: &str = include_str!("data/erf_erfc_c32.txt");
const ERFCX_C32: &str = include_str!("data/erf_erfcx_c32.txt");

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

/// `2^e` for `−1022 ≤ e ≤ 1023`.
fn pow2(e: i32) -> f64 {
    f64::from_bits(((e + 1023) as u64) << 52)
}

/// The rounding of a reference's low word to the least subnormal (at most 2⁻¹⁰⁷⁵), in ulps of a
/// result near `hi`: half an ulp below 2⁻¹⁰²², 2⁻¹⁰²³⁻ᵉ ulps in the binade `2^e` above, negligible
/// from about 2⁻¹⁰⁰⁰ up.
fn quantum(hi: f64) -> f64 {
    let e = ((hi.to_bits() >> 52) & 0x7ff) as i32 - 1023;
    if e < -1022 {
        0.5
    } else if e > -1000 {
        0.0
    } else {
        pow2(-1023 - e)
    }
}

/// A real function of `f64` to the bound of each argument's region (plus the quantisation of a
/// tiny reference) and of `f32` to `b32`.
#[track_caller]
fn real(
    name: &str,
    (t64, b64): (&str, impl Fn(f64) -> f64),
    (t32, b32): (&str, f64),
    f64f: impl Fn(f64) -> f64,
    f32f: impl Fn(f32) -> f32,
) {
    let mut e64 = 0.0f64;
    let cases = cases(&data(t64));
    assert!(!cases.is_empty(), "{name}: no cases");
    for c in &cases {
        let (x, r) = (c.args[0], c.refs[0]);
        let y = f64f(x);
        let e = ulps(y, r, F64);
        let bound = b64(x) + quantum(r.0);
        assert!(
            e <= bound,
            "{name}: {e:.3} ulps > {bound} at {x:?} ({:x}): got {y:e}, want {:e} + {:e}",
            x.to_bits(),
            r.0,
            r.1
        );
        e64 = e64.max(e - quantum(r.0));
    }
    let e32 = check_real(name, &data(t32), F32, b32, |a| f32f(a[0] as f32) as f64);
    eprintln!("{name}: f64 {e64:.3} ulps, f32 {e32:.3} ulps");
}

#[test]
fn erf_data() {
    let b = |x: f64| if x.abs() < 0.5 { 0.86 } else { 0.8 };
    real("erf", (ERF_F64, b), (ERF_F32, 0.501), erf, erf);
}

#[test]
fn erfc_data() {
    let b = |x: f64| if x.abs() < 0.5 { 0.92 } else { 0.85 };
    real("erfc", (ERFC_F64, b), (ERFC_F32, 0.501), erfc, erfc);
}

#[test]
fn erfcx_data() {
    let b = |x: f64| if x.abs() < 0.5 { 0.78 } else { 0.7 };
    real("erfcx", (ERFCX_F64, b), (ERFCX_F32, 0.501), erfcx, erfcx);
}

#[test]
fn erf_inv_data() {
    let b = |x: f64| if x.abs() <= 0.5 { 0.9 } else { 0.98 };
    real(
        "erf_inv",
        (ERF_INV_F64, b),
        (ERF_INV_F32, 0.501),
        erf_inv,
        erf_inv,
    );
}

#[test]
fn erfc_inv_data() {
    real(
        "erfc_inv",
        (ERFC_INV_F64, |_| 0.98),
        (ERFC_INV_F32, 0.501),
        erfc_inv,
        erfc_inv,
    );
}

#[test]
fn dawson_data() {
    let b = |x: f64| if x.abs() < 0.5 { 1.07 } else { 0.86 };
    real(
        "dawson",
        (DAWSON_F64, b),
        (DAWSON_F32, 0.501),
        dawson,
        dawson,
    );
}

#[test]
fn fresnel_data() {
    // S, C below 1, from 1 to 8 and beyond. The bounds are 1.14, 1.34; 1.12, 1.23; 0.85, 0.79 (the
    // worst cases of the roundings); below 8 the data are held to tighter figures above the
    // largest errors found (0.894, 0.934; 0.930, 1.014), so that a loss fails: below 1 the square of
    // the rounded x² for x⁴ (up to 1.008 and 1.022 on these data before 2026-11-21).
    for (k, (name, small, mid, far)) in
        [("fresnel S", 1.0, 1.0, 0.85), ("fresnel C", 1.0, 1.1, 0.79)]
            .into_iter()
            .enumerate()
    {
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
            (&part(FRESNEL_F64), |x: f64| {
                if x.abs() < 1.0 {
                    small
                } else if x.abs() < 8.0 {
                    mid
                } else {
                    far
                }
            }),
            (&part(FRESNEL_F32), 0.501),
            |x| pick(fresnel(x)),
            |x| {
                let (s, c) = fresnel(x);
                if k == 0 { s } else { c }
            },
        );
    }
}

#[test]
fn erfcx_and_dawson_are_correctly_rounded_from_2_to_the_960() {
    // From 2⁹⁶⁰ both are 1/(√π x) and 1/(2x) to the last bit, from a scaled reciprocal rounded
    // once (the plain form's low words were subnormal beyond about 2¹⁰¹⁵: up to 1.32 ulps before
    // 2026-11-21): the result is the reference's nearest double, subnormal ones too. That is hi,
    // and for a subnormal value hi + lo (exact): mpmath rounds a subnormal value to 53 bits before
    // it rounds it to the subnormals, and lo, rounded to the least subnormal, makes up for it.
    let mut n = 0;
    for (name, text, f) in [
        ("erfcx", ERFCX_F64, erfcx::<f64> as fn(f64) -> f64),
        ("dawson", DAWSON_F64, dawson::<f64>),
    ] {
        for c in cases(&data(text)) {
            let x = c.args[0];
            if !(x.abs() >= pow2(960) && x.is_finite()) {
                continue;
            }
            let (hi, lo) = c.refs[0];
            let want = if hi.abs() >= f64::MIN_POSITIVE {
                hi
            } else {
                hi + lo
            };
            let y = f(x);
            assert_eq!(
                y.to_bits(),
                want.to_bits(),
                "{name}({x:e}) [{:x}] = {y:e}, the nearest double is {want:e}",
                x.to_bits()
            );
            n += 1;
        }
    }
    assert!(cfg!(miri) || n >= 200, "{n} cases from 2^960");
}

/// The error of a complex result against a case `[re, im, scale]`, in units of `ε` of `p`,
/// relative to the scale (`K = scale/|v|` times smaller than relative to `|v|`), and `K`. A scale
/// below the least normal number counts as that number, so that a subnormal result is measured in
/// units of the least subnormal. A scale beyond the largest double (stored negated, divided by
/// 2¹⁰²⁴) is applied in that unit. An infinite reference part must be matched exactly (`∞`
/// otherwise), and a finite part next to it is measured as any part is, against the scale: in
/// absolute terms, so that it can be far off relative to itself where it is far below `|v|`.
fn scaled_errors(got: (f64, f64), refs: &[(f64, f64)], p: Prec) -> (f64, f64) {
    let (re, im, sc) = (refs[0], refs[1], refs[2]);
    let eps = pow2(1 - p.bits);
    let tiny = pow2(p.min_exp + p.bits - 1);
    // `unit`: 1, or 2⁻¹⁰²⁴ (in two exact steps) where the scale is beyond the largest double.
    let (unit, sc) = if sc.0 < 0.0 {
        (pow2(-512), -sc.0)
    } else {
        (1.0, sc.0)
    };
    let down = |v: f64| if unit == 1.0 { v } else { v * unit * unit };
    if !(re.0.is_finite() && im.0.is_finite()) {
        let part = |g: f64, r: (f64, f64)| {
            if !r.0.is_finite() {
                if g == r.0 { 0.0 } else { f64::INFINITY }
            } else if !g.is_finite() {
                f64::INFINITY
            } else {
                // (A scale beyond 2²⁰⁴⁸ is stored as −∞: next to it every finite error is 0.)
                assert!(
                    unit != 1.0,
                    "a finite part next to an infinite one: {refs:?}"
                );
                down((g - r.0) - r.1).abs() / sc / eps
            }
        };
        return (part(got.0, re).max(part(got.1, im)), 1.0);
    }
    if got.0.is_nan() || got.1.is_nan() || got.0.is_infinite() || got.1.is_infinite() {
        return (f64::INFINITY, 1.0);
    }
    let d = down((got.0 - re.0) - re.1).hypot(down((got.1 - im.0) - im.1));
    let norm = down(re.0).hypot(down(im.0));
    let scale = sc.max(norm);
    (d / scale.max(tiny) / eps, scale / norm.max(tiny))
}

/// The bound of `docs/math/special.md` of the region of `fn(x + iy)`, in ε of the scale: the
/// worst case of the roundings of the path that computes it (the largest of the analysis found,
/// with 3 % for its search).
fn bound_of(func: &str, x: f64, y: f64) -> f64 {
    let small = x.hypot(y) < 10.0;
    let boxed = x.abs() <= 10.0 && y.abs() <= 10.0;
    match func {
        "faddeeva" if y >= 0.0 => {
            if small {
                19.6
            } else {
                4.3
            }
        }
        "faddeeva" => {
            if boxed && y >= -10.0 {
                19.0
            } else {
                6.2
            }
        }
        "erf" if x.hypot(y) < 1.0 => 3.9,
        "erf" => {
            if boxed {
                27.0
            } else {
                9.4
            }
        }
        "erfc" => {
            if boxed {
                29.0
            } else {
                12.2
            }
        }
        _ => {
            // erfcx(z) = w(iz)
            if boxed {
                19.6
            } else if x > 0.0 {
                4.3
            } else {
                6.2
            }
        }
    }
}

/// What the tests hold a complex case to: the bound of its region, but no more than 8 ε, above
/// the largest error found anywhere (6.7, at erfc(0.5568 + 0.2313i)) and well below the worst
/// case of the trapezoidal rule's roundings, so that a change that loses accuracy fails.
const CPLX_TEST: f64 = 8.0;

/// Checks a complex function of the data `text` to the bound of each case's region (and
/// `CPLX_TEST`); returns the largest error relative to the scale and, where `K < 2` (away from
/// the zeros), relative to `|v|` (at most twice the former there). Below the real axis (`erfcx`:
/// left of the imaginary one) the value is `2e^(∓z²)` but a part below 1, and a finite part next
/// to an infinite one is held to the bound relative to itself too.
#[track_caller]
fn check_scaled(
    name: &str,
    func: &str,
    text: &str,
    p: Prec,
    f: impl Fn(f64, f64) -> (f64, f64),
) -> (f64, f64) {
    let cases = cases(text);
    assert!(!cases.is_empty(), "{name}: no cases");
    let (mut worst_s, mut worst_w) = (0.0f64, 0.0f64);
    for c in &cases {
        assert_eq!(c.refs.len(), 3, "{name}: a case needs re, im and the scale");
        let (x, y) = (c.args[0], c.args[1]);
        let got = f(x, y);
        let (es, k) = scaled_errors(got, &c.refs, p);
        let bound = bound_of(func, x, y).min(CPLX_TEST);
        assert!(
            es <= bound,
            "{name}: {es:.3} ε of the scale > {bound} at {:?} ({:x?}): got {got:?}, want {:?}",
            c.args,
            c.args.iter().map(|a| a.to_bits()).collect::<Vec<_>>(),
            c.refs
        );
        let lower = (func == "faddeeva" && y < 0.0) || (func == "erfcx" && x < 0.0);
        let (re, im) = (c.refs[0], c.refs[1]);
        if lower && p.bits == 53 && re.0.is_finite() != im.0.is_finite() {
            let (g, r) = if re.0.is_finite() {
                (got.0, re)
            } else {
                (got.1, im)
            };
            let e = part_ulps(g, r);
            assert!(
                e <= bound,
                "{name}: the finite part {e:.3} ulps of itself > {bound} at {:?}: got {got:?}, want {:?}",
                c.args,
                c.refs
            );
        }
        worst_s = worst_s.max(es);
        if k < 2.0 {
            worst_w = worst_w.max(es * k);
        }
    }
    (worst_s, worst_w)
}

fn cplx(name: &str, func: &str, text: &str, f: impl Fn(Complex<f64>) -> Complex<f64>) {
    let (es, ew) = check_scaled(name, func, &data(text), F64, |re, im| {
        let w = f(Complex::new(re, im));
        (w.re, w.im)
    });
    eprintln!("{name}: {es:.3} ε of the scale, {ew:.3} ε of |w| where K < 2");
}

#[test]
fn faddeeva_data() {
    // The upper half-plane (no cancellation), then the lower one: 2e^(−z²) − w(−z).
    cplx("faddeeva", "faddeeva", FADDEEVA_C64, faddeeva);
    cplx("faddeeva lower", "faddeeva", FADDEEVA_LHP_C64, faddeeva);
}

#[test]
fn complex_erf_data() {
    // Near the zeros of erf, erfc and erfcx (|z| > 2.3, off the real axis) the value is a
    // difference of terms far larger than it: the bound is relative to the larger one.
    cplx("erf c64", "erf", ERF_C64, erf);
    cplx("erfc c64", "erfc", ERFC_C64, erfc);
    cplx("erfcx c64", "erfcx", ERFCX_C64, erfcx);
}

/// The error of a part in ulps of itself (an infinite reference must be matched), less the
/// quantisation of a tiny reference.
fn part_ulps(got: f64, r: (f64, f64)) -> f64 {
    (ulps(got, r, F64) - quantum(r.0)).max(0.0)
}

#[test]
fn the_real_part_of_w_next_to_the_real_axis_is_right_on_its_own() {
    // Re w(x + iy) is the Voigt function: e^(−x²) on the axis, which the pole correction of the
    // trapezoidal rule carries; it was lost for 6.708 ≤ |x| < 10 and a small y. The bounds are the
    // worst case of the roundings of its terms (33 ulps below |x| = 10, 10.5 beyond), held here to
    // 14 below |x| = 10 (11.8 found) and 10 beyond (4.3).
    let text = data(FADDEEVA_AXIS_C64);
    let mut worst = 0.0f64;
    for c in &cases(&text) {
        let (x, y) = (c.args[0], c.args[1]);
        let w = faddeeva(Complex::new(x, y));
        let e = part_ulps(w.re, c.refs[0]);
        let bound = if x.abs() < 10.0 { 14.0 } else { 10.0 };
        assert!(
            e <= bound,
            "Re w: {e:.3} ulps > {bound} at {:?}: got {:e}, want {:e}",
            c.args,
            w.re,
            c.refs[0].0
        );
        worst = worst.max(e);
    }
    cplx(
        "faddeeva next to the real axis",
        "faddeeva",
        FADDEEVA_AXIS_C64,
        faddeeva,
    );
    eprintln!("Re w next to the real axis: {worst:.3} ulps");
}

#[test]
fn erf_and_erfc_next_to_the_imaginary_axis_are_right_part_by_part() {
    // For |x|·max(|y|, 1) < 2⁻¹⁴ (erf from |z| = 1) both are taken part by part from the Taylor
    // series in x: Re erf (of the size of x) within 0.6 ulps of itself, the imaginary parts
    // within 4, and Re erfc = 1 − Re erf within 0.6 ε of max(|Re|, 1) (its 1 was lost for
    // 6.708 ≤ |Im z| < 10 before 2026-11-21).
    for (name, text, f) in [
        (
            "erf",
            ERF_AXIS_C64,
            erf::<Complex<f64>> as fn(Complex<f64>) -> Complex<f64>,
        ),
        ("erfc", ERFC_AXIS_C64, erfc::<Complex<f64>>),
    ] {
        let text = data(text);
        let (mut wr, mut wi) = (0.0f64, 0.0f64);
        for c in &cases(&text) {
            let v = f(Complex::new(c.args[0], c.args[1]));
            let (hi, lo) = c.refs[0];
            let er = if name == "erf" {
                part_ulps(v.re, c.refs[0])
            } else if hi.is_finite() {
                ((v.re - hi) - lo).abs() / hi.abs().max(1.0) / f64::EPSILON
            } else {
                part_ulps(v.re, c.refs[0])
            };
            let ei = part_ulps(v.im, c.refs[1]);
            assert!(
                er <= 0.6 && ei <= 4.0,
                "{name}: Re {er:.3} (bound 0.6), Im {ei:.3} ulps (bound 4) at {:?}: got {v:?}, want {:?}",
                c.args,
                c.refs
            );
            wr = wr.max(er);
            wi = wi.max(ei);
        }
        cplx(
            &format!("{name} next to the imaginary axis"),
            name,
            &text,
            f,
        );
        eprintln!("{name} next to the imaginary axis: Re {wr:.3}, Im {wi:.3} ulps");
    }
}

#[test]
fn w_below_the_real_axis_next_to_the_imaginary_one_is_right_part_by_part() {
    // For |x|·max(−y, 1) < 2⁻³⁰, w = erfcx(y) + 2ix·(1/√π − y·erfcx(y)): the real part within the
    // bound of erfcx and the imaginary one within 3.1 ulps of itself, also where e^(y²) is beyond
    // any double and one of them is not. erfcx(z) = w(iz) is its image left of the imaginary axis.
    let text = data(FADDEEVA_IMAG_C64);
    let (mut wr, mut wi) = (0.0f64, 0.0f64);
    for c in &cases(&text) {
        let (x, y) = (c.args[0], c.args[1]);
        for (v, er_bound) in [(faddeeva(Complex::new(x, y)), 0.8), {
            let e = erfcx(Complex::new(y, -x));
            (e, 0.8)
        }] {
            let er = part_ulps(v.re, c.refs[0]);
            let ei = part_ulps(v.im, c.refs[1]);
            assert!(
                er <= er_bound && ei <= 3.1,
                "w({x:e} + {y:e}i): Re {er:.3}, Im {ei:.3} ulps: got {v:?}, want {:?}",
                c.refs
            );
            wr = wr.max(er);
            wi = wi.max(ei);
        }
    }
    cplx(
        "faddeeva below, next to the imaginary axis",
        "faddeeva",
        &text,
        faddeeva,
    );
    eprintln!("w below, next to the imaginary axis: Re {wr:.3}, Im {wi:.3} ulps");
}

#[test]
fn complex_f32_data() {
    // The Complex<f64> values rounded once a part: within half an ε of |w| (the data hold no
    // case within a factor 100 of a zero, where the f64 error could show).
    for (name, text, f) in [
        (
            "faddeeva c32",
            FADDEEVA_C32,
            faddeeva::<f32> as fn(Complex<f32>) -> Complex<f32>,
        ),
        ("erf c32", ERF_C32, erf::<Complex<f32>>),
        ("erfc c32", ERFC_C32, erfc::<Complex<f32>>),
        ("erfcx c32", ERFCX_C32, erfcx::<Complex<f32>>),
    ] {
        let e = check_complex(name, &data(text), F32, 0.501, |re, im| {
            let w = f(Complex::new(re as f32, im as f32));
            (w.re as f64, w.im as f64)
        });
        eprintln!("{name}: {e:.4} ε");
    }
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
    // Dawson's integral is odd, at ±0 too.
    assert_eq!(dawson(0.0f64).to_bits(), 0.0f64.to_bits());
    assert_eq!(dawson(-0.0f64).to_bits(), (-0.0f64).to_bits());
    assert_eq!(dawson(0.0f32).to_bits(), 0.0f32.to_bits());
    assert_eq!(dawson(-0.0f32).to_bits(), (-0.0f32).to_bits());
    assert_eq!(dawson(f64::INFINITY), 0.0);
    assert_eq!(dawson(f64::NEG_INFINITY).to_bits(), (-0.0f64).to_bits());
    assert_eq!(fresnel(0.0f64), (0.0, 0.0));
    let (s, c) = fresnel(-0.0f64);
    assert_eq!(
        (s.to_bits(), c.to_bits()),
        ((-0.0f64).to_bits(), (-0.0f64).to_bits())
    );
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
    // |z| → ∞ with Im z bounded below: 0, in the lower half-plane too.
    for (x, y) in [
        (f64::INFINITY, 1.0),
        (f64::NEG_INFINITY, 0.0),
        (f64::INFINITY, -1.0),
        (f64::NEG_INFINITY, -1e300),
        (1.0, f64::INFINITY),
        (f64::INFINITY, f64::INFINITY),
    ] {
        let w = faddeeva(c(x, y));
        assert_eq!((w.re, w.im), (0.0, 0.0), "w({x} + {y}i)");
    }
    // On the imaginary axis erfcx(y), to −i∞; off it −i∞ has no limit.
    let w = faddeeva(c(0.0, f64::NEG_INFINITY));
    assert_eq!((w.re, w.im), (f64::INFINITY, 0.0));
    assert!(faddeeva(c(f64::NAN, 1.0f64)).re.is_nan());
    // A NaN on the imaginary axis is erfcx(NaN), its zero imaginary part kept, as erfcx(NaN ± 0i)
    // has it (erfcx(z) = w(iz)).
    for x in [0.0f64, -0.0] {
        let w = faddeeva(c(x, f64::NAN));
        assert!(
            w.re.is_nan() && w.im.to_bits() == x.to_bits(),
            "w({x} + NaN i)"
        );
    }
    assert!(faddeeva(c(1.0, f64::NEG_INFINITY)).re.is_nan());
    assert!(faddeeva(c(f64::INFINITY, f64::NEG_INFINITY)).re.is_nan());
    let e = erf(c(f64::INFINITY, 2.0));
    assert_eq!((e.re, e.im), (1.0, 0.0));
    let e = erf(c(f64::NEG_INFINITY, -2.0));
    assert_eq!((e.re, e.im), (-1.0, -0.0));
    let e = erfc(c(f64::NEG_INFINITY, 2.0));
    assert_eq!(e.re, 2.0);
    let e = erfcx(c(f64::INFINITY, 3.0));
    assert_eq!((e.re, e.im), (0.0, 0.0));
    let e = erfcx(c(-1.0, f64::INFINITY));
    assert_eq!((e.re, e.im), (0.0, 0.0));
    assert!(erf(c(f64::NAN, 0.5f64)).re.is_nan() && erfc(c(0.5, f64::NAN)).re.is_nan());
    // On the axes the real functions.
    for x in [-3.0f64, -0.3, 0.7, 2.0, 9.0] {
        assert_eq!(erf(c(x, 0.0)).re, erf(x));
        assert_eq!(erfc(c(x, 0.0)).re, erfc(x));
        assert_eq!(erfcx(c(x, 0.0)).re, erfcx(x));
        let w = faddeeva(c(x, 0.0));
        assert!((w.im - core::f64::consts::FRAC_2_SQRT_PI * dawson(x)).abs() <= 1e-16 * w.im.abs());
    }
    // w(iy) = erfcx(y) on the whole imaginary axis, the sign of the zero part that of Re z.
    for y in [
        -1e200f64, -27.0, -26.6, -10.0, -3.0, -0.5, -1e-300, 1e-300, 0.5, 3.0, 30.0, 1e300,
    ] {
        for x in [0.0f64, -0.0] {
            let w = faddeeva(c(x, y));
            assert_eq!(w.re.to_bits(), erfcx(y).to_bits(), "w({x} + {y}i)");
            assert_eq!(w.im.to_bits(), x.to_bits(), "w({x} + {y}i)");
        }
    }
    // erfc(iy) = 1 − i·erfi(y), its 1 also where e^(−y²) underflows; erf(iy) = i·erfi(y).
    for y in [-30.0f64, -26.0, -3.0, -0.25, 0.25, 3.0, 26.0, 30.0, 1e200] {
        let (e, f) = (erf(c(0.0, y)), erfc(c(0.0, y)));
        assert_eq!(
            (f.re, f.im.to_bits()),
            (1.0, (-e.im).to_bits()),
            "erfc({y}i)"
        );
    }
}

#[test]
fn values_beyond_the_square_root_of_the_largest_double_and_on_the_diagonals() {
    // Infinite values with the signs of their phase, finite ones where the squares of the
    // parts overflow (they were zero, tiny or NaN before 2026-11-21).
    let c = Complex::new;
    let check = |name: &str, v: Complex<f64>, want: (f64, f64)| {
        assert_eq!((v.re, v.im), want, "{name}");
    };
    let inf = f64::INFINITY;
    check("erf(1 + 1e200i)", erf(c(1.0, 1e200)), (-inf, inf));
    check("erfc(1 + 1e200i)", erfc(c(1.0, 1e200)), (inf, -inf));
    check("erfc(0 + 1e200i)", erfc(c(0.0, 1e200)), (1.0, -inf));
    check("w(1 − 1e200i)", faddeeva(c(1.0, -1e200)), (inf, -inf));
    check("erfcx(−1e200 + i)", erfcx(c(-1e200, 1.0)), (inf, inf));
    check("erf(1e300 + 1e8i)", erf(c(1e300, 1e8)), (1.0, -0.0));
    check("erf(1.2e154 + 1e154i)", erf(c(1.2e154, 1e154)), (1.0, 0.0));
    assert!(erfc(c(1e300, 1e8)).re == 0.0 && erfc(c(-1e300, 1e8)).re == 2.0);
    let w = faddeeva(c(1e300, -1e8));
    assert_eq!(w.im, 5.641_895_835_477_563e-301);
    let w = faddeeva(c(1e308, -1.0));
    assert_eq!(w.im, 5.641_895_835_477_56e-309);
    // Next to and on the diagonals, |w| = 2 below the real axis: right to a few ε at any |z|.
    for (x, want) in [
        (1.1e12, (-0.529_424_841_179_702_1, -1.928_654_799_994_319_7)),
        (
            12_345_678.9,
            (-0.313_411_728_094_467_2, 1.975_290_660_456_092_3),
        ),
    ] {
        let w = faddeeva(c(x, -x));
        let d = (w.re - want.0).hypot(w.im - want.1) / 2.0;
        assert!(d < 4.0 * f64::EPSILON, "w({x} − {x}i) = {w:?}");
    }
    // Where the exact value overflows but a part of the factor e^(−z²) multiplies is tiny, every
    // part is still infinite (they were finite before the review of 2026-11-21: the power of 2
    // was clamped to 2040), and next to the axes a finite part next to an infinite one is right
    // on its own.
    check("erf(1 + 1e306i)", erf(c(1.0, 1e306)), (inf, -inf));
    check("erfc(1 + 1e306i)", erfc(c(1.0, 1e306)), (-inf, inf));
    check(
        "erf(4.05e-18 + 2.85e306i)",
        erf(c(4.053_377_871_816_522e-18, 2.852_998_099_721_371e306)),
        (inf, -inf),
    );
    check(
        "erf(2.32e-313 + 7.17e242i)",
        erf(c(2.3195109627e-313, 7.165_292_968_072_47e242)),
        (inf, inf),
    );
    check(
        "w(−2e-323 − 5690.35i)",
        faddeeva(c(-2e-323, -5_690.354_600_842_763)),
        (inf, -inf),
    );
    check(
        "erfcx(−12076671.74 − 2.85e-317i)",
        erfcx(c(-12_076_671.741_680_516, -2.851648e-317)),
        (inf, inf),
    );
    let w = faddeeva(c(-1e-320, -30.0));
    assert_eq!(w.re, inf);
    assert!(
        (w.im / -8.794_479_158_349_042e72 - 1.0).abs() < 4e-16,
        "{w:?}"
    );
    let e = erf(c(1e-310, 30.0));
    assert_eq!(e.im, inf);
    assert!(
        (e.re / 8.269_681_287_964_97e80 - 1.0).abs() < 4e-16,
        "{e:?}"
    );
    // Both parts beyond the half of the largest double (Smith's denominator overflowed).
    let w = faddeeva(c(1.2e308, 1.2e308));
    assert_eq!(
        (w.re, w.im),
        (2.350_789_931_448_986e-309, 2.350_789_931_448_986e-309)
    );
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
    let bits = |v: Complex<f64>| (v.re.to_bits(), v.im.to_bits());
    for &(x, y) in &[
        (0.3f64, 0.4f64),
        (-2.5, 1.25),
        (3.0, -2.0),
        (0.01, 5.0),
        (7.0, 0.5),
        (12.0, 3.0),
        (1e-300, 7.0),
        (2e-30, 8.0),
        (148.66, 148.68),
        (1.1e12, 1.1e12),
        (-3e7, 3e7 + 0.5),
        (1e200, -1e200),
        (1.2e308, 1.2e308),
    ] {
        let z = c(x, y);
        // erf(conj z) = conj erf(z), erf(−z) = −erf(z), bit for bit (zeros too).
        let (a, b) = (erf(c(x, -y)), erf(z));
        assert_eq!(bits(a), bits(c(b.re, -b.im)), "{x} {y}");
        let (a, b) = (erf(c(-x, -y)), erf(z));
        assert_eq!(bits(a), bits(c(-b.re, -b.im)), "{x} {y}");
        // erf + erfc = 1 (where erf is not large).
        let (e, f) = (erf(z), erfc(z));
        let n = e.re.hypot(e.im).max(1.0);
        if n < 1e300 {
            assert!(
                (e.re + f.re - 1.0).abs() < 1e-14 * n && (e.im + f.im).abs() < 1e-14 * n,
                "{x} {y}"
            );
        }
        // erfcx(z) = w(iz).
        let (a, b) = (erfcx(z), faddeeva(c(-y, x)));
        assert_eq!(bits(a), bits(b), "{x} {y}");
        // w(−conj z) = conj w(z) in the upper half-plane.
        if y > 0.0 {
            let (a, b) = (faddeeva(c(-x, y)), faddeeva(z));
            assert_eq!(bits(a), bits(c(b.re, -b.im)), "{x} {y}");
        }
    }
    // With infinite parts too (the zero parts with the signs of i/(√π z)): w(−conj z) = conj w(z)
    // everywhere but the one NaN, erfcx(conj z) = conj erfcx(z), erf(−z) = −erf(z).
    let inf = f64::INFINITY;
    // erfcx(z) = w(iz) at NaN ± 0i too: (NaN, ∓0) both (the sign of a NaN is not specified).
    for y in [0.0f64, -0.0] {
        let (a, b) = (erfcx(c(f64::NAN, y)), faddeeva(c(-y, f64::NAN)));
        assert!(a.re.is_nan() && b.re.is_nan(), "{y}");
        assert_eq!(a.im.to_bits(), b.im.to_bits(), "erfcx(NaN + {y}i)");
        assert_eq!(a.im.to_bits(), (-y).to_bits(), "erfcx(NaN + {y}i)");
    }
    for &(x, y) in &[
        (inf, 1.0f64),
        (inf, -1.0),
        (inf, 0.0),
        (inf, inf),
        (2.0, inf),
        (0.5, -inf),
        (inf, -inf),
        (-3.0, 1e300),
    ] {
        let (a, b) = (faddeeva(c(-x, y)), faddeeva(c(x, y)));
        if !(a.re.is_nan() || b.re.is_nan()) {
            assert_eq!(bits(a), bits(c(b.re, -b.im)), "w at {x} {y}");
        }
        let (a, b) = (erfcx(c(x, -y)), erfcx(c(x, y)));
        if !(a.re.is_nan() || b.re.is_nan()) {
            assert_eq!(bits(a), bits(c(b.re, -b.im)), "erfcx at {x} {y}");
        }
        let (a, b) = (erf(c(-x, -y)), erf(c(x, y)));
        if !(a.re.is_nan() || b.re.is_nan()) {
            assert_eq!(bits(a), bits(c(-b.re, -b.im)), "erf at {x} {y}");
        }
    }
}
