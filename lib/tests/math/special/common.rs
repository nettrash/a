//! The reference data of the special-function tests and their error measures.
//!
//! A data file (made by `data/gen_*.py` with mpmath at 256 bits) has one case a line:
//! `a₁ a₂ … : hi₁ lo₁ hi₂ lo₂ …`, every value the 16 hexadecimal digits of the bits of an `f64`.
//! The arguments are exact (for an `f32` file, `f32` values); each reference value is the
//! double-double `hi + lo` (a complex one its real part, then its imaginary part). A NaN
//! reference means the function must return NaN, an infinite one that infinity. Lines starting
//! with `#` are comments.

#![allow(dead_code)]

/// One case: the arguments and the reference values.
pub struct Case {
    pub args: Vec<f64>,
    pub refs: Vec<(f64, f64)>,
}

fn word(w: &str) -> f64 {
    f64::from_bits(u64::from_str_radix(w, 16).unwrap_or_else(|_| panic!("bad word {w:?}")))
}

/// The cases of a data file.
pub fn cases(text: &str) -> Vec<Case> {
    text.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(|l| {
            let (a, r) = l.split_once(':').expect("a line needs a ':'");
            let refs: Vec<f64> = r.split_whitespace().map(word).collect();
            assert!(
                refs.len().is_multiple_of(2),
                "references come in pairs: {l}"
            );
            Case {
                args: a.split_whitespace().map(word).collect(),
                refs: refs.chunks(2).map(|p| (p[0], p[1])).collect(),
            }
        })
        .collect()
}

/// The precision of a result type: its significand bits and the exponent of its least subnormal.
#[derive(Clone, Copy, Debug)]
pub struct Prec {
    pub bits: i32,
    pub min_exp: i32,
}

/// `f64`.
pub const F64: Prec = Prec {
    bits: 53,
    min_exp: -1074,
};

/// `f32`.
pub const F32: Prec = Prec {
    bits: 24,
    min_exp: -149,
};

/// `2^e`, for any `e` in `−1074..=1023`.
fn pow2(e: i32) -> f64 {
    if e >= -1022 {
        f64::from_bits(((e + 1023) as u64) << 52)
    } else {
        f64::from_bits(1u64 << (e + 1074))
    }
}

/// The unit in the last place of the value `hi` in the precision `p`.
pub fn ulp(hi: f64, p: Prec) -> f64 {
    let a = hi.abs();
    if a == 0.0 {
        return pow2(p.min_exp);
    }
    // ⌊log₂ a⌋ from the bits (a subnormal f64 normalised first).
    let e = if a < f64::MIN_POSITIVE {
        let b = (a * pow2(64)).to_bits();
        ((b >> 52) as i32) - 1023 - 64
    } else {
        ((a.to_bits() >> 52) as i32) - 1023
    };
    pow2((e - (p.bits - 1)).max(p.min_exp))
}

/// The error of `y` against the reference `hi + lo`, in ulps of the precision `p`: 0 when both
/// are the same NaN or infinity, infinite when only one is; a reference beyond the range of `p`
/// is that infinity.
pub fn ulps(y: f64, (hi, lo): (f64, f64), p: Prec) -> f64 {
    let max = if p.bits == 24 {
        f32::MAX as f64
    } else {
        f64::MAX
    };
    let hi = if hi.abs() > max {
        f64::INFINITY.copysign(hi)
    } else {
        hi
    };
    if hi.is_nan() || y.is_nan() {
        return if hi.is_nan() && y.is_nan() {
            0.0
        } else {
            f64::INFINITY
        };
    }
    if hi.is_infinite() || y.is_infinite() {
        return if y == hi { 0.0 } else { f64::INFINITY };
    }
    ((y - hi) - lo).abs() / ulp(hi, p)
}

/// The error of a complex `y = (re, im)` against the reference, relative to the reference's
/// absolute value, in units of `2^(1 − bits)` (ε).
pub fn rel_eps(y: (f64, f64), re: (f64, f64), im: (f64, f64), p: Prec) -> f64 {
    if !(re.0.is_finite() && im.0.is_finite()) || y.0.is_nan() || y.1.is_nan() {
        let same = |a: f64, b: f64| a == b || (a.is_nan() && b.is_nan());
        return if same(y.0, re.0) && same(y.1, im.0) {
            0.0
        } else {
            f64::INFINITY
        };
    }
    let dr = (y.0 - re.0) - re.1;
    let di = (y.1 - im.0) - im.1;
    let norm = re.0.hypot(im.0);
    let eps = pow2(1 - p.bits);
    if norm == 0.0 {
        return if dr == 0.0 && di == 0.0 {
            0.0
        } else {
            f64::INFINITY
        };
    }
    // Divided in two steps: `norm · ε` underflows for references below about 1e-292.
    dr.hypot(di) / norm / eps
}

/// The largest error of a set of cases and where it is.
#[derive(Debug)]
pub struct Worst {
    pub n: usize,
    pub max: f64,
    pub args: Vec<f64>,
    pub got: Vec<f64>,
    pub want: Vec<(f64, f64)>,
}

/// The largest error of `f` over the cases, measured by `err(got, refs)`.
pub fn worst(
    cases: &[Case],
    f: impl Fn(&[f64]) -> Vec<f64>,
    err: impl Fn(&[f64], &[(f64, f64)]) -> f64,
) -> Worst {
    let mut w = Worst {
        n: cases.len(),
        max: 0.0,
        args: Vec::new(),
        got: Vec::new(),
        want: Vec::new(),
    };
    for c in cases {
        let got = f(&c.args);
        let e = err(&got, &c.refs);
        if e > w.max || (e.is_nan() && !w.max.is_nan()) {
            w.max = e;
            w.args = c.args.clone();
            w.got = got;
            w.want = c.refs.clone();
        }
    }
    w
}

/// Checks a real function of the data `text` to `bound` ulps of the precision `p`, and returns
/// its largest error.
#[track_caller]
pub fn check_real(name: &str, text: &str, p: Prec, bound: f64, f: impl Fn(&[f64]) -> f64) -> f64 {
    let cases = cases(text);
    assert!(!cases.is_empty(), "{name}: no cases");
    let w = worst(&cases, |a| vec![f(a)], |g, r| ulps(g[0], r[0], p));
    assert!(
        w.max <= bound,
        "{name}: {:.3} ulps > {bound} at {:?} ({:?}): got {:e}, want {:e} + {:e}",
        w.max,
        w.args,
        w.args.iter().map(|a| a.to_bits()).collect::<Vec<_>>(),
        w.got[0],
        w.want[0].0,
        w.want[0].1
    );
    w.max
}

/// Checks a complex function of the data `text` (arguments `re im`, references `re` then `im`) to
/// `bound` ε relative to the reference's absolute value, and returns its largest error.
#[track_caller]
pub fn check_complex(
    name: &str,
    text: &str,
    p: Prec,
    bound: f64,
    f: impl Fn(f64, f64) -> (f64, f64),
) -> f64 {
    let cases = cases(text);
    assert!(!cases.is_empty(), "{name}: no cases");
    let w = worst(
        &cases,
        |a| {
            let (re, im) = f(a[0], a[1]);
            vec![re, im]
        },
        |g, r| rel_eps((g[0], g[1]), r[0], r[1], p),
    );
    assert!(
        w.max <= bound,
        "{name}: {:.3} ε > {bound} at {:?}: got {:?}, want {:?}",
        w.max,
        w.args,
        w.got,
        w.want
    );
    w.max
}
