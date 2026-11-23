//! The small kernels the zeta area shares: fused multiply-adds where the hardware has them,
//! polynomial evaluation, `eˣ` and `ln x` in double-double (to about `2⁻⁷⁵`, and to `2⁻¹⁰⁰` in
//! the slower `_acc` forms the zeta function's phases at large `t` need), `atan` of a
//! double-double, and scalings by powers of two that over- and underflow correctly for any
//! exponent.
//!
//! The tables are printed by `gen_util.py` in this directory (with `gen_fit.py`), but
//! `ATAN_EIGHTHS` and `STIRLING`, which `gen_complex.py` prints.

use crate::math::special::dd::Dd;

/// Whether the target fuses multiply-adds in hardware; elsewhere `mul_add` is a library call.
const FMA: bool = cfg!(any(target_arch = "aarch64", target_feature = "fma"));

/// `a·b + c`, rounded once where the target fuses multiply-adds.
#[inline(always)]
pub(in crate::math::special) fn mla(a: f64, b: f64, c: f64) -> f64 {
    if FMA { a.mul_add(b, c) } else { a * b + c }
}

/// `Σ c[k] uᵏ` by Horner's rule.
#[inline(always)]
pub(in crate::math::special) fn horner(u: f64, c: &[f64]) -> f64 {
    let mut s = c[c.len() - 1];
    for &k in c[..c.len() - 1].iter().rev() {
        s = mla(s, u, k);
    }
    s
}

/// `Σ c[k] uᵏ` with the terms from `k = 1` on: `c[0]` is added last, so that a leading
/// coefficient that dominates is rounded once. Two Horner chains, the even and the odd
/// coefficients, for the latency.
#[inline(always)]
pub(in crate::math::special) fn poly_tail(u: f64, c: &[f64]) -> f64 {
    let n = c.len();
    if n <= 8 {
        return u * horner(u, &c[1..]);
    }
    let u2 = u * u;
    // c[1] u + c[2] u² + … = u (c[1] + c[3] u² + …) + u² (c[2] + c[4] u² + …).
    let mut odd = 0.0;
    let mut even = 0.0;
    let mut k = n - 1;
    loop {
        if k % 2 == 1 {
            odd = mla(odd, u2, c[k]);
        } else {
            even = mla(even, u2, c[k]);
        }
        if k == 1 {
            break;
        }
        k -= 1;
    }
    mla(u, odd, u2 * even)
}

/// `a + b = s + e` exactly.
#[inline(always)]
pub(in crate::math::special) fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    (s, (a - (s - bb)) + (b - bb))
}

/// `a·b = p + e` exactly (barring overflow and underflow): by a fused multiply-add where the
/// hardware has one, else by Dekker's product of Veltkamp's halves.
#[inline(always)]
pub(in crate::math::special) fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    if FMA && !cfg!(miri) {
        (p, a.mul_add(b, -p))
    } else {
        // Veltkamp's split overflows past 2^996: the large factor scaled by 2^-64 there
        let big = |x: f64| x.abs() > 1e290;
        let (a2, b2, k) = if big(a) {
            (
                a * 5.421_010_862_427_522e-20,
                b,
                18_446_744_073_709_551_616.0,
            )
        } else if big(b) {
            (
                a,
                b * 5.421_010_862_427_522e-20,
                18_446_744_073_709_551_616.0,
            )
        } else {
            (a, b, 1.0)
        };
        let split = |x: f64| {
            let t = 134_217_729.0 * x; // 2^27 + 1
            let hi = t - (t - x);
            (hi, x - hi)
        };
        let p2 = a2 * b2;
        let (ah, al) = split(a2);
        let (bh, bl) = split(b2);
        (p, (((ah * bh - p2) + ah * bl + al * bh) + al * bl) * k)
    }
}

/// `c − a·b` with the product exact (`c` near `a·b`, so that the difference is exact too): the
/// residual of a quotient or a square root, by a fused multiply-add where the hardware has one,
/// else by Dekker's product.
#[inline(always)]
pub(in crate::math::special) fn residual(a: f64, b: f64, c: f64) -> f64 {
    if FMA && !cfg!(miri) {
        (-a).mul_add(b, c)
    } else {
        let (p, pe) = two_prod(a, b);
        (c - p) - pe
    }
}

/// `π` as a double-double.
pub(in crate::math::special) const PI_HI: f64 = core::f64::consts::PI;
pub(in crate::math::special) const PI_LO: f64 = 1.224_646_799_147_353_2e-16;

/// `sin(πr)/r − π` in `r²`, minimax on `|r| ≤ 1/4` (relative error `2·10⁻²¹`).
const SIN_PI: [f64; 8] = [
    core::f64::consts::PI,
    -5.16771278004997,
    2.5501640398773415,
    -0.5992645293202866,
    0.08214588657954405,
    -0.007370429872719095,
    0.00046628259019530056,
    -2.171676546847857e-05,
];

/// `(cos(πr) − 1)/r²` in `r²`, minimax on `|r| ≤ 1/4` (relative error `2·10⁻²²`).
const COS_PI: [f64; 8] = [
    -4.934802200544679,
    4.0587121264167685,
    -1.3352627688545888,
    0.23533063035880594,
    -0.025806891384544024,
    0.001929574123091674,
    -0.00010463458716991511,
    4.268451776919289e-06,
];

/// `sin(πr)` for `|r| ≤ 1/4`.
#[inline(always)]
fn sin_pi_kernel(r: f64) -> f64 {
    let z = r * r;
    r * (PI_HI + (PI_LO + poly_tail(z, &SIN_PI)))
}

/// `cos(πr)` for `|r| ≤ 1/4`.
#[inline(always)]
fn cos_pi_kernel(r: f64) -> f64 {
    let z = r * r;
    1.0 + z * horner(z, &COS_PI)
}

/// `(sin(πx), cos(πx))`, the argument reduced exactly: exact zeros at the integers (`sin`) and
/// the half-integers (`cos`), and an error of about half a unit elsewhere. NaN for `±∞`.
#[inline]
pub(in crate::math::special) fn sincos_pi(x: f64) -> (f64, f64) {
    if !x.is_finite() {
        return (f64::NAN, f64::NAN);
    }
    // x = n/2 + r, |r| ≤ 1/4, exactly.
    let n = (2.0 * x).round();
    let r = x - 0.5 * n;
    // Past 2⁶² every n is a multiple of 4.
    let k = if n.abs() < 4.0e18 { (n as i64) & 3 } else { 0 };
    let (s, c) = (sin_pi_kernel(r), cos_pi_kernel(r));
    match k {
        0 => (s, c),
        1 => (c, -s),
        2 => (-s, -c),
        _ => (-c, s),
    }
}

/// `2^(j/32)` as double-doubles, `j = 0, …, 31`.
const EXP2_32: [(f64, f64); 32] = [
    (1.0, 0.0),
    (1.0218971486541166, 5.109225028973444e-17),
    (1.0442737824274138, 8.551889705537965e-17),
    (1.0671404006768237, -7.899853966841582e-17),
    (1.0905077326652577, -3.046782079812471e-17),
    (1.1143867425958924, 1.0410278456845571e-16),
    (1.1387886347566916, 8.912812676025408e-17),
    (1.1637248587775775, 3.8292048369240935e-17),
    (1.189207115002721, 3.982015231465646e-17),
    (1.215247359980469, -7.712630692681488e-17),
    (1.241857812073484, 4.658027591836937e-17),
    (1.2690509571917332, 2.667932131342186e-18),
    (1.2968395546510096, 2.5382502794888315e-17),
    (1.3252366431597413, -2.8587312100388614e-17),
    (1.3542555469368927, 7.70094837980299e-17),
    (1.383909881963832, -6.770511658794786e-17),
    (core::f64::consts::SQRT_2, -9.667293313452913e-17),
    (1.4451808069770467, -3.0237581349939873e-17),
    (1.4768261459394993, -3.483994556892796e-17),
    (1.5091644275934228, -1.016455327754295e-16),
    (1.5422108254079407, 7.949834809697621e-17),
    (1.5759808451078865, -1.0136916471278304e-17),
    (1.6104903319492543, 2.4707192569797888e-17),
    (1.645755478153965, -1.0125679913674773e-16),
    (1.681792830507429, 8.199010020581497e-17),
    (1.718619298122478, -1.851380418263111e-17),
    (1.7562521603732995, 2.960140695448873e-17),
    (1.7947090750031072, 1.8227458427912087e-17),
    (1.8340080864093424, 3.283107224245627e-17),
    (1.8741676341103, -6.122763413004143e-17),
    (1.9152065613971474, -1.0619946056195963e-16),
    (1.9571441241754002, 8.960767791036668e-17),
];

/// `ln 2 / 32` in two parts, the first of 36 bits so that `k` times it is exact for `|k| < 2¹⁷`.
const LN2_32_HI: f64 = 0.021_660_849_392_446_835;
const LN2_32_LO: f64 = 5.145_609_244_655_338e-14;

/// `eˣ = 2^q (e + el)` to about `2⁻⁷⁴` relative: a table of `2^(j/32)` as double-doubles and a
/// polynomial of degree 9 on `|r| ≤ ln 2/64`, its quadratic part exact (the cubic part, below
/// `2.2·10⁻⁷`, is rounded in double; [`exp_parts_acc`] is the slower form good to `2⁻¹⁰²`). For
/// `|x| ≤ 760` (past it the reduction is no longer exact: [`exp_mul`] splits such an `x`).
#[inline]
pub(in crate::math::special) fn exp_parts(x: f64) -> (f64, f64, i32) {
    debug_assert!(
        x.abs() <= 761.0 || x.is_nan(),
        "exp_parts({x}) is outside |x| ≤ 760"
    );
    let kf = (x * 46.166_241_308_446_83).round();
    let k = kf as i64;
    // kf·ln2/32ʰⁱ is exact (36 bits by 17), and so is the difference
    let r_hi = x - kf * LN2_32_HI;
    let r = r_hi - kf * LN2_32_LO;
    let r_lo = (r_hi - r) - kf * LN2_32_LO;
    // eʳ − 1 = r + r²/2 + r³ Q(r) as p + pl: r² exact, the cubic part in double (it is below
    // 2.2·10⁻⁷), and the low part of r to first order.
    let (r2, r2e) = two_prod(r, r);
    let q = r2
        * r
        * horner(
            r,
            &[
                0.16666666666666666,
                0.041666666666666664,
                0.008333333333333333,
                0.001388888888888889,
                0.0001984126984126984,
                2.48015873015873e-05,
                2.7557319223985893e-06,
            ],
        );
    let p = r + 0.5 * r2;
    let pl = ((r - p) + 0.5 * r2) + (0.5 * r2e + q + r_lo * (1.0 + r));
    let (t, tl) = EXP2_32[(k & 31) as usize];
    // t (1 + p + pl) + tl (1 + p)
    let (m, me) = two_prod(t, p);
    let (e, ee) = two_sum(t, m);
    // normalised: |el| ≤ ulp(e)/2
    let s = ee + (me + t * pl + tl * (1.0 + p));
    let h = e + s;
    (h, s - (h - e), (k >> 5) as i32)
}

/// `eˣ · (g + gl)`, rounded once: the product as a double-double and the scaling by a power
/// of two last, so that the result overflows and underflows (to subnormals) correctly, for any
/// `x` (a large `g` may bring an `eˣ` below the range back into it, and a small one an `eˣ`
/// above it).
#[inline]
pub(in crate::math::special) fn exp_mul(x: f64, g: f64, gl: f64) -> f64 {
    if x.abs() > 760.0 || x.is_nan() {
        return exp_mul_far(x, g, gl, 0);
    }
    let (e, el, q) = exp_parts(x);
    let (m, me) = two_prod(e, g);
    scale(m + (me + (e * gl + el * g)), q)
}

/// `eˣ · (g + gl) · 2ᵏ`, rounded once: [`exp_mul`] with a scaling by a power of two inside, so
/// that a result in the range is not lost to an overflow or an underflow of `eˣ g` alone.
#[inline]
pub(in crate::math::special) fn exp_mul_k(x: f64, g: f64, gl: f64, k: i32) -> f64 {
    if x.abs() > 760.0 || x.is_nan() {
        return exp_mul_far(x, g, gl, k);
    }
    let (e, el, q) = exp_parts(x);
    let (m, me) = two_prod(e, g);
    scale(m + (me + (e * gl + el * g)), q + k)
}

/// `e^(x + xl)` as a double-double `(h, l)` (`xl` the low part of the exponent, to first order):
/// from [`exp_parts`], to about `2⁻⁷⁴` relative, for `|x| ≤ 708`; past it [`exp_mul`] rounded once
/// (an infinity, or a result so small that its low part is lost to the subnormals anyway).
#[inline]
pub(in crate::math::special) fn exp_dd(x: f64, xl: f64) -> (f64, f64) {
    if x.abs() > 708.0 || x.is_nan() {
        return (exp_mul(x, 1.0, xl), 0.0);
    }
    let (e, el, q) = exp_parts(x);
    (scale(e, q), scale(el + e * xl, q))
}

/// [`exp_mul_k`] past `|x| = 760`: `eˣ = e^(x − jc) (e^c)ʲ`, `c = ±750`, the products as
/// double-doubles and one scaling.
#[cold]
#[inline(never)]
fn exp_mul_far(x: f64, g: f64, gl: f64, k: i32) -> f64 {
    if x.is_nan() || g.is_nan() {
        return f64::NAN;
    }
    if g == 0.0 {
        return g;
    }
    if g.is_infinite() {
        return if x == f64::NEG_INFINITY { f64::NAN } else { g };
    }
    // No finite g (below 2¹⁰²⁴ < e^710, at least 2⁻¹⁰⁷⁴ > e^−745) brings e^±1500 back.
    let xk = x + k as f64 * core::f64::consts::LN_2;
    if xk > 1500.0 {
        return f64::INFINITY.copysign(g);
    }
    if xk < -1500.0 {
        return 0.0f64.copysign(g);
    }
    // e^c as many times as it takes (|x| ≤ 1500 + |k| ln 2), each x − c exact (Sterbenz)
    let (mut p, mut pl, mut q) = (1.0, 0.0, k);
    let mut x = x;
    while x.abs() > 760.0 {
        let c = 750.0f64.copysign(x);
        let (e1, el1, q1) = exp_parts(c);
        let (h, he) = two_prod(p, e1);
        let (h, l) = two_sum(h, he + (p * el1 + pl * e1));
        (p, pl, q) = (h, l, q + q1);
        x -= c;
    }
    let (e2, el2, q2) = exp_parts(x);
    let (pp, ppe) = two_prod(p, e2);
    let ppl = ppe + (p * el2 + pl * e2);
    let (m, me) = two_prod(pp, g);
    scale(m + (me + (pp * gl + ppl * g)), q + q2)
}

/// `ln x` as a double-double, for a finite `x > 0`: `l = ln x` corrected by
/// `x e⁻ˡ − 1` (with `e⁻ˡ` from [`exp_parts`], whose cubic part in double leaves an absolute
/// error of about `2⁻⁷⁵`; [`ln_dd_acc`] is the slower form good to `2⁻¹⁰⁰`).
#[inline]
pub(in crate::math::special) fn ln_dd(x: f64) -> (f64, f64) {
    let l = x.ln();
    let (e, el, q) = exp_parts(-l);
    let xs = scale(x, q);
    let (m, me) = two_prod(e, xs);
    (l, (m - 1.0) + (me + el * xs))
}

/// The third part of `ln 2/32` (`LN2_32_HI + LN2_32_LO + LN2_32_LO2` to about `2⁻¹⁶⁰`).
const LN2_32_LO2: f64 = -3.149_359_104_970_357_4e-30;

/// `1/(k + 1)!` for `k = 0, …, 5` as double-doubles, and for `k = 6, …, 10` as doubles.
const EXP_DD: [(f64, f64); 6] = [
    (1.0, 0.0),
    (0.5, 0.0),
    (0.16666666666666666, 9.25185853854297e-18),
    (0.041666666666666664, 2.3129646346357427e-18),
    (0.008333333333333333, 1.1564823173178714e-19),
    (0.001388888888888889, -5.300543954373577e-20),
];
const EXP_TAIL: [f64; 5] = [
    0.0001984126984126984,
    2.48015873015873e-05,
    2.7557319223985893e-06,
    2.755731922398589e-07,
    2.505210838544172e-08,
];

/// `eˣ = 2^q (e + el)` to about `2⁻¹⁰²`, for `|x| ≤ 700`: [`exp_parts`] with the reduction
/// carried in three parts and `e^r − 1 = r Σ rᵏ/(k + 1)!` to `k = 10`, its terms to the fifth in
/// double-double (for the phases `t ln n` of the zeta function at large `t`).
pub(in crate::math::special) fn exp_parts_acc(x: f64) -> (f64, f64, i32) {
    let kf = (x * 46.166_241_308_446_83).round();
    let k = kf as i64;
    // r = x − kf·ln2/32 as a double-double: kf·HI exact, kf·LO by an exact product
    let r_hi = x - kf * LN2_32_HI;
    let (p, pe) = two_prod(kf, LN2_32_LO);
    let (r, re) = two_sum(r_hi, -p);
    let r = Dd::new(r).add_f64(re - pe - kf * LN2_32_LO2);
    let mut s = Dd::new(horner(r.hi, &EXP_TAIL));
    for &(h, l) in EXP_DD.iter().rev() {
        s = s.mul(r).add(Dd { hi: h, lo: l });
    }
    // e^r = 1 + r s, times 2^(j/32)
    let er = r.mul(s).add_f64(1.0);
    let (t, tl) = EXP2_32[(k & 31) as usize];
    let e = er.mul(Dd { hi: t, lo: tl });
    (e.hi, e.lo, (k >> 5) as i32)
}

/// `ln x` as a double-double to about `2⁻¹⁰⁰`, for a finite `x > 0`: [`ln_dd`] with `e⁻ˡ` from
/// [`exp_parts_acc`].
pub(in crate::math::special) fn ln_dd_acc(x: f64) -> (f64, f64) {
    let l = x.ln();
    let (e, el, q) = exp_parts_acc(-l);
    let xs = scale(x, q);
    // x e⁻ˡ − 1 = δ, ln x = l + δ − δ²/2 + …, δ about 2⁻⁵³
    let d = Dd { hi: e, lo: el }.mul_f64(xs).add_f64(-1.0);
    let d = d.value();
    (l, d - 0.5 * d * d)
}

/// `y · 2^q`, rounded once, for any `q` (a zero or an infinity of the sign of `y` past the
/// range).
#[inline(always)]
pub(in crate::math::special) fn scale(y: f64, q: i32) -> f64 {
    let pow2 = |e: i32| f64::from_bits(((e + 1023) as u64) << 52);
    if (-1000..=1000).contains(&q) {
        y * pow2(q)
    } else {
        scale_far(y, q)
    }
}

/// [`scale`] past `|q| = 1000`: `y = m·2ᵉ` with `1 ≤ |m| < 2` first, so that no step leaves the
/// range before the last.
#[cold]
#[inline(never)]
fn scale_far(y: f64, q: i32) -> f64 {
    let pow2 = |e: i32| f64::from_bits(((e + 1023) as u64) << 52);
    if y == 0.0 || !y.is_finite() {
        return y;
    }
    // a subnormal y normalised first
    let (y, e0) = if y.abs() < f64::MIN_POSITIVE {
        (y * pow2(64), -64i64)
    } else {
        (y, 0)
    };
    let b = y.to_bits();
    let e = ((b >> 52) & 0x7ff) as i64 - 1023 + e0;
    let m = f64::from_bits((b & !(0x7ffu64 << 52)) | (1023u64 << 52));
    let total = e + q as i64;
    if total > 1023 {
        f64::INFINITY.copysign(y)
    } else if total >= -1022 {
        m * pow2(total as i32)
    } else if total >= -1080 {
        // into the subnormals: the first factor is exact, the second rounds once
        (m * pow2(-1000)) * pow2(total as i32 + 1000)
    } else {
        0.0f64.copysign(y)
    }
}

/// `atan(k/8)`, `k = 0, …, 8`, as double-doubles.
const ATAN_EIGHTHS: [(f64, f64); 9] = [
    (0.0, 0.0),
    (0.12435499454676144, -3.1253241424539383e-18),
    (0.24497866312686414, 1.0698755618734451e-17),
    (0.35877067027057225, -2.4623815582638635e-17),
    (0.4636476090008061, 2.2698777452961687e-17),
    (0.5585993153435624, -5.4556305485916264e-18),
    (0.6435011087932844, 1.5834785051444286e-17),
    (0.7188299996216245, -2.1478388444456983e-17),
    (core::f64::consts::FRAC_PI_4, 3.061616997868383e-17),
];

/// `B₂ₖ/(2k(2k − 1))`, `k = 1, …, 10`: Stirling's series of `ln Γ`.
pub(in crate::math::special) const STIRLING: [f64; 10] = [
    0.08333333333333333,
    -0.002777777777777778,
    0.0007936507936507937,
    -0.0005952380952380953,
    0.0008417508417508417,
    -0.0019175269175269176,
    0.00641025641025641,
    -0.029550653594771242,
    0.17964437236883057,
    -1.3924322169059011,
];

/// `π/2` as a double-double.
const PI_2_DD: Dd = Dd {
    hi: core::f64::consts::FRAC_PI_2,
    lo: 6.123_233_995_736_766e-17,
};

/// `atan r` of a double-double: `atan(k/8) + atan(δ)`, `δ = (r − k/8)/(1 + r k/8)` on `[0, 1]`
/// (`|δ| ≤ 1/16`), `π/2 − atan(1/r)` past 1; `atan δ = δ (1 − δ²/3 + δ⁴ P(δ²))` with `δ` and
/// `δ²/3` in double-double and `P` in double, so that `atan δ` is good to about `2⁻⁷⁰` of
/// itself at `|δ| = 1/16` (`δ⁴ P` is below `2⁻¹⁸` of it; far better for smaller `δ`), and the
/// series' truncation is below `2⁻¹⁰⁸`.
pub(in crate::math::special) fn atan_dd(r: Dd) -> Dd {
    let neg = r.hi < 0.0;
    let x = if neg { r.neg() } else { r };
    let inv = x.hi > 1.0;
    let x = if inv { Dd::new(1.0).div(x) } else { x };
    let k = (8.0 * x.hi).round();
    let c = 0.125 * k;
    let d = x.add_f64(-c).div(x.mul_f64(c).add_f64(1.0));
    let y = d.mul(d);
    // P(y) = 1/5 − y/7 + y²/9 − …, to y¹⁰ (y ≤ 2⁻⁸: the rest is below 2⁻¹⁰⁸ of atan δ; to y⁷
    // it was 2⁻⁸⁴)
    let p = horner(
        -y.hi,
        &[
            0.2,
            1.0 / 7.0,
            1.0 / 9.0,
            1.0 / 11.0,
            1.0 / 13.0,
            1.0 / 15.0,
            1.0 / 17.0,
            1.0 / 19.0,
            1.0 / 21.0,
            1.0 / 23.0,
            1.0 / 25.0,
        ],
    );
    // 1 − y/3 + y² P = 1 + y (−1/3 + y P)
    let q = y.mul_f64(p).add(Dd {
        hi: -1.0 / 3.0,
        lo: -1.850_371_707_708_594e-17,
    });
    let t = d.mul(y.mul(q).add_f64(1.0));
    let (th, tl) = ATAN_EIGHTHS[k as usize];
    let a = Dd { hi: th, lo: tl }.add(t);
    let a = if inv { PI_2_DD.sub(a) } else { a };
    if neg { a.neg() } else { a }
}
