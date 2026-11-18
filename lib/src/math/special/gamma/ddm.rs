//! The elementary functions of the gamma area, in double-double where its functions need more
//! than `f64`: the logarithm, the exponential and `sin(πx)`, `cos(πx)`. They use no `std`
//! transcendental function, so their results are the same on every target.

use super::super::dd::{Dd, fast_two_sum, two_prod, two_sum};
use super::tab::*;

/// Whether `mul_add` is one fused instruction (elsewhere it is a slow call into the library, and
/// under Miri it need not be exact): the crate's switch, as in `dd.rs`.
pub(super) const FMA: bool = cfg!(all(
    any(target_arch = "aarch64", target_feature = "fma"),
    not(miri)
));

/// `a·b + c`, fused where the hardware has it: for terms whose rounding does not matter.
#[inline(always)]
pub(super) fn fma(a: f64, b: f64, c: f64) -> f64 {
    if FMA { a.mul_add(b, c) } else { a * b + c }
}

/// `c − a·b` exactly, where it is representable (a remainder): fused, or by Dekker's product.
#[inline(always)]
pub(super) fn rem(a: f64, b: f64, c: f64) -> f64 {
    if FMA {
        (-a).mul_add(b, c)
    } else {
        let (p, e) = two_prod(a, b);
        (c - p) - e
    }
}

/// `a·b − p` exactly for `p = a·b` rounded (the product's rounding error).
#[inline(always)]
pub(super) fn prod_err(a: f64, b: f64, p: f64) -> f64 {
    if FMA {
        a.mul_add(b, -p)
    } else {
        two_prod(a, b).1
    }
}

/// The polynomial `Σ c[i] x^i` of at most 16 coefficients, by Estrin's scheme (fully unrolled:
/// every condition below is a constant).
#[inline(always)]
pub(super) fn poly<const N: usize>(x: f64, c: &[f64; N]) -> f64 {
    const { assert!(N >= 1 && N <= 16) };
    // Through an opaque reference the coefficients are loaded in pairs, not built from immediates
    // (four or five instructions each on AArch64).
    let c = core::hint::black_box(c);
    if N <= 3 {
        let mut s = c[N - 1];
        let mut i = N - 1;
        while i > 0 {
            i -= 1;
            s = fma(s, x, c[i]);
        }
        return s;
    }
    let x2 = x * x;
    let x4 = x2 * x2;
    let x8 = x4 * x4;
    let mut a = [0.0; 8];
    for (i, ai) in a.iter_mut().enumerate() {
        if 2 * i + 1 < N {
            *ai = fma(x, c[2 * i + 1], c[2 * i]);
        } else if 2 * i < N {
            *ai = c[2 * i];
        }
    }
    let n1 = N.div_ceil(2);
    let mut b = [0.0; 4];
    for (i, bi) in b.iter_mut().enumerate() {
        if 2 * i + 1 < n1 {
            *bi = fma(x2, a[2 * i + 1], a[2 * i]);
        } else if 2 * i < n1 {
            *bi = a[2 * i];
        }
    }
    let n2 = n1.div_ceil(2);
    if n2 == 2 {
        return fma(x4, b[1], b[0]);
    }
    let lo = fma(x4, b[1], b[0]);
    let hi = if n2 == 4 { fma(x4, b[3], b[2]) } else { b[2] };
    fma(x8, hi, lo)
}

/// The rounding constant of `round_int`: `1.5·2⁵²`.
const SHIFT: f64 = 6_755_399_441_055_744.0;

/// `x` rounded to the nearest integer (ties to even), for `|x| < 2⁵¹`.
#[inline(always)]
pub(super) fn round_int(x: f64) -> f64 {
    (x + SHIFT) - SHIFT
}

/// The offset of the subintervals of the logarithm: `z` is reduced to `[OFF, 2·OFF)`.
const OFF: u64 = 0x3fe6_9555_0000_0000;

/// `ln x` as a double-double (relative error below `2⁻⁹⁰`), for finite `x > 0`.
#[inline]
pub(super) fn ln_dd(x: f64) -> Dd {
    let mut ix = x.to_bits();
    let mut k0 = 0i64;
    if ix < 0x0010_0000_0000_0000 {
        // A subnormal: normalised by 2⁵⁴.
        ix = (x * 18_014_398_509_481_984.0).to_bits();
        k0 = -54;
    }
    let tmp = ix.wrapping_sub(OFF);
    let i = ((tmp >> 46) & 63) as usize;
    let k = ((tmp as i64) >> 52) + k0;
    let z = f64::from_bits(ix.wrapping_sub(tmp & (0xfff << 52)));
    // r = z/c − 1, exact: c has 7 significant bits.
    let r = -rem(z, LOG_INVC[i], 1.0);
    // ln(1 + r) = r − r²/2 + r³ P(r).
    let (qh, ql) = two_prod(r, -0.5 * r);
    let tail = r * r * r * poly(r, &LOG1P_TAIL);
    let kf = k as f64;
    // |k ln 2| > |ln c| unless k = 0; |r| may exceed |s| here, but not r²/2 below.
    let (s, e1) = fast_two_sum(kf * LN2_HI, LOG_LOGC_HI[i]);
    let (s, e2) = two_sum(s, r);
    let (s, e3) = fast_two_sum(s, qh);
    let lo = ((e1 + e2) + (e3 + ql)) + (tail + fma(kf, LN2_LO, LOG_LOGC_LO[i]));
    let (hi, lo) = fast_two_sum(s, lo);
    Dd { hi, lo }
}

/// `ln(hi + lo)` of a positive double-double.
#[inline]
pub(super) fn ln_of_dd(x: Dd) -> Dd {
    ln_dd(x.hi).add_f64(x.lo / x.hi)
}

/// `e^x = d·2^m` with `d` a double-double near `[1, 2)` (relative error below `2⁻⁹⁰`), for
/// `|x.hi| < 2¹⁰ · ln 2`.
#[inline]
pub(super) fn exp_parts(x: Dd) -> (Dd, i32) {
    let kf = round_int(x.hi * EXP_INV_C);
    let k = kf as i64;
    // r = x − k·ln2/64, |r| ≤ ln2/128; the first difference is exact.
    let rh = x.hi - kf * EXP_C_HI;
    let rl = fma(-kf, EXP_C_LO, x.lo);
    let (rh, rl) = fast_two_sum(rh, rl);
    // e^r − 1 = r + r²/2 + r³ Q(r).
    let (qh, ql) = two_prod(rh, 0.5 * rh);
    let (ph, pe) = fast_two_sum(rh, qh);
    let pl = pe + ql + rl + rh * rl + rh * rh * rh * poly(rh, &EXP_TAIL);
    let j = (k & 63) as usize;
    let (th, tl) = (EXP_T_HI[j], EXP_T_LO[j]);
    // T (1 + p) = T + T p.
    let (a, ae) = two_prod(th, ph);
    let (s, se) = fast_two_sum(th, a);
    let lo = se + (ae + fma(th, pl, tl * (1.0 + ph)));
    let (hi, lo) = fast_two_sum(s, lo);
    (Dd { hi, lo }, (k >> 6) as i32)
}

/// `2^m` for `−1022 ≤ m ≤ 1023`.
#[inline(always)]
pub(super) fn pow2i(m: i32) -> f64 {
    f64::from_bits(((m + 1023) as u64) << 52)
}

/// `d·2^m` rounded to an `f64` (to ±∞ or zero beyond the range).
#[inline]
pub(super) fn scale(d: Dd, m: i32) -> f64 {
    let v = d.hi + d.lo;
    if (-1021..=1022).contains(&m) {
        v * pow2i(m)
    } else if m > 1022 {
        v * pow2i(1022) * pow2i((m - 1022).min(1023))
    } else {
        // Exact to a normal first, then rounded once into the subnormals.
        v * pow2i((m + 600).max(-1022)) * pow2i(-600)
    }
}

/// `e^x` of a double-double, rounded to an `f64`: ∞ above about 709.78, 0 below about −745.13.
#[inline]
pub(super) fn exp_f64(x: Dd) -> f64 {
    if x.hi.is_nan() {
        return x.hi;
    }
    if x.hi > 709.8 {
        return f64::INFINITY;
    }
    if x.hi < -745.2 {
        return 0.0;
    }
    let (d, m) = exp_parts(x);
    scale(d, m)
}

/// `e^x` of a double-double as a double-double, for `−708 < x < 709`.
#[inline]
pub(super) fn exp_dd(x: Dd) -> Dd {
    let (d, m) = exp_parts(x);
    let s = pow2i(m);
    Dd {
        hi: d.hi * s,
        lo: d.lo * s,
    }
}

/// `sin(πr)` of `|r| ≤ 1/4`, as a double-double (relative error about `2⁻⁶⁰`):
/// `πr − (π³/6) r³ + r·r²·P(r²)` with the first two terms in double-double.
#[inline(always)]
fn sin_pi_core(r: f64) -> Dd {
    let (uh, ul) = two_prod(r, r);
    let (ch, ce) = two_prod(r, uh);
    let cl = fma(r, ul, ce);
    let (ah, ae) = two_prod(PI3_6_HI, ch);
    let al = ae + fma(PI3_6_HI, cl, PI3_6_LO * ch);
    let (sh, sl) = two_prod(PI_HI, r);
    let (s, e) = fast_two_sum(sh, -ah);
    let lo = (e + fma(PI_LO, r, sl)) + (ch * poly(uh, &SINPI) - al);
    let (hi, lo) = fast_two_sum(s, lo);
    Dd { hi, lo }
}

/// `cos(πr)` of `|r| ≤ 1/4`, as a double-double (relative error about `2⁻⁵⁸`).
#[inline(always)]
fn cos_pi_core(r: f64) -> Dd {
    let (uh, ul) = two_prod(r, r);
    // v = π²r²/2.
    let (vh, ve) = two_prod(PI2_2_HI, uh);
    let vl = ve + fma(PI2_2_HI, ul, PI2_2_LO * uh);
    let tail = uh * uh * poly(uh, &COSPI);
    let (s, e) = fast_two_sum(1.0, -vh);
    let (hi, lo) = fast_two_sum(s, (e - vl) + tail);
    Dd { hi, lo }
}

/// The quadrant `q` (mod 4) and the reduced `r ∈ [−1/4, 1/4]` with `x ≡ q/2 + r (mod 2)`,
/// exactly, for finite `x`.
#[inline(always)]
fn reduce_pi(x: f64) -> (i64, f64) {
    // Beyond 2⁵⁰ the period is taken out first (exactly); `round_int` needs `|2x| < 2⁵¹`.
    let x = if x.abs() >= 1_125_899_906_842_624.0 {
        x % 2.0
    } else {
        x
    };
    let n = round_int(2.0 * x);
    (n as i64 & 3, x - 0.5 * n)
}

/// `sin(πx)` as a double-double, for finite `x` (`±0` at the integers, the sign of `x`).
#[inline]
pub(super) fn sin_pi_dd(x: f64) -> Dd {
    let (q, r) = reduce_pi(x);
    let v = match q {
        0 => sin_pi_core(r),
        1 => cos_pi_core(r),
        2 => sin_pi_core(-r),
        _ => cos_pi_core(r).neg(),
    };
    if v.hi == 0.0 {
        return Dd::new(0.0f64.copysign(x));
    }
    v
}

/// `cos(πx)` as a double-double, for finite `x` (`+0` at the half-integers).
#[inline]
pub(super) fn cos_pi_dd(x: f64) -> Dd {
    let (q, r) = reduce_pi(x);
    let v = match q {
        0 => cos_pi_core(r),
        1 => sin_pi_core(-r),
        2 => cos_pi_core(r).neg(),
        _ => sin_pi_core(r),
    };
    if v.hi == 0.0 { Dd::new(0.0) } else { v }
}

/// `cot(πx)` as a double-double, for a finite non-integer `x`: one reduction for both the sine
/// and the cosine, and one division.
#[inline]
pub(super) fn cot_pi_dd(x: f64) -> Dd {
    let (q, r) = reduce_pi(x);
    let (s, c) = (sin_pi_core(r), cos_pi_core(r));
    // cot(π(q/2 + r)) = cot(πr) for an even q, −tan(πr) for an odd one.
    let (n, d) = if q & 1 == 0 { (c, s) } else { (s.neg(), c) };
    let qh = n.hi / d.hi;
    let res = rem(qh, d.hi, n.hi) + fma(-qh, d.lo, n.lo);
    let (hi, lo) = fast_two_sum(qh, res / d.hi);
    Dd { hi, lo }
}
