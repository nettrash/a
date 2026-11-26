//! The Riemann zeta function of a real argument.
//!
//! * `s ≥ 56`: `1 + 2⁻ˢ`.
//! * `3 ≤ s < 56`: `1 + 2⁻ˢ V(s)` with `V = (ζ − 1)·2ˢ` a rational minimax fit on each of
//!   `[3, 8]`, `[8, 20]` and `[20, 56]`, weighted relative to `ζ`.
//! * `−1/2 ≤ s < 3`: `1/(s − 1) + R(s)` with the entire `R = ζ(s) − 1/(s − 1)` a polynomial
//!   minimax fit; the reciprocal, the head `c₀ + c₁(s − 1)` of `R` (the low part of `s − 1`
//!   through `R′`) and the sum are double-doubles, so that the cancellation for `s < 1` costs
//!   nothing.
//! * `s < −1/2`: the reflection `ζ(s) = 2 (2π)^(s−1) sin(πs/2) Γ(1−s) ζ(1−s)`, `1 − s` taken
//!   exactly as a double-double (its low part through `ζ′/ζ` and `ψ − ln 2π`), `(2π)^(s−1)` from
//!   a double-double logarithm, and past the range of `Γ` (`1 − s > 171`)
//!   `ln Γ(1 − s) − (1 − s) ln 2π` in double-double (Stirling's series) as one exponent. The
//!   trivial zeros are exact.
//!
//! The fits were made for this crate (Remez exchange, and Lawson's iteration for the
//! rationals, mpmath at 128 bits).
//!
//! The tables are printed by `gen_real.py` in this directory (with `gen_fit.py`).

use super::util::{
    STIRLING, exp_mul, horner, ln_dd, ln_dd_acc, mla, poly_tail, residual, sincos_pi, two_prod,
    two_sum,
};
use crate::math::special::dd::Dd;
use crate::math::special::gamma::gamma_f64;

/// ζ(s) − 1/(s − 1): degree 16 on [-0.5, 3] in t − 1, error 1.4e-20 (6.9e-18 in doubles).
const ZETA_R: [f64; 17] = [
    0.5772156649015329,
    0.07281584548367673,
    -0.004845181596436159,
    -0.0003423057367172266,
    9.68904193944704e-05,
    -6.611031810831047e-06,
    -3.316240908750168e-07,
    1.0462094582024842e-07,
    -8.733218097960473e-09,
    9.478280648205899e-11,
    5.658421389554015e-11,
    -6.768708708429717e-12,
    3.4921664147382184e-13,
    4.417341885927632e-15,
    -2.4021553052832717e-15,
    2.1549995570398375e-16,
    -8.998911878613857e-18,
];
const ZETA_R_LO: f64 = -4.944548762355127e-18;

/// (ζ(s) − 1)·2ˢ on [3, 8] in s − 5.5: numerator of the rational 6/6, error 4.66e-19 relative to ζ
const ZETA_V1_P: [f64; 7] = [
    1.1406290817866738,
    0.3429763116287159,
    0.04335620908746577,
    0.00306304231205039,
    0.00013874524696037564,
    4.219656463022448e-06,
    6.792404812576828e-08,
];

/// Its denominator.
const ZETA_V1_Q: [f64; 7] = [
    1.0,
    0.36249894518442416,
    0.04244674593985986,
    0.003060778896081972,
    0.00014066466200279094,
    4.141716131207618e-06,
    6.90594226958289e-08,
];

/// (ζ(s) − 1)·2ˢ on [8, 20] in s − 14.0: numerator of the rational 6/6, error 1.03e-18 relative to ζ
const ZETA_V2_P: [f64; 7] = [
    1.0034894448018248,
    0.20039672858833513,
    0.017577190840991103,
    0.0008940900295402814,
    2.818694421753771e-05,
    5.112174243163122e-07,
    3.970695108801994e-09,
];

/// Its denominator.
const ZETA_V2_Q: [f64; 7] = [
    1.0,
    0.2011288487535676,
    0.017506992130634974,
    0.0008980660448198362,
    2.804493791013228e-05,
    5.14296258983444e-07,
    3.938423141157005e-09,
];

/// (ζ(s) − 1)·2ˢ on [20, 56] in s − 38.0: numerator of the rational 4/4, error 2.81e-19 relative to ζ
const ZETA_V3_P: [f64; 5] = [
    1.0000002615720294,
    0.13242306356575095,
    0.006874673199322519,
    0.0001654872966263011,
    1.554611705743463e-06,
];

/// Its denominator.
const ZETA_V3_Q: [f64; 5] = [
    1.0,
    0.1324230843876059,
    0.006874657775573783,
    0.0001654869041840329,
    1.5545233197119332e-06,
];

/// `ln 2π` as a double-double.
const LN_2PI_HI: f64 = 1.837_877_066_409_345_6;
const LN_2PI_LO: f64 = -7.756_588_316_134_483e-17;

/// `ζ(s)` for `s ≥ −1/2`.
#[inline]
pub(super) fn zeta_right(s: f64) -> f64 {
    if s >= 3.0 {
        if s >= 56.0 {
            return 1.0 + (-s).exp2();
        }
        let (p, q, c): (&[f64], &[f64], f64) = if s <= 8.0 {
            (&ZETA_V1_P, &ZETA_V1_Q, 5.5)
        } else if s <= 20.0 {
            (&ZETA_V2_P, &ZETA_V2_Q, 14.0)
        } else {
            (&ZETA_V3_P, &ZETA_V3_Q, 38.0)
        };
        let u = s - c;
        return 1.0 + (-s).exp2() * (horner(u, p) / horner(u, q));
    }
    if s == 1.0 {
        return f64::INFINITY;
    }
    // 1/(s − 1) as q + ql: d + dl = s − 1 exactly, q(d + dl) = 1 − e.
    let (d, dl) = two_sum(s, -1.0);
    let q = 1.0 / d;
    let e = residual(q, d, 1.0) - q * dl;
    let ql = q * e;
    // R = c₀ + c₁d + d² T(d) with c₁d exact and the low part of d through R′ ≈ c₁ + 2c₂d: left of
    // 0 the result is a third of 1/(s − 1), and rounding c₁d and dropping dl cost 1.7 ulps
    let (p1, p1e) = two_prod(ZETA_R[1], d);
    let rest = d * d * horner(d, &ZETA_R[2..]);
    let (h, l) = two_sum(q, ZETA_R[0]);
    let (h2, l2) = two_sum(h, p1);
    let low = ZETA_R_LO + p1e + rest + dl * (ZETA_R[1] + 2.0 * ZETA_R[2] * d);
    h2 + (l2 + (l + (ql + low)))
}

/// `R(s) = ζ(s) − 1/(s − 1)` for `−1/2 ≤ s ≤ 3` (entire).
#[inline]
pub(super) fn zeta_regular(s: f64) -> f64 {
    let d = s - 1.0;
    ZETA_R[0] + (ZETA_R_LO + poly_tail(d, &ZETA_R))
}

/// `ζ′(s)/ζ(s)` to a few digits, for `1.5 ≤ s ≤ 8`, by a central difference.
#[inline(never)]
fn zeta_log_derivative(s: f64) -> f64 {
    let h = 1.0 / 4096.0;
    (zeta_right(s + h) - zeta_right(s - h)) / (2.0 * h * zeta_right(s))
}

/// `ψ(z)` to a few digits, for `z > 0`.
pub(super) fn digamma_rough(z: f64) -> f64 {
    let (mut z, mut acc) = (z, 0.0);
    while z < 6.0 {
        acc -= 1.0 / z;
        z += 1.0;
    }
    let r = 1.0 / (z * z);
    acc + z.ln() - 0.5 / z - r * (1.0 / 12.0 - r * (1.0 / 120.0 - r / 252.0))
}

/// `ln Γ(z + zl)` of a double-double `z + zl > 0` as a double-double: from `z = 15` Stirling's
/// series, its main terms `(z − ½) ln z − z + ½ ln 2π` in double-double (`ln z` by `ln_dd`, whose
/// `2⁻⁷⁵` times `z` stays below `2⁻⁵⁵` to `z = 2²⁰`, then by `ln_dd_acc`) and eight terms of the
/// rest; below, the logarithm of `gamma_f64`; the low part through `ψ(z)`.
pub(super) fn lgamma_dd(z: f64, zl: f64) -> Dd {
    let base = if z >= 15.0 {
        let (l, ll) = if z < 1_048_576.0 {
            ln_dd(z)
        } else {
            ln_dd_acc(z)
        };
        let r = 1.0 / z;
        let r2 = r * r;
        let mut tail = STIRLING[7];
        for &b in STIRLING[..7].iter().rev() {
            tail = mla(tail, r2, b);
        }
        Dd::new(z)
            .add_f64(-0.5)
            .mul(Dd { hi: l, lo: ll })
            .add_f64(-z)
            .add(HALF_LN_2PI)
            .add_f64(tail * r)
    } else {
        let (l, ll) = ln_dd(gamma_f64(z).abs());
        Dd { hi: l, lo: ll }
    };
    if zl == 0.0 {
        base
    } else {
        base.add_f64(zl * digamma_rough(z))
    }
}

/// `½ ln 2π` as a double-double.
const HALF_LN_2PI: Dd = Dd {
    hi: 0.918_938_533_204_672_8,
    lo: -3.878_294_158_067_241_5e-17,
};

/// `e^(−z ln 2π) = (2π)^(−z)` with the logarithm as a double-double.
#[inline]
pub(super) fn pow_2pi_neg(z: f64) -> f64 {
    let (a, ae) = two_prod(-z, LN_2PI_HI);
    let a_lo = ae - z * LN_2PI_LO;
    a.exp() * (1.0 + a_lo)
}

/// `ζ(s)` for `s < −1/2` by the reflection formula.
#[cold]
fn zeta_reflect(s: f64) -> f64 {
    if s == f64::NEG_INFINITY {
        return f64::NAN;
    }
    let h = 0.5 * s;
    if h == h.floor() {
        // The trivial zeros, and every s past 2⁵³.
        return 0.0;
    }
    let (sin, _) = sincos_pi(h);
    if s < -280.0 {
        return f64::INFINITY.copysign(sin);
    }
    // z + zl = 1 − s exactly.
    let (z, zl) = two_sum(1.0, -s);
    let mut zeta1 = zeta_right(z);
    // Γ(z + zl)(2π)^(−z−zl) = Γ(z)(2π)^(−z) (1 + zl (ψ(z) − ln 2π)).
    let mut corr = 1.0;
    if zl != 0.0 {
        if z <= 8.0 {
            zeta1 *= 1.0 + zl * zeta_log_derivative(z);
        }
        corr += zl * (digamma_rough(z) - LN_2PI_HI);
    }
    // The product 2 sin(πs/2) ζ(1 − s) (1 + …) as a double-double, and Γ(…)(2π)^(−z) through
    // exp_mul from a double-double exponent, so that the result is rounded once.
    let (l, ll) = two_prod(2.0 * sin, zeta1);
    if z <= 171.0 {
        let (l, ll) = (l, ll + l * (corr - 1.0));
        let gz = gamma_f64(z);
        let (p, pe) = two_prod(l, gz);
        let (a, ae) = two_prod(-z, LN_2PI_HI);
        let a_lo = ae - z * LN_2PI_LO;
        let pl = pe + ll * gz;
        return exp_mul(a, p, pl + p * a_lo);
    }
    // past the range of Γ: ln Γ(z + zl) − (z + zl) ln 2π in double-double, one exponent
    let e = lgamma_dd(z, zl).sub(
        Dd {
            hi: LN_2PI_HI,
            lo: LN_2PI_LO,
        }
        .mul(Dd { hi: z, lo: zl }),
    );
    exp_mul(e.hi, l, ll + l * e.lo)
}

/// `ζ(s)` of an `f64`.
#[inline]
pub(crate) fn zeta_f64(s: f64) -> f64 {
    if s >= -0.5 {
        zeta_right(s)
    } else if s.is_nan() {
        s
    } else {
        zeta_reflect(s)
    }
}
