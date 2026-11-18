//! The exponential integrals.
//!
//! * `E₁(x)`, `0 < x ≤ 1`: `F(x) − ln x` with the entire `F = E₁(x) + ln x = −γ + x − x²/4 + …` a
//!   polynomial minimax fit with a double-double head (no cancellation: `F ≥ 0` where `ln x`
//!   nears 0). `1 < x ≤ 2.5`: `e⁻ˣ (H₀ + K(x))/x` with `K` a rational fit; `x > 2.5`:
//!   `e⁻ˣ (1 + P(1/x)/Q(1/x))/x`, the rational of degree 10/11 of Boost.Math (`expint.hpp`,
//!   53-bit, Boost Software License 1.0); `e⁻ˣ` to double-double accuracy and the product
//!   rounded once.
//! * `Ei(x)`, `0 < x ≤ 6`: `ln(x/x₀) + (x − x₀) R(x)` about the zero `x₀ = 0.3725…` of `Ei`, the
//!   entire `R` a polynomial minimax fit on each of four pieces, `x − x₀` as a double-double and
//!   `ln(x/x₀)` by `ln_1p` near `x₀`; `x > 6`: `eˣ (1 + K(x))/x` with `K` a rational fit on each
//!   of four pieces to 48 and a polynomial in `1/x` on two past it; `x < 0`: `−E₁(−x)`.
//! * `Eₙ(x)`, `n ≥ 2`: for `x ≤ 2` and `n < 50`, the recurrence `Eₖ₊₁ = (e⁻ˣ − x Eₖ)/k` up from
//!   `E₁` (stable there); else the continued fraction of `Eₙ`, its depth found by Lentz's method
//!   and its value from the bottom up in double-double arithmetic.
//!
//! The fits but Boost's were made for this crate (Remez exchange, and Lawson's iteration for the
//! rationals, mpmath at 128 bits).
//!
//! The tables but Boost's are printed by `zeta/gen_expint.py` (with `zeta/gen_fit.py`).

use super::zeta::util::{
    exp_mul, exp_parts, horner, ln_dd, mla, poly_tail, residual, scale, two_prod, two_sum,
};
use super::{Real, via};

/// E₁(x) + ln x on (0, 1]: degree 14 on [0, 1] in t − 0, error 6.0e-23 (5.8e-19 in doubles).
const E1_SMALL: [f64; 15] = [
    -0.5772156649015329,
    1.0,
    -0.25,
    0.0555555555555555,
    -0.01041666666666574,
    0.0016666666666580166,
    -0.00023148148142876547,
    2.8344670981422386e-05,
    -3.1001977611777963e-06,
    3.0619105129526205e-07,
    -2.7555193879328794e-08,
    2.2751238331387002e-09,
    -1.7216292087534021e-10,
    1.1411592215924614e-11,
    -5.160395889591241e-13,
];
const E1_SMALL_LO: f64 = 4.94297481997456e-18;

/// (Ei(x) − ln(x/x₀))/(x − x₀): degree 12 on [0, 1.5] in t − 0.75, error 6.9e-18 (1.8e-17 in doubles).
const EI_R1: [f64; 13] = [
    1.3444410732887826,
    0.38382815009612264,
    0.09170058338964479,
    0.01799571948719568,
    0.002971073989703034,
    0.0004222210658771982,
    5.261350823550205e-05,
    5.834605375860289e-06,
    5.827310722078529e-07,
    5.292990451935347e-08,
    4.408625761480789e-09,
    3.419545421566401e-10,
    2.4291194561754334e-11,
];
const EI_R1_LO: f64 = -1.447411855851298e-17;

/// (Ei(x) − ln(x/x₀))/(x − x₀): degree 13 on [1.5, 3] in t − 2.25, error 3.7e-19 (1.3e-17 in doubles).
const EI_R2: [f64; 14] = [
    2.2062087147302805,
    0.8341541972759887,
    0.23219087479877454,
    0.050216528067688296,
    0.008872616832158299,
    0.001325937968583732,
    0.00017178250046311582,
    1.9650878308991166e-05,
    2.0131257179289523e-06,
    1.867756167913022e-07,
    1.5837232418854933e-08,
    1.2367857828089116e-09,
    9.025904247977965e-11,
    6.0469622413823245e-12,
];
const EI_R2_LO: f64 = -1.8941512069264095e-16;

/// (Ei(x) − ln(x/x₀))/(x − x₀): degree 13 on [3, 4.5] in t − 3.75, error 7.5e-19 (2.0e-17 in doubles).
const EI_R3: [f64; 14] = [
    4.206688996717705,
    2.032750436980439,
    0.6396526322942202,
    0.149436405519419,
    0.02787436589777165,
    0.0043383648761282905,
    0.0005802244453233335,
    6.810072874763457e-05,
    7.126211346129699e-06,
    6.730839210810414e-07,
    5.7951049603195455e-08,
    4.585909992495313e-09,
    3.3860950308714447e-10,
    2.2873626103983822e-11,
];
const EI_R3_LO: f64 = -2.4569734499876495e-16;

/// (Ei(x) − ln(x/x₀))/(x − x₀): degree 13 on [4.5, 6] in t − 5.25, error 1.3e-18 (4.6e-17 in doubles).
const EI_R4: [f64; 14] = [
    9.381424362588437,
    5.4795442140240675,
    1.8925257536517777,
    0.46934347756540123,
    0.09137790801521019,
    0.014694801861493531,
    0.0020170883360672524,
    0.0002418358950876175,
    2.576056626327005e-05,
    2.4702446176940237e-06,
    2.154775479216984e-07,
    1.7247332560991194e-08,
    1.2864774593571718e-09,
    8.751720547566387e-11,
];
const EI_R4_LO: f64 = 3.564849093755043e-16;

/// x e⁻ˣ Ei(x) − 1 on [6, 10] in u = (x − 8)/2: numerator (u 6 10 c=8.0 h=2.0 7 7 7.02e-18 2.17e-17)
const EI_K1_P: [f64; 8] = [
    0.18184798698720797,
    0.0997621507953273,
    0.03282756157959268,
    0.007176861012563044,
    0.0007003286557098008,
    8.584184725127461e-05,
    1.294326029574128e-07,
    7.859978832187504e-08,
];

/// Its denominator.
const EI_K1_Q: [f64; 8] = [
    1.0,
    0.9238272625737608,
    0.4019400528235068,
    0.10597922569715629,
    0.018314987393667714,
    0.0020678868967899575,
    0.0001402779170662248,
    4.109435971790956e-06,
];

/// x e⁻ˣ Ei(x) − 1 on [10, 20] in u = (x − 15)/5: numerator (u 10 20 c=15.0 h=5.0 8 8 6.17e-18 3.87e-17)
const EI_K2_P: [f64; 9] = [
    0.07810310738616064,
    0.13375238305285364,
    0.11297098429608435,
    0.05905360558946569,
    0.02075175839344994,
    0.004894840789156828,
    0.000727426065542015,
    5.3778777393626515e-05,
    2.933280428246536e-09,
];

/// Its denominator.
const EI_K2_Q: [f64; 9] = [
    1.0,
    2.111314428327374,
    2.1247578851189326,
    1.3273100547791483,
    0.5636833416528872,
    0.1671103512237959,
    0.03392118754262716,
    0.004333866797583724,
    0.0002692123491473454,
];

/// x e⁻ˣ Ei(x) − 1 on [20, 30] in u = (x − 25)/5: numerator (u 20 30 7 7 5.63e-20 2.12e-17)
const EI_K3_P: [f64; 8] = [
    0.04366193626661325,
    0.07155548253903724,
    0.05207610616950055,
    0.0216556260458408,
    0.005426888345082539,
    0.0007756997742267238,
    4.895582135963935e-05,
    1.8161055167369244e-11,
];

/// Its denominator.
const EI_K3_Q: [f64; 8] = [
    1.0,
    1.85820381678355,
    1.551933086593327,
    0.7572443474954816,
    0.23286809118207227,
    0.044956036128706314,
    0.005004469909326457,
    0.00024478099297212927,
];

/// x e⁻ˣ Ei(x) − 1 on [30, 48] in u = (x − 39)/9: numerator (u 30 48 7 7 4.09e-22 5.55e-17)
const EI_K4_P: [f64; 8] = [
    0.02706905348794879,
    0.05429052411223543,
    0.04712686180972114,
    0.02247384425370071,
    0.006156099810262076,
    0.0009099982884847485,
    5.6213415292461145e-05,
    3.239047770468901e-14,
];

/// Its denominator.
const EI_K4_Q: [f64; 8] = [
    1.0,
    2.2496564244697272,
    2.23030645778342,
    1.2549025542235814,
    0.4298866470926823,
    0.08906234339716292,
    0.01026988091953594,
    0.0005059207434708701,
];

/// (x e⁻ˣ Ei(x) − 1)·x in t = 1/x on [1/128, 1/48] (t 0.0078125 0.020833333333333332 10 8.07e-18 1.69e-18)
const EI_T1: [f64; 11] = [
    1.0000000000206153,
    1.999999982629825,
    6.000006543902763,
    23.998546681315084,
    120.21106413946201,
    699.0063995700389,
    6493.285138642058,
    -29183.38541444361,
    2580702.7993993144,
    -39815834.962054595,
    459883222.2880361,
];

/// (x e⁻ˣ Ei(x) − 1)·x in t = 1/x on [1/717, 1/128] (t 0.001394700139470014 0.0078125 7 4.9e-17 7.78e-19)
const EI_T2: [f64; 8] = [
    0.9999999999999678,
    2.0000000000777134,
    5.999999922838292,
    24.00004128452903,
    119.98695459388344,
    722.496056877744,
    4758.347464069249,
    57157.19941203625,
];

/// `x eˣ E₁(x) − H₀` on [1, 2.5] in `u = (x − 1.75)/0.75`, `H₀` its value at 1.75: numerator
/// of the rational 6/6, error 9.29e-18 relative to `x eˣ E₁(x)` (Lawson's iteration).
const E1_MID_P: [f64; 7] = [
    2.5653749302979253e-18,
    0.07475202640456856,
    0.09671347307086288,
    0.04781618967357025,
    0.011203996140949397,
    0.0012326092416137138,
    5.040428885625026e-05,
];
/// Its denominator.
const E1_MID_Q: [f64; 7] = [
    1.0,
    1.5642326915900775,
    0.9845314796441281,
    0.3176953208096724,
    0.055140859123402094,
    0.004851421069812081,
    0.0001678990319029312,
];
/// `H₀ = 1.75 e^1.75 E₁(1.75)` as a double-double.
const E1_MID_H0: f64 = 0.699789598161452;
const E1_MID_H0_LO: f64 = 1.6538127354567527e-17;

/// Boost.Math's numerator for `E₁` past 1, in `1/x`.
const E1_LARGE_P: [f64; 11] = [
    -1.2101319065772557e-19,
    -0.9999999999999988,
    -43.3058660811818,
    -724.5814827914625,
    -6046.825011271104,
    -27182.625446673395,
    -66598.26523454186,
    -86273.15677116495,
    -54844.45872264021,
    -14751.489578612845,
    -1185.4572031520104,
];

/// Its denominator.
const E1_LARGE_Q: [f64; 12] = [
    1.0,
    45.30586608118015,
    809.1932149545503,
    7417.376244546896,
    38129.55944848185,
    113057.05869159632,
    192104.04779022798,
    180329.4983805018,
    86722.34034673347,
    18455.412473772205,
    1229.2078418240305,
    -0.776491285282331,
];

/// The zero `x₀` of `Ei` as a triple-double (for the relative accuracy of `x − x₀` at the
/// doubles nearest it), and `ln x₀`.
const EI_X0_HI: f64 = 0.372_507_410_781_366_6;
const EI_X0_LO: f64 = 1.314_018_341_438_602_8e-17;
const EI_X0_LO2: f64 = 6.472_568_844_595_414_5e-34;
const LN_EI_X0_HI: f64 = -0.987_498_346_645_341_9;
const LN_EI_X0_LO: f64 = 1.050_470_761_430_787_5e-17;

/// `c[0] + lo + Σ_{k≥1} c[k] tᵏ`.
#[inline(always)]
fn full(t: f64, c: &[f64], lo: f64) -> f64 {
    c[0] + (lo + poly_tail(t, c))
}

/// `E₁(x)` of an `f64`.
pub(crate) fn e1_f64(x: f64) -> f64 {
    if x.is_nan() || x <= 0.0 {
        return if x == 0.0 { f64::INFINITY } else { f64::NAN };
    }
    if x <= 1.0 {
        // F(x) − ln x, F's head −γ added to −ln x exactly.
        let (h, l) = two_sum(E1_SMALL[0], -x.ln());
        return h + (l + (E1_SMALL_LO + poly_tail(x, &E1_SMALL)));
    }
    if x > 746.0 {
        return 0.0;
    }
    let (h, hl) = if x <= 2.5 {
        let u = (x - 1.75) / 0.75;
        let k = horner(u, &E1_MID_P) / horner(u, &E1_MID_Q);
        two_sum(E1_MID_H0, k + E1_MID_H0_LO)
    } else {
        let t = 1.0 / x;
        two_sum(1.0, horner(t, &E1_LARGE_P) / horner(t, &E1_LARGE_Q))
    };
    let (g, gl) = div(h, hl, x);
    exp_mul(-x, g, gl)
}

/// `(h + hl)/x` as a double-double.
#[inline(always)]
fn div(h: f64, hl: f64, x: f64) -> (f64, f64) {
    let q = h / x;
    let r = residual(q, x, h) + hl;
    (q, r / x)
}

/// `Ei(x)` of an `f64`.
pub(crate) fn ei_f64(x: f64) -> f64 {
    if x.is_nan() || x <= 0.0 {
        return if x == 0.0 {
            f64::NEG_INFINITY
        } else if x < 0.0 {
            -e1_f64(-x)
        } else {
            x
        };
    }
    if x <= 6.0 {
        // d + dl = x − x₀ (x − x₀ʰⁱ is exact near x₀).
        let (d, dl) = two_sum(x - EI_X0_HI, -EI_X0_LO);
        let dl = dl - EI_X0_LO2;
        let r = if x <= 1.5 {
            full(x - 0.75, &EI_R1, EI_R1_LO)
        } else if x <= 3.0 {
            full(x - 2.25, &EI_R2, EI_R2_LO)
        } else if x <= 4.5 {
            full(x - 3.75, &EI_R3, EI_R3_LO)
        } else {
            full(x - 5.25, &EI_R4, EI_R4_LO)
        };
        let l = if d.abs() < 0.1 {
            // ln(1 + t), t = d/x₀ as a double-double.
            let t = d / EI_X0_HI;
            let tl = (residual(t, EI_X0_HI, d) + dl - t * EI_X0_LO) / EI_X0_HI;
            // ln(1 + t + tl) = l − (eˡ − 1 − t − tl)/(1 + t), eˡ to double-double accuracy
            let l = t.ln_1p();
            let (m, ml, q) = exp_parts(l);
            let (m, ml) = (scale(m, q), scale(ml, q));
            l - (((m - 1.0) - t) + (ml - tl)) / (1.0 + t)
        } else {
            (x.ln() - LN_EI_X0_HI) - LN_EI_X0_LO
        };
        return mla(d, r, l + dl * r);
    }
    ei_large(x)
}

/// `Ei(x)` for `x > 6`: `eˣ (1 + K)/x`.
#[inline]
fn ei_large(x: f64) -> f64 {
    let rat = |u: f64, p: &[f64], q: &[f64]| horner(u, p) / horner(u, q);
    let k = if x <= 10.0 {
        rat((x - 8.0) * 0.5, &EI_K1_P, &EI_K1_Q)
    } else if x <= 20.0 {
        rat((x - 15.0) / 5.0, &EI_K2_P, &EI_K2_Q)
    } else if x <= 30.0 {
        rat((x - 25.0) / 5.0, &EI_K3_P, &EI_K3_Q)
    } else if x <= 48.0 {
        rat((x - 39.0) / 9.0, &EI_K4_P, &EI_K4_Q)
    } else {
        if x > 717.0 {
            return f64::INFINITY;
        }
        let t = 1.0 / x;
        t * horner(t, if x <= 128.0 { &EI_T1 } else { &EI_T2 })
    };
    let (h, hl) = two_sum(1.0, k);
    let (g, gl) = div(h, hl, x);
    exp_mul(x, g, gl)
}

/// `E₁(x)` as a double-double for `0 < x ≤ 2.5`: `F(x) − ln x` with `F` by Horner's rule in
/// double-double arithmetic and `ln x` a double-double, or `e⁻ˣ (H₀ + K)/x` with `e⁻ˣ` and the
/// quotient as double-doubles.
fn e1_dd(x: f64) -> (f64, f64) {
    if x <= 1.0 {
        let c = &E1_SMALL;
        let (mut h, mut l) = (c[c.len() - 1], 0.0);
        for &ck in c[..c.len() - 1].iter().rev() {
            let (p, pe) = two_prod(h, x);
            let (s, se) = two_sum(p, ck);
            h = s;
            l = se + pe + l * x;
        }
        l += E1_SMALL_LO;
        let (lx, lxl) = ln_dd(x);
        let (s, se) = two_sum(h, -lx);
        return two_sum(s, se + l - lxl);
    }
    let u = (x - 1.75) / 0.75;
    let k = horner(u, &E1_MID_P) / horner(u, &E1_MID_Q);
    let (h, hl) = two_sum(E1_MID_H0, k + E1_MID_H0_LO);
    let (g, gl) = div(h, hl, x);
    let (e, el, q) = exp_parts(-x);
    let (e, el) = (scale(e, q), scale(el, q));
    let (m, me) = two_prod(e, g);
    two_sum(m, me + e * gl + el * g)
}

/// `Eₙ(x)` of an `f64`, `n ≥ 2`, `x > 0`.
fn en_f64(n: u32, x: f64) -> f64 {
    if n < 50 && x <= 2.0 {
        // Up from E₁ in double-double arithmetic: Eₖ₊₁ = (e⁻ˣ − x Eₖ)/k, whose error shrinks by
        // x/k at each step but whose difference cancels by up to 2.5 at the first ones.
        let (e, el, q) = exp_parts(-x);
        let (e, el) = (scale(e, q), scale(el, q));
        let (mut v, mut vl) = e1_dd(x);
        for k in 1..n {
            let (p, pe) = two_prod(x, v);
            let (s, se) = two_sum(e, -p);
            let sl = se + el - pe - x * vl;
            let kf = k as f64;
            let qv = s / kf;
            let r = residual(qv, kf, s) + sl;
            v = qv;
            vl = r / kf;
        }
        return v + vl;
    }
    if x > 746.0 {
        return 0.0;
    }
    // Eₙ(x) = e⁻ˣ / (b₀ + a₁/(b₁ + a₂/(b₂ + …))), bₖ = x + n + 2k, aₖ = −k(n − 1 + k): the
    // depth by Lentz's method, then the value from the bottom up, which is stable.
    let nf = n as f64;
    let tiny = 1e-300;
    let mut b = x + nf;
    let mut c = 1.0 / tiny;
    let mut d = 1.0 / b;
    let mut i = 1.0;
    loop {
        let a = -i * (nf - 1.0 + i);
        b += 2.0;
        d = 1.0 / mla(a, d, b);
        c = b + a / c;
        if (c * d - 1.0).abs() <= f64::EPSILON || i >= 5000.0 {
            break;
        }
        i += 1.0;
    }
    // From the bottom up in double-double arithmetic, f = bₖ₋₁ + aₖ/f, a quarter deeper than
    // Lentz's test stopped, for the slowly converging fractions of small x.
    i = (1.25 * i).ceil() + 4.0;
    let (mut f, mut fl) = two_sum(x, nf + 2.0 * (i + 1.0));
    while i >= 1.0 {
        let a = -i * (nf - 1.0 + i);
        let q = a / f;
        let ql = (residual(q, f, a) - q * fl) / f;
        let (bh, bl) = two_sum(x, nf + 2.0 * (i - 1.0));
        let (s, e) = two_sum(bh, q);
        f = s;
        fl = e + (bl + ql);
        i -= 1.0;
    }
    let g = 1.0 / f;
    let gl = (residual(g, f, 1.0) - g * fl) * g;
    exp_mul(-x, g, gl)
}

/// The exponential integral `E₁(x) = ∫₁^∞ e^(−xt)/t dt`, for `x > 0` (`∞` at 0).
pub fn expint_e1<T: Real>(x: T) -> T {
    via(x, e1_f64)
}

/// The exponential integral `Ei(x) = −PV ∫₋ₓ^∞ e^(−t)/t dt`, for `x ≠ 0` (`−∞` at 0).
pub fn expint_ei<T: Real>(x: T) -> T {
    via(x, ei_f64)
}

/// The generalised exponential integral `Eₙ(x) = ∫₁^∞ e^(−xt)/tⁿ dt`, for `x ≥ 0` (`x > 0` when
/// `n ≤ 1`).
pub fn expint_en<T: Real>(n: u32, x: T) -> T {
    via(x, |x| {
        if x.is_nan() || x < 0.0 {
            return f64::NAN;
        }
        match n {
            0 => {
                if x == 0.0 {
                    f64::INFINITY
                } else if x > 746.0 {
                    0.0
                } else {
                    // e⁻ˣ · (1/x as a double-double), rounded once
                    let (g, gl) = div(1.0, 0.0, x);
                    exp_mul(-x, g, gl)
                }
            }
            1 => e1_f64(x),
            _ if x == 0.0 => 1.0 / (n - 1) as f64,
            _ if x == f64::INFINITY => 0.0,
            _ => en_f64(n, x),
        }
    })
}
