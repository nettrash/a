//! The complex error functions through the Faddeeva function `w(z) = e^(−z²) erfc(−iz)`.
//!
//! In the closed upper half-plane, by the symmetry `w(−x + iy) = conj w(x + iy)`, `x ≥ 0`:
//!
//! * `|z| ≥ 10`: the asymptotic series `w(z) = i/(√π z)·Σ (2n − 1)!!/(2z²)ⁿ`, 14 terms;
//! * otherwise the trapezoidal rule of `w(z) = (i/π) ∫ e^(−t²)/(z − t) dt` with step `h = 1/2`
//!   on nodes shifted to `x + (n + ½)h`, so that no node meets the pole, plus its exact pole
//!   correction `2e^(−z²)/(1 + e^(2πy/h))` for `y < π/h`: 27 nodes, an error below 2⁻⁵⁴ of
//!   `|w|` (the aliasing error `e^(−π²/h²)`); every term is a real division;
//! * on the axes the real functions: `w(x) = e^(−x²) + 2i/√π·F(x)` with Dawson's `F`, and
//!   `w(iy) = erfcx(y)`.
//!
//! The lower half-plane is `w(z) = 2e^(−z²) − w(−z)`, with `e^(−z²)` from a double-double `z²`
//! (exact products), so that a large `|z|` costs no accuracy in the exponential.

use super::{
    Dd, dawson_f64, erf_f64, erfc_f64, erfcx_f64, exp_dd, poly, scale, scale_dd, two_prod,
};
use crate::math::complex::Complex;
use core::f64::consts::{FRAC_2_SQRT_PI, PI};

/// `1/√π`.
const FRAC_1_SQRT_PI: f64 = FRAC_2_SQRT_PI * 0.5;

/// `h/π·e^(−k²/4)` with `h = 1/2`, `k = 0..=13`.
static GAUSS: [f64; 14] = [
    0.159_154_943_091_895_35,
    0.123_949_994_309_652_96,
    0.058_549_831_524_319_16,
    0.016_774_807_587_073_417,
    0.002_915_024_465_028_193_5,
    0.000_307_241_318_192_835,
    1.964_128_034_639_743_7e-5,
    7.615_750_862_323_311e-7,
    1.791_052_932_828_018_5e-8,
    2.554_799_797_725_798e-10,
    2.210_334_915_491_785_8e-12,
    1.159_877_313_739_617_6e-14,
    3.691_635_240_477_673e-17,
    7.126_532_519_424_097e-20,
];

/// `4π` as a double-double.
const FOUR_PI: f64 = 12.566_370_614_359_172;
const FOUR_PI_LO: f64 = 4.898_587_196_589_413e-16;

/// `(2n − 1)!!/2ⁿ`, `n = 0..=13`.
static ASYM: [f64; 14] = [
    1.0,
    0.5,
    0.75,
    1.875,
    6.5625,
    29.531_25,
    162.421_875,
    1_055.742_187_5,
    7_918.066_406_25,
    67_303.564_453_125,
    639_383.862_304_687_5,
    6_713_530.554_199_219,
    77_205_601.373_291_02,
    965_070_017.166_137_7,
];

/// `2/√π·(−1)^k/(k!(2k + 1))`, `k = 0..=18`: the Taylor series of `erf`.
static TAYLOR: [f64; 19] = [
    FRAC_2_SQRT_PI,
    -0.376_126_389_031_837_54,
    0.112_837_916_709_551_26,
    -0.026_866_170_645_131_252,
    0.005_223_977_625_442_188,
    -0.000_854_832_702_345_085_3,
    0.000_120_553_329_817_896_64,
    -1.492_565_035_840_625e-5,
    1.646_211_436_588_924_8e-6,
    -1.636_584_469_123_492_4e-7,
    1.480_719_281_587_921_8e-8,
    -1.229_055_530_171_792_8e-9,
    9.422_759_064_650_411e-11,
    -6.711_366_855_164_110_5e-12,
    4.463_224_263_286_477_5e-13,
    -2.783_516_207_210_921_5e-14,
    1.634_261_409_536_715_2e-15,
    -9.063_970_842_808_673e-17,
    4.763_348_040_515_068e-18,
];

#[inline(always)]
fn c(re: f64, im: f64) -> Complex<f64> {
    Complex::new(re, im)
}

#[inline(always)]
fn cmul(a: Complex<f64>, b: Complex<f64>) -> Complex<f64> {
    c(
        a.re.mul_add(b.re, -(a.im * b.im)),
        a.re.mul_add(b.im, a.im * b.re),
    )
}

/// `1/z` without overflow in `|z|²` (Smith's method).
#[inline(always)]
fn recip(x: f64, y: f64) -> Complex<f64> {
    if x.abs() >= y.abs() {
        let r = y / x;
        let d = 1.0 / y.mul_add(r, x);
        c(d, -r * d)
    } else {
        let r = x / y;
        let d = 1.0 / x.mul_add(r, y);
        c(r * d, -d)
    }
}

/// `e^(−x²)` of a real `x`, to about 0.5 ulp.
#[inline(always)]
fn exp_neg_sq_real(x: f64) -> f64 {
    let (h, l) = two_prod(x, x);
    if h > 746.0 {
        return 0.0;
    }
    let (m, k) = exp_dd(-h, -l);
    scale_dd(m, k)
}

/// `e^(−z²) = m·2^k` of `z = x + iy`, `|m|` in about `[0.98, 2.03)`: `x² − y²` as a
/// double-double and `2xy` as one too, so that the phase and the size are exact to the last
/// bits for any `|z|`. `k` is clamped where the value is beyond any range.
#[inline(always)]
fn exp_neg_sq(x: f64, y: f64) -> (Complex<f64>, i64) {
    let (xh, xl) = two_prod(x, x);
    let (yh, yl) = two_prod(y, y);
    let a = Dd { hi: yh, lo: yl }.sub(Dd { hi: xh, lo: xl });
    let (m, k) = if a.hi.abs() < 1400.0 {
        exp_dd(a.hi, a.lo)
    } else {
        (Dd::new(1.0), if a.hi > 0.0 { 2040 } else { -2040 })
    };
    let (ph, pl) = two_prod(x, y);
    // θ = −2xy = −2(ph + pl).
    let (s, co) = sin_cos_dd(-2.0 * ph, -2.0 * pl);
    let mv = m.value();
    (c(mv * co, mv * s), k)
}

/// `(sin θ, cos θ)` of `θ = th + tl` (`|tl|` below an ulp of `th`): Cody and Waite's reduction
/// by a three-part `π/2` for `|θ| < 2²⁰` and fdlibm's kernels (under 1 ulp); beyond, the
/// standard library's reduction.
#[inline(always)]
fn sin_cos_dd(th: f64, tl: f64) -> (f64, f64) {
    if th.is_nan() || th.abs() >= 1_048_576.0 {
        let (s, c) = th.sin_cos();
        return (tl.mul_add(c, s), (-tl).mul_add(s, c));
    }
    // fdlibm's pio2_1 (33 bits), pio2_2 (33 bits) and pio2_2t.
    const P1: f64 = 1.570_796_326_734_125_6;
    const P2: f64 = 6.077_100_506_303_966e-11;
    const P2T: f64 = 2.022_266_248_795_950_6e-21;
    let z = th * core::f64::consts::FRAC_2_PI + super::ROUND;
    let k = (z.to_bits() as i64).wrapping_sub(super::ROUND.to_bits() as i64);
    let kf = z - super::ROUND;
    // th − k·P1 and then − k·P2 exactly; the rest rounded.
    let r = (-kf).mul_add(P1, th);
    let w = kf * P2;
    let rh = r - w;
    let rl = ((r - rh) - w) + (-kf).mul_add(P2T, tl);
    let (rh, rl) = super::fast_two_sum(rh, rl);
    let x2 = rh * rh;
    // __kernel_sin and __kernel_cos.
    let v = x2 * rh;
    let sr = poly(
        x2,
        &[
            8.333_333_333_322_49e-3,
            -1.984_126_982_985_795e-4,
            2.755_731_370_707_006_8e-6,
            -2.505_076_025_340_686_3e-8,
            1.589_690_995_211_55e-10,
        ],
    );
    let sn = rh - ((x2 * (0.5 * rl - v * sr) - rl) + v * 1.666_666_666_666_663_2e-1);
    let cr = x2
        * x2
        * poly(
            x2,
            &[
                4.166_666_666_666_66e-2,
                -1.388_888_888_887_411e-3,
                2.480_158_728_947_673e-5,
                -2.755_731_435_139_066_3e-7,
                2.087_572_321_298_175e-9,
                -1.135_964_755_778_819_5e-11,
            ],
        );
    let hz = 0.5 * x2;
    let wc = 1.0 - hz;
    let cs = wc + (((1.0 - wc) - hz) + (cr - rh * rl));
    match k & 3 {
        0 => (sn, cs),
        1 => (cs, -sn),
        2 => (-sn, -cs),
        _ => (-cs, sn),
    }
}

/// `v·2^k` of a complex `v`.
#[inline(always)]
fn cscale(v: Complex<f64>, k: i64) -> Complex<f64> {
    c(scale(v.re, k), scale(v.im, k))
}

/// `w(z)` for `x ≥ 0`, `y ≥ 0`, both finite.
fn w_quadrant(x: f64, y: f64) -> Complex<f64> {
    if y == 0.0 {
        return c(exp_neg_sq_real(x), FRAC_2_SQRT_PI * dawson_f64(x));
    }
    if x == 0.0 {
        return c(erfcx_f64(y), 0.0);
    }
    if x >= 10.0 || y >= 10.0 || x.mul_add(x, y * y) >= 100.0 {
        return w_asymptotic(x, y);
    }
    w_trapezoid(x, y)
}

/// `w(z)` for `|z| ≥ 10` in the closed first quadrant: the asymptotic series in `1/z²`.
#[inline(always)]
fn w_asymptotic(x: f64, y: f64) -> Complex<f64> {
    let q = recip(x, y);
    let v = cmul(q, q);
    let mut s = c(ASYM[13], 0.0);
    let mut i = 13;
    while i > 0 {
        i -= 1;
        s = cmul(s, v);
        s.re += ASYM[i];
    }
    // i/(√π z)·s.
    let p = cmul(q, s);
    let mut w = c(-FRAC_1_SQRT_PI * p.im, FRAC_1_SQRT_PI * p.re);
    if y < 1e-20 && x < 27.3 {
        // Near the real axis the term e^(−x²) of Re w is not in the series.
        w.re += exp_neg_sq_real(x);
    }
    w
}

/// `w(z)` for `x > 0`, `y > 0`, `|z| < 10`: the shifted trapezoidal rule.
#[inline(always)]
fn w_trapezoid(x: f64, y: f64) -> Complex<f64> {
    // Nodes t = x + s, s = (n + ½)/2; t₀ the node nearest 0, |t₀| ≤ 1/4.
    let n0 = (-2.0 * x - 0.5).round();
    let s0 = (n0 + 0.5) * 0.5;
    let t0 = x + s0;
    // e^(−(t₀ ± k/2)²) = e^(−t₀²)·(e^(∓t₀))^k·e^(−k²/4).
    // e^(−t₀²) and e^(∓t₀) = cosh t₀ ∓ sinh t₀ by their Taylor series (|t₀| ≤ 1/4: the
    // first neglected terms are below 2⁻⁶²).
    let u = t0 * t0;
    let e0 = poly(
        u,
        &[
            1.0,
            -1.0,
            0.5,
            -1.0 / 6.0,
            1.0 / 24.0,
            -1.0 / 120.0,
            1.0 / 720.0,
            -1.0 / 5040.0,
            1.0 / 40320.0,
            -1.0 / 362_880.0,
        ],
    );
    let ch = poly(
        u,
        &[
            1.0,
            0.5,
            1.0 / 24.0,
            1.0 / 720.0,
            1.0 / 40320.0,
            1.0 / 3_628_800.0,
            1.0 / 479_001_600.0,
        ],
    );
    let sh = t0
        * poly(
            u,
            &[
                1.0,
                1.0 / 6.0,
                1.0 / 120.0,
                1.0 / 5040.0,
                1.0 / 362_880.0,
                1.0 / 39_916_800.0,
                1.0 / 6_227_020_800.0,
            ],
        );
    let (p, ip) = (ch - sh, ch + sh);
    let y2 = y * y;
    let mut re = e0 * GAUSS[0] / s0.mul_add(s0, y2);
    let mut im = re * s0;
    let (mut pk, mut mk) = (e0, e0);
    let mut k = 1;
    while k <= 13 {
        pk *= p;
        mk *= ip;
        let g = GAUSS[k];
        let sp = s0 + 0.5 * k as f64;
        let sm = s0 - 0.5 * k as f64;
        // The pair over one division: (E₊D₋ + E₋D₊)/(D₊D₋), and so with the s.
        let (dp, dm) = (sp.mul_add(sp, y2), sm.mul_add(sm, y2));
        let (ep, em) = (pk * g, mk * g);
        let (a, b) = (ep * dm, em * dp);
        let r = 1.0 / (dp * dm);
        re = (a + b).mul_add(r, re);
        im = sp.mul_add(a, sm * b).mul_add(r, im);
        k += 1;
    }
    // (h/π)·(y·Σ E/D − i·Σ E·s/D), h/π in GAUSS.
    let mut w = c(y * re, -im);
    // The pole correction 2e^(−z²)/(1 + e^(4πy)) for y < 2π, its exponent
    // y² − x² − 4πy as a double-double.
    if y < 2.0 * PI && y.mul_add(y - FOUR_PI, -x * x) > -45.0 {
        let (xh, xl) = two_prod(x, x);
        let (yh, yl) = two_prod(y, y);
        let (ph, pl) = two_prod(FOUR_PI, y);
        let a = Dd { hi: yh, lo: yl }.sub(Dd { hi: xh, lo: xl }).sub(Dd {
            hi: ph,
            lo: FOUR_PI_LO.mul_add(y, pl),
        });
        let (m, k) = exp_dd(a.hi, a.lo);
        let (xy, xyl) = two_prod(x, y);
        let (s, co) = sin_cos_dd(2.0 * xy, 2.0 * xyl);
        let (e4, k4) = exp_dd(-ph, -FOUR_PI_LO * y);
        let g = 2.0 * scale(m.value(), k) / (1.0 + scale(e4.value(), k4));
        w.re = g.mul_add(co, w.re);
        w.im = (-g).mul_add(s, w.im);
    }
    w
}

/// `w(z)` for `y ≥ 0` (finite).
#[inline(always)]
fn w_upper(x: f64, y: f64) -> Complex<f64> {
    if x < 0.0 {
        let w = w_quadrant(-x, y);
        c(w.re, -w.im)
    } else {
        w_quadrant(x, y)
    }
}

/// `w(z)`.
pub(crate) fn faddeeva_c64(z: Complex<f64>) -> Complex<f64> {
    let (x, y) = (z.re, z.im);
    if !(x.is_finite() && y.is_finite()) {
        return if x.is_nan() || y.is_nan() || y < 0.0 {
            c(f64::NAN, f64::NAN)
        } else {
            // |z| → ∞ in the closed upper half-plane.
            c(0.0, 0.0)
        };
    }
    if y >= 0.0 {
        return w_upper(x, y);
    }
    // 2e^(−z²) − w(−z).
    let (e, k) = exp_neg_sq(x, y);
    let w = w_upper(-x, -y);
    let e = cscale(c(2.0 * e.re, 2.0 * e.im), k);
    c(e.re - w.re, e.im - w.im)
}

/// `e^(−z²)·w(iz) = erfc(z)` for `x ≥ 0` (finite).
#[inline(always)]
fn erfc_right(x: f64, y: f64) -> Complex<f64> {
    // iz = −y + ix is in the upper half-plane.
    let w = w_upper(-y, x);
    let (e, k) = exp_neg_sq(x, y);
    cscale(cmul(e, w), k)
}

/// `erf(z)` of a `Complex<f64>`.
pub(crate) fn erf_c64(z: Complex<f64>) -> Complex<f64> {
    let (x, y) = (z.re, z.im);
    if y == 0.0 {
        return c(erf_f64(x), y);
    }
    if x == 0.0 {
        return c(x, erfi(y));
    }
    if !(x.is_finite() && y.is_finite()) {
        return if x.is_infinite() && y.is_finite() {
            c(1.0f64.copysign(x), 0.0f64.copysign(y))
        } else {
            c(f64::NAN, f64::NAN)
        };
    }
    if x.mul_add(x, y * y) < 1.0 {
        // The Taylor series in z².
        let u = c((x - y) * (x + y), 2.0 * x * y);
        let mut s = c(TAYLOR[18], 0.0);
        let mut i = 18;
        while i > 0 {
            i -= 1;
            s = cmul(s, u);
            s.re += TAYLOR[i];
        }
        return cmul(z, s);
    }
    // erf is odd: 1 − erfc(z) for x ≥ 0.
    if x >= 0.0 {
        let e = erfc_right(x, y);
        c(1.0 - e.re, -e.im)
    } else {
        let e = erfc_right(-x, -y);
        c(e.re - 1.0, e.im)
    }
}

/// `erfc(z)` of a `Complex<f64>`.
pub(crate) fn erfc_c64(z: Complex<f64>) -> Complex<f64> {
    let (x, y) = (z.re, z.im);
    if y == 0.0 {
        return c(erfc_f64(x), -y);
    }
    if !(x.is_finite() && y.is_finite()) {
        return if x.is_infinite() && y.is_finite() {
            c(if x > 0.0 { 0.0 } else { 2.0 }, -0.0f64.copysign(y))
        } else {
            c(f64::NAN, f64::NAN)
        };
    }
    if x >= 0.0 {
        erfc_right(x, y)
    } else {
        // 2 − erfc(−z).
        let e = erfc_right(-x, -y);
        c(2.0 - e.re, -e.im)
    }
}

/// `erfcx(z) = e^(z²) erfc(z) = w(iz)` of a `Complex<f64>`.
pub(crate) fn erfcx_c64(z: Complex<f64>) -> Complex<f64> {
    let (x, y) = (z.re, z.im);
    if y == 0.0 {
        return c(erfcx_f64(x), -y);
    }
    if x == f64::INFINITY && y.is_finite() {
        return c(0.0, 0.0);
    }
    faddeeva_c64(c(-y, x))
}

/// `erfi(y) = −i·erf(iy) = 2/√π·e^(y²)·F(y)` of a real `y`.
#[inline(always)]
fn erfi(y: f64) -> f64 {
    let a = y.abs();
    if a.is_nan() || a >= 27.0 {
        return if a.is_nan() {
            y
        } else {
            f64::INFINITY.copysign(y)
        };
    }
    let (h, l) = two_prod(a, a);
    let (m, k) = exp_dd(h, l);
    let f = FRAC_2_SQRT_PI * dawson_f64(a);
    scale(m.value() * f, k).copysign(y)
}
