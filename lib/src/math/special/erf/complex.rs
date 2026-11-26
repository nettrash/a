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
//!   `w(iy) = erfcx(y)` (on the whole imaginary axis).
//!
//! The lower half-plane is `w(z) = 2e^(−z²) − w(−z)`, with `e^(−z²)` from a double-double
//! `y² − x²` (exact products) and its phase `−2xy` reduced modulo `π/2` exactly, by Cody and
//! Waite's method below `2²⁰` and by Payne and Hanek's from the exact product of the significands
//! of `x` and `y` beyond, so that a large `|z|` costs no accuracy in the exponential. Its size is
//! kept as an exact power of 2 apart and applied once, saturating, so that a part far smaller
//! than `|w|` over- or underflows where it must and no sooner. Next to the imaginary axis below
//! the real one (and `erf`, `erfc` next to the imaginary axis) the parts come from the real
//! functions by the Taylor series in `Re z`, so that the small part is right on its own.

use super::{
    Dd, dawson_f64, erf_f64, erfc_f64, erfcx_f64, exp_dd, fast_two_sum, mul_dd, poly, pow2, scale,
    scale_dd, two_prod, two_sum,
};
use crate::math::complex::Complex;
use core::f64::consts::{FRAC_2_SQRT_PI, LN_2, PI};

/// `1/√π`.
const FRAC_1_SQRT_PI: f64 = FRAC_2_SQRT_PI * 0.5;

/// `2/√π` as a double-double.
const TWO_SQRT_PI: Dd = Dd {
    hi: FRAC_2_SQRT_PI,
    lo: 1.533_545_961_316_588e-17,
};

/// The exponent of `e^(∓z²)` beyond which every nonzero part of a result over- or underflows: a
/// part of the factor it multiplies is at least 2⁻¹⁰⁷⁴ and at most 4, and `e^1460 > 2²¹⁰⁶`.
/// [`exp_dd`] is exact in its power of 2 well beyond (to about 2839).
const EXP_MAX: f64 = 1460.0;

/// A power of 2 beyond any range: [`scale_sat`] sends every nonzero value to `±∞` with it and to
/// `±0` with its negative.
const BEYOND: i64 = 4096;

/// `|Re z|·max(|Im z|, 1)` below which `erf` and `erfc` are taken part by part from the Taylor
/// series in `Re z` about `i·Im z` ([`erf_near_imag`]).
const NEAR_IMAG: f64 = 6.103_515_625e-5; // 2⁻¹⁴

/// `|Re z|·max(−Im z, 1)` below which `w` below the real axis is taken from the first terms of the
/// Taylor series in `Re z` ([`w_near_neg_imag`]).
const NEAR_NEG_IMAG: f64 = 9.313_225_746_154_785e-10; // 2⁻³⁰

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

/// The bits of `2/π` after the binary point, most significant first, 64 to a word (2,304 bits):
/// enough to reduce the phase `2xy = m·2^e` of any two finite doubles (`m < 2¹⁰⁶`, its units
/// `2^e` below `2¹⁹⁴⁴`, the phase below `2²⁰⁴⁹`) with 254 bits beyond its units.
/// `⌊2/π·2²³⁰⁴⌋` from mpmath at 2,600 bits, the same words as Arb's `arb_const_pi` gives.
static TWO_OVER_PI: [u64; 36] = [
    0xa2f9_836e_4e44_1529,
    0xfc27_57d1_f534_ddc0,
    0xdb62_9599_3c43_9041,
    0xfe51_63ab_debb_c561,
    0xb724_6e3a_424d_d2e0,
    0x0649_2eea_09d1_921c,
    0xfe1d_eb1c_b129_a73e,
    0xe882_35f5_2ebb_4484,
    0xe99c_7026_b45f_7e41,
    0x3991_d639_8353_39f4,
    0x9c84_5f8b_bdf9_283b,
    0x1ff8_97ff_de05_980f,
    0xef2f_118b_5a0a_6d1f,
    0x6d36_7ecf_27cb_09b7,
    0x4f46_3f66_9e5f_ea2d,
    0x7527_bac7_ebe5_f17b,
    0x3d07_39f7_8a52_92ea,
    0x6bfb_5fb1_1f8d_5d08,
    0x5603_3046_fc7b_6bab,
    0xf0cf_bc20_9af4_361d,
    0xa9e3_9161_5ee6_1b08,
    0x6599_855f_14a0_6840,
    0x8dff_d880_4d73_2731,
    0x0606_1556_ca73_a8c9,
    0x60e2_7bc0_8c6b_47c4,
    0x19c3_67cd_dce8_092a,
    0x8359_c476_8b96_1ca6,
    0xddaf_44d1_5719_053e,
    0xa5ff_0705_3f7e_33e8,
    0x32c2_de4f_9832_7dbb,
    0xc33d_26ef_6b1e_5ef8,
    0x9f3a_1f35_caf2_7f1d,
    0x87f1_2190_7c7c_246a,
    0xfa6e_d577_2d30_433b,
    0x15c6_14b5_9d19_c3c2,
    0xc4ad_414d_2c5d_000c,
];

/// The bits `p + 1 ..= p + 64` of `2/π` (bit `k` the coefficient of `2^(−k)`, zero for `k ≤ 0`),
/// for `−∞ < p ≤ 2176`.
#[inline(always)]
fn two_over_pi_bits(p: i64) -> u64 {
    if p <= -64 {
        return 0;
    }
    if p < 0 {
        return TWO_OVER_PI[0] >> -p;
    }
    let (i, s) = ((p / 64) as usize, (p % 64) as u32);
    if s == 0 {
        TWO_OVER_PI[i]
    } else {
        (TWO_OVER_PI[i] << s) | (TWO_OVER_PI[i + 1] >> (64 - s))
    }
}

/// `(sin 2xy, cos 2xy)` for `|xy| ≥ 2¹⁹`, also where `2xy` overflows, by Payne and Hanek's
/// reduction: `2xy = m·2^e` exactly with `m < 2¹⁰⁶` the product of the two significands, and
/// `m·2^e·2/π` modulo 4 needs only the 256 bits of `2/π` from its `2¹` on (the bits above make
/// multiples of 4, those below change it by less than `m·2⁻²⁵⁴ < 2⁻¹⁴⁸`). The quadrant and the
/// fraction come from `m` times those bits modulo `2²⁵⁶`, in integers; then fdlibm's kernels.
#[cold]
#[inline(never)]
fn sin_cos_2xy(x: f64, y: f64) -> (f64, f64) {
    // The integer significand and the exponent of a finite double.
    let split = |v: f64| {
        let b = v.to_bits();
        let e = ((b >> 52) & 0x7ff) as i64;
        let f = b & ((1 << 52) - 1);
        if e == 0 {
            (f, -1074)
        } else {
            (f | (1 << 52), e - 1075)
        }
    };
    let ((mx, ex), (my, ey)) = (split(x), split(y));
    let m = mx as u128 * my as u128;
    let e = ex + ey + 1;
    // I = the bits e − 1 ..= e + 254 of 2/π, most significant first: m·2^e·2/π ≡ m·I·2⁻²⁵⁴ (mod 4).
    let i = [
        two_over_pi_bits(e + 190),
        two_over_pi_bits(e + 126),
        two_over_pi_bits(e + 62),
        two_over_pi_bits(e - 2),
    ];
    // r = m·I mod 2²⁵⁶, least significant word first.
    let mut r = [0u64; 4];
    for (j, mj) in [m as u64, (m >> 64) as u64].into_iter().enumerate() {
        let mut carry = 0u128;
        for k in 0..4 - j {
            // At most (2⁶⁴ − 1)² + 2(2⁶⁴ − 1) = 2¹²⁸ − 1.
            let t = mj as u128 * i[k] as u128 + r[j + k] as u128 + carry;
            r[j + k] = t as u64;
            carry = t >> 64;
        }
    }
    // The top two bits are the quadrant, the 254 below the fraction f, folded into [−1/2, 1/2).
    let mut q = (r[3] >> 62) as i64;
    let mut f = [
        r[3] << 2 | r[2] >> 62,
        r[2] << 2 | r[1] >> 62,
        r[1] << 2 | r[0] >> 62,
        r[0] << 2,
    ];
    let neg = f[0] >> 63 == 1;
    if neg {
        // 1 − f, in two's complement.
        q += 1;
        let mut carry = true;
        for w in f.iter_mut().rev() {
            let (v, c) = (!*w).overflowing_add(carry as u64);
            *w = v;
            carry = c;
        }
    }
    // |f| as a double-double from its 128 leading bits (f = 0 would need 2xy to be a multiple of
    // π/2 to 2⁻¹⁴⁸, which no product of two doubles is; it is kept as 0 all the same).
    let (fh, fl) = match f.iter().position(|&w| w != 0) {
        None => (0.0, 0.0),
        Some(lz) => {
            let (a, b, cw) = (
                f[lz],
                f.get(lz + 1).copied().unwrap_or(0),
                f.get(lz + 2).copied().unwrap_or(0),
            );
            let s = a.leading_zeros();
            let top = if s == 0 {
                (a as u128) << 64 | b as u128
            } else {
                (a as u128) << (64 + s) | (b as u128) << s | (cw >> (64 - s)) as u128
            };
            let shift = 64 * lz as i64 + s as i64;
            fast_two_sum(
                (top >> 75) as f64 * pow2(-53 - shift),
                (top & ((1 << 75) - 1)) as f64 * pow2(-128 - shift),
            )
        }
    };
    // r = f·π/2, π/2 as a double-double.
    const PIO2: f64 = core::f64::consts::FRAC_PI_2;
    const PIO2_LO: f64 = 6.123_233_995_736_766e-17;
    let (ph, pe) = two_prod(fh, PIO2);
    let (rh, rl) = fast_two_sum(ph, fh.mul_add(PIO2_LO, fl.mul_add(PIO2, pe)));
    let (rh, rl) = if neg { (-rh, -rl) } else { (rh, rl) };
    let (sn, cs) = kernel_sin_cos(rh, rl, q);
    if (x < 0.0) != (y < 0.0) {
        (-sn, cs)
    } else {
        (sn, cs)
    }
}

/// `1/z` without overflow in `|z|²` (Smith's method), for `|x|, |y| < 2¹⁰²²`.
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

/// `⌊log₂ y⌋·ln 2`, at most `ln y`, of a positive normal `y`; `−1023·ln 2` for a subnormal one,
/// which is above `ln y` (harmless where it is used: there `x² < 100` keeps the term it decides on
/// whenever `y` is that small).
#[inline(always)]
fn ln_floor(y: f64) -> f64 {
    ((y.to_bits() >> 52) as i64 - 1023) as f64 * LN_2
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

/// `e^(−z²) = m·2^k` of a finite `z = x + iy`, `|m|` in about `[0.98, 2.03)`: `y² − x²` as a
/// double-double from exact squares, and the phase `−2xy` reduced exactly (Cody and Waite below
/// `2²⁰`, Payne and Hanek beyond), so that the size and the phase are right to the last bits for
/// any `|z|`. `k` is exact wherever a part of a result can be finite (`|y² − x²| < 1460`), and
/// `±BEYOND` beyond, also where a square overflows (`y² − x² = (|y| − |x|)(|y| + |x|)` is then 0 or
/// beyond `±2⁹⁷¹`): the caller applies it with [`scale_sat`], which sends every nonzero part to
/// the infinity or the zero of its sign.
#[inline(always)]
fn exp_neg_sq(x: f64, y: f64) -> (Complex<f64>, i64) {
    let (xh, xl) = two_prod(x, x);
    let (yh, yl) = two_prod(y, y);
    let a = Dd { hi: yh, lo: yl }.sub(Dd { hi: xh, lo: xl });
    let (m, k) = if a.hi.abs() < EXP_MAX {
        exp_dd(a.hi, a.lo)
    } else if x.abs() == y.abs() {
        // Both squares overflow (a is NaN) and cancel exactly.
        (Dd::new(1.0), 0)
    } else {
        (
            Dd::new(1.0),
            if y.abs() > x.abs() { BEYOND } else { -BEYOND },
        )
    };
    let (ph, pl) = two_prod(x, y);
    // θ = −2xy = −2(ph + pl).
    let (s, co) = if ph.abs() < 524_288.0 {
        sin_cos_dd(-2.0 * ph, -2.0 * pl)
    } else {
        let (s, co) = sin_cos_2xy(x, y);
        (-s, co)
    };
    let mv = m.value();
    (c(mv * co, mv * s), k)
}

/// `(sin θ, cos θ)` of `θ = th + tl` for `|th| < 2²⁰` (`|tl|` below an ulp of `th`): Cody and
/// Waite's reduction by a three-part `π/2` (`k·P₁` and `k·P₂` exact for `|k| < 2²⁰`) and
/// fdlibm's kernels (under 1 ulp).
#[inline(always)]
fn sin_cos_dd(th: f64, tl: f64) -> (f64, f64) {
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
    let (rh, rl) = fast_two_sum(rh, rl);
    kernel_sin_cos(rh, rl, k)
}

/// `(sin θ, cos θ)` of `θ = k·π/2 + rh + rl`, `|rh| ≤ π/4` (about), `rl` below an ulp of `rh`:
/// fdlibm's `__kernel_sin` and `__kernel_cos`.
#[inline(always)]
fn kernel_sin_cos(rh: f64, rl: f64, k: i64) -> (f64, f64) {
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

/// `v·2^k` for any `k`, rounded once (also where it is subnormal): `±∞` where it overflows and
/// `±0` where it underflows, so that with [`BEYOND`] every nonzero `v` is an infinity.
#[inline(always)]
fn scale_sat(v: f64, k: i64) -> f64 {
    if (-1022..=1023).contains(&k) {
        return v * pow2(k);
    }
    if k > 0 {
        // Up to three steps, each exact until the product overflows.
        let k = k.min(3069);
        let k2 = (k - 1023).min(1023);
        return v * pow2(1023) * pow2(k2) * pow2(k - 1023 - k2);
    }
    // The steps before the last are exact, but where their product is below 2⁻¹⁰²²: then the
    // result is below 2⁻²⁰⁴⁴ and rounds to 0 either way.
    if k >= -2044 {
        return v * pow2(k + 1022) * pow2(-1022);
    }
    let k = k.max(-3066);
    v * pow2(k + 2044) * pow2(-1022) * pow2(-1022)
}

/// `v·2^k` of a complex `v`, each part by [`scale_sat`].
#[inline(always)]
fn cscale(v: Complex<f64>, k: i64) -> Complex<f64> {
    c(scale_sat(v.re, k), scale_sat(v.im, k))
}

/// `w(z)` for `x ≥ 0`, `y ≥ 0`, both finite.
fn w_quadrant(x: f64, y: f64) -> Complex<f64> {
    if y == 0.0 {
        return c(exp_neg_sq_real(x), FRAC_2_SQRT_PI * dawson_f64(x));
    }
    if x == 0.0 {
        // Im w has the sign of x (w(−conj z) = conj w(z)).
        return c(erfcx_f64(y), x);
    }
    if x >= 10.0 || y >= 10.0 || x.mul_add(x, y * y) >= 100.0 {
        return w_asymptotic(x, y);
    }
    w_trapezoid(x, y)
}

/// `w(z)` for `|z| ≥ 10` in the closed first quadrant: the asymptotic series in `1/z²`.
#[inline(always)]
fn w_asymptotic(x: f64, y: f64) -> Complex<f64> {
    if x.max(y) >= 1.123_558_209_288_947_4e307 {
        // Beyond 2¹⁰²⁰ w = i/(√π z) to the last bit (the next term is below 2⁻²⁰⁴⁰ of it), but
        // Smith's denominator can overflow and 1/z is subnormal: 1/z of z·2⁻¹⁰, scaled back once.
        let q = recip(x * pow2(-10), y * pow2(-10));
        return c(
            scale(-FRAC_1_SQRT_PI * q.im, -10),
            scale(FRAC_1_SQRT_PI * q.re, -10),
        );
    }
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
    // y² − x² − 4πy as a double-double. It is left out only where it is below 2⁻⁵⁵ of the
    // smallest Re w can be, Re w = (y/π) ∫ e^(−t²)/((x − t)² + y²) dt ≥ 0.0039·y here: where its
    // exponent is below −45, and for y < 1 below −46 + ln y too (for a small y it is all of
    // Re w but a part of the size of y: e^(−x²) on the real axis).
    let ex = y.mul_add(y - FOUR_PI, -x * x);
    if y < 2.0 * PI && (ex > -45.0 || (y < 1.0 && ex > ln_floor(y) - 46.0)) {
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
        return if x == 0.0 {
            // w(iy) = erfcx(y): 0 at +i∞, +∞ at −i∞, NaN for a NaN y (with the zero imaginary
            // part, as erfcx(z) = w(iz) has it on the real axis).
            c(erfcx_f64(y), x)
        } else if x.is_nan() || y.is_nan() {
            c(f64::NAN, f64::NAN)
        } else if y == f64::NEG_INFINITY {
            // |e^(−z²)| grows without bound and its phase has no limit.
            c(f64::NAN, f64::NAN)
        } else {
            // |z| → ∞ with Im z bounded below: w → 0 (below the real axis e^(−z²) → 0 too, as
            // |Re z| → ∞), as i/(√π z) does: the zero parts with the signs of Im z (+0 on the real
            // axis) and Re z.
            c(if y < 0.0 { -0.0 } else { 0.0 }, 0.0f64.copysign(x))
        };
    }
    if y >= 0.0 {
        return w_upper(x, y);
    }
    if x == 0.0 {
        // The imaginary axis: the real erfcx, also where 2e^(y²) overflows.
        return c(erfcx_f64(y), x);
    }
    if x.abs() * (-y).max(1.0) < NEAR_NEG_IMAG {
        return w_near_neg_imag(x, y);
    }
    // 2e^(−z²) − w(−z).
    let (e, k) = exp_neg_sq(x, y);
    let w = w_upper(-x, -y);
    let e = cscale(c(2.0 * e.re, 2.0 * e.im), k);
    c(e.re - w.re, e.im - w.im)
}

/// `w(x + iy)` below the real axis next to the imaginary one, `0 < |x|·max(−y, 1) < 2⁻³⁰`, by
/// the Taylor series in `x` about `iy`: `w = erfcx(y) + 2ix·(1/√π − y·erfcx(y))`, the terms left
/// out below 2⁻⁵⁸ of each part. So the small imaginary part is right on its own, also where
/// `e^(y²)` is beyond any double and it is not: beyond `−y = 26` it is `4|x|·|y|·e^(y²)` (the rest
/// below `e^(−676)` of it), its power of 2 applied once.
#[inline(never)]
fn w_near_neg_imag(x: f64, y: f64) -> Complex<f64> {
    let a = -y;
    let e = erfcx_f64(y);
    if a <= 26.0 {
        return c(e, (2.0 * x) * a.mul_add(e, FRAC_1_SQRT_PI));
    }
    let (h, l) = two_prod(a, a);
    if h >= EXP_MAX {
        return c(e, f64::INFINITY.copysign(x));
    }
    let (m, k) = exp_dd(h, l);
    // 4x·a, exact as p + pe from x·2⁶⁰⁰ (normal even for a subnormal x), times m, rounded once.
    let (p, pe) = two_prod(x * pow2(600), 4.0 * a);
    let v = p.mul_add(m.hi, p.mul_add(m.lo, pe * m.hi));
    c(e, scale_sat(v, k - 600))
}

/// `erf(x + iy)` next to the imaginary axis, `0 < |x|·max(|y|, 1) < 2⁻¹⁴`, part by part from the
/// Taylor series in `x` about `iy`, with `E = e^(y²)`, `c = 2/√π` and Dawson's `F`:
/// `Re = c·E·x·(1 − (2y² + 1)x²/3 + (3 + 12y² + 4y⁴)x⁴/30)` and
/// `Im = c·E·(F(y) − y·x²·(1 − (3 + 2y²)x²/6))`, the terms left out below 2⁻⁸⁰ of each part. So
/// each part is right on its own, the real one (of the size of `x`) too. Returns `Re = r·2^kr`
/// with `r` a double-double (for the `1 − Re` of `erfc`), and `Im`; both parts are infinite where
/// `E` is beyond any range (`F(y) > y·x²` here).
#[inline(never)]
fn erf_near_imag(x: f64, y: f64) -> (Dd, i64, f64) {
    let a = y.abs();
    let (h, l) = two_prod(a, a);
    if h >= EXP_MAX {
        return (
            Dd::new(f64::INFINITY.copysign(x)),
            0,
            f64::INFINITY.copysign(y),
        );
    }
    let (m, k) = exp_dd(h, l);
    let cm = mul_dd(TWO_SQRT_PI, m);
    let (ch, cl) = fast_two_sum(cm.hi, cm.lo);
    let x2 = x * x;
    // The relative corrections, below 2⁻²⁷.
    let dre = x2 * ((3.0 + h * (12.0 + 4.0 * h)) * x2 / 30.0 - (2.0 * h + 1.0) / 3.0);
    let q = a * x2;
    let f = dawson_f64(a) - (q - q * ((3.0 + 2.0 * h) * x2 / 6.0));
    // Re: c·E·x·(1 + dre), with x·2⁶⁰⁰ (normal even for a subnormal x) as a double-double.
    let (p, pe) = two_prod(ch, x * pow2(600));
    let (rh, rl) = fast_two_sum(p, p.mul_add(dre, cl.mul_add(x * pow2(600), pe)));
    // Im: c·E·f, rounded once.
    let (ih, ie) = two_prod(ch, f);
    let (ih, il) = fast_two_sum(ih, cl.mul_add(f, ie));
    let im = if k > 2040 {
        f64::INFINITY
    } else {
        scale_dd(Dd { hi: ih, lo: il }, k)
    };
    (Dd { hi: rh, lo: rl }, k - 600, im.copysign(y))
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
    if x.abs() * y.abs().max(1.0) < NEAR_IMAG {
        // Next to the imaginary axis: each part on its own.
        let (r, kr, im) = erf_near_imag(x, y);
        return c(scale_dd(r, kr), im);
    }
    // erf is odd: 1 − erfc(z) for x ≥ 0, and its negative (as −(1 − e), so that a zero real
    // part changes sign too).
    if x >= 0.0 {
        let e = erfc_right(x, y);
        c(1.0 - e.re, -e.im)
    } else {
        let e = erfc_right(-x, -y);
        c(-(1.0 - e.re), e.im)
    }
}

/// `erfc(z)` of a `Complex<f64>`.
pub(crate) fn erfc_c64(z: Complex<f64>) -> Complex<f64> {
    let (x, y) = (z.re, z.im);
    if y == 0.0 {
        return c(erfc_f64(x), -y);
    }
    if x == 0.0 && !y.is_nan() {
        // The imaginary axis: 1 − i·erfi(y) (e^(y²)·e^(−y²) of the general path would lose the
        // 1 where e^(−y²) underflows).
        return c(1.0, -erfi(y));
    }
    if !(x.is_finite() && y.is_finite()) {
        return if x.is_infinite() && y.is_finite() {
            c(if x > 0.0 { 0.0 } else { 2.0 }, -0.0f64.copysign(y))
        } else {
            c(f64::NAN, f64::NAN)
        };
    }
    if x.abs() * y.abs().max(1.0) < NEAR_IMAG {
        // Next to the imaginary axis: 1 − erf(z) part by part, the real part rounded once.
        let (r, kr, im) = erf_near_imag(x, y);
        let rh = scale(r.hi, kr);
        if rh.is_infinite() {
            return c(-rh, -im);
        }
        let (s, e) = two_sum(1.0, -rh);
        return c(s + (e - scale(r.lo, kr)), -im);
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
        // 1/(√π z) → 0, its imaginary part with the sign of −y.
        return c(0.0, 0.0f64.copysign(-y));
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
