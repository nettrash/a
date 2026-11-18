//! The elementary kernels of the Bessel and Airy functions, in arithmetic alone (so that every
//! platform, and Miri, gives the same bits): `e^x` and `ln x` to about 2⁻⁶⁸, the reduction of an
//! argument modulo `π/2` to a double-double (Cody and Waite's below 2²⁰, Payne and Hanek's
//! above), and the sine and cosine of a reduced double-double argument (FreeBSD msun's kernels).

#![allow(clippy::excessive_precision)]

use super::dd::{Dd, fast_two_sum, two_prod, two_sum};

/// Whether `mul_add` is one fused instruction (as the integration's `dd.rs`): the polynomials use
/// `mla`, fused there and plain elsewhere (results then differ in the last bit between the two
/// kinds of targets, within the same bounds; Miri takes the plain path).
pub(crate) const FMA: bool = cfg!(all(
    any(target_arch = "aarch64", target_feature = "fma"),
    not(miri)
));

/// `a·b + c`, fused where the hardware has it.
#[inline(always)]
pub(crate) fn mla(a: f64, b: f64, c: f64) -> f64 {
    if FMA { a.mul_add(b, c) } else { a * b + c }
}

/// `Σ c[from + i]·x^i` by Estrin's scheme (pairs, then pairs of pairs with `x²`, `x⁴`, …), for a
/// row of a fixed width, so that the loops unroll into straight code.
#[inline(always)]
pub(crate) fn estrin_at<const W: usize>(x: f64, c: &[f64; W], from: usize) -> f64 {
    let mut v = [0.0f64; W];
    let n = W - from;
    let mut m = 0;
    let mut i = 0;
    while i < n {
        v[m] = if i + 1 < n {
            mla(x, c[from + i + 1], c[from + i])
        } else {
            c[from + i]
        };
        m += 1;
        i += 2;
    }
    let mut p = x * x;
    while m > 1 {
        let mut k = 0;
        let mut j = 0;
        while j < m {
            v[k] = if j + 1 < m {
                mla(p, v[j + 1], v[j])
            } else {
                v[j]
            };
            k += 1;
            j += 2;
        }
        m = k;
        p *= p;
    }
    v[0]
}

/// `Σ c[i]·x^i` by Estrin's scheme, of a table of a fixed length.
#[inline(always)]
pub(crate) fn estrin<const W: usize>(x: f64, c: &[f64; W]) -> f64 {
    estrin_at(x, c, 0)
}

/// `2^e` for `e` in `−1022..=1023`.
#[inline(always)]
pub(crate) fn pow2i(e: i32) -> f64 {
    f64::from_bits(((e + 1023) as u64) << 52)
}

/// `v·2^e` for any `e`, rounded once (also into the subnormals; musl's `scalbn`).
#[inline]
pub(crate) fn ldexp(mut v: f64, mut e: i32) -> f64 {
    if e > 1023 {
        v *= pow2i(1023);
        e -= 1023;
        if e > 1023 {
            v *= pow2i(1023);
            e = (e - 1023).min(1023);
        }
    } else if e < -1022 {
        v *= pow2i(-1022 + 53);
        e += 1022 - 53;
        if e < -1022 {
            v *= pow2i(-1022 + 53);
            e = (e + 1022 - 53).max(-1022);
        }
    }
    v * pow2i(e)
}

/// `ln 2 / 32` in two parts, the first of 32 bits (so `k·LN2_32_1` is exact for `|k| < 2²¹`).
const LN2_32_1: f64 = f64::from_bits(0x3f962e42fee00000);
const LN2_32_2: f64 = 5.963_171_653_970_586_6e-12;
const INV_LN2_32: f64 = 46.166_241_308_446_83;

/// `2^(j/32)` as double-doubles.
const EXP2_32: [(f64, f64); 32] = [
    (1.0, 0.0),
    (1.021_897_148_654_116_6, 5.109_225_028_973_444e-17),
    (1.044_273_782_427_413_8, 8.551_889_705_537_965e-17),
    (1.067_140_400_676_823_7, -7.899_853_966_841_582e-17),
    (1.090_507_732_665_257_7, -3.046_782_079_812_471e-17),
    (1.114_386_742_595_892_4, 1.041_027_845_684_557_1e-16),
    (1.138_788_634_756_691_6, 8.912_812_676_025_408e-17),
    (1.163_724_858_777_577_5, 3.829_204_836_924_093_5e-17),
    (1.189_207_115_002_721, 3.982_015_231_465_646e-17),
    (1.215_247_359_980_469, -7.712_630_692_681_488e-17),
    (1.241_857_812_073_484, 4.658_027_591_836_937e-17),
    (1.269_050_957_191_733_2, 2.667_932_131_342_186e-18),
    (1.296_839_554_651_009_6, 2.538_250_279_488_831_5e-17),
    (1.325_236_643_159_741_3, -2.858_731_210_038_861_4e-17),
    (1.354_255_546_936_892_7, 7.700_948_379_802_99e-17),
    (1.383_909_881_963_832, -6.770_511_658_794_786e-17),
    (core::f64::consts::SQRT_2, -9.667_293_313_452_913e-17),
    (1.445_180_806_977_046_7, -3.023_758_134_993_987_3e-17),
    (1.476_826_145_939_499_3, -3.483_994_556_892_796e-17),
    (1.509_164_427_593_422_8, -1.016_455_327_754_295e-16),
    (1.542_210_825_407_940_7, 7.949_834_809_697_621e-17),
    (1.575_980_845_107_886_5, -1.013_691_647_127_830_4e-17),
    (1.610_490_331_949_254_3, 2.470_719_256_979_788_8e-17),
    (1.645_755_478_153_965, -1.012_567_991_367_477_3e-16),
    (1.681_792_830_507_429, 8.199_010_020_581_497e-17),
    (1.718_619_298_122_478, -1.851_380_418_263_111e-17),
    (1.756_252_160_373_299_5, 2.960_140_695_448_873e-17),
    (1.794_709_075_003_107_2, 1.822_745_842_791_208_7e-17),
    (1.834_008_086_409_342_4, 3.283_107_224_245_627e-17),
    (1.874_167_634_110_3, -6.122_763_413_004_143e-17),
    (1.915_206_561_397_147_4, -1.061_994_605_619_596_3e-16),
    (1.957_144_124_175_400_2, 8.960_767_791_036_668e-17),
];

/// `e^(hi + lo) = m·2^k`, `m` a double-double in about `[0.98, 2.04)` with a relative error
/// below 2⁻⁶⁸, for `|hi| ≤ 2^20` (the caller handles overflow and underflow through `k`).
#[inline]
pub(crate) fn exp_parts(hi: f64, lo: f64) -> (Dd, i32) {
    let kf = (hi * INV_LN2_32).round_ties_even();
    let k = kf as i32;
    let r1 = kf.mul_add(-LN2_32_1, hi);
    let (rh, rl) = two_sum(r1, kf.mul_add(-LN2_32_2, lo));
    // e^r = 1 + r + r²/2 + r³·(1/6 + r/24 + …), to r⁸/8!; r²/2 exactly.
    let (qh, ql) = two_prod(rh, 0.5 * rh);
    let r2 = rh * rh;
    let r4 = r2 * r2;
    let p3 = (r2 * rh)
        * (mla(rh, 1.0 / 24.0, 1.0 / 6.0)
            + r2 * mla(rh, 1.0 / 720.0, 1.0 / 120.0)
            + r4 * mla(rh, 1.0 / 40320.0, 1.0 / 5040.0));
    let (sh, sl) = fast_two_sum(rh, qh);
    let sl = sl + (rl + (ql + (p3 + rh * rl)));
    let (th, tl) = EXP2_32[(k & 31) as usize];
    let (ph, pl) = two_prod(th, sh);
    let (h, l) = fast_two_sum(th, ph);
    let l = l + (pl + th.mul_add(sl, tl.mul_add(sh, tl)));
    let (h, l) = fast_two_sum(h, l);
    (Dd { hi: h, lo: l }, k >> 5)
}

/// `e^x = m·2^k` for an `f64` `|x| ≤ 2^20`, `m` a double-double to about 2⁻⁵⁹
/// (a shorter polynomial than `exp_parts`, for products rounded once at the end).
#[inline(always)]
pub(crate) fn exp_parts_lite(x: f64) -> (Dd, i32) {
    let kf = (x * INV_LN2_32).round_ties_even();
    let k = kf as i32;
    let r = kf.mul_add(-LN2_32_1, x) - kf * LN2_32_2;
    // e^r − 1 = r + r²·(1/2 + r/6 + r²/24 + r³/120 + r⁴/720), to r⁶/6! (error r⁷/7! < 2⁻⁵⁸).
    let r2 = r * r;
    let s = mla(
        r2,
        mla(r, 1.0 / 6.0, 0.5) + r2 * mla(r, 1.0 / 120.0, 1.0 / 24.0) + (r2 * r2) * (1.0 / 720.0),
        r,
    );
    let (th, tl) = EXP2_32[(k & 31) as usize];
    let (hi, lo) = fast_two_sum(th, mla(th, s, tl + tl * s));
    (Dd { hi, lo }, k >> 5)
}

/// `e^x` of an `f64`, about 0.5 ulp, with overflow to `∞` and underflow to 0.
#[inline]
pub(crate) fn exp(x: f64) -> f64 {
    if x > 709.8 {
        return if x.is_nan() { x } else { f64::INFINITY };
    }
    if x < -745.2 {
        return 0.0;
    }
    let (m, k) = exp_parts(x, 0.0);
    ldexp(m.hi + m.lo, k)
}

/// `ln 2` as a double-double, the first part of 42 bits (so `e·LN2_1` is exact for `|e| < 2¹¹`).
const LN2_1: f64 = f64::from_bits(0x3fe62e42fefa3800);
const LN2_2: f64 = f64::from_bits(0x3d2ef35793c76730);

/// `ln x` of a positive finite `x` as a double-double, absolute error below 2⁻⁶⁷ (one step of
/// third order on `e^(−y)·f = 1` from a series seed for the significand `f`).
#[inline]
pub(crate) fn ln_dd(x: f64) -> Dd {
    // x = 2^e·f, f in [√½, √2).
    let (x, e0) = if x < f64::MIN_POSITIVE {
        (x * pow2i(64), -64)
    } else {
        (x, 0)
    };
    let b = x.to_bits();
    let mut e = ((b >> 52) as i32) - 1023 + e0;
    let mut f = f64::from_bits((b & ((1u64 << 52) - 1)) | (1023u64 << 52));
    if f > core::f64::consts::SQRT_2 {
        f *= 0.5;
        e += 1;
    }
    // The seed: 2·atanh((f − 1)/(f + 1)) to s⁹ (error below 10⁻⁹, which the step cubes), in
    // arithmetic alone, so that every platform gets the same bits.
    let sd = (f - 1.0) / (f + 1.0);
    let s2 = sd * sd;
    let y = 2.0 * sd * (1.0 + s2 * (1.0 / 3.0 + s2 * (0.2 + s2 * (1.0 / 7.0 + s2 * (1.0 / 9.0)))));
    let (m, k) = exp_parts(-y, 0.0);
    // δ = f·e^(−y) − 1, with f·m formed exactly and scaled by 2^k (k is 0 or ±1).
    let (ph, pl) = two_prod(f, m.hi);
    let s = pow2i(k);
    let d = (ph * s - 1.0) + (pl + f * m.lo) * s;
    let (hi, lo) = fast_two_sum(y, d - 0.5 * d * d);
    let ef = e as f64;
    let (h2, l2) = two_sum(ef * LN2_1, hi);
    let (hi, lo) = fast_two_sum(h2, l2 + (lo + ef * LN2_2));
    Dd { hi, lo }
}

/// `π/2` in three parts of 33, 33 and 53 bits, and the tail.
const PIO2_1: f64 = f64::from_bits(0x3ff921fb54400000);
const PIO2_2: f64 = f64::from_bits(0x3dd0b4611a600000);
const PIO2_3: f64 = 2.022_266_248_795_950_6e-21;
const PIO2_3T: f64 = 1.008_585_403_587_248_3e-37;
/// `π/2` and `π/4` as double-doubles.
pub(crate) const PIO2: Dd = Dd {
    hi: core::f64::consts::FRAC_PI_2,
    lo: 6.123_233_995_736_766e-17,
};
pub(crate) const PIO4: Dd = Dd {
    hi: core::f64::consts::FRAC_PI_4,
    lo: 3.061_616_997_868_383e-17,
};

/// The bits of `2/π` after the binary point.
const TWO_OVER_PI: [u64; 20] = [
    0xa2f9836e4e441529,
    0xfc2757d1f534ddc0,
    0xdb6295993c439041,
    0xfe5163abdebbc561,
    0xb7246e3a424dd2e0,
    0x06492eea09d1921c,
    0xfe1deb1cb129a73e,
    0xe88235f52ebb4484,
    0xe99c7026b45f7e41,
    0x3991d639835339f4,
    0x9c845f8bbdf9283b,
    0x1ff897ffde05980f,
    0xef2f118b5a0a6d1f,
    0x6d367ecf27cb09b7,
    0x4f463f669e5fea2d,
    0x7527bac7ebe5f17b,
    0x3d0739f78a5292ea,
    0x6bfb5fb11f8d5d08,
    0x56033046fc7b6bab,
    0xf0cfbc209af4361d,
];

/// The 64 bits of `2/π` from the bit `g` after the point on (0 the first; bits before it zero).
#[inline]
fn two_over_pi_bits(g: i32) -> u64 {
    if g <= -64 {
        return 0;
    }
    if g < 0 {
        return TWO_OVER_PI[0] >> (-g);
    }
    let (k, s) = ((g / 64) as usize, g % 64);
    if s == 0 {
        TWO_OVER_PI[k]
    } else {
        (TWO_OVER_PI[k] << s) | (TWO_OVER_PI[k + 1] >> (64 - s))
    }
}

/// `x = n·π/2 + r` with `|r| ≤ π/4` (a little more near the halfway points), `r` as a
/// double-double to about 2⁻¹⁰⁶ relative to `max(|r|, 2⁻⁶⁰)`; `n` modulo 4 only above 2⁶².
pub(crate) fn rem_pio2(x: f64) -> (i64, Dd) {
    let ax = x.abs();
    if ax < 0.785 {
        return (0, Dd::new(x));
    }
    if ax < 1.0e6 {
        let nf = (x * core::f64::consts::FRAC_2_PI).round_ties_even();
        let t1 = nf.mul_add(-PIO2_1, x);
        let (s, e) = two_sum(t1, -nf * PIO2_2);
        let (p, pe) = two_prod(nf, PIO2_3);
        let r = Dd { hi: s, lo: e }.add(Dd {
            hi: -p,
            lo: -pe - nf * PIO2_3T,
        });
        return (nf as i64, r);
    }
    if !ax.is_finite() {
        return (0, Dd::new(f64::NAN));
    }
    // Payne and Hanek: x = m·2^q, x·2/π ≡ m·W·2^(−190) (mod 4) for the 192 bits W of 2/π from
    // the bit q − 2 after the point.
    let b = ax.to_bits();
    let m = (b & ((1u64 << 52) - 1)) | (1u64 << 52);
    let q = ((b >> 52) as i32) - 1075;
    let g0 = q - 2;
    let (w2, w1, w0) = (
        two_over_pi_bits(g0),
        two_over_pi_bits(g0 + 64),
        two_over_pi_bits(g0 + 128),
    );
    let a0 = (m as u128) * (w0 as u128);
    let a1 = (m as u128) * (w1 as u128) + (a0 >> 64);
    let p0 = a0 as u64;
    let p1 = a1 as u64;
    let p2 = m.wrapping_mul(w2).wrapping_add((a1 >> 64) as u64);
    let mut n = (p2 >> 62) as i64;
    // The fraction, 190 bits: f = (p2 mod 2^62)·2^128 + p1·2^64 + p0.
    let mut hi128 = (((p2 & ((1u64 << 62) - 1)) as u128) << 64) | p1 as u128;
    let mut lo64 = p0;
    let mut neg = false;
    if hi128 >> 125 != 0 {
        // f ≥ 1/2: take f − 1, i.e. −(2^190 − f).
        n += 1;
        neg = true;
        let (l, borrow) = 0u64.overflowing_sub(lo64);
        lo64 = l;
        hi128 = (1u128 << 126)
            .wrapping_sub(hi128)
            .wrapping_sub(borrow as u128);
    }
    let lz = hi128.leading_zeros() as i32;
    let g = if lz < 64 {
        (hi128 << lz) | ((lo64 as u128) >> (64 - lz))
    } else {
        hi128 << lz
    };
    // f = g·2^(−126 − lz) quarter turns.
    let h53 = (g >> 75) as u64 as f64;
    let rest = (g & ((1u128 << 75) - 1)) as f64;
    let fh = h53 * pow2i(-51 - lz);
    let fl = rest * pow2i(-126 - lz);
    let f = {
        let (h, l) = fast_two_sum(fh, fl);
        Dd { hi: h, lo: l }
    };
    let mut r = f.mul(PIO2);
    if neg {
        r = r.neg();
    }
    if x < 0.0 { (-n, r.neg()) } else { (n, r) }
}

/// `sin(x + y)` as an unevaluated double-double (the hi part `x`), error about 2⁻⁵⁶ relative.
#[inline(always)]
pub(crate) fn k_sin_dd(x: f64, y: f64) -> Dd {
    const S1: f64 = -1.666_666_666_666_663_243_48e-1;
    const S2: f64 = 8.333_333_333_322_489_461_24e-3;
    const S3: f64 = -1.984_126_982_985_794_931_34e-4;
    const S4: f64 = 2.755_731_370_707_006_767_89e-6;
    const S5: f64 = -2.505_076_025_340_686_341_95e-8;
    const S6: f64 = 1.589_690_995_211_550_102_21e-10;
    let z = x * x;
    let w = z * z;
    let r = S2 + z * (S3 + z * S4) + z * w * (S5 + z * S6);
    let v = z * x;
    let t = -((z * (0.5 * y - v * r) - y) - v * S1);
    let (hi, lo) = fast_two_sum(x, t);
    Dd { hi, lo }
}

/// `cos(x + y)` as a double-double, error about 2⁻⁵⁷.
#[inline(always)]
pub(crate) fn k_cos_dd(x: f64, y: f64) -> Dd {
    const C1: f64 = 4.166_666_666_666_660_190_37e-2;
    const C2: f64 = -1.388_888_888_887_410_957_49e-3;
    const C3: f64 = 2.480_158_728_947_672_941_78e-5;
    const C4: f64 = -2.755_731_435_139_066_330_35e-7;
    const C5: f64 = 2.087_572_321_298_174_827_90e-9;
    const C6: f64 = -1.135_964_755_778_819_482_65e-11;
    let (z, ze) = two_prod(x, x);
    let w = z * z;
    let r = z * (C1 + z * (C2 + z * C3)) + w * w * (C4 + z * (C5 + z * C6));
    let hz = 0.5 * z;
    let w = 1.0 - hz;
    let t = ((1.0 - w) - hz) + (z * r - (x * y + 0.5 * ze));
    let (hi, lo) = fast_two_sum(w, t);
    Dd { hi, lo }
}

/// `cos(n·π/2 + u)` (`sine` false) or `sin(n·π/2 + u)` as a double-double.
#[inline]
pub(crate) fn sin_or_cos_quadrant_dd(n: i64, u: Dd, sine: bool) -> Dd {
    let m = if sine { n.wrapping_sub(1) } else { n };
    // cos(mπ/2 + u): m even → ±cos u, odd → ∓sin u.
    let v = if m & 1 == 0 {
        k_cos_dd(u.hi, u.lo)
    } else {
        k_sin_dd(u.hi, u.lo)
    };
    if (m & 3) == 1 || (m & 3) == 2 {
        v.neg()
    } else {
        v
    }
}

/// `(sin, cos)` of `n·π/2 + u` as double-doubles.
#[inline]
pub(crate) fn sin_cos_quadrant_dd(n: i64, u: Dd) -> (Dd, Dd) {
    let s = k_sin_dd(u.hi, u.lo);
    let c = k_cos_dd(u.hi, u.lo);
    match n & 3 {
        0 => (s, c),
        1 => (c, s.neg()),
        2 => (s.neg(), c.neg()),
        _ => (c.neg(), s),
    }
}

/// `(sin, cos)` of `x + φ` as double-doubles (see `sin_cos_shifted`).
#[inline]
pub(crate) fn sin_cos_shifted_dd(x: f64, phi: Dd) -> (Dd, Dd) {
    let (n, r) = rem_pio2(x);
    let u = r.add(phi);
    let k = (u.hi * core::f64::consts::FRAC_2_PI).round_ties_even();
    let u = u.sub(PIO2.mul_f64(k));
    sin_cos_quadrant_dd(n.wrapping_add(k as i64), u)
}
