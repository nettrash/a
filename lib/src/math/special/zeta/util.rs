//! The small kernels the zeta area shares: fused multiply-adds where the hardware has them, and
//! polynomial evaluation.
//!
//! The tables are printed by `gen_util.py` in this directory (with `gen_fit.py`).

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

/// `eˣ = 2^q (e + el)` to about `2⁻⁹⁰`: a table of `2^(j/32)` as double-doubles and a
/// polynomial of degree 9 on `|r| ≤ ln 2/64`, its quadratic part exact. For `|x| ≤ 760`.
#[inline]
pub(in crate::math::special) fn exp_parts(x: f64) -> (f64, f64, i32) {
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
/// of two last, so that the result overflows and underflows (to subnormals) correctly. For
/// `|x| ≤ 760`.
#[inline]
pub(in crate::math::special) fn exp_mul(x: f64, g: f64, gl: f64) -> f64 {
    let (e, el, q) = exp_parts(x);
    let (m, me) = two_prod(e, g);
    scale(m + (me + (e * gl + el * g)), q)
}

/// `ln x` as a double-double, for a finite `x > 0`: `l = ln x` corrected by
/// `x e⁻ˡ − 1` (with `e⁻ˡ` from [`exp_parts`]).
#[inline]
pub(in crate::math::special) fn ln_dd(x: f64) -> (f64, f64) {
    let l = x.ln();
    let (e, el, q) = exp_parts(-l);
    let xs = scale(x, q);
    let (m, me) = two_prod(e, xs);
    (l, (m - 1.0) + (me + el * xs))
}

/// `xʸ · (g + gl)` for `x > 0`, rounded once: `y ln x` as a double-double.
#[inline]
pub(in crate::math::special) fn pow_mul(x: f64, y: f64, g: f64, gl: f64) -> f64 {
    let (l, ll) = ln_dd(x);
    let (t, te) = two_prod(y, l);
    let tl = te + y * ll;
    if t.abs() > 760.0 {
        return if t > 0.0 { f64::INFINITY * g } else { 0.0 * g };
    }
    // e^(t + tl) = eᵗ (1 + tl)
    exp_mul(t, g, gl + g * tl)
}

/// `y · 2^q`, rounded once (in two steps of powers of two past the normal range).
#[inline(always)]
pub(in crate::math::special) fn scale(y: f64, q: i32) -> f64 {
    let pow2 = |e: i32| f64::from_bits(((e + 1023) as u64) << 52);
    if (-1000..=1000).contains(&q) {
        y * pow2(q)
    } else if q > 0 {
        y * pow2(1000) * pow2(q - 1000)
    } else {
        // Into the subnormals: the first factor keeps y normal, the second rounds once.
        (y * pow2(q + 1000)) * pow2(-1000)
    }
}
