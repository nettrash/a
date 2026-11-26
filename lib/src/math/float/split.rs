//! The sine and the cosine at long working precisions by binary splitting, after Brent's bit-burst
//! method as MPFR's `mpfr_sincos_fast` carries it out: the argument `r` in `[0, 1]` is a sum of
//! pieces `x_k = u_k/2^(2^k)`, `u_k` the bits of `r` from `2^(k - 1) + 1` to `2^k` after the point
//! (the first piece the bits to 32), so that `x_k < 2^(-2^(k - 1))` has a numerator of `2^(k - 1)`
//! bits; the sine of each piece is its Taylor series summed exactly by binary splitting and its
//! cosine `sqrt(1 - sin²)`, and the pieces are put together by the formulas of the sum. About
//! `O(M(F) log² F)` for `F` bits. The arctangent by the bit burst on the point `1 + iz` too
//! ([`atan`]).

use super::limbs;
use super::sincos::{Fx, Lg, div, isqrt};

/// `(P, Q, |T|)` of the terms `a..c` of `sin x = x (1 + Σ_(n >= 1) Π_(k <= n) p(k)/q(k))` for
/// `x = u/2^b`, `p(k) = -u²` and `q(k) = (2k)(2k + 1)·2^(2b)`: `P = Π p`, `Q = Π q` (whose power
/// of two, `2^(2b(c - a))`, is a shift), `T/Q` the sum of the terms; `P` only when `need_p`.
///
/// `T(a, c) = T(a, m) Q(m, c) + P(a, m) T(m, c)`. The terms alternate and decrease (`x <= 1`), so
/// every `T` is below zero, and `|T(a, c)| = |T(a, m)| Q(m, c) ± |P(a, m)| |T(m, c)|`, the sign
/// `(-1)^(m - a)` of `P(a, m)`.
fn split(a: u64, c: u64, u2: &[u64], b2: u64, need_p: bool) -> (Vec<u64>, Vec<u64>, Vec<u64>) {
    if c - a == 1 {
        let q = u128::from(2 * a) * u128::from(2 * a + 1);
        let mut qv = vec![q as u64, (q >> 64) as u64];
        limbs::trim(&mut qv);
        let q = limbs::shl(&qv, b2);
        return (u2.to_vec(), q, u2.to_vec());
    }
    let m = a + (c - a) / 2;
    let (pl, ql, tl) = split(a, m, u2, b2, true);
    let (pr, qr, tr) = split(m, c, u2, b2, need_p);
    let left = limbs::mul(&tl, &qr);
    let right = limbs::mul(&pl, &tr);
    let t = if (m - a).is_multiple_of(2) {
        limbs::add(&left, &right)
    } else {
        limbs::sub(&left, &right)
    };
    let p = if need_p {
        limbs::mul(&pl, &pr)
    } else {
        Vec::new()
    };
    (p, limbs::mul(&ql, &qr), t)
}

/// `sin x` for `x = u·2^-b`, `0 < x <= 1`, at `f` bits after the point, within `1.25` ulps: the
/// `N` terms whose tail `x^(2N + 1)/(2N + 1)!` is below a quarter of an ulp, summed exactly, and
/// `floor(u (Q - |T|) 2^f / (Q 2^b))`.
fn sin_piece(u: &[u64], b: u64, f: u64) -> Vec<u64> {
    let ub = limbs::bit_len(u);
    // `x < 2^-(b - ub)`, `x <= 1`.
    let gap = b as f64 - ub as f64;
    let mut n = 1u64;
    let mut lf = 2.5849f64; // `log2 3!`, rounded down
    while (2 * n + 1) as f64 * gap + lf <= (f + 2) as f64 {
        n += 1;
        lf += ((2 * n) as f64).log2() + ((2 * n + 1) as f64).log2() - 1e-9;
    }
    // The tail starts at term `n`: the terms `1..n` are summed.
    let u2 = limbs::mul(u, u);
    let s = if n == 1 {
        limbs::shl(u, f)
    } else {
        let (_, q, t) = split(1, n, &u2, 2 * b, false);
        let num = limbs::mul(u, &limbs::sub(&q, &t));
        return div(&limbs::shl(&num, f), &limbs::shl(&q, b));
    };
    limbs::shr(&s, b)
}

/// The sine and the cosine of `r = R·2^-g`, `2^-K < |r| <= 1` (the sign plays no part), at `F =
/// 64·fl` bits after the point, as [`sin_cos`](super::sincos::sin_cos) gives them; `cut` when `r`
/// was cut to its bits already (an error of a unit of `2^-g`).
///
/// At `F' = F + 64` bits: each piece's sine within 1.25 ulps, its cosine `floor(sqrt(1 - s²))`
/// within `1.56·1.25 + 1 < 3` (`|d cos/d sin| = sin/cos <= 1.56` below 1); the sum `(S, C) ←
/// (S c + C s, C c - S s)` of the pieces so far `(S, C)` with errors `e_S`, `e_C` and a piece `(s,
/// c)`, `s <= 2^-lo`, adds `e_S c + S e_c + e_C s + C e_s + 2`, at most `e_S + e_C·2^-lo + 6.3`
/// (and likewise for `C`). Then `r`'s own cut to `F'` bits, an ulp.
pub(super) fn sin_cos(
    r: &[u64],
    g: i128,
    cut: bool,
    fl: usize,
    want_sin: bool,
    want_cos: bool,
) -> (Option<Fx>, Option<Fx>) {
    let f = 64 * fl as u64;
    let fp = f + 64;
    // `r` at `F'` bits after the point.
    let sh = i128::from(fp) - g;
    let rf = if sh >= 0 {
        limbs::shl(r, sh as u64)
    } else {
        limbs::shr(r, sh.unsigned_abs() as u64)
    };
    let one = limbs::shl(&[1], fp);
    let (mut s, mut c) = (Vec::new(), one.clone());
    let (mut es, mut ec) = (0f64, 0f64);
    // The first piece the bits to 32 after the point: one binary splitting of a numerator of 33
    // bits costs about half the six of the pieces of 1, 1, 2, 4, 8 and 16 bits it stands for.
    let (mut lo, mut hi) = (0u64, 32u64);
    while lo < fp {
        let hi_c = hi.min(fp);
        // The bits `lo + 1..=hi_c` after the point, `floor(r·2^hi_c) mod 2^(hi_c - lo)`, and for
        // the first piece the bit before the point too (`r = 1`).
        let mut u = limbs::shr(&rf, fp - hi_c);
        if lo > 0 {
            let width = hi_c - lo;
            let k = (width / 64) as usize;
            u.truncate(k + 1);
            if let Some(last) = u.get_mut(k) {
                *last &= (1u64 << (width % 64)) - 1;
            }
            limbs::trim(&mut u);
        }
        if !u.is_empty() {
            let sp = sin_piece(&u, hi_c, fp);
            let cp = isqrt(&limbs::sub(
                &limbs::shl(&[1], 2 * fp),
                &limbs::mul(&sp, &sp),
            ));
            let small = (-(lo as f64)).exp2();
            if s.is_empty() {
                (s, c) = (sp, cp);
                (es, ec) = (1.25, 3.0);
            } else {
                let ns = limbs::add(&mul_fix(&s, &cp, fp), &mul_fix(&c, &sp, fp));
                let nc = limbs::sub(&mul_fix(&c, &cp, fp), &mul_fix(&s, &sp, fp));
                (es, ec) = (es + ec * small + 6.3, ec + es * small + 6.3);
                (s, c) = (ns, nc);
            }
        }
        lo = hi_c;
        hi *= 2;
    }
    // `r`'s cut at `F'` bits and before.
    let extra = if cut { 3.0 } else { 1.0 };
    let lsb = -i128::from(fp);
    let sin = want_sin.then(|| Fx {
        negative: false,
        m: s,
        lsb,
        err: Lg::of(es + extra),
    });
    let cos = want_cos.then(|| Fx {
        negative: false,
        m: c,
        lsb,
        err: Lg::of(ec + extra),
    });
    (sin, cos)
}

/// `floor(a·b·2^-f)`.
fn mul_fix(a: &[u64], b: &[u64], f: u64) -> Vec<u64> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    limbs::shr(&limbs::mul(a, b), f)
}

/// The angle of the point `(re, im)` cut to a dyadic of `hi` bits after the point, for `|im/re| <
/// 1`: `floor(|im| 2^hi / re)`, from `re` and `im` cut to the top `hi - lo + 96` bits of `im`,
/// `lo` the bits that are zero after the point (a quotient of a few more bits than the piece
/// holds, within a relative `2^(1 - (hi - lo + 96))` of the exact one).
fn piece(re: &[u64], im: &[u64], lo: u64, hi: u64) -> Vec<u64> {
    // Both cut by `s` bits, `im` keeps `hi - lo + 96` bits, and so the quotient about as many:
    // `|im/re| < 2^(2 - lo)`, so that it has about `hi - lo` bits.
    let keep = hi - lo.min(hi) + 96;
    let s = limbs::bit_len(im).saturating_sub(keep);
    let r = limbs::shr(re, s);
    let i = limbs::shr(im, s);
    div(&limbs::shl(&i, hi), &r)
}

/// `atan(z)` for the fixed-point `z = Z·2^-F`, `0 < z <= 1`, `F = 64·fl`, by the bit burst on the
/// point `1 + iz`, whose angle it is: the angle is taken off in pieces `θ_k` of doubling lengths,
/// `(re, im) ← (re, im)·(cos θ_k - i sin θ_k)`, the sine of each piece by the binary splitting of
/// its series ([`sin_piece`]) and its cosine `sqrt(1 - sin²)`, as for [`sin_cos`]; each piece
/// read off the point's own angle, `atan(im/re) = im/re` within `|im/re|³/3`, so that no division
/// at the full precision is needed but the last one. The first piece is the `f64` arctangent cut
/// to 32 bits; the pieces stop when `lo`, the bits after the point that the angle left has zero,
/// reaches a third of the precision, and the angle left is `t = im/re` within `t³/3`. Returns the
/// angle at `F` bits after the point and the bound of its error in ulps.
///
/// At `F' = F + 64` bits: a step multiplies the point by `ŵ = (ĉ, -ŝ)` within `e_s + e_c` ulps of
/// `e^(-iθ_k)` (`e_s <= 1.25`, `e_c <= 3`, as in [`sin_cos`]), whose angle is within `asin(e_s +
/// e_c)` ulps of `-θ_k`, and the four products cut add `|ρ| <= 3` ulps, an angle below
/// `asin(|ρ|/|ẑ ŵ|)`; the modulus starts at `sqrt(1 + z²) >= 1` and stays above `0.99`, so each
/// step moves the angle by at most `1.02 (e_s + e_c + 3)` ulps, the first piece's too. The pieces
/// are exact dyadics, the last quotient within two ulps.
pub(super) fn atan(z: &[u64], fl: usize) -> (Vec<u64>, Lg) {
    let f = 64 * fl as u64;
    let fp = f + 64;
    let x = limbs::shl(z, 64);
    // The first piece: the `f64` arctangent of the top bits, cut to 32 bits.
    let bz = limbs::bit_len(z);
    let top = if bz > 64 {
        limbs::shr(z, bz - 64)[0]
    } else {
        z[0] << (64 - bz)
    };
    let zf = (top as f64 * (bz as f64 - f as f64 - 64.0).exp2()).min(1.0);
    let a = zf.atan();
    // `a = m·2^-g0` with `m` of 32 bits.
    let g0 = (32.0 - a.log2().floor()) as u64;
    let u0 = (a * (g0 as f64).exp2()) as u64;
    // Its sine by one binary splitting (one piece of 32 bits costs about half the six pieces of
    // the sine's own bit burst), within 1.25 ulps, and its cosine within 3.
    let s0 = sin_piece(&[u0], g0, fp);
    let c0 = isqrt(&limbs::sub(
        &limbs::shl(&[1], 2 * fp),
        &limbs::mul(&s0, &s0),
    ));
    let mut e = 1.02 * (1.25 + 3.0 + 3.0);
    // `(1 + iX)(c - i s)`.
    let mut re = limbs::add(&c0, &mul_fix(&x, &s0, fp));
    let xc = mul_fix(&x, &c0, fp);
    let (mut neg, mut im) = if limbs::cmp(&xc, &s0) == core::cmp::Ordering::Less {
        (true, limbs::sub(&s0, &xc))
    } else {
        (false, limbs::sub(&xc, &s0))
    };
    // The angle so far, at `F'` bits: the positive and the negative pieces.
    let mut pos = limbs::shl(&[u0], fp - g0);
    let mut negs: Vec<u64> = Vec::new();
    let stop = (fp + 6).div_ceil(3);
    let mut lo = g0 - 1;
    while lo < stop && !im.is_empty() {
        let hi = (2 * lo).min(stop + 1);
        let u = piece(&re, &im, lo, hi);
        if !u.is_empty() {
            let sp = sin_piece(&u, hi, fp);
            let cp = isqrt(&limbs::sub(
                &limbs::shl(&[1], 2 * fp),
                &limbs::mul(&sp, &sp),
            ));
            // `(re + i im)(c - i s)` for `im >= 0`, `(c + i s)` for `im < 0`, in magnitudes.
            let nre = limbs::add(&mul_fix(&re, &cp, fp), &mul_fix(&im, &sp, fp));
            let ic = mul_fix(&im, &cp, fp);
            let rs = mul_fix(&re, &sp, fp);
            let (flip, nim) = if limbs::cmp(&ic, &rs) == core::cmp::Ordering::Less {
                (true, limbs::sub(&rs, &ic))
            } else {
                (false, limbs::sub(&ic, &rs))
            };
            let pu = limbs::shl(&u, fp - hi);
            if neg {
                negs = limbs::add(&negs, &pu);
            } else {
                pos = limbs::add(&pos, &pu);
            }
            neg ^= flip;
            re = nre;
            im = nim;
            e += 1.02 * (1.25 + 3.0 + 3.0);
        }
        lo = hi;
    }
    // The angle left: `t = im/re` within an ulp and `t³/3`.
    if !im.is_empty() {
        // The quotient from the cut `re` and `im`: within an ulp and one more.
        let t = piece(&re, &im, lo, fp);
        let tb = limbs::bit_len(&t) as f64;
        e += 2.0 + (3.0 * tb - 2.0 * fp as f64).exp2();
        if neg {
            negs = limbs::add(&negs, &t);
        } else {
            pos = limbs::add(&pos, &t);
        }
    }
    let y = limbs::sub(&pos, &negs);
    // At `F` bits: the floor adds an ulp of `2^-F`.
    let y = limbs::shr(&y, 64);
    (y, Lg::of(e / 18_446_744_073_709_551_616.0 + 1.0))
}
