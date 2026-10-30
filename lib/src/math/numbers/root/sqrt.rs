//! Square roots of magnitudes held as limbs: native `u64`s, the least significant first.
//!
//! The root of two limbs comes from a floating-point estimate made exact (see
//! [`sqrt_rem_u128`]). Longer magnitudes are rooted by Zimmermann's Karatsuba square root
//! ("Karatsuba Square Root", INRIA research report 3805, 1999), as GMP's `mpn_dc_sqrtrem` arranges
//! it: a magnitude `A` of `2n` limbs whose top limb is at least `B / 4`, for `B = 2^64`, is cut at
//! `l = n / 2` limbs into `A = A3 b^2 + A1 b + A0` with `b = B^l`, `A1` and `A0` of `l` limbs and
//! `A3` of the other `2h = 2(n - l)`, and
//!
//! 1. `(S', R') = sqrtrem(A3)`, by the same method;
//! 2. `(q, u) = divrem(R' b + A1, 2 S')`;
//! 3. `S = S' b + q` and `R = u b + A0 - q^2`;
//! 4. if `R < 0`: `R = R + 2 S - 1` and `S = S - 1`.
//!
//! `S` is then `floor(sqrt(A))` and `R = A - S^2`, so a root of `n` limbs costs a root of half of
//! them, a division of `n` limbs by `h` and a square of `l`: with divisions and squares of
//! Karatsuba's or Toom-Cook's complexity, that of a product of `n` limbs, up to a constant.
//!
//! The divisor `2 S'` of step 2 takes a limb more than `S'`, whose top bit is set: the division is
//! by `S'` instead, whose quotient `Q` is `2q` or `2q + 1`; `q = Q / 2`, and the remainder by
//! `S'` takes `S'` more when `Q` is odd. `R' <= 2 S'` may be a bit longer than `h` limbs; `S'` is
//! then subtracted from it first, and the quotient is `b` more.
//!
//! The step on halves of one limb, four limbs of magnitude, is unrolled in registers (see
//! [`sqrt_rem_wide`]): it ends the recursion, and roots the magnitudes of 17 to 32 bytes.

use super::super::div::limbs::{
    div_2by1, div_qr, div_qr_1, div_qr_2, invert_3by2, invert_limb, qr_itch,
};
use super::super::mul::limbs::{add_1, add_n, sqr, sqr_itch, sub_1, sub_n};
use super::{addmul_1, lshift, rshift};

/// `2^64` as an `f64`.
const TWO_64: f64 = 18_446_744_073_709_551_616.0;

/// The square root of `a`, rounded down, and the remainder: `(s, r)` with `s = floor(sqrt(a))` and
/// `r = a - s^2 <= 2s`.
///
/// The root of `a` in double precision, from its top 53 bits, is right to about 52 bits, within
/// `2^12` of the root, which is below `2^64`. One step of Newton's method, `s + (a - s^2) / 2s` with
/// `a - s^2` computed exactly as an integer, takes it to within about `2^-30`, and its floor is the
/// root but when the root is that close to an integer: the remainder, computed exactly, then tells
/// whether it is one too large or one too small, which a cold path corrects.
#[inline(always)]
pub(in super::super) fn sqrt_rem_u128(a: u128) -> (u64, u128) {
    let (hi, lo) = ((a >> 64) as u64, a as u64);
    if hi == 0 {
        let (s, r) = sqrt_rem_u64(lo);
        return (s, u128::from(r));
    }
    let sf = (hi as f64 * TWO_64 + lo as f64).sqrt();
    // The conversion saturates: a root rounded up to 2^64 is `u64::MAX`.
    let s0 = sf as u64;
    // `|a - s0^2| < 2^78`: `s0` is within `2^12 + 1` of the root, which is below `2^64`.
    let e = a.wrapping_sub(u128::from(s0) * u128::from(s0)) as i128;
    let ef = ((e >> 64) as i64) as f64 * TWO_64 + (e as u64) as f64;
    // `sf` is not below 1 but for `a = 0`, where `e` is 0 too.
    let step = (ef * (0.5 / sf.max(1.0))).floor() as i64;
    let s = (i128::from(s0) + i128::from(step)).clamp(0, i128::from(u64::MAX)) as u64;
    let r = a.wrapping_sub(u128::from(s) * u128::from(s)) as i128;
    if r < 0 || r > 2 * i128::from(s) {
        return corrected(a, s);
    }
    (s, r as u128)
}

/// The square root of the 256-bit `a`, given as limbs, the least significant first, of at least
/// `2^128`, rounded down: `s`, and with `REM` the remainder `a - s^2 <= 2s` as three limbs (zero
/// without `REM`), computed in registers.
///
/// This is one step of [`dc_sqrtrem`] with halves of one limb, unrolled: `a` shifted left by an
/// even number of bits `2c` so that its top limb is at least `B / 4`, the root and remainder of the
/// top two limbs by [`sqrt_rem_u128`], the division of the remainder and the next limb by twice
/// the root by [`div_2by1`], and the correction; then the root shifted back by `c` bits and the
/// remainder as [`sqrtrem`] finds it. Without `REM`, the correction is computed only when the bits
/// shifted out of the root cannot absorb it.
#[inline(always)]
pub(in super::super) fn sqrt_rem_wide<const REM: bool>(a: [u64; 4]) -> (u128, [u64; 3]) {
    let [a0, a1, a2, a3] = a;
    debug_assert!(a3 != 0 || a2 != 0);
    let (hi, lo) = (
        u128::from(a3) << 64 | u128::from(a2),
        u128::from(a1) << 64 | u128::from(a0),
    );
    let c = hi.leading_zeros() / 2;
    // `a 4^c`, its top limb at least `B / 4`: `2c < 128`.
    let sh = 2 * c;
    let (hi, lo) = (hi << sh | lo.checked_shr(128 - sh).unwrap_or(0), lo << sh);
    let (b1, b0) = ((lo >> 64) as u64, lo as u64);
    let (s1, r1) = sqrt_rem_u128(hi);
    // `(r1 B + b1) / s1 = Q`, as `r1 = top s1 + rr` with `top <= 2`, as `r1 <= 2 s1`.
    let d = u128::from(s1);
    let (top, rr): (u64, u128) = if r1 >= 2 * d {
        (2, r1 - 2 * d)
    } else if r1 >= d {
        (1, r1 - d)
    } else {
        (0, r1)
    };
    let (q0, u0) = div_2by1(rr as u64, b1, s1, invert_limb(s1));
    // `q = Q / 2`, at most `B`, and `u`, the remainder by `2 s1`, below `2 s1`.
    let q = (u128::from(top) << 64 | u128::from(q0)) >> 1;
    let u = u128::from(u0) + if q0 & 1 == 1 { d } else { 0 };
    // `S = s1 B + q`, which wraps to 0 only for `q = B` and `s1 = B - 1`, and is then always one
    // too large: the decrement below takes it back.
    let mut root = (d << 64).wrapping_add(q);
    let mask = (1u128 << c) - 1;
    if !REM && c >= 2 && root & mask & !1 != 0 {
        return (root >> c, [0; 3]);
    }
    // `R = u B + b0 - q^2`, three limbs, `q^2 = B^2` when `q = B`.
    let (qh, ql) = ((q >> 64) as u64, q as u64);
    let p = u128::from(ql) * u128::from(ql);
    let (r0, borrow0) = b0.overflowing_sub(p as u64);
    let (mid, borrow1) = (u as u64).overflowing_sub((p >> 64) as u64);
    let (mid, borrow2) = mid.overflowing_sub(u64::from(borrow0));
    let r2 = ((u >> 64) as u64)
        .wrapping_sub(qh)
        .wrapping_sub(u64::from(borrow1 | borrow2));
    let mut r = [r0, mid, r2];
    if (r2 as i64) < 0 {
        // `R + 2S - 1` and `S - 1`: `2 (S - 1) + 1` added.
        root = root.wrapping_sub(1);
        let add = [
            (root << 1) as u64 | 1,
            (root >> 63) as u64,
            (root >> 127) as u64,
        ];
        let mut carry = false;
        for (x, y) in r.iter_mut().zip(add) {
            let (v, c1) = x.overflowing_add(y);
            let (v, c2) = v.overflowing_add(u64::from(carry));
            *x = v;
            carry = c1 | c2;
        }
    }
    if !REM {
        return (root >> c, [0; 3]);
    }
    if c == 0 {
        return (root, r);
    }
    // `(R + 2 s0 S - s0^2) / 4^c` for `s0 = S mod 2^c`, below `2^63`.
    let s0 = root & mask;
    let m = 2 * s0 as u64;
    let low = u128::from(root as u64) * u128::from(m);
    let high = u128::from((root >> 64) as u64) * u128::from(m) + (low >> 64);
    let sq = s0 * s0;
    let mut acc = [0u64; 3];
    let mut carry = 0u128;
    for (i, (x, y)) in [
        (r[0], low as u64),
        (r[1], high as u64),
        (r[2], (high >> 64) as u64),
    ]
    .into_iter()
    .enumerate()
    {
        let t = u128::from(x) + u128::from(y) + carry;
        acc[i] = t as u64;
        carry = t >> 64;
    }
    let mut borrow = 0u64;
    for (i, y) in [sq as u64, (sq >> 64) as u64, 0].into_iter().enumerate() {
        let (v, b1) = acc[i].overflowing_sub(y);
        let (v, b2) = v.overflowing_sub(borrow);
        acc[i] = v;
        borrow = u64::from(b1 | b2);
    }
    // Shifted right by `2c < 128` bits.
    let value = u128::from(acc[1]) << 64 | u128::from(acc[0]);
    let top = acc[2];
    let rem = (value >> sh) | (u128::from(top) << (128 - sh));
    let rem_top = if sh >= 64 { 0 } else { top >> sh };
    (root >> c, [rem as u64, (rem >> 64) as u64, rem_top])
}

/// The square root of `a` rounded down and the remainder, as [`sqrt_rem_u128`]: the root of `a`
/// rounded to double precision is right to about 52 bits, so within one of the root, below
/// `2^32`, and its truncation is the root but when the root is that close to an integer.
#[inline(always)]
fn sqrt_rem_u64(a: u64) -> (u64, u64) {
    // The conversion saturates: no root above `2^32 - 1` is squared.
    let s = ((a as f64).sqrt() as u64).min(u64::from(u32::MAX));
    let r = a.wrapping_sub(s * s) as i64;
    if r < 0 || r as u64 > 2 * s {
        let (s, r) = corrected(u128::from(a), s);
        return (s, r as u64);
    }
    (s, r as u64)
}

/// The rare correction of [`sqrt_rem_u128`]: `s` is within a few units of the root of `a`, which
/// steps of one find.
#[cold]
#[inline(never)]
fn corrected(a: u128, mut s: u64) -> (u64, u128) {
    let mut r = a.wrapping_sub(u128::from(s) * u128::from(s)) as i128;
    while r < 0 {
        // (s - 1)^2 = s^2 - 2s + 1.
        r += 2 * i128::from(s) - 1;
        s -= 1;
    }
    while r > 2 * i128::from(s) {
        // (s + 1)^2 = s^2 + 2s + 1.
        s += 1;
        r -= 2 * i128::from(s) - 1;
    }
    (s, r as u128)
}

/// Divides the `nn` limbs at `n` by the normalized `dn < nn` limbs at `d`: sets the `nn - dn` limbs
/// at `q` to the quotient's low limbs, leaves the remainder in the low `dn` limbs at `n`, and
/// returns the quotient's limb above them, 0 or 1.
///
/// # Safety
///
/// The top `dn` limbs at `n` must be below `2d`; the limbs at `n` must be valid for reads and
/// writes, those at `d` for reads, those at `q` for writes and the [`qr_itch`]`(nn, dn)` at `t`
/// for reads and writes, all apart.
unsafe fn divide(
    q: *mut u64,
    n: *mut u64,
    nn: usize,
    d: *const u64,
    dn: usize,
    t: *mut u64,
) -> u64 {
    // SAFETY: as the caller guarantees.
    unsafe {
        match dn {
            1 => {
                let d0 = *d;
                let mut top = *n.add(nn - 1);
                let qh = top >= d0;
                if qh {
                    top -= d0;
                }
                *n = div_qr_1::<true>(q, n, nn - 1, top, d0, invert_limb(d0));
                u64::from(qh)
            }
            2 => {
                let (d1, d0) = (*d.add(1), *d);
                let (t1, t0) = (*n.add(nn - 1), *n.add(nn - 2));
                let qh = (t1, t0) >= (d1, d0);
                if qh {
                    let (low, borrow) = t0.overflowing_sub(d0);
                    *n.add(nn - 2) = low;
                    *n.add(nn - 1) = t1 - d1 - u64::from(borrow);
                }
                let (r1, r0) = div_qr_2(q, n, nn, d1, d0, invert_3by2(d1, d0));
                (*n, *n.add(1)) = (r0, r1);
                u64::from(qh)
            }
            _ => {
                let v = invert_3by2(*d.add(dn - 1), *d.add(dn - 2));
                u64::from(div_qr(q, n, nn, d, dn, v, t))
            }
        }
    }
}

/// Below this many limbs of root, [`dc_sqrtrem`] divides by the schoolbook method and squares by
/// it, neither of which takes scratch.
const SMALL_ITCH: usize = 64;

/// The limbs of scratch space that [`dc_sqrtrem`] needs for a root of `n` limbs: the quotient of
/// the division and its scratch, or the square and its scratch, or those of the root of the top
/// half, which come first.
pub(super) fn dc_itch(n: usize) -> usize {
    if n < SMALL_ITCH {
        // Neither divide and conquer, which takes 40 limbs of divisor, nor Karatsuba's square,
        // which takes 56: the quotient of `l` limbs or the square of `2l`.
        debug_assert!(qr_itch(n, n - n / 2) == 0 && sqr_itch(n / 2) == 0);
        return n;
    }
    let (l, h) = (n / 2, n - n / 2);
    let division = l + if h >= 3 { qr_itch(n, h) } else { 0 };
    dc_itch(h).max(division).max(2 * l + sqr_itch(l))
}

/// Zimmermann's Karatsuba square root (see the module documentation): sets the `n` limbs at `s` to
/// `floor(sqrt(A))` for the `2n` limbs `A` at `a`, whose top limb is at least `B / 4`, leaves the
/// low `n` limbs of the remainder `A - s^2` in the low `n` limbs at `a`, and returns its top limb,
/// 0 or 1, as the remainder is at most `2s`.
///
/// With `approx`, a mask of bits of the lowest limb of the root, it returns 1 early, the root set
/// but the remainder not computed, when the root before the last correction has one of those bits
/// set: the correction takes one off the root, which changes none of its bits above them, so the
/// root shifted right past them is known without the square of step 3, the last one. Only the
/// outermost call takes a mask.
///
/// # Safety
///
/// `n >= 1`; the limbs at `a` must be valid for reads and writes, those at `s` for writes and the
/// [`dc_itch`]`(n)` at `t` for reads and writes, all apart.
pub(super) unsafe fn dc_sqrtrem(
    s: *mut u64,
    a: *mut u64,
    n: usize,
    approx: u64,
    t: *mut u64,
) -> u64 {
    // SAFETY: every range lies in the `2n` limbs at `a`, the `n` at `s` and the scratch: the root
    // of the top half writes the top `h` limbs of `s` and reads the top `2h` of `a`; the division
    // reads limbs `l..l + n` of `a`, writes its remainder to limbs `l..l + h` and its quotient to
    // the first `l` of the scratch, after which come those of the division; the square of the low
    // `l` limbs of `s` is in the first `2l` of the scratch, with its own after them.
    unsafe {
        if n == 1 {
            let (root, r) = sqrt_rem_u128(u128::from(*a.add(1)) << 64 | u128::from(*a));
            *s = root;
            *a = r as u64;
            return (r >> 64) as u64;
        }
        if n == 2 {
            // The step on halves of one limb, unrolled; with the top limb at least `B / 4`, it
            // shifts nothing.
            let (root, r) = sqrt_rem_wide::<true>([*a, *a.add(1), *a.add(2), *a.add(3)]);
            (*s, *s.add(1)) = (root as u64, (root >> 64) as u64);
            (*a, *a.add(1)) = (r[0], r[1]);
            return r[2];
        }
        let l = n / 2;
        let h = n - l;
        // Step 1: `R'` in limbs `2l..2l + h` and `q`, `S'` in limbs `l..n` of `s`.
        let mut q = dc_sqrtrem(s.add(l), a.add(2 * l), h, 0, t);
        if q != 0 {
            // `R' >= B^h`: `R' - S' <= S'` fits the limbs, the borrow takes the bit above them,
            // and the quotient by `S'` is `b` more.
            sub_n(a.add(2 * l), a.add(2 * l), s.add(l), h);
        }
        // Step 2: `Q = (R' b + A1) / S'`, of `l` limbs and `q` above them, at most 2; the remainder
        // by `S'` in limbs `l..l + h`.
        q += divide(t, a.add(l), n, s.add(l), h, t.add(l));
        let odd = *t & 1;
        // `q`, the low limbs of `S = S' b + Q / 2`: the quotient shifted right by one bit, the
        // bit above its limbs at the top.
        for i in 0..l - 1 {
            *s.add(i) = (*t.add(i) >> 1) | (*t.add(i + 1) << 63);
        }
        *s.add(l - 1) = (*t.add(l - 1) >> 1) | ((q & 1) << 63);
        if *s & approx != 0 {
            return 1;
        }
        // `Q / 2 = b` exactly when `q` is 2: the root is then `S' b + b`, which the correction below
        // always takes back by one, as the remainder is negative.
        q >>= 1;
        // `u`, the remainder by `2 S'`, `S'` more when `Q` is odd, and the top bit of `u b + A0`
        // in limbs `0..n`.
        let mut c: i64 = 0;
        if odd != 0 {
            c = i64::from(add_n(a.add(l), a.add(l), s.add(l), h));
        }
        // Step 3: `R = u b + A0 - q^2`, where `q^2 = b^2` when `q` is `b`, and its low `l` limbs
        // are then zero.
        sqr(t, s, l, t.add(2 * l));
        let borrow = q + u64::from(sub_n(a, a, t, 2 * l));
        c -= if l == h {
            borrow as i64
        } else {
            sub_1(a.add(2 * l), 1, borrow) as i64
        };
        if c < 0 {
            // Step 4: `R + 2S - 1` and `S - 1`, with `S' b + b` first made whole: what carries
            // out of its `n` limbs is taken back by the decrement.
            let carry = add_1(s.add(l), h, q);
            c += i64::from(add_n(a, a, s, n)) + i64::from(add_n(a, a, s, n)) + 2 * carry as i64;
            c -= sub_1(a, n, 1) as i64;
            sub_1(s, n, 1);
        }
        c as u64
    }
}

/// The limbs of working memory that [`sqrtrem`] needs for a magnitude of `an` limbs: the magnitude
/// shifted, a limb above it, and the scratch of [`dc_sqrtrem`].
pub(super) fn sqrtrem_itch(an: usize) -> usize {
    // A root alone may take a limb more (see `sqrtrem`).
    let tn = an.div_ceil(2) + 1;
    2 * tn + 1 + dc_itch(tn)
}

/// From this many limbs of root, a root alone whose bits shifted out would be too few to spare the
/// last square of [`dc_sqrtrem`] is computed with a limb more: that of the magnitude with two zero
/// limbs below it, of which the root's lowest limb is shifted out. Measured on an Apple M4 Max, a
/// root of 4 KiB took 34,800 cycles so against 40,700, and one of 64 KiB 2.41 million against
/// 2.83 million; of 12 limbs 970 against 995, and of 8 limbs 760 against 729.
pub(super) const PAD: usize = 12;

/// Sets the `(an + 1) / 2` limbs at `s`, which has room for one more, to `floor(sqrt(A))` for the
/// `an >= 3` limbs `A` at `a`, whose top limb is not zero; with `REM`, also the limbs at `r` to the
/// remainder `A - s^2`, and returns their number without the zero limbs at the top, 0 for zero (at
/// most `(an + 1) / 2 + 1` are written). Without `REM`, returns 0 and `r` is not used.
///
/// `A` is shifted left by an even number of bits `2c` so that its top limb is at least `B / 4`, and
/// by a limb more when `an` is odd, for an even number of limbs: `A 4^k` for `k = c` or `c + 32`,
/// whose root `S` and remainder `R` [`dc_sqrtrem`] computes. The root of `A` is then `S` shifted
/// right by `k` bits, and with `s0 = S mod 2^k` its remainder is `(R + 2 s0 S - s0^2) / 4^k`, as
/// `A 4^k = (S - s0)^2 + 2 s0 S - s0^2 + R`. Without the remainder, the last square of
/// `dc_sqrtrem` is skipped when the bits shifted out of the root tell it is not needed, which
/// takes at least two of them: from [`PAD`] limbs, a root alone that would have fewer is computed
/// from `A` with two zero limbs below it, `k = c + 64`.
///
/// # Safety
///
/// The limbs at `a` must be valid for reads, those at `s` for writes, with `REM` those at `r` for
/// writes, and the [`sqrtrem_itch`]`(an)` at `w` for reads and writes, all apart.
pub(super) unsafe fn sqrtrem<const REM: bool>(
    s: *mut u64,
    r: *mut u64,
    a: *const u64,
    an: usize,
    w: *mut u64,
) -> usize {
    debug_assert!(an >= 3);
    // SAFETY: as the caller guarantees: the shifted magnitude takes the `2tn + 1` limbs at `w`,
    // the scratch of `dc_sqrtrem` the rest.
    unsafe {
        let c = (*a.add(an - 1)).leading_zeros() / 2;
        let odd = an & 1;
        let tn = an.div_ceil(2);
        // The zero limbs below the magnitude: one when `an` is odd; for a root alone two, when
        // fewer than two bits would be shifted out of the root otherwise.
        let low = if !REM && odd == 0 && c < 2 && tn >= PAD {
            2
        } else {
            odd
        };
        let n = (an + low) / 2;
        let (x, t) = (w, w.add(2 * n + 1));
        core::ptr::write_bytes(x, 0, low);
        lshift(x.add(low), a, an, 2 * c);
        let k = c + 32 * low as u32;
        let mask = if k < 64 { (1u64 << k) - 1 } else { u64::MAX };
        let approx = if REM { 0 } else { mask & !1 };
        let top = dc_sqrtrem(s, x, n, approx, t);
        if !REM {
            if k >= 64 {
                core::ptr::copy(s.add(1), s, n - 1);
            }
            if !k.is_multiple_of(64) {
                rshift(s, s, tn, k % 64);
            }
            return 0;
        }
        let rn = if k == 0 {
            core::ptr::copy_nonoverlapping(x, r, tn);
            *r.add(tn) = top;
            tn + 1
        } else {
            // `R + 2 s0 S - s0^2`, which is not negative, as it is `4^k` times the remainder.
            let s0 = *s & mask;
            let mut top = top + addmul_1(x, s, tn, 2 * s0);
            let p = u128::from(s0) * u128::from(s0);
            let (low, b0) = (*x).overflowing_sub(p as u64);
            let (high, b1) = (*x.add(1)).overflowing_sub((p >> 64) as u64);
            let (high, b2) = high.overflowing_sub(u64::from(b0));
            (*x, *x.add(1)) = (low, high);
            top -= sub_1(x.add(2), tn - 2, u64::from(b1 | b2));
            *x.add(tn) = top;
            rshift(s, s, tn, k);
            // Divided by `4^k`: shifted right by whole limbs, then by the bits left.
            let (whole, bits) = ((2 * k / 64) as usize, 2 * k % 64);
            let rn = tn + 1 - whole;
            if bits == 0 {
                core::ptr::copy_nonoverlapping(x.add(whole), r, rn);
            } else {
                rshift(r, x.add(whole), rn, bits);
            }
            rn
        };
        let mut rn = rn;
        while rn > 0 && *r.add(rn - 1) == 0 {
            rn -= 1;
        }
        rn
    }
}
