//! The division of magnitudes held as limbs: native `u64`s, the least significant first.
//!
//! Every divisor here is normalized: the top bit of its top limb is set, which the caller arranges
//! by shifting both operands left by the same number of bits. A quotient limb is then estimated
//! from the top limbs of the partial remainder without a hardware division, by a multiplication
//! with a precomputed reciprocal of the divisor's top limbs, after Möller and Granlund, "Improved
//! division by invariant integers" (IEEE Transactions on Computers, 2011):
//!
//! - [`div_2by1`] divides two limbs by one, with the reciprocal `floor((B^2 - 1) / d) - B` of
//!   [`invert_limb`], for divisors of one limb.
//! - [`div_3by2`] divides three limbs by two, with the reciprocal `floor((B^3 - 1) / d) - B` of
//!   [`invert_3by2`]; its quotient is exact for divisors of two limbs, and for longer divisors it is
//!   the quotient limb of the schoolbook method (Knuth's Algorithm D), at most one too large.
//!
//! Divisors of at least [`DC_DIV`] limbs are divided by the divide-and-conquer method of Burnikel
//! and Ziegler, as GMP's `mpn_dcpi1_div_qr` carries it out: a quotient of `n` limbs by a divisor
//! of `n` is computed as two halves, each a division of `n` limbs by the top `n / 2` of the divisor
//! followed by a multiplication that corrects the partial remainder, so the work is that of the
//! multiplication, O(n^1.58) with Karatsuba's method instead of O(n^2). [`div_qr`] hands the longest
//! divisions to Newton's method (see [`newton`](super::newton)).

use super::super::mul::limbs::{add_1, add_n, cmp_n, mul, mul_itch, sub_1, sub_n};
use super::kernels::submul_1;
use super::newton::{mu, mu_div_q, mu_div_qr};
use core::cmp::Ordering;

/// From this many limbs in the divisor, a division takes the divide-and-conquer method: measured on
/// an Apple M4 Max, a division of 80 limbs by 40 took 3222 cycles so against 3323 by the
/// schoolbook method, and one of 200 limbs by 40, 12,023 against 12,441; the methods were as fast
/// from 30 to 36 limbs.
pub(in super::super) const DC_DIV: usize = 40;

/// From this many limbs in the divisor, the low limbs of a quotient alone come from the top ones
/// ([`div_q_split`]) instead of a shrinking divisor ([`shrinking`]): measured on an Apple M4 Max
/// with Toom-Cook products, a quotient of 320 limbs by 320 took 63,165 cycles so, against 75,572,
/// and one of 256 limbs by 256 44,164 against 50,600, while one of 192 limbs by 192 took 30,273
/// against 27,684. GMP's threshold for its own pair of methods on an Apple M1 is 242.
pub(super) const DC_DIV_Q: usize = 256;

/// From this many limbs in the dividend, a division by one limb folds the limbs in (see
/// [`div_qr_1`]): measured on an Apple M4 Max, 20 limbs took 150 cycles so against 153 limb by
/// limb, and 32 limbs 231 against 295.
const FOLD_1: usize = 20;

/// The table of [`invert_limb`]: `floor((2^19 - 3 * 2^8) / d9)` for `d9` from 256 to 511, the top
/// nine bits of a normalized limb.
const RECIPROCALS: [u16; 256] = {
    let mut t = [0; 256];
    let mut i = 0;
    while i < 256 {
        t[i] = (((1 << 19) - 3 * (1 << 8)) / (256 + i as u32)) as u16;
        i += 1;
    }
    t
};

/// The reciprocal of the normalized limb `d`, `floor((B^2 - 1) / d) - B`: a table lookup on its top
/// nine bits and three Newton steps, Algorithm 2 of Möller and Granlund, with no division.
#[inline]
pub(in super::super) fn invert_limb(d: u64) -> u64 {
    debug_assert!(d >> 63 == 1);
    let d0 = d & 1;
    let d9 = d >> 55;
    let d40 = (d >> 24) + 1;
    let d63 = (d >> 1) + d0;
    let v0 = u64::from(RECIPROCALS[(d9 - 256) as usize]);
    let v1 = (v0 << 11) - ((v0 * v0 * d40) >> 40) - 1;
    let v2 = (v1 << 13) + ((v1 * ((1 << 60) - v1 * d40)) >> 47);
    let e = ((v2 >> 1) & d0.wrapping_neg()).wrapping_sub(v2.wrapping_mul(d63));
    let v3 = (v2 << 31).wrapping_add(((u128::from(v2) * u128::from(e)) >> 65) as u64);
    let p = u128::from(v3) * u128::from(d) + u128::from(d);
    v3.wrapping_sub(((p >> 64) as u64).wrapping_add(d))
}

/// The reciprocal of the normalized two-limb divisor `d1 B + d0`, `floor((B^3 - 1) / (d1 B + d0)) -
/// B`: the reciprocal of `d1`, corrected for `d0` (Algorithm 6 of Möller and Granlund).
#[inline]
pub(in super::super) fn invert_3by2(d1: u64, d0: u64) -> u64 {
    let mut v = invert_limb(d1);
    let mut p = d1.wrapping_mul(v).wrapping_add(d0);
    if p < d0 {
        v = v.wrapping_sub(1);
        if p >= d1 {
            v = v.wrapping_sub(1);
            p = p.wrapping_sub(d1);
        }
        p = p.wrapping_sub(d1);
    }
    let t = u128::from(d0) * u128::from(v);
    let (t1, t0) = ((t >> 64) as u64, t as u64);
    let (p, carry) = p.overflowing_add(t1);
    if carry {
        v = v.wrapping_sub(1);
        if p >= d1 && (p > d1 || t0 >= d0) {
            v = v.wrapping_sub(1);
        }
    }
    v
}

/// Divides `u1 B + u0` by the normalized limb `d` with its reciprocal `v`: returns the quotient and
/// the remainder. Algorithm 4 of Möller and Granlund: one multiplication gives a quotient that is
/// right or one too large, which a comparison corrects without a branch; a second correction is
/// rare, and is a branch, which the core predicts: as a select it would lengthen the path from one
/// remainder to the next by three instructions, 12 cycles a limb instead of 10 on an Apple M4.
///
/// `u1 < d`, so that the quotient is one limb.
#[inline(always)]
pub(in super::super) fn div_2by1(u1: u64, u0: u64, d: u64, v: u64) -> (u64, u64) {
    debug_assert!(u1 < d);
    let p =
        (u128::from(v) * u128::from(u1)).wrapping_add((u128::from(u1 + 1) << 64) | u128::from(u0));
    let (q, q0) = ((p >> 64) as u64, p as u64);
    let r = u0.wrapping_sub(q.wrapping_mul(d));
    let mask = u64::from(r > q0).wrapping_neg();
    let (q, r) = (q.wrapping_add(mask), r.wrapping_add(mask & d));
    if r >= d {
        return once_more(q, r, d);
    }
    (q, r)
}

/// The rare second correction of [`div_2by1`], out of line, so that the compiler keeps it a
/// branch and does not select its result.
#[cold]
#[inline(never)]
fn once_more(q: u64, r: u64, d: u64) -> (u64, u64) {
    (q + 1, r - d)
}

/// Divides `u2 B^2 + u1 B + u0` by the normalized `d1 B + d0` with its reciprocal `v`: returns the
/// quotient and the two limbs of the remainder. Algorithm 5 of Möller and Granlund.
///
/// `u2 B + u1 < d1 B + d0`, so that the quotient is one limb.
#[inline(always)]
pub(super) fn div_3by2(u2: u64, u1: u64, u0: u64, d1: u64, d0: u64, v: u64) -> (u64, u64, u64) {
    let p = (u128::from(v) * u128::from(u2)).wrapping_add((u128::from(u2) << 64) | u128::from(u1));
    let (q, q0) = ((p >> 64) as u64, p as u64);
    let d = (u128::from(d1) << 64) | u128::from(d0);
    let r1 = u1.wrapping_sub(d1.wrapping_mul(q));
    let r = ((u128::from(r1) << 64) | u128::from(u0))
        .wrapping_sub(d)
        .wrapping_sub(u128::from(d0) * u128::from(q));
    let q = q.wrapping_add(1);
    // The estimate is right or one too large, which the top limb of the remainder tells; it is
    // corrected without a branch.
    let mask = u64::from((r >> 64) as u64 >= q0).wrapping_neg();
    let wide = (u128::from(mask) << 64) | u128::from(mask);
    let (mut q, mut r) = (q.wrapping_add(mask), r.wrapping_add(d & wide));
    if r >= d {
        (q, r) = once_more_2(q, r, d);
    }
    (q, (r >> 64) as u64, r as u64)
}

/// The rare second correction of [`div_3by2`], out of line as [`once_more`] is.
#[cold]
#[inline(never)]
fn once_more_2(q: u64, r: u128, d: u128) -> (u64, u128) {
    (q + 1, r - d)
}

/// Sets the `n` limbs at `q` to the quotient of `top B^n + u` by the normalized limb `d`, for the
/// `n` limbs at `u` and `top < d`, and returns the remainder; without `QUOTIENT`, only returns the
/// remainder, and writes nothing.
///
/// Dividing limb by limb with [`div_2by1`] makes each step wait for the remainder of the one
/// before, through two multiplications and a correction, about 10 cycles on an Apple M4. From
/// [`FOLD_1`] limbs, the limbs are folded in instead, by Möller's method of GMP's
/// `mpn_div_qr_1n_pi1` (and of its `mpn_mod_1_1p` for the remainder alone): with
/// `B^2 = (B + v) d + b2`, where `1 <= b2 <= d`, a partial remainder `r1 B + r0` takes the next
/// limb `u` as
///
/// `(r1 B + r0) B + u = r1 (B + v) d + (r0 B + u + r1 b2)`,
///
/// so the next partial remainder is a two-limb sum, `r0 B + u` plus the product `r1 b2`, and the
/// quotient gains `r1 (B + v)`, whose product `r1 v` is not on the remainder's path. A sum that
/// carries out of its two limbs, `B^2` more, is folded into the next step as `b2` more and `B + v`
/// more of the quotient. A step then waits for one multiplication and two additions, and takes
/// about 7.6 cycles; but it does more work besides, which makes it the slower one below
/// [`FOLD_1`] limbs.
///
/// # Safety
///
/// The limbs at `u` must be valid for reads, and with `QUOTIENT` those at `q` for writes; `q` may
/// be `u`.
pub(in super::super) unsafe fn div_qr_1<const QUOTIENT: bool>(
    q: *mut u64,
    u: *const u64,
    n: usize,
    top: u64,
    d: u64,
    v: u64,
) -> u64 {
    debug_assert!(top < d);
    // SAFETY: as the caller guarantees; limbs `j..` of `u` are read before limbs `j + 1..` of `q`
    // are written.
    unsafe {
        if n < FOLD_1 {
            let mut r = top;
            for i in (0..n).rev() {
                let (qi, ri) = div_2by1(r, *u.add(i), d, v);
                if QUOTIENT {
                    *q.add(i) = qi;
                }
                r = ri;
            }
            return r;
        }
        let b2 = d.wrapping_mul(v).wrapping_neg();
        // `top B^n` folded into the top two limbs: `top (B + v)` of the quotient at limb `n - 2`,
        // whose high limb, `top + hi(top v)`, cannot carry, and `top b2` of the remainder.
        let t = u128::from(top) * u128::from(v);
        // The quotient limb that later steps may still carry into, kept in a register, and the
        // low limb of the last product `r1 v`, which the next step adds one limb up.
        let mut pending = (t >> 64) as u64 + top;
        let mut low = t as u64;
        let sum = (u128::from(*u.add(n - 1)) << 64 | u128::from(*u.add(n - 2)))
            .overflowing_add(u128::from(top) * u128::from(b2));
        let (mut r, mut carry) = sum;
        for j in (0..n - 2).rev() {
            let (r1, r0) = ((r >> 64) as u64, r as u64);
            // The carry out of the last sum, `B^2`: `b2` into the low limb, `B + v` into the
            // quotient. When `b2` carries out of the low limb, that limb is `d` less, and the
            // quotient one more one limb up.
            let (r0, c) = r0.overflowing_add(if carry { b2 } else { 0 });
            let r0 = if c { r0.wrapping_sub(d) } else { r0 };
            if QUOTIENT {
                let p = u128::from(r1) * u128::from(v);
                // Limb `j + 1` of the quotient and what carries out of it into limb `j + 2`:
                // `r1 + hi(r1 v)`, the last step's `lo`, and `v` with the `B` above it for a carry.
                let s = u128::from(r1)
                    + u128::from(low)
                    + u128::from((p >> 64) as u64 + u64::from(c))
                    + if carry { (1 << 64) | u128::from(v) } else { 0 };
                low = p as u64;
                let (above, over) = pending.overflowing_add((s >> 64) as u64);
                *q.add(j + 2) = above;
                if over {
                    add_1(q.add(j + 3), n - j - 3, 1);
                }
                pending = s as u64;
            }
            (r, carry) = (u128::from(r0) << 64 | u128::from(*u.add(j)))
                .overflowing_add(u128::from(r1) * u128::from(b2));
        }
        // The last partial remainder, `carry B^2 + r1 B + r0`, is below `3 d B`: `d B` off for the
        // carry and once more if `r1` is still `d` or more, each one more quotient limb at limb 1,
        // then the last two limbs divided.
        let (mut r1, r0) = ((r >> 64) as u64, r as u64);
        let mut up = u64::from(carry);
        if carry {
            r1 = r1.wrapping_sub(d);
        }
        if r1 >= d {
            r1 -= d;
            up += 1;
        }
        let (t, rem) = div_2by1(r1, r0, d, v);
        if QUOTIENT {
            let (q0, c) = low.overflowing_add(t);
            *q = q0;
            let (q1, over) = pending.overflowing_add(up + u64::from(c));
            *q.add(1) = q1;
            if over {
                add_1(q.add(2), n - 2, 1);
            }
        }
        rem
    }
}

/// Sets the `n - 2` limbs at `q` to the quotient of the `n` limbs at `u` by the normalized
/// `d1 B + d0`, and returns the remainder's two limbs, the high one first.
///
/// # Safety
///
/// `n >= 2`, and the top two limbs of `u` below `d1 B + d0`; the limbs at `u` must be valid for
/// reads and those at `q` for writes.
pub(in super::super) unsafe fn div_qr_2(
    q: *mut u64,
    u: *const u64,
    n: usize,
    d1: u64,
    d0: u64,
    v: u64,
) -> (u64, u64) {
    // SAFETY: as the caller guarantees.
    unsafe {
        let (mut r1, mut r0) = (*u.add(n - 1), *u.add(n - 2));
        for i in (0..n - 2).rev() {
            let (qi, a, b) = div_3by2(r1, r0, *u.add(i), d1, d0, v);
            *q.add(i) = qi;
            (r1, r0) = (a, b);
        }
        (r1, r0)
    }
}

/// The schoolbook division, Knuth's Algorithm D with the quotient limbs of [`div_3by2`], after
/// GMP's `mpn_sbpi1_div_qr`: sets the `nn - dn` limbs at `q` to the quotient of the `nn` limbs at
/// `n` by the `dn` limbs at `d`, leaves the remainder in the low `dn` limbs of `n`, and returns the
/// quotient's limb above them, 0 or 1. `v` is the reciprocal of the divisor's top two limbs.
///
/// # Safety
///
/// `nn >= dn > 2`, the divisor normalized; the limbs at `n` must be valid for reads and writes,
/// those at `d` for reads and those at `q` for writes, all apart.
#[inline(never)]
pub(super) unsafe fn sb_div_qr(
    q: *mut u64,
    n: *mut u64,
    nn: usize,
    d: *const u64,
    dn: usize,
    v: u64,
) -> bool {
    debug_assert!(dn > 2 && nn >= dn);
    // SAFETY: as the caller guarantees: each step reads and writes the `dn + 1` limbs of `n` from
    // `i`, below `nn`, and writes quotient limb `i`, below `nn - dn`.
    unsafe {
        let top = n.add(nn - dn);
        let qh = cmp_n(top, d, dn) != Ordering::Less;
        if qh {
            sub_n(top, top, d, dn);
        }
        // The top limb of the partial remainder, kept in a register from step to step.
        let n1 = sb_steps(q, n, d, dn, v, 0..nn - dn, *n.add(nn - 1));
        *n.add(dn - 1) = n1;
        qh
    }
}

/// The steps of [`sb_div_qr`] for quotient limbs `steps`, from the top, at `q`, on the partial
/// remainder in the limbs at `n`, whose top limb `n1` is kept in a register: returns it.
///
/// # Safety
///
/// As for `sb_div_qr`, for the limbs of `n` below `steps.end + dn`, the top one `n1`.
#[inline(always)]
unsafe fn sb_steps(
    q: *mut u64,
    n: *mut u64,
    d: *const u64,
    dn: usize,
    v: u64,
    steps: core::ops::Range<usize>,
    mut n1: u64,
) -> u64 {
    // SAFETY: as the caller guarantees: each step reads and writes the `dn + 1` limbs of `n` from
    // `i`, and writes quotient limb `i`.
    unsafe {
        let (d1, d0) = (*d.add(dn - 1), *d.add(dn - 2));
        for i in steps.rev() {
            let n2 = *n.add(i + dn - 1);
            let qi = if n1 == d1 && n2 == d0 {
                // The top two limbs equal the divisor's, which `div_3by2` cannot divide: the
                // quotient limb is then `B - 1`, and the top limb cancels.
                submul_1(n.add(i), d, dn, u64::MAX);
                n1 = *n.add(i + dn - 1);
                u64::MAX
            } else {
                let (mut qi, mut r1, r0) = div_3by2(n1, n2, *n.add(i + dn - 2), d1, d0, v);
                // The quotient limb times the divisor's other limbs, from the partial remainder
                // below the two limbs `div_3by2` has accounted for.
                let borrow = submul_1(n.add(i), d, dn - 2, qi);
                let (r0, b0) = r0.overflowing_sub(borrow);
                let b1;
                (r1, b1) = r1.overflowing_sub(u64::from(b0));
                *n.add(i + dn - 2) = r0;
                if b1 {
                    // The quotient limb was one too large: the divisor is added back.
                    let carry = add_n(n.add(i), n.add(i), d, dn - 1);
                    r1 = r1.wrapping_add(d1).wrapping_add(u64::from(carry));
                    qi -= 1;
                }
                n1 = r1;
                qi
            };
            *q.add(i) = qi;
        }
        n1
    }
}

/// The exact division of [`sb_div_qr`], by divide and conquer, [`dc_div_qr`], for a divisor of at
/// least [`DC_DIV`] limbs and a quotient of at least three, and by Newton's method,
/// [`mu_div_qr`], for the longest (see [`mu`]).
///
/// # Safety
///
/// As for `sb_div_qr`, with the [`qr_itch`] limbs at `t` valid for reads and writes, apart from the
/// others.
#[inline(always)]
pub(in super::super) unsafe fn div_qr(
    q: *mut u64,
    n: *mut u64,
    nn: usize,
    d: *const u64,
    dn: usize,
    v: u64,
    t: *mut u64,
) -> bool {
    // SAFETY: as the caller guarantees.
    unsafe {
        if mu(nn, dn) {
            mu_div_qr(q, n, nn, d, dn)
        } else if dn >= DC_DIV && nn >= dn + 3 {
            dc_div_qr(q, n, nn, d, dn, v, t)
        } else {
            sb_div_qr(q, n, nn, d, dn, v)
        }
    }
}

/// [`div_qr`] by the schoolbook method or divide and conquer alone, never Newton's method, whose
/// inverse divides by it: with the [`dc_itch`] limbs at `t` when `dn >= DC_DIV` and `nn >= dn + 3`.
///
/// # Safety
///
/// As for [`div_qr`], with that scratch.
pub(super) unsafe fn div_qr_classic(
    q: *mut u64,
    n: *mut u64,
    nn: usize,
    d: *const u64,
    dn: usize,
    v: u64,
    t: *mut u64,
) -> bool {
    // SAFETY: as the caller guarantees.
    unsafe {
        if dn >= DC_DIV && nn >= dn + 3 {
            dc_div_qr(q, n, nn, d, dn, v, t)
        } else {
            sb_div_qr(q, n, nn, d, dn, v)
        }
    }
}

/// The limbs of scratch space that [`div_qr`] needs for a dividend of `nn` limbs and a divisor of
/// `dn`: none for Newton's method, which allocates its own.
pub(in super::super) fn qr_itch(nn: usize, dn: usize) -> usize {
    if mu(nn, dn) {
        0
    } else if dn >= DC_DIV && nn >= dn + 3 {
        dc_itch(nn, dn)
    } else {
        0
    }
}

/// The last limbs of a quotient alone of a dividend of `nn` limbs by a divisor of `dn` that
/// [`shrinking`] computes: up to `dn - 1`.
pub(super) fn shrink_m(nn: usize, dn: usize) -> usize {
    (nn - dn).min(dn - 1)
}

/// The last `m` limbs of a quotient alone, at `q`, from a divisor that drops a low limb at each
/// step, after the idea of GMP's `mpn_sbpi1_div_q`: the limbs above them divided exactly, the
/// partial remainder `R < B^m D` is in the `m + dn` limbs at `n`, which are lost. Returns `false`,
/// rarely, when a step finds a partial remainder too large for its divisor, the limbs at `q` then
/// unset.
///
/// Quotient limb `i < m` is that of a step of [`sb_div_qr`] on the top `i + 2` limbs of the
/// divisor, `D_i`, and the top `i + 3` of the partial remainder: every step reads and writes the
/// limbs from `c = dn - 2` up, and the products of a quotient limb by the divisor's limbs that it
/// drops are never subtracted. Proof that the quotient is exact, but when the compensation below
/// finds it one too large:
///
/// - A step on the `i + 3` limbs `R_i` by `D_i` is one of `sb_div_qr`'s when the top two limbs of
///   `R_i` are below `D_i`'s, which this one checks, giving up otherwise: then
///   `R_i < (d1 B + d0) B^(i + 1) <= B D_i`, so its quotient limb is below `B`, `div_3by2`'s
///   estimate is at most one too large, and the step leaves `0 <= R_i - q_i D_i < D_i`, the top
///   `i + 2` limbs of the next step.
/// - The last step, on `D_0 = d1 B + d0`, leaves `R_a = (n1 B + n0) B^c <= (D_0 - 1) B^c`. The
///   remainder of the quotient `Q_a` of the steps is `R_t = R - Q_a D = R_a + (R mod B^c) - S`,
///   where `S` sums the dropped products `q_i (D mod B^(c - i)) B^i`, each below `B^(c + 1)`, for
///   the `e = min(m, c)` limbs `i < c` that drop any: `0 <= S < e B^(c + 1)`.
/// - So `R_t < R_a + B^c <= D_0 B^c <= D`, and `Q_a` is not too small; and `R_t > -S > -D`, as
///   `D >= B^dn / 2 > e B^(c + 1)`: `Q_a` is at most one too large, exactly when `R_t < 0`.
/// - `R_t >= R_a - S > (n1 - e) B^(c + 1)`: with `n1 >= e`, `Q_a` is the quotient. Otherwise, the
///   compensation subtracts the dropped products from `R_a + (R mod B^c)`, the low `c` limbs never
///   touched by the steps, and `Q_a` is one too large if they borrow out of the top. It is then not
///   zero, as the quotient of `R` is not negative, and taking one off stays inside its `m` limbs.
///
/// # Safety
///
/// `m < dn` and `dn > 2`, the divisor normalized; the limbs at `n` must be valid for reads and
/// writes, those at `d` for reads and those at `q` for writes, all apart.
#[inline(never)]
pub(super) unsafe fn shrinking(
    q: *mut u64,
    n: *mut u64,
    m: usize,
    d: *const u64,
    dn: usize,
    v: u64,
) -> bool {
    // SAFETY: as the caller guarantees.
    unsafe { shrink_steps(q, n, m, d, dn, v, *n.add(m + dn - 1)) }
}

/// The steps of [`shrinking`], with the partial remainder's top limb `n1` in a register.
///
/// # Safety
///
/// As for `shrinking`, with `n1` the top limb of the partial remainder.
#[inline(always)]
unsafe fn shrink_steps(
    q: *mut u64,
    n: *mut u64,
    m: usize,
    d: *const u64,
    dn: usize,
    v: u64,
    mut n1: u64,
) -> bool {
    debug_assert!(dn > 2 && m < dn);
    // SAFETY: as the caller guarantees: step `i` reads and writes the `i + 3` limbs of `n` from
    // `c`, below `m + dn`, and writes quotient limb `i`, below `m`; the compensation reads and
    // writes the limbs below `dn`.
    unsafe {
        let (d1, d0) = (*d.add(dn - 1), *d.add(dn - 2));
        let c = dn - 2;
        for i in (0..m).rev() {
            let n2 = *n.add(i + dn - 1);
            if n1 > d1 || (n1 == d1 && n2 >= d0) {
                return false;
            }
            let (mut qi, mut r1, r0) = div_3by2(n1, n2, *n.add(i + dn - 2), d1, d0, v);
            // The divisor's limbs from `c - i` below its top two, from the partial remainder's
            // limbs from `c`.
            let borrow = if i > 0 {
                submul_1(n.add(c), d.add(c - i), i, qi)
            } else {
                0
            };
            let (r0, b0) = r0.overflowing_sub(borrow);
            let b1;
            (r1, b1) = r1.overflowing_sub(u64::from(b0));
            *n.add(i + dn - 2) = r0;
            if b1 {
                let carry = add_n(n.add(c), n.add(c), d.add(c - i), i + 1);
                r1 = r1.wrapping_add(d1).wrapping_add(u64::from(carry));
                qi -= 1;
            }
            n1 = r1;
            *q.add(i) = qi;
        }
        let e = m.min(c);
        if n1 < e as u64 {
            *n.add(c + 1) = n1;
            for i in 0..e {
                let borrow = submul_1(n.add(i), d, c - i, *q.add(i));
                let (r0, b0) = (*n.add(c)).overflowing_sub(borrow);
                let (r1, b1) = (*n.add(c + 1)).overflowing_sub(u64::from(b0));
                (*n.add(c), *n.add(c + 1)) = (r0, r1);
                if b1 {
                    // Below zero, and so for the rest: the quotient is one too large.
                    sub_1(q, m, 1);
                    break;
                }
            }
        }
        true
    }
}

/// The quotient alone of [`sb_div_qr`]: its top limbs by `sb_div_qr`'s steps, and its last
/// [`shrink_m`] by those of [`shrinking`], in one function; the limbs at `n` are lost. Returns the
/// quotient's limb above the `nn - dn` at `q`, or `None` when `shrinking` would give up.
///
/// # Safety
///
/// As for `sb_div_qr`.
#[inline(never)]
unsafe fn sb_div_q(
    q: *mut u64,
    n: *mut u64,
    nn: usize,
    d: *const u64,
    dn: usize,
    v: u64,
) -> Option<bool> {
    let (qn, m) = (nn - dn, shrink_m(nn, dn));
    // SAFETY: as the caller guarantees.
    unsafe {
        let top = n.add(qn);
        let qh = cmp_n(top, d, dn) != Ordering::Less;
        if qh {
            sub_n(top, top, d, dn);
        }
        let n1 = sb_steps(q, n, d, dn, v, m..qn, *n.add(nn - 1));
        shrink_steps(q, n, m, d, dn, v, n1).then_some(qh)
    }
}

/// The quotient alone of [`div_qr`]: its top limbs by `div_qr`, and its last [`shrink_m`] from a
/// shrinking divisor, by [`sb_div_q`] or [`dc_div_qr`] and [`shrinking`]; the limbs at `n` are
/// lost. Returns the quotient's limb above the `nn - dn` at `q`, or `None`, rarely, when
/// `shrinking` gives up: the quotient is then to be computed by `div_qr` from the dividend again.
///
/// # Safety
///
/// As for `div_qr`, with the [`q_itch`] limbs at `t`.
#[inline(always)]
pub(super) unsafe fn div_q(
    q: *mut u64,
    n: *mut u64,
    nn: usize,
    d: *const u64,
    dn: usize,
    v: u64,
    t: *mut u64,
) -> Option<bool> {
    let m = shrink_m(nn, dn);
    // SAFETY: as the caller guarantees.
    unsafe {
        if dn >= DC_DIV && nn - m >= dn + 3 {
            let qh = div_qr(q.add(m), n.add(m), nn - m, d, dn, v, t);
            shrinking(q, n, m, d, dn, v).then_some(qh)
        } else {
            sb_div_q(q, n, nn, d, dn, v)
        }
    }
}

/// The limbs of scratch space that [`div_q`] needs for a dividend of `nn` limbs and a divisor of
/// `dn`.
pub(super) fn q_itch(nn: usize, dn: usize) -> usize {
    qr_itch(nn - shrink_m(nn, dn), dn)
}

/// The quotient alone is computed from the top limbs of the operands by [`div_q_top`] when the
/// divisor has at least this many limbs more than twice the quotient's, and two: below, dividing
/// the top limbs costs about what the whole division does.
const TOP: usize = 4;

/// Whether the quotient alone of a dividend of `nn` limbs, the top one below the divisor's, by a
/// divisor of `dn` is computed by [`div_q_top`]: the divisor has [`TOP`] limbs more than twice the
/// quotient's `nn - dn`, and two.
pub(super) fn from_top(nn: usize, dn: usize) -> bool {
    dn >= 2 * (nn - dn) + 2 + TOP
}

/// The limbs of scratch space that [`div_q_top`] needs for a quotient of `k` limbs, after its
/// operands.
pub(super) fn top_itch(k: usize) -> usize {
    if mu(2 * k + 2, k + 1) {
        // Newton's method allocates its own.
        0
    } else if k + 1 < DC_DIV_Q {
        // The copy of the top limbs, and the exact division of them or of their own top limbs.
        (2 * k + 2) + qr_itch(2 * k + 2, k + 1).max(q_itch(2 * k + 2, k + 1))
    } else {
        match split_low(2 * k + 2, k + 1) {
            0 => qr_itch(2 * k + 2, k + 1),
            l => split_itch(2 * k + 2, k + 1, l),
        }
    }
}

/// The limbs of scratch space that [`check_q_top`] needs for a dividend of `nn` limbs and a divisor
/// of `dn`.
pub(super) fn check_itch(nn: usize, dn: usize) -> usize {
    nn + mul_itch(dn, nn - dn)
}

/// What [`div_q_top`] found of the quotient: its top limbs are the quotient, or the quotient or one
/// more, which [`check_q_top`] tells, or the quotient is all ones.
#[derive(Clone, Copy, PartialEq)]
pub(super) enum Found {
    Exact,
    Check,
    AllOnes,
}

/// The quotient of a dividend of `nn` limbs by a divisor of `dn`, for a quotient of `k = nn - dn`
/// limbs much shorter than the divisor, from the top limbs of both: the lower limbs of the divisor
/// change the quotient only rarely, and only by one. Sets the `k + 1` limbs at `h` to the quotient
/// of the `2k + 2` limbs at `w`, the top ones of the dividend, which it clobbers, by the `k + 1` at
/// `dt`, the top ones of the divisor, and returns what the top `k` of them, `H`, are.
///
/// The top `2k + 2` limbs `N'` of the dividend `n` are divided exactly by the top `k + 1` limbs
/// `D'` of the divisor `d`, by the schoolbook method or divide and conquer, for a quotient `Q'`,
/// `H B + g` with a guard limb `g` below the quotient's `k`. With `t = dn - k - 1` limbs of the
/// divisor left out, and `t - 1` of the dividend, `N' B^(t - 1) <= n < (N' + 1) B^(t - 1)` and
/// `D' B^t <= d < (D' + 1) B^t`, so the true quotient `Q = floor(n / d)` satisfies
///
/// `Q B <= n B / d < (N' + 1) / D' <= Q' + 1`, and
/// `(Q + 1) B > n B / d >= N' / (D' + 1) > Q' - 2`,
///
/// the last as `N' / (D' + 1) = Q' + f - (Q' + f) / (D' + 1)` for the fraction `f` of `N' / D'`,
/// and `Q' < 2 D' + 2`: `D'` is normalized, `D' >= B^(k + 1) / 2`, and the top `k + 1` limbs of
/// `N'` are at most `D'`, as the dividend's top limbs are below the divisor's, so
/// `Q' < (D' + 1) B^(k + 1) / D' <= B^(k + 1) + 2`. So `Q B <= Q' < (Q + 1) B + 2`:
///
/// - `g >= 2`: `H = Q`.
/// - `g <= 1`: `H` is `Q` or `Q + 1`, which [`check_q_top`] tells. For random operands a chance of
///   2 in `2^64`; for a dividend that is a multiple of the divisor it is usual, and costs a product.
/// - `Q'` carries out of its `k + 1` limbs, `Q' >= B^(k + 1)`, possible only when the top `k + 1`
///   limbs of `N'` equal `D'`: as `Q < B^k`, the bounds leave `Q = B^k - 1`.
///
/// # Safety
///
/// `k >= 1`, `D'` normalized, the top `k + 1` limbs of `N'` at most `D'`, and the quotient below
/// `B^k`; `v` the reciprocal of `D'`'s top two limbs; the limbs at `w` and `dt` valid for reads,
/// `w`'s for writes too, those at `h` for writes, and the [`top_itch`] limbs at `s` for reads and
/// writes, all apart.
pub(super) unsafe fn div_q_top(
    h: *mut u64,
    w: *mut u64,
    k: usize,
    dt: *const u64,
    v: u64,
    s: *mut u64,
) -> Found {
    // SAFETY: as the caller guarantees.
    unsafe {
        let top = w.add(k + 1);
        if cmp_n(top, dt, k + 1) != Ordering::Less {
            return Found::AllOnes;
        }
        if k == 1 {
            div_qr_2(h, w, 4, *dt.add(1), *dt, v);
        } else {
            // Only the quotient of the top limbs counts, exactly: by Newton's method for the
            // longest; below `DC_DIV_Q` limbs, its last limbs from a shrinking divisor, on a copy,
            // and from the limbs themselves when that gives up; from it, its low limbs from the top
            // ones in turn, when there are enough.
            let high = if mu(2 * k + 2, k + 1) {
                mu_div_q(h, w, 2 * k + 2, dt, k + 1)
            } else if k + 1 < DC_DIV_Q {
                let (c, s) = (s, s.add(2 * k + 2));
                core::ptr::copy_nonoverlapping(w, c, 2 * k + 2);
                match div_q(h, c, 2 * k + 2, dt, k + 1, v, s) {
                    Some(qh) => qh,
                    None => div_qr(h, w, 2 * k + 2, dt, k + 1, v, s),
                }
            } else if split_low(2 * k + 2, k + 1) > 0 {
                div_q_split(h, w, 2 * k + 2, dt, k + 1, v, s)
            } else {
                div_qr(h, w, 2 * k + 2, dt, k + 1, v, s)
            };
            debug_assert!(
                !high,
                "the top limbs of a dividend were below the divisor's"
            );
        }
        if *h < 2 && (1..=k).any(|i| *h.add(i) != 0) {
            Found::Check
        } else {
            Found::Exact
        }
    }
}

/// The fewest low limbs of a quotient alone for which [`div_q_split`] computes them from the top
/// limbs.
const SPLIT_MIN: usize = 4;

/// The low limbs of a quotient alone of a dividend of `nn` limbs by a divisor of `dn` that
/// [`div_q_split`] computes from the top limbs: as many as leave the divisor [`TOP`] limbs more than
/// twice them, and two, and fewer than the quotient's; none below [`SPLIT_MIN`].
pub(super) fn split_low(nn: usize, dn: usize) -> usize {
    let k = nn - dn;
    let l = (dn.saturating_sub(2 + TOP) / 2).min(k.saturating_sub(1));
    if l >= SPLIT_MIN { l } else { 0 }
}

/// The limbs of scratch space that [`div_q_split`] needs to split off `l` low limbs of the
/// quotient of a dividend of `nn` limbs by a divisor of `dn`.
pub(super) fn split_itch(nn: usize, dn: usize, l: usize) -> usize {
    let high = qr_itch(nn - l, dn);
    let low = (2 * l + 2) + (l + 1) + top_itch(l).max(check_itch(l + dn, dn));
    high.max(low)
}

/// The quotient alone of the `nn` limbs at `n`, which it clobbers, by the `dn` limbs at `d`, as
/// [`sb_div_qr`] computes it with the remainder: the quotient's limbs from `l` up exactly, by the
/// schoolbook method or divide and conquer on the dividend's limbs from `l`, which leaves the
/// partial remainder in limbs `l..l + dn`; then the low `l` limbs, the quotient of the dividend's
/// limbs below `l + dn` by the divisor, from the top limbs of both by [`div_q_top`], which needs
/// the divisor's limbs above `2l + 1` only. That skips the products of the low `l` quotient limbs
/// and the divisor's low limbs, about `l dn - l^2` of them.
///
/// # Safety
///
/// As for [`sb_div_qr`], with `l = split_low(nn, dn) > 0`, and the [`split_itch`] limbs at `t`
/// valid for reads and writes, apart from the others.
pub(super) unsafe fn div_q_split(
    q: *mut u64,
    n: *mut u64,
    nn: usize,
    d: *const u64,
    dn: usize,
    v: u64,
    t: *mut u64,
) -> bool {
    let l = split_low(nn, dn);
    debug_assert!(l > 0);
    // SAFETY: as the caller guarantees: the upper division reads and writes the dividend's limbs
    // from `l`, the lower one copies of those below `l + dn`, and both stay in the scratch.
    unsafe {
        let qh = div_qr(q.add(l), n.add(l), nn - l, d, dn, v, t);
        let (w, h) = (t, t.add(2 * l + 2));
        let s = h.add(l + 1);
        core::ptr::copy_nonoverlapping(n.add(dn - l - 2), w, 2 * l + 2);
        match div_q_top(h, w, l, d.add(dn - l - 1), v, s) {
            Found::AllOnes => core::ptr::write_bytes(q, 0xFF, l),
            top => {
                core::ptr::copy_nonoverlapping(h.add(1), q, l);
                if top == Found::Check {
                    check_q_top(q, n, l + dn, d, dn, s);
                }
            }
        }
        qh
    }
}

/// Checks the quotient `H` of [`div_q_top`], in the `k = nn - dn` limbs at `q`, against the `nn`
/// limbs of the dividend at `n` and the `dn` of the divisor at `d`: takes it one down when `H d`
/// is larger than the dividend.
///
/// # Safety
///
/// `dn > k >= 1`; the limbs at `n` and `d` valid for reads, those at `q` for reads and writes, and
/// the [`check_itch`] limbs at `s` for reads and writes, all apart.
pub(super) unsafe fn check_q_top(
    q: *mut u64,
    n: *const u64,
    nn: usize,
    d: *const u64,
    dn: usize,
    s: *mut u64,
) {
    let k = nn - dn;
    // SAFETY: as the caller guarantees: the product of `nn` limbs at `s`, its scratch after it.
    unsafe {
        mul(s, d, dn, q, k, s.add(nn));
        if cmp_n(s, n, nn) == Ordering::Greater {
            sub_1(q, k, 1);
        }
    }
}

/// The limbs of scratch space that [`dc_div_qr`] needs for a dividend of `nn` limbs and a divisor
/// of `dn`, after the divisor's `dn` for the products.
pub(super) fn dc_itch(nn: usize, dn: usize) -> usize {
    let qn = nn - dn;
    let first = if qn > dn { (qn - 1) % dn + 1 } else { qn };
    let mut itch = if qn > dn { dc_n_itch(dn) } else { 0 };
    if first >= 3 {
        if first >= DC_DIV {
            itch = itch.max(dc_n_itch(first));
        }
        if first < dn {
            let (long, short) = if first > dn - first {
                (first, dn - first)
            } else {
                (dn - first, first)
            };
            itch = itch.max(mul_itch(long, short));
        }
    }
    dn + itch
}

/// The scratch space of [`dc_div_qr_n`] for `n` limbs, after the `n` for the products.
fn dc_n_itch(n: usize) -> usize {
    let (lo, hi) = (n / 2, n - n / 2);
    let mut itch = mul_itch(hi, lo);
    if hi >= DC_DIV {
        itch = itch.max(dc_n_itch(hi));
    }
    if lo >= DC_DIV {
        itch = itch.max(dc_n_itch(lo));
    }
    itch
}

/// The divide-and-conquer division, after GMP's `mpn_dcpi1_div_qr`: as [`sb_div_qr`] for a
/// divisor of at least [`DC_DIV`] limbs. The quotient is computed in blocks of `dn` limbs from the
/// top, after a first block of the limbs left over, each by [`dc_div_qr_n`].
///
/// # Safety
///
/// `dn >= DC_DIV` and `nn >= dn + 3`, the divisor normalized; the limbs as for [`sb_div_qr`], and
/// the [`dc_itch`] limbs at `t` valid for reads and writes, apart from the others.
pub(super) unsafe fn dc_div_qr(
    q: *mut u64,
    n: *mut u64,
    nn: usize,
    d: *const u64,
    dn: usize,
    v: u64,
    t: *mut u64,
) -> bool {
    debug_assert!(dn >= 6 && nn >= dn + 3);
    let qn = nn - dn;
    // The first block: the top `first` limbs of the quotient, of a dividend of `dn + first` limbs.
    let first = if qn > dn { (qn - 1) % dn + 1 } else { qn };
    // SAFETY: every block lies in the operands: the first block's dividend is limbs
    // `nn - dn - first..nn` of `n`, its quotient limbs `qn - first..qn` of `q`, and each block after
    // it `2 dn` limbs of `n` below the one before, with `dn` quotient limbs.
    let qh = unsafe {
        let (qf, nf) = (q.add(qn - first), n.add(nn - dn - first));
        if first < 3 {
            // A short block: the schoolbook steps on the whole divisor.
            sb_div_qr(qf, nf, dn + first, d, dn, v)
        } else {
            // The top `2 first` limbs by the divisor's top `first`, then the quotient times the
            // divisor's other limbs subtracted from the rest of the block, and corrected.
            let (top, dtop) = (n.add(nn - 2 * first), d.add(dn - first));
            let mut qh = if first < DC_DIV {
                sb_div_qr(qf, top, 2 * first, dtop, first, v)
            } else {
                dc_div_qr_n(qf, top, dtop, first, v, t)
            };
            if first < dn {
                let low = dn - first;
                if first > low {
                    mul(t, qf, first, d, low, t.add(dn));
                } else {
                    mul(t, d, low, qf, first, t.add(dn));
                }
                let mut borrow = u64::from(sub_n(nf, nf, t, dn));
                if qh {
                    borrow += u64::from(sub_n(nf.add(first), nf.add(first), d, low));
                }
                while borrow != 0 {
                    // The quotient limbs are decremented whatever the limb above them is.
                    let borrow_out = sub_1(qf, first, 1);
                    qh = qh && borrow_out == 0;
                    borrow -= u64::from(add_n(nf, nf, d, dn));
                }
            }
            qh
        }
    };
    // SAFETY: as above.
    unsafe {
        let mut done = first;
        while done < qn {
            done += dn;
            let block = qn - done;
            let high = dc_div_qr_n(q.add(block), n.add(block), d, dn, v, t);
            debug_assert!(
                !high,
                "a block's partial remainder was not below the divisor"
            );
        }
    }
    qh
}

/// Divides the `2n` limbs at `u` by the `n` limbs at `d`: sets the `n` limbs at `q` to the
/// quotient, leaves the remainder in the low `n` limbs of `u`, and returns the quotient's limb
/// above them, 0 or 1. The quotient's high half is the quotient of the top `2 hi` limbs by the
/// divisor's top `hi`, corrected by the product of that half and the divisor's low `lo` limbs;
/// its low half likewise from the partial remainder left.
///
/// # Safety
///
/// `n >= 6` and the divisor normalized; the limbs as for [`sb_div_qr`], and the `n` limbs at `t`
/// for the products, then the scratch of their multiplications, valid for reads and writes.
unsafe fn dc_div_qr_n(
    q: *mut u64,
    u: *mut u64,
    d: *const u64,
    n: usize,
    v: u64,
    t: *mut u64,
) -> bool {
    let (lo, hi) = (n / 2, n - n / 2);
    // SAFETY: every range lies in the `2n` limbs of `u`, the `n` of `d`, the `n` of `q` and the
    // scratch at `t`.
    unsafe {
        let mut qh = if hi < DC_DIV {
            sb_div_qr(q.add(lo), u.add(2 * lo), 2 * hi, d.add(lo), hi, v)
        } else {
            dc_div_qr_n(q.add(lo), u.add(2 * lo), d.add(lo), hi, v, t)
        };
        mul(t, q.add(lo), hi, d, lo, t.add(n));
        let mut borrow = u64::from(sub_n(u.add(lo), u.add(lo), t, n));
        if qh {
            borrow += u64::from(sub_n(u.add(n), u.add(n), d, lo));
        }
        while borrow != 0 {
            // The quotient limbs are decremented whatever the limb above them is.
            let borrow_out = sub_1(q.add(lo), hi, 1);
            qh = qh && borrow_out == 0;
            borrow -= u64::from(add_n(u.add(lo), u.add(lo), d, n));
        }

        let mut ql = if lo < DC_DIV {
            sb_div_qr(q, u.add(hi), 2 * lo, d.add(hi), lo, v)
        } else {
            dc_div_qr_n(q, u.add(hi), d.add(hi), lo, v, t)
        };
        mul(t, d, hi, q, lo, t.add(n));
        let mut borrow = u64::from(sub_n(u, u, t, n));
        if ql {
            borrow += u64::from(sub_n(u.add(lo), u.add(lo), d, hi));
        }
        while borrow != 0 {
            // The quotient limbs are decremented whatever the limb above them is.
            let borrow_out = sub_1(q, lo, 1);
            ql = ql && borrow_out == 0;
            borrow -= u64::from(add_n(u, u, d, n));
        }
        debug_assert!(!ql, "the low half of a quotient carried out of its limbs");
        qh
    }
}

/// The inverse of the odd limb `d` modulo `B = 2^64`: the `v` for which `d v = 1 mod B`. `3d XOR 2`
/// is the inverse modulo `2^5`, and each Newton step `v (2 - d v)` doubles the bits that are right.
#[inline]
pub(in super::super) fn binvert_limb(d: u64) -> u64 {
    debug_assert!(d & 1 == 1);
    let mut v = d.wrapping_mul(3) ^ 2;
    for _ in 0..4 {
        v = v.wrapping_mul(2u64.wrapping_sub(d.wrapping_mul(v)));
    }
    v
}

/// The exact division by Hensel's method, from the low end, as GMP's `mpn_sbpi1_bdiv_q` carries
/// it out with additions: sets the `qn` limbs at `q` to the quotient of the `qn` limbs at `n`,
/// which it clobbers, by the `dn` limbs at `d`, odd, modulo `B^qn`, with `v` the inverse of `d`'s
/// low limb modulo `B` ([`binvert_limb`]). When `d` divides the dividend, of which these are the
/// low limbs, and the quotient has at most `qn` limbs, that is the quotient.
///
/// Each limb of the quotient is the one that makes the low limb of the partial remainder zero,
/// `n_i v mod B`, exact, with no estimate to correct; it is then subtracted times the divisor,
/// whose limbs only up to the quotient's top are needed: the partial remainder above them never
/// counts. So a quotient of `qn` limbs costs about `qn dn - dn^2 / 2` limb products when it is
/// longer than the divisor, and `qn^2 / 2` otherwise, against `qn dn` and a division of three
/// limbs by two for each limb of an ordinary division.
///
/// # Safety
///
/// `dn >= 1` and `d` odd; the `qn` limbs at `n` must be valid for reads and writes, the
/// `min(dn, qn)` at `d` for reads and the `qn` at `q` for writes, all apart.
pub(in super::super) unsafe fn bdiv_q(
    q: *mut u64,
    n: *mut u64,
    qn: usize,
    d: *const u64,
    dn: usize,
    v: u64,
) {
    // SAFETY: as the caller guarantees: step `i` reads and writes limbs `i..qn` of `n` and the
    // limbs of `d` below `qn - i`.
    unsafe {
        for i in 0..qn {
            let qi = (*n.add(i)).wrapping_mul(v);
            *q.add(i) = qi;
            let len = (qn - i).min(dn);
            let borrow = submul_1(n.add(i), d, len, qi);
            if i + len < qn {
                sub_1(n.add(i + len), qn - i - len, borrow);
            }
        }
    }
}

/// Shifts the `n` limbs at `x` right by `s` bits, `0 < s < 64`, in place.
///
/// # Safety
///
/// The limbs must be valid for reads and writes.
pub(in super::super) unsafe fn rshift(x: *mut u64, n: usize, s: u32) {
    debug_assert!(0 < s && s < 64);
    // SAFETY: as the caller guarantees.
    unsafe {
        for i in 0..n - 1 {
            *x.add(i) = (*x.add(i) >> s) | (*x.add(i + 1) << (64 - s));
        }
        *x.add(n - 1) >>= s;
    }
}
