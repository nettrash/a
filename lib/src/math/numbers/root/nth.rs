//! The `k`-th roots of magnitudes held as limbs, for `k >= 3`: native `u64`s, the least significant
//! first.
//!
//! The root of `U` is found by Newton's method at doubling precision, as GMP's `mpn_rootrem` does
//! it after Brent and Zimmermann ("Modern Computer Arithmetic", 2010, section 1.5.2). With `T(p)`
//! the top bits of `U` whose root has `p` bits, `U` shifted right by `k` times the number of bits
//! of the root below them, and `S` the exact root of `T(p)`, the root of `T(p + b)` is `S 2^b + d`
//! for some `0 <= d < 2^b`, and
//!
//! `d <= (T(p + b) - S^k 2^(kb)) / (k S^(k-1) 2^((k-1) b))`,
//!
//! as `(S 2^b + d)^k` is at least the first two terms of its binomial expansion. The right side,
//! rounded down, is computed exactly, from `R = T(p) - S^k`, the next `b` bits of `U` and
//! `k S^(k-1)`: it is never below the root's `d`, and above it by less than one when `S >= 2 k 2^b`
//! (by the mean value theorem: `y^k - x^k <= k y^(k-1) (y - x)` for `y >= x`, and
//! `(y / x)^(k-1) < (1 + 1 / S)^(k-1)`), so the new root is exact or one too large. The next step
//! starts by computing `S^(k-1)` and `S^k`, which it needs anyway, and takes one off the root when
//! `S^k > T`. So the precision grows from `p` to at most `2p - ceil(log2 k) - 2` bits a step; the
//! steps are planned from the root's last bit back.
//!
//! The first `p <= 30` bits, and the whole of a root that short, come from a double-precision
//! estimate, `2^(log2(T) / k)`, made exact by comparing powers of it with `T(p)`: first powers of
//! its top 64 bits truncated at each step, which bound the exact powers within a known factor (see
//! [`compare_power`]), and the exact powers only when those do not decide. When `U` has more than
//! `3k` limbs, the root is computed with 64 bits more, as that of `U 2^(64k)`, and not checked at
//! the end: the root of `U` is the top bits of that one, unless its 64 extra bits are 0 or 1,
//! where it may be one too large and is checked, which is rare but for perfect powers. Otherwise
//! the root is checked against `U` as the first estimate is.
//!
//! The numbers of the steps are kept in one buffer of working memory, of which [`root_itch`] tells
//! the size; the scratch of the products and divisions grows in a vector of its own, which they
//! need only from about 24 limbs.

use super::super::div::limbs::{div_qr, div_qr_1, div_qr_2, invert_3by2, invert_limb, qr_itch};
use super::super::mul::limbs::{add_1, cmp_n, mul, mul_itch, sqr, sqr_itch, sub_1, sub_n};
use super::{ROUGH_BITS, lshift, mul_1, rough_root, rshift};
use core::cmp::Ordering;

/// The number of bits of a root up to which it is computed from a double-precision estimate.
const ESTIMATE_BITS: u64 = 30;

/// The extra bits of the root of `U 2^(64k)` that spare the check at the end (see the module
/// documentation).
const GUARD_BITS: u64 = 64;

/// The longest plan of steps: the precision at least halves back from the root's last bit, but by
/// one bit at a time below `ceil(log2 k) + 2`, which is at most 34.
const MAX_STEPS: usize = 128;

/// The plan of a root: its number of bits, the extra bits computed below them, and the
/// precisions of the steps.
struct Plan {
    /// The number of bits of the root.
    xb: u64,
    /// The extra bits computed below the root's: 0 or [`GUARD_BITS`].
    guard: u64,
    /// The limbs of the numbers of the steps, `U` shifted and the powers of the root.
    cap: usize,
    /// The limbs of the root with the extra bits, and two to spare.
    sn: usize,
}

impl Plan {
    /// The plan of the root of degree `k` of a magnitude of `ub` bits in `un` limbs.
    fn of(ub: u64, un: usize, k: u32) -> Plan {
        let xb = (ub - 1) / u64::from(k) + 1;
        let guard = if xb > ESTIMATE_BITS && un > 3 * k as usize && logk(k) + 2 < ESTIMATE_BITS {
            GUARD_BITS
        } else {
            0
        };
        let sn = (xb + guard).div_ceil(64) as usize + 2;
        // `U` shifted left by at most `(k + 1)` times the extra bits, or by the root's bits when
        // there are none, with a limb or two for the powers of a root one too large.
        let cap = un + if guard > 0 { k as usize + 2 } else { sn } + 4;
        Plan { xb, guard, cap, sn }
    }
}

/// `ceil(log2 k) + 2`: from `p` bits, a step can reach `2p - logk(k)` bits.
fn logk(k: u32) -> u64 {
    u64::from(32 - (k - 1).leading_zeros()) + 2
}

/// The limbs of working memory that [`root`] needs for the root of degree `k >= 3` of a magnitude
/// of `un` limbs: the root and eight numbers of the steps.
pub(super) fn root_itch(un: usize, k: u32) -> usize {
    let plan = Plan::of(64 * un as u64, un, k);
    plan.sn + 8 * (plan.cap + 2)
}

/// The buffers of the steps, each of `cap + 2` limbs, in the working memory.
#[derive(Clone, Copy)]
struct Buffers {
    /// The limbs of each buffer.
    stride: usize,
    t: *mut u64,
    tb: *mut u64,
    pa: *mut u64,
    pb: *mut u64,
    pk: *mut u64,
    nbuf: *mut u64,
    dbuf: *mut u64,
    q: *mut u64,
}

/// Sets the limbs at `s` to the root of degree `k >= 3` of the `un` limbs `U` at `u`, the top one
/// not zero, rounded down, and returns their number.
///
/// # Safety
///
/// The limbs at `u` must be valid for reads, and the [`root_itch`]`(un, k)` at `w` for reads and
/// writes, apart from them; the root is written in the first limbs of `w`, at `s = w`.
pub(super) unsafe fn root(u: *const u64, un: usize, k: u32, w: *mut u64) -> usize {
    // SAFETY: as the caller guarantees: the root takes the first `sn` limbs of the working memory,
    // and each buffer `cap + 2` after them.
    unsafe {
        debug_assert!(k >= 3 && *u.add(un - 1) != 0);
        let ub = 64 * un as u64 - u64::from((*u.add(un - 1)).leading_zeros());
        let plan = Plan::of(ub, un, k);
        // `root_itch` plans for `64 un` bits, at least `ub`: as long a root, and as many limbs.
        debug_assert!(plan.sn + 8 * (plan.cap + 2) <= root_itch(un, k));
        let s = w;
        let stride = plan.cap + 2;
        let base = w.add(plan.sn);
        let buffers = Buffers {
            stride,
            t: base,
            tb: base.add(stride),
            pa: base.add(2 * stride),
            pb: base.add(3 * stride),
            pk: base.add(4 * stride),
            nbuf: base.add(5 * stride),
            dbuf: base.add(6 * stride),
            q: base.add(7 * stride),
        };
        let mut scratch = Vec::new();
        let k64 = u64::from(k);
        let xb = plan.xb;
        if xb <= ESTIMATE_BITS {
            *s = settle(
                estimate(u, un, ub, 0, k),
                k,
                u,
                un,
                ub,
                0,
                &buffers,
                &mut scratch,
            );
            return 1;
        }
        // The precisions of the steps, from the last back.
        let lk = logk(k);
        let mut sizes = [0u64; MAX_STEPS];
        let mut steps = 0;
        let mut p = xb + plan.guard;
        loop {
            sizes[steps] = p;
            steps += 1;
            if p <= ESTIMATE_BITS {
                break;
            }
            let q = (p + lk).div_ceil(2);
            p = if q < p { q } else { p - 1 };
        }
        // `U` shifted for a root of `p` bits: right by `k (xb - p)` bits, left when that is negative.
        let at = |p: u64| (i128::from(xb) - i128::from(p)) * i128::from(k);
        steps -= 1;
        let mut p = sizes[steps];
        *s = settle(
            estimate(u, un, ub, at(p), k),
            k,
            u,
            un,
            ub,
            at(p),
            &buffers,
            &mut scratch,
        );
        let mut sn = 1;
        while steps > 0 {
            steps -= 1;
            let next = sizes[steps];
            let b = next - p;
            let Buffers {
                stride,
                t,
                tb,
                pa,
                pb,
                pk,
                nbuf,
                dbuf,
                q,
            } = buffers;
            // `T b` is `T(p)` with the next `b` bits of `U` below it.
            let tbn = shift_into(tb, u, un, at(p) - i128::from(b));
            debug_assert!(tbn < stride);
            let tn = shift_into(t, tb, tbn, i128::from(b));
            // `W = S^(k-1)` and `P = S^k`, with `S` one less while `P > T`.
            let (mut wp, mut wn, mut pkn);
            loop {
                (wp, wn) = power_into(s, sn, k64 - 1, pa, pb, stride, &mut scratch);
                debug_assert!(wn + sn <= stride);
                pkn = mul_into(pk, wp, wn, s, sn, &mut scratch);
                if compare(pk, pkn, t, tn) != Ordering::Greater {
                    break;
                }
                sub_1(s, sn, 1);
                sn = normalized(s, sn);
            }
            // `N = (T - P) 2^b + the next b bits`, in place of `T b`.
            let pbn = shift_into(pk, pk, pkn, -i128::from(b));
            debug_assert!(pbn <= tbn && wn < stride && sn + (b.div_ceil(64) as usize) < plan.sn);
            let borrow = sub_n(tb, tb, pk, pbn);
            sub_1(tb.add(pbn), tbn - pbn, u64::from(borrow));
            let nn = normalized(tb, tbn);
            // `D = k W`, in place.
            let carry = mul_1(wp, wp, wn, k64);
            *wp.add(wn) = carry;
            let dn = wn + usize::from(carry != 0);
            // The quotient has at most `nn - dn + 1` limbs; a divisor much longer is cut to its top
            // `qn + 2` limbs, and the dividend by as many, plus one, which keeps `d` a bound from
            // above: `N / D < (N' + 1) / D'` for `N'` and `D'` without their low `cut` limbs, by
            // less than one more, as `D' >= 2^64 B^qn` and `N / D < B^qn`.
            let qn = (nn + 1).saturating_sub(dn);
            let cut = dn.saturating_sub(qn + 2).min(nn);
            let (nt, nnt, dt, dnt) = if cut > 0 {
                *tb.add(nn) = add_1(tb.add(cut), nn - cut, 1);
                (
                    tb.add(cut),
                    normalized(tb.add(cut), nn - cut + 1),
                    wp.add(cut),
                    dn - cut,
                )
            } else {
                (tb, nn, wp, dn)
            };
            let mut qn = quotient_into(q, nt, nnt, dt, dnt, nbuf, dbuf, &mut scratch);
            if bit_len(q, qn) > b {
                // The root is below `(S + 1) 2^b`: `d` is at most `2^b - 1`.
                qn = b.div_ceil(64) as usize;
                core::ptr::write_bytes(q, u8::MAX, qn);
                if b % 64 != 0 {
                    *q.add(qn - 1) = (1 << (b % 64)) - 1;
                }
            }
            // `S 2^b + d`.
            sn = shift_into(s, s, sn, -i128::from(b));
            let low = b.div_ceil(64) as usize;
            if sn < low {
                core::ptr::write_bytes(s.add(sn), 0, low - sn);
                sn = low;
            }
            for i in 0..qn {
                *s.add(i) |= *q.add(i);
            }
            p = next;
        }
        if plan.guard == 0 {
            return settle_long(s, sn, k, u, un, &buffers, &mut scratch);
        }
        // `S` is the root of `U 2^(64k)` or one too large.
        let low = *s;
        core::ptr::copy(s.add(1), s, sn - 1);
        sn -= 1;
        if low >= 2 {
            sn
        } else {
            settle_long(s, sn, k, u, un, &buffers, &mut scratch)
        }
    }
}

/// An estimate of the root of degree `k` of `T = U >> sh` (left for a negative `sh`), not zero, of
/// `ub - sh` bits, for a root of at most [`ESTIMATE_BITS`] bits: `2^(log2(T) / k)` in double
/// precision, from the top 64 bits of `T`, which are those of `U`.
///
/// # Safety
///
/// The `un` limbs of `U` at `u` must be valid for reads.
unsafe fn estimate(u: *const u64, un: usize, ub: u64, sh: i128, k: u32) -> u64 {
    let tb = (i128::from(ub) - sh) as u64;
    // SAFETY: as the caller guarantees.
    let top = unsafe { top_bits(u, un) };
    // `T` is `top 2^(tb - 64)`, rounded down when it has more bits.
    let root = if tb.div_ceil(u64::from(k)) <= u64::from(ROUGH_BITS) {
        rough_root(top, tb, k)
    } else {
        let log = (top as f64).log2() + tb as f64 - 64.0;
        (log / f64::from(k)).exp2()
    };
    (root as u64).clamp(1, (1 << ESTIMATE_BITS) - 1)
}

/// The top 64 bits of the `un` limbs at `u`: shifted so that the top bit is bit 63, with the bits
/// of the limb below.
///
/// # Safety
///
/// The limbs must be valid for reads, the top one not zero.
unsafe fn top_bits(u: *const u64, un: usize) -> u64 {
    // SAFETY: as the caller guarantees.
    unsafe {
        let top = *u.add(un - 1);
        let z = top.leading_zeros();
        if z == 0 || un == 1 {
            return top << z;
        }
        (top << z) | (*u.add(un - 2) >> (64 - z))
    }
}

/// The root of degree `k` of `T = U >> sh`, from an estimate within a few units of it: `s` one less
/// while `s^k > T`, and one more while `(s + 1)^k <= T`, each decided by [`compare_power`] from the
/// top bits of `T`, which are those of `U`, or by the exact power when that cannot tell. `T` is
/// that of the first step, or `U` itself, of a root of at most [`ESTIMATE_BITS`] bits.
///
/// # Safety
///
/// As for [`root`]: the buffers must be those of the working memory of `U`.
#[allow(clippy::too_many_arguments)]
unsafe fn settle(
    mut s: u64,
    k: u32,
    u: *const u64,
    un: usize,
    ub: u64,
    sh: i128,
    buffers: &Buffers,
    scratch: &mut Vec<u64>,
) -> u64 {
    // SAFETY: as the caller guarantees.
    unsafe {
        let k64 = u64::from(k);
        let tb = (i128::from(ub) - sh) as u64;
        // The top 64 bits of `T`: those of `U`, of which a `T` of fewer bits has only its own.
        let mut top = top_bits(u, un);
        if tb < 64 {
            top &= !(u64::MAX >> tb);
        }
        // Whether `x^k > T`, from the bounds or exactly.
        let mut tn = 0;
        let mut above = |x: u64| -> bool {
            match compare_power(x, k64, top, tb) {
                Some(order) => order == Ordering::Greater,
                None => {
                    if tn == 0 {
                        tn = shift_into(buffers.t, u, un, sh);
                    }
                    let (pp, pn) = power_into(
                        &raw const x,
                        1,
                        k64,
                        buffers.pa,
                        buffers.pb,
                        buffers.stride,
                        scratch,
                    );
                    compare(pp, pn, buffers.t, tn) == Ordering::Greater
                }
            }
        };
        // Both bounds of the estimate, side by side: the root when `s^k <= T < (s + 1)^k`, which
        // is the usual answer; otherwise steps of one.
        let (low, high) = (s > 1 && above(s), above(s + 1));
        if !low && high {
            return s;
        }
        if low {
            s -= 1;
            while s > 1 && above(s) {
                s -= 1;
            }
            return s;
        }
        s += 1;
        while !above(s + 1) {
            s += 1;
        }
        s
    }
}

/// The root of degree `k` of the `un` limbs `U` at `u` from the `sn` limbs at `s`, which are it or
/// one more: one less when `s^k > U`, decided by [`compare_power`] from the top bits of both, or
/// by the exact power when that cannot tell. Returns the root's number of limbs.
///
/// # Safety
///
/// As for [`root`].
unsafe fn settle_long(
    s: *mut u64,
    sn: usize,
    k: u32,
    u: *const u64,
    un: usize,
    buffers: &Buffers,
    scratch: &mut Vec<u64>,
) -> usize {
    // SAFETY: as the caller guarantees.
    unsafe {
        let k64 = u64::from(k);
        let ub = bit_len(u, un);
        let sb = bit_len(s, sn);
        let top = top_bits(u, un);
        // The top 64 bits of the root, truncated: below it, by less than a part in 2^63.
        let st = top_bits(s, sn);
        let mut sn = sn;
        let mut over = match compare_power_scaled(st, sb, k64, top, ub) {
            Some(order) => order == Ordering::Greater,
            None => {
                let (pp, pn) =
                    power_into(s, sn, k64, buffers.pa, buffers.pb, buffers.stride, scratch);
                compare(pp, pn, u, un) == Ordering::Greater
            }
        };
        while over {
            // One less, and again while the power is above, which a quotient from cut operands
            // makes possible, if rarely.
            sub_1(s, sn, 1);
            sn = normalized(s, sn);
            let sb = bit_len(s, sn);
            over = match compare_power_scaled(top_bits(s, sn), sb, k64, top, ub) {
                Some(order) => order == Ordering::Greater,
                None => {
                    let (pp, pn) =
                        power_into(s, sn, k64, buffers.pa, buffers.pb, buffers.stride, scratch);
                    compare(pp, pn, u, un) == Ordering::Greater
                }
            };
        }
        sn
    }
}

/// Compares `x^k` with `T`, of `tb` bits whose top 64 are `top`, from bounds of `x^k`: `Some`
/// order when they tell it, `None` when they cannot. See [`compare_power_scaled`].
fn compare_power(x: u64, k: u64, top: u64, tb: u64) -> Option<Ordering> {
    let xb = 64 - u64::from(x.leading_zeros());
    compare_power_scaled(x << (64 - xb), xb, k, top, tb)
}

/// Compares `x^k` with `T`, for an `x` of `xb` bits whose top 64, truncated, are `xm`, and a `T`
/// of `tb` bits whose top 64, truncated, are `tm`: `Some` order when the bounds below tell it,
/// `None` when they cannot.
///
/// `x^k` is computed on 64-bit mantissas, each square and product truncated to its top 64 bits,
/// which keeps a lower bound `L = m 2^e`. A truncation takes off less than a part in `2^63`, and a
/// part `d` off before a square is `2d` after it, so from a mantissa of `x` truncated too, `L` is
/// below `x^k` by a part `d < 4k 2^-63 = k 2^-61`; the exact power is then below `L (1 + 2d)`, in
/// `[m, m + 16k + 1] 2^e`. `T` is in `[tm, tm + 1) 2^(tb - 64)`, exactly `tm` below 65 bits. When
/// the two ranges do not meet, their order is that of `x^k` and `T`.
fn compare_power_scaled(xm: u64, xb: u64, k: u64, tm: u64, tb: u64) -> Option<Ordering> {
    debug_assert!(xm >> 63 == 1 && tm >> 63 == 1 && k >= 1);
    // `L = m 2^e`, `m` with its top bit set; `x = xm 2^(xb - 64)`.
    // The exponents are below the bits of `T`, which fit memory, far below `2^63`.
    let mut m = xm;
    let mut e = xb as i64 - 64;
    // `m y` truncated to its top 64 bits, the top one set: the product is at least `2^126`, and
    // shifted left by one more when its top bit is clear, without a branch.
    let step = |m: u64, e: i64, y: u64, ey: i64| -> (u64, i64) {
        let p = u128::from(m) * u128::from(y);
        let s = ((p >> 127) as u32) ^ 1;
        (((p << s) >> 64) as u64, e + ey + 64 - i64::from(s))
    };
    let ex = e;
    for i in (0..63 - k.leading_zeros()).rev() {
        (m, e) = step(m, e, m, e);
        if (k >> i) & 1 == 1 {
            (m, e) = step(m, e, xm, ex);
        }
    }
    // Both on the exponent of the smaller: the ranges of `x^k` and `T` in units of `2^low`.
    let et = tb as i64 - 64;
    let low = e.min(et);
    let (dp, dt) = (e - low, et - low);
    if dp > 2 || dt > 2 {
        // Mantissas with their top bits set: the larger exponent is the larger number, by more
        // than the bounds.
        return Some(e.cmp(&et));
    }
    let margin = 16 * u128::from(k) + 2;
    let (p_lo, p_hi) = (u128::from(m) << dp, (u128::from(m) + margin) << dp);
    // `T` is exact when it has at most 64 bits.
    let width = u128::from(tb > 64);
    let (t_lo, t_hi) = (u128::from(tm) << dt, (u128::from(tm) + width) << dt);
    if p_hi < t_lo {
        Some(Ordering::Less)
    } else if p_lo >= t_hi && (width == 1 || p_lo > t_lo) {
        Some(Ordering::Greater)
    } else {
        None
    }
}

/// The number of limbs of the `n` limbs at `x` without the zero limbs at the top.
///
/// # Safety
///
/// The limbs must be valid for reads.
unsafe fn normalized(x: *const u64, mut n: usize) -> usize {
    // SAFETY: as the caller guarantees.
    unsafe {
        while n > 0 && *x.add(n - 1) == 0 {
            n -= 1;
        }
    }
    n
}

/// The number of bits of the `n` limbs at `x`, the top one not zero, or none.
///
/// # Safety
///
/// The limbs must be valid for reads.
unsafe fn bit_len(x: *const u64, n: usize) -> u64 {
    if n == 0 {
        return 0;
    }
    // SAFETY: as the caller guarantees.
    64 * n as u64 - u64::from(unsafe { *x.add(n - 1) }.leading_zeros())
}

/// Compares the `xn` limbs at `x` with the `yn` at `y`, both without zero limbs at the top.
///
/// # Safety
///
/// The limbs must be valid for reads.
unsafe fn compare(x: *const u64, xn: usize, y: *const u64, yn: usize) -> Ordering {
    match xn.cmp(&yn) {
        // SAFETY: as the caller guarantees.
        Ordering::Equal => unsafe { cmp_n(x, y, xn) },
        other => other,
    }
}

/// Sets the limbs at `dst` to the `xn` limbs at `x` shifted right by `sh` bits, or left by `-sh`
/// when `sh` is negative, and returns their number without zero limbs at the top.
///
/// # Safety
///
/// The limbs at `x` must be valid for reads, and those at `dst` for writing the result, which has
/// at most `xn + ceil(-sh / 64)` limbs; `dst` may be `x`.
unsafe fn shift_into(dst: *mut u64, x: *const u64, xn: usize, sh: i128) -> usize {
    // SAFETY: as the caller guarantees; a right shift reads each limb before writing the limb at or
    // below it, a left shift from the top down, each before writing the limb at or above it.
    unsafe {
        if xn == 0 {
            return 0;
        }
        if sh >= 0 {
            let (limbs, bits) = (
                usize::try_from(sh / 64).unwrap_or(usize::MAX),
                (sh % 64) as u32,
            );
            if limbs >= xn {
                return 0;
            }
            let n = xn - limbs;
            if bits == 0 {
                core::ptr::copy(x.add(limbs), dst, n);
            } else {
                rshift(dst, x.add(limbs), n, bits);
            }
            normalized(dst, n)
        } else {
            let sh = -sh;
            let (limbs, bits) = ((sh / 64) as usize, (sh % 64) as u32);
            let top = lshift(dst.add(limbs), x, xn, bits);
            *dst.add(limbs + xn) = top;
            core::ptr::write_bytes(dst, 0, limbs);
            normalized(dst, limbs + xn + 1)
        }
    }
}

/// Makes `scratch` hold at least `n` limbs, and returns a pointer to them.
fn scratch_of(scratch: &mut Vec<u64>, n: usize) -> *mut u64 {
    if scratch.len() < n {
        scratch.resize(n, 0);
    }
    scratch.as_mut_ptr()
}

/// Sets the limbs at `r` to the product of the `xn` limbs at `x` and the `yn` at `y`, not zero and
/// without zero limbs at the top, and returns its number of limbs likewise.
///
/// # Safety
///
/// The limbs at `x` and `y` must be valid for reads and the `xn + yn` at `r` for writes, `r` apart
/// from both.
unsafe fn mul_into(
    r: *mut u64,
    x: *const u64,
    xn: usize,
    y: *const u64,
    yn: usize,
    scratch: &mut Vec<u64>,
) -> usize {
    let ((a, an), (b, bn)) = if xn >= yn {
        ((x, xn), (y, yn))
    } else {
        ((y, yn), (x, xn))
    };
    let t = scratch_of(scratch, mul_itch(an, bn));
    // SAFETY: as the caller guarantees, with the scratch the product needs.
    unsafe {
        mul(r, a, an, b, bn, t);
        normalized(r, an + bn)
    }
}

/// Sets the limbs at `out` to `x^e` for the `xn` limbs at `x`, not zero, and `e >= 1`, by squares
/// and products from the top bit of `e` down, in the buffers `a` and `b` in turn; returns the
/// buffer that holds the power and its number of limbs.
///
/// # Safety
///
/// The limbs at `x` must be valid for reads, and the `cap` limbs at `a` and at `b` for writing the
/// power and the powers on the way, which must fit them, all apart.
#[allow(clippy::too_many_arguments)]
unsafe fn power_into(
    x: *const u64,
    xn: usize,
    e: u64,
    a: *mut u64,
    b: *mut u64,
    cap: usize,
    scratch: &mut Vec<u64>,
) -> (*mut u64, usize) {
    // SAFETY: as the caller guarantees; each square or product goes from one buffer to the other.
    unsafe {
        let (mut cur, mut other) = (a, b);
        core::ptr::copy_nonoverlapping(x, cur, xn);
        let mut n = xn;
        for i in (0..63 - e.leading_zeros()).rev() {
            debug_assert!(2 * n <= cap, "a power outgrew its buffer");
            let t = scratch_of(scratch, sqr_itch(n));
            sqr(other, cur, n, t);
            n = normalized(other, 2 * n);
            core::mem::swap(&mut cur, &mut other);
            if (e >> i) & 1 == 1 {
                debug_assert!(n + xn <= cap, "a power outgrew its buffer");
                n = mul_into(other, cur, n, x, xn, scratch);
                core::mem::swap(&mut cur, &mut other);
            }
        }
        (cur, n)
    }
}

/// Sets the limbs at `q` to `floor(N / D)` for the `nn` limbs `N` at `n` and the `dn` limbs `D` at
/// `d`, not zero, both without zero limbs at the top, and returns the quotient's number of limbs
/// likewise. `N` and `D` are shifted into `nbuf` and `dbuf` so that the divisor's top bit is set.
///
/// # Safety
///
/// The limbs at `n` and `d` must be valid for reads; `nbuf` for writing `nn + 1` limbs, `dbuf`
/// `dn`, and `q` `nn`, all apart.
#[allow(clippy::too_many_arguments)]
unsafe fn quotient_into(
    q: *mut u64,
    n: *const u64,
    nn: usize,
    d: *const u64,
    dn: usize,
    nbuf: *mut u64,
    dbuf: *mut u64,
    scratch: &mut Vec<u64>,
) -> usize {
    // SAFETY: as the caller guarantees; the dividend's top limb, the bits shifted out of `N`, is
    // below the divisor's, whose top bit is set.
    unsafe {
        if compare(n, nn, d, dn) == Ordering::Less {
            return 0;
        }
        let s = (*d.add(dn - 1)).leading_zeros();
        lshift(dbuf, d, dn, s);
        *nbuf.add(nn) = lshift(nbuf, n, nn, s);
        let qn = match dn {
            1 => {
                div_qr_1::<true>(q, nbuf, nn, *nbuf.add(nn), *dbuf, invert_limb(*dbuf));
                nn
            }
            2 => {
                let v = invert_3by2(*dbuf.add(1), *dbuf);
                div_qr_2(q, nbuf, nn + 1, *dbuf.add(1), *dbuf, v);
                nn - 1
            }
            _ => {
                let len = if *nbuf.add(nn) != 0 { nn + 1 } else { nn };
                let t = scratch_of(scratch, qr_itch(len, dn));
                let v = invert_3by2(*dbuf.add(dn - 1), *dbuf.add(dn - 2));
                *q.add(len - dn) = u64::from(div_qr(q, nbuf, len, dbuf, dn, v, t));
                len - dn + 1
            }
        };
        normalized(q, qn)
    }
}

/// `floor(n / d)` for magnitudes in vectors without zero limbs at the top, `d` not zero.
pub(in super::super) fn quotient(n: &[u64], d: &[u64]) -> Vec<u64> {
    let mut q = vec![0u64; n.len().max(1)];
    let mut nbuf = vec![0u64; n.len() + 1];
    let mut dbuf = vec![0u64; d.len()];
    let mut scratch = Vec::new();
    // SAFETY: the vectors hold the operands and room for the shifted copies and the quotient.
    let qn = unsafe {
        quotient_into(
            q.as_mut_ptr(),
            n.as_ptr(),
            n.len(),
            d.as_ptr(),
            d.len(),
            nbuf.as_mut_ptr(),
            dbuf.as_mut_ptr(),
            &mut scratch,
        )
    };
    q.truncate(qn);
    q
}
