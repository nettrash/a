//! The arithmetic of numbers of one and two limbs in registers, after MPFR's `mpfr_add1sp1`,
//! `mpfr_sub1sp1`, `mpfr_mul_1`, `mpfr_div_1`, `mpfr_sqrt1` and their two-limb forms: every
//! operand's mantissa is a `u64` or a `u128` aligned at its top bit, the exact result (or the
//! result truncated, with a sticky bit for what lies below it) a `u128` or a pair of them, also
//! aligned at its top bit, and the rounding keeps its top bits, with no vector.
//!
//! A truncated result `S` with a sticky bit stands for a value strictly between `S` and one unit
//! of its lowest bit more. Each kernel keeps at least the precision and one bit more, so that the
//! round bit lies in `S` and the halfway point is a whole number of those units: the rounding is
//! then decided by the bits of `S` below the precision and the sticky bit.

use super::Round;
use super::round::Rounded;
use crate::math::numbers::nat;
use core::cmp::Ordering;

/// Whether the mode rounds a magnitude of the sign `negative` up when it is not exact, to
/// nearest aside.
#[inline(always)]
fn away(negative: bool, round: Round) -> bool {
    match round {
        Round::Nearest | Round::TowardZero => false,
        Round::Up => !negative,
        Round::Down => negative,
    }
}

/// The magnitude `0.s·2^exp` for `s` with its top bit set, or with `sticky` one strictly between
/// it and one unit more of the lowest bit that `s` keeps, rounded to `prec ≤ 64` bits. The round
/// bit, bit `127 - prec`, must lie at or above that lowest bit, so that the halfway point is a
/// whole number of its units.
#[inline(always)]
pub(crate) fn round_top(
    negative: bool,
    s: u128,
    exp: i128,
    sticky: bool,
    prec: u64,
    round: Round,
) -> Rounded {
    debug_assert!(s >> 127 == 1 && (1..=64).contains(&prec));
    let unit = 1u128 << (128 - prec);
    let half = unit >> 1;
    let low = s & (unit - 1);
    let mut m = s ^ low;
    let inexact = low != 0 || sticky;
    let up = match round {
        Round::Nearest => low > half || (low == half && (sticky || m & unit != 0)),
        _ => away(negative, round) && inexact,
    };
    let mut e = exp;
    if up {
        m = m.wrapping_add(unit);
        if m == 0 {
            m = 1 << 127;
            e += 1;
        }
    }
    Rounded {
        negative,
        m,
        exp: e,
        e0: exp,
        magnitude: if !inexact {
            Ordering::Equal
        } else if up {
            Ordering::Greater
        } else {
            Ordering::Less
        },
    }
}

/// [`round_top`] for 256 bits `hi·2^128 + lo`, the top bit of `hi` set, to `prec ≤ 128` bits.
#[inline(always)]
pub(crate) fn round_top2(
    negative: bool,
    hi: u128,
    lo: u128,
    exp: i128,
    sticky: bool,
    prec: u64,
    round: Round,
) -> Rounded {
    debug_assert!(hi >> 127 == 1 && (1..=128).contains(&prec));
    // Above, at or below the halfway point, and whether anything is dropped at all.
    let (mut m, unit, above, tie, inexact) = if prec < 128 {
        let unit = 1u128 << (128 - prec);
        let half = unit >> 1;
        let low = hi & (unit - 1);
        let rest = lo != 0 || sticky;
        (
            hi ^ low,
            unit,
            low > half || (low == half && rest),
            low == half && !rest,
            low != 0 || rest,
        )
    } else {
        let h = lo >> 127 == 1;
        let rest = lo << 1 != 0 || sticky;
        (hi, 1, h && rest, h && !rest, lo != 0 || sticky)
    };
    let up = match round {
        Round::Nearest => above || (tie && m & unit != 0),
        _ => away(negative, round) && inexact,
    };
    let mut e = exp;
    if up {
        m = m.wrapping_add(unit);
        if m == 0 {
            m = 1 << 127;
            e += 1;
        }
    }
    Rounded {
        negative,
        m,
        exp: e,
        e0: exp,
        magnitude: if !inexact {
            Ordering::Equal
        } else if up {
            Ordering::Greater
        } else {
            Ordering::Less
        },
    }
}

/// The 128-bit `v` aligned at its top bit, with the exponent `exp` of `0.v·2^exp` moved by the
/// shift; `v` not zero and exact.
#[inline(always)]
fn normal1(v: u128, exp: i128) -> (u128, i128) {
    let z = v.leading_zeros();
    (v << z, exp - i128::from(z))
}

/// The 256-bit `(hi, lo)` aligned at its top bit, exact, with the exponent moved by the shift.
#[inline(always)]
fn normal2(hi: u128, lo: u128, exp: i128) -> (u128, u128, i128) {
    if hi == 0 {
        let z = lo.leading_zeros();
        (lo << z, 0, exp - 128 - i128::from(z))
    } else {
        let z = hi.leading_zeros();
        if z == 0 {
            (hi, lo, exp)
        } else {
            (hi << z | lo >> (128 - z), lo << z, exp - i128::from(z))
        }
    }
}

/// `v·2^-d` and whether a bit of `v` was shifted out, for a 128-bit `v`.
#[inline(always)]
fn shr1(v: u128, d: u128) -> (u128, bool) {
    if d >= 128 {
        (0, v != 0)
    } else {
        (v >> d, v & ((1u128 << d) - 1) != 0)
    }
}

/// `(h·2^128 + l)·2^-d` and whether a bit was shifted out, for 256 bits.
#[inline(always)]
fn shr2(h: u128, l: u128, d: u128) -> (u128, u128, bool) {
    if d == 0 {
        (h, l, false)
    } else if d < 128 {
        (h >> d, h << (128 - d) | l >> d, l & ((1u128 << d) - 1) != 0)
    } else if d < 256 {
        let e = d - 128;
        (0, h >> e, l != 0 || h & ((1u128 << e) - 1) != 0)
    } else {
        (0, 0, h != 0 || l != 0)
    }
}

/// `±x ± y` for mantissas of one limb, of the exponents `ex` and `ey` (`0.x·2^ex`), and the signs
/// `nx` and `ny`, rounded to `prec ≤ 64` bits; `None` for an exact zero.
///
/// The larger term (in magnitude for a difference) is `x`, at the top of 128 bits, and `y` is
/// shifted right by the difference of the exponents, the bits shifted out a sticky bit, which a
/// difference takes as one unit off. A sum that carries is shifted right by one bit, the bit out
/// joining the sticky bit. A difference with a sticky bit (exponents more than 64 apart) keeps
/// its top bit unless `x` is a power of two: then it is computed one bit further.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn add1(
    nx: bool,
    x: u64,
    ex: i128,
    ny: bool,
    y: u64,
    ey: i128,
    prec: u64,
    round: Round,
) -> Option<Rounded> {
    let ((nx, x, ex), (ny, y, ey)) = if ex > ey || (ex == ey && (nx == ny || x >= y)) {
        ((nx, x, ex), (ny, y, ey))
    } else {
        ((ny, y, ey), (nx, x, ex))
    };
    let d = (ex - ey) as u128;
    let xs = u128::from(x) << 64;
    if nx == ny {
        let (ys, sticky) = shr1(u128::from(y) << 64, d);
        let (s, carry) = xs.overflowing_add(ys);
        return Some(if carry {
            round_top(
                nx,
                1 << 127 | s >> 1,
                ex + 1,
                sticky || s & 1 == 1,
                prec,
                round,
            )
        } else {
            round_top(nx, s, ex, sticky, prec, round)
        });
    }
    if d <= 64 {
        // Exact.
        let s = xs - ((u128::from(y) << 64) >> d);
        if s == 0 {
            return None;
        }
        let (s, e) = normal1(s, ex);
        return Some(round_top(nx, s, e, false, prec, round));
    }
    if x != 1 << 63 {
        // `x·2^64 - y·2^(64-d)` stays above `2^127`, as `y·2^(64-d) < 2^63`.
        let (ys, sticky) = shr1(u128::from(y) << 64, d);
        let s = xs - ys - u128::from(sticky);
        return Some(round_top(nx, s, ex, sticky, prec, round));
    }
    // `2^128 - y·2^(65-d)`, one bit further, above `2^127`.
    let (ys, sticky) = shr1(u128::from(y) << 64, d - 1);
    let s = 0u128.wrapping_sub(ys + u128::from(sticky));
    Some(round_top(nx, s, ex - 1, sticky, prec, round))
}

/// `±x ± y` for mantissas of at most two limbs as `u128`s aligned at their top bits, rounded to
/// `prec ≤ 128` bits; `None` for an exact zero. As [`add1`], on 256 bits.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn add2(
    nx: bool,
    x: u128,
    ex: i128,
    ny: bool,
    y: u128,
    ey: i128,
    prec: u64,
    round: Round,
) -> Option<Rounded> {
    let ((nx, x, ex), (ny, y, ey)) = if ex > ey || (ex == ey && (nx == ny || x >= y)) {
        ((nx, x, ex), (ny, y, ey))
    } else {
        ((ny, y, ey), (nx, x, ex))
    };
    let d = (ex - ey) as u128;
    if nx == ny {
        let (yh, yl, sticky) = shr2(y, 0, d);
        let (hi, carry) = x.overflowing_add(yh);
        return Some(if carry {
            round_top2(
                nx,
                1 << 127 | hi >> 1,
                hi << 127 | yl >> 1,
                ex + 1,
                sticky || yl & 1 == 1,
                prec,
                round,
            )
        } else {
            round_top2(nx, hi, yl, ex, sticky, prec, round)
        });
    }
    if d <= 128 {
        // Exact.
        let (yh, yl, _) = shr2(y, 0, d);
        let (lo, b) = 0u128.overflowing_sub(yl);
        let hi = x - yh - u128::from(b);
        if hi == 0 && lo == 0 {
            return None;
        }
        let (hi, lo, e) = normal2(hi, lo, ex);
        return Some(round_top2(nx, hi, lo, e, false, prec, round));
    }
    let (yh, yl, sticky) = shr2(y, 0, d);
    debug_assert!(yh == 0);
    if x != 1 << 127 {
        // `x·2^128 - y·2^(128-d)` stays above `2^255`.
        let (lo, b0) = 0u128.overflowing_sub(yl);
        let (lo, b1) = lo.overflowing_sub(u128::from(sticky));
        let hi = x - u128::from(b0) - u128::from(b1);
        return Some(round_top2(nx, hi, lo, ex, sticky, prec, round));
    }
    // `2^256 - y·2^(129-d)`, one bit further.
    let (_, yl, sticky) = shr2(y, 0, d - 1);
    let (lo, b0) = 0u128.overflowing_sub(yl);
    let (lo, b1) = lo.overflowing_sub(u128::from(sticky));
    let hi = 0u128.wrapping_sub(u128::from(b0) + u128::from(b1));
    Some(round_top2(nx, hi, lo, ex - 1, sticky, prec, round))
}

/// `x·y` for mantissas of one limb, rounded to `prec ≤ 64` bits: the exact product of 128 bits,
/// its top bit at 127 or 126.
#[inline(always)]
pub(crate) fn mul1(
    negative: bool,
    x: u64,
    ex: i128,
    y: u64,
    ey: i128,
    prec: u64,
    round: Round,
) -> Rounded {
    let p = u128::from(x) * u128::from(y);
    let e = ex + ey;
    if p >> 127 == 0 {
        round_top(negative, p << 1, e - 1, false, prec, round)
    } else {
        round_top(negative, p, e, false, prec, round)
    }
}

/// `x·y` for mantissas of at most two limbs, rounded to `prec ≤ 128` bits: the exact product of
/// 256 bits, from four products of limbs.
#[inline(always)]
pub(crate) fn mul2(
    negative: bool,
    x: u128,
    ex: i128,
    y: u128,
    ey: i128,
    prec: u64,
    round: Round,
) -> Rounded {
    let (hi, lo) = wide_mul(x, y);
    let e = ex + ey;
    if hi >> 127 == 0 {
        round_top2(
            negative,
            hi << 1 | lo >> 127,
            lo << 1,
            e - 1,
            false,
            prec,
            round,
        )
    } else {
        round_top2(negative, hi, lo, e, false, prec, round)
    }
}

/// The 256-bit product of two `u128`s, high half first.
#[inline(always)]
pub(crate) fn wide_mul(x: u128, y: u128) -> (u128, u128) {
    let (x1, x0) = (x >> 64, x as u64 as u128);
    let (y1, y0) = (y >> 64, y as u64 as u128);
    let p00 = x0 * y0;
    let p01 = x0 * y1;
    let p10 = x1 * y0;
    let p11 = x1 * y1;
    let mid = (p00 >> 64) + (p01 as u64 as u128) + (p10 as u64 as u128);
    let lo = (p00 as u64 as u128) | mid << 64;
    let hi = p11 + (p01 >> 64) + (p10 >> 64) + (mid >> 64);
    (hi, lo)
}

/// `x/y` for mantissas of one limb, rounded to `prec ≤ 64` bits: the quotient of `x·2^64` by `y`
/// (64 bits, or 65 from `x ≥ y`), with a limb more for 63 and 64 bits, by the division of two
/// limbs by one with the reciprocal of `y`; the remainder is the sticky bit.
#[inline(always)]
pub(crate) fn div1(
    negative: bool,
    x: u64,
    ex: i128,
    y: u64,
    ey: i128,
    prec: u64,
    round: Round,
) -> Rounded {
    let v = nat::invert_limb(y);
    let base = ex - ey;
    let top = x >= y;
    // `x - y < 2^63 ≤ y`, so the top limb of each division is below the divisor.
    let x1 = if top { x - y } else { x };
    let (q1, r1) = nat::div_2by1(x1, 0, y, v);
    if prec <= 62 {
        // At least 64 bits of quotient, two more than the precision.
        return if top {
            round_top(
                negative,
                1 << 127 | u128::from(q1) << 63,
                base + 1,
                r1 != 0,
                prec,
                round,
            )
        } else {
            round_top(negative, u128::from(q1) << 64, base, r1 != 0, prec, round)
        };
    }
    let (q0, r0) = nat::div_2by1(r1, 0, y, v);
    if top {
        // The quotient `2^128 + q1·2^64 + q0`, shifted right by one bit into 128.
        let s = 1u128 << 127 | u128::from(q1) << 63 | u128::from(q0 >> 1);
        round_top(negative, s, base + 1, q0 & 1 == 1 || r0 != 0, prec, round)
    } else {
        let q = u128::from(q1) << 64 | u128::from(q0);
        round_top(negative, q, base, r0 != 0, prec, round)
    }
}

/// `x/y` for mantissas of at most two limbs, rounded to `prec ≤ 128` bits: the quotient of
/// `x·2^128` by `y` by two divisions of three limbs by two (and a third for 128 bits when `x <
/// y`), with the reciprocal of `y`'s two limbs; the remainder is the sticky bit.
#[inline(always)]
pub(crate) fn div2(
    negative: bool,
    x: u128,
    ex: i128,
    y: u128,
    ey: i128,
    prec: u64,
    round: Round,
) -> Rounded {
    let (y1, y0) = ((y >> 64) as u64, y as u64);
    let v = nat::invert_3by2(y1, y0);
    let base = ex - ey;
    let top = x >= y;
    let xr = if top { x - y } else { x };
    let u = [0, 0, xr as u64, (xr >> 64) as u64];
    let mut q = [0u64; 2];
    let (r1, r0) = nat::div_qr_2(&mut q, &u, y1, y0, v);
    let qv = u128::from(q[1]) << 64 | u128::from(q[0]);
    if top {
        // `2^128 + qv`, 129 bits.
        return round_top2(
            negative,
            1 << 127 | qv >> 1,
            qv << 127,
            base + 1,
            r1 | r0 != 0,
            prec,
            round,
        );
    }
    if prec < 128 {
        return round_top2(negative, qv, 0, base, r1 | r0 != 0, prec, round);
    }
    let mut q2 = [0u64; 1];
    let (s1, s0) = nat::div_qr_2(&mut q2, &[0, r0, r1], y1, y0, v);
    round_top2(
        negative,
        qv,
        u128::from(q2[0]) << 64,
        base,
        s1 | s0 != 0,
        prec,
        round,
    )
}

/// The square root of the 128-bit radicand `r` (its top bit at 127 or 126, for an even `ex` of
/// `0.r·2^ex`), with `tail` for bits below it, rounded to `prec ≤ 63` bits: a root of 64 bits,
/// the remainder and the tail the sticky bit. A root is never halfway between two numbers of the
/// precision, so its round bit and the sticky bit decide.
#[inline(always)]
fn sqrt_top(r: u128, ex: i128, tail: bool, prec: u64, round: Round) -> Rounded {
    let (s, rem) = nat::sqrt_rem_u128(r);
    round_top(
        false,
        u128::from(s) << 64,
        ex / 2,
        rem != 0 || tail,
        prec,
        round,
    )
}

/// The square root of a mantissa of one limb, rounded to `prec ≤ 64` bits: the root of `x·2^64`
/// or `x·2^63`, whichever makes the exponent even, in registers (with a limb more for 64 bits).
#[inline(always)]
pub(crate) fn sqrt1(x: u64, ex: i128, prec: u64, round: Round) -> Rounded {
    let odd = ex & 1 != 0;
    if prec <= 63 {
        let r = if odd {
            u128::from(x) << 63
        } else {
            u128::from(x) << 64
        };
        return sqrt_top(r, ex + i128::from(odd), false, prec, round);
    }
    let a = if odd {
        [0, 0, x << 63, x >> 1]
    } else {
        [0, 0, 0, x]
    };
    let (s, rem) = nat::sqrt_rem_wide(a);
    round_top(
        false,
        s,
        (ex + i128::from(odd)) / 2,
        rem != [0; 3],
        prec,
        round,
    )
}

/// The square root of a mantissa of more limbs, `top` and `next` the top two and `rest` whether
/// any of the others is not zero, rounded to `prec ≤ 63` bits: the top 128 bits of the radicand,
/// the others a tail, as the floor of the root of a truncated radicand is that of the exact one.
#[inline(always)]
pub(crate) fn sqrt_long(
    top: u64,
    next: u64,
    rest: bool,
    ex: i128,
    prec: u64,
    round: Round,
) -> Rounded {
    let r = u128::from(top) << 64 | u128::from(next);
    if ex & 1 != 0 {
        sqrt_top(r >> 1, ex + 1, rest || next & 1 == 1, prec, round)
    } else {
        sqrt_top(r, ex, rest, prec, round)
    }
}

/// The square root of a mantissa of at most two limbs, rounded to `prec ≤ 127` bits: the root of
/// `x·2^128` or `x·2^127`, of 128 bits, in registers.
#[inline(always)]
pub(crate) fn sqrt2(x: u128, ex: i128, prec: u64, round: Round) -> Rounded {
    debug_assert!(prec <= 127);
    let odd = ex & 1 != 0;
    let a = if odd {
        let (h, l) = (x >> 1, x << 127);
        [l as u64, (l >> 64) as u64, h as u64, (h >> 64) as u64]
    } else {
        [0, 0, x as u64, (x >> 64) as u64]
    };
    let (s, rem) = nat::sqrt_rem_wide(a);
    round_top2(
        false,
        s,
        0,
        (ex + i128::from(odd)) / 2,
        rem != [0; 3],
        prec,
        round,
    )
}

/// The mantissa of one or two limbs `m` as a `u128` aligned at its top bit.
#[inline(always)]
pub(crate) fn wide(m: &[u64]) -> u128 {
    match *m {
        [x] => u128::from(x) << 64,
        [lo, hi] => u128::from(hi) << 64 | u128::from(lo),
        _ => unreachable!("a mantissa of one or two limbs"),
    }
}
