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

/// The mantissa of one or two limbs `m` as a `u128` aligned at its top bit.
#[inline(always)]
pub(crate) fn wide(m: &[u64]) -> u128 {
    match *m {
        [x] => u128::from(x) << 64,
        [lo, hi] => u128::from(hi) << 64 | u128::from(lo),
        _ => unreachable!("a mantissa of one or two limbs"),
    }
}

// The kernels in words, after MPFR's `mpfr_add1sp1`, `mpfr_sub1sp1`, `mpfr_mul_1`, `mpfr_div_1`
// and their two-limb forms, for the forms into an existing number and the new results: a result
// in a word `W` of one limb (`u64`), two (`u128`), four or eight (`Pair`), of a precision
// `p = BITS - sh` (`sh ≥ 0` bits of the word below it), the operands' mantissas aligned at the
// top of words of the same kind (of any bits), the exponents `i64`s. Each kernel is exact up to
// the word below its result and a sticky bit, and returns the sign, the exponent of `0.m·2^exp`,
// the rounded mantissa and the direction of the rounding, or `None` for an exact zero or a result
// out of the exponent range, which the general path computes again.

/// A limb, or two as one `u128`: the word of a mantissa of one or two limbs.
pub(crate) trait Word:
    Copy
    + Eq
    + Ord
    + core::ops::Shl<u32, Output = Self>
    + core::ops::Shr<u32, Output = Self>
    + core::ops::BitAnd<Output = Self>
    + core::ops::BitOr<Output = Self>
    + core::ops::BitXor<Output = Self>
    + core::ops::Not<Output = Self>
    + core::ops::Sub<Output = Self>
    + core::ops::Add<Output = Self>
{
    /// The bits of the word.
    const BITS: u32;
    const ZERO: Self;
    const ONE: Self;
    /// The top bit.
    const HIGH: Self;
    fn lz(self) -> u32;
    fn oadd(self, other: Self) -> (Self, bool);
    fn wneg(self) -> Self;
    fn wadd(self, other: Self) -> Self;
    fn of_bool(b: bool) -> Self;
    /// The double-word product, high word first.
    fn wmul(self, other: Self) -> (Self, Self);
    /// The mantissa of at most the word's limbs `m`, aligned at the top of the word.
    fn of_top(m: &[u64]) -> Self;
    /// Writes the word into the mantissa `m` of as many limbs.
    fn store(self, m: &mut [u64]);
    /// The lowest limb.
    fn limb0(self) -> u64;
    /// The word with its lowest limb replaced.
    fn with_limb0(self, l: u64) -> Self;
    /// The word of the value `l`.
    fn of_limb(l: u64) -> Self;
}

impl Word for u64 {
    const BITS: u32 = 64;
    const ZERO: u64 = 0;
    const ONE: u64 = 1;
    const HIGH: u64 = 1 << 63;
    #[inline(always)]
    fn lz(self) -> u32 {
        self.leading_zeros()
    }
    #[inline(always)]
    fn oadd(self, other: u64) -> (u64, bool) {
        self.overflowing_add(other)
    }
    #[inline(always)]
    fn wneg(self) -> u64 {
        self.wrapping_neg()
    }
    #[inline(always)]
    fn wadd(self, other: u64) -> u64 {
        self.wrapping_add(other)
    }
    #[inline(always)]
    fn of_bool(b: bool) -> u64 {
        u64::from(b)
    }
    #[inline(always)]
    fn wmul(self, other: u64) -> (u64, u64) {
        let p = u128::from(self) * u128::from(other);
        ((p >> 64) as u64, p as u64)
    }
    #[inline(always)]
    fn of_top(m: &[u64]) -> u64 {
        m[0]
    }
    #[inline(always)]
    fn store(self, m: &mut [u64]) {
        m[0] = self;
    }
    #[inline(always)]
    fn limb0(self) -> u64 {
        self
    }
    #[inline(always)]
    fn with_limb0(self, l: u64) -> u64 {
        l
    }
    #[inline(always)]
    fn of_limb(l: u64) -> u64 {
        l
    }
}

impl Word for u128 {
    const BITS: u32 = 128;
    const ZERO: u128 = 0;
    const ONE: u128 = 1;
    const HIGH: u128 = 1 << 127;
    #[inline(always)]
    fn lz(self) -> u32 {
        self.leading_zeros()
    }
    #[inline(always)]
    fn oadd(self, other: u128) -> (u128, bool) {
        self.overflowing_add(other)
    }
    #[inline(always)]
    fn wneg(self) -> u128 {
        self.wrapping_neg()
    }
    #[inline(always)]
    fn wadd(self, other: u128) -> u128 {
        self.wrapping_add(other)
    }
    #[inline(always)]
    fn of_bool(b: bool) -> u128 {
        u128::from(b)
    }
    #[inline(always)]
    fn wmul(self, other: u128) -> (u128, u128) {
        wide_mul(self, other)
    }
    #[inline(always)]
    fn of_top(m: &[u64]) -> u128 {
        wide(m)
    }
    #[inline(always)]
    fn store(self, m: &mut [u64]) {
        m[0] = self as u64;
        m[1] = (self >> 64) as u64;
    }
    #[inline(always)]
    fn limb0(self) -> u64 {
        self as u64
    }
    #[inline(always)]
    fn with_limb0(self, l: u64) -> u128 {
        self & !u128::from(u64::MAX) | u128::from(l)
    }
    #[inline(always)]
    fn of_limb(l: u64) -> u128 {
        u128::from(l)
    }
}

/// A result of a kernel into an existing number: `±0.m·2^exp`, the bits of `m` below the
/// precision zero, and the direction of the rounding (of the signed value).
#[derive(Clone, Copy)]
pub(crate) struct Fit<W> {
    pub(crate) negative: bool,
    pub(crate) exp: i64,
    pub(crate) m: W,
    pub(crate) o: Ordering,
}

/// The rounding of `0.a·2^exp` (`a`'s top bit set) followed by the word `low` and, below it, the
/// sticky bit `extra`, to the precision `BITS - sh` (`sh` from 0 up): the round bit is bit
/// `sh - 1` of `a`, or the top bit of `low` for `sh = 0`, and the sticky bit every bit below it.
/// `None` when the exponent leaves the range.
#[inline(always)]
pub(crate) fn fit<W: Word>(
    negative: bool,
    a: W,
    low: W,
    exp: i64,
    extra: bool,
    sh: u32,
    round: Round,
) -> Option<Fit<W>> {
    debug_assert!(a >= W::HIGH && sh < W::BITS);
    // The bits below the precision lie in the lowest limb for `sh < 64`, the masks a limb's.
    let (mut m, rb, sb) = if sh == 0 {
        (a, low >= W::HIGH, extra || low << 1 != W::ZERO)
    } else if sh < 64 {
        let l = a.limb0();
        let half = 1u64 << (sh - 1);
        let low_bits = l & ((half << 1) - 1);
        (
            a.with_limb0(l ^ low_bits),
            low_bits & half != 0,
            extra || low != W::ZERO || low_bits & (half - 1) != 0,
        )
    } else {
        let half = W::ONE << (sh - 1);
        let low_bits = a & ((half << 1) - W::ONE);
        (
            a ^ low_bits,
            low_bits & half != W::ZERO,
            extra || low != W::ZERO || low_bits & (half - W::ONE) != W::ZERO,
        )
    };
    let mut e = exp;
    let o = if rb || sb {
        let up = match round {
            Round::Nearest => {
                rb && (sb
                    || if sh < 64 {
                        m.limb0() >> sh & 1 == 1
                    } else {
                        m & W::ONE << sh != W::ZERO
                    })
            }
            _ => away(negative, round),
        };
        if up {
            let unit = if sh < 64 {
                W::of_limb(1 << sh)
            } else {
                W::ONE << sh
            };
            let carry;
            (m, carry) = m.oadd(unit);
            if carry {
                m = W::HIGH;
                e += 1;
            }
        }
        if up == negative {
            Ordering::Less
        } else {
            Ordering::Greater
        }
    } else {
        Ordering::Equal
    };
    if !(super::BigFloat::EXP_MIN..=super::BigFloat::EXP_MAX).contains(&e) {
        return None;
    }
    Some(Fit {
        negative,
        exp: e,
        m,
        o,
    })
}

/// `±(x + y)` for mantissas `x` and `y` of the exponents `ex` and `ey`: the smaller shifted right
/// by the difference `d` of the exponents, the bits shifted out the word below (a sticky bit for
/// `d > BITS`), and a sum that carries shifted right by one, its lowest bit into that word. For
/// `d = 0` the sum always carries.
#[inline(always)]
pub(crate) fn add_fit<W: Word>(
    negative: bool,
    ex: i64,
    x: W,
    ey: i64,
    y: W,
    sh: u32,
    round: Round,
) -> Option<Fit<W>> {
    let (ex, x, ey, y) = if ex >= ey {
        (ex, x, ey, y)
    } else {
        (ey, y, ex, x)
    };
    let d = ex.abs_diff(ey);
    let b = W::BITS - 1;
    if d == 0 {
        let s = x.wadd(y);
        return fit(negative, W::HIGH | s >> 1, s << b, ex + 1, false, sh, round);
    }
    if d >= u64::from(W::BITS) {
        let (low, extra) = if d == u64::from(W::BITS) {
            (y, false)
        } else {
            (W::ZERO, true)
        };
        return fit(negative, x, low, ex, extra, sh, round);
    }
    let d = d as u32;
    let lost = y << (W::BITS - d);
    let (s, carry) = x.oadd(y >> d);
    if carry {
        fit(
            negative,
            W::HIGH | s >> 1,
            s << b | lost >> 1,
            ex + 1,
            lost & W::ONE != W::ZERO,
            sh,
            round,
        )
    } else {
        fit(negative, s, lost, ex, false, sh, round)
    }
}

/// `±(x - y)`, of the sign `nx` of `x` turned over when `y` is the larger, for mantissas of the
/// exponents `ex` and `ey`: the difference of `x·2^BITS` and `y·2^(BITS - d)`, `d` the difference
/// of the exponents, in two words, `a` and the word below it. For `d < BITS` it is exact: the
/// word below is `-(y << (BITS - d))`, with one taken off `a` when that is not zero. For
/// `d ≥ BITS` the bits of `y` from `d - BITS` up are subtracted from the word below, and the
/// fraction below them, when it is not zero, takes one unit more off and is a sticky bit, `T + (1
/// - f)` for the truncation `T`. Then `a` is shifted left until its top bit is set, the bits of
/// the word below coming in: at most one bit unless `d ≤ 1`, where the difference is exact; and
/// for `d = 1` `a` may be zero, the difference then the word below alone.
#[inline(always)]
pub(crate) fn sub_fit<W: Word>(
    nx: bool,
    ex: i64,
    x: W,
    ey: i64,
    y: W,
    sh: u32,
    round: Round,
) -> Option<Fit<W>> {
    let (negative, ex, x, ey, y) = if ex > ey || (ex == ey && x > y) {
        (nx, ex, x, ey, y)
    } else if ex == ey && x == y {
        return None;
    } else {
        (!nx, ey, y, ex, x)
    };
    let d = ex.abs_diff(ey);
    let bits = u64::from(W::BITS);
    let (mut a, mut f, mut e, extra) = if d == 0 {
        (x - y, W::ZERO, ex, false)
    } else if d < bits {
        let d = d as u32;
        let f = (y << (W::BITS - d)).wneg();
        (x - (y >> d) - W::of_bool(f != W::ZERO), f, ex, false)
    } else {
        // `y·2^(BITS - d)`: its whole part `l` from the word below, its fraction a sticky bit.
        let k = d - bits;
        let (l, frac) = if k == 0 {
            (y, false)
        } else if k < bits {
            let k = k as u32;
            (y >> k, y << (W::BITS - k) != W::ZERO)
        } else {
            (W::ZERO, true)
        };
        let f = l.wneg().wadd(W::of_bool(frac).wneg());
        let borrow = l != W::ZERO || frac;
        (x - W::of_bool(borrow), f, ex, frac)
    };
    if a == W::ZERO {
        // Only for `d = 1`: the difference is the word below, exactly.
        a = f;
        f = W::ZERO;
        e -= i64::from(W::BITS);
    }
    let z = a.lz();
    if z > 0 {
        a = a << z | f >> (W::BITS - z);
        f = f << z;
    }
    fit(negative, a, f, e - i64::from(z), extra, sh, round)
}

/// `±x·y` for mantissas of at most the word's limbs (aligned at their tops in it), rounded to
/// `BITS - sh` bits, `sh` from 0 up: the exact double-word product, its top bit at the top or
/// one below, shifted into place.
#[inline(always)]
pub(crate) fn mul_fit<W: Word>(
    negative: bool,
    ex: i64,
    x: W,
    ey: i64,
    y: W,
    sh: u32,
    round: Round,
) -> Option<Fit<W>> {
    let (hi, lo) = x.wmul(y);
    let e = ex + ey;
    if hi < W::HIGH {
        fit(
            negative,
            hi << 1 | lo >> (W::BITS - 1),
            lo << 1,
            e - 1,
            false,
            sh,
            round,
        )
    } else {
        fit(negative, hi, lo, e, false, sh, round)
    }
}

/// `±x/y` for mantissas of one limb, rounded to `64 - sh` bits: the quotient of `x·2^64` by `y`
/// by the division of two limbs by one with the reciprocal of `y`, as [`div1`]: for `x ≥ y` a
/// quotient of 65 bits, its lowest shifted into the word below, else of 64, with a second limb of
/// quotient below for 64 bits; the remainder the sticky bit.
#[inline(always)]
pub(crate) fn div_fit1(
    negative: bool,
    ex: i64,
    x: u64,
    ey: i64,
    y: u64,
    sh: u32,
    round: Round,
) -> Option<Fit<u64>> {
    let v = nat::invert_limb(y);
    let e = ex - ey;
    let top = x >= y;
    let x1 = if top { x - y } else { x };
    let (q1, r1) = nat::div_2by1(x1, 0, y, v);
    if top {
        return fit(
            negative,
            1 << 63 | q1 >> 1,
            q1 << 63,
            e + 1,
            r1 != 0,
            sh,
            round,
        );
    }
    if sh == 0 {
        let (q0, r0) = nat::div_2by1(r1, 0, y, v);
        return fit(negative, q1, q0, e, r0 != 0, sh, round);
    }
    fit(negative, q1, 0, e, r1 != 0, sh, round)
}

/// `±x/y` for mantissas of at most two limbs as `u128`s, rounded to `128 - sh` bits: the quotient
/// of `x·2^128` by `y` by two divisions of three limbs by two with the reciprocal of `y`, as
/// [`div2`], in registers: for `x ≥ y` a quotient of 129 bits, its lowest shifted into the word
/// below, else of 128, with a third limb of quotient for 128 bits; the remainder the sticky bit.
#[inline(always)]
pub(crate) fn div_fit2(
    negative: bool,
    ex: i64,
    x: u128,
    ey: i64,
    y: u128,
    sh: u32,
    round: Round,
) -> Option<Fit<u128>> {
    let (y1, y0) = ((y >> 64) as u64, y as u64);
    let v = nat::invert_3by2(y1, y0);
    let e = ex - ey;
    let top = x >= y;
    let xr = if top { x - y } else { x };
    let (q1, r1, r0) = nat::div_3by2((xr >> 64) as u64, xr as u64, 0, y1, y0, v);
    let (q0, r1, r0) = nat::div_3by2(r1, r0, 0, y1, y0, v);
    let qv = u128::from(q1) << 64 | u128::from(q0);
    if top {
        return fit(
            negative,
            1 << 127 | qv >> 1,
            qv << 127,
            e + 1,
            r1 | r0 != 0,
            sh,
            round,
        );
    }
    if sh == 0 {
        let (q, s1, s0) = nat::div_3by2(r1, r0, 0, y1, y0, v);
        return fit(
            negative,
            qv,
            u128::from(q) << 64,
            e,
            s1 | s0 != 0,
            sh,
            round,
        );
    }
    fit(negative, qv, 0, e, r1 | r0 != 0, sh, round)
}

/// Two words as one of twice their bits, `hi·2^BITS + lo`: the exact products of operands of two
/// limbs (`Pair<u128>`, 256 bits) and of four (`Pair<Pair<u128>>`, 512 bits), and the sums with
/// them, in registers, every shift straight-line on the halves.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct Pair<W> {
    pub(crate) hi: W,
    pub(crate) lo: W,
}

impl<W: Word> PartialOrd for Pair<W> {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<W: Word> Ord for Pair<W> {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> Ordering {
        self.hi.cmp(&other.hi).then(self.lo.cmp(&other.lo))
    }
}

impl<W: Word> core::ops::Shl<u32> for Pair<W> {
    type Output = Pair<W>;
    /// The word shifted left by `s < 2·BITS` bits.
    #[inline(always)]
    fn shl(self, s: u32) -> Pair<W> {
        let b = W::BITS;
        if s == 0 {
            self
        } else if s < b {
            Pair {
                hi: self.hi << s | self.lo >> (b - s),
                lo: self.lo << s,
            }
        } else {
            Pair {
                hi: self.lo << (s - b),
                lo: W::ZERO,
            }
        }
    }
}

impl<W: Word> core::ops::Shr<u32> for Pair<W> {
    type Output = Pair<W>;
    /// The word shifted right by `s < 2·BITS` bits.
    #[inline(always)]
    fn shr(self, s: u32) -> Pair<W> {
        let b = W::BITS;
        if s == 0 {
            self
        } else if s < b {
            Pair {
                hi: self.hi >> s,
                lo: self.lo >> s | self.hi << (b - s),
            }
        } else {
            Pair {
                hi: W::ZERO,
                lo: self.hi >> (s - b),
            }
        }
    }
}

macro_rules! halfwise {
    ($Trait:ident, $f:ident, $op:tt) => {
        impl<W: Word> core::ops::$Trait for Pair<W> {
            type Output = Pair<W>;
            #[inline(always)]
            fn $f(self, other: Pair<W>) -> Pair<W> {
                Pair {
                    hi: self.hi $op other.hi,
                    lo: self.lo $op other.lo,
                }
            }
        }
    };
}

halfwise!(BitAnd, bitand, &);
halfwise!(BitOr, bitor, |);
halfwise!(BitXor, bitxor, ^);

impl<W: Word> core::ops::Not for Pair<W> {
    type Output = Pair<W>;
    #[inline(always)]
    fn not(self) -> Pair<W> {
        Pair {
            hi: !self.hi,
            lo: !self.lo,
        }
    }
}

impl<W: Word> core::ops::Add for Pair<W> {
    type Output = Pair<W>;
    /// The sum, which must not carry out.
    #[inline(always)]
    fn add(self, other: Pair<W>) -> Pair<W> {
        let (r, carry) = self.oadd(other);
        debug_assert!(!carry);
        r
    }
}

impl<W: Word> core::ops::Sub for Pair<W> {
    type Output = Pair<W>;
    /// The difference, which must not borrow: the high halves then do not either.
    #[inline(always)]
    fn sub(self, other: Pair<W>) -> Pair<W> {
        debug_assert!(self >= other);
        let borrow = self.lo < other.lo;
        Pair {
            hi: self.hi - other.hi - W::of_bool(borrow),
            lo: self.lo.wadd(other.lo.wneg()),
        }
    }
}

impl<W: Word> Word for Pair<W> {
    const BITS: u32 = 2 * W::BITS;
    const ZERO: Pair<W> = Pair {
        hi: W::ZERO,
        lo: W::ZERO,
    };
    const ONE: Pair<W> = Pair {
        hi: W::ZERO,
        lo: W::ONE,
    };
    const HIGH: Pair<W> = Pair {
        hi: W::HIGH,
        lo: W::ZERO,
    };
    #[inline(always)]
    fn lz(self) -> u32 {
        if self.hi != W::ZERO {
            self.hi.lz()
        } else {
            W::BITS + self.lo.lz()
        }
    }
    #[inline(always)]
    fn oadd(self, other: Pair<W>) -> (Pair<W>, bool) {
        let (lo, c1) = self.lo.oadd(other.lo);
        let (hi, c2) = self.hi.oadd(other.hi);
        let (hi, c3) = hi.oadd(W::of_bool(c1));
        (Pair { hi, lo }, c2 | c3)
    }
    #[inline(always)]
    fn wneg(self) -> Pair<W> {
        let hi = if self.lo == W::ZERO {
            self.hi.wneg()
        } else {
            !self.hi
        };
        Pair {
            hi,
            lo: self.lo.wneg(),
        }
    }
    #[inline(always)]
    fn wadd(self, other: Pair<W>) -> Pair<W> {
        self.oadd(other).0
    }
    #[inline(always)]
    fn of_bool(b: bool) -> Pair<W> {
        Pair {
            hi: W::ZERO,
            lo: W::of_bool(b),
        }
    }
    /// The product of four products of halves, high word first.
    #[inline(always)]
    fn wmul(self, other: Pair<W>) -> (Pair<W>, Pair<W>) {
        let (p11h, p11l) = self.hi.wmul(other.hi);
        let (p10h, p10l) = self.hi.wmul(other.lo);
        let (p01h, p01l) = self.lo.wmul(other.hi);
        let (p00h, p00l) = self.lo.wmul(other.lo);
        // The middle: p00h + p10l + p01l, with its carries into the top.
        let (m, c1) = p00h.oadd(p10l);
        let (m, c2) = m.oadd(p01l);
        let (t, c3) = p11l.oadd(p10h);
        let (t, c4) = t.oadd(p01h);
        let (t, c5) = t.oadd(W::of_bool(c1).wadd(W::of_bool(c2)));
        let top = p11h
            .wadd(W::of_bool(c3))
            .wadd(W::of_bool(c4))
            .wadd(W::of_bool(c5));
        (Pair { hi: top, lo: t }, Pair { hi: m, lo: p00l })
    }
    #[inline(always)]
    fn of_top(m: &[u64]) -> Pair<W> {
        let half = (W::BITS / 64) as usize;
        if m.len() <= half {
            Pair {
                hi: W::of_top(m),
                lo: W::ZERO,
            }
        } else {
            let (lo, hi) = m.split_at(m.len() - half);
            Pair {
                hi: W::of_top(hi),
                lo: W::of_top(lo),
            }
        }
    }
    #[inline(always)]
    fn store(self, m: &mut [u64]) {
        let half = (W::BITS / 64) as usize;
        self.lo.store(&mut m[..half]);
        self.hi.store(&mut m[half..]);
    }
    #[inline(always)]
    fn limb0(self) -> u64 {
        self.lo.limb0()
    }
    #[inline(always)]
    fn with_limb0(self, l: u64) -> Pair<W> {
        Pair {
            lo: self.lo.with_limb0(l),
            ..self
        }
    }
    #[inline(always)]
    fn of_limb(l: u64) -> Pair<W> {
        Pair {
            hi: W::ZERO,
            lo: W::of_limb(l),
        }
    }
}

/// The limbs of a result `r` of a word into `out`: its top `out.len()` limbs, or with `long` (a
/// result of more limbs than the word) all of them followed by zeros, when the result is exact in
/// the word, `None` otherwise. The sign, the exponent and the direction of the rounding.
#[inline(always)]
pub(crate) fn put_top<W: Word>(
    r: Fit<W>,
    long: bool,
    out: &mut [u64],
) -> Option<(bool, i64, Ordering)> {
    let (n, k) = (out.len(), (W::BITS / 64) as usize);
    let mut t = [0u64; 8];
    r.m.store(&mut t[..k]);
    if long {
        if r.o != Ordering::Equal {
            return None;
        }
        for v in &mut out[..n - k] {
            *v = 0;
        }
        out[n - k..].copy_from_slice(&t[..k]);
    } else {
        out.copy_from_slice(&t[k - n..k]);
    }
    Some((r.negative, r.exp, r.o))
}

/// A term of a sum: the sign, the exponent of `0.m·2^exp` and the mantissa as a word.
#[derive(Clone, Copy)]
pub(crate) struct Term<W> {
    pub(crate) negative: bool,
    pub(crate) exp: i64,
    pub(crate) m: W,
}

impl<W: Word> Term<W> {
    /// The term one unit of the word's lowest bit larger in magnitude: a carry out of the top
    /// makes it the next power of two.
    #[inline(always)]
    pub(crate) fn bump(self) -> Term<W> {
        let (m, carry) = self.m.oadd(W::ONE);
        if carry {
            Term {
                m: W::HIGH,
                exp: self.exp + 1,
                ..self
            }
        } else {
            Term { m, ..self }
        }
    }
}

/// `x + y` for signed terms, rounded to `BITS - sh` bits: [`add_fit`] or [`sub_fit`].
#[inline(always)]
pub(crate) fn sum_fit<W: Word>(x: Term<W>, y: Term<W>, sh: u32, round: Round) -> Option<Fit<W>> {
    if x.negative == y.negative {
        add_fit(x.negative, x.exp, x.m, y.exp, y.m, sh, round)
    } else {
        sub_fit(x.negative, x.exp, x.m, y.exp, y.m, sh, round)
    }
}

/// The rounding of a value known to lie strictly between the values `lo` and `hi`, given their
/// roundings: when both round to the same number `R` and `R` is not above `lo` (nor below `hi`),
/// `R` is below (above) the value too. `None` when they leave it undecided.
#[inline(always)]
pub(crate) fn between<W: Word>(lo: Fit<W>, hi: Fit<W>) -> Option<Fit<W>> {
    if lo.m != hi.m || lo.exp != hi.exp || lo.negative != hi.negative {
        return None;
    }
    if lo.o != Ordering::Greater {
        Some(Fit {
            o: Ordering::Less,
            ..lo
        })
    } else if hi.o != Ordering::Less {
        Some(Fit {
            o: Ordering::Greater,
            ..lo
        })
    } else {
        None
    }
}

/// A kernel into `n = ceil(prec / 64)` limbs run on the vector `out`, made that long: its result
/// as a value with the vector as its mantissa, or `None` (the vector then of no use).
#[inline(always)]
pub(crate) fn with_vec(
    prec: u64,
    out: &mut Vec<u64>,
    f: impl FnOnce(&mut [u64]) -> Option<(bool, i64, Ordering)>,
) -> Option<(super::Kind, Ordering)> {
    let n = prec.div_ceil(64) as usize;
    out.clear();
    out.resize(n, 0);
    let (negative, exp, o) = f(out)?;
    Some((
        super::Kind::Finite {
            negative,
            exp,
            mant: core::mem::take(out),
        },
        o,
    ))
}

/// The term of a mantissa as a word: its top `BITS / 64` limbs, and whether any limb below them
/// is not zero.
#[inline(always)]
fn term<W: Word>(negative: bool, m: &[u64], exp: i64) -> (Term<W>, bool) {
    let k = m.len().min((W::BITS / 64) as usize);
    let (low, top) = m.split_at(m.len() - k);
    (
        Term {
            negative,
            exp,
            m: W::of_top(top),
        },
        low.iter().any(|&v| v != 0),
    )
}

/// `x + y` of the signs `nx` and `ny` in a word, rounded to `BITS - sh` bits: exactly when
/// neither has limbs below the word; otherwise each lies between its truncation `T` and `T` plus
/// one unit of its lowest limb, strictly when its other limbs are not all zero, so the sum lies
/// strictly between the sum of the lower ends and that of the upper ends, and [`between`] tells
/// when their roundings settle the rounding of the sum.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn add_word<W: Word>(
    nx: bool,
    x: &[u64],
    ex: i64,
    ny: bool,
    y: &[u64],
    ey: i64,
    sh: u32,
    round: Round,
) -> Option<Fit<W>> {
    let ((tx, ax), (ty, ay)) = (term::<W>(nx, x, ex), term::<W>(ny, y, ey));
    if !(ax || ay) {
        return sum_fit(tx, ty, sh, round);
    }
    // An operand with a tail is larger in magnitude than its truncation.
    let end = |t: Term<W>, tail: bool, up: bool| {
        if tail && up != t.negative {
            t.bump()
        } else {
            t
        }
    };
    let lo = sum_fit(end(tx, ax, false), end(ty, ay, false), sh, round)?;
    let hi = sum_fit(end(tx, ax, true), end(ty, ay, true), sh, round)?;
    between(lo, hi)
}

/// `x + y` of the signs `nx` and `ny` (a difference when they differ) for mantissas of other
/// lengths than the result's, into its `n = out.len()` limbs, `prec` bits, on the stack: in a
/// word of two limbs for one limb of result, of four for two, from the operands' top limbs (see
/// [`add_word`]); operands of at most four limbs exactly in a word of four, a result of more limbs
/// when the sum is exact in it. The sign, the exponent and the direction of the rounding; `None`
/// for the general path.
#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn add_short(
    nx: bool,
    x: &[u64],
    ex: i64,
    ny: bool,
    y: &[u64],
    ey: i64,
    prec: u64,
    round: Round,
    out: &mut [u64],
) -> Option<(bool, i64, Ordering)> {
    type U256 = Pair<u128>;
    let n = out.len();
    if n == 1 {
        let r = add_word::<u128>(nx, x, ex, ny, y, ey, (128 - prec) as u32, round)?;
        out[0] = (r.m >> 64) as u64;
        return Some((r.negative, r.exp, r.o));
    }
    if n == 2 || (x.len() <= 4 && y.len() <= 4) {
        let long = n > 4;
        let sh = if long { 0 } else { (256 - prec) as u32 };
        let r = add_word::<U256>(nx, x, ex, ny, y, ey, sh, round)?;
        return put_top(r, long, out);
    }
    None
}

/// `±x·y` into the `n = out.len()` limbs of a result of `prec` bits, `n ≥ 3`, on the stack: for
/// operands of at most two limbs the exact product in a word of four limbs, a longer result
/// taking it whole; for operands of at most four and a result of at most four, [`mul_fit`] in a
/// word of eight. `None` for the general path.
#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn mul_short(
    negative: bool,
    x: &[u64],
    ex: i64,
    y: &[u64],
    ey: i64,
    prec: u64,
    round: Round,
    out: &mut [u64],
) -> Option<(bool, i64, Ordering)> {
    type U256 = Pair<u128>;
    let n = out.len();
    if n < 3 {
        return None;
    }
    if x.len() <= 2 && y.len() <= 2 {
        let (hi, lo) = wide_mul(wide(x), wide(y));
        let p = Pair { hi, lo };
        let (p, e) = if hi >> 127 == 0 {
            (p << 1, ex + ey - 1)
        } else {
            (p, ex + ey)
        };
        let long = n > 4;
        let sh = if long { 0 } else { (256 - prec) as u32 };
        let r = fit(negative, p, U256::ZERO, e, false, sh, round)?;
        return put_top(r, long, out);
    }
    if x.len() <= 4 && y.len() <= 4 && n <= 4 {
        let r = mul_fit(
            negative,
            ex,
            U256::of_top(x),
            ey,
            U256::of_top(y),
            (256 - prec) as u32,
            round,
        )?;
        return put_top(r, false, out);
    }
    None
}

/// `±x/y` for a divisor of one limb and a dividend of at most two, into the `n = out.len() ≥ 3`
/// limbs of a result of `prec` bits: `N`, the top `n + 1` limbs of `0.x·2^-c` (`c = 1` when
/// `0.x ≥ 0.y`, so that its top limb is below `y`), divided limb by limb by divisions of two
/// limbs by one with the reciprocal of `y`, every limb of `N` below `x`'s zero; then one limb more
/// of quotient, the word below, and the last remainder the sticky bit. The quotient has its top
/// bit set, `0.x·2^-c / 0.y` being in `[1/2, 1)`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn div_long1(
    negative: bool,
    x: &[u64],
    ex: i64,
    y: u64,
    ey: i64,
    prec: u64,
    round: Round,
    out: &mut [u64],
) -> super::same::Put {
    let n = out.len();
    debug_assert!(n >= 3 && x.len() <= 2 && y >> 63 == 1);
    let xw = wide(x);
    let c = u32::from(xw >> 64 >= u128::from(y));
    // The top limbs of `N`: `xw·2^-c` in two limbs, then zeros, with the bit shifted out.
    let t = xw >> c;
    let spill = if c == 1 { (xw as u64) << 63 } else { 0 };
    let v = nat::invert_limb(y);
    let mut r = (t >> 64) as u64;
    debug_assert!(r < y);
    let feed = [t as u64, spill];
    for i in (0..n).rev() {
        let u = if n - 1 - i < 2 { feed[n - 1 - i] } else { 0 };
        let (q, rr) = nat::div_2by1(r, u, y, v);
        out[i] = q;
        r = rr;
    }
    let (low, r) = nat::div_2by1(r, 0, y, v);
    super::same::finish(
        negative,
        out,
        low,
        r != 0,
        i128::from(ex - ey + i64::from(c)),
        prec,
        round,
    )
}
