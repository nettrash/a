//! The kernels of the short operations and of the forms into an existing number: results written
//! into the vectors of the parts they replace, exact sums of products on the stack, and the
//! roundings of one- and two-limb results in registers.
//!
//! - Products of parts of one limb each are exact in 128 bits, and a part `ac - bd` is the sum of
//!   two of them rounded once by the two-limb addition of [`small`](super::super::small), which
//!   takes both terms exactly and keeps what lies beyond its 256 bits as a sticky bit.
//! - Sums of the exact products of longer parts (to [`FACTOR`] limbs each), and of parts, are
//!   exact in a two's-complement accumulator of at most [`ACC`] limbs when their bits span it (a
//!   vector the thread keeps), and are then rounded once into the result's vector; [`exact_sum`]
//!   says when they do not.
//! - A square root of such an exact sum `S` takes the floor of the root of its top 256 bits (or
//!   of `S` moved up to 255 or 256 bits), the shift of even parity with the exponent of the lowest
//!   bit: `⌊√⌊x⌋⌋ = ⌊√x⌋` for a real `x ≥ 0`, so the root of 128 bits is that of `S` cut, and the
//!   remainder and the bits cut off are its sticky bit; the round bit lies in the root for a
//!   precision to 127 bits.

use super::super::general::{self, Op};
use super::super::limbs::{self, Work};
use super::super::round::{Rounded, round_into, seal, signed};
use super::super::small::{add2, div1, div2, mul2, round_top, round_top2, wide_mul};
use super::super::sum::Zeros;
use super::super::{BigFloat, Kind, Round};
use crate::math::numbers::nat;
use core::cmp::Ordering;

/// The longest factor of a product the accumulator takes, in limbs (1024 bits): to there the
/// schoolbook products are those the crate's multiplication takes too.
pub(super) const FACTOR: usize = 16;

/// The most limbs of the accumulator of an exact sum: 10240 bits, its sign bit included.
pub(super) const ACC: usize = 160;

std::thread_local! {
    /// The vectors of the thread's accumulators, kept between sums.
    static POOL: core::cell::Cell<Vec<Vec<u64>>> = const { core::cell::Cell::new(Vec::new()) };
}

/// The vector of the mantissa of `x`, taken out (`x` is left NaN), or one of the thread's pool
/// when `x` has none: for a result written in place.
#[inline]
pub(crate) fn take_mant(x: &mut BigFloat) -> Vec<u64> {
    match core::mem::replace(&mut x.kind, Kind::Nan) {
        Kind::Finite { mant, .. } => mant,
        _ => pool_take(),
    }
}

/// Sets `out` to the value `kind` that holds no mantissa (NaN, an infinity or a zero), giving
/// its vector to the thread's pool.
#[inline]
pub(crate) fn put_kind(out: &mut BigFloat, kind: Kind) {
    debug_assert!(!matches!(kind, Kind::Finite { .. }));
    if let Kind::Finite { mant, .. } = core::mem::replace(&mut out.kind, kind) {
        pool_give(mant);
    }
}

/// Sets `out` to the zero of the sign: exact.
#[inline]
pub(crate) fn put_zero(out: &mut BigFloat, negative: bool) -> Ordering {
    put_kind(out, Kind::Zero { negative });
    Ordering::Equal
}

/// Sets `out` to the result `r` rounded in registers at the precision of `out` (at most 128 bits),
/// into its limbs when it has a mantissa, and the direction of the rounding.
#[inline]
pub(crate) fn put(out: &mut BigFloat, r: Rounded, round: Round) -> Ordering {
    let prec = out.prec;
    debug_assert!(prec <= 128 && r.m >> 127 == 1);
    let fits = (i128::from(BigFloat::EXP_MIN)..=i128::from(BigFloat::EXP_MAX)).contains(&r.exp);
    if fits
        && let Kind::Finite {
            negative,
            exp,
            mant,
        } = &mut out.kind
    {
        // A finite number of the precision has `ceil(prec / 64)` limbs.
        debug_assert_eq!(mant.len(), prec.div_ceil(64) as usize);
        *negative = r.negative;
        *exp = r.exp as i64;
        if let [lo, hi] = mant.as_mut_slice() {
            *lo = r.m as u64;
            *hi = (r.m >> 64) as u64;
        } else {
            mant[0] = (r.m >> 64) as u64;
        }
        return signed(r.negative, r.magnitude);
    }
    let v = take_mant(out);
    let (kind, o) = seal(r, prec, round, v);
    out.kind = kind;
    o
}

/// Sets `out` to `±m·2^lsb` (with `sticky`, a number strictly between it and one unit more)
/// rounded at the precision of `out`, into its vector, and the direction of the rounding.
#[inline]
pub(crate) fn put_limbs(
    out: &mut BigFloat,
    negative: bool,
    m: &[u64],
    lsb: i128,
    sticky: bool,
    round: Round,
) -> Ordering {
    let v = take_mant(out);
    let (kind, o) = round_into(negative, m, lsb, sticky, out.prec, round, v);
    out.kind = kind;
    o
}

/// Sets `out` to `x` rounded at the precision of `out`, the sign turned over when `negate`, and
/// returns the direction of the rounding, as MPFR's `mpfr_set` (and `mpfr_neg`).
pub(super) fn set_part(out: &mut BigFloat, x: &BigFloat, negate: bool, round: Round) -> Ordering {
    match &x.kind {
        Kind::Finite {
            negative,
            exp,
            mant,
        } => {
            let negative = *negative != negate;
            if mant.len() as u64 == out.prec.div_ceil(64)
                && let Kind::Finite {
                    negative: n,
                    exp: e,
                    mant: m,
                } = &mut out.kind
                && out.prec >= x.prec
            {
                // Exact: the bits below the precision of `x` are zeros.
                *n = negative;
                *e = *exp;
                m.copy_from_slice(mant);
                return Ordering::Equal;
            }
            let lsb = i128::from(*exp) - 64 * mant.len() as i128;
            put_limbs(out, negative, mant, lsb, false, round)
        }
        Kind::Nan => {
            put_kind(out, Kind::Nan);
            Ordering::Equal
        }
        Kind::Inf { negative } => {
            put_kind(
                out,
                Kind::Inf {
                    negative: *negative != negate,
                },
            );
            Ordering::Equal
        }
        Kind::Zero { negative } => put_zero(out, *negative != negate),
    }
}

/// An exact real value of at most 128 bits: a signed zero, or `±0.m·2^exp` with the top bit of
/// `m` set.
#[derive(Clone, Copy)]
pub(super) enum Short {
    Zero(bool),
    Val { negative: bool, m: u128, exp: i128 },
}

/// The exact product of two finite parts of one limb each (to 64 bits), `None` for a longer part.
#[inline(always)]
pub(super) fn short_product(x: &BigFloat, y: &BigFloat, negate: bool) -> Option<Short> {
    let negative = (x.is_sign_negative() != y.is_sign_negative()) != negate;
    match (&x.kind, &y.kind) {
        (
            Kind::Finite {
                exp: ex, mant: mx, ..
            },
            Kind::Finite {
                exp: ey, mant: my, ..
            },
        ) => {
            let ([a], [b]) = (mx.as_slice(), my.as_slice()) else {
                return None;
            };
            let p = u128::from(*a) * u128::from(*b);
            let e = i128::from(*ex) + i128::from(*ey);
            // Both top bits set: the product's top bit at 127 or 126.
            Some(if p >> 127 == 0 {
                Short::Val {
                    negative,
                    m: p << 1,
                    exp: e - 1,
                }
            } else {
                Short::Val {
                    negative,
                    m: p,
                    exp: e,
                }
            })
        }
        (Kind::Zero { .. }, Kind::Finite { mant, .. })
        | (Kind::Finite { mant, .. }, Kind::Zero { .. })
            if mant.len() == 1 =>
        {
            Some(Short::Zero(negative))
        }
        (Kind::Zero { .. }, Kind::Zero { .. }) => Some(Short::Zero(negative)),
        _ => None,
    }
}

/// The exact value `x·2^k` of 128 bits at most rounded to `prec ≤ 128` bits in registers.
#[inline(always)]
fn round_short(negative: bool, m: u128, exp: i128, prec: u64, round: Round) -> Rounded {
    if prec <= 64 {
        round_top(negative, m, exp, false, prec, round)
    } else {
        round_top2(negative, m, 0, exp, false, prec, round)
    }
}

/// The sum `p + q` of two exact values of 128 bits rounded once in registers at the precision of
/// the result (at most 128 bits): `Ok` of the rounding, or `Err` of the sign of an exact zero
/// (IEEE 754's: two zeros of one sign that sign, otherwise `+0`, or `-0` rounding down).
#[inline(always)]
pub(super) fn short_sum(p: Short, q: Short, prec: u64, round: Round) -> Result<Rounded, bool> {
    match (p, q) {
        (
            Short::Val {
                negative: nx,
                m: x,
                exp: ex,
            },
            Short::Val {
                negative: ny,
                m: y,
                exp: ey,
            },
        ) => add2(nx, x, ex, ny, y, ey, prec, round).ok_or(round == Round::Down),
        (Short::Val { negative, m, exp }, Short::Zero(_))
        | (Short::Zero(_), Short::Val { negative, m, exp }) => {
            Ok(round_short(negative, m, exp, prec, round))
        }
        (Short::Zero(a), Short::Zero(b)) => Err(if a == b { a } else { round == Round::Down }),
    }
}

/// [`short_sum`] scaled by `2^k`, exactly.
#[inline(always)]
pub(super) fn short_scaled(s: Short, k: i128) -> Short {
    match s {
        Short::Val { negative, m, exp } => Short::Val {
            negative,
            m,
            exp: exp + k,
        },
        z => z,
    }
}

/// An exact real value of at most 256 bits: a signed zero, or `±0.(h, l)·2^exp` with the top bit
/// of `h` set.
#[derive(Clone, Copy)]
pub(super) enum Long {
    Zero(bool),
    Val {
        negative: bool,
        h: u128,
        l: u128,
        exp: i128,
    },
}

/// The mantissa of one or two limbs `m` as a `u128` aligned at its top bit.
#[inline(always)]
fn wide(m: &[u64]) -> Option<u128> {
    match *m {
        [x] => Some(u128::from(x) << 64),
        [lo, hi] => Some(u128::from(hi) << 64 | u128::from(lo)),
        _ => None,
    }
}

/// The exact product of two finite parts of at most two limbs each (to 128 bits), of 256 bits;
/// `None` for a longer part.
#[inline(always)]
pub(super) fn long_product(x: &BigFloat, y: &BigFloat, negate: bool) -> Option<Long> {
    let negative = (x.is_sign_negative() != y.is_sign_negative()) != negate;
    match (&x.kind, &y.kind) {
        (
            Kind::Finite {
                exp: ex, mant: mx, ..
            },
            Kind::Finite {
                exp: ey, mant: my, ..
            },
        ) => {
            let (a, b) = (wide(mx)?, wide(my)?);
            let (h, l) = wide_mul(a, b);
            let e = i128::from(*ex) + i128::from(*ey);
            Some(if h >> 127 == 0 {
                Long::Val {
                    negative,
                    h: h << 1 | l >> 127,
                    l: l << 1,
                    exp: e - 1,
                }
            } else {
                Long::Val {
                    negative,
                    h,
                    l,
                    exp: e,
                }
            })
        }
        (Kind::Zero { .. }, Kind::Finite { mant, .. })
        | (Kind::Finite { mant, .. }, Kind::Zero { .. })
            if mant.len() <= 2 =>
        {
            Some(Long::Zero(negative))
        }
        (Kind::Zero { .. }, Kind::Zero { .. }) => Some(Long::Zero(negative)),
        _ => None,
    }
}

/// `(a2, a1, a0)·2^-d` over 384 bits, `d ≤ 512`, and whether a bit was shifted out.
#[inline(always)]
fn shr3(a2: u128, a1: u128, a0: u128, d: u128) -> (u128, u128, u128, bool) {
    // Whether the bits of `v`, bits `from..from + 128` of the 384, below bit `d` hold a one.
    let out = |v: u128, from: u128| -> bool {
        if d <= from {
            false
        } else if d - from >= 128 {
            v != 0
        } else {
            v & ((1u128 << (d - from)) - 1) != 0
        }
    };
    let sticky = out(a0, 0) || out(a1, 128) || out(a2, 256);
    let (q, r) = (d / 128, (d % 128) as u32);
    let limb = [a0, a1, a2, 0, 0, 0, 0];
    let at = |i: usize| -> u128 {
        let i = i + q as usize;
        let lo = limb[i.min(6)] >> r;
        if r == 0 {
            lo
        } else {
            lo | limb[(i + 1).min(6)] << (128 - r)
        }
    };
    (at(2), at(1), at(0), sticky)
}

/// `±x ± y` for the exact 256-bit values `x = (xh, xl)` and `y = (yh, yl)`, aligned at their top
/// bits, `0.x·2^ex`, rounded to `prec ≤ 128` bits; `None` for an exact zero.
///
/// The larger term (in magnitude for a difference) `x` is at the top of a window of 384 bits and
/// `y` shifted right in it by the difference of the exponents, the bits shifted out a sticky bit,
/// which only a difference of more than 128 makes; the sum or difference is exact in the window
/// otherwise. A difference with a sticky bit is computed as one unit less, strictly below the
/// value by less than a unit, and keeps its top bit within one, so that after it is moved back to
/// the top the value lies strictly between it and two units more: no halfway point of the
/// precision (multiples of `2^127` units) lies there, and the window and the sticky bit round it.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn add4(
    nx: bool,
    xh: u128,
    xl: u128,
    ex: i128,
    ny: bool,
    yh: u128,
    yl: u128,
    ey: i128,
    prec: u64,
    round: Round,
) -> Option<Rounded> {
    let swap = ex < ey || (ex == ey && nx != ny && (xh, xl) < (yh, yl));
    let ((nx, xh, xl, ex), (ny, yh, yl, ey)) = if swap {
        ((ny, yh, yl, ey), (nx, xh, xl, ex))
    } else {
        ((nx, xh, xl, ex), (ny, yh, yl, ey))
    };
    let d = ex - ey;
    let d = if d > 512 { 512 } else { d as u128 };
    let (y2, y1, y0, sticky) = shr3(yh, yl, 0, d);
    if nx == ny {
        let (s1, c1) = xl.overflowing_add(y1);
        let (s2, c2) = xh.overflowing_add(y2);
        let (s2, c3) = s2.overflowing_add(u128::from(c1));
        let (s0, s1, s2, e, sticky) = if c2 || c3 {
            (
                s1 << 127 | y0 >> 1,
                s2 << 127 | s1 >> 1,
                1 << 127 | s2 >> 1,
                ex + 1,
                sticky || y0 & 1 == 1,
            )
        } else {
            (y0, s1, s2, ex, sticky)
        };
        return Some(round_top2(nx, s2, s1, e, sticky || s0 != 0, prec, round));
    }
    // `x - y - sticky`, three limbs.
    let (s0, b0) = 0u128.overflowing_sub(y0);
    let (s0, b0b) = s0.overflowing_sub(u128::from(sticky));
    let (s1, b1) = xl.overflowing_sub(y1);
    let (s1, b1b) = s1.overflowing_sub(u128::from(b0 || b0b));
    let s2 = xh - y2 - u128::from(b1 || b1b);
    if !sticky && s2 == 0 && s1 == 0 && s0 == 0 {
        return None;
    }
    // Moved up to the top.
    let (s2, s1, s0, e) = if s2 != 0 {
        let z = s2.leading_zeros();
        if z == 0 {
            (s2, s1, s0, ex)
        } else {
            (
                s2 << z | s1 >> (128 - z),
                s1 << z | s0 >> (128 - z),
                s0 << z,
                ex - i128::from(z),
            )
        }
    } else if s1 != 0 {
        let z = s1.leading_zeros();
        let (h, m) = if z == 0 {
            (s1, s0)
        } else {
            (s1 << z | s0 >> (128 - z), s0 << z)
        };
        (h, m, 0, ex - 128 - i128::from(z))
    } else {
        let z = s0.leading_zeros();
        (s0 << z, 0, 0, ex - 256 - i128::from(z))
    };
    Some(round_top2(nx, s2, s1, e, sticky || s0 != 0, prec, round))
}

/// The sum `p + q` of two exact values of 256 bits rounded once at the precision of the result
/// (at most 128 bits), as [`short_sum`].
#[inline(always)]
pub(super) fn long_sum(p: Long, q: Long, prec: u64, round: Round) -> Result<Rounded, bool> {
    match (p, q) {
        (
            Long::Val {
                negative: nx,
                h: xh,
                l: xl,
                exp: ex,
            },
            Long::Val {
                negative: ny,
                h: yh,
                l: yl,
                exp: ey,
            },
        ) => add4(nx, xh, xl, ex, ny, yh, yl, ey, prec, round).ok_or(round == Round::Down),
        (
            Long::Val {
                negative,
                h,
                l,
                exp,
            },
            Long::Zero(_),
        )
        | (
            Long::Zero(_),
            Long::Val {
                negative,
                h,
                l,
                exp,
            },
        ) => Ok(round_top2(negative, h, l, exp, false, prec, round)),
        (Long::Zero(a), Long::Zero(b)) => Err(if a == b { a } else { round == Round::Down }),
    }
}

/// An exact value of 256 bits at most, times `2^k`, rounded once to `prec ≤ 128` bits.
#[inline(always)]
pub(super) fn long_round(s: Long, k: i128, prec: u64, round: Round) -> Result<Rounded, bool> {
    match s {
        Long::Val {
            negative,
            h,
            l,
            exp,
        } => Ok(round_top2(negative, h, l, exp + k, false, prec, round)),
        Long::Zero(negative) => Err(negative),
    }
}

/// Sets `out` to a result of [`short_sum`].
#[inline(always)]
pub(crate) fn put_short(out: &mut BigFloat, r: Result<Rounded, bool>, round: Round) -> Ordering {
    match r {
        Ok(r) => put(out, r, round),
        Err(negative) => put_zero(out, negative),
    }
}

/// One term of an exact sum: `±x`, or `±x·y`, of finite numbers.
#[derive(Clone, Copy)]
pub(crate) struct Term<'a> {
    pub(crate) x: &'a BigFloat,
    pub(crate) y: Option<&'a BigFloat>,
    pub(crate) negate: bool,
}

impl<'a> Term<'a> {
    /// `±x·y`, `-` when `negate`.
    #[inline(always)]
    pub(crate) fn product(x: &'a BigFloat, y: &'a BigFloat, negate: bool) -> Term<'a> {
        Term {
            x,
            y: Some(y),
            negate,
        }
    }

    /// `±x`.
    #[inline(always)]
    pub(crate) fn one(x: &'a BigFloat, negate: bool) -> Term<'a> {
        Term { x, y: None, negate }
    }
}

/// An exact sum: a zero (and how it came), or `±m·2^lsb` for the limbs `m = limbs[..len]`, the
/// top one not zero; the limbs after them hold a product while the sum is made. Its vector comes
/// from the thread's pool and goes back to it.
pub(crate) struct Acc {
    limbs: Vec<u64>,
    len: usize,
    pub(crate) lsb: i128,
    pub(crate) negative: bool,
    pub(crate) zero: Option<Zeros>,
}

impl Acc {
    /// An accumulator of a sum of no terms.
    #[inline]
    pub(crate) fn new() -> Acc {
        Acc {
            limbs: pool_take(),
            len: 0,
            lsb: 0,
            negative: false,
            zero: Some(Zeros::Positive),
        }
    }

    /// The magnitude's limbs.
    #[inline(always)]
    pub(super) fn m(&self) -> &[u64] {
        &self.limbs[..self.len]
    }
}

/// A vector of the thread's pool, empty or not.
#[inline]
pub(crate) fn pool_take() -> Vec<u64> {
    let mut pool = POOL.take();
    let v = pool.pop().unwrap_or_default();
    POOL.set(pool);
    v
}

/// Gives a vector back to the thread's pool, which keeps [`POOL_LEN`] of them: four of up to
/// `2^17` limbs, the others of up to 64 (the mantissas of the matrices' factors).
#[inline]
pub(crate) fn pool_give(v: Vec<u64>) {
    let mut pool = POOL.take();
    let cap = v.capacity();
    if cap > 0 && (pool.len() < 4 && cap <= 1 << 17 || pool.len() < POOL_LEN && cap <= 64) {
        pool.push(v);
    }
    POOL.set(pool);
}

/// The most vectors the thread's pool keeps.
const POOL_LEN: usize = 1024;

/// `x` at its own precision with its mantissa in a vector of the thread's pool: a copy that
/// does not allocate when the pool has one.
pub(crate) fn pooled_copy(x: &BigFloat) -> BigFloat {
    match &x.kind {
        Kind::Finite {
            negative,
            exp,
            mant,
        } => {
            let mut v = pool_take();
            v.clear();
            v.extend_from_slice(mant);
            BigFloat {
                prec: x.prec,
                kind: Kind::Finite {
                    negative: *negative,
                    exp: *exp,
                    mant: v,
                },
            }
        }
        _ => x.clone(),
    }
}

/// Gives the vector of `x` to the thread's pool.
#[inline]
pub(crate) fn recycle(x: BigFloat) {
    if let Kind::Finite { mant, .. } = x.kind {
        pool_give(mant);
    }
}

impl Drop for Acc {
    #[inline]
    fn drop(&mut self) {
        pool_give(core::mem::take(&mut self.limbs));
    }
}

/// The significant limbs of a finite number that is not zero (those below the precision that
/// are zero left out) and the exponent of their lowest bit.
#[inline(always)]
fn significant(x: &BigFloat) -> (bool, &[u64], i128) {
    let (negative, m, lsb) = x.parts().expect("a finite number that is not zero");
    let z = limbs::low_zeros(m);
    (negative, &m[z..], lsb + 64 * z as i128)
}

/// `p = x·y` for the first `x.len() + y.len()` limbs of `p`, by the schoolbook method on the
/// crate's row kernel.
#[inline(always)]
fn mul_to(p: &mut [u64], x: &[u64], y: &[u64]) {
    let n = x.len();
    p[..n].fill(0);
    for (j, &b) in y.iter().enumerate() {
        p[j + n] = nat::addmul_1(&mut p[j..j + n], x, b);
    }
}

/// `acc += m·2^r` (or `-=` when `sub`), two's complement, the carry out of the top dropped, for
/// `r < 64` and `acc` longer than `m`.
#[inline(always)]
fn add_shifted(acc: &mut [u64], m: &[u64], r: u32, sub: bool) {
    debug_assert!(acc.len() > m.len() && r < 64);
    let mut out = 0u64;
    let mut c = false;
    let step = |slot: &mut u64, v: u64, c: &mut bool| {
        if sub {
            let (s, b1) = slot.overflowing_sub(v);
            let (s, b2) = s.overflowing_sub(u64::from(*c));
            *slot = s;
            *c = b1 || b2;
        } else {
            let (s, c1) = slot.overflowing_add(v);
            let (s, c2) = s.overflowing_add(u64::from(*c));
            *slot = s;
            *c = c1 || c2;
        }
    };
    let n = m.len();
    for i in 0..n {
        let v = if r == 0 { m[i] } else { m[i] << r | out };
        out = if r == 0 { 0 } else { m[i] >> (64 - r) };
        step(&mut acc[i], v, &mut c);
    }
    step(&mut acc[n], out, &mut c);
    for slot in &mut acc[n + 1..] {
        if !c {
            break;
        }
        step(slot, 0, &mut c);
    }
}

/// Bits `k..k + 256` of the magnitude `m` (the bits outside it zero), the least significant limb
/// first, and whether a bit below bit `k` is set, for `|k| < 2^62`: [`limbs::extract`] on
/// 64-bit positions.
#[inline(always)]
fn window4(m: &[u64], k: i64) -> ([u64; 4], bool) {
    let (q, b) = (k >> 6, (k & 63) as u32);
    let at = |i: i64| -> u64 {
        if (0..m.len() as i64).contains(&i) {
            m[i as usize]
        } else {
            0
        }
    };
    let mut w = [0u64; 4];
    for (j, v) in w.iter_mut().enumerate() {
        let i = q + j as i64;
        *v = if b == 0 {
            at(i)
        } else {
            at(i) >> b | at(i + 1) << (64 - b)
        };
    }
    let below = if k <= 0 {
        false
    } else {
        let full = (q as usize).min(m.len());
        m[..full].iter().any(|&v| v != 0)
            || (b != 0 && (q as usize) < m.len() && m[q as usize] << (64 - b) != 0)
    };
    (w, below)
}

/// The number of bits of a bound `2^b ≥ n` of `n ≥ 1`.
#[inline(always)]
fn log2_ceil(n: usize) -> i128 {
    if n <= 1 {
        0
    } else {
        i128::from(usize::BITS - (n - 1).leading_zeros())
    }
}

/// The exact sum of `terms`, all finite, into `acc`: `false` (and `acc` as it was) when a factor
/// of a product is longer than [`FACTOR`] limbs or the bits of the terms span more than the
/// accumulator holds ([`ACC`] limbs, or twice the longest term).
///
/// The terms that are not zero are added at their places in a two's-complement accumulator from
/// the lowest bit of all to past the top bit of all, `|Σ| < count·2^top` and a sign bit above, so
/// that nothing carries out; a sum of zeros only is the zero IEEE 754's sums give.
pub(crate) fn exact_sum(terms: &[Term<'_>], acc: &mut Acc) -> bool {
    exact_sum_by(terms.len(), |i| terms[i], acc)
}

/// [`exact_sum`] of the `len` terms `term(0)`, `term(1)`, …
pub(crate) fn exact_sum_by<'a>(
    len: usize,
    term: impl Fn(usize) -> Term<'a>,
    acc: &mut Acc,
) -> bool {
    let mut zeros: Option<Zeros> = None;
    let mut top = i128::MIN;
    let mut low = i128::MAX;
    let mut count = 0usize;
    let mut longest = 0usize;
    let mut widest = 0usize;
    for t in (0..len).map(&term) {
        let zero = t.x.is_zero() || t.y.is_some_and(BigFloat::is_zero);
        if zero {
            let negative =
                (t.x.is_sign_negative() != t.y.is_some_and(BigFloat::is_sign_negative)) != t.negate;
            let z = if negative {
                Zeros::Negative
            } else {
                Zeros::Positive
            };
            zeros = Some(match zeros {
                Some(w) if w != z => Zeros::Mixed,
                _ => z,
            });
            continue;
        }
        let (_, mx, lx) = significant(t.x);
        if mx.len() > FACTOR && t.y.is_some() {
            return false;
        }
        widest = widest.max(mx.len());
        let (mut l, mut e) = (lx, lx + 64 * mx.len() as i128);
        if let Some(y) = t.y {
            let (_, my, ly) = significant(y);
            if my.len() > FACTOR {
                return false;
            }
            l += ly;
            e += ly + 64 * my.len() as i128;
            longest = longest.max(mx.len() + my.len());
        }
        top = top.max(e);
        low = low.min(l);
        count += 1;
    }
    if count == 0 {
        acc.zero = Some(zeros.unwrap_or(Zeros::Positive));
        acc.len = 0;
        return true;
    }
    let bits = top - low + log2_ceil(count) + 2;
    let n = (bits as usize).div_ceil(64) + 1;
    if n > ACC.max(2 * widest + 4) {
        return false;
    }
    acc.limbs.clear();
    acc.limbs.resize(n + longest, 0);
    let (a, p) = acc.limbs.split_at_mut(n);
    for t in (0..len).map(&term) {
        if t.x.is_zero() || t.y.is_some_and(BigFloat::is_zero) {
            continue;
        }
        let (nx, mx, lx) = significant(t.x);
        let (m, lsb, negative): (&[u64], i128, bool) = match t.y {
            None => (mx, lx, nx != t.negate),
            Some(y) => {
                let (ny, my, ly) = significant(y);
                let k = mx.len() + my.len();
                if let ([x], [y]) = (mx, my) {
                    let q = u128::from(*x) * u128::from(*y);
                    p[0] = q as u64;
                    p[1] = (q >> 64) as u64;
                } else {
                    mul_to(p, mx, my);
                }
                (&p[..k], lx + ly, (nx != ny) != t.negate)
            }
        };
        let off = (lsb - low) as u64;
        let q = (off / 64) as usize;
        add_shifted(&mut a[q..], m, (off % 64) as u32, negative);
    }
    let negative = a[n - 1] >> 63 == 1;
    if negative {
        // `-a = !a + 1`.
        let mut carry = true;
        for x in a.iter_mut() {
            let (s, c) = (!*x).overflowing_add(u64::from(carry));
            *x = s;
            carry = c;
        }
    }
    let mut len = n;
    while len > 0 && a[len - 1] == 0 {
        len -= 1;
    }
    acc.len = len;
    acc.lsb = low;
    acc.negative = negative;
    acc.zero = (len == 0).then_some(Zeros::Mixed);
    true
}

/// Sets `out` to the exact sum `acc` rounded at its precision, and the direction of the rounding;
/// the sum scaled by `2^k` first, exactly.
#[inline]
pub(crate) fn put_acc(out: &mut BigFloat, acc: &Acc, k: i128, round: Round) -> Ordering {
    match acc.zero {
        Some(z) => put_zero(out, z.negative(round)),
        None => put_limbs(out, acc.negative, acc.m(), acc.lsb + k, false, round),
    }
}

/// The square root of the exact sum `acc`, not below zero and not zero, rounded to `prec ≤ 127`
/// bits in registers (see the [module](self)).
pub(super) fn sqrt_acc(acc: &Acc, prec: u64, round: Round) -> Rounded {
    debug_assert!(acc.zero.is_none() && !acc.negative && prec <= 127);
    let m = acc.m();
    let bits = limbs::bit_len(m) as i64;
    // `R = ⌊S / 2^k⌋` of 255 or 256 bits, `k + lsb` even.
    let mut k = bits - 256;
    if (i128::from(k) + acc.lsb).rem_euclid(2) == 1 {
        k += 1;
    }
    let (r, cut) = window4(m, k);
    let (s, rem) = nat::sqrt_rem_wide(r);
    // `√(S·2^lsb) = √(S/2^k)·2^((k + lsb)/2)` and `s` of 128 bits, its top bit set.
    let exp = 128 + (i128::from(k) + acc.lsb) / 2;
    round_top2(false, s, 0, exp, rem != [0; 3] || cut, prec, round)
}

/// An exact value of 128 bits at most rounded once to `prec ≤ 128` bits in registers: `Err` of
/// the sign of a zero.
#[inline(always)]
pub(super) fn short_round(s: Short, prec: u64, round: Round) -> Result<Rounded, bool> {
    match s {
        Short::Val { negative, m, exp } => Ok(round_short(negative, m, exp, prec, round)),
        Short::Zero(negative) => Err(negative),
    }
}

/// The parts of `(a + bi)(c + di)` for finite parts of one limb each and results to 128 bits,
/// `ac - bd` and `ad + bc` each the exact sum of two exact products rounded once in registers;
/// `None` for longer parts or results.
#[inline(always)]
pub(super) fn short_mul(
    [a, b, c, d]: [&BigFloat; 4],
    prec: (u64, u64),
    round: (Round, Round),
) -> Option<[Result<Rounded, bool>; 2]> {
    if prec.0 > 128 || prec.1 > 128 {
        return None;
    }
    if let (Some(ac), Some(bd), Some(ad), Some(bc)) = (
        short_product(a, c, false),
        short_product(b, d, true),
        short_product(a, d, false),
        short_product(b, c, false),
    ) {
        return Some([
            short_sum(ac, bd, prec.0, round.0),
            short_sum(ad, bc, prec.1, round.1),
        ]);
    }
    let ac = long_product(a, c, false)?;
    let bd = long_product(b, d, true)?;
    let ad = long_product(a, d, false)?;
    let bc = long_product(b, c, false)?;
    Some([
        long_sum(ac, bd, prec.0, round.0),
        long_sum(ad, bc, prec.1, round.1),
    ])
}

/// [`short_mul`] for `(a + bi)²`: `a² - b²` and `2ab`.
#[inline(always)]
pub(super) fn short_sqr(
    [a, b]: [&BigFloat; 2],
    prec: (u64, u64),
    round: (Round, Round),
) -> Option<[Result<Rounded, bool>; 2]> {
    if prec.0 > 128 || prec.1 > 128 {
        return None;
    }
    if let (Some(aa), Some(bb), Some(ab)) = (
        short_product(a, a, false),
        short_product(b, b, true),
        short_product(a, b, false),
    ) {
        return Some([
            short_sum(aa, bb, prec.0, round.0),
            short_round(short_scaled(ab, 1), prec.1, round.1),
        ]);
    }
    let aa = long_product(a, a, false)?;
    let bb = long_product(b, b, true)?;
    let ab = long_product(a, b, false)?;
    Some([
        long_sum(aa, bb, prec.0, round.0),
        long_round(ab, 1, prec.1, round.1),
    ])
}

/// The exact parts `ac - bd` and `ad + bc` of `(a + bi)(c + di)` for finite parts into the
/// accumulators, `false` when they do not hold them.
#[inline]
pub(super) fn exact_mul([a, b, c, d]: [&BigFloat; 4], acc: &mut [Acc; 2]) -> bool {
    let [x, y] = acc;
    let re = [Term::product(a, c, false), Term::product(b, d, true)];
    let im = [Term::product(a, d, false), Term::product(b, c, false)];
    exact_sum(&re, x) && exact_sum(&im, y)
}

/// [`exact_mul`] for `(a + bi)²`: `a² - b²` and `2ab`.
#[inline]
pub(super) fn exact_sqr([a, b]: [&BigFloat; 2], acc: &mut [Acc; 2]) -> bool {
    let [x, y] = acc;
    let re = [Term::product(a, a, false), Term::product(b, b, true)];
    let im = [Term::product(a, b, false)];
    if !(exact_sum(&re, x) && exact_sum(&im, y)) {
        return false;
    }
    y.lsb += 1;
    true
}

/// The exact product `x·y` of two finite numbers that are not zero into `p`, aligned at its top
/// bit: the sign, the exponent of `0.p·2^exp`.
#[inline]
fn product_into(
    p: &mut Vec<u64>,
    scratch: &mut Vec<u64>,
    x: &BigFloat,
    y: &BigFloat,
    negate: bool,
) -> Op<'static> {
    let (Some(a), Some(b)) = (x.op(), y.op()) else {
        unreachable!("finite numbers that are not zero");
    };
    let (a, b) = (a.strip(), b.strip());
    if core::ptr::eq(x, y) {
        nat::sqr_into(p, a.m, scratch);
    } else {
        nat::mul_into(p, a.m, b.m, scratch);
    }
    let mut exp = a.exp + b.exp;
    if p[p.len() - 1] >> 63 == 0 {
        // The top bit one below (both factors' top bits are set): moved up, nothing out.
        let mut carry = 0u64;
        for v in p.iter_mut() {
            let next = *v >> 63;
            *v = *v << 1 | carry;
            carry = next;
        }
        exp -= 1;
    }
    Op {
        negative: (a.negative != b.negative) != negate,
        m: &[],
        exp,
    }
}

/// Sets `out` to `±x1·y1 ± x2·y2` (`-` for a term when its flag is set) for finite numbers that
/// are not zero, rounded once at the precision of `out`: both products exact in the thread's
/// working memory, then the crate's window addition, which reads of them only the limbs the
/// rounding needs (MPFR's `mpfr_fmma`).
pub(super) fn sum_products(
    out: &mut BigFloat,
    (x1, y1, n1): (&BigFloat, &BigFloat, bool),
    (x2, y2, n2): (&BigFloat, &BigFloat, bool),
    round: Round,
) -> Ordering {
    let v = take_mant(out);
    let prec = out.prec;
    let (kind, o) = limbs::with_work(|w| {
        let Work { a, b, scratch, .. } = w;
        let p = product_into(a, scratch, x1, y1, n1);
        let q = product_into(b, scratch, x2, y2, n2);
        general::add(Op { m: a, ..p }, Op { m: b, ..q }, prec, round, v)
    });
    out.kind = kind;
    o
}

/// Sets `out` to `±x·y·2^k` for finite numbers that are not zero, rounded once at its precision.
pub(super) fn scaled_product(
    out: &mut BigFloat,
    x: &BigFloat,
    y: &BigFloat,
    k: i128,
    round: Round,
) -> Ordering {
    let v = take_mant(out);
    let prec = out.prec;
    let (kind, o) = limbs::with_work(|w| {
        let Work { a, scratch, .. } = w;
        let p = product_into(a, scratch, x, y, false);
        let lsb = p.exp + k - 64 * a.len() as i128;
        round_into(p.negative, a, lsb, false, prec, round, v)
    });
    out.kind = kind;
    o
}

/// Sets `out` to `√(x² + y²)` for finite numbers that are not zero, rounded once at its
/// precision: the sum of the exact squares cut toward zero to `q = 2·prec + 4` bits at least
/// (moved by an even power of two `2^(2k)` into the exponent range), and when it is not exact,
/// the number halfway between the cut sum and the next, whose root rounds as the sum's (see
/// [`sqrt_summed`](super::super::sum::sqrt_summed)): no root of a number of the precision, nor a
/// halfway point between two, has its square strictly between them, as those squares are
/// multiples of `2^(2(e - prec - 1))` for the root's exponent `e`, and the cut sum's last place is
/// below that by `2^(q - 2·prec - 2)` at least.
pub(super) fn hypot_into(out: &mut BigFloat, x: &BigFloat, y: &BigFloat, round: Round) -> Ordering {
    let prec = out.prec;
    let mut q = 2 * prec + 4;
    if q.is_multiple_of(64) {
        // A bit below the last place in its limbs, for the halfway point.
        q += 1;
    }
    let (Some(ex), Some(ey)) = (x.exponent(), y.exponent()) else {
        unreachable!("finite numbers that are not zero");
    };
    let k = i128::from(ex.max(ey));
    let mut s = Acc::new();
    let mut v = core::mem::take(&mut s.limbs);
    let (kind, o) = limbs::with_work(|w| {
        let Work { a, b, scratch, .. } = w;
        let p = product_into(a, scratch, x, x, false);
        let r = product_into(b, scratch, y, y, false);
        let (p, r) = (
            Op {
                m: a,
                exp: p.exp - 2 * k,
                ..p
            },
            Op {
                m: b,
                exp: r.exp - 2 * k,
                ..r
            },
        );
        general::add(p, r, q, Round::TowardZero, core::mem::take(&mut v))
    });
    let Kind::Finite { exp, mut mant, .. } = kind else {
        unreachable!("a sum of squares near 1 that is not zero");
    };
    if o != Ordering::Equal {
        // The bit below the last place: `(2M + 1)·2^(lsb - 1)`.
        mant[0] |= 1u64 << (64 * mant.len() as u64 - q - 1);
    }
    let v = take_mant(out);
    let r = limbs::with_work(|w| {
        let m = Op {
            negative: false,
            m: &mant,
            exp: i128::from(exp) + 2 * k,
        };
        general::sqrt(m, prec, round, w, v)
    });
    s.limbs = mant;
    out.kind = r.0;
    r.1
}

/// The limbs of the accumulator of [`short_dot`]: 768 bits, its sign bit included.
const SHORT_ACC: usize = 12;

/// Sets `out` to the exact sum of `len` terms `term(0)`, `term(1)`, … rounded once at its
/// precision (at most 128 bits) when every factor is finite and of at most two limbs (to 128
/// bits), and the bits of the terms span at most the 768 of an accumulator on the stack: each
/// product of 256 bits (128 for factors of one limb) added at its place in two's complement, then
/// the top 256 bits of the sum and a sticky bit for the rest rounded in registers (the round bit
/// lies in them). The zeros are [`exact_sum`]'s. `None` when the terms are not so.
pub(crate) fn short_dot<'a>(
    out: &mut BigFloat,
    len: usize,
    term: impl Fn(usize) -> Term<'a>,
    round: Round,
) -> Option<Ordering> {
    let r = short_dot_value(len, term, out.prec, round)?;
    Some(put_short(out, r, round))
}

/// A factor of [`short_dot`]: `None` for a zero, `Err` for one that is not finite or is longer.
#[inline(always)]
fn short_factor(x: &BigFloat) -> Result<Option<(u128, bool, i128)>, ()> {
    match &x.kind {
        Kind::Finite { exp, mant, .. } => match *mant.as_slice() {
            [m] => Ok(Some((u128::from(m), false, i128::from(*exp) - 64))),
            [lo, hi] => Ok(Some((
                u128::from(hi) << 64 | u128::from(lo),
                true,
                i128::from(*exp) - 128,
            ))),
            _ => Err(()),
        },
        Kind::Zero { .. } => Ok(None),
        _ => Err(()),
    }
}

/// A term of [`short_dot`] exactly: its sign, its four limbs and the exponent of their lowest
/// bit; `None` for a zero.
#[inline(always)]
fn short_term(t: &Term<'_>) -> Result<Option<(bool, [u64; 4], i128)>, ()> {
    let negative =
        (t.x.is_sign_negative() != t.y.is_some_and(BigFloat::is_sign_negative)) != t.negate;
    let x = short_factor(t.x)?;
    let y = match t.y {
        None => return Ok(x.map(|(m, _, l)| (negative, [m as u64, (m >> 64) as u64, 0, 0], l))),
        Some(y) => short_factor(y)?,
    };
    Ok(match (x, y) {
        (Some((a, wa, la)), Some((b, wb, lb))) => {
            let (h, l) = if wa || wb { wide_mul(a, b) } else { (0, a * b) };
            Some((
                negative,
                [l as u64, (l >> 64) as u64, h as u64, (h >> 64) as u64],
                la + lb,
            ))
        }
        _ => None,
    })
}

/// The rounding of [`short_dot`] to `prec` bits before it is written: `Err` of the sign of an
/// exact zero.
pub(crate) fn short_dot_value<'a>(
    len: usize,
    term: impl Fn(usize) -> Term<'a>,
    prec: u64,
    round: Round,
) -> Option<Result<Rounded, bool>> {
    if prec > 128 {
        return None;
    }
    let mut zeros: Option<Zeros> = None;
    let (mut top, mut low, mut count) = (i128::MIN, i128::MAX, 0usize);
    for i in 0..len {
        let t = term(i);
        match short_term(&t).ok()? {
            Some((_, _, lsb)) => {
                top = top.max(lsb + 256);
                low = low.min(lsb);
                count += 1;
            }
            None => {
                let negative = (t.x.is_sign_negative()
                    != t.y.is_some_and(BigFloat::is_sign_negative))
                    != t.negate;
                let z = if negative {
                    Zeros::Negative
                } else {
                    Zeros::Positive
                };
                zeros = Some(match zeros {
                    Some(w) if w != z => Zeros::Mixed,
                    _ => z,
                });
            }
        }
    }
    if count == 0 {
        return Some(Err(zeros.unwrap_or(Zeros::Positive).negative(round)));
    }
    let bits = top - low + log2_ceil(count) + 2;
    let n = (bits as usize).div_ceil(64) + 1;
    if n > SHORT_ACC {
        return None;
    }
    let mut acc = [0u64; SHORT_ACC];
    for i in 0..len {
        let Ok(Some((negative, v, lsb))) = short_term(&term(i)) else {
            continue;
        };
        let off = (lsb - low) as u64;
        let (q, r) = ((off / 64) as usize, (off % 64) as u32);
        let k = if v[2] | v[3] == 0 { 2 } else { 4 };
        add_shifted(&mut acc[q..n], &v[..k], r, negative);
    }
    let a = &mut acc[..n];
    let negative = a[n - 1] >> 63 == 1;
    if negative {
        let mut carry = true;
        for x in a.iter_mut() {
            let (s, c) = (!*x).overflowing_add(u64::from(carry));
            *x = s;
            carry = c;
        }
    }
    let mut len = n;
    while len > 0 && a[len - 1] == 0 {
        len -= 1;
    }
    if len == 0 {
        return Some(Err(Zeros::Mixed.negative(round)));
    }
    let m = &a[..len];
    let bits = limbs::bit_len(m) as i64;
    let (w, sticky) = window4(m, bits - 256);
    let (hi, lo) = (
        u128::from(w[3]) << 64 | u128::from(w[2]),
        u128::from(w[1]) << 64 | u128::from(w[0]),
    );
    let b = i128::from(bits);
    Some(Ok(if prec <= 64 {
        round_top(negative, hi, low + b, sticky || lo != 0, prec, round)
    } else {
        round_top2(negative, hi, lo, low + b, sticky, prec, round)
    }))
}

/// A finite number of at most 128 bits in registers, for the factorisations of matrices: a zero
/// of the sign when `m` is zero, otherwise `±0.m·2^exp` with the top bit of `m` set. The results
/// of its operations are those of [`BigFloat`]'s, correctly rounded to nearest at a precision to
/// 128 bits, as long as they stay in the exponent range (`None` otherwise).
#[derive(Clone, Copy)]
pub(crate) struct Reg {
    negative: bool,
    m: u128,
    exp: i128,
}

impl Reg {
    /// A finite number of at most two limbs, `None` for the others.
    #[inline]
    pub(crate) fn of(x: &BigFloat) -> Option<Reg> {
        match &x.kind {
            Kind::Finite {
                negative,
                exp,
                mant,
            } => Some(Reg {
                negative: *negative,
                m: wide(mant)?,
                exp: i128::from(*exp),
            }),
            Kind::Zero { negative } => Some(Reg {
                negative: *negative,
                m: 0,
                exp: 0,
            }),
            _ => None,
        }
    }

    /// The number as a [`BigFloat`] of the precision `prec`, which holds it exactly.
    #[inline]
    pub(crate) fn to_float(self, prec: u64) -> BigFloat {
        let kind = if self.m == 0 {
            Kind::Zero {
                negative: self.negative,
            }
        } else {
            let mant = if prec <= 64 {
                vec![(self.m >> 64) as u64]
            } else {
                vec![self.m as u64, (self.m >> 64) as u64]
            };
            Kind::Finite {
                negative: self.negative,
                exp: self.exp as i64,
                mant,
            }
        };
        BigFloat { prec, kind }
    }

    /// Whether the number is a zero.
    #[inline]
    pub(crate) fn is_zero(self) -> bool {
        self.m == 0
    }

    /// Whether `|self| > |other|`.
    #[inline]
    pub(crate) fn larger(self, other: Reg) -> bool {
        match (self.m == 0, other.m == 0) {
            (true, _) => false,
            (false, true) => true,
            _ => (self.exp, self.m) > (other.exp, other.m),
        }
    }

    /// The number with its sign turned over.
    #[inline]
    pub(crate) fn neg(self) -> Reg {
        Reg {
            negative: !self.negative,
            ..self
        }
    }

    /// A result rounded in registers, or `None` beyond the exponent range.
    #[inline]
    fn rounded(r: Rounded) -> Option<Reg> {
        (i128::from(BigFloat::EXP_MIN)..=i128::from(BigFloat::EXP_MAX))
            .contains(&r.exp)
            .then_some(Reg {
                negative: r.negative,
                m: r.m,
                exp: r.exp,
            })
    }

    /// `self / d` rounded to nearest at `prec ≤ 128` bits, for `d` not zero.
    #[inline]
    pub(crate) fn div(self, d: Reg, prec: u64) -> Option<Reg> {
        debug_assert!(d.m != 0 && prec <= 128);
        let negative = self.negative != d.negative;
        if self.m == 0 {
            return Some(Reg {
                negative,
                m: 0,
                exp: 0,
            });
        }
        let n = Round::Nearest;
        let short = self.m as u64 == 0 && d.m as u64 == 0 && prec <= 64;
        Reg::rounded(if short {
            div1(
                negative,
                (self.m >> 64) as u64,
                self.exp,
                (d.m >> 64) as u64,
                d.exp,
                prec,
                n,
            )
        } else {
            div2(negative, self.m, self.exp, d.m, d.exp, prec, n)
        })
    }

    /// `self · y` rounded to nearest at `prec ≤ 128` bits.
    #[inline]
    pub(crate) fn mul(self, y: Reg, prec: u64) -> Option<Reg> {
        let negative = self.negative != y.negative;
        if self.m == 0 || y.m == 0 {
            return Some(Reg {
                negative,
                m: 0,
                exp: 0,
            });
        }
        Reg::rounded(mul2(
            negative,
            self.m,
            self.exp,
            y.m,
            y.exp,
            prec,
            Round::Nearest,
        ))
    }

    /// The exact value as four limbs and the exponent of their lowest bit: the number, or the
    /// product with `y`.
    #[inline(always)]
    fn limbs(self, y: Option<Reg>) -> ([u64; 4], i128) {
        match y {
            None => ([self.m as u64, (self.m >> 64) as u64, 0, 0], self.exp - 128),
            Some(y) => {
                let (h, l) = if self.m as u64 == 0 && y.m as u64 == 0 {
                    (0, (self.m >> 64) * (y.m >> 64))
                } else {
                    let (h, l) = wide_mul(self.m, y.m);
                    (h, l)
                };
                let lsb = self.exp + y.exp - 256 + if h == 0 { 128 } else { 0 };
                let v = if h == 0 {
                    [l as u64, (l >> 64) as u64, 0, 0]
                } else {
                    [l as u64, (l >> 64) as u64, h as u64, (h >> 64) as u64]
                };
                (v, lsb)
            }
        }
    }
}

/// The exact sum of `len` terms `±x` or `±x·y` (`term(i)` gives `x`, `y` and whether to take
/// `-`), rounded once to nearest at `prec ≤ 128` bits, as [`short_dot`] has it: `None` when the
/// terms span more than its accumulator or the result leaves the exponent range.
pub(crate) fn reg_dot(
    len: usize,
    term: impl Fn(usize) -> (Reg, Option<Reg>, bool),
    prec: u64,
) -> Option<Reg> {
    let round = Round::Nearest;
    let zero_term = |x: Reg, y: Option<Reg>| x.m == 0 || y.is_some_and(|y| y.m == 0);
    let sign = |x: Reg, y: Option<Reg>, negate: bool| {
        (x.negative != y.is_some_and(|y| y.negative)) != negate
    };
    let mut zeros: Option<Zeros> = None;
    let (mut top, mut low, mut count) = (i128::MIN, i128::MAX, 0usize);
    for i in 0..len {
        let (x, y, negate) = term(i);
        if zero_term(x, y) {
            let z = if sign(x, y, negate) {
                Zeros::Negative
            } else {
                Zeros::Positive
            };
            zeros = Some(match zeros {
                Some(w) if w != z => Zeros::Mixed,
                _ => z,
            });
            continue;
        }
        let e = x.exp + y.map_or(0, |y| y.exp);
        top = top.max(e);
        low = low.min(e - if y.is_some() { 256 } else { 128 });
        count += 1;
    }
    let zero = |negative: bool| {
        Some(Reg {
            negative,
            m: 0,
            exp: 0,
        })
    };
    if count == 0 {
        return zero(zeros.unwrap_or(Zeros::Positive).negative(round));
    }
    let bits = top - low + log2_ceil(count) + 2;
    let n = (bits as usize).div_ceil(64) + 1;
    if n > SHORT_ACC {
        return None;
    }
    let mut acc = [0u64; SHORT_ACC];
    for i in 0..len {
        let (x, y, negate) = term(i);
        if zero_term(x, y) {
            continue;
        }
        let (v, lsb) = x.limbs(y);
        let off = (lsb - low) as u64;
        let (q, r) = ((off / 64) as usize, (off % 64) as u32);
        let k = if v[2] | v[3] == 0 { 2 } else { 4 };
        add_shifted(&mut acc[q..n], &v[..k], r, sign(x, y, negate));
    }
    let a = &mut acc[..n];
    let negative = a[n - 1] >> 63 == 1;
    if negative {
        let mut carry = true;
        for x in a.iter_mut() {
            let (s, c) = (!*x).overflowing_add(u64::from(carry));
            *x = s;
            carry = c;
        }
    }
    let mut len = n;
    while len > 0 && a[len - 1] == 0 {
        len -= 1;
    }
    if len == 0 {
        return zero(Zeros::Mixed.negative(round));
    }
    let m = &a[..len];
    let bits = limbs::bit_len(m) as i64;
    let (w, sticky) = window4(m, bits - 256);
    let (hi, lo) = (
        u128::from(w[3]) << 64 | u128::from(w[2]),
        u128::from(w[1]) << 64 | u128::from(w[0]),
    );
    let b = i128::from(bits);
    Reg::rounded(if prec <= 64 {
        round_top(negative, hi, low + b, sticky || lo != 0, prec, round)
    } else {
        round_top2(negative, hi, lo, low + b, sticky, prec, round)
    })
}
