//! Square roots of one and two limbs in registers, after MPFR's `mpfr_sqrt1`, `mpfr_sqrt1n` and
//! `mpfr_sqrt2`: a root that is known within a unit or two decides the rounding unless its bits
//! below the round bit are within those units of a boundary, and only then is it made exact, with
//! its remainder.
//!
//! The radicand is the mantissa `0.x·2^e` made of an even exponent, `x·2^64` or `x·2^63` (two
//! limbs: `x·2^128` or `x·2^127`, or the top four limbs of a longer one, the others a tail), its
//! top limb at least `2^62`, so that its root has its top bit set. The root of the top limb comes
//! from the square root of an `f64` and one step of Newton's method on the integer (see
//! [`approx`]); a root of two limbs takes one step of Zimmermann's Karatsuba square root on it,
//! the division by the reciprocal of the root's top limb. A root is halfway between two numbers of
//! the precision only when the remainder is zero, which the exact paths see.

use super::Round;
use super::round::Rounded;
use super::small::{round_top, round_top2};
use crate::math::numbers::nat;

/// `2^32` as an `f64`.
const TWO_32: f64 = 4_294_967_296.0;

/// A root within one of `F = floor(sqrt(r))` for the 128-bit `r` whose top limb `t` is at least
/// `2^62`: `s` with `F ∈ {s - 1, s, s + 1}` (as a `u64`, `s = F + 1` wraps to zero for `F =
/// 2^64 - 1`).
///
/// `σ = sqrt(f64(t))·2^32` is within `1.51·2^-53` of `sqrt(r)` relatively (the conversion, the
/// limb below `t`, which is below `2^-62` of it, and the square root), so `s0 = trunc(σ)` is
/// within 3,101 of it, and the remainder `e = r - s0^2`, computed exactly, is below `2^77` in
/// magnitude. Newton's step `e / 2s0` differs from `sqrt(r) - s0` by `(sqrt(r) - s0)^2 / 2s0 <
/// 2^-40`, and it is computed as `f64(e >> 16)·(2^-17 / σ·2^-32)`: the 16 bits dropped, the
/// conversions, the division, the product and `σ` against `s0` (one unit of `2^63`) take it at
/// most `4·2^-53` of its size away, below `2^-38`. Its floor is then that of `sqrt(r) - s0` or one
/// away from it.
///
/// Also `2^-17 / σ·2^-32`, for [`reciprocal`].
#[inline(always)]
fn approx(r: u128) -> (u64, f64) {
    let t = (r >> 64) as u64;
    let sx = (t as f64).sqrt();
    // The conversion saturates: a root rounded up to `2^64` is `u64::MAX`.
    let s0 = (sx * TWO_32) as u64;
    let e = r.wrapping_sub(u128::from(s0) * u128::from(s0)) as i128;
    let inv = (1.0 / 131_072.0) / sx;
    let step = ((e >> 16) as i64 as f64 * inv).floor() as i64;
    (s0.wrapping_add(step as u64), inv)
}

/// `2^112` as an `f64`.
const TWO_112: f64 = 5_192_296_858_534_827_628_530_496_329_220_096.0;

/// `v ≈ T = 2^127 / s1` for the root `s1 = floor(sqrt(h))` of [`sqrt2`], from `inv` of [`approx`]
/// for `h`: `v ∈ [T - 1 - 2^-38, T + 2^-38]`.
///
/// `v0 = trunc(inv·2^112) = trunc(2^127 / σ)` is within `2.6·2^-53` of `T` relatively, `σ` within
/// `1.51·2^-53` of `sqrt(h)` and `s1` within one unit of `2^63` of it (saturated at `2^64 - 1`,
/// which keeps the bound, as `T ≤ 2^64`); with `s1·v0 = 2^127(1 + δ)`, `|δ| < 2^-51.6`, Newton's
/// step `v0 + v0(2^127 - s1·v0)/2^127 = T(1 - δ^2)` is within `2^-38` of `T`, and it is computed
/// with the remainder's 14 low bits dropped (below `2^-49`) and the product's floor.
#[inline(always)]
fn reciprocal(s1: u64, inv: f64) -> u64 {
    let v0 = (inv * TWO_112) as u64;
    // `|2^127 - s1·v0| < 2^75.4`, so its top bits fit an `i64` after the shift.
    let e = (1u128 << 127).wrapping_sub(u128::from(s1) * u128::from(v0)) as i128;
    let step = (i128::from(v0) * i128::from((e >> 14) as i64)) >> 113;
    (i128::from(v0) + step).clamp(0, i128::from(u64::MAX)) as u64
}

/// `F = floor(sqrt(r))` and the remainder `r - F^2`, from `s` within one of `F` (see [`approx`]):
/// one step up or down, chosen without a branch, and the exact root of the crate's integers when
/// `s` is not that close (which the bounds of [`approx`] exclude; the check keeps it safe).
#[inline(always)]
fn exact(s: u64, r: u128) -> (u64, u128) {
    let e = r.wrapping_sub(u128::from(s) * u128::from(s)) as i128;
    let two = 2 * i128::from(s);
    let (below, above) = (e < 0, e > two);
    // `(s - 1)^2 = s^2 - 2s + 1` and `(s + 1)^2 = s^2 + 2s + 1`.
    let s = s
        .wrapping_sub(u64::from(below))
        .wrapping_add(u64::from(above));
    let e = e + if below { two - 1 } else { 0 } - if above { two + 1 } else { 0 };
    if e < 0 || e > 2 * i128::from(s) {
        return far(r);
    }
    (s, e as u128)
}

/// The exact root and remainder, out of line, for an estimate that is not within one.
#[cold]
#[inline(never)]
fn far(r: u128) -> (u64, u128) {
    nat::sqrt_rem_u128(r)
}

/// Whether the root `s` (of `bits` bits, its top bit set but for the root `F` just below a power
/// of two) known within `lo` below and `hi` above of `F`, rounds to `prec` bits as `s` does with a
/// sticky bit: its `k = bits - 1 - prec` bits below the round bit are at least `lo + 1` and at
/// most `2^k - 1 - hi`, so that every candidate has the same bits from the round bit up and some
/// bit set below it.
#[inline(always)]
fn decided(low: u128, k: u64, lo: u128, hi: u128) -> bool {
    k >= 1 && low > lo && low < (1u128 << k) - hi
}

/// The square root of a mantissa of one limb, rounded to `prec ≤ 64` bits: the root `F` of 64 bits
/// of `x·2^64` or `x·2^63`, whichever makes the exponent even, and its remainder `ρ` when the bits
/// of the estimate below the round bit do not decide. For 64 bits, the round bit lies below `F`:
/// `sqrt(r) ≥ F + 1/2` exactly when `ρ ≥ F + 1/4`, that is `ρ > F`, which takes the place of the
/// round bit (no root is halfway, as the remainder of one would be `F + 1/4`).
#[inline(always)]
pub(crate) fn sqrt1(x: u64, ex: i128, prec: u64, round: Round) -> Rounded {
    let odd = ex & 1 != 0;
    let r = if odd {
        u128::from(x) << 63
    } else {
        u128::from(x) << 64
    };
    root1(r, false, (ex + i128::from(odd)) / 2, prec, round)
}

/// The square root of a mantissa of two limbs or more, `top` and `next` the top two and `rest`
/// whether any of the others is not zero, rounded to `prec ≤ 63` bits: the root of the top 128
/// bits of the radicand, the others a tail, as the floor of the root of a truncated radicand is
/// that of the exact one.
#[inline(always)]
pub(crate) fn sqrt_long(
    top: u64,
    next: u64,
    rest: bool,
    ex: i128,
    prec: u64,
    round: Round,
) -> Rounded {
    debug_assert!(prec <= 63);
    let r = u128::from(top) << 64 | u128::from(next);
    if ex & 1 != 0 {
        root1(r >> 1, rest || next & 1 == 1, (ex + 1) / 2, prec, round)
    } else {
        root1(r, rest, ex / 2, prec, round)
    }
}

/// The root of the 128-bit radicand `r` (its top limb at least `2^62`) with `tail` for bits
/// below it, as `0.F·2^exp` rounded to `prec ≤ 64` bits; no tail for 64 bits.
#[inline(always)]
pub(crate) fn root1(r: u128, tail: bool, exp: i128, prec: u64, round: Round) -> Rounded {
    let (s, _) = approx(r);
    if prec <= 61 {
        let k = 63 - prec;
        if decided(u128::from(s) & ((1 << k) - 1), k, 1, 1) {
            return round_top(false, u128::from(s) << 64, exp, true, prec, round);
        }
    }
    let (s, rem) = exact(s, r);
    let sticky = rem != 0 || tail;
    if prec == 64 {
        debug_assert!(!tail);
        let half = u128::from(rem > u128::from(s)) << 63;
        return round_top(false, u128::from(s) << 64 | half, exp, sticky, prec, round);
    }
    round_top(false, u128::from(s) << 64, exp, sticky, prec, round)
}

/// The square root of a mantissa `m` of any length, rounded to `prec ≤ 128` bits (127 when `m`
/// has more than two limbs): the root `S` of 128 bits of the top 256 bits `R` of the radicand
/// (`x·2^128` or `x·2^127` for one or two limbs), the others a tail.
///
/// With `h` the top two limbs of `R`, `a1` and `a0` the others, `s1 = floor(sqrt(h))` and its
/// remainder `r1 ≤ 2s1` (exact, from [`approx`]), and `q`, `u` the quotient and remainder of
/// `r1·2^64 + a1` by `2s1` (of `(r1·2^64 + a1)/2` by `s1`, with the reciprocal of `s1`), `S =
/// s1·2^64 + q` is the root or one more, as the Karatsuba square root has it: `R - S^2 = u·2^64 +
/// a0 - q^2`, below zero exactly when `S` is one too large. The quotient is first estimated by a
/// product with the reciprocal of `s1`, which decides the rounding but within a few units of a
/// boundary; otherwise it is computed by the division and the remainder with it. For 128 bits,
/// the round bit lies below `S`: the remainder above `S` takes its place, as in [`sqrt1`].
#[inline(always)]
pub(crate) fn sqrt2(m: &[u64], ex: i128, prec: u64, round: Round) -> Rounded {
    let odd = ex & 1 != 0;
    let exp = (ex + i128::from(odd)) / 2;
    let (h, a1, a0, tail) = match *m {
        [x0] => {
            let x = u128::from(x0) << 64;
            (if odd { x >> 1 } else { x }, 0, 0, false)
        }
        [x0, x1] => {
            let x = u128::from(x1) << 64 | u128::from(x0);
            if odd {
                (x >> 1, x0 << 63, 0, false)
            } else {
                (x, 0, 0, false)
            }
        }
        _ => {
            let mut d = [0u64; 4];
            let tail = super::limbs::top_limbs(m, u32::from(odd), &mut d);
            (u128::from(d[3]) << 64 | u128::from(d[2]), d[1], d[0], tail)
        }
    };
    root2(h, a1, a0, tail, exp, prec, round)
}

/// The root of the 256-bit radicand `h·2^128 + a1·2^64 + a0` (`h ≥ 2^126`) with `tail` for bits
/// below it, as `0.S·2^exp` rounded to `prec ≤ 128` bits, as [`sqrt2`] describes; no tail for 128
/// bits.
#[inline(always)]
pub(crate) fn root2(
    h: u128,
    a1: u64,
    a0: u64,
    tail: bool,
    exp: i128,
    prec: u64,
    round: Round,
) -> Rounded {
    debug_assert!(prec <= 127 || !tail);
    let (s, inv) = approx(h);
    let (s1, r1) = exact(s, h);
    // `(r1·2^64 + a1) / 2`, of two limbs as `r1 ≤ 2s1 < 2^65`, below `s1·2^64` but for `r1 ≥ 2s1 -
    // 1`, where the quotient may reach `2^64`.
    let (n1, n0) = ((r1 >> 1) as u64, (r1 as u64) << 63 | a1 >> 1);
    if n1 >= s1 {
        return wide(h, a1, a0, tail, exp, prec, round);
    }
    if prec <= 125 {
        // The quotient `qt` estimated with `v` of [`reciprocal`]: `(n1·2^64 + n0)·v / 2^127`, the
        // low product's low limb and the bits below the result dropped, is within `(n1·2^64 +
        // n0)·(1 + 2^-37)/2^127 < 2.001` below the exact quotient and `2^-38` above it, so that the
        // estimate is `qt - 3` to `qt + 1`, and the root `F` lies in `[S - 2, S + 3]`. (`S` wraps
        // to zero only for `F = 2^128 - 1`, and is then not decided.)
        let v = reciprocal(s1, inv);
        let p = u128::from(n1) * u128::from(v) + ((u128::from(n0) * u128::from(v)) >> 64);
        let s = (u128::from(s1) << 64).wrapping_add(p >> 63);
        let k = 127 - prec;
        if decided(s & ((1 << k) - 1), k, 2, 3) {
            return round_top2(false, s, 0, exp, true, prec, round);
        }
    }
    let (q, u) = nat::div_2by1(n1, n0, s1, nat::invert_limb(s1));
    let s = u128::from(s1) << 64 | u128::from(q);
    // The remainder by `2s1` is `2u` and the bit of `a1` that the halving dropped, below `2^65`;
    // `R - S^2 = u'·2^64 + a0 - q^2` is `(hi, lo)`, `hi` the top limb in two's complement.
    let u2 = u128::from(u) << 1 | u128::from(a1 & 1);
    let (lo, borrow) = (u2 << 64 | u128::from(a0)).overflowing_sub(u128::from(q) * u128::from(q));
    let hi = ((u2 >> 64) as u64).wrapping_sub(u64::from(borrow));
    // One too large: `S - 1` and the remainder `2(S - 1) + 1` more, without a branch.
    let over = u128::from((hi as i64) < 0).wrapping_neg();
    let s = s.wrapping_sub(over & 1);
    let (lo, carry) = lo.overflowing_add((s << 1 | 1) & over);
    let hi = hi
        .wrapping_add((s >> 127) as u64 & over as u64)
        .wrapping_add(u64::from(carry));
    finish(s, hi, lo, tail, exp, prec, round)
}

/// The root `S` with the remainder `hi·2^128 + lo` (at most `2S`) rounded, for [`sqrt2`].
#[inline(always)]
fn finish(s: u128, hi: u64, lo: u128, tail: bool, exp: i128, prec: u64, round: Round) -> Rounded {
    let sticky = hi != 0 || lo != 0 || tail;
    let half = if prec == 128 {
        u128::from(hi != 0 || lo > s) << 127
    } else {
        0
    };
    round_top2(false, s, half, exp, sticky, prec, round)
}

/// [`sqrt2`] when the quotient of its step may reach `2^64`, by the root of four limbs of the
/// crate's integers, out of line.
#[cold]
#[inline(never)]
fn wide(h: u128, a1: u64, a0: u64, tail: bool, exp: i128, prec: u64, round: Round) -> Rounded {
    let (s, rem) = nat::sqrt_rem_wide([a0, a1, h as u64, (h >> 64) as u64]);
    let lo = u128::from(rem[1]) << 64 | u128::from(rem[0]);
    finish(s, rem[2], lo, tail, exp, prec, round)
}
