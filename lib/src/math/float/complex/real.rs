//! The real numbers the complex functions are built from, at a working precision `w`, each with a
//! bound of its error: [`Fx`]s, a magnitude `±M·2^lsb` whose exponent is an `i128` (outside the
//! exponent range of [`BigFloat`] as an intermediate may be) and whose error is at most
//! `2^err·2^lsb`, `err` a [`Lg`], the logarithm of a bound in units of the last place.
//!
//! The bounds are kept in units of the last place of each value, never as absolute exponents in
//! floating point, so that the `f64`s of the bounds stay small (a few thousand at most) and exact
//! enough for any exponent the values have. Every operation rounds its result toward zero to
//! about `w + 16` bits and adds the unit of the last place it drops, or the bound of what it
//! leaves out; a function built from them has a sound bound by construction, which
//! [`to_approx`] hands to Ziv's rounding ([`ziv2`]).
//!
//! The functions of the parts are [`exp`], [`expm1`], [`sin_cos`], [`sinh_cosh`], [`ln`],
//! [`ln1p`], [`atan2`], the constants [`pi`], [`ln2`] and [`ln10`], and series for the parts of the
//! functions of tiny arguments that differ from their first terms by less than the precision.

use super::super::atan::{atan_unit, plus_pi_part};
use super::super::ball::{Ball, Rad, fast_round};
use super::super::consts::{ln2_approx, ln10_approx, pi_approx};
use super::super::exp::{Range, exp_ball};
use super::super::limbs;
use super::super::log::ln_raw;
use super::super::round::round_mag;
use super::super::sincos::{self, Fx, Lg, lg_down, lg_up};
use super::super::trig;
use super::super::ziv::{Approx, guard, round_approx};
use super::super::{BigFloat, Round};
use core::cmp::Ordering;

/// A part of a result and the direction of its rounding.
pub(super) type Part = (BigFloat, Ordering);

/// What an attempt of [`ziv2`] gives for a part: an approximation to round, or the part itself.
pub(super) type Step = Result<Approx, Part>;

/// The slack each bound computed here takes on, as [`Lg`]'s own operations do.
const SLACK: f64 = 1.0 / 256.0;

// The values.

/// A finite float that is not zero, exactly.
pub(super) fn of(x: &BigFloat) -> Fx {
    let (negative, m, lsb) = super::raw::significant(x);
    let mut m = m.to_vec();
    limbs::trim(&mut m);
    Fx {
        negative,
        m,
        lsb,
        err: Lg::ZERO,
    }
}

/// The exact `±m·2^lsb`.
pub(super) fn exact(negative: bool, m: Vec<u64>, lsb: i128) -> Fx {
    let mut m = m;
    limbs::trim(&mut m);
    Fx {
        negative,
        m,
        lsb,
        err: Lg::ZERO,
    }
}

/// The exact `1`.
pub(super) fn one() -> Fx {
    exact(false, vec![1], 0)
}

/// The exact `0`.
pub(super) fn zero() -> Fx {
    exact(false, Vec::new(), 0)
}

/// `-a`.
pub(super) fn neg(mut a: Fx) -> Fx {
    a.negative = !a.negative;
    a
}

/// `|a|`.
pub(super) fn abs(mut a: Fx) -> Fx {
    a.negative = false;
    a
}

/// `a·2^k`, exactly: the bound in units of the last place is unchanged.
pub(super) fn shl(mut a: Fx, k: i128) -> Fx {
    a.lsb += k;
    a
}

/// The bound of `|a|`'s value, in units of `a`'s last place: of the magnitude and the error.
#[inline]
fn bound(a: &Fx) -> Lg {
    lg_up(&a.m).add(a.err)
}

/// The least `e` with `|v| < 2^e` for the value `v` that `a` stands for (a bound of its magnitude,
/// from above).
pub(super) fn top(a: &Fx) -> i128 {
    // An exact zero far below anything; a bound that is not a number or is too large to mean
    // anything clamped.
    let b = bound(a).0;
    if b == f64::NEG_INFINITY {
        return i128::MIN / 8;
    }
    a.lsb + b.min(1e15).ceil() as i128 + 1
}

/// Whether the bound of `a` means nothing: not a number, or too large for the exponents.
pub(super) fn bad(a: &Fx) -> bool {
    a.err.0.is_nan() || a.err.0 > 1e12
}

/// A value whose bound means nothing, which never decides.
pub(super) fn useless() -> Fx {
    Fx {
        negative: false,
        m: vec![1],
        lsb: 0,
        err: Lg(f64::INFINITY),
    }
}

/// Whether the sign of the value is known: its error is below its approximation.
pub(super) fn sign_known(a: &Fx) -> bool {
    !a.m.is_empty() && (a.err.0 == f64::NEG_INFINITY || a.err.0 < lg_down(&a.m) - 1.0)
}

/// The error bound moved to the units of `2^lsb` from those of `2^from`.
#[inline]
fn moved(err: Lg, from: i128, lsb: i128) -> Lg {
    if err.0 == f64::NEG_INFINITY {
        err
    } else {
        Lg(err.0 + (from - lsb) as f64 + SLACK)
    }
}

/// The absolute bound `2^e` in units of `2^lsb`.
#[inline]
fn pow2_in(e: i128, lsb: i128) -> Lg {
    Lg((e - lsb) as f64 + SLACK)
}

/// `a` cut toward zero at the bit of the weight `2^floor` (nothing when its last bit lies above).
fn cut_at(a: &Fx, floor: i128) -> Fx {
    if a.lsb >= floor || a.m.is_empty() {
        return a.clone();
    }
    let keep = (a.exponent() - floor).max(1) as u64;
    a.clone().cut(keep)
}

/// `a·b`, about `w + 16` bits: `|ãb̃ - ab| <= |ã|β + |b̃|α + αβ` for the errors `α`, `β`, in units
/// of the product's last place, and the unit dropped by the cut.
pub(super) fn mul(a: &Fx, b: &Fx, w: u64) -> Fx {
    let m = limbs::mul(&a.m, &b.m);
    let err = lg_up(&a.m)
        .mul(b.err)
        .add(lg_up(&b.m).mul(a.err))
        .add(a.err.mul(b.err));
    Fx {
        negative: a.negative != b.negative,
        m,
        lsb: a.lsb + b.lsb,
        err,
    }
    .cut(w + 16)
}

/// `a²`, as [`mul`].
pub(super) fn sqr(a: &Fx, w: u64) -> Fx {
    let m = limbs::mul(&a.m, &a.m);
    let err = lg_up(&a.m).mul(a.err).scale(1).add(a.err.mul(a.err));
    Fx {
        negative: false,
        m,
        lsb: 2 * a.lsb,
        err,
    }
    .cut(w + 16)
}

/// `a + b` to about `w` bits of the larger: a term whose value is below `2^(e - w - 16)`, `e` the
/// exponent of the larger, goes into the bound whole; otherwise both are cut at `2^(e - w - 24)`
/// (each cut adding its unit) and added exactly. A sum that cancels keeps the absolute error, so
/// that its relative error grows; Ziv's rounding then takes a larger `w`.
pub(super) fn add(a: &Fx, b: &Fx, w: u64) -> Fx {
    let (x, y) = if a.m.is_empty() || (!b.m.is_empty() && b.exponent() > a.exponent()) {
        (b, a)
    } else {
        (a, b)
    };
    if x.m.is_empty() {
        // Both approximations zero: the larger bound in the units of the lower last place.
        let lsb = x.lsb.min(y.lsb);
        return Fx {
            negative: false,
            m: Vec::new(),
            lsb,
            err: moved(x.err, x.lsb, lsb).add(moved(y.err, y.lsb, lsb)),
        };
    }
    let ex = x.exponent();
    let floor = ex - i128::from(w) - 24;
    if top(y) < floor + 8 {
        let mut r = cut_at(x, floor);
        r.err = r.err.add(moved(bound(y), y.lsb, r.lsb));
        return r;
    }
    let (xc, yc) = (cut_at(x, floor), cut_at(y, floor));
    let mut s = sincos::sum(&[&xc, &yc]);
    limbs::trim(&mut s.m);
    s
}

/// `a - b`, as [`add`].
pub(super) fn sub(a: &Fx, b: &Fx, w: u64) -> Fx {
    let mut nb = b.clone();
    nb.negative = !nb.negative;
    add(a, &nb, w)
}

/// `a/b` to about `w + 16` bits, `None` when `b`'s error is not below a quarter of it. With the
/// relative errors `ρa`, `ρb <= 1/4` of the approximations, `|a/b - ã/b̃| <= |ã/b̃|·(ρa + ρb)/(1 -
/// ρb) <= |ã/b̃|·(4/3)(ρa + ρb)`, and the floor of the quotient one unit more.
pub(super) fn div(a: &Fx, b: &Fx, w: u64) -> Option<Fx> {
    if b.m.is_empty() {
        return None;
    }
    let lb = lg_down(&b.m);
    if b.err.0 > lb - 2.0 {
        return None;
    }
    let negative = a.negative != b.negative;
    let (ba, bb) = (limbs::bit_len(&a.m), limbs::bit_len(&b.m));
    if a.m.is_empty() {
        // `|a/b| <= α/(|b̃|(1 - ρb))`, in units of `2^(a.lsb - b.lsb)` and the bits of `b`.
        let lsb = a.lsb - b.lsb - i128::from(bb);
        let err = Lg(a.err.0 - lb + i128::from(bb) as f64 + 0.42 + SLACK);
        return Some(Fx {
            negative,
            m: Vec::new(),
            lsb,
            err,
        });
    }
    let s = (w + 16 + 2 + bb).saturating_sub(ba);
    let q = sincos::div(&limbs::shl(&a.m, s), &b.m);
    let lsb = a.lsb - i128::from(s) - b.lsb;
    let rho = Lg(a.err.0 - lg_down(&a.m)).add(Lg(b.err.0 - lb));
    // `4/3 < 2^0.42`.
    let err = Lg(lg_up(&q).0 + rho.0 + 0.42 + SLACK).add(Lg(0.0));
    Some(Fx {
        negative,
        m: q,
        lsb,
        err,
    })
}

/// `a/k` for an integer `k >= 1`, to about `w + 16` bits: the floor one unit more.
pub(super) fn div_int(a: &Fx, k: u64, w: u64) -> Fx {
    let ba = limbs::bit_len(&a.m);
    let s = (w + 16 + 64).saturating_sub(ba);
    let m = sincos::div_small(&limbs::shl(&a.m, s), k);
    let err = Lg(a.err.0 + s as f64 - (k as f64).log2() + SLACK).add(Lg(0.0));
    Fx {
        negative: a.negative,
        m,
        lsb: a.lsb - i128::from(s),
        err,
    }
}

/// `a·k` for an integer `k`, exactly (the error times `k`).
pub(super) fn mul_int(a: &Fx, k: u64) -> Fx {
    let m = sincos::mul_small(&a.m, k);
    let mut m = m;
    limbs::trim(&mut m);
    Fx {
        negative: a.negative,
        m,
        lsb: a.lsb,
        err: Lg(a.err.0 + (k as f64).log2() + SLACK),
    }
}

/// `√a` to about `w + 16` bits, `None` when the value may be below zero (its error not below its
/// approximation): `|√a - √ã| = |a - ã|/(√a + √ã) <= α/√ã`, and the floor of the root one unit.
pub(super) fn sqrt(a: &Fx, w: u64) -> Option<Fx> {
    if a.negative || !sign_known(a) {
        return None;
    }
    // `a` with `2(w + 16)` bits or more and an even last exponent; `a` cut if it is longer.
    let want = 2 * (w + 16) + 2;
    let mut a = a.clone().cut(want + 2);
    let bits = limbs::bit_len(&a.m);
    let mut s = want.saturating_sub(bits);
    if (a.lsb - i128::from(s)).rem_euclid(2) != 0 {
        s += 1;
    }
    a.m = limbs::shl(&a.m, s);
    a.lsb -= i128::from(s);
    a.err = Lg(a.err.0 + s as f64);
    let r = sincos::isqrt(&a.m);
    let err = if a.err.0 == f64::NEG_INFINITY {
        Lg(0.0)
    } else {
        // `α/√ã` in units of `2^(lsb/2)`: `err·2^lsb / (√M·2^(lsb/2)) = err·2^(lsb/2)/√M`, and
        // `√M >= r`.
        Lg(a.err.0 - lg_down(&r) + SLACK).add(Lg(0.0))
    };
    Some(Fx {
        negative: false,
        m: r,
        lsb: a.lsb / 2,
        err,
    })
}

/// The [`Approx`] of a value for Ziv's rounding, or one that decides nothing when the bound is
/// not below the value.
pub(super) fn to_approx(f: &Fx) -> Approx {
    if f.m.is_empty() || bad(f) {
        return undecided();
    }
    let e = f.exponent();
    let err = if f.err.0 == f64::NEG_INFINITY {
        f.lsb - 2
    } else {
        (f.lsb + f.err.0.ceil() as i128).max(f.lsb - 2)
    };
    if err >= e - 1 {
        return undecided();
    }
    Approx {
        negative: f.negative,
        m: f.m.clone(),
        lsb: f.lsb,
        err,
    }
}

/// An approximation that decides nothing.
pub(super) fn undecided() -> Approx {
    Approx {
        negative: false,
        m: Vec::new(),
        lsb: 0,
        err: 0,
    }
}

/// The part `l + d` for an exact `l` (not zero) and a `d` of a known sign that is small beside it,
/// rounded to `prec` bits in the mode `round` when `|d|` lies below both the last place of `l` and
/// the units at which `l` has two bits more than the precision (MPFR's rounding of a result near
/// its argument, [`trig::near`]); otherwise the approximation of the sum.
pub(super) fn lead(l: &Fx, d: &Fx, w: u64, prec: u64, round: Round) -> Step {
    if d.m.is_empty() && d.err.0 == f64::NEG_INFINITY {
        unreachable!("an exact lead is found before Ziv's loop");
    }
    if sign_known(d) {
        let e = top(d) + 1;
        if let Some(r) = trig::near(
            l.negative,
            &l.m,
            l.lsb,
            d.negative == l.negative,
            e,
            prec,
            round,
        ) {
            return Err(r);
        }
    }
    Ok(to_approx(&add(l, d, w + 32)))
}

// The functions of the parts.

/// `e^x` to about `w` bits relatively, its exponent unbounded, or where it lies beyond the
/// exponent range (`|x| >= 2^62`, see [`exp_ball`]). A tiny `x`, `|x| < 2^-(w + 16)`, gives 1
/// within `e^|x| - 1 <= 2|x|`; otherwise `x` is a ball of a midpoint of `w + 32` bits (its rounding
/// and `x`'s error the radius) and [`exp_ball`] takes it.
pub(super) fn exp(x: &Fx, w: u64) -> Result<Fx, Range> {
    if bad(x) {
        return Ok(useless());
    }
    let t = top(x);
    if x.m.is_empty() && t > 0 {
        return Ok(useless());
    }
    if x.m.is_empty() || t < -i128::from(w) - 16 {
        // `|e^x - 1| <= 2|x|` for `|x| <= 1`.
        let mut r = one();
        r.err = pow2_in(t + 1, 0);
        return Ok(r);
    }
    if x.exponent() >= 63 {
        return Err(if x.negative {
            Range::Under
        } else {
            Range::Over
        });
    }
    let b = ball(x, w + 32);
    let raw = exp_ball(&b, w + 8)?;
    let mut m = raw.m;
    limbs::trim(&mut m);
    let bits = limbs::bit_len(&m) as i128;
    let err = if raw.rel.is_zero() {
        Lg::ZERO
    } else {
        Lg((raw.rel.exp2_ceil() + bits) as f64 + SLACK)
    };
    Ok(Fx {
        negative: false,
        m,
        lsb: raw.lsb,
        err,
    })
}

/// The ball of `x` (exponent in the range of [`BigFloat`]), the midpoint rounded to nearest at `p`
/// bits, the radius its error and the rounding's.
fn ball(x: &Fx, p: u64) -> Ball {
    let (mid, o) = round_mag(x.negative, &x.m, x.lsb, false, p, Round::Nearest);
    let mut rad = Rad::ZERO;
    if x.err.0 != f64::NEG_INFINITY {
        rad = Rad::pow2(x.lsb + x.err.0.ceil() as i128);
    }
    if o != Ordering::Equal {
        rad = rad.add(Rad::pow2(x.exponent() - i128::from(p)));
    }
    Ball::new(mid, rad)
}

/// `e^x - 1` to about `w` bits relatively, for `|x| < 2^62`: `e^x` less 1 at `w + 16 - e` bits for
/// `-w/3 <= e < 0` (`|x| < 2^e`; the cancellation loses `-e` bits), the series `Σ x^k/k!` with a
/// bound of its tail for smaller `x` (`|tail| <= 2|x|^(K + 1)/(K + 1)!` for `|x| <= 1/2`), and
/// `e^x - 1` as it is for `|x| >= 1/2`.
pub(super) fn expm1(x: &Fx, w: u64) -> Result<Fx, Range> {
    if x.m.is_empty() && x.err.0 == f64::NEG_INFINITY {
        return Ok(zero());
    }
    let e = top(x);
    if e >= 0 {
        let y = exp(x, w)?;
        return Ok(sub(&y, &one(), w + 8));
    }
    if -e <= i128::from(w / 3) + 2 {
        let wx = w + 16 + (-e) as u64;
        let y = exp(x, wx)?;
        return Ok(sub(&y, &one(), wx + 8));
    }
    // Terms `x^k/k!` while above `2^-(w + 24)` of `|x|`: `K - 1 >= (w + 24)/(-e)`.
    let k_max = (u128::from(w + 24) / (-e) as u128) as u64 + 2;
    let mut sum = x.clone();
    let mut term = x.clone();
    for k in 2..=k_max {
        term = div_int(&mul(&term, x, w + 8), k, w + 8);
        sum = add(&sum, &term, w + 8);
    }
    // The tail: `2|x|^(K + 1)/(K + 1)! <= 2^(1 + (K + 1)e)`.
    let tail = 1 + (k_max as i128 + 1) * e;
    sum.err = sum.err.add(pow2_in(tail, sum.lsb));
    Ok(sum)
}

/// `sin x` and `cos x` to about `w` bits relatively each, by the reduction and the kernels of
/// [`trig::eval`]; `x`'s error added to both (`|sin'|, |cos'| <= 1`).
pub(super) fn sin_cos(x: &Fx, w: u64) -> (Fx, Fx) {
    if bad(x) {
        return (useless(), useless());
    }
    let t = top(x);
    if !x.m.is_empty() && t < -i128::from(w / 2) - 16 {
        // `|sin x - x| <= |x|³/6 < 2^(3t - 2)` and `0 <= 1 - cos x <= x²/2 < 2^(2t - 1)`, with
        // `x`'s error.
        let mut s = x.clone();
        s.err = s.err.add(pow2_in(3 * t - 2, s.lsb));
        let mut c = one();
        c.err = pow2_in(2 * t - 1, 0).add(moved(x.err, x.lsb, 0));
        return (s, c);
    }
    if x.m.is_empty() {
        // `|sin x| <= α`, `|1 - cos x| <= α²/2 <= α` for `α <= 1`.
        let mut s = zero();
        s.lsb = x.lsb;
        s.err = x.err;
        let mut c = one();
        c.err = moved(x.err, x.lsb, 0);
        return (s, c);
    }
    let e = trig::eval(&x.m, x.lsb, w + 8, |_| (true, true));
    let s = e.sin.expect("asked");
    let c = e.cos.expect("asked");
    let (mut sin, mut cos) = match e.q {
        0 => (s, c),
        1 => (c, neg(s)),
        2 => (neg(s), neg(c)),
        _ => (neg(c), s),
    };
    if x.negative {
        sin = neg(sin);
    }
    for f in [&mut sin, &mut cos] {
        limbs::trim(&mut f.m);
        f.err = f.err.add(moved(x.err, x.lsb, f.lsb));
    }
    (sin, cos)
}

/// `sinh x` and `cosh x` to about `w` bits relatively each, or where they lie beyond the exponent
/// range: with `E = e^|x|`, `cosh = (E + 1/E)/2`, and `sinh = (E - 1/E)/2` for `|x| >= 1/2`, `(u +
/// u/(u + 1))/2` with `u = e^|x| - 1` ([`expm1`]) below, which cancels nothing; the sign of `x`'s.
pub(super) fn sinh_cosh(x: &Fx, w: u64) -> Result<(Fx, Fx), Range> {
    let ax = abs(x.clone());
    let w2 = w + 8;
    let (s, c) = if top(&ax) >= 0 {
        let big = exp(&ax, w2).map_err(|_| Range::Over)?;
        let inv = div(&one(), &big, w2).expect("e^|x| >= 1");
        let s = shl(sub(&big, &inv, w2), -1);
        let c = shl(add(&big, &inv, w2), -1);
        (s, c)
    } else {
        let u = expm1(&ax, w2).map_err(|_| Range::Over)?;
        let u1 = add(&u, &one(), w2);
        let q = div(&u, &u1, w2).expect("u + 1 >= 1");
        let s = shl(add(&u, &q, w2), -1);
        let inv = div(&one(), &u1, w2).expect("u + 1 >= 1");
        let c = shl(add(&u1, &inv, w2), -1);
        (s, c)
    };
    Ok((if x.negative { neg(s) } else { s }, c))
}

/// `ln x` for `x > 0` to about `w` bits absolutely (relatively when `|ln x| >= 1/4`, and near 1 as
/// precisely as `x` itself is known: [`ln_raw`] takes the bits of `x - 1` it needs), `None` when
/// the sign of `x` is not known. `x = 2^k·x'` with `x'` in `[1/2, 1)` when its exponent leaves the
/// range of [`BigFloat`], and `ln x = ln x' + k·ln 2`.
pub(super) fn ln(x: &Fx, w: u64) -> Option<Fx> {
    if bad(x) || x.negative || !sign_known(x) {
        return None;
    }
    let ex = x.exponent();
    let k = if ex.abs() < 1 << 60 { 0 } else { ex };
    let xs = shl(x.clone(), -k);
    let b = ball(&xs, w + 32);
    let (negative, m, lsb, rad) = ln_raw(&b, w + 8)?;
    let mut r = Fx {
        negative,
        m,
        lsb,
        err: if rad.is_zero() {
            Lg::ZERO
        } else {
            pow2_in(rad.exp2_ceil(), lsb)
        },
    };
    limbs::trim(&mut r.m);
    if k != 0 {
        let kb = 128 - k.unsigned_abs().leading_zeros() as u64;
        let l2 = mul(&ln2(w + kb + 16), &int(k), w + kb + 16);
        r = add(&r, &l2, w + 8);
    }
    Some(r)
}

/// The exact integer `k`.
pub(super) fn int(k: i128) -> Fx {
    let u = k.unsigned_abs();
    exact(k < 0, vec![u as u64, (u >> 64) as u64], 0)
}

/// `ln(1 + d)` for `d > -1` to about `w` bits relatively, `None` when that is not known to hold:
/// for `|d| >= 1/4` the logarithm of `1 + d`; for `2^-(w/3 + 2) <= |d| < 2^e`, `e < 0`, of `1 + d`
/// formed at `w - e + 32` bits; below, the series `Σ (-1)^(k + 1) d^k/k` with its tail, at most
/// `2|d|^(K + 1)` for `|d| <= 1/2`.
pub(super) fn ln1p(d: &Fx, w: u64) -> Option<Fx> {
    let e = top(d);
    if bad(d) || (d.m.is_empty() && e > -1) {
        return None;
    }
    if d.m.is_empty() {
        // `|ln(1 + d)| <= 2|d|` for `|d| <= 1/2`.
        let mut r = d.clone();
        r.err = r.err.scale(1);
        return Some(r);
    }
    if e >= -1 {
        let x = add(&one(), d, w + 16);
        return ln(&x, w);
    }
    if -e <= i128::from(w / 3) + 2 {
        // `ln` rounds its argument to `wx + 40` bits: within `2^-(w + 72 - e)` of it, below
        // `2^-(w + 64)` of the result, near `d`.
        let wx = w + 32 + (-e) as u64;
        let x = add(&one(), d, wx);
        return ln(&x, wx);
    }
    let k_max = (u128::from(w + 24) / (-e) as u128) as u64 + 2;
    let mut sum = d.clone();
    let mut power = d.clone();
    for k in 2..=k_max {
        power = mul(&power, d, w + 8);
        let mut t = div_int(&power, k, w + 8);
        if k % 2 == 0 {
            t = neg(t);
        }
        sum = add(&sum, &t, w + 8);
    }
    let tail = 1 + (k_max as i128 + 1) * e;
    sum.err = sum.err.add(pow2_in(tail, sum.lsb));
    Some(sum)
}

/// The angle of the point `(x, y)`, in `(-π, π]`, to about `w` bits relatively, for `x` and `y` of
/// known signs (`None` otherwise): `atan(z)` of the ratio `z` of the smaller magnitude to the larger,
/// at most 1, and `π/2 - atan z`, `π - atan z` or `π/2 + atan z` as the quadrant and the order of
/// the magnitudes say, the sign of `y`'s. No sum with a multiple of `π/2` cancels, as the sums are
/// above `π/4`; the ratio's error passes through the arctangent, `|atan'| <= 1`.
pub(super) fn atan2(y: &Fx, x: &Fx, w: u64) -> Option<Fx> {
    if !sign_known(y) || !sign_known(x) {
        return None;
    }
    let (ay, ax) = (abs(y.clone()), abs(x.clone()));
    let big = cmp_mag(&ay, &ax) == Ordering::Greater;
    let z = if big {
        div(&ax, &ay, w + 64)?
    } else {
        div(&ay, &ax, w + 64)?
    };
    if z.m.is_empty() {
        return None;
    }
    let a = atan_unit(&z, w + 8);
    let part = match (big, x.negative) {
        (false, false) => 0,
        (false, true) => 2,
        (true, false) => 1,
        (true, true) => 3,
    };
    let mut r = plus_pi_part(a, part, w + 8);
    limbs::trim(&mut r.m);
    r.negative = y.negative;
    Some(r)
}

/// The order of the approximations of two magnitudes.
pub(super) fn cmp_mag(a: &Fx, b: &Fx) -> Ordering {
    match (a.m.is_empty(), b.m.is_empty()) {
        (true, true) => return Ordering::Equal,
        (true, false) => return Ordering::Less,
        (false, true) => return Ordering::Greater,
        _ => {}
    }
    match a.exponent().cmp(&b.exponent()) {
        Ordering::Equal => {}
        o => return o,
    }
    let l = a.lsb.min(b.lsb);
    let x = limbs::shl(&a.m, (a.lsb - l) as u64);
    let y = limbs::shl(&b.m, (b.lsb - l) as u64);
    limbs::cmp(&x, &y)
}

/// A constant of [`super::super::consts`] as an [`Fx`].
fn constant(a: Approx) -> Fx {
    let mut m = a.m;
    limbs::trim(&mut m);
    Fx {
        negative: false,
        m,
        lsb: a.lsb,
        err: pow2_in(a.err, a.lsb),
    }
}

/// π to about `w` bits.
pub(super) fn pi(w: u64) -> Fx {
    constant(pi_approx(w + 8))
}

/// `ln 2` to about `w` bits.
pub(super) fn ln2(w: u64) -> Fx {
    constant(ln2_approx(w + 8))
}

/// `ln 10` to about `w` bits.
pub(super) fn ln10(w: u64) -> Fx {
    constant(ln10_approx(w + 8))
}

// Ziv's rounding of the two parts.

/// The correct roundings of the two parts of a result, each to its precision in its mode, and
/// their directions, from `approx(w, need)`: at the working precision `w`, an attempt for each part
/// that `need` asks for (the parts not yet decided). The attempts start at the larger precision
/// and [`guard`] bits more and grow by half each time; a part is decided by
/// [`fast_round`] or [`round_approx`], or is given by the attempt itself.
pub(super) fn ziv2(
    prec: (u64, u64),
    round: (Round, Round),
    mut approx: impl FnMut(u64, [bool; 2]) -> [Option<Step>; 2],
) -> (Part, Part) {
    let precs = [prec.0, prec.1];
    let rounds = [round.0, round.1];
    let top = prec.0.max(prec.1);
    let mut w = top + guard(top);
    let mut parts: [Option<Part>; 2] = [None, None];
    loop {
        let need = [parts[0].is_none(), parts[1].is_none()];
        let got = approx(w, need);
        for (i, g) in got.into_iter().enumerate() {
            if !need[i] {
                continue;
            }
            parts[i] = match g {
                Some(Err(p)) => Some(p),
                Some(Ok(a)) => fast_round(&a, precs[i], rounds[i])
                    .or_else(|| round_approx(&a, precs[i], rounds[i])),
                None => None,
            };
        }
        if parts.iter().all(Option::is_some) {
            let [re, im] = parts;
            return (re.expect("decided"), im.expect("decided"));
        }
        w += (w / 2).max(64);
    }
}

/// The correct rounding of one real value to `prec` bits in the mode `round`, and its direction,
/// from `approx(w)` at the working precision `w`, as [`ziv2`] for one part.
pub(super) fn ziv1(prec: u64, round: Round, mut approx: impl FnMut(u64) -> Step) -> Part {
    let mut w = prec + guard(prec);
    loop {
        match approx(w) {
            Err(p) => return p,
            Ok(a) => {
                if let Some(r) =
                    fast_round(&a, prec, round).or_else(|| round_approx(&a, prec, round))
                {
                    return r;
                }
            }
        }
        w += (w / 2).max(64);
    }
}

/// The step of a part from an approximation of it.
#[inline]
pub(super) fn step(f: &Fx) -> Step {
    Ok(to_approx(f))
}

/// The step of a part from an approximation that may be beyond the exponent range: the overflow
/// or the underflow of the sign `negative` by the mode, as [`super::super::exp::out_of_range`].
pub(super) fn range_step(r: Range, negative: bool, prec: u64, round: Round) -> Step {
    Err(super::super::exp::out_of_range(r, negative, prec, round))
}

// Series of the parts of the functions of tiny arguments.

/// The odd series `Σ_(k >= 1) s_k·c_k·x^(2k + 1)/(2k + 1)!` for `|x| < 2^-4`, `s_k = (-1)^k` when
/// `alternate` (otherwise 1) and `c_k = 2k` when `weighted` (otherwise 1), to about `w` bits
/// relatively, with the bound of its tail: `|Σ_(k > K)| <= Σ_(k > K) |x|^(2k + 1) <= 2|x|^(2K +
/// 3)`, as `2k/(2k + 1)! <= 1`. These are `sin x - x` (`alternate`), `sinh x - x`, `x cos x - sin x`
/// (`alternate` and `weighted`, the sign turned over) and `x cosh x - sinh x` (`weighted`).
pub(super) fn odd_series(x: &Fx, w: u64, alternate: bool, weighted: bool) -> Fx {
    if x.m.is_empty() && x.err.0 == f64::NEG_INFINITY {
        return zero();
    }
    let e = top(x);
    if e >= -3 || bad(x) {
        return useless();
    }
    // Terms while above `2^-(w + 24)` of the first: `2K(-e) >= w + 24`.
    let k_max = u128::from(w + 24).div_ceil(2 * (-e) as u128) as u64 + 1;
    let x2 = sqr(x, w + 8);
    // `x^(2k + 1)/(2k + 1)!`, from `x³/6`.
    let mut t = div_int(&mul(&x2, x, w + 8), 6, w + 8);
    let mut sum = zero();
    for k in 1..=k_max {
        if k > 1 {
            t = div_int(&mul(&t, &x2, w + 8), (2 * k) * (2 * k + 1), w + 8);
        }
        let mut term = if weighted {
            mul_int(&t, 2 * k)
        } else {
            t.clone()
        };
        if alternate && k % 2 == 1 {
            term = neg(term);
        }
        sum = if k == 1 {
            term
        } else {
            add(&sum, &term, w + 8)
        };
    }
    let tail = 1 + (2 * k_max as i128 + 3) * e;
    sum.err = sum.err.add(pow2_in(tail, sum.lsb));
    sum
}

/// `ln(1 + d) - d` for `|d| < 2^-4`, to about `w` bits relatively: the series `Σ_(k >= 2)
/// (-1)^(k + 1) d^k/k` with its tail, at most `2|d|^(K + 1)`; for `|d| >= 2^-(w/3)`, `ln(1 + d)`
/// at `w - 2e + 16` bits less `d`, which cancels `-e` bits.
pub(super) fn ln1p_minus(d: &Fx, w: u64) -> Option<Fx> {
    if d.m.is_empty() && d.err.0 == f64::NEG_INFINITY {
        return Some(zero());
    }
    let e = top(d);
    if e >= -3 || bad(d) {
        return None;
    }
    if -e <= i128::from(w / 3) + 2 {
        let wx = w + 16 + 2 * (-e) as u64;
        return Some(sub(&ln1p(d, wx)?, d, wx));
    }
    let k_max = (u128::from(w + 24) / (-e) as u128) as u64 + 3;
    let mut sum = zero();
    let mut power = d.clone();
    for k in 2..=k_max {
        power = mul(&power, d, w + 8);
        let mut t = div_int(&power, k, w + 8);
        if k % 2 == 0 {
            t = neg(t);
        }
        sum = if k == 2 { t } else { add(&sum, &t, w + 8) };
    }
    let tail = 1 + (k_max as i128 + 1) * e;
    sum.err = sum.err.add(pow2_in(tail, sum.lsb));
    Some(sum)
}

// Parts and pairs.

/// The exact part `x`.
#[inline]
pub(super) fn exactly(x: BigFloat) -> Part {
    (x, Ordering::Equal)
}

/// NaN of the precision.
#[inline]
pub(super) fn nan(prec: u64) -> BigFloat {
    BigFloat::nan(prec)
}

/// `±∞` of the precision.
#[inline]
pub(super) fn inf(prec: u64, negative: bool) -> BigFloat {
    if negative {
        BigFloat::neg_infinity(prec)
    } else {
        BigFloat::infinity(prec)
    }
}

/// `±0` of the precision.
#[inline]
pub(super) fn zero_of(prec: u64, negative: bool) -> BigFloat {
    super::raw::zero(prec, negative)
}

/// Both parts NaN.
#[inline]
pub(super) fn nans(prec: (u64, u64)) -> (Part, Part) {
    (exactly(nan(prec.0)), exactly(nan(prec.1)))
}

/// `x` rounded to `prec` bits in the mode `round`.
#[inline]
pub(super) fn set(x: &BigFloat, prec: u64, round: Round) -> Part {
    x.prec_round(prec, round)
}

/// `-v` for the rounding `(v, o)` of a value in the mode for `-v`.
#[inline]
pub(super) fn negated((x, o): Part) -> Part {
    (-x, o.reverse())
}

/// `±f` for a function `f` of a mode that rounds a value that is not below zero: `f` in the mode
/// for the value turned over, then turned over, when `negative`.
#[inline]
pub(super) fn signed(negative: bool, round: Round, f: impl FnOnce(Round) -> Part) -> Part {
    if negative {
        negated(f(super::super::arith::round_for(round, true)))
    } else {
        f(round)
    }
}

/// `±π·num/4` rounded, `num` 1 to 4: by the arctangents of the points `(±∞, ±∞)`, `(±1, 0)` and
/// `(±0, -1)`, which [`BigFloat::atan2_round`] rounds.
pub(super) fn pi_part(num: u8, negative: bool, prec: u64, round: Round) -> Part {
    let one = BigFloat::from(if negative { -1 } else { 1 });
    let (y, x) = match num {
        1 => (inf(2, negative), BigFloat::infinity(2)),
        2 => (one, BigFloat::zero(2)),
        3 => (inf(2, negative), BigFloat::neg_infinity(2)),
        _ => (zero_of(2, negative), BigFloat::from(-1)),
    };
    y.atan2_round(&x, prec, round)
}

/// The precisions checked.
#[track_caller]
pub(super) fn check(prec: (u64, u64)) -> (u64, u64) {
    (BigFloat::check_prec(prec.0), BigFloat::check_prec(prec.1))
}

/// The number of the two parts and the directions of their roundings.
#[inline]
pub(super) fn pair((re, im): (Part, Part)) -> (super::BigComplex, super::Ternary) {
    (super::BigComplex::new(re.0, im.0), (re.1, im.1))
}

/// `x·y` exactly, without the cut of [`mul`].
pub(super) fn mul_exact(a: &Fx, b: &Fx) -> Fx {
    debug_assert!(a.err.0 == f64::NEG_INFINITY && b.err.0 == f64::NEG_INFINITY);
    exact(
        a.negative != b.negative,
        limbs::mul(&a.m, &b.m),
        a.lsb + b.lsb,
    )
}

/// `x + y` exactly, for exact values.
pub(super) fn add_exact(a: &Fx, b: &Fx) -> Fx {
    let mut s = sincos::sum(&[a, b]);
    limbs::trim(&mut s.m);
    s
}

/// `a + b` exactly when both are exact and the sum takes at most `4w + 4096` bits, otherwise as
/// [`add`].
pub(super) fn xadd(a: &Fx, b: &Fx, w: u64) -> Fx {
    let exact = a.err.0 == f64::NEG_INFINITY && b.err.0 == f64::NEG_INFINITY;
    if exact {
        if a.m.is_empty() {
            return b.clone();
        }
        if b.m.is_empty() {
            return a.clone();
        }
        let span = a.exponent().max(b.exponent()) - a.lsb.min(b.lsb);
        if span <= 4 * i128::from(w) + 4096 {
            return add_exact(a, b);
        }
    }
    add(a, b, w)
}

/// `x² + y² - 1` for `|x|`, `|y|` exact: for a larger part of the exponent 0 or 1, the square
/// of the larger less 1 exactly (it has at most twice its bits and two more), then the smaller's
/// square added as [`xadd`] does; otherwise the sum of the squares at `w` bits, less 1.
pub(super) fn norm_m1(x: &Fx, y: &Fx, w: u64) -> Fx {
    let (big, small) = if x.exponent() >= y.exponent() {
        (x, y)
    } else {
        (y, x)
    };
    let k = big.exponent();
    if k == 0 || k == 1 {
        let d1 = add_exact(&mul_exact(big, big), &neg(one()));
        let s2 = mul_exact(small, small);
        return xadd(&d1, &s2, w + 16);
    }
    let n = add(&sqr(big, w + 8), &sqr(small, w + 8), w + 8);
    sub(&n, &one(), w + 8)
}

/// The rounding of a part of an odd function `f(z) = z + c·z³ + Σ_(k >= 2) a_k z^(2k + 1)`, with
/// `|a_k|·(2k + 1) <= 1` (`sin`, `tan`, `asin`, `atan`, `c = cn/6` for `cn` -1, 2, 1, -2), of a
/// tiny `z = x + yi`, beside its first term, when that decides it: `Re f = x·(1 + c(x² - 3y²) +
/// θ)` and `Im f = y·(1 + c(3x² - y²) + θ)` with `|θ| <= 2|z|^4` for `|z| <= 1/16`, as `|Re
/// z^n| <= n·|z|^(n - 1)·|x|` and `|Im z^n| <= n·|z|^(n - 1)·|y|` for an odd `n` (`|cos nφ| <=
/// n|cos φ|`, `|sin nφ| <= n|sin φ|`). `None` when the difference is not known to lie below the
/// last place, or its sign is not known.
pub(super) fn odd_tiny(
    x: &Fx,
    y: &Fx,
    cn: i64,
    imag: bool,
    prec: u64,
    round: Round,
) -> Option<Part> {
    let m = top(x).max(top(y));
    if 2 * m + 8 > -i128::from(prec) {
        return None;
    }
    let (l, o) = if imag { (y, x) } else { (x, y) };
    // `x² - 3y²` (`3x² - y²` for the imaginary part): the lead's square less three times the
    // other's, or the other way round.
    let (l2, o2) = (mul_exact(l, l), mul_exact(o, o));
    let u = if imag {
        add_exact(&mul_int(&o2, 3), &neg(l2))
    } else {
        add_exact(&l2, &neg(mul_int(&o2, 3)))
    };
    if u.m.is_empty() {
        return None;
    }
    let w = prec + 64;
    let mut d = div_int(&mul_int(&mul_exact(l, &u), cn.unsigned_abs()), 6, w);
    if cn < 0 {
        d = neg(d);
    }
    // `2|l||z|^4 < 2^(top(l) + 3 + 4m)`, `|z| < √2·2^m`.
    d.err = d.err.add(pow2_in(top(l) + 3 + 4 * m, d.lsb));
    if !sign_known(&d) {
        return None;
    }
    trig::near(
        l.negative,
        &l.m,
        l.lsb,
        d.negative == l.negative,
        top(&d) + 1,
        prec,
        round,
    )
}
