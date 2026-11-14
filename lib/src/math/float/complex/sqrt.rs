//! The principal square root, each part correctly rounded, as MPC's `mpc_sqrt`: for
//! `z = a + b·i`, `t = √((|z| + |a|)/2)` and `u = |b|/(2t)`, the root is `t ± u·i` for `a > 0`
//! and `u ± t·i` for `a < 0`, the sign of `b`'s.
//!
//! The parts are approximated at a working precision `w` from the exact squares, every step cut
//! toward zero to `w` bits, and rounded by Ziv's strategy. With `η = 2^(1 - w)`: `a² + b²` cut is
//! within `2η` of it, relatively; its root within `2.02η`; the sum with `|a|`, cut, within
//! `4.03η`; `t` within `3.03η`; and `u` within `4.05η`. A part `v` below `2^e` is then within
//! `4.1η·|v|·(1 + 4.2η) < 2^(e + 4 - w)` of the root's part.
//!
//! When the first attempt does not decide, the root is tried exactly: `t` and `u` are dyadic
//! numbers, and Ziv's intervals would hold them at every precision, exactly when `a² + b²` is the
//! square of a dyadic `s` and `(s + |a|)/2` the square of `t` (then `u = |b|/(2t)` is dyadic too,
//! as `t² - u²` and `2tu` are). That needs `|a|` and `|b|` within a factor `2^(p + 2)` of each
//! other, `p` the larger of their precisions: they are then a Pythagorean pair `k(m² - n²)` and
//! `2kmn` up to powers of two, whose odd parts bound `m + n` by `2^p` and so their ratio by
//! `2^(p + 1)`.
//!
//! Parts whose exponents are further apart than about twice the precisions are decided apart
//! from Ziv's strategy (see [`far_root`]): each part of the root is then a square root of one
//! part, `√|a|`, `|b|/(2√|a|)` or `√(|b|/2)`, within far less than that root can lie from a
//! number of the precision or a halfway point unless it is one, and when it is one the side the
//! part lies on decides it.

use super::BigComplex;
use super::Ternary;
use super::fast::{self, put};
use super::raw::{
    Exact, Val, add_vals, div_exact, div_trunc, exact, sqrt_exact, sqrt_trunc, square, sum_exact,
    sum_trunc, zero,
};
use crate::math::float::arith::{round_for, sqrt_mag};
use crate::math::float::round::Rounded;
use crate::math::float::round::round_mag;
use crate::math::float::small::{add2, div2, mul2, round_top2, wide};
use crate::math::float::sqrt_small;
use crate::math::float::ziv::{Approx, guard, round_approx};
use crate::math::float::{BigFloat, Kind, Round};
use core::cmp::Ordering;

/// `±∞` of the precision.
#[inline]
fn inf(prec: u64, negative: bool) -> BigFloat {
    BigFloat {
        prec,
        kind: Kind::Inf { negative },
    }
}

/// `-r` for a root `r` rounded in the mode for `-r`: the result and its direction for `-r`.
#[inline]
fn negated((x, o): (BigFloat, Ordering)) -> (BigFloat, Ordering) {
    (-x, o.reverse())
}

impl BigComplex {
    /// The principal square root, each part rounded to `prec` bits in the mode `round`, and the
    /// directions of the roundings, as MPC's `mpc_sqrt`: its real part is not below zero, and
    /// its imaginary part has the sign of `self`'s, signed zeros included, so that the branch
    /// cut along the negative real axis has `√(-4 + 0i) = 2i` and `√(-4 - 0i) = -2i`. The special
    /// values are MPC's: `√(x ± ∞i) = +∞ ± ∞i` for any `x`, NaN included; `√(-∞ ± yi) = +0 ± ∞i`
    /// and `√(+∞ ± yi) = +∞ ± 0i` for a finite `y`; `√(-∞ + NaN·i) = NaN ± ∞i`,
    /// `√(+∞ + NaN·i) = +∞ + NaN·i`; NaN in both parts for the other numbers with a NaN.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat, Round};
    /// use core::cmp::Ordering;
    ///
    /// let z = BigComplex::new(BigFloat::from(3), BigFloat::from(4));
    /// let (r, o) = z.sqrt_round(53, Round::Nearest);
    /// assert_eq!(r, BigComplex::new(BigFloat::from(2), BigFloat::from(1)));
    /// assert_eq!(o, (Ordering::Equal, Ordering::Equal));
    ///
    /// let (r, _) = BigComplex::new(BigFloat::from(-4), BigFloat::neg_zero(53)).sqrt_round(53, Round::Nearest);
    /// assert_eq!(r, BigComplex::new(BigFloat::from(0), BigFloat::from(-2)));
    /// ```
    ///
    /// # Panics
    ///
    /// When `prec` is not in [`MIN_PREC`](BigFloat::MIN_PREC)..=[`MAX_PREC`](BigFloat::MAX_PREC).
    #[track_caller]
    pub fn sqrt_round(&self, prec: u64, round: Round) -> (BigComplex, (Ordering, Ordering)) {
        self.sqrt_round_parts((prec, prec), (round, round))
    }

    /// [`sqrt_round`](BigComplex::sqrt_round) with a precision and a mode for each part.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat, Round};
    /// use core::cmp::Ordering;
    ///
    /// let i = BigComplex::i(53);
    /// let (r, o) = i.sqrt_round_parts((53, 10), (Round::Up, Round::Down));
    /// assert_eq!(o, (Ordering::Greater, Ordering::Less)); // √2/2·(1 + i)
    /// ```
    #[track_caller]
    pub fn sqrt_round_parts(
        &self,
        prec: (u64, u64),
        round: (Round, Round),
    ) -> (BigComplex, (Ordering, Ordering)) {
        let prec = (BigFloat::check_prec(prec.0), BigFloat::check_prec(prec.1));
        let mut z = BigComplex::new(zero(prec.0, false), zero(prec.1, false));
        let o = root_into(&mut z, self, round);
        (z, o)
    }

    /// The principal square root, to nearest at the precisions of the parts.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat};
    ///
    /// let z = BigComplex::new(BigFloat::from(-3), BigFloat::from(-4));
    /// assert_eq!(z.sqrt(), BigComplex::new(BigFloat::from(1), BigFloat::from(-2)));
    /// ```
    pub fn sqrt(&self) -> BigComplex {
        self.sqrt_round_parts(self.prec(), (Round::Nearest, Round::Nearest))
            .0
    }
}

type Part = (BigFloat, Ordering);

/// Sets `out` to the root of `z` at the precisions of its parts: in registers when the parts of
/// `z` have at most two limbs each and the working precision is at most 127 bits and decides
/// both parts at once ([`short_root`]), otherwise as [`root`] has it.
pub(super) fn root_into(out: &mut BigComplex, z: &BigComplex, round: (Round, Round)) -> Ternary {
    let prec = out.prec();
    if let Some([re, im]) = short_root(z, prec, round) {
        return (put(&mut out.re, re, round.0), put(&mut out.im, im, round.1));
    }
    if let Some(o) = long_root(out, z, round) {
        return o;
    }
    let (re, im) = root(z, prec, round);
    *out = BigComplex::new(re.0, im.0);
    (re.1, im.1)
}

/// A number of the precision `prec` to compute in, its mantissa a vector of the thread's pool.
fn scratch(prec: u64) -> BigFloat {
    let mut v = fast::pool_take();
    v.clear();
    v.resize(prec.div_ceil(64) as usize, 0);
    let n = v.len();
    v[n - 1] = 1 << 63;
    BigFloat {
        prec,
        kind: Kind::Finite {
            negative: false,
            exp: 0,
            mant: v,
        },
    }
}

/// Gives the vector of a number of [`scratch`] back to the pool.
fn release(x: BigFloat) {
    if let Kind::Finite { mant, .. } = x.kind {
        fast::pool_give(mant);
    }
}

/// `x·2^k` in place, for a finite `x` far inside the exponent range.
#[inline]
fn halve(x: &mut BigFloat) {
    if let Kind::Finite { exp, .. } = &mut x.kind {
        *exp -= 1;
    }
}

/// Sets `out` to the value that the finite `x` of the precision `w` approximates within
/// `2^(e + 4 - w)` (`e` its exponent), of the sign `negative`, rounded at the precision `p` of
/// `out`, when that decides it: when the interval holds no multiple of half a unit of the last
/// place of `p` bits at the exponent `e`, so that no number of the precision, no halfway point
/// between two and no power of two (where the unit changes) lies in it. Then the value and `x`
/// round alike, inexactly, in every mode, and `x` is rounded. `false` when it does not decide.
///
/// In units of the lowest bit of the mantissa `M` (`n` limbs) the bound is `E = 2^b`,
/// `b = 64n - w + 4`, and half a unit of the last place `H = 2^h`, `h = 64n - p - 1 > b`; with
/// `L = M mod H`, the interval `[M - E, M + E]` holds no multiple of `H` exactly when
/// `E < L < H - E`: some bit of `L` from `b + 1` up, or bit `b` and a bit below it, is set, and the
/// bits from `b` to `h - 1` are not all ones.
fn decide_into(
    out: &mut BigFloat,
    negative: bool,
    x: &BigFloat,
    w: u64,
    round: Round,
) -> Option<Ordering> {
    let (_, m, lsb) = x.parts()?;
    let n = m.len() as u64;
    let p = out.prec;
    if w < p + 8 {
        return None;
    }
    let (b, h) = (64 * n - w + 4, 64 * n - p - 1);
    let bit = |i: u64| m[(i / 64) as usize] >> (i % 64) & 1 == 1;
    // Bits `from..to` of `M`: whether one is set, whether all are.
    let any = |from: u64, to: u64| (from..to).any(&bit);
    let all = |from: u64, to: u64| (from..to).all(&bit);
    let above = any(b + 1, h) || (bit(b) && any(0, b));
    if !above || all(b, h) {
        return None;
    }
    Some(fast::put_limbs(out, negative, m, lsb, false, round))
}

/// The root of `z = a + b·i` as [`short_root`] has it, for parts of any length, at the working
/// precision `w = p + guard(p)` on numbers of the thread's pool by the real operations into an
/// existing number, each cut toward zero to `w` bits: the bound of [`short_root`] holds, and
/// [`decide_into`] rounds each part into `out`. `None` when the parts are not both finite and not
/// zero, their exponents are not within `2^60` of zero (so that the squares stay in the exponent
/// range), or a part is not decided: then [`root`] decides.
fn long_root(out: &mut BigComplex, z: &BigComplex, round: (Round, Round)) -> Option<Ternary> {
    let (a, b) = (&z.re, &z.im);
    let inside = |x: &BigFloat| x.exponent().is_some_and(|e| e.unsigned_abs() < 1 << 60);
    if !(a.is_normal() && b.is_normal() && inside(a) && inside(b)) {
        return None;
    }
    let prec = out.prec();
    let top = prec.0.max(prec.1);
    let w = top + guard(top);
    let tz = Round::TowardZero;
    let (mut x, mut y) = (scratch(w), scratch(w));
    x.set_mul(a, a, tz);
    y.set_mul(b, b, tz);
    x.add_assign_round(&y, tz);
    // `x = a² + b²`, `y = |z|`, `x = |z| + |a|`, `y = t`, `x = u`.
    y.set_sqrt(&x, tz);
    if a.is_sign_negative() {
        x.set_sub(&y, a, tz);
    } else {
        x.set_add(&y, a, tz);
    }
    halve(&mut x);
    y.set_sqrt(&x, tz);
    x.set_div(b, &y, tz);
    if x.is_sign_negative() {
        x.set_sign(false);
    }
    halve(&mut x);
    let (t, u) = (&y, &x);
    let (re, im) = if a.is_sign_negative() { (u, t) } else { (t, u) };
    let neg = b.is_sign_negative();
    let r = match decide_into(&mut out.re, false, re, w, round.0) {
        Some(o) => decide_into(&mut out.im, neg, im, w, round.1).map(|p| (o, p)),
        None => None,
    };
    release(x);
    release(y);
    r
}

/// The value `0.m·2^exp` (`m` of `w` bits) within `2^(exp + 4 - w)` of a value that is not zero,
/// rounded to `prec ≤ 128` bits in registers as [`round_approx`] rounds it: when both ends of the
/// interval round alike and the result lies outside it.
#[inline(always)]
fn decide(negative: bool, m: u128, exp: i128, w: u64, prec: u64, round: Round) -> Option<Rounded> {
    debug_assert!(m >> 127 == 1 && (16..=127).contains(&w));
    // The bound in units of the lowest bit of `m`, `2^(exp - 128)`.
    let e = 1u128 << (132 - w);
    let lo = m - e;
    let (lo, elo) = if lo >> 127 == 0 {
        (lo << 1, exp - 1)
    } else {
        (lo, exp)
    };
    let (hi, carry) = m.overflowing_add(e);
    let (hh, hl, ehi) = if carry {
        (1 << 127 | hi >> 1, (hi & 1) << 127, exp + 1)
    } else {
        (hi, 0, exp)
    };
    let rl = round_top2(negative, lo, 0, elo, false, prec, round);
    let rh = round_top2(negative, hh, hl, ehi, false, prec, round);
    if rl.m != rh.m || rl.exp != rh.exp {
        return None;
    }
    if rl.magnitude == Ordering::Less {
        Some(rl)
    } else if rh.magnitude == Ordering::Greater {
        Some(rh)
    } else {
        None
    }
}

/// The root of `z = a + b·i` for `a` and `b` finite, not zero and of at most two limbs each, in
/// registers at the working precision `w = min(p + guard(p), 127)` for `p ≤ 113` (`p` the larger precision of the
/// result), each step cut toward zero to `w` bits, below the exact value by less than
/// `η = 2^(1 - w)` of it: `a²` and `b²` each, their sum, its root `|z|`, the sum `|z| + |a|`,
/// `t = √((|z| + |a|)/2)` and `u = |b|/(2t)`. Then `a² + b²` is within `2η` (all terms are above
/// zero), `|z|` within `2η`, the sum with `|a|` within `3η`, `t` within `2.5η`, below, and `u`
/// within `2.6η`, above or below, of their exact values: each part `v`, below `2^e`, within
/// `2.7η·2^e < 2^(e + 4 - w)`, the bound of the [module](self). `None` when the parts are not
/// so, the working precision would be larger, or a part is not decided (an exact root among
/// them): then [`root`] decides.
fn short_root(z: &BigComplex, prec: (u64, u64), round: (Round, Round)) -> Option<[Rounded; 2]> {
    let (
        Kind::Finite {
            negative: na,
            exp: ea,
            mant: ma,
        },
        Kind::Finite {
            negative: nb,
            exp: eb,
            mant: mb,
        },
    ) = (&z.re.kind, &z.im.kind)
    else {
        return None;
    };
    // Up to 113 bits, 14 guard bits at least: an attempt fails with a probability near
    // `2^(6 + p - w) ≤ 2^-8` (the bound's interval, of `2^(5 + e - w)`, over a halfway point's
    // spacing `2^(e - p - 1)`), and then [`long_root`] tries with more.
    let top = prec.0.max(prec.1);
    let w = (top + guard(top)).min(127);
    if ma.len() > 2 || mb.len() > 2 || top > 113 {
        return None;
    }
    let tz = Round::TowardZero;
    let (xa, ea) = (wide(ma), i128::from(*ea));
    let (xb, eb) = (wide(mb), i128::from(*eb));
    let aa = mul2(false, xa, ea, xa, ea, w, tz);
    let bb = mul2(false, xb, eb, xb, eb, w, tz);
    let n = add2(false, aa.m, aa.exp, false, bb.m, bb.exp, w, tz)?;
    let r = sqrt2(n.m, n.exp, w, tz);
    let s = add2(false, r.m, r.exp, false, xa, ea, w, tz)?;
    let t = sqrt2(s.m, s.exp - 1, w, tz);
    let u = div2(false, xb, eb, t.m, t.exp, w, tz);
    // `(re, im)` is `(t, ±u)` for `a > 0` and `(u, ±t)` for `a < 0`, the sign `b`'s.
    let (x, y) = if *na { (u, t) } else { (t, u) };
    let y_exp = if *na { y.exp } else { y.exp - 1 };
    let x_exp = if *na { x.exp - 1 } else { x.exp };
    let re = decide(false, x.m, x_exp, w, prec.0, round.0)?;
    let im = decide(*nb, y.m, y_exp, w, prec.1, round.1)?;
    Some([re, im])
}

/// The exact value of a part, `(value, Equal)`.
#[inline]
fn exactly(x: BigFloat) -> Part {
    (x, Ordering::Equal)
}

fn root(z: &BigComplex, prec: (u64, u64), round: (Round, Round)) -> (Part, Part) {
    let (a, b) = (&z.re, &z.im);
    let neg = b.is_sign_negative();
    if !z.is_finite() {
        if b.is_infinite() {
            return (exactly(inf(prec.0, false)), exactly(inf(prec.1, neg)));
        }
        if a.is_infinite() {
            return match (a.is_sign_negative(), b.is_finite()) {
                (true, true) => (exactly(zero(prec.0, false)), exactly(inf(prec.1, neg))),
                (true, false) => (exactly(BigFloat::nan(prec.0)), exactly(inf(prec.1, neg))),
                (false, true) => (exactly(inf(prec.0, false)), exactly(zero(prec.1, neg))),
                (false, false) => (exactly(inf(prec.0, false)), exactly(BigFloat::nan(prec.1))),
            };
        }
        return (
            exactly(BigFloat::nan(prec.0)),
            exactly(BigFloat::nan(prec.1)),
        );
    }
    if b.is_zero() {
        if a.is_zero() {
            return (exactly(zero(prec.0, false)), exactly(zero(prec.1, neg)));
        }
        if !a.is_sign_negative() {
            return (a.sqrt_round(prec.0, round.0), exactly(zero(prec.1, neg)));
        }
        let m = a.abs();
        let im = if neg {
            negated(m.sqrt_round(prec.1, round_for(round.1, true)))
        } else {
            m.sqrt_round(prec.1, round.1)
        };
        return (exactly(zero(prec.0, false)), im);
    }
    if a.is_zero() {
        // `√(|b|/2)·(1 ± i)`.
        let (_, m, lsb) = b.parts().expect("finite and not zero");
        let re = sqrt_mag(m, lsb - 1, prec.0, round.0);
        let im = if neg {
            negated(sqrt_mag(m, lsb - 1, prec.1, round_for(round.1, true)))
        } else {
            sqrt_mag(m, lsb - 1, prec.1, round.1)
        };
        return (re, im);
    }
    if let Some(r) = far_root(a, b, prec, round) {
        return r;
    }
    general(a, b, prec, round)
}

/// The root of `a + b·i` for `a` and `b` finite and not zero whose exponents are so far apart
/// that each part of the root is a square root of one part within less than that root can lie
/// from a number of the precision or a halfway point, unless it is one; then Ziv's intervals
/// would hold that number at every working precision short of about `2(e_a - e_b)` (or
/// `e_b - e_a`) bits, astronomically many for exponents near the ends of the range, and the root
/// is decided here instead. `None` when the exponents are not so far apart (or the precision
/// needed is above [`MAX_PREC`](BigFloat::MAX_PREC)): Ziv's strategy then needs at most about
/// the precision named below.
///
/// Both cases rest on one fact: the square root of a rational `q` between `1/8` and `8` whose
/// denominator is below `2^n`, when it is not a number of `p + 1` bits, lies further than
/// `2^-(n + 2p + 8)`, relatively, from every number `g` of `p + 1` bits, as
/// `|√q - g| = |q - g²|/(√q + g)` and `q - g²` is a rational of a denominator below
/// `2^(n + 2p + 4)` that is not zero. So a value within `2^-(n + 2p + 20)` of `√q`, relatively,
/// lies in the same open interval between two numbers of `p + 1` bits (the numbers of the
/// precision and the halfway points: in every mode the points where the rounding or its
/// direction changes) as `√q`, and so does the value strictly between `r` (`√q` cut toward zero
/// to `P ≥ n + 2p + 32` bits) and one unit of `r`'s last place more: they round alike, inexactly.
/// When `√q` is a number of `p + 1` bits it is exact at `P` bits, `r = √q`, and a value just
/// above (or just below) it rounds as `r` plus (or less one unit and plus) a sticky bit.
///
/// `|b|` far below `|a|`: with `ρ = |b|/|a| < 2^(e_b - e_a + 1)`,
/// `t = √|a|·√((1 + √(1 + ρ²))/2)` lies in `(√|a|, √|a|·(1 + ρ²/8))` and `u = |b|/(2t)` in
/// `(u₀·(1 - ρ²/8), u₀)`, `u₀ = |b|/(2√|a|)`. With `P = 2p + p_a + 2p_b + 32` (`p` the larger
/// precision of the result, `p_a`, `p_b` the parts'), the operands moved to exponents near zero
/// (`|a| = a'·2^(2k)`, `|b| = b'·2^e_b`): `r ≤ √a' < r·(1 + 2^(1-P))`, and
/// `u₀ = (b'/√a')·2^(e_b - k - 1)` from `v = b'²/a'` cut (`b'²` exact) and `w = √v` cut,
/// `w ≤ b'/√a' < w·(1 + 3·2^(1-P))`. `a'` is a dyadic of `p_a` bits, and `b'²/a'` a rational
/// whose denominator is below `2^(p_a + 2p_b)`, so the fact holds for both with
/// `n = p_a + 2p_b`; the terms in `ρ²` are below `2^(-P-6)` for `e_a - e_b > P/2 + 4`. `t` rounds as `r` with a
/// sticky bit (just above `√|a|` when that is exact); `u` as `w` with a sticky bit, or just
/// below `w` when `v` and `w` are exact (`u₀` dyadic).
///
/// `|a|` far below `|b|`: with `ρ = |a|/|b| < 2^(e_a - e_b + 1)` and `s = √(|b|/2)`,
/// `t² = s²·(√(1 + ρ²) + ρ)` with `1 < √(1 + ρ²) + ρ < (1 + ρ)²`, so `t` lies in
/// `(s, s·(1 + ρ))` and `u = s²/t` in `(s·(1 - ρ), s)`: both are `s` within `ρ < 2^(-P-6)`
/// for `e_b - e_a > P + 7`, `P = 2p + p_b + 32`, `s` the root of a dyadic of `p_b` bits. `t`
/// rounds as `r` (`s` cut to `P` bits) with a sticky bit, just above `s` when that is exact;
/// `u` as `r` with a sticky bit, or just below `s` when that is exact (`n = p_b`).
///
/// For `a > 0` the root is `t ± u·i`, for `a < 0` it is `u ± t·i`, the imaginary part of the
/// sign of `b` (rounded as a negative value, in the mode for it, when `b < 0`).
fn far_root(
    a: &BigFloat,
    b: &BigFloat,
    prec: (u64, u64),
    round: (Round, Round),
) -> Option<(Part, Part)> {
    let (ea, eb) = (i128::from(a.exponent()?), i128::from(b.exponent()?));
    let p = prec.0.max(prec.1);
    let tz = Round::TowardZero;
    let neg = b.is_sign_negative();
    // `|x|` with the exponent `e`.
    let moved = |x: &BigFloat, e: i128| -> BigFloat {
        let (_, m, _) = x.parts().expect("finite and not zero");
        BigFloat {
            prec: x.prec(),
            kind: Kind::Finite {
                negative: false,
                exp: e as i64,
                mant: m.to_vec(),
            },
        }
    };
    // `x·2^shift` as a value strictly between it and one unit of its last place more, or, when
    // `x` is the exact value and the part lies below it (`below`), strictly between one unit
    // less and it: the mantissa less one unit, with a sticky bit. The mantissa has the `P > p`
    // bits of `x`'s precision, so the unit and the sticky bit lie below the precision.
    let part = |x: &BigFloat, shift: i128, below: bool, negative: bool, prec: u64, round: Round| {
        let (_, m, lsb) = x.parts().expect("finite and not zero");
        let mut m = m.to_vec();
        if below {
            // The top bit is set, so the top limb stays nonzero.
            for v in m.iter_mut() {
                let (d, borrow) = v.overflowing_sub(1);
                *v = d;
                if !borrow {
                    break;
                }
            }
        }
        round_mag(negative, &m, lsb + shift, true, prec, round)
    };
    let big = 2 * p + a.prec() + 2 * b.prec() + 32;
    if ea - eb > i128::from(big / 2 + 4) && big <= BigFloat::MAX_PREC {
        // `|a| = a'·2^(2k)` with the exponent of `a'` 0 or 1, `|b| = b'·2^e_b`.
        let k = ea.div_euclid(2);
        let (a1, b1) = (moved(a, ea - 2 * k), moved(b, 0));
        let (r, _) = a1.sqrt_round(big, tz);
        let (b2, _) = b1.mul_round(&b1, 2 * b1.prec(), tz);
        let (v, vo) = b2.div_round(&a1, big, tz);
        let (w, wo) = v.sqrt_round(big, tz);
        let u_below = vo == Ordering::Equal && wo == Ordering::Equal;
        // `u₀ = (b'/√a')·2^(e_b - k - 1)` and `w ≈ b'/√a'`.
        return Some(if a.is_sign_negative() {
            (
                part(&w, eb - k - 1, u_below, false, prec.0, round.0),
                part(&r, k, false, neg, prec.1, round.1),
            )
        } else {
            (
                part(&r, k, false, false, prec.0, round.0),
                part(&w, eb - k - 1, u_below, neg, prec.1, round.1),
            )
        });
    }
    let big = 2 * p + b.prec() + 32;
    if eb - ea > i128::from(big + 7) && big <= BigFloat::MAX_PREC {
        // `|b|/2 = b'·2^(2k)` with the exponent of `b'` 0 or 1.
        let k = (eb - 1).div_euclid(2);
        let (r, ro) = moved(b, eb - 1 - 2 * k).sqrt_round(big, tz);
        let exact = ro == Ordering::Equal;
        return Some(if a.is_sign_negative() {
            (
                part(&r, k, exact, false, prec.0, round.0),
                part(&r, k, false, neg, prec.1, round.1),
            )
        } else {
            (
                part(&r, k, false, false, prec.0, round.0),
                part(&r, k, exact, neg, prec.1, round.1),
            )
        });
    }
    None
}

/// The value of a nonzero finite part, exactly.
fn val(x: &BigFloat) -> Val {
    match exact(x) {
        Exact::Val(v) => v,
        Exact::Zero(_) => unreachable!("a part that is not zero"),
    }
}

/// The root of `a + b·i` for `a` and `b` finite and not zero, by Ziv's strategy (see the
/// [module](self)).
fn general(a: &BigFloat, b: &BigFloat, prec: (u64, u64), round: (Round, Round)) -> (Part, Part) {
    let neg = b.is_sign_negative();
    let positive = !a.is_sign_negative();
    let (va, vb) = (Exact::Val(val(a).abs()), val(b).abs());
    let (aa, bb) = (square(a), square(b));
    // The signs of the parts `t` and `u` as they land in the root: `(t, u)` is `(re, im)` for
    // `a > 0` and `(im, re)` for `a < 0`.
    let (pt, pu) = if positive { (0, 1) } else { (1, 0) };
    let precs = [prec.0, prec.1];
    let rounds = [round.0, round.1];
    let negs = [false, neg];
    let mut parts: [Option<Part>; 2] = [None, None];
    let top = prec.0.max(prec.1);
    let mut w = top + guard(top);
    let mut tried = false;
    loop {
        let n = match sum_trunc(&aa, &bb, w, Round::Nearest) {
            Exact::Val(v) => v,
            Exact::Zero(_) => unreachable!("a² + b² > 0"),
        };
        let m = Exact::Val(sqrt_trunc(&n, w));
        let s = match sum_trunc(&m, &va, w, Round::Nearest) {
            Exact::Val(v) => v,
            Exact::Zero(_) => unreachable!("|z| + |a| > 0"),
        };
        let t = sqrt_trunc(&s.scale(-1), w);
        let u = div_trunc(false, &vb, &t, w).scale(-1);
        for (i, v) in [(pt, &t), (pu, &u)] {
            if parts[i].is_none() {
                let approx = Approx {
                    negative: negs[i],
                    m: v.m.clone(),
                    lsb: v.lsb,
                    err: v.top() + 4 - i128::from(w),
                };
                parts[i] = round_approx(&approx, precs[i], rounds[i]);
            }
        }
        if parts.iter().all(Option::is_some) {
            break;
        }
        if !tried {
            tried = true;
            let (t, u) = exact_root(a, b, &aa, &bb);
            for (i, v) in [(pt, t), (pu, u)] {
                if parts[i].is_none()
                    && let Some(v) = v
                {
                    parts[i] = Some(round_mag(negs[i], &v.m, v.lsb, false, precs[i], rounds[i]));
                }
            }
            if parts.iter().all(Option::is_some) {
                break;
            }
        }
        w += w / 2;
    }
    let [re, im] = parts;
    (re.expect("decided"), im.expect("decided"))
}

/// The parts `t` and `u` of the root exactly, when they are dyadic numbers (see the
/// [module](self)).
fn exact_root(a: &BigFloat, b: &BigFloat, aa: &Exact, bb: &Exact) -> (Option<Val>, Option<Val>) {
    let (ea, eb) = (
        a.exponent().expect("finite and not zero"),
        b.exponent().expect("finite and not zero"),
    );
    let p = a.prec().max(b.prec());
    if ea.abs_diff(eb) > p + 4 {
        return (None, None);
    }
    let n = match sum_exact(aa, bb, Round::Nearest) {
        Exact::Val(v) => v,
        Exact::Zero(_) => unreachable!("a² + b² > 0"),
    };
    let Some(s) = sqrt_exact(&n) else {
        return (None, None);
    };
    let Some(t2) = add_vals(&s, &val(a).abs()) else {
        unreachable!("|z| + |a| > 0")
    };
    let Some(t) = sqrt_exact(&t2.scale(-1)) else {
        return (None, None);
    };
    let u = div_exact(false, &val(b).abs(), &t).map(|u| u.scale(-1));
    (Some(t), u)
}

/// The square root of a mantissa of two limbs as a `u128` aligned at its top bit, `0.x·2^ex`,
/// rounded to `prec ≤ 127` bits: [`sqrt_small::sqrt2`] on its limbs.
#[inline(always)]
fn sqrt2(x: u128, ex: i128, prec: u64, round: Round) -> Rounded {
    sqrt_small::sqrt2(&[x as u64, (x >> 64) as u64], ex, prec, round)
}
