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

use super::BigComplex;
use super::raw::{
    Exact, Val, add_vals, div_exact, div_trunc, exact, sqrt_exact, sqrt_trunc, square, sum_exact,
    sum_trunc, zero,
};
use crate::math::float::arith::{round_for, sqrt_mag};
use crate::math::float::round::round_mag;
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
        let (re, im) = root(self, prec, round);
        (BigComplex::new(re.0, im.0), (re.1, im.1))
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
    general(a, b, prec, round)
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
