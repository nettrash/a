//! The exact real numbers that the complex arithmetic works on: the products of the parts, their
//! sums and differences, exactly or cut to a working precision, outside the exponent range of
//! [`BigFloat`] (the lowest bit's exponent is an `i128`), and their roundings.

use super::super::arith::{Mag, add_mags, div_mags, sqrt_mag};
use super::super::limbs;
use super::super::round::round_mag;
use super::super::{BigFloat, Kind, Round};
use core::cmp::Ordering;

/// A signed magnitude `±m·2^lsb`, `m` not zero and without zero limbs at the top.
#[derive(Clone, Debug)]
pub(super) struct Val {
    pub(super) negative: bool,
    pub(super) m: Vec<u64>,
    pub(super) lsb: i128,
}

/// An exact real number: a signed zero or a [`Val`].
#[derive(Clone, Debug)]
pub(super) enum Exact {
    Zero(bool),
    Val(Val),
}

impl Val {
    /// The exponent `e` of the value, `2^(e - 1) ≤ |v| < 2^e`.
    #[inline]
    pub(super) fn top(&self) -> i128 {
        self.lsb + i128::from(limbs::bit_len(&self.m))
    }

    /// The number of bits of the magnitude.
    #[inline]
    pub(super) fn bits(&self) -> u64 {
        limbs::bit_len(&self.m)
    }

    #[inline]
    pub(super) fn mag(&self) -> Mag<'_> {
        Mag {
            negative: self.negative,
            m: &self.m,
            lsb: self.lsb,
        }
    }

    /// The value times `2^k`, exactly.
    #[inline]
    pub(super) fn scale(mut self, k: i128) -> Val {
        self.lsb += k;
        self
    }

    /// The absolute value.
    #[inline]
    pub(super) fn abs(mut self) -> Val {
        self.negative = false;
        self
    }
}

impl Exact {
    /// The number with its sign turned over.
    #[inline]
    pub(super) fn neg(self) -> Exact {
        match self {
            Exact::Zero(n) => Exact::Zero(!n),
            Exact::Val(mut v) => {
                v.negative = !v.negative;
                Exact::Val(v)
            }
        }
    }

    /// Whether the sign bit is set.
    #[inline]
    pub(super) fn is_negative(&self) -> bool {
        match self {
            Exact::Zero(n) => *n,
            Exact::Val(v) => v.negative,
        }
    }
}

/// The number of zero limbs at the bottom of a mantissa.
#[inline]
fn lead(m: &[u64]) -> usize {
    m.iter().position(|&x| x != 0).unwrap_or(0)
}

/// The significant limbs of a finite number that is not zero and the exponent of their lowest
/// bit, the zero limbs below the precision left out.
#[inline]
pub(super) fn significant(x: &BigFloat) -> (bool, &[u64], i128) {
    let (negative, m, lsb) = x.parts().expect("a finite number that is not zero");
    let z = lead(m);
    (negative, &m[z..], lsb + 64 * z as i128)
}

/// A finite float exactly.
pub(super) fn exact(x: &BigFloat) -> Exact {
    match x.kind {
        Kind::Zero { negative } => Exact::Zero(negative),
        Kind::Finite { .. } => {
            let (negative, m, lsb) = significant(x);
            Exact::Val(Val {
                negative,
                m: m.to_vec(),
                lsb,
            })
        }
        _ => unreachable!("a finite number"),
    }
}

/// `x·y` exactly, for finite `x` and `y`; a zero of the sign of the product when either is zero.
pub(super) fn prod(x: &BigFloat, y: &BigFloat) -> Exact {
    let negative = x.is_sign_negative() != y.is_sign_negative();
    if x.is_zero() || y.is_zero() {
        return Exact::Zero(negative);
    }
    let (_, mx, lx) = significant(x);
    let (_, my, ly) = significant(y);
    let mut m = if core::ptr::eq(x, y) {
        limbs::mul(mx, mx)
    } else {
        limbs::mul(mx, my)
    };
    limbs::trim(&mut m);
    Exact::Val(Val {
        negative,
        m,
        lsb: lx + ly,
    })
}

/// `x²` exactly, for finite `x`; `+0` for a zero.
pub(super) fn square(x: &BigFloat) -> Exact {
    prod(x, x)
}

/// The zero of an exact sum of the zeros of the signs `a` and `b`, IEEE 754's: their sign when
/// they agree, otherwise `-0` when rounding down and `+0` in the other modes.
#[inline]
pub(super) fn zero_sum(a: bool, b: bool, round: Round) -> bool {
    if a == b { a } else { round == Round::Down }
}

/// `p + q` rounded once to `prec` bits in the mode `round`, and the direction of the rounding;
/// an exact zero is IEEE 754's: the sign of two zeros of one sign, otherwise `+0`, or `-0` when
/// rounding down.
pub(super) fn sum_round(p: &Exact, q: &Exact, prec: u64, round: Round) -> (BigFloat, Ordering) {
    match (p, q) {
        (Exact::Zero(a), Exact::Zero(b)) => (zero(prec, zero_sum(*a, *b, round)), Ordering::Equal),
        (Exact::Zero(_), Exact::Val(v)) | (Exact::Val(v), Exact::Zero(_)) => {
            round_mag(v.negative, &v.m, v.lsb, false, prec, round)
        }
        (Exact::Val(a), Exact::Val(b)) => add_mags(a.mag(), b.mag(), prec, round),
    }
}

/// The number of bits an exact sum of `p` and `q` may need.
pub(super) fn sum_len(p: &Exact, q: &Exact) -> u64 {
    match (p, q) {
        (Exact::Val(a), Exact::Val(b)) => (a.top().max(b.top()) - a.lsb.min(b.lsb) + 1) as u64,
        (Exact::Val(a), _) | (_, Exact::Val(a)) => a.bits(),
        _ => 0,
    }
}

/// `a + b` exactly, `None` when they cancel.
pub(super) fn add_vals(a: &Val, b: &Val) -> Option<Val> {
    let s = a.lsb.min(b.lsb);
    let x = limbs::shl(&a.m, (a.lsb - s) as u64);
    let y = limbs::shl(&b.m, (b.lsb - s) as u64);
    if a.negative == b.negative {
        return Some(Val {
            negative: a.negative,
            m: limbs::add(&x, &y),
            lsb: s,
        });
    }
    match limbs::cmp(&x, &y) {
        Ordering::Equal => None,
        Ordering::Greater => Some(Val {
            negative: a.negative,
            m: limbs::sub(&x, &y),
            lsb: s,
        }),
        Ordering::Less => Some(Val {
            negative: b.negative,
            m: limbs::sub(&y, &x),
            lsb: s,
        }),
    }
}

/// `p + q` exactly; an exact zero as [`sum_round`] has it in the mode `round`.
pub(super) fn sum_exact(p: &Exact, q: &Exact, round: Round) -> Exact {
    match (p, q) {
        (Exact::Zero(a), Exact::Zero(b)) => Exact::Zero(zero_sum(*a, *b, round)),
        (Exact::Zero(_), v) | (v, Exact::Zero(_)) => v.clone(),
        (Exact::Val(a), Exact::Val(b)) => match add_vals(a, b) {
            Some(v) => Exact::Val(v),
            None => Exact::Zero(round == Round::Down),
        },
    }
}

/// The value cut toward zero to `w` bits: below it by less than `2^(1 - w)` of it.
pub(super) fn trunc(v: Val, w: u64) -> Val {
    let bits = v.bits();
    if bits <= w {
        return v;
    }
    Val {
        negative: v.negative,
        m: limbs::shr(&v.m, bits - w),
        lsb: v.lsb + i128::from(bits - w),
    }
}

/// `p + q` to a working precision of `w ≥ 8` bits: an exact zero only when the sum is zero, and
/// otherwise within `2^(2 - w)` of the sum, relatively.
///
/// When the smaller term `y` has `|y| < 2^(e_x - w - 4)`, `e_x` the exponent of the larger `x`,
/// it is left out: `|y| < 2^(-w - 3)·|x|`, and `x` cut to `w` bits is within `2^(1 - w)·|x|`;
/// with `|x + y| ≥ (1 - 2^(-w - 3))·|x|`, the error is below `1.13·2^(1 - w)·|x + y|`. Otherwise
/// the sum is exact before it is cut, within `2^(1 - w)`.
pub(super) fn sum_trunc(p: &Exact, q: &Exact, w: u64, round: Round) -> Exact {
    match (p, q) {
        (Exact::Val(a), Exact::Val(b)) => {
            let (x, y) = if a.top() >= b.top() { (a, b) } else { (b, a) };
            if y.top() < x.top() - i128::from(w) - 4 {
                return Exact::Val(trunc(x.clone(), w));
            }
            match add_vals(a, b) {
                Some(v) => Exact::Val(trunc(v, w)),
                None => Exact::Zero(round == Round::Down),
            }
        }
        _ => match sum_exact(p, q, round) {
            Exact::Val(v) => Exact::Val(trunc(v, w)),
            z => z,
        },
    }
}

/// The value of a float of the precision `w` that is finite and not zero, the exponent `k` added.
fn of_float(x: BigFloat, k: i128) -> Val {
    let (negative, m, lsb) = x.parts().expect("a finite number that is not zero");
    let z = lead(m);
    let mut m = m[z..].to_vec();
    limbs::trim(&mut m);
    Val {
        negative,
        m,
        lsb: lsb + 64 * z as i128 + k,
    }
}

/// `|n| / |d|` cut toward zero to `w` bits, within `2^(1 - w)` of it, relatively; of the sign
/// `negative`.
pub(super) fn div_trunc(negative: bool, n: &Val, d: &Val, w: u64) -> Val {
    // At the exponents 0, far inside the exponent range, and moved back after.
    let (q, _) = div_mags(negative, &n.m, 0, &d.m, 0, w, Round::TowardZero);
    of_float(q, n.lsb - d.lsb)
}

/// `sqrt(|v|)` cut toward zero to `w` bits, within `2^(1 - w)` of it, relatively.
pub(super) fn sqrt_trunc(v: &Val, w: u64) -> Val {
    let odd = v.lsb.rem_euclid(2);
    let (r, _) = sqrt_mag(&v.m, odd, w, Round::TowardZero);
    of_float(r, (v.lsb - odd) / 2)
}

/// The exact square root of `|v|` when it is a dyadic number, `None` otherwise.
pub(super) fn sqrt_exact(v: &Val) -> Option<Val> {
    let (m, lsb) = if v.lsb.rem_euclid(2) == 1 {
        (limbs::shl(&v.m, 1), v.lsb - 1)
    } else {
        (v.m.clone(), v.lsb)
    };
    let n = limbs::to_number(&m, false, crate::math::numbers::ByteOrder::LittleEndian);
    let (s, r) = n.isqrt_rem();
    if !r.is_zero() {
        return None;
    }
    let mut s = limbs::of_number(&s);
    limbs::trim(&mut s);
    Some(Val {
        negative: false,
        m: s,
        lsb: lsb / 2,
    })
}

/// `|n| / |d|` exactly when it is a dyadic number, `None` otherwise; of the sign `negative`.
pub(super) fn div_exact(negative: bool, n: &Val, d: &Val) -> Option<Val> {
    // `d = D·2^t` with `D` odd divides `n` exactly when `D` divides `N`.
    let tz = limbs::trailing_zeros(&d.m);
    let dm = limbs::shr(&d.m, tz);
    let order = crate::math::numbers::ByteOrder::LittleEndian;
    let (q, r) = limbs::to_number(&n.m, false, order).div_rem(&limbs::to_number(&dm, false, order));
    if !r.is_zero() {
        return None;
    }
    let mut q = limbs::of_number(&q);
    limbs::trim(&mut q);
    Some(Val {
        negative,
        m: q,
        lsb: n.lsb - d.lsb - i128::from(tz),
    })
}

/// A zero of the sign at the precision.
#[inline]
pub(super) fn zero(prec: u64, negative: bool) -> BigFloat {
    BigFloat {
        prec,
        kind: Kind::Zero { negative },
    }
}

/// An exact number rounded to `prec` bits in the mode `round`, and the direction.
pub(super) fn round_exact(e: &Exact, prec: u64, round: Round) -> (BigFloat, Ordering) {
    match e {
        Exact::Zero(negative) => (zero(prec, *negative), Ordering::Equal),
        Exact::Val(v) => round_mag(v.negative, &v.m, v.lsb, false, prec, round),
    }
}
