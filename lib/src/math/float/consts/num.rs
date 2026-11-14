//! The numbers of the binary splittings: integers that may be cut to a working number of bits, with
//! a bound of the relative error that the cuts made, and the fixed-point quotient that ends each
//! evaluation.
//!
//! A [`Num`] is `±m·2^e` for limbs `m`; exact while `err` is zero, otherwise known only to lie
//! within a relative error of `err` units of `2^-W` of the value it stands for, `W` the working
//! number of bits of the splitting ([`Cut`]). Cutting a magnitude of more than `W` bits to its top
//! `W` lowers it by less than `2^(1-W)` of itself: 2 units. A product of two numbers within `ε1`
//! and `ε2` is within `ε1 + ε2 + ε1 ε2`, and `ε1 ε2 ≤ 2^-W` while both counts stay below `2^32`
//! and `W ≥ 64`: one unit more. A sum of two numbers of the same sign is within the larger of
//! their errors. The sums of numbers of both signs, whose relative errors cancellation could
//! enlarge, are only ever made of exact numbers (Chudnovsky's series is not cut).

use super::super::limbs;
use crate::math::numbers::{ByteOrder, Number, nat};

/// `±m·2^e`, within `err` units of `2^-W` relative (see the [module](self)).
#[derive(Clone, Debug, Default)]
pub(super) struct Num {
    /// The sign; never set for zero.
    pub(super) neg: bool,
    /// The magnitude, without zero limbs at the top; empty for zero.
    pub(super) m: Vec<u64>,
    /// The exponent of the magnitude's lowest bit.
    pub(super) e: u64,
    /// The bound of the relative error, in units of `2^-W`.
    pub(super) err: u64,
}

impl Num {
    /// The exact number of the limbs `m` and the sign `neg`.
    pub(super) fn exact(neg: bool, mut m: Vec<u64>) -> Num {
        limbs::trim(&mut m);
        Num {
            neg: neg && !m.is_empty(),
            m,
            e: 0,
            err: 0,
        }
    }

    /// The exact number `v`.
    pub(super) fn of_i128(v: i128) -> Num {
        let a = v.unsigned_abs();
        Num::exact(v < 0, vec![a as u64, (a >> 64) as u64])
    }

    /// Whether the number is zero.
    #[inline]
    pub(super) fn is_zero(&self) -> bool {
        self.m.is_empty()
    }

    /// The position of the bit above the top one, `bits(m) + e`.
    #[inline]
    pub(super) fn top(&self) -> u64 {
        limbs::bit_len(&self.m) + self.e
    }

    /// The magnitude `m·2^e` as limbs, shifted out.
    pub(super) fn to_limbs(&self) -> Vec<u64> {
        limbs::shl(&self.m, self.e)
    }
}

/// The working number of bits `W` of a splitting, and the scratch of its products.
pub(super) struct Cut {
    /// The bits kept of a number cut, at least 64; `u64::MAX` for an exact splitting.
    pub(super) w: u64,
    /// The working memory of the products.
    pub(super) scratch: Vec<u64>,
}

impl Cut {
    /// A splitting that keeps `w` bits, or `None` for an exact one.
    pub(super) fn new(w: Option<u64>) -> Cut {
        Cut {
            w: w.map_or(u64::MAX, |w| w.max(64)),
            scratch: Vec::new(),
        }
    }

    /// Cuts `x` to the top `W` bits when it is longer: 2 units more of error.
    #[inline]
    pub(super) fn cut(&self, x: &mut Num) {
        let bits = limbs::bit_len(&x.m);
        if bits > self.w {
            let s = bits - self.w;
            x.m = limbs::shr(&x.m, s);
            x.e += s;
            x.err += 2;
        }
    }

    /// `a·b`, cut.
    pub(super) fn mul(&mut self, a: &Num, b: &Num) -> Num {
        if a.is_zero() || b.is_zero() {
            return Num::default();
        }
        let mut m = Vec::new();
        if core::ptr::eq(a, b) {
            nat::sqr_into(&mut m, &a.m, &mut self.scratch);
        } else {
            nat::mul_into(&mut m, &a.m, &b.m, &mut self.scratch);
        }
        limbs::trim(&mut m);
        let mut r = Num {
            neg: a.neg != b.neg,
            m,
            e: a.e + b.e,
            err: product_err(a.err, b.err),
        };
        self.cut(&mut r);
        r
    }

    /// `a²`, cut.
    pub(super) fn sqr(&mut self, a: &Num) -> Num {
        self.mul(a, a)
    }

    /// `a·2^s`.
    #[inline]
    pub(super) fn shl(a: &mut Num, s: u64) {
        if !a.is_zero() {
            a.e += s;
        }
    }

    /// `a + b`, cut; numbers of both signs must both be exact.
    pub(super) fn add(&mut self, a: Num, b: Num) -> Num {
        if a.is_zero() {
            return b;
        }
        if b.is_zero() {
            return a;
        }
        debug_assert!(a.neg == b.neg || a.err == 0 && b.err == 0 && self.w == u64::MAX);
        // The sum is aligned at the lower exponent, by shifting the other left, unless that would
        // keep more than `W + 2` bits below the top of the larger: the operands are then cut at
        // `t = top - W - 2`, each by less than `2^t`, a relative error of the sum (at least
        // `2^(top - 1)`, as the operands have one sign) below `2^(t + 2 - top) = 2^-W`: one
        // unit.
        let top = a.top().max(b.top());
        let mut e = a.e.min(b.e);
        let mut err = a.err.max(b.err);
        if self.w != u64::MAX && top - e > self.w + 2 {
            e = top - self.w - 2;
            err += 1;
        }
        let align = |x: &Num| -> Vec<u64> {
            if x.e >= e {
                limbs::shl(&x.m, x.e - e)
            } else {
                limbs::shr(&x.m, e - x.e)
            }
        };
        let (ma, mb) = (align(&a), align(&b));
        let (neg, m) = if a.neg == b.neg {
            (a.neg, limbs::add(&ma, &mb))
        } else {
            match limbs::cmp(&ma, &mb) {
                core::cmp::Ordering::Less => (b.neg, limbs::sub(&mb, &ma)),
                _ => (a.neg, limbs::sub(&ma, &mb)),
            }
        };
        let mut r = Num { neg, m, e, err };
        limbs::trim(&mut r.m);
        if r.m.is_empty() {
            return Num::default();
        }
        self.cut(&mut r);
        r
    }
}

/// The error count of a product of numbers within `a` and `b` units: `a + b`, and one more when
/// both are inexact (see the [module](self)).
#[inline]
fn product_err(a: u64, b: u64) -> u64 {
    debug_assert!(
        a < 1 << 32 && b < 1 << 32,
        "error counts this large are not bounded"
    );
    a + b + u64::from(a != 0 && b != 0)
}

/// The number of the magnitude `m`.
pub(super) fn number(m: &[u64]) -> Number {
    limbs::to_number(m, false, ByteOrder::LittleEndian)
}

/// `floor(a·2^s / b)` within less than `1.25` of `a·2^s / b`, for magnitudes `a` and `b` that are
/// not zero: both cut first to the bits the quotient needs.
///
/// With `q = a 2^s / b < 2^qb` for `qb = bits(a) + s - bits(b) + 1`, `b` is cut to `b'` of
/// `qb + 4` bits, `b = 2^c (b' + β)`, and `a 2^(s - c) = a' + α`, `α, β ∈ [0, 1)`. Then
/// `|q - a'/b'| = |α b' - a' β| / (b' (b' + β)) ≤ (1 + a'/b') / b' ≤ (1 + 2^qb) / 2^(qb + 3)`,
/// at most `1/4` for `qb ≥ 0`, and the floor of `a'/b'` is below it by less than 1.
pub(super) fn quotient(a: &[u64], s: i128, b: &[u64]) -> Vec<u64> {
    let (ba, bb) = (limbs::bit_len(a) as i128, limbs::bit_len(b) as i128);
    let qb = ba + s - bb + 1;
    if qb <= 0 {
        // `q < 1`.
        return Vec::new();
    }
    let c = (bb - (qb + 4)).max(0);
    let b = limbs::shr(b, c as u64);
    let t = s - c;
    let a = if t >= 0 {
        limbs::shl(a, t as u64)
    } else {
        limbs::shr(a, (-t) as u64)
    };
    if a.is_empty() {
        return Vec::new();
    }
    let q = &number(&a) / &number(&b);
    limbs::of_number(&q)
}

/// `floor(sqrt(n·4^g))`, within less than 1 below `sqrt(n)·2^g`.
pub(super) fn sqrt_fixed(n: u64, g: u64) -> Vec<u64> {
    let x = limbs::shl(&[n], 2 * g);
    limbs::of_number(&number(&x).isqrt())
}
