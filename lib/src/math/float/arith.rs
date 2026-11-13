//! `+`, `-`, `*`, `/`, the fused multiply-add and the square root, each correctly rounded, with
//! the special values of IEEE 754: NaN for `∞ - ∞`, `0·∞`, `0/0`, `∞/∞` and the square root of a
//! number below zero; `x/0` an infinity of the sign of the quotient; and an exact zero sum `+0`,
//! or `-0` when rounding down, unless both terms are zeros of the same sign.

use super::limbs;
use super::round::round_mag;
use super::{BigFloat, Kind, Round};
use crate::math::numbers::ByteOrder;
use core::cmp::Ordering;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// A result that is exact: `x` and [`Ordering::Equal`].
#[inline]
fn exact(x: BigFloat) -> (BigFloat, Ordering) {
    (x, Ordering::Equal)
}

/// The value `kind` at the precision `prec`.
#[inline]
fn of(prec: u64, kind: Kind) -> BigFloat {
    BigFloat { prec, kind }
}

/// The zero of an exact sum of zero, of terms of the signs `a` and `b`: their sign when they
/// agree, otherwise `-0` when rounding down and `+0` in the other modes.
#[inline]
fn zero_sum(a: bool, b: bool, round: Round) -> bool {
    if a == b { a } else { round == Round::Down }
}

/// A signed magnitude `±M·2^lsb`, `M` not zero and without zero limbs at the top.
#[derive(Clone, Copy)]
pub(crate) struct Mag<'a> {
    pub(crate) negative: bool,
    pub(crate) m: &'a [u64],
    pub(crate) lsb: i128,
}

impl BigFloat {
    /// The signed magnitude of a finite number that is not zero.
    #[inline]
    pub(crate) fn mag(&self) -> Option<Mag<'_>> {
        self.parts()
            .map(|(negative, m, lsb)| Mag { negative, m, lsb })
    }
}

/// `a + b` for the signed magnitudes `a` and `b` rounded to `prec` in the mode `round`.
///
/// When the smaller term lies below a quarter of the result's last place and below the larger
/// term's last bit, it changes the result only as a sticky bit would: the sum lies strictly
/// between the larger term, put at a scale `2^k` below both, and one more (or one less) unit of
/// `2^k`, and no number of the precision nor any halfway point between two lies there. Otherwise
/// the sum is exact, on at most the precision and the operands' lengths together.
pub(crate) fn add_mags(a: Mag<'_>, b: Mag<'_>, prec: u64, round: Round) -> (BigFloat, Ordering) {
    let ea = a.lsb + i128::from(limbs::bit_len(a.m));
    let eb = b.lsb + i128::from(limbs::bit_len(b.m));
    let ((nx, x, lx, ex), (ny, y, ly, ey)) = if ea >= eb {
        ((a.negative, a.m, a.lsb, ea), (b.negative, b.m, b.lsb, eb))
    } else {
        ((b.negative, b.m, b.lsb, eb), (a.negative, a.m, a.lsb, ea))
    };
    let k = lx.min(ex - i128::from(prec) - 3);
    if ey <= k {
        let mut m = limbs::shl(x, (lx - k) as u64);
        if nx != ny {
            m = limbs::sub(&m, &[1]);
        }
        return round_mag(nx, &m, k, true, prec, round);
    }
    let s = lx.min(ly);
    let xs = limbs::shl(x, (lx - s) as u64);
    let ys = limbs::shl(y, (ly - s) as u64);
    if nx == ny {
        return round_mag(nx, &limbs::add(&xs, &ys), s, false, prec, round);
    }
    match limbs::cmp(&xs, &ys) {
        Ordering::Equal => exact(of(
            prec,
            Kind::Zero {
                negative: round == Round::Down,
            },
        )),
        Ordering::Greater => round_mag(nx, &limbs::sub(&xs, &ys), s, false, prec, round),
        Ordering::Less => round_mag(ny, &limbs::sub(&ys, &xs), s, false, prec, round),
    }
}

/// `a ± b`, `b`'s sign turned over when `negate`.
fn add_signed(
    a: &BigFloat,
    b: &BigFloat,
    negate: bool,
    prec: u64,
    round: Round,
) -> (BigFloat, Ordering) {
    let prec = BigFloat::check_prec(prec);
    match (&a.kind, &b.kind) {
        (Kind::Nan, _) | (_, Kind::Nan) => exact(BigFloat::nan(prec)),
        (Kind::Inf { negative: x }, Kind::Inf { negative: y }) => {
            if *x == (*y != negate) {
                exact(of(prec, Kind::Inf { negative: *x }))
            } else {
                exact(BigFloat::nan(prec))
            }
        }
        (Kind::Inf { negative }, _) => exact(of(
            prec,
            Kind::Inf {
                negative: *negative,
            },
        )),
        (_, Kind::Inf { negative }) => exact(of(
            prec,
            Kind::Inf {
                negative: *negative != negate,
            },
        )),
        (Kind::Zero { negative: x }, Kind::Zero { negative: y }) => exact(of(
            prec,
            Kind::Zero {
                negative: zero_sum(*x, *y != negate, round),
            },
        )),
        (Kind::Zero { .. }, _) => {
            let (mut r, o) = b.prec_round(prec, round_for(round, negate));
            if negate {
                r.set_sign(!r.is_sign_negative());
                (r, o.reverse())
            } else {
                (r, o)
            }
        }
        (_, Kind::Zero { .. }) => a.prec_round(prec, round),
        _ => {
            let (x, mut y) = (a.mag().expect("finite"), b.mag().expect("finite"));
            y.negative ^= negate;
            add_mags(x, y, prec, round)
        }
    }
}

/// The mode that rounds `-x` as `round` rounds `x` when `negate`: up and down swapped.
#[inline]
pub(crate) fn round_for(round: Round, negate: bool) -> Round {
    match (round, negate) {
        (Round::Up, true) => Round::Down,
        (Round::Down, true) => Round::Up,
        (r, _) => r,
    }
}

/// `x·y` of two numbers that are not NaN, as a sign and, unless it is an infinity or a zero, the
/// exact magnitude `M·2^lsb`.
enum Product {
    Nan,
    Inf(bool),
    Zero(bool),
    Finite(bool, Vec<u64>, i128),
}

fn product(x: &BigFloat, y: &BigFloat) -> Product {
    let negative = x.is_sign_negative() != y.is_sign_negative();
    match (&x.kind, &y.kind) {
        (Kind::Nan, _) | (_, Kind::Nan) => Product::Nan,
        (Kind::Inf { .. }, Kind::Zero { .. }) | (Kind::Zero { .. }, Kind::Inf { .. }) => {
            Product::Nan
        }
        (Kind::Inf { .. }, _) | (_, Kind::Inf { .. }) => Product::Inf(negative),
        (Kind::Zero { .. }, _) | (_, Kind::Zero { .. }) => Product::Zero(negative),
        _ => {
            let (_, mx, lx) = x.parts().expect("finite");
            let (_, my, ly) = y.parts().expect("finite");
            // The limbs below the precision are zero: multiply the significant ones only.
            let (zx, zy) = (lead(mx), lead(my));
            let m = limbs::mul(&mx[zx..], &my[zy..]);
            Product::Finite(negative, m, lx + ly + 64 * (zx + zy) as i128)
        }
    }
}

/// The number of zero limbs at the bottom of a mantissa.
#[inline]
fn lead(m: &[u64]) -> usize {
    m.iter().position(|&x| x != 0).unwrap_or(0)
}

impl BigFloat {
    /// `self + other` rounded to `prec` bits in the mode `round`, and the direction of the
    /// rounding.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigFloat, Round};
    /// use core::cmp::Ordering;
    ///
    /// let (x, o) = BigFloat::from(1).add_round(&BigFloat::from(0.25), 2, Round::Down);
    /// assert_eq!(x, BigFloat::from(1));
    /// assert_eq!(o, Ordering::Less);
    /// ```
    ///
    /// # Panics
    ///
    /// When `prec` is not in [`MIN_PREC`](BigFloat::MIN_PREC)..=[`MAX_PREC`](BigFloat::MAX_PREC).
    #[track_caller]
    pub fn add_round(&self, other: &BigFloat, prec: u64, round: Round) -> (BigFloat, Ordering) {
        add_signed(self, other, false, prec, round)
    }

    /// `self - other` rounded to `prec` bits in the mode `round`, and the direction of the
    /// rounding.
    #[track_caller]
    pub fn sub_round(&self, other: &BigFloat, prec: u64, round: Round) -> (BigFloat, Ordering) {
        add_signed(self, other, true, prec, round)
    }

    /// `self · other` rounded to `prec` bits in the mode `round`, and the direction of the
    /// rounding.
    #[track_caller]
    pub fn mul_round(&self, other: &BigFloat, prec: u64, round: Round) -> (BigFloat, Ordering) {
        let prec = BigFloat::check_prec(prec);
        match product(self, other) {
            Product::Nan => exact(BigFloat::nan(prec)),
            Product::Inf(negative) => exact(of(prec, Kind::Inf { negative })),
            Product::Zero(negative) => exact(of(prec, Kind::Zero { negative })),
            Product::Finite(negative, m, lsb) => round_mag(negative, &m, lsb, false, prec, round),
        }
    }

    /// `self / other` rounded to `prec` bits in the mode `round`, and the direction of the
    /// rounding; a number divided by zero is an infinity of the sign of the quotient, and `0/0`
    /// NaN.
    #[track_caller]
    pub fn div_round(&self, other: &BigFloat, prec: u64, round: Round) -> (BigFloat, Ordering) {
        let prec = BigFloat::check_prec(prec);
        let negative = self.is_sign_negative() != other.is_sign_negative();
        match (&self.kind, &other.kind) {
            (Kind::Nan, _) | (_, Kind::Nan) => exact(BigFloat::nan(prec)),
            (Kind::Inf { .. }, Kind::Inf { .. }) | (Kind::Zero { .. }, Kind::Zero { .. }) => {
                exact(BigFloat::nan(prec))
            }
            (Kind::Inf { .. }, _) | (_, Kind::Zero { .. }) => {
                exact(of(prec, Kind::Inf { negative }))
            }
            (_, Kind::Inf { .. }) | (Kind::Zero { .. }, _) => {
                exact(of(prec, Kind::Zero { negative }))
            }
            _ => {
                let (_, a, la) = self.parts().expect("finite");
                let (_, b, lb) = other.parts().expect("finite");
                div_mags(negative, a, la, b, lb, prec, round)
            }
        }
    }

    /// `self · a + b` rounded once to `prec` bits in the mode `round`, and the direction of the
    /// rounding, as IEEE 754's `fusedMultiplyAdd`.
    #[track_caller]
    pub fn mul_add_round(
        &self,
        a: &BigFloat,
        b: &BigFloat,
        prec: u64,
        round: Round,
    ) -> (BigFloat, Ordering) {
        let prec = BigFloat::check_prec(prec);
        match (product(self, a), &b.kind) {
            (Product::Nan, _) | (_, Kind::Nan) => exact(BigFloat::nan(prec)),
            (Product::Inf(x), Kind::Inf { negative: y }) => {
                if x == *y {
                    exact(of(prec, Kind::Inf { negative: x }))
                } else {
                    exact(BigFloat::nan(prec))
                }
            }
            (Product::Inf(negative), _) => exact(of(prec, Kind::Inf { negative })),
            (_, Kind::Inf { negative }) => exact(of(
                prec,
                Kind::Inf {
                    negative: *negative,
                },
            )),
            (Product::Zero(x), Kind::Zero { negative: y }) => exact(of(
                prec,
                Kind::Zero {
                    negative: zero_sum(x, *y, round),
                },
            )),
            (Product::Zero(_), _) => b.prec_round(prec, round),
            (Product::Finite(negative, m, lsb), Kind::Zero { .. }) => {
                round_mag(negative, &m, lsb, false, prec, round)
            }
            (Product::Finite(negative, m, lsb), _) => {
                let p = Mag {
                    negative,
                    m: &m,
                    lsb,
                };
                add_mags(p, b.mag().expect("finite"), prec, round)
            }
        }
    }

    /// `self · a + b` rounded once to nearest at the largest precision of the three, as
    /// `f64::mul_add`.
    pub fn mul_add(&self, a: &BigFloat, b: &BigFloat) -> BigFloat {
        let prec = self.prec.max(a.prec).max(b.prec);
        self.mul_add_round(a, b, prec, Round::Nearest).0
    }

    /// The square root rounded to `prec` bits in the mode `round`, and the direction of the
    /// rounding: NaN below zero, `-0` for `-0`, as IEEE 754 has it.
    #[track_caller]
    pub fn sqrt_round(&self, prec: u64, round: Round) -> (BigFloat, Ordering) {
        let prec = BigFloat::check_prec(prec);
        match &self.kind {
            Kind::Nan | Kind::Inf { negative: true } | Kind::Finite { negative: true, .. } => {
                exact(BigFloat::nan(prec))
            }
            Kind::Inf { negative: false } => exact(BigFloat::infinity(prec)),
            Kind::Zero { negative } => exact(of(
                prec,
                Kind::Zero {
                    negative: *negative,
                },
            )),
            Kind::Finite { .. } => {
                let (_, a, la) = self.parts().expect("finite");
                sqrt_mag(a, la, prec, round)
            }
        }
    }

    /// The square root, to nearest at the precision of `self`.
    pub fn sqrt(&self) -> BigFloat {
        self.sqrt_round(self.prec, Round::Nearest).0
    }
}

/// `±(A·2^la) / (B·2^lb)` rounded: the quotient of `A·2^s` by `B` with at least two bits more
/// than the precision, its remainder the sticky bit.
pub(crate) fn div_mags(
    negative: bool,
    a: &[u64],
    la: i128,
    b: &[u64],
    lb: i128,
    prec: u64,
    round: Round,
) -> (BigFloat, Ordering) {
    let (ba, bb) = (limbs::bit_len(a), limbs::bit_len(b));
    let s = (prec + 2 + bb).saturating_sub(ba);
    let n = limbs::to_number(&limbs::shl(a, s), false, ByteOrder::LittleEndian);
    let d = limbs::to_number(b, false, ByteOrder::LittleEndian);
    let (q, r) = n.div_rem(&d);
    let q = limbs::of_number(&q);
    round_mag(
        negative,
        &q,
        la - lb - i128::from(s),
        !r.is_zero(),
        prec,
        round,
    )
}

/// `sqrt(A·2^la)` rounded: the integer square root of `A·2^(2t)`, with the exponent made even,
/// to at least two bits more than the precision, its remainder the sticky bit.
pub(crate) fn sqrt_mag(a: &[u64], la: i128, prec: u64, round: Round) -> (BigFloat, Ordering) {
    let (a, la) = if la % 2 != 0 {
        (limbs::shl(a, 1), la - 1)
    } else {
        (a.to_vec(), la)
    };
    let bits = limbs::bit_len(&a);
    let t = (2 * prec + 4).saturating_sub(bits).div_ceil(2);
    let n = limbs::to_number(&limbs::shl(&a, 2 * t), false, ByteOrder::LittleEndian);
    let (s, r) = n.isqrt_rem();
    let s = limbs::of_number(&s);
    round_mag(
        false,
        &s,
        (la - 2 * i128::from(t)) / 2,
        !r.is_zero(),
        prec,
        round,
    )
}

macro_rules! binary {
    ($Op:ident, $op:ident, $Assign:ident, $assign:ident, $method:ident) => {
        impl $Op<&BigFloat> for &BigFloat {
            type Output = BigFloat;
            /// Rounded to nearest at the larger precision of the operands.
            #[inline]
            fn $op(self, rhs: &BigFloat) -> BigFloat {
                self.$method(rhs, self.prec.max(rhs.prec), Round::Nearest).0
            }
        }
        impl $Op<BigFloat> for &BigFloat {
            type Output = BigFloat;
            #[inline]
            fn $op(self, rhs: BigFloat) -> BigFloat {
                self.$op(&rhs)
            }
        }
        impl $Op<&BigFloat> for BigFloat {
            type Output = BigFloat;
            #[inline]
            fn $op(self, rhs: &BigFloat) -> BigFloat {
                (&self).$op(rhs)
            }
        }
        impl $Op<BigFloat> for BigFloat {
            type Output = BigFloat;
            #[inline]
            fn $op(self, rhs: BigFloat) -> BigFloat {
                (&self).$op(&rhs)
            }
        }
        impl $Assign<&BigFloat> for BigFloat {
            /// Rounded to nearest at the larger precision of the operands.
            #[inline]
            fn $assign(&mut self, rhs: &BigFloat) {
                *self = (&*self).$op(rhs);
            }
        }
        impl $Assign<BigFloat> for BigFloat {
            #[inline]
            fn $assign(&mut self, rhs: BigFloat) {
                *self = (&*self).$op(&rhs);
            }
        }
    };
}

binary!(Add, add, AddAssign, add_assign, add_round);
binary!(Sub, sub, SubAssign, sub_assign, sub_round);
binary!(Mul, mul, MulAssign, mul_assign, mul_round);
binary!(Div, div, DivAssign, div_assign, div_round);

impl Neg for BigFloat {
    type Output = BigFloat;
    /// The number with its sign turned over, exactly; NaN stays NaN.
    #[inline]
    fn neg(mut self) -> BigFloat {
        let negative = self.is_sign_negative();
        self.set_sign(!negative);
        self
    }
}

impl Neg for &BigFloat {
    type Output = BigFloat;
    #[inline]
    fn neg(self) -> BigFloat {
        -self.clone()
    }
}
