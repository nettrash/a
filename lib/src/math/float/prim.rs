//! The operators with a primitive number or a [`Number`] on either side: `x + 1`, `2.0 * &x`,
//! `&x / &n`, `x -= 3u8`. The other operand is made a float exactly (an integer at the bits of its
//! type, a float at its own precision, a `Number` at its bits), and the operation is rounded to
//! nearest at the precision of the `BigFloat`, once.
//!
//! A primitive operand is made a float in a number that each thread keeps for it, so that it
//! allocates nothing; the result is the operation's own, and an assigning operator writes it into
//! the vector of the number it changes.

use super::{BigFloat, Kind, Round};
use crate::math::numbers::Number;
use core::cell::Cell;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

std::thread_local! {
    /// The float that a primitive operand is made in, kept between operators.
    static OPERAND: Cell<Option<BigFloat>> = const { Cell::new(None) };
}

/// Runs `f` on the primitive `v` as a float, built in the thread's number.
#[inline]
fn with_operand<T: Primitive, R>(v: T, f: impl FnOnce(&BigFloat) -> R) -> R {
    let mut x = OPERAND.take().unwrap_or_else(|| BigFloat::zero(1));
    v.store(&mut x);
    let r = f(&x);
    OPERAND.set(Some(x));
    r
}

/// Sets `x` to `±m·2^e` exactly at the precision `prec`, which holds the bits of `m`, reusing
/// the vector of its mantissa.
#[inline]
fn set_exact(x: &mut BigFloat, negative: bool, m: u128, e: i64, prec: u64) {
    x.prec = prec;
    let mut mant = match core::mem::replace(&mut x.kind, Kind::Nan) {
        Kind::Finite { mant, .. } => mant,
        _ => Vec::new(),
    };
    if m == 0 {
        x.kind = Kind::Zero { negative };
        return;
    }
    let lz = m.leading_zeros();
    let top = m << lz;
    mant.clear();
    if prec <= 64 {
        mant.push((top >> 64) as u64);
    } else {
        mant.push(top as u64);
        mant.push((top >> 64) as u64);
    }
    x.kind = Kind::Finite {
        negative,
        exp: e + 128 - i64::from(lz),
        mant,
    };
}

/// A primitive number that an operand of the operators can be.
trait Primitive: Copy {
    /// Sets `x` to the number exactly, as `BigFloat::from` makes it.
    fn store(self, x: &mut BigFloat);
}

macro_rules! integer {
    ($($t:ty => $bits:expr),*) => {$(
        impl Primitive for $t {
            #[inline]
            #[allow(unused_comparisons)]
            fn store(self, x: &mut BigFloat) {
                set_exact(x, self < 0, (self as i128).unsigned_abs(), 0, $bits);
            }
        }
    )*};
}

integer!(i8 => 8, i16 => 16, i32 => 32, i64 => 64, isize => 64, u8 => 8, u16 => 16,
    u32 => 32, u64 => 64, usize => 64, i128 => 128);

impl Primitive for u128 {
    #[inline]
    fn store(self, x: &mut BigFloat) {
        set_exact(x, false, self, 0, 128);
    }
}

impl Primitive for f64 {
    #[inline]
    fn store(self, x: &mut BigFloat) {
        let bits = self.to_bits();
        let negative = bits >> 63 == 1;
        let (field, frac) = ((bits >> 52 & 0x7ff) as i64, bits & ((1 << 52) - 1));
        match field {
            0x7ff => {
                x.prec = 53;
                x.kind = if frac != 0 {
                    Kind::Nan
                } else {
                    Kind::Inf { negative }
                };
            }
            0 => set_exact(x, negative, u128::from(frac), -1074, 53),
            _ => set_exact(x, negative, u128::from(frac | 1 << 52), field - 1075, 53),
        }
    }
}

impl Primitive for f32 {
    #[inline]
    fn store(self, x: &mut BigFloat) {
        let bits = self.to_bits();
        let negative = bits >> 31 == 1;
        let (field, frac) = ((bits >> 23 & 0xff) as i64, bits & ((1 << 23) - 1));
        match field {
            0xff => {
                x.prec = 24;
                x.kind = if frac != 0 {
                    Kind::Nan
                } else {
                    Kind::Inf { negative }
                };
            }
            0 => set_exact(x, negative, u128::from(frac), -149, 24),
            _ => set_exact(x, negative, u128::from(frac | 1 << 23), field - 150, 24),
        }
    }
}

macro_rules! operators {
    ($($t:ty),*) => {$(
        operators!(@one $t, Add, add, AddAssign, add_assign, add_round, add_assign_round);
        operators!(@one $t, Sub, sub, SubAssign, sub_assign, sub_round, sub_assign_round);
        operators!(@one $t, Mul, mul, MulAssign, mul_assign, mul_round, mul_assign_round);
        operators!(@one $t, Div, div, DivAssign, div_assign, div_round, div_assign_round);
    )*};
    (@one $t:ty, $Op:ident, $op:ident, $Assign:ident, $assign:ident, $round:ident,
        $assign_round:ident) => {
        impl $Op<$t> for &BigFloat {
            type Output = BigFloat;
            /// The primitive exactly, the result rounded to nearest at the precision of `self`.
            #[inline]
            fn $op(self, rhs: $t) -> BigFloat {
                with_operand(rhs, |y| self.$round(y, self.prec, Round::Nearest).0)
            }
        }
        impl $Op<$t> for BigFloat {
            type Output = BigFloat;
            #[inline]
            fn $op(mut self, rhs: $t) -> BigFloat {
                with_operand(rhs, |y| self.$assign_round(y, Round::Nearest));
                self
            }
        }
        impl $Op<&BigFloat> for $t {
            type Output = BigFloat;
            /// The primitive exactly, the result rounded to nearest at the precision of `rhs`.
            #[inline]
            fn $op(self, rhs: &BigFloat) -> BigFloat {
                with_operand(self, |x| x.$round(rhs, rhs.prec, Round::Nearest).0)
            }
        }
        impl $Op<BigFloat> for $t {
            type Output = BigFloat;
            #[inline]
            fn $op(self, rhs: BigFloat) -> BigFloat {
                self.$op(&rhs)
            }
        }
        impl $Assign<$t> for BigFloat {
            /// The primitive exactly, the result rounded to nearest at the precision of `self`.
            #[inline]
            fn $assign(&mut self, rhs: $t) {
                with_operand(rhs, |y| self.$assign_round(y, Round::Nearest));
            }
        }
    };
}

operators!(
    i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize, f32, f64
);

macro_rules! numbers {
    ($($Op:ident, $op:ident, $Assign:ident, $assign:ident, $round:ident, $assign_round:ident);*) => {$(
        impl $Op<&Number> for &BigFloat {
            type Output = BigFloat;
            /// The integer exactly, the result rounded to nearest at the precision of `self`.
            #[inline]
            fn $op(self, rhs: &Number) -> BigFloat {
                self.$round(&BigFloat::from(rhs), self.prec, Round::Nearest).0
            }
        }
        impl $Op<Number> for &BigFloat {
            type Output = BigFloat;
            #[inline]
            fn $op(self, rhs: Number) -> BigFloat {
                self.$op(&rhs)
            }
        }
        impl $Op<&Number> for BigFloat {
            type Output = BigFloat;
            #[inline]
            fn $op(mut self, rhs: &Number) -> BigFloat {
                self.$assign_round(&BigFloat::from(rhs), Round::Nearest);
                self
            }
        }
        impl $Op<Number> for BigFloat {
            type Output = BigFloat;
            #[inline]
            fn $op(self, rhs: Number) -> BigFloat {
                self.$op(&rhs)
            }
        }
        impl $Op<&BigFloat> for &Number {
            type Output = BigFloat;
            /// The integer exactly, the result rounded to nearest at the precision of `rhs`.
            #[inline]
            fn $op(self, rhs: &BigFloat) -> BigFloat {
                BigFloat::from(self).$round(rhs, rhs.prec, Round::Nearest).0
            }
        }
        impl $Op<BigFloat> for &Number {
            type Output = BigFloat;
            #[inline]
            fn $op(self, rhs: BigFloat) -> BigFloat {
                self.$op(&rhs)
            }
        }
        impl $Op<&BigFloat> for Number {
            type Output = BigFloat;
            #[inline]
            fn $op(self, rhs: &BigFloat) -> BigFloat {
                (&self).$op(rhs)
            }
        }
        impl $Op<BigFloat> for Number {
            type Output = BigFloat;
            #[inline]
            fn $op(self, rhs: BigFloat) -> BigFloat {
                (&self).$op(&rhs)
            }
        }
        impl $Assign<&Number> for BigFloat {
            /// The integer exactly, the result rounded to nearest at the precision of `self`.
            #[inline]
            fn $assign(&mut self, rhs: &Number) {
                self.$assign_round(&BigFloat::from(rhs), Round::Nearest);
            }
        }
        impl $Assign<Number> for BigFloat {
            #[inline]
            fn $assign(&mut self, rhs: Number) {
                self.$assign(&rhs);
            }
        }
    )*};
}

numbers!(
    Add, add, AddAssign, add_assign, add_round, add_assign_round;
    Sub, sub, SubAssign, sub_assign, sub_round, sub_assign_round;
    Mul, mul, MulAssign, mul_assign, mul_round, mul_assign_round;
    Div, div, DivAssign, div_assign, div_round, div_assign_round
);
