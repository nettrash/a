//! The arithmetic operators of rationals: `+`, `-`, `*` and `/` and their assignments, with a
//! rational, a [`Number`] or a primitive integer on either side, each operand owned or borrowed,
//! and `-` to negate.
//!
//! Each takes the register path of [`small`](super::small) when every part fits in 64 bits, and
//! otherwise reduces as Knuth's section 4.5.1 does on limbs, in the kernels of
//! [`fraction`](crate::math::numbers::fraction): the four parts read once, the gcds, exact
//! divisions and products on limbs in working memory, and the two parts of the result written
//! once. The result is in the byte order of the left operand; the vectors of the operand assigned
//! to, or taken by value, take it when they have room.

use super::Rational;
use super::small::{self, Small, Wide};
use crate::math::numbers::fraction::{self, Fraction};
use crate::math::numbers::{ByteOrder, Magnitude, Number};
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// The operation of an operator.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Op {
    Add,
    Sub,
    Mul,
    Div,
}

/// `x op y` on the register path.
#[inline(always)]
fn small_op(x: Small, y: Small, op: Op) -> Wide {
    match op {
        Op::Add => small::add(x, y, false),
        Op::Sub => small::add(x, y, true),
        Op::Mul => small::mul(x, y, false),
        Op::Div => small::mul(x, y, true),
    }
}

/// Panics as an integer division by zero does.
#[cold]
#[inline(never)]
#[track_caller]
fn by_zero() -> ! {
    panic!("attempt to divide by zero")
}

/// `x op y` for fractions given by their parts, each in lowest terms with a positive denominator
/// (`None` for 1), on limbs (see [`fraction`]): the numerator and the denominator of the result,
/// in lowest terms, in the byte order `order`, in new vectors.
#[inline(never)]
fn general(x: Fraction, y: Fraction, op: Op, order: ByteOrder) -> (Number, Number) {
    match op {
        Op::Add | Op::Sub => fraction::sum_new(x, y, op == Op::Sub, order),
        Op::Mul | Op::Div => fraction::product_new(x, y, op == Op::Div, order),
    }
}

/// `t = t op y` (`t = y op t` when `left` is not set), on limbs, in the vectors of `t` where it
/// has them with room, and in words when the result fits in them.
#[inline(never)]
fn general_into(t: &mut Rational, y: Fraction, left: bool, op: Op, order: ByteOrder) {
    let Some((tn, td)) = t.numbers_mut() else {
        // `t` in words has no vectors to write in.
        let x = parts(t);
        let (num, den) = if left {
            general(x, y, op, order)
        } else {
            general(y, x, op, order)
        };
        *t = Rational::from_reduced(num, den);
        return;
    };
    match op {
        Op::Add | Op::Sub => fraction::sum_assign(tn, td, y, left, op == Op::Sub, order),
        Op::Mul | Op::Div => fraction::product_assign(tn, td, y, left, op == Op::Div, order),
    }
    t.settle();
}

/// The parts of `r` as a fraction, read in place.
#[inline(always)]
fn parts(r: &Rational) -> Fraction<'_> {
    let (num, den) = r.magnitudes();
    Fraction {
        num,
        den: Some(den),
    }
}

/// The integer `n` as a fraction.
#[inline(always)]
fn integer(n: &Number) -> Fraction<'_> {
    Fraction {
        num: Magnitude::of(n),
        den: None,
    }
}

/// `a op b` in a new rational, in the byte order of `a`.
///
/// Only the register path is inlined into each operator; the limbs' path, with the reading of
/// the parts of numbers, is out of line.
#[inline(always)]
#[track_caller]
pub(super) fn apply(a: &Rational, b: &Rational, op: Op) -> Rational {
    if let (Some(x), Some(y)) = (a.word(), b.word()) {
        if op == Op::Div && y.num == 0 {
            by_zero();
        }
        return small_op(x, y, op).rational(a.byte_order());
    }
    apply_long(a, b, op)
}

/// [`apply`] when a part does not fit in 64 bits.
#[inline(never)]
#[track_caller]
fn apply_long(a: &Rational, b: &Rational, op: Op) -> Rational {
    if op == Op::Div && b.is_zero() {
        by_zero();
    }
    let (num, den) = general(parts(a), parts(b), op, a.byte_order());
    Rational::from_reduced(num, den)
}

/// `a = a op b`, in the byte order of `a`, in its words, or its vectors where they have room.
#[inline(always)]
#[track_caller]
pub(super) fn apply_left(a: &mut Rational, b: &Rational, op: Op) {
    if let (Some(x), Some(y)) = (a.word(), b.word()) {
        if op == Op::Div && y.num == 0 {
            by_zero();
        }
        let order = a.byte_order();
        small_op(x, y, op).write(a, order);
        return;
    }
    apply_left_long(a, b, op);
}

/// [`apply_left`] when a part does not fit in 64 bits.
#[inline(never)]
#[track_caller]
fn apply_left_long(a: &mut Rational, b: &Rational, op: Op) {
    if op == Op::Div && b.is_zero() {
        by_zero();
    }
    let order = a.byte_order();
    general_into(a, parts(b), true, op, order);
}

/// `b = a op b`, in the byte order of `a`, in the words or the vectors of `b` where they have
/// room.
#[inline(always)]
#[track_caller]
pub(super) fn apply_right(a: &Rational, b: &mut Rational, op: Op) {
    if let (Some(x), Some(y)) = (a.word(), b.word()) {
        if op == Op::Div && y.num == 0 {
            by_zero();
        }
        small_op(x, y, op).write(b, a.byte_order());
        return;
    }
    apply_right_long(a, b, op);
}

/// [`apply_right`] when a part does not fit in 64 bits.
#[inline(never)]
#[track_caller]
fn apply_right_long(a: &Rational, b: &mut Rational, op: Op) {
    if op == Op::Div && b.is_zero() {
        by_zero();
    }
    general_into(b, parts(a), false, op, a.byte_order());
}

/// `a op n` for an integer `n`, in a new rational in the byte order of `a`.
#[inline]
#[track_caller]
pub(super) fn apply_int(a: &Rational, n: &Number, op: Op) -> Rational {
    let order = a.byte_order();
    if op == Op::Div && n.is_zero() {
        by_zero();
    }
    if let (Some(x), Some(y)) = (Small::of(a), Small::integer(n)) {
        let w = small_op(x, y, op);
        return w.rational(order);
    }
    let (num, den) = general(parts(a), integer(n), op, order);
    Rational::from_reduced(num, den)
}

/// `a = a op n` for an integer `n`, in the vectors of `a` where they have room.
#[inline]
#[track_caller]
pub(super) fn apply_int_left(a: &mut Rational, n: &Number, op: Op) {
    let order = a.byte_order();
    if op == Op::Div && n.is_zero() {
        by_zero();
    }
    if let (Some(x), Some(y)) = (Small::of(a), Small::integer(n)) {
        let w = small_op(x, y, op);
        w.write(a, order);
        return;
    }
    general_into(a, integer(n), true, op, order);
}

/// `n op a` for an integer `n`, in a new rational in the byte order of `n`.
#[inline]
#[track_caller]
pub(super) fn apply_int_right(n: &Number, a: &Rational, op: Op) -> Rational {
    let order = n.byte_order();
    if op == Op::Div && a.is_zero() {
        by_zero();
    }
    if let (Some(x), Some(y)) = (Small::integer(n), Small::of(a)) {
        let w = small_op(x, y, op);
        return w.rational(order);
    }
    let (num, den) = general(integer(n), parts(a), op, order);
    Rational::from_reduced(num, den)
}

/// `a = n op a` for an integer `n`, in the vectors of `a` where they have room, in the byte order
/// of `n`.
#[inline]
#[track_caller]
pub(super) fn apply_int_right_into(n: &Number, a: &mut Rational, op: Op) {
    let order = n.byte_order();
    if op == Op::Div && a.is_zero() {
        by_zero();
    }
    if let (Some(x), Some(y)) = (Small::integer(n), Small::of(a)) {
        let w = small_op(x, y, op);
        w.write(a, order);
        return;
    }
    general_into(a, integer(n), false, op, order);
}

/// The operators between two rationals.
macro_rules! rational_ops {
    ($($trait:ident $method:ident $assign:ident $assign_method:ident $op:ident),*) => {$(
        impl $trait<&Rational> for &Rational {
            type Output = Rational;
            #[inline]
            #[track_caller]
            fn $method(self, rhs: &Rational) -> Rational {
                apply(self, rhs, Op::$op)
            }
        }
        impl $trait<&Rational> for Rational {
            type Output = Rational;
            #[inline]
            #[track_caller]
            fn $method(mut self, rhs: &Rational) -> Rational {
                apply_left(&mut self, rhs, Op::$op);
                self
            }
        }
        impl $trait<Rational> for &Rational {
            type Output = Rational;
            #[inline]
            #[track_caller]
            fn $method(self, mut rhs: Rational) -> Rational {
                apply_right(self, &mut rhs, Op::$op);
                rhs
            }
        }
        impl $trait<Rational> for Rational {
            type Output = Rational;
            #[inline]
            #[track_caller]
            fn $method(mut self, rhs: Rational) -> Rational {
                apply_left(&mut self, &rhs, Op::$op);
                self
            }
        }
        impl $assign<&Rational> for Rational {
            #[inline]
            #[track_caller]
            fn $assign_method(&mut self, rhs: &Rational) {
                apply_left(self, rhs, Op::$op);
            }
        }
        impl $assign<Rational> for Rational {
            #[inline]
            #[track_caller]
            fn $assign_method(&mut self, rhs: Rational) {
                apply_left(self, &rhs, Op::$op);
            }
        }

        impl $trait<&Number> for &Rational {
            type Output = Rational;
            #[inline]
            #[track_caller]
            fn $method(self, rhs: &Number) -> Rational {
                apply_int(self, rhs, Op::$op)
            }
        }
        impl $trait<Number> for &Rational {
            type Output = Rational;
            #[inline]
            #[track_caller]
            fn $method(self, rhs: Number) -> Rational {
                apply_int(self, &rhs, Op::$op)
            }
        }
        impl $trait<&Number> for Rational {
            type Output = Rational;
            #[inline]
            #[track_caller]
            fn $method(mut self, rhs: &Number) -> Rational {
                apply_int_left(&mut self, rhs, Op::$op);
                self
            }
        }
        impl $trait<Number> for Rational {
            type Output = Rational;
            #[inline]
            #[track_caller]
            fn $method(mut self, rhs: Number) -> Rational {
                apply_int_left(&mut self, &rhs, Op::$op);
                self
            }
        }
        impl $assign<&Number> for Rational {
            #[inline]
            #[track_caller]
            fn $assign_method(&mut self, rhs: &Number) {
                apply_int_left(self, rhs, Op::$op);
            }
        }
        impl $assign<Number> for Rational {
            #[inline]
            #[track_caller]
            fn $assign_method(&mut self, rhs: Number) {
                apply_int_left(self, &rhs, Op::$op);
            }
        }
        impl $trait<&Rational> for &Number {
            type Output = Rational;
            #[inline]
            #[track_caller]
            fn $method(self, rhs: &Rational) -> Rational {
                apply_int_right(self, rhs, Op::$op)
            }
        }
        impl $trait<Rational> for &Number {
            type Output = Rational;
            #[inline]
            #[track_caller]
            fn $method(self, mut rhs: Rational) -> Rational {
                apply_int_right_into(self, &mut rhs, Op::$op);
                rhs
            }
        }
        impl $trait<&Rational> for Number {
            type Output = Rational;
            #[inline]
            #[track_caller]
            fn $method(self, rhs: &Rational) -> Rational {
                apply_int_right(&self, rhs, Op::$op)
            }
        }
        impl $trait<Rational> for Number {
            type Output = Rational;
            #[inline]
            #[track_caller]
            fn $method(self, mut rhs: Rational) -> Rational {
                apply_int_right_into(&self, &mut rhs, Op::$op);
                rhs
            }
        }
    )*};
}

rational_ops! {
    Add add AddAssign add_assign Add,
    Sub sub SubAssign sub_assign Sub,
    Mul mul MulAssign mul_assign Mul,
    Div div DivAssign div_assign Div
}

/// The sign and magnitude of a primitive integer.
pub(super) trait Primitive: Copy {
    /// The sign and the magnitude.
    fn parts(self) -> (bool, u128);
}

macro_rules! unsigned_primitive {
    ($($t:ty),*) => {$(
        impl Primitive for $t {
            #[inline(always)]
            fn parts(self) -> (bool, u128) {
                (false, self as u128)
            }
        }
    )*};
}

macro_rules! signed_primitive {
    ($($t:ty),*) => {$(
        impl Primitive for $t {
            #[inline(always)]
            fn parts(self) -> (bool, u128) {
                (self < 0, self.unsigned_abs() as u128)
            }
        }
    )*};
}

unsigned_primitive!(u8, u16, u32, u64, u128, usize);
signed_primitive!(i8, i16, i32, i64, i128, isize);

/// A primitive integer as a number of the magnitude `m` and the sign `negative`, little-endian.
#[inline]
pub(super) fn primitive_number(negative: bool, m: u128) -> Number {
    small::number(ByteOrder::LittleEndian, negative, m)
}

/// `a op v` for a primitive integer `v`, in a new rational.
#[inline(always)]
#[track_caller]
fn prim_op<T: Primitive>(a: &Rational, v: T, op: Op) -> Rational {
    let (negative, m) = v.parts();
    if op == Op::Div && m == 0 {
        by_zero();
    }
    if m >> 64 == 0
        && let Some(x) = Small::of(a)
    {
        let w = small_op(
            x,
            Small {
                negative: negative && m != 0,
                num: m as u64,
                den: 1,
            },
            op,
        );
        return w.rational(a.byte_order());
    }
    apply_int(a, &primitive_number(negative, m), op)
}

/// `a = a op v` for a primitive integer `v`.
#[inline(always)]
#[track_caller]
fn prim_op_left<T: Primitive>(a: &mut Rational, v: T, op: Op) {
    let (negative, m) = v.parts();
    if op == Op::Div && m == 0 {
        by_zero();
    }
    if m >> 64 == 0
        && let Some(x) = Small::of(a)
    {
        let w = small_op(
            x,
            Small {
                negative: negative && m != 0,
                num: m as u64,
                den: 1,
            },
            op,
        );
        let order = a.byte_order();
        w.write(a, order);
        return;
    }
    apply_int_left(a, &primitive_number(negative, m), op);
}

/// `v op a` for a primitive integer `v`, in the byte order of `a`.
#[inline(always)]
#[track_caller]
fn prim_op_right<T: Primitive>(v: T, a: &Rational, op: Op) -> Rational {
    let (negative, m) = v.parts();
    if op == Op::Div && a.is_zero() {
        by_zero();
    }
    let order = a.byte_order();
    if m >> 64 == 0
        && let Some(y) = Small::of(a)
    {
        let w = small_op(
            Small {
                negative: negative && m != 0,
                num: m as u64,
                den: 1,
            },
            y,
            op,
        );
        return w.rational(order);
    }
    apply_int_right(&small::number(order, negative, m), a, op)
}

/// `a = v op a` for a primitive integer `v`, in the byte order of `a`.
#[inline(always)]
#[track_caller]
fn prim_op_right_into<T: Primitive>(v: T, a: &mut Rational, op: Op) {
    let (negative, m) = v.parts();
    if op == Op::Div && a.is_zero() {
        by_zero();
    }
    let order = a.byte_order();
    if m >> 64 == 0
        && let Some(y) = Small::of(a)
    {
        let w = small_op(
            Small {
                negative: negative && m != 0,
                num: m as u64,
                den: 1,
            },
            y,
            op,
        );
        w.write(a, order);
        return;
    }
    apply_int_right_into(&small::number(order, negative, m), a, op);
}

/// The operators between rationals and primitive integers, on either side.
macro_rules! primitive_ops {
    ($($t:ty),*) => {$(
        primitive_ops!(@ops $t;
            Add add AddAssign add_assign Add,
            Sub sub SubAssign sub_assign Sub,
            Mul mul MulAssign mul_assign Mul,
            Div div DivAssign div_assign Div);
    )*};
    (@ops $t:ty; $($trait:ident $method:ident $assign:ident $assign_method:ident $op:ident),*) => {$(
        impl $trait<$t> for &Rational {
            type Output = Rational;
            #[inline]
            #[track_caller]
            fn $method(self, rhs: $t) -> Rational {
                prim_op(self, rhs, Op::$op)
            }
        }
        impl $trait<$t> for Rational {
            type Output = Rational;
            #[inline]
            #[track_caller]
            fn $method(mut self, rhs: $t) -> Rational {
                prim_op_left(&mut self, rhs, Op::$op);
                self
            }
        }
        impl $assign<$t> for Rational {
            #[inline]
            #[track_caller]
            fn $assign_method(&mut self, rhs: $t) {
                prim_op_left(self, rhs, Op::$op);
            }
        }
        impl $trait<&Rational> for $t {
            type Output = Rational;
            #[inline]
            #[track_caller]
            fn $method(self, rhs: &Rational) -> Rational {
                prim_op_right(self, rhs, Op::$op)
            }
        }
        impl $trait<Rational> for $t {
            type Output = Rational;
            #[inline]
            #[track_caller]
            fn $method(self, mut rhs: Rational) -> Rational {
                prim_op_right_into(self, &mut rhs, Op::$op);
                rhs
            }
        }
    )*};
}

primitive_ops!(
    u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize
);

impl Neg for Rational {
    type Output = Rational;
    /// `-x`, in the words or the vectors of `x`.
    #[inline]
    fn neg(mut self) -> Rational {
        self.negate();
        self
    }
}

impl Neg for &Rational {
    type Output = Rational;
    /// `-x`, in new words or vectors.
    #[inline]
    fn neg(self) -> Rational {
        if let Some(x) = self.word() {
            return Rational::from_word(!x.negative && x.num != 0, x.num, x.den, self.byte_order());
        }
        -self.clone()
    }
}
