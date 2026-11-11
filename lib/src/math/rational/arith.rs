//! The arithmetic operators of rationals: `+`, `-`, `*` and `/` and their assignments, with a
//! rational, a [`Number`] or a primitive integer on either side, each operand owned or borrowed,
//! and `-` to negate.
//!
//! Each takes the register path of [`small`](super::small) when every part fits in 64 bits, and
//! otherwise reduces as Knuth's section 4.5.1 does, on numbers. The result is in the byte order
//! of the left operand; an owned operand's vectors take it when they have room.

use super::small::{self, Small, Wide};
use super::{Rational, in_order, is_one, one};
use crate::math::numbers::{ByteOrder, Number};
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// The operation of an operator.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Op {
    Add,
    Sub,
    Mul,
    Div,
}

/// `x op y` on the register path, or `None` when the numerator of a sum does not fit in 128 bits.
#[inline(always)]
fn small_op(x: Small, y: Small, op: Op) -> Option<Wide> {
    match op {
        Op::Add => small::add(x, y, false),
        Op::Sub => small::add(x, y, true),
        Op::Mul => Some(small::mul(x, y, false)),
        Op::Div => Some(small::mul(x, y, true)),
    }
}

/// Panics as an integer division by zero does.
#[cold]
#[inline(never)]
#[track_caller]
fn by_zero() -> ! {
    panic!("attempt to divide by zero")
}

/// `x op y` for fractions given by their parts, each in lowest terms with a positive denominator,
/// on numbers: the numerator and the denominator of the result, in lowest terms, in the byte order
/// `order`.
#[track_caller]
fn general(
    xn: &Number,
    xd: &Number,
    yn: &Number,
    yd: &Number,
    op: Op,
    order: ByteOrder,
) -> (Number, Number) {
    let (num, den) = match op {
        Op::Add | Op::Sub => sum(xn, xd, yn, yd, op == Op::Sub),
        Op::Mul => product(xn, xd, yn, yd, yn.is_negative()),
        Op::Div => {
            if yn.is_zero() {
                by_zero();
            }
            // `(xn/xd) / (yn/yd) = (xn/xd) · (yd/|yn|)`, of the sign of `xn·yn`.
            product(xn, xd, yd, yn, yn.is_negative())
        }
    };
    finish(num, den, order)
}

/// `x ± y`, Knuth's way: the numerator and a positive denominator, in lowest terms, the
/// numerator in the byte order of `xn` or `xd`.
fn sum(xn: &Number, xd: &Number, yn: &Number, yd: &Number, subtract: bool) -> (Number, Number) {
    let combine = |a: &Number, b: &Number| if subtract { a - b } else { a + b };
    let (x1, y1) = (is_one(xd), is_one(yd));
    if x1 && y1 {
        return (combine(xn, yn), one(xd.byte_order()));
    }
    if y1 {
        // `(xn ± yn·xd)/xd`, prime to `xd` as `xn` is.
        return (combine(xn, &(yn * xd)), xd.clone());
    }
    if x1 {
        return (combine(&(xn * yd), yn), yd.clone());
    }
    if xd == yd {
        let num = combine(xn, yn);
        if num.is_zero() {
            return (num, one(xd.byte_order()));
        }
        let g = num.gcd(xd);
        if is_one(&g) {
            return (num, xd.clone());
        }
        return (num.div_exact(&g), xd.div_exact(&g));
    }
    let g = xd.gcd(yd);
    if is_one(&g) {
        let num = combine(&(xn * yd), &(yn * xd));
        return (num, xd * yd);
    }
    let s = xd.div_exact(&g);
    let t = yd.div_exact(&g);
    let num = combine(&(xn * &t), &(yn * &s));
    if num.is_zero() {
        return (num, one(xd.byte_order()));
    }
    let g2 = num.gcd(&g);
    if is_one(&g2) {
        (num, s * yd)
    } else {
        (num.div_exact(&g2), s * yd.div_exact(&g2))
    }
}

/// `(xn/xd) · (yn/yd)` for `yd` positive and `yn` of the sign `negative_y` (the parts of a
/// quotient are given turned around, so `yn` is then positive and the sign is that of the
/// divisor): the numerator and a positive denominator, in lowest terms.
fn product(
    xn: &Number,
    xd: &Number,
    yn: &Number,
    yd: &Number,
    negative_y: bool,
) -> (Number, Number) {
    let negative = xn.is_negative() != negative_y;
    if xn.is_zero() || yn.is_zero() {
        return (Number::default(), one(xd.byte_order()));
    }
    // `gcd(xn, yd)` and `gcd(yn, xd)`, each 1 without a division when its denominator is.
    let g1 = (!is_one(yd)).then(|| xn.gcd(yd)).filter(|g| !is_one(g));
    let g2 = (!is_one(xd)).then(|| yn.gcd(xd)).filter(|g| !is_one(g));
    let a = match &g1 {
        Some(g) => xn.div_exact(g),
        None => xn.clone(),
    };
    let num = match &g2 {
        Some(g) => a * &yn.div_exact(g),
        None => a * yn,
    };
    let b = match &g2 {
        Some(g) => xd.div_exact(g),
        None => xd.clone(),
    };
    let mut den = match &g1 {
        Some(g) => b * &yd.div_exact(g),
        None => b * yd,
    };
    let mut num = num;
    num.negative = negative && !num.is_zero();
    den.negative = false;
    (num, den)
}

/// The parts in the byte order `order`, a zero numerator over 1.
#[inline]
fn finish(mut num: Number, den: Number, order: ByteOrder) -> (Number, Number) {
    super::trim(&mut num);
    let mut den = in_order(den, order);
    super::trim(&mut den);
    (in_order(num, order), den)
}

/// `a op b` in a new rational, in the byte order of `a`.
#[inline]
#[track_caller]
pub(super) fn apply(a: &Rational, b: &Rational, op: Op) -> Rational {
    let order = a.byte_order();
    if op == Op::Div && b.is_zero() {
        by_zero();
    }
    if let (Some(x), Some(y)) = (Small::of(a), Small::of(b))
        && let Some(w) = small_op(x, y, op)
    {
        return w.rational(order);
    }
    let (num, den) = general(&a.num, &a.den, &b.num, &b.den, op, order);
    Rational::from_reduced(num, den)
}

/// `a = a op b`, in the byte order of `a`, in its vectors where they have room.
#[inline]
#[track_caller]
pub(super) fn apply_left(a: &mut Rational, b: &Rational, op: Op) {
    let order = a.byte_order();
    if op == Op::Div && b.is_zero() {
        by_zero();
    }
    if let (Some(x), Some(y)) = (Small::of(a), Small::of(b))
        && let Some(w) = small_op(x, y, op)
    {
        w.write(a, order);
        return;
    }
    let (num, den) = general(&a.num, &a.den, &b.num, &b.den, op, order);
    *a = Rational::from_reduced(num, den);
}

/// `b = a op b`, in the byte order of `a`, in the vectors of `b` where they have room.
#[inline]
#[track_caller]
pub(super) fn apply_right(a: &Rational, b: &mut Rational, op: Op) {
    let order = a.byte_order();
    if op == Op::Div && b.is_zero() {
        by_zero();
    }
    if let (Some(x), Some(y)) = (Small::of(a), Small::of(b))
        && let Some(w) = small_op(x, y, op)
    {
        w.write(b, order);
        return;
    }
    let (num, den) = general(&a.num, &a.den, &b.num, &b.den, op, order);
    *b = Rational::from_reduced(num, den);
}

/// `a op n` for an integer `n`, in a new rational in the byte order of `a`.
#[inline]
#[track_caller]
pub(super) fn apply_int(a: &Rational, n: &Number, op: Op) -> Rational {
    let order = a.byte_order();
    if op == Op::Div && n.is_zero() {
        by_zero();
    }
    if let (Some(x), Some(y)) = (Small::of(a), Small::integer(n))
        && let Some(w) = small_op(x, y, op)
    {
        return w.rational(order);
    }
    let (num, den) = general(&a.num, &a.den, n, &one(n.byte_order()), op, order);
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
    if let (Some(x), Some(y)) = (Small::of(a), Small::integer(n))
        && let Some(w) = small_op(x, y, op)
    {
        w.write(a, order);
        return;
    }
    let (num, den) = general(&a.num, &a.den, n, &one(n.byte_order()), op, order);
    *a = Rational::from_reduced(num, den);
}

/// `n op a` for an integer `n`, in a new rational in the byte order of `n`.
#[inline]
#[track_caller]
pub(super) fn apply_int_right(n: &Number, a: &Rational, op: Op) -> Rational {
    let order = n.byte_order();
    if op == Op::Div && a.is_zero() {
        by_zero();
    }
    if let (Some(x), Some(y)) = (Small::integer(n), Small::of(a))
        && let Some(w) = small_op(x, y, op)
    {
        return w.rational(order);
    }
    let (num, den) = general(n, &one(order), &a.num, &a.den, op, order);
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
    if let (Some(x), Some(y)) = (Small::integer(n), Small::of(a))
        && let Some(w) = small_op(x, y, op)
    {
        w.write(a, order);
        return;
    }
    let (num, den) = general(n, &one(order), &a.num, &a.den, op, order);
    *a = Rational::from_reduced(num, den);
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
#[inline]
#[track_caller]
fn prim_op<T: Primitive>(a: &Rational, v: T, op: Op) -> Rational {
    let (negative, m) = v.parts();
    if op == Op::Div && m == 0 {
        by_zero();
    }
    if m >> 64 == 0
        && let Some(x) = Small::of(a)
        && let Some(w) = small_op(
            x,
            Small {
                negative: negative && m != 0,
                num: m as u64,
                den: 1,
            },
            op,
        )
    {
        return w.rational(a.byte_order());
    }
    apply_int(a, &primitive_number(negative, m), op)
}

/// `a = a op v` for a primitive integer `v`.
#[inline]
#[track_caller]
fn prim_op_left<T: Primitive>(a: &mut Rational, v: T, op: Op) {
    let (negative, m) = v.parts();
    if op == Op::Div && m == 0 {
        by_zero();
    }
    if m >> 64 == 0
        && let Some(x) = Small::of(a)
        && let Some(w) = small_op(
            x,
            Small {
                negative: negative && m != 0,
                num: m as u64,
                den: 1,
            },
            op,
        )
    {
        let order = a.byte_order();
        w.write(a, order);
        return;
    }
    apply_int_left(a, &primitive_number(negative, m), op);
}

/// `v op a` for a primitive integer `v`, in the byte order of `a`.
#[inline]
#[track_caller]
fn prim_op_right<T: Primitive>(v: T, a: &Rational, op: Op) -> Rational {
    let (negative, m) = v.parts();
    if op == Op::Div && a.is_zero() {
        by_zero();
    }
    let order = a.byte_order();
    if m >> 64 == 0
        && let Some(y) = Small::of(a)
        && let Some(w) = small_op(
            Small {
                negative: negative && m != 0,
                num: m as u64,
                den: 1,
            },
            y,
            op,
        )
    {
        return w.rational(order);
    }
    apply_int_right(&small::number(order, negative, m), a, op)
}

/// `a = v op a` for a primitive integer `v`, in the byte order of `a`.
#[inline]
#[track_caller]
fn prim_op_right_into<T: Primitive>(v: T, a: &mut Rational, op: Op) {
    let (negative, m) = v.parts();
    if op == Op::Div && a.is_zero() {
        by_zero();
    }
    let order = a.byte_order();
    if m >> 64 == 0
        && let Some(y) = Small::of(a)
        && let Some(w) = small_op(
            Small {
                negative: negative && m != 0,
                num: m as u64,
                den: 1,
            },
            y,
            op,
        )
    {
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
    /// `-x`, in the vectors of `x`.
    #[inline]
    fn neg(mut self) -> Rational {
        self.num.negative = !self.num.negative && !self.num.is_zero();
        self
    }
}

impl Neg for &Rational {
    type Output = Rational;
    /// `-x`, in new vectors.
    #[inline]
    fn neg(self) -> Rational {
        -self.clone()
    }
}
