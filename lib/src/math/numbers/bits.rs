//! Bitwise operations on [`Number`]s: `&`, `|`, `^`, `&=`, `|=` and `^=`, with either operand
//! owned or borrowed, `!`, and the bits of a number: [`Number::bits`], [`Number::count_ones`],
//! [`Number::trailing_zeros`], [`Number::bit`] and [`Number::set_bit`].
//!
//! The operators work as if both operands were in two's complement with infinitely many bits, as
//! those of `i32` do on 32 and GMP's `mpz_and`, `mpz_ior`, `mpz_xor` and `mpz_com` on any number:
//! a negative number has infinitely many ones above its magnitude's bits, `-x` being `!x + 1`. The
//! result is stored back as a sign and a magnitude. With `X` and `Y` the magnitudes, a negative
//! operand is `!(X - 1)` and a negative result `!(R - 1)`, so every combination of signs comes down
//! to one operation `g` on the bytes of `X` or `X - 1` and `Y` or `Y - 1`, plus one when the result
//! is negative:
//!
//! | | `x & y` | `x \| y` | `x ^ y` |
//! | --- | --- | --- | --- |
//! | `+ +` | `X & Y` | `X \| Y` | `X ^ Y` |
//! | `+ -` | `X & !(Y - 1)` | `-((Y - 1) & !X) - 1` | `-(X ^ (Y - 1)) - 1` |
//! | `- +` | `!(X - 1) & Y` | `-((X - 1) & !Y) - 1` | `-((X - 1) ^ Y) - 1` |
//! | `- -` | `-((X - 1) \| (Y - 1)) - 1` | `-((X - 1) & (Y - 1)) - 1` | `(X - 1) ^ (Y - 1)` |
//!
//! `!x` is `-x - 1`: `-(X + 1)` for `x >= 0`, and `X - 1` for `x < 0`.
//!
//! The ones taken from the magnitudes and the one added to the result are borrows and a carry
//! that run through the low bytes only: `X - 1` differs from `X` below its lowest nonzero byte
//! and at it, and the carry of `R + 1` runs through the bytes of `R` that are `0xFF`. So the
//! result is computed limb by limb (eight bytes, the least significant first) with the borrows
//! and the carry in flags, from the low end until all three are spent, which for random operands
//! is the first limb: the head. Above it, each byte of the result is `g` of the bytes of the
//! magnitudes at its position, with no carry from one byte to the next, so the rest is a loop
//! over the bytes in the order of memory, the same for both byte orders: NEON on aarch64, sixteen
//! bytes a step (`and`, `orr`, `eor` or `bic`), and words of eight bytes elsewhere. Above the
//! shorter operand, the result is the longer one's bytes or zero, by the operation.
//!
//! The paths, by the operands:
//!
//! - Both vectors of at most 16 bytes: in registers, as `u128`s in two's complement.
//! - Neither negative, in the same byte order: the operation, known at compile time, runs over
//!   the shorter magnitude's bytes with the kernel, with no borrow or carry, straight into the
//!   vector the result goes to; its length is the longer magnitude's for `|` and `^` of different
//!   lengths, and is otherwise read from the top down, one read for random operands: of the
//!   result, in place little-endian, and of the operands otherwise.
//! - Both of at most 64 bytes otherwise: in registers, as up to eight limbs each, as the
//!   addition's medium path reads them, with no branch on the signs or the length.
//! - Longer ones, in an operand's vector with room: the head of one limb, the length from the top
//!   of the operands (one read), one kernel call and a copy of the longer operand's top bytes.
//! - Otherwise (a new vector, a longer head, or a big-endian result of another length than the
//!   vector it is built in), everything about the result but its bytes is planned first, as for
//!   the other operators: the head, by a pass over its limbs that only reads them, and the exact
//!   length. Only then is a vector chosen, and the result written in it, each byte once; in place,
//!   the bytes the longer operand gives are left where they are, and a big-endian result that
//!   moves is written in the order that reads every byte of the operand before it is written over.

// The kernels of the loops over the bytes, under the name `kernels`: NEON on the 64-bit
// little-endian aarch64 targets whose operating system allows unaligned accesses (those of the
// addition's assembly), where NEON is enabled, as it is by default; the portable ones elsewhere and
// under Miri, which runs no SIMD of this kind.
#[cfg(all(
    target_arch = "aarch64",
    target_endian = "little",
    target_pointer_width = "64",
    target_feature = "neon",
    any(
        target_vendor = "apple",
        target_os = "linux",
        target_os = "android",
        target_os = "windows",
        target_os = "freebsd",
        target_os = "netbsd",
        target_os = "openbsd",
        target_os = "fuchsia"
    ),
    not(miri)
))]
#[path = "bits/aarch64.rs"]
mod kernels;
#[cfg(not(all(
    target_arch = "aarch64",
    target_endian = "little",
    target_pointer_width = "64",
    target_feature = "neon",
    any(
        target_vendor = "apple",
        target_os = "linux",
        target_os = "android",
        target_os = "windows",
        target_os = "freebsd",
        target_os = "netbsd",
        target_os = "openbsd",
        target_os = "fuchsia"
    ),
    not(miri)
)))]
#[path = "bits/portable.rs"]
mod kernels;

use super::add::{
    SMALL, long, medium, read, read_limbs, read_small, significant_len, small, top_of, write,
    write_small,
};
use super::{ByteOrder, Number};
use core::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};
use core::{mem, ptr};

// The operations on the bytes of two magnitudes `x` and `y` that the operators come down to.

/// `x & y`.
const AND: u8 = 0;
/// `x | y`.
const OR: u8 = 1;
/// `x ^ y`.
const XOR: u8 = 2;
/// `x & !y`.
const AND_NOT: u8 = 3;
/// `!x & y`.
const NOT_AND: u8 = 4;

/// The operation `g` on the words `x` and `y`.
#[inline(always)]
fn apply(g: u8, x: u64, y: u64) -> u64 {
    match g {
        AND => x & y,
        OR => x | y,
        XOR => x ^ y,
        AND_NOT => x & !y,
        _ => !x & y,
    }
}

/// A bitwise operator on two numbers.
#[derive(Clone, Copy)]
#[repr(u8)]
enum Op {
    And,
    Or,
    Xor,
}

/// The operation on the magnitudes and the sign of the result of `op` for operands with the signs
/// `nx` and `ny`, as in the table of the module: the result is `g(X', Y')`, plus one when it is
/// negative, with `X' = X - 1` for a negative `x` and `X` otherwise, and `Y'` likewise. Read from
/// a table, with no branch.
#[inline(always)]
fn select(op: Op, nx: bool, ny: bool) -> (u8, bool) {
    /// `g` in the low bits and the sign of the result in the top one, by the operator and the
    /// signs `nx ny` of the operands: `+ +`, `+ -`, `- +`, `- -`.
    const TABLE: [[u8; 4]; 3] = [
        [AND, AND_NOT, NOT_AND, OR | 0x80],
        [OR, NOT_AND | 0x80, AND_NOT | 0x80, AND | 0x80],
        [XOR, XOR | 0x80, XOR | 0x80, XOR],
    ];
    let t = TABLE[op as usize][usize::from(nx) * 2 + usize::from(ny)];
    (t & 0x7F, t & 0x80 != 0)
}

// The operators

/// Implements a bitwise operator: `$Op` with its method `$op` for each form of the operands, and
/// `$Assign` with `$assign`. `$sym` is the operator; the examples compute `12 $sym -10`, which is
/// `$value` (`$magnitude` and `$negative`), `0x0F_00 $sym 0x00_F0` (`$wide`, as stored
/// big-endian) and `0xFF $sym 0x01_00` (`$by_b`, stored as `$in_b`, in the vector of the right
/// operand when `$built_in_b`).
macro_rules! bitwise {
    (
        $Op:ident, $op:ident, $Assign:ident, $assign:ident, $kind:expr, $OP:expr, $sym:literal,
        $what:literal, $value:literal, $magnitude:literal, $negative:literal, $wide:literal,
        $by_b:literal, $in_b:literal, $built_in_b:literal
    ) => {
        impl $Op<&Number> for &Number {
            type Output = Number;

            #[doc = concat!("Returns `self ", $sym, " rhs`, the bitwise ", $what, " of the numbers in")]
            /// infinite two's complement, in the byte order of `self`, in the shortest form.
            ///
            /// The result is built in one new vector of exactly its length (one byte for zero);
            /// both operands are left as they were. The operands may have any signs, byte orders
            /// and lengths, and need not be in the shortest form.
            ///
            /// # Examples
            ///
            /// ```
            /// use a::math::numbers::{ByteOrder, Number};
            ///
            /// let a = Number::new_from_i32(12, ByteOrder::BigEndian);
            /// let b = Number::new_from_i32(-10, ByteOrder::LittleEndian);
            #[doc = concat!("let c = &a ", $sym, " &b;")]
            #[doc = concat!("// As for `i32`: `12 ", $sym, " -10` is ", $value, ".")]
            #[doc = concat!("assert_eq!(c.value, [", $magnitude, "]);")]
            #[doc = concat!("assert_eq!(c.is_negative(), ", $negative, ");")]
            /// assert!(c.byte_order() == ByteOrder::BigEndian);
            /// ```
            fn $op(self, rhs: &Number) -> Number {
                if small(self, rhs) {
                    // SAFETY: both vectors hold at most 16 bytes.
                    let s = unsafe { small_result(self, rhs, $kind) };
                    return small_number(self.byte_order, s);
                }
                if let Some(n) = plus_new::<{ $OP }>(self, rhs) {
                    return n;
                }
                if medium(self, rhs) {
                    // SAFETY: both vectors hold at most 64 bytes.
                    let m = unsafe { medium_result::<{ $OP }>(self, rhs) };
                    return medium_number(self.byte_order, &m);
                }
                ref_ref(self, rhs, $kind)
            }
        }

        impl $Op<&Number> for Number {
            type Output = Number;

            #[doc = concat!("Returns `self ", $sym, " rhs`, in the byte order of `self`, in the shortest")]
            #[doc = concat!("form, as `self ", $sym, "= rhs` does.")]
            ///
            /// The result is built in the vector of `self` when its capacity has room for it,
            /// which allocates nothing, and otherwise in one new vector of exactly its length.
            /// `rhs` is left as it was.
            ///
            /// # Examples
            ///
            /// ```
            /// use a::math::numbers::{ByteOrder, Number};
            ///
            /// let mut value = Vec::with_capacity(4);
            /// value.extend([0x0F, 0x00]);
            /// let a = Number::new(value, ByteOrder::BigEndian);
            /// let pointer = a.value.as_ptr();
            ///
            #[doc = concat!("let c = a ", $sym, " &Number::new_from_u8(0xF0, ByteOrder::LittleEndian);")]
            #[doc = concat!("assert_eq!(c.value, [", $wide, "]);")]
            /// assert_eq!(c.value.as_ptr(), pointer);
            /// ```
            fn $op(mut self, rhs: &Number) -> Number {
                $Assign::$assign(&mut self, rhs);
                self
            }
        }

        impl $Op<Number> for &Number {
            type Output = Number;

            #[doc = concat!("Returns `self ", $sym, " rhs`, in the byte order of `self`, in the shortest")]
            /// form.
            ///
            /// The result is built in the vector of `rhs` when its capacity has room for it, also
            /// when `rhs` is in the other byte order, and otherwise in one new vector of exactly
            /// its length. `self` is left as it was.
            ///
            /// # Examples
            ///
            /// ```
            /// use a::math::numbers::{ByteOrder, Number};
            ///
            /// let a = Number::new_from_u16(0x0F00, ByteOrder::BigEndian);
            /// let mut value = Vec::with_capacity(4);
            /// value.extend([0xF0]);
            /// let b = Number::new(value, ByteOrder::LittleEndian);
            /// let pointer = b.value.as_ptr();
            ///
            #[doc = concat!("let c = &a ", $sym, " b;")]
            #[doc = concat!("assert_eq!(c.value, [", $wide, "]);")]
            /// assert_eq!(c.value.as_ptr(), pointer);
            /// assert!(c.byte_order() == ByteOrder::BigEndian);
            /// ```
            fn $op(self, mut rhs: Number) -> Number {
                if small(self, &rhs) {
                    // SAFETY: both vectors hold at most 16 bytes.
                    let s = unsafe { small_result(self, &rhs, $kind) };
                    let order = self.byte_order;
                    put_small(&mut rhs.value, order, s);
                    return Number {
                        value: rhs.value,
                        negative: s.negative,
                        byte_order: order,
                    };
                }
                // The operations are symmetric: `rhs op self` over `rhs`.
                if plus_over::<{ $OP }>(&mut rhs, self) {
                    return rhs;
                }
                if medium(self, &rhs) {
                    // SAFETY: both vectors hold at most 64 bytes.
                    let m = unsafe { medium_result::<{ $OP }>(self, &rhs) };
                    let order = self.byte_order;
                    put_medium(&mut rhs.value, order, &m);
                    return Number {
                        value: rhs.value,
                        negative: m.negative,
                        byte_order: order,
                    };
                }
                // `rhs` is dropped on return, with the vector it holds then, if any.
                ref_value(self, &mut rhs, $kind)
            }
        }

        impl $Op<Number> for Number {
            type Output = Number;

            #[doc = concat!("Returns `self ", $sym, " rhs`, in the byte order of `self`, in the shortest")]
            #[doc = concat!("form, as `self ", $sym, "= rhs` does.")]
            ///
            /// The result is built in the vector of `self` when its capacity has room for it, or
            /// else in the vector of `rhs` when that has room, also when `rhs` is in the other
            /// byte order, either of which allocates nothing; otherwise it gets one new vector of
            /// exactly its length.
            ///
            /// # Examples
            ///
            /// ```
            /// use a::math::numbers::{ByteOrder, Number};
            ///
            /// let a = Number::new_from_i32(12, ByteOrder::LittleEndian);
            /// let b = Number::new_from_i32(-10, ByteOrder::BigEndian);
            #[doc = concat!("let c = a ", $sym, " b;")]
            #[doc = concat!("assert_eq!(c.value, [", $magnitude, "]);")]
            #[doc = concat!("assert_eq!(c.is_negative(), ", $negative, ");")]
            /// ```
            fn $op(mut self, rhs: Number) -> Number {
                $Assign::$assign(&mut self, rhs);
                self
            }
        }

        impl $Assign<&Number> for Number {
            #[doc = concat!("Sets `self` to `self ", $sym, " rhs`, keeping its byte order, in the shortest form.")]
            ///
            /// The result is built in the vector of `self` when its capacity has room for it,
            /// which allocates nothing, and otherwise in one new vector of exactly its length,
            /// which replaces it. `rhs` is left as it was.
            ///
            /// # Examples
            ///
            /// ```
            /// use a::math::numbers::{ByteOrder, Number};
            ///
            /// let mut n = Number::new_from_i32(12, ByteOrder::LittleEndian);
            #[doc = concat!("n ", $sym, "= &Number::new_from_i32(-10, ByteOrder::BigEndian);")]
            #[doc = concat!("assert_eq!(n.value, [", $magnitude, "]);")]
            #[doc = concat!("assert_eq!(n.is_negative(), ", $negative, ");")]
            /// ```
            #[inline]
            fn $assign(&mut self, rhs: &Number) {
                if small(self, rhs) {
                    // SAFETY: both vectors hold at most 16 bytes.
                    let s = unsafe { small_result(self, rhs, $kind) };
                    put_small(&mut self.value, self.byte_order, s);
                    self.negative = s.negative;
                    return;
                }
                if plus_over::<{ $OP }>(self, rhs) {
                    return;
                }
                if medium(self, rhs) {
                    // SAFETY: both vectors hold at most 64 bytes.
                    let m = unsafe { medium_result::<{ $OP }>(self, rhs) };
                    put_medium(&mut self.value, self.byte_order, &m);
                    self.negative = m.negative;
                    return;
                }
                assign_ref(self, rhs, $kind);
            }
        }

        impl $Assign<Number> for Number {
            #[doc = concat!("Sets `self` to `self ", $sym, " rhs`, keeping its byte order, in the shortest form.")]
            ///
            /// The result is built in the vector of `self` when its capacity has room for it, or
            /// else in the vector of `rhs` when that has room, also when `rhs` is in the other
            /// byte order, either of which allocates nothing; otherwise it gets one new vector of
            /// exactly its length.
            ///
            /// # Examples
            ///
            /// ```
            /// use a::math::numbers::{ByteOrder, Number};
            ///
            /// // `n` has room for one byte only; the vector of the right operand has more.
            /// let mut value = Vec::with_capacity(1);
            /// value.push(0xFF);
            /// let mut n = Number::new(value, ByteOrder::BigEndian);
            /// let mut value = Vec::with_capacity(4);
            /// value.extend([0x00, 0x01]);
            /// let b = Number::new(value, ByteOrder::LittleEndian);
            /// let pointer = b.value.as_ptr();
            #[doc = concat!("n ", $sym, "= b;")]
            #[doc = concat!("// `0xFF ", $sym, " 0x01_00` is ", $by_b, ".")]
            #[doc = concat!("assert_eq!(n.value, [", $in_b, "]);")]
            #[doc = concat!("assert_eq!(n.value.as_ptr() == pointer, ", $built_in_b, ");")]
            /// ```
            #[inline]
            fn $assign(&mut self, mut rhs: Number) {
                if small(self, &rhs) {
                    // SAFETY: both vectors hold at most 16 bytes.
                    let s = unsafe { small_result(self, &rhs, $kind) };
                    if self.value.capacity() < s.len && rhs.value.capacity() >= s.len {
                        // The vector of `self` goes to `rhs`, and is dropped with it.
                        mem::swap(&mut self.value, &mut rhs.value);
                    }
                    put_small(&mut self.value, self.byte_order, s);
                    self.negative = s.negative;
                    return;
                }
                if plus_over::<{ $OP }>(self, &rhs) {
                    return;
                }
                if medium(self, &rhs) {
                    // SAFETY: both vectors hold at most 64 bytes.
                    let m = unsafe { medium_result::<{ $OP }>(self, &rhs) };
                    if self.value.capacity() < m.len && rhs.value.capacity() >= m.len {
                        // The vector of `self` goes to `rhs`, and is dropped with it.
                        mem::swap(&mut self.value, &mut rhs.value);
                    }
                    put_medium(&mut self.value, self.byte_order, &m);
                    self.negative = m.negative;
                    return;
                }
                // `rhs` is dropped on return, with the vector it holds then.
                assign_value(self, &mut rhs, $kind);
            }
        }
    };
}

bitwise!(
    BitAnd,
    bitand,
    BitAndAssign,
    bitand_assign,
    Op::And,
    AND,
    "&",
    "and",
    "4",
    "4",
    "false",
    "0",
    "0",
    "0",
    "false"
);
bitwise!(
    BitOr,
    bitor,
    BitOrAssign,
    bitor_assign,
    Op::Or,
    OR,
    "|",
    "or",
    "-2",
    "2",
    "true",
    "0x0F, 0xF0",
    "0x01_FF",
    "0x01, 0xFF",
    "true"
);
bitwise!(
    BitXor,
    bitxor,
    BitXorAssign,
    bitxor_assign,
    Op::Xor,
    XOR,
    "^",
    "exclusive or",
    "-6",
    "6",
    "true",
    "0x0F, 0xF0",
    "0x01_FF",
    "0x01, 0xFF",
    "true"
);

impl Not for Number {
    type Output = Number;

    /// Returns `!self`, every bit of the number in infinite two's complement inverted, which is
    /// `-self - 1`, in the byte order of `self`, in the shortest form.
    ///
    /// The result is built in the vector of `self` when its capacity has room for it, which
    /// allocates nothing, and otherwise in one new vector of exactly its length; it is at most one
    /// byte longer. Only the bytes the carry of `+ 1` (or the borrow of `- 1`) passes through
    /// are written, which for random numbers is the lowest one: a number in the shortest form
    /// takes constant time.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// // As for `i32`: `!0xFF` is `-0x100`, and `!-0x100` is `0xFF`.
    /// let n = !Number::new_from_i32(0xFF, ByteOrder::BigEndian);
    /// assert_eq!(n.value, [0x01, 0x00]);
    /// assert!(n.is_negative());
    /// let n = !n;
    /// assert_eq!(n.value, [0xFF]);
    /// assert!(!n.is_negative());
    /// ```
    #[inline]
    fn not(mut self) -> Number {
        not_in_place(&mut self);
        self
    }
}

impl Not for &Number {
    type Output = Number;

    /// Returns `!self`, which is `-self - 1`, in the byte order of `self`, in the shortest form.
    ///
    /// The result is built in one new vector of exactly its length; `self` is left as it was.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// let a = Number::new_from_i32(-1, ByteOrder::LittleEndian);
    /// let n = !&a;
    /// assert_eq!(n.value, [0]);
    /// assert!(!n.is_negative());
    /// ```
    #[inline]
    fn not(self) -> Number {
        not_new(self)
    }
}

// Small numbers

/// The result of a bitwise operation on two small operands, computed in registers.
#[derive(Clone, Copy)]
struct Small {
    /// The low eight bytes of the magnitude.
    lo: u64,
    /// The next eight bytes of the magnitude.
    hi: u64,
    /// The length of the result in the shortest form, from 1 to 17: one byte for zero, and 17 for
    /// a magnitude of `2^128`, whose seventeenth byte is 1 and the others 0.
    len: usize,
    /// Whether the result is negative; never for zero.
    negative: bool,
}

/// The magnitude `m` with the sign `negative`, or `2^128` when `top` is set, as a [`Small`].
#[inline(always)]
fn small_of(m: u128, top: bool, negative: bool) -> Small {
    Small {
        lo: m as u64,
        hi: (m >> 64) as u64,
        len: if top {
            SMALL + 1
        } else {
            SMALL - (m | 1).leading_zeros() as usize / 8
        },
        negative,
    }
}

/// `a op b` for small operands, in two's complement on `u128`s: a negative magnitude `x` is
/// `-x` there, with infinitely many ones above, and a negative result `r` is the magnitude
/// `2^128 - r`, which is `2^128` itself, 17 bytes, for `r = 0`.
///
/// # Safety
///
/// Both vectors must hold at most [`SMALL`] bytes.
#[inline(always)]
unsafe fn small_result(a: &Number, b: &Number, op: Op) -> Small {
    // SAFETY: as the caller guarantees.
    let (x, y) = unsafe {
        (
            read_small(&a.value, a.byte_order),
            read_small(&b.value, b.byte_order),
        )
    };
    // A magnitude of zero is zero, whatever the sign says.
    let (nx, ny) = (a.negative && x != 0, b.negative && y != 0);
    let tx = if nx { x.wrapping_neg() } else { x };
    let ty = if ny { y.wrapping_neg() } else { y };
    let (r, negative) = match op {
        Op::And => (tx & ty, nx & ny),
        Op::Or => (tx | ty, nx | ny),
        Op::Xor => (tx ^ ty, nx ^ ny),
    };
    let m = if negative { r.wrapping_neg() } else { r };
    small_of(m, negative && r == 0, negative)
}

/// Writes the small result `s` as the whole value of `v`, in its vector when it has room, in a
/// new one otherwise.
#[inline(always)]
fn put_small(v: &mut Vec<u8>, order: ByteOrder, s: Small) {
    if v.capacity() >= s.len {
        // SAFETY: the vector has room for the `len` bytes, from 1 to 17, all of which
        // `write_small` writes before the length is set.
        unsafe {
            write_small(v.as_mut_ptr(), order, s.lo, s.hi, s.len);
            v.set_len(s.len);
        }
    } else {
        replace_small(v, order, s);
    }
}

/// Replaces `v` with one new vector of exactly the small result's length, holding it. Out of
/// line and cold, so that the small path saves no registers for it.
#[cold]
#[inline(never)]
fn replace_small(v: &mut Vec<u8>, order: ByteOrder, s: Small) {
    *v = new_small(order, s);
}

/// The small result `s` in one new vector of exactly its length.
#[inline(always)]
fn new_small(order: ByteOrder, s: Small) -> Vec<u8> {
    let mut v = Vec::with_capacity(s.len);
    // SAFETY: the new vector has room for the `len` bytes, from 1 to 17, all of which
    // `write_small` writes before the length is set.
    unsafe {
        write_small(v.as_mut_ptr(), order, s.lo, s.hi, s.len);
        v.set_len(s.len);
    }
    v
}

/// The small result `s` as a number in one new vector, in the byte order `order`. Out of line,
/// so that the small path of `&a op &b` saves no registers for the allocation.
#[inline(never)]
fn small_number(order: ByteOrder, s: Small) -> Number {
    Number {
        value: new_small(order, s),
        negative: s.negative,
        byte_order: order,
    }
}

// Operands that are not negative

/// `a op b` written over `a`, for `a` and `b` that are not negative (as their signs say), in the
/// same byte order, of which at least one has more than 16 bytes, when `a`'s vector has room for
/// the longest the result can be, and big-endian, the result's positions are those of `a`:
/// `op` is the operation on the magnitudes' bytes, known here, with no borrow or carry, so the
/// kernel runs straight over the bytes of the shorter magnitude, and the longer one's top bytes
/// follow for `|` and `^`. The length of the result is the longer magnitude's then, and otherwise
/// found from the top down, one read for random operands (of the operands big-endian, of the
/// result little-endian); big-endian, a shorter result is moved to the start of the vector. Returns whether it did, with nothing written
/// otherwise; zero operands are left to the general path.
///
/// Measured on an Apple M4 Max over operands in the level-1 cache, `a ^= &b` of 24, 64 and 128
/// bytes took 13, 16 and 17 cycles so, against 17, 28 and 28 through the paths for any signs, and
/// GMP 6.3's `mpz_xor(a, a, b)` 11, 13 and 20.
#[inline(always)]
fn plus_over<const OP: u8>(a: &mut Number, b: &Number) -> bool {
    if a.negative | b.negative || a.byte_order != b.byte_order {
        return false;
    }
    if a.byte_order == ByteOrder::BigEndian {
        plus_over_in::<OP, true>(&mut a.value, &b.value)
    } else {
        plus_over_in::<OP, false>(&mut a.value, &b.value)
    }
}

/// [`plus_over`] for the layout `BE`.
#[inline(never)]
fn plus_over_in<const OP: u8, const BE: bool>(v: &mut Vec<u8>, y: &[u8]) -> bool {
    let n = v.len();
    let (lx, ly) = (sig_len::<BE>(v), sig_len::<BE>(y));
    let (long, short) = (lx.max(ly), lx.min(ly));
    // The bound of the result's length, which is its length but for `&`, and `^` of magnitudes
    // of the same length.
    let u = if OP == AND { short } else { long };
    if short == 0 || v.capacity() < u || (BE && u != n) {
        return false;
    }
    let ye = end::<BE>(y);
    // The length of a result that can be shorter than `u`: big-endian, from the top of the
    // operands down, before anything is written; little-endian, read back from the top of the
    // result. Read back big-endian, it waited for the kernel's stores (measured on an Apple M4
    // Max, a big-endian `^=` of 40 to 64 bytes took 30 to 34 cycles so, against 19 to 20), and
    // read first little-endian, it took two cycles more than read back.
    let cut = OP == AND || (OP == XOR && lx == ly);
    let first = if cut && BE {
        // SAFETY: the positions below `short` are significant in both magnitudes.
        unsafe { highest::<BE, BE>(OP, end::<BE>(v), ye, 0, short) }.unwrap_or(1)
    } else {
        long
    };
    let base = v.as_mut_ptr();
    // SAFETY: the reads are of the significant bytes of each magnitude, the writes of the
    // `u <= capacity` bytes of `v`, at the positions of `x` (big-endian, `x` ends at `n = u`),
    // each block of `x` read before it is written over at the same positions.
    unsafe {
        let dst = if BE { base.add(n) } else { base };
        kernels::logic::<OP, false, false>(
            at_mut::<BE>(dst, 0, short),
            at::<BE>(dst, 0, short),
            at::<BE>(ye, 0, short),
            short,
        );
        let len = if cut && BE {
            if first < u {
                ptr::copy(base.add(u - first), base, first);
            }
            first
        } else if cut {
            highest_of::<BE>(dst, u)
        } else {
            if ly > lx {
                let k = long - short;
                ptr::copy_nonoverlapping(at::<BE>(ye, short, k), at_mut::<BE>(dst, short, k), k);
            }
            long
        };
        v.set_len(len);
    }
    true
}

/// The length up to the highest nonzero byte of the `u >= 1` bytes at the least significant end
/// `end` in the layout `BE`, or 1 when they are all zero: from the top down, eight at a time.
///
/// # Safety
///
/// The `u` bytes must be valid for reads.
#[inline(always)]
unsafe fn highest_of<const BE: bool>(end: *const u8, u: usize) -> usize {
    let mut q = u;
    // SAFETY: every read is of positions below `u`.
    unsafe {
        while q >= 8 {
            let w = load::<BE>(end, q - 8);
            if w != 0 {
                return q - w.leading_zeros() as usize / 8;
            }
            q -= 8;
        }
        while q > 1 && get::<BE>(end, q - 1) == 0 {
            q -= 1;
        }
        q.max(1)
    }
}

/// `a op b` in one new vector of exactly its length, for `a` and `b` that are not negative, in
/// the same byte order, of which at least one has more than 16 bytes, as [`plus_over`] computes
/// it, with the length found first: the longer magnitude's, or from the top down for `&`, and
/// `^` of magnitudes of the same length. `None` for the other operands, and zero ones.
#[inline(always)]
fn plus_new<const OP: u8>(a: &Number, b: &Number) -> Option<Number> {
    if a.negative | b.negative || a.byte_order != b.byte_order {
        return None;
    }
    let value = if a.byte_order == ByteOrder::BigEndian {
        plus_new_in::<OP, true>(&a.value, &b.value)?
    } else {
        plus_new_in::<OP, false>(&a.value, &b.value)?
    };
    Some(Number {
        value,
        negative: false,
        byte_order: a.byte_order,
    })
}

/// [`plus_new`] for the layout `BE`.
#[inline(never)]
fn plus_new_in<const OP: u8, const BE: bool>(x: &[u8], y: &[u8]) -> Option<Vec<u8>> {
    let (lx, ly) = (sig_len::<BE>(x), sig_len::<BE>(y));
    let (long, short) = (lx.max(ly), lx.min(ly));
    let (xe, ye) = (end::<BE>(x), end::<BE>(y));
    let len = if OP == AND || (OP == XOR && lx == ly) {
        // SAFETY: the positions below `short` are significant in both magnitudes.
        unsafe { highest::<BE, BE>(OP, xe, ye, 0, short)? }
    } else if short > 0 {
        long
    } else {
        return None;
    };
    let mut v = Vec::<u8>::with_capacity(len);
    // SAFETY: the new vector has room for the `len` bytes, all of which are written: positions
    // below `min(short, len)` by the kernel, the others, for `|` and `^`, from the longer
    // magnitude.
    unsafe {
        let base = v.as_mut_ptr();
        let dst = if BE { base.add(len) } else { base };
        let k = short.min(len);
        kernels::logic::<OP, false, false>(
            at_mut::<BE>(dst, 0, k),
            at::<BE>(xe, 0, k),
            at::<BE>(ye, 0, k),
            k,
        );
        if len > short {
            let (from, k) = (if lx > ly { xe } else { ye }, len - short);
            ptr::copy_nonoverlapping(at::<BE>(from, short, k), at_mut::<BE>(dst, short, k), k);
        }
        v.set_len(len);
    }
    Some(v)
}

// Operands of up to 64 bytes

/// The result of a bitwise operation on two operands of up to 64 bytes, computed in registers.
struct Medium {
    /// The limbs of the magnitude, the least significant first: up to eight, then the ninth,
    /// which is 1 for a magnitude of `2^(64 L)`, then zeros.
    limbs: [u64; 9],
    /// The length of the result in the shortest form: one byte for zero.
    len: usize,
    /// The number `L` of limbs the operands were read as, when the result is at least
    /// `8 (L - 1)` bytes long, with its top eight positions in `top`; zero otherwise.
    long: usize,
    /// The eight positions of the result that end at its top, when `long` is set.
    top: u64,
    /// Whether the result is negative; never for zero.
    negative: bool,
}

/// `a op b` for operands of at most 64 bytes, `op` one of [`AND`], [`OR`] and [`XOR`], by the
/// number of limbs of the longer vector, from 3 to 8.
///
/// The limit is the addition's medium path's, whose reading of limbs this shares. Measured on an
/// Apple M4 Max with negative operands, `a &= &b` and `a ^= &b` of 40 and 48 bytes took 28 to 32
/// cycles in registers, against 36 to 40 through the general path; at 64 bytes the two are about
/// even (36 to 41 cycles against 35 to 48).
///
/// # Safety
///
/// Both vectors must hold at most [`MEDIUM`](super::add::MEDIUM) bytes.
#[inline(always)]
unsafe fn medium_result<const OP: u8>(a: &Number, b: &Number) -> Medium {
    // SAFETY: as the caller guarantees, the vectors hold at most `8 L` bytes for the `L` chosen.
    unsafe {
        match a.value.len().max(b.value.len()).div_ceil(8) {
            ..=3 => medium_of::<OP, 3>(a, b),
            4 => medium_of::<OP, 4>(a, b),
            5 => medium_of::<OP, 5>(a, b),
            6 => medium_of::<OP, 6>(a, b),
            7 => medium_of::<OP, 7>(a, b),
            _ => medium_of::<OP, 8>(a, b),
        }
    }
}

/// [`medium_result`] for vectors of at most `L` limbs: the magnitudes read into `L` limbs each as
/// the addition reads them, turned into two's complement where they are negative (`!x + 1`, one
/// chain of carries, the limbs above the vector all ones), `op` on the limbs, and a negative
/// result turned back into its magnitude the same way, all without a branch.
///
/// # Safety
///
/// Both vectors must hold at most `8 L` bytes.
#[inline(never)]
unsafe fn medium_of<const OP: u8, const L: usize>(a: &Number, b: &Number) -> Medium {
    // SAFETY: as the caller guarantees.
    let (x, y) = unsafe {
        (
            read_limbs::<L>(&a.value, a.byte_order),
            read_limbs::<L>(&b.value, b.byte_order),
        )
    };
    let nonzero = |v: &[u64; 9]| v.iter().take(L).fold(0, |o, &l| o | l) != 0;
    // A magnitude of zero is zero, whatever the sign says.
    let (nx, ny) = (a.negative && nonzero(&x), b.negative && nonzero(&y));
    let negative = match OP {
        AND => nx & ny,
        OR => nx | ny,
        _ => nx ^ ny,
    };
    let (mx, my, mr) = (
        0u64.wrapping_sub(u64::from(nx)),
        0u64.wrapping_sub(u64::from(ny)),
        0u64.wrapping_sub(u64::from(negative)),
    );
    let (mut cx, mut cy, mut cr) = (nx, ny, negative);
    let mut limbs = [0u64; 9];
    for i in 0..L {
        let (tx, c) = (x[i] ^ mx).overflowing_add(u64::from(cx));
        cx = c;
        let (ty, c) = (y[i] ^ my).overflowing_add(u64::from(cy));
        cy = c;
        let r = match OP {
            AND => tx & ty,
            OR => tx | ty,
            _ => tx ^ ty,
        };
        let (m, c) = (r ^ mr).overflowing_add(u64::from(cr));
        cr = c;
        limbs[i] = m;
    }
    // Above the limbs, the result is `mr` in two's complement, and its magnitude zero, but for
    // the carry out of the top limb: a negative result that is zero in all `L` limbs is
    // `-2^(64 L)`.
    limbs[L] = u64::from(cr);
    if long::<L>(&limbs) {
        // The length and the top eight positions without a branch, as the addition finds them:
        // whether `&` or `|` of operands of a sign each loses a top byte is a coin toss.
        let (len, top) = top_of::<L>(&limbs);
        return Medium {
            limbs,
            len,
            long: L,
            top,
            negative,
        };
    }
    let mut t = L;
    while t > 0 && limbs[t] == 0 {
        t -= 1;
    }
    Medium {
        limbs,
        len: 8 * t + 8 - (limbs[t] | 1).leading_zeros() as usize / 8,
        long: 0,
        top: 0,
        negative,
    }
}

/// Writes the medium result `m` over the first `m.len` bytes at `p`, in the layout of `order`.
///
/// A result of at least `8 (L - 1)` bytes, as for operands of `L` limbs that do not nearly
/// cancel, is written by stores chosen by `L`, not by its length, which `&` and `|` make a coin
/// toss: limbs `0..L - 1` at their places, limb `L - 1` at its place or, for a result shorter
/// than `8 L` bytes, below the top eight positions, and then those, as the addition writes a
/// medium sum. Measured on an Apple M4 Max, `a &= &b` of a positive `a` and a negative `b` of 24
/// and 48 bytes took 20 and 29 cycles so, against 25 and 32 with stores chosen by the length. Otherwise: up to 16 bytes as the addition writes a short sum; beyond, the whole
/// limbs, then the eight positions that end at the top, over the limb below.
///
/// # Safety
///
/// `p` must be valid for writing `m.len` bytes.
#[inline(always)]
unsafe fn write_medium(p: *mut u8, order: ByteOrder, m: &Medium) {
    let (r, len) = (&m.limbs, m.len);
    let be = order == ByteOrder::BigEndian;
    // SAFETY: every write lies inside the `len` bytes at `p`: a long result has `len >= 8 (L - 1)`
    // bytes and `L >= 3`, so `len >= 16`.
    unsafe {
        if m.long > 0 {
            let l = m.long;
            let c = (8 * (l - 1)).min(len - 8);
            for (i, limb) in r.iter().enumerate().take(l - 1) {
                if be {
                    write(p.add(len - 8 - 8 * i), limb.to_be_bytes());
                } else {
                    write(p.add(8 * i), limb.to_le_bytes());
                }
            }
            if be {
                write(p.add(len - 8 - c), r[l - 1].to_be_bytes());
                write(p, m.top.to_be_bytes());
            } else {
                write(p.add(c), r[l - 1].to_le_bytes());
                write(p.add(len - 8), m.top.to_le_bytes());
            }
            return;
        }
        if len <= SMALL {
            write_small(p, order, r[0], r[1], len);
            return;
        }
        let (full, rem) = (len / 8, len % 8);
        for (i, limb) in r.iter().enumerate().take(full) {
            if be {
                write(p.add(len - 8 - 8 * i), limb.to_be_bytes());
            } else {
                write(p.add(8 * i), limb.to_le_bytes());
            }
        }
        if rem > 0 {
            let k = 8 * rem as u32;
            let top = (r[full - 1] >> k) | (r[full] << (64 - k));
            if be {
                write(p, top.to_be_bytes());
            } else {
                write(p.add(len - 8), top.to_le_bytes());
            }
        }
    }
}

/// Writes the medium result `m` as the whole value of `v`, in its vector when it has room, in a
/// new one otherwise.
#[inline(always)]
fn put_medium(v: &mut Vec<u8>, order: ByteOrder, m: &Medium) {
    if v.capacity() < m.len {
        *v = Vec::with_capacity(m.len);
    }
    // SAFETY: the vector has room for the `len` bytes, all of which are written before the length
    // is set.
    unsafe {
        write_medium(v.as_mut_ptr(), order, m);
        v.set_len(m.len);
    }
}

/// The medium result `m` as a number in one new vector, in the byte order `order`.
#[inline(never)]
fn medium_number(order: ByteOrder, m: &Medium) -> Number {
    let mut value = Vec::new();
    put_medium(&mut value, order, m);
    Number {
        value,
        negative: m.negative,
        byte_order: order,
    }
}

// The general path: operands of more than 64 bytes, out of line, so that the small path saves no
// registers for it.

/// `a op b` for borrowed operands, of which at least one has more than 64 bytes.
#[inline(never)]
fn ref_ref(a: &Number, b: &Number, op: Op) -> Number {
    let plan = plan(a, b, op);
    Number {
        value: new_result(a, b, &plan),
        negative: plan.negative,
        byte_order: a.byte_order,
    }
}

/// `a op b` for a borrowed `a` and an owned `b`, of which at least one has more than 64 bytes.
/// The result takes the vector of `b` when it is built there, leaving `b` an empty one; the
/// caller drops `b` either way.
#[inline(never)]
fn ref_value(a: &Number, b: &mut Number, op: Op) -> Number {
    let order = a.byte_order;
    // The operations are symmetric: `b op a` over `b`, when it is in the result's byte order.
    if b.byte_order == order
        && let Some(negative) = over(&mut b.value, order, b.negative, a, op)
    {
        return Number {
            value: mem::take(&mut b.value),
            negative,
            byte_order: order,
        };
    }
    let plan = plan(a, b, op);
    let value = if b.value.capacity() >= plan.len {
        if b.byte_order != order {
            b.value.reverse();
            b.byte_order = order;
        }
        // SAFETY: the vector of `b`, in the byte order of the result now, has room for it, and the
        // plan with the operands swapped is that of `b` and `a`.
        unsafe { in_vector(&mut b.value, order, a, &plan.swapped()) };
        mem::take(&mut b.value)
    } else {
        new_result(a, b, &plan)
    };
    Number {
        value,
        negative: plan.negative,
        byte_order: order,
    }
}

/// `a op= b` for a borrowed `b`, where at least one of them has more than 64 bytes.
#[inline(never)]
fn assign_ref(a: &mut Number, b: &Number, op: Op) {
    if let Some(negative) = over(&mut a.value, a.byte_order, a.negative, b, op) {
        a.negative = negative;
        return;
    }
    let plan = plan(a, b, op);
    if a.value.capacity() >= plan.len {
        // SAFETY: the vector of `a` has room for the result, and the plan is that of `a` and `b`.
        unsafe { in_vector(&mut a.value, a.byte_order, b, &plan) };
    } else {
        a.value = new_result(a, b, &plan);
    }
    a.negative = plan.negative;
}

/// `a op= b` for an owned `b`, where at least one of them has more than 64 bytes. When the
/// result is built in the vector of `b`, the vector of `a` goes to `b`; the caller drops `b`
/// either way.
#[inline(never)]
fn assign_value(a: &mut Number, b: &mut Number, op: Op) {
    let order = a.byte_order;
    if let Some(negative) = over(&mut a.value, order, a.negative, b, op) {
        a.negative = negative;
        return;
    }
    let plan = plan(a, b, op);
    if a.value.capacity() >= plan.len {
        // SAFETY: the vector of `a` has room for the result, and the plan is that of `a` and `b`.
        unsafe { in_vector(&mut a.value, order, b, &plan) };
    } else if b.value.capacity() >= plan.len {
        if b.byte_order != order {
            b.value.reverse();
            b.byte_order = order;
        }
        // SAFETY: the vector of `b`, in the byte order of the result now, has room for it, and the
        // plan with the operands swapped is that of `b` and `a`.
        unsafe { in_vector(&mut b.value, order, a, &plan.swapped()) };
        mem::swap(&mut a.value, &mut b.value);
    } else {
        a.value = new_result(a, b, &plan);
    }
    a.negative = plan.negative;
}

// Over an operand

/// `x op y` written over `x`, the magnitude in `v` in the layout `order` with the sign
/// `negative_x`, as the general path's first try, when it is simple: `v` has room for the result
/// in the shortest form, which big-endian also ends where `x` does, and a borrow or a carry runs
/// through one limb at most, as for random operands. Then the plan is cut short: the length of the
/// result is the longer magnitude's when the result takes its top bytes, and otherwise found from
/// the top down, one read for random operands; the head is one limb, computed and written first;
/// and the rest is one call to the kernel and a copy. Returns the sign of the result, or `None`,
/// with nothing written, for the cases left to the plan.
///
/// Measured on an Apple M4 Max over operands in the level-1 cache, the plan and its second pass
/// over the head cost a `^=` of 256 bytes about 340 instructions a call, 57 cycles; this path,
/// 250 instructions and 35 to 38 cycles.
#[inline]
fn over(v: &mut Vec<u8>, order: ByteOrder, negative_x: bool, y: &Number, op: Op) -> Option<bool> {
    // SAFETY: `v` and `y` are a valid vector and number.
    unsafe {
        match (
            order == ByteOrder::BigEndian,
            y.byte_order == ByteOrder::BigEndian,
        ) {
            (true, true) => over_in::<true, true>(v, negative_x, y, op),
            (true, false) => over_in::<true, false>(v, negative_x, y, op),
            (false, true) => over_in::<false, true>(v, negative_x, y, op),
            (false, false) => over_in::<false, false>(v, negative_x, y, op),
        }
    }
}

/// [`over`] for `x` in the layout `XBE` and `y` in `YBE`.
///
/// # Safety
///
/// None beyond the validity of `v` and `y`; unsafe for the raw reads and writes inside.
#[inline(never)]
unsafe fn over_in<const XBE: bool, const YBE: bool>(
    v: &mut Vec<u8>,
    negative_x: bool,
    y: &Number,
    op: Op,
) -> Option<bool> {
    let n = v.len();
    let lx = sig_len::<XBE>(v);
    let ly = sig_len::<YBE>(&y.value);
    let (nx, ny) = (negative_x && lx > 0, y.negative && ly > 0);
    let (g, negative) = select(op, nx, ny);
    let tail = tail_of(g, lx, ly);
    let (long, short) = (lx.max(ly), lx.min(ly));
    let x = end::<XBE>(v);
    let ye = end::<YBE>(&y.value);
    // SAFETY: the reads are of the significant bytes of each magnitude.
    let (head, lo) = unsafe {
        if nx | ny | negative {
            let mut flags = Flags {
                x: nx,
                y: ny,
                carry: negative,
            };
            let w = flags.next(g, limb::<XBE>(x, lx, 0), limb::<YBE>(ye, ly, 0));
            if flags.running() || short < 8 {
                return None;
            }
            (w, 8)
        } else {
            (0, 0)
        }
    };
    let len = if tail != Tail::Zero {
        long
    } else {
        // SAFETY: the positions below `short` are significant in both magnitudes.
        match unsafe { highest::<XBE, YBE>(g, x, ye, lo, short) } {
            Some(len) => len,
            None if lo == 0 => return None,
            // The head's limb, alone.
            None => 8 - (head | 1).leading_zeros() as usize / 8,
        }
    };
    if v.capacity() < len || (XBE && len != n) {
        return None;
    }
    let base = v.as_mut_ptr();
    // SAFETY: the result ends where `x` does, big-endian, as `len == n`; the reads are of the
    // significant bytes of each magnitude, the writes of the `len <= capacity` bytes of `v`, each
    // limb or block of `x` read before it is written over at the same positions. The bulk ends at
    // `len` when that is below `short`, and the head is at most `len` long but for its zero bytes.
    unsafe {
        let dst = if XBE { base.add(n) } else { base };
        let x = dst.cast_const();
        if lo > 0 {
            put_limb::<XBE>(dst, len, 0, head);
        }
        let bulk = short.min(len);
        if bulk > lo {
            let k = bulk - lo;
            logic::<false>(
                g,
                XBE != YBE,
                at_mut::<XBE>(dst, lo, k),
                at::<XBE>(x, lo, k),
                at::<YBE>(ye, lo, k),
                k,
            );
        }
        if tail == Tail::Y {
            let k = long - short;
            if XBE != YBE {
                kernels::copy_reversed(at_mut::<XBE>(dst, short, k), at::<YBE>(ye, short, k), k);
            } else {
                ptr::copy_nonoverlapping(at::<YBE>(ye, short, k), at_mut::<XBE>(dst, short, k), k);
            }
        }
        v.set_len(len);
    }
    Some(negative)
}

/// What the bytes of the result of `g` above the shorter magnitude are, for magnitudes of `lx`
/// and `ly` significant bytes: `g(x, 0)` is `x` for `|`, `^` and `x & !y`, and zero for the
/// others; `g(0, y)` likewise.
#[inline(always)]
fn tail_of(g: u8, lx: usize, ly: usize) -> Tail {
    // Bit `g` of each mask: whether `g` keeps the longer `x`, or the longer `y`.
    const KEEPS_X: u8 = 1 << OR | 1 << XOR | 1 << AND_NOT;
    const KEEPS_Y: u8 = 1 << OR | 1 << XOR | 1 << NOT_AND;
    if lx > ly && KEEPS_X >> g & 1 != 0 {
        Tail::X
    } else if ly > lx && KEEPS_Y >> g & 1 != 0 {
        Tail::Y
    } else {
        Tail::Zero
    }
}

/// The significant length of the magnitude in `v` in the layout `BE`: its length when its top
/// byte is not zero, as in the shortest form, at the cost of one read, and otherwise as
/// [`significant_len`] finds it.
#[inline(always)]
fn sig_len<const BE: bool>(v: &[u8]) -> usize {
    let n = v.len();
    // SAFETY: the top byte is read when there is one.
    if n > 0 && unsafe { get::<BE>(end::<BE>(v), n - 1) } != 0 {
        n
    } else {
        significant_len::<BE>(v)
    }
}

// Planning

/// What the bytes of the result above the shorter magnitude are.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Tail {
    /// Zero: the result ends below them.
    Zero,
    /// Those of `x`.
    X,
    /// Those of `y`.
    Y,
}

/// Everything about the result of a bitwise operation on `x` and `y` but its bytes.
#[derive(Clone, Copy)]
struct Plan {
    /// The operation on the bytes of the magnitudes (or of the magnitudes less one).
    g: u8,
    /// The significant length of `x`.
    lx: usize,
    /// The significant length of `y`.
    ly: usize,
    /// Whether `x` is negative and not zero: `X - 1` is taken instead of `X`.
    nx: bool,
    /// Whether `y` is negative and not zero.
    ny: bool,
    /// Whether the result is negative: one is added to what `g` gives.
    negative: bool,
    /// The number of limbs at the low end through which a borrow or the carry runs, worked out
    /// one at a time; above them, each byte of the result is `g` of the operands' bytes.
    head: usize,
    /// The bytes of the result above the shorter magnitude.
    tail: Tail,
    /// The length of the result in the shortest form: one byte for zero.
    len: usize,
}

impl Plan {
    /// The plan of the same operation with `x` and `y` swapped.
    #[inline(always)]
    fn swapped(&self) -> Plan {
        Plan {
            g: match self.g {
                AND_NOT => NOT_AND,
                NOT_AND => AND_NOT,
                g => g,
            },
            lx: self.ly,
            ly: self.lx,
            nx: self.ny,
            ny: self.nx,
            tail: match self.tail {
                Tail::X => Tail::Y,
                Tail::Y => Tail::X,
                Tail::Zero => Tail::Zero,
            },
            ..*self
        }
    }
}

/// The borrows of `X - 1` and `Y - 1` and the carry of the result's `+ 1`, from one limb to the
/// next.
struct Flags {
    x: bool,
    y: bool,
    carry: bool,
}

impl Flags {
    /// The limb of the result from the limbs `x` and `y` of the magnitudes at the same position,
    /// and the flags for the next limb. A borrow out of `X - 1` stops at the lowest nonzero limb
    /// of `X`, and the carry at the lowest limb of `g` that is not all ones.
    #[inline(always)]
    fn next(&mut self, g: u8, x: u64, y: u64) -> u64 {
        let xm = x.wrapping_sub(u64::from(self.x));
        self.x &= x == 0;
        let ym = y.wrapping_sub(u64::from(self.y));
        self.y &= y == 0;
        let m = apply(g, xm, ym);
        let r = m.wrapping_add(u64::from(self.carry));
        self.carry &= m == u64::MAX;
        r
    }

    /// Whether a borrow or the carry is still running.
    #[inline(always)]
    fn running(&self) -> bool {
        self.x | self.y | self.carry
    }
}

/// The plan of `a op b`, with `x` = `a` and `y` = `b`.
#[inline]
fn plan(a: &Number, b: &Number, op: Op) -> Plan {
    match (
        a.byte_order == ByteOrder::BigEndian,
        b.byte_order == ByteOrder::BigEndian,
    ) {
        (true, true) => plan_in::<true, true>(a, b, op),
        (true, false) => plan_in::<true, false>(a, b, op),
        (false, true) => plan_in::<false, true>(a, b, op),
        (false, false) => plan_in::<false, false>(a, b, op),
    }
}

/// [`plan`] for `a` in the layout `XBE` and `b` in `YBE`.
#[inline(always)]
fn plan_in<const XBE: bool, const YBE: bool>(a: &Number, b: &Number, op: Op) -> Plan {
    let lx = sig_len::<XBE>(&a.value);
    let ly = sig_len::<YBE>(&b.value);
    let (nx, ny) = (a.negative && lx > 0, b.negative && ly > 0);
    let (g, negative) = select(op, nx, ny);
    let tail = tail_of(g, lx, ly);
    let (xe, ye) = (end::<XBE>(&a.value), end::<YBE>(&b.value));
    // The head, and the length of its part of the result.
    let (mut head, mut top) = (0, 0);
    if nx | ny | negative {
        let mut flags = Flags {
            x: nx,
            y: ny,
            carry: negative,
        };
        loop {
            // SAFETY: `limb` reads the significant bytes of each magnitude only. The loop ends:
            // past both magnitudes, whose borrows end at their lowest nonzero byte, `g(0, 0)` is
            // zero and ends the carry.
            let r = unsafe { flags.next(g, limb::<XBE>(xe, lx, head), limb::<YBE>(ye, ly, head)) };
            if r != 0 {
                top = 8 * head + 8 - r.leading_zeros() as usize / 8;
            }
            head += 1;
            if !flags.running() {
                break;
            }
        }
    }
    let (long, short) = (lx.max(ly), lx.min(ly));
    let len = if 8 * head >= long {
        top.max(1)
    } else if tail != Tail::Zero {
        // The top byte of the longer magnitude, which is not zero.
        long
    } else {
        // SAFETY: the positions below `short` are significant in both magnitudes.
        unsafe { highest::<XBE, YBE>(g, xe, ye, 8 * head, short) }.unwrap_or(top.max(1))
    };
    Plan {
        g,
        lx,
        ly,
        nx,
        ny,
        negative,
        head,
        tail,
        len,
    }
}

/// The length up to the highest nonzero byte of `g(x, y)` at positions `lo..hi`, or `None` when
/// they are all zero: from the top down, eight positions at a time, so that for random operands
/// the first read decides.
///
/// # Safety
///
/// Positions `lo..hi` of the magnitudes at the least significant ends `x` (layout `XBE`) and `y`
/// (layout `YBE`) must be valid for reads.
#[inline(always)]
unsafe fn highest<const XBE: bool, const YBE: bool>(
    g: u8,
    x: *const u8,
    y: *const u8,
    lo: usize,
    hi: usize,
) -> Option<usize> {
    let mut q = hi;
    // SAFETY: every read is of positions from `lo` to `hi`.
    unsafe {
        while q >= lo + 8 {
            let m = apply(g, load::<XBE>(x, q - 8), load::<YBE>(y, q - 8));
            if m != 0 {
                return Some(q - m.leading_zeros() as usize / 8);
            }
            q -= 8;
        }
        while q > lo {
            let m = apply(
                g,
                u64::from(get::<XBE>(x, q - 1)),
                u64::from(get::<YBE>(y, q - 1)),
            );
            if m as u8 != 0 {
                return Some(q);
            }
            q -= 1;
        }
    }
    None
}

// Building the result

/// Where the result is written, relative to the magnitude `x` it is computed from.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Place {
    /// In a vector of its own.
    New,
    /// Over `x`, at the same positions.
    Same,
    /// Over `x`, `d > 0` bytes further on in memory: a big-endian result longer than the vector
    /// of `x`, whose least significant byte is at the end.
    Above,
    /// Over `x`, `d > 0` bytes further back in memory: a big-endian result shorter than the vector
    /// of `x`.
    Below,
}

/// The limbs of the head a result written [`Place::Below`] its operand holds back until the rest
/// is written: as many as random operands need, one, and more. A longer head moves the operand
/// first.
const HELD: usize = 4;

/// The result of `plan` in one new vector of exactly its length, from `x` = `a` and `y` = `b`.
fn new_result(a: &Number, b: &Number, plan: &Plan) -> Vec<u8> {
    let mut v = Vec::<u8>::with_capacity(plan.len);
    let p = v.as_mut_ptr();
    // SAFETY: the new vector has room for the `len` bytes of the result, all of which `compute`
    // writes, apart from the operands.
    unsafe {
        match (
            a.byte_order == ByteOrder::BigEndian,
            b.byte_order == ByteOrder::BigEndian,
        ) {
            (true, true) => compute::<true, true>(
                p.add(plan.len),
                end::<true>(&a.value),
                end::<true>(&b.value),
                plan,
                Place::New,
            ),
            (true, false) => compute::<true, false>(
                p.add(plan.len),
                end::<true>(&a.value),
                end::<false>(&b.value),
                plan,
                Place::New,
            ),
            (false, true) => compute::<false, true>(
                p,
                end::<false>(&a.value),
                end::<true>(&b.value),
                plan,
                Place::New,
            ),
            (false, false) => compute::<false, false>(
                p,
                end::<false>(&a.value),
                end::<false>(&b.value),
                plan,
                Place::New,
            ),
        }
        v.set_len(plan.len);
    }
    v
}

/// Computes the result of `plan` over `x`, the magnitude in `v` in the layout `order`, and `y`,
/// and sets the length of `v` to it.
///
/// # Safety
///
/// The capacity of `v` must be at least the length of the result, and the plan that of `x` and
/// `y`.
#[inline(always)]
unsafe fn in_vector(v: &mut Vec<u8>, order: ByteOrder, y: &Number, plan: &Plan) {
    let (n, len) = (v.len(), plan.len);
    let base = v.as_mut_ptr();
    // SAFETY: as the caller guarantees: the vector has room for the `len` bytes, all of which
    // `compute` writes. Big-endian, `x` ends at `n` and the result at `len`.
    unsafe {
        if order == ByteOrder::BigEndian {
            let place = match len.cmp(&n) {
                core::cmp::Ordering::Equal => Place::Same,
                core::cmp::Ordering::Greater => Place::Above,
                core::cmp::Ordering::Less => Place::Below,
            };
            let (dst, x) = (base.add(len), base.add(n).cast_const());
            if y.byte_order == ByteOrder::BigEndian {
                compute::<true, true>(dst, x, end::<true>(&y.value), plan, place);
            } else {
                compute::<true, false>(dst, x, end::<false>(&y.value), plan, place);
            }
        } else if y.byte_order == ByteOrder::BigEndian {
            compute::<false, true>(base, base, end::<true>(&y.value), plan, Place::Same);
        } else {
            compute::<false, false>(base, base, end::<false>(&y.value), plan, Place::Same);
        }
        v.set_len(len);
    }
}

/// Writes the `len` bytes of the result of `plan` at the least significant end `dst` in the
/// layout `XBE`, from the magnitudes at the least significant ends `x` (layout `XBE`) and `y`
/// (`YBE`), with `dst` placed as `place` says relative to `x`: the head limb by limb, the bytes
/// above it up to the shorter magnitude by the kernels, and the tail copied.
///
/// Over `x`, every byte of it is read before it is written over: at the same positions each
/// limb or block is read before it is written; for a result further on in memory, the head
/// first and the rest from the least significant byte up; for one further back, the head is
/// held back (or `x` moved first, for a long head), the tail first and the rest from the most
/// significant byte down.
///
/// # Safety
///
/// The magnitudes' significant bytes must be valid for reads, `dst` for writing the `len` bytes,
/// and `place` true of them; the plan must be that of `x` and `y`.
#[inline(never)]
unsafe fn compute<const XBE: bool, const YBE: bool>(
    dst: *mut u8,
    x: *const u8,
    y: *const u8,
    plan: &Plan,
    place: Place,
) {
    let mixed = XBE != YBE;
    let len = plan.len;
    let short = plan.lx.min(plan.ly);
    let lo = 8 * plan.head;
    // The head's limbs below the end of the result, the bulk `lo..bulk` and the tail `rest..len`.
    let held = plan.head.min(len.div_ceil(8));
    let bulk = short.min(len);
    let rest = short.max(lo);
    // SAFETY: as the caller guarantees; every position written is below `len`, and every one
    // read is significant in its magnitude. The order of the steps is that of the doc comment.
    unsafe {
        match place {
            Place::Below if held > HELD => {
                // Moves the bytes of `x` the result needs to its positions first: those below
                // the result's end, which is all that the result depends on.
                let m = plan.lx.min(len);
                ptr::copy(x.sub(m), dst.sub(m), m);
                let moved = Plan { lx: m, ..*plan };
                compute::<XBE, YBE>(dst, dst, y, &moved, Place::Same);
            }
            Place::Below => {
                let mut limbs = [0u64; HELD];
                let mut flags = Flags {
                    x: plan.nx,
                    y: plan.ny,
                    carry: plan.negative,
                };
                for (i, w) in limbs.iter_mut().enumerate().take(held) {
                    *w = flags.next(
                        plan.g,
                        limb::<XBE>(x, plan.lx, i),
                        limb::<YBE>(y, plan.ly, i),
                    );
                }
                tail::<XBE, YBE>(dst, x, y, plan, rest, place);
                if bulk > lo {
                    let n = bulk - lo;
                    logic::<false>(
                        plan.g,
                        mixed,
                        at_mut::<XBE>(dst, lo, n),
                        at::<XBE>(x, lo, n),
                        at::<YBE>(y, lo, n),
                        n,
                    );
                }
                for (i, &w) in limbs.iter().enumerate().take(held) {
                    put_limb::<XBE>(dst, len, i, w);
                }
            }
            _ => {
                let mut flags = Flags {
                    x: plan.nx,
                    y: plan.ny,
                    carry: plan.negative,
                };
                for i in 0..held {
                    let w = flags.next(
                        plan.g,
                        limb::<XBE>(x, plan.lx, i),
                        limb::<YBE>(y, plan.ly, i),
                    );
                    put_limb::<XBE>(dst, len, i, w);
                }
                if bulk > lo {
                    let n = bulk - lo;
                    let (d, xs, ys) = (
                        at_mut::<XBE>(dst, lo, n),
                        at::<XBE>(x, lo, n),
                        at::<YBE>(y, lo, n),
                    );
                    if place == Place::Above {
                        logic::<true>(plan.g, mixed, d, xs, ys, n);
                    } else {
                        logic::<false>(plan.g, mixed, d, xs, ys, n);
                    }
                }
                tail::<XBE, YBE>(dst, x, y, plan, rest, place);
            }
        }
    }
}

/// Writes the tail of the result, positions `rest..len`, from the longer magnitude, unless they
/// are already there; or the zero byte of a zero result that nothing else writes.
///
/// # Safety
///
/// As for [`compute`].
#[inline(always)]
unsafe fn tail<const XBE: bool, const YBE: bool>(
    dst: *mut u8,
    x: *const u8,
    y: *const u8,
    plan: &Plan,
    rest: usize,
    place: Place,
) {
    let len = plan.len;
    if len <= rest {
        return;
    }
    let n = len - rest;
    // SAFETY: as the caller guarantees: positions `rest..len` are significant in the longer
    // magnitude, and below `len` in the result.
    unsafe {
        match plan.tail {
            Tail::X if place != Place::Same => {
                ptr::copy(at::<XBE>(x, rest, n), at_mut::<XBE>(dst, rest, n), n);
            }
            Tail::Y if XBE != YBE => {
                kernels::copy_reversed(at_mut::<XBE>(dst, rest, n), at::<YBE>(y, rest, n), n);
            }
            Tail::Y => {
                ptr::copy_nonoverlapping(at::<YBE>(y, rest, n), at_mut::<XBE>(dst, rest, n), n);
            }
            Tail::Zero => {
                // Zero, from operands of which one is zero and neither is negative: its byte is
                // all that the head and the bulk leave out.
                ptr::write_bytes(at_mut::<XBE>(dst, rest, n), 0, n);
            }
            Tail::X => {}
        }
    }
}

/// A kernel: `dst[k] = g(x[k], y[k])` over `n` bytes.
type Kernel = unsafe fn(*mut u8, *const u8, *const u8, usize);

/// The kernels by direction, `y` reversed or not, and the operation `g`.
const KERNELS: [[[Kernel; 5]; 2]; 2] = [
    [
        [
            kernels::logic::<AND, false, false>,
            kernels::logic::<OR, false, false>,
            kernels::logic::<XOR, false, false>,
            kernels::logic::<AND_NOT, false, false>,
            kernels::logic::<NOT_AND, false, false>,
        ],
        [
            kernels::logic::<AND, true, false>,
            kernels::logic::<OR, true, false>,
            kernels::logic::<XOR, true, false>,
            kernels::logic::<AND_NOT, true, false>,
            kernels::logic::<NOT_AND, true, false>,
        ],
    ],
    [
        [
            kernels::logic::<AND, false, true>,
            kernels::logic::<OR, false, true>,
            kernels::logic::<XOR, false, true>,
            kernels::logic::<AND_NOT, false, true>,
            kernels::logic::<NOT_AND, false, true>,
        ],
        [
            kernels::logic::<AND, true, true>,
            kernels::logic::<OR, true, true>,
            kernels::logic::<XOR, true, true>,
            kernels::logic::<AND_NOT, true, true>,
            kernels::logic::<NOT_AND, true, true>,
        ],
    ],
];

/// Runs the kernel of the operation `g` over the `n` bytes at `dst`, `x` and `y`, in the order
/// of memory, `y` reversed when `mixed`; from the top down when `DOWN`. One call through a table,
/// whose target is the same from one call to the next for the same operation and signs.
///
/// # Safety
///
/// As for the kernels.
#[inline(always)]
unsafe fn logic<const DOWN: bool>(
    g: u8,
    mixed: bool,
    dst: *mut u8,
    x: *const u8,
    y: *const u8,
    n: usize,
) {
    // SAFETY: as the caller guarantees; `g` is one of the five operations.
    unsafe { KERNELS[usize::from(DOWN)][usize::from(mixed)][usize::from(g)](dst, x, y, n) }
}

// Not

/// `!a` for a borrowed `a`, in one new vector: `-(A + 1)` for `a >= 0`, `A - 1` for `a < 0`.
#[inline(never)]
fn not_new(a: &Number) -> Number {
    let order = a.byte_order;
    if a.value.len() <= SMALL {
        // SAFETY: the vector holds at most 16 bytes.
        let s = unsafe { small_not(a) };
        return small_number(order, s);
    }
    let (value, negative) = if order == ByteOrder::BigEndian {
        not_copy::<true>(a)
    } else {
        not_copy::<false>(a)
    };
    Number {
        value,
        negative,
        byte_order: order,
    }
}

/// `!a` for a small `a`, in registers.
///
/// # Safety
///
/// The vector of `a` must hold at most [`SMALL`] bytes.
#[inline(always)]
unsafe fn small_not(a: &Number) -> Small {
    // SAFETY: as the caller guarantees.
    let x = unsafe { read_small(&a.value, a.byte_order) };
    if a.negative && x != 0 {
        small_of(x - 1, false, false)
    } else {
        let (m, carry) = x.overflowing_add(1);
        small_of(m, carry, true)
    }
}

/// `!a` in a new vector, for `a` in the layout `BE`: its bytes and its sign.
///
/// For a magnitude of more than eight bytes whose eight least significant ones the `+ 1` (or the
/// `- 1`) does not carry (or borrow) out of, as for random numbers, the magnitude is copied and
/// those eight bytes written with the one added (or taken); otherwise as [`step`] writes it.
fn not_copy<const BE: bool>(a: &Number) -> (Vec<u8>, bool) {
    let l = sig_len::<BE>(&a.value);
    let src = end::<BE>(&a.value);
    let negative = !(a.negative && l > 0);
    if l > 8 {
        // SAFETY: the eight least significant bytes are significant.
        let w = unsafe { load::<BE>(src, 0) };
        let (low, out) = if negative {
            w.overflowing_add(1)
        } else {
            w.overflowing_sub(1)
        };
        if !out {
            let mut v = Vec::<u8>::with_capacity(l);
            // SAFETY: the new vector has room for the `l` bytes of the result, the magnitude's with
            // the eight least significant ones replaced, all of which are written before the
            // length is set.
            unsafe {
                let base = v.as_mut_ptr();
                ptr::copy_nonoverlapping(at::<BE>(src, 0, l), base, l);
                store::<BE>(if BE { base.add(l) } else { base }, 0, low);
                v.set_len(l);
            }
            return (v, negative);
        }
    }
    // SAFETY: `run` reads the significant bytes only.
    let (r, len) = unsafe { run::<BE>(src, l, negative) };
    let mut v = Vec::<u8>::with_capacity(len);
    let base = v.as_mut_ptr();
    let dst = if BE {
        // SAFETY: the end of the new vector's `len` bytes.
        unsafe { base.add(len) }
    } else {
        base
    };
    // SAFETY: the new vector has room for the `len` bytes, all of which `step` writes.
    unsafe {
        step::<BE>(dst, src, l, r, len, negative, false);
        v.set_len(len);
    }
    (v, negative)
}

/// `!a` in the vector of `a`.
#[inline(never)]
fn not_in_place(a: &mut Number) {
    if a.value.len() <= SMALL {
        // SAFETY: the vector holds at most 16 bytes.
        let s = unsafe { small_not(a) };
        put_small(&mut a.value, a.byte_order, s);
        a.negative = s.negative;
        return;
    }
    if a.byte_order == ByteOrder::BigEndian {
        not_in::<true>(a);
    } else {
        not_in::<false>(a);
    }
}

/// [`not_in_place`] for `a` in the layout `BE`.
///
/// A magnitude of more than eight bytes whose eight least significant ones the `+ 1` (or the
/// `- 1`) does not carry (or borrow) out of, as for random numbers, keeps its length, and only
/// those eight bytes are written, in place (big-endian, when no zero byte comes before the
/// magnitude). Measured on an Apple M4 Max, `!a` took 15 to 16 cycles so, from 17 bytes to
/// 64 KiB, against 18 to 19 with the run and the step of the general case; most of it is the
/// moving of `a` in and out, as num-bigint's `!a` takes as long, and GMP's `mpz_com(a, a)`, in
/// place, 7.
fn not_in<const BE: bool>(a: &mut Number) {
    let n = a.value.len();
    let l = sig_len::<BE>(&a.value);
    let negative = !(a.negative && l > 0);
    if l > 8 && (!BE || l == n) {
        let base = a.value.as_mut_ptr();
        let e = if BE {
            // SAFETY: the end of the vector, `l == n` bytes on.
            unsafe { base.add(n) }
        } else {
            base
        };
        // SAFETY: the eight least significant bytes are significant.
        let w = unsafe { load::<BE>(e, 0) };
        let (low, out) = if negative {
            w.overflowing_add(1)
        } else {
            w.overflowing_sub(1)
        };
        if !out {
            // SAFETY: the same eight bytes, and `l <= n` bytes stay.
            unsafe {
                store::<BE>(e, 0, low);
                a.value.set_len(l);
            }
            a.negative = negative;
            return;
        }
    }
    // SAFETY: `run` reads the significant bytes only.
    let (r, len) = unsafe { run::<BE>(end::<BE>(&a.value), l, negative) };
    if a.value.capacity() < len {
        // Only `0xFF…FF + 1`, one byte longer, which is all zeros but its top byte.
        let (value, negative) = not_copy::<BE>(a);
        a.value = value;
        a.negative = negative;
        return;
    }
    a.negative = negative;
    let base = a.value.as_mut_ptr();
    // SAFETY: the vector has room for the `len` bytes of the result, all of which `step` writes.
    // Big-endian, a result whose length differs from `l` has all its bytes written, as `step`
    // says, so the bytes of `a` need not move; one of the same length as `l` is written over
    // them, moved to the start of the vector first when zero bytes come before them.
    unsafe {
        if BE {
            if len == l && n > l {
                ptr::copy(base.add(n - l), base, l);
            }
            let in_place = len == l;
            step::<true>(
                base.add(len),
                base.add(l).cast_const(),
                l,
                r,
                len,
                negative,
                in_place,
            );
        } else {
            step::<false>(base, base.cast_const(), l, r, len, negative, true);
        }
        a.value.set_len(len);
    }
}

/// For `!a` with a magnitude of `l` significant bytes at the least significant end `src` in the
/// layout `BE`: the position `r` of the lowest byte that the `+ 1` (when `plus`, for a result
/// that is negative) or the `- 1` changes, the bytes below it being `0xFF` or zero, and the
/// length of the result.
///
/// # Safety
///
/// The `l` bytes must be valid for reads.
#[inline(always)]
unsafe fn run<const BE: bool>(src: *const u8, l: usize, plus: bool) -> (usize, usize) {
    // The bytes the run passes through: `0xFF` for `+ 1`, zero for `- 1`.
    let pass = if plus { u64::MAX } else { 0 };
    let mut r = 0;
    // SAFETY: every read is of positions below `l`.
    unsafe {
        while r + 8 <= l && load::<BE>(src, r) == pass {
            r += 8;
        }
        while r < l && get::<BE>(src, r) == pass as u8 {
            r += 1;
        }
        let len = if plus {
            // `0xFF…FF + 1` is one byte longer, and `0 + 1` is 1.
            if r == l { l + 1 } else { l }
        } else if r + 1 == l && get::<BE>(src, r) == 1 {
            // `0x01_00…00 - 1` is one byte shorter, and `1 - 1` is zero, one byte.
            (l - 1).max(1)
        } else {
            l
        };
        (r, len)
    }
}

/// Writes the `len` bytes of `A + 1` (when `plus`) or `A - 1` at the least significant end `dst`
/// in the layout `BE`, for `A` of `l` significant bytes at `src` whose run of `0xFF` (or zero)
/// bytes ends at `r`: those bytes turn to zero (or `0xFF`), byte `r` gains (or loses) one, and the
/// others are copied, unless `in_place`, where they are already there. A result of another
/// length than `l` is all zeros but a top byte of 1, or all `0xFF`, and reads nothing.
///
/// # Safety
///
/// `r` and `len` must be those [`run`] gives. The `l` bytes at `src` must be valid for reads and
/// the `len` bytes at `dst` for writes; `in_place` only when `dst` is `src`, or `len != l`.
#[inline(always)]
unsafe fn step<const BE: bool>(
    dst: *mut u8,
    src: *const u8,
    l: usize,
    r: usize,
    len: usize,
    plus: bool,
    in_place: bool,
) {
    let fill = if plus { 0 } else { 0xFF };
    // SAFETY: as the caller guarantees; every write is of positions below `len`, every read of
    // positions below `l`.
    unsafe {
        if len != l {
            // `0xFF…FF + 1` (or `0 + 1`), all zeros but a top byte of 1, or `0x01_00…00 - 1`,
            // all `0xFF`.
            ptr::write_bytes(at_mut::<BE>(dst, 0, len), fill, len);
            if plus {
                set::<BE>(dst, len - 1, 1);
            }
            return;
        }
        // Of the same length, the run ends inside the magnitude.
        ptr::write_bytes(at_mut::<BE>(dst, 0, r), fill, r);
        let byte = get::<BE>(src, r);
        set::<BE>(dst, r, if plus { byte + 1 } else { byte - 1 });
        if !in_place && l > r + 1 {
            let n = l - r - 1;
            ptr::copy_nonoverlapping(at::<BE>(src, r + 1, n), at_mut::<BE>(dst, r + 1, n), n);
        }
    }
}

// The bits of a number

impl Number {
    /// Returns the number of bits of the magnitude, `|self|.ilog2() + 1`, or 0 for zero: the
    /// position of the highest one plus one, as `BigInt::bits` in num-bigint and
    /// `mpz_sizeinbase(x, 2)` in GMP (which gives 1 for zero).
    ///
    /// It takes constant time for a number in the shortest form: the length of the magnitude and
    /// its top byte decide.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// assert_eq!(Number::new_from_i32(-0x1FF, ByteOrder::BigEndian).bits(), 9);
    /// assert_eq!(Number::new_from_u64(1 << 40, ByteOrder::LittleEndian).bits(), 41);
    /// assert_eq!(Number::new_from_u8(0, ByteOrder::BigEndian).bits(), 0);
    /// ```
    pub fn bits(&self) -> u64 {
        let (l, top) = if self.byte_order == ByteOrder::BigEndian {
            top_byte::<true>(&self.value)
        } else {
            top_byte::<false>(&self.value)
        };
        if l == 0 {
            0
        } else {
            8 * (l as u64 - 1) + u64::from(8 - top.leading_zeros())
        }
    }

    /// Returns the number of ones in the magnitude, as Python's `int.bit_count` and num-bigint's
    /// `BigUint::count_ones` count them, and GMP's `mpz_popcount` for numbers that are not
    /// negative: the sign is not counted. (In two's complement a negative number has infinitely
    /// many ones, which `mpz_popcount` gives as its largest value.)
    ///
    /// The bytes are counted as they are stored, in either byte order, sixteen at a time with
    /// NEON's `cnt` on aarch64.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// assert_eq!(Number::new_from_i32(0b1011, ByteOrder::BigEndian).count_ones(), 3);
    /// // The magnitude of -0b1011 has as many.
    /// assert_eq!(Number::new_from_i32(-0b1011, ByteOrder::LittleEndian).count_ones(), 3);
    /// ```
    pub fn count_ones(&self) -> u64 {
        let v = &self.value;
        if v.len() <= SMALL {
            // As the addition reads a short magnitude, inline; the order of the bytes does not
            // matter.
            // SAFETY: the vector holds at most 16 bytes.
            return u64::from(unsafe { read_small(v, ByteOrder::LittleEndian) }.count_ones());
        }
        // SAFETY: the bytes of the vector are valid for reads.
        unsafe { kernels::count_ones(v.as_ptr(), v.len()) }
    }

    /// Returns the number of zeros below the lowest one of the magnitude, the position of that
    /// one, or `None` for zero, as num-bigint's `BigInt::trailing_zeros` and GMP's `mpz_scan1(x, 0)`
    /// give it. The same for `x` and `-x`: in two's complement, `-x` has its lowest one where
    /// `x` has.
    ///
    /// The bytes are read from the least significant up, eight at a time while they are zero, so
    /// for random numbers the first read decides.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// assert_eq!(Number::new_from_i32(-0x500, ByteOrder::BigEndian).trailing_zeros(), Some(8));
    /// assert_eq!(Number::new_from_u8(0, ByteOrder::LittleEndian).trailing_zeros(), None);
    /// ```
    pub fn trailing_zeros(&self) -> Option<u64> {
        // SAFETY: `lowest` reads the bytes of the vector only.
        unsafe {
            if self.byte_order == ByteOrder::BigEndian {
                lowest::<true>(&self.value)
            } else {
                lowest::<false>(&self.value)
            }
        }
    }

    /// Returns bit `i` of the number in infinite two's complement, as num-bigint's `BigInt::bit`
    /// and GMP's `mpz_tstbit`: for a negative number, `-x` is `!x + 1`, so its bits are those of
    /// `!(|x| - 1)`, and all those above the magnitude are ones.
    ///
    /// For a number that is not negative this takes constant time; for a negative one, the bytes
    /// below bit `i` are read from the nearest down until one is not zero, which for random
    /// numbers is the first.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// // -4 is …11111100.
    /// let n = Number::new_from_i32(-4, ByteOrder::BigEndian);
    /// assert!(!n.bit(0) && !n.bit(1) && n.bit(2) && n.bit(3) && n.bit(1000));
    /// let n = Number::new_from_i32(4, ByteOrder::LittleEndian);
    /// assert!(n.bit(2) && !n.bit(3) && !n.bit(1000));
    /// ```
    pub fn bit(&self, i: u64) -> bool {
        if self.byte_order == ByteOrder::BigEndian {
            bit_of::<true>(self, i)
        } else {
            bit_of::<false>(self, i)
        }
    }

    /// Sets bit `i` of the number in infinite two's complement to `value`, as num-bigint's
    /// `BigInt::set_bit` and GMP's `mpz_setbit` and `mpz_clrbit` do, and stores the result back
    /// as a sign and a magnitude, in the byte order of `self`, in the shortest form.
    ///
    /// For a number that is not negative, the bit of the magnitude is set or cleared. For a
    /// negative one, whose magnitude `X` has its lowest one at position `t`: above `t` the bits are
    /// those of `!X`, so setting bit `i > t` clears it in `X` and clearing it sets it; bit `t` is
    /// one, and clearing it adds `2^t` to `X`; below `t` the bits are zeros, and setting bit `i`
    /// subtracts `2^i` from `X`.
    ///
    /// The vector grows as needed, by `Vec`'s rules, and shrinks to the shortest form; a
    /// big-endian magnitude that grows or shrinks at its top moves its bytes, as its most
    /// significant byte comes first. Otherwise a number in the shortest form costs constant time
    /// (but for the borrows and carries of a negative number, which for random numbers end in the
    /// first byte).
    ///
    /// # Panics
    ///
    /// When the result is too long to be allocated.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// let mut n = Number::new_from_i32(5, ByteOrder::BigEndian);
    /// n.set_bit(9, true);
    /// assert_eq!(n.value, [0x02, 0x05]);
    ///
    /// // -4 is …11111100: setting bit 0 gives …11111101, -3.
    /// let mut n = Number::new_from_i32(-4, ByteOrder::LittleEndian);
    /// n.set_bit(0, true);
    /// assert_eq!(n.value, [3]);
    /// assert!(n.is_negative());
    /// // Clearing bit 2 gives …11111001, -7.
    /// n.set_bit(2, false);
    /// assert_eq!(n.value, [7]);
    /// assert!(n.is_negative());
    /// ```
    #[track_caller]
    pub fn set_bit(&mut self, i: u64, value: bool) {
        if self.byte_order == ByteOrder::BigEndian {
            set_bit_in::<true>(self, i, value);
        } else {
            set_bit_in::<false>(self, i, value);
        }
    }
}

/// The significant length of the magnitude in `v` in the layout `BE`, and its top byte (0 for
/// zero).
#[inline(always)]
fn top_byte<const BE: bool>(v: &[u8]) -> (usize, u8) {
    let l = significant_len::<BE>(v);
    if l == 0 {
        return (0, 0);
    }
    // SAFETY: position `l - 1` is in the vector.
    (l, unsafe { get::<BE>(end::<BE>(v), l - 1) })
}

/// The position of the lowest one of the magnitude in `v` in the layout `BE`, or `None` for
/// zero.
///
/// # Safety
///
/// None beyond the validity of `v`; unsafe for the raw reads inside.
#[inline(always)]
unsafe fn lowest<const BE: bool>(v: &[u8]) -> Option<u64> {
    let (e, n) = (end::<BE>(v), v.len());
    let mut p = 0;
    // SAFETY: every read is of positions below `n`.
    unsafe {
        while p + 8 <= n {
            let w = load::<BE>(e, p);
            if w != 0 {
                return Some(8 * p as u64 + u64::from(w.trailing_zeros()));
            }
            p += 8;
        }
        while p < n {
            let b = get::<BE>(e, p);
            if b != 0 {
                return Some(8 * p as u64 + u64::from(b.trailing_zeros()));
            }
            p += 1;
        }
    }
    None
}

/// Whether any of the positions `0..q` of the magnitude at the least significant end `e` in the
/// layout `BE` is not zero: from `q - 1` down, eight at a time.
///
/// # Safety
///
/// The `q` positions must be valid for reads.
#[inline(always)]
unsafe fn any_below<const BE: bool>(e: *const u8, q: usize) -> bool {
    let mut p = q;
    // SAFETY: every read is of positions below `q`.
    unsafe {
        while p >= 8 {
            if load::<BE>(e, p - 8) != 0 {
                return true;
            }
            p -= 8;
        }
        while p > 0 {
            if get::<BE>(e, p - 1) != 0 {
                return true;
            }
            p -= 1;
        }
    }
    false
}

/// Bit `k` of position `q` of the magnitude at the least significant end `e` in the layout `BE`,
/// and whether any bit below it is set: one read of the eight positions that end with `q`, whose
/// test a random number passes every time, and only when they are all zero below that bit are
/// the positions further down read. (A test of the bits of the byte at `q` alone would be a coin
/// toss for a random `k`.) Measured on an Apple M4 Max, `set_bit` of negative numbers of 1 KiB
/// at random positions took 8 to 10 cycles so, in either byte order, against 8 and 19 with a
/// read of the byte and then of the eight below it, and GMP's `mpz_setbit` 7.
///
/// # Safety
///
/// Positions `0..=q` must be valid for reads.
#[inline(always)]
unsafe fn bit_and_below<const BE: bool>(e: *const u8, q: usize, k: u32) -> (bool, bool) {
    // SAFETY: as the caller guarantees; below position 7, `limb` reads the `q + 1` positions.
    unsafe {
        let (word, s) = if q >= 7 {
            (load::<BE>(e, q - 7), 56 + k)
        } else {
            (limb::<BE>(e, q + 1, 0), 8 * q as u32 + k)
        };
        let own = (word >> s) & 1 == 1;
        let below = word & ((1 << s) - 1) != 0 || (q > 7 && any_below::<BE>(e, q - 7));
        (own, below)
    }
}

/// [`Number::bit`] for a number in the layout `BE`.
#[inline(always)]
fn bit_of<const BE: bool>(a: &Number, i: u64) -> bool {
    let (e, n) = (end::<BE>(&a.value), a.value.len());
    let (q, k) = (i / 8, (i % 8) as u32);
    if q >= n as u64 {
        // Above the magnitude: ones for a negative number, which zero is not.
        return a.negative && significant_len::<BE>(&a.value) > 0;
    }
    let q = q as usize;
    // SAFETY: position `q` is below `n`.
    unsafe {
        if !a.negative {
            return (get::<BE>(e, q) >> k) & 1 == 1;
        }
        // Bit `i` of `-x` is that of `x` up to its lowest one, and the opposite above it.
        let (own, below) = bit_and_below::<BE>(e, q, k);
        own ^ below
    }
}

/// [`Number::set_bit`] for a number in the layout `BE`.
#[track_caller]
fn set_bit_in<const BE: bool>(a: &mut Number, i: u64, value: bool) {
    let l = sig_len::<BE>(&a.value);
    if l != a.value.len() || l == 0 {
        // The shortest form first: the bytes above the magnitude go, and a zero is not negative.
        trim::<BE>(&mut a.value, l);
        if l == 0 {
            a.negative = false;
        }
    }
    if !a.negative {
        if value {
            set_magnitude_bit::<BE>(&mut a.value, i);
        } else {
            clear_magnitude_bit::<BE>(&mut a.value, i);
        }
        return;
    }
    // A negative number's bit `i` is that of `!(x - 1)`: the opposite of its magnitude's above
    // the lowest one, which is at `i` when no bit below is set and bit `i` is.
    let (e, n) = (end::<BE>(&a.value), a.value.len());
    let (q, k) = (i / 8, (i % 8) as u32);
    let (below, own) = if q >= n as u64 {
        (true, false)
    } else {
        // SAFETY: position `q` is below `n`.
        let (own, below) = unsafe { bit_and_below::<BE>(e, q as usize, k) };
        (below, own)
    };
    match (below, own, value) {
        // Above the lowest one, the bits are those of `!x`.
        (true, _, true) => clear_magnitude_bit::<BE>(&mut a.value, i),
        (true, _, false) => set_magnitude_bit::<BE>(&mut a.value, i),
        // The lowest one, cleared: `-x - 2^i`, a magnitude `x + 2^i`.
        (false, true, false) => add_power::<BE>(&mut a.value, i),
        // Below it, set: `-x + 2^i`, a magnitude `x - 2^i`.
        (false, false, true) => subtract_power::<BE>(&mut a.value, i),
        _ => {}
    }
}

/// Drops the bytes of `v` above its significant length `l`, keeping one byte for zero.
fn trim<const BE: bool>(v: &mut Vec<u8>, l: usize) {
    let keep = l.max(1);
    if v.len() > keep {
        if BE {
            v.drain(..v.len() - keep);
        } else {
            v.truncate(keep);
        }
    } else if v.is_empty() {
        v.push(0);
    }
}

/// Grows the magnitude in `v`, in the shortest form, with zero bytes above it to `len` bytes:
/// big-endian, its bytes move up to make room for them at the start.
///
/// # Panics
///
/// When it cannot be allocated.
#[track_caller]
fn grow<const BE: bool>(v: &mut Vec<u8>, len: usize) {
    let (n, more) = (v.len(), len - v.len());
    if v.try_reserve(more).is_err() {
        too_long(len);
    }
    // SAFETY: the vector has room for `len` bytes; big-endian its `n` bytes move up by `more`,
    // and the `more` bytes before them, all of the new ones, are set to zero before the length is.
    unsafe {
        let p = v.as_mut_ptr();
        if BE {
            ptr::copy(p, p.add(more), n);
            ptr::write_bytes(p, 0, more);
        } else {
            ptr::write_bytes(p.add(n), 0, more);
        }
        v.set_len(len);
    }
}

/// Panics for a `set_bit` whose result cannot be allocated.
#[cold]
#[inline(never)]
#[track_caller]
fn too_long(len: usize) -> ! {
    panic!("a number of {len} bytes cannot be allocated")
}

/// Panics for a `set_bit` at a position beyond any vector.
#[cold]
#[inline(never)]
#[track_caller]
fn too_far(i: u64) -> ! {
    panic!("a number with bit {i} set cannot be allocated")
}

/// Sets bit `i` of the magnitude in `v`, in the shortest form, growing it as needed.
#[track_caller]
fn set_magnitude_bit<const BE: bool>(v: &mut Vec<u8>, i: u64) {
    let n = v.len();
    let q = i / 8;
    if q >= n as u64 {
        let Some(len) = usize::try_from(q).ok().and_then(|q| q.checked_add(1)) else {
            too_far(i)
        };
        grow::<BE>(v, len);
    }
    let n = v.len();
    // SAFETY: position `q` is in the vector.
    unsafe {
        let e = v.as_mut_ptr().add(if BE { n } else { 0 });
        let q = q as usize;
        set::<BE>(e, q, get::<BE>(e, q) | (1 << (i % 8)));
    }
}

/// Clears bit `i` of the magnitude in `v`, in the shortest form, shrinking it to the shortest
/// form.
fn clear_magnitude_bit<const BE: bool>(v: &mut Vec<u8>, i: u64) {
    let n = v.len();
    let q = i / 8;
    if q >= n as u64 {
        return;
    }
    let q = q as usize;
    // SAFETY: position `q` is in the vector.
    let byte = unsafe {
        let e = v.as_mut_ptr().add(if BE { n } else { 0 });
        let byte = get::<BE>(e, q) & !(1 << (i % 8));
        set::<BE>(e, q, byte);
        byte
    };
    if byte == 0 && q + 1 == n {
        let l = significant_len::<BE>(v);
        trim::<BE>(v, l);
    }
}

/// Adds `2^t` to the magnitude in `v`, in the shortest form, whose bit `t` is set: the carry runs
/// through the bytes above that are `0xFF`, and past the top makes it one byte longer.
#[track_caller]
fn add_power<const BE: bool>(v: &mut Vec<u8>, t: u64) {
    let n = v.len();
    let mut q = (t / 8) as usize;
    // SAFETY: every position read and written is below `n`.
    unsafe {
        let e = v.as_mut_ptr().add(if BE { n } else { 0 });
        let (byte, carry) = get::<BE>(e, q).overflowing_add(1 << (t % 8));
        set::<BE>(e, q, byte);
        if !carry {
            return;
        }
        q += 1;
        while q < n && get::<BE>(e, q) == 0xFF {
            set::<BE>(e, q, 0);
            q += 1;
        }
        if q < n {
            set::<BE>(e, q, get::<BE>(e, q) + 1);
            return;
        }
    }
    grow::<BE>(v, n + 1);
    // SAFETY: position `n` is the new top byte.
    unsafe {
        let e = v.as_mut_ptr().add(if BE { n + 1 } else { 0 });
        set::<BE>(e, n, 1);
    }
}

/// Subtracts `2^i` from the magnitude in `v`, in the shortest form, whose bits below `i` are
/// zero and which is above `2^i`: the borrow runs through the zero bytes above, and the result is
/// left in the shortest form.
fn subtract_power<const BE: bool>(v: &mut Vec<u8>, i: u64) {
    let n = v.len();
    let mut q = (i / 8) as usize;
    // SAFETY: every position read and written is below `n`: the magnitude is above `2^i`, so the
    // borrow ends inside it.
    unsafe {
        let e = v.as_mut_ptr().add(if BE { n } else { 0 });
        let (byte, borrow) = get::<BE>(e, q).overflowing_sub(1 << (i % 8));
        set::<BE>(e, q, byte);
        if borrow {
            q += 1;
            while get::<BE>(e, q) == 0 {
                set::<BE>(e, q, 0xFF);
                q += 1;
            }
            set::<BE>(e, q, get::<BE>(e, q) - 1);
        }
    }
    if q + 1 == n {
        let l = significant_len::<BE>(v);
        trim::<BE>(v, l);
    }
}

// Positions

/// The least significant end of the magnitude in `v` in the layout `BE`: the address of its
/// first byte little-endian, and the address one past its last byte big-endian.
#[inline(always)]
fn end<const BE: bool>(v: &[u8]) -> *const u8 {
    if BE { v.as_ptr_range().end } else { v.as_ptr() }
}

/// The lowest address of positions `p..p + n` of the magnitude at the least significant end
/// `end` in the layout `BE`: `end + p` little-endian, `end - p - n` big-endian.
///
/// # Safety
///
/// The address must lie in the allocation of `end`.
#[inline(always)]
unsafe fn at<const BE: bool>(end: *const u8, p: usize, n: usize) -> *const u8 {
    // SAFETY: as the caller guarantees.
    unsafe { if BE { end.sub(p + n) } else { end.add(p) } }
}

/// [`at`] for writing.
///
/// # Safety
///
/// As for [`at`].
#[inline(always)]
unsafe fn at_mut<const BE: bool>(end: *mut u8, p: usize, n: usize) -> *mut u8 {
    // SAFETY: as the caller guarantees.
    unsafe { if BE { end.sub(p + n) } else { end.add(p) } }
}

/// Reads position `p` of the magnitude at the least significant end `end` in the layout `BE`.
///
/// # Safety
///
/// The byte must be valid for reading.
#[inline(always)]
unsafe fn get<const BE: bool>(end: *const u8, p: usize) -> u8 {
    // SAFETY: as the caller guarantees.
    unsafe { *at::<BE>(end, p, 1) }
}

/// Writes position `p` of the magnitude at the least significant end `end` in the layout `BE`.
///
/// # Safety
///
/// The byte must be valid for writing.
#[inline(always)]
unsafe fn set<const BE: bool>(end: *mut u8, p: usize, v: u8) {
    // SAFETY: as the caller guarantees.
    unsafe { *at_mut::<BE>(end, p, 1) = v };
}

/// Reads positions `p..p + 8` of the magnitude at the least significant end `end` in the layout
/// `BE` as a `u64` whose least significant byte is position `p`.
///
/// # Safety
///
/// The eight bytes must be valid for reading; they need no alignment.
#[inline(always)]
unsafe fn load<const BE: bool>(end: *const u8, p: usize) -> u64 {
    // SAFETY: as the caller guarantees.
    unsafe {
        let bytes = read(at::<BE>(end, p, 8));
        if BE {
            u64::from_be_bytes(bytes)
        } else {
            u64::from_le_bytes(bytes)
        }
    }
}

/// Writes positions `p..p + 8` of the magnitude at the least significant end `end` in the layout
/// `BE`, as [`load`] reads them.
///
/// # Safety
///
/// The eight bytes must be valid for writing; they need no alignment.
#[inline(always)]
unsafe fn store<const BE: bool>(end: *mut u8, p: usize, v: u64) {
    // SAFETY: as the caller guarantees.
    unsafe {
        let bytes = if BE { v.to_be_bytes() } else { v.to_le_bytes() };
        write(at_mut::<BE>(end, p, 8), bytes);
    }
}

/// Limb `i`, positions `8i..8i + 8`, of the magnitude of `l` significant bytes at the least
/// significant end `end` in the layout `BE`, its positions from `l` up zero: a limb inside it as
/// it is, the top one from the eight positions that end at `l` (or, below 8, as the addition
/// reads a short magnitude), shifted down.
///
/// # Safety
///
/// The `l` bytes must be valid for reads.
#[inline(always)]
unsafe fn limb<const BE: bool>(end: *const u8, l: usize, i: usize) -> u64 {
    let p = 8 * i;
    // SAFETY: every read is of positions below `l`.
    unsafe {
        if p + 8 <= l {
            load::<BE>(end, p)
        } else if p >= l {
            0
        } else if l >= 8 {
            load::<BE>(end, l - 8) >> (8 * (p + 8 - l))
        } else {
            let order = if BE {
                ByteOrder::BigEndian
            } else {
                ByteOrder::LittleEndian
            };
            read_small(&*ptr::slice_from_raw_parts(at::<BE>(end, 0, l), l), order) as u64
        }
    }
}

/// Writes limb `i` of a result of `len` bytes at the least significant end `end` in the layout
/// `BE`: its positions below `len`.
///
/// # Safety
///
/// `8 i < len`, and the `len` bytes must be valid for writes.
#[inline(always)]
unsafe fn put_limb<const BE: bool>(end: *mut u8, len: usize, i: usize, v: u64) {
    let p = 8 * i;
    // SAFETY: as the caller guarantees; a part of a limb, `n` from 1 to 7 bytes, is written as
    // `write_small` writes so short a sum.
    unsafe {
        if p + 8 <= len {
            store::<BE>(end, p, v);
        } else {
            let n = len - p;
            let order = if BE {
                ByteOrder::BigEndian
            } else {
                ByteOrder::LittleEndian
            };
            write_small(at_mut::<BE>(end, p, n), order, v, 0, n);
        }
    }
}
