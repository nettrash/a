//! Subtraction of [`Number`]s: `-` and `-=`, with either operand owned or borrowed.
//!
//! `a - b` is `a + (-b)`, and it is computed as that sum, by the addition and with everything it
//! does: the short path in registers for operands of at most 16 bytes, the plan of the exact
//! length, and the kernels in inline assembly on `x86_64` (`sbb` and `adc` chains through the
//! carry flag) and on 64-bit little-endian `aarch64` (`sbcs` and `adcs`, in several interleaved
//! chains for long operands), portable Rust elsewhere. So a difference costs what a sum of the
//! same magnitudes does, and the operators keep the promises of `+` and `+=`: the difference is
//! exact, in the shortest form and in the byte order of the left operand, and it is built in
//! the vector of an operand taken by value, or of the target of `-=`, when that has room for it
//! (the left operand's first), and otherwise in one new vector of exactly its length. The
//! magnitudes are subtracted when the signs of `a` and `-b` differ, as for `5 - 3`, and added
//! when they agree, as for `5 - (-3)`.
//!
//! The sign of `b` is turned over without changing the code of the addition:
//!
//! - Operands of at most 16 bytes take the short path of the addition, inlined, with the sign of
//!   `b` turned over as it is read into a register.
//! - An owned `b` has its sign turned over in place, and is then added.
//! - `a -= &b` on longer operands cannot change `b`, so it turns over the sign of `a` instead:
//!   `a - b` is `-(-a + b)`. The sign of `a` is turned over, `b` is added, and the sign of the
//!   sum is turned back, unless the sum is zero, which stays not negative. Only the sign of `a`
//!   goes through memory on its way to the addition, and it decides only branches, which the
//!   core predicts; the vectors of both operands are read where they are.
//! - `&a - &b` on longer operands can change neither, so it adds to `a` a copy of the fields of
//!   `b` with the sign turned over, which shares the vector of `b` and is only read (see
//!   [`Negated`]). The addition reads the vector's address and length from that copy, a store
//!   and a load later than from `b` itself, which the allocation of the difference's vector
//!   hides; a long difference built in place, whose kernel starts before the plan is done, would
//!   wait for them.

use super::Number;
use super::add::{add_assign_ref, ref_plus_ref, small, small_add_to, small_number_of};
use core::marker::PhantomData;
use core::mem::ManuallyDrop;
use core::ops::{Deref, Sub, SubAssign};
use core::ptr;

impl Sub<&Number> for &Number {
    type Output = Number;

    /// Returns `self - rhs`, in the byte order of `self`, in the shortest form.
    ///
    /// The difference is built in one new vector of exactly its length (one byte for zero), the
    /// only allocation; both operands are left as they were. The operands may have any signs,
    /// byte orders and lengths, and need not be in the shortest form.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// // 0x01_00 - 0x01_01 = -1, in the byte order of the left operand.
    /// let a = Number::new_from_i32(0x0100, ByteOrder::BigEndian);
    /// let b = Number::new_from_i32(0x0101, ByteOrder::LittleEndian);
    /// let difference = &a - &b;
    /// assert_eq!(difference.value, [1]);
    /// assert!(difference.is_negative());
    /// assert!(difference.byte_order() == ByteOrder::BigEndian);
    ///
    /// // A number minus itself is zero, one zero byte and not negative.
    /// let zero = &a - &a;
    /// assert_eq!(zero.value, [0]);
    /// assert!(!zero.is_negative());
    /// ```
    fn sub(self, rhs: &Number) -> Number {
        if small(self, rhs) {
            return small_number_of(self, rhs, !rhs.negative);
        }
        ref_plus_ref(self, &Negated::new(rhs))
    }
}

impl Sub<&Number> for Number {
    type Output = Number;

    /// Returns `self - rhs`, in the byte order of `self`, in the shortest form, as
    /// `self -= rhs` does.
    ///
    /// The difference is built in the vector of `self` when its capacity has room for it, which
    /// allocates nothing, and otherwise in one new vector of exactly its length. `rhs` is left
    /// as it was.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// let a = Number::new(vec![1, 0, 0], ByteOrder::BigEndian);
    /// let pointer = a.value.as_ptr();
    /// let one = Number::new_from_u8(1, ByteOrder::BigEndian);
    ///
    /// // 0x01_00_00 - 1 = 0xFF_FF, in the vector of `a`.
    /// let difference = a - &one;
    /// assert_eq!(difference.value, [0xFF, 0xFF]);
    /// assert_eq!(difference.value.as_ptr(), pointer);
    /// ```
    fn sub(mut self, rhs: &Number) -> Number {
        self -= rhs;
        self
    }
}

impl Sub<Number> for &Number {
    type Output = Number;

    /// Returns `self - rhs`, in the byte order of `self`, in the shortest form.
    ///
    /// The difference is built in the vector of `rhs` when its capacity has room for it, which
    /// allocates nothing, also when `rhs` is in the other byte order; otherwise it gets one new
    /// vector of exactly its length. `self` is left as it was.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// let a = Number::new_from_u32(0x01_02_03, ByteOrder::BigEndian);
    /// let mut value = Vec::with_capacity(3);
    /// value.extend([0x10, 0x20]);
    /// let b = Number::new(value, ByteOrder::LittleEndian);
    /// let pointer = b.value.as_ptr();
    ///
    /// // 0x01_02_03 - 0x20_10 = 0xE1_F3, big-endian, in the vector of `b`.
    /// let difference = &a - b;
    /// assert_eq!(difference.value, [0xE1, 0xF3]);
    /// assert_eq!(difference.value.as_ptr(), pointer);
    /// ```
    fn sub(self, mut rhs: Number) -> Number {
        rhs.negative = !rhs.negative;
        self + rhs
    }
}

impl Sub<Number> for Number {
    type Output = Number;

    /// Returns `self - rhs`, in the byte order of `self`, in the shortest form, as
    /// `self -= rhs` does.
    ///
    /// The difference is built in the vector of `self` when its capacity has room for it, or
    /// else in the vector of `rhs` when that has room, also when `rhs` is in the other byte
    /// order, either of which allocates nothing; otherwise it gets one new vector of exactly
    /// its length.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// // 0x01_00 - 0x02_FF = -0x01_FF.
    /// let a = Number::new_from_i32(0x0100, ByteOrder::LittleEndian);
    /// let b = Number::new_from_i32(0x02FF, ByteOrder::BigEndian);
    /// let difference = a - b;
    /// assert_eq!(difference.value, [0xFF, 0x01]);
    /// assert!(difference.is_negative());
    /// ```
    fn sub(mut self, rhs: Number) -> Number {
        self -= rhs;
        self
    }
}

impl SubAssign<&Number> for Number {
    /// Subtracts `rhs` from `self`, keeping the byte order of `self`, in the shortest form.
    ///
    /// The difference is built in the vector of `self` when its capacity has room for it, which
    /// allocates nothing, and otherwise in one new vector of exactly its length, which replaces
    /// it. `rhs` is left as it was.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// let mut n = Number::new_from_u16(0x0200, ByteOrder::LittleEndian);
    /// n -= &Number::new_from_u8(1, ByteOrder::BigEndian);
    /// assert_eq!(n.value, [0xFF, 0x01]);
    /// n -= &Number::new_from_u16(0x0200, ByteOrder::BigEndian);
    /// assert_eq!(n.value, [1]);
    /// assert!(n.is_negative());
    /// ```
    #[inline]
    fn sub_assign(&mut self, rhs: &Number) {
        if small(self, rhs) {
            small_add_to(self, rhs, !rhs.negative);
            return;
        }
        // a - b = -(-a + b). A zero `a` turned negative is zero to the addition, and a zero sum
        // is `[0]` and not negative, which it stays. The sign of the sum is turned back without
        // a branch on it, a coin toss for random operands of the same length; the branch on the
        // length is not, as a sum this long is one byte long only when its operands nearly
        // cancel, and then only is its byte read, which may still be on its way to memory.
        self.negative = !self.negative;
        add_assign_ref(self, rhs);
        let zero = self.value.len() == 1 && self.value[0] == 0;
        self.negative = !self.negative & !zero;
    }
}

impl SubAssign<Number> for Number {
    /// Subtracts `rhs` from `self`, keeping the byte order of `self`, in the shortest form.
    ///
    /// The difference is built in the vector of `self` when its capacity has room for it, or
    /// else in the vector of `rhs` when that has room, also when `rhs` is in the other byte
    /// order, either of which allocates nothing; otherwise it gets one new vector of exactly
    /// its length.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// // `n` has no room for 0xFF - (-1) = 0x01_00, but the vector of the right operand has.
    /// let mut n = Number::new_from_u8(0xFF, ByteOrder::BigEndian);
    /// let mut value = Vec::with_capacity(2);
    /// value.push(1);
    /// let minus_one = Number::new_signed(value, true, ByteOrder::BigEndian);
    /// let pointer = minus_one.value.as_ptr();
    /// n -= minus_one;
    /// assert_eq!(n.value, [1, 0]);
    /// assert_eq!(n.value.as_ptr(), pointer);
    /// ```
    #[inline]
    fn sub_assign(&mut self, mut rhs: Number) {
        rhs.negative = !rhs.negative;
        *self += rhs;
    }
}

/// A borrowed number with the opposite sign, for the addition: a copy of its fields that shares
/// its vector, which it lets be read only and never drops.
struct Negated<'a> {
    number: ManuallyDrop<Number>,
    borrowed: PhantomData<&'a Number>,
}

impl<'a> Negated<'a> {
    /// `n` with the opposite sign, for as long as `n` is borrowed.
    #[inline(always)]
    fn new(n: &'a Number) -> Self {
        Negated {
            number: ManuallyDrop::new(Number {
                // SAFETY: the copy shares the vector of `n`, which stays borrowed, and so
                // unchanged and not freed, while it lives. It is never written through: `Negated`
                // gives out shared references only. It is never dropped: the vector is freed
                // once, by `n`.
                value: unsafe { ptr::read(&n.value) },
                negative: !n.negative,
                byte_order: n.byte_order,
            }),
            borrowed: PhantomData,
        }
    }
}

impl Deref for Negated<'_> {
    type Target = Number;

    #[inline(always)]
    fn deref(&self) -> &Number {
        &self.number
    }
}
