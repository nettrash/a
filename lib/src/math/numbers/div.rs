//! Division of [`Number`]s: `/` and `/=` for the quotient and `%` and `%=` for the remainder, with
//! either operand owned or borrowed, and [`Number::div_rem`], [`Number::div_euclid`] and
//! [`Number::rem_euclid`].
//!
//! The rules are those of Rust's primitive integers. The quotient `a / b` is truncated toward
//! zero, and the remainder `a % b` is what is left, `a - (a / b) b`: it has the sign of `a` and a
//! magnitude below that of `b`, so `-7 / 2 = -3` and `-7 % 2 = -1`. `div_euclid` and `rem_euclid`
//! give Euclid's division instead, whose remainder is never negative: `-7 = 2 (-4) + 1`. Dividing by
//! zero panics, as it does for `i32`. A zero magnitude is zero whatever `negative` says, and zero
//! bytes above the most significant one count for nothing.
//!
//! Operands whose vectors both hold at most 16 bytes take a short path in registers: their
//! magnitudes are read into `u128`s as the addition reads them and divided, and the result is
//! written in the byte order of the left operand. A dividend of up to 64 bytes by a divisor of up
//! to 16 is divided in registers too, by the steps below.
//!
//! Longer operands are read into limbs of eight bytes, native `u64`s, the least significant first,
//! in working memory on the stack, or on the heap for the longest. Both are shifted left by the
//! same number of bits, so that the divisor's top bit is set: its top limbs then give each limb of
//! the quotient by a multiplication with their precomputed reciprocal (see [`limbs`]). A divisor of
//! one limb or two divides limb by limb; a longer one by the schoolbook method, each quotient limb
//! times the divisor subtracted from the partial remainder by kernels in inline assembly on 64-bit
//! little-endian `aarch64` and, with BMI2, on `x86_64`, and portable Rust elsewhere and under Miri;
//! a divisor of 40 limbs or more by divide and conquer, on the multiplication. The remainder is
//! shifted back, and the quotient and the remainder are written as the operators promise. `/`
//! computes the quotient alone, from the top limbs of the operands when it is much shorter than
//! the divisor, and `%` by a divisor of one limb the remainder alone.
//!
//! A quotient is never longer than the dividend, nor a remainder than the divisor or the dividend,
//! so the dividend's vector, when it is taken by value or is the target of `/=` or `%=`, has room
//! for the result, except for zero in an empty vector, and so has the divisor's for a remainder.
//!
//! [`Number::div_exact`] and [`Number::is_multiple_of`] are in [`exact`], and a divisor prepared
//! for many divisions, [`Divisor`](super::Divisor), divides by the same steps.

// The kernel of the schoolbook method under the name `kernels`, chosen as those of the addition
// are: assembly on 64-bit little-endian aarch64 where the system allows unaligned accesses and on
// x86_64, and portable Rust elsewhere and under Miri.
mod exact;
#[cfg(all(
    target_arch = "aarch64",
    target_endian = "little",
    target_pointer_width = "64",
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
#[path = "div/aarch64.rs"]
mod kernels;
#[cfg(all(target_arch = "x86_64", not(miri)))]
#[path = "div/x86_64.rs"]
mod kernels;
#[cfg(not(any(
    all(
        target_arch = "aarch64",
        target_endian = "little",
        target_pointer_width = "64",
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
    ),
    all(target_arch = "x86_64", not(miri))
)))]
#[path = "div/portable.rs"]
mod kernels;
mod limbs;

pub(super) use limbs::{div_2by1 as divide_2by1, invert_limb as reciprocal};

use super::add::{SMALL, read, read_small, small, write_small};
use super::mul::limbs::{add_1, sub_n};
use super::mul::{
    Operand, Product, STACK_LIMBS, WIDE, Wide, on_heap, on_stack, put_wide, read_wide,
};
use super::{ByteOrder, Number};
use core::cmp::Ordering::Less;
use core::mem::{self, MaybeUninit};
use core::ops::{Div, DivAssign, Rem, RemAssign};
use core::ptr;
use limbs::{
    DC_DIV, DC_DIV_Q, Found, check_itch, check_q_top, div_2by1, div_3by2, div_q, div_q_split,
    div_q_top, div_qr, div_qr_1, div_qr_2, from_top, invert_3by2, invert_limb, q_itch, qr_itch,
    rshift, split_itch, split_low, top_itch,
};

impl Div<&Number> for &Number {
    type Output = Number;

    /// Returns `self / rhs`, truncated toward zero, in the byte order of `self`, in the shortest
    /// form.
    ///
    /// The quotient is built in one new vector of exactly its length (one byte for zero); both
    /// operands are left as they were. The operands may have any signs, byte orders and lengths,
    /// and need not be in the shortest form.
    ///
    /// # Panics
    ///
    /// When `rhs` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// // -0x01_00 / 3 = -0x55, truncated toward zero, in the byte order of the left operand.
    /// let a = Number::new_from_i32(-0x0100, ByteOrder::BigEndian);
    /// let b = Number::new_from_i32(3, ByteOrder::LittleEndian);
    /// let quotient = &a / &b;
    /// assert_eq!(quotient.value, [0x55]);
    /// assert!(quotient.is_negative());
    /// assert!(quotient.byte_order() == ByteOrder::BigEndian);
    /// ```
    #[track_caller]
    fn div(self, rhs: &Number) -> Number {
        if small(self, rhs) {
            let (q, _) = small_division::<TRUNCATED, false>(self, rhs);
            return small_number(self.byte_order, q);
        }
        ref_by_ref::<TRUNCATED, false>(self, rhs)
    }
}

impl Div<&Number> for Number {
    type Output = Number;

    /// Returns `self / rhs`, truncated toward zero, in the byte order of `self`, in the shortest
    /// form, as `self /= rhs` does.
    ///
    /// The quotient is built in the vector of `self`, which has room for it unless `self` is an
    /// empty vector; `rhs` is left as it was.
    ///
    /// # Panics
    ///
    /// When `rhs` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// let a = Number::new(vec![0x01, 0x00, 0x00], ByteOrder::BigEndian);
    /// let pointer = a.value.as_ptr();
    ///
    /// // 0x01_00_00 / 0x01_01 = 0xFF, in the vector of `a`.
    /// let quotient = a / &Number::new_from_u16(0x0101, ByteOrder::LittleEndian);
    /// assert_eq!(quotient.value, [0xFF]);
    /// assert_eq!(quotient.value.as_ptr(), pointer);
    /// ```
    #[track_caller]
    fn div(mut self, rhs: &Number) -> Number {
        self /= rhs;
        self
    }
}

impl Div<Number> for &Number {
    type Output = Number;

    /// Returns `self / rhs`, truncated toward zero, in the byte order of `self`, in the shortest
    /// form.
    ///
    /// The quotient is built in the vector of `rhs` when its capacity has room for it, also when
    /// `rhs` is in the other byte order; otherwise it gets one new vector of exactly its length.
    /// `self` is left as it was.
    ///
    /// # Panics
    ///
    /// When `rhs` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// let a = Number::new_from_u16(0x0300, ByteOrder::BigEndian);
    /// let mut value = Vec::with_capacity(2);
    /// value.push(3);
    /// let b = Number::new(value, ByteOrder::LittleEndian);
    /// let pointer = b.value.as_ptr();
    ///
    /// // 0x03_00 / 3 = 0x01_00, big-endian, in the vector of `b`.
    /// let quotient = &a / b;
    /// assert_eq!(quotient.value, [0x01, 0x00]);
    /// assert_eq!(quotient.value.as_ptr(), pointer);
    /// ```
    #[track_caller]
    fn div(self, mut rhs: Number) -> Number {
        if small(self, &rhs) {
            let (q, _) = small_division::<TRUNCATED, false>(self, &rhs);
            put_small(&mut rhs.value, self.byte_order, q);
            return Number {
                value: rhs.value,
                negative: q.negative,
                byte_order: self.byte_order,
            };
        }
        // `rhs` is dropped on return, with the vector it holds then, if any.
        ref_by_value::<false>(self, &mut rhs)
    }
}

impl Div<Number> for Number {
    type Output = Number;

    /// Returns `self / rhs`, truncated toward zero, in the byte order of `self`, in the shortest
    /// form, as `self /= rhs` does.
    ///
    /// The quotient is built in the vector of `self`, which has room for it unless `self` is an
    /// empty vector, and then in the vector of `rhs` when that has room.
    ///
    /// # Panics
    ///
    /// When `rhs` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// // -7 / -2 = 3.
    /// let a = Number::new_from_i32(-7, ByteOrder::LittleEndian);
    /// let b = Number::new_from_i32(-2, ByteOrder::BigEndian);
    /// let quotient = a / b;
    /// assert_eq!(quotient.value, [3]);
    /// assert!(!quotient.is_negative());
    /// ```
    #[track_caller]
    fn div(mut self, rhs: Number) -> Number {
        self /= rhs;
        self
    }
}

impl DivAssign<&Number> for Number {
    /// Divides `self` by `rhs`, truncating toward zero, keeping the byte order of `self`, in the
    /// shortest form.
    ///
    /// The quotient is built in the vector of `self`, which has room for it unless `self` is an
    /// empty vector; `rhs` is left as it was.
    ///
    /// # Panics
    ///
    /// When `rhs` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// let mut n = Number::new_from_u16(0x1000, ByteOrder::LittleEndian);
    /// n /= &Number::new_from_u8(0x10, ByteOrder::BigEndian);
    /// assert_eq!(n.value, [0x00, 0x01]);
    /// n /= &Number::new_from_i8(-3, ByteOrder::BigEndian);
    /// assert_eq!(n.value, [0x55]);
    /// assert!(n.is_negative());
    /// ```
    #[inline]
    #[track_caller]
    fn div_assign(&mut self, rhs: &Number) {
        if small(self, rhs) {
            let (q, _) = small_division::<TRUNCATED, false>(self, rhs);
            put_small(&mut self.value, self.byte_order, q);
            self.negative = q.negative;
            return;
        }
        assign_ref::<false>(self, rhs);
    }
}

impl DivAssign<Number> for Number {
    /// Divides `self` by `rhs`, truncating toward zero, keeping the byte order of `self`, in the
    /// shortest form.
    ///
    /// The quotient is built in the vector of `self`, which has room for it unless `self` is an
    /// empty vector, and then in the vector of `rhs` when that has room.
    ///
    /// # Panics
    ///
    /// When `rhs` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// let mut n = Number::new_from_u32(1_000_000, ByteOrder::BigEndian);
    /// n /= Number::new_from_u16(1000, ByteOrder::BigEndian);
    /// assert_eq!(n.value, [0x03, 0xE8]);
    /// ```
    #[inline]
    #[track_caller]
    fn div_assign(&mut self, mut rhs: Number) {
        if small(self, &rhs) {
            let (q, _) = small_division::<TRUNCATED, false>(self, &rhs);
            if self.value.capacity() < q.len && rhs.value.capacity() >= q.len {
                // The vector of `self` goes to `rhs`, and is dropped with it.
                mem::swap(&mut self.value, &mut rhs.value);
            }
            put_small(&mut self.value, self.byte_order, q);
            self.negative = q.negative;
            return;
        }
        // `rhs` is dropped on return, with the vector it holds then.
        assign_value::<false>(self, &mut rhs);
    }
}

impl Rem<&Number> for &Number {
    type Output = Number;

    /// Returns `self % rhs`, the remainder of the division truncated toward zero, in the byte
    /// order of `self`, in the shortest form: it has the sign of `self`, unless it is zero, and a
    /// magnitude below that of `rhs`.
    ///
    /// The remainder is built in one new vector of exactly its length (one byte for zero); both
    /// operands are left as they were.
    ///
    /// # Panics
    ///
    /// When `rhs` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// // -7 % 2 = -1 and 7 % -2 = 1, as for `i32`.
    /// let two = Number::new_from_i32(2, ByteOrder::BigEndian);
    /// let remainder = &Number::new_from_i32(-7, ByteOrder::BigEndian) % &two;
    /// assert_eq!(remainder.value, [1]);
    /// assert!(remainder.is_negative());
    ///
    /// let minus_two = Number::new_from_i32(-2, ByteOrder::BigEndian);
    /// let remainder = &Number::new_from_i32(7, ByteOrder::BigEndian) % &minus_two;
    /// assert_eq!(remainder.value, [1]);
    /// assert!(!remainder.is_negative());
    /// ```
    #[track_caller]
    fn rem(self, rhs: &Number) -> Number {
        if small(self, rhs) {
            let (_, r) = small_division::<TRUNCATED, true>(self, rhs);
            return small_number(self.byte_order, r);
        }
        ref_by_ref::<TRUNCATED, true>(self, rhs)
    }
}

impl Rem<&Number> for Number {
    type Output = Number;

    /// Returns `self % rhs` as `&self % rhs` does, as `self %= rhs` does.
    ///
    /// The remainder is built in the vector of `self`, which has room for it unless `self` is an
    /// empty vector; `rhs` is left as it was.
    ///
    /// # Panics
    ///
    /// When `rhs` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// let a = Number::new_from_u32(1_000_003, ByteOrder::LittleEndian);
    /// let pointer = a.value.as_ptr();
    /// let remainder = a % &Number::new_from_u16(1000, ByteOrder::BigEndian);
    /// assert_eq!(remainder.value, [3]);
    /// assert_eq!(remainder.value.as_ptr(), pointer);
    /// ```
    #[track_caller]
    fn rem(mut self, rhs: &Number) -> Number {
        self %= rhs;
        self
    }
}

impl Rem<Number> for &Number {
    type Output = Number;

    /// Returns `self % rhs` as `&self % &rhs` does.
    ///
    /// The remainder is built in the vector of `rhs`, which always has room for it, as it is
    /// shorter than `rhs`, also when `rhs` is in the other byte order. `self` is left as it was.
    ///
    /// # Panics
    ///
    /// When `rhs` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// let a = Number::new_from_u32(0x01_02_03, ByteOrder::BigEndian);
    /// let b = Number::new_from_u16(0x0100, ByteOrder::LittleEndian);
    /// let pointer = b.value.as_ptr();
    ///
    /// // 0x01_02_03 % 0x01_00 = 3, big-endian, in the vector of `b`.
    /// let remainder = &a % b;
    /// assert_eq!(remainder.value, [3]);
    /// assert_eq!(remainder.value.as_ptr(), pointer);
    /// ```
    #[track_caller]
    fn rem(self, mut rhs: Number) -> Number {
        if small(self, &rhs) {
            let (_, r) = small_division::<TRUNCATED, true>(self, &rhs);
            put_small(&mut rhs.value, self.byte_order, r);
            return Number {
                value: rhs.value,
                negative: r.negative,
                byte_order: self.byte_order,
            };
        }
        ref_by_value::<true>(self, &mut rhs)
    }
}

impl Rem<Number> for Number {
    type Output = Number;

    /// Returns `self % rhs` as `&self % &rhs` does, as `self %= rhs` does.
    ///
    /// The remainder is built in the vector of `self`, which has room for it unless `self` is an
    /// empty vector, and then in the vector of `rhs`.
    ///
    /// # Panics
    ///
    /// When `rhs` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// let a = Number::new_from_i32(-1000, ByteOrder::BigEndian);
    /// let b = Number::new_from_i32(7, ByteOrder::BigEndian);
    /// let remainder = a % b;
    /// // -1000 = 7 (-142) - 6.
    /// assert_eq!(remainder.value, [6]);
    /// assert!(remainder.is_negative());
    /// ```
    #[track_caller]
    fn rem(mut self, rhs: Number) -> Number {
        self %= rhs;
        self
    }
}

impl RemAssign<&Number> for Number {
    /// Sets `self` to `self % rhs`, keeping the byte order of `self`, in the shortest form.
    ///
    /// The remainder is built in the vector of `self`, which has room for it unless `self` is an
    /// empty vector; `rhs` is left as it was.
    ///
    /// # Panics
    ///
    /// When `rhs` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// let mut n = Number::new_from_u32(0x0102_0304, ByteOrder::LittleEndian);
    /// n %= &Number::new_from_u32(0x0001_0000, ByteOrder::BigEndian);
    /// assert_eq!(n.value, [0x04, 0x03]);
    /// ```
    #[inline]
    #[track_caller]
    fn rem_assign(&mut self, rhs: &Number) {
        if small(self, rhs) {
            let (_, r) = small_division::<TRUNCATED, true>(self, rhs);
            put_small(&mut self.value, self.byte_order, r);
            self.negative = r.negative;
            return;
        }
        assign_ref::<true>(self, rhs);
    }
}

impl RemAssign<Number> for Number {
    /// Sets `self` to `self % rhs`, keeping the byte order of `self`, in the shortest form.
    ///
    /// The remainder is built in the vector of `self`, which has room for it unless `self` is an
    /// empty vector, and then in the vector of `rhs`.
    ///
    /// # Panics
    ///
    /// When `rhs` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// let mut n = Number::new_from_i32(-0x0105, ByteOrder::BigEndian);
    /// n %= Number::new_from_i32(-0x0100, ByteOrder::BigEndian);
    /// assert_eq!(n.value, [5]);
    /// assert!(n.is_negative());
    /// ```
    #[inline]
    #[track_caller]
    fn rem_assign(&mut self, mut rhs: Number) {
        if small(self, &rhs) {
            let (_, r) = small_division::<TRUNCATED, true>(self, &rhs);
            if self.value.capacity() < r.len {
                mem::swap(&mut self.value, &mut rhs.value);
            }
            put_small(&mut self.value, self.byte_order, r);
            self.negative = r.negative;
            return;
        }
        assign_value::<true>(self, &mut rhs);
    }
}

impl Number {
    /// Returns the quotient and the remainder of `self` divided by `rhs`, `(self / rhs, self %
    /// rhs)`, computed together: the quotient truncated toward zero, the remainder with the sign of
    /// `self`, both in the byte order of `self` and in the shortest form.
    ///
    /// Each is built in one new vector of exactly its length (one byte for zero); both operands
    /// are left as they were.
    ///
    /// # Panics
    ///
    /// When `rhs` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// // -1000 = 7 (-142) - 6.
    /// let a = Number::new_from_i32(-1000, ByteOrder::BigEndian);
    /// let (q, r) = a.div_rem(&Number::new_from_i32(7, ByteOrder::BigEndian));
    /// assert_eq!(q.value, [142]);
    /// assert!(q.is_negative());
    /// assert_eq!(r.value, [6]);
    /// assert!(r.is_negative());
    /// ```
    #[track_caller]
    pub fn div_rem(&self, rhs: &Number) -> (Number, Number) {
        if small(self, rhs) {
            let (q, r) = small_division::<TRUNCATED, false>(self, rhs);
            return (
                small_number(self.byte_order, q),
                small_number(self.byte_order, r),
            );
        }
        both::<TRUNCATED>(self, rhs)
    }

    /// Returns the quotient of Euclid's division of `self` by `rhs`: the `q` for which `self = rhs
    /// q + r` with `0 <= r < |rhs|`, in the byte order of `self` and in the shortest form, as
    /// `i32::div_euclid` gives it.
    ///
    /// It is `self / rhs` unless the remainder `self % rhs` is below zero, and then one further
    /// from zero: `-7 / 2 = -3`, but the Euclidean quotient is `-4`. It is built in one new vector
    /// of exactly its length.
    ///
    /// # Panics
    ///
    /// When `rhs` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// let a = Number::new_from_i32(-7, ByteOrder::BigEndian);
    /// let q = a.div_euclid(&Number::new_from_i32(2, ByteOrder::BigEndian));
    /// assert_eq!(q.value, [4]);
    /// assert!(q.is_negative());
    ///
    /// let q = a.div_euclid(&Number::new_from_i32(-2, ByteOrder::BigEndian));
    /// assert_eq!(q.value, [4]);
    /// assert!(!q.is_negative());
    /// ```
    #[track_caller]
    pub fn div_euclid(&self, rhs: &Number) -> Number {
        if small(self, rhs) {
            let (q, _) = small_division::<EUCLID, false>(self, rhs);
            return small_number(self.byte_order, q);
        }
        ref_by_ref::<EUCLID, false>(self, rhs)
    }

    /// Returns the remainder of Euclid's division of `self` by `rhs`, `self` modulo `|rhs|`: the
    /// `r` for which `self = rhs q + r` with `0 <= r < |rhs|`, in the byte order of `self` and in
    /// the shortest form, as `i32::rem_euclid` gives it. It is never negative.
    ///
    /// It is `self % rhs` when that is not below zero, and `self % rhs + |rhs|` otherwise. It is
    /// built in one new vector of exactly its length.
    ///
    /// # Panics
    ///
    /// When `rhs` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// let a = Number::new_from_i32(-7, ByteOrder::BigEndian);
    /// let r = a.rem_euclid(&Number::new_from_i32(2, ByteOrder::BigEndian));
    /// assert_eq!(r.value, [1]);
    /// assert!(!r.is_negative());
    ///
    /// let r = a.rem_euclid(&Number::new_from_i32(-2, ByteOrder::BigEndian));
    /// assert_eq!(r.value, [1]);
    /// assert!(!r.is_negative());
    /// ```
    #[track_caller]
    pub fn rem_euclid(&self, rhs: &Number) -> Number {
        if small(self, rhs) {
            let (_, r) = small_division::<EUCLID, true>(self, rhs);
            return small_number(self.byte_order, r);
        }
        ref_by_ref::<EUCLID, true>(self, rhs)
    }
}

/// The division whose quotient is truncated toward zero, and whose remainder has the sign of the
/// dividend: `/`, `%` and `div_rem`.
pub(super) const TRUNCATED: bool = false;

/// Euclid's division, whose remainder is never negative: `div_euclid` and `rem_euclid`.
pub(super) const EUCLID: bool = true;

/// Panics for a division by zero, with the message of the primitive integers: of `%` when
/// `REM`, of `/` otherwise.
#[cold]
#[inline(never)]
#[track_caller]
pub(super) fn by_zero<const REM: bool>() -> ! {
    if REM {
        panic!("attempt to calculate the remainder with a divisor of zero")
    } else {
        panic!("attempt to divide by zero")
    }
}

// Small numbers

/// A quotient or a remainder of two small operands, as the small path computes it in registers.
#[derive(Clone, Copy)]
struct Small {
    /// The low eight bytes of the magnitude.
    lo: u64,
    /// The next eight bytes of the magnitude.
    hi: u64,
    /// The length in the shortest form, from 1 to 16: one byte for zero.
    len: usize,
    /// Whether it is negative; never for zero.
    negative: bool,
}

impl Small {
    /// The magnitude `m` with the sign `negative`, which zero does not keep.
    #[inline(always)]
    fn of(m: u128, negative: bool) -> Small {
        Small {
            lo: m as u64,
            hi: (m >> 64) as u64,
            len: 16 - (m | 1).leading_zeros() as usize / 8,
            negative: negative && m != 0,
        }
    }
}

/// The quotient and the remainder of the small operands `a` and `b`: truncated, or Euclid's with
/// `EUCLIDEAN`. Panics with the message of `%` when `REM`, of `/` otherwise, when `b` is zero.
#[inline(always)]
#[track_caller]
fn small_division<const EUCLIDEAN: bool, const REM: bool>(
    a: &Number,
    b: &Number,
) -> (Small, Small) {
    // SAFETY: both vectors hold at most 16 bytes, as `small` checked.
    let (x, y) = unsafe {
        (
            read_small(&a.value, a.byte_order),
            read_small(&b.value, b.byte_order),
        )
    };
    if y == 0 {
        by_zero::<REM>();
    }
    let (q, r) = div_u128(x, y);
    let negative_a = a.negative && x != 0;
    if EUCLIDEAN && negative_a && r != 0 {
        // One further from zero, which leaves the remainder `y - r`: with `r != 0`, `y >= 2`, and
        // `q <= x / 2` does not overflow.
        return (
            Small::of(q + 1, a.negative != b.negative),
            Small::of(y - r, false),
        );
    }
    (
        Small::of(q, a.negative != b.negative),
        Small::of(r, negative_a && !EUCLIDEAN),
    )
}

/// The quotient and the remainder of `x` by `y > 0`: by the hardware's division when both fit in
/// a limb; by a divisor of one limb, normalized, its reciprocal computed and the two limbs of `x`
/// divided by [`div_2by1`] steps, when only `y` does, which took 30 cycles instead of the 56 of the
/// `u128` division of the compiler's runtime on an Apple M4 Max; and by that division otherwise,
/// whose quotient is then a single limb.
#[inline(always)]
fn div_u128(x: u128, y: u128) -> (u128, u128) {
    if y >> 64 == 0 {
        let y = y as u64;
        if x >> 64 == 0 {
            let x = x as u64;
            return (u128::from(x / y), u128::from(x % y));
        }
        let s = y.leading_zeros();
        let (d, u) = (y << s, x << s);
        let v = invert_limb(d);
        // The bits shifted out of the top, below `2^s`, are below `d`.
        let top = ((x >> 64) as u64 >> 1) >> (63 - s);
        let (q1, t) = div_2by1(top, (u >> 64) as u64, d, v);
        let (q0, r) = div_2by1(t, u as u64, d, v);
        return ((u128::from(q1) << 64) | u128::from(q0), u128::from(r >> s));
    }
    (x / y, x % y)
}

/// Writes the small result `s` as the whole value of `v`, in its vector when it has room, in a new
/// one otherwise.
#[inline(always)]
fn put_small(v: &mut Vec<u8>, order: ByteOrder, s: Small) {
    if v.capacity() >= s.len {
        // SAFETY: the vector has room for the `s.len` bytes, all of which are written before the
        // length is set.
        unsafe {
            write_small(v.as_mut_ptr(), order, s.lo, s.hi, s.len);
            v.set_len(s.len);
        }
    } else {
        replace_small(v, order, s);
    }
}

/// Replaces `v` with one new vector of exactly the small result's length, holding it. Out of line
/// and cold, so that the small path saves no registers for it.
#[cold]
#[inline(never)]
fn replace_small(v: &mut Vec<u8>, order: ByteOrder, s: Small) {
    *v = new_small(order, s);
}

/// The small result `s` in one new vector of exactly its length.
#[inline(always)]
fn new_small(order: ByteOrder, s: Small) -> Vec<u8> {
    let mut v = Vec::with_capacity(s.len);
    // SAFETY: the new vector has room for the `s.len` bytes, all of which are written before the
    // length is set.
    unsafe {
        write_small(v.as_mut_ptr(), order, s.lo, s.hi, s.len);
        v.set_len(s.len);
    }
    v
}

/// The small result `s` as a number in one new vector, in the byte order `order`. Out of line, so
/// that the small path saves no registers for the allocation.
#[inline(never)]
fn small_number(order: ByteOrder, s: Small) -> Number {
    Number {
        value: new_small(order, s),
        negative: s.negative,
        byte_order: order,
    }
}

// Dividends of up to 64 bytes by divisors of up to 32

/// The longest vector, in bytes, of a dividend that [`wider_division`] divides in registers.
pub(super) const WIDER: usize = 64;

/// Whether `a / b` is computed in registers by [`register_division`]: the dividend's vector holds
/// at most [`WIDER`] bytes and the divisor's at most [`WIDE`].
#[inline(always)]
fn wide(a: &Number, b: &Number) -> bool {
    a.value.len() <= WIDER && b.value.len() <= WIDE
}

/// The quotient and the remainder of a dividend of up to 64 bytes by a divisor of up to 32: by a
/// divisor of up to 16 significant bytes, as [`wide_division`] computes them up to 32 bytes and
/// [`wider_division`] beyond; by a longer one, as [`wider_by_wide`] does. Panics with the message
/// of `%` when `REM`, of `/` otherwise, when the divisor is zero.
///
/// # Safety
///
/// As for [`wider_division`], and the divisor's vector must hold at most [`WIDE`] bytes.
#[inline(always)]
#[track_caller]
unsafe fn register_division<const EUCLIDEAN: bool, const REM: bool>(
    a: &Number,
    b: &Number,
) -> (Wide, Wide) {
    // SAFETY: as the caller guarantees, with the length checked for the shorter one.
    unsafe {
        let y = if b.value.len() <= SMALL {
            read_small(&b.value, b.byte_order)
        } else {
            let y = read_wide(&b.value, b.byte_order);
            if y[2] | y[3] != 0 {
                return by_medium::<EUCLIDEAN>(a, Medium::of(y, b.negative));
            }
            u128::from(y[0]) | (u128::from(y[1]) << 64)
        };
        if y == 0 {
            by_zero::<REM>();
        }
        let y = Short::of(y, b.negative);
        if a.value.len() <= WIDE {
            wide_division::<EUCLIDEAN, 4>(a, y)
        } else {
            wider_division::<EUCLIDEAN>(a, y)
        }
    }
}

/// A divisor of up to 16 bytes, not zero, as the divisions in registers use it: its magnitude
/// `y`, the shift `s` that sets its top bit and the magnitude so shifted, `d`, the reciprocal `v`
/// of `d`'s top limb when `y` is one limb and of both otherwise, and its sign.
#[derive(Clone, Copy)]
pub(super) struct Short {
    y: u128,
    s: u32,
    d: u128,
    v: u64,
    negative: bool,
}

impl Short {
    /// The magnitude `y`, not zero, and the sign `negative`, prepared.
    #[inline(always)]
    pub(super) fn of(y: u128, negative: bool) -> Short {
        let (y1, y0) = ((y >> 64) as u64, y as u64);
        let s = if y1 == 0 {
            y0.leading_zeros()
        } else {
            y1.leading_zeros()
        };
        let d = y << s;
        let v = if y1 == 0 {
            invert_limb(d as u64)
        } else {
            invert_3by2((d >> 64) as u64, d as u64)
        };
        Short {
            y,
            s,
            d,
            v,
            negative,
        }
    }
}

/// The quotient and the remainder of a dividend of up to `8 LIMBS` bytes, 16 or 32, by the short
/// divisor `y`, truncated, or Euclid's with `EUCLIDEAN`, computed in registers. The dividend is
/// read as limbs, as the addition or the multiplication reads it, shifted so that the divisor's top
/// bit is set, and its limbs divided by [`div_2by1`] steps for a divisor of one limb, or
/// [`div_3by2`] steps for two, unrolled.
///
/// # Safety
///
/// The dividend's vector must hold at most `8 LIMBS` bytes.
#[inline(always)]
pub(super) unsafe fn wide_division<const EUCLIDEAN: bool, const LIMBS: usize>(
    a: &Number,
    y: Short,
) -> (Wide, Wide) {
    // SAFETY: as the caller guarantees.
    let x = unsafe {
        if LIMBS == 2 {
            let x = read_small(&a.value, a.byte_order);
            [x as u64, (x >> 64) as u64, 0, 0]
        } else {
            read_wide(&a.value, a.byte_order)
        }
    };
    let u = shifted(x, y.s);
    let mut q = [0; 4];
    let r = if y.y >> 64 == 0 {
        let d = y.d as u64;
        let mut t = u[LIMBS];
        for i in (0..LIMBS).rev() {
            (q[i], t) = div_2by1(t, u[i], d, y.v);
        }
        u128::from(t >> y.s)
    } else {
        let (d1, d0) = ((y.d >> 64) as u64, y.d as u64);
        let (mut t1, mut t0) = (u[LIMBS], u[LIMBS - 1]);
        for i in (0..LIMBS - 1).rev() {
            (q[i], t1, t0) = div_3by2(t1, t0, u[i], d1, d0, y.v);
        }
        ((u128::from(t1) << 64) | u128::from(t0)) >> y.s
    };
    let negative_a = a.negative && x != [0; 4];
    if EUCLIDEAN && negative_a && r != 0 {
        // One further from zero, which leaves the remainder `y - r`: with `r != 0`, `y >= 2`, and
        // the quotient, at most half the dividend, does not overflow.
        let mut carry = true;
        for l in &mut q {
            (*l, carry) = l.overflowing_add(u64::from(carry));
        }
        return wide_results::<true>(q4(q), y.y - r, a.negative != y.negative, false);
    }
    // The quotient's length without a branch by one limb, whose quotients of 24 or 25 bytes for
    // a dividend of 32 are a coin toss; by two, where it took 82 cycles against 75 for 32 bytes
    // by 16, with one.
    if y.y >> 64 == 0 {
        wide_results::<true>(q4(q), r, a.negative != y.negative, negative_a && !EUCLIDEAN)
    } else {
        wide_results::<false>(q4(q), r, a.negative != y.negative, negative_a && !EUCLIDEAN)
    }
}

/// The four limbs `q` as eight.
#[inline(always)]
fn q4(q: [u64; 4]) -> [u64; 8] {
    [q[0], q[1], q[2], q[3], 0, 0, 0, 0]
}

/// The quotient `q` and the remainder `r` of a division in registers as results to write, with the
/// signs `negative_q` and `negative_r`, which zero does not keep; the quotient's length found
/// without a branch with `BRANCHLESS` (see [`wide_of`]).
#[inline(always)]
fn wide_results<const BRANCHLESS: bool>(
    q: [u64; 8],
    r: u128,
    negative_q: bool,
    negative_r: bool,
) -> (Wide, Wide) {
    (
        wide_of::<BRANCHLESS>(q, negative_q),
        wide_of::<false>([r as u64, (r >> 64) as u64, 0, 0, 0, 0, 0, 0], negative_r),
    )
}

/// [`wide_division`] for a dividend of 33 to 64 bytes: its limbs are read shifted into memory, and
/// divided by a loop of steps.
///
/// # Safety
///
/// The dividend's vector must hold at most [`WIDER`] bytes.
#[inline(always)]
pub(super) unsafe fn wider_division<const EUCLIDEAN: bool>(a: &Number, y: Short) -> (Wide, Wide) {
    let x = Operand {
        ptr: a.value.as_ptr(),
        len: a.value.len(),
        big_endian: a.byte_order == ByteOrder::BigEndian,
        significant: a.value.len(),
    };
    let n = x.len.div_ceil(8);
    let mut u = [0; 9];
    // SAFETY: the dividend's `n <= 8` limbs and the one above them fit.
    unsafe { u[n] = to_limbs_shifted(x, u.as_mut_ptr(), y.s) };
    let mut q = [0; 8];
    let r = if y.y >> 64 == 0 {
        let d = y.d as u64;
        let mut t = u[n];
        for i in (0..n).rev() {
            (q[i], t) = div_2by1(t, u[i], d, y.v);
        }
        u128::from(t >> y.s)
    } else {
        let (d1, d0) = ((y.d >> 64) as u64, y.d as u64);
        let (mut t1, mut t0) = (u[n], u[n - 1]);
        for i in (0..n - 1).rev() {
            (q[i], t1, t0) = div_3by2(t1, t0, u[i], d1, d0, y.v);
        }
        ((u128::from(t1) << 64) | u128::from(t0)) >> y.s
    };
    let negative_a = a.negative && u != [0; 9];
    if EUCLIDEAN && negative_a && r != 0 {
        // As in `wide_division`.
        let mut carry = true;
        for l in &mut q {
            (*l, carry) = l.overflowing_add(u64::from(carry));
        }
        return wide_results::<false>(q, y.y - r, a.negative != y.negative, false);
    }
    wide_results::<false>(q, r, a.negative != y.negative, negative_a && !EUCLIDEAN)
}

/// A divisor of three or four limbs, 17 to 32 significant bytes, as [`wider_by_wide`] uses it: its
/// limbs `d` shifted left by `s` bits so that the top bit of the top one is set, their number
/// `n`, the reciprocal `v` of their top two, and its sign.
#[derive(Clone, Copy)]
pub(super) struct Medium {
    d: [u64; 4],
    s: u32,
    n: usize,
    v: u64,
    negative: bool,
}

impl Medium {
    /// The magnitude of the limbs `y`, of which the third or the fourth is not zero, and the sign
    /// `negative`, prepared.
    #[inline(always)]
    pub(super) fn of(y: [u64; 4], negative: bool) -> Medium {
        let n = if y[3] != 0 { 4 } else { 3 };
        let s = y[n - 1].leading_zeros();
        let above = |l: u64| (l >> 1) >> (63 - s);
        let d = [
            y[0] << s,
            (y[1] << s) | above(y[0]),
            (y[2] << s) | above(y[1]),
            (y[3] << s) | above(y[2]),
        ];
        let v = invert_3by2(d[n - 1], d[n - 2]);
        Medium {
            d,
            s,
            n,
            v,
            negative,
        }
    }

    /// The divisor normalized in the `n` limbs at `d` by the shift `s`, `n` three or four, their
    /// reciprocal `v`, and the sign `negative`.
    ///
    /// # Safety
    ///
    /// The limbs at `d` must be valid for reads.
    #[inline(always)]
    pub(super) unsafe fn normalized(
        d: *const u64,
        n: usize,
        s: u32,
        v: u64,
        negative: bool,
    ) -> Medium {
        let mut limbs = [0; 4];
        // SAFETY: as the caller guarantees.
        unsafe { ptr::copy_nonoverlapping(d, limbs.as_mut_ptr(), n) };
        Medium {
            d: limbs,
            s,
            n,
            v,
            negative,
        }
    }
}

/// The quotient and the remainder of a dividend of up to 64 bytes by the divisor `y`, by
/// [`wider_by_wide`] for its number of limbs.
///
/// # Safety
///
/// The dividend's vector must hold at most [`WIDER`] bytes.
#[inline(always)]
pub(super) unsafe fn by_medium<const EUCLIDEAN: bool>(a: &Number, y: Medium) -> (Wide, Wide) {
    // SAFETY: as the caller guarantees.
    unsafe {
        if y.n == 4 {
            wider_by_wide::<EUCLIDEAN, 4>(a, y)
        } else {
            wider_by_wide::<EUCLIDEAN, 3>(a, y)
        }
    }
}

/// The quotient and the remainder of a dividend of up to 64 bytes by the divisor `y` of `NY`
/// limbs, truncated, or Euclid's with `EUCLIDEAN`: the steps of the schoolbook division on limbs
/// in arrays, the divisor's top two and their reciprocal in registers, and the products of a
/// quotient limb by its other one or two limbs inline.
///
/// # Safety
///
/// The dividend's vector must hold at most [`WIDER`] bytes.
#[inline(always)]
unsafe fn wider_by_wide<const EUCLIDEAN: bool, const NY: usize>(
    a: &Number,
    y: Medium,
) -> (Wide, Wide) {
    debug_assert!(y.n == NY);
    let (ny, d, s, v) = (NY, y.d, y.s, y.v);
    let x = Operand {
        ptr: a.value.as_ptr(),
        len: a.value.len(),
        big_endian: a.byte_order == ByteOrder::BigEndian,
        significant: a.value.len(),
    };
    let n = x.len.div_ceil(8);
    let mut u = [0; 9];
    // SAFETY: the dividend's `n <= 8` limbs and the one above them fit.
    unsafe { u[n] = to_limbs_shifted(x, u.as_mut_ptr(), s) };
    let negative_a = a.negative && u != [0; 9];
    let (d1, d0) = (d[ny - 1], d[ny - 2]);
    let mut q = [0; 8];
    let nn = n.max(ny);
    // Without bits shifted out above the dividend, the top quotient limb is 0 or 1, from a
    // comparison instead of a step, as for the general path.
    let steps = if u[nn] == 0 {
        let k = nn - ny;
        if (0..ny)
            .rev()
            .map(|j| u[k + j].cmp(&d[j]))
            .find(|o| o.is_ne())
            != Some(Less)
        {
            let mut borrow = false;
            for j in 0..ny {
                let (l, b1) = u[k + j].overflowing_sub(d[j]);
                let (l, b2) = l.overflowing_sub(u64::from(borrow));
                (u[k + j], borrow) = (l, b1 | b2);
            }
            q[k] = 1;
        }
        k
    } else {
        nn - ny + 1
    };
    // The limb above the dividend's is below `d1`, whose top bit is set, and each step leaves the
    // partial remainder below the divisor: the top two limbs of a step are at most `d1 B + d0`.
    for i in (0..steps).rev() {
        let (u2, u1) = (u[i + ny], u[i + ny - 1]);
        if u2 == d1 && u1 == d0 {
            q[i] = all_ones_step(&mut u, i, &d, ny);
            continue;
        }
        let (mut qi, mut r1, mut r0) = div_3by2(u2, u1, u[i + ny - 2], d1, d0, v);
        // The quotient limb times the divisor's limbs below its top two.
        let mut borrow = 0;
        for j in 0..ny - 2 {
            let p = u128::from(qi) * u128::from(d[j]) + u128::from(borrow);
            let (l, b) = u[i + j].overflowing_sub(p as u64);
            u[i + j] = l;
            borrow = (p >> 64) as u64 + u64::from(b);
        }
        let b0;
        (r0, b0) = r0.overflowing_sub(borrow);
        let b1;
        (r1, b1) = r1.overflowing_sub(u64::from(b0));
        if b1 {
            // One too large: the divisor is added back.
            let mut carry = false;
            for j in 0..ny - 2 {
                (u[i + j], carry) = carrying(u[i + j], d[j], carry);
            }
            (r0, carry) = carrying(r0, d0, carry);
            r1 = r1.wrapping_add(d1).wrapping_add(u64::from(carry));
            qi -= 1;
        }
        (u[i + ny - 2], u[i + ny - 1], u[i + ny]) = (r0, r1, 0);
        q[i] = qi;
    }
    let remainder = u[..ny] != [0; NY];
    if EUCLIDEAN && negative_a && remainder {
        // One further from zero, which leaves the remainder `y - r`, from the limbs as they are
        // shifted: the quotient, below `B^6` as the divisor is at least `B^2`, does not overflow.
        let mut carry = true;
        for l in &mut q {
            (*l, carry) = l.overflowing_add(u64::from(carry));
        }
        let mut borrow = false;
        for j in 0..ny {
            let (l, b1) = d[j].overflowing_sub(u[j]);
            let (l, b2) = l.overflowing_sub(u64::from(borrow));
            (u[j], borrow) = (l, b1 | b2);
        }
    }
    // The remainder, shifted back: `(l << 1) << (63 - s)` is `l << (64 - s)`, and zero for `s = 0`.
    let below = |l: u64| (l << 1) << (63 - s);
    let mut r = [0; 8];
    for j in 0..ny {
        r[j] = (u[j] >> s) | below(u[j + 1]);
    }
    (
        wide_of::<false>(q, a.negative != y.negative),
        wide_of::<false>(r, negative_a && !EUCLIDEAN),
    )
}

/// `a + b + carry` and its carry out.
#[inline(always)]
fn carrying(a: u64, b: u64, carry: bool) -> (u64, bool) {
    let (s, c1) = a.overflowing_add(b);
    let (s, c2) = s.overflowing_add(u64::from(carry));
    (s, c1 | c2)
}

/// The step of [`wider_by_wide`] whose partial remainder, in limbs `i..=i + ny` of `u`, has the
/// top two limbs of the normalized divisor `d` of `ny` limbs, which `div_3by2` cannot divide: its
/// quotient limb is `B - 1`, as in `sb_div_qr`, and `(B - 1) d` is subtracted, which cancels the
/// top limb. Returns the quotient limb.
#[cold]
#[inline(never)]
fn all_ones_step(u: &mut [u64; 9], i: usize, d: &[u64; 4], ny: usize) -> u64 {
    // `(B - 1) d = d B - d`: `d` added, and `d` subtracted one limb up.
    let mut carry = false;
    for j in 0..ny {
        (u[i + j], carry) = carrying(u[i + j], d[j], carry);
    }
    u[i + ny] = u[i + ny].wrapping_add(u64::from(carry));
    let mut borrow = false;
    for j in 0..ny {
        let (l, b1) = u[i + j + 1].overflowing_sub(d[j]);
        let (l, b2) = l.overflowing_sub(u64::from(borrow));
        (u[i + j + 1], borrow) = (l, b1 | b2);
    }
    debug_assert!(u[i + ny] == 0, "the top limb cancels");
    u64::MAX
}

/// The four limbs `x` shifted left by `s` bits, `s < 64`, into five.
#[inline(always)]
fn shifted(x: [u64; 4], s: u32) -> [u64; 5] {
    // `(l >> 1) >> (63 - s)` is `l >> (64 - s)`, and zero for `s = 0`, with shifts below 64.
    let out = |l: u64| (l >> 1) >> (63 - s);
    [
        x[0] << s,
        (x[1] << s) | out(x[0]),
        (x[2] << s) | out(x[1]),
        (x[3] << s) | out(x[2]),
        out(x[3]),
    ]
}

/// The limbs `l` as a result of a division in registers, with its length in the shortest form
/// and the sign `negative`, which zero does not keep; the length found without a branch with
/// `BRANCHLESS`.
///
/// Whether a quotient's top limb is zero is as good as random, and a mispredicted branch on it
/// cost a third of a division of 32 bytes by 8, measured on an Apple M4 Max: 77 cycles against
/// 65. That is for the quotient of [`wide_division`], whose limbs are in registers; for those of
/// [`wider_division`] and [`wider_by_wide`], in memory, the length found without a branch made
/// the stores of the result wait for all of them, and a division of 64 bytes by 8 took 160
/// cycles against 110.
#[inline(always)]
fn wide_of<const BRANCHLESS: bool>(l: [u64; 8], negative: bool) -> Wide {
    let len = if BRANCHLESS {
        // The top limb that is not zero, `t`, from a bit for each, and its bytes; zero is one
        // byte, of limb 0.
        let mut nonzero = 1;
        for (i, &x) in l.iter().enumerate() {
            nonzero |= u32::from(x != 0) << i;
        }
        let t = 31 - nonzero.leading_zeros() as usize;
        (8 * t + 8 - l[t].leading_zeros() as usize / 8).max(1)
    } else {
        match l.iter().rposition(|&x| x != 0) {
            Some(t) => 8 * t + 8 - l[t].leading_zeros() as usize / 8,
            None => 1,
        }
    };
    Wide {
        limbs: l,
        len,
        negative: negative && l != [0; 8],
    }
}

/// The result `w` of [`wide_division`] as a number in one new vector of exactly its length.
#[inline(always)]
pub(super) fn wide_number(order: ByteOrder, w: &Wide) -> Number {
    let mut value = Vec::new();
    put_wide(&mut value, order, w);
    Number {
        value,
        negative: w.negative,
        byte_order: order,
    }
}

// The general path: operands of more than 16 bytes, out of line, so that the small path saves no
// registers for it.

/// The operands of a division with at least one of more than 16 bytes, read for the general path;
/// panics with the message of `%` when `REM`, of `/` otherwise, when the divisor is zero.
#[inline(always)]
#[track_caller]
fn operands<const REM: bool>(a: &Number, b: &Number) -> (Operand, Operand) {
    let (x, y) = (Operand::of(a), Operand::of(b));
    if y.significant == 0 {
        by_zero::<REM>();
    }
    (x, y)
}

/// `a / b` (`a % b` when `REM`; Euclid's with `EUCLIDEAN`) for borrowed operands, in one new
/// vector.
#[inline(never)]
#[track_caller]
fn ref_by_ref<const EUCLIDEAN: bool, const REM: bool>(a: &Number, b: &Number) -> Number {
    let order = a.byte_order;
    if wide(a, b) {
        // SAFETY: the vectors hold at most 64 and 32 bytes.
        let (q, r) = unsafe { register_division::<EUCLIDEAN, REM>(a, b) };
        return wide_number(order, if REM { &r } else { &q });
    }
    let (x, y) = operands::<REM>(a, b);
    // SAFETY: the operands are those of `a` and `b`, which are borrowed until the result is
    // computed.
    unsafe {
        divide::<EUCLIDEAN, _>(
            x,
            y,
            a.negative,
            b.negative,
            Want::of(EUCLIDEAN, REM),
            |q, r| new_number(if REM { r } else { q }, order),
        )
    }
}

/// `div_rem` for borrowed operands: both results, in one new vector each.
#[inline(never)]
#[track_caller]
fn both<const EUCLIDEAN: bool>(a: &Number, b: &Number) -> (Number, Number) {
    let order = a.byte_order;
    if wide(a, b) {
        // SAFETY: the vectors hold at most 64 and 32 bytes.
        let (q, r) = unsafe { register_division::<EUCLIDEAN, false>(a, b) };
        return (wide_number(order, &q), wide_number(order, &r));
    }
    let (x, y) = operands::<false>(a, b);
    // SAFETY: as in `ref_by_ref`.
    unsafe {
        divide::<EUCLIDEAN, _>(x, y, a.negative, b.negative, Want::Both, |q, r| {
            (new_number(q, order), new_number(r, order))
        })
    }
}

/// `a / b` (`a % b` when `REM`) for a borrowed `a` and an owned `b`: the result takes the vector of
/// `b` when it is built there, leaving `b` an empty one; the caller drops `b` either way.
#[inline(never)]
#[track_caller]
fn ref_by_value<const REM: bool>(a: &Number, b: &mut Number) -> Number {
    let order = a.byte_order;
    if wide(a, b) {
        // SAFETY: the vectors hold at most 64 and 32 bytes.
        let (q, r) = unsafe { register_division::<TRUNCATED, REM>(a, b) };
        let w = if REM { &r } else { &q };
        put_wide(&mut b.value, order, w);
        return Number {
            value: mem::take(&mut b.value),
            negative: w.negative,
            byte_order: order,
        };
    }
    let (x, y) = operands::<REM>(a, b);
    // SAFETY: the operands are those of `a` and `b`; `b`'s vector is written only once both are
    // read into limbs.
    unsafe {
        divide::<TRUNCATED, _>(
            x,
            y,
            a.negative,
            b.negative,
            Want::of(TRUNCATED, REM),
            |q, r| {
                let p = if REM { r } else { q };
                p.put_in(&mut b.value, order);
                Number {
                    value: mem::take(&mut b.value),
                    negative: p.negative,
                    byte_order: order,
                }
            },
        )
    }
}

/// `a /= b` (`a %= b` when `REM`) for a borrowed `b`.
#[inline(never)]
#[track_caller]
fn assign_ref<const REM: bool>(a: &mut Number, b: &Number) {
    let order = a.byte_order;
    if wide(a, b) {
        // SAFETY: the vectors hold at most 64 and 32 bytes.
        let (q, r) = unsafe { register_division::<TRUNCATED, REM>(a, b) };
        let w = if REM { &r } else { &q };
        put_wide(&mut a.value, order, w);
        a.negative = w.negative;
        return;
    }
    let (x, y) = operands::<REM>(a, b);
    // SAFETY: the operands are those of `a` and `b`; `a`'s vector is written only once both are
    // read into limbs.
    unsafe {
        divide::<TRUNCATED, _>(
            x,
            y,
            a.negative,
            b.negative,
            Want::of(TRUNCATED, REM),
            |q, r| {
                let p = if REM { r } else { q };
                p.put_in(&mut a.value, order);
                a.negative = p.negative;
            },
        );
    }
}

/// `a /= b` (`a %= b` when `REM`) for an owned `b`: when the result is built in the vector of `b`,
/// the vector of `a` goes to `b`; the caller drops `b` either way.
#[inline(never)]
#[track_caller]
fn assign_value<const REM: bool>(a: &mut Number, b: &mut Number) {
    let order = a.byte_order;
    if wide(a, b) {
        // SAFETY: the vectors hold at most 64 and 32 bytes.
        let (q, r) = unsafe { register_division::<TRUNCATED, REM>(a, b) };
        let w = if REM { &r } else { &q };
        if a.value.capacity() < w.len && b.value.capacity() >= w.len {
            mem::swap(&mut a.value, &mut b.value);
        }
        put_wide(&mut a.value, order, w);
        a.negative = w.negative;
        return;
    }
    let (x, y) = operands::<REM>(a, b);
    // SAFETY: as in `assign_ref`.
    unsafe {
        divide::<TRUNCATED, _>(
            x,
            y,
            a.negative,
            b.negative,
            Want::of(TRUNCATED, REM),
            |q, r| {
                let p = if REM { r } else { q };
                if a.value.capacity() < p.len && b.value.capacity() >= p.len {
                    mem::swap(&mut a.value, &mut b.value);
                }
                p.put_in(&mut a.value, order);
                a.negative = p.negative;
            },
        );
    }
}

/// The result `p` as a number in one new vector of exactly its length, in the byte order `order`.
///
/// # Safety
///
/// As for [`Product::write`].
#[inline(always)]
unsafe fn new_number(p: &Product, order: ByteOrder) -> Number {
    let mut v = Vec::with_capacity(p.len);
    // SAFETY: the new vector has room for the `len` bytes, all of which are written before the
    // length is set.
    unsafe {
        p.write(v.as_mut_ptr(), order);
        v.set_len(p.len);
    }
    Number {
        value: v,
        negative: p.negative,
        byte_order: order,
    }
}

/// Divides the operand `x` by the operand `y`, of signs `negative_x` and `negative_y`, and hands
/// the quotient and the remainder to `place`, which writes them where they belong; returns what
/// `place` returns. The division is truncated, or Euclid's with `EUCLIDEAN`. When `want` asks for
/// one result alone, the other is not computed, and `place` gets zero for it.
///
/// # Safety
///
/// The operands must describe vectors that stay valid for reads until `place` is called, and `y`
/// must not be zero.
#[inline(always)]
unsafe fn divide<const EUCLIDEAN: bool, R>(
    x: Operand,
    y: Operand,
    negative_x: bool,
    negative_y: bool,
    want: Want,
    place: impl FnOnce(&Product, &Product) -> R,
) -> R {
    let ny = y.significant.div_ceil(8);
    let nx = x.significant.div_ceil(8).max(ny);
    // The divisor, normalized, and a limb to start it on 16 bytes, then what `compute_by` needs.
    let need = ny + 1 + working_memory(nx, ny, want);
    let compute = |s: *mut MaybeUninit<u64>| {
        // SAFETY: the working memory holds `need` limbs; the rest as the caller guarantees.
        unsafe { compute::<EUCLIDEAN, R>(s.cast(), x, y, negative_x, negative_y, want, place) }
    };
    if need <= STACK_LIMBS {
        on_stack(compute)
    } else {
        on_heap(need, compute)
    }
}

/// The limbs of working memory that [`compute_by`] needs for a dividend of `nx` limbs, at least as
/// many as the divisor's `ny`: the dividend shifted, with a limb above it, the quotient with a limb
/// above it for Euclid's, and the scratch of divide and conquer, or of a quotient alone computed
/// from the top limbs of the operands.
#[inline(always)]
pub(super) fn working_memory(nx: usize, ny: usize, want: Want) -> usize {
    let qn = nx - ny + 1;
    let mut need = (nx + 1) + (qn + 1);
    if want == Want::Quotient && ny >= 3 && from_top(nx + 1, ny) {
        // The top limbs of both operands, a limb to align the divisor's, their quotient and the
        // scratch of their division; or the product that checks it.
        let k = qn;
        let top = (2 * k + 2) + 1 + 2 * (k + 1) + top_itch(k);
        need += top.max(check_itch(nx + 1, ny));
    } else if ny >= DC_DIV {
        // The exact division by divide and conquer, of the dividend with the limb above it or
        // without (see `compute_by`), and for a quotient alone of its top limbs, or with the low
        // ones from the top ones; below `DC_DIV` limbs, none.
        let itch = |nn: usize| {
            let mut itch = qr_itch(nn, ny);
            if want == Want::Quotient && ny >= 3 {
                if ny < DC_DIV_Q {
                    itch = itch.max(q_itch(nn, ny));
                } else if let l @ 1.. = split_low(nn, ny) {
                    itch = itch.max(split_itch(nn, ny, l));
                }
            }
            itch
        };
        need += itch(nx + 1).max(itch(nx));
    }
    need
}

/// A divisor of the general path as the division reads it: its `n` limbs at `d`, the least
/// significant first, shifted left by `shift` bits so that the top bit is set, starting on 16
/// bytes, and their reciprocal `v`, of the top limb for one limb and of the top two otherwise.
#[derive(Clone, Copy)]
pub(super) struct Normal {
    pub(super) d: *const u64,
    pub(super) n: usize,
    pub(super) shift: u32,
    pub(super) v: u64,
}

/// The shift that sets the top bit of the operand `y`, not zero, from its most significant byte,
/// which is its top limb's `y.significant - 8 (n - 1)`-th.
///
/// # Safety
///
/// The operand's vector must be valid for reads.
#[inline(always)]
unsafe fn normalizing_shift(y: Operand) -> u32 {
    // SAFETY: the most significant byte lies inside the vector.
    let top = unsafe {
        if y.big_endian {
            *y.ptr.add(y.len - y.significant)
        } else {
            *y.ptr.add(y.significant - 1)
        }
    };
    8 * (8 * y.significant.div_ceil(8) - y.significant) as u32 + top.leading_zeros()
}

/// Reads the divisor `y`, not zero, normalized into its limbs at `d`, which start on 16 bytes, and
/// computes their reciprocal.
///
/// # Safety
///
/// The operand's vector must be valid for reads and `d` for writing its limbs.
#[inline(always)]
pub(super) unsafe fn normalize(y: Operand, d: *mut u64) -> Normal {
    let n = y.significant.div_ceil(8);
    // SAFETY: as the caller guarantees.
    unsafe {
        let shift = normalizing_shift(y);
        to_limbs_shifted(y, d, shift);
        let v = if n == 1 {
            invert_limb(*d)
        } else {
            invert_3by2(*d.add(n - 1), *d.add(n - 2))
        };
        Normal { d, n, shift, v }
    }
}

/// [`divide`] in the working memory at `s`: the divisor is read into its first limbs, normalized,
/// and the dividend divided by it with [`compute_by`] in the rest, but for a quotient alone much
/// shorter than the divisor, which needs only the divisor's top limbs.
///
/// # Safety
///
/// As for [`divide`], with `s` valid for the limbs that `divide` counts.
#[inline(always)]
unsafe fn compute<const EUCLIDEAN: bool, R>(
    s: *mut u64,
    x: Operand,
    y: Operand,
    negative_x: bool,
    negative_y: bool,
    want: Want,
    place: impl FnOnce(&Product, &Product) -> R,
) -> R {
    let ny = y.significant.div_ceil(8);
    let nx = x.significant.div_ceil(8).max(ny);
    // SAFETY: the working memory holds the divisor's `ny` limbs from an address on 16 bytes, and
    // after them the limbs of `compute_by`.
    unsafe {
        // The divisor starts on 16 bytes, which the kernel reads in pairs of limbs for every limb
        // of the quotient: at 8 bytes off, a division of 72 limbs by 36 took 4100 cycles instead of
        // 2760 on an Apple M4 Max, whenever the stack put it there.
        let d = s.add((s.addr() / 8) & 1);
        let w = s.add(ny + 1);
        if want == Want::Quotient && ny >= 3 && from_top(nx + 1, ny) {
            let k = nx + 1 - ny;
            let (u, q) = (w, w.add(nx + 1));
            let t = q.add(k + 1);
            let shift = normalizing_shift(y);
            quotient_from_top(x, Top::Operand(y, d), ny, shift, u, q, t);
            let quotient = result(q, k, (negative_x && x.significant != 0) != negative_y);
            return place(&quotient, &ZERO);
        }
        let normal = normalize(y, d);
        compute_by::<EUCLIDEAN, R>(w, x, normal, negative_x, negative_y, want, place)
    }
}

/// Divides the operand `x`, of sign `negative_x`, by the normalized divisor `y`, of sign
/// `negative_y`, in the working memory at `s`, and hands the results to `place`, as [`divide`]
/// does.
///
/// # Safety
///
/// As for [`divide`], with `s` valid for the [`working_memory`] limbs of `x` and `y`.
#[inline(always)]
pub(super) unsafe fn compute_by<const EUCLIDEAN: bool, R>(
    s: *mut u64,
    x: Operand,
    y: Normal,
    negative_x: bool,
    negative_y: bool,
    want: Want,
    place: impl FnOnce(&Product, &Product) -> R,
) -> R {
    let (d, ny, shift) = (y.d, y.n, y.shift);
    let lx = x.significant.div_ceil(8);
    let nx = lx.max(ny);
    let qn = nx - ny + 1;
    let negative_x = negative_x && x.significant != 0;
    // SAFETY: the working memory holds the `nx + 1` limbs of the dividend, the `qn + 1` of the
    // quotient and the scratch, which every step stays inside.
    unsafe {
        let u = s;
        let q = u.add(nx + 1);
        let t = q.add(qn + 1);
        if want == Want::Quotient && ny >= 3 && from_top(nx + 1, ny) {
            quotient_from_top(x, Top::Normal(d), ny, shift, u, q, t);
            let quotient = result(q, qn, negative_x != negative_y);
            return place(&quotient, &ZERO);
        }
        read_dividend(x, u, nx, shift);
        *q.add(qn) = 0;
        match ny {
            1 => {
                *u = if want == Want::Remainder {
                    div_qr_1::<false>(q, u, nx, *u.add(nx), *d, y.v)
                } else {
                    div_qr_1::<true>(q, u, nx, *u.add(nx), *d, y.v)
                };
            }
            2 => {
                let (r1, r0) = div_qr_2(q, u, nx + 1, *d.add(1), *d, y.v);
                (*u, *u.add(1)) = (r0, r1);
            }
            _ => {
                // The limb above the dividend takes part only when the shift put bits in it; the
                // quotient's top limb then comes out of the steps, and otherwise from the
                // comparison that precedes them.
                let nn = if *u.add(nx) != 0 { nx + 1 } else { nx };
                let qh = if want != Want::Quotient {
                    div_qr(q, u, nn, d, ny, y.v, t)
                } else if ny < DC_DIV_Q {
                    // The quotient alone: its last limbs from a shrinking divisor, the others
                    // exactly.
                    match div_q(q, u, nn, d, ny, y.v, t) {
                        Some(qh) => qh,
                        None => exactly_after_all(q, u, nn, x, y, t),
                    }
                } else if split_low(nn, ny) > 0 {
                    // The quotient alone of a long division: its low limbs from the top ones.
                    div_q_split(q, u, nn, d, ny, y.v, t)
                } else {
                    div_qr(q, u, nn, d, ny, y.v, t)
                };
                if nn == nx {
                    *q.add(nx - ny) = u64::from(qh);
                }
            }
        }
        if want == Want::Quotient {
            let quotient = result(q, qn + 1, negative_x != negative_y);
            return place(&quotient, &ZERO);
        }
        let remainder = (0..ny).any(|i| *u.add(i) != 0);
        if EUCLIDEAN && negative_x && remainder {
            // Euclid's quotient is one further from zero, and its remainder `|y| - r`: shifted, as
            // the divisor and the remainder still are.
            sub_n(u, d, u, ny);
            if want != Want::Remainder {
                add_1(q, qn + 1, 1);
            }
        }
        if shift != 0 {
            rshift(u, ny, shift);
        }
        // Without the quotient, the limbs of a quotient of a divisor of one limb are not written.
        let quotient = if want == Want::Remainder {
            ZERO
        } else {
            result(q, qn + 1, negative_x != negative_y)
        };
        let remainder = result(u, ny, negative_x && !EUCLIDEAN);
        place(&quotient, &remainder)
    }
}

/// Reads the dividend `x` into the `nx + 1` limbs at `u`, shifted left by `shift` bits, `nx` at
/// least its limbs: a dividend shorter than the divisor gets zero limbs up to the divisor's length,
/// and its quotient is then zero, its remainder the dividend.
///
/// # Safety
///
/// The operand's vector must be valid for reads, and `u` for writing `nx + 1` limbs.
#[inline(always)]
unsafe fn read_dividend(x: Operand, u: *mut u64, nx: usize, shift: u32) {
    let lx = x.significant.div_ceil(8);
    // SAFETY: as the caller guarantees.
    unsafe {
        *u.add(lx) = to_limbs_shifted(x, u, shift);
        if lx < nx {
            ptr::write_bytes(u.add(lx + 1), 0, nx - lx);
        }
    }
}

/// The quotient of [`div_qr`] after all, when [`div_q`] gave up on the dividend `x` of `nn`
/// limbs by the divisor `y`: the dividend is read again into the limbs at `u`. Returns the
/// quotient's limb above the `nn - y.n` at `q`.
///
/// # Safety
///
/// As for `div_qr`, with the operand's vector valid for reads.
#[cold]
#[inline(never)]
unsafe fn exactly_after_all(
    q: *mut u64,
    u: *mut u64,
    nn: usize,
    x: Operand,
    y: Normal,
    t: *mut u64,
) -> bool {
    // SAFETY: as the caller guarantees: the limbs are those `compute_by` read, of which the
    // division takes `nn`.
    unsafe {
        read_dividend(x, u, x.significant.div_ceil(8).max(y.n), y.shift);
        div_qr(q, u, nn, y.d, y.n, y.v, t)
    }
}

/// Where [`quotient_from_top`] gets the divisor from: an operand, of which it reads only the top
/// limbs, and the whole, into the limbs at the pointer, for the rare check; or normalized limbs.
#[derive(Clone, Copy)]
enum Top {
    Operand(Operand, *mut u64),
    Normal(*const u64),
}

/// The quotient alone of a division whose quotient is much shorter than the divisor
/// ([`from_top`]), in the `k` limbs at `q`: from the top `2k + 2` limbs of the dividend and the top
/// `k + 1` of the divisor of `ny` limbs, shifted by `shift`, by [`div_q_top`]. The whole dividend
/// is read, into the limbs at `u`, only for the rare check. Out of line, so that the shorter
/// divisions keep their registers.
///
/// # Safety
///
/// As for [`compute_by`], with `u` valid for the dividend's `nx + 1` limbs, `q` for the quotient's
/// `k`, and `t` for the scratch of `div_q_top` and of `check_q_top`, as [`working_memory`] counts
/// them, and the divisor's limbs as `y` describes them.
#[inline(never)]
unsafe fn quotient_from_top(
    x: Operand,
    y: Top,
    ny: usize,
    shift: u32,
    u: *mut u64,
    q: *mut u64,
    t: *mut u64,
) {
    let lx = x.significant.div_ceil(8);
    let nx = lx.max(ny);
    let k = nx + 1 - ny;
    // SAFETY: the copies of the top limbs, their quotient and the scratch of their division lie in
    // the scratch at `t`, as `working_memory` counts it, and so does the check's.
    unsafe {
        let w = t;
        let dt = w.add(2 * k + 2);
        let dt = dt.add((dt.addr() / 8) & 1);
        let h = dt.add(k + 1);
        let top: *const u64 = match y {
            Top::Operand(y, _) => {
                shifted_limbs(y, ny - (k + 1), ny, shift, dt);
                dt
            }
            Top::Normal(d) => d.add(ny - (k + 1)),
        };
        shifted_limbs(x, nx + 1 - (2 * k + 2), nx + 1, shift, w);
        let v = invert_3by2(*top.add(k), *top.add(k - 1));
        let found = div_q_top(h, w, k, top, v, h.add(k + 1));
        debug_assert!(
            found != Found::AllOnes,
            "the dividend's top limb is below the divisor's"
        );
        ptr::copy_nonoverlapping(h.add(1), q, k);
        if found == Found::Check {
            read_dividend(x, u, nx, shift);
            let d = match y {
                Top::Operand(y, d) => {
                    to_limbs_shifted(y, d, shift);
                    d.cast_const()
                }
                Top::Normal(d) => d,
            };
            check_q_top(q, u, nx + 1, d, ny, t);
        }
    }
}

/// What a division on the general path computes: both results, or the quotient or the remainder
/// alone, which spares the steps that only the other needs.
#[derive(Clone, Copy, PartialEq)]
pub(super) enum Want {
    Both,
    Quotient,
    Remainder,
}

impl Want {
    /// What `/` and `/=` (`%` and `%=` with `REM`) need, or, with `EUCLIDEAN`, `div_euclid` and
    /// `rem_euclid`: a Euclidean result is adjusted with both.
    #[inline(always)]
    pub(super) const fn of(euclidean: bool, rem: bool) -> Want {
        match (euclidean, rem) {
            (false, false) => Want::Quotient,
            (_, true) => Want::Remainder,
            (true, false) => Want::Both,
        }
    }
}

/// The zero that [`compute`] hands to `place` as the result it does not compute.
const ZERO: Product = Product {
    r: ptr::null(),
    len: 1,
    negative: false,
};

/// Limb `i` of the magnitude of the operand `x`: its `x.significant.div_ceil(8)` limbs read as
/// [`to_limbs_shifted`] reads them, and zero above them.
///
/// # Safety
///
/// The operand's vector must be valid for reads.
#[inline(always)]
unsafe fn limb(x: Operand, i: usize) -> u64 {
    let l = x.significant;
    let (full, rem) = (l / 8, l % 8);
    // SAFETY: every read lies inside the significant bytes, as in `to_limbs_shifted`.
    unsafe {
        let end = x.ptr.add(x.len);
        if i < full {
            if x.big_endian {
                u64::from_be_bytes(read(end.sub(8 * i + 8)))
            } else {
                u64::from_le_bytes(read(x.ptr.add(8 * i)))
            }
        } else if i == full && rem > 0 {
            if l >= 8 {
                let v = if x.big_endian {
                    u64::from_be_bytes(read(end.sub(l)))
                } else {
                    u64::from_le_bytes(read(x.ptr.add(l - 8)))
                };
                v >> (8 * (8 - rem))
            } else if x.big_endian {
                read_small(
                    core::slice::from_raw_parts(end.sub(l), rem),
                    ByteOrder::BigEndian,
                ) as u64
            } else {
                read_small(
                    core::slice::from_raw_parts(x.ptr, rem),
                    ByteOrder::LittleEndian,
                ) as u64
            }
        } else {
            0
        }
    }
}

/// Writes limbs `from..to` of the magnitude of the operand `x` shifted left by `s` bits, `s < 64`,
/// at `out`: each from the limb at its place and the one below it. The whole limbs go through a
/// loop of their own, for either byte order, and the partial limb above them and the zero limbs
/// through [`limb`].
///
/// # Safety
///
/// The operand's vector must be valid for reads, and `out` for writing `to - from` limbs.
#[inline(always)]
unsafe fn shifted_limbs(x: Operand, from: usize, to: usize, s: u32, out: *mut u64) {
    let above = |v: u64| (v >> 1) >> (63 - s);
    let full = (x.significant / 8).clamp(from, to);
    // SAFETY: as the caller guarantees; the limbs below `full` are whole limbs of the operand.
    unsafe {
        let mut below = if from > 0 { limb(x, from - 1) } else { 0 };
        let end = x.ptr.add(x.len);
        if x.big_endian {
            for i in from..full {
                let v = u64::from_be_bytes(read(end.sub(8 * i + 8)));
                *out.add(i - from) = (v << s) | above(below);
                below = v;
            }
        } else {
            for i in from..full {
                let v = u64::from_le_bytes(read(x.ptr.add(8 * i)));
                *out.add(i - from) = (v << s) | above(below);
                below = v;
            }
        }
        for i in full..to {
            let v = limb(x, i);
            *out.add(i - from) = (v << s) | above(below);
            below = v;
        }
    }
}

/// Reads the significant bytes of the operand `x` into `x.significant.div_ceil(8)` limbs at `out`,
/// the least significant first, as the multiplication's `to_limbs` does, shifted left by `s` bits
/// on the way, `s < 64`; returns the bits shifted out of the top limb.
///
/// # Safety
///
/// The operand's vector must be valid for reads, and `out` for writing its limbs.
#[inline(always)]
unsafe fn to_limbs_shifted(x: Operand, out: *mut u64, s: u32) -> u64 {
    let l = x.significant;
    let (full, rem) = (l / 8, l % 8);
    // The bits of `v` above `64 - s`, `v >> (64 - s)`, and none for `s = 0`, with shifts below 64.
    let above = |v: u64| (v >> 1) >> (63 - s);
    let mut below = 0;
    // SAFETY: every read lies inside the significant bytes: big-endian the last `l` of the
    // vector, little-endian the first `l`.
    unsafe {
        let end = x.ptr.add(x.len);
        for i in 0..full {
            let v = if x.big_endian {
                u64::from_be_bytes(read(end.sub(8 * i + 8)))
            } else {
                u64::from_le_bytes(read(x.ptr.add(8 * i)))
            };
            *out.add(i) = (v << s) | above(below);
            below = v;
        }
        if rem > 0 {
            // The `rem` bytes above the whole limbs: from 8 bytes up, the eight that end at the
            // top, shifted down to them; below, as the addition reads a short magnitude.
            let v = if l >= 8 {
                let v = if x.big_endian {
                    u64::from_be_bytes(read(end.sub(l)))
                } else {
                    u64::from_le_bytes(read(x.ptr.add(l - 8)))
                };
                v >> (8 * (8 - rem))
            } else if x.big_endian {
                read_small(
                    core::slice::from_raw_parts(end.sub(l), rem),
                    ByteOrder::BigEndian,
                ) as u64
            } else {
                read_small(
                    core::slice::from_raw_parts(x.ptr, rem),
                    ByteOrder::LittleEndian,
                ) as u64
            };
            *out.add(full) = (v << s) | above(below);
            below = v;
        }
    }
    above(below)
}

/// The `n` limbs at `r` as a result to write: their length in the shortest form, and the sign
/// `negative`, which zero does not keep.
///
/// # Safety
///
/// The limbs must be valid for reads.
#[inline(always)]
unsafe fn result(r: *const u64, n: usize, negative: bool) -> Product {
    // SAFETY: as the caller guarantees.
    unsafe {
        let mut top = n;
        while top > 0 && *r.add(top - 1) == 0 {
            top -= 1;
        }
        if top == 0 {
            return Product {
                r: ptr::null(),
                len: 1,
                negative: false,
            };
        }
        let len = 8 * top - (*r.add(top - 1)).leading_zeros() as usize / 8;
        Product { r, len, negative }
    }
}
