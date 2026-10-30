//! The sign of a number: `-a` with [`Neg`], [`Number::abs`], [`Number::signum`],
//! [`Number::is_zero`] and [`Number::is_positive`].

use super::add::significant_len;
use super::{ByteOrder, Number};
use core::ops::Neg;

impl Number {
    /// Returns whether the number is zero: whether every byte of its magnitude is zero, or it has
    /// none.
    ///
    /// A number in the shortest form costs one read of its top byte.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// assert!(Number::new_from_u8(0, ByteOrder::BigEndian).is_zero());
    /// assert!(Number::new(Vec::new(), ByteOrder::LittleEndian).is_zero());
    /// assert!(!Number::new_from_i8(-1, ByteOrder::BigEndian).is_zero());
    /// ```
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.significant() == 0
    }

    /// Returns whether the number is above zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// assert!(Number::new_from_i8(1, ByteOrder::BigEndian).is_positive());
    /// assert!(!Number::new_from_i8(0, ByteOrder::BigEndian).is_positive());
    /// assert!(!Number::new_from_i8(-1, ByteOrder::BigEndian).is_positive());
    /// ```
    #[inline]
    pub fn is_positive(&self) -> bool {
        !self.negative && !self.is_zero()
    }

    /// Returns the absolute value of the number, in its byte order and in the shortest form, in
    /// one new vector of exactly its length.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// let n = Number::new_from_i32(-1000, ByteOrder::BigEndian).abs();
    /// assert_eq!(n, 1000);
    /// assert!(!n.is_negative());
    /// ```
    #[inline]
    pub fn abs(&self) -> Number {
        self.with_sign(false)
    }

    /// Returns the sign of the number as a number, in its byte order: -1 below zero, 0 for zero
    /// and 1 above, as [`i32::signum`] does.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// assert_eq!(Number::new_from_i32(-1000, ByteOrder::BigEndian).signum(), -1);
    /// assert_eq!(Number::new_from_i32(0, ByteOrder::BigEndian).signum(), 0);
    /// assert_eq!(Number::new_from_i32(1000, ByteOrder::LittleEndian).signum(), 1);
    /// ```
    #[inline]
    pub fn signum(&self) -> Number {
        let zero = self.is_zero();
        Number {
            value: vec![u8::from(!zero)],
            negative: self.negative && !zero,
            byte_order: self.byte_order,
        }
    }

    /// The length of the magnitude without the zero bytes above its most significant non-zero
    /// byte.
    #[inline(always)]
    pub(super) fn significant(&self) -> usize {
        if self.byte_order == ByteOrder::BigEndian {
            significant_len::<true>(&self.value)
        } else {
            significant_len::<false>(&self.value)
        }
    }

    /// The magnitude of the number with the sign `negative` (which zero does not keep), in its
    /// byte order and in the shortest form, in one new vector of exactly its length.
    #[inline]
    fn with_sign(&self, negative: bool) -> Number {
        let s = self.significant();
        let value = if s == 0 {
            vec![0]
        } else if self.byte_order == ByteOrder::BigEndian {
            self.value[self.value.len() - s..].to_vec()
        } else {
            self.value[..s].to_vec()
        };
        Number {
            value,
            negative: negative && s != 0,
            byte_order: self.byte_order,
        }
    }
}

/// Trims the magnitude of `n` to the shortest form, in place: drops the zero bytes above its most
/// significant non-zero byte, and makes zero `[0]` and not negative.
#[inline]
pub(super) fn trim(n: &mut Number) {
    let s = n.significant();
    let len = n.value.len();
    if s == 0 {
        n.value.clear();
        n.value.push(0);
        n.negative = false;
    } else if s != len {
        if n.byte_order == ByteOrder::BigEndian {
            n.value.drain(..len - s);
        } else {
            n.value.truncate(s);
        }
    }
}

impl Neg for Number {
    type Output = Number;

    /// Returns `-self`: the same magnitude with the other sign, in the shortest form, in the
    /// vector of `self`; zero stays zero, never negative.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// let n = -Number::new_from_i32(1000, ByteOrder::BigEndian);
    /// assert_eq!(n, -1000);
    /// assert_eq!(-n, 1000);
    /// assert!(!(-Number::new_from_u8(0, ByteOrder::BigEndian)).is_negative());
    /// ```
    #[inline]
    fn neg(mut self) -> Number {
        trim(&mut self);
        self.negative = !self.negative && self.value != [0];
        self
    }
}

impl Neg for &Number {
    type Output = Number;

    /// Returns `-self`: the same magnitude with the other sign, in the byte order of `self` and in
    /// the shortest form, in one new vector of exactly its length; zero stays zero.
    #[inline]
    fn neg(self) -> Number {
        self.with_sign(!self.negative)
    }
}
