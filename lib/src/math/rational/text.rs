//! Rationals as text: [`Display`](fmt::Display) writes `-3/4`, or `5` for an integer, and
//! [`Debug`](fmt::Debug) writes the same; and the conversions from integers.

use super::{Rational, one, trim};
use crate::math::numbers::{ByteOrder, Number};
use core::fmt;

impl fmt::Display for Rational {
    /// Writes the number in decimal as `numerator/denominator`, `-` in front of a negative one, and
    /// as the numerator alone for an integer; the formatter's flags, width and fill apply to the
    /// whole, as they do to the primitive integers.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::rational::Rational;
    ///
    /// let r = Rational::from(-3) / 4u32;
    /// assert_eq!(r.to_string(), "-3/4");
    /// assert_eq!(format!("{:>7}", r), "   -3/4");
    /// assert_eq!(format!("{:+}", Rational::from(5)), "+5");
    /// ```
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let digits = if super::is_one(&self.den) {
            self.num.abs().to_string()
        } else {
            let mut s = self.num.abs().to_string();
            s.push('/');
            s.push_str(&self.den.to_string());
            s
        };
        f.pad_integral(!self.num.is_negative(), "", &digits)
    }
}

impl fmt::Debug for Rational {
    /// Writes the number as [`Display`](fmt::Display) does.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl From<Number> for Rational {
    /// The integer `n` as `n/1`, as [`Rational::from_integer`] makes it.
    #[inline]
    fn from(n: Number) -> Rational {
        Rational::from_integer(n)
    }
}

impl From<&Number> for Rational {
    /// The integer `n` as `n/1`, in new vectors.
    #[inline]
    fn from(n: &Number) -> Rational {
        Rational::from_integer(n.clone())
    }
}

macro_rules! from_primitive {
    ($($t:ty),*) => {$(
        impl From<$t> for Rational {
            #[doc = concat!("The `", stringify!($t), "` `v` as `v/1`, little-endian.")]
            #[inline]
            fn from(v: $t) -> Rational {
                let mut num = Number::from(v);
                trim(&mut num);
                Rational::from_reduced(num, one(ByteOrder::LittleEndian))
            }
        }
    )*};
}

from_primitive!(
    u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize
);
