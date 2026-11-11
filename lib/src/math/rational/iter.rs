//! Sums and products of iterators of rationals: [`Sum`] and [`Product`], of rationals owned or
//! borrowed.

use super::Rational;
use core::iter::{Product, Sum};

impl Sum for Rational {
    /// The sum, `0` for no rationals, in the byte order of the first; each added in place.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::rational::Rational;
    ///
    /// let h: Rational = (1..=4u32).map(|k| Rational::from(1) / k).sum();
    /// assert_eq!(h.to_string(), "25/12");
    /// ```
    fn sum<I: Iterator<Item = Rational>>(mut iter: I) -> Rational {
        let Some(mut total) = iter.next() else {
            return Rational::default();
        };
        for r in iter {
            total += &r;
        }
        total
    }
}

impl<'a> Sum<&'a Rational> for Rational {
    /// The sum, `0` for no rationals, in the byte order of the first.
    fn sum<I: Iterator<Item = &'a Rational>>(mut iter: I) -> Rational {
        let Some(first) = iter.next() else {
            return Rational::default();
        };
        let mut total = first.clone();
        for r in iter {
            total += r;
        }
        total
    }
}

impl Product for Rational {
    /// The product, `1` for no rationals, in the byte order of the first; each multiplied in
    /// place.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::rational::Rational;
    ///
    /// let p: Rational = (2..=5u32).map(|k| Rational::from(k - 1) / k).product();
    /// assert_eq!(p.to_string(), "1/5");
    /// ```
    fn product<I: Iterator<Item = Rational>>(mut iter: I) -> Rational {
        let Some(mut total) = iter.next() else {
            return Rational::from(1u8);
        };
        for r in iter {
            total *= &r;
        }
        total
    }
}

impl<'a> Product<&'a Rational> for Rational {
    /// The product, `1` for no rationals, in the byte order of the first.
    fn product<I: Iterator<Item = &'a Rational>>(mut iter: I) -> Rational {
        let Some(first) = iter.next() else {
            return Rational::from(1u8);
        };
        let mut total = first.clone();
        for r in iter {
            total *= r;
        }
        total
    }
}
