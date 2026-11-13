//! Comparisons by value, as IEEE 754 has them: NaN is unordered and equal to nothing, itself
//! included, and `-0 == +0`; whatever the precisions. [`BigFloat::total_cmp`] orders every value.

use super::{BigFloat, Kind};
use core::cmp::Ordering;

/// The order of the magnitudes `0.a` and `0.b` of two mantissas aligned at their top bits, of
/// any lengths.
fn cmp_mantissas(a: &[u64], b: &[u64]) -> Ordering {
    let (na, nb) = (a.len(), b.len());
    for i in 0..na.max(nb) {
        let x = if i < na { a[na - 1 - i] } else { 0 };
        let y = if i < nb { b[nb - 1 - i] } else { 0 };
        if x != y {
            return x.cmp(&y);
        }
    }
    Ordering::Equal
}

/// The order of the magnitudes of two numbers that are not NaN.
fn cmp_abs(a: &Kind, b: &Kind) -> Ordering {
    match (a, b) {
        (Kind::Inf { .. }, Kind::Inf { .. }) | (Kind::Zero { .. }, Kind::Zero { .. }) => {
            Ordering::Equal
        }
        (Kind::Inf { .. }, _) | (_, Kind::Zero { .. }) => Ordering::Greater,
        (_, Kind::Inf { .. }) | (Kind::Zero { .. }, _) => Ordering::Less,
        (
            Kind::Finite {
                exp: ea, mant: ma, ..
            },
            Kind::Finite {
                exp: eb, mant: mb, ..
            },
        ) => ea.cmp(eb).then_with(|| cmp_mantissas(ma, mb)),
        (Kind::Nan, _) | (_, Kind::Nan) => unreachable!("NaN has no magnitude"),
    }
}

impl BigFloat {
    /// The order of the numbers by value, `None` when either is NaN.
    pub(crate) fn cmp_value(&self, other: &BigFloat) -> Option<Ordering> {
        if self.is_nan() || other.is_nan() {
            return None;
        }
        let (a, b) = (
            self.is_sign_negative() && !self.is_zero(),
            other.is_sign_negative() && !other.is_zero(),
        );
        Some(match (a, b) {
            (false, true) => Ordering::Greater,
            (true, false) => Ordering::Less,
            (false, false) => cmp_abs(&self.kind, &other.kind),
            (true, true) => cmp_abs(&self.kind, &other.kind).reverse(),
        })
    }

    /// The order of the magnitudes, `None` when either is NaN.
    pub fn cmp_abs(&self, other: &BigFloat) -> Option<Ordering> {
        if self.is_nan() || other.is_nan() {
            None
        } else {
            Some(cmp_abs(&self.kind, &other.kind))
        }
    }

    /// A total order of every value, as IEEE 754's `totalOrder`: `-∞`, the negative numbers,
    /// `-0`, `+0`, the positive numbers, `+∞`, then NaN; equal values of different precisions
    /// are equal.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::BigFloat;
    /// use core::cmp::Ordering;
    ///
    /// assert_eq!(BigFloat::neg_zero(10).total_cmp(&BigFloat::zero(10)), Ordering::Less);
    /// assert_eq!(BigFloat::nan(10).total_cmp(&BigFloat::infinity(10)), Ordering::Greater);
    /// ```
    pub fn total_cmp(&self, other: &BigFloat) -> Ordering {
        match (self.is_nan(), other.is_nan()) {
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Greater,
            (false, true) => Ordering::Less,
            (false, false) => {
                let (a, b) = (self.is_sign_negative(), other.is_sign_negative());
                match (a, b) {
                    (false, true) => Ordering::Greater,
                    (true, false) => Ordering::Less,
                    (false, false) => cmp_abs(&self.kind, &other.kind),
                    (true, true) => cmp_abs(&self.kind, &other.kind).reverse(),
                }
            }
        }
    }
}

impl PartialEq for BigFloat {
    /// Whether the numbers are equal in value: NaN equals nothing, `-0` equals `+0`, and the
    /// precisions do not count.
    fn eq(&self, other: &BigFloat) -> bool {
        self.cmp_value(other) == Some(Ordering::Equal)
    }
}

impl PartialOrd for BigFloat {
    /// The order by value, `None` when either is NaN.
    fn partial_cmp(&self, other: &BigFloat) -> Option<Ordering> {
        self.cmp_value(other)
    }
}
