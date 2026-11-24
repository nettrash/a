//! `GT`: the `r`-th roots of unity of `F_p¹²`, where the pairing takes its values.

use super::curve::PairingCurve;
use crate::math::field::Fp12;
use crate::math::numbers::Number;
use core::fmt;
use core::hash::{Hash, Hasher};
use core::ops::{Div, DivAssign, Mul, MulAssign};

/// An element of `GT`, the group of order `r` of the `r`-th roots of unity in `F_p¹²*`, written
/// multiplicatively: `*` is the group's operation, [`pow`](Gt::pow) its power, and the inverse is
/// the conjugate. It borrows its [`PairingCurve`].
///
/// # Panics
///
/// The operators panic for elements of different curves.
///
/// # Examples
///
/// ```
/// use a::math::pairing::PairingCurve;
/// use a::math::numbers::Number;
///
/// let c = PairingCurve::bn254();
/// let g = c.gt_generator();
/// assert_eq!(&g * &g.inverse(), c.gt_identity());
/// assert_eq!(g.pow(c.order()), c.gt_identity());
/// assert_eq!(g.pow(&Number::from(-1)), g.inverse());
/// ```
#[derive(Clone)]
pub struct Gt<'c> {
    /// The curve.
    pub(super) curve: &'c PairingCurve,
    /// The value, of order dividing `r`.
    pub(super) f: Fp12<'c>,
}

impl<'c> Gt<'c> {
    /// The element of `f`, which the caller knows to be of order dividing `r`.
    pub(super) fn from_fp12_unchecked(curve: &'c PairingCurve, f: Fp12<'c>) -> Gt<'c> {
        Gt { curve, f }
    }

    /// The element of `GT` that `f` is, if `f^r = 1` and `f ≠ 0`: a test of the cyclotomic
    /// subgroup by two Frobenius maps, then a power by `r` by cyclotomic squares.
    ///
    /// # Panics
    ///
    /// Panics if `f` is not of the curve's tower.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::Fp12;
    /// use a::math::pairing::{Gt, PairingCurve};
    ///
    /// let c = PairingCurve::bls12_381();
    /// let g = c.gt_generator();
    /// assert_eq!(Gt::from_fp12(c, g.value().clone()), Some(g.clone()));
    /// assert_eq!(Gt::from_fp12(c, g.value() + g.value()), None);
    /// ```
    #[track_caller]
    #[must_use]
    pub fn from_fp12(curve: &'c PairingCurve, f: Fp12<'c>) -> Option<Gt<'c>> {
        assert!(
            f.tower().xi() == curve.tower().xi(),
            "elements of different towers"
        );
        if !f.is_cyclotomic() || !f.cyclotomic_pow(&curve.r).is_one() {
            return None;
        }
        Some(Gt { curve, f })
    }

    /// The curve.
    #[must_use]
    pub fn curve(&self) -> &'c PairingCurve {
        self.curve
    }

    /// The element of `F_p¹²`.
    #[must_use]
    pub fn value(&self) -> &Fp12<'c> {
        &self.f
    }

    /// Whether it is 1, the neutral element.
    #[must_use]
    pub fn is_identity(&self) -> bool {
        self.f.is_one()
    }

    /// The inverse, the conjugate.
    #[must_use]
    pub fn inverse(&self) -> Gt<'c> {
        Gt {
            curve: self.curve,
            f: self.f.conjugate(),
        }
    }

    /// The square, by the cyclotomic squaring of Granger and Scott.
    #[must_use]
    pub fn square(&self) -> Gt<'c> {
        Gt {
            curve: self.curve,
            f: self.f.cyclotomic_square(),
        }
    }

    /// The `k`-th power for any integer `k`, reduced modulo `r` first, by cyclotomic squares over
    /// the non-adjacent form of the exponent.
    #[must_use]
    pub fn pow(&self, k: &Number) -> Gt<'c> {
        let k = k.rem_euclid(&self.curve.r);
        Gt {
            curve: self.curve,
            f: self.f.cyclotomic_pow(&k),
        }
    }

    /// The product.
    #[track_caller]
    fn mul_ref(a: &Gt<'c>, b: &Gt<'c>) -> Gt<'c> {
        a.curve.check(b.curve);
        Gt {
            curve: a.curve,
            f: &a.f * &b.f,
        }
    }

    /// The quotient, by the conjugate.
    #[track_caller]
    fn div_ref(a: &Gt<'c>, b: &Gt<'c>) -> Gt<'c> {
        a.curve.check(b.curve);
        Gt {
            curve: a.curve,
            f: &a.f * &b.f.conjugate(),
        }
    }
}

/// `*` and `/` for the four combinations of owned and borrowed operands, and `*=` and `/=`.
macro_rules! gt_binary {
    ($tr:ident, $m:ident, $atr:ident, $am:ident, $f:ident) => {
        impl<'c> $tr<&Gt<'c>> for &Gt<'c> {
            type Output = Gt<'c>;
            #[track_caller]
            fn $m(self, rhs: &Gt<'c>) -> Gt<'c> {
                Gt::$f(self, rhs)
            }
        }
        impl<'c> $tr<Gt<'c>> for &Gt<'c> {
            type Output = Gt<'c>;
            #[track_caller]
            fn $m(self, rhs: Gt<'c>) -> Gt<'c> {
                Gt::$f(self, &rhs)
            }
        }
        impl<'c> $tr<&Gt<'c>> for Gt<'c> {
            type Output = Gt<'c>;
            #[track_caller]
            fn $m(self, rhs: &Gt<'c>) -> Gt<'c> {
                Gt::$f(&self, rhs)
            }
        }
        impl<'c> $tr<Gt<'c>> for Gt<'c> {
            type Output = Gt<'c>;
            #[track_caller]
            fn $m(self, rhs: Gt<'c>) -> Gt<'c> {
                Gt::$f(&self, &rhs)
            }
        }
        impl<'c> $atr<&Gt<'c>> for Gt<'c> {
            #[track_caller]
            fn $am(&mut self, rhs: &Gt<'c>) {
                *self = Gt::$f(self, rhs);
            }
        }
        impl<'c> $atr<Gt<'c>> for Gt<'c> {
            #[track_caller]
            fn $am(&mut self, rhs: Gt<'c>) {
                *self = Gt::$f(self, &rhs);
            }
        }
    };
}

gt_binary!(Mul, mul, MulAssign, mul_assign, mul_ref);
gt_binary!(Div, div, DivAssign, div_assign, div_ref);

impl PartialEq for Gt<'_> {
    #[track_caller]
    fn eq(&self, other: &Self) -> bool {
        self.curve.check(other.curve);
        self.f == other.f
    }
}

impl Eq for Gt<'_> {}

impl Hash for Gt<'_> {
    fn hash<H: Hasher>(&self, h: &mut H) {
        self.f.hash(h);
    }
}

impl fmt::Debug for Gt<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.f, f)
    }
}

impl fmt::Display for Gt<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.f, f)
    }
}
