//! `GT`: the `r`-th roots of unity of `F_p¹²`, where the pairing takes its values.

use super::curve::PairingCurve;
use super::fast::{self, Engine, with_engine};
use crate::math::field::Fp12;
use crate::math::numbers::Number;
use core::fmt;
use core::hash::{Hash, Hasher};
use core::ops::{Div, DivAssign, Mul, MulAssign};
use std::sync::OnceLock;

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
    /// The value, of order dividing `r`, in the engine's arithmetic: the twelve coefficients'
    /// Montgomery limbs (4 of the 6 on BN254).
    v: [[u64; 6]; 12],
    /// The value in the public tower, made when it is asked for.
    public: OnceLock<Fp12<'c>>,
}

impl<'c> Gt<'c> {
    /// The element of `f`, which the caller knows to be of order dividing `r`.
    pub(super) fn from_fp12_unchecked(curve: &'c PairingCurve, f: Fp12<'c>) -> Gt<'c> {
        with_engine!(curve, E, N => Gt::from_fast::<E, N>(curve, &fast::Fp12::from_public(&f)))
    }

    /// The element of the engine's `f`, of order dividing `r`.
    pub(super) fn from_fast<E: Engine<N>, const N: usize>(
        curve: &'c PairingCurve,
        f: &fast::Fp12<E, N>,
    ) -> Gt<'c> {
        let c = f.coefficients();
        let mut v = [[0u64; 6]; 12];
        for (d, x) in v.iter_mut().zip(c.iter()) {
            d[..N].copy_from_slice(&x.0);
        }
        Gt {
            curve,
            v,
            public: OnceLock::new(),
        }
    }

    /// The value in the engine's arithmetic.
    pub(super) fn fast<E: Engine<N>, const N: usize>(&self) -> fast::Fp12<E, N> {
        let c: [fast::Fp<E, N>; 12] = core::array::from_fn(|i| {
            let mut l = [0u64; N];
            l.copy_from_slice(&self.v[i][..N]);
            fast::Fp::raw(l)
        });
        fast::Fp12::from_coefficients(&c)
    }

    /// The element of `GT` that `f` is, if `f^r = 1` and `f ≠ 0`: a test of the cyclotomic
    /// subgroup by two Frobenius maps, then `f^p = f^x` on BLS12 (Scott, ePrint 2021/1130) or
    /// `f^p = f^(6x²)` on BN, whose exponents differ from `p` by multiples of `r` that leave no
    /// other element of the cyclotomic subgroup (checked when the curve's tables are built; a
    /// power by `r` otherwise).
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
        with_engine!(curve, E, N => {
            let g = fast::Fp12::<E, N>::from_public(&f);
            if !fast::ops::gt_in_group(&g) {
                return None;
            }
            let gt = Gt::from_fast::<E, N>(curve, &g);
            let _ = gt.public.set(f);
            Some(gt)
        })
    }

    /// The curve.
    #[must_use]
    pub fn curve(&self) -> &'c PairingCurve {
        self.curve
    }

    /// The element of `F_p¹²`.
    #[must_use]
    pub fn value(&self) -> &Fp12<'c> {
        self.public
            .get_or_init(|| with_engine!(self.curve, E, N => self.fast::<E, N>().to_public()))
    }

    /// Whether it is 1, the neutral element.
    #[must_use]
    pub fn is_identity(&self) -> bool {
        with_engine!(self.curve, E, N => self.fast::<E, N>().is_one())
    }

    /// The inverse, the conjugate.
    #[must_use]
    pub fn inverse(&self) -> Gt<'c> {
        let c = self.curve;
        // The conjugate: the six coefficients of w negated in place, the others copied.
        let mut v = self.v;
        with_engine!(c, E, N => {
            for limbs in &mut v[6..] {
                let mut l = [0u64; N];
                l.copy_from_slice(&limbs[..N]);
                limbs[..N].copy_from_slice(&fast::Fp::<E, N>::raw(l).neg().0);
            }
        });
        Gt {
            curve: c,
            v,
            public: OnceLock::new(),
        }
    }

    /// The square, by the cyclotomic squaring of Granger and Scott.
    #[must_use]
    pub fn square(&self) -> Gt<'c> {
        let c = self.curve;
        with_engine!(c, E, N => Gt::from_fast::<E, N>(c, &self.fast::<E, N>().cyclotomic_square()))
    }

    /// The `k`-th power for any integer `k`, reduced modulo `r` first, by the decomposition of the
    /// exponent along the Frobenius map (`f^p = f^x` on BLS12, `f^(6x²)` on BN): four digits of
    /// 64 bits on BLS12-381, four of about 64 bits from a reduced lattice basis on BN254, raised
    /// together by cyclotomic squares.
    #[must_use]
    pub fn pow(&self, k: &Number) -> Gt<'c> {
        let c = self.curve;
        with_engine!(c, E, N => Gt::from_fast::<E, N>(c, &fast::ops::gt_pow(&self.fast::<E, N>(), k)))
    }

    /// The product.
    #[track_caller]
    fn mul_ref(a: &Gt<'c>, b: &Gt<'c>) -> Gt<'c> {
        a.curve.check(b.curve);
        let c = a.curve;
        with_engine!(c, E, N => Gt::from_fast::<E, N>(c, &a.fast::<E, N>().mul(&b.fast::<E, N>())))
    }

    /// The quotient, by the conjugate.
    #[track_caller]
    fn div_ref(a: &Gt<'c>, b: &Gt<'c>) -> Gt<'c> {
        a.curve.check(b.curve);
        let c = a.curve;
        with_engine!(c, E, N => {
            Gt::from_fast::<E, N>(c, &a.fast::<E, N>().mul(&b.fast::<E, N>().conjugate()))
        })
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
        self.v == other.v
    }
}

impl Eq for Gt<'_> {}

impl Hash for Gt<'_> {
    fn hash<H: Hasher>(&self, h: &mut H) {
        self.value().hash(h);
    }
}

impl fmt::Debug for Gt<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.value(), f)
    }
}

impl fmt::Display for Gt<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.value(), f)
    }
}
