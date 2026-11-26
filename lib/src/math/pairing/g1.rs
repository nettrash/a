//! `G1`: points of `E(F_p)`, `y² = x³ + b`.

use super::curve::{Family, PairingCurve};
use super::encoding::{
    COMPRESSED, INFINITY, SORT, fast_fp_from_be, fast_fp_is_largest, fast_fp_to_be, field_bytes,
    require_ethereum, require_zcash, zcash_flags,
};
use super::error::PointError;
use super::fast::group::Jac;
use super::fast::{self, Engine, ops, with_engine};
use super::point::group_ops;
use crate::math::field::Fp;
use crate::math::numbers::Number;

/// A point of the curve `E: y² = x³ + b` over `F_p` of a [`PairingCurve`], which it borrows: an
/// element of `G1`, the subgroup of order `r`, when it comes from the generator, from
/// [`G1::new`] or from bytes, which check that; any point of `E(F_p)` when it comes from
/// [`G1::new_on_curve`], as hashing to the curve needs before [`clear_cofactor`](G1::clear_cofactor).
///
/// It is held in Jacobian coordinates of the pairing's own arithmetic (see the
/// [module](crate::math::pairing)), so that the group's operations take no inversion. The
/// operators take the points owned or borrowed; scalars are [`Number`]s of any sign and length,
/// not reduced modulo `r`, so that they act on points outside `G1` too.
///
/// **Nothing here is constant-time**: the time of a product depends on the bits of the scalar.
///
/// # Panics
///
/// The operators panic for points of different curves.
///
/// # Examples
///
/// ```
/// use a::math::pairing::PairingCurve;
/// use a::math::numbers::Number;
///
/// let c = PairingCurve::bn254();
/// let g = c.g1_generator();
/// let p = &g * &Number::from(5u8);
/// assert_eq!(&p - &g, g.double().double());
/// assert_eq!(&g * &(c.order() - 1u8), -&g);
/// let (x, y) = g.coordinates().unwrap();
/// assert_eq!((x.value(), y.value()), (Number::from(1u8), Number::from(2u8)));
/// ```
#[derive(Clone)]
pub struct G1<'c> {
    /// The curve.
    pub(super) curve: &'c PairingCurve,
    /// The point: Jacobian `X`, `Y`, `Z` in the engine's Montgomery limbs.
    pub(super) p: [[u64; 6]; 3],
    /// Whether the point is known to be in `G1`, which lets products use the endomorphism.
    pub(super) sub: bool,
}

impl<'c> G1<'c> {
    /// The point of `curve` held as the engine's `j`, known to be in `G1` if `sub`.
    pub(super) fn from_jac<E: Engine<N>, const N: usize>(
        curve: &'c PairingCurve,
        j: &Jac<fast::Fp<E, N>>,
        sub: bool,
    ) -> G1<'c> {
        let mut p = [[0u64; 6]; 3];
        j.store(&mut p);
        G1 { curve, p, sub }
    }

    /// The point in the engine's arithmetic.
    pub(super) fn jac<E: Engine<N>, const N: usize>(&self) -> Jac<fast::Fp<E, N>> {
        Jac::load(&self.p)
    }

    /// The point at infinity.
    pub(super) fn identity(curve: &'c PairingCurve) -> G1<'c> {
        with_engine!(curve, E, N => G1::from_jac::<E, N>(curve, &Jac::identity(), true))
    }

    /// The affine point `(x, y)` of the public field, not checked.
    pub(super) fn from_affine_unchecked(
        curve: &'c PairingCurve,
        x: &Fp<'_>,
        y: &Fp<'_>,
        sub: bool,
    ) -> G1<'c> {
        with_engine!(curve, E, N => {
            let j = Jac::from_affine(fast::Fp::<E, N>::from_public(x), fast::Fp::from_public(y));
            G1::from_jac::<E, N>(curve, &j, sub)
        })
    }

    /// `self + o`.
    #[track_caller]
    pub(super) fn sum(&self, o: &G1<'c>) -> G1<'c> {
        self.curve.check(o.curve);
        let c = self.curve;
        with_engine!(c, E, N => G1::from_jac::<E, N>(c, &self.jac::<E, N>().add(&o.jac::<E, N>()), self.sub && o.sub))
    }

    /// `−self`.
    pub(super) fn negated(&self) -> G1<'c> {
        let c = self.curve;
        with_engine!(c, E, N => G1::from_jac::<E, N>(c, &self.jac::<E, N>().neg(), self.sub))
    }

    /// `[k] self`.
    pub(super) fn times(&self, k: &Number) -> G1<'c> {
        let c = self.curve;
        with_engine!(c, E, N => {
            G1::from_jac::<E, N>(c, &ops::g1_mul(&self.jac::<E, N>(), k, self.sub), self.sub)
        })
    }

    /// Whether the points are equal.
    #[track_caller]
    pub(super) fn same_point(&self, o: &G1<'c>) -> bool {
        self.curve.check(o.curve);
        with_engine!(self.curve, E, N => self.jac::<E, N>().equals(&o.jac::<E, N>()))
    }

    /// The point `(x, y)` of `G1`.
    ///
    /// # Errors
    ///
    /// [`PointError::NotOnCurve`] if `y² ≠ x³ + b`, [`PointError::NotInSubgroup`] if the point is
    /// on the curve but not of order `r` (never on BN254, whose cofactor is 1).
    ///
    /// # Panics
    ///
    /// Panics if `x` or `y` is not an element of the curve's base field.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::pairing::{G1, PairingCurve, PointError};
    ///
    /// let c = PairingCurve::bls12_381();
    /// let f = c.base_field();
    /// let (x, y) = c.g1_generator().coordinates().unwrap();
    /// assert_eq!(G1::new(c, x.clone(), y.clone()).unwrap(), c.g1_generator());
    /// assert_eq!(G1::new(c, x, y.double()), Err(PointError::NotOnCurve));
    /// // (0, 2) is on y² = x³ + 4, of order 3.
    /// assert_eq!(G1::new(c, f.zero(), f.from_u64(2)), Err(PointError::NotInSubgroup));
    /// ```
    #[track_caller]
    pub fn new(curve: &'c PairingCurve, x: Fp<'c>, y: Fp<'c>) -> Result<G1<'c>, PointError> {
        let p = G1::new_on_curve(curve, x, y)?;
        if p.is_in_subgroup() {
            Ok(G1 { sub: true, ..p })
        } else {
            Err(PointError::NotInSubgroup)
        }
    }

    /// The point `(x, y)` of `E(F_p)`, in `G1` or not.
    ///
    /// # Errors
    ///
    /// [`PointError::NotOnCurve`] if `y² ≠ x³ + b`.
    ///
    /// # Panics
    ///
    /// Panics if `x` or `y` is not an element of the curve's base field.
    #[track_caller]
    pub fn new_on_curve(
        curve: &'c PairingCurve,
        x: Fp<'c>,
        y: Fp<'c>,
    ) -> Result<G1<'c>, PointError> {
        let p = curve.base_field().modulus();
        assert!(
            x.field().modulus() == p && y.field().modulus() == p,
            "elements of different fields"
        );
        with_engine!(curve, E, N => {
            let (x, y) = (fast::Fp::<E, N>::from_public(&x), fast::Fp::from_public(&y));
            if !Jac::on_curve(&x, &y, &E::tables().b) {
                return Err(PointError::NotOnCurve);
            }
            // On BN, E(F_p) is G1.
            let sub = curve.family == Family::Bn;
            Ok(G1::from_jac::<E, N>(curve, &Jac::from_affine(x, y), sub))
        })
    }

    /// The curve.
    #[must_use]
    pub fn curve(&self) -> &'c PairingCurve {
        self.curve
    }

    /// Whether this is the point at infinity, the neutral element.
    #[must_use]
    pub fn is_identity(&self) -> bool {
        with_engine!(self.curve, E, N => self.jac::<E, N>().is_identity())
    }

    /// The affine coordinates `(x, y)`, or `None` for the point at infinity: one inversion.
    #[must_use]
    pub fn coordinates(&self) -> Option<(Fp<'c>, Fp<'c>)> {
        with_engine!(self.curve, E, N => {
            let (x, y) = self.jac::<E, N>().to_affine()?;
            Some((x.to_public(), y.to_public()))
        })
    }

    /// `2P`.
    #[must_use]
    pub fn double(&self) -> G1<'c> {
        let c = self.curve;
        with_engine!(c, E, N => G1::from_jac::<E, N>(c, &self.jac::<E, N>().double(), self.sub))
    }

    /// Whether the point lies in `G1`, the subgroup of order `r`.
    ///
    /// On BN254, whose `E(F_p)` has prime order, every point does. On BLS12-381 by the
    /// endomorphism `φ(x, y) = (βx, y)` with `β` a cube root of unity: the point is in `G1` if and
    /// only if `φ(P) = [−x²] P` (Scott, *A note on group membership tests for G1, G2 and GT on BLS
    /// pairing-friendly curves*, ePrint 2021/1130, §6; proved by El Housni, Guillevic and
    /// Piellard, ePrint 2022/352). Two products by the 64-bit `x` instead of one by the 255-bit
    /// `r`.
    #[must_use]
    pub fn is_in_subgroup(&self) -> bool {
        with_engine!(self.curve, E, N => ops::g1_in_subgroup(&self.jac::<E, N>()))
    }

    /// The point in `G1` that the cofactor clearing of RFC 9380 maps it to: `[1 − x] P`, its
    /// `h_eff` for BLS12-381 (§8.8.1), which kills the part outside `G1` since
    /// `E(F_p) ≅ Z/(|x − 1| r) × Z/(|x − 1|/3)` (PARI's `ellgroup`): `x − 1` is the exponent of that
    /// part, a divisor of the cofactor `(x − 1)²/3`, not the cofactor itself. On BN254 the point
    /// itself.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::pairing::{G1, PairingCurve};
    ///
    /// let c = PairingCurve::bls12_381();
    /// let f = c.base_field();
    /// let t = G1::new_on_curve(c, f.zero(), f.from_u64(2)).unwrap(); // of order 3
    /// let p = &c.g1_generator() + &t;
    /// assert!(!p.is_in_subgroup());
    /// assert!(p.clear_cofactor().is_in_subgroup());
    /// ```
    #[must_use]
    pub fn clear_cofactor(&self) -> G1<'c> {
        let c = self.curve;
        with_engine!(c, E, N => G1::from_jac::<E, N>(c, &ops::g1_clear(&self.jac::<E, N>()), true))
    }

    /// `[h] P` for the cofactor `h` of `G1` ([`PairingCurve::g1_cofactor`]): a point of
    /// `G1` for every point of the curve, by a product by the whole cofactor, as arkworks' default
    /// clearing computes it (and its clearing of BN254's `G2`).
    #[must_use]
    pub fn mul_by_cofactor(&self) -> G1<'c> {
        let c = self.curve;
        with_engine!(c, E, N => {
            let h = &E::groups().h1;
            // Width 4: measured on an Apple M4 Max (BLS12-381's 126-bit cofactor, widths 2 to 6
            // alternated), widths 3 and 5 take 1.01 times its time.
            G1::from_jac::<E, N>(c, &ops::mul_wnaf(&self.jac::<E, N>(), h, 4), true)
        })
    }

    /// The compressed ZCash encoding of BLS12-381, 48 bytes: `x` in big-endian order with three
    /// flags in its top bits, `0x80` for compression, `0x40` for the point at infinity (then all
    /// else is zero) and `0x20` when `y` is the larger of `±y`.
    ///
    /// # Panics
    ///
    /// Panics on BN254, whose prime leaves no room for three flags.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::pairing::{G1, PairingCurve};
    ///
    /// let c = PairingCurve::bls12_381();
    /// let b = c.g1_generator().to_zcash_compressed();
    /// assert_eq!(b[..4], [0x97, 0xf1, 0xd3, 0xa7]);
    /// assert_eq!(G1::from_zcash(c, &b).unwrap(), c.g1_generator());
    /// ```
    #[track_caller]
    #[must_use]
    pub fn to_zcash_compressed(&self) -> Vec<u8> {
        require_zcash(self.curve);
        let n = field_bytes(self.curve.base_field());
        with_engine!(self.curve, E, N => match self.jac::<E, N>().to_affine() {
            None => {
                let mut b = vec![0u8; n];
                b[0] = COMPRESSED | INFINITY;
                b
            }
            Some((x, y)) => {
                let mut b = fast_fp_to_be(&x, n);
                b[0] |= COMPRESSED;
                if fast_fp_is_largest(&y) {
                    b[0] |= SORT;
                }
                b
            }
        })
    }

    /// The uncompressed ZCash encoding of BLS12-381, 96 bytes: `x` and `y` in big-endian order,
    /// `0x40` and zeros for the point at infinity.
    ///
    /// # Panics
    ///
    /// Panics on BN254.
    #[track_caller]
    #[must_use]
    pub fn to_zcash_uncompressed(&self) -> Vec<u8> {
        require_zcash(self.curve);
        let n = field_bytes(self.curve.base_field());
        with_engine!(self.curve, E, N => match self.jac::<E, N>().to_affine() {
            None => {
                let mut b = vec![0u8; 2 * n];
                b[0] = INFINITY;
                b
            }
            Some((x, y)) => {
                let mut b = fast_fp_to_be(&x, n);
                b.extend(fast_fp_to_be(&y, n));
                b
            }
        })
    }

    /// The point of a ZCash encoding of BLS12-381, compressed (48 bytes) or uncompressed (96), as
    /// the `bls12_381` crate reads them: the flags must agree with the length, the infinity flag
    /// allows no other bit, the sort flag only on a compressed finite point, the coordinates must
    /// be below `p`, the point on the curve and in `G1`.
    ///
    /// # Errors
    ///
    /// The [`PointError`] of the first check that fails.
    ///
    /// # Panics
    ///
    /// Panics on BN254.
    #[track_caller]
    pub fn from_zcash(curve: &'c PairingCurve, bytes: &[u8]) -> Result<G1<'c>, PointError> {
        require_zcash(curve);
        let n = field_bytes(curve.base_field());
        if bytes.len() != n && bytes.len() != 2 * n {
            return Err(PointError::Length);
        }
        let compressed = bytes.len() == n;
        let (c, inf, sort, b) = zcash_flags(bytes);
        if c != compressed {
            return Err(PointError::Flags);
        }
        if inf {
            if sort || b.iter().any(|&x| x != 0) {
                return Err(PointError::Flags);
            }
            return Ok(curve.g1_identity());
        }
        with_engine!(curve, E, N => {
            let x = fast_fp_from_be::<E, N>(&b[..n])?;
            let bb = &E::tables().b;
            let y = if compressed {
                let y = x.square().mul(&x).add(bb).sqrt().ok_or(PointError::NotOnCurve)?;
                if fast_fp_is_largest(&y) == sort { y } else { y.neg() }
            } else {
                if sort {
                    return Err(PointError::Flags);
                }
                let y = fast_fp_from_be::<E, N>(&b[n..])?;
                if !Jac::on_curve(&x, &y, bb) {
                    return Err(PointError::NotOnCurve);
                }
                y
            };
            let j = Jac::from_affine(x, y);
            if !ops::g1_in_subgroup(&j) {
                return Err(PointError::NotInSubgroup);
            }
            Ok(G1::from_jac::<E, N>(curve, &j, true))
        })
    }

    /// The encoding of Ethereum's EIP-196 for BN254, 64 bytes: `x` and `y` in big-endian order,
    /// zeros for the point at infinity.
    ///
    /// # Panics
    ///
    /// Panics on BLS12-381.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::pairing::{G1, PairingCurve};
    ///
    /// let c = PairingCurve::bn254();
    /// let b = c.g1_generator().to_ethereum();
    /// assert_eq!((b[31], b[63]), (1, 2));
    /// assert_eq!(G1::from_ethereum(c, &b).unwrap(), c.g1_generator());
    /// assert!(G1::from_ethereum(c, &[0; 64]).unwrap().is_identity());
    /// ```
    #[track_caller]
    #[must_use]
    pub fn to_ethereum(&self) -> Vec<u8> {
        require_ethereum(self.curve);
        let n = field_bytes(self.curve.base_field());
        with_engine!(self.curve, E, N => match self.jac::<E, N>().to_affine() {
            None => vec![0u8; 2 * n],
            Some((x, y)) => {
                let mut b = fast_fp_to_be(&x, n);
                b.extend(fast_fp_to_be(&y, n));
                b
            }
        })
    }

    /// The point of an EIP-196 encoding of BN254: 64 bytes, two coordinates below `p`, `(0, 0)`
    /// for the point at infinity, and otherwise a point of the curve.
    ///
    /// # Errors
    ///
    /// [`PointError::Length`], [`PointError::NotInField`] or [`PointError::NotOnCurve`].
    ///
    /// # Panics
    ///
    /// Panics on BLS12-381.
    #[track_caller]
    pub fn from_ethereum(curve: &'c PairingCurve, bytes: &[u8]) -> Result<G1<'c>, PointError> {
        require_ethereum(curve);
        let n = field_bytes(curve.base_field());
        if bytes.len() != 2 * n {
            return Err(PointError::Length);
        }
        with_engine!(curve, E, N => {
            let x = fast_fp_from_be::<E, N>(&bytes[..n])?;
            let y = fast_fp_from_be::<E, N>(&bytes[n..])?;
            if x.is_zero() && y.is_zero() {
                return Ok(curve.g1_identity());
            }
            if !Jac::on_curve(&x, &y, &E::tables().b) {
                return Err(PointError::NotOnCurve);
            }
            let j = Jac::from_affine(x, y);
            if !ops::g1_in_subgroup(&j) {
                return Err(PointError::NotInSubgroup);
            }
            Ok(G1::from_jac::<E, N>(curve, &j, true))
        })
    }
}

group_ops!(G1);
