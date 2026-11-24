//! `G2`: points of the sextic twist `E'(F_p²)`, `y² = x³ + b'`.

use super::curve::{Family, PairingCurve};
use super::encoding::{
    COMPRESSED, INFINITY, SORT, field_bytes, fp2_from_bytes, fp2_is_largest, fp2_to_bytes,
    require_ethereum, require_zcash, zcash_flags,
};
use super::error::PointError;
use super::point::{Jacobian, group_ops, on_curve};
use crate::math::field::Fp2;
use crate::math::numbers::Number;

/// A point of the sextic twist `E': y² = x³ + b'` over `F_p²` of a [`PairingCurve`], which it
/// borrows: an element of `G2`, the subgroup of order `r`, when it comes from the generator, from
/// [`G2::new`] or from bytes, which check that; any point of `E'(F_p²)` when it comes from
/// [`G2::new_on_curve`].
///
/// `G2` is the image on the twist of the eigenspace of the Frobenius map for the eigenvalue `p`
/// in `E(F_p¹²)[r]`, where its coordinates fit in `F_p²` instead of `F_p¹²`. Held in Jacobian
/// coordinates; scalars are [`Number`]s of any sign and length, not reduced modulo `r`.
///
/// **Nothing here is constant-time**.
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
/// let c = PairingCurve::bls12_381();
/// let q = c.g2_generator();
/// assert_eq!(&q * &Number::from(3u8), &q.double() + &q);
/// // ψ acts on G2 as the product by p, which is x modulo r for BLS12.
/// assert_eq!(q.psi(), &q * c.x());
/// assert!(q.is_in_subgroup());
/// ```
#[derive(Clone)]
pub struct G2<'c> {
    /// The curve.
    pub(super) curve: &'c PairingCurve,
    /// The point.
    pub(super) p: Jacobian<Fp2<'c>>,
}

impl<'c> G2<'c> {
    /// The point of `curve` held as `p`.
    pub(super) fn from_jacobian(curve: &'c PairingCurve, p: Jacobian<Fp2<'c>>) -> G2<'c> {
        G2 { curve, p }
    }

    /// The point `(x, y)` of `G2`.
    ///
    /// # Errors
    ///
    /// [`PointError::NotOnCurve`] if `y² ≠ x³ + b'`, [`PointError::NotInSubgroup`] if the point is
    /// on the twist but not of order `r`.
    ///
    /// # Panics
    ///
    /// Panics if `x` or `y` is not an element of the curve's tower.
    #[track_caller]
    pub fn new(curve: &'c PairingCurve, x: Fp2<'c>, y: Fp2<'c>) -> Result<G2<'c>, PointError> {
        let q = G2::new_on_curve(curve, x, y)?;
        if q.is_in_subgroup() {
            Ok(q)
        } else {
            Err(PointError::NotInSubgroup)
        }
    }

    /// The point `(x, y)` of `E'(F_p²)`, in `G2` or not.
    ///
    /// # Errors
    ///
    /// [`PointError::NotOnCurve`] if `y² ≠ x³ + b'`.
    ///
    /// # Panics
    ///
    /// Panics if `x` or `y` is not an element of the curve's tower.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::Fp2;
    /// use a::math::pairing::{G2, PairingCurve};
    ///
    /// let c = PairingCurve::bn254();
    /// let t = c.tower();
    /// // The point of the twist with the least x = k + u, outside G2.
    /// let mut x = Fp2::new(t, t.base().zero(), t.base().one());
    /// let y = loop {
    ///     if let Some(y) = (&(&x.square() * &x) + &c.twist_b()).sqrt() {
    ///         break y;
    ///     }
    ///     x = &x + &Fp2::one(t);
    /// };
    /// let q = G2::new_on_curve(c, x, y).unwrap();
    /// assert!(!q.is_in_subgroup());
    /// assert!(q.clear_cofactor().is_in_subgroup());
    /// ```
    #[track_caller]
    pub fn new_on_curve(
        curve: &'c PairingCurve,
        x: Fp2<'c>,
        y: Fp2<'c>,
    ) -> Result<G2<'c>, PointError> {
        let t = curve.tower();
        assert!(
            x.tower().base().modulus() == t.base().modulus()
                && y.tower().base().modulus() == t.base().modulus()
                && x.tower().xi() == t.xi()
                && y.tower().xi() == t.xi(),
            "elements of different towers"
        );
        if !on_curve(&x, &y, &curve.b_twist) {
            return Err(PointError::NotOnCurve);
        }
        Ok(G2::from_jacobian(curve, Jacobian::from_affine(x, y)))
    }

    /// The curve.
    #[must_use]
    pub fn curve(&self) -> &'c PairingCurve {
        self.curve
    }

    /// Whether this is the point at infinity, the neutral element.
    #[must_use]
    pub fn is_identity(&self) -> bool {
        self.p.is_identity()
    }

    /// The affine coordinates `(x, y)`, or `None` for the point at infinity: one inversion.
    #[must_use]
    pub fn coordinates(&self) -> Option<(Fp2<'c>, Fp2<'c>)> {
        self.p.to_affine()
    }

    /// `2Q`.
    #[must_use]
    pub fn double(&self) -> G2<'c> {
        G2::from_jacobian(self.curve, self.p.double())
    }

    /// The endomorphism `ψ` of the twist: untwist to `E(F_p¹²)`, apply the Frobenius map
    /// `(x, y) ↦ (xᵖ, yᵖ)`, twist back. In coordinates `ψ(x, y) = (x̄ γₓ, ȳ γᵧ)` with
    /// `γₓ = ξ^((p − 1)/3)`, `γᵧ = ξ^((p − 1)/2)` on a D-type twist and their inverses on an M-type
    /// one: two products of `F_p²` and no inversion. On `G2` it is the product by `p`.
    #[must_use]
    pub fn psi(&self) -> G2<'c> {
        let [gx, gy] = &self.curve.psi;
        G2::from_jacobian(
            self.curve,
            Jacobian {
                x: &self.p.x.conjugate() * gx,
                y: &self.p.y.conjugate() * gy,
                z: self.p.z.conjugate(),
            },
        )
    }

    /// `ψ²`, with coefficients in `F_p`: `ψ²(x, y) = (x N(γₓ), y N(γᵧ))`.
    pub(super) fn psi2(&self) -> G2<'c> {
        let [gx, gy] = &self.curve.psi2;
        G2::from_jacobian(
            self.curve,
            Jacobian {
                x: self.p.x.mul_by_base(gx),
                y: self.p.y.mul_by_base(gy),
                z: self.p.z.clone(),
            },
        )
    }

    /// Whether the point lies in `G2`, the subgroup of order `r`: by the endomorphism, `ψ(Q) = [p]Q`
    /// on `G2`, and `p ≡ 6x² (mod r)` for BN, `p ≡ x` for BLS12; the converse, that no other point
    /// of the twist passes `ψ(Q) = [6x²] Q` on BN254 or `ψ(Q) = [x] Q` on BLS12-381, is shown by
    /// Scott (ePrint 2021/1130, §4) and El Housni, Guillevic and Piellard (ePrint 2022/352, §4).
    /// A product by a scalar of 64 or 128 bits instead of the 255-bit `r`.
    #[must_use]
    pub fn is_in_subgroup(&self) -> bool {
        self.psi().p.equals(&self.p.mul(&self.curve.g2_test))
    }

    /// A point of `G2` from any point of the twist, by an endomorphism that kills the part outside
    /// `G2`:
    ///
    /// * BLS12-381: Budroni and Pintore's `[x² − x − 1] Q + [x − 1] ψ(Q) + ψ²(2Q)` (*Efficient
    ///   hash maps to G2 on BLS curves*, ePrint 2017/419), the product by `h_eff` of RFC 9380
    ///   (§8.8.2, appendix G.3), as hashing to `G2` needs it.
    /// * BN254: Fuentes-Castañeda, Knapp and Rodríguez-Henríquez's
    ///   `[x] Q + ψ([3x] Q) + ψ²([x] Q) + ψ³(Q)` (*Faster hashing to G2*, SAC 2011, §6.1), which
    ///   on `G2` is the product by `x + 3xp + xp² + p³` modulo `r`.
    #[must_use]
    pub fn clear_cofactor(&self) -> G2<'c> {
        let x = &self.curve.x;
        match self.curve.family {
            Family::Bls12 => {
                let t1 = self * x;
                let t2 = self.psi();
                let t3 = self.double().psi2();
                let t3 = &t3 - &t2;
                let t2 = &(&t1 + &t2) * x;
                &(&(&t3 + &t2) - &t1) - self
            }
            Family::Bn => {
                let xq = self * x;
                let a = (&xq * &Number::from(3u8)).psi();
                let b = xq.psi2();
                let c = self.psi2().psi();
                &(&(&xq + &a) + &b) + &c
            }
        }
    }

    /// `[h] P` for the cofactor `h` of `G2` ([`PairingCurve::g2_cofactor`]): a point of
    /// `G2` for every point of the curve, by a product by the whole cofactor, as arkworks' default
    /// clearing computes it (and its clearing of BN254's `G2`).
    #[must_use]
    pub fn mul_by_cofactor(&self) -> G2<'c> {
        G2::from_jacobian(self.curve, self.p.mul(&self.curve.h2))
    }

    /// The compressed ZCash encoding of BLS12-381, 96 bytes: `x₁ ‖ x₀` of `x = x₀ + x₁u` in
    /// big-endian order, with the flags of [`G1::to_zcash_compressed`](super::G1::to_zcash_compressed)
    /// in the top bits; `y` is the larger of `±y` when its `c₁` is, or when `c₁ = 0` and its `c₀`
    /// is.
    ///
    /// # Panics
    ///
    /// Panics on BN254.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::pairing::{G2, PairingCurve};
    ///
    /// let c = PairingCurve::bls12_381();
    /// let b = c.g2_generator().to_zcash_compressed();
    /// assert_eq!(b[..4], [0x93, 0xe0, 0x2b, 0x60]);
    /// assert_eq!(G2::from_zcash(c, &b).unwrap(), c.g2_generator());
    /// ```
    #[track_caller]
    #[must_use]
    pub fn to_zcash_compressed(&self) -> Vec<u8> {
        require_zcash(self.curve);
        match self.coordinates() {
            None => {
                let mut b = vec![0u8; 2 * field_bytes(self.curve.base_field())];
                b[0] = COMPRESSED | INFINITY;
                b
            }
            Some((x, y)) => {
                let mut b = fp2_to_bytes(&x);
                b[0] |= COMPRESSED;
                if fp2_is_largest(&y) {
                    b[0] |= SORT;
                }
                b
            }
        }
    }

    /// The uncompressed ZCash encoding of BLS12-381, 192 bytes: `x₁ ‖ x₀ ‖ y₁ ‖ y₀`, `0x40` and
    /// zeros for the point at infinity.
    ///
    /// # Panics
    ///
    /// Panics on BN254.
    #[track_caller]
    #[must_use]
    pub fn to_zcash_uncompressed(&self) -> Vec<u8> {
        require_zcash(self.curve);
        match self.coordinates() {
            None => {
                let mut b = vec![0u8; 4 * field_bytes(self.curve.base_field())];
                b[0] = INFINITY;
                b
            }
            Some((x, y)) => {
                let mut b = fp2_to_bytes(&x);
                b.extend(fp2_to_bytes(&y));
                b
            }
        }
    }

    /// The point of a ZCash encoding of BLS12-381, compressed (96 bytes) or uncompressed (192),
    /// with the checks of [`G1::from_zcash`](super::G1::from_zcash), the root of `x³ + b'` taken by
    /// [`Fp2::sqrt`].
    ///
    /// # Errors
    ///
    /// The [`PointError`] of the first check that fails.
    ///
    /// # Panics
    ///
    /// Panics on BN254.
    #[track_caller]
    pub fn from_zcash(curve: &'c PairingCurve, bytes: &[u8]) -> Result<G2<'c>, PointError> {
        require_zcash(curve);
        let n = 2 * field_bytes(curve.base_field());
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
            return Ok(curve.g2_identity());
        }
        let x = fp2_from_bytes(curve, &b[..n])?;
        let y = if compressed {
            let y = (&(&x.square() * &x) + &curve.b_twist)
                .sqrt()
                .ok_or(PointError::NotOnCurve)?;
            if fp2_is_largest(&y) == sort { y } else { -y }
        } else {
            if sort {
                return Err(PointError::Flags);
            }
            fp2_from_bytes(curve, &b[n..])?
        };
        G2::new(curve, x, y)
    }

    /// The encoding of Ethereum's EIP-197 for BN254, 128 bytes: `x₁ ‖ x₀ ‖ y₁ ‖ y₀` of
    /// `x = x₀ + x₁u`, `y = y₀ + y₁u` in big-endian order (the coefficient of `u` first), zeros for
    /// the point at infinity.
    ///
    /// # Panics
    ///
    /// Panics on BLS12-381.
    #[track_caller]
    #[must_use]
    pub fn to_ethereum(&self) -> Vec<u8> {
        require_ethereum(self.curve);
        match self.coordinates() {
            None => vec![0u8; 4 * field_bytes(self.curve.base_field())],
            Some((x, y)) => {
                let mut b = fp2_to_bytes(&x);
                b.extend(fp2_to_bytes(&y));
                b
            }
        }
    }

    /// The point of an EIP-197 encoding of BN254: 128 bytes, four coefficients below `p`, all zero
    /// for the point at infinity, and otherwise a point of the twist in `G2`.
    ///
    /// # Errors
    ///
    /// [`PointError::Length`], [`PointError::NotInField`], [`PointError::NotOnCurve`] or
    /// [`PointError::NotInSubgroup`].
    ///
    /// # Panics
    ///
    /// Panics on BLS12-381.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::pairing::{G2, PairingCurve};
    ///
    /// let c = PairingCurve::bn254();
    /// let b = c.g2_generator().to_ethereum();
    /// assert_eq!(b[..4], [0x19, 0x8e, 0x93, 0x93]); // x₁ of the generator of EIP-197
    /// assert_eq!(G2::from_ethereum(c, &b).unwrap(), c.g2_generator());
    /// ```
    #[track_caller]
    pub fn from_ethereum(curve: &'c PairingCurve, bytes: &[u8]) -> Result<G2<'c>, PointError> {
        require_ethereum(curve);
        let n = 2 * field_bytes(curve.base_field());
        if bytes.len() != 2 * n {
            return Err(PointError::Length);
        }
        let x = fp2_from_bytes(curve, &bytes[..n])?;
        let y = fp2_from_bytes(curve, &bytes[n..])?;
        if x.is_zero() && y.is_zero() {
            return Ok(curve.g2_identity());
        }
        G2::new(curve, x, y)
    }
}

group_ops!(G2);
