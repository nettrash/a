//! Pairings on elliptic curves: the optimal ate pairing `e: G1 × G2 → GT` on the curves BN254 and
//! BLS12-381, with their groups.
//!
//! * [`PairingCurve`]: a curve and its parameters, [`PairingCurve::bn254`] and
//!   [`PairingCurve::bls12_381`], with [`pairing`](PairingCurve::pairing),
//!   [`multi_pairing`](PairingCurve::multi_pairing), and the two halves of a pairing,
//!   [`miller_loop`](PairingCurve::miller_loop) and
//!   [`final_exponentiation`](PairingCurve::final_exponentiation).
//! * [`G1`]: points of `E(F_p)`; [`G2`]: points of the sextic twist `E'(F_p²)` (D-type for BN254,
//!   M-type for BLS12-381); both add with `+`, negate and subtract with `-` and multiply by a
//!   [`Number`](crate::math::numbers::Number) with `*`, with subgroup checks by endomorphisms,
//!   cofactor clearings (RFC 9380's on BLS12-381; on BN254, whose `G1` needs none, Fuentes-Castañeda,
//!   Knapp and Rodríguez-Henríquez's map on `G2`), and the ZCash (BLS12-381) and Ethereum (BN254)
//!   encodings.
//! * [`Gt`]: the values of the pairing, `r`-th roots of unity of `F_p¹²`, written
//!   multiplicatively.
//! * [`ethereum`]: Ethereum's precompiles for BN254 on bytes, `ecAdd`, `ecMul` and `ecPairing`
//!   (EIP-196, EIP-197).
//! * [`hash`]: hashing to `G1` and `G2` as RFC 9380 specifies: its four suites for BLS12-381
//!   (section 8.8), by the simplified SWU map, over `F_p²` for `G2`, and the 11- and 3-isogenies;
//!   and gnark-crypto's two for BN254's `G1`, by the Shallue–van de Woestijne map, as RFC 9380 has
//!   no suite for BN254. BN254's `G2` has no hashing here.
//!
//! The coordinates and values are given and taken in the towers of
//! [`a::math::field`](crate::math::field): [`Fp`](crate::math::field::Fp),
//! [`Fp2`](crate::math::field::Fp2) and [`Fp12`](crate::math::field::Fp12) of
//! [`Tower::bn254`](crate::math::field::Tower::bn254) and
//! [`Tower::bls12_381`](crate::math::field::Tower::bls12_381). Inside, the pairing computes on
//! arithmetic of its own: `F_p` in 4 or 6 limbs with the prime fixed at compile time (Montgomery's
//! products without the extra carry word that a full-width prime needs), `F_p²` products with two
//! reductions instead of three, and the same tower, bases and formulas above it, so that its
//! elements are the public ones. The values of the pairing are those of arkworks on both curves,
//! and of the `bls12_381` crate, blst and RELIC on BLS12-381, exactly.
//!
//! Like the rest of the library this is not constant-time: it is for mathematics and for verifying
//! with public data, not for secret keys.
//!
//! # Examples
//!
//! ```
//! use a::math::pairing::{G1, PairingCurve, pairing};
//! use a::math::numbers::Number;
//!
//! let c = PairingCurve::bls12_381();
//! let (p, q) = (c.g1_generator(), c.g2_generator());
//! let (a, b) = (Number::from(1234u32), Number::from(5678u32));
//! // Bilinearity: e(aP, bQ) = e(P, Q)^(ab).
//! assert_eq!(pairing(&(&p * &a), &(&q * &b)), pairing(&p, &q).pow(&(&a * &b)));
//! // A point travels as ZCash's 48 bytes.
//! let bytes = (&p * &a).to_zcash_compressed();
//! assert_eq!(G1::from_zcash(c, &bytes).unwrap(), &p * &a);
//! ```

mod curve;
mod encoding;
mod error;
pub mod ethereum;
mod fast;
mod g1;
mod g2;
mod gt;
pub mod hash;
mod miller;
mod point;

pub use curve::{Family, PairingCurve, Twist};
pub use error::PointError;
pub use g1::G1;
pub use g2::G2;
pub use gt::Gt;

/// The optimal ate pairing `e(P, Q)` on the curve of the points; see [`PairingCurve::pairing`].
///
/// # Panics
///
/// Panics if the points are of different curves.
///
/// # Examples
///
/// ```
/// use a::math::pairing::{PairingCurve, pairing};
///
/// let c = PairingCurve::bn254();
/// let e = pairing(&c.g1_generator(), &c.g2_generator());
/// assert_eq!(e, c.gt_generator());
/// assert_eq!(pairing(&c.g1_identity(), &c.g2_generator()), c.gt_identity());
/// assert_eq!(pairing(&-c.g1_generator(), &c.g2_generator()), e.inverse());
/// ```
#[track_caller]
#[must_use]
pub fn pairing<'c>(p: &G1<'c>, q: &G2<'c>) -> Gt<'c> {
    p.curve().pairing(p, q)
}
