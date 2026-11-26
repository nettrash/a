//! Why a point could not be made from coordinates or bytes.

use core::fmt;

/// Why coordinates or an encoding do not give a point of `G1` or `G2`.
///
/// # Examples
///
/// ```
/// use a::math::pairing::{G1, PairingCurve, PointError};
///
/// let c = PairingCurve::bls12_381();
/// assert_eq!(G1::from_zcash(c, &[0u8; 47]), Err(PointError::Length));
/// // The compressed point at infinity is 0xc0 and zeros; without the compression flag, or with
/// // the sort flag, it is not an encoding.
/// let mut b = [0u8; 48];
/// b[0] = 0xc0;
/// assert!(G1::from_zcash(c, &b).unwrap().is_identity());
/// b[0] = 0xe0;
/// assert_eq!(G1::from_zcash(c, &b), Err(PointError::Flags));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PointError {
    /// The input has a length that the encoding does not have.
    Length,
    /// The flag bits of a ZCash encoding contradict each other or the length: the compression flag
    /// on an uncompressed encoding or not on a compressed one, the infinity flag with other bits
    /// set, or the sort flag where it has no meaning.
    Flags,
    /// A coordinate is not below `p`.
    NotInField,
    /// The coordinates do not satisfy the equation of the curve; or, compressed, `x³ + b` has no
    /// square root.
    NotOnCurve,
    /// The point is on the curve but not in the subgroup of order `r`.
    NotInSubgroup,
}

impl fmt::Display for PointError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            PointError::Length => "the encoding has the wrong length",
            PointError::Flags => "the flag bits of the encoding are invalid",
            PointError::NotInField => "a coordinate is not below the prime",
            PointError::NotOnCurve => "the point is not on the curve",
            PointError::NotInSubgroup => "the point is not in the subgroup of prime order",
        })
    }
}

impl std::error::Error for PointError {}
