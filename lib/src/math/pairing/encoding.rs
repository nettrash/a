//! The byte encodings of points: ZCash's for BLS12-381 (its specification, and appendix C of the
//! IETF draft on pairing-friendly curves) and Ethereum's for BN254 (EIP-196 and EIP-197).

use super::curve::{Family, PairingCurve};
use super::error::PointError;
use super::point::to_be_bytes;
use crate::math::field::{Fp, Fp2, PrimeField};
use crate::math::numbers::{ByteOrder, Number};

/// The ZCash flag of a compressed encoding.
pub(super) const COMPRESSED: u8 = 0x80;
/// The ZCash flag of the point at infinity.
pub(super) const INFINITY: u8 = 0x40;
/// The ZCash flag of the larger `y` of a compressed encoding.
pub(super) const SORT: u8 = 0x20;

/// Panics unless the curve is BLS12-381, the only one whose prime leaves the three bits ZCash's
/// flags need.
#[track_caller]
pub(super) fn require_zcash(curve: &PairingCurve) {
    assert!(
        curve.family() == Family::Bls12,
        "the ZCash encoding is that of BLS12-381, not of {}",
        curve.name()
    );
}

/// Panics unless the curve is BN254, the curve of EIP-196 and EIP-197.
#[track_caller]
pub(super) fn require_ethereum(curve: &PairingCurve) {
    assert!(
        curve.family() == Family::Bn,
        "the Ethereum encoding of EIP-196 and EIP-197 is that of BN254, not of {}",
        curve.name()
    );
}

/// The number of bytes of an element of `F_p`.
pub(super) fn field_bytes(f: &PrimeField) -> usize {
    f.bits().div_ceil(8) as usize
}

/// `x` in big-endian order, in [`field_bytes`].
pub(super) fn fp_to_bytes(x: &Fp<'_>) -> Vec<u8> {
    to_be_bytes(&x.value(), field_bytes(x.field()))
}

/// The element of the big-endian `bytes`, if it is below `p`.
pub(super) fn fp_from_bytes<'f>(f: &'f PrimeField, bytes: &[u8]) -> Result<Fp<'f>, PointError> {
    let n = Number::new(bytes.to_vec(), ByteOrder::BigEndian);
    if &n >= f.modulus() {
        return Err(PointError::NotInField);
    }
    Ok(f.element(&n))
}

/// Whether `y` is the larger of `±y`: above `(p − 1)/2`.
pub(super) fn fp_is_largest(y: &Fp<'_>) -> bool {
    let p = y.field().modulus();
    y.value() * 2u8 > p - 1u8
}

/// ZCash's order on `F_p²`: `y` is the larger of `±y` if `c₁` is, or if `c₁ = 0` and `c₀` is.
pub(super) fn fp2_is_largest(y: &Fp2<'_>) -> bool {
    fp_is_largest(y.c1()) || (y.c1().is_zero() && fp_is_largest(y.c0()))
}

/// The bytes of an element of `F_p²` for ZCash, `c₁` first, as for Ethereum's EIP-197.
pub(super) fn fp2_to_bytes(x: &Fp2<'_>) -> Vec<u8> {
    let mut out = fp_to_bytes(x.c1());
    out.extend(fp_to_bytes(x.c0()));
    out
}

/// The element of `F_p²` of `c₁ ‖ c₀`, each below `p`.
pub(super) fn fp2_from_bytes<'c>(
    curve: &'c PairingCurve,
    bytes: &[u8],
) -> Result<Fp2<'c>, PointError> {
    let n = bytes.len() / 2;
    let f = curve.base_field();
    let c1 = fp_from_bytes(f, &bytes[..n])?;
    let c0 = fp_from_bytes(f, &bytes[n..])?;
    Ok(Fp2::new(curve.tower(), c0, c1))
}

/// The three flags of the first byte of a ZCash encoding, and the bytes with them cleared.
pub(super) fn zcash_flags(bytes: &[u8]) -> (bool, bool, bool, Vec<u8>) {
    let mut b = bytes.to_vec();
    let flags = b[0];
    b[0] &= 0x1f;
    (
        flags & COMPRESSED != 0,
        flags & INFINITY != 0,
        flags & SORT != 0,
        b,
    )
}
