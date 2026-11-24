//! Ethereum's precompiled contracts for BN254 (alt_bn128), on their bytes: `ecAdd` (address 6)
//! and `ecMul` (address 7) of EIP-196, `ecPairing` (address 8) of EIP-197.
//!
//! A point of `G1` is 64 bytes, `x ‖ y` in big-endian order, and one of `G2` 128 bytes,
//! `x₁ ‖ x₀ ‖ y₁ ‖ y₀` with the coefficient of `u` first; zeros are the point at infinity. As the
//! EIPs specify, the inputs of `ecAdd` and `ecMul` are read as if padded with zeros to 128 and 96
//! bytes, the bytes after those ignored, while that of `ecPairing` must be a multiple of 192
//! bytes. An input that is not a valid encoding (a coordinate not below `p`, a point off the curve,
//! or for `G2` outside its subgroup) is an error, where the contract fails.
//!
//! # Examples
//!
//! ```
//! use a::math::pairing::ethereum::{ec_add, ec_mul, ec_pairing};
//!
//! let mut g = [0u8; 64]; // (1, 2), the generator
//! g[31] = 1;
//! g[63] = 2;
//! let mut input = g.to_vec();
//! input.extend_from_slice(&g);
//! let twice = ec_add(&input).unwrap();
//! let mut by_two = g.to_vec();
//! by_two.extend_from_slice(&[0; 31]);
//! by_two.push(2);
//! assert_eq!(ec_mul(&by_two).unwrap(), twice);
//! // The empty product of pairings is 1.
//! assert_eq!(ec_pairing(&[]).unwrap()[31], 1);
//! ```

use super::curve::PairingCurve;
use super::error::PointError;
use super::g1::G1;
use super::g2::G2;
use crate::math::numbers::{ByteOrder, Number};

/// `input` cut or padded with zeros to `n` bytes.
fn padded(input: &[u8], n: usize) -> Vec<u8> {
    let mut b = input[..input.len().min(n)].to_vec();
    b.resize(n, 0);
    b
}

/// `ecAdd`: the sum of the two points of `G1` of the first 128 bytes, as 64 bytes.
///
/// # Errors
///
/// [`PointError::NotInField`] or [`PointError::NotOnCurve`] for an invalid point.
///
/// # Examples
///
/// ```
/// use a::math::pairing::ethereum::ec_add;
/// use a::math::pairing::{PairingCurve, PointError};
///
/// let g = PairingCurve::bn254().g1_generator();
/// let mut input = g.to_ethereum();
/// input.extend((-&g).to_ethereum());
/// assert_eq!(ec_add(&input).unwrap(), [0; 64]); // G − G
/// assert_eq!(ec_add(&[0xff; 128]), Err(PointError::NotInField));
/// ```
pub fn ec_add(input: &[u8]) -> Result<[u8; 64], PointError> {
    let c = PairingCurve::bn254();
    let b = padded(input, 128);
    let p = G1::from_ethereum(c, &b[..64])?;
    let q = G1::from_ethereum(c, &b[64..])?;
    Ok(to_array(&(&p + &q).to_ethereum()))
}

/// `ecMul`: the product of the point of `G1` of the first 64 bytes by the 256-bit big-endian
/// scalar of the next 32, as 64 bytes.
///
/// # Errors
///
/// [`PointError::NotInField`] or [`PointError::NotOnCurve`] for an invalid point.
///
/// # Examples
///
/// ```
/// use a::math::pairing::ethereum::ec_mul;
/// use a::math::pairing::PairingCurve;
/// use a::math::numbers::Number;
///
/// let g = PairingCurve::bn254().g1_generator();
/// let mut input = g.to_ethereum();
/// input.extend([0; 31]);
/// input.push(3);
/// assert_eq!(ec_mul(&input).unwrap().to_vec(), (&g * &Number::from(3u8)).to_ethereum());
/// ```
pub fn ec_mul(input: &[u8]) -> Result<[u8; 64], PointError> {
    let c = PairingCurve::bn254();
    let b = padded(input, 96);
    let p = G1::from_ethereum(c, &b[..64])?;
    let k = Number::new(b[64..].to_vec(), ByteOrder::BigEndian);
    Ok(to_array(&(&p * &k).to_ethereum()))
}

/// `ecPairing`: whether the product of the pairings of the pairs `(G1, G2)` of 192 bytes each is 1,
/// as a 32-byte word, 1 or 0. No pairs give 1.
///
/// # Errors
///
/// [`PointError::Length`] if the length is not a multiple of 192; the error of an invalid point
/// otherwise, [`PointError::NotInSubgroup`] included for `G2`.
///
/// # Examples
///
/// ```
/// use a::math::pairing::ethereum::ec_pairing;
/// use a::math::pairing::PairingCurve;
///
/// let c = PairingCurve::bn254();
/// // e(P, Q) · e(−P, Q) = 1.
/// let mut input = c.g1_generator().to_ethereum();
/// input.extend(c.g2_generator().to_ethereum());
/// input.extend((-c.g1_generator()).to_ethereum());
/// input.extend(c.g2_generator().to_ethereum());
/// assert_eq!(ec_pairing(&input).unwrap()[31], 1);
/// assert_eq!(ec_pairing(&input[..192]).unwrap()[31], 0);
/// ```
pub fn ec_pairing(input: &[u8]) -> Result<[u8; 32], PointError> {
    if !input.len().is_multiple_of(192) {
        return Err(PointError::Length);
    }
    let c = PairingCurve::bn254();
    let mut points = Vec::with_capacity(input.len() / 192);
    for chunk in input.chunks(192) {
        let p = G1::from_ethereum(c, &chunk[..64])?;
        let q = G2::from_ethereum(c, &chunk[64..])?;
        points.push((p, q));
    }
    let pairs: Vec<(&G1<'_>, &G2<'_>)> = points.iter().map(|(p, q)| (p, q)).collect();
    let mut out = [0u8; 32];
    out[31] = u8::from(c.multi_pairing(&pairs).is_identity());
    Ok(out)
}

/// The 64 bytes of a point.
fn to_array(b: &[u8]) -> [u8; 64] {
    let mut out = [0u8; 64];
    out.copy_from_slice(b);
    out
}
