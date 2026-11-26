//! The byte encodings of points: ZCash's for BLS12-381 (its specification, and appendix C of the
//! IETF draft on pairing-friendly curves) and Ethereum's for BN254 (EIP-196 and EIP-197).

use super::curve::{Family, PairingCurve};
use super::error::PointError;
use super::fast::{self, Engine};
use crate::math::field::PrimeField;

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

/// An element of the engine's `F_p` from big-endian bytes, `Err(NotInField)` unless below `p`.
pub(super) fn fast_fp_from_be<E: Engine<N>, const N: usize>(
    bytes: &[u8],
) -> Result<fast::Fp<E, N>, PointError> {
    let mut l = [0u64; N];
    if bytes.len() == 8 * N {
        // The usual length: big-endian words from the end, no copy.
        for (i, w) in l.iter_mut().enumerate() {
            let at = bytes.len() - 8 * (i + 1);
            *w = u64::from_be_bytes(bytes[at..at + 8].try_into().expect("8 bytes"));
        }
        return below_p::<E, N>(l);
    }
    let mut le = bytes.to_vec();
    le.reverse();
    for (i, chunk) in le.chunks(8).enumerate() {
        let mut w = [0u8; 8];
        w[..chunk.len()].copy_from_slice(chunk);
        if i >= N {
            if w != [0; 8] {
                return Err(PointError::NotInField);
            }
            continue;
        }
        l[i] = u64::from_le_bytes(w);
    }
    below_p::<E, N>(l)
}

/// The element of the canonical limbs `l`, `Err(NotInField)` unless below `p`.
fn below_p<E: Engine<N>, const N: usize>(l: [u64; N]) -> Result<fast::Fp<E, N>, PointError> {
    // Compare from the top limb.
    for i in (0..N).rev() {
        match l[i].cmp(&E::P[i]) {
            core::cmp::Ordering::Less => return Ok(fast::Fp::from_canonical(l)),
            core::cmp::Ordering::Greater => return Err(PointError::NotInField),
            core::cmp::Ordering::Equal => {}
        }
    }
    Err(PointError::NotInField)
}

/// The big-endian bytes of an element of the engine's `F_p`, `len` of them.
pub(super) fn fast_fp_to_be<E: Engine<N>, const N: usize>(
    x: &fast::Fp<E, N>,
    len: usize,
) -> Vec<u8> {
    let c = x.canonical();
    let mut le: Vec<u8> = c.iter().flat_map(|l| l.to_le_bytes()).collect();
    le.truncate(len);
    le.reverse();
    le
}

/// Whether `y` is the larger of `±y`: above `(p − 1)/2`.
pub(super) fn fast_fp_is_largest<E: Engine<N>, const N: usize>(y: &fast::Fp<E, N>) -> bool {
    let c = y.canonical();
    let half = &E::tables().legendre_exp;
    for i in (0..N).rev() {
        match c[i].cmp(&half[i]) {
            core::cmp::Ordering::Greater => return true,
            core::cmp::Ordering::Less => return false,
            core::cmp::Ordering::Equal => {}
        }
    }
    false
}

/// ZCash's order on `F_p²`: by `y₁`, then by `y₀` when `y₁ = 0`.
pub(super) fn fast_fp2_is_largest<E: Engine<N>, const N: usize>(y: &fast::Fp2<E, N>) -> bool {
    fast_fp_is_largest(&y.c1) || (y.c1.is_zero() && fast_fp_is_largest(&y.c0))
}

/// `x₁ ‖ x₀` in big-endian bytes, `len` each.
pub(super) fn fast_fp2_to_be<E: Engine<N>, const N: usize>(
    x: &fast::Fp2<E, N>,
    len: usize,
) -> Vec<u8> {
    let mut out = fast_fp_to_be(&x.c1, len);
    out.extend(fast_fp_to_be(&x.c0, len));
    out
}

/// An element of `F_p²` from `x₁ ‖ x₀`.
pub(super) fn fast_fp2_from_be<E: Engine<N>, const N: usize>(
    bytes: &[u8],
) -> Result<fast::Fp2<E, N>, PointError> {
    let n = bytes.len() / 2;
    let c1 = fast_fp_from_be(&bytes[..n])?;
    let c0 = fast_fp_from_be(&bytes[n..])?;
    Ok(fast::Fp2::new(c0, c1))
}
