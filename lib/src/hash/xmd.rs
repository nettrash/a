//! `expand_message_xmd` of RFC 9380, section 5.3.1, with the hashing of long domain separation
//! tags of section 5.3.3.

use super::Digest;

/// The prefix of a domain separation tag longer than 255 bytes, before it is hashed (RFC 9380,
/// section 5.3.3).
const OVERSIZE: &[u8] = b"H2C-OVERSIZE-DST-";

/// `len_in_bytes` uniform bytes from the message `msg` and the domain separation tag `dst`, by
/// RFC 9380's `expand_message_xmd` (section 5.3.1) with the hash `H`:
///
/// ```text
/// b_0 = H(Z_pad || msg || I2OSP(len_in_bytes, 2) || I2OSP(0, 1) || DST_prime)
/// b_1 = H(b_0 || I2OSP(1, 1) || DST_prime)
/// b_i = H(strxor(b_0, b_(i - 1)) || I2OSP(i, 1) || DST_prime)
/// ```
///
/// with `Z_pad` a block of zeros and `DST_prime = DST || I2OSP(len(DST), 1)`; the output is the
/// first `len_in_bytes` bytes of `b_1 || ... || b_ell`. A tag longer than 255 bytes is first
/// replaced by `H("H2C-OVERSIZE-DST-" || DST)`, as section 5.3.3 requires. It hashes the
/// message once, in `b_0`, after a block of zeros, and then `ell = ceil(len_in_bytes /
/// OUTPUT_BYTES)` short strings. Asking for no bytes gives none, without hashing.
///
/// RFC 9380 requires a hash of at least `2·k` bits for a security level of `k` bits: SHA-256 for
/// 128, SHA-384 for 192 and SHA-512 for 256. That choice is the caller's.
///
/// Not constant-time: for public data, not secrets (see [`a::hash`](crate::hash)).
///
/// # Panics
///
/// Panics if `dst` is empty (section 3.1: "tags MUST have nonzero length"), if `len_in_bytes` is
/// above 65535 or needs more than 255 blocks of the hash's output (8160 bytes for SHA-256, 12240
/// for SHA-384, 16320 for SHA-512), and for a [`Digest`] whose output is longer than its block or
/// than 255 bytes, which the RFC's construction does not allow.
///
/// # Examples
///
/// ```
/// use a::hash::{Sha256, Sha512, expand_message_xmd};
///
/// // RFC 9380, appendix K.1: msg = "abc", 32 bytes.
/// let u = expand_message_xmd::<Sha256>(b"abc", b"QUUX-V01-CS02-with-expander-SHA256-128", 32);
/// assert_eq!(u[..4], [0xd8, 0xcc, 0xab, 0x23]);
///
/// // Appendix K.3: SHA-512, an empty message, 32 bytes.
/// let u = expand_message_xmd::<Sha512>(b"", b"QUUX-V01-CS02-with-expander-SHA512-256", 32);
/// assert_eq!(u[..4], [0x6b, 0x9a, 0x73, 0x12]);
///
/// // The length is hashed in: a longer output does not begin with a shorter one.
/// let long = expand_message_xmd::<Sha256>(b"abc", b"DST", 64);
/// assert_eq!(long.len(), 64);
/// assert_ne!(long[..32], expand_message_xmd::<Sha256>(b"abc", b"DST", 32)[..]);
/// ```
#[track_caller]
#[must_use]
pub fn expand_message_xmd<H: Digest>(msg: &[u8], dst: &[u8], len_in_bytes: usize) -> Vec<u8> {
    let b = H::OUTPUT_BYTES;
    assert!(
        !dst.is_empty(),
        "the domain separation tag must not be empty (RFC 9380, section 3.1)"
    );
    assert!(
        b > 0 && b <= H::BLOCK_BYTES && b <= 255,
        "expand_message_xmd needs a hash whose output is not longer than its block or 255 bytes"
    );
    let ell = len_in_bytes.div_ceil(b);
    assert!(
        len_in_bytes <= 65535 && ell <= 255,
        "expand_message_xmd gives at most 65535 bytes and 255 blocks of the hash, not {len_in_bytes} bytes"
    );
    if len_in_bytes == 0 {
        return Vec::new();
    }
    let hashed;
    let dst = if dst.len() > 255 {
        let mut h = H::new();
        h.update(OVERSIZE);
        h.update(dst);
        hashed = h.finalize();
        hashed.as_ref()
    } else {
        dst
    };
    // Both fit: the checks above bound them by 255 and 65535.
    let dst_len = [dst.len() as u8];
    let lib_str = (len_in_bytes as u16).to_be_bytes();

    let mut h = H::new();
    // Z_pad, a block of zeros. Its state could be computed once per hash; that is for a later pass.
    let zeros = [0u8; 128];
    let mut pad = H::BLOCK_BYTES;
    while pad > 0 {
        let n = pad.min(zeros.len());
        h.update(&zeros[..n]);
        pad -= n;
    }
    h.update(msg);
    h.update(&lib_str);
    h.update(&[0]);
    h.update(dst);
    h.update(&dst_len);
    let b0 = h.finalize();

    let mut out = Vec::with_capacity(ell * b);
    let mut chained = b0;
    for i in 1..=ell {
        if i > 1 {
            // strxor(b_0, b_(i - 1)), in place of the previous block.
            for (c, z) in chained.as_mut().iter_mut().zip(b0.as_ref()) {
                *c ^= z;
            }
        }
        let mut h = H::new();
        h.update(chained.as_ref());
        // `i` is at most 255 by the check above.
        h.update(&[i as u8]);
        h.update(dst);
        h.update(&dst_len);
        chained = h.finalize();
        out.extend_from_slice(chained.as_ref());
    }
    out.truncate(len_in_bytes);
    out
}
