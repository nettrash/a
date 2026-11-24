//! Hash functions: SHA-224, SHA-256, SHA-384 and SHA-512 of FIPS 180-4, and the
//! `expand_message_xmd` of RFC 9380 that hashing to curves builds on.
//!
//! * [`Sha256`] and [`Sha512`], with [`Sha224`] and [`Sha384`], the same functions from other
//!   initial values with their outputs cut short: streaming hashers that take the message in
//!   pieces of any length, by [`update`](Sha256::update), and give the digest by
//!   [`finalize`](Sha256::finalize); and the one-shot functions [`sha224`], [`sha256`], [`sha384`]
//!   and [`sha512`].
//! * [`Digest`]: what the four share, for code that is generic over the hash, as
//!   [`expand_message_xmd`] is.
//! * [`expand_message_xmd`]: RFC 9380's expansion of a message and a domain separation tag into
//!   as many uniform bytes as asked for, with the hashing of tags longer than 255 bytes of its
//!   section 5.3.3. [`a::math::ec::hash`](crate::math::ec::hash) and
//!   [`a::math::pairing::hash`](crate::math::pairing::hash) hash to fields, to curves and to the
//!   groups of pairings with it.
//!
//! The compression functions are those of FIPS 180-4, section 6, in portable Rust.
//!
//! **Nothing here is constant-time**, like the rest of the library: the hashers branch on the
//! lengths of what they are given, and no care is taken against timing or cache attacks. They are
//! for hashing public data (checksums, test vectors, the messages of hashing to curves), not for
//! keys or other secrets an attacker could time.
//!
//! # Examples
//!
//! ```
//! use a::hash::{Digest, Sha256, expand_message_xmd, sha256};
//!
//! // FIPS 180-4's first example, in one call and in pieces.
//! let d = sha256(b"abc");
//! assert_eq!(d[..4], [0xba, 0x78, 0x16, 0xbf]);
//! let mut h = Sha256::new();
//! h.update(b"a");
//! h.update(b"bc");
//! assert_eq!(h.finalize(), d);
//! assert_eq!(Sha256::digest(b"abc"), d);
//!
//! // RFC 9380, appendix K.1: 32 uniform bytes from an empty message.
//! let u = expand_message_xmd::<Sha256>(b"", b"QUUX-V01-CS02-with-expander-SHA256-128", 32);
//! assert_eq!(u[..4], [0x68, 0xa9, 0x85, 0xb8]);
//! ```

mod block;
mod sha256;
mod sha512;
mod xmd;

pub use sha256::{Sha224, Sha256, sha224, sha256};
pub use sha512::{Sha384, Sha512, sha384, sha512};
pub use xmd::expand_message_xmd;

/// A hash function that takes its message in pieces: what [`Sha224`], [`Sha256`], [`Sha384`] and
/// [`Sha512`] share, so that code such as [`expand_message_xmd`] can be generic over the hash.
///
/// An implementation for another hash must give digests of exactly
/// [`OUTPUT_BYTES`](Digest::OUTPUT_BYTES) bytes, whatever pieces the message came in, and a
/// fresh, empty hasher from [`new`](Digest::new) and [`Default`].
///
/// # Examples
///
/// ```
/// use a::hash::{Digest, Sha256, Sha512};
///
/// fn hex<H: Digest>(data: &[u8]) -> String {
///     H::digest(data).as_ref().iter().map(|b| format!("{b:02x}")).collect()
/// }
/// assert_eq!(hex::<Sha256>(b"").len(), 2 * Sha256::OUTPUT_BYTES);
/// assert!(hex::<Sha512>(b"abc").starts_with("ddaf35a1"));
/// assert_eq!(Sha512::BLOCK_BYTES, 128);
/// ```
pub trait Digest: Clone + Default {
    /// The length of a digest in bytes: 28, 32, 48 or 64 for SHA-224, SHA-256, SHA-384 and
    /// SHA-512.
    const OUTPUT_BYTES: usize;

    /// The length in bytes of the blocks the compression function takes, RFC 9380's
    /// `s_in_bytes`: 64 for SHA-224 and SHA-256, 128 for SHA-384 and SHA-512.
    const BLOCK_BYTES: usize;

    /// The digest: an array of [`OUTPUT_BYTES`](Digest::OUTPUT_BYTES) bytes.
    type Output: AsRef<[u8]> + AsMut<[u8]> + Copy + Eq + core::hash::Hash + core::fmt::Debug;

    /// A hasher that has taken nothing yet.
    fn new() -> Self;

    /// Takes the next piece of the message.
    ///
    /// # Panics
    ///
    /// Panics if the message grows longer than the hash can take: 2⁶¹ − 1 bytes for SHA-224 and
    /// SHA-256 (FIPS 180-4 counts its length in 64 bits), 2¹²⁵ − 1 for SHA-384 and SHA-512.
    fn update(&mut self, data: &[u8]);

    /// The digest of the whole message.
    fn finalize(self) -> Self::Output;

    /// The digest of `data`, in one call.
    #[must_use]
    fn digest(data: &[u8]) -> Self::Output {
        let mut h = Self::new();
        h.update(data);
        h.finalize()
    }
}
