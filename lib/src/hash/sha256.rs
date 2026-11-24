//! SHA-256 and SHA-224 of FIPS 180-4 (sections 6.2 and 6.3): one compression function on blocks
//! of 64 bytes and 32-bit words, from two initial values.

use super::Digest;
use super::block::Blocks;

/// The round constants: the first 32 bits of the fractional parts of the cube roots of the first
/// 64 primes (FIPS 180-4, section 4.2.2).
const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

/// SHA-256's initial value: the first 32 bits of the fractional parts of the square roots of the
/// first 8 primes (section 5.3.3).
const IV256: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

/// SHA-224's initial value: the second 32 bits of the fractional parts of the square roots of the
/// 9th to 16th primes (section 5.3.2).
const IV224: [u32; 8] = [
    0xc1059ed8, 0x367cd507, 0x3070dd17, 0xf70e5939, 0xffc00b31, 0x68581511, 0x64f98fa7, 0xbefa4fa4,
];

/// The longest message FIPS 180-4 allows, `2^64 - 1` bits, in whole bytes.
const LIMIT: u128 = (1 << 61) - 1;

/// The state of the compression function and the message's buffer, which SHA-256 and SHA-224
/// share.
#[derive(Clone, Debug)]
struct Core {
    /// The eight working words between blocks.
    state: [u32; 8],
    /// The partial block and the length.
    blocks: Blocks<64>,
}

impl Core {
    /// A core from the initial value `iv`.
    const fn new(iv: [u32; 8]) -> Core {
        Core {
            state: iv,
            blocks: Blocks::new(LIMIT),
        }
    }

    /// Takes the next piece of the message.
    #[track_caller]
    fn update(&mut self, data: &[u8]) {
        let state = &mut self.state;
        self.blocks.update(data, |b| compress(state, b));
    }

    /// The final state, after the padding.
    fn finish(self) -> [u32; 8] {
        let mut state = self.state;
        self.blocks.finish::<8>(|b| compress(&mut state, b));
        state
    }
}

/// The compression function of section 6.2.2 on each block of 64 bytes in `blocks`.
fn compress(state: &mut [u32; 8], blocks: &[u8]) {
    for block in blocks.as_chunks::<64>().0 {
        // The message schedule.
        let mut w = [0u32; 64];
        for (t, word) in block.as_chunks::<4>().0.iter().enumerate() {
            w[t] = u32::from_be_bytes(*word);
        }
        for t in 16..64 {
            let s0 = w[t - 15].rotate_right(7) ^ w[t - 15].rotate_right(18) ^ (w[t - 15] >> 3);
            let s1 = w[t - 2].rotate_right(17) ^ w[t - 2].rotate_right(19) ^ (w[t - 2] >> 10);
            w[t] = w[t - 16]
                .wrapping_add(s0)
                .wrapping_add(w[t - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
        for t in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[t])
                .wrapping_add(w[t]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (s, v) in state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
            *s = s.wrapping_add(v);
        }
    }
}

/// The first `N` bytes of the words, each in big-endian order.
fn output<const N: usize>(state: &[u32; 8]) -> [u8; N] {
    let mut out = [0; N];
    for (chunk, word) in out.chunks_mut(4).zip(state) {
        chunk.copy_from_slice(&word.to_be_bytes()[..chunk.len()]);
    }
    out
}

/// SHA-256 of FIPS 180-4: a digest of 32 bytes of a message of any length below 2⁶¹ bytes, taken
/// in pieces of any length.
///
/// Not constant-time: for public data, not secrets (see [`a::hash`](crate::hash)).
///
/// # Examples
///
/// ```
/// use a::hash::{Sha256, sha256};
///
/// let mut h = Sha256::new();
/// h.update(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq");
/// let d = h.finalize();
/// // FIPS 180-4's example of two blocks.
/// assert_eq!(d[..4], [0x24, 0x8d, 0x6a, 0x61]);
/// assert_eq!(d[28..], [0x19, 0xdb, 0x06, 0xc1]);
/// assert_eq!(sha256(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"), d);
/// ```
#[derive(Clone, Debug)]
pub struct Sha256(Core);

impl Sha256 {
    /// A hasher that has taken nothing yet.
    #[must_use]
    pub const fn new() -> Sha256 {
        Sha256(Core::new(IV256))
    }

    /// Takes the next piece of the message.
    ///
    /// # Panics
    ///
    /// Panics if the message grows to 2⁶¹ bytes or more, beyond the `2^64 − 1` bits of FIPS
    /// 180-4.
    #[track_caller]
    pub fn update(&mut self, data: &[u8]) {
        self.0.update(data);
    }

    /// The digest of the whole message: the eight words of the state in big-endian order.
    #[must_use]
    pub fn finalize(self) -> [u8; 32] {
        output(&self.0.finish())
    }
}

impl Default for Sha256 {
    fn default() -> Sha256 {
        Sha256::new()
    }
}

impl Digest for Sha256 {
    const OUTPUT_BYTES: usize = 32;
    const BLOCK_BYTES: usize = 64;
    type Output = [u8; 32];

    fn new() -> Sha256 {
        Sha256::new()
    }

    #[track_caller]
    fn update(&mut self, data: &[u8]) {
        Sha256::update(self, data);
    }

    fn finalize(self) -> [u8; 32] {
        Sha256::finalize(self)
    }
}

/// SHA-224 of FIPS 180-4: SHA-256 from another initial value, its digest cut to 28 bytes.
///
/// Not constant-time: for public data, not secrets (see [`a::hash`](crate::hash)).
///
/// # Examples
///
/// ```
/// use a::hash::{Sha224, sha224};
///
/// let mut h = Sha224::new();
/// h.update(b"ab");
/// h.update(b"c");
/// let d = h.finalize();
/// assert_eq!(d.len(), 28);
/// assert_eq!(d[..4], [0x23, 0x09, 0x7d, 0x22]);
/// assert_eq!(sha224(b"abc"), d);
/// ```
#[derive(Clone, Debug)]
pub struct Sha224(Core);

impl Sha224 {
    /// A hasher that has taken nothing yet.
    #[must_use]
    pub const fn new() -> Sha224 {
        Sha224(Core::new(IV224))
    }

    /// Takes the next piece of the message.
    ///
    /// # Panics
    ///
    /// Panics if the message grows to 2⁶¹ bytes or more, beyond the `2^64 − 1` bits of FIPS
    /// 180-4.
    #[track_caller]
    pub fn update(&mut self, data: &[u8]) {
        self.0.update(data);
    }

    /// The digest of the whole message: the first seven words of the state in big-endian order.
    #[must_use]
    pub fn finalize(self) -> [u8; 28] {
        output(&self.0.finish())
    }
}

impl Default for Sha224 {
    fn default() -> Sha224 {
        Sha224::new()
    }
}

impl Digest for Sha224 {
    const OUTPUT_BYTES: usize = 28;
    const BLOCK_BYTES: usize = 64;
    type Output = [u8; 28];

    fn new() -> Sha224 {
        Sha224::new()
    }

    #[track_caller]
    fn update(&mut self, data: &[u8]) {
        Sha224::update(self, data);
    }

    fn finalize(self) -> [u8; 28] {
        Sha224::finalize(self)
    }
}

/// The SHA-256 digest of `data`.
///
/// # Examples
///
/// ```
/// use a::hash::sha256;
///
/// // FIPS 180-4's example: SHA-256("abc").
/// assert_eq!(
///     sha256(b"abc"),
///     [
///         0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea, 0x41, 0x41, 0x40, 0xde, 0x5d, 0xae,
///         0x22, 0x23, 0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c, 0xb4, 0x10, 0xff, 0x61,
///         0xf2, 0x00, 0x15, 0xad,
///     ]
/// );
/// ```
#[must_use]
pub fn sha256(data: &[u8]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(data);
    h.finalize()
}

/// The SHA-224 digest of `data`.
///
/// # Examples
///
/// ```
/// use a::hash::sha224;
///
/// // SHA-224 of the empty message.
/// assert_eq!(sha224(b"")[..4], [0xd1, 0x4a, 0x02, 0x8c]);
/// ```
#[must_use]
pub fn sha224(data: &[u8]) -> [u8; 28] {
    let mut h = Sha224::new();
    h.update(data);
    h.finalize()
}
