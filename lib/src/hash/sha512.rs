//! SHA-512 and SHA-384 of FIPS 180-4 (sections 6.4 and 6.5): one compression function on blocks
//! of 128 bytes and 64-bit words, from two initial values.

use super::Digest;
use super::block::Blocks;

/// The round constants: the first 64 bits of the fractional parts of the cube roots of the first
/// 80 primes (FIPS 180-4, section 4.2.3).
pub(super) const K: [u64; 80] = [
    0x428a2f98d728ae22,
    0x7137449123ef65cd,
    0xb5c0fbcfec4d3b2f,
    0xe9b5dba58189dbbc,
    0x3956c25bf348b538,
    0x59f111f1b605d019,
    0x923f82a4af194f9b,
    0xab1c5ed5da6d8118,
    0xd807aa98a3030242,
    0x12835b0145706fbe,
    0x243185be4ee4b28c,
    0x550c7dc3d5ffb4e2,
    0x72be5d74f27b896f,
    0x80deb1fe3b1696b1,
    0x9bdc06a725c71235,
    0xc19bf174cf692694,
    0xe49b69c19ef14ad2,
    0xefbe4786384f25e3,
    0x0fc19dc68b8cd5b5,
    0x240ca1cc77ac9c65,
    0x2de92c6f592b0275,
    0x4a7484aa6ea6e483,
    0x5cb0a9dcbd41fbd4,
    0x76f988da831153b5,
    0x983e5152ee66dfab,
    0xa831c66d2db43210,
    0xb00327c898fb213f,
    0xbf597fc7beef0ee4,
    0xc6e00bf33da88fc2,
    0xd5a79147930aa725,
    0x06ca6351e003826f,
    0x142929670a0e6e70,
    0x27b70a8546d22ffc,
    0x2e1b21385c26c926,
    0x4d2c6dfc5ac42aed,
    0x53380d139d95b3df,
    0x650a73548baf63de,
    0x766a0abb3c77b2a8,
    0x81c2c92e47edaee6,
    0x92722c851482353b,
    0xa2bfe8a14cf10364,
    0xa81a664bbc423001,
    0xc24b8b70d0f89791,
    0xc76c51a30654be30,
    0xd192e819d6ef5218,
    0xd69906245565a910,
    0xf40e35855771202a,
    0x106aa07032bbd1b8,
    0x19a4c116b8d2d0c8,
    0x1e376c085141ab53,
    0x2748774cdf8eeb99,
    0x34b0bcb5e19b48a8,
    0x391c0cb3c5c95a63,
    0x4ed8aa4ae3418acb,
    0x5b9cca4f7763e373,
    0x682e6ff3d6b2b8a3,
    0x748f82ee5defb2fc,
    0x78a5636f43172f60,
    0x84c87814a1f0ab72,
    0x8cc702081a6439ec,
    0x90befffa23631e28,
    0xa4506cebde82bde9,
    0xbef9a3f7b2c67915,
    0xc67178f2e372532b,
    0xca273eceea26619c,
    0xd186b8c721c0c207,
    0xeada7dd6cde0eb1e,
    0xf57d4f7fee6ed178,
    0x06f067aa72176fba,
    0x0a637dc5a2c898a6,
    0x113f9804bef90dae,
    0x1b710b35131c471b,
    0x28db77f523047d84,
    0x32caab7b40c72493,
    0x3c9ebe0a15c9bebc,
    0x431d67c49c100d4c,
    0x4cc5d4becb3e42b6,
    0x597f299cfc657e2a,
    0x5fcb6fab3ad6faec,
    0x6c44198c4a475817,
];

/// SHA-512's initial value: the first 64 bits of the fractional parts of the square roots of the
/// first 8 primes (section 5.3.5).
const IV512: [u64; 8] = [
    0x6a09e667f3bcc908,
    0xbb67ae8584caa73b,
    0x3c6ef372fe94f82b,
    0xa54ff53a5f1d36f1,
    0x510e527fade682d1,
    0x9b05688c2b3e6c1f,
    0x1f83d9abfb41bd6b,
    0x5be0cd19137e2179,
];

/// SHA-384's initial value: the first 64 bits of the fractional parts of the square roots of the
/// 9th to 16th primes (section 5.3.4).
const IV384: [u64; 8] = [
    0xcbbb9d5dc1059ed8,
    0x629a292a367cd507,
    0x9159015a3070dd17,
    0x152fecd8f70e5939,
    0x67332667ffc00b31,
    0x8eb44a8768581511,
    0xdb0c2e0d64f98fa7,
    0x47b5481dbefa4fa4,
];

/// The longest message FIPS 180-4 allows, `2^128 - 1` bits, in whole bytes.
const LIMIT: u128 = (1 << 125) - 1;

/// The state of the compression function and the message's buffer, which SHA-512 and SHA-384
/// share.
#[derive(Clone, Debug)]
struct Core {
    /// The eight working words between blocks.
    state: [u64; 8],
    /// The partial block and the length.
    blocks: Blocks<128>,
}

impl Core {
    /// A core from the initial value `iv`.
    const fn new(iv: [u64; 8]) -> Core {
        Core {
            state: iv,
            blocks: Blocks::new(LIMIT),
        }
    }

    /// A core that has taken one block of zeros, from the state `zero` after it.
    const fn after_zero_block(zero: [u64; 8]) -> Core {
        Core {
            state: zero,
            blocks: Blocks::after_one_block(LIMIT),
        }
    }

    /// Takes the next piece of the message.
    #[track_caller]
    fn update(&mut self, data: &[u8]) {
        let state = &mut self.state;
        self.blocks.update(data, |b| compress(state, b));
    }

    /// The final state, after the padding.
    fn finish(self) -> [u64; 8] {
        let mut state = self.state;
        self.blocks.finish::<16>(|b| compress(&mut state, b));
        state
    }
}

/// The compression function of section 6.4.2 on each block of 128 bytes in `blocks`: on the
/// SHA-512 instructions of ARMv8 where the CPU has them, else in portable Rust.
fn compress(state: &mut [u64; 8], blocks: &[u8]) {
    #[cfg(all(target_arch = "aarch64", not(miri)))]
    if super::aarch64::has_sha512() {
        // SAFETY: the CPU has FEAT_SHA512.
        unsafe { super::aarch64::sha512(state, blocks, &K) };
        return;
    }
    portable(state, blocks);
}

/// The compression function in portable Rust: the schedule kept as its last sixteen words, the
/// rounds unrolled eight at a time with the working words renamed rather than moved.
fn portable(state: &mut [u64; 8], blocks: &[u8]) {
    for block in blocks.as_chunks::<128>().0 {
        let mut w = [0u64; 16];
        for (t, word) in block.as_chunks::<8>().0.iter().enumerate() {
            w[t] = u64::from_be_bytes(*word);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
        // Round `t`, as in SHA-256: `d += t1` and `h = t1 + t2`, the names turning.
        macro_rules! round {
            ($a:ident, $b:ident, $c:ident, $d:ident, $e:ident, $f:ident, $g:ident, $h:ident, $t:expr, $wt:expr) => {{
                let s1 = $e.rotate_right(14) ^ $e.rotate_right(18) ^ $e.rotate_right(41);
                let ch = ($e & $f) ^ (!$e & $g);
                let t1 = $h
                    .wrapping_add(s1)
                    .wrapping_add(ch)
                    .wrapping_add(K[$t])
                    .wrapping_add($wt);
                let s0 = $a.rotate_right(28) ^ $a.rotate_right(34) ^ $a.rotate_right(39);
                let maj = ($a & $b) ^ ($c & ($a ^ $b));
                $d = $d.wrapping_add(t1);
                $h = t1.wrapping_add(s0.wrapping_add(maj));
            }};
        }
        macro_rules! given {
            ($t:expr) => {
                w[$t & 15]
            };
        }
        macro_rules! scheduled {
            ($t:expr) => {{
                let t = $t;
                let w15 = w[(t + 1) & 15];
                let w2 = w[(t + 14) & 15];
                let s0 = w15.rotate_right(1) ^ w15.rotate_right(8) ^ (w15 >> 7);
                let s1 = w2.rotate_right(19) ^ w2.rotate_right(61) ^ (w2 >> 6);
                w[t & 15] = w[t & 15]
                    .wrapping_add(s0)
                    .wrapping_add(w[(t + 9) & 15])
                    .wrapping_add(s1);
                w[t & 15]
            }};
        }
        macro_rules! eight {
            ($t:expr, $word:ident) => {{
                let t = $t;
                round!(a, b, c, d, e, f, g, h, t, $word!(t));
                round!(h, a, b, c, d, e, f, g, t + 1, $word!(t + 1));
                round!(g, h, a, b, c, d, e, f, t + 2, $word!(t + 2));
                round!(f, g, h, a, b, c, d, e, t + 3, $word!(t + 3));
                round!(e, f, g, h, a, b, c, d, t + 4, $word!(t + 4));
                round!(d, e, f, g, h, a, b, c, t + 5, $word!(t + 5));
                round!(c, d, e, f, g, h, a, b, t + 6, $word!(t + 6));
                round!(b, c, d, e, f, g, h, a, t + 7, $word!(t + 7));
            }};
        }
        eight!(0, given);
        eight!(8, given);
        for t in [16, 24, 32, 40, 48, 56, 64, 72] {
            eight!(t, scheduled);
        }
        for (s, v) in state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
            *s = s.wrapping_add(v);
        }
    }
}

/// The state after one block of zeros from `iv`, computed when the crate is built: the
/// `Z_pad` with which `expand_message_xmd` starts.
const fn after_zero_block(iv: [u64; 8]) -> [u64; 8] {
    let mut w = [0u64; 80];
    let mut t = 16;
    while t < 80 {
        let s0 = w[t - 15].rotate_right(1) ^ w[t - 15].rotate_right(8) ^ (w[t - 15] >> 7);
        let s1 = w[t - 2].rotate_right(19) ^ w[t - 2].rotate_right(61) ^ (w[t - 2] >> 6);
        w[t] = w[t - 16]
            .wrapping_add(s0)
            .wrapping_add(w[t - 7])
            .wrapping_add(s1);
        t += 1;
    }
    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = iv;
    let mut t = 0;
    while t < 80 {
        let s1 = e.rotate_right(14) ^ e.rotate_right(18) ^ e.rotate_right(41);
        let ch = (e & f) ^ (!e & g);
        let t1 = h
            .wrapping_add(s1)
            .wrapping_add(ch)
            .wrapping_add(K[t])
            .wrapping_add(w[t]);
        let s0 = a.rotate_right(28) ^ a.rotate_right(34) ^ a.rotate_right(39);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        h = g;
        g = f;
        f = e;
        e = d.wrapping_add(t1);
        d = c;
        c = b;
        b = a;
        a = t1.wrapping_add(s0.wrapping_add(maj));
        t += 1;
    }
    let v = [a, b, c, d, e, f, g, h];
    let mut out = iv;
    let mut i = 0;
    while i < 8 {
        out[i] = out[i].wrapping_add(v[i]);
        i += 1;
    }
    out
}

/// The first `N` bytes of the words, each in big-endian order.
fn output<const N: usize>(state: &[u64; 8]) -> [u8; N] {
    let mut out = [0; N];
    for (chunk, word) in out.chunks_mut(8).zip(state) {
        chunk.copy_from_slice(&word.to_be_bytes()[..chunk.len()]);
    }
    out
}

/// SHA-512 of FIPS 180-4: a digest of 64 bytes of a message of any length below 2¹²⁵ bytes, taken
/// in pieces of any length.
///
/// Not constant-time: for public data, not secrets (see [`a::hash`](crate::hash)).
///
/// # Examples
///
/// ```
/// use a::hash::{Sha512, sha512};
///
/// let mut h = Sha512::new();
/// h.update(b"a");
/// h.update(b"bc");
/// let d = h.finalize();
/// // FIPS 180-4's example: SHA-512("abc") begins ddaf35a1 and ends a54ca49f.
/// assert_eq!(d[..4], [0xdd, 0xaf, 0x35, 0xa1]);
/// assert_eq!(d[60..], [0xa5, 0x4c, 0xa4, 0x9f]);
/// assert_eq!(sha512(b"abc"), d);
/// ```
#[derive(Clone, Debug)]
pub struct Sha512(Core);

impl Sha512 {
    /// A hasher that has taken nothing yet.
    #[must_use]
    pub const fn new() -> Sha512 {
        Sha512(Core::new(IV512))
    }

    /// Takes the next piece of the message.
    ///
    /// # Panics
    ///
    /// Panics if the message grows to 2¹²⁵ bytes or more, beyond the `2^128 − 1` bits of FIPS
    /// 180-4.
    #[track_caller]
    pub fn update(&mut self, data: &[u8]) {
        self.0.update(data);
    }

    /// The digest of the whole message: the eight words of the state in big-endian order.
    #[must_use]
    pub fn finalize(self) -> [u8; 64] {
        output(&self.0.finish())
    }
}

impl Default for Sha512 {
    fn default() -> Sha512 {
        Sha512::new()
    }
}

impl Digest for Sha512 {
    const OUTPUT_BYTES: usize = 64;
    const BLOCK_BYTES: usize = 128;
    type Output = [u8; 64];

    fn new() -> Sha512 {
        Sha512::new()
    }

    #[track_caller]
    fn update(&mut self, data: &[u8]) {
        Sha512::update(self, data);
    }

    fn finalize(self) -> [u8; 64] {
        Sha512::finalize(self)
    }

    fn after_zero_block() -> Sha512 {
        const ZERO: [u64; 8] = after_zero_block(IV512);
        Sha512(Core::after_zero_block(ZERO))
    }
}

/// SHA-384 of FIPS 180-4: SHA-512 from another initial value, its digest cut to 48 bytes.
///
/// Not constant-time: for public data, not secrets (see [`a::hash`](crate::hash)).
///
/// # Examples
///
/// ```
/// use a::hash::{Sha384, sha384};
///
/// let mut h = Sha384::new();
/// h.update(b"abc");
/// let d = h.finalize();
/// // FIPS 180-4's example: SHA-384("abc") begins cb00753f and ends 34c825a7.
/// assert_eq!(d[..4], [0xcb, 0x00, 0x75, 0x3f]);
/// assert_eq!(d[44..], [0x34, 0xc8, 0x25, 0xa7]);
/// assert_eq!(sha384(b"abc"), d);
/// ```
#[derive(Clone, Debug)]
pub struct Sha384(Core);

impl Sha384 {
    /// A hasher that has taken nothing yet.
    #[must_use]
    pub const fn new() -> Sha384 {
        Sha384(Core::new(IV384))
    }

    /// Takes the next piece of the message.
    ///
    /// # Panics
    ///
    /// Panics if the message grows to 2¹²⁵ bytes or more, beyond the `2^128 − 1` bits of FIPS
    /// 180-4.
    #[track_caller]
    pub fn update(&mut self, data: &[u8]) {
        self.0.update(data);
    }

    /// The digest of the whole message: the first six words of the state in big-endian order.
    #[must_use]
    pub fn finalize(self) -> [u8; 48] {
        output(&self.0.finish())
    }
}

impl Default for Sha384 {
    fn default() -> Sha384 {
        Sha384::new()
    }
}

impl Digest for Sha384 {
    const OUTPUT_BYTES: usize = 48;
    const BLOCK_BYTES: usize = 128;
    type Output = [u8; 48];

    fn new() -> Sha384 {
        Sha384::new()
    }

    #[track_caller]
    fn update(&mut self, data: &[u8]) {
        Sha384::update(self, data);
    }

    fn finalize(self) -> [u8; 48] {
        Sha384::finalize(self)
    }

    fn after_zero_block() -> Sha384 {
        const ZERO: [u64; 8] = after_zero_block(IV384);
        Sha384(Core::after_zero_block(ZERO))
    }
}

/// The SHA-512 digest of `data`.
///
/// # Examples
///
/// ```
/// use a::hash::sha512;
///
/// // SHA-512 of the empty message begins cf83e135.
/// assert_eq!(sha512(b"")[..4], [0xcf, 0x83, 0xe1, 0x35]);
/// ```
#[must_use]
pub fn sha512(data: &[u8]) -> [u8; 64] {
    let mut h = Sha512::new();
    h.update(data);
    h.finalize()
}

/// The SHA-384 digest of `data`.
///
/// # Examples
///
/// ```
/// use a::hash::sha384;
///
/// // SHA-384 of the empty message begins 38b060a7.
/// assert_eq!(sha384(b"")[..4], [0x38, 0xb0, 0x60, 0xa7]);
/// ```
#[must_use]
pub fn sha384(data: &[u8]) -> [u8; 48] {
    let mut h = Sha384::new();
    h.update(data);
    h.finalize()
}
