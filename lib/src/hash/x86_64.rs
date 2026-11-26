//! SHA-256's compression function on the SHA extensions of x86_64 (`SHA256RNDS2`, `SHA256MSG1`,
//! `SHA256MSG2`), detected at run time; the portable function remains for the CPUs without them
//! and under Miri. (x86_64 has no SHA-512 instructions in this sense: SHA-384 and SHA-512 stay
//! portable there.)
//!
//! The state is kept as the vectors `(a, b, e, f)` and `(c, d, g, h)` that `SHA256RNDS2` takes:
//! each call advances two rounds, and the second call of a group of four takes the upper half of
//! the words plus constants. `SHA256MSG1` and `SHA256MSG2` give four words of the schedule from
//! the sixteen before them, in two steps between which the words seven back are added.

use core::arch::x86_64::{
    __m128i, _mm_add_epi32, _mm_alignr_epi8, _mm_blend_epi16, _mm_loadu_si128, _mm_set_epi64x,
    _mm_sha256msg1_epu32, _mm_sha256msg2_epu32, _mm_sha256rnds2_epu32, _mm_shuffle_epi8,
    _mm_shuffle_epi32, _mm_storeu_si128,
};

/// Whether this CPU has the SHA extensions and the SSSE3 and SSE4.1 shuffles and blends the
/// function also takes.
#[inline]
pub(super) fn has_sha() -> bool {
    !cfg!(a_baseline)
        && std::is_x86_feature_detected!("sha")
        && std::is_x86_feature_detected!("ssse3")
        && std::is_x86_feature_detected!("sse4.1")
}

/// SHA-256's compression of each block of 64 bytes of `blocks` into `state`.
///
/// # Safety
///
/// The CPU must have the SHA extensions, SSSE3 and SSE4.1 ([`has_sha`]).
#[target_feature(enable = "sha,sse2,ssse3,sse4.1")]
pub(super) unsafe fn sha256(state: &mut [u32; 8], blocks: &[u8], k: &[u32; 64]) {
    // SAFETY: the unaligned loads and stores read and write the 8 words of `state`, 16 bytes at
    // a time of a block of 64 and 4 words of `k` at offsets below 64, all in bounds; the
    // instructions are those of the target features this function is compiled with.
    unsafe {
        // The bytes of each 32-bit word reversed: the message's words are big-endian.
        let swap = _mm_set_epi64x(0x0c0d_0e0f_0809_0a0b, 0x0405_0607_0001_0203);
        let dcba = _mm_shuffle_epi32(_mm_loadu_si128(state.as_ptr().cast()), 0xb1);
        let efgh = _mm_shuffle_epi32(_mm_loadu_si128(state.as_ptr().add(4).cast()), 0x1b);
        let mut abef = _mm_alignr_epi8(dcba, efgh, 8);
        let mut cdgh = _mm_blend_epi16(efgh, dcba, 0xf0);
        for block in blocks.as_chunks::<64>().0 {
            let (abef0, cdgh0) = (abef, cdgh);
            let p = block.as_ptr().cast::<__m128i>();
            let mut m: [__m128i; 4] = [
                _mm_shuffle_epi8(_mm_loadu_si128(p), swap),
                _mm_shuffle_epi8(_mm_loadu_si128(p.add(1)), swap),
                _mm_shuffle_epi8(_mm_loadu_si128(p.add(2)), swap),
                _mm_shuffle_epi8(_mm_loadu_si128(p.add(3)), swap),
            ];
            // Rounds 4i to 4i + 3 on the words m[i mod 4]. From i = 3 to 14 the words of group
            // i + 1 are completed (their first step was taken two groups before); from i = 1 to
            // 12 the first step of group i + 3 is taken.
            macro_rules! quad {
                // The last group, which updates no words.
                (last) => {{
                    let wk = _mm_add_epi32(m[3], _mm_loadu_si128(k.as_ptr().add(60).cast()));
                    cdgh = _mm_sha256rnds2_epu32(cdgh, abef, wk);
                    abef = _mm_sha256rnds2_epu32(abef, cdgh, _mm_shuffle_epi32(wk, 0x0e));
                }};
                ($i:expr) => {{
                    const I: usize = $i;
                    let wk = _mm_add_epi32(m[I % 4], _mm_loadu_si128(k.as_ptr().add(4 * I).cast()));
                    // (a, b, e, f) two rounds on, which is (c, d, g, h) two rounds later.
                    cdgh = _mm_sha256rnds2_epu32(cdgh, abef, wk);
                    if (3..=14).contains(&I) {
                        let seven = _mm_alignr_epi8(m[I % 4], m[(I + 3) % 4], 4);
                        m[(I + 1) % 4] =
                            _mm_sha256msg2_epu32(_mm_add_epi32(m[(I + 1) % 4], seven), m[I % 4]);
                    }
                    abef = _mm_sha256rnds2_epu32(abef, cdgh, _mm_shuffle_epi32(wk, 0x0e));
                    if (1..=12).contains(&I) {
                        m[(I + 3) % 4] = _mm_sha256msg1_epu32(m[(I + 3) % 4], m[I % 4]);
                    }
                }};
            }
            quad!(0);
            quad!(1);
            quad!(2);
            quad!(3);
            quad!(4);
            quad!(5);
            quad!(6);
            quad!(7);
            quad!(8);
            quad!(9);
            quad!(10);
            quad!(11);
            quad!(12);
            quad!(13);
            quad!(14);
            quad!(last);
            abef = _mm_add_epi32(abef, abef0);
            cdgh = _mm_add_epi32(cdgh, cdgh0);
        }
        let feba = _mm_shuffle_epi32(abef, 0x1b);
        let dchg = _mm_shuffle_epi32(cdgh, 0xb1);
        _mm_storeu_si128(state.as_mut_ptr().cast(), _mm_blend_epi16(feba, dchg, 0xf0));
        _mm_storeu_si128(
            state.as_mut_ptr().add(4).cast(),
            _mm_alignr_epi8(dchg, feba, 8),
        );
    }
}
