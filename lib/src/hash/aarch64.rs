//! The compression functions of SHA-256 and SHA-512 on the instructions of the ARMv8
//! cryptographic extensions: FEAT_SHA256 (`SHA256H`, `SHA256H2`, `SHA256SU0`, `SHA256SU1`) and
//! FEAT_SHA512 (`SHA512H`, `SHA512H2`, `SHA512SU0`, `SHA512SU1`), which Rust calls the target
//! features `sha2` and `sha3`. They are used when the target has them at compile time (as Apple's
//! do) or the CPU reports them at run time; the portable functions remain for the others and
//! under Miri.
//!
//! SHA-256 keeps the state as the two vectors `(a, b, c, d)` and `(e, f, g, h)`: `SHA256H` and
//! `SHA256H2` advance them by four rounds, and `SHA256SU0`/`SHA256SU1` give the next four words of
//! the schedule from the sixteen before them. SHA-512 keeps `(a, b)`, `(c, d)`, `(e, f)` and
//! `(g, h)`: `SHA512H` and `SHA512H2` advance two rounds, after which the four pairs play each
//! other's parts, so that the code below cycles through four forms of the double round, and
//! `SHA512SU0`/`SHA512SU1` give two words of the schedule at a time.

use core::arch::aarch64::{
    uint32x4_t, uint64x2_t, vaddq_u32, vaddq_u64, vextq_u64, vld1q_u8, vld1q_u32, vld1q_u64,
    vreinterpretq_u32_u8, vreinterpretq_u64_u8, vrev32q_u8, vrev64q_u8, vsha256h2q_u32,
    vsha256hq_u32, vsha256su0q_u32, vsha256su1q_u32, vsha512h2q_u64, vsha512hq_u64,
    vsha512su0q_u64, vsha512su1q_u64, vst1q_u32, vst1q_u64,
};

/// Whether this CPU has FEAT_SHA256.
#[inline]
pub(super) fn has_sha256() -> bool {
    !cfg!(a_baseline)
        && (cfg!(target_feature = "sha2") || std::arch::is_aarch64_feature_detected!("sha2"))
}

/// Whether this CPU has FEAT_SHA512.
#[inline]
pub(super) fn has_sha512() -> bool {
    !cfg!(a_baseline)
        && (cfg!(target_feature = "sha3") || std::arch::is_aarch64_feature_detected!("sha3"))
}

/// SHA-256's compression of each block of 64 bytes of `blocks` into `state`.
///
/// # Safety
///
/// The CPU must have FEAT_SHA256 ([`has_sha256`]).
#[target_feature(enable = "sha2")]
pub(super) unsafe fn sha256(state: &mut [u32; 8], blocks: &[u8], k: &[u32; 64]) {
    // SAFETY: the loads and stores read and write the 8 words of `state`, 16 bytes of a block of
    // 64 and 4 words of `k` at offsets below 64, all in bounds; the instructions are those of the
    // target feature this function is compiled with.
    unsafe {
        let mut abcd = vld1q_u32(state.as_ptr());
        let mut efgh = vld1q_u32(state.as_ptr().add(4));
        let kv: [uint32x4_t; 16] = core::array::from_fn(|i| vld1q_u32(k.as_ptr().add(4 * i)));
        for block in blocks.as_chunks::<64>().0 {
            let (abcd0, efgh0) = (abcd, efgh);
            let p = block.as_ptr();
            let mut m: [uint32x4_t; 4] = [
                vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(p))),
                vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(p.add(16)))),
                vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(p.add(32)))),
                vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(p.add(48)))),
            ];
            // Sixteen times four rounds on `wk`, the words plus constants of the group, computed
            // one group ahead; the first twelve groups also give the words four groups on.
            let mut wk = vaddq_u32(m[0], kv[0]);
            macro_rules! quad {
                ($i:expr, $w:expr) => {{
                    let next = vaddq_u32(m[($w + 1) % 4], kv[$i + 1]);
                    let prev = abcd;
                    abcd = vsha256hq_u32(abcd, efgh, wk);
                    efgh = vsha256h2q_u32(efgh, prev, wk);
                    wk = next;
                }};
                (last) => {{
                    let prev = abcd;
                    abcd = vsha256hq_u32(abcd, efgh, wk);
                    efgh = vsha256h2q_u32(efgh, prev, wk);
                }};
                ($i:expr, $w:expr, next) => {{
                    let next = vaddq_u32(m[($w + 1) % 4], kv[$i + 1]);
                    m[$w] = vsha256su1q_u32(
                        vsha256su0q_u32(m[$w], m[($w + 1) % 4]),
                        m[($w + 2) % 4],
                        m[($w + 3) % 4],
                    );
                    let prev = abcd;
                    abcd = vsha256hq_u32(abcd, efgh, wk);
                    efgh = vsha256h2q_u32(efgh, prev, wk);
                    wk = next;
                }};
            }
            quad!(0, 0, next);
            quad!(1, 1, next);
            quad!(2, 2, next);
            quad!(3, 3, next);
            quad!(4, 0, next);
            quad!(5, 1, next);
            quad!(6, 2, next);
            quad!(7, 3, next);
            quad!(8, 0, next);
            quad!(9, 1, next);
            quad!(10, 2, next);
            quad!(11, 3, next);
            quad!(12, 0);
            quad!(13, 1);
            quad!(14, 2);
            quad!(last);
            abcd = vaddq_u32(abcd, abcd0);
            efgh = vaddq_u32(efgh, efgh0);
        }
        vst1q_u32(state.as_mut_ptr(), abcd);
        vst1q_u32(state.as_mut_ptr().add(4), efgh);
    }
}

/// SHA-512's compression of each block of 128 bytes of `blocks` into `state`.
///
/// # Safety
///
/// The CPU must have FEAT_SHA512 ([`has_sha512`]).
#[target_feature(enable = "sha3")]
pub(super) unsafe fn sha512(state: &mut [u64; 8], blocks: &[u8], k: &[u64; 80]) {
    // SAFETY: as in `sha256`: the 8 words of `state`, 16 bytes at a time of a block of 128, two
    // words of `k` at even offsets below 80; the instructions of the target feature.
    unsafe {
        let mut ab = vld1q_u64(state.as_ptr());
        let mut cd = vld1q_u64(state.as_ptr().add(2));
        let mut ef = vld1q_u64(state.as_ptr().add(4));
        let mut gh = vld1q_u64(state.as_ptr().add(6));
        for block in blocks.as_chunks::<128>().0 {
            let saved = (ab, cd, ef, gh);
            let p = block.as_ptr();
            let load = |i: usize| vreinterpretq_u64_u8(vrev64q_u8(vld1q_u8(p.add(16 * i))));
            let mut s: [uint64x2_t; 8] = [
                load(0),
                load(1),
                load(2),
                load(3),
                load(4),
                load(5),
                load(6),
                load(7),
            ];
            // Two rounds on the words `w` with the constants from `t`; the pairs named `a`,
            // `c`, `e`, `g` stand for (a, b), (c, d), (e, f), (g, h) of this double round.
            macro_rules! double {
                ($t:expr, $w:expr, $a:ident, $c:ident, $e:ident, $g:ident) => {{
                    let wk = vaddq_u64($w, vld1q_u64(k.as_ptr().add($t)));
                    let sum = vaddq_u64(vextq_u64(wk, wk, 1), $g);
                    let mid = vsha512hq_u64(sum, vextq_u64($e, $g, 1), vextq_u64($c, $e, 1));
                    $g = vsha512h2q_u64(mid, $c, $a);
                    $c = vaddq_u64($c, mid);
                }};
            }
            // The next two words of the schedule into `s[i]`, from the sixteen before them.
            macro_rules! next {
                ($i:expr) => {{
                    s[$i] = vsha512su1q_u64(
                        vsha512su0q_u64(s[$i], s[($i + 1) % 8]),
                        s[($i + 7) % 8],
                        vextq_u64(s[($i + 4) % 8], s[($i + 5) % 8], 1),
                    );
                }};
            }
            // Eight double rounds, the pairs' parts turning every two rounds.
            macro_rules! eight {
                ($t:expr) => {{
                    double!($t, s[0], ab, cd, ef, gh);
                    double!($t + 2, s[1], gh, ab, cd, ef);
                    double!($t + 4, s[2], ef, gh, ab, cd);
                    double!($t + 6, s[3], cd, ef, gh, ab);
                    double!($t + 8, s[4], ab, cd, ef, gh);
                    double!($t + 10, s[5], gh, ab, cd, ef);
                    double!($t + 12, s[6], ef, gh, ab, cd);
                    double!($t + 14, s[7], cd, ef, gh, ab);
                }};
            }
            eight!(0);
            for t in [16, 32, 48, 64] {
                for i in 0..8 {
                    next!(i);
                }
                eight!(t);
            }
            ab = vaddq_u64(ab, saved.0);
            cd = vaddq_u64(cd, saved.1);
            ef = vaddq_u64(ef, saved.2);
            gh = vaddq_u64(gh, saved.3);
        }
        vst1q_u64(state.as_mut_ptr(), ab);
        vst1q_u64(state.as_mut_ptr().add(2), cd);
        vst1q_u64(state.as_mut_ptr().add(4), ef);
        vst1q_u64(state.as_mut_ptr().add(6), gh);
    }
}
