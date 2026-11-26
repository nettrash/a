//! The kernels of the hard leaves in NEON.

use super::super::tables::UP_TO;
use super::{WORDS, WORDS_BEFORE};
use core::arch::aarch64::{
    uint16x8_t, vaddlvq_u8, vaddq_u8, vaddq_u16, vaddq_u32, vaddvq_u16, vaddvq_u32, vand_u8,
    vandq_u8, vandq_u16, vcgtq_u16, vcnt_u8, vcntq_u8, vcombine_u8, vcreate_u8, vdup_n_u8,
    vdupq_laneq_u32, vdupq_n_u16, vdupq_n_u32, vextq_u16, vget_low_u16, vgetq_lane_u32, vld1_u8,
    vld1q_u8, vld1q_u16, vmovl_high_u16, vmovl_u8, vmovl_u16, vpadalq_u16, vst1q_u16, vst1q_u32,
    vsubq_u16,
};

/// The sums of the eight lanes of `v` up to each lane: three shifted additions.
#[inline(always)]
fn scan8(v: uint16x8_t) -> uint16x8_t {
    // SAFETY: NEON is enabled on the targets this module is compiled for.
    unsafe {
        let z = vdupq_n_u16(0);
        let v = vaddq_u16(v, vextq_u16::<7>(z, v));
        let v = vaddq_u16(v, vextq_u16::<6>(z, v));
        vaddq_u16(v, vextq_u16::<4>(z, v))
    }
}

/// Writes into `before` the sums of the counts of `blocks` before each, `before[i]` that of
/// `blocks[..i]`: the quarters' sums first, then four chains over the quarters, eight blocks a
/// step, the sums within the eight by three shifted additions and widened to 32 bits, which the
/// processor runs side by side; the rest of each quarter one block at a time. The slices have the
/// same length, a multiple of 4; a block counts at most 512 bits, so that eight sum below `2^16`.
#[inline]
pub(super) fn prefix(before: &mut [u32], blocks: &[u16]) {
    let n = blocks.len();
    assert!(before.len() == n && n.is_multiple_of(4));
    let q = n / 4;
    let steps = q / 8;
    // The sums of the quarters before each.
    let mut start = [0u32; 4];
    for k in 1..4 {
        let quarter = &blocks[(k - 1) * q..k * q];
        let (eights, rest) = quarter.as_chunks::<8>();
        let mut t = rest.iter().map(|&c| u32::from(c)).sum::<u32>();
        for e in eights {
            // SAFETY: the eight counts are read from the chunk; NEON is enabled.
            t += u32::from(unsafe { vaddvq_u16(vld1q_u16(e.as_ptr())) });
        }
        start[k] = start[k - 1] + t;
    }
    let mut ends = start;
    // SAFETY: each step reads the eight counts at `k q + 8 i` and writes the eight sums there, with
    // `8 i + 8 <= 8 steps <= q`, so within the slices of `n = 4 q` elements; NEON is enabled.
    unsafe {
        let (src, dst) = (blocks.as_ptr(), before.as_mut_ptr());
        let mut carry = start.map(|c| vdupq_n_u32(c));
        for i in 0..steps {
            for (k, carry) in carry.iter_mut().enumerate() {
                let at = k * q + 8 * i;
                let v = vld1q_u16(src.add(at));
                let inc = scan8(v);
                let exc = vsubq_u16(inc, v);
                vst1q_u32(dst.add(at), vaddq_u32(vmovl_u16(vget_low_u16(exc)), *carry));
                vst1q_u32(dst.add(at + 4), vaddq_u32(vmovl_high_u16(exc), *carry));
                *carry = vaddq_u32(*carry, vdupq_laneq_u32::<3>(vmovl_high_u16(inc)));
            }
        }
        for (end, carry) in ends.iter_mut().zip(carry) {
            *end = vgetq_lane_u32::<0>(carry);
        }
    }
    for (k, &end) in ends.iter().enumerate() {
        let mut c = end;
        for at in k * q + 8 * steps..(k + 1) * q {
            before[at] = c;
            c += u32::from(blocks[at]);
        }
    }
}

/// For each mask of eight bits, the positions of its set bits, in increasing order, then zeros.
static COMPACT: [[u8; 8]; 256] = {
    let mut t = [[0u8; 8]; 256];
    let mut m = 0;
    while m < 256 {
        let (mut b, mut n) = (0, 0);
        while b < 8 {
            if m >> b & 1 == 1 {
                t[m][n] = b as u8;
                n += 1;
            }
            b += 1;
        }
        m += 1;
    }
    t
};

/// The bit of each lane in a mask of eight lanes.
static LANE_BITS: [u16; 8] = [1, 2, 4, 8, 16, 32, 64, 128];

/// Writes into `leaves` the indices of the entries of `chunk`, at most 256, above `prime`, in
/// increasing order, and returns their number: eight entries a step, compared at once, the mask of
/// those above `prime` turned into their positions by [`COMPACT`], and the eight indices stored
/// whole at the next free place, the count moved on by the mask's bits. The rest one at a time.
#[inline]
pub(super) fn gather(chunk: &[u16], prime: u16, leaves: &mut [u16; 256]) -> usize {
    assert!(chunk.len() <= 256);
    let (eights, rest) = chunk.as_chunks::<8>();
    let mut n = 0;
    // SAFETY: each step stores eight indices from `n`, at most the eight entries before it (the
    // indices stored so far, at most 8 a step), so at most `256 - 8`; the reads are of the step's
    // eight entries and of the tables; NEON is enabled.
    unsafe {
        let threshold = vdupq_n_u16(prime);
        let weights = vld1q_u16(LANE_BITS.as_ptr());
        let mut base = vdupq_n_u16(0);
        let eight = vdupq_n_u16(8);
        for e in eights {
            let above = vcgtq_u16(vld1q_u16(e.as_ptr()), threshold);
            let mask = vaddvq_u16(vandq_u16(above, weights)) as usize;
            let positions = vmovl_u8(vld1_u8(COMPACT[mask].as_ptr()));
            vst1q_u16(leaves.as_mut_ptr().add(n), vaddq_u16(base, positions));
            n += mask.count_ones() as usize;
            base = vaddq_u16(base, eight);
        }
    }
    for (k, &e) in (8 * eights.len()..).zip(rest) {
        leaves[n] = k as u16;
        n += usize::from(e > prime);
    }
    n
}

/// The bits of the numbers from the segment's start to `low + stop`: those before the block of
/// `stop`, from `before`, plus those of the block's eight words before its word and of its word up
/// to it ([`in_block`]). The word's index is kept to the last word, so that every access is within
/// the slices whatever `stop` is.
///
/// # Safety
///
/// `words` must be a non-empty whole number of blocks of [`WORDS`] words, and `before` must hold
/// one count for each block.
#[inline(always)]
pub(super) unsafe fn rank(words: &[[u8; 8]], before: &[u32], stop: u64) -> u64 {
    debug_assert!(!words.is_empty() && words.len().is_multiple_of(WORDS));
    debug_assert_eq!(before.len(), words.len() / WORDS);
    let w = ((stop / 240) as usize).min(words.len() - 1);
    // SAFETY: `w` is below the words' length, a multiple of `WORDS`, so that its block is below
    // `before`'s length, as the caller guarantees.
    unsafe { u64::from(*before.get_unchecked(w / WORDS)) + in_block(words, w, stop) }
}

/// [`rank`] with no counts before the blocks prepared: those of the blocks before the block of
/// `stop` summed from their counters, eight a step, for a prime with few leaves in the segment.
///
/// # Safety
///
/// As for [`rank`], with `blocks` for `before`.
#[inline(always)]
pub(super) unsafe fn rank_direct(words: &[[u8; 8]], blocks: &[u16], stop: u64) -> u64 {
    debug_assert!(!words.is_empty() && words.len().is_multiple_of(WORDS));
    debug_assert_eq!(blocks.len(), words.len() / WORDS);
    let w = ((stop / 240) as usize).min(words.len() - 1);
    let block = w / WORDS;
    let eights = block / 8;
    // SAFETY: the counters read are those before `block`, below `blocks`' length, as the caller
    // guarantees; NEON is enabled.
    unsafe {
        let mut acc = vdupq_n_u32(0);
        for i in 0..eights {
            acc = vpadalq_u16(acc, vld1q_u16(blocks.as_ptr().add(8 * i)));
        }
        let mut sum = u64::from(vaddvq_u32(acc));
        for j in 8 * eights..block {
            sum += u64::from(*blocks.get_unchecked(j));
        }
        sum + in_block(words, w, stop)
    }
}

/// The bits of the words of the block of the word `w` before it, and of `w` up to `stop`: the
/// eight words masked by [`WORDS_BEFORE`] and the word by [`UP_TO`], counted by bytes and summed
/// once.
///
/// # Safety
///
/// `w` must be below the length of `words`, a multiple of [`WORDS`].
#[inline(always)]
unsafe fn in_block(words: &[[u8; 8]], w: usize, stop: u64) -> u64 {
    let first = w / WORDS * WORDS;
    // SAFETY: the block's `WORDS` words from `first` are within the words, as the caller
    // guarantees; the mask rows are of `WORDS` words; NEON is enabled.
    unsafe {
        let p = words.as_ptr().add(first).cast::<u8>();
        let m = WORDS_BEFORE.get_unchecked(w - first).as_ptr().cast::<u8>();
        let a0 = vcntq_u8(vandq_u8(vld1q_u8(p), vld1q_u8(m)));
        let a1 = vcntq_u8(vandq_u8(vld1q_u8(p.add(16)), vld1q_u8(m.add(16))));
        let a2 = vcntq_u8(vandq_u8(vld1q_u8(p.add(32)), vld1q_u8(m.add(32))));
        let a3 = vcntq_u8(vandq_u8(vld1q_u8(p.add(48)), vld1q_u8(m.add(48))));
        let up_to = vcreate_u8(*UP_TO.get_unchecked((stop % 240) as usize));
        let x = vcnt_u8(vand_u8(vld1_u8(words.as_ptr().add(w).cast::<u8>()), up_to));
        // At most 4 * 8 + 8 bits a byte.
        let c = vaddq_u8(vaddq_u8(a0, a1), vaddq_u8(a2, a3));
        let c = vaddq_u8(c, vcombine_u8(x, vdup_n_u8(0)));
        u64::from(vaddlvq_u8(c))
    }
}
