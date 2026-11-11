//! The kernels of the hard leaves in portable Rust.

use super::super::tables::UP_TO;
use super::{BLOCK, WORDS, WORDS_BEFORE};

/// Writes into `before` the sums of the counts of `blocks` before each, `before[i]` that of
/// `blocks[..i]`: four chains of sums, over the four quarters of the blocks, which the processor
/// runs side by side, and the sums of the quarters before each added after. The slices have the same
/// length, a multiple of 4.
#[inline]
pub(super) fn prefix(before: &mut [u32], blocks: &[u16]) {
    debug_assert!(before.len() == blocks.len() && blocks.len().is_multiple_of(4));
    let q = blocks.len() / 4;
    let (b0, rest) = before.split_at_mut(q);
    let (b1, rest) = rest.split_at_mut(q);
    let (b2, b3) = rest.split_at_mut(q);
    let (c0, rest) = blocks.split_at(q);
    let (c1, rest) = rest.split_at(q);
    let (c2, c3) = rest.split_at(q);
    let mut n = [0u32; 4];
    for i in 0..q {
        b0[i] = n[0];
        b1[i] = n[1];
        b2[i] = n[2];
        b3[i] = n[3];
        n[0] += u32::from(c0[i]);
        n[1] += u32::from(c1[i]);
        n[2] += u32::from(c2[i]);
        n[3] += u32::from(c3[i]);
    }
    let mut carry = 0u32;
    for (k, part) in [b1, b2, b3].into_iter().enumerate() {
        carry += n[k];
        part.iter_mut().for_each(|b| *b += carry);
    }
}

/// Writes into `leaves` the indices of the entries of `chunk`, at most 256, above `prime`, in
/// increasing order, and returns their number: with no branch on whether each is one, which would be
/// mispredicted for about one entry in three.
#[inline]
pub(super) fn gather(chunk: &[u16], prime: u16, leaves: &mut [u16; 256]) -> usize {
    debug_assert!(chunk.len() <= 256);
    let mut n = 0;
    for (k, &e) in chunk.iter().enumerate() {
        leaves[n] = k as u16;
        n += usize::from(e > prime);
    }
    n
}

/// The bits of the numbers from the segment's start to `low + stop`: those before the block of
/// `stop`, from `before`, and those of the block's words before its word and of its word up to it
/// ([`in_block`]).
///
/// # Safety
///
/// None here: the accesses are checked. [`aarch64`](super) states what its version needs.
#[inline(always)]
pub(super) unsafe fn rank(words: &[[u8; 8]], before: &[u32], stop: u64) -> u64 {
    let w = (stop / 240) as usize;
    u64::from(before[w >> (BLOCK - 3)]) + in_block(words, w, stop)
}

/// [`rank`] with no counts before the blocks prepared: those of the blocks before the block of
/// `stop` summed from their counters, for a prime with few leaves in the segment.
///
/// # Safety
///
/// None here: the accesses are checked.
#[inline(always)]
pub(super) unsafe fn rank_direct(words: &[[u8; 8]], blocks: &[u16], stop: u64) -> u64 {
    let w = (stop / 240) as usize;
    let before: u64 = blocks[..w >> (BLOCK - 3)]
        .iter()
        .map(|&c| u64::from(c))
        .sum();
    before + in_block(words, w, stop)
}

/// The bits of the words of the block of the word `w` before it, and of `w` up to `stop`.
#[inline(always)]
fn in_block(words: &[[u8; 8]], w: usize, stop: u64) -> u64 {
    // The masks come from a table: a mask formed from the index, a select, is turned back into a
    // loop to the index, with a branch for each word.
    let first = (w >> (BLOCK - 3)) << (BLOCK - 3);
    let ws: &[[u8; 8]; WORDS] = words[first..first + WORDS].try_into().expect("a block");
    let masks = &WORDS_BEFORE[w - first];
    let bits: u32 = ws
        .iter()
        .zip(masks)
        .map(|(word, &m)| (u64::from_le_bytes(*word) & m).count_ones())
        .sum();
    let x = u64::from_le_bytes(words[w]) & UP_TO[(stop % 240) as usize];
    u64::from(bits) + u64::from(x.count_ones())
}
