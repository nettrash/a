//! The aarch64 kernels of the shifts, on NEON: sixteen bytes at a time in byte lanes, through the
//! intrinsics of `core::arch`, for the 64-bit little-endian targets whose operating system allows
//! unaligned accesses (those of the addition's assembly) with NEON enabled. Unlike the carry
//! chains of the addition, nothing here needs the flags, and LLVM's code for the intrinsics is
//! what assembly would be: a block of sixteen bytes is a load, an `ext`, an `ushl`, an `mla` and
//! a store.
//!
//! # Byte lanes
//!
//! A shift by `r` bits, from 1 to 7, makes each byte of the result from two bytes of the input:
//! a left shift `x[i] << r | x[i - 1] >> (8 - r)` from the byte at the same position and the one
//! below it, a right shift `x[i] >> r | x[i + 1] << (8 - r)` from the one above. No carry passes
//! from one byte to the next, so the lanes are independent. Each block of sixteen is loaded once;
//! its neighbours, the same bytes one position off, are made from it and the block next to it,
//! which the loop loads anyway, by an `ext`. The byte shifted right is shifted by `ushl` with a
//! negative count; the one shifted left is multiplied by `2^r` (or `2^(8 - r)`), which in a byte
//! drops the bits shifted out as a shift does, and added to the other by `mla`: they have no bit
//! in common, so their sum is their `or`. That is three vector operations a block, with the
//! count in a register: `ushr` and `sli` with immediates are three too with the `ext`, for seven
//! copies of each kernel, and a second load of the neighbours instead of the `ext` saves an
//! operation but loads each byte twice.
//!
//! The kernels work in the order of memory, so that one kernel serves both byte orders: the byte
//! "below" a position is the one before it in memory little-endian and the one after it
//! big-endian. Big-endian magnitudes need no byte reversal, which limbs of eight bytes would
//! (two `rev`s per limb), and they shift as fast as little-endian ones.
//!
//! Measured on an Apple M4 Max, a left shift by 5 bits in place, over buffers that fit in the
//! level-1 cache, took 0.045 to 0.048 cycles a byte so, against 0.051 to 0.056 with two loads,
//! `ushl` and `orr`, and 0.057 to 0.063 with the `ext` but `ushl` and `orr`; over 1 MiB of
//! buffers, read from the level-2 cache, 0.058 to 0.072 against 0.077 to 0.096 with two loads,
//! whose second, unaligned, load is what costs there. A loop of `lsl`, `lsr` and `orr` on limbs,
//! the shape of GMP 6.3's `mpn_lshift`, took 0.086 to 0.097, one of 64-bit lanes with `ext` 0.063
//! to 0.070, and `extr`, whose shift is an immediate, issues about one a cycle on this core.
//!
//! # In place
//!
//! A kernel may write its result over its input. Each kernel loads every block before it stores
//! a block over any of its bytes: a left shift little-endian moves the bytes `q` positions up,
//! towards higher addresses, and runs from the top down; the other shifts move them down or, a
//! left shift big-endian growing by a byte, one byte towards the end of the vector, and run from
//! low addresses up, loading the block after the next ones before they store them. A block at an
//! end, whose neighbour byte is outside the input, is loaded first and stored last (or first,
//! where its neighbour is below the start); the block at the other end of the regular ones, whose
//! neighbouring block may run past the input, loads its neighbours instead.

use core::arch::aarch64::{
    int8x16_t, uint8x16_t, vbslq_u8, vdupq_n_s8, vdupq_n_u8, vextq_u8, vld1q_u8, vmlaq_u8,
    vorrq_u8, vshlq_u8, vst1q_u8,
};

/// From this many bytes, the kernels store their regular blocks at addresses on 16 bytes: a store
/// across them costs more than a load, and a shift of whole bytes besides the bits would put every
/// store across them, 30% slower for 64 KiB from the level-2 cache. Below, [`up_with_next`] loads
/// the next bytes of each block, two loads a block and a shift and an `or` to combine them, the
/// shortest path from the loads to the stores, for one to three blocks. Measured on an Apple M4
/// Max, aligning from 128 bytes instead saved about a cycle for a big-endian `<<=` of 64 to 96
/// bytes, but the code the compiler made then took a third longer for a right shift of 1 KiB,
/// which has nothing to align.
const ALIGNED: usize = 64;

/// A shift in byte lanes: the factor of the bytes shifted left, `2^r` for a left shift by `r`
/// bits and `2^(8 - r)` for a right shift, the same as a count, `r` or `8 - r`, and the negative
/// count of the bytes shifted right, `r - 8` or `-r`.
#[derive(Clone, Copy)]
struct Lanes {
    times: uint8x16_t,
    left: int8x16_t,
    right: int8x16_t,
}

impl Lanes {
    /// The lanes of a shift whose bytes shifted left go `left` bits up, from 1 to 7.
    #[inline(always)]
    unsafe fn new(left: u32) -> Lanes {
        // SAFETY: NEON is enabled on the targets this module is compiled for.
        unsafe {
            Lanes {
                times: vdupq_n_u8(1 << left),
                left: vdupq_n_s8(left as i8),
                right: vdupq_n_s8(left as i8 - 8),
            }
        }
    }
}

/// Writes positions `q..q + l` of `x << (8q + r)` at the least significant end `dst` in the
/// layout `BE`, for `x` of `l >= 16` significant bytes at the least significant end `src`, and
/// `r` from 1 to 7: the low `l` bytes of `x << r`. The bits shifted out of the top are left to
/// the caller.
///
/// # Safety
///
/// The `l` bytes at `src` must be valid for reads and the bytes at positions `q..q + l` at `dst`
/// for writes. If they overlap, `dst` must be `src` little-endian, and big-endian at most
/// `src + 1`.
#[inline(always)]
pub(super) unsafe fn shl_bits<const BE: bool>(
    dst: *mut u8,
    src: *const u8,
    l: usize,
    q: usize,
    r: u32,
) {
    // SAFETY: as the caller guarantees. Little-endian the result lies `q` bytes above the input
    // and is written from the top down; big-endian the result's bytes from the most significant
    // are at `dst - q - l` and the input's at `src - l`, at most one byte higher, written upwards.
    unsafe {
        let lanes = Lanes::new(r);
        if BE {
            up_with_next::<true>(dst.sub(q + l), src.sub(l), l, l, lanes, 0);
        } else {
            down_with_previous(dst.add(q), src, l, lanes);
        }
    }
}

/// Writes positions `0..n` of `x >> (8q + r)` at the least significant end `dst` in the layout
/// `BE`, for `x` of `l` significant bytes at the least significant end `src`, `r` from 1 to 7 and
/// `n >= 16` the length of the quotient, `l - q` or `l - q - 1`: positions `q..q + n` of `x` and
/// `above`, its byte at position `q + n` (zero when that is above its top).
///
/// The blocks are counted from `l - q`, so that the branches do not depend on which of the two
/// lengths the quotient has, a coin toss for random numbers and some shifts.
///
/// # Safety
///
/// Positions `q..l` at `src` must be valid for reads and positions `0..n` at `dst` for writes. If
/// they overlap, `dst` must be `src` little-endian, and big-endian `src - (l - q - n)`, the
/// significant bytes of `x` at the start of the vector.
#[inline(always)]
pub(super) unsafe fn shr_bits<const BE: bool>(
    dst: *mut u8,
    src: *const u8,
    l: usize,
    q: usize,
    r: u32,
    n: usize,
    above: u8,
) {
    // SAFETY: as the caller guarantees. Little-endian the result lies `q` bytes below the input,
    // its byte above the top `above`, which is the input's byte there when the quotient is one
    // byte shorter; big-endian the quotient's bytes from the most significant are at `dst - n`
    // and the input's at `src - q - n`, at most one byte higher, the byte before them `above`.
    // Both are written upwards, and `l - q` is `n` or `n + 1`.
    unsafe {
        let lanes = Lanes::new(8 - r);
        if BE {
            up_with_previous(dst.sub(n), src.sub(q + n), n, l - q, lanes, above);
        } else {
            up_with_next::<false>(dst, src.add(q), n, l - q, lanes, above);
        }
    }
}

/// The bytes of `block` and of its `neighbour` combined in each lane: `block` shifted left and
/// the neighbour right when `BLOCK_UP` (a left shift), the other way round otherwise (a right
/// shift); the one shifted left multiplied by `lanes.times` and added, two vector operations.
#[inline(always)]
unsafe fn combine<const BLOCK_UP: bool>(
    block: uint8x16_t,
    neighbour: uint8x16_t,
    lanes: Lanes,
) -> uint8x16_t {
    // SAFETY: NEON is enabled on the targets this module is compiled for.
    unsafe {
        if BLOCK_UP {
            vmlaq_u8(vshlq_u8(neighbour, lanes.right), block, lanes.times)
        } else {
            vmlaq_u8(vshlq_u8(block, lanes.right), neighbour, lanes.times)
        }
    }
}

/// [`combine`] for the blocks at the ends, by two shifts and an `or`: one operation more, but a
/// cycle shorter from the loads to the store than the multiplication, which counts when a
/// number has one or two blocks.
#[inline(always)]
unsafe fn combine_end<const BLOCK_UP: bool>(
    block: uint8x16_t,
    neighbour: uint8x16_t,
    lanes: Lanes,
) -> uint8x16_t {
    let (up, down) = if BLOCK_UP {
        (block, neighbour)
    } else {
        (neighbour, block)
    };
    // SAFETY: NEON is enabled on the targets this module is compiled for.
    unsafe { vorrq_u8(vshlq_u8(up, lanes.left), vshlq_u8(down, lanes.right)) }
}

/// Sixteen bytes at `p`, at any alignment.
///
/// # Safety
///
/// The bytes must be valid for reads.
#[inline(always)]
unsafe fn load(p: *const u8) -> uint8x16_t {
    // SAFETY: as the caller guarantees.
    unsafe { vld1q_u8(p) }
}

/// Writes sixteen bytes at `p`, at any alignment.
///
/// # Safety
///
/// The bytes must be valid for writes.
#[inline(always)]
unsafe fn store(p: *mut u8, v: uint8x16_t) {
    // SAFETY: as the caller guarantees.
    unsafe { vst1q_u8(p, v) }
}

/// The sixteen bytes one position after the block `v`, whose next block is `w`.
#[inline(always)]
unsafe fn after(v: uint8x16_t, w: uint8x16_t) -> uint8x16_t {
    // SAFETY: NEON is enabled on the targets this module is compiled for.
    unsafe { vextq_u8::<1>(v, w) }
}

/// The sixteen bytes one position before the block `v`, whose block before is `u`.
#[inline(always)]
unsafe fn before(u: uint8x16_t, v: uint8x16_t) -> uint8x16_t {
    // SAFETY: NEON is enabled on the targets this module is compiled for.
    unsafe { vextq_u8::<15>(u, v) }
}

/// Writes `out[i]`, `x[i]` combined with `x[i + 1]` (see [`combine`]), for `i` in `0..n`, `x[n]`
/// being `edge`, in byte lanes from low addresses up, `n >= 16`, where `m` bytes of `x` may be
/// read: `n`, or `n + 1` with `x[n]` equal to `edge`.
///
/// From [`ALIGNED`] bytes, the blocks of sixteen whose stores are aligned, at `g, g + 16, …` from
/// the first such `g`, and whose next byte lies in the `m` bytes, are regular; below them, the
/// block at 0, when `g` is not 0, loads its next bytes and is stored after the first regular one
/// is loaded. Below, the regular blocks are at `0, 16, …` and load their next bytes; the top block, at `n - 16` (over the last regular one), takes `edge` as its next
/// byte, and is loaded first and stored last. The regular blocks end at or before `m - 1`, which
/// is at most `n`, and the branches depend on `m` and the address of `out` alone. All regular
/// blocks but the last take their next bytes from the block after them, which is regular too; the
/// last one loads them.
///
/// # Safety
///
/// The `m` bytes at `x` must be valid for reads and the `n` at `out` for writes; `out` may lie
/// anywhere below `x`, or up to one byte above.
#[inline(always)]
unsafe fn up_with_next<const BLOCK_UP: bool>(
    out: *mut u8,
    x: *const u8,
    n: usize,
    m: usize,
    lanes: Lanes,
    edge: u8,
) {
    // SAFETY: every load lies in `x[0..m]`: a regular block at `i` is followed by `x[i + 16]`
    // with `i + 17 <= m`, and all but the last are followed by a regular block, at `i + 16`;
    // the last one reads `x[i + 1..i + 17]`, and the block at 0 `x[1..17]` when a regular one
    // follows it. Every store lies in `out[0..n]`: a regular block's ends at
    // `i + 16 <= m - 1 <= n`. Before a block's store, the blocks it reaches into have been
    // loaded: `out` at most one byte above `x` reaches at most the first byte of the next block,
    // which the loop loads before the store; the block at 0 reaches into the first regular one,
    // loaded before it is stored, and the top block is loaded first.
    unsafe {
        let top = load(x.add(n - 16));
        let top_next = after(top, vdupq_n_u8(edge));
        if n < ALIGNED {
            // Up to three regular blocks, whose latency counts more than their operations: each
            // loads its next bytes, before the block below it is stored.
            let regular = (m - 1) / 16;
            if regular > 0 {
                let mut i = 0;
                let (mut v, mut w) = (load(x), load(x.add(1)));
                for _ in 1..regular {
                    let (v1, w1) = (load(x.add(i + 16)), load(x.add(i + 17)));
                    store(out.add(i), combine_end::<BLOCK_UP>(v, w, lanes));
                    (v, w) = (v1, w1);
                    i += 16;
                }
                store(out.add(i), combine_end::<BLOCK_UP>(v, w, lanes));
            }
            store(
                out.add(n - 16),
                combine_end::<BLOCK_UP>(top, top_next, lanes),
            );
            return;
        }
        // The first aligned block; from `ALIGNED` bytes, at least two blocks are regular.
        let g = out.addr().wrapping_neg() % 16;
        // The regular blocks: those at `i = g + 16 k` with `i + 17 <= m`.
        let regular = (m - 1).saturating_sub(g) / 16;
        if regular > 0 {
            let mut i = g;
            let mut v = load(x.add(i));
            if g > 0 {
                store(out, combine_end::<BLOCK_UP>(load(x), load(x.add(1)), lanes));
            }
            let mut left = regular - 1;
            while left >= 2 {
                let (v1, v2) = (load(x.add(i + 16)), load(x.add(i + 32)));
                store(out.add(i), combine::<BLOCK_UP>(v, after(v, v1), lanes));
                store(
                    out.add(i + 16),
                    combine::<BLOCK_UP>(v1, after(v1, v2), lanes),
                );
                v = v2;
                i += 32;
                left -= 2;
            }
            if left == 1 {
                let v1 = load(x.add(i + 16));
                store(out.add(i), combine::<BLOCK_UP>(v, after(v, v1), lanes));
                v = v1;
                i += 16;
            }
            store(
                out.add(i),
                combine_end::<BLOCK_UP>(v, load(x.add(i + 1)), lanes),
            );
        }
        store(
            out.add(n - 16),
            combine_end::<BLOCK_UP>(top, top_next, lanes),
        );
    }
}

/// Writes `out[i] = x[i] >> r | x[i - 1] << (8 - r)` for `i` in `0..n`, `x[-1]` being `edge`, in
/// byte lanes from low addresses up, `n >= 16`, with `m` either `n` or `n + 1`, for the right
/// shift in `lanes`.
///
/// The bottom block, at 0, takes `edge` as its previous byte and is stored first; the blocks at
/// `16, 32, …` below `16 ((m - 1) / 16)` are regular, which ends at or before `n`, and take their
/// previous bytes from the block before them; the top block, at `n - 16`, over the last of them
/// (or over the bottom one, for 16 bytes), is loaded first and stored last. The branches depend on
/// `m` alone, as in [`up_with_next`].
///
/// # Safety
///
/// The `n` bytes at `x` must be valid for reads and the `n` at `out` for writes; `out` may lie
/// anywhere below `x`, or at it.
#[inline(always)]
unsafe fn up_with_previous(out: *mut u8, x: *const u8, n: usize, m: usize, lanes: Lanes, edge: u8) {
    // SAFETY: every load lies in `x[0..n]`: a block at `i >= 16` reads `x[i..i + 16]` with
    // `i + 16 <= 16 ((m - 1) / 16) <= m - 1 <= n`, and the top block's previous bytes are read
    // from `x + max(n, 17) - 17`, with `n >= 16`. Every store lies in `out[0..n]`. A block's
    // store reaches no byte of `x` above its own, and the top block is loaded first.
    unsafe {
        let mut v = load(x);
        let mut w = before(vdupq_n_u8(edge), v);
        // The top block's previous bytes, from one byte before it, or for 16 bytes, where that is
        // before `x`, those of the bottom block: chosen without a branch, as `n` is 16 or 17 for
        // 17 bytes shifted, a coin toss.
        let top = load(x.add(n - 16));
        let loaded = load(x.add(n.max(17) - 17));
        let top_previous = vbslq_u8(vdupq_n_u8(0u8.wrapping_sub(u8::from(n == 16))), w, loaded);
        let mut i = 0;
        // The regular blocks, after the bottom one.
        let mut left = ((m - 1) / 16).saturating_sub(1);
        while left >= 2 {
            let (v1, v2) = (load(x.add(i + 16)), load(x.add(i + 32)));
            store(out.add(i), combine::<false>(v, w, lanes));
            store(out.add(i + 16), combine::<false>(v1, before(v, v1), lanes));
            (v, w) = (v2, before(v1, v2));
            i += 32;
            left -= 2;
        }
        if left == 1 {
            let v1 = load(x.add(i + 16));
            store(out.add(i), combine::<false>(v, w, lanes));
            (v, w) = (v1, before(v, v1));
            i += 16;
        }
        store(out.add(i), combine_end::<false>(v, w, lanes));
        store(
            out.add(n - 16),
            combine_end::<false>(top, top_previous, lanes),
        );
    }
}

/// Writes `out[i] = x[i] << r | x[i - 1] >> (8 - r)` for `i` in `0..n`, `x[-1]` being zero, in
/// byte lanes from high addresses down, `n >= 16`, for the left shift in `lanes`.
///
/// The blocks of sixteen whose stores are aligned, at `t, t - 16, …` from the highest such `t` at
/// or below `n - 16`, that begin after the first byte of `x`, are regular; above them, from
/// [`ALIGNED`] bytes, the block at `n - 16`, when `t` is not `n - 16`, loads its previous bytes
/// and is stored after the first regular one is loaded; the bottom block, at 0 (under the last
/// regular one), takes zero as its previous byte, and is loaded first and stored last. All
/// regular blocks but the lowest take their previous bytes from the block below them, which is
/// regular too; the lowest one loads them.
///
/// # Safety
///
/// The `n` bytes at `x` must be valid for reads and the `n` at `out` for writes; `out` may lie
/// anywhere above `x`, or at it.
#[inline(always)]
unsafe fn down_with_previous(out: *mut u8, x: *const u8, n: usize, lanes: Lanes) {
    // SAFETY: every load lies in `x[0..n]`: a regular block at `i` begins at `i >= 1`, and all
    // but the lowest at `i >= 17`, with a regular block at `i - 16`; the lowest reads
    // `x[i - 1..i + 15]`, and the block at `n - 16`, not regular, `x[n - 17..n - 1]` with
    // `n - 16 > t >= 1`. Every store lies in `out[0..n]`. `out` at or above `x` reaches only
    // bytes of `x` at or above the block's own, which have been loaded: the block at `n - 16`
    // reaches into the first regular block, loaded before it is stored, and the bottom block is
    // loaded first.
    unsafe {
        let bottom = load(x);
        let bottom_previous = before(vdupq_n_u8(0), bottom);
        // How far the block at `n - 16` is above the highest aligned one: none below `ALIGNED`
        // bytes, where the stores are few.
        let skew = if n >= ALIGNED {
            out.add(n - 16).addr() % 16
        } else {
            0
        };
        if skew > 0 {
            let (v, w) = (load(x.add(n - 16)), load(x.add(n - 17)));
            let t = n - 16 - skew;
            // The first regular block, which the top one's store reaches into.
            let first = load(x.add(t));
            store(out.add(n - 16), combine_end::<true>(v, w, lanes));
            regular_down(out, x, t, first, lanes);
        } else {
            regular_down(out, x, n - 16, load(x.add(n - 16)), lanes);
        }
        store(out, combine_end::<true>(bottom, bottom_previous, lanes));
    }
}

/// The regular blocks of [`down_with_previous`], at `t, t - 16, …` down to 1, the one at `t`
/// loaded as `v`: none when `t` is 0.
///
/// # Safety
///
/// As for [`down_with_previous`], with `t + 16 <= n`.
#[inline(always)]
unsafe fn regular_down(out: *mut u8, x: *const u8, t: usize, mut v: uint8x16_t, lanes: Lanes) {
    // SAFETY: as in `down_with_previous`.
    unsafe {
        if t == 0 {
            return;
        }
        let mut i = t;
        let mut left = (t - 1) / 16;
        while left >= 2 {
            let (v1, v2) = (load(x.add(i - 16)), load(x.add(i - 32)));
            store(out.add(i), combine::<true>(v, before(v1, v), lanes));
            store(out.add(i - 16), combine::<true>(v1, before(v2, v1), lanes));
            v = v2;
            i -= 32;
            left -= 2;
        }
        if left == 1 {
            let v1 = load(x.add(i - 16));
            store(out.add(i), combine::<true>(v, before(v1, v), lanes));
            v = v1;
            i -= 16;
        }
        store(
            out.add(i),
            combine_end::<true>(v, load(x.add(i - 1)), lanes),
        );
    }
}
