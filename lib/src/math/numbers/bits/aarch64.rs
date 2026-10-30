//! The aarch64 kernels of the bitwise operations, on NEON: sixteen bytes at a time in byte lanes,
//! through the intrinsics of `core::arch`, for the 64-bit little-endian targets whose operating
//! system allows unaligned accesses (those of the addition's assembly) with NEON enabled.
//!
//! A bitwise operation needs no byte order: each byte of the result comes from the bytes at its
//! position only. So the kernels run in the order of memory, the same for both byte orders, and a
//! block of an operand in the other byte order is the sixteen bytes at the mirrored place,
//! reversed by a `rev64` and an `ext`. A block is two loads, the operation (`and`, `orr`, `eor`
//! or `bic`) and a store.
//!
//! Ones are counted by `cnt`, which gives the ones of each byte; eight blocks are summed by `add`s
//! in a tree and added in pairs into sixteen-bit lanes by `uadalp`, into two accumulators, so that
//! no step waits for the one before.

use super::super::ByteOrder;
use super::super::add::read_small;
use super::{AND, AND_NOT, NOT_AND, OR, XOR};
use core::arch::aarch64::{
    uint8x16_t, uint16x8_t, vaddlvq_u8, vaddq_u8, vaddq_u32, vaddvq_u32, vandq_u8, vbicq_u8,
    vcgeq_u8, vcntq_u8, vdupq_n_u8, vdupq_n_u16, veorq_u8, vextq_u8, vld1q_u8, vorrq_u8,
    vpadalq_u8, vpaddlq_u16, vrev64q_u8, vst1q_u8,
};

/// The operation `OP` on the blocks `x` and `y`.
#[inline(always)]
unsafe fn op<const OP: u8>(x: uint8x16_t, y: uint8x16_t) -> uint8x16_t {
    // SAFETY: NEON is enabled on the targets this module is compiled for.
    unsafe {
        match OP {
            AND => vandq_u8(x, y),
            OR => vorrq_u8(x, y),
            XOR => veorq_u8(x, y),
            AND_NOT => vbicq_u8(x, y),
            _ => {
                debug_assert!(OP == NOT_AND);
                vbicq_u8(y, x)
            }
        }
    }
}

/// The operation `OP` on the words `x` and `y`.
#[inline(always)]
fn op64<const OP: u8>(x: u64, y: u64) -> u64 {
    match OP {
        AND => x & y,
        OR => x | y,
        XOR => x ^ y,
        AND_NOT => x & !y,
        _ => !x & y,
    }
}

/// The sixteen bytes of `v` in the reverse order.
#[inline(always)]
unsafe fn reversed(v: uint8x16_t) -> uint8x16_t {
    // SAFETY: NEON is enabled on the targets this module is compiled for.
    unsafe {
        let r = vrev64q_u8(v);
        vextq_u8::<8>(r, r)
    }
}

/// The sixteen bytes of the result at `k`: `OP` of those of `x` at `k` and those of `y` at `k`,
/// or, when `MIXED`, at the mirrored place `n - 16 - k`, reversed.
///
/// # Safety
///
/// `k + 16 <= n`, and the bytes must be valid for reads.
#[inline(always)]
unsafe fn block<const OP: u8, const MIXED: bool>(
    x: *const u8,
    y: *const u8,
    n: usize,
    k: usize,
) -> uint8x16_t {
    // SAFETY: as the caller guarantees.
    unsafe {
        let a = vld1q_u8(x.add(k));
        let b = if MIXED {
            reversed(vld1q_u8(y.add(n - 16 - k)))
        } else {
            vld1q_u8(y.add(k))
        };
        op::<OP>(a, b)
    }
}

/// Reads the eight bytes at `p` as a `u64` in the order of memory.
///
/// # Safety
///
/// The bytes must be valid for reading; they need no alignment.
#[inline(always)]
unsafe fn read8(p: *const u8) -> u64 {
    // SAFETY: as the caller guarantees.
    unsafe { u64::from_ne_bytes(p.cast::<[u8; 8]>().read()) }
}

/// Writes the `u64` `v` as the eight bytes at `p`, in the order of memory.
///
/// # Safety
///
/// The bytes must be valid for writing; they need no alignment.
#[inline(always)]
unsafe fn write8(p: *mut u8, v: u64) {
    // SAFETY: as the caller guarantees.
    unsafe { p.cast::<[u8; 8]>().write(v.to_ne_bytes()) }
}

/// Writes `dst[k] = OP(x[k], y[k])` for `k` in `0..n`, or with `y[n - 1 - k]` when `MIXED`,
/// sixteen bytes at a time, four blocks a step, in increasing order of the addresses, or
/// decreasing when `DOWN`.
///
/// `dst` may lie over `x`: at the same address, below it when the order is increasing, or above
/// it when it is decreasing. Each block is read before a write reaches it: the block at the far
/// end of the run, which overlaps the one before it when `n` is not a multiple of sixteen, is
/// read first and written last. Below sixteen bytes, two words of eight that overlap, both read
/// before either is written, or single bytes below eight.
///
/// # Safety
///
/// The `n` bytes at `x` and `y` must be valid for reads and those at `dst` for writes; `dst` may
/// overlap `x` only as above, and not `y`.
#[inline]
pub(super) unsafe fn logic<const OP: u8, const MIXED: bool, const DOWN: bool>(
    dst: *mut u8,
    x: *const u8,
    y: *const u8,
    n: usize,
) {
    // SAFETY: as the caller guarantees: every read and write is of the `n` bytes, and every block
    // is read before a write reaches it.
    unsafe {
        if n >= 16 {
            let last = if DOWN { 0 } else { n - 16 };
            let w = block::<OP, MIXED>(x, y, n, last);
            if DOWN {
                let mut e = n;
                while e > 64 {
                    for j in 1..=4 {
                        let k = e - 16 * j;
                        vst1q_u8(dst.add(k), block::<OP, MIXED>(x, y, n, k));
                    }
                    e -= 64;
                }
                while e > 16 {
                    e -= 16;
                    vst1q_u8(dst.add(e), block::<OP, MIXED>(x, y, n, e));
                }
            } else {
                let mut k = 0;
                while k + 64 < n {
                    for j in 0..4 {
                        let b = k + 16 * j;
                        vst1q_u8(dst.add(b), block::<OP, MIXED>(x, y, n, b));
                    }
                    k += 64;
                }
                while k + 16 < n {
                    vst1q_u8(dst.add(k), block::<OP, MIXED>(x, y, n, k));
                    k += 16;
                }
            }
            vst1q_u8(dst.add(last), w);
        } else if n >= 8 {
            let (lo, hi) = if MIXED {
                (
                    op64::<OP>(read8(x), read8(y.add(n - 8)).swap_bytes()),
                    op64::<OP>(read8(x.add(n - 8)), read8(y).swap_bytes()),
                )
            } else {
                (
                    op64::<OP>(read8(x), read8(y)),
                    op64::<OP>(read8(x.add(n - 8)), read8(y.add(n - 8))),
                )
            };
            write8(dst, lo);
            write8(dst.add(n - 8), hi);
        } else {
            for i in 0..n {
                let k = if DOWN { n - 1 - i } else { i };
                let b = if MIXED { *y.add(n - 1 - k) } else { *y.add(k) };
                *dst.add(k) = op64::<OP>(u64::from(*x.add(k)), u64::from(b)) as u8;
            }
        }
    }
}

/// Writes `dst[k] = src[n - 1 - k]` for `k` in `0..n`.
///
/// # Safety
///
/// The `n` bytes at `src` must be valid for reads and those at `dst` for writes, apart from them.
#[inline(never)]
pub(super) unsafe fn copy_reversed(dst: *mut u8, src: *const u8, n: usize) {
    // SAFETY: every read is of the `n` bytes at `src` and every write of the `n` at `dst`; the
    // last block or word overlaps the one before it, with the same bytes.
    unsafe {
        if n >= 16 {
            let mut k = 0;
            while k + 16 < n {
                vst1q_u8(dst.add(k), reversed(vld1q_u8(src.add(n - 16 - k))));
                k += 16;
            }
            vst1q_u8(dst.add(n - 16), reversed(vld1q_u8(src)));
        } else if n >= 8 {
            write8(dst, read8(src.add(n - 8)).swap_bytes());
            write8(dst.add(n - 8), read8(src).swap_bytes());
        } else {
            for k in 0..n {
                *dst.add(k) = *src.add(n - 1 - k);
            }
        }
    }
}

/// The steps of 128 bytes whose counts two sixteen-bit accumulators take before they are added
/// up: each step adds at most 64 to a lane (eight bytes of at most eight ones, summed in pairs).
const STEPS: usize = 1023;

/// The number of ones in the `n` bytes at `p`.
///
/// Eight blocks of sixteen bytes a step: their `cnt`s summed by `add`s in a tree, at most 32 a
/// byte, and then into two accumulators of sixteen-bit lanes by `uadalp`, one each a step, so
/// that the chain of an accumulator, about three cycles a `uadalp`, waits for no step. One
/// `uadalp` a block into four accumulators, as GMP 6.3's `mpn_popcount` for arm64 adds them up,
/// was held by those chains to about three cycles every 64 bytes; measured on an Apple M4 Max over
/// 1 KiB in the level-1 cache, this loop takes 34 cycles, 30 bytes a cycle, and GMP's 44. The
/// blocks left, up to seven, add their `cnt`s in bytes, in two accumulators, and the bytes past the
/// last whole block are counted from the sixteen that end the run, those counted already masked
/// off.
///
/// # Safety
///
/// The `n` bytes must be valid for reads.
#[inline]
pub(super) unsafe fn count_ones(p: *const u8, n: usize) -> u64 {
    // SAFETY: every read is of the `n` bytes; NEON is enabled on the targets this module is
    // compiled for.
    unsafe {
        if n <= 16 {
            // As the addition reads a short magnitude; the order of the bytes does not matter.
            let v = read_small(
                &*core::ptr::slice_from_raw_parts(p, n),
                ByteOrder::LittleEndian,
            );
            return u64::from(v.count_ones());
        }
        let count = |k: usize| vcntq_u8(vld1q_u8(p.add(k)));
        let mut ones = 0u64;
        let mut k = 0;
        while n - k >= 128 {
            let steps = ((n - k) / 128).min(STEPS);
            let (mut a0, mut a1): (uint16x8_t, uint16x8_t) = (vdupq_n_u16(0), vdupq_n_u16(0));
            for _ in 0..steps {
                let t0 = vaddq_u8(
                    vaddq_u8(count(k), count(k + 16)),
                    vaddq_u8(count(k + 32), count(k + 48)),
                );
                let t1 = vaddq_u8(
                    vaddq_u8(count(k + 64), count(k + 80)),
                    vaddq_u8(count(k + 96), count(k + 112)),
                );
                a0 = vpadalq_u8(a0, t0);
                a1 = vpadalq_u8(a1, t1);
                k += 128;
            }
            // Up to 65,472 a lane, widened before they are added up.
            ones += u64::from(vaddvq_u32(vaddq_u32(vpaddlq_u16(a0), vpaddlq_u16(a1))));
        }
        // Up to seven blocks, at most 56 a byte with the masked one, in two accumulators so that
        // their chains of `add`s are half as long.
        let (mut b, mut c) = (vdupq_n_u8(0), vdupq_n_u8(0));
        while n - k >= 32 {
            b = vaddq_u8(b, count(k));
            c = vaddq_u8(c, count(k + 16));
            k += 32;
        }
        if n - k >= 16 {
            b = vaddq_u8(b, count(k));
            k += 16;
        }
        if k < n {
            // The sixteen bytes that end at `n`, those already counted masked off: lanes from
            // `16 - (n - k)` up are new.
            const LANES: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
            let lanes = vld1q_u8(LANES.as_ptr());
            let new = vcgeq_u8(lanes, vdupq_n_u8((16 - (n - k)) as u8));
            b = vaddq_u8(b, vandq_u8(count(n - 16), new));
        }
        ones + u64::from(vaddlvq_u8(vaddq_u8(b, c)))
    }
}
