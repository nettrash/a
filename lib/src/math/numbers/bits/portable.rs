//! The portable kernels of the bitwise operations, for Miri and for the targets without NEON
//! kernels: all but the 64-bit little-endian aarch64 targets whose operating system allows
//! unaligned accesses. The bytes go eight at a time as `u64`s, read from and written as
//! `[u8; 8]`, which has alignment 1. A bitwise operation needs no byte order: each byte of the
//! result comes from the bytes at its position only, so the words are taken in the order of
//! memory, and a word of an operand in the other byte order is the eight bytes at the mirrored
//! place, byte-swapped.

use super::super::add::{read, write};
use super::{AND, AND_NOT, NOT_AND, OR, XOR};

/// The operation `OP` on the words `x` and `y`.
#[inline(always)]
fn op<const OP: u8>(x: u64, y: u64) -> u64 {
    match OP {
        AND => x & y,
        OR => x | y,
        XOR => x ^ y,
        AND_NOT => x & !y,
        _ => {
            debug_assert!(OP == NOT_AND);
            !x & y
        }
    }
}

/// The eight bytes of the result at `k`: `OP` of those of `x` at `k` and those of `y` at `k`, or,
/// when `MIXED`, at the mirrored place `n - 8 - k`, reversed.
///
/// # Safety
///
/// `k + 8 <= n`, and the bytes must be valid for reads.
#[inline(always)]
unsafe fn word<const OP: u8, const MIXED: bool>(
    x: *const u8,
    y: *const u8,
    n: usize,
    k: usize,
) -> u64 {
    // SAFETY: as the caller guarantees.
    unsafe {
        let a = u64::from_ne_bytes(read(x.add(k)));
        let b = if MIXED {
            u64::from_ne_bytes(read(y.add(n - 8 - k))).swap_bytes()
        } else {
            u64::from_ne_bytes(read(y.add(k)))
        };
        op::<OP>(a, b)
    }
}

/// Writes `dst[k] = OP(x[k], y[k])` for `k` in `0..n`, or with `y[n - 1 - k]` when `MIXED`, eight
/// bytes at a time, in increasing order of the addresses, or decreasing when `DOWN`.
///
/// `dst` may lie over `x`: at the same address, below it when the order is increasing, or above
/// it when it is decreasing. Each group of eight is read before a write reaches it: the group at
/// the far end of the run, which overlaps the one before it when `n` is not a multiple of eight,
/// is read first and written last.
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
    // SAFETY: as the caller guarantees: every read and write is of the `n` bytes, and every group
    // is read before a write reaches it.
    unsafe {
        if n >= 8 {
            let last = if DOWN { 0 } else { n - 8 };
            let w = word::<OP, MIXED>(x, y, n, last);
            if DOWN {
                let mut k = n;
                while k > 8 {
                    k -= 8;
                    write(dst.add(k), word::<OP, MIXED>(x, y, n, k).to_ne_bytes());
                }
            } else {
                let mut k = 0;
                while k + 8 < n {
                    write(dst.add(k), word::<OP, MIXED>(x, y, n, k).to_ne_bytes());
                    k += 8;
                }
            }
            write(dst.add(last), w.to_ne_bytes());
        } else {
            for i in 0..n {
                let k = if DOWN { n - 1 - i } else { i };
                let b = if MIXED { *y.add(n - 1 - k) } else { *y.add(k) };
                *dst.add(k) = op::<OP>(u64::from(*x.add(k)), u64::from(b)) as u8;
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
    let mut k = 0;
    // SAFETY: every byte read is below `src + n` and every byte written below `dst + n`.
    unsafe {
        while k + 8 <= n {
            let w = u64::from_ne_bytes(read(src.add(n - 8 - k))).swap_bytes();
            write(dst.add(k), w.to_ne_bytes());
            k += 8;
        }
        while k < n {
            *dst.add(k) = *src.add(n - 1 - k);
            k += 1;
        }
    }
}

/// The number of ones in the `n` bytes at `p`.
///
/// # Safety
///
/// The `n` bytes must be valid for reads.
#[inline(never)]
pub(super) unsafe fn count_ones(p: *const u8, n: usize) -> u64 {
    let (mut k, mut ones) = (0, 0u64);
    // SAFETY: every read is of the `n` bytes.
    unsafe {
        while k + 8 <= n {
            ones += u64::from(u64::from_ne_bytes(read(p.add(k))).count_ones());
            k += 8;
        }
        while k < n {
            ones += u64::from((*p.add(k)).count_ones());
            k += 1;
        }
    }
    ones
}
