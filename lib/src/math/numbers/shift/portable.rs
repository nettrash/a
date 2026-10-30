//! The portable kernels of the shifts, for Miri and for the targets without NEON kernels: all
//! but the 64-bit little-endian aarch64 targets whose operating system allows unaligned
//! accesses. The bytes go eight at a time as `u64`s, read from and written as `[u8; 8]`, which
//! has alignment 1, in the layout of the magnitude: big-endian ones are put through a byte swap
//! on the way in and out, which the compiler folds into the access where the target has one
//! (`movbe`, a reversing load).
//!
//! Each group of eight bytes of the result takes the bits of its own group of the input and
//! the bits shifted out of the group next to it, which the loop carries in a register, so that
//! every byte is read once.

use super::{load, store};

/// Writes positions `q..q + l` of `x << (8q + r)` at the least significant end `dst` in the
/// layout `BE`, for `x` of `l >= 16` significant bytes at the least significant end `src`, and
/// `r` from 1 to 7: the low `l` bytes of `x << r`. The bits shifted out of the top are left to
/// the caller.
///
/// The groups are written from the most significant down, each read before a write reaches it:
/// in place little-endian the result lies `q` bytes above the input, and big-endian it ends at
/// most one byte further towards the end of the vector than the input does.
///
/// # Safety
///
/// The `l` bytes at `src` must be valid for reads and the bytes at positions `q..q + l` at `dst`
/// for writes. If they overlap, `dst` must be `src` little-endian, and big-endian at most
/// `src + 1`.
#[inline(never)]
pub(super) unsafe fn shl_bits<const BE: bool>(
    dst: *mut u8,
    src: *const u8,
    l: usize,
    q: usize,
    r: u32,
) {
    // SAFETY: as the caller guarantees: every read is of positions below `l`, every write of
    // positions from `q` to `q + l`, and every group is read before a write reaches it.
    unsafe {
        let mut hi = load::<BE>(src, l - 8);
        let mut x = l - 8;
        while x >= 8 {
            let lo = load::<BE>(src, x - 8);
            store::<BE>(dst, q + x, (hi << r) | (lo >> (64 - r)));
            hi = lo;
            x -= 8;
        }
        if x > 0 {
            // The `x` bytes below the last group, as the low end of the eight at position 0.
            let lo = load::<BE>(src, 0);
            let below = lo << (8 * (8 - x));
            store::<BE>(dst, q + x, (hi << r) | (below >> (64 - r)));
            hi = lo;
        }
        store::<BE>(dst, q, hi << r);
    }
}

/// Writes positions `0..n` of `x >> (8q + r)` at the least significant end `dst` in the layout
/// `BE`, for `x` of `l` significant bytes at the least significant end `src`, `r` from 1 to 7 and
/// `n >= 16` the length of the quotient, `l - q` or `l - q - 1`: positions `q..q + n` of `x` and
/// `above`, its byte at position `q + n` (zero when that is above its top).
///
/// The groups of eight are taken from the top of the quotient down, positions `n - 8`,
/// `n - 16`, and so on, and written from the least significant up: first the `n % 8` bytes
/// below them, if any, as the eight at position 0, over which the group above is written. Each
/// group is read before a write reaches it: in place little-endian the result lies `q` bytes
/// below the input; big-endian the input ends at the end of the vector, and the result `q` or
/// `q + 1` bytes before.
///
/// # Safety
///
/// Positions `q..q + n` at `src` must be valid for reads and positions `0..n` at `dst` for
/// writes. If they overlap, `dst` must be `src` little-endian, and big-endian
/// `src - (l - q - n)` with `l` the significant length of `x`, its significant bytes at the start
/// of the vector.
#[inline(never)]
pub(super) unsafe fn shr_bits<const BE: bool>(
    dst: *mut u8,
    src: *const u8,
    _l: usize,
    q: usize,
    r: u32,
    n: usize,
    above: u8,
) {
    // SAFETY: as the caller guarantees: every read is of positions `q..q + n` and every write
    // of positions below `n`, each group read before a write reaches it.
    unsafe {
        let above = u64::from(above);
        let m = n % 8;
        let mut lo = load::<BE>(src, q + m);
        if m > 0 {
            // Its upper neighbour, position `q + 8`, is byte `8 - m` of `lo`.
            let bottom = load::<BE>(src, q);
            store::<BE>(dst, 0, (bottom >> r) | ((lo >> (8 * (8 - m))) << (64 - r)));
        }
        let mut p = m;
        while p + 8 < n {
            let hi = load::<BE>(src, q + p + 8);
            store::<BE>(dst, p, (lo >> r) | (hi << (64 - r)));
            lo = hi;
            p += 8;
        }
        store::<BE>(dst, p, (lo >> r) | (above << (64 - r)));
    }
}
