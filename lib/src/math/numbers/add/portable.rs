//! The portable kernels, for Miri and for the targets without assembly kernels: all but
//! `x86_64` and the 64-bit little-endian aarch64 targets whose operating system allows
//! unaligned accesses, so also the strict-alignment aarch64 ones (hermit, nuttx, bare metal).
//! The limbs are `u64`s read from and written as `[u8; 8]`, which has alignment 1: the
//! pointers need no alignment, and the compiler makes only the accesses the target allows. The
//! carry between limbs is taken from `overflowing_add` (or `overflowing_sub`) pairs, which
//! compilers lower to add-with-carry where the target has a carry flag.
//!
//! The loop handles four limbs per round, so that four additions chain directly. The test that
//! closes the loop clobbers the carry flag on the targets that have one, and the compiler
//! rebuilds the carry from a register after it, once per round instead of once per limb: on
//! aarch64, where it was measured, about 1.8 cycles per limb against 4 for a loop of one limb.

/// Reads limb `i` of an input: the eight bytes at `p + 8i` as a little-endian `u64`, or,
/// with `BE`, the eight bytes ending at `p - 8i` as a big-endian one.
///
/// # Safety
///
/// The eight bytes of limb `i` must be valid for reads.
#[inline(always)]
unsafe fn load<const BE: bool>(p: *const u8, i: usize) -> u64 {
    // SAFETY: the caller guarantees the eight bytes; `[u8; 8]` has alignment 1, so the read
    // needs none, whatever the alignment of `p`.
    unsafe {
        if BE {
            u64::from_be_bytes(p.sub(8 * i + 8).cast::<[u8; 8]>().read())
        } else {
            u64::from_le_bytes(p.add(8 * i).cast::<[u8; 8]>().read())
        }
    }
}

/// Writes `v` as limb `i` of the output, in the layout of [`load`].
///
/// # Safety
///
/// The eight bytes of limb `i` must be valid for writes.
#[inline(always)]
unsafe fn store<const BE: bool>(p: *mut u8, i: usize, v: u64) {
    // SAFETY: as for `load`.
    unsafe {
        if BE {
            p.sub(8 * i + 8).cast::<[u8; 8]>().write(v.to_be_bytes());
        } else {
            p.add(8 * i).cast::<[u8; 8]>().write(v.to_le_bytes());
        }
    }
}

/// Computes limb `i` of `r = x + y`, or of `r = x - y` with `SUB`, given the carry (or
/// borrow) `c` out of limb `i - 1`, and returns the carry out of limb `i`.
///
/// # Safety
///
/// Limb `i` of `x` and of `y` must be valid for reads, and limb `i` of `r` for writes.
#[inline(always)]
unsafe fn limb<const BE: bool, const SUB: bool>(
    r: *mut u8,
    x: *const u8,
    y: *const u8,
    i: usize,
    c: bool,
) -> bool {
    // SAFETY: as the caller guarantees. Both inputs are read before the output is written,
    // so an output that overlaps them overwrites them only once they have been read.
    unsafe {
        let (a, b) = (load::<BE>(x, i), load::<BE>(y, i));
        // At most one of the two steps carries (or borrows): when `a + b` wraps, its sum is
        // at most `2^64 - 2`, so adding the carry cannot wrap again; likewise for `a - b`.
        let (v, c) = if SUB {
            let (d, b1) = a.overflowing_sub(b);
            let (d, b2) = d.overflowing_sub(u64::from(c));
            (d, b1 | b2)
        } else {
            let (s, c1) = a.overflowing_add(b);
            let (s, c2) = s.overflowing_add(u64::from(c));
            (s, c1 | c2)
        };
        store::<BE>(r, i, v);
        c
    }
}

/// The loop of the four kernels: limbs `0..n` in order, four per round, then the last
/// `n mod 4` one at a time.
///
/// # Safety
///
/// As for the kernel that calls it.
#[inline(always)]
unsafe fn run<const BE: bool, const SUB: bool>(
    r: *mut u8,
    x: *const u8,
    y: *const u8,
    n: usize,
    mut c: bool,
) -> bool {
    let mut i = 0;
    // SAFETY: every limb passed to `limb` is below `n`, so it is valid as the caller
    // guarantees. The limbs go from 0 upwards, so an output equal to an input, or above it
    // in the big-endian layout, only overwrites input bytes already read.
    unsafe {
        while n - i >= 4 {
            c = limb::<BE, SUB>(r, x, y, i, c);
            c = limb::<BE, SUB>(r, x, y, i + 1, c);
            c = limb::<BE, SUB>(r, x, y, i + 2, c);
            c = limb::<BE, SUB>(r, x, y, i + 3, c);
            i += 4;
        }
        while i < n {
            c = limb::<BE, SUB>(r, x, y, i, c);
            i += 1;
        }
    }
    c
}

/// Sets the `n` little-endian limbs at `r` to `x + y + carry` and returns the carry out of
/// the top limb.
///
/// # Safety
///
/// The `8 * n` bytes at `x` and at `y` must be valid for reads, and those at `r` for
/// writes; no alignment is needed. `r` may equal `x`, `y` or both; otherwise the output
/// must not overlap an input.
#[inline]
pub(super) unsafe fn add_n_le(
    r: *mut u8,
    x: *const u8,
    y: *const u8,
    n: usize,
    carry: bool,
) -> bool {
    // SAFETY: as the caller guarantees.
    unsafe { run::<false, false>(r, x, y, n, carry) }
}

/// Sets the `n` little-endian limbs at `r` to `x - y - borrow` and returns the borrow out
/// of the top limb.
///
/// # Safety
///
/// As for [`add_n_le`].
#[inline]
pub(super) unsafe fn sub_n_le(
    r: *mut u8,
    x: *const u8,
    y: *const u8,
    n: usize,
    borrow: bool,
) -> bool {
    // SAFETY: as the caller guarantees.
    unsafe { run::<false, true>(r, x, y, n, borrow) }
}

/// Sets the `n` big-endian limbs below `r_end` to `x + y + carry` and returns the carry
/// out of the top limb.
///
/// # Safety
///
/// The `8 * n` bytes below `x_end` and below `y_end` must be valid for reads, and those
/// below `r_end` for writes; no alignment is needed. `r_end` may equal `x_end`, `y_end` or
/// both, or lie above either (`r_end = x_end + s` with `s >= 1`); otherwise the output must
/// not overlap an input.
#[inline]
pub(super) unsafe fn add_n_be(
    r_end: *mut u8,
    x_end: *const u8,
    y_end: *const u8,
    n: usize,
    carry: bool,
) -> bool {
    // SAFETY: as the caller guarantees.
    unsafe { run::<true, false>(r_end, x_end, y_end, n, carry) }
}

/// Sets the `n` big-endian limbs below `r_end` to `x - y - borrow` and returns the borrow
/// out of the top limb.
///
/// # Safety
///
/// As for [`add_n_be`].
#[inline]
pub(super) unsafe fn sub_n_be(
    r_end: *mut u8,
    x_end: *const u8,
    y_end: *const u8,
    n: usize,
    borrow: bool,
) -> bool {
    // SAFETY: as the caller guarantees.
    unsafe { run::<true, true>(r_end, x_end, y_end, n, borrow) }
}
