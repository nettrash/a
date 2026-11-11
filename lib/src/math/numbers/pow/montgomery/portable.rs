//! No kernels of the products in registers, for Miri and the targets without assembly kernels:
//! each function gives `None`, and the products are the Rust rows of [`super::rows`].

/// `a b / R mod m` below `m`: no kernel.
#[inline(always)]
pub(in crate::math::numbers) fn mul<const N: usize>(
    _: &[u64; N],
    _: &[u64; N],
    _: &[u64; N],
    _: u64,
) -> Option<[u64; N]> {
    None
}

/// `a^2 / R mod m` below `m`: no kernel.
#[inline(always)]
pub(in crate::math::numbers) fn sqr<const N: usize>(
    _: &[u64; N],
    _: &[u64; N],
    _: u64,
) -> Option<[u64; N]> {
    None
}

/// `a b / R mod m` below `2m`, lazily: no kernel.
#[inline(always)]
pub(in crate::math::numbers) fn lazy_mul<const N: usize>(
    _: &[u64; N],
    _: &[u64; N],
    _: &[u64; N],
    _: u64,
) -> Option<[u64; N]> {
    None
}

/// `a^2 / R mod m` below `2m`, lazily: no kernel.
#[inline(always)]
pub(in crate::math::numbers) fn lazy_sqr<const N: usize>(
    _: &[u64; N],
    _: &[u64; N],
    _: u64,
) -> Option<[u64; N]> {
    None
}

/// Whether [`long`] takes residues of `n` limbs: never, without the kernels.
#[inline(always)]
pub(in crate::math::numbers) const fn has_long(_: usize) -> bool {
    false
}

/// No kernel for long residues: never called, as [`has_long`] takes no length.
///
/// # Safety
///
/// Never to be called.
#[inline(always)]
pub(in crate::math::numbers) unsafe fn long(
    _: *mut u64,
    _: *const u64,
    _: *const u64,
    _: *const u64,
    _: usize,
) {
    unreachable!()
}

/// Whether [`low`] takes residues of `n` limbs: never, without the kernels.
#[inline(always)]
pub(in crate::math::numbers) const fn has_low(_: usize) -> bool {
    false
}

/// No kernel for the low halves: never called, as [`has_low`] takes no length.
///
/// # Safety
///
/// Never to be called.
#[inline(always)]
pub(in crate::math::numbers) unsafe fn low(_: *mut u64, _: *const u64, _: *const u64, _: usize) {
    unreachable!()
}
