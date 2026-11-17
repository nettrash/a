//! No vector kernels (other targets and Miri): every pass runs in portable Rust.

/// No instruction set beyond the portable code.
#[derive(Clone, Copy, Debug)]
pub(in crate::math::fft) struct Isa;

impl Isa {
    pub(in crate::math::fft) fn detect() -> Isa {
        Isa
    }
}

/// Leaves every pair to the portable code.
///
/// # Safety
///
/// None needed; `unsafe` as the vector kernels are.
#[inline(always)]
pub(in crate::math::fft) unsafe fn pairs_f64(
    _: Isa,
    _: *mut f64,
    _: *const f64,
    _: usize,
    _: *const f64,
    _: f64,
) -> usize {
    1
}

/// Leaves every pair to the portable code.
///
/// # Safety
///
/// None needed.
#[inline(always)]
pub(in crate::math::fft) unsafe fn pairs_f32(
    _: Isa,
    _: *mut f32,
    _: *const f32,
    _: usize,
    _: *const f32,
    _: f32,
) -> usize {
    1
}
