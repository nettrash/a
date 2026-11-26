//! The portable kernel of the doubling's linear part, for Miri and for the targets without the
//! assembly one: [`super::Chains::limb`] a limb at a time.

use super::Chains;

/// Runs the chains of `state` over the `4 blocks` limbs at the pointers: `(4x - y, 4x - y - (x +
/// y))` at `a` and `b` when `ONE`, and `(4x - y - (x + y), x + y)` otherwise.
///
/// # Safety
///
/// The `4 blocks` limbs at `x` and `y` must be valid for reads, those at `a` and `b` for writes,
/// all apart.
#[inline(always)]
pub(super) unsafe fn blocks<const ONE: bool>(
    a: *mut u64,
    b: *mut u64,
    x: *const u64,
    y: *const u64,
    blocks: usize,
    state: &mut Chains,
) {
    // SAFETY: as the caller guarantees.
    unsafe {
        for i in 0..4 * blocks {
            let (d, f, s) = state.limb(*x.add(i), *y.add(i), 0, 0);
            if ONE {
                *a.add(i) = d;
                *b.add(i) = f;
            } else {
                *a.add(i) = f;
                *b.add(i) = s;
            }
        }
    }
}
