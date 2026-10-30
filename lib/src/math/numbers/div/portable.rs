//! The portable kernel of the division, for Miri and for the targets without an assembly kernel.
//! A product of two limbs is taken as a `u128`, which compilers lower to the target's widening
//! multiplication.

/// Subtracts the `n` limbs at `x` times `y` from the `n` limbs at `r`, and returns the limb that
/// borrows out of them: what is still to be subtracted from the limb above.
///
/// # Safety
///
/// `n >= 1`; the `n` limbs at `x` must be valid for reads and those at `r` for reads and writes,
/// and apart from them.
#[inline]
pub(super) unsafe fn submul_1(r: *mut u64, x: *const u64, n: usize, y: u64) -> u64 {
    let mut borrow = 0;
    // SAFETY: as the caller guarantees. `x * y + borrow` is at most `(2^64 - 1)^2 + 2^64 - 1`,
    // a `u128` whose high limb is at most `2^64 - 2` unless its low limb is zero, so adding the
    // borrow of the low limb to the high one does not overflow.
    unsafe {
        for i in 0..n {
            let p = u128::from(*x.add(i)) * u128::from(y) + u128::from(borrow);
            let (v, b) = (*r.add(i)).overflowing_sub(p as u64);
            *r.add(i) = v;
            borrow = (p >> 64) as u64 + u64::from(b);
        }
    }
    borrow
}
