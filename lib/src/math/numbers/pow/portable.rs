//! The portable kernel of Montgomery's reduction, for Miri and for the targets without assembly
//! kernels: a row of limb products added into the limbs, a product of two limbs taken as a `u128`,
//! which compilers lower to the target's widening multiplication.

/// Adds the `n` limbs at `x` times `y` to the `n` limbs at `r`, and returns the limb that carries
/// out of them.
///
/// # Safety
///
/// `n >= 1`; the `n` limbs at `x` must be valid for reads and those at `r` for reads and writes,
/// and apart from them.
#[inline]
pub(super) unsafe fn addmul_1(r: *mut u64, x: *const u64, n: usize, y: u64) -> u64 {
    let mut carry = 0;
    // SAFETY: as the caller guarantees. `x * y + r + carry` is at most
    // `(2^64 - 1)^2 + 2 (2^64 - 1) = 2^128 - 1`, a `u128`.
    unsafe {
        for i in 0..n {
            let p =
                u128::from(*x.add(i)) * u128::from(y) + u128::from(*r.add(i)) + u128::from(carry);
            *r.add(i) = p as u64;
            carry = (p >> 64) as u64;
        }
    }
    carry
}

/// The row of [`addmul_1`], for the low halves of products.
pub(super) use addmul_1 as row;

/// REDC a limb at a time, after GMP's `mpn_redc_1`: sets the `n` limbs at `r` to `u / B^n mod m`
/// less `m` when it returns `true`, for the `2n` limbs at `u`, below `B^n m`, which it clobbers,
/// with `minv[0] = -1/m mod B`.
///
/// Limb `j`'s quotient `q = u_j (-1/m) mod B` makes `u + q m B^j` zero at limb `j`; the row's carry
/// out, which belongs at limb `j + n`, is kept in limb `j`, now free, and the carries are added to
/// the high half at the end, in one pass, whose carry out it returns.
///
/// # Safety
///
/// `n >= 1`; the limbs must be valid, `r` apart from `u`, `m` apart from both, and `minv` must be
/// `-1/m mod B^2`.
#[inline(always)]
pub(super) unsafe fn redc(
    r: *mut u64,
    u: *mut u64,
    m: *const u64,
    n: usize,
    minv: [u64; 2],
) -> bool {
    // SAFETY: as the caller guarantees: row `j` reads and writes limbs `j..j + n` of `u`.
    unsafe {
        for j in 0..n {
            let q = (*u.add(j)).wrapping_mul(minv[0]);
            *u.add(j) = addmul_1(u.add(j), m, n, q);
        }
        super::super::mul::limbs::add_n(r, u.add(n), u, n)
    }
}
