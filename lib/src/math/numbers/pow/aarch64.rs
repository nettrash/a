//! The aarch64 kernel of Montgomery's reduction: two limbs of the quotient a pass, each pass one
//! call of the multiplication's two-row kernel, `addmul_2`, in inline assembly (four carry chains
//! a step, see the multiplication's kernels).

use super::super::mul::kernels::{addmul_1, addmul_2};

pub(super) use super::super::mul::kernels::addmul_1 as row;

/// REDC two limbs at a time: sets the `n` limbs at `r` to `u / B^n mod m`, below `B^n`, for the
/// `2n` limbs at `u`, below `B^n m`, which it clobbers, with `minv = -1/m mod B^2` (low limb first).
///
/// Each pass takes the quotient `q` of limbs `j` and `j + 1` together, `(u_j + u_{j+1} B) minv mod
/// B^2`, which makes `u + q m B^j` zero at both, and adds `q m` by one pass of `addmul_2` over the
/// `n` limbs from `j`. The kernel writes the pass's limb `n` instead of adding to it, so limb
/// `j + n` is kept aside and put back; the pass's two limbs above `n`, which belong at `j + n` and
/// `j + n + 1`, go into limbs `j` and `j + 1`, now free, and are added to the high half at the
/// end, in one pass, as GMP's `mpn_redc_2` arranges it. An odd last limb takes one row.
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
    let [n0, n1] = minv;
    // SAFETY: as the caller guarantees: pass `j` reads limbs `j..j + n` of `u` and writes limbs
    // `j..j + n + 1`, of which it restores the last.
    unsafe {
        let mut j = 0;
        while j + 2 <= n {
            let (u0, u1) = (*u.add(j), *u.add(j + 1));
            let p = u128::from(u0) * u128::from(n0);
            let q0 = p as u64;
            let q1 = ((p >> 64) as u64)
                .wrapping_add(u0.wrapping_mul(n1))
                .wrapping_add(u1.wrapping_mul(n0));
            let kept = *u.add(j + n);
            let top = addmul_2(u.add(j), m, n, q0, q1, 0);
            *u.add(j) = *u.add(j + n);
            *u.add(j + 1) = top;
            *u.add(j + n) = kept;
            j += 2;
        }
        if j < n {
            let q = (*u.add(j)).wrapping_mul(n0);
            *u.add(j) = addmul_1(u.add(j), m, n, q);
        }
        super::super::mul::limbs::add_n(r, u.add(n), u, n)
    }
}
