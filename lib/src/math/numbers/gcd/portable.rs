//! The portable kernels of the greatest common divisor, for Miri and for the targets without an
//! assembly kernel: a product of two limbs is taken as a `u128`, which compilers lower to the
//! target's widening multiplication.

use crate::math::numbers::gcd::limbs::{Matrix1, gcd_11};

/// Sets `(a, b)` to `(u11 a - u01 b, u00 b - u10 a)` over the `n` limbs at `a` and `b`, in place:
/// the pair reduced by the inverse of the matrix `m`, whose entries are below `2^63`. Both results
/// must be non-negative and fit in `n` limbs, as they do when `m` is a reduction of the pair.
///
/// Each position takes the two products of each result in an `i128`, whose magnitude stays below
/// `2^127`: a product of a limb and an entry is below `2^127 - 2^64`, and the carry that the
/// position passes on, the high limb of the difference, is within `2^63 + 1` of zero.
///
/// # Safety
///
/// `n >= 1`; the `n` limbs at `a` and at `b` must be valid for reads and writes, and apart.
#[inline]
pub(in crate::math::numbers) unsafe fn reduce_pair(a: *mut u64, b: *mut u64, n: usize, m: Matrix1) {
    let (mut ca, mut cb) = (0i128, 0i128);
    // SAFETY: as the caller guarantees; each position is read before it is written.
    unsafe {
        for i in 0..n {
            let (x, y) = (u128::from(*a.add(i)), u128::from(*b.add(i)));
            let p = (u128::from(m.u11) * x) as i128 - (u128::from(m.u01) * y) as i128 + ca;
            let q = (u128::from(m.u00) * y) as i128 - (u128::from(m.u10) * x) as i128 + cb;
            *a.add(i) = p as u64;
            *b.add(i) = q as u64;
            ca = p >> 64;
            cb = q >> 64;
        }
    }
    debug_assert!(ca == 0 && cb == 0, "a reduced pair is negative or too long");
}

/// Sets `(x, y)` to `(u00 x + u10 y, u01 x + u11 y)` over the `n` limbs at `x` and `y`, in place:
/// the row `(x, y)` times the matrix `m`, whose entries are below `2^63`. Returns the limbs that
/// carry out of the two results, each below `2^64`: two products of a limb and an entry and a carry
/// add up to less than `2^128`.
///
/// # Safety
///
/// `n >= 1`; the `n` limbs at `x` and at `y` must be valid for reads and writes, and apart.
#[inline]
pub(in crate::math::numbers) unsafe fn extend_row(
    x: *mut u64,
    y: *mut u64,
    n: usize,
    m: Matrix1,
) -> (u64, u64) {
    let (mut cx, mut cy) = (0u64, 0u64);
    // SAFETY: as the caller guarantees; each position is read before it is written.
    unsafe {
        for i in 0..n {
            let (u, v) = (u128::from(*x.add(i)), u128::from(*y.add(i)));
            let p = u128::from(m.u00) * u + u128::from(m.u10) * v + u128::from(cx);
            let q = u128::from(m.u01) * u + u128::from(m.u11) * v + u128::from(cy);
            *x.add(i) = p as u64;
            *y.add(i) = q as u64;
            cx = (p >> 64) as u64;
            cy = (q >> 64) as u64;
        }
    }
    (cx, cy)
}

/// Adds the `n` limbs at `x` times `y` to the `n` limbs at `r`, and returns the limb that carries
/// out of them.
///
/// # Safety
///
/// `n >= 1`; the `n` limbs at `x` must be valid for reads and those at `r` for reads and writes,
/// and apart from them.
#[inline]
pub(in crate::math::numbers) unsafe fn addmul_1(
    r: *mut u64,
    x: *const u64,
    n: usize,
    y: u64,
) -> u64 {
    let mut carry = 0;
    // SAFETY: as the caller guarantees. `x * y + r + carry` is at most `(2^64 - 1)^2 + 2 (2^64 -
    // 1) = 2^128 - 1`, a `u128`.
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

/// The quotient and the remainder of `n` by `d`, for `n < 8 d` and `d < 2^61`: three conditional
/// subtractions of `4 d`, `2 d` and `d`, each a select that the compiler keeps without a branch, as
/// whether it is taken is as good as random.
#[inline(always)]
pub(in crate::math::numbers) fn small_q1(n: u64, d: u64) -> (u64, u64) {
    let mut r = n;
    let mut q = 0;
    for k in [2, 1, 0] {
        let (t, borrow) = r.overflowing_sub(d << k);
        r = core::hint::select_unpredictable(borrow, r, t);
        q = (q << 1) | u64::from(!borrow);
    }
    (q, r)
}

/// [`small_q1`] on two limbs, for `n < 8 d` and `d < 2^125`.
#[inline(always)]
pub(in crate::math::numbers) fn small_q2(n: u128, d: u128) -> (u64, u128) {
    let mut r = n;
    let mut q = 0;
    for k in [2, 1, 0] {
        let (t, borrow) = r.overflowing_sub(d << k);
        r = core::hint::select_unpredictable(borrow, r, t);
        q = (q << 1) | u64::from(!borrow);
    }
    (q, r)
}

/// The greatest common divisor of the odd `u` and `v` of up to two limbs: [`gcd_11`]'s steps on
/// two limbs while either number has a second one, then on one.
///
/// [`gcd_11`]: crate::math::numbers::gcd::limbs::gcd_11
#[inline]
pub(in crate::math::numbers) fn gcd_22(mut u: u128, mut v: u128) -> u128 {
    debug_assert!(u & v & 1 == 1);
    while (u | v) >> 64 != 0 {
        let (d, below) = u.overflowing_sub(v);
        if d == 0 {
            return u;
        }
        let zeros = d.trailing_zeros();
        if below {
            v = u;
        }
        u = if below { d.wrapping_neg() } else { d } >> zeros;
    }
    u128::from(gcd_11(u as u64, v as u64))
}
