//! The portable kernels of the multiplication, for Miri and for the targets without assembly
//! kernels. The limbs are native `u64`s; a product of two limbs is taken as a `u128`, which
//! compilers lower to the target's widening multiplication (`mul` and `umulh` on aarch64, `mul`
//! on x86_64).

/// Sets the `n` limbs at `r` to the `n` limbs at `x` times `y`, and returns the limb above them.
///
/// # Safety
///
/// `n >= 1`; the `n` limbs at `x` must be valid for reads and those at `r` for writes, and `r`
/// must be `x` or apart from it.
#[inline]
pub(super) unsafe fn mul_1(r: *mut u64, x: *const u64, n: usize, y: u64) -> u64 {
    let mut carry = 0;
    // SAFETY: as the caller guarantees; limb `i` of `x` is read before limb `i` of `r` is
    // written.
    unsafe {
        for i in 0..n {
            let p = u128::from(*x.add(i)) * u128::from(y) + u128::from(carry);
            *r.add(i) = p as u64;
            carry = (p >> 64) as u64;
        }
    }
    carry
}

/// Adds the `n` limbs at `x` times `y` to the `n` limbs at `r`, and returns the limb that
/// carries out of them.
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

/// Sets the `nx + ny` limbs at `r` to the product of the `nx` limbs at `x` and the `ny` at `y`,
/// one row of `x` per limb of `y`.
///
/// # Safety
///
/// `nx >= ny >= 1`; the limbs of `x` and `y` must be valid for reads and those of `r` for
/// writes, and `r` apart from both.
#[inline]
pub(super) unsafe fn mul_basecase(r: *mut u64, x: *const u64, nx: usize, y: *const u64, ny: usize) {
    // SAFETY: as the caller guarantees; row `j` adds into limbs `j..j + nx` of `r`, all written
    // by the rows before it but the top one, which it writes.
    unsafe {
        *r.add(nx) = mul_1(r, x, nx, *y);
        for j in 1..ny {
            *r.add(nx + j) = addmul_1(r.add(j), x, nx, *y.add(j));
        }
    }
}

/// Sets the `2n` limbs at `r` to the square of the `n` limbs at `x`: the products of two
/// different limbs once, then doubled, then the squares of the limbs added.
///
/// # Safety
///
/// `n >= 1`; the limbs of `x` must be valid for reads and those of `r` for writes, and apart from
/// them.
#[inline]
pub(super) unsafe fn sqr_basecase(r: *mut u64, x: *const u64, n: usize) {
    // SAFETY: as the caller guarantees; every position written is below `2n`.
    unsafe {
        if n == 1 {
            let p = u128::from(*x) * u128::from(*x);
            *r = p as u64;
            *r.add(1) = (p >> 64) as u64;
            return;
        }
        // The products `x_i x_j` for `i < j`, at positions `i + j`: row `i` adds `x_i` times
        // the limbs above it at position `2i + 1`, and its carry goes to position `n + i`.
        *r.add(n) = mul_1(r.add(1), x.add(1), n - 1, *x);
        for i in 1..n - 1 {
            *r.add(n + i) = addmul_1(r.add(2 * i + 1), x.add(i + 1), n - i - 1, *x.add(i));
        }
        *r = 0;
        *r.add(2 * n - 1) = 0;
        double_and_add_squares(r, x, n);
    }
}

/// Doubles the `2n` limbs at `r` and adds the squares of the `n` limbs at `x` at the even
/// positions: the last passes of a square, after the products of two different limbs.
///
/// # Safety
///
/// `n >= 1`; the `2n` limbs at `r` must be valid for reads and writes, the `n` at `x` for reads,
/// apart from them, and the top bit of `r` clear.
#[inline]
pub(super) unsafe fn double_and_add_squares(r: *mut u64, x: *const u64, n: usize) {
    // SAFETY: as the caller guarantees; every position is below `2n`.
    unsafe {
        // Doubled: shifted up by one bit.
        let mut top = 0;
        for i in 0..2 * n {
            let v = *r.add(i);
            *r.add(i) = (v << 1) | top;
            top = v >> 63;
        }
        // The squares, at positions `2i`.
        let mut carry = 0u64;
        for i in 0..n {
            let p = u128::from(*x.add(i)) * u128::from(*x.add(i));
            let lo = u128::from(*r.add(2 * i)) + u128::from(p as u64) + u128::from(carry);
            *r.add(2 * i) = lo as u64;
            let hi = u128::from(*r.add(2 * i + 1)) + (p >> 64) + (lo >> 64);
            *r.add(2 * i + 1) = hi as u64;
            carry = (hi >> 64) as u64;
        }
    }
}
