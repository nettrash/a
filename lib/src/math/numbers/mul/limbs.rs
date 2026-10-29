//! The multiplication of magnitudes held as limbs: native `u64`s, the least significant first.
//!
//! Short operands are multiplied by the schoolbook method, in the kernels. From
//! [`KARATSUBA_MUL`] limbs in the shorter operand (and [`KARATSUBA_SQR`] for a square), an
//! operand is split into halves, `x = x1 B^n + x0`, and the product is made of three products of
//! halves instead of four, each computed the same way:
//!
//! `x y = x1 y1 B^2n + (x0 y0 + x1 y1 - (x0 - x1)(y0 - y1)) B^n + x0 y0`.
//!
//! The middle term takes the differences, whose signs are kept apart from their magnitudes, so
//! that no product is longer than its halves allow. A shorter operand of at most half the
//! longer one's limbs multiplies the longer one in pieces of its own length.

#[cfg(target_endian = "little")]
use super::super::add::{add_limbs as add_n, sub_limbs as sub_n};
use super::kernels::{mul_basecase, sqr_basecase};

/// From this many limbs in the shorter operand, a product is computed by Karatsuba's method:
/// measured on an Apple M4 Max, a balanced product of 24 limbs took 529 cycles so, against 570 by
/// the schoolbook method, and one of 20 limbs 411 against 404.
pub(super) const KARATSUBA_MUL: usize = 24;

/// From this many limbs, a square is computed by Karatsuba's method: the schoolbook square needs
/// only about half the limb products, and measured on an Apple M4 Max it was faster up to 56 limbs
/// (1857 cycles against 1925), and as fast at 64.
pub(super) const KARATSUBA_SQR: usize = 64;

/// The limbs of scratch space that [`mul`] needs for operands of `nx >= ny` limbs.
pub(super) fn mul_itch(nx: usize, ny: usize) -> usize {
    if ny < KARATSUBA_MUL {
        0
    } else if ny <= nx.div_ceil(2) {
        let last = nx % ny;
        let rest = if last == 0 { 0 } else { mul_itch(ny, last) };
        2 * ny + mul_itch(ny, ny).max(rest)
    } else {
        let n = nx.div_ceil(2);
        let (sx, ty) = (nx - n, ny - n);
        2 * n + mul_itch(n, n).max(mul_itch(sx, ty))
    }
}

/// The limbs of scratch space that [`sqr`] needs for an operand of `n` limbs.
pub(super) fn sqr_itch(n: usize) -> usize {
    if n < KARATSUBA_SQR {
        0
    } else {
        let h = n.div_ceil(2);
        2 * h + sqr_itch(h)
    }
}

/// Sets the `nx + ny` limbs at `r` to the product of the `nx` limbs at `x` and the `ny` limbs at
/// `y`.
///
/// # Safety
///
/// `nx >= ny >= 1`; the limbs of `x` and `y` must be valid for reads, those of `r` for writes,
/// and the [`mul_itch`]`(nx, ny)` limbs at `s` for reads and writes, all apart from each other.
pub(super) unsafe fn mul(
    r: *mut u64,
    x: *const u64,
    nx: usize,
    y: *const u64,
    ny: usize,
    s: *mut u64,
) {
    debug_assert!(nx >= ny && ny >= 1);
    // SAFETY: forwarded.
    unsafe {
        if ny < KARATSUBA_MUL {
            mul_basecase(r, x, nx, y, ny);
        } else if ny <= nx.div_ceil(2) {
            mul_pieces(r, x, nx, y, ny, s);
        } else {
            karatsuba(r, x, nx, y, ny, s);
        }
    }
}

/// Sets the `2n` limbs at `r` to the square of the `n` limbs at `x`.
///
/// # Safety
///
/// `n >= 1`; the limbs of `x` must be valid for reads, those of `r` for writes, and the
/// [`sqr_itch`]`(n)` limbs at `s` for reads and writes, all apart from each other.
pub(super) unsafe fn sqr(r: *mut u64, x: *const u64, n: usize, s: *mut u64) {
    debug_assert!(n >= 1);
    // SAFETY: forwarded.
    unsafe {
        if n < KARATSUBA_SQR {
            sqr_basecase(r, x, n);
        } else {
            karatsuba_sqr(r, x, n, s);
        }
    }
}

/// [`mul`] for a shorter operand of at most half the longer one's limbs: the longer one is
/// multiplied in pieces of `ny` limbs, each product added in at the position of its piece.
///
/// # Safety
///
/// As for [`mul`], with `ny <= nx.div_ceil(2)`.
unsafe fn mul_pieces(r: *mut u64, x: *const u64, nx: usize, y: *const u64, ny: usize, s: *mut u64) {
    // SAFETY: every piece lies in `x`, every product of a piece in the scratch, and every
    // position of `r` written is below `nx + ny`; the scratch holds the `2ny` limbs of a product
    // and what multiplying a piece needs after them.
    unsafe {
        mul(r, x, ny, y, ny, s);
        let (t, rest) = (s, s.add(2 * ny));
        let mut done = ny;
        while done < nx {
            let len = (nx - done).min(ny);
            mul(t, y, ny, x.add(done), len, rest);
            // The top `ny` limbs of the product so far are at `done`; above them nothing yet.
            let carry = add_n(r.add(done), r.add(done), t, ny);
            core::ptr::copy_nonoverlapping(t.add(ny), r.add(done + ny), len);
            let out = add_1(r.add(done + ny), len, u64::from(carry));
            debug_assert!(out == 0, "a product of pieces carried out of its limbs");
            done += len;
        }
    }
}

/// [`mul`] by Karatsuba's method, for `nx >= ny > nx.div_ceil(2)`: `x` and `y` are split at
/// `n = nx.div_ceil(2)` limbs, into `x0` and `y0` of `n` limbs and `x1` and `y1` of the rest.
///
/// The differences `|x0 - x1|` and `|y0 - y1|` are put in the low `2n` limbs of `r` and their
/// product in the scratch, then `x1 y1` and `x0 y0` are computed in `r` over them, and the three
/// are recombined by [`middle`].
///
/// # Safety
///
/// As for [`mul`], with `ny > nx.div_ceil(2)`.
unsafe fn karatsuba(r: *mut u64, x: *const u64, nx: usize, y: *const u64, ny: usize, s: *mut u64) {
    let n = nx.div_ceil(2);
    let (sx, ty) = (nx - n, ny - n);
    debug_assert!(0 < ty && ty <= sx && sx <= n);
    // SAFETY: `x0`, `x1`, `y0` and `y1` lie in their operands; `2n + sx + ty = nx + ny` limbs of
    // `r` are written; the scratch holds the `2n` limbs of the product of the differences, then
    // the scratch of the products of halves.
    unsafe {
        let (x1, y1) = (x.add(n), y.add(n));
        let negative_x = abs_diff(r, x, n, x1, sx);
        let negative_y = abs_diff(r.add(n), y, n, y1, ty);
        let rest = s.add(2 * n);
        mul(s, r, n, r.add(n), n, rest);
        mul(r.add(2 * n), x1, sx, y1, ty, rest);
        mul(r, x, n, y, n, rest);
        middle(r, n, nx + ny, s, negative_x == negative_y);
    }
}

/// [`sqr`] by Karatsuba's method: as [`karatsuba`], with `(x0 - x1)^2`, which is never negative.
///
/// # Safety
///
/// As for [`sqr`], with `n >= KARATSUBA_SQR`.
unsafe fn karatsuba_sqr(r: *mut u64, x: *const u64, n: usize, s: *mut u64) {
    let h = n.div_ceil(2);
    let sx = n - h;
    // SAFETY: as in `karatsuba`, with both operands `x`.
    unsafe {
        let x1 = x.add(h);
        abs_diff(r, x, h, x1, sx);
        let rest = s.add(2 * h);
        sqr(s, r, h, rest);
        sqr(r.add(2 * h), x1, sx, rest);
        sqr(r, x, h, rest);
        middle(r, h, 2 * n, s, true);
    }
}

/// Recombines Karatsuba's three products in place: with `v0 = x0 y0` in limbs `0..2n` of `r`,
/// `vinf = x1 y1` in limbs `2n..len`, and the product of the differences in the `2n` limbs at `s`,
/// makes `r = v0 + (v0 + vinf -+ s) B^n + vinf B^2n`, minus when `subtract`.
///
/// Splitting each product into its halves of `n` limbs, the sum `H(v0) + L(vinf)` appears both at
/// `B^n` and at `B^2n`: it is computed once, at limb `2n` over `L(vinf)`, then `L(v0)` added to it
/// at limb `n` over `H(v0)`, and `H(vinf)` at limb `2n`. The product of the differences is then
/// added or subtracted over limbs `n..3n`, and the carries of the passes go in last, at limbs `2n`
/// and `3n`: five passes of `n` limbs in all, after GMP's `mpn_toom22_mul`.
///
/// # Safety
///
/// The `len >= 3n` limbs at `r` and the `2n` at `s` must be valid for reads and writes, and
/// apart, with `len <= 4n`.
unsafe fn middle(r: *mut u64, n: usize, len: usize, s: *mut u64, subtract: bool) {
    // SAFETY: as the caller guarantees; every range read or written lies in `r[0..len]` or
    // `s[0..2n]`.
    unsafe {
        let high = len - 3 * n;
        let c = add_n(r.add(2 * n), r.add(n), r.add(2 * n), n);
        let c2 = u64::from(c) + u64::from(add_n(r.add(n), r.add(2 * n), r, n));
        let mut cy =
            i64::from(c) + i64::from(add_ext(r.add(2 * n), r.add(2 * n), n, r.add(3 * n), high));
        if subtract {
            cy -= i64::from(sub_n(r.add(n), r.add(n), s, 2 * n));
        } else {
            cy += i64::from(add_n(r.add(n), r.add(n), s, 2 * n));
        }
        // The two carries go in one after the other: the product fits its limbs, so what one of
        // them carries out of the top, the other takes back (`cy` is -1 after a borrow).
        add_1(r.add(2 * n), len - 2 * n, c2);
        if cy >= 0 {
            add_1(r.add(3 * n), high, cy as u64);
        } else {
            sub_1(r.add(3 * n), high, 1);
        }
    }
}

/// Sets the `na` limbs at `r` to `|a - b|` for the `na` limbs at `a` and the `nb <= na` limbs at
/// `b`, and returns whether `a < b`.
///
/// # Safety
///
/// The limbs of `a` and `b` must be valid for reads and those of `r` for writes, `r` apart from
/// both.
unsafe fn abs_diff(r: *mut u64, a: *const u64, na: usize, b: *const u64, nb: usize) -> bool {
    // SAFETY: as the caller guarantees.
    unsafe {
        let above = (nb..na).any(|i| *a.add(i) != 0);
        let less = !above && cmp_n(a, b, nb) == core::cmp::Ordering::Less;
        if less {
            sub_n(r, b, a, nb);
            core::ptr::write_bytes(r.add(nb), 0, na - nb);
        } else {
            let borrow = sub_n(r, a, b, nb);
            core::ptr::copy_nonoverlapping(a.add(nb), r.add(nb), na - nb);
            let out = sub_1(r.add(nb), na - nb, u64::from(borrow));
            debug_assert!(out == 0);
        }
        less
    }
}

/// Compares the `n` limbs at `a` with the `n` at `b`, from the top.
///
/// # Safety
///
/// The limbs must be valid for reads.
unsafe fn cmp_n(a: *const u64, b: *const u64, n: usize) -> core::cmp::Ordering {
    // SAFETY: as the caller guarantees.
    unsafe {
        for i in (0..n).rev() {
            let (x, y) = (*a.add(i), *b.add(i));
            if x != y {
                return x.cmp(&y);
            }
        }
    }
    core::cmp::Ordering::Equal
}

/// Sets the `na` limbs at `r` to `a + b` for the `na` limbs at `a` and the `nb <= na` limbs at
/// `b`, and returns the carry out.
///
/// # Safety
///
/// The limbs of `a` and `b` must be valid for reads and those of `r` for writes; `r` may be `a`.
unsafe fn add_ext(r: *mut u64, a: *const u64, na: usize, b: *const u64, nb: usize) -> bool {
    // SAFETY: as the caller guarantees.
    unsafe {
        let carry = add_n(r, a, b, nb);
        if r.cast_const() != a {
            core::ptr::copy_nonoverlapping(a.add(nb), r.add(nb), na - nb);
        }
        add_1(r.add(nb), na - nb, u64::from(carry)) != 0
    }
}

/// Adds `c` to the `n` limbs at `r`, and returns what carries out of them.
///
/// # Safety
///
/// The limbs must be valid for reads and writes.
unsafe fn add_1(r: *mut u64, n: usize, mut c: u64) -> u64 {
    // SAFETY: as the caller guarantees.
    unsafe {
        let mut i = 0;
        while c != 0 && i < n {
            let (v, o) = (*r.add(i)).overflowing_add(c);
            *r.add(i) = v;
            c = u64::from(o);
            i += 1;
        }
    }
    c
}

/// Subtracts `c` from the `n` limbs at `r`, and returns what borrows out of them.
///
/// # Safety
///
/// The limbs must be valid for reads and writes.
unsafe fn sub_1(r: *mut u64, n: usize, mut c: u64) -> u64 {
    // SAFETY: as the caller guarantees.
    unsafe {
        let mut i = 0;
        while c != 0 && i < n {
            let (v, o) = (*r.add(i)).overflowing_sub(c);
            *r.add(i) = v;
            c = u64::from(o);
            i += 1;
        }
    }
    c
}

/// Sets the `n` limbs at `r` to `a + b`, and returns the carry out: on a big-endian target, where
/// the kernels of the addition read little-endian limbs, not native ones.
///
/// # Safety
///
/// The limbs must be valid; `r` may be `a` or `b`, and is otherwise apart from them.
#[cfg(target_endian = "big")]
unsafe fn add_n(r: *mut u64, a: *const u64, b: *const u64, n: usize) -> bool {
    let mut carry = false;
    // SAFETY: as the caller guarantees; limb `i` of the inputs is read before limb `i` of `r`
    // is written.
    unsafe {
        for i in 0..n {
            let (v, c1) = (*a.add(i)).overflowing_add(*b.add(i));
            let (v, c2) = v.overflowing_add(u64::from(carry));
            *r.add(i) = v;
            carry = c1 | c2;
        }
    }
    carry
}

/// Sets the `n` limbs at `r` to `a - b`, and returns the borrow out, as [`add_n`] adds.
///
/// # Safety
///
/// As for [`add_n`].
#[cfg(target_endian = "big")]
unsafe fn sub_n(r: *mut u64, a: *const u64, b: *const u64, n: usize) -> bool {
    let mut borrow = false;
    // SAFETY: as in `add_n`.
    unsafe {
        for i in 0..n {
            let (v, b1) = (*a.add(i)).overflowing_sub(*b.add(i));
            let (v, b2) = v.overflowing_sub(u64::from(borrow));
            *r.add(i) = v;
            borrow = b1 | b2;
        }
    }
    borrow
}
