//! Toom-Cook's methods for long products, after Karatsuba's: an operand cut into `p` parts of
//! `k` limbs is a polynomial of degree `p - 1` at `X = B^k`, and the product of two such
//! polynomials, of degree `2p - 2`, is found from its values at `2p - 1` points, each the product
//! of the operands' values there, which are numbers of about `k` limbs.
//!
//! Toom-3 cuts the operands in three parts and multiplies their values at `0`, `1`, `-1`, `2` and
//! infinity (the product of the top parts): five products of a third of the length instead of
//! nine, in O(n^1.47). Toom-4 cuts them in four and takes seven points, `0`, `1`, `-1`, `2`, `-2`,
//! `1/2` (as `2^6` times the value there, which is a whole number) and infinity, in O(n^1.40).
//!
//! Evaluating takes additions and shifts of parts, and interpolating, which recovers the
//! coefficients of the product from its values, takes additions, shifts and exact divisions by
//! 3, 9 and 15, all linear in the length: the sequences of Marco Bodrato, which GMP 6.3 uses too
//! (`mpn_toom_interpolate_5pts` and `_7pts`). Values at negative points can be negative: their
//! magnitudes are multiplied, and the sign is kept apart. Some steps of Toom-4's interpolation
//! go through negative numbers, which are held in two's complement over the length of a value;
//! exact divisions by odd numbers hold there as well, while halving is done only on numbers that
//! are not negative.
//!
//! The values at the points other than 0 and infinity have one limb more than a part, a small
//! one (at most 2 for Toom-3's `x(1)`, 14 for Toom-4's `x(2)`), and are multiplied as numbers of
//! `k + 1` limbs; a product of two of them is below `B^(2k + 1)`. The coefficients come out in
//! `2k + 1` limbs each and are added at their positions in the end, the high part of each folded
//! into the next one so that every position of the product is written once.

use super::kernels::{divexact_finish, mul_1};
use super::limbs::{
    abs_diff, add_1, add_ext, add_n, cmp_n, mul, mul_itch, sqr, sqr_itch, sub_ext, sub_n,
};
use core::cmp::Ordering;

// Shapes and scratch

/// The limbs a value of `k + 1` limbs takes in a layout: rounded up to an even number, so that
/// every value and product starts on 16 bytes when the layout does, as the kernels read and write
/// limbs in pairs.
fn stride(k: usize) -> usize {
    (k + 2) & !1
}

/// Whether Toom-3 can multiply operands of `nx >= ny` limbs: cut at `k = nx.div_ceil(3)` limbs,
/// the shorter operand must reach into its third part.
pub(super) fn toom33_fits(nx: usize, ny: usize) -> bool {
    nx >= 12 && ny > 2 * nx.div_ceil(3)
}

/// Whether Toom-4 can multiply operands of `nx >= ny` limbs: the shorter operand must reach into
/// the fourth part, of `nx.div_ceil(4)` limbs a part.
pub(super) fn toom44_fits(nx: usize, ny: usize) -> bool {
    nx >= 16 && ny > 3 * nx.div_ceil(4)
}

/// The size of the parts of Toom-4x2 for operands of `nx >= ny` limbs: a quarter of the longer
/// one, or half of the shorter one when that is more.
fn toom42_part(nx: usize, ny: usize) -> usize {
    if nx >= 2 * ny {
        nx.div_ceil(4)
    } else {
        ny.div_ceil(2)
    }
}

/// Whether Toom-4x2 can multiply operands of `nx >= ny` limbs: the longer one cut into four parts
/// and the shorter one into two, each top part of at least one limb and at most a part.
pub(super) fn toom42_fits(nx: usize, ny: usize) -> bool {
    let k = toom42_part(nx, ny);
    k >= 4 && nx > 3 * k && nx <= 4 * k && ny > k && ny <= 2 * k
}

/// The scratch of the products of parts: of `k + 1` limbs each, of `k`, and of the top parts.
fn parts_itch(k: usize, sx: usize, ty: usize) -> usize {
    mul_itch(k + 1, k + 1)
        .max(mul_itch(k, k))
        .max(mul_itch(sx, ty))
}

/// The scratch that [`toom33`] needs for operands of `nx >= ny` limbs: two products of `2k + 2`
/// limbs, three values of `k + 1`, and the scratch of the products of parts.
pub(super) fn toom33_itch(nx: usize, ny: usize) -> usize {
    let k = nx.div_ceil(3);
    4 * k + 4 + 3 * stride(k) + parts_itch(k, nx - 2 * k, ny - 2 * k)
}

/// The scratch that [`toom42`] needs for operands of `nx >= ny` limbs, as [`toom33`] does.
pub(super) fn toom42_itch(nx: usize, ny: usize) -> usize {
    let k = toom42_part(nx, ny);
    let (sx, ty) = (nx - 3 * k, ny - k);
    4 * k + 4 + 3 * stride(k) + parts_itch(k, sx.max(ty), sx.min(ty))
}

/// The scratch that [`toom3_sqr`] needs for an operand of `n` limbs.
pub(super) fn toom3_sqr_itch(n: usize) -> usize {
    let k = n.div_ceil(3);
    4 * k + 4 + 2 * stride(k) + sqr_itch(k + 1).max(sqr_itch(k)).max(sqr_itch(n - 2 * k))
}

/// The scratch that [`toom44`] needs for operands of `nx >= ny` limbs: four products of `2k + 2`
/// limbs, two values of `k + 1`, and the scratch of the products of parts.
pub(super) fn toom44_itch(nx: usize, ny: usize) -> usize {
    let k = nx.div_ceil(4);
    8 * k + 8 + 2 * stride(k) + parts_itch(k, nx - 3 * k, ny - 3 * k)
}

/// The scratch that [`toom4_sqr`] needs for an operand of `n` limbs.
pub(super) fn toom4_sqr_itch(n: usize) -> usize {
    let k = n.div_ceil(4);
    8 * k + 8 + 2 * stride(k) + sqr_itch(k + 1).max(sqr_itch(k)).max(sqr_itch(n - 3 * k))
}

// Toom-3

/// Sets the `nx + ny` limbs at `r` to the product of the `nx` limbs at `x` and the `ny` limbs at
/// `y` by Toom-3: `x = x0 + x1 X + x2 X^2` with `X = B^k`, `k = nx.div_ceil(3)`, the top part
/// `x2` of `sx` limbs, and `y` likewise with a top part of `ty`.
///
/// The values of the operands at 1, -1 and 2 are taken into the scratch and into the low limbs of
/// `r`, the products at -1 and 2 into the scratch, and those at 1, infinity and 0 into `r`, where
/// the product at 1 lies at limb `2k`, its top limb kept aside, then [`interpolate5`] makes the
/// product of them.
///
/// # Safety
///
/// [`toom33_fits`]`(nx, ny)`; the limbs of `x` and `y` must be valid for reads, those of `r` for
/// writes, and the [`toom33_itch`]`(nx, ny)` limbs at `s` for reads and writes, all apart.
pub(super) unsafe fn toom33(
    r: *mut u64,
    x: *const u64,
    nx: usize,
    y: *const u64,
    ny: usize,
    s: *mut u64,
) {
    let k = nx.div_ceil(3);
    let (sx, ty) = (nx - 2 * k, ny - 2 * k);
    debug_assert!(0 < ty && ty <= sx && sx <= k);
    // SAFETY: the scratch holds the two products of `2k + 2` limbs, the three values of `k + 1`
    // in `stride(k) <= k + 2` limbs each and, after them, the scratch of the products of parts;
    // the values in `r`, at limbs `0..3 stride(k)`, lie below `nx + ny = 4k + sx + ty` as
    // `k >= 4`, and each is read before the product that is written over it. Every product is
    // written apart from its operands.
    unsafe {
        let v = stride(k);
        let (vm1, v2) = (s, s.add(2 * k + 2));
        let (xp1, xm1, ym1) = (
            s.add(4 * k + 4),
            s.add(4 * k + 4 + v),
            s.add(4 * k + 4 + 2 * v),
        );
        let rest = s.add(4 * k + 4 + 3 * v);
        let (yp1, xp2, yp2) = (r, r.add(v), r.add(2 * v));
        let negative_x = eval3(xp1, xm1, xp2, x, k, sx);
        let negative_y = eval3(yp1, ym1, yp2, y, k, ty);
        product_of_values(vm1, xm1, ym1, k, rest);
        mul(v2, xp2, k + 1, yp2, k + 1, rest);
        mul(r.add(2 * k), xp1, k + 1, yp1, k + 1, rest);
        let v1_top = *r.add(4 * k);
        mul(r.add(4 * k), x.add(2 * k), sx, y.add(2 * k), ty, rest);
        mul(r, x, k, y, k, rest);
        interpolate5(r, k, sx + ty, vm1, v2, v1_top, negative_x != negative_y);
    }
}

/// Sets the `2n` limbs at `r` to the square of the `n` limbs at `x` by Toom-3, as [`toom33`]
/// multiplies: the values of `x` alone, squared, and the square at -1 never negative.
///
/// # Safety
///
/// `n >= 3`; the limbs of `x` must be valid for reads, those of `r` for writes, and the
/// [`toom3_sqr_itch`]`(n)` limbs at `s` for reads and writes, all apart.
pub(super) unsafe fn toom3_sqr(r: *mut u64, x: *const u64, n: usize, s: *mut u64) {
    let k = n.div_ceil(3);
    let sx = n - 2 * k;
    debug_assert!(0 < sx && sx <= k);
    // SAFETY: as in `toom33`, with the value at 2 in `r`, at limbs `0..k + 1`, below `2n`.
    unsafe {
        let v = stride(k);
        let (vm1, v2) = (s, s.add(2 * k + 2));
        let (p1, m1) = (s.add(4 * k + 4), s.add(4 * k + 4 + v));
        let rest = s.add(4 * k + 4 + 2 * v);
        let p2 = r;
        eval3(p1, m1, p2, x, k, sx);
        if *m1.add(k) == 0 {
            sqr(vm1, m1, k, rest);
            *vm1.add(2 * k) = 0;
        } else {
            sqr(vm1, m1, k + 1, rest);
        }
        sqr(v2, p2, k + 1, rest);
        sqr(r.add(2 * k), p1, k + 1, rest);
        let v1_top = *r.add(4 * k);
        sqr(r.add(4 * k), x.add(2 * k), sx, rest);
        sqr(r, x, k, rest);
        interpolate5(r, k, 2 * sx, vm1, v2, v1_top, false);
    }
}

/// Sets the `k + 1` limbs at `p1` to `x(1) = x0 + x1 + x2`, those at `m1` to `|x(-1)|`, the
/// magnitude of `x0 - x1 + x2`, and those at `p2` to `x(2) = x0 + 2 x1 + 4 x2`, for the parts
/// `x0` and `x1` of `k` limbs at `x` and `x2` of `s` limbs after them; returns whether `x(-1)` is
/// negative. `x(2)` is made as `2 (x(1) + x2) - x0`.
///
/// # Safety
///
/// `1 <= s <= k`; the `2k + s` limbs at `x` must be valid for reads, and the `k + 1` at each of
/// `p1`, `m1` and `p2` for writes, apart from each other and from `x`.
unsafe fn eval3(
    p1: *mut u64,
    m1: *mut u64,
    p2: *mut u64,
    x: *const u64,
    k: usize,
    s: usize,
) -> bool {
    // SAFETY: as the caller guarantees; `p2` holds `x0 + x2` until `x(-1)` is taken.
    unsafe {
        let (x0, x1, x2) = (x, x.add(k), x.add(2 * k));
        *p2.add(k) = u64::from(add_ext(p2, x0, k, x2, s));
        *p1.add(k) = *p2.add(k) + u64::from(add_n(p1, p2, x1, k));
        let negative = *p2.add(k) == 0 && cmp_n(p2, x1, k) == Ordering::Less;
        if negative {
            sub_n(m1, x1, p2, k);
            *m1.add(k) = 0;
        } else {
            *m1.add(k) = *p2.add(k) - u64::from(sub_n(m1, p2, x1, k));
        }
        *p2.add(k) = *p1.add(k) + u64::from(add_ext(p2, p1, k, x2, s));
        lshift::<1>(p2, p2, k + 1);
        *p2.add(k) -= u64::from(sub_n(p2, p2, x0, k));
        negative
    }
}

/// Sets the `nx + ny` limbs at `r` to the product of the `nx` limbs at `x` and the `ny` limbs at
/// `y` by Toom-4x2, for a longer operand of about twice the shorter one's limbs or more: `x` is
/// cut into four parts of `k` limbs, the top one of `sx`, and `y` into two, the top one of `ty`.
/// The product, of degree 4, takes the five points of Toom-3, and the values are laid out as
/// [`toom33`] lays them out.
///
/// # Safety
///
/// [`toom42_fits`]`(nx, ny)`; the limbs of `x` and `y` must be valid for reads, those of `r` for
/// writes, and the [`toom42_itch`]`(nx, ny)` limbs at `s` for reads and writes, all apart.
pub(super) unsafe fn toom42(
    r: *mut u64,
    x: *const u64,
    nx: usize,
    y: *const u64,
    ny: usize,
    s: *mut u64,
) {
    let k = toom42_part(nx, ny);
    let (sx, ty) = (nx - 3 * k, ny - k);
    debug_assert!(0 < sx && sx <= k && 0 < ty && ty <= k);
    // SAFETY: as in `toom33`; the magnitude of `x(-2)`, which is not used, and the scratch of its
    // evaluation go where the product at 2 goes later.
    unsafe {
        let v = stride(k);
        let (vm1, v2) = (s, s.add(2 * k + 2));
        let (xp1, xm1, ym1) = (
            s.add(4 * k + 4),
            s.add(4 * k + 4 + v),
            s.add(4 * k + 4 + 2 * v),
        );
        let rest = s.add(4 * k + 4 + 3 * v);
        let (yp1, xp2, yp2) = (r, r.add(v), r.add(2 * v));
        let negative_x = eval4_pm1(xp1, xm1, x, k, sx, v2);
        eval4_pm2(xp2, v2, x, k, sx, v2.add(k + 1));
        let (y0, y1) = (y, y.add(k));
        *yp1.add(k) = u64::from(add_ext(yp1, y0, k, y1, ty));
        let negative_y = abs_diff(ym1, y0, k, y1, ty);
        *yp2.add(k) = *yp1.add(k) + u64::from(add_ext(yp2, yp1, k, y1, ty));
        if *xm1.add(k) == 0 {
            mul(vm1, xm1, k, ym1, k, rest);
            *vm1.add(2 * k) = 0;
        } else {
            mul(vm1, xm1, k + 1, ym1, k, rest);
        }
        mul(v2, xp2, k + 1, yp2, k + 1, rest);
        mul(r.add(2 * k), xp1, k + 1, yp1, k + 1, rest);
        let v1_top = *r.add(4 * k);
        let (x3, y1) = (x.add(3 * k), y.add(k));
        if sx >= ty {
            mul(r.add(4 * k), x3, sx, y1, ty, rest);
        } else {
            mul(r.add(4 * k), y1, ty, x3, sx, rest);
        }
        mul(r, x, k, y, k, rest);
        interpolate5(r, k, sx + ty, vm1, v2, v1_top, negative_x != negative_y);
    }
}

/// Recovers the product from Toom-3's five values and writes it: with `v0` in the `2k` limbs at
/// `r`, the product at 1 in the `2k` limbs at `r + 2k` with `v1_top` above them, the product at
/// infinity in the `w` limbs at `r + 4k`, and the magnitude of the product at -1 (negative when
/// `negative`) and the product at 2 in the `2k + 1` limbs at `vm1` and `v2`, makes
/// `r = r0 + r1 X + r2 X^2 + r3 X^3 + r4 X^4` in its `4k + w` limbs.
///
/// Bodrato's sequence, on `m = 2k + 1` limbs, where `vm1` is the signed value:
///
/// 1. `v2 = (v2 - vm1) / 3`, which is `r1 + r2 + 3 r3 + 5 r4`;
/// 2. `vm1 = (v1 - vm1) / 2`, `r1 + r3`;
/// 3. `v1 = v1 - v0`, `r1 + r2 + r3 + r4`;
/// 4. `v2 = (v2 - v1) / 2`, `r3 + 2 r4`;
/// 5. `v1 = v1 - vm1`, `r2 + r4`;
/// 6. `v2 = v2 - 2 r4`, `r3`;
/// 7. `v1 = v1 - r4`, `r2`;
/// 8. `vm1 = vm1 - v2`, `r1`.
///
/// None of them is negative. `r0`, `r2` and `r4` are then in place; `r1` and `r3` are added at
/// limbs `k` and `3k`, the high part of `r3` with the top limb of `r2` and the carry folded in,
/// so that each limb above `3k` is added once.
///
/// # Safety
///
/// `k >= 1`, `2 <= w <= 2k`; the `4k + w` limbs at `r` and the `2k + 1` at each of `vm1` and `v2`
/// must be valid for reads and writes, and apart.
unsafe fn interpolate5(
    r: *mut u64,
    k: usize,
    w: usize,
    vm1: *mut u64,
    v2: *mut u64,
    v1_top: u64,
    negative: bool,
) {
    let m = 2 * k + 1;
    let mut top = v1_top;
    // SAFETY: as the caller guarantees; every range lies in `r[0..4k + w]`, `vm1[0..m]` or
    // `v2[0..m]`.
    unsafe {
        let v1 = r.add(2 * k);
        let vinf = r.add(4 * k);
        if negative {
            add_n(v2, v2, vm1, m);
        } else {
            sub_n(v2, v2, vm1, m);
        }
        divexact::<3>(v2, m);
        if negative {
            let carry = add_n(vm1, v1, vm1, 2 * k);
            *vm1.add(2 * k) += top + u64::from(carry);
        } else {
            let borrow = sub_n(vm1, v1, vm1, 2 * k);
            *vm1.add(2 * k) = top - *vm1.add(2 * k) - u64::from(borrow);
        }
        rshift::<1>(vm1, vm1, m);
        top -= u64::from(sub_n(v1, v1, r, 2 * k));
        let borrow = sub_n(v2, v2, v1, 2 * k);
        *v2.add(2 * k) = *v2.add(2 * k) - top - u64::from(borrow);
        rshift::<1>(v2, v2, m);
        let borrow = sub_n(v1, v1, vm1, 2 * k);
        top = top - *vm1.add(2 * k) - u64::from(borrow);
        sub_ext(v2, v2, m, vinf, w);
        sub_ext(v2, v2, m, vinf, w);
        top -= u64::from(sub_ext(v1, v1, 2 * k, vinf, w));
        sub_n(vm1, vm1, v2, m);

        // r1 at limb k: what carries out of limb 4k - 1 goes to the top of r2.
        let carry = add_n(r.add(k), r.add(k), vm1, m);
        top += add_1(r.add(3 * k + 1), k - 1, u64::from(carry));
        // r3 at limb 3k: its low k limbs over r2, the rest, with the top of r2, over r4.
        let carry = add_n(r.add(3 * k), r.add(3 * k), v2, k);
        let out = add_1(v2.add(k), k + 1, top + u64::from(carry));
        debug_assert!(out == 0);
        if w > k + 1 {
            let carry = add_n(vinf, vinf, v2.add(k), k + 1);
            add_1(vinf.add(k + 1), w - k - 1, u64::from(carry));
        } else {
            add_n(vinf, vinf, v2.add(k), w);
        }
    }
}

// Toom-4

/// Sets the `nx + ny` limbs at `r` to the product of the `nx` limbs at `x` and the `ny` limbs at
/// `y` by Toom-4: `x = x0 + x1 X + x2 X^2 + x3 X^3` with `X = B^k`, `k = nx.div_ceil(4)`, the top
/// part `x3` of `sx` limbs, and `y` likewise with a top part of `ty`.
///
/// The values are taken a pair of points at a time, into the low limbs of `r` and the scratch,
/// and multiplied into the scratch: at 2 and -2, at 1/2, then at -1. The product at 1 goes to
/// limb `2k` of `r`, those at infinity and 0 to limbs `6k` and 0, and [`interpolate7`] makes the
/// product of them.
///
/// # Safety
///
/// [`toom44_fits`]`(nx, ny)`; the limbs of `x` and `y` must be valid for reads, those of `r` for
/// writes, and the [`toom44_itch`]`(nx, ny)` limbs at `s` for reads and writes, all apart.
pub(super) unsafe fn toom44(
    r: *mut u64,
    x: *const u64,
    nx: usize,
    y: *const u64,
    ny: usize,
    s: *mut u64,
) {
    let k = nx.div_ceil(4);
    let (sx, ty) = (nx - 3 * k, ny - 3 * k);
    debug_assert!(0 < ty && ty <= sx && sx <= k);
    // SAFETY: the scratch holds the four products of `2k + 2` limbs, two values of `k + 1` in
    // `stride(k) <= k + 2` limbs each, which serve [`interpolate7`] as its `2k + 1` limbs of
    // scratch, and, after them, the scratch of the products of parts; the values in `r`, at limbs
    // `0..3 stride(k)`, lie below `nx + ny = 6k + sx + ty`, and each is read before the product
    // that is written over it. Every product is written apart from its operands.
    unsafe {
        let (n, v) = (2 * k + 2, stride(k));
        let (w1, w3, w4, w5) = (s, s.add(n), s.add(2 * n), s.add(3 * n));
        let (ya, g) = (s.add(4 * n), s.add(4 * n + v));
        let rest = s.add(4 * n + 2 * v);
        let (xa, xb, yb) = (r, r.add(v), r.add(2 * v));
        let negative_x = eval4_pm2(xa, xb, x, k, sx, g);
        let negative_y = eval4_pm2(ya, yb, y, k, ty, g);
        mul(w4, xa, k + 1, ya, k + 1, rest);
        mul(w1, xb, k + 1, yb, k + 1, rest);
        let negative_2 = negative_x != negative_y;
        eval4_half(xa, x, k, sx);
        eval4_half(ya, y, k, ty);
        mul(w5, xa, k + 1, ya, k + 1, rest);
        let negative_x = eval4_pm1(xa, xb, x, k, sx, g);
        let negative_y = eval4_pm1(ya, yb, y, k, ty, g);
        product_of_values(w3, xb, yb, k, rest);
        mul(r.add(2 * k), xa, k + 1, ya, k + 1, rest);
        mul(r.add(6 * k), x.add(3 * k), sx, y.add(3 * k), ty, rest);
        mul(r, x, k, y, k, rest);
        interpolate7(
            r,
            k,
            sx + ty,
            [w1, w3, w4, w5],
            negative_2,
            negative_x != negative_y,
            ya,
        );
    }
}

/// Sets the `2n` limbs at `r` to the square of the `n` limbs at `x` by Toom-4, as [`toom44`]
/// multiplies.
///
/// # Safety
///
/// `n >= 4`; the limbs of `x` must be valid for reads, those of `r` for writes, and the
/// [`toom4_sqr_itch`]`(n)` limbs at `s` for reads and writes, all apart.
pub(super) unsafe fn toom4_sqr(r: *mut u64, x: *const u64, n: usize, s: *mut u64) {
    let k = n.div_ceil(4);
    let sx = n - 3 * k;
    debug_assert!(0 < sx && sx <= k);
    // SAFETY: as in `toom44`, with the values at limbs `0..2 stride(k)` of `r`, and the scratch
    // of the evaluations and of `interpolate7` in the `2 stride(k)` limbs after the products.
    unsafe {
        let (m, v) = (2 * k + 2, stride(k));
        let (w1, w3, w4, w5) = (s, s.add(m), s.add(2 * m), s.add(3 * m));
        let g = s.add(4 * m);
        let rest = s.add(4 * m + 2 * v);
        let (a, b) = (r, r.add(v));
        eval4_pm2(a, b, x, k, sx, g);
        sqr(w4, a, k + 1, rest);
        sqr(w1, b, k + 1, rest);
        eval4_half(a, x, k, sx);
        sqr(w5, a, k + 1, rest);
        eval4_pm1(a, b, x, k, sx, g);
        if *b.add(k) == 0 {
            sqr(w3, b, k, rest);
            *w3.add(2 * k) = 0;
        } else {
            sqr(w3, b, k + 1, rest);
        }
        sqr(r.add(2 * k), a, k + 1, rest);
        sqr(r.add(6 * k), x.add(3 * k), sx, rest);
        sqr(r, x, k, rest);
        interpolate7(r, k, 2 * sx, [w1, w3, w4, w5], false, false, g);
    }
}

/// Sets the `k + 1` limbs at `p` to `x(1) = x0 + x1 + x2 + x3` and those at `m` to `|x(-1)|`, for
/// the parts `x0`, `x1`, `x2` of `k` limbs at `x` and `x3` of `s` limbs after them, as
/// `(x0 + x2) +- (x1 + x3)`; returns whether `x(-1)` is negative.
///
/// # Safety
///
/// `1 <= s <= k`; the `3k + s` limbs at `x` must be valid for reads, and the `k + 1` at each of
/// `p`, `m` and `g` (which this clobbers) for writes, apart from each other and from `x`.
unsafe fn eval4_pm1(
    p: *mut u64,
    m: *mut u64,
    x: *const u64,
    k: usize,
    s: usize,
    g: *mut u64,
) -> bool {
    // SAFETY: as the caller guarantees.
    unsafe {
        *p.add(k) = u64::from(add_n(p, x, x.add(2 * k), k));
        *g.add(k) = u64::from(add_ext(g, x.add(k), k, x.add(3 * k), s));
        sum_and_difference(p, m, g, k + 1)
    }
}

/// Sets the `k + 1` limbs at `p` to `x(2) = x0 + 2 x1 + 4 x2 + 8 x3` and those at `m` to
/// `|x(-2)|`, as `(x0 + 4 x2) +- 2 (x1 + 4 x3)`, for the parts of [`eval4_pm1`]; returns whether
/// `x(-2)` is negative.
///
/// # Safety
///
/// As for [`eval4_pm1`].
unsafe fn eval4_pm2(
    p: *mut u64,
    m: *mut u64,
    x: *const u64,
    k: usize,
    s: usize,
    g: *mut u64,
) -> bool {
    // SAFETY: as the caller guarantees; `m` holds `4 x3` until the difference is taken.
    unsafe {
        *p.add(k) = lshift::<2>(p, x.add(2 * k), k);
        *p.add(k) += u64::from(add_n(p, p, x, k));
        *m.add(s) = lshift::<2>(m, x.add(3 * k), s);
        *g.add(k) = if s < k {
            u64::from(add_ext(g, x.add(k), k, m, s + 1))
        } else {
            *m.add(k) + u64::from(add_n(g, x.add(k), m, k))
        };
        lshift::<1>(g, g, k + 1);
        sum_and_difference(p, m, g, k + 1)
    }
}

/// Sets the `k + 1` limbs at `p` to `2^3 x(1/2) = 8 x0 + 4 x1 + 2 x2 + x3`, for the parts of
/// [`eval4_pm1`], as `2 (2 (2 x0 + x1) + x2) + x3`.
///
/// # Safety
///
/// `1 <= s <= k`; the `3k + s` limbs at `x` must be valid for reads, and the `k + 1` at `p` for
/// writes, apart from them.
unsafe fn eval4_half(p: *mut u64, x: *const u64, k: usize, s: usize) {
    // SAFETY: as the caller guarantees.
    unsafe {
        *p.add(k) = lshift::<1>(p, x, k);
        *p.add(k) += u64::from(add_n(p, p, x.add(k), k));
        lshift::<1>(p, p, k + 1);
        *p.add(k) += u64::from(add_n(p, p, x.add(2 * k), k));
        lshift::<1>(p, p, k + 1);
        add_ext(p, p, k + 1, x.add(3 * k), s);
    }
}

/// Sets the `n` limbs at `m` to `|p - g|` and those at `p` to `p + g`; returns whether
/// `p < g`. The sum must fit in `n` limbs.
///
/// # Safety
///
/// The `n` limbs at each pointer must be valid for reads and writes, and apart.
unsafe fn sum_and_difference(p: *mut u64, m: *mut u64, g: *const u64, n: usize) -> bool {
    // SAFETY: as the caller guarantees.
    unsafe {
        let negative = cmp_n(p, g, n) == Ordering::Less;
        if negative {
            sub_n(m, g, p, n);
        } else {
            sub_n(m, p, g, n);
        }
        add_n(p, p, g, n);
        negative
    }
}

/// Recovers the product from Toom-4's seven values and writes it: with `w0 = v0` in the `2k`
/// limbs at `r`, `w2`, the product at 1, in the `2k + 1` at `r + 2k`, `w6`, the product at
/// infinity, in the `w` at `r + 6k`, and in the `2k + 1` limbs at each of `ws`, `w1`, the
/// magnitude of the product at -2 (negative when `negative_2`), `w3`, that at -1 (negative when
/// `negative_1`), `w4`, the product at 2, and `w5`, `2^6` times the product at 1/2, makes
/// `r = r0 + r1 X + ... + r6 X^6` in its `6k + w` limbs.
///
/// Bodrato's sequence, on `m = 2k + 1` limbs, with `w1` and `w3` the signed values; the values
/// marked may be negative, in two's complement, and only those that are not are halved:
///
/// 1. `w5 = w5 + w4`; `w1 = (w4 - w1) / 2`; `w4 = w4 - w0`;
/// 2. `w4 = (w4 - w1) / 4 - 16 w6`; `w3 = (w2 - w3) / 2`; `w2 = w2 - w3`;
/// 3. `w5 = w5 - 65 w2` (may be negative); `w2 = w2 - w6 - w0`; `w5 = (w5 + 45 w2) / 2`;
/// 4. `w4 = (w4 - w2) / 3`; `w2 = w2 - w4`; `w1 = w5 - w1` (may be negative);
/// 5. `w5 = (w5 - 8 w3) / 9`; `w3 = w3 - w5`; `w1 = (w1 / 15 + w5) / 2`; `w5 = w5 - w1`;
///
/// which leaves `wi = ri`. `r0`, `r2` and `r6` are then in place, and the others are added at
/// limb `ik`: `r1` over `r0` and `r2`, then each of `r3`, `r4` and `r5` its low `k` limbs at its
/// position and its high part, with the carry and the top limb below folded in, after them.
///
/// # Safety
///
/// `k >= 1`, `2 <= w <= 2k`; the `6k + w` limbs at `r`, the `2k + 1` at each of `ws` and the
/// `2k + 1` at `t` (scratch) must be valid for reads and writes, and apart.
unsafe fn interpolate7(
    r: *mut u64,
    k: usize,
    w: usize,
    ws: [*mut u64; 4],
    negative_2: bool,
    negative_1: bool,
    t: *mut u64,
) {
    let m = 2 * k + 1;
    let [w1, w3, w4, w5] = ws;
    // SAFETY: as the caller guarantees; every range lies in `r[0..6k + w]`, `t[0..m]` or one of
    // the values.
    unsafe {
        let (w0, w2, w6) = (r, r.add(2 * k), r.add(6 * k));
        add_n(w5, w5, w4, m);
        if negative_2 {
            add_n(w1, w4, w1, m);
        } else {
            sub_n(w1, w4, w1, m);
        }
        rshift::<1>(w1, w1, m);
        sub_ext(w4, w4, m, w0, 2 * k);
        sub_n(w4, w4, w1, m);
        rshift::<2>(w4, w4, m);
        *t.add(w) = lshift::<4>(t, w6, w);
        sub_ext(w4, w4, m, t, w + 1);
        if negative_1 {
            add_n(w3, w2, w3, m);
        } else {
            sub_n(w3, w2, w3, m);
        }
        rshift::<1>(w3, w3, m);
        sub_n(w2, w2, w3, m);
        mul_1(t, w2, m, 65);
        sub_n(w5, w5, t, m);
        sub_ext(w2, w2, m, w6, w);
        sub_ext(w2, w2, m, w0, 2 * k);
        mul_1(t, w2, m, 45);
        add_n(w5, w5, t, m);
        rshift::<1>(w5, w5, m);
        sub_n(w4, w4, w2, m);
        divexact::<3>(w4, m);
        sub_n(w2, w2, w4, m);
        sub_n(w1, w5, w1, m);
        lshift::<3>(t, w3, m);
        sub_n(w5, w5, t, m);
        divexact::<3>(w5, m);
        divexact::<3>(w5, m);
        sub_n(w3, w3, w5, m);
        divexact::<15>(w1, m);
        add_n(w1, w1, w5, m);
        rshift::<1>(w1, w1, m);
        sub_n(w5, w5, w1, m);

        // r1 at limb k, over r0 and r2, up to the top of r2 at limb 4k.
        let carry = add_n(r.add(k), r.add(k), w1, m);
        add_1(r.add(3 * k + 1), k, u64::from(carry));
        // r3, r4 and r5 at limbs 3k, 4k and 5k: limb 4k on is written, not added.
        let carry = add_n(r.add(3 * k), r.add(3 * k), w3, k);
        add_1(w3.add(k), k + 1, *r.add(4 * k) + u64::from(carry));
        let carry = add_n(r.add(4 * k), w3.add(k), w4, k);
        add_1(w4.add(k), k + 1, *w3.add(2 * k) + u64::from(carry));
        let carry = add_n(r.add(5 * k), w4.add(k), w5, k);
        add_1(w5.add(k), k + 1, *w4.add(2 * k) + u64::from(carry));
        if w > k + 1 {
            let carry = add_n(w6, w6, w5.add(k), k + 1);
            add_1(w6.add(k + 1), w - k - 1, u64::from(carry));
        } else {
            add_n(w6, w6, w5.add(k), w);
        }
    }
}

// Shared steps

/// Sets the `2k + 1` limbs at `r` to the product of the values of `k + 1` limbs at `a` and `b`
/// whose top limbs are at most 1, as for the values at -1: when both top limbs are 0, a product
/// of `k` limbs each and a zero limb above it.
///
/// # Safety
///
/// The `k + 1` limbs at `a` and `b` must be valid for reads, the `2k + 2` at `r` for writes, and
/// the `mul_itch(k + 1, k + 1)` at `s` for reads and writes, all apart.
unsafe fn product_of_values(r: *mut u64, a: *const u64, b: *const u64, k: usize, s: *mut u64) {
    // SAFETY: as the caller guarantees.
    unsafe {
        if *a.add(k) | *b.add(k) == 0 {
            mul(r, a, k, b, k, s);
            *r.add(2 * k) = 0;
        } else {
            mul(r, a, k + 1, b, k + 1, s);
        }
    }
}

/// Sets the `n` limbs at `r` to the `n` limbs at `a` shifted up by `C` bits, and returns the bits
/// shifted out of the top, at the bottom of a limb. From the top down, so `r` may be `a`.
///
/// # Safety
///
/// `n >= 1`, `1 <= C < 64`; the limbs must be valid, and `r` equal to `a` or apart from it.
#[inline]
unsafe fn lshift<const C: u32>(r: *mut u64, a: *const u64, n: usize) -> u64 {
    // SAFETY: as the caller guarantees; limb `i` of `r` is written after limbs `i` and `i - 1` of
    // `a` are read, and before any limb below it.
    unsafe {
        let out = *a.add(n - 1) >> (64 - C);
        for i in (1..n).rev() {
            *r.add(i) = (*a.add(i) << C) | (*a.add(i - 1) >> (64 - C));
        }
        *r = *a << C;
        out
    }
}

/// Sets the `n` limbs at `r` to the `n` limbs at `a` shifted down by `C` bits, the bits shifted
/// out at the bottom being zero. From the bottom up, so `r` may be `a`.
///
/// # Safety
///
/// `n >= 1`, `1 <= C < 64`; the limbs must be valid, and `r` equal to `a` or apart from it.
#[inline]
unsafe fn rshift<const C: u32>(r: *mut u64, a: *const u64, n: usize) {
    // SAFETY: as the caller guarantees; limb `i` of `r` is written after limbs `i` and `i + 1` of
    // `a` are read, and before any limb above it.
    unsafe {
        debug_assert!(*a & ((1 << C) - 1) == 0, "an inexact halving");
        for i in 0..n - 1 {
            *r.add(i) = (*a.add(i) >> C) | (*a.add(i + 1) << (64 - C));
        }
        *r.add(n - 1) = *a.add(n - 1) >> C;
    }
}

/// Divides the `n` limbs at `r` in place by `D`, a divisor of `B - 1`, which divides them
/// exactly; in two's complement too, for a negative value whose quotient fits.
///
/// With `D q = a` and `D d = B - 1` for `d = (B - 1) / D`, `q (B - 1) = a d`, so `q = q B - a d`:
/// limb `i` of `q` is limb `i - 1` of `q` less limb `i` of `a d` and the borrow from below, one
/// borrow chain after a multiplication by one limb.
///
/// # Safety
///
/// `n >= 1`; the limbs must be valid for reads and writes.
#[inline]
unsafe fn divexact<const D: u64>(r: *mut u64, n: usize) {
    // SAFETY: as the caller guarantees; `mul_1` may work in place.
    unsafe {
        mul_1(r, r, n, u64::MAX / D);
        divexact_finish(r, n);
    }
}
