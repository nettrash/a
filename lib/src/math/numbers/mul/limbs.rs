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
//! that no product is longer than its halves allow.
//!
//! From [`TOOM3_MUL`] limbs in the shorter operand ([`TOOM3_SQR`] for a square) Toom-3 takes over,
//! five products of thirds, and from [`TOOM4_MUL`] ([`TOOM4_SQR`]) Toom-4, seven products of
//! quarters (see [`toom`]). Each needs operands of about the same length: Toom-4 a
//! shorter one of more than three quarters of the longer one, Toom-3 of more than two thirds.
//! Below that, down to a quarter, the longer operand is cut in four parts and the shorter one in
//! two (Toom-4x2, five products); a shorter operand shorter still, or of at most half the longer
//! one below Toom-3, multiplies the longer one in pieces, each about twice its own length in the
//! range of Toom's methods and its own length below.

#[cfg(target_endian = "little")]
pub(in super::super) use super::super::add::{add_limbs as add_n, sub_limbs as sub_n};
use super::kernels::{mul_basecase, sqr_basecase};
use super::toom;

/// From this many limbs in the shorter operand, a product is computed by Karatsuba's method:
/// measured on an Apple M4 Max, a balanced product of 24 limbs took 529 cycles so, against 570 by
/// the schoolbook method, and one of 20 limbs 411 against 404.
pub(super) const KARATSUBA_MUL: usize = 24;

/// From this many limbs, a square is computed by Karatsuba's method: the schoolbook square needs
/// only about half the limb products, and measured on an Apple M4 Max it was faster up to 48 limbs
/// (1331 cycles against 1359), and slower from 56 (1780 against 1694).
pub(super) const KARATSUBA_SQR: usize = 56;

/// From this many limbs in the shorter operand, a product of operands that Toom-3 can cut is
/// computed by it: measured on an Apple M4 Max, Toom-3 over products of thirds by Karatsuba's
/// method took 7954 cycles for 120 limbs against 8108 by Karatsuba's method alone, and 9943
/// against 10,620 for 136; at 112 limbs, 7339 against 7195.
pub(super) const TOOM3_MUL: usize = 120;

/// From this many limbs in the shorter operand, a product of operands that Toom-4 can cut is
/// computed by it: measured on an Apple M4 Max, 20,663 cycles for 224 limbs against 21,820 by
/// Toom-3, 70,030 against 74,210 for 512, and 16,678 against 16,602 for 192. Where the parts of
/// one land just above a threshold and those of the other just below, either can be a few percent
/// ahead: 384 limbs took 49,053 cycles by Toom-4 and 47,798 by Toom-3.
pub(super) const TOOM4_MUL: usize = 224;

/// From this many limbs, a square is computed by Toom-3: measured on an Apple M4 Max, 9860 cycles
/// for 168 limbs against 10,203 by Karatsuba's method, and 9494 against 9373 for 160.
pub(super) const TOOM3_SQR: usize = 168;

/// From this many limbs, a square is computed by Toom-4: measured on an Apple M4 Max, 15,419
/// cycles for 224 limbs against 15,486 by Toom-3, 26,214 against 29,286 for 320, and 14,009
/// against 13,847 for 208.
pub(super) const TOOM4_SQR: usize = 224;

/// The method of a product, or of a square.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Method {
    /// The schoolbook method, in the kernels.
    Schoolbook,
    /// The longer operand in pieces, each multiplied by the shorter one.
    Pieces,
    /// Karatsuba's method, three products of halves.
    Karatsuba,
    /// Toom-3, five products of thirds.
    Toom3,
    /// Toom-4, seven products of quarters.
    Toom4,
    /// Toom-4x2, the longer operand in four parts and the shorter one in two: five products.
    Toom42,
}

/// The method [`mul`] takes for operands of `nx >= ny` limbs.
#[inline]
fn method(nx: usize, ny: usize) -> Method {
    if ny < KARATSUBA_MUL {
        Method::Schoolbook
    } else if ny < TOOM3_MUL {
        if ny <= nx.div_ceil(2) {
            Method::Pieces
        } else {
            Method::Karatsuba
        }
    } else if ny >= TOOM4_MUL && toom::toom44_fits(nx, ny) {
        Method::Toom4
    } else if toom::toom33_fits(nx, ny) {
        Method::Toom3
    } else if toom::toom42_fits(nx, ny) {
        Method::Toom42
    } else if ny <= nx.div_ceil(2) {
        Method::Pieces
    } else {
        Method::Karatsuba
    }
}

/// The method [`sqr`] takes for an operand of `n` limbs.
#[inline]
fn sqr_method(n: usize) -> Method {
    if n < KARATSUBA_SQR {
        Method::Schoolbook
    } else if n < TOOM3_SQR {
        Method::Karatsuba
    } else if n < TOOM4_SQR {
        Method::Toom3
    } else {
        Method::Toom4
    }
}

/// The limbs of scratch space that [`mul`] needs for operands of `nx >= ny` limbs.
pub(in super::super) fn mul_itch(nx: usize, ny: usize) -> usize {
    match method(nx, ny) {
        Method::Schoolbook => 0,
        Method::Pieces => {
            let (p, full, last) = pieces(nx, ny);
            let itch = |len: usize| mul_itch(len.max(ny), len.min(ny));
            let rest = if full > 0 { itch(p) } else { 0 }.max(itch(last));
            let t = if full > 0 { p.max(last) } else { last };
            itch(p).max(t + ny + rest)
        }
        Method::Karatsuba => {
            let n = nx.div_ceil(2);
            let (sx, ty) = (nx - n, ny - n);
            2 * n + mul_itch(n, n).max(mul_itch(sx, ty))
        }
        Method::Toom3 => toom::toom33_itch(nx, ny),
        Method::Toom4 => toom::toom44_itch(nx, ny),
        Method::Toom42 => toom::toom42_itch(nx, ny),
    }
}

/// The limbs of scratch space that [`sqr`] needs for an operand of `n` limbs.
pub(in super::super) fn sqr_itch(n: usize) -> usize {
    match sqr_method(n) {
        Method::Schoolbook | Method::Pieces | Method::Toom42 => 0,
        Method::Karatsuba => {
            let h = n.div_ceil(2);
            2 * h + sqr_itch(h).max(sqr_itch(n - h))
        }
        Method::Toom3 => toom::toom3_sqr_itch(n),
        Method::Toom4 => toom::toom4_sqr_itch(n),
    }
}

/// Sets the `nx + ny` limbs at `r` to the product of the `nx` limbs at `x` and the `ny` limbs at
/// `y`.
///
/// # Safety
///
/// `nx >= ny >= 1`; the limbs of `x` and `y` must be valid for reads, those of `r` for writes,
/// and the [`mul_itch`]`(nx, ny)` limbs at `s` for reads and writes, all apart from each other.
pub(in super::super) unsafe fn mul(
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
        match method(nx, ny) {
            Method::Schoolbook => mul_basecase(r, x, nx, y, ny),
            Method::Pieces => mul_pieces(r, x, nx, y, ny, s),
            Method::Karatsuba => karatsuba(r, x, nx, y, ny, s),
            Method::Toom3 => toom::toom33(r, x, nx, y, ny, s),
            Method::Toom4 => toom::toom44(r, x, nx, y, ny, s),
            Method::Toom42 => toom::toom42(r, x, nx, y, ny, s),
        }
    }
}

/// Sets the `2n` limbs at `r` to the square of the `n` limbs at `x`.
///
/// # Safety
///
/// `n >= 1`; the limbs of `x` must be valid for reads, those of `r` for writes, and the
/// [`sqr_itch`]`(n)` limbs at `s` for reads and writes, all apart from each other.
pub(in super::super) unsafe fn sqr(r: *mut u64, x: *const u64, n: usize, s: *mut u64) {
    debug_assert!(n >= 1);
    // SAFETY: forwarded.
    unsafe {
        match sqr_method(n) {
            Method::Schoolbook | Method::Pieces | Method::Toom42 => sqr_basecase(r, x, n),
            Method::Karatsuba => karatsuba_sqr(r, x, n, s),
            Method::Toom3 => toom::toom3_sqr(r, x, n, s),
            Method::Toom4 => toom::toom4_sqr(r, x, n, s),
        }
    }
}

/// How [`mul_pieces`] cuts the longer operand: into pieces of `p` limbs, the first one and `full`
/// more, then a last one of `last` limbs. Pieces are of `ny` limbs, or of `2ny` in the range of
/// Toom's methods, where Toom-4x2 multiplies a piece; the last one, what is left, is at least half
/// of `ny` and less than half of it more than a piece.
fn pieces(nx: usize, ny: usize) -> (usize, usize, usize) {
    let half = ny / 2;
    let p = if ny >= TOOM3_MUL && nx >= 2 * ny + half {
        2 * ny
    } else {
        ny
    };
    let left = nx - p;
    let full = if left >= p + half {
        (left - half) / p
    } else {
        0
    };
    (p, full, left - full * p)
}

/// Sets the `nx + ny` limbs at `r` to the product of the `nx` limbs at `x` and the `ny` at `y`,
/// either of which may be the longer: [`mul`] with the longer one first.
///
/// # Safety
///
/// As for [`mul`], with `nx, ny >= 1` in either order, and the scratch that [`mul_itch`] gives for
/// the longer one first.
unsafe fn mul_any(r: *mut u64, x: *const u64, nx: usize, y: *const u64, ny: usize, s: *mut u64) {
    // SAFETY: forwarded.
    unsafe {
        if nx >= ny {
            mul(r, x, nx, y, ny, s);
        } else {
            mul(r, y, ny, x, nx, s);
        }
    }
}

/// [`mul`] for a shorter operand of at most half the longer one's limbs: the longer one is
/// multiplied in pieces (see [`pieces`]), each product added in at the position of its piece.
///
/// # Safety
///
/// As for [`mul`], with `ny <= nx.div_ceil(2)`.
unsafe fn mul_pieces(r: *mut u64, x: *const u64, nx: usize, y: *const u64, ny: usize, s: *mut u64) {
    let (p, full, last) = pieces(nx, ny);
    // SAFETY: every piece lies in `x`, every product of a piece but the first in the scratch, and
    // every position of `r` written is below `nx + ny`; the scratch holds the product of the
    // longest of the other pieces and what multiplying a piece needs after it.
    unsafe {
        mul_any(r, x, p, y, ny, s);
        let longest = if full > 0 { p.max(last) } else { last };
        let (t, rest) = (s, s.add(longest + ny));
        let mut done = p;
        for i in 0..=full {
            let len = if i < full { p } else { last };
            mul_any(t, x.add(done), len, y, ny, rest);
            // The top `ny` limbs of the product so far are at `done`; above them nothing yet.
            let carry = add_n(r.add(done), r.add(done), t, ny);
            core::ptr::copy_nonoverlapping(t.add(ny), r.add(done + ny), len);
            let out = add_1(r.add(done + ny), len, u64::from(carry));
            debug_assert!(out == 0, "a product of pieces carried out of its limbs");
            done += len;
        }
        debug_assert!(done == nx);
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
/// As for [`sqr`], with `n >= 2`.
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
pub(super) unsafe fn abs_diff(
    r: *mut u64,
    a: *const u64,
    na: usize,
    b: *const u64,
    nb: usize,
) -> bool {
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
pub(in super::super) unsafe fn cmp_n(
    a: *const u64,
    b: *const u64,
    n: usize,
) -> core::cmp::Ordering {
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
pub(super) unsafe fn add_ext(
    r: *mut u64,
    a: *const u64,
    na: usize,
    b: *const u64,
    nb: usize,
) -> bool {
    // SAFETY: as the caller guarantees.
    unsafe {
        let carry = add_n(r, a, b, nb);
        if r.cast_const() != a {
            core::ptr::copy_nonoverlapping(a.add(nb), r.add(nb), na - nb);
        }
        add_1(r.add(nb), na - nb, u64::from(carry)) != 0
    }
}

/// Sets the `na` limbs at `r` to `a - b` for the `na` limbs at `a` and the `nb <= na` limbs at
/// `b`, and returns the borrow out.
///
/// # Safety
///
/// The limbs of `a` and `b` must be valid for reads and those of `r` for writes; `r` may be `a`.
pub(super) unsafe fn sub_ext(
    r: *mut u64,
    a: *const u64,
    na: usize,
    b: *const u64,
    nb: usize,
) -> bool {
    // SAFETY: as the caller guarantees.
    unsafe {
        let borrow = sub_n(r, a, b, nb);
        if r.cast_const() != a {
            core::ptr::copy_nonoverlapping(a.add(nb), r.add(nb), na - nb);
        }
        sub_1(r.add(nb), na - nb, u64::from(borrow)) != 0
    }
}

/// Adds `c` to the `n` limbs at `r`, and returns what carries out of them.
///
/// # Safety
///
/// The limbs must be valid for reads and writes.
pub(in super::super) unsafe fn add_1(r: *mut u64, n: usize, mut c: u64) -> u64 {
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
pub(in super::super) unsafe fn sub_1(r: *mut u64, n: usize, mut c: u64) -> u64 {
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
pub(in super::super) unsafe fn add_n(r: *mut u64, a: *const u64, b: *const u64, n: usize) -> bool {
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
pub(in super::super) unsafe fn sub_n(r: *mut u64, a: *const u64, b: *const u64, n: usize) -> bool {
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
