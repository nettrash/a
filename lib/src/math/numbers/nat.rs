//! Arithmetic on magnitudes held as native limbs, the least significant first, for the algorithms
//! of the crate outside `numbers` that keep their integers as limbs (the exact eliminations of
//! matrices): products by the method their lengths call for, exact quotients by Hensel's method,
//! and sums and differences, over slices.

use super::div::limbs::{
    bdiv_q, binvert_limb, div_3by2 as div_3by2_limbs, div_q as div_q_limbs, div_qr as div_qr_limbs,
    div_qr_1, div_qr_2 as div_qr_2_limbs, invert_3by2 as invert_3by2_limbs, q_itch, qr_itch,
};
pub(crate) use super::div::limbs::{div_2by1, invert_limb};
use super::mul::basecase;
use super::mul::limbs::{add_n, mul as mul_limbs, mul_itch, sqr as sqr_limbs, sqr_itch, sub_n};
use super::root::{
    root_itch, root_limbs, sqrt_rem_u128 as sqrt_u128, sqrt_rem_wide as sqrt_wide, sqrtrem,
    sqrtrem_itch,
};
use super::scratch::Scratch;

/// `r = x·y` in the vector `r`, of the operands' lengths together, for `x` and `y` of at least
/// one limb, with the working memory of the longer methods in `scratch`, which grows as they
/// need: below [`basecase::SCHOOLBOOK`] limbs in the shorter operand straight to the schoolbook
/// kernel, beyond by the method the lengths call for.
///
/// # Panics
///
/// Panics if an operand is empty.
#[inline]
pub(crate) fn mul_into(r: &mut Vec<u64>, x: &[u64], y: &[u64], scratch: &mut Vec<u64>) {
    let (x, y) = if x.len() >= y.len() { (x, y) } else { (y, x) };
    let (nx, ny) = (x.len(), y.len());
    assert!(ny >= 1);
    r.clear();
    r.reserve(nx + ny);
    if ny < basecase::SCHOOLBOOK {
        // SAFETY: `nx >= ny >= 1`; the vector has room for the `nx + ny` limbs, which the kernel
        // writes before it reads any of them, and is apart from the operands, as a unique
        // borrow; they are all written when the length is set.
        unsafe {
            basecase::mul_unbalanced(r.as_mut_ptr(), x.as_ptr(), nx, y.as_ptr(), ny);
            r.set_len(nx + ny);
        }
        return;
    }
    let itch = mul_itch(nx, ny);
    if scratch.len() < itch {
        scratch.resize(itch, 0);
    }
    // SAFETY: as above, and the scratch holds the `itch` limbs the method of these lengths uses.
    unsafe {
        mul_limbs(
            r.as_mut_ptr(),
            x.as_ptr(),
            nx,
            y.as_ptr(),
            ny,
            Scratch::new(scratch.as_mut_ptr(), itch),
        );
        r.set_len(nx + ny);
    }
}

/// `r = x²` in the vector `r`, of twice the operand's length, for `x` of at least one limb, with the
/// working memory of the longer methods in `scratch`, which grows as they need.
///
/// # Panics
///
/// Panics if `x` is empty.
#[inline]
pub(crate) fn sqr_into(r: &mut Vec<u64>, x: &[u64], scratch: &mut Vec<u64>) {
    let n = x.len();
    assert!(n >= 1);
    r.clear();
    r.reserve(2 * n);
    let itch = sqr_itch(n);
    if scratch.len() < itch {
        scratch.resize(itch, 0);
    }
    // SAFETY: `n >= 1`; the vector has room for the `2n` limbs, which the method writes before it
    // reads any of them, apart from the operand as a unique borrow, and the scratch holds the
    // `itch` limbs it uses; they are all written when the length is set.
    unsafe {
        sqr_limbs(
            r.as_mut_ptr(),
            x.as_ptr(),
            n,
            Scratch::new(scratch.as_mut_ptr(), itch),
        );
        r.set_len(2 * n);
    }
}

/// `r = x²` in the slice `r` of twice the operand's length, for `x` of at least one limb, with the
/// working memory of the longer methods in `scratch`, which grows as they need: [`sqr_into`] into
/// limbs the caller holds (on the stack, say).
///
/// # Panics
///
/// Panics if `x` is empty or `r` is not twice as long.
#[inline]
pub(crate) fn sqr_to(r: &mut [u64], x: &[u64], scratch: &mut Vec<u64>) {
    let n = x.len();
    assert!(n >= 1 && r.len() == 2 * n);
    let itch = sqr_itch(n);
    if scratch.len() < itch {
        scratch.resize(itch, 0);
    }
    // SAFETY: `n >= 1`; `r` holds the `2n` limbs, which the method writes before it reads any of
    // them, apart from the operand as a unique borrow, and the scratch holds the `itch` limbs it
    // uses.
    unsafe {
        sqr_limbs(
            r.as_mut_ptr(),
            x.as_ptr(),
            n,
            Scratch::new(scratch.as_mut_ptr(), itch),
        );
    }
}

/// `r = x·y` in the slice `r` of the operands' lengths together, for `x` and `y` of at least one
/// limb, with the working memory of the longer methods in `scratch`: [`mul_into`] into limbs the
/// caller holds.
///
/// # Panics
///
/// Panics if an operand is empty or `r` is not as long as both.
#[inline]
pub(crate) fn mul_to(r: &mut [u64], x: &[u64], y: &[u64], scratch: &mut Vec<u64>) {
    let (x, y) = if x.len() >= y.len() { (x, y) } else { (y, x) };
    let (nx, ny) = (x.len(), y.len());
    assert!(ny >= 1 && r.len() == nx + ny);
    if ny < basecase::SCHOOLBOOK {
        // SAFETY: `nx >= ny >= 1`; `r` holds the `nx + ny` limbs, which the kernel writes before
        // it reads any of them, apart from the operands as a unique borrow.
        unsafe { basecase::mul_unbalanced(r.as_mut_ptr(), x.as_ptr(), nx, y.as_ptr(), ny) };
        return;
    }
    let itch = mul_itch(nx, ny);
    if scratch.len() < itch {
        scratch.resize(itch, 0);
    }
    // SAFETY: as above, and the scratch holds the `itch` limbs the method of these lengths uses.
    unsafe {
        mul_limbs(
            r.as_mut_ptr(),
            x.as_ptr(),
            nx,
            y.as_ptr(),
            ny,
            Scratch::new(scratch.as_mut_ptr(), itch),
        );
    }
}

/// `r += x·y` for `x` as long as `r`: the limb that carries out.
///
/// # Panics
///
/// Panics if `x` is empty or not as long as `r`.
#[inline]
pub(crate) fn addmul_1(r: &mut [u64], x: &[u64], y: u64) -> u64 {
    assert!(!x.is_empty() && x.len() == r.len());
    // SAFETY: `n >= 1` limbs of each, `r` a unique borrow apart from `x`.
    unsafe { basecase::addmul_1(r.as_mut_ptr(), x.as_ptr(), x.len(), y) }
}

/// `r += a` for `a` no longer than `r`: the carry out of `r`.
#[inline]
pub(crate) fn add_assign(r: &mut [u64], a: &[u64]) -> bool {
    let n = a.len();
    assert!(n <= r.len());
    if n == 0 {
        return false;
    }
    // SAFETY: the `n` limbs of `r` and `a` are valid, and the output is the first input.
    let mut carry = unsafe { add_n(r.as_mut_ptr(), r.as_ptr(), a.as_ptr(), n) };
    for x in &mut r[n..] {
        if !carry {
            break;
        }
        (*x, carry) = x.overflowing_add(1);
    }
    carry
}

/// `r -= a` for `a` no longer than `r`: the borrow out of `r`.
#[inline]
pub(crate) fn sub_assign(r: &mut [u64], a: &[u64]) -> bool {
    let n = a.len();
    assert!(n <= r.len());
    if n == 0 {
        return false;
    }
    // SAFETY: as for `add_assign`.
    let mut borrow = unsafe { sub_n(r.as_mut_ptr(), r.as_ptr(), a.as_ptr(), n) };
    for x in &mut r[n..] {
        if !borrow {
            break;
        }
        (*x, borrow) = x.overflowing_sub(1);
    }
    borrow
}

/// `r = a + b` for `r`, `a` and `b` of the same length: the carry out.
#[inline]
pub(crate) fn add_n_into(r: &mut [u64], a: &[u64], b: &[u64]) -> bool {
    let n = r.len();
    assert!(a.len() == n && b.len() == n);
    if n == 0 {
        return false;
    }
    // SAFETY: the `n` limbs of each are valid, and the output, a unique borrow, is apart from
    // the inputs.
    unsafe { add_n(r.as_mut_ptr(), a.as_ptr(), b.as_ptr(), n) }
}

/// `r = a - b` for `r`, `a` and `b` of the same length: the borrow out.
#[inline]
pub(crate) fn sub_n_into(r: &mut [u64], a: &[u64], b: &[u64]) -> bool {
    let n = r.len();
    assert!(a.len() == n && b.len() == n);
    if n == 0 {
        return false;
    }
    // SAFETY: as for `add_n_into`.
    unsafe { sub_n(r.as_mut_ptr(), a.as_ptr(), b.as_ptr(), n) }
}

/// `r = a - r` for `r` and `a` of the same length: the borrow out.
#[inline]
pub(crate) fn sub_from(r: &mut [u64], a: &[u64]) -> bool {
    let n = a.len();
    assert!(n == r.len());
    if n == 0 {
        return false;
    }
    // SAFETY: the `n` limbs of `r` and `a` are valid, and the output is the second input.
    unsafe { sub_n(r.as_mut_ptr(), a.as_ptr(), r.as_ptr(), n) }
}

/// The inverse of the odd limb `d` modulo `2^64`.
#[inline]
pub(crate) fn binvert(d: u64) -> u64 {
    binvert_limb(d)
}

/// `q = n / d mod B^qn` in the vector `q`, `B = 2^64`, for the odd `d` with its inverse `v` modulo
/// `B` ([`binvert`]) and `n` of at least `qn` limbs, by Hensel's method from the low end: the
/// quotient when `d` divides `n` and the quotient has at most `qn` limbs. Changes the low `qn`
/// limbs of `n`. About `qn·m - m²/2` limb products for `m = min(qn, d.len())`.
///
/// # Panics
///
/// Panics if `d` is empty or even, or `n` is shorter than `qn`.
pub(crate) fn bdiv_into(q: &mut Vec<u64>, qn: usize, n: &mut [u64], d: &[u64], v: u64) {
    assert!(!d.is_empty() && d[0] & 1 == 1 && n.len() >= qn);
    q.clear();
    if qn == 0 {
        return;
    }
    q.reserve(qn);
    // SAFETY: the vector has room for `qn` limbs, which are written one by one before any is
    // read, apart from `n` and `d` (unique and shared borrows); the first `qn` limbs of `n` are
    // valid for reads and writes, and the `min(d.len(), qn) >= 1` of `d` for reads.
    unsafe {
        bdiv_q(
            q.as_mut_ptr(),
            n.as_mut_ptr(),
            qn,
            d.as_ptr(),
            d.len().min(qn),
            v,
        );
        q.set_len(qn);
    }
}

/// `x = d⁻¹ mod B^l`, `B = 2^64`, in the vector `x`, for the odd `d` with its inverse `v` modulo
/// `B`: by Newton's iteration, `x ← x - x·(d·x - 1)`, which doubles the limbs that are right; `t`
/// and `scratch` are working memory. About three products of `l` limbs.
///
/// # Panics
///
/// Panics if `d` is empty or even, or `l` is zero.
pub(crate) fn inverse_into(
    x: &mut Vec<u64>,
    d: &[u64],
    v: u64,
    l: usize,
    t: &mut Vec<u64>,
    scratch: &mut Vec<u64>,
) {
    assert!(!d.is_empty() && d[0] & 1 == 1 && l >= 1);
    x.clear();
    x.push(v);
    let mut h = Vec::new();
    while x.len() < l {
        let p = x.len();
        let q = (2 * p).min(l);
        // `d·x = 1 + B^p·h mod B^q`.
        mul_into(t, &d[..d.len().min(q)], x, scratch);
        t.resize(q.max(t.len()), 0);
        h.clear();
        h.extend_from_slice(&t[p..q]);
        // The limbs `p..q` of the next `x`: `-(x·h) mod B^(q - p)`.
        let k = q - p;
        mul_into(t, &x[..p.min(k)], &h, scratch);
        t.truncate(k);
        t.resize(k, 0);
        // `-t mod B^k`: the complement plus one.
        let mut carry = true;
        for limb in t.iter_mut() {
            let (y, c) = (!*limb).overflowing_add(carry as u64);
            *limb = y;
            carry = c;
        }
        x.extend_from_slice(t);
    }
}

/// The reciprocal `floor((B^3 - 1) / (d1 B + d0)) - B` of the normalized two limbs `d1 B + d0`
/// (the top bit of `d1` set), for [`div_qr_2`].
#[inline]
pub(crate) fn invert_3by2(d1: u64, d0: u64) -> u64 {
    debug_assert!(d1 >> 63 == 1);
    invert_3by2_limbs(d1, d0)
}

/// `(u2 B^2 + u1 B + u0) / (d1 B + d0)` for the normalized divisor with its reciprocal `v` of
/// [`invert_3by2`]: the quotient limb and the remainder's two limbs, the high one first.
/// `u2 B + u1` must be below the divisor.
#[inline(always)]
pub(crate) fn div_3by2(u2: u64, u1: u64, u0: u64, d1: u64, d0: u64, v: u64) -> (u64, u64, u64) {
    debug_assert!((u2, u1) < (d1, d0));
    div_3by2_limbs(u2, u1, u0, d1, d0, v)
}

/// `q = u / (d1 B + d0)` for the normalized divisor `d1 B + d0` with its reciprocal `v` of
/// [`invert_3by2`], and the remainder's two limbs, the high one first: `q` of `u.len() - 2` limbs.
///
/// # Panics
///
/// Panics if `u` has fewer than two limbs or `q` is not two limbs shorter. The top two limbs of
/// `u` must be below the divisor.
#[inline]
pub(crate) fn div_qr_2(q: &mut [u64], u: &[u64], d1: u64, d0: u64, v: u64) -> (u64, u64) {
    let n = u.len();
    assert!(n >= 2 && q.len() == n - 2 && d1 >> 63 == 1);
    debug_assert!((u[n - 1], u[n - 2]) < (d1, d0));
    // SAFETY: `n >= 2` limbs of `u` for reads, `n - 2` of `q` for writes, apart as borrows.
    unsafe { div_qr_2_limbs(q.as_mut_ptr(), u.as_ptr(), n, d1, d0, v) }
}

/// The square root of `a` rounded down and the remainder `a - s^2`, in registers.
#[inline]
pub(crate) fn sqrt_rem_u128(a: u128) -> (u64, u128) {
    sqrt_u128(a)
}

/// The square root rounded down and the remainder (three limbs) of the four limbs `a`, the least
/// significant first, of at least `2^128`, in registers.
#[inline]
pub(crate) fn sqrt_rem_wide(a: [u64; 4]) -> (u128, [u64; 3]) {
    assert!(a[3] != 0 || a[2] != 0);
    sqrt_wide::<true>(a)
}

/// `q = n / d` and the remainder in the low `d.len()` limbs of `n`, for the normalized divisor
/// `d` (the top bit of its top limb set) and `n` at least as long: `q` takes the `n.len() -
/// d.len()` limbs below the quotient's top bit, which is returned. By the method the lengths call
/// for: one or two limbs in registers, the schoolbook method, divide and conquer, or Newton's
/// method for the longest; `scratch` grows as it needs.
///
/// # Panics
///
/// Panics if `d` is empty or not normalized, `n` is shorter than `d`, or `q` is not `n.len() -
/// d.len()` limbs long.
pub(crate) fn div_qr(q: &mut [u64], n: &mut [u64], d: &[u64], scratch: &mut Vec<u64>) -> bool {
    let (nn, dn) = (n.len(), d.len());
    assert!(dn >= 1 && nn >= dn && q.len() == nn - dn && d[dn - 1] >> 63 == 1);
    match dn {
        1 => {
            let d0 = d[0];
            let qh = n[nn - 1] >= d0;
            let top = if qh { n[nn - 1] - d0 } else { n[nn - 1] };
            let v = invert_limb(d0);
            // SAFETY: `nn - 1` limbs of `n` for reads and of `q` for writes, apart as borrows;
            // `top < d0`.
            let r = unsafe { div_qr_1::<true>(q.as_mut_ptr(), n.as_ptr(), nn - 1, top, d0, v) };
            n[0] = r;
            qh
        }
        2 => {
            let (d1, d0) = (d[1], d[0]);
            let qh = (n[nn - 1], n[nn - 2]) >= (d1, d0);
            if qh {
                let (lo, b) = n[nn - 2].overflowing_sub(d0);
                n[nn - 2] = lo;
                n[nn - 1] = n[nn - 1] - d1 - u64::from(b);
            }
            let v = invert_3by2_limbs(d1, d0);
            // SAFETY: as in `div_qr_2`; the top two limbs are below the divisor.
            let (r1, r0) = unsafe { div_qr_2_limbs(q.as_mut_ptr(), n.as_ptr(), nn, d1, d0, v) };
            (n[0], n[1]) = (r0, r1);
            qh
        }
        _ => {
            let itch = qr_itch(nn, dn);
            if scratch.len() < itch {
                scratch.resize(itch, 0);
            }
            let v = invert_3by2_limbs(d[dn - 1], d[dn - 2]);
            // SAFETY: `nn >= dn > 2`, the divisor normalized; the limbs of `n` for reads and writes,
            // of `d` for reads and the `nn - dn` of `q` for writes, apart as borrows, and the
            // scratch holds the `qr_itch` limbs the method needs.
            unsafe {
                div_qr_limbs(
                    q.as_mut_ptr(),
                    n.as_mut_ptr(),
                    nn,
                    d.as_ptr(),
                    dn,
                    v,
                    Scratch::new(scratch.as_mut_ptr(), itch),
                )
            }
        }
    }
}

/// `q = n / d` alone, as [`div_qr`] computes it but without the remainder, for the normalized
/// divisor `d` of at least three limbs and `n` at least as long, whose limbs it clobbers: its top
/// limbs by the schoolbook method or divide and conquer, its last ones from a divisor that drops a
/// limb at each step (about half the products). Returns the quotient's top bit, as `div_qr`, or
/// `None`, rarely, when the shrinking divisor gives up: the quotient is then to be computed by
/// `div_qr` from the dividend again. `scratch` grows as it needs.
///
/// # Panics
///
/// Panics if `d` has fewer than three limbs or is not normalized, `n` is shorter than `d`, or `q`
/// is not `n.len() - d.len()` limbs long.
pub(crate) fn div_q(
    q: &mut [u64],
    n: &mut [u64],
    d: &[u64],
    scratch: &mut Vec<u64>,
) -> Option<bool> {
    let (nn, dn) = (n.len(), d.len());
    assert!(dn >= 3 && nn >= dn && q.len() == nn - dn && d[dn - 1] >> 63 == 1);
    let itch = q_itch(nn, dn);
    if scratch.len() < itch {
        scratch.resize(itch, 0);
    }
    let v = invert_3by2_limbs(d[dn - 1], d[dn - 2]);
    // SAFETY: `nn >= dn > 2`, the divisor normalized; the limbs of `n` for reads and writes, of `d`
    // for reads and the `nn - dn` of `q` for writes, apart as borrows, and the scratch holds the
    // `q_itch` limbs the method needs.
    unsafe {
        div_q_limbs(
            q.as_mut_ptr(),
            n.as_mut_ptr(),
            nn,
            d.as_ptr(),
            dn,
            v,
            Scratch::new(scratch.as_mut_ptr(), itch),
        )
    }
}

/// `s = floor(sqrt(a))` in the vector `s`, of `(a.len() + 1) / 2` limbs, for `a` of at least three
/// limbs with its top limb not zero, by Zimmermann's Karatsuba square root; with `r`, the
/// remainder `a - s^2` too, without zero limbs at the top. `scratch` grows as it needs.
///
/// # Panics
///
/// Panics if `a` has fewer than three limbs or its top limb is zero.
pub(crate) fn sqrt_rem(
    s: &mut Vec<u64>,
    r: Option<&mut Vec<u64>>,
    a: &[u64],
    scratch: &mut Vec<u64>,
) {
    let an = a.len();
    assert!(an >= 3 && a[an - 1] != 0);
    let tn = an.div_ceil(2);
    let itch = sqrtrem_itch(an);
    if scratch.len() < itch {
        scratch.resize(itch, 0);
    }
    s.clear();
    s.resize(tn + 1, 0);
    // SAFETY: `an >= 3` limbs of `a` for reads, the root's `tn` and one more for writes, the
    // remainder's at most `tn + 1` for writes, and the scratch's `sqrtrem_itch` limbs, all apart as
    // borrows.
    unsafe {
        match r {
            Some(r) => {
                r.clear();
                r.resize(tn + 1, 0);
                let w = Scratch::new(scratch.as_mut_ptr(), itch);
                let rn = sqrtrem::<true>(s.as_mut_ptr(), r.as_mut_ptr(), a.as_ptr(), an, w);
                r.truncate(rn);
            }
            None => {
                let w = Scratch::new(scratch.as_mut_ptr(), itch);
                let p = core::ptr::NonNull::dangling().as_ptr();
                sqrtrem::<false>(s.as_mut_ptr(), p, a.as_ptr(), an, w);
            }
        }
    }
    s.truncate(tn);
}

/// `s = floor(a^(1/k))` in the vector `s`, without zero limbs at the top, for `k >= 3` and `a`
/// with its top limb not zero, by Newton's method; `scratch` grows as it needs.
///
/// # Panics
///
/// Panics if `k < 3`, `a` is empty or its top limb is zero.
pub(crate) fn root(s: &mut Vec<u64>, a: &[u64], k: u32, scratch: &mut Vec<u64>) {
    let an = a.len();
    assert!(k >= 3 && an >= 1 && a[an - 1] != 0);
    let itch = root_itch(an, k);
    if scratch.len() < itch {
        scratch.resize(itch, 0);
    }
    // SAFETY: the `an` limbs of `a` for reads and the `root_itch` of the scratch for reads and
    // writes, apart as borrows; the root is written in the first limbs of the scratch.
    let sn = unsafe { root_limbs(a.as_ptr(), an, k, Scratch::new(scratch.as_mut_ptr(), itch)) };
    s.clear();
    s.extend_from_slice(&scratch[..sn]);
    while s.last() == Some(&0) {
        s.pop();
    }
}

/// `q = floor(n/d)` and `r = n mod d` in the vectors `q` and `r`, without zero limbs at the top,
/// for `d` without zero limbs at the top and not zero, by the division of the crate's integers
/// (the schoolbook method, divide and conquer, or Newton's method by the lengths), with the
/// working memory in `scratch`, which grows as it needs.
///
/// # Panics
///
/// Panics if `d` is empty or its top limb is zero.
pub(crate) fn div_rem_into(
    q: &mut Vec<u64>,
    r: &mut Vec<u64>,
    n: &[u64],
    d: &[u64],
    scratch: &mut Vec<u64>,
) {
    assert!(d.last().is_some_and(|&t| t != 0));
    let (nn, dn) = (n.len(), d.len());
    q.clear();
    r.clear();
    if nn < dn {
        r.extend_from_slice(n);
        return;
    }
    let itch = super::gcd::limbs::tdiv_itch(nn, dn);
    if scratch.len() < itch {
        scratch.resize(itch, 0);
    }
    r.extend_from_slice(n);
    q.resize(nn - dn + 1, 0);
    // SAFETY: `nn >= dn >= 1` and `d`'s top limb is not zero; `r` holds the `nn` limbs of the
    // dividend, valid for reads and writes, `q` the `nn - dn + 1` of the quotient, `d` is valid
    // for reads and the scratch holds the `tdiv_itch` limbs the division uses, all apart (unique
    // borrows and a shared one).
    unsafe {
        super::gcd::limbs::tdiv_qr(
            q.as_mut_ptr(),
            r.as_mut_ptr(),
            nn,
            d.as_ptr(),
            dn,
            Scratch::new(scratch.as_mut_ptr(), itch),
        );
    }
    r.truncate(dn);
    while q.last() == Some(&0) {
        q.pop();
    }
    while r.last() == Some(&0) {
        r.pop();
    }
}

/// `s = floor(sqrt(a))` in the vector `s`, without zero limbs at the top, for `a` without zero
/// limbs at the top: in registers up to two limbs, and otherwise by the crate's Karatsuba square
/// root, with the working memory in `scratch`, which grows as it needs.
pub(crate) fn sqrt_into(s: &mut Vec<u64>, a: &[u64], scratch: &mut Vec<u64>) {
    s.clear();
    let an = a.len();
    debug_assert!(a.last() != Some(&0));
    if an <= 2 {
        let v = u128::from(a.first().copied().unwrap_or(0))
            | u128::from(a.get(1).copied().unwrap_or(0)) << 64;
        let (root, _) = super::root::sqrt::sqrt_rem_u128(v);
        if root != 0 {
            s.push(root);
        }
        return;
    }
    let itch = super::root::sqrt::sqrtrem_itch(an);
    if scratch.len() < itch {
        scratch.resize(itch, 0);
    }
    let tn = an.div_ceil(2);
    s.resize(tn + 1, 0);
    // SAFETY: `an >= 3` and the top limb is not zero; `s` has room for the `tn` limbs of the root
    // and one more, `a` is valid for reads, the scratch holds the `sqrtrem_itch(an)` limbs the
    // root uses, all apart; without the remainder its pointer is not used.
    unsafe {
        super::root::sqrt::sqrtrem::<false>(
            s.as_mut_ptr(),
            core::ptr::null_mut(),
            a.as_ptr(),
            an,
            Scratch::new(scratch.as_mut_ptr(), itch),
        );
    }
    s.truncate(tn);
    while s.last() == Some(&0) {
        s.pop();
    }
}
