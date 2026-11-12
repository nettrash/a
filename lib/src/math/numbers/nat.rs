//! Arithmetic on magnitudes held as native limbs, the least significant first, for the algorithms
//! of the crate outside `numbers` that keep their integers as limbs (the exact eliminations of
//! matrices): products by the method their lengths call for, exact quotients by Hensel's method,
//! and sums and differences, over slices.

use super::div::limbs::{bdiv_q, binvert_limb};
pub(crate) use super::div::limbs::{div_2by1, invert_limb};
use super::mul::basecase;
use super::mul::limbs::{add_n, mul as mul_limbs, mul_itch, sqr as sqr_limbs, sqr_itch, sub_n};
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
