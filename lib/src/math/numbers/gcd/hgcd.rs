//! The subquadratic greatest common divisor: Möller's half gcd ("On Schönhage's algorithm and
//! subquadratic integer gcd computation", Math. Comp. 77, 2008), as GMP's `mpn_hgcd`, `mpn_gcd`
//! and `mpn_gcdext` arrange it.
//!
//! [`hgcd`] reduces two numbers of `n` limbs by Euclid's steps until they differ by less than
//! `B^s`, `s = n / 2 + 1`, never taking either below `B^s`, and returns the matrix of the steps,
//! whose entries have about `n / 2` limbs. It finds the steps from the top half of the numbers:
//! the steps that reduce the top `n / 2` limbs by half reduce the whole numbers too, for the same
//! reason as [`hgcd2`](super::limbs::hgcd2)'s, so it calls itself on the top half, applies the
//! matrix to the low half ([`Matrix::adjust`]), takes a few single steps, and calls itself again on
//! what is left of the top, the two matrices multiplied together. A call on `n` limbs is two calls
//! on about `n / 2` and a few products of `n / 2` limbs, so the whole is `O(M(n) log n)` for
//! products of `M(n)`, and the products are the crate's multiplication, with Karatsuba's and
//! Toom-Cook's methods.
//!
//! The greatest common divisor calls it on the top two thirds of the numbers, as GMP's `mpn_gcd`
//! does, and applies the matrix to the rest, over and over until the numbers are short enough for
//! Lehmer's method; the extended algorithm on the top half first and then the top two thirds,
//! keeping a row of the product of the matrices, the cofactor of one number.
//!
//! Under Miri the thresholds are a few limbs, so that its runs, which take numbers of a few limbs,
//! go through these paths too; the algorithms are right for any threshold, the thresholds only
//! decide their speed, and the fuzzing ran with these small ones as well.

use super::super::mul::limbs::{add_n, cmp_n, mul};
use super::kernels::{addmul_1, extend_row, reduce_pair};
use super::limbs::{
    Row, Step, StepExt, add_to, divexact, divexact_itch, gcdext_lehmer, hgcd2_hinted,
    lehmer_ext_itch, mul_bound, normalized, sub_from, subdiv, subdiv_ext, tdiv_itch, tdiv_qr,
    top_limbs, with_scratch,
};
use core::cmp::Ordering;
use core::ptr;

/// From this many limbs, [`hgcd`] calls itself on the top half; below, it takes Lehmer's steps
/// one at a time. Measured on an Apple M4 Max, the greatest common divisor of 2048 and of 8192
/// limbs took the least time from 180 to 350, 1 % more at 130 and 3 to 5 % more at 60.
pub(super) const HGCD_THRESHOLD: usize = if cfg!(miri) { 8 } else { 250 };

/// From this many limbs, the greatest common divisor reduces the numbers by [`hgcd`] before
/// Lehmer's method: measured on an Apple M4 Max, numbers of 768 limbs took as long either way, of
/// 640 limbs 1.7 % longer by `hgcd` and of 1024 limbs 2.3 % longer without. GMP's threshold on an
/// Apple M1 is 435, its Lehmer steps being slower.
pub(super) const GCD_DC_THRESHOLD: usize = if cfg!(miri) { 6 } else { 800 };

/// From this many limbs, the extended algorithm reduces the numbers by [`hgcd`] before Lehmer's
/// method, and down to this many it goes on doing so. Measured on an Apple M4 Max, numbers of 448
/// and 512 limbs took the least time so, 6 to 12 % more with 350 to 450 and 5 to 10 % more by
/// Lehmer's method alone; numbers of 1024 and 2048 limbs took 3 % more than with 400 or 450, which
/// costs the shorter ones more, and Lehmer's method alone 1.27 and 1.54 times as long. GMP's
/// threshold on an Apple M1 is 199.
pub(super) const GCDEXT_DC_THRESHOLD: usize = if cfg!(miri) { 6 } else { 300 };

/// `r = x y` for the `xn` limbs at `x` and the `yn` at `y`, either the longer, both at least one.
///
/// # Safety
///
/// The `xn + yn` limbs at `r` must be valid for writes and the [`mul_bound`] of their lengths at
/// `s` for reads and writes, apart from the operands.
#[inline(always)]
unsafe fn mul_any(r: *mut u64, x: *const u64, xn: usize, y: *const u64, yn: usize, s: *mut u64) {
    // SAFETY: forwarded.
    unsafe {
        if xn >= yn {
            mul(r, x, xn, y, yn, s);
        } else {
            mul(r, y, yn, x, xn, s);
        }
    }
}

/// The scratch of a product of operands of at most `xn` and `yn` limbs, in either order:
/// [`mul_bound`].
fn mul_any_itch(xn: usize, yn: usize) -> usize {
    mul_bound(xn, yn)
}

/// A reduction matrix of many limbs, `(p00, p01; p10, p11)`, of determinant 1 and non-negative
/// entries, each of `n` limbs, in room for `alloc`.
pub(super) struct Matrix {
    /// The limbs each entry has room for.
    pub(super) alloc: usize,
    /// The limbs each entry is taken to have, the longest one's; the others are zero above theirs.
    pub(super) n: usize,
    /// The entries, `p[i][j]` for `pij`.
    pub(super) p: [[*mut u64; 2]; 2],
    /// Whether the last of [`hgcd2`](super::limbs::hgcd2)'s matrices took so many steps that the
    /// quotients are mostly 1 (see [`hgcd2_hinted`]).
    pub(super) ones: bool,
}

/// The limbs that [`Matrix::new`] takes for numbers of `n` limbs: four entries of `(n + 1) / 2 +
/// 1`, which the entries of [`hgcd`]'s matrix of such numbers fit with a limb to spare.
pub(super) fn matrix_itch(n: usize) -> usize {
    4 * (n.div_ceil(2) + 1)
}

impl Matrix {
    /// The identity matrix in the [`matrix_itch`]`(n)` limbs at `mem`, for numbers of `n` limbs.
    ///
    /// # Safety
    ///
    /// The limbs must be valid for reads and writes as long as the matrix is used.
    pub(super) unsafe fn new(n: usize, mem: *mut u64) -> Matrix {
        let alloc = n.div_ceil(2) + 1;
        // SAFETY: as the caller guarantees.
        unsafe {
            ptr::write_bytes(mem, 0, 4 * alloc);
            *mem = 1;
            *mem.add(3 * alloc) = 1;
            Matrix {
                alloc,
                n: 1,
                ones: false,
                p: [
                    [mem, mem.add(alloc)],
                    [mem.add(2 * alloc), mem.add(3 * alloc)],
                ],
            }
        }
    }

    /// Records Euclid's step of quotient `q`, of `qn` limbs: column `col` gains `q` times the other
    /// column, as `a - q b` does to column 1 and `b - q a` to column 0.
    ///
    /// # Safety
    ///
    /// The `qn` limbs at `q` must be valid for reads, and the [`update_itch`] limbs at `t` for
    /// writes; the entries must have room for the sum.
    unsafe fn update(&mut self, q: *const u64, qn: usize, col: usize, t: *mut u64) {
        // SAFETY: as the caller guarantees.
        unsafe {
            if qn == 1 {
                let n = self.n;
                let c0 = addmul_1(self.p[0][col], self.p[0][1 - col], n, *q);
                let c1 = addmul_1(self.p[1][col], self.p[1][1 - col], n, *q);
                *self.p[0][col].add(n) = c0;
                *self.p[1][col].add(n) = c1;
                self.n = n + usize::from(c0 | c1 != 0);
                debug_assert!(self.n < self.alloc);
                return;
            }
            // The other column's length, as short as its top limbs allow, so that the sum fits.
            let mut n = self.n;
            while n + qn > self.n
                && *self.p[0][1 - col].add(n - 1) | *self.p[1][1 - col].add(n - 1) == 0
            {
                n -= 1;
            }
            debug_assert!(n + qn < self.alloc + 1);
            let mut carry = [0u64; 2];
            for (row, c) in carry.iter_mut().enumerate() {
                mul_any(t, self.p[row][1 - col], n, q, qn, t.add(n + qn));
                *c = add_to(self.p[row][col], t, n + qn, self.p[row][col], self.n);
            }
            let mut len = n + qn;
            if carry[0] | carry[1] != 0 {
                *self.p[0][col].add(len) = carry[0];
                *self.p[1][col].add(len) = carry[1];
                len += 1;
            } else if *self.p[0][col].add(len - 1) | *self.p[1][col].add(len - 1) == 0 {
                len -= 1;
            }
            // The other column is zero above its length.
            for row in 0..2 {
                for i in self.n..len {
                    *self.p[row][1 - col].add(i) = 0;
                }
            }
            self.n = len.max(self.n);
            debug_assert!(self.n < self.alloc);
        }
    }

    /// `M = M M1` for a matrix of single limbs: each row times `M1`, in place.
    ///
    /// # Safety
    ///
    /// The entries must have room for one more limb.
    unsafe fn mul_1(&mut self, m1: super::limbs::Matrix1) {
        // SAFETY: as the caller guarantees.
        unsafe {
            let n = self.n;
            let mut grow = false;
            for row in 0..2 {
                let (c0, c1) = extend_row(self.p[row][0], self.p[row][1], n, m1);
                *self.p[row][0].add(n) = c0;
                *self.p[row][1].add(n) = c1;
                grow |= c0 | c1 != 0;
            }
            self.n = n + usize::from(grow);
            debug_assert!(self.n < self.alloc);
        }
    }

    /// `M = M M1` for matrices of many limbs: eight products of the entries, the two of each new
    /// entry added; a new entry is at most `n + m1.n + 1` limbs long.
    ///
    /// # Safety
    ///
    /// The [`mul_matrix_itch`]`(self.n, m1.n)` limbs at `t` must be valid for reads and writes, and
    /// `self.n + m1.n < self.alloc`.
    unsafe fn mul(&mut self, m1: &Matrix, t: *mut u64) {
        let (rn, mn) = (self.n, m1.n);
        let len = rn + mn;
        debug_assert!(len < self.alloc);
        // SAFETY: as the caller guarantees: `x` takes `len + 1` limbs of the scratch, `y` `len`,
        // and the products' scratch the rest.
        unsafe {
            let (x, y) = (t, t.add(len + 1));
            let w = y.add(len);
            for row in 0..2 {
                let (r0, r1) = (self.p[row][0], self.p[row][1]);
                // r0' = r0 q00 + r1 q10 and r1' = r0 q01 + r1 q11.
                mul_any(x, r0, rn, m1.p[0][0], mn, w);
                mul_any(y, r1, rn, m1.p[1][0], mn, w);
                *x.add(len) = u64::from(add_n(x, x, y, len));
                mul_any(y, r0, rn, m1.p[0][1], mn, w);
                mul_any(r0, r1, rn, m1.p[1][1], mn, w);
                *r1.add(len) = u64::from(add_n(r1, r0, y, len));
                ptr::copy_nonoverlapping(x, r0, len + 1);
            }
            let mut n = len + 1;
            while n > 1
                && *self.p[0][0].add(n - 1)
                    | *self.p[0][1].add(n - 1)
                    | *self.p[1][0].add(n - 1)
                    | *self.p[1][1].add(n - 1)
                    == 0
            {
                n -= 1;
            }
            self.n = n;
        }
    }

    /// Applies the inverse of the matrix, found for the numbers' limbs from `p` up, to the `p`
    /// limbs below: with the reduced top limbs at `a + p` and `b + p`, `n - p` of them in all, the
    /// numbers become `a = a_top B^p + m11 a_low - m01 b_low` and `b = b_top B^p + m00 b_low -
    /// m10 a_low`. Returns their new length, which can be one limb more than `n`, or one less.
    ///
    /// # Safety
    ///
    /// `p + self.n < n`; the `n + 1` limbs at `a` and `b` must be valid for reads and writes, and
    /// the [`adjust_itch`]`(p, self.n)` limbs at `t`, apart from them.
    pub(super) unsafe fn adjust(
        &self,
        n: usize,
        a: *mut u64,
        b: *mut u64,
        p: usize,
        t: *mut u64,
    ) -> usize {
        let mn = self.n;
        debug_assert!(p + mn < n);
        // SAFETY: as the caller guarantees: each product takes `p + mn` limbs, of which the low `p`
        // replace the low limbs and the rest add into the reduced top limbs.
        unsafe {
            let (t0, t1) = (t, t.add(p + mn));
            let w = t1.add(p + mn);
            mul_any(t0, self.p[1][1], mn, a, p, w);
            mul_any(t1, self.p[1][0], mn, a, p, w);
            ptr::copy_nonoverlapping(t0, a, p);
            let mut ah = add_to(a.add(p), a.add(p), n - p, t0.add(p), mn);
            mul_any(t0, self.p[0][1], mn, b, p, w);
            let borrow = sub_from(a, a, n, t0, p + mn);
            debug_assert!(borrow <= ah);
            ah -= borrow;
            mul_any(t0, self.p[0][0], mn, b, p, w);
            ptr::copy_nonoverlapping(t0, b, p);
            let mut bh = add_to(b.add(p), b.add(p), n - p, t0.add(p), mn);
            let borrow = sub_from(b, b, n, t1, p + mn);
            debug_assert!(borrow <= bh);
            bh -= borrow;
            if ah | bh != 0 {
                *a.add(n) = ah;
                *b.add(n) = bh;
                n + 1
            } else if *a.add(n - 1) | *b.add(n - 1) == 0 {
                n - 1
            } else {
                n
            }
        }
    }
}

/// The limbs of scratch space that [`Matrix::update`] needs for a quotient of `qn` limbs and a
/// matrix of `n`.
fn update_itch(n: usize, qn: usize) -> usize {
    if qn == 1 {
        0
    } else {
        n + qn + mul_any_itch(n, qn)
    }
}

/// The limbs of scratch space that [`Matrix::mul`] needs for matrices of `rn` and `mn` limbs.
fn mul_matrix_itch(rn: usize, mn: usize) -> usize {
    2 * (rn + mn) + 1 + mul_any_itch(rn, mn)
}

/// The limbs of scratch space that [`Matrix::adjust`] needs for `p` low limbs and a matrix of
/// `mn`.
pub(super) fn adjust_itch(p: usize, mn: usize) -> usize {
    2 * (p + mn) + mul_any_itch(p, mn)
}

/// One of Euclid's steps for [`hgcd`], when [`hgcd2`](super::limbs::hgcd2) makes none, on the
/// numbers of `n` limbs at `a` and `b`, which it never takes to `s` limbs or fewer: the smaller is
/// subtracted from the larger once and, if that leaves the larger the larger, divides it, the
/// quotient one less if the remainder is too short. The steps are recorded in `m`. Returns the
/// numbers' new length, or 0 when no step can be taken so, the numbers left as they were.
///
/// # Safety
///
/// The `n` limbs at `a` and `b` must be valid for reads and writes, with a limb to spare, and the
/// `avail` limbs at `t` for reads and writes, apart from them and from the matrix, whose entries
/// must have room for the steps.
unsafe fn subdiv_hgcd(
    a: *mut u64,
    b: *mut u64,
    n: usize,
    s: usize,
    m: &mut Matrix,
    t: *mut u64,
    avail: usize,
) -> usize {
    // SAFETY: as the caller guarantees.
    unsafe {
        let (mut pa, mut pb) = (a, b);
        let mut an = normalized(pa, n);
        let mut bn = normalized(pb, n);
        // The column of the number reduced: 0 for `b`, 1 for `a`.
        let mut col = 0;
        let swap = match an.cmp(&bn) {
            Ordering::Equal => match cmp_n(pa, pb, an) {
                Ordering::Equal => return 0,
                order => order == Ordering::Greater,
            },
            order => order == Ordering::Greater,
        };
        if swap {
            core::mem::swap(&mut pa, &mut pb);
            core::mem::swap(&mut an, &mut bn);
            col = 1;
        }
        if an <= s {
            return 0;
        }
        let borrow = sub_from(pb, pb, bn, pa, an);
        debug_assert!(borrow == 0);
        bn = normalized(pb, bn);
        if bn <= s {
            // Too short: the subtraction is undone.
            let carry = add_to(pb, pa, an, pb, bn);
            *pb.add(an) = carry;
            return 0;
        }
        let order = if bn == an {
            cmp_n(pb, pa, an)
        } else {
            bn.cmp(&an)
        };
        m.update(&1, 1, col, t);
        if order == Ordering::Equal {
            // The numbers are equal now: a step was taken, and no other can be.
            return an;
        }
        if order == Ordering::Less {
            core::mem::swap(&mut pa, &mut pb);
            core::mem::swap(&mut an, &mut bn);
            col ^= 1;
        }
        let qn = bn - an + 1;
        let (q, w) = (t, t.add(qn));
        let room = avail.saturating_sub(qn);
        with_scratch(w, room, tdiv_itch(bn, an), |w| {
            tdiv_qr(q, pb, bn, pa, an, w)
        });
        let rn = normalized(pb, an);
        let mut size = an;
        if rn <= s {
            // The quotient is one too large: one less, and `a` added back.
            let mut i = 0;
            while *q.add(i) == 0 {
                *q.add(i) = u64::MAX;
                i += 1;
            }
            *q.add(i) -= 1;
            let carry = add_to(pb, pa, an, pb, rn);
            if carry != 0 {
                *pb.add(an) = carry;
                size += 1;
            }
        }
        let qn = normalized(q, qn);
        if qn > 0 {
            with_scratch(w, room, update_itch(m.n, qn), |w| m.update(q, qn, col, w));
        }
        size
    }
}

/// The limbs of scratch space that a step of [`hgcd`] on `n` limbs plans for: the quotient, and
/// the division or the product of the update. A division by divide and conquer needs more, and
/// takes it from [`with_scratch`].
fn step_itch(n: usize) -> usize {
    (n + 1) + tdiv_itch(n, 2).max(update_itch(n / 2 + 2, n))
}

/// One step of [`hgcd`] on the numbers of `n` limbs at `a` and `b`:
/// [`hgcd2`](super::limbs::hgcd2)'s matrix of the top limbs, recorded in `m` and applied to the
/// numbers, or else [`subdiv_hgcd`]. At `n = s + 1` the
/// top limbs are taken as they are, not shifted, so that the reduced numbers stay above `B^s`.
/// Returns the new length, or 0 when no step can be taken.
///
/// # Safety
///
/// As for [`subdiv_hgcd`].
unsafe fn hgcd_step(
    n: usize,
    a: *mut u64,
    b: *mut u64,
    s: usize,
    m: &mut Matrix,
    t: *mut u64,
    avail: usize,
) -> usize {
    // SAFETY: as the caller guarantees.
    unsafe {
        let top = if n == s + 1 {
            let mask = *a.add(n - 1) | *b.add(n - 1);
            if mask < 4 {
                None
            } else {
                Some((*a.add(n - 1), *a.add(n - 2), *b.add(n - 1), *b.add(n - 2)))
            }
        } else {
            Some(top_limbs(a, b, n))
        };
        if let Some(m1) = top.and_then(|(ah, al, bh, bl)| hgcd2_hinted(ah, al, bh, bl, &mut m.ones))
        {
            m.mul_1(m1);
            reduce_pair(a, b, n, m1);
            return n - usize::from(*a.add(n - 1) | *b.add(n - 1) == 0);
        }
        subdiv_hgcd(a, b, n, s, m, t, avail)
    }
}

/// The limbs of scratch space that [`hgcd`] needs for numbers of `n` limbs, besides its matrix.
pub(super) fn hgcd_itch(n: usize) -> usize {
    if n < HGCD_THRESHOLD {
        return step_itch(n);
    }
    let p = n / 2;
    let s = n / 2 + 1;
    let n2 = 3 * n / 4 + 1;
    // The second call is on at most `2 n2 - 2 s - 1` limbs, about `n / 2`.
    let half = (n - p).max(2 * n2 - 2 * s);
    let mn = half.div_ceil(2) + 1;
    let inner = hgcd_itch(half);
    let first = inner.max(adjust_itch(p, mn));
    let second = matrix_itch(half)
        + inner
            .max(adjust_itch(n, mn))
            .max(mul_matrix_itch(n.div_ceil(2) + 1, mn));
    first.max(second).max(step_itch(n))
}

/// Möller's half gcd of the numbers of `n` limbs at `a` and `b`, not both zero at the top: reduces
/// them by Euclid's steps, recorded in `m`, which starts as the identity for numbers of `n` limbs,
/// until they differ by less than `B^s`, `s = n / 2 + 1`, neither taken to `s` limbs or fewer.
/// Returns their new length, or 0 when no step can be taken so.
///
/// # Safety
///
/// The `n` limbs at `a` and `b` must be valid for reads and writes, with a limb to spare, and the
/// [`hgcd_itch`]`(n)` limbs at `t`, apart from them and from the matrix.
pub(super) unsafe fn hgcd(
    a: *mut u64,
    b: *mut u64,
    mut n: usize,
    m: &mut Matrix,
    t: *mut u64,
) -> usize {
    let s = n / 2 + 1;
    if n <= s {
        return 0;
    }
    let avail = hgcd_itch(n);
    let mut success = false;
    // SAFETY: as the caller guarantees; every call stays inside the numbers' limbs and the
    // scratch, as `hgcd_itch` counts it.
    unsafe {
        if n >= HGCD_THRESHOLD {
            let n2 = 3 * n / 4 + 1;
            // The top half first, its matrix the result's, applied to the low half.
            let p = n / 2;
            let nn = hgcd(a.add(p), b.add(p), n - p, m, t);
            if nn > 0 {
                n = m.adjust(p + nn, a, b, p, t);
                success = true;
            }
            while n > n2 {
                let nn = hgcd_step(n, a, b, s, m, t, avail);
                if nn == 0 {
                    return if success { n } else { 0 };
                }
                n = nn;
                success = true;
            }
            if n > s + 2 {
                // What is left above `B^s`, reduced by a second call on its top limbs.
                let p = 2 * s - n + 1;
                let mm = matrix_itch(n - p);
                let mut m1 = Matrix::new(n - p, t);
                let nn = hgcd(a.add(p), b.add(p), n - p, &mut m1, t.add(mm));
                if nn > 0 {
                    n = m1.adjust(p + nn, a, b, p, t.add(mm));
                    m.mul(&m1, t.add(mm));
                    success = true;
                }
            }
        }
        loop {
            let nn = hgcd_step(n, a, b, s, m, t, avail);
            if nn == 0 {
                return if success { n } else { 0 };
            }
            n = nn;
            success = true;
        }
    }
}

/// The limbs of scratch space that [`gcd_dc`] needs for numbers of `n` limbs, besides what Lehmer's
/// method needs after it.
pub(super) fn gcd_dc_itch(n: usize) -> usize {
    let p = 2 * n / 3;
    let mn = (n - p).div_ceil(2) + 1;
    matrix_itch(n - p) + hgcd_itch(n - p).max(adjust_itch(p, mn))
}

/// Reduces the numbers of `n` limbs at `a` and `b`, not both zero at the top, by [`hgcd`] on their
/// top two thirds, the matrix applied to the rest, until they are shorter than
/// [`GCD_DC_THRESHOLD`] limbs. Returns their length, or the greatest common divisor when a step
/// finds it.
///
/// # Safety
///
/// The `n` limbs at `a` and `b` must be valid for reads and writes, with a limb to spare, and the
/// `avail` limbs at `t`, at least [`gcd_dc_itch`]`(n)`, apart from them.
pub(super) unsafe fn gcd_dc(
    a: &mut *mut u64,
    b: &mut *mut u64,
    mut n: usize,
    t: *mut u64,
    avail: usize,
) -> Step {
    // SAFETY: as the caller guarantees.
    unsafe {
        while n >= GCD_DC_THRESHOLD {
            let p = 2 * n / 3;
            let mut m = Matrix::new(n - p, t);
            let w = t.add(matrix_itch(n - p));
            let nn = hgcd(a.add(p), b.add(p), n - p, &mut m, w);
            if nn > 0 {
                n = m.adjust(p + nn, *a, *b, p, w);
            } else {
                match subdiv(a, b, n, t, avail) {
                    Step::Size(k) => n = k,
                    found => return found,
                }
            }
        }
        Step::Size(n)
    }
}

/// The limbs of scratch space that [`gcdext_dc`] needs for numbers of `n` limbs, besides the row.
pub(super) fn gcdext_dc_itch(n: usize) -> usize {
    let p = n / 3;
    let mn = (n - p).div_ceil(2) + 1;
    let hg = matrix_itch(n - p)
        + hgcd_itch(n - p)
            .max(adjust_itch(p, mn))
            .max(row_itch(n + 2, mn));
    hg.max(lehmer_ext_itch(n))
        .max(tail_itch(GCDEXT_DC_THRESHOLD.min(n), n + 4))
}

/// The limbs of scratch space that [`row_times`] needs for a row of `un` limbs and a matrix of
/// `mn`.
fn row_itch(un: usize, mn: usize) -> usize {
    2 * (un + mn + 1) + mul_any_itch(un, mn)
}

/// `(u0, u1) = (u0 m00 + u1 m10, u0 m01 + u1 m11)`: the row times the matrix `m`.
///
/// # Safety
///
/// The row's entries must have room for `un + m.n + 1` limbs, zero above its length, and the
/// [`row_itch`] limbs at `t` must be valid for reads and writes.
unsafe fn row_times(row: &mut Row, m: &Matrix, t: *mut u64) {
    let (un, mn) = (row.un, m.n);
    let len = un + mn;
    // SAFETY: as the caller guarantees.
    unsafe {
        let (x, y) = (t, t.add(len + 1));
        let w = y.add(len + 1);
        mul_any(x, row.u0, un, m.p[0][0], mn, w);
        mul_any(y, row.u1, un, m.p[1][0], mn, w);
        *x.add(len) = u64::from(add_n(x, x, y, len));
        mul_any(y, row.u0, un, m.p[0][1], mn, w);
        mul_any(row.u0, row.u1, un, m.p[1][1], mn, w);
        *row.u1.add(len) = u64::from(add_n(row.u1, row.u0, y, len));
        ptr::copy_nonoverlapping(x, row.u0, len + 1);
        let mut n = len + 1;
        while n > 1 && *row.u0.add(n - 1) | *row.u1.add(n - 1) == 0 {
            n -= 1;
        }
        row.un = n;
    }
}

/// Reduces the numbers of `n` limbs at `a` and `b`, both not zero, by [`hgcd`], first on their top
/// half and then on their top two thirds, until they are shorter than [`GCDEXT_DC_THRESHOLD`]
/// limbs, and multiplies the row by each matrix. Returns their length, or the greatest common
/// divisor with its cofactor when a step finds it.
///
/// # Safety
///
/// The `n` limbs at `a` and `b` must be valid for reads and writes, with a limb to spare; the row's
/// entries for `n + 2` limbs, zero above its length; and the `avail` limbs at `t`, at least
/// [`gcdext_dc_itch`]`(n)`, apart from all of them.
pub(super) unsafe fn gcdext_dc(
    a: *mut u64,
    b: *mut u64,
    mut n: usize,
    row: &mut Row,
    t: *mut u64,
    avail: usize,
) -> StepExt {
    let mut first = true;
    // SAFETY: as the caller guarantees.
    unsafe {
        while n >= GCDEXT_DC_THRESHOLD {
            let p = if first { n / 2 } else { n / 3 };
            first = false;
            let mut m = Matrix::new(n - p, t);
            let w = t.add(matrix_itch(n - p));
            let nn = hgcd(a.add(p), b.add(p), n - p, &mut m, w);
            if nn > 0 {
                n = m.adjust(p + nn, a, b, p, w);
                row_times(row, &m, w);
            } else {
                match subdiv_ext(a, b, n, row, t, avail) {
                    StepExt::Size(k) => n = k,
                    found => return found,
                }
            }
        }
        StepExt::Size(n)
    }
}

/// The limbs of scratch space that [`tail`] needs for numbers of `n` limbs and a row of `un`.
fn tail_itch(n: usize, un: usize) -> usize {
    let lehmer = 2 * (n + 1) + 2 * (n + 4) + (n + 4) + lehmer_ext_itch(n);
    let v = (2 * n + 4) + (n + 3) + mul_any_itch(n, n + 3).max(divexact_itch(2 * n + 4, n));
    let combine = 2 * (un + n + 5) + mul_any_itch(un, n + 3);
    lehmer + v + combine
}

/// The end of the extended algorithm once [`gcdext_dc`] has reduced the numbers to `n` limbs and
/// built the row `(u0, u1)`: Lehmer's method on copies of the numbers with a row of their own,
/// which gives the cofactor `u` of the reduced `a`, the cofactor `v = (g - u a) / b` of the reduced
/// `b` from it, an exact division, and the cofactor of the original `a` as `u u1 - v u0`, as GMP's
/// `mpn_gcdext` arranges it: the row is multiplied once, instead of at every one of Lehmer's
/// steps. When the numbers are equal, the cofactor is the row's smaller one, as in
/// [`gcdext_lehmer`].
///
/// # Safety
///
/// As for [`gcdext`], with the numbers left unchanged, and the [`tail_itch`] limbs at `t`.
#[allow(clippy::too_many_arguments)]
unsafe fn tail(
    g: *mut u64,
    c: *mut u64,
    a: *mut u64,
    b: *mut u64,
    n: usize,
    row: &mut Row,
    t: *mut u64,
    avail: usize,
) -> (usize, usize, bool) {
    // SAFETY: as the caller guarantees: the copies, the row of their own, its cofactor, the
    // product and the other cofactor, the two products of the end, and the scratch lie one after
    // the other in the scratch, as `tail_itch` counts them.
    unsafe {
        if cmp_n(a, b, n) == Ordering::Equal {
            let f = row.smaller();
            ptr::copy_nonoverlapping(a, g, n);
            ptr::copy(f.p, c, f.n);
            return (normalized(g, n), f.n, f.negative);
        }
        let (ac, bc) = (t, t.add(n + 1));
        let (r0, r1) = (bc.add(n + 1), bc.add(n + 1 + n + 4));
        let cq = r1.add(n + 4);
        let prod = cq.add(n + 4);
        let v = prod.add(2 * n + 4);
        let p1 = v.add(n + 3);
        let p2 = p1.add(row.un + n + 5);
        let w = p2.add(row.un + n + 5);
        let room = avail - (w.offset_from(t) as usize);
        ptr::copy_nonoverlapping(a, ac, n);
        ptr::copy_nonoverlapping(b, bc, n);
        ptr::write_bytes(r0, 0, 2 * (n + 4));
        *r1 = 1;
        let mut fresh = Row {
            u0: r0,
            u1: r1,
            un: 1,
        };
        let (gn, un_, negative) = gcdext_lehmer(g, cq, ac, bc, n, &mut fresh, w, room);
        let (u0, u1) = (row.u0, row.u1);
        let (u0n, u1n) = (normalized(u0, row.un), normalized(u1, row.un));
        if un_ == 0 {
            // u = 0, so v = g / b = 1: the cofactor is `-u0`.
            ptr::copy(u0, c, u0n);
            return (gn, u0n, u0n != 0);
        }
        // v = (g - u a) / b, whose sign is the opposite of `u`'s.
        let an = normalized(a, n);
        let bn = normalized(b, n);
        let mut pn = an + un_;
        mul_any(prod, a, an, cq, un_, w);
        if negative {
            *prod.add(pn) = add_to(prod, prod, pn, g, gn);
            pn += 1;
        } else {
            let borrow = sub_from(prod, prod, pn, g, gn);
            debug_assert!(borrow == 0);
        }
        let pn = normalized(prod, pn);
        let vn = if pn == 0 {
            0
        } else {
            divexact(v, prod, pn, b, bn, w, room);
            normalized(v, pn - bn + 1)
        };
        // The cofactor: sgn(u) (|u| u1 + |v| u0).
        let n1 = un_ + u1n;
        mul_any(p1, cq, un_, u1, u1n, w);
        let len = if vn == 0 || u0n == 0 {
            n1
        } else {
            let n2 = vn + u0n;
            mul_any(p2, v, vn, u0, u0n, w);
            let (long, ln, short, sn) = if n1 >= n2 {
                (p1, n1, p2, n2)
            } else {
                (p2, n2, p1, n1)
            };
            *long.add(ln) = add_to(long, long, ln, short, sn);
            if long != p1 {
                ptr::copy_nonoverlapping(long, p1, ln + 1);
            }
            ln + 1
        };
        let len = normalized(p1, len);
        ptr::copy_nonoverlapping(p1, c, len);
        (gn, len, negative && len != 0)
    }
}

/// The extended algorithm on the numbers of `n` limbs at `a` and `b`, both not zero: [`gcdext_dc`]
/// down to [`GCDEXT_DC_THRESHOLD`] limbs, then [`gcdext_lehmer`], or [`tail`] when the row is not
/// the identity's any more; writes the greatest common divisor at `g` and its cofactor at `c`, and
/// returns their lengths and the cofactor's sign, as `gcdext_lehmer` does.
///
/// # Safety
///
/// As for [`gcdext_lehmer`], with `avail` at least [`gcdext_dc_itch`]`(n)` and the numbers with a
/// limb to spare.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn gcdext(
    g: *mut u64,
    c: *mut u64,
    a: *mut u64,
    b: *mut u64,
    n: usize,
    row: &mut Row,
    t: *mut u64,
    avail: usize,
) -> (usize, usize, bool) {
    // SAFETY: as the caller guarantees.
    unsafe {
        if n < GCDEXT_DC_THRESHOLD {
            return gcdext_lehmer(g, c, a, b, n, row, t, avail);
        }
        let identity = row.un == 1 && *row.u0 == 0 && *row.u1 == 1;
        let n = match gcdext_dc(a, b, n, row, t, avail) {
            StepExt::Size(k) => k,
            StepExt::Gcd(found) => {
                ptr::copy(found.g, g, found.gn);
                let f = found.cofactor;
                ptr::copy(f.p, c, f.n);
                return (found.gn, f.n, f.negative);
            }
        };
        if identity && row.un == 1 && *row.u0 == 0 && *row.u1 == 1 {
            return gcdext_lehmer(g, c, a, b, n, row, t, avail);
        }
        tail(g, c, a, b, n, row, t, avail)
    }
}
