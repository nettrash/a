//! Products of many linear factors, remainders and the values of a polynomial at many points,
//! modulo `n`, for the elliptic curve method's second stage: the fast arithmetic of polynomials
//! (Borodin and Moenck, 1974) on the products of [`Kron`].
//!
//! The coefficients are held in Montgomery's form for `R' = B^(k + 1)`, `x R' mod n`, below `n`:
//! [`Kron`] reduces each coefficient of a product by REDC over `k + 1` limbs, which in that form is
//! the exact product, so that monic polynomials stay monic and remainders are exact. A residue of
//! the methods' arithmetic, `x R`, takes that form by one product with the residue of `B`; the
//! lazy arithmetics hold it below `2n`, which [`Kron`] takes as it is and [`Coeffs::neg`], which
//! makes the leaves from the points, reduces first.
//!
//! - The subproduct tree of the points `a_j`: the leaves `X - a_j`, and each node above the
//!   product of two, the last of an odd level carried up.
//! - The inverse of the reversed root `rev(F)` modulo `X^d`, by Newton's iteration
//!   `v <- v - v (rev(F) v - 1)`, which doubles the exact coefficients each step.
//! - `P mod F` for `P` of degree below `2d`: the quotient's reversal is the low half of
//!   `rev(P) rev(F)^(-1)`, and the remainder the low half of `P - q F`.
//! - The values `H(a_j)` at the roots of `F`, by the transposed algorithm (Bostan, Lecerf and
//!   Schost, "Tellegen's principle into practice", 2003): at the root, the reversal of the low
//!   `d` coefficients of `rev(F)^(-1) rev(H)`; at a node of children `M_1` of `d_1` leaves and
//!   `M_2` of `d_2`, the vector `u` goes to the first child as the coefficients `d_2 ..
//!   d_1 + d_2` of `M_2 u`, and to the second as the coefficients `d_1 .. d_1 + d_2` of `M_1 u`;
//!   at a leaf it is the value. Each node takes two products, as many as the tree's building.

use super::super::mul::limbs::{add_n, cmp_n, sub_n};
use super::poly::Kron;
use core::cmp::Ordering;

/// Coefficients modulo `n`, `k` limbs each, below `n`.
pub(super) struct Coeffs<'a> {
    /// `n`.
    pub(super) n: &'a [u64],
}

impl Coeffs<'_> {
    /// `r = a - b mod n`.
    #[inline]
    pub(super) fn sub(&self, r: &mut [u64], a: &[u64], b: &[u64]) {
        let k = self.n.len();
        debug_assert!(r.len() == k && a.len() == k && b.len() == k);
        // SAFETY: `k` limbs each; the kernels allow `r` at an operand, and `r` is apart from `a`
        // and `b` as a mutable borrow.
        unsafe {
            if sub_n(r.as_mut_ptr(), a.as_ptr(), b.as_ptr(), k) {
                add_n(r.as_mut_ptr(), r.as_ptr(), self.n.as_ptr(), k);
            }
        }
    }

    /// `r = -r mod n`, below `n`, for `r` below `2n`: the points come from the arithmetic, whose
    /// lazy kinds hold residues below `2n`.
    #[inline]
    pub(super) fn neg(&self, r: &mut [u64]) {
        let k = self.n.len();
        debug_assert_eq!(r.len(), k);
        // SAFETY: `k` limbs each; the kernels allow `r` at an operand.
        unsafe {
            if cmp_n(r.as_ptr(), self.n.as_ptr(), k) != Ordering::Less {
                sub_n(r.as_mut_ptr(), r.as_ptr(), self.n.as_ptr(), k);
            }
        }
        if r.iter().all(|&l| l == 0) {
            return;
        }
        // SAFETY: `k` limbs each, `r` now below `n`.
        unsafe { sub_n(r.as_mut_ptr(), self.n.as_ptr(), r.as_ptr(), k) };
    }
}

/// The reversal of the coefficients of `a`, `k` limbs each.
pub(super) fn reverse(a: &[u64], k: usize) -> Vec<u64> {
    let mut r = Vec::with_capacity(a.len());
    for c in a.chunks_exact(k).rev() {
        r.extend_from_slice(c);
    }
    r
}

/// The subproduct tree of points: `levels[0]` the leaves `X - a_j`, each level above the products
/// of pairs of the one below, the last of an odd level carried up; the root alone at the top.
/// Each node is monic, its coefficients the constant first.
pub(super) struct Tree {
    pub(super) levels: Vec<Vec<Vec<u64>>>,
}

impl Tree {
    /// The tree of the points `roots`, `k` limbs each and below `2n`, with `one` the coefficient 1.
    pub(super) fn new(kron: &mut Kron, roots: &[u64], one: &[u64], c: &Coeffs<'_>) -> Tree {
        let k = kron.limbs();
        let mut leaves = Vec::with_capacity(roots.len() / k);
        for a in roots.chunks_exact(k) {
            let mut p = vec![0u64; 2 * k];
            p[..k].copy_from_slice(a);
            c.neg(&mut p[..k]);
            p[k..].copy_from_slice(one);
            leaves.push(p);
        }
        let mut levels = vec![leaves];
        while levels.last().is_some_and(|l| l.len() > 1) {
            let below = levels.last().unwrap();
            let mut next = Vec::with_capacity(below.len().div_ceil(2));
            for pair in below.chunks(2) {
                if let [a, b] = pair {
                    let len = a.len() / k + b.len() / k - 1;
                    let mut p = vec![0u64; len * k];
                    kron.mul(a, b, 0, &mut p);
                    next.push(p);
                } else {
                    next.push(pair[0].clone());
                }
            }
            levels.push(next);
        }
        Tree { levels }
    }

    /// The root: the product of all the leaves.
    pub(super) fn root(&self) -> &[u64] {
        &self.levels.last().unwrap()[0]
    }

    /// The product alone, without keeping the levels: the points as for [`Tree::new`].
    pub(super) fn product(kron: &mut Kron, roots: &[u64], one: &[u64], c: &Coeffs<'_>) -> Vec<u64> {
        let k = kron.limbs();
        let mut level: Vec<Vec<u64>> = Vec::with_capacity(roots.len() / k);
        for a in roots.chunks_exact(k) {
            let mut p = vec![0u64; 2 * k];
            p[..k].copy_from_slice(a);
            c.neg(&mut p[..k]);
            p[k..].copy_from_slice(one);
            level.push(p);
        }
        while level.len() > 1 {
            let mut next = Vec::with_capacity(level.len().div_ceil(2));
            let mut it = level.into_iter();
            while let Some(a) = it.next() {
                match it.next() {
                    Some(b) => {
                        let len = a.len() / k + b.len() / k - 1;
                        let mut p = vec![0u64; len * k];
                        kron.mul(&a, &b, 0, &mut p);
                        next.push(p);
                    }
                    None => next.push(a),
                }
            }
            level = next;
        }
        level.pop().unwrap_or_default()
    }

    /// The values at the leaves of the polynomial whose transposed vector at the root is `u`, by
    /// the descent: `out` gets them in the leaves' order.
    pub(super) fn descend(&self, kron: &mut Kron, u: Vec<u64>, out: &mut Vec<u64>) {
        let k = kron.limbs();
        let top = self.levels.len() - 1;
        let mut cur = vec![u];
        for level in (1..=top).rev() {
            let below = &self.levels[level - 1];
            let mut next = Vec::with_capacity(below.len());
            for (i, v) in cur.into_iter().enumerate() {
                let (l, r) = (2 * i, 2 * i + 1);
                if r >= below.len() {
                    // A node carried up: the same vector.
                    next.push(v);
                    continue;
                }
                let (m1, m2) = (&below[l], &below[r]);
                let (d1, d2) = (m1.len() / k - 1, m2.len() / k - 1);
                debug_assert_eq!(v.len() / k, d1 + d2);
                let mut u1 = vec![0u64; d1 * k];
                kron.mul(m2, &v, d2, &mut u1);
                let mut u2 = vec![0u64; d2 * k];
                kron.mul(m1, &v, d1, &mut u2);
                next.push(u1);
                next.push(u2);
            }
            cur = next;
        }
        out.clear();
        for v in cur {
            debug_assert_eq!(v.len(), k);
            out.extend_from_slice(&v);
        }
    }
}

/// `rev(F)^(-1) mod X^d` for the monic `F` of degree `d`, whose reversal starts with 1: Newton's
/// iteration from `v = 1`, each step `v - X^m (v e mod X^m)` for `e` the coefficients `m .. 2m` of
/// `rev(F) v`.
pub(super) fn inverse(kron: &mut Kron, f: &[u64], one: &[u64], c: &Coeffs<'_>) -> Vec<u64> {
    let k = kron.limbs();
    let d = f.len() / k - 1;
    let rev = reverse(f, k);
    let mut v = one.to_vec();
    let mut m = 1;
    while m < d {
        let m2 = (2 * m).min(d);
        // `e`: the coefficients `m .. m2` of `rev(F) v`, the low `m` being those of 1.
        let mut e = vec![0u64; (m2 - m) * k];
        kron.mul(&rev[..m2 * k], &v, m, &mut e);
        // `v e mod X^(m2 - m)`, subtracted at `m`.
        let mut t = vec![0u64; (m2 - m) * k];
        kron.mul(&v, &e, 0, &mut t);
        v.resize(m2 * k, 0);
        for (dst, src) in v[m * k..].chunks_exact_mut(k).zip(t.chunks_exact(k)) {
            dst.copy_from_slice(src);
            c.neg(dst);
        }
        m = m2;
    }
    v
}

/// `P mod F` for `P` of fewer than `2d` coefficients (padded to `2d`) and the monic `F` of degree
/// `d`, with `inv = rev(F)^(-1) mod X^d`: `d` coefficients.
pub(super) fn reduce(
    kron: &mut Kron,
    p: &[u64],
    f: &[u64],
    inv: &[u64],
    c: &Coeffs<'_>,
) -> Vec<u64> {
    let k = kron.limbs();
    let d = f.len() / k - 1;
    let lp = p.len() / k;
    debug_assert!(lp <= 2 * d);
    if lp <= d {
        let mut r = p.to_vec();
        r.resize(d * k, 0);
        return r;
    }
    // The quotient, of degree `lp - 1 - d`: its reversal is the low `lp - d` coefficients of
    // `rev(P) inv`, `rev(P)` taken to `lp - d` coefficients.
    let q_len = lp - d;
    let top = reverse(&p[d * k..], k);
    let mut rq = vec![0u64; q_len * k];
    kron.mul(&top, &inv[..q_len.min(d) * k], 0, &mut rq);
    let q = reverse(&rq, k);
    // `P - q F` below `X^d`.
    let mut qf = vec![0u64; d * k];
    kron.mul(&q, f, 0, &mut qf);
    let mut r = vec![0u64; d * k];
    for ((dst, a), b) in r
        .chunks_exact_mut(k)
        .zip(p.chunks_exact(k))
        .zip(qf.chunks_exact(k))
    {
        c.sub(dst, a, b);
    }
    r
}

/// The vector at the root of the descent for the values of `H`, `d` coefficients, at the roots of
/// `F`: the reversal of the low `d` coefficients of `inv rev(H)`.
pub(super) fn transposed_root(kron: &mut Kron, h: &[u64], inv: &[u64]) -> Vec<u64> {
    let k = kron.limbs();
    let d = inv.len() / k;
    let rh = reverse(h, k);
    let mut w = vec![0u64; d * k];
    kron.mul(inv, &rh, 0, &mut w);
    reverse(&w, k)
}
