//! Products of polynomials modulo an `n` of `k ≥ 2` limbs: Kronecker substitution into the
//! crate's multiplication with one division by `n` a coefficient, and the schoolbook method for
//! short operands, its sums of products of residues kept in `2k + 1` limbs and divided once.
//!
//! The division works for every `n`, even or odd: a field of the product, below `m (n - 1)^2`
//! for `m` products, is shifted as `n` is normalized (its top bit set) and divided by the crate's
//! division of limbs (two or three limbs in registers, the schoolbook method beyond), which
//! leaves the remainder.

use super::lred::LimbRed;
use super::pack;
use crate::math::numbers::nat;

/// The reduction modulo `n` of `k ≥ 2` limbs.
pub(crate) struct LimbMod<'a> {
    /// The limbs of `n`.
    k: usize,
    /// The bits of `n - 1`.
    bits_n1: usize,
    /// The division by `n`.
    red: &'a LimbRed,
}

impl LimbMod<'_> {
    /// The reduction modulo the `k ≥ 2` limbs `n` (the top one not zero) by `red`.
    pub(crate) fn new<'a>(n: &[u64], red: &'a LimbRed) -> LimbMod<'a> {
        let k = n.len();
        debug_assert!(k >= 2 && n[k - 1] != 0);
        // n - 1 has the bits of n unless n is a power of 2.
        let pow2 = n[..k - 1].iter().all(|&l| l == 0) && n[k - 1].is_power_of_two();
        let bits_n = 64 * k - n[k - 1].leading_zeros() as usize;
        LimbMod {
            k,
            bits_n1: if pow2 { bits_n - 1 } else { bits_n },
            red,
        }
    }

    /// Sets the `k` limbs `out` to `x mod n` for the limbs `x`, any number of them.
    #[inline]
    pub(crate) fn reduce(&self, x: &[u64], out: &mut [u64]) {
        self.red.reduce(x, out);
    }
}

/// The bits of `len`.
#[inline]
fn bits_of(len: usize) -> usize {
    (usize::BITS - len.leading_zeros()) as usize
}

/// The product of the polynomials `a` and `b` of packed coefficients of `k` limbs (the square
/// of `a` when `b` is empty) by Kronecker substitution: `(la + lb - 1) k` limbs.
pub(crate) fn ks(a: &[u64], b: &[u64], m: &LimbMod) -> Vec<u64> {
    let k = m.k;
    let (la, lb) = (a.len() / k, b.len() / k);
    let square = b.is_empty();
    let lb = if square { la } else { lb };
    let len = la + lb - 1;
    let s = 2 * m.bits_n1 + bits_of(la.min(lb));
    let pack_all = |x: &[u64], l: usize| {
        let mut v = vec![0u64; (l * s).div_ceil(64) + 1];
        for (i, c) in x.chunks_exact(k).enumerate() {
            pack::put(&mut v, i * s, c);
        }
        while v.last() == Some(&0) {
            v.pop();
        }
        v
    };
    let mut out = vec![0u64; len * k];
    let xa = pack_all(a, la);
    if xa.is_empty() {
        return out;
    }
    let mut p = Vec::new();
    let mut scratch = Vec::new();
    if square {
        nat::sqr_into(&mut p, &xa, &mut scratch);
    } else {
        let xb = pack_all(b, lb);
        if xb.is_empty() {
            return out;
        }
        nat::mul_into(&mut p, &xa, &xb, &mut scratch);
    }
    let mut field = vec![0u64; s.div_ceil(64)];
    for (j, o) in out.chunks_exact_mut(k).enumerate() {
        pack::get(&p, j * s, s, &mut field);
        m.reduce(&field, o);
    }
    out
}

/// The schoolbook product of `a` and `b` of packed coefficients of `k` limbs, both not empty:
/// each coefficient a sum of products of residues in `2k + 1` limbs (or more, for very long
/// operands), divided by `n` once.
pub(crate) fn school(a: &[u64], b: &[u64], m: &LimbMod) -> Vec<u64> {
    let k = m.k;
    let (la, lb) = (a.len() / k, b.len() / k);
    let len = la + lb - 1;
    let w = 2 * k + 1 + bits_of(la.min(lb)).div_ceil(64);
    let mut out = vec![0u64; len * k];
    // The sums and the products on the stack for up to 8 limbs.
    let (mut accb, mut tb) = ([0u64; 24], [0u64; 16]);
    let (mut accv, mut tv) = (Vec::new(), Vec::new());
    let (acc, t): (&mut [u64], &mut [u64]) = if w <= 24 && k <= 8 {
        (&mut accb[..w], &mut tb[..2 * k])
    } else {
        accv.resize(w, 0);
        tv.resize(2 * k, 0);
        (&mut accv, &mut tv)
    };
    let mut scratch = Vec::new();
    for (j, o) in out.chunks_exact_mut(k).enumerate() {
        acc.fill(0);
        let lo = j.saturating_sub(lb - 1);
        let hi = j.min(la - 1);
        for i in lo..=hi {
            let x = &a[i * k..(i + 1) * k];
            let y = &b[(j - i) * k..(j - i + 1) * k];
            nat::mul_to(t, x, y, &mut scratch);
            let carry = nat::add_assign(acc, t);
            debug_assert!(!carry);
        }
        m.reduce(acc, o);
    }
    out
}

/// The product of `a` and `b` of packed coefficients of `k` limbs (the square of `a` when `b` is
/// empty) by the transforms on pieces of the coefficients (see `nat::limb_poly_mul`), each
/// coefficient's exact value reduced once: `None` past the transforms.
pub(crate) fn split(a: &[u64], b: &[u64], m: &LimbMod) -> Option<Vec<u64>> {
    let k = m.k;
    let (la, lb) = (a.len() / k, b.len() / k);
    let p = nat::limb_poly_plan(la, lb, m.bits_n1.max(1) as u32)?;
    let len = if b.is_empty() {
        2 * la - 1
    } else {
        la + lb - 1
    };
    let ow = p.out_limbs();
    let mut v = vec![0u64; len * ow];
    nat::limb_poly_mul(&mut v, ow, a, b, k, &p);
    let mut out = vec![0u64; len * k];
    for (o, x) in out.chunks_exact_mut(k).zip(v.chunks_exact(ow)) {
        m.reduce(x, o);
    }
    Some(out)
}

/// The low `n_out` coefficients of the product of `a` and `b` (packed, `k` limbs a coefficient,
/// not empty, `n_out` at most the product's length): by the transforms on pieces when they are
/// the method, the theorem applied to those coefficients only; `None` otherwise.
pub(crate) fn mul_low(
    a: &[u64],
    b: &[u64],
    n_out: usize,
    n: &[u64],
    red: &LimbRed,
) -> Option<Vec<u64>> {
    let k = n.len();
    let (la, lb) = (a.len() / k, b.len() / k);
    if !use_split(la.min(lb), la.max(lb), k) {
        return None;
    }
    let m = LimbMod::new(n, red);
    let p = nat::limb_poly_plan(la, lb, m.bits_n1.max(1) as u32)?;
    let ow = p.out_limbs();
    let mut v = vec![0u64; n_out * ow];
    nat::limb_poly_mul(&mut v, ow, a, b, k, &p);
    let mut out = vec![0u64; n_out * k];
    for (o, x) in out.chunks_exact_mut(k).zip(v.chunks_exact(ow)) {
        m.reduce(x, o);
    }
    Some(out)
}

/// Whether a product of `short` by `long` coefficients of `k` limbs goes by the transforms on
/// pieces of the coefficients rather than Kronecker substitution: measured on an Apple M4 (under
/// load), 0.7 to 0.9 times its time from 48 coefficients up to products of about 6,000 for 2 to
/// 9 limbs, where the integers' transforms, of lengths that fit the product closer, catch up.
pub(crate) fn use_split(short: usize, long: usize, k: usize) -> bool {
    if cfg!(miri) {
        return short >= 3;
    }
    short >= 48 && k <= 9 && short + long <= 6000
}

/// Up to this many coefficients in the shorter operand, the schoolbook method.
fn school_max(k: usize) -> usize {
    if cfg!(miri) {
        2
    } else if k <= 2 {
        6
    } else {
        4
    }
}

/// The product of `a` and `b` (packed, `k` limbs a coefficient, not empty), by the method their
/// lengths call for; a square when `b` is empty.
pub(crate) fn mul(a: &[u64], b: &[u64], n: &[u64], red: &LimbRed) -> Vec<u64> {
    let m = LimbMod::new(n, red);
    let k = n.len();
    let short = if b.is_empty() {
        a.len()
    } else {
        a.len().min(b.len())
    } / k;
    if short <= school_max(k) {
        return school(a, if b.is_empty() { a } else { b }, &m);
    }
    let long = if b.is_empty() {
        a.len()
    } else {
        a.len().max(b.len())
    } / k;
    if use_split(short, long, k)
        && let Some(c) = split(a, b, &m)
    {
        return c;
    }
    ks(a, b, &m)
}

/// The quotient and the remainder of `a` by `d` (packed, `k` limbs a coefficient, `d` not empty)
/// with `inv` the inverse of the leading coefficient of `d`, by the schoolbook method with delayed
/// reduction: the remainder's coefficients kept as sums `a_j + Σ (n - t_i) d_(j-i)` in `2k + 2`
/// limbs or more, only the one that gives the next quotient coefficient divided by `n` at each
/// step, the others once at the end. The remainder has `ld - 1` coefficients, not trimmed.
pub(crate) fn div_rem_school(
    a: &[u64],
    d: &[u64],
    inv: &[u64],
    n: &[u64],
    red: &LimbRed,
) -> (Vec<u64>, Vec<u64>) {
    let k = n.len();
    let m = LimbMod::new(n, red);
    let (la, ld) = (a.len() / k, d.len() / k);
    if la < ld {
        let mut r = a.to_vec();
        r.resize((ld - 1) * k, 0);
        return (Vec::new(), r);
    }
    let lq = la - ld + 1;
    let steps = lq.min(ld - 1) + 1;
    let w = 2 * k + 1 + bits_of(steps).div_ceil(64);
    let mut acc = vec![0u64; la * w];
    for (j, c) in a.chunks_exact(k).enumerate() {
        acc[j * w..j * w + k].copy_from_slice(c);
    }
    let mut q = vec![0u64; lq * k];
    let mut top = vec![0u64; k];
    let mut t = vec![0u64; k];
    let mut nt = vec![0u64; k];
    let mut p = vec![0u64; 2 * k];
    let mut prod = vec![0u64; 2 * k];
    let mut scratch = Vec::new();
    for i in (0..lq).rev() {
        let h = i + ld - 1;
        m.reduce(&acc[h * w..(h + 1) * w], &mut top);
        nat::mul_to(&mut prod, &top, inv, &mut scratch);
        m.reduce(&prod, &mut t);
        q[i * k..(i + 1) * k].copy_from_slice(&t);
        if t.iter().all(|&l| l == 0) {
            continue;
        }
        // nt = n - t.
        nt.copy_from_slice(n);
        nat::sub_assign(&mut nt, &t);
        for j in 0..ld - 1 {
            nat::mul_to(&mut p, &nt, &d[j * k..(j + 1) * k], &mut scratch);
            let carry = nat::add_assign(&mut acc[(i + j) * w..(i + j + 1) * w], &p);
            debug_assert!(!carry);
        }
    }
    let mut r = vec![0u64; (ld - 1) * k];
    for (j, o) in r.chunks_exact_mut(k).enumerate() {
        m.reduce(&acc[j * w..(j + 1) * w], o);
    }
    (q, r)
}
