//! Products and divisions of polynomials modulo any `n ≥ 2` on the packed coefficients of a
//! [`ModPolynomial`], for the loops of the factorisation (Hensel's lifting modulo `pᵃ`, the
//! arithmetic modulo a prime of several limbs), on the arith area's kernels: the products of
//! `mmul` (Kronecker substitution with one reduction a coefficient, the schoolbook method for
//! short operands), its delayed-reduction schoolbook division, and the division by a monic
//! polynomial by the Newton inverse of its reversal past a crossover. A modulus of one word takes
//! the word kernels of [`nmod`](super::nmod).

use super::super::arith::mmul;
use super::super::modular::{Modulus, Work};
use super::nmod;

/// The length of a monic divisor from which a remainder goes by its Newton inverse, for a
/// modulus of several limbs.
pub(crate) const NEWTON_DIV: usize = 24;

/// The same for a word modulus.
pub(crate) const NEWTON_DIV_WORD: usize = 160;

/// Drops the zero coefficients at the top of the packed `a`.
#[inline]
pub(crate) fn trim(a: &mut Vec<u64>, k: usize) {
    while a.len() >= k && a[a.len() - k..].iter().all(|&l| l == 0) {
        a.truncate(a.len() - k);
    }
}

/// The product of the packed `a` and `b` modulo `m` (each possibly with zeros at the top; the
/// square when `b` is `None`): `la + lb - 1` coefficients, none when one is empty. The products
/// of [`arith`](super::super::arith): modulo a word those of `wmul`, modulo several limbs
/// Kronecker substitution with one reduction a coefficient (`mmul`).
pub(crate) fn mul_raw(m: &Modulus, a: &[u64], b: Option<&[u64]>) -> Vec<u64> {
    if let Some(w) = m.word_mod() {
        return match b {
            Some(b) => nmod::mul_raw(w, a, b),
            None => nmod::sqr_raw(w, a),
        };
    }
    if a.is_empty() || b.is_some_and(<[u64]>::is_empty) {
        return Vec::new();
    }
    mmul::mul(
        a,
        b.unwrap_or(&[]),
        &m.d,
        m.l.as_ref().expect("several limbs"),
    )
}

/// The product, trimmed.
pub(crate) fn mul(m: &Modulus, a: &[u64], b: &[u64]) -> Vec<u64> {
    let mut r = mul_raw(m, a, Some(b));
    trim(&mut r, m.k());
    r
}

/// The inverse of the packed power series `h`, `h₀` a unit, modulo `xⁿ`: `n` coefficients,
/// possibly with zeros at the top.
pub(crate) fn inv_series(m: &Modulus, h: &[u64], n: usize) -> Vec<u64> {
    let k = m.k();
    if let Some(w) = m.word_mod() {
        return nmod::inv_series(w, h, n);
    }
    let h0 = m
        .inv(&h[..k])
        .expect("the constant coefficient is not a unit modulo n");
    let mut g = h0;
    let mut kk = 1;
    let lh = h.len() / k;
    while kk < n {
        let k2 = (2 * kk).min(n);
        let mut e = mul_raw(m, &h[..k2.min(lh) * k], Some(&g));
        e.resize(k2 * k, 0);
        let gl = (k2 - kk).min(g.len() / k);
        let mut t = mul_raw(m, &g[..gl * k], Some(&e[kk * k..k2 * k]));
        t.resize((k2 - kk) * k, 0);
        for c in t.chunks_exact_mut(k) {
            m.neg(c);
        }
        g.extend_from_slice(&t);
        kk = k2;
    }
    g
}

/// Division by a fixed monic `h` modulo `m`, with the Newton inverse of its reversal when it is
/// long.
pub(crate) struct Div {
    /// `h`, monic, packed.
    h: Vec<u64>,
    /// The inverse of the reversal of `h` modulo `x^(deg h)`, or empty.
    hinv: Vec<u64>,
}

impl Div {
    /// Division by the packed monic `h` of degree at least 1.
    pub(crate) fn new(m: &Modulus, h: &[u64]) -> Div {
        let k = m.k();
        let n = h.len() / k;
        let threshold = if k == 1 { NEWTON_DIV_WORD } else { NEWTON_DIV };
        let hinv = if n >= threshold {
            let rev: Vec<u64> = h.chunks_exact(k).rev().flatten().copied().collect();
            inv_series(m, &rev, n - 1)
        } else {
            Vec::new()
        };
        Div {
            h: h.to_vec(),
            hinv,
        }
    }

    /// The quotient (when `want_q`) and the remainder of the packed `a`, both trimmed.
    pub(crate) fn divrem(&self, m: &Modulus, a: &[u64], want_q: bool) -> (Vec<u64>, Vec<u64>) {
        let k = m.k();
        let n = self.h.len() / k;
        let mut la = a.len() / k;
        if la < n {
            let mut r = a.to_vec();
            trim(&mut r, k);
            return (Vec::new(), r);
        }
        if let Some(w) = m.word_mod()
            && self.hinv.is_empty()
        {
            if want_q {
                return nmod::divrem(w, a, &self.h);
            }
            return (Vec::new(), nmod::rem(w, a, &self.h));
        }
        if self.hinv.is_empty() {
            return schoolbook(m, a, &self.h, want_q);
        }
        // Chunks of at most n - 1 coefficients of the quotient, from the top.
        let mut r = a.to_vec();
        let mut q = if want_q {
            vec![0u64; (la + 1 - n) * k]
        } else {
            Vec::new()
        };
        while la >= n {
            let qn = (la + 1 - n).min(n - 1);
            let lo = la - qn - (n - 1); // the chunk's quotient times h covers lo..la
            let top: Vec<u64> = r[(la - qn) * k..la * k]
                .chunks_exact(k)
                .rev()
                .flatten()
                .copied()
                .collect();
            let mut qr = mul_raw(m, &top, Some(&self.hinv[..qn * k]));
            qr.truncate(qn * k);
            let qc: Vec<u64> = qr.chunks_exact(k).rev().flatten().copied().collect();
            // r[lo..la] -= qc·h; the top qn coefficients become 0.
            let prod = mul_raw(m, &qc, Some(&self.h[..(n - 1) * k]));
            for (i, c) in prod.chunks_exact(k).enumerate() {
                if lo + i < la - qn {
                    m.sub(&mut r[(lo + i) * k..(lo + i + 1) * k], c);
                }
            }
            if want_q {
                q[lo * k..(lo + qn) * k].copy_from_slice(&qc);
            }
            la -= qn;
            r.truncate(la * k);
        }
        trim(&mut r, k);
        trim(&mut q, k);
        (q, r)
    }
}

/// The schoolbook division of the packed `a` by the monic `h`, `k ≥ 2` limbs a coefficient,
/// with delayed reduction (`mmul`).
fn schoolbook(m: &Modulus, a: &[u64], h: &[u64], want_q: bool) -> (Vec<u64>, Vec<u64>) {
    let k = m.k();
    let mut one = vec![0u64; k];
    one[0] = 1;
    let (mut q, mut r) =
        mmul::div_rem_school(a, h, &one, &m.d, m.l.as_ref().expect("several limbs"));
    trim(&mut r, k);
    if want_q {
        trim(&mut q, k);
    } else {
        q.clear();
    }
    (q, r)
}

/// The quotient and the remainder of the packed `a` by `b`, whose leading coefficient is a unit:
/// the delayed-reduction schoolbook division for a short quotient or divisor, Newton's method
/// beyond (the inverse of the reversed `b` modulo `x^lq`, only as long as the quotient).
pub(crate) fn divrem(m: &Modulus, a: &[u64], b: &[u64]) -> (Vec<u64>, Vec<u64>) {
    let k = m.k();
    let n = b.len() / k;
    assert!(n > 0, "division by the zero polynomial");
    if let Some(w) = m.word_mod() {
        return nmod::divrem(w, a, b);
    }
    let inv = m
        .inv(&b[(n - 1) * k..])
        .expect("the leading coefficient is not a unit modulo n");
    let la = a.len() / k;
    if la < n {
        let mut r = a.to_vec();
        trim(&mut r, k);
        return (Vec::new(), r);
    }
    let lq = la + 1 - n;
    if lq < NEWTON_DIV || n < NEWTON_DIV {
        let (mut q, mut r) =
            mmul::div_rem_school(a, b, &inv, &m.d, m.l.as_ref().expect("several limbs"));
        trim(&mut q, k);
        trim(&mut r, k);
        return (q, r);
    }
    // The reversed b's first lq coefficients, monic, inverted modulo x^lq.
    let bm = scale(m, b, &inv);
    let rev: Vec<u64> = bm
        .chunks_exact(k)
        .rev()
        .take(lq)
        .flatten()
        .copied()
        .collect();
    let mut binv = inv_series(m, &rev, lq);
    binv.resize(lq * k, 0);
    let top: Vec<u64> = a[(n - 1) * k..]
        .chunks_exact(k)
        .rev()
        .flatten()
        .copied()
        .collect();
    let mut qr = mul_raw(m, &top, Some(&binv));
    qr.truncate(lq * k);
    let qm: Vec<u64> = qr.chunks_exact(k).rev().flatten().copied().collect();
    // r = a - qm·bm, its low n - 1 coefficients.
    let mut r = a[..(n - 1) * k].to_vec();
    if n > 1 {
        let prod = mul_raw(m, &qm, Some(&bm[..(n - 1) * k]));
        for (x, y) in r.chunks_exact_mut(k).zip(prod.chunks_exact(k)) {
            m.sub(x, y);
        }
    }
    trim(&mut r, k);
    let mut q = scale(m, &qm, &inv);
    trim(&mut q, k);
    (q, r)
}

/// The packed `a` times the residue `c`.
pub(crate) fn scale(m: &Modulus, a: &[u64], c: &[u64]) -> Vec<u64> {
    let k = m.k();
    if c[0] == 1 && c[1..].iter().all(|&l| l == 0) {
        return a.to_vec();
    }
    let mut out = vec![0u64; a.len()];
    let mut work = Work::default();
    for (o, x) in out.chunks_exact_mut(k).zip(a.chunks_exact(k)) {
        m.mul(o, x, c, &mut work);
    }
    trim(&mut out, k);
    out
}

/// Up to this many coefficients in the shorter operand, [`gcd`] keeps this module's Euclid: the
/// gcd area's engine costs its setting up there (measured 15% at 4 coefficients, even from 8).
const OWN_GCD: usize = 6;

/// The monic greatest common divisor of the packed `a` and `b`, modulo a prime.
///
/// The gcd seam (moduli of several limbs): the gcd area's Euclid (fraction-free steps for short
/// divisors, the arithmetic area's delayed-reduction remainders otherwise) and half gcd, except
/// for the shortest operands.
pub(crate) fn gcd(m: &Modulus, a: &[u64], b: &[u64]) -> Vec<u64> {
    if let Some(w) = m.word_mod() {
        return nmod::gcd(w, a, b);
    }
    if a.len().min(b.len()) <= OWN_GCD * m.k() {
        return euclid(m, a, b);
    }
    super::super::gcd::limbs_gcd(m, a, b)
}

/// The monic gcd by Euclid's algorithm on this module's division.
fn euclid(m: &Modulus, a: &[u64], b: &[u64]) -> Vec<u64> {
    let k = m.k();
    let (mut a, mut b) = if a.len() >= b.len() {
        (a.to_vec(), b.to_vec())
    } else {
        (b.to_vec(), a.to_vec())
    };
    while !b.is_empty() {
        let r = divrem(m, &a, &b).1;
        a = b;
        b = r;
    }
    if a.is_empty() {
        return a;
    }
    let inv = m
        .inv(&a[a.len() - k..])
        .expect("the leading coefficient is not a unit modulo n");
    scale(m, &a, &inv)
}

/// `a + b` of the packed `a` and `b`, trimmed.
pub(crate) fn add(m: &Modulus, a: &[u64], b: &[u64]) -> Vec<u64> {
    let (a, b) = if a.len() >= b.len() { (a, b) } else { (b, a) };
    let mut r = a.to_vec();
    let k = m.k();
    for (x, y) in r.chunks_exact_mut(k).zip(b.chunks_exact(k)) {
        m.add(x, y);
    }
    trim(&mut r, k);
    r
}

/// `a - b` of the packed `a` and `b`, trimmed.
pub(crate) fn sub(m: &Modulus, a: &[u64], b: &[u64]) -> Vec<u64> {
    let mut r = a.to_vec();
    if r.len() < b.len() {
        r.resize(b.len(), 0);
    }
    let k = m.k();
    for (x, y) in r.chunks_exact_mut(k).zip(b.chunks_exact(k)) {
        m.sub(x, y);
    }
    trim(&mut r, k);
    r
}
