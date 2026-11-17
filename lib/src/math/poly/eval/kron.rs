//! Products of polynomials modulo `n` on packed residues, for the trees of this module, on the
//! polynomials' fast arithmetic: modulo a word the products of [`wmul`] (the schoolbook method,
//! Kronecker substitution, the transforms) and middle products by the cyclic transforms;
//! modulo several limbs the products of [`mmul`] (Kronecker substitution with one division a
//! coefficient) and the schoolbook method for the coefficients wanted.
//!
//! A middle product, the coefficients `lo..hi` of `a·b`, is the cyclic product modulo `x^L - 1`
//! for the least power of two `L` at least `hi`, the operands' lengths and `la + lb - 1 - lo`:
//! every coefficient past the `L`-th then wraps around below `lo`. In the down pass of the
//! scaled remainder tree, `σ_v` of `2^ℓ` coefficients times the half `M_w` of `2^(ℓ-1)`, that is
//! half the transform of the full product.

use super::word::WordMod;
use crate::math::numbers::nat;
use crate::math::poly::arith::lred::LimbRed;
use crate::math::poly::arith::{mmul, wmul};

/// Up to this many coefficients in the shorter operand, a middle product modulo several limbs
/// is made coefficient by coefficient (only the wanted ones).
const SCHOOL_MULTI: usize = 16;

/// The products modulo `n`.
pub(crate) struct Kron<'a> {
    /// The limbs of a residue.
    k: usize,
    /// The limbs of `n`.
    d: &'a [u64],
    /// `n` as a word, when it is one.
    word: Option<WordMod>,
    /// The reversed operand of the schoolbook middle products, and the cyclic product's room.
    rev: Vec<u64>,
    /// A sum of products, a product and the scratch of the residues of several limbs, and their
    /// reduction.
    acc: Vec<u64>,
    t: Vec<u64>,
    s: Vec<u64>,
    lm: Option<LimbRed>,
}

impl<'a> Kron<'a> {
    /// The products modulo the `n ≥ 2` of the limbs `d`, the top one not zero.
    pub(crate) fn new(d: &'a [u64]) -> Kron<'a> {
        let k = d.len();
        Kron {
            k,
            d,
            word: (k == 1).then(|| WordMod::new(d[0])),
            rev: Vec::new(),
            acc: Vec::new(),
            t: Vec::new(),
            s: Vec::new(),
            lm: (k >= 2).then(|| LimbRed::new(d)),
        }
    }

    /// Sets `out` to the product `a·b`, `la + lb - 1` residues of `k` limbs; `a` and `b` packed
    /// residues, at least one each.
    pub(crate) fn mul(&mut self, a: &[u64], b: &[u64], out: &mut [u64]) {
        let k = self.k;
        debug_assert!(out.len() == (a.len() / k + b.len() / k - 1) * k);
        if let Some(m) = &self.word {
            wmul::mul_into(out, a, b, m);
            return;
        }
        let red = self.lm.as_ref().expect("several limbs");
        out.copy_from_slice(&mmul::mul(a, b, self.d, red));
    }

    /// Sets `out` to the product of the monic `a` and `b` (`la, lb ≥ 2` coefficients, the top
    /// ones 1): `(x^da + a')(x^db + b') = x^(da+db) + x^da b' + x^db a' + a' b'`, the product of
    /// the parts below the tops, whose transform is half as long for lengths of powers of two.
    pub(crate) fn mul_monic(&mut self, a: &[u64], b: &[u64], out: &mut [u64]) {
        let k = self.k;
        let (da, db) = (a.len() / k - 1, b.len() / k - 1);
        debug_assert!(out.len() == (da + db + 1) * k);
        let (lo, top) = out.split_at_mut((da + db - 1) * k);
        self.mul(&a[..da * k], &b[..db * k], lo);
        top.fill(0);
        top[k] = 1;
        self.add_at(out, da, &b[..db * k]);
        self.add_at(out, db, &a[..da * k]);
    }

    /// `out[at..] += x`, residues.
    #[inline]
    pub(crate) fn add_at(&self, out: &mut [u64], at: usize, x: &[u64]) {
        let k = self.k;
        if let Some(m) = &self.word {
            for (o, &v) in out[at..].iter_mut().zip(x) {
                *o = m.add(*o, v);
            }
            return;
        }
        for (o, v) in out[at * k..].chunks_exact_mut(k).zip(x.chunks_exact(k)) {
            let carry = nat::add_assign(o, v);
            if carry || !lt(o, self.d) {
                nat::sub_assign(o, self.d);
            }
        }
    }

    /// Sets `out` to the coefficients `lo..hi` of `a·b`, `hi - lo` residues of `k` limbs, 0 past
    /// the product's last; `a` and `b` packed residues, at least one each.
    pub(crate) fn mul_range(
        &mut self,
        a: &[u64],
        b: &[u64],
        lo: usize,
        hi: usize,
        out: &mut [u64],
    ) {
        let k = self.k;
        let (la, lb) = (a.len() / k, b.len() / k);
        debug_assert!(la > 0 && lb > 0 && out.len() == (hi - lo) * k);
        let total = la + lb - 1;
        if lo >= total {
            out.fill(0);
            return;
        }
        let end = hi.min(total);
        out[(end - lo) * k..].fill(0);
        let out = &mut out[..(end - lo) * k];
        if lo == 0 && end == total {
            self.mul(a, b, out);
            return;
        }
        if let Some(m) = self.word {
            match wmul::method(la, lb, &m) {
                wmul::Method::School => {
                    self.school_word(a, b, lo, end, out, &m);
                    return;
                }
                wmul::Method::Transforms => {
                    let need = end.max(total - lo).max(la).max(lb);
                    let lk = usize::BITS - (need - 1).leading_zeros();
                    if (1usize << lk) < total
                        && (wmul::NEON || cfg!(miri))
                        && let Some(p) = nat::word_poly_plan_cyclic(la.min(lb), lk, m.bits())
                    {
                        nat::word_poly_mul_cyclic_range(out, a, b, m.n, p, lo);
                        return;
                    }
                }
                wmul::Method::Kronecker => {}
            }
            let mut p = core::mem::take(&mut self.rev);
            p.clear();
            p.resize(total, 0);
            wmul::mul_into(&mut p, a, b, &m);
            out.copy_from_slice(&p[lo..end]);
            self.rev = p;
            return;
        }
        if la.min(lb) <= if cfg!(miri) { 4 } else { SCHOOL_MULTI } {
            self.school(a, b, lo, end, out);
            return;
        }
        let p = mmul::mul(a, b, self.d, self.lm.as_ref().expect("several limbs"));
        out.copy_from_slice(&p[lo * k..end * k]);
    }

    /// The coefficients `lo..hi` (all below the product's length) of `a·b` modulo the word `m`,
    /// each a sum of products reduced once ([`wmul::dot_mod`]) against `b` reversed.
    fn school_word(
        &mut self,
        a: &[u64],
        b: &[u64],
        lo: usize,
        hi: usize,
        out: &mut [u64],
        m: &WordMod,
    ) {
        let (la, lb) = (a.len(), b.len());
        let rev = &mut self.rev;
        rev.clear();
        rev.extend(b.iter().rev());
        for (j, o) in (lo..hi).zip(out.iter_mut()) {
            // Σ a_i b_(j-i) for i0 ≤ i ≤ i1; b_(j-i) is rev[lb - 1 - j + i].
            let i0 = j.saturating_sub(lb - 1);
            let i1 = j.min(la - 1);
            let r0 = lb - 1 - (j - i0);
            *o = wmul::dot_mod(&a[i0..=i1], &rev[r0..r0 + (i1 - i0 + 1)], m);
        }
    }

    /// The coefficients `lo..hi` of `a·b` modulo several limbs, coefficient by coefficient: the
    /// products summed in `2k + 2` limbs, then one reduction each.
    fn school(&mut self, a: &[u64], b: &[u64], lo: usize, hi: usize, out: &mut [u64]) {
        let k = self.k;
        let (la, lb) = (a.len() / k, b.len() / k);
        for (j, o) in (lo..hi).zip(out.chunks_exact_mut(k)) {
            let i0 = j.saturating_sub(lb - 1);
            let i1 = j.min(la - 1);
            if i0 > i1 {
                o.fill(0);
                continue;
            }
            let mut acc = core::mem::take(&mut self.acc);
            acc.clear();
            // A sum of fewer than 2^64 products below 2^128k fits 2k + 1 limbs.
            acc.resize(2 * k + 1, 0);
            self.t.resize(2 * k, 0);
            for i in i0..=i1 {
                let (x, y) = (&a[i * k..(i + 1) * k], &b[(j - i) * k..(j - i + 1) * k]);
                nat::mul_to(&mut self.t, x, y, &mut self.s);
                nat::add_assign(&mut acc, &self.t);
            }
            self.reduce(&acc, o);
            self.acc = acc;
        }
    }

    /// The residue of the limbs `x` into the `k` limbs `out`, modulo several limbs.
    #[inline]
    fn reduce(&mut self, x: &[u64], out: &mut [u64]) {
        self.lm.as_ref().expect("several limbs").reduce(x, out);
    }
}

/// Whether `a < b`, both of the same length.
#[inline]
fn lt(a: &[u64], b: &[u64]) -> bool {
    for (x, y) in a.iter().rev().zip(b.iter().rev()) {
        if x != y {
            return x < y;
        }
    }
    false
}
