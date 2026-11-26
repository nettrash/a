//! The subproduct tree of points modulo `n` on packed residues, and the algorithms on it:
//! the product of the linear factors, the values of a polynomial at the points by the scaled
//! remainder tree (the transposed algorithm of Bostan, Lecerf and Schost, Bernstein's scaled
//! remainders), and the linear combination of interpolation going up the tree.
//!
//! The tree of `m` points `x₀ … x_{m-1}` has the levels `ℓ = 1 … h`, `2^h ≥ m`: the node `j` of the
//! level `ℓ` is the monic `Π (x - xᵢ)` over the leaves `j·2^ℓ ≤ i < min((j + 1)·2^ℓ, m)`, held in
//! one vector a level, `2^ℓ + 1` coefficients apart. The leaves themselves, the linear factors, are
//! only the points.

use super::kron::Kron;
use super::mont::MontCtx;
use super::word::{WordMod, mul_shoup_any};
use crate::math::poly::ModPolynomial;
use crate::math::poly::arith::wrem;
use crate::math::poly::modular::{Modulus, Work};
use std::sync::Arc;

/// The residue arithmetic and the products the trees need, with their working memory.
pub(crate) struct Ctx<'a> {
    /// The modulus.
    pub(crate) m: &'a Arc<Modulus>,
    /// The limbs of a residue.
    pub(crate) k: usize,
    /// The modulus as a word, when it is one.
    pub(crate) word: Option<WordMod>,
    /// The products of polynomials.
    pub(crate) kr: Kron<'a>,
    /// Montgomery's products, for an odd `n` of several limbs (the modulus's own).
    mc: Option<&'a MontCtx>,
    /// The scratch of the products of residues.
    w: Work,
    /// A residue.
    t: Vec<u64>,
}

impl<'a> Ctx<'a> {
    /// The arithmetic modulo `m`.
    pub(crate) fn new(m: &'a Arc<Modulus>) -> Ctx<'a> {
        let k = m.k();
        Ctx {
            m,
            k,
            word: m.word_mod().copied(),
            kr: Kron::new(m),
            mc: m.mont(),
            w: Work::default(),
            t: Vec::new(),
        }
    }

    /// `out = a·b`.
    #[inline]
    pub(crate) fn mul(&mut self, out: &mut [u64], a: &[u64], b: &[u64]) {
        match (&self.word, &self.mc) {
            (Some(m), _) => out[0] = m.mul(a[0], b[0]),
            (None, Some(mc)) => mc.mul(out, a, b),
            (None, None) => self.m.mul(out, a, b, &mut self.w),
        }
    }

    /// `r += a·b`.
    #[inline]
    pub(crate) fn mul_add(&mut self, r: &mut [u64], a: &[u64], b: &[u64]) {
        match &self.word {
            Some(m) => r[0] = m.add(r[0], m.mul(a[0], b[0])),
            None => {
                let mut t = core::mem::take(&mut self.t);
                t.resize(self.k, 0);
                self.mul(&mut t, a, b);
                self.m.add(r, &t);
                self.t = t;
            }
        }
    }

    /// `r -= a·b`.
    #[inline]
    pub(crate) fn mul_sub(&mut self, r: &mut [u64], a: &[u64], b: &[u64]) {
        match &self.word {
            Some(m) => r[0] = m.sub(r[0], m.mul(a[0], b[0])),
            None => {
                let mut t = core::mem::take(&mut self.t);
                t.resize(self.k, 0);
                self.mul(&mut t, a, b);
                self.m.sub(r, &t);
                self.t = t;
            }
        }
    }

    /// The factor of `a` for [`mul_sub_factor`](Ctx::mul_sub_factor): `a R mod n` for
    /// Montgomery's products, `a` itself otherwise.
    pub(crate) fn factor(&self, a: &[u64]) -> Vec<u64> {
        let mut f = a.to_vec();
        if let Some(mc) = &self.mc {
            mc.factor(&mut f, a);
        }
        f
    }

    /// `r -= x·a` for the [`factor`](Ctx::factor) `af` of `a`: one Montgomery product.
    #[inline]
    pub(crate) fn mul_sub_factor(&mut self, r: &mut [u64], x: &[u64], af: &[u64]) {
        match &self.mc {
            Some(mc) => {
                let mut t = core::mem::take(&mut self.t);
                t.resize(self.k, 0);
                mc.mul_factor(&mut t, x, af);
                self.m.sub(r, &t);
                self.t = t;
            }
            None => self.mul_sub(r, x, af),
        }
    }

    /// `r += a`.
    #[inline]
    pub(crate) fn add(&self, r: &mut [u64], a: &[u64]) {
        self.m.add(r, a);
    }

    /// `r -= a`.
    #[inline]
    pub(crate) fn sub(&self, r: &mut [u64], a: &[u64]) {
        self.m.sub(r, a);
    }

    /// `r = -r`.
    #[inline]
    pub(crate) fn neg(&self, r: &mut [u64]) {
        self.m.neg(r);
    }

    /// The residue of the word `x`, into `out`.
    #[inline]
    pub(crate) fn small(&mut self, out: &mut [u64], x: u64) {
        out.fill(0);
        match &self.word {
            Some(m) => out[0] = if x < m.n { x } else { x % m.n },
            None => out[0] = x,
        }
    }

    /// The inverses of the `len` residues `a`, in place, by Montgomery's trick; `Err(i)` with the
    /// first `i` that is not a unit.
    pub(crate) fn batch_inv(&mut self, a: &mut [u64]) -> Result<(), usize> {
        let k = self.k;
        if let Some(m) = &self.word {
            return super::word::batch_inv(m, a);
        }
        let len = a.len() / k;
        if len == 0 {
            return Ok(());
        }
        let mut pre = vec![0u64; len * k];
        let mut acc = vec![0u64; k];
        acc[0] = 1;
        let mut t = vec![0u64; k];
        for i in 0..len {
            let x = &a[i * k..(i + 1) * k];
            if x.iter().all(|&l| l == 0) {
                return Err(i);
            }
            pre[i * k..(i + 1) * k].copy_from_slice(&acc);
            self.mul(&mut t, &acc, x);
            acc.copy_from_slice(&t);
        }
        let Some(mut inv) = self.m.inv(&acc) else {
            let i = (0..len)
                .find(|&i| self.m.inv(&a[i * k..(i + 1) * k]).is_none())
                .unwrap_or(0);
            return Err(i);
        };
        let mut x = vec![0u64; k];
        for i in (0..len).rev() {
            x.copy_from_slice(&a[i * k..(i + 1) * k]);
            self.mul(&mut t, &inv, &pre[i * k..(i + 1) * k]);
            a[i * k..(i + 1) * k].copy_from_slice(&t);
            self.mul(&mut t, &inv, &x);
            inv.copy_from_slice(&t);
        }
        Ok(())
    }

    /// The first `prec` coefficients of the power series inverse of the packed `r`, whose constant
    /// is 1: the polynomials' Newton iteration (modulo a word [`wrem::inv_series`], otherwise
    /// [`ModPolynomial::inv_series`]).
    pub(crate) fn inv_series(&mut self, r: &[u64], prec: usize) -> Vec<u64> {
        let k = self.k;
        let mut g = match &self.word {
            Some(m) => wrem::inv_series(r, prec, 1, m),
            None => ModPolynomial::of(self.m.clone(), r.to_vec())
                .inv_series(prec)
                .c
                .into_vec(),
        };
        g.resize(prec * k, 0);
        g
    }
}

/// The number of nodes of the level `l` of the tree of `m` points.
#[inline]
fn nodes(m: usize, l: usize) -> usize {
    m.div_ceil(1 << l)
}

/// The degree of the node `j` of the level `l` of the tree of `m` points.
#[inline]
fn deg(m: usize, l: usize, j: usize) -> usize {
    (((j + 1) << l).min(m)) - (j << l)
}

/// The subproduct tree of `m` points.
pub(crate) struct Tree {
    /// The number of points.
    pub(crate) m: usize,
    /// The points, packed.
    pub(crate) pts: Vec<u64>,
    /// The levels `1..=h` (the first is empty): the node `j` of the level `ℓ`, `deg + 1`
    /// coefficients from `j·(2^ℓ + 1)`.
    lv: Vec<Vec<u64>>,
    /// The height, `2^h ≥ m`.
    h: usize,
}

impl Tree {
    /// The packed coefficients of the node `j` of the level `l ≥ 1`.
    #[inline]
    fn node(&self, k: usize, l: usize, j: usize) -> &[u64] {
        let off = j * ((1 << l) + 1) * k;
        &self.lv[l][off..off + (deg(self.m, l, j) + 1) * k]
    }

    /// The monic product of all the linear factors, `m + 1` coefficients.
    pub(crate) fn root(&self, ctx: &Ctx<'_>) -> Vec<u64> {
        let k = ctx.k;
        if self.h == 0 {
            let mut r = vec![0u64; 2 * k];
            r[..k].copy_from_slice(&self.pts[..k]);
            ctx.neg(&mut r[..k]);
            r[k] = 1;
            return r;
        }
        self.node(k, self.h, 0).to_vec()
    }
}

/// The tree of the packed points `pts`, at least one, all its levels.
pub(crate) fn build(ctx: &mut Ctx<'_>, pts: &[u64]) -> Tree {
    let k = ctx.k;
    let m = pts.len() / k;
    debug_assert!(m >= 1);
    let mut h = 0;
    while (1usize << h) < m {
        h += 1;
    }
    let mut lv: Vec<Vec<u64>> = vec![Vec::new()];
    if h >= 1 {
        // Level 1: (x - a)(x - b) = x² - (a + b)x + ab, or a lone x - a.
        let mut v = vec![0u64; nodes(m, 1) * 3 * k];
        for (j, o) in v.chunks_exact_mut(3 * k).enumerate() {
            let a = &pts[2 * j * k..(2 * j + 1) * k];
            if 2 * j + 1 < m {
                let b = &pts[(2 * j + 1) * k..(2 * j + 2) * k];
                let (c0, rest) = o.split_at_mut(k);
                let (c1, c2) = rest.split_at_mut(k);
                ctx.mul(c0, a, b);
                c1.copy_from_slice(a);
                ctx.add(c1, b);
                ctx.neg(c1);
                c2[0] = 1;
            } else {
                let (c0, c1) = o.split_at_mut(k);
                c0.copy_from_slice(a);
                ctx.neg(c0);
                c1[0] = 1;
            }
        }
        lv.push(v);
    }
    for l in 1..h {
        let (cs, ps) = ((1usize << l) + 1, (1usize << (l + 1)) + 1);
        let mut v = vec![0u64; nodes(m, l + 1) * ps * k];
        {
            let below = &lv[l];
            let nb = nodes(m, l);
            for (j, o) in v.chunks_mut(ps * k).enumerate() {
                let (du, u) = (deg(m, l, 2 * j), 2 * j * cs * k);
                let a = &below[u..u + (du + 1) * k];
                if 2 * j + 1 < nb {
                    let dw = deg(m, l, 2 * j + 1);
                    let w = (2 * j + 1) * cs * k;
                    let b = &below[w..w + (dw + 1) * k];
                    ctx.kr.mul_monic(a, b, &mut o[..(du + dw + 1) * k]);
                } else {
                    o[..a.len()].copy_from_slice(a);
                }
            }
        }
        lv.push(v);
    }
    Tree {
        m,
        pts: pts.to_vec(),
        lv,
        h,
    }
}

/// The product `Π (x - xᵢ)` of the packed points `pts`, `m + 1` coefficients, monic: blocks of
/// [`LINEAR_ROOTS`] roots one linear factor after the other, then their product tree.
pub(crate) fn from_roots(ctx: &mut Ctx<'_>, pts: &[u64]) -> Vec<u64> {
    let k = ctx.k;
    let m = pts.len() / k;
    let block = match ctx.word {
        _ if cfg!(miri) => 4,
        Some(w) if w.n < 1 << 30 => LINEAR_ROOTS.0,
        Some(_) => LINEAR_ROOTS.1,
        None => LINEAR_ROOTS.2,
    };
    if m <= block {
        // One block: its linear factors one after the other.
        return linear(ctx, pts);
    }
    let mut level: Vec<Vec<u64>> = if m == 0 {
        vec![linear(ctx, pts)]
    } else {
        pts.chunks(block * k).map(|b| linear(ctx, b)).collect()
    };
    while level.len() > 1 {
        let mut next = Vec::with_capacity(level.len().div_ceil(2));
        for pair in level.chunks(2) {
            match pair {
                [a, b] => {
                    let len = a.len() / k + b.len() / k - 1;
                    let mut o = vec![0u64; len * k];
                    ctx.kr.mul_monic(a, b, &mut o);
                    next.push(o);
                }
                [a] => next.push(a.clone()),
                _ => unreachable!(),
            }
        }
        level = next;
    }
    level.pop().expect("one product")
}

/// `Π (x - xᵢ)` one linear factor after the other, `r ← r·(x - a)` from the top down: modulo a
/// word by Shoup's products by `a`.
fn linear(ctx: &mut Ctx<'_>, pts: &[u64]) -> Vec<u64> {
    let k = ctx.k;
    let m = pts.len() / k;
    let mut r = vec![0u64; (m + 1) * k];
    r[0] = 1;
    if let Some(wm) = ctx.word {
        for (i, &a) in pts.iter().enumerate() {
            // r[j] = r[j-1] - a·r[j], r[0] = -a·r[0], r[i+1] = r[i] (= 1).
            r[i + 1] = r[i];
            let (na, nap) = (wm.neg(a), wm.shoup(wm.neg(a)));
            for j in (1..=i).rev() {
                r[j] = wm.add(r[j - 1], mul_shoup_any(&wm, r[j], na, nap));
            }
            r[0] = mul_shoup_any(&wm, r[0], na, nap);
        }
        return r;
    }
    let mut t = vec![0u64; k];
    if let Some(mc) = &ctx.mc {
        // r[j] = r[j-1] + r[j]·(-a), the factor of -a made once: one Montgomery product a step.
        let mut na = vec![0u64; k];
        for i in 0..m {
            na.copy_from_slice(&pts[i * k..(i + 1) * k]);
            ctx.m.neg(&mut na);
            mc.factor(&mut t, &na);
            let naf = t.clone();
            r.copy_within(i * k..(i + 1) * k, (i + 1) * k);
            for j in (0..=i).rev() {
                let (lo, hi) = r.split_at_mut(j * k);
                let rj = &mut hi[..k];
                mc.mul_factor(&mut na, rj, &naf);
                rj.copy_from_slice(&na);
                if j > 0 {
                    ctx.m.add(rj, &lo[(j - 1) * k..]);
                }
            }
        }
        return r;
    }
    for i in 0..m {
        let a = &pts[i * k..(i + 1) * k];
        r.copy_within(i * k..(i + 1) * k, (i + 1) * k);
        for j in (1..=i).rev() {
            t.copy_from_slice(&r[j * k..(j + 1) * k]);
            let (lo, hi) = r.split_at_mut(j * k);
            hi[..k].copy_from_slice(&lo[(j - 1) * k..]);
            ctx.mul_sub(&mut hi[..k], a, &t);
        }
        t.copy_from_slice(&r[..k]);
        r[..k].fill(0);
        ctx.mul_sub(&mut r[..k], a, &t);
    }
    r
}

/// The roots a block of [`from_roots`] takes one after the other, modulo a word below `2^30`, a
/// larger word and several limbs (measured on an Apple M4 against the tree's products).
const LINEAR_ROOTS: (usize, usize, usize) = (16, 32, 16);

impl Tree {
    /// The scaled remainders of the root for the packed polynomial `f` of `len ≥ 1` coefficients:
    /// `σ[t]`, `t < m`, the coefficient of `x^-(m - t)` in the expansion in `1/x` of
    /// `(f mod M) / M`, `M` the root; that is `(F·I)[len - 1 - t]` for `F` the reversal of `f` and
    /// `I` the power series inverse, to `len` terms, of the reversal of `M`.
    fn top(&self, ctx: &mut Ctx<'_>, f: &[u64]) -> Vec<u64> {
        let k = ctx.k;
        let (m, len) = (self.m, f.len() / k);
        let root = self.root(ctx);
        // The reversal of M (monic, so its constant is 1), cut to `len` terms.
        let rl = (m + 1).min(len);
        let mut rev = vec![0u64; rl * k];
        for i in 0..rl {
            rev[i * k..(i + 1) * k].copy_from_slice(&root[(m - i) * k..(m - i + 1) * k]);
        }
        let inv = ctx.inv_series(&rev, len);
        let mut frev = vec![0u64; len * k];
        for i in 0..len {
            frev[i * k..(i + 1) * k].copy_from_slice(&f[(len - 1 - i) * k..(len - i) * k]);
        }
        // The coefficients lo..len of F·I, then reversed into σ (zeros past len).
        let lo = len.saturating_sub(m);
        let mut mid = vec![0u64; (len - lo) * k];
        ctx.kr.mul_range(&frev, &inv, lo, len, &mut mid);
        let mut s = vec![0u64; m * k];
        for t in 0..m.min(len) {
            let i = len - 1 - t - lo;
            s[t * k..(t + 1) * k].copy_from_slice(&mid[i * k..(i + 1) * k]);
        }
        s
    }

    /// The values of the packed polynomial `f`, not zero, at the points, packed: the scaled
    /// remainders of the root carried down the tree by middle products,
    /// `σ_u = (σ_v · M_w)[deg w .. deg v]` for the child `u` of `v` and its sibling `w`.
    pub(crate) fn eval(&self, ctx: &mut Ctx<'_>, f: &[u64]) -> Vec<u64> {
        let k = ctx.k;
        let m = self.m;
        let mut s = self.top(ctx, f);
        let mut next = Vec::new();
        for l in (2..=self.h).rev() {
            // From the level l (nodes 2^l coefficients apart in s) to l - 1.
            let half = 1usize << (l - 1);
            next.clear();
            next.resize(nodes(m, l - 1) * half * k, 0);
            let nb = nodes(m, l - 1);
            for j in 0..nodes(m, l) {
                let dv = deg(m, l, j);
                let sv = &s[j * 2 * half * k..(j * 2 * half + dv) * k];
                let (u, w) = (2 * j, 2 * j + 1);
                let du = deg(m, l - 1, u);
                if w < nb {
                    let dw = deg(m, l - 1, w);
                    let (mu, mw) = (self.node(k, l - 1, u), self.node(k, l - 1, w));
                    let (ou, ow) = next[u * half * k..].split_at_mut(half * k);
                    // σ_v·M_w = σ_v·x^dw + σ_v·(M_w - x^dw): its coefficients dw..dv are those
                    // of σ_v below du and the middle product by the part below the top.
                    let (ou, ow) = (&mut ou[..du * k], &mut ow[..dw * k]);
                    ctx.kr.mul_range(sv, &mw[..dw * k], dw, dv, ou);
                    ctx.kr.add_at(ou, 0, &sv[..du * k]);
                    ctx.kr.mul_range(sv, &mu[..du * k], du, dv, ow);
                    ctx.kr.add_at(ow, 0, &sv[..dw * k]);
                } else {
                    next[u * half * k..(u * half + du) * k].copy_from_slice(sv);
                }
            }
            core::mem::swap(&mut s, &mut next);
        }
        if self.h == 0 {
            return s;
        }
        // The level 1: f(a) = σ₀ - b·σ₁ and f(b) = σ₀ - a·σ₁ for the node (x - a)(x - b).
        let mut out = vec![0u64; m * k];
        if let Some(wm) = ctx.word {
            for j in 0..nodes(m, 1) {
                let (s0, a) = (s[2 * j], self.pts[2 * j]);
                if 2 * j + 1 < m {
                    let (s1, b) = (s[2 * j + 1], self.pts[2 * j + 1]);
                    out[2 * j] = wm.sub(s0, wm.mul(b, s1));
                    out[2 * j + 1] = wm.sub(s0, wm.mul(a, s1));
                } else {
                    out[2 * j] = s0;
                }
            }
            return out;
        }
        for j in 0..nodes(m, 1) {
            let i0 = 2 * j;
            out[i0 * k..(i0 + 1) * k].copy_from_slice(&s[i0 * k..(i0 + 1) * k]);
            if i0 + 1 < m {
                out[(i0 + 1) * k..(i0 + 2) * k].copy_from_slice(&s[i0 * k..(i0 + 1) * k]);
                let s1 = s[(i0 + 1) * k..(i0 + 2) * k].to_vec();
                let (a, b) = (
                    &self.pts[i0 * k..(i0 + 1) * k],
                    &self.pts[(i0 + 1) * k..(i0 + 2) * k],
                );
                let (oa, ob) = out[i0 * k..(i0 + 2) * k].split_at_mut(k);
                ctx.mul_sub(oa, b, &s1);
                ctx.mul_sub(ob, a, &s1);
            }
        }
        out
    }

    /// `Σ cᵢ · M / (x - xᵢ)` for the packed `c`, one a point, `m` coefficients: going up the
    /// tree, `r_v = r_u · M_w + r_w · M_u` for the children `u` and `w` of `v`.
    pub(crate) fn combine(&self, ctx: &mut Ctx<'_>, c: &[u64]) -> Vec<u64> {
        let k = ctx.k;
        let m = self.m;
        if self.h == 0 {
            return c.to_vec();
        }
        // The level 1: c_a (x - b) + c_b (x - a) = (c_a + c_b) x - (c_a b + c_b a).
        let mut r = vec![0u64; nodes(m, 1) * 2 * k];
        for j in 0..nodes(m, 1) {
            let (i0, o) = (2 * j, &mut r[2 * j * k..(2 * j + 2) * k]);
            let ca = &c[i0 * k..(i0 + 1) * k];
            if i0 + 1 < m {
                let cb = &c[(i0 + 1) * k..(i0 + 2) * k];
                let (a, b) = (
                    &self.pts[i0 * k..(i0 + 1) * k],
                    &self.pts[(i0 + 1) * k..(i0 + 2) * k],
                );
                let (o0, o1) = o.split_at_mut(k);
                ctx.mul_sub(o0, ca, b);
                ctx.mul_sub(o0, cb, a);
                o1.copy_from_slice(ca);
                ctx.add(o1, cb);
            } else {
                o[..k].copy_from_slice(ca);
            }
        }
        let mut next = Vec::new();
        let mut t = Vec::new();
        for l in 1..self.h {
            let half = 1usize << l;
            next.clear();
            next.resize(nodes(m, l + 1) * 2 * half * k, 0);
            let nb = nodes(m, l);
            for j in 0..nodes(m, l + 1) {
                let (u, w) = (2 * j, 2 * j + 1);
                let du = deg(m, l, u);
                let ru = &r[u * half * k..(u * half + du) * k];
                let o = &mut next[j * 2 * half * k..];
                if w < nb {
                    let dw = deg(m, l, w);
                    let dv = du + dw;
                    let rw = &r[w * half * k..(w * half + dw) * k];
                    let (mu, mw) = (self.node(k, l, u), self.node(k, l, w));
                    // r_u·(x^dw + w') + r_w·(x^du + u'), the products of the parts below the
                    // tops, dv - 1 coefficients each.
                    let o = &mut o[..dv * k];
                    o[(dv - 1) * k..].fill(0);
                    ctx.kr.mul(ru, &mw[..dw * k], &mut o[..(dv - 1) * k]);
                    t.clear();
                    t.resize((dv - 1) * k, 0);
                    ctx.kr.mul(rw, &mu[..du * k], &mut t);
                    ctx.kr.add_at(o, 0, &t);
                    ctx.kr.add_at(o, dw, ru);
                    ctx.kr.add_at(o, du, rw);
                } else {
                    o[..du * k].copy_from_slice(ru);
                }
            }
            core::mem::swap(&mut r, &mut next);
        }
        r.truncate(m * k);
        r
    }
}

/// The derivative of the packed `f`.
pub(crate) fn derivative(ctx: &mut Ctx<'_>, f: &[u64]) -> Vec<u64> {
    let k = ctx.k;
    let len = f.len() / k;
    if len <= 1 {
        return Vec::new();
    }
    let mut out = vec![0u64; (len - 1) * k];
    let mut t = vec![0u64; k];
    for i in 1..len {
        // i mod n, then the product.
        if let Some(wm) = ctx.word {
            out[i - 1] = wm.mul(
                if (i as u64) < wm.n {
                    i as u64
                } else {
                    i as u64 % wm.n
                },
                f[i],
            );
        } else {
            ctx.small(&mut t, i as u64);
            let tt = t.clone();
            ctx.mul(&mut out[(i - 1) * k..i * k], &tt, &f[i * k..(i + 1) * k]);
        }
    }
    out
}
