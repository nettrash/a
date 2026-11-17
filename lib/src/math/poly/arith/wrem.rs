//! Remainders, products and powers modulo a fixed polynomial `f` modulo a word, on vectors of
//! residues reused from step to step: the schoolbook division for a short `f`, and beyond the
//! product by the inverse of the reversed `f`, computed once.

use super::nmod::WordMod;
use super::wmul::{self, add3};

/// Up to this many coefficients in `f`, remainders by the schoolbook division.
pub(crate) fn school_max() -> usize {
    if cfg!(miri) { 8 } else { 160 }
}

/// The remainders modulo `f`, of `lf ≥ 2` coefficients, its leading one a unit.
#[derive(Clone, Debug)]
pub(crate) struct WordRem {
    pub(crate) m: WordMod,
    /// The coefficients of `f`.
    pub(crate) f: Vec<u64>,
    /// The inverse of the leading coefficient, and `n` minus it with Shoup's constant.
    inv: u64,
    /// The inverse of the reversed `f` modulo `x^(lf - 1)`, empty for the schoolbook division.
    finv: Vec<u64>,
    /// The low `lf - 1` coefficients of `f` as 32-bit values for a modulus up to `2^32`.
    f32: Vec<u32>,
}

/// The working memory of the remainders and products.
#[derive(Default)]
pub(crate) struct Work {
    p: Vec<u64>,
    q: Vec<u64>,
    t: Vec<u64>,
    u: Vec<u64>,
    w: Vec<u128>,
    c: Vec<u64>,
}

impl WordRem {
    /// The remainders modulo `f` (without zeros at the top, at least 2 coefficients) whose
    /// leading coefficient has the inverse `inv`, the inverse of the reversal `finv` (modulo
    /// `x^(lf - 1)`, `None` to compute it).
    pub(crate) fn new(f: &[u64], inv: u64, m: WordMod, finv: Option<Vec<u64>>) -> WordRem {
        let lf = f.len();
        debug_assert!(lf >= 2);
        let finv = if lf <= school_max() {
            Vec::new()
        } else {
            let mut v = finv.unwrap_or_else(|| {
                let rev: Vec<u64> = f.iter().rev().copied().collect();
                inv_series(&rev, lf - 1, inv, &m)
            });
            v.resize(lf - 1, 0);
            v
        };
        let f32 = if m.n <= 1 << 32 {
            f[..lf - 1].iter().map(|&x| x as u32).collect()
        } else {
            Vec::new()
        };
        WordRem {
            m,
            f: f.to_vec(),
            inv,
            finv,
            f32,
        }
    }

    /// `a mod f` in place: `a` left with `lf - 1` coefficients (zeros at the top kept).
    pub(crate) fn rem(&self, a: &mut Vec<u64>, w: &mut Work) {
        let lf = self.f.len();
        if a.len() < lf {
            a.resize(lf - 1, 0);
            return;
        }
        if self.finv.is_empty() {
            wmul::rem_school_in(a, &self.f, &self.f32, self.inv, &self.m, &mut w.w, &mut w.c);
            return;
        }
        while a.len() > 2 * lf - 2 {
            // The top 2 lf - 2 coefficients reduced at a time.
            let cut = a.len() - (2 * lf - 2);
            let mut top = a.split_off(cut);
            self.rem_newton(&mut top, w);
            a.extend_from_slice(&top);
        }
        self.rem_newton(a, w);
    }

    /// The remainder of `a` of at most `2 lf - 2` coefficients by the inverse of the reversal:
    /// the quotient's reversal is that of `a`'s top times the inverse, modulo `x^(la - lf + 1)`.
    fn rem_newton(&self, a: &mut Vec<u64>, w: &mut Work) {
        let m = &self.m;
        let lf = self.f.len();
        let la = a.len();
        if la < lf {
            a.resize(lf - 1, 0);
            return;
        }
        let lq = la - lf + 1;
        // The top lq coefficients of a, reversed.
        w.t.clear();
        w.t.extend(a[la - lq..].iter().rev());
        w.q.resize(2 * lq - 1, 0);
        wmul::mul_into(&mut w.q, &w.t, &self.finv[..lq], m);
        // q = the reversal of the low lq coefficients.
        w.u.clear();
        w.u.extend(w.q[..lq].iter().rev());
        // r = a - q f, its low lf - 1 coefficients.
        // Modulo x^L - 1 for L >= lf: the coefficients of q f from lf - 1 on are those of a,
        // so r_j = a_j - (q f mod x^L - 1)_j + a_(j + L).
        let k = usize::BITS - (lf - 1).leading_zeros();
        if let Some(c) = wmul::mul_cyclic(&w.u, &self.f, k, m) {
            let l = 1usize << k;
            for j in 0..lf - 1 {
                let hi = if j + l < la { a[j + l] } else { 0 };
                a[j] = m.add(m.sub(a[j], c[j]), hi);
            }
            a.truncate(lf - 1);
            return;
        }
        let fl = &self.f[..lf - 1];
        w.p.resize(lq + lf - 2, 0);
        wmul::mul_into(&mut w.p, &w.u, fl, m);
        a.truncate(lf - 1);
        for (x, &y) in a.iter_mut().zip(&w.p) {
            *x = m.sub(*x, y);
        }
    }

    /// Up to this many coefficients in `f`, products modulo `f` in one pass on the stack.
    const TINY: usize = 16;

    /// `a·b mod f` into `out` (`lf - 1` coefficients) for `lf <= TINY`, `a` and `b` of fewer than
    /// `lf` coefficients, in one pass without allocating: the sums of products in three words,
    /// then the schoolbook division on the same sums (each step adding `(n - t) f` to them), and
    /// one reduction a coefficient at the end.
    fn mul_tiny(&self, out: &mut Vec<u64>, a: &[u64], b: &[u64]) {
        let m = &self.m;
        let lf = self.f.len();
        let len = a.len() + b.len() - 1;
        // Each sum takes fewer than 2 lf products: in a word when they fit.
        let sq = u128::from(m.n - 1) * u128::from(m.n - 1);
        if sq.checked_mul(2 * lf as u128).is_some_and(|b| b < 1 << 64) {
            let mut w = [0u64; 2 * Self::TINY];
            for (i, &x) in a.iter().enumerate() {
                for (o, &y) in w[i..].iter_mut().zip(b) {
                    *o += x * y;
                }
            }
            let fl = &self.f[..lf - 1];
            for top in (lf - 1..len).rev() {
                let t = m.mul(m.reduce1(w[top]), self.inv);
                if t == 0 {
                    continue;
                }
                let nt = m.n - t;
                for (o, &y) in w[top + 1 - lf..].iter_mut().zip(fl) {
                    *o += nt * y;
                }
            }
            out.clear();
            out.extend(w[..lf - 1].iter().map(|&x| m.reduce1(x)));
            return;
        }
        let mut s = [0u128; 2 * Self::TINY];
        let mut c = [0u64; 2 * Self::TINY];
        for (i, &x) in a.iter().enumerate() {
            for (j, &y) in b.iter().enumerate() {
                add3(&mut s[i + j], &mut c[i + j], u128::from(x) * u128::from(y));
            }
        }
        let fl = &self.f[..lf - 1];
        for top in (lf - 1..len).rev() {
            let v = m.reduce3(c[top], (s[top] >> 64) as u64, s[top] as u64);
            let t = m.mul(v, self.inv);
            if t == 0 {
                continue;
            }
            let nt = m.n - t;
            let base = top + 1 - lf;
            for (j, &y) in fl.iter().enumerate() {
                add3(
                    &mut s[base + j],
                    &mut c[base + j],
                    u128::from(nt) * u128::from(y),
                );
            }
        }
        out.clear();
        out.extend((0..lf - 1).map(|j| m.reduce3(c[j], (s[j] >> 64) as u64, s[j] as u64)));
    }

    /// `a·b mod f` into `out` for `a`, `b` not empty.
    pub(crate) fn mul(&self, out: &mut Vec<u64>, a: &[u64], b: &[u64], w: &mut Work) {
        let lf = self.f.len();
        if lf <= Self::TINY && a.len() < lf && b.len() < lf {
            self.mul_tiny(out, a, b);
            return;
        }
        out.resize(a.len() + b.len() - 1, 0);
        wmul::mul_into(out, a, b, &self.m);
        self.rem(out, w);
    }

    /// `a² mod f` into `out` for `a` not empty.
    pub(crate) fn sqr(&self, out: &mut Vec<u64>, a: &[u64], w: &mut Work) {
        let lf = self.f.len();
        if lf <= Self::TINY && a.len() < lf {
            self.mul_tiny(out, a, a);
            return;
        }
        out.resize(2 * a.len() - 1, 0);
        wmul::sqr_into(out, a, &self.m);
        self.rem(out, w);
    }

    /// `a·x mod f` in place, for `a` of `lf - 1` coefficients: a shift and one step of the
    /// division.
    pub(crate) fn mul_x(&self, a: &mut Vec<u64>) {
        let m = &self.m;
        let lf = self.f.len();
        a.insert(0, 0);
        let top = a.pop().expect("lf - 1 + 1 coefficients");
        if top == 0 {
            return;
        }
        let t = m.mul(top, self.inv);
        let nt = m.n - t;
        for (x, &y) in a.iter_mut().zip(&self.f[..lf - 1]) {
            *x = m.add(*x, m.mul(y, nt));
        }
    }
}

impl WordRem {
    /// `b^e mod f` for `b` reduced (below `lf - 1` coefficients, not empty), by squares and
    /// shifts for `b = x`, and sliding windows of odd powers otherwise; `lf - 1` coefficients.
    pub(crate) fn pow(&self, b: &[u64], e: &crate::math::numbers::Number) -> Vec<u64> {
        let lf = self.f.len();
        let mut w = Work::default();
        let bits = e.bits();
        let mut one = vec![0u64; lf - 1];
        one[0] = 1;
        if bits == 0 {
            return one;
        }
        let mut b = b.to_vec();
        b.resize(lf - 1, 0);
        let is_x = lf >= 3 && b[1] == 1 && b[0] == 0 && b[2..].iter().all(|&c| c == 0);
        let mut r = b.clone();
        let mut t = Vec::new();
        if is_x {
            for i in (0..bits - 1).rev() {
                self.sqr(&mut t, trimmed(&r), &mut w);
                core::mem::swap(&mut r, &mut t);
                if e.bit(i) {
                    self.mul_x(&mut r);
                }
            }
            return r;
        }
        let win: u64 = match bits {
            0..=24 => 1,
            25..=80 => 3,
            81..=240 => 4,
            241..=700 => 5,
            _ => 6,
        };
        let mut b2 = Vec::new();
        self.sqr(&mut b2, trimmed(&b), &mut w);
        let mut odd = vec![b];
        for _ in 1..(1usize << (win - 1)) {
            let mut next = Vec::new();
            self.mul(
                &mut next,
                trimmed(odd.last().expect("not empty")),
                trimmed(&b2),
                &mut w,
            );
            odd.push(next);
        }
        let mut started = false;
        let mut i = bits as i64 - 1;
        while i >= 0 {
            if !e.bit(i as u64) {
                self.sqr(&mut t, trimmed(&r), &mut w);
                core::mem::swap(&mut r, &mut t);
                i -= 1;
                continue;
            }
            let mut j = (i - win as i64 + 1).max(0);
            while !e.bit(j as u64) {
                j += 1;
            }
            let mut val = 0usize;
            for k in (j..=i).rev() {
                val = (val << 1) | usize::from(e.bit(k as u64));
            }
            if started {
                for _ in j..=i {
                    self.sqr(&mut t, trimmed(&r), &mut w);
                    core::mem::swap(&mut r, &mut t);
                }
                self.mul(&mut t, trimmed(&r), trimmed(&odd[val >> 1]), &mut w);
                core::mem::swap(&mut r, &mut t);
            } else {
                r = odd[val >> 1].clone();
                started = true;
            }
            i = j - 1;
        }
        r
    }
}

impl WordRem {
    /// `a(g) mod f` for `g` reduced (fewer than `lf` coefficients) by Brent and Kung's method on
    /// vectors of words: with `m = ⌈√len a⌉`, the powers `g^0..g^m` modulo `f`, each block of
    /// `m` coefficients of `a` evaluated as a combination of them (dot products with the powers by
    /// columns, reduced once), and the blocks joined by Horner's rule in `g^m`. `lf - 1`
    /// coefficients.
    pub(crate) fn compose(&self, a: &[u64], g: &[u64]) -> Vec<u64> {
        let m = &self.m;
        let width = self.f.len() - 1;
        let la = a.len();
        let mut r = vec![0u64; width];
        if la == 0 {
            return r;
        }
        let mut w = Work::default();
        let bm = ((la as f64).sqrt().ceil() as usize).max(1);
        // The powers by columns: cols[t bm + i] the coefficient of x^t in g^i, i < bm, and g^bm.
        let mut cols = vec![0u64; width * bm];
        cols[0] = 1;
        let mut p: Vec<u64> = g.to_vec();
        p.resize(width, 0);
        let mut next = Vec::new();
        for i in 1..=bm {
            if i > 1 {
                self.mul(&mut next, trimmed(&p), trimmed(g), &mut w);
                next.resize(width, 0);
                core::mem::swap(&mut p, &mut next);
            }
            if i < bm {
                for (t, &c) in p.iter().enumerate() {
                    cols[t * bm + i] = c;
                }
            }
        }
        let gm = p;
        let gm = trimmed(&gm);
        let blocks = la.div_ceil(bm);
        let mut acc = Vec::new();
        for j in (0..blocks).rev() {
            let cs = &a[j * bm..((j + 1) * bm).min(la)];
            // r = r g^bm + the block's combination.
            if j + 1 < blocks && r.iter().any(|&c| c != 0) {
                self.mul(&mut acc, trimmed(&r), gm, &mut w);
                acc.resize(width, 0);
                core::mem::swap(&mut r, &mut acc);
            } else {
                r.fill(0);
            }
            for (t, o) in r.iter_mut().enumerate() {
                let v = wmul::dot_mod(cs, &cols[t * bm..t * bm + cs.len()], m);
                *o = m.add(*o, v);
            }
        }
        r
    }
}

/// `a` without zeros at the top, one coefficient at least.
#[inline]
fn trimmed(a: &[u64]) -> &[u64] {
    let n = a.iter().rposition(|&c| c != 0).map_or(1, |i| i + 1);
    &a[..n.min(a.len()).max(1)]
}

/// The inverse of the power series `f` (its constant with the inverse `inv`) modulo `x^n`, by
/// the schoolbook recurrence and Newton's iteration.
pub(crate) fn inv_series(f: &[u64], n: usize, inv: u64, m: &WordMod) -> Vec<u64> {
    if n <= if cfg!(miri) { 8 } else { 160 } {
        let mut g = vec![0u64; n];
        g[0] = inv;
        let ninv = m.neg(inv);
        // rf[n - 1 - i] = f_i, so that f_(j - t) = rf[n - 1 - j + t] against g_t.
        let mut rf: Vec<u64> = vec![0u64; n];
        for i in 1..n.min(f.len()) {
            rf[n - 1 - i] = f[i];
        }
        for j in 1..n {
            let s = wmul::dot_mod(&rf[n - 1 - j..n - 1], &g[..j], m);
            g[j] = m.mul(s, ninv);
        }
        while g.last() == Some(&0) {
            g.pop();
        }
        return g;
    }
    let h = n.div_ceil(2);
    let mut g = inv_series(f, h, inv, m);
    g.resize(h, 0);
    // e = f g mod x^n; its coefficients from h on are t.
    let fl = &f[..f.len().min(n)];
    // Modulo x^L - 1 for L >= n the wrapped coefficients land below h, which are known.
    let k = usize::BITS - (n - 1).leading_zeros();
    let t: Vec<u64> = match wmul::mul_cyclic(fl, &g, k, m) {
        Some(c) => c[h..n].to_vec(),
        None => {
            let mut e = vec![0u64; fl.len() + h - 1];
            wmul::mul_into(&mut e, fl, &g, m);
            e.resize(n.max(e.len()), 0);
            e[h..n].to_vec()
        }
    };
    // u = t g mod x^(n - h).
    let tn = t.iter().rposition(|&x| x != 0).map_or(0, |i| i + 1);
    let mut out = g;
    out.resize(n, 0);
    if tn > 0 {
        let mut u = vec![0u64; tn + h - 1];
        wmul::mul_into(&mut u, &t[..tn], &out[..h], m);
        for (o, &v) in out[h..].iter_mut().zip(&u) {
            *o = m.sub(*o, v);
        }
    }
    while out.last() == Some(&0) {
        out.pop();
    }
    out
}
