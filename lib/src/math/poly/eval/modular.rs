//! Evaluation at many points and interpolation of [`ModPolynomial`]s.

use super::modtree::{self, Ctx};
use super::mont::MAX_LIMBS;
use super::word;
use crate::math::numbers::Number;
use crate::math::poly::ModPolynomial;
use crate::math::poly::modular::{Modulus, Work};
use std::sync::Arc;

/// Up to this many coefficients of the polynomial, or points, `eval_many` takes Horner's rule at
/// every point, for a word modulus below `2^30` (NEON), `2^62`, `2^63` and `2^64`: measured on
/// an Apple M4 against the trees on the polynomials' fast products.
const WORD_HORNER_LEN: [usize; 4] = [1408, 1280, 448, 448];
/// The same for the odd moduli of `k ≥ 2` limbs, by `k` (Horner's rule by Montgomery's
/// products; measured for 2, 3, 4, 6, 8, 9, 12 and 16 limbs), and `8 + 2k` for the even ones (by
/// the products of residues by division).
const MULTI_HORNER_LEN: [usize; 17] = [
    0, 0, 960, 448, 384, 384, 368, 336, 304, 224, 220, 215, 208, 192, 176, 164, 152,
];
/// Up to this many points, [`interpolate`](ModPolynomial::interpolate) takes Lagrange's form on
/// Horner's rule at eight points (word moduli, by the classes of [`WORD_HORNER_LEN`]).
const WORD_INTERP_QUAD: [usize; 4] = [96, 176, 160, 112];
/// Up to this many points, modulo several limbs, Newton's form: for an odd modulus of up to 8, 12
/// and 16 limbs (by Montgomery's products), and for the others.
const MULTI_INTERP_NEWTON: [usize; 4] = [52, 44, 32, 8];

/// The class of the word modulus `n` in the tables of the crossovers.
fn class(n: u64) -> usize {
    match n {
        _ if n < 1 << 30 => 0,
        _ if n < 1 << 62 => 1,
        _ if n < 1 << 63 => 2,
        _ => 3,
    }
}

/// The values of the packed polynomial `f` at the packed points `xs`, packed (in the vector of
/// the points for Horner's rule).
pub(crate) fn eval_packed(m: &Arc<Modulus>, f: &[u64], mut xs: Vec<u64>) -> Vec<u64> {
    let k = m.k();
    let (len, npts) = (f.len() / k, xs.len() / k);
    if len == 0 || npts == 0 {
        xs.fill(0);
        return xs;
    }
    let cross = match m.word() {
        // Under Miri the trees from a few points, so that they run there.
        _ if cfg!(miri) => 12,
        Some(n) => WORD_HORNER_LEN[class(n)],
        None if k <= MAX_LIMBS && m.d[0] & 1 == 1 => MULTI_HORNER_LEN[k],
        None => 8 + 2 * k,
    };
    if len <= cross || npts <= cross {
        if let Some(wm) = m.word_mod() {
            word::horner_many(wm, f, &mut xs);
            return xs;
        }
        return horner(m, f, &xs);
    }
    let mut ctx = Ctx::new(m);
    // Trees of `chunk` points each: as many as the coefficients, at least the crossover.
    let chunk = len.max(cross);
    let mut out = Vec::with_capacity(xs.len());
    for c in xs.chunks(chunk * k) {
        if c.len() / k <= cross {
            out.extend(horner(m, f, c));
            continue;
        }
        let tree = modtree::build(&mut ctx, c);
        out.extend(tree.eval(&mut ctx, f));
    }
    out
}

/// Horner's rule at each point: modulo a word by [`word::horner_many`], modulo an odd `n` of
/// several limbs by Montgomery's products
/// ([`MontCtx::horner_many`](super::mont::MontCtx::horner_many)), otherwise on the residues'
/// products.
fn horner(m: &Modulus, f: &[u64], xs: &[u64]) -> Vec<u64> {
    if let Some(wm) = m.word_mod() {
        let mut v = xs.to_vec();
        word::horner_many(wm, f, &mut v);
        return v;
    }
    let mut out = vec![0u64; xs.len()];
    if let Some(mc) = m.mont() {
        mc.horner_many(f, xs, &mut out);
        return out;
    }
    let k = m.k();
    let len = f.len() / k;
    let mut v = vec![0u64; k];
    let mut t = vec![0u64; k];
    let mut w = Work::default();
    for (o, x) in out.chunks_exact_mut(k).zip(xs.chunks_exact(k)) {
        v.copy_from_slice(&f[(len - 1) * k..]);
        for i in (0..len - 1).rev() {
            m.mul(&mut t, &v, x, &mut w);
            v.copy_from_slice(&t);
            m.add(&mut v, &f[i * k..(i + 1) * k]);
        }
        o.copy_from_slice(&v);
    }
    out
}

/// The interpolating polynomial of the packed points and values, at least one, by the tree:
/// the weights `1 / M'(xᵢ)` by the values of the derivative of the root, then the linear
/// combination up the tree. Panics on a weight that is not a unit.
#[track_caller]
fn interp_tree(ctx: &mut Ctx<'_>, xs: &[u64], ys: &[u64]) -> Vec<u64> {
    let tree = modtree::build(ctx, xs);
    let root = tree.root(ctx);
    let d = modtree::derivative(ctx, &root);
    let mut w = tree.eval(ctx, &d);
    if let Err(i) = ctx.batch_inv(&mut w) {
        repeated(i);
    }
    let k = ctx.k;
    let mut c = vec![0u64; w.len()];
    for i in 0..tree.m {
        ctx.mul(
            &mut c[i * k..(i + 1) * k],
            &w[i * k..(i + 1) * k],
            &ys[i * k..(i + 1) * k],
        );
    }
    tree.combine(ctx, &c)
}

/// The panic of a point that is repeated, or whose differences to the others are not units.
#[track_caller]
fn repeated(i: usize) -> ! {
    panic!(
        "the point {i} is repeated, or a difference to it is not a unit modulo n: the points must be distinct modulo n"
    )
}

/// Newton's divided differences, then the expansion of the Newton form: `O(m²)`, the inverses of
/// all the differences of points by one Montgomery's trick (one inverse).
#[track_caller]
fn interp_newton(ctx: &mut Ctx<'_>, xs: &[u64], ys: &[u64]) -> Vec<u64> {
    let k = ctx.k;
    let m = xs.len() / k;
    let mut c = ys.to_vec();
    // The column j of differences x_i - x_{i-j}, i = j..m, from off(j).
    let off = |j: usize| (j - 1) * m - (j - 1) * j / 2;
    let mut d = vec![0u64; off(m) * k];
    for j in 1..m {
        for i in j..m {
            let o = &mut d[(off(j) + i - j) * k..(off(j) + i - j + 1) * k];
            o.copy_from_slice(&xs[i * k..(i + 1) * k]);
            ctx.sub(o, &xs[(i - j) * k..(i - j + 1) * k]);
        }
    }
    if let Err(e) = ctx.batch_inv(&mut d) {
        // The column and the row of the first difference that is not a unit.
        let j = (1..m).rev().find(|&j| off(j) <= e).unwrap_or(1);
        repeated(e - off(j) + j);
    }
    let mut t = vec![0u64; k];
    for j in 1..m {
        let dd = &d[off(j) * k..off(j + 1) * k];
        for i in (j..m).rev() {
            // c_i = (c_i - c_{i-1}) / (x_i - x_{i-j})
            let (lo, hi) = c.split_at_mut(i * k);
            t.copy_from_slice(&hi[..k]);
            ctx.sub(&mut t, &lo[(i - 1) * k..]);
            ctx.mul(&mut hi[..k], &t, &dd[(i - j) * k..(i - j + 1) * k]);
        }
    }
    expand_newton(ctx, xs, &c)
}

/// The polynomial `c₀ + c₁(x - x₀) + c₂(x - x₀)(x - x₁) + …` of the Newton form `c` at the
/// packed points `xs`, by Horner's rule on polynomials: `p ← p·(x - xᵢ) + cᵢ` from the top.
fn expand_newton(ctx: &mut Ctx<'_>, xs: &[u64], c: &[u64]) -> Vec<u64> {
    let k = ctx.k;
    let m = c.len() / k;
    let mut p = vec![0u64; m * k];
    let mut t = vec![0u64; k];
    if m == 0 {
        return p;
    }
    // p has `len` coefficients, from `m - len`... kept at the bottom: p[0..len].
    p[..k].copy_from_slice(&c[(m - 1) * k..]);
    for (len, i) in (1..m).zip((0..m - 1).rev()) {
        let af = ctx.factor(&xs[i * k..(i + 1) * k]);
        let a = &af[..];
        // p ← p·(x - a): p[len] = p[len-1], p[j] = p[j-1] - a p[j], p[0] = -a p[0].
        p.copy_within((len - 1) * k..len * k, len * k);
        for j in (1..len).rev() {
            t.copy_from_slice(&p[j * k..(j + 1) * k]);
            let (lo, hi) = p.split_at_mut(j * k);
            hi[..k].copy_from_slice(&lo[(j - 1) * k..]);
            ctx.mul_sub_factor(&mut hi[..k], &t, a);
        }
        t.copy_from_slice(&p[..k]);
        p[..k].fill(0);
        ctx.mul_sub_factor(&mut p[..k], &t, a);
        ctx.add(&mut p[..k], &c[i * k..(i + 1) * k]);
    }
    p
}

/// Lagrange's form: `Σ yᵢ/wᵢ · M/(x - xᵢ)` with `wᵢ = Π_{j≠i}(xᵢ - xⱼ)`, each quotient by
/// synthetic division of `M`: `O(m²)`.
#[track_caller]
fn interp_lagrange(ctx: &mut Ctx<'_>, xs: &[u64], ys: &[u64]) -> Vec<u64> {
    let k = ctx.k;
    let m = xs.len() / k;
    let big = modtree::from_roots(ctx, xs);
    if let Some(wm) = ctx.word {
        // The weights M'(xᵢ) by Horner's rule at all the points, inverted together, then the
        // quotients' sum at eight points at once.
        let mut w = xs.to_vec();
        word::horner_many(&wm, &modtree::derivative(ctx, &big), &mut w);
        if let Err(i) = word::batch_inv(&wm, &mut w) {
            repeated(i);
        }
        for (w, &y) in w.iter_mut().zip(ys) {
            *w = wm.mul(*w, y);
        }
        let mut out = vec![0u64; m];
        word::lagrange_sum(&wm, &big, xs, &w, &mut out);
        return out;
    }
    // The weights.
    let mut w = vec![0u64; m * k];
    let mut t = vec![0u64; k];
    for i in 0..m {
        let o = &mut w[i * k..(i + 1) * k];
        o.fill(0);
        o[0] = 1;
        for j in 0..m {
            if j == i {
                continue;
            }
            t.copy_from_slice(&xs[i * k..(i + 1) * k]);
            ctx.sub(&mut t, &xs[j * k..(j + 1) * k]);
            let prev = o.to_vec();
            ctx.mul(o, &prev, &t);
        }
    }
    if let Err(i) = ctx.batch_inv(&mut w) {
        repeated(i);
    }
    let mut out = vec![0u64; m * k];
    let mut q = vec![0u64; k];
    let mut c = vec![0u64; k];
    for i in 0..m {
        ctx.mul(&mut c, &w[i * k..(i + 1) * k], &ys[i * k..(i + 1) * k]);
        let a = &xs[i * k..(i + 1) * k];
        // M / (x - a): q_{m-1} = 1, q_{j-1} = M_j + a·q_j.
        q.fill(0);
        q[0] = 1;
        for j in (0..m).rev() {
            ctx.mul_add(&mut out[j * k..(j + 1) * k], &c, &q);
            if j > 0 {
                let prev = q.clone();
                ctx.mul(&mut q, &prev, a);
                ctx.add(&mut q, &big[j * k..(j + 1) * k]);
            }
        }
    }
    out
}

/// The residues of the numbers `xs` modulo `m`, packed.
fn residues(m: &Modulus, xs: &[Number]) -> Vec<u64> {
    if m.word().is_some() {
        // Without the limbs of each point unless one is longer than a word.
        let mut v = Vec::with_capacity(xs.len());
        for x in xs {
            v.push(match m.word_residue(x) {
                Some(r) => r,
                None => m.residue(x)[0],
            });
        }
        return v;
    }
    let k = m.k();
    let mut v = vec![0u64; xs.len() * k];
    let mut l = Vec::new();
    let mut w = Work::default();
    for (x, o) in xs.iter().zip(v.chunks_exact_mut(k)) {
        // The magnitude's limbs, divided only when it is not below n already.
        crate::math::poly::arith::pack::limbs(x, &mut l);
        if l.len() < k || (l.len() == k && below(&l, &m.d)) {
            o[..l.len()].copy_from_slice(&l);
        } else {
            m.reduce(o, &l, &mut w);
        }
        if x.is_negative() {
            m.neg(o);
        }
    }
    v
}

/// Whether `a < b`, of the same length.
fn below(a: &[u64], b: &[u64]) -> bool {
    for (x, y) in a.iter().rev().zip(b.iter().rev()) {
        if x != y {
            return x < y;
        }
    }
    false
}

/// The numbers of the packed residues `v`.
fn numbers(m: &Modulus, v: &[u64]) -> Vec<Number> {
    v.chunks_exact(m.k()).map(|r| m.number(r)).collect()
}

/// The word modulus `n`, as [`Modulus::new`] makes it.
///
/// # Panics
///
/// Panics if `n` is below 2.
#[track_caller]
fn word_modulus(n: u64) -> Arc<Modulus> {
    assert!(n >= 2, "the modulus must be at least 2");
    crate::math::poly::modular::cached_word_modulus(n)
}

/// Which interpolation.
#[derive(Clone, Copy)]
enum Interp {
    Fast,
    Lagrange,
    Newton,
}

/// The interpolating polynomial of the packed `xs` and `ys` modulo `m`, by `how`.
#[track_caller]
fn interpolate_packed(m: Arc<Modulus>, xs: &[u64], ys: Vec<u64>, how: Interp) -> ModPolynomial {
    assert_eq!(
        xs.len(),
        ys.len(),
        "as many values as points are needed to interpolate"
    );
    if xs.is_empty() {
        return ModPolynomial {
            m,
            c: Vec::new().into(),
        };
    }
    if xs.len() == m.k() {
        // One point: the constant.
        return ModPolynomial::of(m, ys);
    }
    let mut ctx = Ctx::new(&m);
    let npts = xs.len() / ctx.k;
    let small = match ctx.word {
        _ if cfg!(miri) => 8,
        Some(w) => WORD_INTERP_QUAD[class(w.n)],
        None if m.d[0] & 1 == 0 || ctx.k > MAX_LIMBS => MULTI_INTERP_NEWTON[3],
        None if ctx.k <= 8 => MULTI_INTERP_NEWTON[0],
        None if ctx.k <= 12 => MULTI_INTERP_NEWTON[1],
        None => MULTI_INTERP_NEWTON[2],
    };
    let c = match how {
        Interp::Fast if npts > small => interp_tree(&mut ctx, xs, &ys),
        // Modulo a word Lagrange's form on Horner's rule at eight points is the fastest of the
        // quadratic ones; modulo several limbs Newton's.
        Interp::Fast if ctx.word.is_some() => interp_lagrange(&mut ctx, xs, &ys),
        Interp::Fast | Interp::Newton => interp_newton(&mut ctx, xs, &ys),
        Interp::Lagrange => interp_lagrange(&mut ctx, xs, &ys),
    };
    drop(ctx);
    ModPolynomial::of(m, c)
}

impl ModPolynomial {
    /// The values modulo `n` at each of the points `xs`, in `0..n`: by Horner's rule for short
    /// polynomials or few points, otherwise by the subproduct tree of the points and the scaled
    /// remainder tree (the transposed algorithm), the points in groups of as many as the
    /// coefficients. Any modulus `n ≥ 2`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::poly::ModPolynomial;
    /// use a::math::numbers::Number;
    ///
    /// // x² + 1 modulo 7 at 0, 1, 2, 3.
    /// let p = ModPolynomial::from_u64(&[1, 0, 1], 7);
    /// let xs: Vec<Number> = (0..4).map(Number::from).collect();
    /// let v: Vec<Number> = [1, 2, 5, 3].into_iter().map(Number::from).collect();
    /// assert_eq!(p.eval_many(&xs), v);
    /// ```
    pub fn eval_many(&self, xs: &[Number]) -> Vec<Number> {
        if self.length() <= 1 {
            // A constant: its value at every point, the points not read.
            return vec![self.coeff(0); xs.len()];
        }
        let pts = residues(&self.m, xs);
        numbers(&self.m, &eval_packed(&self.m, &self.c, pts))
    }

    /// [`eval_many`](ModPolynomial::eval_many) on words, for a modulus below `2^64`: the points
    /// reduced, the values in `0..n`.
    ///
    /// # Panics
    ///
    /// Panics if the modulus is `2^64` or more.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::poly::ModPolynomial;
    ///
    /// let p = ModPolynomial::from_u64(&[1, 0, 1], 7);
    /// assert_eq!(p.eval_many_u64(&[0, 1, 2, 10]), [1, 2, 5, 3]);
    /// ```
    #[track_caller]
    pub fn eval_many_u64(&self, xs: &[u64]) -> Vec<u64> {
        let n = self
            .m
            .word()
            .expect("eval_many_u64 needs a modulus below 2^64");
        let pts: Vec<u64> = xs.iter().map(|&x| if x < n { x } else { x % n }).collect();
        eval_packed(&self.m, &self.c, pts)
    }

    /// The monic `Π (x - rᵢ)` modulo `modulus`, by the product tree of the linear factors (one
    /// factor after the other for a few). The constant 1 for no roots.
    ///
    /// # Panics
    ///
    /// Panics if `modulus` is below 2.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::poly::ModPolynomial;
    /// use a::math::numbers::Number;
    ///
    /// // (x - 1)(x - 2) = x² - 3x + 2 = x² + 4x + 2 modulo 7.
    /// let r = [Number::from(1), Number::from(2)];
    /// let p = ModPolynomial::from_roots(&r, &Number::from(7));
    /// assert_eq!(p, ModPolynomial::from_u64(&[2, 4, 1], 7));
    /// ```
    #[track_caller]
    pub fn from_roots(roots: &[Number], modulus: &Number) -> ModPolynomial {
        let m = crate::math::poly::modular::cached_modulus(modulus);
        if let [r] = roots {
            // x - r, without the tree's context.
            let mut c = m.residue(r);
            m.neg(&mut c);
            c.resize(2 * m.k(), 0);
            c[m.k()] = 1;
            return ModPolynomial::of(m, c);
        }
        let pts = residues(&m, roots);
        let c = modtree::from_roots(&mut Ctx::new(&m), &pts);
        ModPolynomial::of(m, c)
    }

    /// [`from_roots`](ModPolynomial::from_roots) on words: the roots reduced modulo the word
    /// `modulus`.
    ///
    /// # Panics
    ///
    /// Panics if `modulus` is below 2.
    #[track_caller]
    pub fn from_roots_u64(roots: &[u64], modulus: u64) -> ModPolynomial {
        let m = word_modulus(modulus);
        if let [r] = roots {
            // x - r, without the tree's context.
            let r = r % modulus;
            return ModPolynomial::of(m, vec![if r == 0 { 0 } else { modulus - r }, 1]);
        }
        let pts: Vec<u64> = roots.iter().map(|&x| x % modulus).collect();
        let c = modtree::from_roots(&mut Ctx::new(&m), &pts);
        ModPolynomial::of(m, c)
    }

    /// The polynomial of degree below `xs.len()` modulo `modulus` that takes the value `ys[i]` at
    /// `xs[i]`, for a prime `modulus` (or any whose differences of the points are units): by the
    /// subproduct tree, the weights `1 / M'(xᵢ)` by the scaled remainder tree and the linear
    /// combination up the tree, or for a few hundred points modulo a word Lagrange's form (a few
    /// modulo several limbs, Newton's). The zero polynomial for no points.
    ///
    /// # Panics
    ///
    /// Panics if `modulus` is below 2, if `xs` and `ys` differ in length, or if two points are
    /// the same modulo `modulus` (or, for a composite modulus, differ by a non-unit).
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::poly::ModPolynomial;
    /// use a::math::numbers::Number;
    ///
    /// let n = Number::from(101);
    /// let p = ModPolynomial::from_u64(&[5, 0, 3, 1], 101);
    /// let xs: Vec<Number> = (1..=4).map(Number::from).collect();
    /// let ys = p.eval_many(&xs);
    /// assert_eq!(ModPolynomial::interpolate(&xs, &ys, &n), p);
    /// ```
    #[track_caller]
    pub fn interpolate(xs: &[Number], ys: &[Number], modulus: &Number) -> ModPolynomial {
        let m = crate::math::poly::modular::cached_modulus(modulus);
        if let ([_], [y]) = (xs, ys) {
            // One point: the constant.
            let c = m.residue(y);
            return ModPolynomial::of(m, c);
        }
        let (px, py) = (residues(&m, xs), residues(&m, ys));
        interpolate_packed(m, &px, py, Interp::Fast)
    }

    /// [`interpolate`](ModPolynomial::interpolate) on words: the points and values reduced modulo
    /// the word `modulus`.
    ///
    /// # Panics
    ///
    /// As [`interpolate`](ModPolynomial::interpolate).
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::poly::ModPolynomial;
    ///
    /// // The line through (1, 3) and (2, 5) modulo 7: 2x + 1.
    /// let p = ModPolynomial::interpolate_u64(&[1, 2], &[3, 5], 7);
    /// assert_eq!(p, ModPolynomial::from_u64(&[1, 2], 7));
    /// ```
    #[track_caller]
    pub fn interpolate_u64(xs: &[u64], ys: &[u64], modulus: u64) -> ModPolynomial {
        let m = word_modulus(modulus);
        if let ([_], [y]) = (xs, ys) {
            // One point: the constant.
            return ModPolynomial::of(m, vec![y % modulus]);
        }
        let px: Vec<u64> = xs.iter().map(|&x| x % modulus).collect();
        let py: Vec<u64> = ys.iter().map(|&x| x % modulus).collect();
        interpolate_packed(m, &px, py, Interp::Fast)
    }

    /// [`interpolate`](ModPolynomial::interpolate) by Lagrange's form, `Σ yᵢ · Π_{j≠i} (x - xⱼ) /
    /// (xᵢ - xⱼ)`: `O(m²)` for `m` points.
    ///
    /// # Panics
    ///
    /// As [`interpolate`](ModPolynomial::interpolate).
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::poly::ModPolynomial;
    /// use a::math::numbers::Number;
    ///
    /// let xs = [Number::from(1), Number::from(2)];
    /// let ys = [Number::from(3), Number::from(5)];
    /// let p = ModPolynomial::interpolate_lagrange(&xs, &ys, &Number::from(7));
    /// assert_eq!(p, ModPolynomial::from_u64(&[1, 2], 7));
    /// ```
    #[track_caller]
    pub fn interpolate_lagrange(xs: &[Number], ys: &[Number], modulus: &Number) -> ModPolynomial {
        let m = crate::math::poly::modular::cached_modulus(modulus);
        if let ([_], [y]) = (xs, ys) {
            // One point: the constant.
            let c = m.residue(y);
            return ModPolynomial::of(m, c);
        }
        let (px, py) = (residues(&m, xs), residues(&m, ys));
        interpolate_packed(m, &px, py, Interp::Lagrange)
    }

    /// [`interpolate`](ModPolynomial::interpolate) by Newton's divided differences and the
    /// expansion of Newton's form: `O(m²)` for `m` points.
    ///
    /// # Panics
    ///
    /// As [`interpolate`](ModPolynomial::interpolate).
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::poly::ModPolynomial;
    /// use a::math::numbers::Number;
    ///
    /// let xs = [Number::from(1), Number::from(2)];
    /// let ys = [Number::from(3), Number::from(5)];
    /// let p = ModPolynomial::interpolate_newton(&xs, &ys, &Number::from(7));
    /// assert_eq!(p, ModPolynomial::from_u64(&[1, 2], 7));
    /// ```
    #[track_caller]
    pub fn interpolate_newton(xs: &[Number], ys: &[Number], modulus: &Number) -> ModPolynomial {
        let m = crate::math::poly::modular::cached_modulus(modulus);
        if let ([_], [y]) = (xs, ys) {
            // One point: the constant.
            let c = m.residue(y);
            return ModPolynomial::of(m, c);
        }
        let (px, py) = (residues(&m, xs), residues(&m, ys));
        interpolate_packed(m, &px, py, Interp::Newton)
    }
}
