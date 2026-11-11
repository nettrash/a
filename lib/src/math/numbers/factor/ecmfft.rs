//! The elliptic curve method's second stage by fast polynomial arithmetic: Montgomery's FFT
//! continuation (his thesis, 1992) as GMP-ECM's `ecm2.c` arranges it (Zimmermann and Dodson,
//! "20 years of ECM", 2006).
//!
//! For `D` made of small primes and the `d = phi(D)/2` odd `j < D/2` prime to `D`, the baby steps'
//! `x(j Q)` are the roots of `F`, and the giant steps' `x(k D Q)`, `d` of them a block, the roots
//! of `G_i`; then `H = prod_i G_i mod F` and the product of `H` at the roots of `F` is the product
//! of all the differences `x(j Q) - x(k D Q)`, each 0 modulo a prime `p` when `(kD - j) Q` or
//! `(kD + j) Q` is the identity modulo `p`: every number prime to `D` from `k_1 D - D/2` to the
//! last giant step's `+ D/2` is covered, primes and others, at the cost of a few products of
//! polynomials of degree `d` a block (see [`tree`](super::tree)) instead of a product of residues
//! for each prime.

use super::field::Arith;
use super::poly::Kron;
use super::rho::{Gcd, gcd};
use super::tree::{Coeffs, Tree, inverse, reduce, transposed_root};

/// The candidates for `D`: `210 t`, `2310 t`, `30030 t` for the `t` made of the primes to 11, whose
/// `phi(D)/2` are `24 t`, `240 t` and `2880 t` (`210 t` with 11 in `t` is `2310 t/11`).
const TS: [u64; 22] = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 18, 20, 21, 22, 24, 25, 27,
];

/// The parameters of the continuation for its bounds.
#[derive(Clone, Debug)]
pub(super) struct FftPlan {
    /// `D`.
    pub(super) d: u64,
    /// The odd `j < D/2` prime to `D`.
    pub(super) js: Vec<u64>,
    /// The first giant step, and the blocks of `js.len()` giant steps.
    pub(super) k_first: u64,
    pub(super) blocks: u64,
    /// The estimated time, in the nanoseconds of [`product_ns`](super::mk::product_ns).
    pub(super) cost: f64,
}

/// Euler's totient of `d`.
fn phi(mut d: u64) -> u64 {
    let mut t = d;
    let mut p = 2;
    while p * p <= d {
        if d.is_multiple_of(p) {
            t = t / p * (p - 1);
            while d.is_multiple_of(p) {
                d /= p;
            }
        }
        p += 1;
    }
    if d > 1 {
        t = t / d * (d - 1);
    }
    t
}

/// The time of a tree of products of `d` linear factors with coefficients of `k` limbs, in
/// nanoseconds: measured on an Apple M4 Max, about 0.4 of a product of all the coefficients a level
/// (the low levels are short products).
fn tree_ns(k: usize, d: usize) -> f64 {
    let levels = (d.max(2) as f64).log2().ceil();
    0.4 * levels * product_poly_ns(k, d)
}

/// A product of two polynomials of `d` coefficients.
fn product_poly_ns(k: usize, d: usize) -> f64 {
    let limbs = 2 * d * (2 * k + 1);
    let per_limb = 15.0 + f64::from((limbs / 1000).max(1).ilog2());
    let redc = 10.0 + 0.4 * ((k + 1) * (k + 1)) as f64;
    limbs as f64 * per_limb + 2.0 * d as f64 * redc
}

impl FftPlan {
    /// The parameters that cost least for the bounds `b1 < b2` (the primes to `b1` in the first
    /// stage) and a modulus of `k` limbs, with `D/2 <= b1`.
    pub(super) fn new(b1: u64, b2: u64, k: usize) -> Option<FftPlan> {
        if b2 <= b1 {
            return None;
        }
        let mm = super::mk::product_ns(k);
        let mut best: Option<(f64, u64, u64, u64)> = None;
        for base in [210u64, 2310, 30030] {
            for &t in &TS {
                if base == 210 && (t > 10 || t % 11 == 0) {
                    continue;
                }
                let d = base * t;
                if d / 2 > b1 {
                    continue;
                }
                let n = phi(d) / 2;
                let k_first = (b1 + 1 + d / 2) / d;
                let k_last = (b2 / d).saturating_add(1);
                let steps = k_last + 1 - k_first;
                let blocks = steps.div_ceil(n);
                // Memory: the tree of `F`, `n` coefficients a level, at most 64 MiB.
                let levels = u64::from(n.max(2).ilog2()) + 2;
                if n.saturating_mul(levels).saturating_mul(k as u64 * 8) > 64 << 20 {
                    continue;
                }
                let n = n as usize;
                // Babies: `D/4` additions and the normalisation; `F`'s tree, its inverse and the
                // descent; per block, `n` additions and their normalisation, `G`'s tree, `H G`
                // and its reduction.
                // Measured: the inverse about 3 products, the descent 2.5 trees; the first block's
                // `G mod F` is `G - F`.
                let babies = (d as f64 / 4.0 * 6.0 + 3.0 * n as f64) * mm;
                let tree = tree_ns(k, n);
                let product = product_poly_ns(k, n);
                let once = 3.5 * tree + 3.0 * product;
                let block = 9.0 * n as f64 * mm + tree + 3.0 * product;
                let cost = babies + once + blocks as f64 * block - 3.0 * product + 60_000.0;
                if best.is_none_or(|b| cost < b.0) {
                    best = Some((cost, d, k_first, blocks));
                }
            }
        }
        let (cost, d, k_first, blocks) = best?;
        let js = (1..d / 2)
            .step_by(2)
            .filter(|&j| gcd_small(j, d) == 1)
            .collect();
        Some(FftPlan {
            d,
            js,
            k_first,
            blocks,
            cost,
        })
    }
}

/// The greatest common divisor of small numbers.
fn gcd_small(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// The evaluation: `F` from the baby steps, then `H` block by block.
pub(super) struct Eval<'a, A: Arith> {
    f: &'a A,
    kron: Kron,
    /// The residue of `B`, which takes a residue `x R` to `x R'`.
    b: A::V,
    /// 1 as a coefficient, `R' mod n`.
    one: Vec<u64>,
    /// The tree of `F`, and `rev(F)^(-1) mod X^d`.
    tree: Tree,
    inv: Vec<u64>,
    /// `H`, once a block is in.
    h: Option<Vec<u64>>,
    /// The roots of a block, as coefficients.
    roots: Vec<u64>,
}

impl<'a, A: Arith> Eval<'a, A> {
    /// The evaluation for the affine `x` of the baby steps.
    pub(super) fn new(f: &'a A, babies: &[A::V]) -> Self {
        let k = f.n().len();
        let mut kron = Kron::new(f.n());
        // `B mod n` as a residue: `R B mod n`, from the residue of `B` read as limbs.
        let b = f.residue(&b_limbs(f));
        let mut one = vec![0u64; k];
        f.put(&b, &mut one);
        let mut roots = vec![0u64; babies.len() * k];
        let mut t = f.zero();
        for (x, c) in babies.iter().zip(roots.chunks_exact_mut(k)) {
            f.mul(&mut t, x, &b);
            f.put(&t, c);
        }
        let coeffs = Coeffs { n: f.n() };
        let tree = Tree::new(&mut kron, &roots, &one, &coeffs);
        let inv = inverse(&mut kron, tree.root(), &one, &coeffs);
        Eval {
            f,
            kron,
            b,
            one,
            tree,
            inv,
            h: None,
            roots,
        }
    }

    /// The degree of `F`: the giant steps of a block.
    pub(super) fn degree(&self) -> usize {
        self.tree.root().len() / self.f.n().len() - 1
    }

    /// `H = H G mod F` for the giant steps' affine `x`, as many as `F`'s degree at most.
    pub(super) fn block(&mut self, giants: &[A::V]) {
        let f = self.f;
        let k = f.n().len();
        let d = self.degree();
        debug_assert!(giants.len() <= d);
        self.roots.clear();
        self.roots.resize(giants.len() * k, 0);
        let mut t = f.zero();
        for (x, c) in giants.iter().zip(self.roots.chunks_exact_mut(k)) {
            f.mul(&mut t, x, &self.b);
            f.put(&t, c);
        }
        let coeffs = Coeffs { n: f.n() };
        let g = Tree::product(&mut self.kron, &self.roots, &self.one, &coeffs);
        let root = self.tree.root();
        let next = match self.h.take() {
            // `G mod F`.
            None => reduce(&mut self.kron, &g, root, &self.inv, &coeffs),
            Some(h) => {
                let mut p = vec![0u64; (d + g.len() / k - 1) * k];
                self.kron.mul(&h, &g, 0, &mut p);
                reduce(&mut self.kron, &p, root, &self.inv, &coeffs)
            }
        };
        self.h = Some(next);
    }

    /// The product of `H` at the roots of `F`, and its greatest common divisor with `n`.
    pub(super) fn finish(mut self) -> Gcd {
        let f = self.f;
        let Some(h) = self.h.take() else {
            return Gcd::One;
        };
        let u = transposed_root(&mut self.kron, &h, &self.inv);
        let mut values = Vec::new();
        self.tree.descend(&mut self.kron, u, &mut values);
        let mut acc = f.one();
        for c in values.chunks_exact(f.n().len()) {
            f.mul_raw(&mut acc, c);
        }
        gcd(f, &acc)
    }
}

/// The limbs of `B mod n` for the modulus of `f`: `B` itself below two limbs' worth, as `residue`
/// takes at most the modulus's limbs.
fn b_limbs<A: Arith>(f: &A) -> Vec<u64> {
    let n = f.n();
    if n.len() >= 2 {
        vec![0, 1]
    } else {
        vec![((1u128 << 64) % u128::from(n[0])) as u64]
    }
}
