//! The transformation of the Hermite form of `[A | I]` from its structure, in time linear in the
//! number of rows.
//!
//! For `A` of rank `r`, with `M = A[R, J]` the minor of its rank profiles, `δ = |det M|`, `S` the
//! rows outside `R` and `Y = A[S, J]·M⁻¹ = N / δ` with `N` of integers, the transformation `U` of
//! the Hermite form of `[A | I]` (see [`super::hermite`]) is made of two parts:
//!
//! - its last `m − r` rows, `(−z·Y, z)` on the columns `R` and `S`, for the rows `z` of the Hermite
//!   form of the lattice `Λ = {z ∈ Z^{m−r} : z·N ≡ 0 (mod δ)}` of the left kernel on `S`;
//! - its first `r` rows, `((Z_i − w·N) / δ, w)` for `Z = δ·H[:, J]·M⁻¹` and `w` the one solution of
//!   `w·N ≡ Z_i (mod δ)` that the rows of the form of `Λ` leave reduced; with every row in `R`
//!   (no `S`), simply `H·M⁻¹`. Both products by `M⁻¹`, matrices of integers, come from their
//!   residues modulo word primes ([`super::inverse`]).
//!
//! The quotient `Z^{m−r} / Λ` is the subgroup of `(Z/δ)^r` that the rows `N_k` span, of order at
//! most `δ^r`, so the form of `Λ` has at most `log₂ |Z^{m−r} / Λ|` pivots above 1: its pivot at
//! `k` is the order `d_k` of `N_k` modulo the subgroup `G_k` that the rows after it span, and its row
//! there is `d_k` at `k` and elements below the pivots above 1 after `k` alone, as the other ones
//! are 1. The rows are found from the last one up: `G_k` is held as a triangular basis of the
//! lattice `G_k + δ·Z^r`, each of its rows with its coefficients on the `N_p` of the pivots `p`
//! above 1 so far; the order of `N_k` and the coefficients that give `d_k·N_k` come from one pass
//! down that basis, and a pivot above 1 joins its row to the basis by extended Euclidean steps.
//! With `t` pivots above 1, each row costs `O(r² + r·t + t²)` operations on numbers below `δ`,
//! where the Hermite form of the `m × m` matrix `[A[:, J] | E_S]` would cost `O(m³)`.

use super::super::Matrix;
use super::inverse::times_inverse;
use super::profile::Basis;
use crate::math::numbers::{Divisor, Number};
use crate::math::rational::Rational;

/// A unimodular `size × size` matrix held as its first `r` rows, dense, and its other rows,
/// sparse, each with its pivot: the transformation of [`transform`], whose last rows are a basis of
/// a lattice of the left kernel in Hermite form on the columns of their pivots.
pub(super) struct Transform {
    /// The number of columns.
    pub(super) size: usize,
    /// The first `r` rows.
    pub(super) top: Vec<Vec<Number>>,
    /// The other rows, in the order of their pivots' columns, which ascend.
    pub(super) kernel: Vec<Sparse>,
}

/// A row of the kernel part of a [`Transform`]: its pivot, positive, in a column where the rows
/// before it are zero, and its other elements, none of them in the column of an earlier pivot.
pub(super) struct Sparse {
    /// The column of the pivot.
    pub(super) column: usize,
    /// The pivot.
    pub(super) pivot: Number,
    /// The elements but the pivot: `(column, value)`, each value not zero.
    pub(super) rest: Vec<(usize, Number)>,
}

impl Transform {
    /// Each of the first rows reduced modulo the kernel rows, in the order of their pivots: its
    /// element in the column of each pivot into `[0, pivot)`. The rows stay in the same coset of
    /// the lattice of the kernel rows, which leaves a product `U·A` as it was when the kernel rows
    /// are in the left kernel of `A`.
    pub(super) fn reduce_top(&mut self) {
        for row in &mut self.top {
            for k in &self.kernel {
                let q = row[k.column].div_floor(&k.pivot);
                if q.is_zero() {
                    continue;
                }
                row[k.column] -= &q * &k.pivot;
                for (c, x) in &k.rest {
                    row[*c] -= &q * x;
                }
            }
        }
    }

    /// The matrix, or its transpose when `transposed` is set.
    pub(super) fn into_matrix(self, transposed: bool) -> Matrix<Number> {
        let n = self.size;
        let mut out = Matrix::zeros(n, n);
        let r = self.top.len();
        for (i, row) in self.top.into_iter().enumerate() {
            for (j, x) in row.into_iter().enumerate() {
                if !x.is_zero() {
                    let at = if transposed { (j, i) } else { (i, j) };
                    out[at] = x;
                }
            }
        }
        for (t, k) in self.kernel.into_iter().enumerate() {
            let i = r + t;
            let at = |j: usize| if transposed { (j, i) } else { (i, j) };
            out[at(k.column)] = k.pivot;
            for (j, x) in k.rest {
                out[at(j)] = x;
            }
        }
        out
    }
}

/// `x` modulo `d`, in `[0, |d|)`.
#[inline]
fn reduce(x: &mut Number, d: &Divisor) {
    if !x.is_zero() {
        *x = d.rem_euclid(x);
    }
}

/// The rows of the Hermite form of `Λ` with pivots above 1 found so far, from the last up: the
/// form restricted to their columns, which are the only ones their rows reach.
struct Pivots {
    /// The pivot of each, in the order found (descending columns).
    d: Vec<Number>,
    /// The row of each over the pivots found up to it: `rows[q][q] = d[q]` and `rows[q][p]` for
    /// `p < q`, the pivots after it, in `[0, d[p])`.
    rows: Vec<Vec<Number>>,
}

impl Pivots {
    /// `x`, over the pivots found, reduced by their rows from the leftmost column on: each element
    /// into `[0, pivot)`. `x` stays in its coset of `Λ`.
    fn reduce(&self, x: &mut [Number]) {
        for q in (0..x.len()).rev() {
            let t = x[q].div_floor(&self.d[q]);
            if t.is_zero() {
                continue;
            }
            for (y, z) in x[..=q].iter_mut().zip(&self.rows[q]) {
                if !z.is_zero() {
                    *y -= &t * z;
                }
            }
        }
    }
}

/// The subgroup `G` of `(Z/δ)^r` that the rows `N_p` of the pivots above 1 span: a triangular
/// basis `t` of the lattice `G + δ·Z^r`, row `i` zero before column `i`, with `t[i] ≡ c[i]·N_P
/// (mod δ)` for coefficients `c[i]` on those pivots.
struct Group {
    /// δ prepared for the reductions.
    by: Divisor,
    /// The basis.
    t: Vec<Vec<Number>>,
    /// The coefficients of its rows on the pivots above 1.
    c: Vec<Vec<Number>>,
}

impl Group {
    /// The zero subgroup: `t = δ·I`.
    fn new(r: usize, delta: &Number) -> Group {
        let t = (0..r)
            .map(|i| {
                let mut row = vec![Number::default(); r];
                row[i] = delta.clone();
                row
            })
            .collect();
        Group {
            by: Divisor::new(delta),
            t,
            c: vec![Vec::new(); r],
        }
    }

    /// The order `d` of `v` modulo `G`, and coefficients `c` on the pivots above 1 with
    /// `d·v ≡ c·N_P (mod δ)`: down the basis, `v` times what makes its element in each column a
    /// multiple of the basis's there, less that multiple of the basis row.
    fn express(&self, v: &[Number], pivots: usize) -> (Number, Vec<Number>) {
        let r = self.t.len();
        let mut w: Vec<Number> = v.to_vec();
        w.iter_mut().for_each(|x| reduce(x, &self.by));
        let mut d = Number::from(1u8);
        let mut lam = vec![Number::default(); r];
        for i in 0..r {
            let ti = &self.t[i][i];
            let g = ti.div_exact(&w[i].gcd(ti));
            if g != 1u8 {
                d *= &g;
                for x in &mut w[i..] {
                    *x = &*x * &g;
                    reduce(x, &self.by);
                }
                for l in &mut lam[..i] {
                    *l = &*l * &g;
                    reduce(l, &self.by);
                }
            }
            if w[i].is_zero() {
                continue;
            }
            let q = w[i].div_exact(ti);
            for (x, y) in w[i..].iter_mut().zip(&self.t[i][i..]) {
                if !y.is_zero() {
                    *x -= &q * y;
                    reduce(x, &self.by);
                }
            }
            lam[i] = q;
        }
        debug_assert!(w.iter().all(Number::is_zero), "a multiple in the group");
        let mut c = vec![Number::default(); pivots];
        for (l, ci) in lam.iter().zip(&self.c) {
            if l.is_zero() {
                continue;
            }
            for (x, y) in c.iter_mut().zip(ci) {
                if !y.is_zero() {
                    *x += l * y;
                }
            }
        }
        (d, c)
    }

    /// `G` with `v` added, `v` the row of the pivot above 1 found last, the coefficients reduced
    /// by `pivots`, which already hold its row.
    fn add(&mut self, v: &[Number], pivots: &Pivots) {
        let p = pivots.d.len();
        for c in &mut self.c {
            c.resize(p, Number::default());
        }
        let mut v: Vec<Number> = v.to_vec();
        v.iter_mut().for_each(|x| reduce(x, &self.by));
        let mut cv = vec![Number::default(); p];
        cv[p - 1] = Number::from(1u8);
        let r = self.t.len();
        for i in 0..r {
            if v[i].is_zero() {
                continue;
            }
            let (ti, vi) = (self.t[i][i].clone(), v[i].clone());
            let (q, rem) = vi.div_rem(&ti);
            if rem.is_zero() {
                for (x, y) in v[i..].iter_mut().zip(&self.t[i][i..]) {
                    if !y.is_zero() {
                        *x -= &q * y;
                        reduce(x, &self.by);
                    }
                }
                for (x, y) in cv.iter_mut().zip(&self.c[i]) {
                    if !y.is_zero() {
                        *x -= &q * y;
                    }
                }
                continue;
            }
            let (e, x, y) = ti.gcd_ext(&vi);
            let (a, b) = (ti.div_exact(&e), vi.div_exact(&e));
            for (tj, vj) in self.t[i][i..].iter_mut().zip(v[i..].iter_mut()) {
                let mut nt = &x * &*tj + &y * &*vj;
                let mut nv = &a * &*vj - &b * &*tj;
                reduce(&mut nt, &self.by);
                reduce(&mut nv, &self.by);
                *tj = nt;
                *vj = nv;
            }
            for (cj, vj) in self.c[i].iter_mut().zip(cv.iter_mut()) {
                let nc = &x * &*cj + &y * &*vj;
                let nv = &a * &*vj - &b * &*cj;
                *cj = nc;
                *vj = nv;
            }
            pivots.reduce(&mut self.c[i]);
            pivots.reduce(&mut cv);
        }
        // `v` is zero now; its coefficients `cv` are a relation, and go.
        debug_assert!(v.iter().all(Number::is_zero), "a row the basis takes in");
    }
}

/// `y·δ`, an integer.
pub(super) fn times(y: &Rational, delta: &Number) -> Number {
    let (p, q) = (y.numer(), y.denom());
    p * &delta.div_exact(q)
}

/// The transformation of the Hermite form of `[A | I]` for the `m` rows of `A`, from its rank
/// profiles `b` (with `Y`; its rows `R` need only be independent for a transformation of the form
/// of `[A[:, J] | E_S]`, the one of `[A | I]` when they are the profile from the bottom) and the
/// Hermite form `hj` of the rows of `A[:, J]`.
pub(super) fn transform(m: usize, b: &Basis, hj: &Matrix<Number>) -> Transform {
    let r = b.rank;
    let s = &b.others;
    let delta = b.det();
    let n = b.n.as_ref().expect("the rows of the profile");
    // `N = δ·Y`, exactly and modulo δ.
    let n_exact: Vec<Vec<Number>> = (0..s.len())
        .map(|k| (0..r).map(|j| n[(k, j)].clone()).collect())
        .collect();
    let by = Divisor::new(delta);
    let n_mod: Vec<Vec<Number>> = n_exact
        .iter()
        .map(|row| {
            let mut row = row.clone();
            row.iter_mut().for_each(|x| reduce(x, &by));
            row
        })
        .collect();
    if s.is_empty() {
        // Every row in `R`: `U = H·M⁻¹` on the columns `R`, and no kernel.
        let u = times_inverse(hj, &b.minor, delta, &Number::from(1u8));
        let top = (0..r)
            .map(|i| {
                let mut row = vec![Number::default(); m];
                for (j, &col) in b.rows.iter().enumerate() {
                    row[col] = u[(i, j)].clone();
                }
                row
            })
            .collect();
        return Transform {
            size: m,
            top,
            kernel: Vec::new(),
        };
    }
    // `Z = δ·H[:, J]·M⁻¹`, of integers.
    let z_top: Vec<Vec<Number>> = if r == 0 {
        Vec::new()
    } else {
        let z = times_inverse(hj, &b.minor, delta, delta);
        (0..r)
            .map(|i| (0..r).map(|j| z[(i, j)].clone()).collect())
            .collect()
    };
    // The form of `Λ`, from its last row up.
    let mut group = Group::new(r, delta);
    let mut pivots = Pivots {
        d: Vec::new(),
        rows: Vec::new(),
    };
    // The positions of the pivots above 1, in the order found.
    let mut found: Vec<usize> = Vec::new();
    // For each row of `Λ`'s form: its pivot and its elements on the pivots above 1 after it.
    let mut rows: Vec<(Number, Vec<Number>)> = Vec::with_capacity(s.len());
    for k in (0..s.len()).rev() {
        let (d, c) = group.express(&n_mod[k], found.len());
        let mut z: Vec<Number> = c.into_iter().map(|x| -x).collect();
        pivots.reduce(&mut z);
        if d != 1u8 {
            let mut row = z.clone();
            row.push(d.clone());
            pivots.d.push(d.clone());
            pivots.rows.push(row);
            found.push(k);
            group.add(&n_mod[k], &pivots);
        }
        rows.push((d, z));
    }
    rows.reverse();
    // The `R` part of a row `(x on S)`: `(target − x·N) / δ`.
    let r_part = |target: Option<&[Number]>, x: &[(usize, &Number)]| -> Vec<Number> {
        (0..r)
            .map(|j| {
                let mut acc = target.map_or_else(Number::default, |t| t[j].clone());
                for (k, v) in x {
                    let e = &n_exact[*k][j];
                    if !e.is_zero() {
                        acc -= *v * e;
                    }
                }
                acc.div_exact(delta)
            })
            .collect()
    };
    let mut top = Vec::with_capacity(r);
    for zi in &z_top {
        let (d, mut w) = group.express(zi, found.len());
        assert!(
            d == 1u8,
            "a row of the form in the span of the rows of the matrix"
        );
        pivots.reduce(&mut w);
        let on_s: Vec<(usize, &Number)> = found
            .iter()
            .zip(&w)
            .filter(|(_, x)| !x.is_zero())
            .map(|(&k, x)| (k, x))
            .collect();
        let mut row = vec![Number::default(); m];
        for (&col, x) in b.rows.iter().zip(r_part(Some(zi), &on_s)) {
            row[col] = x;
        }
        for (k, x) in &on_s {
            row[s[*k]] = (*x).clone();
        }
        top.push(row);
    }
    let kernel = rows
        .iter()
        .enumerate()
        .map(|(k, (d, z))| {
            let mut on_s: Vec<(usize, &Number)> = vec![(k, d)];
            on_s.extend(
                found
                    .iter()
                    .zip(z)
                    .filter(|(_, x)| !x.is_zero())
                    .map(|(&p, x)| (p, x)),
            );
            let mut rest: Vec<(usize, Number)> = b
                .rows
                .iter()
                .zip(r_part(None, &on_s))
                .filter(|(_, x)| !x.is_zero())
                .map(|(&c, x)| (c, x))
                .collect();
            rest.extend(on_s[1..].iter().map(|(p, x)| (s[*p], (*x).clone())));
            Sparse {
                column: s[k],
                pivot: d.clone(),
                rest,
            }
        })
        .collect();
    Transform {
        size: m,
        top,
        kernel,
    }
}
