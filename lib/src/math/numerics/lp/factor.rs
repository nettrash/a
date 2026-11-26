//! The basis of the dual simplex method, factorised: a sparse LU decomposition by Markowitz's
//! rule with threshold pivoting, finished densely once what is left is dense, and the changes of
//! basis since in the product form.
//!
//! The decomposition is `P·B·Q = L·U` (Markowitz, 1957; Suhl and Suhl, 1990): pivot `k` is the
//! element of row `prow[k]` and basis position `pcol[k]` of the basis `B`, `L` is unit lower
//! triangular and `U` upper triangular in the order of the pivots. The pivots are chosen one at a
//! time from the part of the matrix not yet eliminated: among the elements whose magnitude is at
//! least [`THRESHOLD`] times the largest of their column, one that makes the product of the other
//! elements of its row and its column, the bound of Markowitz on the fill it can cause, small,
//! searched over the columns and the rows of the fewest elements first and stopped after
//! [`SEARCH`] of them. A column or a row of one element is a pivot that causes no fill, so the
//! columns of the logical variables and the triangular parts of a basis are taken first, at no
//! cost. Once the elements left fill more than [`DENSE`] of the square they lie in, that square
//! is factorised as a dense matrix by [`Matrix::lu`], with partial pivoting, and its factors are
//! kept dense.
//!
//! A change of basis replaces one column; it is absorbed as one elementary factor of the product
//! form (Dantzig and Orchard-Hays, 1954), `B = B₀·E₁·E₂⋯`, `Eₖ` the identity with one column
//! replaced by the entering column as the basis before it saw it, until the basis is factorised
//! afresh. The solves, `B·x = b` and `Bᵀ·y = c`, pass over the zeros of what they solve for, so
//! that a sparse right-hand side costs only the parts of the factors it meets.

use super::{Real, mag};
use crate::math::matrix::blas::{Op, Side, Uplo, gemm, trsm};
use crate::math::matrix::{Matrix, Scalar};

/// The fraction of the largest element of its column that a pivot of the sparse part has to
/// reach: the threshold of the rule of Markowitz, which trades the growth of the elements, and so
/// the stability of the factors, against fill.
// Measured on an Apple M4 Max: 0.01, 0.05 and 0.3 change the times of the scoreboard's, the
// netlib-like and the battery's programmes, and of random ones of densities 0.15 to 0.8, by at
// most 0.5 % and their pivots hardly at all (Suhl and Suhl use 0.01, HiGHS 0.1).
const THRESHOLD: f64 = 0.1;

/// How many columns and rows the search of a pivot examines, once it has found one that will do.
// Measured on an Apple M4 Max: 2 and 8 change the times of the same programmes by less than
// 0.7 % (sweep 2 to 8).
const SEARCH: usize = 4;

/// The fraction of its square that the part left to factorise has to fill before it is factorised
/// as a dense matrix.
// Measured on an Apple M4 Max: 0.2 is within 1 % of it on the same programmes, 0.1 makes the
// scoreboard's of density 0.1 2.8 % slower, 0.4 and 0.5 the netlib-like equality-heavy ones 2.6
// and 6 % (sweep 0.1 to 0.5).
const DENSE: f64 = 0.3;

/// The fraction of its rectangle that the part of `U` in the rows of the sparse part and the
/// columns of the dense part has to fill to be kept dense.
// Measured on an Apple M4 Max: 0.1 makes the scoreboard's programmes of density 0.1 2.5 %
// slower, 0.5 random ones of densities 0.3 to 0.5 6 to 16 % (sweep 0.1 to 0.5).
const DENSE_ABOVE: f64 = 0.25;

/// The fraction of its elements that an elementary factor of the product form has to fill to be
/// kept as a dense vector.
// Measured on an Apple M4 Max: 0.2 makes some programmes up to 2 % slower, 0.6 is within 1.2 %
// on every kind (sweep 0.2 to 0.6).
const DENSE_ETA: f64 = 0.4;

/// Why a basis could not be factorised: a column that is, to rounding, a combination of others.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Singular;

/// The working memory of a factorisation: the part not yet eliminated, as a dense array of values
/// with the patterns of its rows and its columns beside it.
struct Active<T> {
    /// `m`.
    m: usize,
    /// The values, element `(i, c)` at `i + c·m`; zero outside the patterns.
    val: Vec<T>,
    /// Whether element `(i, c)` is in the patterns.
    present: Vec<bool>,
    /// The columns of each row that hold an element; eliminated columns stay until the row is
    /// cleared, and are passed over.
    row_pat: Vec<Vec<usize>>,
    /// The rows of each column that hold an element, likewise.
    col_pat: Vec<Vec<usize>>,
    /// How many elements each row has in the columns not yet eliminated.
    row_count: Vec<usize>,
    /// How many elements each column has in the rows not yet eliminated.
    col_count: Vec<usize>,
    /// Whether each row has not been eliminated.
    row_active: Vec<bool>,
    /// Whether each column has not been eliminated.
    col_active: Vec<bool>,
    /// The lists of the columns and the rows by their counts, doubly linked: the first of each
    /// count, and the next and previous of each.
    col_head: Vec<usize>,
    col_next: Vec<usize>,
    col_prev: Vec<usize>,
    row_head: Vec<usize>,
    row_next: Vec<usize>,
    row_prev: Vec<usize>,
    /// The size below which an element of each column is taken for zero: the rounding of the
    /// largest element of the column as it was given.
    tiny: Vec<T>,
    /// How many elements the part not yet eliminated holds.
    nnz: usize,
}

/// The end of a list.
const NONE: usize = usize::MAX;

/// `Σ a_i·b_i` over the shorter of the two, in eight partial sums that the compiler keeps in
/// vector registers: a sum of one accumulator waits for each addition before the next.
#[inline]
pub(super) fn dot<T: Real>(a: &[T], b: &[T]) -> T {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let mut s = [T::ZERO; 8];
    let (ca, ra) = a.as_chunks::<8>();
    let (cb, rb) = b.as_chunks::<8>();
    for (x, y) in ca.iter().zip(cb) {
        for l in 0..8 {
            s[l] += x[l] * y[l];
        }
    }
    let mut t = T::ZERO;
    for (x, y) in ra.iter().zip(rb) {
        t += *x * *y;
    }
    ((s[0] + s[1]) + (s[2] + s[3])) + ((s[4] + s[5]) + (s[6] + s[7])) + t
}

/// `Σ_p value_p·v[index_p]`, in four partial sums.
#[inline]
pub(super) fn dot_gather<T: Real>(value: &[T], index: &[usize], v: &[T]) -> T {
    let mut s = [T::ZERO; 4];
    let (cv, rv) = value.as_chunks::<4>();
    let (ci, ri) = index.as_chunks::<4>();
    for (x, i) in cv.iter().zip(ci) {
        for l in 0..4 {
            s[l] += x[l] * v[i[l]];
        }
    }
    let mut t = T::ZERO;
    for (x, &i) in rv.iter().zip(ri) {
        t += *x * v[i];
    }
    (s[0] + s[1]) + (s[2] + s[3]) + t
}

/// `y ← y − a·v` over the shorter of the two.
#[inline]
fn axpy<T: Real>(y: &mut [T], a: &[T], v: T) {
    for (y, a) in y.iter_mut().zip(a) {
        *y -= *a * v;
    }
}

impl<T: Real> Active<T> {
    fn new() -> Self {
        Self {
            m: 0,
            val: Vec::new(),
            present: Vec::new(),
            row_pat: Vec::new(),
            col_pat: Vec::new(),
            row_count: Vec::new(),
            col_count: Vec::new(),
            row_active: Vec::new(),
            col_active: Vec::new(),
            col_head: Vec::new(),
            col_next: Vec::new(),
            col_prev: Vec::new(),
            row_head: Vec::new(),
            row_next: Vec::new(),
            row_prev: Vec::new(),
            tiny: Vec::new(),
            nnz: 0,
        }
    }

    /// Loads the `m` columns of `start`, `index` and `value` (compressed by columns).
    fn load(&mut self, m: usize, start: &[usize], index: &[usize], value: &[T]) {
        if self.m != m {
            self.m = m;
            self.val = vec![T::ZERO; m * m];
            self.present = vec![false; m * m];
            self.row_pat = vec![Vec::new(); m];
            self.col_pat = vec![Vec::new(); m];
        }
        self.row_count.clear();
        self.row_count.resize(m, 0);
        self.col_count.clear();
        self.col_count.resize(m, 0);
        self.row_active.clear();
        self.row_active.resize(m, true);
        self.col_active.clear();
        self.col_active.resize(m, true);
        self.tiny.clear();
        let rounding = T::EPSILON * T::from_f64(m as f64);
        self.nnz = 0;
        for c in 0..m {
            let mut largest = T::ZERO;
            for p in start[c]..start[c + 1] {
                let (i, v) = (index[p], value[p]);
                let at = i + c * m;
                if v == T::ZERO {
                    continue;
                }
                if self.present[at] {
                    self.val[at] += v;
                } else {
                    self.present[at] = true;
                    self.val[at] = v;
                    self.row_pat[i].push(c);
                    self.col_pat[c].push(i);
                    self.row_count[i] += 1;
                    self.col_count[c] += 1;
                    self.nnz += 1;
                }
                largest = largest.max(mag(self.val[at]));
            }
            self.tiny.push(largest * rounding);
        }
        self.col_head.clear();
        self.col_head.resize(m + 1, NONE);
        self.col_next.clear();
        self.col_next.resize(m, NONE);
        self.col_prev.clear();
        self.col_prev.resize(m, NONE);
        self.row_head.clear();
        self.row_head.resize(m + 1, NONE);
        self.row_next.clear();
        self.row_next.resize(m, NONE);
        self.row_prev.clear();
        self.row_prev.resize(m, NONE);
        for c in (0..m).rev() {
            self.col_insert(c);
        }
        for i in (0..m).rev() {
            self.row_insert(i);
        }
    }

    /// Clears the values and the patterns, so that the memory is all zeros again.
    fn clear(&mut self) {
        let m = self.m;
        for c in 0..m {
            for &i in &self.col_pat[c] {
                self.val[i + c * m] = T::ZERO;
                self.present[i + c * m] = false;
            }
            self.col_pat[c].clear();
        }
        for row in &mut self.row_pat {
            row.clear();
        }
    }

    fn col_insert(&mut self, c: usize) {
        let t = self.col_count[c];
        let head = self.col_head[t];
        self.col_next[c] = head;
        self.col_prev[c] = NONE;
        if head != NONE {
            self.col_prev[head] = c;
        }
        self.col_head[t] = c;
    }

    fn col_remove(&mut self, c: usize) {
        let (prev, next) = (self.col_prev[c], self.col_next[c]);
        if prev == NONE {
            self.col_head[self.col_count[c]] = next;
        } else {
            self.col_next[prev] = next;
        }
        if next != NONE {
            self.col_prev[next] = prev;
        }
    }

    fn row_insert(&mut self, i: usize) {
        let t = self.row_count[i];
        let head = self.row_head[t];
        self.row_next[i] = head;
        self.row_prev[i] = NONE;
        if head != NONE {
            self.row_prev[head] = i;
        }
        self.row_head[t] = i;
    }

    fn row_remove(&mut self, i: usize) {
        let (prev, next) = (self.row_prev[i], self.row_next[i]);
        if prev == NONE {
            self.row_head[self.row_count[i]] = next;
        } else {
            self.row_next[prev] = next;
        }
        if next != NONE {
            self.row_prev[next] = prev;
        }
    }

    /// The largest magnitude of the elements of column `c` in the rows not yet eliminated.
    fn column_max(&self, c: usize) -> T {
        let m = self.m;
        let mut largest = T::ZERO;
        for &i in &self.col_pat[c] {
            if self.row_active[i] {
                largest = largest.max(mag(self.val[i + c * m]));
            }
        }
        largest
    }

    /// The pivot by the rule of Markowitz with threshold pivoting, as `(row, column)`, among the
    /// rows and columns not yet eliminated; `Err` when a column has no element that is not
    /// rounding, which makes the basis singular.
    fn choose(&self, remaining: usize) -> Result<(usize, usize), Singular> {
        let m = self.m;
        let threshold = T::from_f64(THRESHOLD);
        // A column or a row of no element left: singular.
        if self.col_head[0] != NONE || self.row_head[0] != NONE {
            return Err(Singular);
        }
        let mut best: Option<(usize, usize, usize)> = None;
        let mut examined = 0;
        for t in 1..=remaining {
            // The columns of `t` elements.
            let mut c = self.col_head[t];
            while c != NONE {
                let largest = self.column_max(c);
                if largest <= self.tiny[c] {
                    return Err(Singular);
                }
                for &i in &self.col_pat[c] {
                    if !self.row_active[i] {
                        continue;
                    }
                    let v = mag(self.val[i + c * m]);
                    if v >= threshold * largest && v > self.tiny[c] {
                        let cost = (self.row_count[i] - 1) * (t - 1);
                        if best.is_none_or(|(b, _, _)| cost < b) {
                            best = Some((cost, i, c));
                        }
                    }
                }
                examined += 1;
                if let Some((cost, i, c)) = best
                    && (examined >= SEARCH || cost <= (t - 1) * (t - 1))
                {
                    return Ok((i, c));
                }
                c = self.col_next[c];
            }
            // The rows of `t` elements.
            let mut i = self.row_head[t];
            while i != NONE {
                for &c in &self.row_pat[i] {
                    if !self.col_active[c] {
                        continue;
                    }
                    let v = mag(self.val[i + c * m]);
                    if v <= self.tiny[c] {
                        continue;
                    }
                    let largest = self.column_max(c);
                    if v >= threshold * largest {
                        let cost = (t - 1) * (self.col_count[c] - 1);
                        if best.is_none_or(|(b, _, _)| cost < b) {
                            best = Some((cost, i, c));
                        }
                    }
                }
                examined += 1;
                if let Some((cost, i, c)) = best
                    && (examined >= SEARCH || cost <= (t - 1) * t)
                {
                    return Ok((i, c));
                }
                i = self.row_next[i];
            }
        }
        best.map(|(_, i, c)| (i, c)).ok_or(Singular)
    }
}

/// The factors of a basis and the changes of basis since: see the [module](self).
pub(super) struct Factor<T> {
    /// `m`, the order of the basis.
    m: usize,
    /// The row of the basis of each pivot.
    prow: Vec<usize>,
    /// The basis position of each pivot.
    pcol: Vec<usize>,
    /// How many pivots the sparse part has; the pivots from there to `m` are the dense part.
    k0: usize,
    /// The pivots of the sparse part, the diagonal of `U` there.
    diag: Vec<T>,
    /// The columns of `L` of the sparse part, in the order of the pivots: the elements of column
    /// `k` below the diagonal are `l_idx`/`l_val` from `l_start[k]` to `l_start[k + 1]`.
    l_start: Vec<usize>,
    l_idx: Vec<usize>,
    l_val: Vec<T>,
    /// The same elements by rows: those of row `i` (any pivot) in the columns of the sparse part.
    lr_start: Vec<usize>,
    lr_idx: Vec<usize>,
    lr_val: Vec<T>,
    /// The rows of `U` of the sparse part, right of the diagonal.
    u_start: Vec<usize>,
    u_idx: Vec<usize>,
    u_val: Vec<T>,
    /// The same elements by columns: those of column `j` (any pivot) in the rows of the sparse
    /// part.
    uc_start: Vec<usize>,
    uc_idx: Vec<usize>,
    uc_val: Vec<T>,
    /// The dense part, `d × d` for `d = m − k0`, column-major: `L` below the diagonal (its
    /// diagonal ones not stored) and `U` on and above it.
    dense: Vec<T>,
    /// The part of `U` in the rows of the sparse part and the columns of the dense part, `k0 × d`
    /// column-major, when it is dense enough to be kept so ([`DENSE_ABOVE`]); empty otherwise,
    /// and then among the sparse rows of `U`.
    above: Vec<T>,
    /// The elementary factors of the product form, oldest first: the basis position each
    /// replaces, its pivot, and its other elements: `eta_val` from `eta_start[k]` to
    /// `eta_start[k + 1]`, by the positions `eta_idx` from `eta_istart[k]` when they are listed,
    /// or every position, zero at its own, when they are not (a dense factor, [`DENSE_ETA`]).
    eta_pos: Vec<usize>,
    eta_piv: Vec<T>,
    eta_start: Vec<usize>,
    eta_istart: Vec<usize>,
    eta_idx: Vec<usize>,
    eta_val: Vec<T>,
    /// Vectors in the order of the pivots, for the solves.
    work: Vec<T>,
    /// A copy of the dense parts of them.
    copy: Vec<T>,
    /// The working memory of a factorisation.
    active: Active<T>,
}

impl<T: Real + Scalar> Factor<T> {
    /// An empty factorisation, for `factor` to fill.
    pub(super) fn new() -> Self {
        Self {
            m: 0,
            prow: Vec::new(),
            pcol: Vec::new(),
            k0: 0,
            diag: Vec::new(),
            l_start: vec![0],
            l_idx: Vec::new(),
            l_val: Vec::new(),
            lr_start: vec![0],
            lr_idx: Vec::new(),
            lr_val: Vec::new(),
            u_start: vec![0],
            u_idx: Vec::new(),
            u_val: Vec::new(),
            uc_start: vec![0],
            uc_idx: Vec::new(),
            uc_val: Vec::new(),
            dense: Vec::new(),
            above: Vec::new(),
            eta_pos: Vec::new(),
            eta_piv: Vec::new(),
            eta_start: vec![0],
            eta_istart: vec![0],
            eta_idx: Vec::new(),
            eta_val: Vec::new(),
            work: Vec::new(),
            copy: Vec::new(),
            active: Active::new(),
        }
    }

    /// The number of changes of basis since the basis was factorised.
    pub(super) fn updates(&self) -> usize {
        self.eta_pos.len()
    }

    /// The number of elements the elementary factors hold.
    pub(super) fn eta_elements(&self) -> usize {
        self.eta_val.len()
    }

    /// The number of elements the factors hold, `L` and `U` together, with the dense part.
    pub(super) fn elements(&self) -> usize {
        let d = self.m - self.k0;
        self.k0 + self.l_idx.len() + self.u_idx.len() + self.above.len() + d * d
    }

    /// Factorises the basis of the `m` columns given compressed by columns (`start`, `index`,
    /// `value`: the elements of basis position `c` are from `start[c]` to `start[c + 1]`), and
    /// forgets the changes of basis; `Err` when it is singular to working precision.
    pub(super) fn factor(
        &mut self,
        m: usize,
        start: &[usize],
        index: &[usize],
        value: &[T],
    ) -> Result<(), Singular> {
        self.m = m;
        self.prow.clear();
        self.pcol.clear();
        self.diag.clear();
        self.l_start.clear();
        self.l_start.push(0);
        self.l_idx.clear();
        self.l_val.clear();
        self.u_start.clear();
        self.u_start.push(0);
        self.u_idx.clear();
        self.u_val.clear();
        self.dense.clear();
        self.above.clear();
        self.eta_pos.clear();
        self.eta_piv.clear();
        self.eta_start.clear();
        self.eta_start.push(0);
        self.eta_istart.clear();
        self.eta_istart.push(0);
        self.eta_idx.clear();
        self.eta_val.clear();
        self.work.clear();
        self.work.resize(m, T::ZERO);
        self.active.load(m, start, index, value);
        let result = self.eliminate();
        self.active.clear();
        result
    }

    /// The elimination itself, on the loaded working memory.
    fn eliminate(&mut self) -> Result<(), Singular> {
        let m = self.m;
        let a = &mut self.active;
        let mut k = 0;
        let mut urow: Vec<(usize, T)> = Vec::new();
        let mut lcol: Vec<(usize, T)> = Vec::new();
        while k < m {
            let remaining = m - k;
            // A column of one element is a pivot of no fill and no work: those are always taken
            // before the rest is judged dense.
            if a.col_head[1] == NONE && a.nnz as f64 >= DENSE * (remaining * remaining) as f64 {
                break;
            }
            let (p, q) = a.choose(remaining)?;
            let pivot = a.val[p + q * m];
            urow.clear();
            for &c in &a.row_pat[p] {
                if a.col_active[c] && c != q {
                    urow.push((c, a.val[p + c * m]));
                }
            }
            lcol.clear();
            for &i in &a.col_pat[q] {
                if a.row_active[i] && i != p {
                    lcol.push((i, a.val[i + q * m] / pivot));
                }
            }
            a.row_remove(p);
            a.row_active[p] = false;
            a.col_remove(q);
            a.col_active[q] = false;
            a.nnz -= 1 + urow.len() + lcol.len();
            for &(c, _) in &urow {
                a.col_remove(c);
                a.col_count[c] -= 1;
            }
            for &(i, _) in &lcol {
                a.row_remove(i);
                a.row_count[i] -= 1;
            }
            for &(i, l) in &lcol {
                for &(c, u) in &urow {
                    let at = i + c * m;
                    if a.present[at] {
                        a.val[at] -= l * u;
                    } else {
                        a.present[at] = true;
                        a.val[at] = -(l * u);
                        a.row_pat[i].push(c);
                        a.col_pat[c].push(i);
                        a.row_count[i] += 1;
                        a.col_count[c] += 1;
                        a.nnz += 1;
                    }
                }
            }
            for &(c, _) in &urow {
                a.col_insert(c);
            }
            for &(i, _) in &lcol {
                a.row_insert(i);
            }
            // Row `p` and column `q` are done with: their patterns can go.
            a.row_pat[p].clear();
            self.prow.push(p);
            self.pcol.push(q);
            self.diag.push(pivot);
            // Row and column indices stay those of the basis until every pivot is known.
            for &(c, u) in &urow {
                self.u_idx.push(c);
                self.u_val.push(u);
            }
            self.u_start.push(self.u_idx.len());
            for &(i, l) in &lcol {
                self.l_idx.push(i);
                self.l_val.push(l);
            }
            self.l_start.push(self.l_idx.len());
            k += 1;
        }
        self.k0 = k;
        let d = m - k;
        if d > 0 {
            let a = &self.active;
            let rows: Vec<usize> = (0..m).filter(|&i| a.row_active[i]).collect();
            let cols: Vec<usize> = (0..m).filter(|&c| a.col_active[c]).collect();
            let block = Matrix::from_fn(d, d, |s, t| a.val[rows[s] + cols[t] * m]);
            let lu = block.lu();
            let (lower, upper) = (lu.l(), lu.u());
            for t in 0..d {
                let pivot = upper[(t, t)];
                if !super::finite(pivot) || mag(pivot) <= a.tiny[cols[t]] {
                    return Err(Singular);
                }
            }
            self.dense.resize(d * d, T::ZERO);
            for t in 0..d {
                for s in 0..d {
                    self.dense[s + t * d] = if s > t { lower[(s, t)] } else { upper[(s, t)] };
                }
            }
            for (t, &c) in cols.iter().enumerate() {
                self.prow.push(rows[lu.permutation()[t]]);
                self.pcol.push(c);
            }
        }
        self.renumber();
        Ok(())
    }

    /// Turns the row and column indices of the sparse factors into positions of pivots, and
    /// makes the copies by rows of `L` and by columns of `U`.
    fn renumber(&mut self) {
        let m = self.m;
        let mut row_pos = vec![0; m];
        let mut col_pos = vec![0; m];
        for k in 0..m {
            row_pos[self.prow[k]] = k;
            col_pos[self.pcol[k]] = k;
        }
        for i in &mut self.l_idx {
            *i = row_pos[*i];
        }
        for c in &mut self.u_idx {
            *c = col_pos[*c];
        }
        // The part of `U` above the dense part, kept dense when it is dense.
        let (k0, d) = (self.k0, m - self.k0);
        let above = self.u_idx.iter().filter(|&&j| j >= k0).count();
        if d > 0 && k0 > 0 && above as f64 >= DENSE_ABOVE * (k0 * d) as f64 {
            self.above.clear();
            self.above.resize(k0 * d, T::ZERO);
            let mut kept = 0;
            for k in 0..k0 {
                let (s, e) = (self.u_start[k], self.u_start[k + 1]);
                self.u_start[k] = kept;
                for p in s..e {
                    let (j, v) = (self.u_idx[p], self.u_val[p]);
                    if j >= k0 {
                        self.above[k + (j - k0) * k0] = v;
                    } else {
                        self.u_idx[kept] = j;
                        self.u_val[kept] = v;
                        kept += 1;
                    }
                }
            }
            self.u_start[k0] = kept;
            self.u_idx.truncate(kept);
            self.u_val.truncate(kept);
        }
        // `L` by rows.
        let mut count = vec![0usize; m + 1];
        for &i in &self.l_idx {
            count[i + 1] += 1;
        }
        for i in 0..m {
            count[i + 1] += count[i];
        }
        self.lr_start.clear();
        self.lr_start.extend_from_slice(&count);
        self.lr_idx.clear();
        self.lr_idx.resize(self.l_idx.len(), 0);
        self.lr_val.clear();
        self.lr_val.resize(self.l_idx.len(), T::ZERO);
        for k in 0..self.k0 {
            for p in self.l_start[k]..self.l_start[k + 1] {
                let i = self.l_idx[p];
                let at = count[i];
                count[i] += 1;
                self.lr_idx[at] = k;
                self.lr_val[at] = self.l_val[p];
            }
        }
        // `U` by columns.
        let mut count = vec![0usize; m + 1];
        for &j in &self.u_idx {
            count[j + 1] += 1;
        }
        for j in 0..m {
            count[j + 1] += count[j];
        }
        self.uc_start.clear();
        self.uc_start.extend_from_slice(&count);
        self.uc_idx.clear();
        self.uc_idx.resize(self.u_idx.len(), 0);
        self.uc_val.clear();
        self.uc_val.resize(self.u_idx.len(), T::ZERO);
        for k in 0..self.k0 {
            for p in self.u_start[k]..self.u_start[k + 1] {
                let j = self.u_idx[p];
                let at = count[j];
                count[j] += 1;
                self.uc_idx[at] = k;
                self.uc_val[at] = self.u_val[p];
            }
        }
    }

    /// Records that the variable of basis position `r` has been replaced by the one whose column
    /// is `B·column`, `column` in basis positions.
    pub(super) fn update(&mut self, r: usize, column: &[T]) {
        self.eta_pos.push(r);
        self.eta_piv.push(column[r]);
        let nonzero = column.iter().filter(|v| **v != T::ZERO).count();
        if nonzero as f64 > DENSE_ETA * self.m as f64 {
            self.eta_val.extend_from_slice(column);
            let at = self.eta_val.len() - self.m + r;
            self.eta_val[at] = T::ZERO;
        } else {
            for (i, &v) in column.iter().enumerate() {
                if v != T::ZERO && i != r {
                    self.eta_idx.push(i);
                    self.eta_val.push(v);
                }
            }
        }
        self.eta_start.push(self.eta_val.len());
        self.eta_istart.push(self.eta_idx.len());
    }

    /// `x ← B⁻¹·x`: `x` holds a right-hand side by rows and is left holding the solution by
    /// basis positions.
    pub(super) fn ftran(&mut self, x: &mut [T]) {
        self.ftran_many(&mut [x]);
    }

    /// [`ftran`](Self::ftran) of several right-hand sides at once, in one pass over the factors:
    /// each column of them is read once, from memory, for all of them, and the dense part is a
    /// blocked triangular solve of all of them together.
    pub(super) fn ftran_many(&mut self, xs: &mut [&mut [T]]) {
        let m = self.m;
        if m == 0 {
            return;
        }
        let k0 = self.k0;
        let d = m - k0;
        let count = xs.len();
        self.work.resize(count * m, T::ZERO);
        let w = &mut self.work[..count * m];
        for (r, x) in xs.iter().enumerate() {
            for (v, &i) in w[r * m..(r + 1) * m].iter_mut().zip(&self.prow) {
                *v = x[i];
            }
        }
        // `L` of the sparse part, a column at a time.
        for k in 0..k0 {
            let (s, e) = (self.l_start[k], self.l_start[k + 1]);
            if s == e {
                continue;
            }
            for r in 0..count {
                let wr = &mut w[r * m..(r + 1) * m];
                let v = wr[k];
                if v != T::ZERO {
                    for p in s..e {
                        wr[self.l_idx[p]] -= self.l_val[p] * v;
                    }
                }
            }
        }
        if d > 0 {
            // The dense part: `L`, then `U`, every right-hand side together.
            let b = &mut w[k0..];
            trsm(
                Side::Left,
                Uplo::Lower,
                Op::N,
                true,
                d,
                count,
                &self.dense,
                d,
                b,
                m,
            );
            trsm(
                Side::Left,
                Uplo::Upper,
                Op::N,
                false,
                d,
                count,
                &self.dense,
                d,
                b,
                m,
            );
            // What the dense part's values take off the rows of the sparse part: a product of
            // blocks, on a copy of those values.
            if !self.above.is_empty() {
                self.copy.clear();
                for r in 0..count {
                    self.copy.extend_from_slice(&w[r * m + k0..(r + 1) * m]);
                }
                gemm(
                    k0,
                    count,
                    d,
                    -T::ONE,
                    &self.above,
                    k0,
                    Op::N,
                    &self.copy,
                    d,
                    Op::N,
                    T::ONE,
                    w,
                    m,
                );
            } else {
                for j in k0..m {
                    let (s, e) = (self.uc_start[j], self.uc_start[j + 1]);
                    if s == e {
                        continue;
                    }
                    for r in 0..count {
                        let wr = &mut w[r * m..(r + 1) * m];
                        let v = wr[j];
                        if v != T::ZERO {
                            for p in s..e {
                                wr[self.uc_idx[p]] -= self.uc_val[p] * v;
                            }
                        }
                    }
                }
            }
        }
        // `U` of the sparse part, a column at a time from the last.
        for k in (0..k0).rev() {
            let (s, e) = (self.uc_start[k], self.uc_start[k + 1]);
            let pivot = self.diag[k];
            for r in 0..count {
                let wr = &mut w[r * m..(r + 1) * m];
                let v = wr[k] / pivot;
                wr[k] = v;
                if v != T::ZERO {
                    for p in s..e {
                        wr[self.uc_idx[p]] -= self.uc_val[p] * v;
                    }
                }
            }
        }
        for (r, x) in xs.iter_mut().enumerate() {
            for (v, &c) in w[r * m..(r + 1) * m].iter().zip(&self.pcol) {
                x[c] = *v;
            }
        }
        // The elementary factors, the oldest first.
        for e in 0..self.eta_pos.len() {
            let r = self.eta_pos[e];
            let values = &self.eta_val[self.eta_start[e]..self.eta_start[e + 1]];
            let (si, ei) = (self.eta_istart[e], self.eta_istart[e + 1]);
            let dense = si == ei && values.len() == m;
            for x in xs.iter_mut() {
                let v = x[r] / self.eta_piv[e];
                x[r] = v;
                if v == T::ZERO {
                    continue;
                }
                if dense {
                    axpy(x, values, v);
                } else {
                    for (&i, &a) in self.eta_idx[si..ei].iter().zip(values) {
                        x[i] -= a * v;
                    }
                }
            }
        }
    }

    /// `y ← B⁻ᵀ·y`: `y` holds a right-hand side by basis positions and is left holding the
    /// solution by rows.
    pub(super) fn btran(&mut self, y: &mut [T]) {
        let m = self.m;
        let k0 = self.k0;
        // The elementary factors, the newest first.
        for e in (0..self.eta_pos.len()).rev() {
            let r = self.eta_pos[e];
            let values = &self.eta_val[self.eta_start[e]..self.eta_start[e + 1]];
            let (si, ei) = (self.eta_istart[e], self.eta_istart[e + 1]);
            let others = if si == ei && values.len() == m {
                dot(values, y)
            } else {
                dot_gather(values, &self.eta_idx[si..ei], y)
            };
            y[r] = (y[r] - others) / self.eta_piv[e];
        }
        let w = &mut self.work[..m];
        for (v, &c) in w.iter_mut().zip(&self.pcol) {
            *v = y[c];
        }
        // `Uᵀ` of the sparse part, a row of `U` at a time.
        for k in 0..k0 {
            let v = w[k] / self.diag[k];
            w[k] = v;
            if v != T::ZERO {
                for p in self.u_start[k]..self.u_start[k + 1] {
                    w[self.u_idx[p]] -= self.u_val[p] * v;
                }
            }
        }
        let d = m - k0;
        if d > 0 {
            // What the rows of the sparse part take off the dense part, kept dense.
            if !self.above.is_empty() {
                let (top, bottom) = w.split_at_mut(k0);
                for (t, v) in bottom.iter_mut().enumerate() {
                    *v -= dot(&self.above[t * k0..(t + 1) * k0], top);
                }
            }
            // The dense part: `Uᵀ`, then `Lᵀ`, by inner products with the columns of the factors
            // (for one right-hand side these beat the blocked solve).
            let (dw, dense) = (&mut w[k0..], &self.dense);
            for t in 0..d {
                let column = &dense[t * d..t * d + t];
                dw[t] = (dw[t] - dot(column, &dw[..t])) / dense[t * d + t];
            }
            for t in (0..d).rev() {
                let column = &dense[t * d + t + 1..(t + 1) * d];
                dw[t] -= dot(column, &dw[t + 1..]);
            }
        }
        // `Lᵀ` of the sparse part, a row of `L` at a time from the last.
        for i in (0..m).rev() {
            let v = w[i];
            if v != T::ZERO {
                for p in self.lr_start[i]..self.lr_start[i + 1] {
                    w[self.lr_idx[p]] -= self.lr_val[p] * v;
                }
            }
        }
        for (v, &i) in w.iter().zip(&self.prow) {
            y[i] = *v;
        }
    }
}
