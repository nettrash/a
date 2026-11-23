//! The supernodal numeric factorisation that [`SparseCholesky`] and [`SparseLdl`] share, and the
//! triangular solves on its blocks.
//!
//! The factor is held supernode by supernode ([`Symbolic`]): the columns of a supernode have the
//! same rows, so they are kept as two dense blocks, column after column — the square block on the
//! diagonal, of which the lower triangle is the factor's, and the block of the rows below it. The
//! method is the left-looking one of Ng and Peyton ("Block sparse Cholesky algorithms on advanced
//! uniprocessor computers", SIAM J. Sci. Comput. 14, 1993), which CHOLMOD's supernodal
//! factorisation follows too. For each supernode in turn:
//!
//! 1. its blocks receive the elements of the matrix;
//! 2. every earlier supernode with rows among its columns — a descendant in the elimination tree —
//!    takes its part out: the product of two blocks of the descendant, by `gemm`, subtracted at
//!    the rows the two share;
//! 3. the block on the diagonal is factored, by the dense `cholesky_in_place` or by the dense
//!    [`ldl_in_place`];
//! 4. the block below is solved against it, by `trsm`.
//!
//! All the arithmetic is thus that of dense kernels on blocks; what is sparse is which blocks
//! meet. The descendants of a supernode are not searched for: a finished supernode waits in the
//! list of the supernode that holds its next row, and moves on to the next when it has been used,
//! as the columns of the classical sparse column Cholesky do.
//!
//! `P·A·Pᵀ = L·D·Lᴴ` goes the same way with three changes: the products carry `D` between the
//! two blocks, the block on the diagonal is factored without square roots, and the block below
//! is divided by the pivots after its solve.
//!
//! [`SparseCholesky`]: super::SparseCholesky
//! [`SparseLdl`]: super::SparseLdl

use super::SparseMatrix;
use super::symbolic::{Lower, NONE, Symbolic};
use crate::math::complex::Float;
use crate::math::matrix::blas::{Op, Side, Uplo, gemm, trsm};
use crate::math::matrix::{Matrix, Scalar, cholesky_in_place};

/// The supernodal factor of `P·A·Pᵀ`: `L` with `L·Lᴴ`, or the unit `L` and `D` with `L·D·Lᴴ`.
#[derive(Clone)]
pub(super) struct Factor<T: Scalar> {
    /// The supernodes and where their blocks are.
    pub(super) symbolic: Symbolic,
    /// Whether the factorisation is `L·D·Lᴴ`, with ones on the diagonal of `L`.
    unit: bool,
    /// The square blocks on the diagonal, each column after column: `L` in the lower triangle,
    /// zeros above it. With `unit`, the diagonal holds `D` and the ones of `L` are not stored.
    diagonal: Vec<T>,
    /// The blocks below them, each column after column, a row for each row of the supernode.
    below: Vec<T>,
    /// `D` in the inner numbering, with `unit`; empty otherwise.
    pivots: Vec<T::Real>,
}

impl<T: Scalar> Factor<T> {
    /// Factors the matrix whose lower triangle is `lower`, laid out as `symbolic` says: `L·Lᴴ`,
    /// or `L·D·Lᴴ` with `unit`. `None` when a pivot stops it: one that is not positive for
    /// `L·Lᴴ`, one that is zero for `L·D·Lᴴ`, and one that is not finite for either.
    pub(super) fn new(symbolic: Symbolic, lower: &Lower<T>, unit: bool) -> Option<Self> {
        let s = &symbolic;
        let count = s.supernodes();
        let mut diagonal = vec![T::zero(); s.diagonal_ptrs[count]];
        let mut below = vec![T::zero(); s.below_ptrs[count]];
        let mut pivots = vec![T::Real::ZERO; if unit { s.n } else { 0 }];
        // Where each row lies among the rows below the supernode at hand.
        let mut position = vec![0usize; s.n];
        // The supernodes waiting to update each supernode, as lists: the first, and the one
        // after each; and how many of its rows each has been used for already.
        let mut head = vec![NONE; count];
        let mut next = vec![NONE; count];
        let mut used = vec![0usize; count];
        // Working memory: the product of an update, the rows of a descendant times `D`, and
        // what the dense `L·D·Lᴴ` asks for.
        let mut product: Vec<T> = Vec::new();
        let mut scaled: Vec<T> = Vec::new();
        let mut work: Vec<T> = Vec::new();
        for sn in 0..count {
            let (first, end) = (s.first[sn], s.first[sn + 1]);
            let width = end - first;
            let rows = &s.rows[s.row_ptrs[sn]..s.row_ptrs[sn + 1]];
            let height = rows.len();
            let block = &mut diagonal[s.diagonal_ptrs[sn]..s.diagonal_ptrs[sn + 1]];
            let (finished, rest) = below.split_at_mut(s.below_ptrs[sn]);
            let panel = &mut rest[..height * width];
            for (t, &r) in rows.iter().enumerate() {
                position[r] = t;
            }
            // The elements of the matrix: a row is one of the supernode's own columns, or one of
            // the rows below them.
            for j in first..end {
                let c = j - first;
                for p in lower.col_ptrs[j]..lower.col_ptrs[j + 1] {
                    let (i, x) = (lower.rows[p], lower.values[p]);
                    if i < end {
                        block[(i - first) + c * width] = x;
                    } else {
                        panel[position[i] + c * height] = x;
                    }
                }
            }
            // The updates. Of the rows of a descendant not yet used, the first ones lie among
            // the columns here and the others below them: the product of the rows from the first
            // on with the conjugate transpose of the first ones is what the descendant takes out
            // of this supernode, its lower triangle at least, which is all that is kept.
            let mut d = core::mem::replace(&mut head[sn], NONE);
            while d != NONE {
                let following = next[d];
                let d_rows = &s.rows[s.row_ptrs[d]..s.row_ptrs[d + 1]];
                let d_width = s.first[d + 1] - s.first[d];
                let d_height = d_rows.len();
                let d_panel = &finished[s.below_ptrs[d]..s.below_ptrs[d + 1]];
                let from = used[d];
                let mut to = from;
                while to < d_height && d_rows[to] < end {
                    to += 1;
                }
                let (m, k) = (d_height - from, to - from);
                if product.len() < m * k {
                    product.resize(m * k, T::zero());
                }
                let update = &mut product[..m * k];
                if unit {
                    // `L·D·Lᴴ`: the first rows times `D`, so that the product has it inside.
                    if scaled.len() < k * d_width {
                        scaled.resize(k * d_width, T::zero());
                    }
                    let with_d = &mut scaled[..k * d_width];
                    for q in 0..d_width {
                        let pivot = pivots[s.first[d] + q];
                        let column = &d_panel[q * d_height + from..q * d_height + to];
                        for (w, &x) in with_d[q * k..(q + 1) * k].iter_mut().zip(column) {
                            *w = x.scale(pivot);
                        }
                    }
                    gemm(
                        m,
                        k,
                        d_width,
                        T::one(),
                        &d_panel[from..],
                        d_height,
                        Op::N,
                        with_d,
                        k,
                        Op::C,
                        T::zero(),
                        update,
                        m,
                    );
                } else {
                    gemm(
                        m,
                        k,
                        d_width,
                        T::one(),
                        &d_panel[from..],
                        d_height,
                        Op::N,
                        &d_panel[from..],
                        d_height,
                        Op::C,
                        T::zero(),
                        update,
                        m,
                    );
                }
                for jj in 0..k {
                    let c = d_rows[from + jj] - first;
                    let column = &update[jj * m..(jj + 1) * m];
                    for ii in jj..k {
                        block[(d_rows[from + ii] - first) + c * width] -= column[ii];
                    }
                    for ii in k..m {
                        panel[position[d_rows[from + ii]] + c * height] -= column[ii];
                    }
                }
                // On to the supernode of its next row, if it has one.
                used[d] = to;
                if to < d_height {
                    let target = s.supernode[d_rows[to]];
                    next[d] = head[target];
                    head[target] = d;
                }
                d = following;
            }
            // The block on the diagonal, then the block below it: `X·L₁₁ᴴ = B`, and for
            // `L·D·Lᴴ`, where that leaves `L₂₁·D`, the division by the pivots.
            if unit {
                if !ldl_in_place(block, width, width, &mut work) {
                    return None;
                }
                for j in 0..width {
                    pivots[first + j] = block[j + j * width].re();
                    block[j * width..j * width + j].fill(T::zero());
                }
            } else {
                cholesky_in_place(block, width, width)?;
                // The dense method takes an infinite pivot for a positive one.
                if (0..width).any(|j| !block[j + j * width].is_finite()) {
                    return None;
                }
            }
            if height > 0 {
                trsm(
                    Side::Right,
                    Uplo::Lower,
                    Op::C,
                    unit,
                    height,
                    width,
                    block,
                    width,
                    panel,
                    height,
                );
                if unit {
                    for j in 0..width {
                        let pivot = pivots[first + j];
                        for x in &mut panel[j * height..(j + 1) * height] {
                            *x = over(*x, pivot);
                        }
                    }
                }
                let target = s.supernode[rows[0]];
                next[sn] = head[target];
                head[target] = sn;
            }
        }
        Some(Self {
            symbolic,
            unit,
            diagonal,
            below,
            pivots,
        })
    }

    /// The square block on the diagonal of supernode `sn`.
    fn block(&self, sn: usize) -> &[T] {
        let s = &self.symbolic;
        &self.diagonal[s.diagonal_ptrs[sn]..s.diagonal_ptrs[sn + 1]]
    }

    /// The block of the rows below supernode `sn`.
    fn panel(&self, sn: usize) -> &[T] {
        let s = &self.symbolic;
        &self.below[s.below_ptrs[sn]..s.below_ptrs[sn + 1]]
    }

    /// The rows below supernode `sn`.
    fn rows(&self, sn: usize) -> &[usize] {
        let s = &self.symbolic;
        &s.rows[s.row_ptrs[sn]..s.row_ptrs[sn + 1]]
    }

    /// The real parts of the diagonal of what is held, in the inner numbering: the diagonal of
    /// `L` for `L·Lᴴ`, and `D` for `L·D·Lᴴ`.
    pub(super) fn diagonal(&self) -> impl Iterator<Item = T::Real> + '_ {
        let s = &self.symbolic;
        (0..s.supernodes()).flat_map(move |sn| {
            let width = s.first[sn + 1] - s.first[sn];
            let block = self.block(sn);
            (0..width).map(move |j| block[j + j * width].re())
        })
    }

    /// `D` in the numbering of `P·A·Pᵀ`, for `L·D·Lᴴ`.
    pub(super) fn pivots(&self) -> Vec<T::Real> {
        let mut d = vec![T::Real::ZERO; self.symbolic.n];
        for (&pivot, &j) in self.pivots.iter().zip(&self.symbolic.postorder) {
            d[j] = pivot;
        }
        d
    }

    /// `L` in the numbering of `P·A·Pᵀ`, as a sparse matrix, with its ones on the diagonal for
    /// `L·D·Lᴴ`. An element that is zero is not stored: one the amalgamation of supernodes
    /// holds, and one that the arithmetic happened to cancel.
    pub(super) fn l(&self) -> SparseMatrix<T> {
        let s = &self.symbolic;
        let n = s.n;
        // A row of the result wants its elements in ascending order of the column, so the columns
        // are visited in the order of `P·A·Pᵀ` and not in the order they are held in.
        let mut place = vec![0usize; n];
        for (k, &j) in s.postorder.iter().enumerate() {
            place[j] = k;
        }
        let mut row_ptrs = vec![0usize; n + 1];
        self.for_each(&place, |i, _, _| row_ptrs[i + 1] += 1);
        for i in 0..n {
            row_ptrs[i + 1] += row_ptrs[i];
        }
        let mut col_indices = vec![0usize; row_ptrs[n]];
        let mut values = vec![T::zero(); row_ptrs[n]];
        let mut cursor = row_ptrs[..n].to_vec();
        self.for_each(&place, |i, j, x| {
            col_indices[cursor[i]] = j;
            values[cursor[i]] = x;
            cursor[i] += 1;
        });
        SparseMatrix::from_csr(n, n, row_ptrs, col_indices, values)
    }

    /// Calls `f(i, j, x)` for every nonzero `x` of `L`, at row `i` and column `j` of `P·A·Pᵀ`,
    /// column after column in ascending order; `place` is the inverse of the postorder.
    fn for_each(&self, place: &[usize], mut f: impl FnMut(usize, usize, T)) {
        let s = &self.symbolic;
        for (j, &k) in place.iter().enumerate() {
            let sn = s.supernode[k];
            let (first, end) = (s.first[sn], s.first[sn + 1]);
            let (width, c) = (end - first, k - first);
            let column = &self.block(sn)[c * width..(c + 1) * width];
            if self.unit {
                f(j, j, T::one());
            } else {
                f(j, j, column[c]);
            }
            for i in k + 1..end {
                let x = column[i - first];
                if !x.is_zero() {
                    f(s.postorder[i], j, x);
                }
            }
            let rows = self.rows(sn);
            let height = rows.len();
            let column = &self.panel(sn)[c * height..(c + 1) * height];
            for (&i, &x) in rows.iter().zip(column) {
                if !x.is_zero() {
                    f(s.postorder[i], j, x);
                }
            }
        }
    }

    /// The solution `X` of `A·X = B`, for `B` of any number of columns.
    ///
    /// # Panics
    ///
    /// Panics if `b` has not as many rows as the matrix.
    #[track_caller]
    pub(super) fn solve(&self, b: &Matrix<T>) -> Matrix<T> {
        let n = self.symbolic.n;
        assert_eq!(b.rows(), n, "the right-hand side has not as many rows");
        Matrix::from_column_major(n, b.cols(), self.solve_columns(b.as_slice(), b.cols()))
    }

    /// The solution `x` of `A·x = b`.
    ///
    /// # Panics
    ///
    /// Panics if `b` is not as long as the matrix has rows.
    #[track_caller]
    pub(super) fn solve_vec(&self, b: &[T]) -> Vec<T> {
        assert_eq!(
            b.len(),
            self.symbolic.n,
            "the right-hand side has not as many elements as the matrix has rows"
        );
        self.solve_columns(b, 1)
    }

    /// The solutions of `A·x = b` for the `columns` right-hand sides in `b`, one after another:
    /// `Pᵀ·L⁻ᴴ·D⁻¹·L⁻¹·P·b`, the permutation being the fill-reducing one and the postorder
    /// together.
    fn solve_columns(&self, b: &[T], columns: usize) -> Vec<T> {
        let s = &self.symbolic;
        let n = s.n;
        let mut y = vec![T::zero(); b.len()];
        if n == 0 || columns == 0 {
            return y;
        }
        for (y, b) in y.chunks_exact_mut(n).zip(b.chunks_exact(n)) {
            for (y, &old) in y.iter_mut().zip(&s.original) {
                *y = b[old];
            }
        }
        self.solve_in_place(&mut y, columns);
        let mut x = vec![T::zero(); b.len()];
        for (x, y) in x.chunks_exact_mut(n).zip(y.chunks_exact(n)) {
            for (&y, &old) in y.iter().zip(&s.original) {
                x[old] = y;
            }
        }
        x
    }

    /// Solves `L·D·Lᴴ·x = y` in place (`D` the identity for `L·Lᴴ`), for the `columns`
    /// right-hand sides in `y`, in the inner numbering, with `n > 0`. The rows of a supernode's
    /// own columns lie together in `y`, so the dense kernels work on them where they are; the
    /// rows below are gathered into working memory and scattered back.
    fn solve_in_place(&self, y: &mut [T], columns: usize) {
        let s = &self.symbolic;
        let n = s.n;
        let count = s.supernodes();
        let tallest = (0..count).map(|sn| self.rows(sn).len()).max().unwrap_or(0);
        let mut gathered = vec![T::zero(); tallest * columns];
        // `L·z = y`, the supernodes in order: the block on the diagonal, then its effect on the
        // rows below.
        for sn in 0..count {
            let (first, width) = (s.first[sn], s.first[sn + 1] - s.first[sn]);
            let (block, panel, rows) = (self.block(sn), self.panel(sn), self.rows(sn));
            let height = rows.len();
            trsm(
                Side::Left,
                Uplo::Lower,
                Op::N,
                self.unit,
                width,
                columns,
                block,
                width,
                &mut y[first..],
                n,
            );
            if height > 0 {
                let product = &mut gathered[..height * columns];
                gemm(
                    height,
                    columns,
                    width,
                    T::one(),
                    panel,
                    height,
                    Op::N,
                    &y[first..],
                    n,
                    Op::N,
                    T::zero(),
                    product,
                    height,
                );
                for (y, product) in y.chunks_exact_mut(n).zip(product.chunks_exact(height)) {
                    for (&i, &x) in rows.iter().zip(product) {
                        y[i] -= x;
                    }
                }
            }
        }
        if self.unit {
            for y in y.chunks_exact_mut(n) {
                for (y, &pivot) in y.iter_mut().zip(&self.pivots) {
                    *y = over(*y, pivot);
                }
            }
        }
        // `Lᴴ·x = z`, the supernodes in reverse: the rows below are solved already and are taken
        // out of the supernode's own, then the block on the diagonal.
        for sn in (0..count).rev() {
            let (first, width) = (s.first[sn], s.first[sn + 1] - s.first[sn]);
            let (block, panel, rows) = (self.block(sn), self.panel(sn), self.rows(sn));
            let height = rows.len();
            if height > 0 {
                let solved = &mut gathered[..height * columns];
                for (y, solved) in y.chunks_exact(n).zip(solved.chunks_exact_mut(height)) {
                    for (&i, x) in rows.iter().zip(solved) {
                        *x = y[i];
                    }
                }
                gemm(
                    width,
                    columns,
                    height,
                    -T::one(),
                    panel,
                    height,
                    Op::C,
                    solved,
                    height,
                    Op::N,
                    T::one(),
                    &mut y[first..],
                    n,
                );
            }
            trsm(
                Side::Left,
                Uplo::Lower,
                Op::C,
                self.unit,
                width,
                columns,
                block,
                width,
                &mut y[first..],
                n,
            );
        }
    }
}

/// `x / r` for a real `r`, each part divided: correctly rounded, which the product by the
/// reciprocal is not, and without the reciprocal's overflow for a tiny `r`.
#[inline]
fn over<T: Scalar>(x: T, r: T::Real) -> T {
    T::from_parts(x.re() / r, x.im() / r)
}

/// The order up to which [`ldl_in_place`] factors a block column by column.
const LEAF: usize = 32;

/// The width of the column blocks in which [`ldl_in_place`] updates the trailing block.
const UPDATE: usize = 64;

/// The `L·D·Lᴴ` decomposition, without pivoting, of the Hermitian `n × n` block at the start of
/// `a`, column after column with the leading dimension `ld`, in place: the part of the block
/// below the diagonal becomes that of the unit lower triangular `L`, and the diagonal `D`, which
/// is real. `false` when a pivot is zero or not finite, the block then left half done. Only the
/// lower triangle is read, and of the diagonal the real parts; what is above the diagonal may be
/// overwritten. `work` is working memory, which is made as long as is needed.
///
/// A block of more than [`LEAF`] columns is halved, as the dense Cholesky decomposition is:
/// `A₁₁ = L₁₁·D₁·L₁₁ᴴ`, then `L₂₁·D₁ = A₂₁·L₁₁⁻ᴴ` by a triangular solve on the right, then
/// `A₂₂ − L₂₁·D₁·L₂₁ᴴ` on its lower triangle, by products of blocks, factored the same way. So
/// nearly all the work of a large block is `gemm`'s.
pub(super) fn ldl_in_place<T: Scalar>(a: &mut [T], n: usize, ld: usize, work: &mut Vec<T>) -> bool {
    if n <= LEAF {
        return ldl_leaf(a, n, ld);
    }
    let n1 = n / 2;
    let n2 = n - n1;
    if !ldl_in_place(a, n1, ld, work) {
        return false;
    }
    // Two blocks of working memory: `L₁₁`, which the solve cannot read where it is, as it shares
    // its columns with the block being solved; and `L₂₁·D₁`, the solve's result, which the
    // product needs beside `L₂₁`.
    if work.len() < n1 * n {
        work.resize(n1 * n, T::zero());
    }
    let (l11, with_d) = work[..n1 * n].split_at_mut(n1 * n1);
    let (left, right) = a.split_at_mut(n1 * ld);
    for j in 0..n1 {
        l11[j * n1 + j..(j + 1) * n1].copy_from_slice(&left[j * ld + j..j * ld + n1]);
    }
    trsm(
        Side::Right,
        Uplo::Lower,
        Op::C,
        true,
        n2,
        n1,
        l11,
        n1,
        &mut left[n1..],
        ld,
    );
    for j in 0..n1 {
        let pivot = left[j + j * ld].re();
        let column = &mut left[j * ld + n1..j * ld + n];
        with_d[j * n2..(j + 1) * n2].copy_from_slice(column);
        for x in column {
            *x = over(*x, pivot);
        }
    }
    // `A₂₂ −= L₂₁·(L₂₁·D₁)ᴴ`, a block of columns at a time from its diagonal down.
    let l21 = &left[n1..];
    let a22 = &mut right[n1..];
    let mut j0 = 0;
    while j0 < n2 {
        let width = UPDATE.min(n2 - j0);
        gemm(
            n2 - j0,
            width,
            n1,
            -T::one(),
            &l21[j0..],
            ld,
            Op::N,
            &with_d[j0..],
            n2,
            Op::C,
            T::one(),
            &mut a22[j0 + j0 * ld..],
            ld,
        );
        j0 += width;
    }
    ldl_in_place(a22, n2, ld, work)
}

/// [`ldl_in_place`] of a small block, column by column: the pivot, then what the column takes out
/// of the columns after it, then its division by the pivot.
fn ldl_leaf<T: Scalar>(a: &mut [T], n: usize, ld: usize) -> bool {
    for j in 0..n {
        // The imaginary part of a pivot is the rounding of products of conjugates: dropped.
        let pivot = a[j + j * ld].re();
        if pivot == T::Real::ZERO || !pivot.is_finite() {
            return false;
        }
        a[j + j * ld] = T::from_real(pivot);
        // Below the pivot the column still holds `L·D`, so the element `(i, k)` after it loses
        // `(l_ij·d_j)·conj(l_kj)`.
        for k in j + 1..n {
            let (before, from) = a.split_at_mut(k * ld);
            let column = &before[j * ld..];
            let multiplier = over(column[k], pivot).conj();
            for (x, &y) in from[k..n].iter_mut().zip(&column[k..n]) {
                *x = x.mul_sub(y, multiplier);
            }
        }
        for x in &mut a[j * ld + j + 1..j * ld + n] {
            *x = over(*x, pivot);
        }
    }
    true
}
