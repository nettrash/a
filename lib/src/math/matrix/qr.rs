//! The QR decomposition by Householder reflections.
//!
//! # The decomposition
//!
//! A matrix of at most [`SMALL`] elements, and a complex one of at most 64 columns and 32768
//! elements, is decomposed one reflection at a time ([`unblocked`], LAPACK's `geqr2`): each column's
//! reflection taken from it, then applied to the columns right of it four at a time, each read for
//! its products with the reflection's vector and again, from the cache, for the update
//! ([`kernels::reflect`]). Larger matrices go by blocks of `nb` columns ([`block_width`]): a panel
//! of `nb` columns is decomposed and gives the triangular factor `T` of its block of reflections,
//! `H_0·…·H_nb-1 = I - V·T·V*` (the compact WY form), and the rest of the matrix takes the block
//! by three products, `C -= V·(T*·(V*·C))`, with `V` read where the panel left it, its unit
//! triangle written over `R` for the time ([`with_v`]).
//!
//! A panel is halved down to leaves of [`LEAF`] columns as LAPACK's `geqrt3` does ([`panel`]):
//! the left half, its block applied to the right half by products of blocks, the right half, and
//! the halves' factors joined, `T12 = -T1·V1*·V2·T2`. A leaf is decomposed one reflection at a time,
//! and each reflection's products with the columns left of it, which make its column of `T`, are
//! taken in the pass that applies it to those right of it.
//!
//! The block widens with the work of the update against that of the panel: 8 columns up to 384
//! columns of `f64` (192 of `f32`, 96 complex), then 16, 24 and 32, measured with each width from
//! 8 to 64 on square matrices and ones of twice and eight times as many rows as columns. At
//! `1024 × 1024` 24 took 30.3 ms, 16 30.9, 8 32.1 and 64 32.3, and 32 as 24 in other runs; at
//! `256 × 256` 8 took 556 µs and 32 610. A matrix of four times as many rows as columns takes at
//! least 16 columns from 512 rows, 48 from 2048 and 64 from 4096: the product `V*·C` of so tall
//! a block of 32 columns or fewer ran at 41 GFLOP/s against 54 for 64, and an `8192 × 1024`
//! decomposition took 337 ms with 64 against 372 with 32.
//!
//! The decomposition keeps the blocks' `T` with the reflections: forming `Q` and the least
//! squares reuse them.
//!
//! # `Q`
//!
//! `Q` is formed from the last block back, as LAPACK's `orgqr` ([`Qr::form_q`]): each block gives
//! its own columns directly, `[I; 0] - V·T·E1*` (`E1` the block's unit triangle), and updates
//! those right of it, whose rows of the block are still zeros, by three products. The columns
//! right of the last block are those of the identity, so that their product with `V*` is a copy
//! of it. The whole `Q` of a matrix of at least four times as many rows as columns is formed in
//! one block of all the reflections, whose `T` is built from `V*·V` ([`Qr::whole_t`]): most of
//! the work is the columns right of the reflections, and in one block it is the copy and one
//! product, half of the work of applying the blocks in turn: the whole `Q` of a `1024 × 128`
//! matrix took 6.4 ms, against 12.3 by blocks of 32 before and faer's 10.9. A decomposition
//! without blocks forms `Q` one reflection at a time.
//!
//! # Least squares
//!
//! `Q*·B` takes the blocks' `T` for more than [`FOURS`] columns of `B`, by products of blocks;
//! for up to four, each column takes four reflections at a time by the diagonal blocks of their
//! blocks' `T`, `w = T*·V*·y` in one pass over the column and `y -= V·w` in another
//! ([`Qr::reflect_fours`]), half the passes of one reflection at a time. `R·X = Q*·B` is solved
//! by columns of `R` for one column of `B` or an `R` of at most [`SUBSTITUTE`] columns
//! ([`back_substitute`]), and by the blocked triangular solve otherwise.
//!
//! # Speed
//!
//! Measured on an Apple M4 Max, one core, the median of interleaved runs while other processes
//! ran, against faer 0.24 (`Par::Seq`), nalgebra 0.35 and Accelerate (`dgeqrf`, `dorgqr`,
//! `dgels`, `dormqr` with `dtrtrs`; on the matrix unit); `f64`, µs unless ms. The largest varied
//! by up to 10 % from run to run, their ratios by less. `solve(b)` is of one column, given the
//! decomposition; `qr().solve(b)` includes it. nalgebra has no whole `Q` of a non-square matrix.
//!
//! | `f64` | ours | faer | nalgebra | Accelerate |
//! | --- | --- | --- | --- | --- |
//! | `qr()` 2 × 2 | 0.05 | 0.15 | 0.07 | 0.11 |
//! | `qr()` 8 × 8 | 0.22 | 0.50 | 0.35 | 0.48 |
//! | `qr()` 32 × 32 | 2.52 | 7.18 | 4.40 | 4.20 |
//! | `qr()` 128 × 128 | 82.2 | 147 | 176 | 109 |
//! | `qr()` 512 × 512 | 4.44 ms | 5.15 ms | 13.56 ms | 2.30 ms |
//! | `qr()` 1024 × 1024 | 32.47 ms | 35.42 ms | 102.36 ms | 12.25 ms |
//! | `qr()` 2048 × 1024 | 79.24 ms | 83.14 ms | 247.62 ms | 27.22 ms |
//! | `qr()` 8192 × 1024 | 365.6 ms | 381.5 ms | 1259.7 ms | 114.7 ms |
//! | `q()` 32 × 32 | 2.18 | 5.24 | 3.56 | 4.08 |
//! | `q()` 1024 × 1024 | 28.12 ms | 44.29 ms | 95.46 ms | 11.12 ms |
//! | `q()` 8192 × 1024 | 310.3 ms | 601.5 ms | 1189.7 ms | 106.3 ms |
//! | `q_full()` 1024 × 128 | 6.39 ms | 10.85 ms | - | 8.84 ms |
//! | `q_full()` 8192 × 1024 | 2.93 s | 4.72 s | - | 3.85 s |
//! | `qr().solve(b)` 32 × 32 | 2.94 | 8.60 | 5.77 | 6.82 |
//! | `qr().solve(b)` 1024 × 1024 | 30.16 ms | 32.87 ms | 96.37 ms | 12.76 ms |
//! | `qr().solve(b)` 8192 × 1024 | 347.4 ms | 369.9 ms | 1181.6 ms | 124.2 ms |
//! | `solve(b)` 16 × 16 | 0.23 | 0.30 | 0.56 | 0.83 |
//! | `solve(b)` 1024 × 1024 | 229 | 389 | 2.00 ms | 1.06 ms |
//! | `solve(b)` 8192 × 1024 | 1.87 ms | 2.63 ms | 3.95 ms | 6.22 ms |
//!
//! `f32` and complex matrices compare alike: `qr()` of `512 × 512` took 2.18 ms in `f32` against
//! faer's 2.63 and nalgebra's 7.36, and 15.9 ms in `Complex<f64>` against 19.3 and 55.1.

mod kernels;

use super::blas::{Op, Side, Uplo, gemm, trsm};
use super::{Matrix, Scalar};
use kernels::{axpy, axpys4, dots, dots4, reflect, reflector};

/// The QR decomposition `A = Q·R` of an `m × n` matrix by Householder reflections: `Q` of
/// `m × min(m, n)` with orthonormal columns (unitary for complex elements), `R` of
/// `min(m, n) × n` upper triangular with a real diagonal. From [`Matrix::qr`].
///
/// # Examples
///
/// ```
/// use a::math::matrix::Matrix;
///
/// let a: Matrix<f64> = Matrix::from_rows(&[[3.0, 1.0], [4.0, 2.0]]);
/// let qr = a.qr();
/// let (q, r) = (qr.q(), qr.r());
/// assert!((&q * &r - &a).as_slice().iter().all(|x| x.abs() < 1e-15));
/// assert!((&q.transpose() * &q - Matrix::identity(2)).as_slice().iter().all(|x| x.abs() < 1e-15));
/// assert!((r[(0, 0)].abs() - 5.0).abs() < 1e-15);
/// ```
#[derive(Clone, Debug)]
pub struct Qr<T> {
    /// `R` on and above the diagonal, the reflections' vectors below it.
    qr: Matrix<T>,
    /// The reflections' factors.
    taus: Vec<T>,
    /// The triangular factors of the blocks of reflections, `H_j0·…·H_j1-1 = I - V·T·V*` for
    /// the block of columns `j0..j1`, whose `T` starts at `j0 · nb` with the leading dimension
    /// `nb`; empty when the decomposition was not blocked.
    ts: Vec<T>,
    /// The width `nb` of the blocks, 0 when the decomposition was not blocked.
    nb: usize,
}

/// The most elements of a matrix decomposed one reflection at a time. Beyond, blocks of 8
/// columns were faster: `f64` matrices of `64 × 64` took 14.3 µs by them against 14.6, of
/// `256 × 32` 22.7 against 25.8, while `48 × 48` took 7.3 against 7.5 and `128 × 16` 3.6 against
/// 4.2; `f32` and complex ones of the same shapes broke even at about the same size, but complex
/// ones of few columns stayed faster one reflection at a time up to `512 × 64`.
const SMALL: usize = 4096;

/// The columns of the leaves of a panel, decomposed one reflection at a time: 4 and 16 were as
/// fast as 8 within 5 % from `64 × 64` to `8192 × 1024`, 32 up to 10 % slower.
const LEAF: usize = 8;

/// The most columns of `B` whose `Q*·B` goes a column at a time, four reflections at a time;
/// with more, products of blocks were faster: for 8 columns of 1024 rows 916 µs against 963.
const FOURS: usize = 4;

/// The most columns of `R` solved for by columns for any number of columns of `B`: for 4 columns
/// of 128 rows 9.4 µs against 10.2 by the triangular solve, of 192 rows 22.7 against 20.5.
const SUBSTITUTE: usize = 128;

/// The fewest rows of a matrix whose whole `Q` is formed in one block: `16 × 4` took 0.41 µs so
/// against 0.25 one reflection at a time, `64 × 16` 6.5 against 8.8.
const ONE_BLOCK: usize = 48;

/// The reflections whose triangular factor [`Qr::whole_t`] builds from `V*·V` directly, before
/// joining them by products of blocks.
const LEAF_T: usize = 64;

/// The width of the blocks of the decomposition of an `m × n` matrix, 0 for none: see the
/// module's documentation.
fn block_width<T: Scalar>(m: usize, n: usize) -> usize {
    let size = m * n;
    if size <= SMALL || (T::IS_COMPLEX && n <= 64 && size <= 8 * SMALL) {
        return 0;
    }
    // A complex element takes four times the work of a real one, and an `f32` half the time of
    // an `f64`, at the same speed of the products of blocks.
    let e = n * if T::IS_COMPLEX {
        4
    } else if core::mem::size_of::<T>() == 4 {
        2
    } else {
        1
    };
    let nb = match e {
        0..=384 => 8,
        385..=640 => 16,
        641..=896 => 24,
        _ => 32,
    };
    if m >= 4 * n {
        let tall = match m {
            0..=511 => 8,
            512..=2047 => 16,
            2048..=4095 => 48,
            _ => 64,
        };
        nb.max(tall)
    } else {
        nb
    }
}

impl<T: Scalar> Matrix<T> {
    /// The QR decomposition `A = Q·R`: see [`Qr`]. About `2mn² - 2n³/3` operations for
    /// `m ≥ n`, most of them, for large matrices, in products of blocks of reflections.
    pub fn qr(&self) -> Qr<T> {
        let (m, n) = self.shape();
        let mut qr = self.clone();
        let k = m.min(n);
        let mut taus = vec![T::zero(); k];
        let nb = block_width::<T>(m, n);
        if nb == 0 {
            unblocked(&mut qr.data, m, m, n, &mut taus, None, &mut []);
            return Qr {
                qr,
                taus,
                ts: Vec::new(),
                nb: 0,
            };
        }
        let mut ts = vec![T::zero(); nb * k];
        let mut work = Work::new(nb, n);
        let mut j0 = 0;
        while j0 < k {
            let kb = nb.min(k - j0);
            let j1 = j0 + kb;
            let (left, right) = qr.data.split_at_mut(j1 * m);
            let block = &mut left[j0 + j0 * m..];
            panel(
                block,
                m,
                m - j0,
                kb,
                &mut taus[j0..j1],
                &mut ts[j0 * nb..],
                nb,
                &mut work,
            );
            if j1 < n {
                let Work { save, w, .. } = &mut work;
                let t = &ts[j0 * nb..];
                with_v(block, m, kb, save, |v| {
                    apply_block(v, m, m - j0, kb, t, nb, &mut right[j0..], m, n - j1, w);
                });
            }
            j0 = j1;
        }
        Qr { qr, taus, ts, nb }
    }
}

impl<T: Scalar> Qr<T> {
    /// `Q`, `m × min(m, n)`, with orthonormal columns.
    pub fn q(&self) -> Matrix<T> {
        self.form_q(self.taus.len())
    }

    /// The whole `m × m` unitary `Q`, whose first `min(m, n)` columns are [`q`](Qr::q).
    pub fn q_full(&self) -> Matrix<T> {
        self.form_q(self.qr.rows())
    }

    /// `R`, `min(m, n) × n`, upper triangular.
    pub fn r(&self) -> Matrix<T> {
        let (m, n) = self.qr.shape();
        let k = m.min(n);
        let mut r = vec![T::zero(); k * n];
        for (j, col) in r.chunks_exact_mut(k.max(1)).take(n).enumerate() {
            let h = k.min(j + 1);
            col[..h].copy_from_slice(&self.qr.data[j * m..j * m + h]);
        }
        Matrix::from_column_major(k, n, r)
    }

    /// The `X` of the least squares, which makes `‖A·X - B‖` the smallest, for `m ≥ n`: the
    /// solution of `R·X = Q*·B`, or `None` when a diagonal element of `R` is zero, the
    /// columns of `A` dependent. For a square matrix it solves `A·X = B`.
    ///
    /// # Panics
    ///
    /// Panics if the matrix has fewer rows than columns, or `B` not as many rows as it.
    #[track_caller]
    pub fn solve(&self, b: &Matrix<T>) -> Option<Matrix<T>> {
        let (m, n) = self.qr.shape();
        assert!(m >= n, "the matrix has fewer rows than columns");
        assert_eq!(b.rows(), m, "the right-hand side has not as many rows");
        if (0..n).any(|i| self.qr[(i, i)] == T::zero()) {
            return None;
        }
        let cols = b.cols();
        let mut y = b.clone();
        let a = &self.qr.data;
        if cols == 0 {
        } else if self.nb == 0 {
            // One reflection at a time, `y ← H_j*·y`.
            for (j, &tau) in self.taus.iter().enumerate() {
                if tau != T::zero() {
                    let v = &a[j * m + j + 1..(j + 1) * m];
                    reflect(v, tau.conj(), &mut y.data[j..], m, cols);
                }
            }
        } else if cols <= FOURS {
            for y in y.data.chunks_exact_mut(m) {
                self.reflect_fours(y);
            }
        } else {
            self.apply_blocks(&mut y.data, cols);
        }
        // The first `n` rows of `y`, moved up in place.
        for j in 1..cols {
            y.data.copy_within(j * m..j * m + n, j * n);
        }
        y.data.truncate(n * cols);
        let mut x = Matrix::from_column_major(n, cols, y.data);
        if n == 0 {
        } else if cols == 1 || n <= SUBSTITUTE {
            for x in x.data.chunks_exact_mut(n) {
                back_substitute(a, m, n, x);
            }
        } else {
            trsm(
                Side::Left,
                Uplo::Upper,
                Op::N,
                false,
                n,
                cols,
                a,
                m,
                &mut x.data,
                n,
            );
        }
        Some(x)
    }

    /// `Y ← Q*·Y` for the `m × cols` block `y` (leading dimension `m`), a block of reflections at
    /// a time: `W = V*·Y` by the block's unit triangle and the rows below it, `W ← T*·W`, and
    /// `Y -= V·W`.
    fn apply_blocks(&self, y: &mut [T], cols: usize) {
        let m = self.qr.rows();
        let (k, nb) = (self.taus.len(), self.nb);
        let a = &self.qr.data;
        let mut e1 = vec![T::zero(); nb * nb];
        let mut w = vec![T::zero(); 2 * nb * cols];
        let mut j0 = 0;
        while j0 < k {
            let kb = nb.min(k - j0);
            let j1 = j0 + kb;
            unit_lower(&a[j0 + j0 * m..], m, kb, &mut e1);
            let v2 = &a[j1 + j0 * m..];
            let t = &self.ts[j0 * nb..];
            let (w1, w2) = w.split_at_mut(kb * cols);
            let w2 = &mut w2[..kb * cols];
            gemm(
                kb,
                cols,
                kb,
                T::one(),
                &e1,
                kb,
                Op::C,
                &y[j0..],
                m,
                Op::N,
                T::zero(),
                w1,
                kb,
            );
            gemm(
                kb,
                cols,
                m - j1,
                T::one(),
                v2,
                m,
                Op::C,
                &y[j1..],
                m,
                Op::N,
                T::one(),
                w1,
                kb,
            );
            gemm(
                kb,
                cols,
                kb,
                T::one(),
                t,
                nb,
                Op::C,
                w1,
                kb,
                Op::N,
                T::zero(),
                w2,
                kb,
            );
            gemm(
                kb,
                cols,
                kb,
                -T::one(),
                &e1,
                kb,
                Op::N,
                w2,
                kb,
                Op::N,
                T::one(),
                &mut y[j0..],
                m,
            );
            gemm(
                m - j1,
                cols,
                kb,
                -T::one(),
                v2,
                m,
                Op::N,
                w2,
                kb,
                Op::N,
                T::one(),
                &mut y[j1..],
                m,
            );
            j0 = j1;
        }
    }

    /// `y ← Q*·y` for a column `y`, four reflections at a time by the diagonal blocks of their
    /// blocks' `T`, `I - V·T·V*` of four: `w = T*·V*·y` in one pass over `y` and `y -= V·w` in
    /// another, the rows below the four's triangle by kernels; the reflections left over one at a
    /// time.
    fn reflect_fours(&self, y: &mut [T]) {
        let m = self.qr.rows();
        let (k, nb) = (self.taus.len(), self.nb);
        let a = &self.qr.data;
        let mut j0 = 0;
        while j0 < k {
            let j1 = (j0 + nb).min(k);
            let t = &self.ts[j0 * nb..];
            let mut j = j0;
            while j + 4 <= j1 && j + 4 <= m {
                // `V`'s rows `j..j + 4`, unit lower triangular, and the rows below them.
                let at = |r: usize, c: usize| a[j + r + (j + c) * m];
                let col = |c: usize| &a[(j + c) * m + j + 4..(j + c + 1) * m];
                let cols = [col(0), col(1), col(2), col(3)];
                let (top, bottom) = y[j..].split_at_mut(4);
                let mut w = dots4(bottom, cols);
                for (c, w) in w.iter_mut().enumerate() {
                    *w += top[c];
                    for (r, &y) in top.iter().enumerate().skip(c + 1) {
                        *w += at(r, c).conj() * y;
                    }
                }
                // `w ← T*·w` for the four's block of `T`, upper triangular.
                let d = j - j0;
                let tt = |r: usize, c: usize| t[d + r + (d + c) * nb];
                for c in (0..4).rev() {
                    let mut x = T::zero();
                    for (r, &w) in w.iter().enumerate().take(c + 1) {
                        x += tt(r, c).conj() * w;
                    }
                    w[c] = x;
                }
                for r in 0..4 {
                    let mut x = w[r];
                    for (c, &w) in w.iter().enumerate().take(r) {
                        x += at(r, c) * w;
                    }
                    top[r] -= x;
                }
                axpys4(bottom, cols, w);
                j += 4;
            }
            for j in j..j1 {
                let tau = self.taus[j];
                if tau != T::zero() {
                    reflect(
                        &a[j * m + j + 1..(j + 1) * m],
                        tau.conj(),
                        &mut y[j..],
                        m,
                        1,
                    );
                }
            }
            j0 = j1;
        }
    }

    /// The triangular factor of all the `k` reflections in one block, `k × k`: `S = V*·V` by
    /// products of blocks, the factors of blocks of up to [`LEAF_T`] reflections from it as
    /// LAPACK's `larft` builds them, `T[0..j, j] = -τ_j·T[0..j, 0..j]·S[0..j, j]`, and those
    /// joined from the left by products of blocks, `T12 = -T1·S12·T2`.
    fn whole_t(&self) -> Vec<T> {
        let m = self.qr.rows();
        let k = self.taus.len();
        let a = &self.qr.data;
        let mut s = vec![T::zero(); k * k];
        {
            let mut e = vec![T::zero(); k * k];
            unit_lower(a, m, k, &mut e);
            gemm(
                k,
                k,
                k,
                T::one(),
                &e,
                k,
                Op::C,
                &e,
                k,
                Op::N,
                T::zero(),
                &mut s,
                k,
            );
            gemm(
                k,
                k,
                m - k,
                T::one(),
                &a[k..],
                m,
                Op::C,
                &a[k..],
                m,
                Op::N,
                T::one(),
                &mut s,
                k,
            );
        }
        let mut t = vec![T::zero(); k * k];
        let mut y = vec![T::zero(); k * LEAF_T];
        let mut z = vec![T::zero(); k * LEAF_T];
        let mut j0 = 0;
        while j0 < k {
            let j1 = (j0 + LEAF_T).min(k);
            let kb = j1 - j0;
            for j in j0..j1 {
                let tau = self.taus[j];
                t[j + j * k] = tau;
                if tau == T::zero() {
                    continue;
                }
                for i in j0..j {
                    let mut x = T::zero();
                    for p in i..j {
                        x += t[i + p * k] * s[p + j * k];
                    }
                    t[i + j * k] = -tau * x;
                }
            }
            if j0 > 0 {
                let (y, z) = (&mut y[..j0 * kb], &mut z[..j0 * kb]);
                gemm(
                    j0,
                    kb,
                    j0,
                    T::one(),
                    &t,
                    k,
                    Op::N,
                    &s[j0 * k..],
                    k,
                    Op::N,
                    T::zero(),
                    y,
                    j0,
                );
                gemm(
                    j0,
                    kb,
                    kb,
                    -T::one(),
                    y,
                    j0,
                    Op::N,
                    &t[j0 + j0 * k..],
                    k,
                    Op::N,
                    T::zero(),
                    z,
                    j0,
                );
                for (c, z) in z.chunks_exact(j0).enumerate() {
                    t[(j0 + c) * k..(j0 + c) * k + j0].copy_from_slice(z);
                }
            }
            j0 = j1;
        }
        t
    }

    /// The first `cols` columns of `Q = H_0·H_1·…·H_k-1`, `cols` either `k` or `m`, from the
    /// last reflection back, as LAPACK's `orgqr`: each block of reflections gives its own columns
    /// directly and updates those right of it. The matrix starts as zeros: the identity's columns
    /// from `k` on are taken in by the last block, or set for the reflections one at a time.
    fn form_q(&self, cols: usize) -> Matrix<T> {
        let m = self.qr.rows();
        let k = self.taus.len();
        let a = &self.qr.data;
        let mut q = Matrix::from_column_major(m, cols, vec![T::zero(); m * cols]);
        // The whole `Q` of a tall matrix: all the reflections in one block, whose product with
        // the columns of the identity right of them, most of the work, is a copy.
        let one = cols > k && m >= 4 * k && m >= ONE_BLOCK;
        let nb = if one { k } else { self.nb };
        if nb == 0 || k == 0 {
            for j in k..cols {
                q.data[j + j * m] = T::one();
            }
            for j in (0..k).rev() {
                let tau = self.taus[j];
                let v = &a[j * m + j + 1..(j + 1) * m];
                let (left, right) = q.data.split_at_mut((j + 1) * m);
                let rest = cols - j - 1;
                if rest > 0 && tau != T::zero() {
                    // Row `j` of the columns right of `j` is still zero.
                    reflect(v, tau, &mut right[j..], m, rest);
                }
                left[j + j * m] = T::one() - tau;
                for (q, &v) in left[j * m + j + 1..].iter_mut().zip(v) {
                    *q = -tau * v;
                }
            }
            return q;
        }
        let built;
        let ts: &[T] = if one {
            built = self.whole_t();
            &built
        } else {
            &self.ts
        };
        let mut e1 = vec![T::zero(); nb * nb];
        let mut y = vec![T::zero(); nb * nb];
        let mut w = vec![T::zero(); 2 * nb * cols];
        let mut end = k;
        while end > 0 {
            let j0 = (end - 1) / nb * nb;
            let j1 = end;
            let kb = j1 - j0;
            unit_lower(&a[j0 + j0 * m..], m, kb, &mut e1);
            let v2 = &a[j1 + j0 * m..];
            let t = &ts[j0 * nb..];
            let (left, right) = q.data.split_at_mut(j1 * m);
            let rest = cols - j1;
            // The last block, whose columns right of it are still those of the identity.
            let last = j1 == k;
            if rest > 0 {
                // The columns right of the block, zeros in its rows: `W = T·V2*·C2`, then
                // `C1 = -E1·W` and `C2 -= V2·W`.
                let (w1, w2) = w.split_at_mut(kb * rest);
                let w2 = &mut w2[..kb * rest];
                if last {
                    // `V2*·I`.
                    for (c, w) in w1.chunks_exact_mut(kb).enumerate() {
                        for (i, w) in w.iter_mut().enumerate() {
                            *w = v2[c + i * m].conj();
                        }
                    }
                } else {
                    gemm(
                        kb,
                        rest,
                        m - j1,
                        T::one(),
                        v2,
                        m,
                        Op::C,
                        &right[j1..],
                        m,
                        Op::N,
                        T::zero(),
                        w1,
                        kb,
                    );
                }
                gemm(
                    kb,
                    rest,
                    kb,
                    T::one(),
                    t,
                    nb,
                    Op::N,
                    w1,
                    kb,
                    Op::N,
                    T::zero(),
                    w2,
                    kb,
                );
                gemm(
                    kb,
                    rest,
                    kb,
                    -T::one(),
                    &e1,
                    kb,
                    Op::N,
                    w2,
                    kb,
                    Op::N,
                    T::zero(),
                    &mut right[j0..],
                    m,
                );
                // `C2 = I - V2·W` for the last block, the identity not held yet.
                gemm(
                    m - j1,
                    rest,
                    kb,
                    -T::one(),
                    v2,
                    m,
                    Op::N,
                    w2,
                    kb,
                    Op::N,
                    if last { T::zero() } else { T::one() },
                    &mut right[j1..],
                    m,
                );
                if last {
                    for c in 0..rest.min(m - j1) {
                        right[j1 + c + c * m] += T::one();
                    }
                }
            }
            // The block's own columns, `[I; 0] - V·T·E1*`.
            gemm(
                kb,
                kb,
                kb,
                T::one(),
                t,
                nb,
                Op::N,
                &e1,
                kb,
                Op::C,
                T::zero(),
                &mut y,
                kb,
            );
            gemm(
                kb,
                kb,
                kb,
                -T::one(),
                &e1,
                kb,
                Op::N,
                &y,
                kb,
                Op::N,
                T::zero(),
                &mut left[j0 + j0 * m..],
                m,
            );
            gemm(
                m - j1,
                kb,
                kb,
                -T::one(),
                v2,
                m,
                Op::N,
                &y,
                kb,
                Op::N,
                T::zero(),
                &mut left[j1 + j0 * m..],
                m,
            );
            for i in j0..j1 {
                left[i + i * m] += T::one();
            }
            end = j0;
        }
        q
    }
}

/// `x ← R⁻¹·x` for the upper triangle `R` of the `n × n` block `r`, its diagonal real, and a
/// column `x`, by columns of `R`: each element of `x`, once known, taken from those above it.
fn back_substitute<T: Scalar>(r: &[T], ldr: usize, n: usize, x: &mut [T]) {
    for j in (0..n).rev() {
        let d = r[j + j * ldr].re();
        let xj = T::from_parts(x[j].re() / d, x[j].im() / d);
        x[j] = xj;
        axpy(&r[j * ldr..j * ldr + j], xj, &mut x[..j]);
    }
}

/// Scratch of the blocked decomposition.
struct Work<T> {
    /// The elements of `R` a block's unit triangle stands in for while it is used.
    save: Vec<T>,
    /// Products of blocks of reflections with the rest of the matrix.
    w: Vec<T>,
    /// The products of a reflection's vector with the columns of a leaf.
    u: Vec<T>,
}

impl<T: Scalar> Work<T> {
    /// Scratch for blocks of `nb` reflections of a matrix of `n` columns.
    fn new(nb: usize, n: usize) -> Work<T> {
        Work {
            save: vec![T::zero(); nb * nb],
            w: vec![T::zero(); 2 * nb * n.max(nb)],
            u: vec![T::zero(); nb],
        }
    }
}

/// Factors the `m × n` panel `a`, `m ≥ n`, into its reflections and the triangular factor `T`
/// (`ldt`, zeros below its diagonal on entry) of their block, as LAPACK's `geqrt3`: the left
/// half, then its block applied to the right half, the right half, and the two halves' `T`
/// joined by products of blocks; leaves of [`LEAF`] columns one reflection at a time.
#[allow(clippy::too_many_arguments)]
fn panel<T: Scalar>(
    a: &mut [T],
    lda: usize,
    m: usize,
    n: usize,
    taus: &mut [T],
    t: &mut [T],
    ldt: usize,
    work: &mut Work<T>,
) {
    if n <= LEAF {
        unblocked(a, lda, m, n, taus, Some((t, ldt)), &mut work.u);
        return;
    }
    let n1 = n / 2;
    let n2 = n - n1;
    panel(a, lda, m, n1, &mut taus[..n1], t, ldt, work);
    let (l, r) = a.split_at_mut(n1 * lda);
    {
        let Work { save, w, .. } = &mut *work;
        let t1: &[T] = t;
        with_v(l, lda, n1, save, |v| {
            apply_block(v, lda, m, n1, t1, ldt, r, lda, n2, w);
        });
    }
    let (t1, t2) = t.split_at_mut(n1 * ldt);
    panel(
        &mut r[n1..],
        lda,
        m - n1,
        n2,
        &mut taus[n1..],
        &mut t2[n1..],
        ldt,
        work,
    );
    // `T12 = -T1·(V1*·V2)·T2`, `V1` from row `n1` on, where `V2` starts.
    let Work { save, w, .. } = work;
    let (x, y) = w.split_at_mut(n1 * n2);
    let y = &mut y[..n1 * n2];
    let l: &[T] = l;
    with_v(&mut r[n1..], lda, n2, save, |v2| {
        gemm(
            n1,
            n2,
            m - n1,
            T::one(),
            &l[n1..],
            lda,
            Op::C,
            v2,
            lda,
            Op::N,
            T::zero(),
            x,
            n1,
        );
    });
    gemm(
        n1,
        n2,
        n1,
        T::one(),
        t1,
        ldt,
        Op::N,
        x,
        n1,
        Op::N,
        T::zero(),
        y,
        n1,
    );
    gemm(
        n1,
        n2,
        n2,
        -T::one(),
        y,
        n1,
        Op::N,
        &t2[n1..],
        ldt,
        Op::N,
        T::zero(),
        x,
        n1,
    );
    for (t, x) in t2.chunks_mut(ldt).zip(x.chunks(n1)) {
        t[..n1].copy_from_slice(x);
    }
}

/// Factors the `m × n` block `a` by `min(m, n)` reflections, one at a time, as LAPACK's
/// `geqr2`, and with `t` (`ldt`, zeros below its diagonal on entry) also the triangular factor
/// of their block, as `larft`, from the same kind of products: for the reflection of column `j`,
/// the products of its vector with the columns right of it, which apply it, and with those left
/// of it, the reflections' vectors, which make the column `j` of `T`. `u` holds `n` elements of
/// scratch when `t` is given.
fn unblocked<T: Scalar>(
    a: &mut [T],
    lda: usize,
    m: usize,
    n: usize,
    taus: &mut [T],
    mut t: Option<(&mut [T], usize)>,
    u: &mut [T],
) {
    let k = m.min(n);
    for j in 0..k {
        let (left, right) = a.split_at_mut(((j + 1) * lda).min(a.len()));
        let col = &mut left[j * lda..j * lda + m];
        let (tau, beta) = reflector(col[j], &mut col[j + 1..]);
        col[j] = beta;
        taus[j] = tau;
        if tau == T::zero() {
            continue;
        }
        let v = &left[j * lda + j + 1..j * lda + m];
        let rest = n - j - 1;
        if rest > 0 {
            reflect(v, tau.conj(), &mut right[j..], lda, rest);
        }
        if let Some((t, ldt)) = &mut t {
            let ldt = *ldt;
            t[j + j * ldt] = tau;
            if j > 0 {
                // `T[0..j, j] = -τ·T[0..j, 0..j]·V[:, 0..j]*·v_j`, the products from row `j`,
                // where `v_j` starts.
                let s = &mut u[..j];
                dots(v, &left[j + 1..], lda, j, s);
                for (i, s) in s.iter_mut().enumerate() {
                    *s = (left[j + i * lda] + *s).conj();
                }
                for i in 0..j {
                    let mut x = T::zero();
                    for p in i..j {
                        x += t[i + p * ldt] * s[p];
                    }
                    t[i + j * ldt] = -tau * x;
                }
            }
        }
    }
}

/// `c ← (I - V·T·V*)*·c` for the `m × n` block `c`: `V` of `m × k` with its unit lower triangle
/// held, `T` of `k × k` upper triangular with zeros below; `W = V*·c`, `W ← T*·W`, `c -= V·W`.
/// `w` holds `2·k·n` elements of scratch.
#[allow(clippy::too_many_arguments)]
fn apply_block<T: Scalar>(
    v: &[T],
    ldv: usize,
    m: usize,
    k: usize,
    t: &[T],
    ldt: usize,
    c: &mut [T],
    ldc: usize,
    n: usize,
    w: &mut [T],
) {
    if n == 0 || k == 0 {
        return;
    }
    let (w1, w2) = w[..2 * k * n].split_at_mut(k * n);
    gemm(
        k,
        n,
        m,
        T::one(),
        v,
        ldv,
        Op::C,
        c,
        ldc,
        Op::N,
        T::zero(),
        w1,
        k,
    );
    gemm(
        k,
        n,
        k,
        T::one(),
        t,
        ldt,
        Op::C,
        w1,
        k,
        Op::N,
        T::zero(),
        w2,
        k,
    );
    gemm(
        m,
        n,
        k,
        -T::one(),
        v,
        ldv,
        Op::N,
        w2,
        k,
        Op::N,
        T::one(),
        c,
        ldc,
    );
}

/// Runs `f` on the `m × k` block `a` of reflections' vectors as `V`, its top `k × k` made the
/// unit lower triangle for the time: the elements of `R` there are kept in `save` and put back.
fn with_v<T: Scalar, X>(
    a: &mut [T],
    lda: usize,
    k: usize,
    save: &mut [T],
    f: impl FnOnce(&[T]) -> X,
) -> X {
    for j in 0..k {
        let col = &mut a[j * lda..j * lda + j + 1];
        save[j * k..j * k + j + 1].copy_from_slice(col);
        col[..j].fill(T::zero());
        col[j] = T::one();
    }
    let x = f(a);
    for j in 0..k {
        a[j * lda..j * lda + j + 1].copy_from_slice(&save[j * k..j * k + j + 1]);
    }
    x
}

/// The unit lower triangle of the top `k × k` of the reflections' vectors `a`, into `e`, `k × k`.
fn unit_lower<T: Scalar>(a: &[T], lda: usize, k: usize, e: &mut [T]) {
    for j in 0..k {
        let col = &mut e[j * k..(j + 1) * k];
        col[..j].fill(T::zero());
        col[j] = T::one();
        col[j + 1..].copy_from_slice(&a[j * lda + j + 1..j * lda + k]);
    }
}
