//! What the tests of the sparse module share: the element types under test, the random matrices,
//! and comparisons up to rounding against the dense ones.

#![allow(dead_code, unused_imports, unused_macros)] // CORE: until every area's tests use them.

use a::math::complex::{Complex, Float};
use a::math::matrix::Matrix;
use a::math::matrix::Scalar;
use a::math::sparse::{CscMatrix, SparseMatrix};

/// Scales a test down when it runs under Miri.
pub(super) fn size(normal: usize, miri: usize) -> usize {
    if cfg!(miri) { miri } else { normal }
}

/// Small deterministic generator.
pub(super) struct Rng(pub(super) u64);

impl Rng {
    /// The next 64 bits.
    pub(super) fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// A uniform float in `[-1, 1)`.
    pub(super) fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    }

    /// A uniform index below `n`.
    pub(super) fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// The element types the module is tested with.
macro_rules! each_type {
    ($f:ident) => {
        $f::<f32>();
        $f::<f64>();
        $f::<a::math::complex::Complex<f32>>();
        $f::<a::math::complex::Complex<f64>>();
    };
}
pub(super) use each_type;

/// An element type, seen through what the tests need of it.
pub(super) trait Num: Scalar + Copy + core::fmt::Debug {
    /// The unit roundoff of the type.
    const EPS: f64;
    /// A random element, its parts uniform in `[-1, 1)`.
    fn random(r: &mut Rng) -> Self;
    /// The real number `x`.
    fn real(x: f64) -> Self;
}

impl Num for f32 {
    const EPS: f64 = f32::EPSILON as f64;
    fn random(r: &mut Rng) -> Self {
        r.unit() as f32
    }
    fn real(x: f64) -> Self {
        x as f32
    }
}

impl Num for f64 {
    const EPS: f64 = f64::EPSILON;
    fn random(r: &mut Rng) -> Self {
        r.unit()
    }
    fn real(x: f64) -> Self {
        x
    }
}

impl Num for Complex<f32> {
    const EPS: f64 = f32::EPSILON as f64;
    fn random(r: &mut Rng) -> Self {
        Complex::new(r.unit() as f32, r.unit() as f32)
    }
    fn real(x: f64) -> Self {
        Complex::new(x as f32, 0.0)
    }
}

impl Num for Complex<f64> {
    const EPS: f64 = f64::EPSILON;
    fn random(r: &mut Rng) -> Self {
        Complex::new(r.unit(), r.unit())
    }
    fn real(x: f64) -> Self {
        Complex::new(x, 0.0)
    }
}

/// The absolute value of an element, as an `f64` to compare tolerances in.
pub(super) fn magnitude<T: Num>(x: T) -> f64 {
    T::abs(x).to_f64()
}

/// A matrix of the shape with about `density` stored elements, some of them repeated cells that
/// `from_coo` sums: the pattern a factorization is handed.
pub(super) fn random_sparse<T: Num>(
    r: &mut Rng,
    rows: usize,
    cols: usize,
    density: usize,
) -> SparseMatrix<T> {
    let mut entries = Vec::new();
    for _ in 0..density {
        let (i, j) = (r.below(rows), r.below(cols));
        entries.push((i, j, T::random(r)));
    }
    SparseMatrix::from_coo(rows, cols, &entries)
}

/// A Hermitian positive definite matrix: a random symmetric pattern with negative off-diagonal
/// elements and a diagonal larger than the row, which makes it positive definite.
pub(super) fn positive_definite<T: Num>(r: &mut Rng, n: usize, density: usize) -> SparseMatrix<T> {
    let mut entries = Vec::new();
    let mut sums = vec![0.0f64; n];
    for _ in 0..density {
        let (i, j) = (r.below(n), r.below(n));
        if i == j {
            continue;
        }
        let x = -1.0 - r.unit().abs();
        entries.push((i, j, T::real(x)));
        entries.push((j, i, T::real(x)));
        sums[i] += x.abs();
        sums[j] += x.abs();
    }
    for (i, &sum) in sums.iter().enumerate() {
        entries.push((i, i, T::real(sum + 1.0)));
    }
    SparseMatrix::from_coo(n, n, &entries)
}

/// A nonsingular matrix with a sparse pattern, for the methods and factors that need no symmetry.
pub(super) fn nonsingular<T: Num>(r: &mut Rng, n: usize, density: usize) -> SparseMatrix<T> {
    for _ in 0..20 {
        let a = random_sparse(r, n, n, density);
        if !a.to_matrix().lu().is_singular() {
            return a;
        }
    }
    // A dense matrix is nonsingular with probability 1, which the loop above can only miss by bad
    // luck with a pattern too thin to hold a full set of pivots.
    let entries: Vec<T> = (0..n * n).map(|_| T::random(r)).collect();
    SparseMatrix::from_matrix(&Matrix::from_column_major(n, n, entries))
}

/// The dense form, named for what the tests compare against.
pub(super) fn dense<T: Num>(a: &SparseMatrix<T>) -> Matrix<T> {
    a.to_matrix()
}

/// Whether two dense matrices agree to `factor` times the roundoff of the type, scaled by their
/// sizes and their magnitudes.
pub(super) fn assert_close<T: Num>(a: &Matrix<T>, b: &Matrix<T>, factor: f64) {
    let tolerance = factor * T::EPS * (a.rows().max(a.cols()) as f64);
    let scale = norm(a) + norm(b) + 1.0;
    let d = norm(&(a - b));
    assert!(
        d <= tolerance * scale,
        "the matrices differ by {d}, more than the tolerated {} of {scale}",
        tolerance * scale
    );
}

/// The Frobenius norm of a dense matrix, as an `f64` to compare tolerances in.
pub(super) fn norm<T: Num>(a: &Matrix<T>) -> f64 {
    a.norm(a::math::matrix::Norm::Frobenius).to_f64()
}

/// The invariant every constructor restores and every algorithm here relies on: the column indices
/// of a row ascend, no stored element is zero, and the three vectors agree with each other.
pub(super) fn assert_csr<T: Num>(a: &SparseMatrix<T>) {
    let pointers = a.row_pointers();
    assert_eq!(
        pointers.len(),
        a.rows() + 1,
        "the row pointers are the wrong length"
    );
    assert_eq!(
        pointers.first().copied().unwrap_or(0),
        0,
        "the first row pointer isn't 0"
    );
    assert_eq!(
        pointers.last().copied().unwrap_or(0),
        a.nnz(),
        "the last row pointer isn't nnz"
    );
    assert_eq!(
        a.col_indices().len(),
        a.nnz(),
        "the column indices don't match nnz"
    );
    assert_eq!(a.values().len(), a.nnz(), "the values don't match nnz");
    for (i, window) in pointers.windows(2).enumerate() {
        assert!(
            window[0] <= window[1],
            "the row pointers decrease at row {i}"
        );
        let mut previous: Option<usize> = None;
        for (j, x) in a.row(i) {
            assert!(
                j < a.cols(),
                "row {i} reaches column {j} of {} columns",
                a.cols()
            );
            assert!(!x.is_zero(), "row {i} stores a zero in column {j}");
            assert!(
                previous.is_none_or(|p| p < j),
                "the indices of row {i} don't ascend"
            );
            previous = Some(j);
        }
        assert_eq!(
            window[1] - window[0],
            a.row(i).count(),
            "row {i} holds what it promises"
        );
    }
}

/// The same invariant of the compressed sparse column form: the row indices of a column strictly
/// ascend, no stored element is zero, and the three vectors agree with each other.
pub(super) fn assert_csc<T: Num>(a: &CscMatrix<T>) {
    let pointers = a.col_pointers();
    assert_eq!(
        pointers.len(),
        a.cols() + 1,
        "the column pointers are the wrong length"
    );
    assert_eq!(pointers[0], 0, "the first column pointer isn't 0");
    assert_eq!(
        pointers[a.cols()],
        a.nnz(),
        "the last column pointer isn't nnz"
    );
    assert_eq!(
        a.row_indices().len(),
        a.nnz(),
        "the row indices don't match nnz"
    );
    assert_eq!(a.values().len(), a.nnz(), "the values don't match nnz");
    for (j, window) in pointers.windows(2).enumerate() {
        assert!(
            window[0] <= window[1],
            "the column pointers decrease at column {j}"
        );
        let rows = &a.row_indices()[window[0]..window[1]];
        assert!(
            rows.windows(2).all(|w| w[0] < w[1]),
            "the indices of column {j} don't ascend"
        );
        assert!(
            rows.iter().all(|&i| i < a.rows()),
            "column {j} reaches beyond the {} rows",
            a.rows()
        );
        assert!(
            a.values()[window[0]..window[1]]
                .iter()
                .all(|x| !x.is_zero()),
            "column {j} stores a zero"
        );
    }
}

/// A dense matrix of random elements.
pub(super) fn random_dense<T: Num>(r: &mut Rng, rows: usize, cols: usize) -> Matrix<T> {
    let data: Vec<T> = (0..rows * cols).map(|_| T::random(r)).collect();
    Matrix::from_column_major(rows, cols, data)
}

/// A vector of random elements.
pub(super) fn random_vector<T: Num>(r: &mut Rng, n: usize) -> Vec<T> {
    (0..n).map(|_| T::random(r)).collect()
}
