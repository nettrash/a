//! How a matrix in compressed sparse row form is built and read back: every constructor leaves the
//! form in its invariant, and every accessor returns what the dense matrix of the same elements
//! holds.

use super::common::{Num, Rng, assert_csr, each_type, random_sparse, size};
use a::math::complex::{Complex, Float};
use a::math::matrix::Matrix;
use a::math::sparse::{CooMatrix, CscMatrix, CsrMatrix, SparseMatrix};
use core::hash::Hash;
use core::marker::PhantomData;
use std::collections::BTreeMap;
use std::time::{Duration, Instant};

fn constructors_of<T: Num>() {
    // The empty matrix stores nothing and reads as zeros everywhere.
    let z = SparseMatrix::<T>::zeros(3, 5);
    assert_eq!((z.rows(), z.cols(), z.nnz()), (3, 5, 0));
    assert_eq!(z.shape(), (3, 5));
    assert!(z.is_empty() && !z.is_square());
    assert_eq!(z.density(), T::Real::ZERO);
    assert_eq!(z.to_matrix(), Matrix::zeros(3, 5));
    assert_csr(&z);

    // The identity has n stored elements and reads back as the dense one.
    let i = SparseMatrix::<T>::identity(4);
    assert_eq!(i.nnz(), 4);
    assert!(i.is_square() && !i.is_empty());
    assert_eq!(i.to_matrix(), Matrix::identity(4));
    assert_csr(&i);

    // A diagonal whose zeros aren't stored, so it is a smaller matrix than its length.
    let d = SparseMatrix::from_diagonal(&[T::real(1.0), T::zero(), T::real(4.0)]);
    assert_eq!((d.rows(), d.cols(), d.nnz()), (3, 3, 2));
    assert_eq!(d.diagonal(), vec![T::real(1.0), T::zero(), T::real(4.0)]);
    assert_eq!(
        d.to_matrix(),
        Matrix::from_diagonal(&[T::real(1.0), T::zero(), T::real(4.0)])
    );
    assert_csr(&d);

    // Three diagonals of length n − 1, n and n − 1 make the n × n band matrix.
    let a = SparseMatrix::tridiagonal(&[T::real(-1.0); 3], &[T::real(2.0); 4], &[T::real(-3.0); 3]);
    assert_eq!(a.nnz(), 10);
    assert_eq!(a[(1, 0)], T::real(-1.0));
    assert_eq!(a[(1, 1)], T::real(2.0));
    assert_eq!(a[(1, 2)], T::real(-3.0));
    assert_eq!(a[(0, 2)], T::zero());
    assert_csr(&a);
    let want = Matrix::from_fn(4, 4, |i, j| match j {
        k if k + 1 == i => T::real(-1.0),
        k if k == i => T::real(2.0),
        k if k == i + 1 => T::real(-3.0),
        _ => T::zero(),
    });
    assert_eq!(a.to_matrix(), want);

    // The zeros a rule returns aren't stored either.
    let off = SparseMatrix::from_fn(4, 4, |i, j| if i == j { T::zero() } else { T::real(1.0) });
    assert_eq!(off.nnz(), 12);
    assert_csr(&off);
    assert_eq!(
        off.to_matrix(),
        Matrix::from_fn(4, 4, |i, j| if i == j { T::zero() } else { T::real(1.0) })
    );

    // The matrices with no rows, no columns, or neither.
    for (rows, cols) in [(0, 0), (0, 4), (4, 0)] {
        let empty = SparseMatrix::<T>::zeros(rows, cols);
        assert_csr(&empty);
        assert_eq!(empty.row_pointers().len(), rows + 1);
        assert_eq!(empty.to_matrix().shape(), (rows, cols));
        assert_eq!(SparseMatrix::from_matrix(&empty.to_matrix()), empty);
        assert_eq!(SparseMatrix::from_coo(rows, cols, &[]), empty);
        assert_eq!(SparseMatrix::from_fn(rows, cols, |_, _| T::one()), empty);
        assert_eq!(empty.transpose().shape(), (cols, rows));
        assert_eq!(empty.density(), T::Real::ZERO);
        assert_eq!(empty.diagonal(), Vec::<T>::new());
    }
    assert_eq!(SparseMatrix::<T>::default(), SparseMatrix::<T>::zeros(0, 0));
    assert_eq!(
        SparseMatrix::<T>::identity(0),
        SparseMatrix::<T>::zeros(0, 0)
    );
    assert_eq!(
        SparseMatrix::<T>::tridiagonal(&[], &[T::real(5.0)], &[]).to_matrix(),
        Matrix::from_rows(&[[T::real(5.0)]])
    );
    // The alias names the same type.
    let alias: CsrMatrix<T> = SparseMatrix::identity(2);
    assert_eq!(alias, SparseMatrix::identity(2));
}

#[test]
fn the_constructors_build_what_the_dense_ones_build() {
    each_type!(constructors_of);
}

fn coo_sums_repeated_cells_of<T: Num>() {
    // Repeated cells are summed, whatever order the entries come in, and a zero sum is dropped.
    let a = SparseMatrix::from_coo(
        2,
        2,
        &[
            (1, 1, T::real(1.0)),
            (0, 1, T::real(2.0)),
            (1, 1, T::real(3.0)),
            (0, 1, T::real(-2.0)),
            (1, 0, T::real(5.0)),
            (0, 0, T::real(4.0)),
            (0, 0, T::real(-4.0)),
            (0, 1, T::real(2.0)),
        ],
    );
    assert_eq!(a.nnz(), 3);
    assert_eq!(a[(0, 0)], T::zero());
    assert_eq!(a[(0, 1)], T::real(2.0));
    assert_eq!(a[(1, 0)], T::real(5.0));
    assert_eq!(a[(1, 1)], T::real(4.0));
    assert_csr(&a);

    // The order of the entries of different cells changes nothing.
    let mut r = Rng(0x5EED_2201);
    let mut entries: Vec<(usize, usize, T)> = Vec::new();
    for _ in 0..60 {
        entries.push((r.below(7), r.below(5), T::real(r.below(9) as f64 - 4.0)));
    }
    let mut shuffled = entries.clone();
    for k in (1..shuffled.len()).rev() {
        shuffled.swap(k, r.below(k + 1));
    }
    let x = SparseMatrix::from_coo(7, 5, &entries);
    let y = SparseMatrix::from_coo(7, 5, &shuffled);
    assert_eq!(x, y);
    assert_csr(&x);
    // Against the sums made one cell at a time.
    let mut cells: BTreeMap<(usize, usize), T> = BTreeMap::new();
    for &(i, j, v) in &entries {
        *cells.entry((i, j)).or_insert(T::zero()) += v;
    }
    let want = Matrix::from_fn(7, 5, |i, j| {
        cells.get(&(i, j)).copied().unwrap_or(T::zero())
    });
    assert_eq!(x.to_matrix(), want);
}

#[test]
fn from_coo_sums_the_entries_of_a_cell_and_drops_zero_sums() {
    each_type!(coo_sums_repeated_cells_of);
}

#[test]
fn from_coo_sums_the_entries_of_a_cell_in_the_order_given() {
    // In floats the order of a sum shows: 1 + 1e-17 − 1 is 0, and 1 − 1 + 1e-17 is 1e-17.
    let lost = SparseMatrix::from_coo(1, 1, &[(0, 0, 1.0), (0, 0, 1e-17), (0, 0, -1.0)]);
    assert_eq!(lost.nnz(), 0);
    let kept = SparseMatrix::from_coo(1, 1, &[(0, 0, 1.0), (0, 0, -1.0), (0, 0, 1e-17)]);
    assert_eq!(kept.values(), [1e-17]);
    // The same within a row whose columns come out of order, which is where a sort could mix
    // the entries of one cell up.
    let mixed = SparseMatrix::from_coo(
        1,
        3,
        &[
            (0, 2, 1.0),
            (0, 0, 5.0),
            (0, 2, 1e-17),
            (0, 1, 7.0),
            (0, 2, -1.0),
            (0, 0, -5.0),
            (0, 0, 1e-300),
        ],
    );
    assert_eq!(mixed.col_indices(), [0, 1]);
    assert_eq!(mixed.values(), [1e-300, 7.0]);
}

#[test]
fn signed_zeros_and_cancelled_sums_are_not_stored() {
    let a = SparseMatrix::from_coo(2, 2, &[(0, 0, -0.0f64), (0, 1, 2.0), (1, 1, 0.0)]);
    assert_eq!(a.nnz(), 1);
    assert_eq!(a[(0, 0)].to_bits(), 0.0f64.to_bits());
    let c = SparseMatrix::from_coo(1, 1, &[(0, 0, Complex::new(0.0, -0.0))]);
    assert_eq!(c.nnz(), 0);
    assert_eq!(SparseMatrix::from_diagonal(&[-0.0, 1.0]).nnz(), 1);
    assert_eq!(
        SparseMatrix::from_matrix(&Matrix::from_rows(&[[-0.0, 3.0]])).nnz(),
        1
    );
    // A NaN is not a zero: it is stored, and read back.
    let n = SparseMatrix::from_coo(1, 2, &[(0, 1, f64::NAN)]);
    assert_eq!(n.nnz(), 1);
    assert!(n[(0, 1)].is_nan());
    assert!(SparseMatrix::from_matrix(&n.to_matrix())[(0, 1)].is_nan());
}

#[test]
fn a_long_row_in_any_order_is_assembled_in_linear_time() {
    // One row of n entries in descending order of column took an insertion sort n²/2 moves: at
    // this size some 10¹⁰, minutes in a test build, against the fraction of a second two counting
    // passes take. The bound is generous because the machine may be busy.
    let n = size(200_000, 300);
    let budget = Duration::from_secs(30);
    let begun = Instant::now();
    let descending: Vec<(usize, usize, f64)> =
        (0..n).rev().map(|j| (0, j, (j + 1) as f64)).collect();
    let row = SparseMatrix::from_coo(1, n, &descending);
    assert_eq!(row.nnz(), n);
    assert!(row.col_indices().windows(2).all(|w| w[0] < w[1]));
    assert!(
        row.values()
            .iter()
            .enumerate()
            .all(|(j, &x)| x == (j + 1) as f64)
    );

    // An arrow: a full first row, a full first column and the diagonal, the entries shuffled.
    let mut r = Rng(0x5EED_2207);
    let mut arrow: Vec<(usize, usize, f64)> = Vec::with_capacity(3 * n);
    for k in 1..n {
        arrow.push((0, k, 1.0));
        arrow.push((k, 0, 2.0));
        arrow.push((k, k, 3.0));
    }
    arrow.push((0, 0, 4.0));
    for k in (1..arrow.len()).rev() {
        arrow.swap(k, r.below(k + 1));
    }
    let a = SparseMatrix::from_coo(n, n, &arrow);
    assert_eq!(a.nnz(), 3 * n - 2);
    assert_eq!(a.row(0).count(), n);
    assert!(
        a.row_pointers()
            .windows(2)
            .skip(1)
            .all(|w| w[1] - w[0] == 2)
    );
    assert!(
        a.col_indices()[..n]
            .iter()
            .enumerate()
            .all(|(j, &c)| c == j)
    );
    assert!(a.is_symmetric_pattern() && !a.is_symmetric());

    // The compressed arrays themselves, given with that row backwards.
    let reversed: Vec<usize> = (0..n).rev().collect();
    let values: Vec<f64> = (0..n).rev().map(|j| (j + 1) as f64).collect();
    let b = SparseMatrix::from_csr(1, n, vec![0, n], reversed, values);
    assert_eq!(b, row);
    assert!(
        cfg!(miri) || begun.elapsed() < budget,
        "assembling took {:?}",
        begun.elapsed()
    );
}

#[test]
fn a_matrix_with_far_more_columns_than_elements_costs_what_its_elements_cost() {
    // Nothing here may allocate or walk a vector as long as the matrix is wide: 2⁶³ columns.
    let cols = usize::MAX / 2;
    let mut r = Rng(0x5EED_2208);
    let n = size(4_000, 60);
    let mut entries: Vec<(usize, usize, f64)> = Vec::new();
    let mut cells: BTreeMap<(usize, usize), f64> = BTreeMap::new();
    for k in 0..n {
        // Columns spread over all 63 bits, with repeats, so that every digit of the radix sort
        // has something to do.
        let j = if k % 7 == 3 {
            entries[k - 1].1
        } else {
            (r.next() as usize) % cols
        };
        let (i, x) = (r.below(3), r.below(9) as f64 - 4.0);
        entries.push((i, j, x));
        *cells.entry((i, j)).or_insert(0.0) += x;
    }
    let a = SparseMatrix::from_coo(3, cols, &entries);
    assert_csr(&a);
    let want: Vec<((usize, usize), f64)> = cells.into_iter().filter(|&(_, x)| x != 0.0).collect();
    let got: Vec<((usize, usize), f64)> = (0..3)
        .flat_map(|i| a.row(i).map(move |(j, x)| ((i, j), x)))
        .collect();
    assert_eq!(got, want);
    assert_eq!(a.density(), a.nnz() as f64 / (3.0 * cols as f64));
    assert!(a.density() > 0.0);
    // The same through the compressed arrays, whose rows arrive unsorted.
    let (rows, cols, pointers, indices, values) = a.clone().into_parts();
    let mut backwards_indices = indices.clone();
    let mut backwards_values = values.clone();
    for i in 0..rows {
        backwards_indices[pointers[i]..pointers[i + 1]].reverse();
        backwards_values[pointers[i]..pointers[i + 1]].reverse();
    }
    assert_eq!(
        SparseMatrix::from_csr(rows, cols, pointers, backwards_indices, backwards_values),
        a
    );
    // And the operations that touch the stored elements alone.
    assert_eq!(a.map(|_, _, x| 2.0 * x), &a + &a);
    assert_eq!((&a - &a.clone()).nnz(), 0);
    assert_eq!(a.to_coo().to_csr(), a);
    assert_eq!(a.select(&[2, 0], &[entries[0].1]).shape(), (2, 1));
    assert_eq!(a.submatrix(0..3, 0..cols), a);
    assert_eq!(
        a.norm(a::math::matrix::Norm::One),
        a.transpose_free_column_sums()
    );
}

/// The largest column sum computed the slow way, for the test above.
trait ColumnSums {
    fn transpose_free_column_sums(&self) -> f64;
}

impl ColumnSums for SparseMatrix<f64> {
    fn transpose_free_column_sums(&self) -> f64 {
        let mut sums: BTreeMap<usize, f64> = BTreeMap::new();
        for i in 0..self.rows() {
            for (j, x) in self.row(i) {
                *sums.entry(j).or_insert(0.0) += x.abs();
            }
        }
        sums.values().fold(0.0, |m, &s| m.max(s))
    }
}

fn from_csr_restores_the_canonical_form_of<T: Num>() {
    // Rows out of order, a repeated column and a zero: sorted, summed and dropped.
    let a = SparseMatrix::from_csr(
        3,
        4,
        vec![0, 4, 4, 7],
        vec![3, 0, 3, 1, 2, 0, 2],
        vec![
            T::real(1.0),
            T::real(2.0),
            T::real(3.0),
            T::zero(),
            T::real(5.0),
            T::real(6.0),
            T::real(-5.0),
        ],
    );
    assert_csr(&a);
    assert_eq!(a.row_pointers(), [0, 2, 2, 3]);
    assert_eq!(a.col_indices(), [0, 3, 0]);
    assert_eq!(a.values(), [T::real(2.0), T::real(4.0), T::real(6.0)]);
    // Arrays that are canonical come back as they went in.
    let mut r = Rng(0x5EED_2209);
    for (rows, cols) in [(1, 1), (5, 9), (9, 5), (12, 12)] {
        let b: SparseMatrix<T> = random_sparse(&mut r, rows, cols, rows * cols / 3 + 1);
        let (m, n, pointers, indices, values) = b.clone().into_parts();
        assert_eq!((m, n), (rows, cols));
        assert_eq!(pointers, b.row_pointers());
        assert_eq!(indices, b.col_indices());
        assert_eq!(values, b.values());
        let again = SparseMatrix::from_csr(m, n, pointers, indices, values);
        assert_eq!(again, b);
        assert_csr(&again);
        // Shuffled within its rows, the same matrix.
        let (_, _, pointers, mut indices, mut values) = b.clone().into_parts();
        for i in 0..rows {
            for k in (pointers[i] + 1..pointers[i + 1]).rev() {
                let other = pointers[i] + r.below(k - pointers[i] + 1);
                indices.swap(k, other);
                values.swap(k, other);
            }
        }
        assert_eq!(SparseMatrix::from_csr(m, n, pointers, indices, values), b);
    }
    // No rows, and no elements.
    let none = SparseMatrix::<T>::from_csr(0, 3, vec![0], vec![], vec![]);
    assert_eq!(none, SparseMatrix::zeros(0, 3));
}

#[test]
fn from_csr_sorts_sums_and_drops_and_into_parts_is_its_inverse() {
    each_type!(from_csr_restores_the_canonical_form_of);
}

fn accessors_read_the_dense_matrix_of<T: Num>() {
    let mut r = Rng(0x5EED_2202);
    let a: SparseMatrix<T> = random_sparse(&mut r, 8, 6, 30);
    let d = a.to_matrix();
    assert_csr(&a);
    assert!(a.nnz() <= 48);
    assert!(a.density() <= T::Real::ONE);
    assert_eq!(a.density(), T::Real::from_f64(a.nnz() as f64 / 48.0));
    for i in 0..8 {
        for j in 0..6 {
            assert_eq!(a[(i, j)], d[(i, j)], "the cell ({i}, {j}) differs");
            match a.find(i, j) {
                Some(at) => {
                    assert_eq!(a.values()[at], d[(i, j)]);
                    assert_eq!(a.col_indices()[at], j);
                    assert!(!d[(i, j)].is_zero(), "a stored cell holds the zero");
                }
                None => assert!(d[(i, j)].is_zero(), "a cell of the matrix isn't stored"),
            }
        }
        let row: Vec<(usize, T)> = (0..6)
            .map(|j| (j, d[(i, j)]))
            .filter(|&(_, x)| !x.is_zero())
            .collect();
        assert_eq!(a.row(i).collect::<Vec<_>>(), row, "row {i} differs");
    }
    for j in 0..6 {
        let column: Vec<(usize, T)> = (0..8)
            .map(|i| (i, d[(i, j)]))
            .filter(|&(_, x)| !x.is_zero())
            .collect();
        assert_eq!(a.column(j), column, "column {j} differs");
    }
    let diagonal: Vec<T> = (0..6).map(|i| d[(i, i)]).collect();
    assert_eq!(a.diagonal(), diagonal);
    assert_eq!(a.find(8, 0), None);
    assert_eq!(a.find(0, 6), None);
}

#[test]
fn the_accessors_read_what_the_dense_matrix_holds() {
    each_type!(accessors_read_the_dense_matrix_of);
}

fn transpose_and_map_follow_the_dense_matrix_of<T: Num>() {
    let mut r = Rng(0x5EED_2203);
    let a: SparseMatrix<T> = random_sparse(&mut r, 7, 4, 20);
    let d = a.to_matrix();
    assert_eq!(a.transpose().to_matrix(), d.transpose());
    assert_csr(&a.transpose());
    assert_eq!(a.conjugate_transpose().to_matrix(), d.conjugate_transpose());
    assert_csr(&a.conjugate_transpose());
    // Transposing twice is the matrix itself, down to what it stores.
    assert_eq!(a.transpose().transpose(), a);
    assert_eq!(a.conjugate_transpose().conjugate_transpose(), a);
    // A map that zeroes some cells stores fewer of them.
    let mapped = a.map(|i, j, x| if (i + j) % 3 == 0 { T::zero() } else { x });
    assert_csr(&mapped);
    let want = Matrix::from_fn(7, 4, |i, j| {
        if (i + j) % 3 == 0 {
            T::zero()
        } else {
            d[(i, j)]
        }
    });
    assert_eq!(mapped.to_matrix(), want);
    // The identity map changes nothing.
    assert_eq!(a.map(|_, _, x| x), a);
}

#[test]
fn the_transposes_and_map_follow_the_dense_matrix() {
    each_type!(transpose_and_map_follow_the_dense_matrix_of);
}

fn equality_follows_the_dense_matrix_of<T: Num>() {
    let mut r = Rng(0x5EED_2204);
    let a: SparseMatrix<T> = random_sparse(&mut r, 6, 6, 18);
    assert_eq!(a, a.clone());
    assert_eq!(a.to_matrix(), a.transpose().transpose().to_matrix());
    // The same elements assembled another way compare equal.
    let entries: Vec<(usize, usize, T)> = (0..a.rows())
        .flat_map(|i| a.row(i).map(move |(j, x)| (i, j, x)))
        .collect();
    let b = SparseMatrix::from_coo(a.rows(), a.cols(), &entries);
    assert_eq!(a, b);
    // A matrix of another shape is never equal, even when it stores the same elements.
    assert_ne!(a, SparseMatrix::from_coo(a.rows(), a.cols() + 1, &entries));
    assert_ne!(a, SparseMatrix::from_coo(a.rows() + 1, a.cols(), &entries));
    assert_ne!(
        SparseMatrix::<T>::zeros(2, 3),
        SparseMatrix::<T>::zeros(3, 2)
    );
    // One element changed, or dropped, and the matrices differ.
    let (i, j, x) = entries[entries.len() / 2];
    assert_ne!(a, a.map(|p, q, y| if (p, q) == (i, j) { y + y } else { y }));
    assert_ne!(
        a,
        a.map(|p, q, y| if (p, q) == (i, j) { T::zero() } else { y })
    );
    assert!(!x.is_zero());
}

#[test]
fn equality_is_that_of_the_dense_matrices() {
    each_type!(equality_follows_the_dense_matrix_of);
}

#[test]
fn a_matrix_holding_a_nan_is_not_equal_to_itself_as_the_dense_one_is_not() {
    // `==` is the elements' own, which is why the type is `PartialEq` and no more: neither
    // `Matrix<f64>` nor a sparse matrix of floats is `Eq` or `Hash`.
    let a = SparseMatrix::from_coo(2, 2, &[(0, 0, f64::NAN), (1, 1, 1.0)]);
    let same = a.clone();
    assert!(a != same);
    assert!(a.to_matrix() != same.to_matrix());
    let c = SparseMatrix::from_coo(1, 1, &[(0, 0, Complex::new(1.0f32, f32::NAN))]);
    assert!(c != c.clone());
}

fn symmetry_of<T: Num>() {
    let mut r = Rng(0x5EED_2205);
    let square = SparseMatrix::from_coo(
        3,
        3,
        &[
            (0, 2, T::real(1.0)),
            (2, 0, T::real(1.0)),
            (1, 1, T::real(2.0)),
        ],
    );
    assert!(square.is_symmetric());
    assert!(square.is_hermitian());
    assert!(square.is_symmetric_pattern());
    assert_csr(&square);
    // One element out of place, or one missing mirror, and it stops being either.
    let lopsided = square.map(|i, j, x| {
        if (i, j) == (0, 2) {
            x + T::real(1.0)
        } else {
            x
        }
    });
    assert!(!lopsided.is_symmetric() && !lopsided.is_hermitian());
    assert!(lopsided.is_symmetric_pattern());
    let missing = square.map(|i, j, x| if (i, j) == (0, 2) { T::zero() } else { x });
    assert!(!missing.is_symmetric() && !missing.is_hermitian());
    assert!(!missing.is_symmetric_pattern());
    // A rectangular matrix has no diagonal to be symmetric about.
    let wide = random_sparse::<T>(&mut r, 3, 5, 10);
    assert!(!wide.is_symmetric() && !wide.is_hermitian() && !wide.is_symmetric_pattern());
    // The empty square matrices are symmetric, the empty rectangular ones are not.
    assert!(SparseMatrix::<T>::zeros(0, 0).is_symmetric());
    assert!(SparseMatrix::<T>::zeros(4, 4).is_hermitian());
    assert!(!SparseMatrix::<T>::zeros(0, 4).is_symmetric_pattern());

    // Against the dense matrices, on random matrices that are symmetric, Hermitian, nearly
    // either, or neither.
    for case in 0..size(120, 12) {
        let n = 1 + r.below(9);
        let count = 1 + r.below(2 * n);
        let base: SparseMatrix<T> = random_sparse(&mut r, n, n, count);
        let a = match case % 4 {
            0 => &base + &base.transpose(),
            1 => &base + &base.conjugate_transpose(),
            2 => {
                // Symmetric but for one element changed or removed.
                let s = &base + &base.transpose();
                let (i, j) = (r.below(n), r.below(n));
                let drop = r.below(2) == 0;
                s.map(|p, q, x| {
                    if (p, q) != (i, j) {
                        x
                    } else if drop {
                        T::zero()
                    } else {
                        x + T::real(1.0)
                    }
                })
            }
            _ => base,
        };
        let d = a.to_matrix();
        assert_eq!(a.is_symmetric(), d == d.transpose(), "case {case}");
        assert_eq!(
            a.is_hermitian(),
            d == d.conjugate_transpose(),
            "case {case}"
        );
        let stored = |i: usize, j: usize| a.find(i, j).is_some();
        let pattern = (0..n).all(|i| (0..n).all(|j| stored(i, j) == stored(j, i)));
        assert_eq!(a.is_symmetric_pattern(), pattern, "case {case}");
    }
}

#[test]
fn the_symmetries_are_those_of_the_dense_matrix() {
    each_type!(symmetry_of);
}

#[test]
fn a_hermitian_matrix_needs_conjugates_and_a_real_diagonal() {
    // Real parts symmetric, imaginary parts antisymmetric: Hermitian, and not symmetric.
    let h = SparseMatrix::from_coo(
        2,
        2,
        &[
            (0, 1, Complex::new(1.0, -2.0)),
            (1, 0, Complex::new(1.0, 2.0)),
            (0, 0, Complex::new(3.0, 0.0)),
            (1, 1, Complex::new(4.0, 0.0)),
        ],
    );
    assert!(h.is_hermitian());
    assert!(!h.is_symmetric());
    assert_eq!(h.to_matrix(), h.to_matrix().conjugate_transpose());
    // Swapping the imaginary parts without conjugating leaves both.
    let s = SparseMatrix::from_coo(
        2,
        2,
        &[
            (0, 1, Complex::new(1.0, -2.0)),
            (1, 0, Complex::new(1.0, -2.0)),
            (0, 0, Complex::new(3.0, 0.0)),
            (1, 1, Complex::new(4.0, 0.0)),
        ],
    );
    assert!(s.is_symmetric());
    assert!(!s.is_hermitian());
    // An imaginary part on the diagonal is symmetric and not Hermitian.
    let d = SparseMatrix::from_diagonal(&[Complex::new(1.0f32, 1.0)]);
    assert!(d.is_symmetric() && !d.is_hermitian());
}

fn converts_between_the_two_forms_of<T: Num>() {
    let mut r = Rng(0x5EED_2206);
    for (rows, cols) in [(1, 1), (2, 7), (9, 3), (6, 6)] {
        let a: SparseMatrix<T> = random_sparse(&mut r, rows, cols, rows * cols / 2);
        let d = a.to_matrix();
        assert_eq!(d.shape(), (rows, cols));
        assert_eq!(
            SparseMatrix::from_matrix(&d),
            a,
            "a round trip changed the matrix"
        );
        assert_csr(&SparseMatrix::from_matrix(&d));
        // A dense matrix's zeros are left out on the way in.
        let stored = (0..rows)
            .flat_map(|i| (0..cols).map(move |j| (i, j)))
            .filter(|&(i, j)| !d[(i, j)].is_zero())
            .count();
        assert_eq!(a.nnz(), stored, "a zero was stored, or a nonzero left out");
        // A full matrix goes through unchanged too.
        let full = Matrix::from_fn(rows, cols, |_, _| T::random(&mut r));
        assert_eq!(SparseMatrix::from_matrix(&full).to_matrix(), full);
    }
}

#[test]
fn a_matrix_goes_to_the_dense_form_and_back_unchanged() {
    each_type!(converts_between_the_two_forms_of);
}

#[test]
fn the_dense_form_of_a_wide_matrix_takes_one_step_for_each_cell() {
    // 3 000 000 cells, of which a handful are stored. A binary search for every cell was within
    // the budget as well, so this checks the result more than the time; the count of steps is in
    // the code.
    let (rows, cols) = (size(1500, 20), size(2000, 30));
    let a = SparseMatrix::from_coo(
        rows,
        cols,
        &[
            (0, cols - 1, 1.5f32),
            (rows - 1, 0, -2.0),
            (rows / 2, 7, 3.0),
        ],
    );
    let d = a.to_matrix();
    assert_eq!(d.as_slice().iter().filter(|&&x| x != 0.0).count(), 3);
    assert_eq!(
        (d[(0, cols - 1)], d[(rows - 1, 0)], d[(rows / 2, 7)]),
        (1.5, -2.0, 3.0)
    );
    assert_eq!(SparseMatrix::from_matrix(&d), a);
}

#[test]
fn the_formats_print_the_shape_and_the_stored_elements() {
    let a = SparseMatrix::from_coo(2, 3, &[(1, 0, -2.0), (0, 2, 1.5)]);
    assert_eq!(format!("{a:?}"), "2 × 3 {(0, 2): 1.5, (1, 0): -2.0}");
    assert_eq!(format!("{a}"), "(0, 2): 1.5\n(1, 0): -2\n");
    assert_eq!(
        format!("{:?}", SparseMatrix::<f32>::zeros(0, 4)),
        "0 × 4 {}"
    );
    assert_eq!(format!("{}", SparseMatrix::<f32>::zeros(2, 2)), "");
    let c = SparseMatrix::from_coo(1, 1, &[(0, 0, Complex::new(1.0, -2.0))]);
    assert_eq!(format!("{c}"), "(0, 0): 1-2i\n");
}

#[test]
fn a_very_wide_matrix_has_a_density_and_no_overflow() {
    let a = SparseMatrix::<f64>::zeros(3, usize::MAX);
    assert_eq!((a.nnz(), a.density()), (0, 0.0));
    let b = SparseMatrix::from_coo(3, usize::MAX, &[(2, usize::MAX - 1, 1.0)]);
    assert!(b.density() > 0.0 && b.density() < 1e-19);
    assert_eq!(b[(2, usize::MAX - 1)], 1.0);
    assert_eq!(b.find(2, usize::MAX - 1), Some(0));
}

/// Whether a type is `Eq`, by the stable probe of an inherent constant, which exists only where its
/// bound holds and otherwise leaves the trait's: a type that is not `Eq` reads `false`.
struct IsEq<T>(PhantomData<T>);

trait NotEq {
    const YES: bool = false;
}

impl<T> NotEq for IsEq<T> {}

impl<T: Eq> IsEq<T> {
    const YES: bool = true;
}

/// Whether a type is `Hash`, by the same probe.
struct IsHash<T>(PhantomData<T>);

trait NotHash {
    const YES: bool = false;
}

impl<T> NotHash for IsHash<T> {}

impl<T: Hash> IsHash<T> {
    const YES: bool = true;
}

#[test]
fn no_sparse_matrix_of_floats_claims_an_equality_that_a_nan_breaks() {
    // The audit's p_storage-5: `Eq` was implemented for every element type, none of them `Eq`,
    // and `Hash` under a bound no element type meets. Like the dense matrix, the sparse forms are
    // `PartialEq` and no more.
    let floats = [
        <IsEq<SparseMatrix<f32>>>::YES,
        <IsEq<SparseMatrix<f64>>>::YES,
        <IsEq<SparseMatrix<Complex<f32>>>>::YES,
        <IsEq<SparseMatrix<Complex<f64>>>>::YES,
        <IsEq<CscMatrix<f64>>>::YES,
        <IsEq<CooMatrix<Complex<f32>>>>::YES,
        <IsEq<Matrix<f64>>>::YES,
        <IsHash<SparseMatrix<f64>>>::YES,
        <IsHash<CscMatrix<f32>>>::YES,
    ];
    assert_eq!(floats, [false; 9]);
    // The probes see the types that are.
    assert_eq!(
        [<IsEq<Vec<usize>>>::YES, <IsHash<Vec<usize>>>::YES],
        [true; 2]
    );
}

#[test]
#[should_panic(expected = "more rows or columns than a vector of offsets can hold")]
fn rejects_more_rows_than_a_vector_of_offsets_can_hold() {
    let _ = SparseMatrix::<f64>::zeros(usize::MAX, 3);
}

#[test]
#[should_panic(expected = "more rows or columns than a vector of offsets can hold")]
fn rejects_triplets_of_more_rows_than_a_vector_of_offsets_can_hold() {
    let _ = SparseMatrix::<f64>::from_coo(usize::MAX, 3, &[]);
}

#[test]
#[should_panic(expected = "more rows or columns than a vector of offsets can hold")]
fn from_fn_rejects_more_rows_than_a_vector_of_offsets_can_hold_before_it_begins() {
    // A release build used to push offsets until memory ran out.
    let _ = SparseMatrix::<f64>::from_fn(usize::MAX, 1, |_, _| panic!("began to build"));
}

#[test]
#[should_panic(expected = "more rows or columns than a vector of offsets can hold")]
fn from_matrix_rejects_a_dense_matrix_of_more_rows_than_a_vector_of_offsets_can_hold() {
    let _ = SparseMatrix::from_matrix(&Matrix::<f64>::zeros(usize::MAX, 0));
}

#[test]
fn the_density_is_right_where_the_cells_outnumber_what_a_usize_counts() {
    // The cases the verification of p_storage-7 lists, written with usize::MAX and usize::BITS so
    // that they hold on a 32-bit target too, where 65 536 × 65 536 cells already overflow.
    let a = SparseMatrix::from_coo(3, usize::MAX, &[(0, 7, 1.0f64)]);
    let want = 1.0 / (3.0 * usize::MAX as f64);
    assert!(
        (a.density() - want).abs() <= 1e-15 * want,
        "{}",
        a.density()
    );
    let b = SparseMatrix::from_coo(2, 1 << (usize::BITS - 1), &[(0, 7, 1.0f64)]);
    assert!(b.density() > 0.0);
    let c = SparseMatrix::from_coo(
        3,
        usize::MAX / 3 + 1,
        &[(0, 0, 1.0f64), (1, 1, 1.0), (2, 2, 1.0), (0, 9, 1.0)],
    );
    assert!(c.density() > 0.0 && c.density() <= 1.0, "{}", c.density());
    let d = SparseMatrix::<f64>::identity(65_537);
    assert!((d.density() * 65_537.0 - 1.0).abs() < 1e-12);
}

#[test]
#[should_panic(expected = "more rows or columns than a vector of offsets can hold")]
fn rejects_the_transpose_of_more_columns_than_a_vector_of_offsets_can_hold() {
    let _ = SparseMatrix::<f64>::zeros(3, usize::MAX).transpose();
}

#[test]
#[should_panic(expected = "outside the matrix")]
fn rejects_entries_outside_the_matrix() {
    let _ = SparseMatrix::from_coo(2, 2, &[(0, 2, 1.0)]);
}

#[test]
#[should_panic(expected = "outside the matrix")]
fn rejects_entries_in_a_row_outside_the_matrix() {
    let _ = SparseMatrix::from_coo(2, 2, &[(2, 0, 1.0)]);
}

#[test]
#[should_panic(expected = "the subdiagonal is the wrong length")]
fn rejects_a_subdiagonal_of_the_wrong_length() {
    let _ = SparseMatrix::tridiagonal(&[-1.0; 4], &[2.0; 4], &[-1.0; 3]);
}

#[test]
#[should_panic(expected = "the superdiagonal is the wrong length")]
fn rejects_a_superdiagonal_of_the_wrong_length() {
    let _ = SparseMatrix::tridiagonal(&[-1.0; 3], &[2.0; 4], &[-1.0; 4]);
}

#[test]
#[should_panic(expected = "the element is outside the matrix")]
fn rejects_an_index_outside_the_matrix() {
    let a = SparseMatrix::<f64>::zeros(2, 3);
    let _ = a[(2, 0)];
}

#[test]
#[should_panic(expected = "the row is outside the matrix")]
fn rejects_a_row_outside_the_matrix() {
    let a = SparseMatrix::<f64>::zeros(2, 3);
    let _ = a.row(2).count();
}

#[test]
#[should_panic(expected = "the column is outside the matrix")]
fn rejects_a_column_outside_the_matrix() {
    let a = SparseMatrix::<f64>::zeros(2, 3);
    let _ = a.column(3);
}

#[test]
#[should_panic(expected = "there isn't an offset for each row and the end")]
fn from_csr_rejects_offsets_of_the_wrong_length() {
    let _ = SparseMatrix::from_csr(2, 2, vec![0, 1], vec![0], vec![1.0]);
}

#[test]
#[should_panic(expected = "the columns and the values differ in length")]
fn from_csr_rejects_columns_and_values_of_different_lengths() {
    let _ = SparseMatrix::from_csr(1, 2, vec![0, 2], vec![0, 1], vec![1.0]);
}

#[test]
#[should_panic(expected = "the first offset isn't 0")]
fn from_csr_rejects_a_first_offset_that_is_not_zero() {
    let _ = SparseMatrix::from_csr(1, 2, vec![1, 1], vec![0], vec![1.0]);
}

#[test]
#[should_panic(expected = "the last offset isn't the number of elements")]
fn from_csr_rejects_a_last_offset_that_is_not_the_number_of_elements() {
    let _ = SparseMatrix::from_csr(1, 2, vec![0, 2], vec![0], vec![1.0]);
}

#[test]
#[should_panic(expected = "the offsets decrease")]
fn from_csr_rejects_offsets_that_decrease() {
    let _ = SparseMatrix::from_csr(3, 2, vec![0, 2, 1, 2], vec![0, 1], vec![1.0, 2.0]);
}

#[test]
#[should_panic(expected = "a column lies outside the matrix")]
fn from_csr_rejects_a_column_outside_the_matrix() {
    let _ = SparseMatrix::from_csr(1, 2, vec![0, 1], vec![2], vec![1.0]);
}
