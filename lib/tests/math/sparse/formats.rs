//! The three forms of a sparse matrix hold the same matrix: the triplets of a [`CooMatrix`], the
//! compressed rows of a [`SparseMatrix`] and the compressed columns of a [`CscMatrix`]. Every
//! conversion between them and the dense form keeps every element, costs what the elements and the
//! dimensions cost, and leaves a compressed form canonical.

use super::common::{
    Num, Rng, assert_close, assert_csc, assert_csr, each_type, random_dense, random_sparse,
    random_vector, size,
};
use a::math::complex::Complex;
use a::math::matrix::Matrix;
use a::math::sparse::{CooMatrix, CscMatrix, CsrMatrix, FnOperator, LinearOperator, SparseMatrix};
use std::collections::BTreeMap;
use std::time::{Duration, Instant};

/// Random triplets inside `rows × cols`: repeated cells, zeros, and pairs that cancel exactly.
fn random_triplets<T: Num>(
    r: &mut Rng,
    rows: usize,
    cols: usize,
    count: usize,
) -> Vec<(usize, usize, T)> {
    let mut list: Vec<(usize, usize, T)> = Vec::with_capacity(count);
    if rows == 0 || cols == 0 {
        return list;
    }
    for k in 0..count {
        if k % 5 == 1 {
            // The triplet before, negated: the two cancel, unless a third falls in the cell.
            let (i, j, x) = list[k - 1];
            list.push((i, j, -x));
        } else if k % 7 == 3 {
            list.push((r.below(rows), r.below(cols), T::zero()));
        } else {
            list.push((r.below(rows), r.below(cols), T::random(r)));
        }
    }
    list
}

/// The dense matrix the triplets stand for: each cell the sum of its triplets, in the order they
/// are given in, which is the sum the conversions form.
fn dense_of<T: Num>(rows: usize, cols: usize, triplets: &[(usize, usize, T)]) -> Matrix<T> {
    let mut cells: BTreeMap<(usize, usize), T> = BTreeMap::new();
    for &(i, j, x) in triplets {
        cells.entry((i, j)).and_modify(|s| *s += x).or_insert(x);
    }
    Matrix::from_fn(rows, cols, |i, j| {
        cells.get(&(i, j)).copied().unwrap_or(T::zero())
    })
}

/// The matrix of triplets, pushed one at a time.
fn pushed<T: Num>(rows: usize, cols: usize, triplets: &[(usize, usize, T)]) -> CooMatrix<T> {
    let mut a = CooMatrix::new(rows, cols);
    for &(i, j, x) in triplets {
        a.push(i, j, x);
    }
    a
}

fn triplets_of<T: Num>() {
    let mut r = Rng(0x5EED_2230);
    for (rows, cols) in [(1, 1), (3, 7), (7, 3), (6, 6), (0, 4), (4, 0), (0, 0)] {
        let list: Vec<(usize, usize, T)> = random_triplets(&mut r, rows, cols, 3 * (rows + cols));
        let a = pushed(rows, cols, &list);
        let want = dense_of(rows, cols, &list);
        // The triplets are kept as they were given, in their order, zeros and repeats too.
        assert_eq!(a.shape(), (rows, cols));
        assert_eq!((a.rows(), a.cols()), (rows, cols));
        assert_eq!(a.nnz(), list.len());
        assert_eq!(a.is_empty(), list.is_empty());
        assert_eq!(a.iter().len(), list.len());
        assert_eq!(a.iter().collect::<Vec<_>>(), list);
        let rows_of: Vec<usize> = list.iter().map(|t| t.0).collect();
        let cols_of: Vec<usize> = list.iter().map(|t| t.1).collect();
        let values_of: Vec<T> = list.iter().map(|t| t.2).collect();
        assert_eq!(a.row_indices(), rows_of);
        assert_eq!(a.col_indices(), cols_of);
        assert_eq!(a.values(), values_of);
        // The same triplets handed over as three vectors, and taken back.
        let b = CooMatrix::from_triplets(rows, cols, rows_of, cols_of, values_of);
        assert_eq!(a, b);
        let (m, n, row_indices, col_indices, values) = b.into_parts();
        assert_eq!(
            CooMatrix::from_triplets(m, n, row_indices, col_indices, values),
            a
        );
        // Room made beforehand changes nothing.
        let mut c = CooMatrix::with_capacity(rows, cols, list.len());
        for &(i, j, x) in &list {
            c.push(i, j, x);
        }
        assert_eq!(c, a);

        // Every conversion sums the cells, in the order the triplets are in, and drops zeros.
        assert_eq!(a.to_matrix(), want);
        let csr = a.to_csr();
        assert_csr(&csr);
        assert_eq!(csr.to_matrix(), want);
        assert_eq!(csr, SparseMatrix::from_coo(rows, cols, &list));
        assert_eq!(csr, SparseMatrix::from_matrix(&want));
        let csc = a.to_csc();
        assert_csc(&csc);
        assert_eq!(csc.to_matrix(), want);
        assert_eq!(csc, CscMatrix::from_matrix(&want));
        assert_eq!(csc, csr.to_csc());

        // The transpose swaps the rows and the columns of every triplet, at no cost.
        let t = a.transpose();
        assert_eq!(t.shape(), (cols, rows));
        assert_eq!(t.row_indices(), a.col_indices());
        assert_eq!(t.col_indices(), a.row_indices());
        assert_eq!(t, a.clone().into_transpose());
        assert_eq!(t.to_csr(), csr.transpose());
        assert_eq!(t.transpose(), a);

        // The triplets of a dense matrix are its nonzero elements, by rows.
        let from_dense = CooMatrix::from_matrix(&want);
        assert_eq!(from_dense.shape(), (rows, cols));
        assert_eq!(from_dense.to_matrix(), want);
        assert_eq!(from_dense.nnz(), csr.nnz());
        assert_eq!(from_dense, csr.to_coo());
    }
    assert_eq!(CooMatrix::<T>::default(), CooMatrix::new(0, 0));
    assert_eq!(CooMatrix::<T>::default().to_csr(), SparseMatrix::default());
}

#[test]
fn a_matrix_of_triplets_stands_for_the_sums_of_its_cells() {
    each_type!(triplets_of);
}

#[test]
fn triplets_of_one_cell_are_summed_in_the_order_they_are_in() {
    // 1 + 1e-17 − 1 is 0, and 1 − 1 + 1e-17 is 1e-17: the order of the triplets shows, and every
    // conversion keeps it.
    let lost = CooMatrix::from_triplets(1, 2, vec![0; 3], vec![1; 3], vec![1.0, 1e-17, -1.0]);
    let kept = CooMatrix::from_triplets(1, 2, vec![0; 3], vec![1; 3], vec![1.0, -1.0, 1e-17]);
    assert_eq!(lost.to_csr().nnz(), 0);
    assert_eq!(lost.to_csc().nnz(), 0);
    assert_eq!(lost.to_matrix().as_slice(), &[0.0, 0.0]);
    assert_eq!(kept.to_csr().values(), [1e-17]);
    assert_eq!(kept.to_csc().values(), [1e-17]);
    assert_eq!(kept.to_matrix().as_slice(), &[0.0, 1e-17]);
    // Equal triplets in another order are another list, and the same matrix where nothing rounds.
    let a = CooMatrix::from_triplets(2, 2, vec![0, 1], vec![1, 0], vec![2.0, 3.0]);
    let b = CooMatrix::from_triplets(2, 2, vec![1, 0], vec![0, 1], vec![3.0, 2.0]);
    assert_ne!(a, b);
    assert_eq!(a.to_csr(), b.to_csr());
}

fn columns_of<T: Num>() {
    let mut r = Rng(0x5EED_2231);
    for (rows, cols) in [(1, 1), (8, 6), (6, 8), (7, 7), (1, 9), (9, 1)] {
        let a: SparseMatrix<T> = random_sparse(&mut r, rows, cols, rows * cols / 2 + 1);
        let d = a.to_matrix();
        let c = a.to_csc();
        assert_csc(&c);
        assert_eq!(c.shape(), (rows, cols));
        assert_eq!((c.rows(), c.cols(), c.nnz()), (rows, cols, a.nnz()));
        assert_eq!(c.is_empty(), a.is_empty());
        assert_eq!(c.to_matrix(), d);
        assert_eq!(c.to_csr(), a);
        // The arrays are the compressed rows of the transpose.
        let t = a.transpose();
        assert_eq!(c.col_pointers(), t.row_pointers());
        assert_eq!(c.row_indices(), t.col_indices());
        assert_eq!(c.values(), t.values());
        for j in 0..cols {
            let column: Vec<(usize, T)> = (0..rows)
                .map(|i| (i, d[(i, j)]))
                .filter(|&(_, x)| !x.is_zero())
                .collect();
            assert_eq!(c.column(j).collect::<Vec<_>>(), column, "column {j}");
            assert_eq!(a.column(j), column, "column {j} of the rows");
        }
        for i in 0..rows {
            for j in 0..cols {
                assert_eq!(c[(i, j)], d[(i, j)], "the cell ({i}, {j}) differs");
                match c.find(i, j) {
                    Some(at) => {
                        assert_eq!(c.row_indices()[at], i);
                        assert_eq!(c.values()[at], d[(i, j)]);
                    }
                    None => assert!(d[(i, j)].is_zero()),
                }
            }
        }
        assert_eq!(c.find(rows, 0), None);
        assert_eq!(c.find(0, cols), None);
        // The dense matrix by columns, and the transposes.
        assert_eq!(CscMatrix::from_matrix(&d), c);
        assert_eq!(c.transpose().to_matrix(), d.transpose());
        assert_eq!(c.conjugate_transpose().to_matrix(), d.conjugate_transpose());
        assert_csc(&c.transpose());
        assert_csc(&c.conjugate_transpose());
        assert_eq!(c.transpose().transpose(), c);
        assert_eq!(c.conjugate_transpose().conjugate_transpose(), c);
        assert_eq!(c.transpose(), a.transpose().to_csc());
        assert_eq!(c.clone(), c);
    }
    // The matrices with no rows, no columns, or neither.
    for (rows, cols) in [(0, 0), (0, 4), (4, 0)] {
        let z = CscMatrix::<T>::zeros(rows, cols);
        assert_csc(&z);
        assert_eq!(z.col_pointers().len(), cols + 1);
        assert_eq!(z.to_matrix().shape(), (rows, cols));
        assert_eq!(CscMatrix::from_matrix(&z.to_matrix()), z);
        assert_eq!(z.to_csr(), SparseMatrix::zeros(rows, cols));
        assert_eq!(z.to_coo(), CooMatrix::new(rows, cols));
        assert_eq!(z.transpose(), CscMatrix::zeros(cols, rows));
    }
    assert_eq!(CscMatrix::<T>::default(), CscMatrix::zeros(0, 0));
}

#[test]
fn the_compressed_columns_read_what_the_dense_matrix_holds() {
    each_type!(columns_of);
}

fn from_csc_of<T: Num>() {
    // Rows out of order, a repeated row and a zero: sorted, summed and dropped.
    let a = CscMatrix::from_csc(
        4,
        3,
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
    assert_csc(&a);
    assert_eq!(a.col_pointers(), [0, 2, 2, 3]);
    assert_eq!(a.row_indices(), [0, 3, 0]);
    assert_eq!(a.values(), [T::real(2.0), T::real(4.0), T::real(6.0)]);
    // Canonical arrays come back as they went in, and shuffled ones give the same matrix.
    let mut r = Rng(0x5EED_2232);
    for (rows, cols) in [(1, 1), (5, 9), (9, 5), (12, 12)] {
        let b: CscMatrix<T> = random_sparse(&mut r, rows, cols, rows * cols / 3 + 1).to_csc();
        let (m, n, pointers, indices, values) = b.clone().into_parts();
        assert_eq!((m, n), (rows, cols));
        assert_eq!(pointers, b.col_pointers());
        assert_eq!(indices, b.row_indices());
        assert_eq!(values, b.values());
        assert_eq!(CscMatrix::from_csc(m, n, pointers, indices, values), b);
        let (_, _, pointers, mut indices, mut values) = b.clone().into_parts();
        for j in 0..cols {
            for k in (pointers[j] + 1..pointers[j + 1]).rev() {
                let other = pointers[j] + r.below(k - pointers[j] + 1);
                indices.swap(k, other);
                values.swap(k, other);
            }
        }
        let shuffled = CscMatrix::from_csc(m, n, pointers, indices, values);
        assert_csc(&shuffled);
        assert_eq!(shuffled, b);
    }
    assert_eq!(
        CscMatrix::<T>::from_csc(3, 0, vec![0], vec![], vec![]),
        CscMatrix::zeros(3, 0)
    );
}

#[test]
fn from_csc_sorts_sums_and_drops_and_into_parts_is_its_inverse() {
    each_type!(from_csc_of);
}

/// A random sparse matrix of small integers, whose products no element type rounds.
fn integers<T: Num>(r: &mut Rng, rows: usize, cols: usize, count: usize) -> SparseMatrix<T> {
    let list: Vec<(usize, usize, T)> = (0..count)
        .map(|_| {
            let x = T::real(r.below(7) as f64 - 3.0);
            (r.below(rows), r.below(cols), x)
        })
        .collect();
    SparseMatrix::from_coo(rows, cols, &list)
}

fn column_products_of<T: Num>() {
    let mut r = Rng(0x5EED_2233);
    for (rows, cols) in [(1, 1), (5, 9), (9, 5), (7, 7)] {
        let a: SparseMatrix<T> = random_sparse(&mut r, rows, cols, 2 * (rows + cols));
        let c = a.to_csc();
        let d = a.to_matrix();
        let x: Vec<T> = random_vector(&mut r, cols);
        let w: Vec<T> = random_vector(&mut r, rows);
        // By columns a product is the scatter of the rows and by rows the four-sum gather (and
        // the other way round for the transposes): the same terms in other orders, the same to
        // rounding; and the same bits where the gathered rows or columns hold under four.
        let column = |v: &[T]| Matrix::from_column_major(v.len(), 1, v.to_vec());
        let close = |p: &[T], q: &[T]| assert_close(&column(p), &column(q), 16.0);
        close(&c.mul_vec(&x), &a.mul_vec(&x));
        close(&c.mul_vec_transpose(&w), &a.mul_vec_transpose(&w));
        close(&c.mul_vec_adjoint(&w), &a.mul_vec_adjoint(&w));
        let mut y = vec![T::real(9.0); rows];
        c.mul_vec_into(&x, &mut y);
        close(&y, &a.mul_vec(&x));
        assert_eq!(y, c.mul_vec(&x));
        let mut z = vec![T::real(-9.0); cols];
        c.mul_vec_transpose_into(&w, &mut z);
        assert_eq!(z, a.transpose().mul_vec(&w));
        z.fill(T::real(5.0));
        c.mul_vec_adjoint_into(&w, &mut z);
        assert_eq!(z, a.conjugate_transpose().mul_vec(&w));
        close(&c.mul_vec_adjoint(&w), &c.conjugate_transpose().mul_vec(&w));
        let short = |m: &SparseMatrix<T>| (0..m.rows()).all(|i| m.row(i).count() < 4);
        if short(&a) {
            assert_eq!(c.mul_vec(&x), a.mul_vec(&x));
        }
        if short(&a.transpose()) {
            assert_eq!(c.mul_vec_transpose(&w), a.mul_vec_transpose(&w));
        }
        assert_close(&column(&c.mul_vec(&x)), &(&d * &column(&x)), 16.0);
        assert_close(
            &column(&c.mul_vec_adjoint(&w)),
            &(&d.conjugate_transpose() * &column(&w)),
            16.0,
        );
        // Small integers: exactly the dense products.
        let e: SparseMatrix<T> = integers(&mut r, rows, cols, rows + cols);
        let (ce, de) = (e.to_csc(), e.to_matrix());
        let xi: Vec<T> = (0..cols).map(|k| T::real(k as f64 - 2.0)).collect();
        let wi: Vec<T> = (0..rows).map(|k| T::real(3.0 - k as f64)).collect();
        assert_eq!(column(&ce.mul_vec(&xi)), &de * &column(&xi));
        assert_eq!(
            column(&ce.mul_vec_transpose(&wi)),
            &de.transpose() * &column(&wi)
        );
    }
}

#[test]
fn the_products_by_compressed_columns_are_those_by_compressed_rows() {
    each_type!(column_products_of);
}

#[test]
fn the_adjoint_product_by_columns_conjugates() {
    let i = Complex::new(0.0f32, 1.0);
    let c = SparseMatrix::from_coo(2, 1, &[(0, 0, i), (1, 0, Complex::new(2.0, 0.0))]).to_csc();
    let x = [Complex::new(1.0, 0.0), i];
    // cᵀ·x = i + 2i, and cᴴ·x = −i + 2i.
    assert_eq!(c.mul_vec_transpose(&x), [Complex::new(0.0, 3.0)]);
    assert_eq!(c.mul_vec_adjoint(&x), [Complex::new(0.0, 1.0)]);
    assert_eq!(c.conjugate_transpose()[(0, 0)], Complex::new(0.0, -1.0));
    assert_eq!(c.transpose()[(0, 0)], i);
}

fn conversions_of<T: Num>() {
    let mut r = Rng(0x5EED_2234);
    for (rows, cols) in [(1, 1), (4, 9), (9, 4), (8, 8), (0, 3), (3, 0), (0, 0)] {
        let a: SparseMatrix<T> = if rows * cols == 0 {
            SparseMatrix::zeros(rows, cols)
        } else {
            random_sparse(&mut r, rows, cols, rows * cols / 2)
        };
        let d = a.to_matrix();
        let csc = a.to_csc();
        let coo = a.to_coo();
        // Every direction, from a borrowed form and from an owned one.
        assert_eq!(CscMatrix::from(&a), csc);
        assert_eq!(CscMatrix::from(a.clone()), csc);
        assert_eq!(CooMatrix::from(&a), coo);
        assert_eq!(CooMatrix::from(a.clone()), coo);
        assert_eq!(SparseMatrix::from(&csc), a);
        assert_eq!(SparseMatrix::from(csc.clone()), a);
        assert_eq!(CooMatrix::from(&csc), csc.to_coo());
        assert_eq!(CooMatrix::from(csc.clone()), csc.to_coo());
        assert_eq!(SparseMatrix::from(&coo), a);
        assert_eq!(SparseMatrix::from(coo.clone()), a);
        assert_eq!(CscMatrix::from(&coo), csc);
        assert_eq!(CscMatrix::from(coo.clone()), csc);
        let alias: CsrMatrix<T> = coo.to_csr();
        assert_eq!(alias, a);
        // The triplets of the compressed rows come by rows, and those of the columns by columns.
        let by_rows: Vec<(usize, usize, T)> = (0..rows)
            .flat_map(|i| (0..cols).map(move |j| (i, j)))
            .filter(|&(i, j)| !d[(i, j)].is_zero())
            .map(|(i, j)| (i, j, d[(i, j)]))
            .collect();
        let by_columns: Vec<(usize, usize, T)> = (0..cols)
            .flat_map(|j| (0..rows).map(move |i| (i, j)))
            .filter(|&(i, j)| !d[(i, j)].is_zero())
            .map(|(i, j)| (i, j, d[(i, j)]))
            .collect();
        assert_eq!(coo.iter().collect::<Vec<_>>(), by_rows);
        assert_eq!(csc.to_coo().iter().collect::<Vec<_>>(), by_columns);
        // Round trips through every form, and through the dense one.
        assert_eq!(csc.to_coo().to_csr(), a);
        assert_eq!(coo.to_csc().to_csr().to_coo(), coo);
        assert_eq!(csc.to_csr().to_coo().to_csc(), csc);
        assert_eq!(
            SparseMatrix::from_matrix(&CscMatrix::from_matrix(&d).to_matrix()),
            a
        );
        assert_eq!(CooMatrix::from_matrix(&d).to_csc(), csc);
        for m in [csc.to_coo().to_csr(), coo.to_csr()] {
            assert_csr(&m);
        }
        for m in [coo.to_csc(), a.to_csc(), CscMatrix::from_matrix(&d)] {
            assert_csc(&m);
        }
        // A full matrix goes through unchanged as well.
        if rows * cols > 0 {
            let full: Matrix<T> = random_dense(&mut r, rows, cols);
            assert_eq!(CscMatrix::from_matrix(&full).to_matrix(), full);
            assert_eq!(CooMatrix::from_matrix(&full).to_matrix(), full);
            assert_eq!(CooMatrix::from_matrix(&full).nnz(), rows * cols);
        }
    }
}

#[test]
fn every_form_converts_to_every_other_and_back_unchanged() {
    each_type!(conversions_of);
}

#[test]
fn the_arrays_change_hands_without_a_copy() {
    let a = SparseMatrix::from_coo(3, 4, &[(0, 1, 1.0), (2, 3, -2.0), (2, 0, 4.0)]);
    let (columns, values) = (a.col_indices().as_ptr(), a.values().as_ptr());
    // Taken apart, and put together again from canonical arrays: the same buffers.
    let (rows, cols, pointers, col_indices, elements) = a.clone().into_parts();
    let (pointers_at, columns_at, values_at) =
        (pointers.as_ptr(), col_indices.as_ptr(), elements.as_ptr());
    let b = SparseMatrix::from_csr(rows, cols, pointers, col_indices, elements);
    assert_eq!(b, a);
    assert_eq!(b.row_pointers().as_ptr(), pointers_at);
    assert_eq!(b.col_indices().as_ptr(), columns_at);
    assert_eq!(b.values().as_ptr(), values_at);
    // The triplets of an owned matrix by rows keep its columns and values.
    let owned = a.clone();
    let (columns_owned, values_owned) = (owned.col_indices().as_ptr(), owned.values().as_ptr());
    let coo = CooMatrix::from(owned);
    assert_eq!(coo.col_indices().as_ptr(), columns_owned);
    assert_eq!(coo.values().as_ptr(), values_owned);
    assert_ne!(columns_owned, columns);
    assert_ne!(values_owned, values);
    // The triplets: handed over, transposed and taken back without a copy.
    let (r, c, v) = (vec![0usize, 2], vec![1usize, 3], vec![1.5f64, -1.0]);
    let (r_at, c_at, v_at) = (r.as_ptr(), c.as_ptr(), v.as_ptr());
    let t = CooMatrix::from_triplets(3, 4, r, c, v).into_transpose();
    assert_eq!(t.row_indices().as_ptr(), c_at);
    assert_eq!(t.col_indices().as_ptr(), r_at);
    let (_, _, rows_back, cols_back, values_back) = t.into_parts();
    assert_eq!(
        (rows_back.as_ptr(), cols_back.as_ptr(), values_back.as_ptr()),
        (c_at, r_at, v_at)
    );
    // The columns: taken apart, put together, and turned into triplets keeping rows and values.
    let csc = a.to_csc();
    let (m, n, col_pointers, row_indices, elements) = csc.clone().into_parts();
    let (rows_at, values_at) = (row_indices.as_ptr(), elements.as_ptr());
    let again = CscMatrix::from_csc(m, n, col_pointers, row_indices, elements);
    assert_eq!(again, csc);
    assert_eq!(
        (again.row_indices().as_ptr(), again.values().as_ptr()),
        (rows_at, values_at)
    );
    let triplets = CooMatrix::from(again);
    assert_eq!(triplets.row_indices().as_ptr(), rows_at);
    assert_eq!(triplets.values().as_ptr(), values_at);
    assert_eq!(triplets, csc.to_coo());
}

#[test]
fn a_long_column_in_any_order_is_converted_in_linear_time() {
    // One column of n triplets in descending order of row, and its compressed forms. A sort of
    // the column by insertion would take n²/2 steps, some 10¹⁰ at this size: minutes in a test
    // build, where the stable radix sort of the conversions takes a fraction of a second.
    let n = size(200_000, 300);
    let budget = Duration::from_secs(30);
    let begun = Instant::now();
    let rows: Vec<usize> = (0..n).rev().collect();
    let values: Vec<f64> = (0..n).rev().map(|i| (i + 1) as f64).collect();
    let coo = CooMatrix::from_triplets(n, 1, rows.clone(), vec![0; n], values.clone());
    let c = coo.to_csc();
    assert_eq!(c.nnz(), n);
    assert!(c.row_indices().windows(2).all(|w| w[0] < w[1]));
    assert!(c.column(0).all(|(i, x)| x == (i + 1) as f64));
    assert_eq!(coo.to_csr().to_csc(), c);
    assert_eq!(coo.to_csr(), c.to_csr());
    // The compressed column arrays themselves, given backwards.
    let b = CscMatrix::from_csc(n, 1, vec![0, n], rows, values);
    assert_eq!(b, c);
    // And a row and a column together: an arrow, its triplets shuffled.
    let mut r = Rng(0x5EED_2235);
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
    let mut coo = CooMatrix::with_capacity(n, n, arrow.len());
    for &(i, j, x) in &arrow {
        coo.push(i, j, x);
    }
    let by_columns = coo.to_csc();
    assert_csc(&by_columns);
    assert_eq!(by_columns.column(0).count(), n);
    assert_eq!(by_columns.to_csr(), coo.to_csr());
    assert_eq!(by_columns.transpose().to_csr(), coo.to_csr().transpose());
    assert!(
        cfg!(miri) || begun.elapsed() < budget,
        "converting took {:?}",
        begun.elapsed()
    );
}

#[test]
fn a_matrix_with_far_more_rows_than_elements_turns_into_columns_in_the_time_of_its_elements() {
    // 2⁶³ rows and three columns: the compressed columns need offsets for the columns alone, and
    // the sort of the rows must not walk a vector as long as the matrix is tall.
    let rows = usize::MAX / 2;
    let mut r = Rng(0x5EED_2236);
    let n = size(4_000, 60);
    let mut list: Vec<(usize, usize, f64)> = Vec::new();
    let mut cells: BTreeMap<(usize, usize), f64> = BTreeMap::new();
    for k in 0..n {
        // Rows spread over all 63 bits, with repeats, so that every digit of the sort counts.
        let i = if k % 7 == 3 {
            list[k - 1].0
        } else {
            (r.next() as usize) % rows
        };
        let (j, x) = (r.below(3), r.below(9) as f64 - 4.0);
        list.push((i, j, x));
        *cells.entry((j, i)).or_insert(0.0) += x;
    }
    let coo = CooMatrix::from_triplets(
        rows,
        3,
        list.iter().map(|t| t.0).collect(),
        list.iter().map(|t| t.1).collect(),
        list.iter().map(|t| t.2).collect(),
    );
    let c = coo.to_csc();
    assert_csc(&c);
    let want: Vec<((usize, usize), f64)> = cells.into_iter().filter(|&(_, x)| x != 0.0).collect();
    let got: Vec<((usize, usize), f64)> = (0..3)
        .flat_map(|j| c.column(j).map(move |(i, x)| ((j, i), x)))
        .collect();
    assert_eq!(got, want);
    assert_eq!(c.to_coo().to_csc(), c);
    let (m, n, pointers, indices, values) = c.clone().into_parts();
    assert_eq!((m, n, pointers.len()), (rows, 3, 4));
    assert_eq!(CscMatrix::from_csc(m, n, pointers, indices, values), c);
}

fn operators_of<T: Num>() {
    let mut r = Rng(0x5EED_2237);
    let a: SparseMatrix<T> = random_sparse(&mut r, 7, 5, 20);
    let c = a.to_csc();
    let d = a.to_matrix();
    let f = FnOperator::new(7, 5, |x: &[T], y: &mut [T]| a.mul_vec_into(x, y));
    let x: Vec<T> = random_vector(&mut r, 5);
    let want = a.mul_vec(&x);
    let operators: [(&str, &dyn LinearOperator<T>); 4] = [
        ("rows", &a),
        ("columns", &c),
        ("dense", &d),
        ("function", &f),
    ];
    for (name, op) in operators {
        assert_eq!((op.rows(), op.cols()), (7, 5), "{name}");
        let mut y = vec![T::real(9.0); 7];
        op.apply(&x, &mut y);
        let column = |v: &[T]| Matrix::from_column_major(v.len(), 1, v.to_vec());
        assert_close(&column(&y), &column(&want), 4.0);
        if name != "dense" {
            assert_eq!(y, want, "{name}");
        }
    }
    // The operators of the empty shapes.
    let empty = CscMatrix::<T>::zeros(3, 0);
    let mut y = vec![T::real(1.0); 3];
    empty.apply(&[], &mut y);
    assert_eq!(y, vec![T::zero(); 3]);
}

#[test]
fn every_form_and_a_function_act_as_the_same_linear_operator() {
    each_type!(operators_of);
}

#[test]
fn the_forms_print_their_shape_and_their_elements() {
    let a = SparseMatrix::from_coo(2, 3, &[(1, 0, -2.0), (0, 2, 1.5)]);
    assert_eq!(
        format!("{:?}", a.to_csc()),
        "2 × 3 {(1, 0): -2.0, (0, 2): 1.5}"
    );
    let mut coo = CooMatrix::new(2, 3);
    coo.push(1, 2, 4.0);
    coo.push(0, 0, 1.0);
    assert_eq!(format!("{coo:?}"), "2 × 3 [(1, 2, 4.0), (0, 0, 1.0)]");
    assert_eq!(format!("{:?}", CscMatrix::<f32>::zeros(0, 2)), "0 × 2 {}");
    assert_eq!(format!("{:?}", CooMatrix::<f32>::new(4, 0)), "4 × 0 []");
}

#[test]
#[should_panic(expected = "the entry lies outside the matrix")]
fn rejects_a_triplet_pushed_outside_the_matrix() {
    let mut a = CooMatrix::new(2, 2);
    a.push(2, 0, 1.0);
}

#[test]
#[should_panic(expected = "the rows, the columns and the values differ in length")]
fn rejects_triplets_of_different_lengths() {
    let _ = CooMatrix::from_triplets(2, 2, vec![0], vec![0, 1], vec![1.0]);
}

#[test]
#[should_panic(expected = "an entry lies outside the matrix")]
fn rejects_triplets_outside_the_matrix() {
    let _ = CooMatrix::from_triplets(2, 2, vec![0], vec![2], vec![1.0]);
}

#[test]
#[should_panic(expected = "there isn't an offset for each column and the end")]
fn from_csc_rejects_offsets_of_the_wrong_length() {
    let _ = CscMatrix::from_csc(2, 2, vec![0, 1], vec![0], vec![1.0]);
}

#[test]
#[should_panic(expected = "the rows and the values differ in length")]
fn from_csc_rejects_rows_and_values_of_different_lengths() {
    let _ = CscMatrix::from_csc(2, 1, vec![0, 2], vec![0, 1], vec![1.0]);
}

#[test]
#[should_panic(expected = "the first offset isn't 0")]
fn from_csc_rejects_a_first_offset_that_is_not_zero() {
    let _ = CscMatrix::from_csc(2, 1, vec![1, 1], vec![0], vec![1.0]);
}

#[test]
#[should_panic(expected = "the last offset isn't the number of elements")]
fn from_csc_rejects_a_last_offset_that_is_not_the_number_of_elements() {
    let _ = CscMatrix::from_csc(2, 1, vec![0, 2], vec![0], vec![1.0]);
}

#[test]
#[should_panic(expected = "the offsets decrease")]
fn from_csc_rejects_offsets_that_decrease() {
    let _ = CscMatrix::from_csc(2, 3, vec![0, 2, 1, 2], vec![0, 1], vec![1.0, 2.0]);
}

#[test]
#[should_panic(expected = "a row lies outside the matrix")]
fn from_csc_rejects_a_row_outside_the_matrix() {
    let _ = CscMatrix::from_csc(2, 1, vec![0, 1], vec![2], vec![1.0]);
}

#[test]
#[should_panic(expected = "the column is outside the matrix")]
fn rejects_a_column_outside_the_compressed_columns() {
    let _ = CscMatrix::<f64>::zeros(2, 3).column(3).count();
}

#[test]
#[should_panic(expected = "the element is outside the matrix")]
fn rejects_an_index_outside_the_compressed_columns() {
    let _ = CscMatrix::<f64>::zeros(2, 3)[(2, 0)];
}

#[test]
#[should_panic(expected = "the vector has the wrong length")]
fn rejects_a_vector_of_the_wrong_length_for_the_compressed_columns() {
    let _ = CscMatrix::<f64>::zeros(2, 3).mul_vec(&[1.0, 2.0]);
}

#[test]
#[should_panic(expected = "the result has the wrong length")]
fn rejects_a_result_of_the_wrong_length_for_the_compressed_columns() {
    CscMatrix::<f64>::zeros(2, 3).mul_vec_into(&[1.0; 3], &mut [0.0; 3]);
}

#[test]
#[should_panic(expected = "more rows or columns than a vector of offsets can hold")]
fn rejects_more_columns_than_a_vector_of_offsets_can_hold() {
    let _ = CscMatrix::<f64>::zeros(3, usize::MAX);
}

#[test]
#[should_panic(expected = "the vector has the wrong length for the operator")]
fn a_function_operator_rejects_a_vector_of_the_wrong_length() {
    let f = FnOperator::new(2, 3, |_: &[f64], _: &mut [f64]| {});
    f.apply(&[1.0; 2], &mut [0.0; 2]);
}

#[test]
#[should_panic(expected = "the result has the wrong length for the matrix")]
fn a_dense_operator_rejects_a_result_of_the_wrong_length() {
    let d = Matrix::<f64>::zeros(2, 3);
    LinearOperator::apply(&d, &[1.0; 3], &mut [0.0; 3]);
}
