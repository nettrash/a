//! The operations on a sparse matrix are the dense one's, done over the stored elements: the sums,
//! the products, the norms and the trace, and the sub-matrices, triangles and permutations taken
//! out of one.

use super::common::{
    Num, Rng, assert_close, assert_csr, each_type, random_dense, random_sparse, random_vector, size,
};
use a::math::complex::{Complex, Float};
use a::math::matrix::{Matrix, Norm};
use a::math::sparse::SparseMatrix;

/// A random matrix of small integers, whose sums and products are exact in every element type.
fn random_integers<T: Num>(
    r: &mut Rng,
    rows: usize,
    cols: usize,
    entries: usize,
) -> SparseMatrix<T> {
    let mut list = Vec::new();
    if rows > 0 && cols > 0 {
        for _ in 0..entries {
            list.push((
                r.below(rows),
                r.below(cols),
                T::real(r.below(7) as f64 - 3.0),
            ));
        }
    }
    SparseMatrix::from_coo(rows, cols, &list)
}

/// The dense matrix of small integers.
fn dense_integers<T: Num>(r: &mut Rng, rows: usize, cols: usize) -> Matrix<T> {
    let data: Vec<T> = (0..rows * cols)
        .map(|_| T::real(r.below(7) as f64 - 3.0))
        .collect();
    Matrix::from_column_major(rows, cols, data)
}

fn addition_of<T: Num>() {
    let mut r = Rng(0x5EED_2210);
    let a: SparseMatrix<T> = random_sparse(&mut r, 9, 7, 30);
    let b: SparseMatrix<T> = random_sparse(&mut r, 9, 7, 30);
    let (da, db) = (a.to_matrix(), b.to_matrix());
    // A sum of two elements is one rounding in either form: the same bits.
    assert_eq!((&a + &b).to_matrix(), &da + &db);
    assert_eq!((&a - &b).to_matrix(), &da - &db);
    assert_eq!((-&a).to_matrix(), -&da);
    assert_csr(&(&a + &b));
    assert_csr(&(&a - &b));
    // The borrowed and the owned operands give the same result.
    assert_eq!(&a + &b, a.clone() + b.clone());
    assert_eq!(&a + b.clone(), a.clone() + &b);
    assert_eq!(&a - &b, a.clone() - b.clone());
    assert_eq!(&a - b.clone(), a.clone() - &b);
    assert_eq!(-a.clone(), -&a);
    // The assigned forms do what the plain ones do.
    let mut sum = a.clone();
    sum += &b;
    assert_eq!(sum, &a + &b);
    sum += b.clone();
    assert_eq!(sum, &(&a + &b) + &b);
    let mut difference = a.clone();
    difference -= b.clone();
    assert_eq!(difference, &a - &b);
    difference -= &b;
    assert_eq!(difference, &(&a - &b) - &b);
    // Subtracting what was added returns the matrix, to rounding.
    assert_close(&(&(&a + &b) - &b).to_matrix(), &da, 32.0);
    // The empty matrix is the sum's neutral, and a sum stores no more than both together.
    assert_eq!(&a + &SparseMatrix::<T>::zeros(9, 7), a);
    assert_eq!(&SparseMatrix::<T>::zeros(9, 7) - &a, -&a);
    assert!((&a + &b).nnz() <= a.nnz() + b.nnz());
    // What cancels leaves the pattern.
    let same = a.clone();
    assert_eq!((&a - &same).nnz(), 0);
    assert_csr(&(&a - &same));
    assert_eq!(&a + &(-&a), SparseMatrix::zeros(9, 7));
}

#[test]
fn sums_and_differences_are_those_of_the_dense_matrices() {
    each_type!(addition_of);
}

// The operators that take the element by reference are among those tested.
#[allow(clippy::op_ref)]
fn scaling_of<T: Num>() {
    let mut r = Rng(0x5EED_2211);
    let a: SparseMatrix<T> = random_sparse(&mut r, 6, 6, 20);
    let alpha = T::real(2.5);
    let d = a.to_matrix();
    assert_eq!((&a * alpha).to_matrix(), &d * alpha);
    assert_eq!((&a / alpha).to_matrix(), &d / alpha);
    assert_eq!(a.scale(alpha), &a * alpha);
    assert_eq!(&a * &alpha, &a * alpha);
    assert_eq!(a.clone() * alpha, &a * alpha);
    assert_eq!(a.clone() * &alpha, &a * alpha);
    assert_eq!(a.clone() / alpha, &a / alpha);
    assert_eq!(&a * T::real(1.0), a);
    assert_eq!(&a / T::real(1.0), a);
    let mut scaled = a.clone();
    scaled *= alpha;
    assert_eq!(scaled, &a * alpha);
    let mut divided = a.clone();
    divided /= alpha;
    assert_eq!(divided, &a / alpha);
    // Scaling a cell to the zero leaves it unstored, so the matrix shrinks rather than fills.
    assert_eq!(&a * T::zero(), SparseMatrix::<T>::zeros(6, 6));
    assert_eq!(&a * T::real(-1.0), -&a);
    assert_csr(&(&a * alpha));
}

#[test]
fn products_and_quotients_by_an_element_are_those_of_the_dense_matrix() {
    each_type!(scaling_of);
}

#[test]
fn an_unstored_element_stays_zero_whatever_it_is_multiplied_by() {
    let a = SparseMatrix::from_coo(1, 3, &[(0, 0, f64::INFINITY), (0, 1, 1.0)]);
    // ∞ · 0 is a NaN where the infinity is stored, as in the dense matrix; 1 · 0 leaves the
    // pattern; and the cell that stores nothing stays as it is.
    let zeroed = &a * 0.0;
    assert_eq!(zeroed.col_indices(), [0]);
    assert!(zeroed[(0, 0)].is_nan());
    assert_eq!(a.scale(0.0).nnz(), 1);
    // By a NaN and by zero under the bar, the stored elements alone change.
    let by_nan = &a * f64::NAN;
    assert_eq!(by_nan.col_indices(), [0, 1]);
    assert!(by_nan.values().iter().all(|x| x.is_nan()));
    let over_zero = &a / 0.0;
    assert_eq!(over_zero.values(), [f64::INFINITY, f64::INFINITY]);
    assert_eq!(over_zero[(0, 2)], 0.0);
    // An underflow to zero leaves the pattern too.
    let tiny = SparseMatrix::from_coo(1, 2, &[(0, 0, 1e-200), (0, 1, 1e100)]);
    assert_eq!((&tiny * 1e-200).col_indices(), [1]);
    assert_eq!((&tiny / 1e200).col_indices(), [1]);
    // A sum of opposite infinities is a NaN, and stored.
    let b = SparseMatrix::from_coo(1, 3, &[(0, 0, f64::NEG_INFINITY)]);
    assert!((&a + &b)[(0, 0)].is_nan());
}

fn products_of<T: Num>() {
    let mut r = Rng(0x5EED_2212);
    let a: SparseMatrix<T> = random_sparse(&mut r, 6, 8, 24);
    let x: Matrix<T> = random_dense(&mut r, 8, 3);
    let y: Matrix<T> = random_dense(&mut r, 4, 6);
    let da = a.to_matrix();
    assert_close(&a.mul_matrix(&x), &(&da * &x), 16.0);
    assert_close(&(&a * &x), &(&da * &x), 16.0);
    assert_close(&(&y * &a), &(&y * &da), 16.0);
    // The owned and the borrowed operands agree, on either side.
    assert_eq!(&a * x.clone(), &a * &x);
    assert_eq!(a.clone() * &x, &a * &x);
    assert_eq!(a.clone() * x.clone(), &a * &x);
    assert_eq!(&y * a.clone(), &y * &a);
    assert_eq!(y.clone() * &a, &y * &a);
    assert_eq!(y.clone() * a.clone(), &y * &a);
    // The vector product is the matrix one of the same column.
    let v: Vec<T> = random_vector(&mut r, 8);
    let product = a.mul_vec(&v);
    assert_eq!(product.len(), a.rows());
    let column = Matrix::from_column_major(8, 1, v.clone());
    assert_eq!(product, a.mul_matrix(&column).into_vec());
    assert_close(
        &Matrix::from_column_major(6, 1, product.clone()),
        &(&da * &column),
        16.0,
    );
    let mut into = vec![T::real(9.0); 6];
    a.mul_vec_into(&v, &mut into);
    assert_eq!(into, product);
    // The identity is the product's neutral, on either side, and stores nothing extra.
    assert_eq!(&a * &SparseMatrix::<T>::identity(8), a);
    assert_eq!(&SparseMatrix::<T>::identity(6) * &a, a);
    assert_eq!(&Matrix::<T>::identity(6) * &a, da);
    assert_eq!(&a * &Matrix::<T>::identity(8), da);
}

#[test]
fn products_with_dense_matrices_and_vectors_are_the_dense_ones() {
    each_type!(products_of);
}

fn exact_products_of<T: Num>() {
    // Small integers, whose products and sums no element type rounds: every product must equal
    // the dense one bit for bit, on shapes of every kind, the empty ones too. The product of a
    // dense matrix by a sparse one read the rows of the sparse factor as its columns, which went
    // unseen on square matrices only where the result happened to be symmetric.
    let mut r = Rng(0x5EED_2215);
    for case in 0..size(300, 20) {
        let (m, k, n) = (r.below(7), r.below(7), r.below(7));
        let a: SparseMatrix<T> = random_integers(&mut r, m, k, 2 * (m + k));
        let b: SparseMatrix<T> = random_integers(&mut r, k, n, 2 * (k + n));
        let x: Matrix<T> = dense_integers(&mut r, k, n);
        let y: Matrix<T> = dense_integers(&mut r, n, m);
        let (da, db) = (a.to_matrix(), b.to_matrix());
        assert_eq!(
            (&a * &b).to_matrix(),
            &da * &db,
            "case {case}: sparse × sparse"
        );
        assert_csr(&(&a * &b));
        assert_eq!(&a * &x, &da * &x, "case {case}: sparse × dense");
        assert_eq!(&y * &a, &y * &da, "case {case}: dense × sparse");
        let v: Vec<T> = (0..k).map(|_| T::real(r.below(7) as f64 - 3.0)).collect();
        let w: Vec<T> = (0..m).map(|_| T::real(r.below(7) as f64 - 3.0)).collect();
        let column = |v: &[T]| Matrix::from_column_major(v.len(), 1, v.to_vec());
        assert_eq!(a.mul_vec(&v), (&da * &column(&v)).into_vec(), "case {case}");
        assert_eq!(
            a.mul_vec_transpose(&w),
            (&da.transpose() * &column(&w)).into_vec(),
            "case {case}"
        );
        assert_eq!(
            a.mul_vec_adjoint(&w),
            (&da.conjugate_transpose() * &column(&w)).into_vec(),
            "case {case}"
        );
    }
}

#[test]
fn products_of_integers_are_exactly_the_dense_ones_on_every_shape() {
    each_type!(exact_products_of);
}

fn transposed_products_of<T: Num>() {
    let mut r = Rng(0x5EED_2216);
    for (rows, cols) in [(1, 1), (5, 9), (9, 5), (7, 7), (3, 0), (0, 3)] {
        let a: SparseMatrix<T> = if rows * cols == 0 {
            SparseMatrix::zeros(rows, cols)
        } else {
            random_sparse(&mut r, rows, cols, 2 * (rows + cols))
        };
        let x: Vec<T> = random_vector(&mut r, rows);
        // The products without a transposed copy add the same terms as the product by the
        // transpose that is formed, in another order (that one sums each row in four running
        // sums): the same to rounding, at the bound the documentation states.
        let as_column = |v: Vec<T>| Matrix::from_column_major(v.len(), 1, v);
        let close = |p: Vec<T>, q: Vec<T>| assert_close(&as_column(p), &as_column(q), 16.0);
        close(a.mul_vec_transpose(&x), a.transpose().mul_vec(&x));
        close(a.mul_vec_adjoint(&x), a.conjugate_transpose().mul_vec(&x));
        let mut y = vec![T::real(7.0); cols];
        a.mul_vec_transpose_into(&x, &mut y);
        close(y.clone(), a.transpose().mul_vec(&x));
        y.fill(T::real(-3.0));
        a.mul_vec_adjoint_into(&x, &mut y);
        close(y.clone(), a.conjugate_transpose().mul_vec(&x));
        // The same input gives the same bits, every time.
        assert_eq!(a.transpose().mul_vec(&x), a.transpose().mul_vec(&x));
        // Where no column holds four elements the two orders are one: the same bits.
        if (0..cols).all(|j| (0..rows).filter(|&i| a.find(i, j).is_some()).count() < 4) {
            assert_eq!(a.mul_vec_transpose(&x), a.transpose().mul_vec(&x));
        }
        // And they are the dense products.
        let column = Matrix::from_column_major(rows, 1, x.clone());
        let d = a.to_matrix();
        assert_close(
            &Matrix::from_column_major(cols, 1, a.mul_vec_transpose(&x)),
            &(&d.transpose() * &column),
            16.0,
        );
        assert_close(
            &Matrix::from_column_major(cols, 1, a.mul_vec_adjoint(&x)),
            &(&d.conjugate_transpose() * &column),
            16.0,
        );
    }
}

#[test]
fn products_by_the_transpose_and_the_adjoint_need_no_transposed_copy() {
    each_type!(transposed_products_of);
}

#[test]
fn the_adjoint_product_conjugates_and_the_transposed_one_does_not() {
    let i = Complex::new(0.0, 1.0);
    let a = SparseMatrix::from_coo(2, 1, &[(0, 0, i), (1, 0, Complex::new(2.0, 0.0))]);
    let x = [Complex::new(1.0, 0.0), i];
    // aᵀ·x = i + 2i, and aᴴ·x = −i + 2i.
    assert_eq!(a.mul_vec_transpose(&x), [Complex::new(0.0, 3.0)]);
    assert_eq!(a.mul_vec_adjoint(&x), [Complex::new(0.0, 1.0)]);
    // A real matrix has one transpose.
    let b = SparseMatrix::from_coo(2, 1, &[(0, 0, 3.0f32), (1, 0, -1.0)]);
    assert_eq!(
        b.mul_vec_transpose(&[1.0, 2.0]),
        b.mul_vec_adjoint(&[1.0, 2.0])
    );
}

fn sparse_products_of<T: Num>() {
    let mut r = Rng(0x5EED_2213);
    let a: SparseMatrix<T> = random_sparse(&mut r, 5, 7, 18);
    let b: SparseMatrix<T> = random_sparse(&mut r, 7, 4, 16);
    assert_close(
        &(&a * &b).to_matrix(),
        &(a.to_matrix() * b.to_matrix()),
        16.0,
    );
    assert_csr(&(&a * &b));
    // The borrowed and owned operands agree, as do the assigned forms.
    assert_eq!(&a * &b, a.clone() * b.clone());
    assert_eq!(&a * b.clone(), a.clone() * &b);
    let mut product = a.clone();
    product *= &b;
    assert_eq!(product, &a * &b);
    let mut product = a.clone();
    product *= b.clone();
    assert_eq!(product, &a * &b);
    // A Gram matrix has the symmetric pattern and the dense Gram's elements.
    let gram = &a * &a.transpose();
    assert!(gram.is_symmetric_pattern());
    assert_close(
        &gram.to_matrix(),
        &(a.to_matrix() * a.to_matrix().transpose()),
        24.0,
    );
    // The product stores no more than the dense one has nonzeros, and a zero row stays empty.
    assert!(gram.nnz() <= a.rows() * a.rows());
    assert_eq!((&SparseMatrix::<T>::zeros(5, 7) * &b).nnz(), 0);
    // A product whose terms cancel stores nothing: [1 1]·[1; −1].
    let row = SparseMatrix::from_coo(1, 2, &[(0, 0, T::real(1.0)), (0, 1, T::real(1.0))]);
    let col = SparseMatrix::from_coo(2, 1, &[(0, 0, T::real(1.0)), (1, 0, T::real(-1.0))]);
    assert_eq!((&row * &col).nnz(), 0);
    assert_csr(&(&row * &col));
    assert_eq!((&col * &row).nnz(), 4);
}

#[test]
fn products_of_two_sparse_matrices_are_the_dense_ones() {
    each_type!(sparse_products_of);
}

fn norms_and_trace_of<T: Num>() {
    let mut r = Rng(0x5EED_2214);
    for (rows, cols) in [(7, 5), (5, 7), (6, 6), (1, 9), (9, 1)] {
        let a = random_sparse::<T>(&mut r, rows, cols, 26);
        let d = a.to_matrix();
        for kind in [Norm::One, Norm::Infinity, Norm::Frobenius, Norm::Max] {
            let (sparse, dense) = (a.norm(kind).to_f64(), d.norm(kind).to_f64());
            let tolerance = 8.0 * T::EPS * dense;
            assert!(
                (sparse - dense).abs() <= tolerance,
                "{kind:?} differs: {sparse} against {dense}"
            );
        }
        // The 2-norm and the nuclear norm go through the dense matrix: the same numbers.
        assert_eq!(a.norm(Norm::Two).to_f64(), d.norm(Norm::Two).to_f64());
        assert_eq!(
            a.norm(Norm::Nuclear).to_f64(),
            d.norm(Norm::Nuclear).to_f64()
        );
        // The diagonal has min(rows, cols) elements, and the trace of a square matrix is its
        // sum, as the dense trace is.
        let diagonal = a.diagonal();
        assert_eq!(diagonal.len(), rows.min(cols));
        if rows == cols {
            let sum = diagonal.iter().fold(T::zero(), |s, &x| s + x);
            assert_eq!(a.trace(), sum);
            assert_eq!(a.trace(), d.trace());
        }
    }
    let square = random_sparse::<T>(&mut r, 6, 6, 30);
    assert_eq!(square.trace(), square.to_matrix().trace());
    // An empty matrix has no size and no trace, and the identity of n has both equal to n.
    for kind in [Norm::One, Norm::Infinity, Norm::Frobenius, Norm::Max] {
        assert_eq!(SparseMatrix::<T>::zeros(4, 4).norm(kind), T::Real::ZERO);
        assert_eq!(SparseMatrix::<T>::zeros(0, 4).norm(kind), T::Real::ZERO);
        assert_eq!(SparseMatrix::<T>::zeros(4, 0).norm(kind), T::Real::ZERO);
    }
    assert_eq!(SparseMatrix::<T>::zeros(4, 4).trace(), T::zero());
    assert_eq!(SparseMatrix::<T>::zeros(0, 0).trace(), T::zero());
    assert_eq!(SparseMatrix::<T>::identity(3).trace(), T::real(3.0));
    assert_eq!(SparseMatrix::<T>::identity(3).norm(Norm::One), T::Real::ONE);
    assert_eq!(
        SparseMatrix::<T>::identity(4).norm(Norm::Frobenius),
        T::Real::TWO
    );
}

#[test]
fn the_norms_and_the_trace_are_those_of_the_dense_matrix() {
    each_type!(norms_and_trace_of);
}

#[test]
fn a_norm_of_a_matrix_holding_a_nan_is_a_nan() {
    let a = SparseMatrix::from_coo(2, 2, &[(0, 0, f64::NAN), (1, 1, 1.0)]);
    let d = a.to_matrix();
    for kind in [Norm::One, Norm::Infinity, Norm::Frobenius, Norm::Max] {
        assert!(a.norm(kind).is_nan(), "{kind:?} of a NaN");
        assert!(d.norm(kind).is_nan(), "{kind:?} of a dense NaN");
    }
    let alone = SparseMatrix::from_coo(1, 1, &[(0, 0, f32::NAN)]);
    for kind in [Norm::One, Norm::Infinity, Norm::Frobenius, Norm::Max] {
        assert!(alone.norm(kind).is_nan(), "{kind:?} of a lone NaN");
    }
    let c = SparseMatrix::from_coo(1, 2, &[(0, 1, Complex::new(1.0, f64::NAN))]);
    for kind in [Norm::One, Norm::Infinity, Norm::Frobenius, Norm::Max] {
        assert!(c.norm(kind).is_nan(), "{kind:?} of a complex NaN");
    }
    // An infinity makes every norm infinite, and Frobenius's also beside a NaN, as hypot has it.
    let inf = SparseMatrix::from_coo(1, 2, &[(0, 0, f64::NEG_INFINITY), (0, 1, 2.0)]);
    for kind in [Norm::One, Norm::Infinity, Norm::Frobenius, Norm::Max] {
        assert_eq!(inf.norm(kind), f64::INFINITY, "{kind:?} of an infinity");
    }
    let both = SparseMatrix::from_coo(1, 2, &[(0, 0, f64::INFINITY), (0, 1, f64::NAN)]);
    assert_eq!(both.norm(Norm::Frobenius), f64::INFINITY);
    assert_eq!(f64::INFINITY.hypot(f64::NAN), f64::INFINITY);
}

#[test]
fn the_frobenius_norm_neither_overflows_nor_underflows() {
    let big = SparseMatrix::from_coo(1, 2, &[(0, 0, 1e200), (0, 1, 1e200)]);
    assert_eq!(big.norm(Norm::Frobenius), 1e200 * 2f64.sqrt());
    let small = SparseMatrix::from_coo(1, 2, &[(0, 0, 3e-200f64), (0, 1, 4e-200)]);
    assert!((small.norm(Norm::Frobenius) - 5e-200).abs() <= 5e-200 * 2.0 * f64::EPSILON);
    let single = SparseMatrix::from_coo(1, 2, &[(0, 0, 3e30f32), (0, 1, 4e30)]);
    assert_eq!(single.norm(Norm::Frobenius), 5e30);
    // The largest and the smallest numbers there are.
    let top = SparseMatrix::from_coo(1, 2, &[(0, 0, f64::MAX), (0, 1, f64::MAX)]);
    assert_eq!(top.norm(Norm::Frobenius), f64::INFINITY);
    assert_eq!(top.norm(Norm::Max), f64::MAX);
    let half = SparseMatrix::from_coo(1, 2, &[(0, 0, f64::MAX / 2.0), (0, 1, f64::MAX / 2.0)]);
    assert_eq!(half.norm(Norm::Frobenius), (f64::MAX / 2.0) * 2f64.sqrt());
    let least = f64::from_bits(1);
    let bottom = SparseMatrix::from_coo(1, 2, &[(0, 0, 3.0 * least), (0, 1, 4.0 * least)]);
    assert_eq!(bottom.norm(Norm::Frobenius), 5.0 * least);
    // |3e200 − 4e200·i| is 4.99999999999999984866…e200 (mpmath, 300 bits), which rounds to
    // 4.9999999999999995e200, one ulp below 5e200: the decimals are not exactly 3 and 4 times
    // a power of ten.
    let c = SparseMatrix::from_coo(1, 1, &[(0, 0, Complex::new(3e200, -4e200))]);
    assert_eq!(c.norm(Norm::Frobenius), 4.999_999_999_999_999_5e200);
    assert_eq!(c.norm(Norm::Max), 4.999_999_999_999_999_5e200);
}

#[test]
fn the_norms_of_long_sums_keep_their_accuracy() {
    // A row and a column of 2²⁴ followed by a thousand ones. Summed in `f32`, every one is lost
    // in the rounding, 2²⁴ + 1 being no `f32`: the norms came out as 2²⁴, and as 2¹² for
    // Frobenius's, of [4096, 1, …, 1]. They are 16 778 216 and √(2²⁴ + 1000).
    let ones = 1000;
    let top = 16_777_216.0f32;
    let mut row: Vec<(usize, usize, f32)> = vec![(0, 0, top)];
    row.extend((1..=ones).map(|j| (0, j, 1.0)));
    let wide = SparseMatrix::from_coo(1, ones + 1, &row);
    let tall = wide.transpose();
    assert_eq!(wide.norm(Norm::Infinity), 16_778_216.0);
    assert_eq!(tall.norm(Norm::One), 16_778_216.0);
    assert_eq!(wide.norm(Norm::One), top);
    assert_eq!(tall.norm(Norm::Infinity), top);
    let wide = wide.map(|_, j, x| if j == 0 { 4096.0 } else { x });
    let exact = (16_777_216.0f64 + ones as f64).sqrt();
    assert_eq!(wide.norm(Norm::Frobenius), exact as f32);
    assert_eq!(wide.transpose().norm(Norm::Frobenius), exact as f32);

    // The same in `f64`, where a plain sum loses the ones beside 2⁵³: the compensated sum keeps
    // them.
    let top = 9_007_199_254_740_992.0f64;
    let mut row: Vec<(usize, usize, f64)> = vec![(0, 0, top)];
    row.extend((1..=ones).map(|j| (0, j, 1.0)));
    let wide = SparseMatrix::from_coo(1, ones + 1, &row);
    assert_eq!(wide.norm(Norm::Infinity), top + ones as f64);
    assert_eq!(wide.transpose().norm(Norm::One), top + ones as f64);
    // √(2⁵² + 1000) = 2²⁶·√(1 + 1000/2⁵²), which to the last bit is 2²⁶ + 1000/2²⁷.
    let wide = wide.map(|_, j, x| if j == 0 { 67_108_864.0 } else { x });
    let exact = 67_108_864.0 + 1000.0 / 134_217_728.0;
    assert_eq!(wide.norm(Norm::Frobenius), exact);

    // The 2-D Laplacian of a k × k grid has the Frobenius norm √(20k² − 4k), exactly: 5k² − 4k
    // elements of −1 and k² of 4. In `f32` the squares pass 2²⁴ long before the end.
    let k = size(96, 6);
    let index = |i: usize, j: usize| i * k + j;
    let mut entries: Vec<(usize, usize, f32)> = Vec::new();
    for i in 0..k {
        for j in 0..k {
            entries.push((index(i, j), index(i, j), 4.0));
            if i > 0 {
                entries.push((index(i, j), index(i - 1, j), -1.0));
            }
            if i + 1 < k {
                entries.push((index(i, j), index(i + 1, j), -1.0));
            }
            if j > 0 {
                entries.push((index(i, j), index(i, j - 1), -1.0));
            }
            if j + 1 < k {
                entries.push((index(i, j), index(i, j + 1), -1.0));
            }
        }
    }
    let laplacian = SparseMatrix::from_coo(k * k, k * k, &entries);
    assert_eq!(laplacian.nnz(), 5 * k * k - 4 * k);
    let exact = ((20 * k * k - 4 * k) as f64).sqrt();
    assert_eq!(laplacian.norm(Norm::Frobenius), exact as f32);
    assert_eq!(laplacian.norm(Norm::One), 8.0);
    assert_eq!(laplacian.norm(Norm::Infinity), 8.0);
    assert_eq!(laplacian.norm(Norm::Max), 4.0);
}

/// The dense matrix of the rows and the columns two lists name.
fn dense_select<T: Num>(d: &Matrix<T>, rows: &[usize], cols: &[usize]) -> Matrix<T> {
    Matrix::from_fn(rows.len(), cols.len(), |i, j| d[(rows[i], cols[j])])
}

fn submatrices_of<T: Num>() {
    let mut r = Rng(0x5EED_2217);
    for case in 0..size(200, 16) {
        let (rows, cols) = (1 + r.below(9), 1 + r.below(9));
        let count = r.below(rows * cols + 1);
        let a: SparseMatrix<T> = random_sparse(&mut r, rows, cols, count);
        let d = a.to_matrix();
        // A range of rows and one of columns, which may be empty or everything.
        let (r0, c0) = (r.below(rows + 1), r.below(cols + 1));
        let (r1, c1) = (r0 + r.below(rows - r0 + 1), c0 + r.below(cols - c0 + 1));
        let sub = a.submatrix(r0..r1, c0..c1);
        assert_csr(&sub);
        let row_list: Vec<usize> = (r0..r1).collect();
        let col_list: Vec<usize> = (c0..c1).collect();
        assert_eq!(
            sub.to_matrix(),
            dense_select(&d, &row_list, &col_list),
            "case {case}"
        );
        // The same by the lists of the indices.
        assert_eq!(a.select(&row_list, &col_list), sub, "case {case}");
        // Any lists: out of order, with repeats, longer than the matrix, or empty.
        let row_list: Vec<usize> = (0..r.below(2 * rows + 1)).map(|_| r.below(rows)).collect();
        let col_list: Vec<usize> = (0..r.below(2 * cols + 1)).map(|_| r.below(cols)).collect();
        let picked = a.select(&row_list, &col_list);
        assert_csr(&picked);
        assert_eq!(picked.shape(), (row_list.len(), col_list.len()));
        assert_eq!(
            picked.to_matrix(),
            dense_select(&d, &row_list, &col_list),
            "case {case}"
        );
    }
    let a: SparseMatrix<T> = random_sparse(&mut r, 6, 8, 30);
    assert_eq!(a.submatrix(0..6, 0..8), a);
    assert_eq!(a.submatrix(2..2, 0..8), SparseMatrix::zeros(0, 8));
    assert_eq!(a.submatrix(0..6, 8..8), SparseMatrix::zeros(6, 0));
    assert_eq!(a.select(&[], &[]), SparseMatrix::zeros(0, 0));
    let everything: (Vec<usize>, Vec<usize>) = ((0..6).collect(), (0..8).collect());
    assert_eq!(a.select(&everything.0, &everything.1), a);
    // One row and one column at a time are the rows and the columns.
    for i in 0..6 {
        let row: Vec<(usize, T)> = a.submatrix(i..i + 1, 0..8).row(0).collect();
        assert_eq!(row, a.row(i).collect::<Vec<_>>());
    }
    for j in 0..8 {
        let column = a.select(&everything.0, &[j]);
        let want: Vec<(usize, T)> = a.column(j);
        let got: Vec<(usize, T)> = (0..6)
            .flat_map(|i| column.row(i).map(move |(_, x)| (i, x)))
            .collect();
        assert_eq!(got, want);
    }
}

#[test]
fn submatrices_and_selections_are_those_of_the_dense_matrix() {
    each_type!(submatrices_of);
}

#[test]
fn a_selection_searches_the_shorter_of_a_row_and_the_columns_in_the_longer() {
    // Long rows and few columns, and short rows and many columns: the two branches of the search,
    // each against the dense selection.
    let mut r = Rng(0x5EED_2218);
    let n = size(400, 40);
    let full: SparseMatrix<f64> = SparseMatrix::from_fn(4, n, |i, j| ((i + 1) * (j + 1)) as f64);
    let thin: SparseMatrix<f64> = random_sparse(&mut r, 4, n, n / 10 + 4);
    for a in [&full, &thin] {
        let d = a.to_matrix();
        for cols in [
            vec![n - 1, 0, n / 2, 0],
            (0..n).rev().collect::<Vec<_>>(),
            (0..3 * n).map(|k| (k * 7) % n).collect(),
            (0..n).step_by(3).collect(),
        ] {
            let rows = [3, 3, 0];
            let picked = a.select(&rows, &cols);
            assert_csr(&picked);
            assert_eq!(picked.to_matrix(), dense_select(&d, &rows, &cols));
        }
    }
}

#[test]
fn a_selection_costs_what_it_takes_and_not_what_the_matrix_holds() {
    // A million rows and columns, a few of them taken, a few thousand times: proportional to the
    // matrix, each selection would walk its 3·10⁶ elements or a table of 10⁶ columns, hours in
    // all. The bound is generous because the machine may be busy.
    let n = size(1_000_000, 500);
    let a = SparseMatrix::tridiagonal(&vec![-1.0; n - 1], &vec![2.0; n], &vec![-1.0; n - 1]);
    let begun = std::time::Instant::now();
    let mut r = Rng(0x5EED_2219);
    let mut total = 0.0;
    for _ in 0..size(5_000, 20) {
        let at = 1 + r.below(n - 2);
        let picked = a.select(&[at, at - 1], &[at + 1, at, at - 1]);
        assert_eq!(
            picked.to_matrix().as_slice(),
            &[-1.0, 0.0, 2.0, -1.0, -1.0, 2.0]
        );
        let sub = a.submatrix(at - 1..at + 1, at - 1..at + 2);
        assert_eq!(
            sub.to_matrix().as_slice(),
            &[2.0, -1.0, -1.0, 2.0, 0.0, -1.0]
        );
        total += picked.diagonal().iter().sum::<f64>() + sub.diagonal().iter().sum::<f64>();
    }
    assert!(total != 0.0);
    assert!(
        cfg!(miri) || begun.elapsed() < std::time::Duration::from_secs(30),
        "the selections took {:?}",
        begun.elapsed()
    );
}

fn triangles_of<T: Num>() {
    let mut r = Rng(0x5EED_221A);
    for (rows, cols) in [(1, 1), (6, 6), (4, 9), (9, 4), (0, 3), (3, 0)] {
        let a: SparseMatrix<T> = if rows * cols == 0 {
            SparseMatrix::zeros(rows, cols)
        } else {
            random_sparse(&mut r, rows, cols, rows * cols)
        };
        let d = a.to_matrix();
        let (upper, lower) = (a.upper_triangle(), a.lower_triangle());
        assert_csr(&upper);
        assert_csr(&lower);
        let zero = T::zero();
        assert_eq!(
            upper.to_matrix(),
            Matrix::from_fn(rows, cols, |i, j| if j >= i { d[(i, j)] } else { zero })
        );
        assert_eq!(
            lower.to_matrix(),
            Matrix::from_fn(rows, cols, |i, j| if j <= i { d[(i, j)] } else { zero })
        );
        // The two share the diagonal and nothing else, and between them hold the matrix.
        let diagonal = a.map(|i, j, x| if i == j { x } else { zero });
        assert_eq!(&(&upper + &lower) - &diagonal, a);
        assert_eq!(upper.nnz() + lower.nnz(), a.nnz() + diagonal.nnz());
        // The triangle of a triangle is itself, and the other triangle of it the diagonal.
        assert_eq!(upper.upper_triangle(), upper);
        assert_eq!(upper.lower_triangle(), diagonal);
        assert_eq!(a.transpose().upper_triangle(), lower.transpose());
    }
}

#[test]
fn the_triangles_are_those_of_the_dense_matrix_with_the_diagonal() {
    each_type!(triangles_of);
}

/// A random permutation of `0..n`.
fn random_permutation(r: &mut Rng, n: usize) -> Vec<usize> {
    let mut p: Vec<usize> = (0..n).collect();
    for k in (1..n).rev() {
        p.swap(k, r.below(k + 1));
    }
    p
}

fn permutations_of<T: Num>() {
    let mut r = Rng(0x5EED_221B);
    for case in 0..size(100, 10) {
        let (rows, cols) = (1 + r.below(9), 1 + r.below(9));
        let count = r.below(rows * cols + 1);
        let a: SparseMatrix<T> = random_sparse(&mut r, rows, cols, count);
        let d = a.to_matrix();
        let (p, q) = (
            random_permutation(&mut r, rows),
            random_permutation(&mut r, cols),
        );
        let b = a.permute(&p, &q);
        assert_csr(&b);
        // `p[new] = old`: the element (i, j) of the result is the element (p[i], q[j]).
        assert_eq!(b.to_matrix(), dense_select(&d, &p, &q), "case {case}");
        assert_eq!(b, a.select(&p, &q), "case {case}");
        assert_eq!(b.nnz(), a.nnz());
        // The inverse permutations undo it.
        let invert = |p: &[usize]| {
            let mut inverse = vec![0; p.len()];
            for (new, &old) in p.iter().enumerate() {
                inverse[old] = new;
            }
            inverse
        };
        assert_eq!(b.permute(&invert(&p), &invert(&q)), a, "case {case}");
        // The identity changes nothing.
        let identity = |n: usize| (0..n).collect::<Vec<usize>>();
        assert_eq!(a.permute(&identity(rows), &identity(cols)), a);
    }
    // With the matrices P and Q that have a 1 at (i, p[i]) and at (j, q[j]): P·A·Qᵀ.
    let a: SparseMatrix<T> = random_sparse(&mut r, 5, 7, 20);
    let (p, q) = (random_permutation(&mut r, 5), random_permutation(&mut r, 7));
    let one = T::one();
    let pm = SparseMatrix::from_coo(5, 5, &(0..5).map(|i| (i, p[i], one)).collect::<Vec<_>>());
    let qm = SparseMatrix::from_coo(7, 7, &(0..7).map(|j| (j, q[j], one)).collect::<Vec<_>>());
    assert_eq!(a.permute(&p, &q), &(&pm * &a) * &qm.transpose());
    // One permutation of the rows and the columns of a symmetric matrix keeps it symmetric.
    let s = &(&a * &a.transpose());
    let p = random_permutation(&mut r, 5);
    assert_eq!(
        s.permute(&p, &p).is_symmetric_pattern(),
        s.is_symmetric_pattern()
    );
    assert_eq!(s.permute(&p, &p).diagonal().len(), 5);
    // Nothing to permute.
    assert_eq!(
        SparseMatrix::<T>::zeros(0, 0).permute(&[], &[]),
        SparseMatrix::zeros(0, 0)
    );
    assert_eq!(
        SparseMatrix::<T>::zeros(0, 2).permute(&[], &[1, 0]),
        SparseMatrix::zeros(0, 2)
    );
}

#[test]
fn a_permutation_puts_the_old_row_and_column_where_the_new_ones_say() {
    each_type!(permutations_of);
}

#[test]
fn a_dense_matrix_times_a_sparse_one_is_that_product_and_not_the_one_by_the_transpose() {
    // The cases of the audit (p_storage-1): the operator returned D·Sᵀ, a wrong matrix of the
    // right shape for a tall S, and panicked on a wide one.
    let s = SparseMatrix::from_coo(2, 2, &[(0, 1, 1.0f64)]);
    let d = Matrix::from_rows(&[[1.0, 2.0], [3.0, 4.0]]);
    assert_eq!(&d * &s, Matrix::from_rows(&[[0.0, 1.0], [0.0, 3.0]]));
    let tall = SparseMatrix::from_coo(3, 2, &[(0, 1, 1.0f64), (2, 0, 7.0)]);
    let r3 = Matrix::from_rows(&[[1.0, 2.0, 3.0]]);
    assert_eq!(&r3 * &tall, Matrix::from_rows(&[[21.0, 1.0]]));
    let wide = SparseMatrix::from_coo(2, 3, &[(0, 0, 1.0f64), (1, 2, 5.0)]);
    let r2 = Matrix::from_rows(&[[1.0, 2.0]]);
    assert_eq!(&r2 * &wide, Matrix::from_rows(&[[1.0, 0.0, 10.0]]));
    // Every form of the operands, against the dense product.
    assert_eq!(r2.clone() * wide.clone(), &r2 * &wide.to_matrix());
    assert_eq!(&r2 * wide.clone(), &r2 * &wide.to_matrix());
    assert_eq!(r2.clone() * &wide, &r2 * &wide.to_matrix());
}

#[test]
fn the_f32_frobenius_norm_keeps_its_accuracy_over_millions_of_elements() {
    // The cases of the audit (p_storage-2): summed by hypot in f32, the matrix of ones of
    // 2048 × 2048 came out as 2020.7085 for 2048, and the Laplacian of the 256 × 256 grid as
    // 1145.6541 for 1144.4196, the value the dense norm and scipy give.
    let n = size(2048, 16);
    let ones = SparseMatrix::<f32>::from_fn(n, n, |_, _| 1.0);
    assert_eq!(ones.norm(Norm::Frobenius), n as f32);
    let k = size(256, 8);
    let mut entries: Vec<(usize, usize, f32)> = Vec::with_capacity(5 * k * k);
    for p in 0..k {
        for q in 0..k {
            let i = p * k + q;
            entries.push((i, i, 4.0));
            if p > 0 {
                entries.push((i, i - k, -1.0));
            }
            if q > 0 {
                entries.push((i, i - 1, -1.0));
            }
            if q + 1 < k {
                entries.push((i, i + 1, -1.0));
            }
            if p + 1 < k {
                entries.push((i, i + k, -1.0));
            }
        }
    }
    let laplacian = SparseMatrix::from_coo(k * k, k * k, &entries);
    let exact = ((16 * k * k + 4 * k * (k - 1)) as f64).sqrt();
    assert_eq!(laplacian.norm(Norm::Frobenius), exact as f32);
    if k == 256 {
        assert!((laplacian.norm(Norm::Frobenius) - 1144.4196).abs() < 1e-4);
    }
}

fn nan_norms_of<T: Num>() {
    // The cases the verification of p_storage-4 lists: the One, Infinity and Max norms ignored a
    // NaN, where the dense norm and scipy give NaN.
    let nan = T::real(f64::NAN);
    let (one, two, three, five) = (T::real(1.0), T::real(2.0), T::real(3.0), T::real(5.0));
    let mut cases = vec![
        SparseMatrix::from_coo(2, 2, &[(0, 0, nan), (1, 1, one)]),
        SparseMatrix::from_coo(1, 1, &[(0, 0, nan)]),
        SparseMatrix::from_coo(2, 2, &[(0, 0, three), (0, 1, five), (1, 1, nan)]),
        SparseMatrix::from_coo(2, 3, &[(0, 0, one), (0, 2, two), (1, 1, nan)]),
        SparseMatrix::from_fn(2, 2, |_, _| nan),
    ];
    if T::IS_COMPLEX {
        let imaginary = T::from_parts(T::Real::ONE, T::Real::NAN);
        cases.push(SparseMatrix::from_coo(1, 2, &[(0, 1, imaginary)]));
    }
    for (case, a) in cases.iter().enumerate() {
        for kind in [Norm::One, Norm::Infinity, Norm::Max, Norm::Frobenius] {
            assert!(a.norm(kind).is_nan(), "case {case}: {kind:?}");
            assert!(
                a.to_matrix().norm(kind).is_nan(),
                "case {case}: dense {kind:?}"
            );
        }
    }
    // An infinity is infinite, and nothing is nothing.
    let inf = SparseMatrix::from_coo(2, 2, &[(0, 0, T::real(f64::INFINITY)), (1, 1, one)]);
    for kind in [Norm::One, Norm::Infinity, Norm::Max, Norm::Frobenius] {
        assert_eq!(inf.norm(kind), T::Real::INFINITY, "{kind:?}");
        assert_eq!(
            inf.to_matrix().norm(kind),
            T::Real::INFINITY,
            "dense {kind:?}"
        );
        assert_eq!(SparseMatrix::<T>::zeros(3, 5).norm(kind), T::Real::ZERO);
    }
}

#[test]
fn a_norm_of_a_matrix_holding_a_nan_is_a_nan_for_every_element_type() {
    each_type!(nan_norms_of);
}

/// Whether an element holds a NaN in either part.
fn is_nan<T: Num>(x: T) -> bool {
    x.re().to_f64().is_nan() || x.im().to_f64().is_nan()
}

// The operators that take the element by reference are among those tested.
#[allow(clippy::op_ref)]
fn zero_scaling_of<T: Num>() {
    // The case the verification of p_storage-6 lists: a zero factor emptied the matrix, where
    // the stored infinities and NaN become NaN in IEEE arithmetic, as in the dense matrix.
    let inf = T::real(f64::INFINITY);
    let a = SparseMatrix::from_coo(
        1,
        5,
        &[
            (0, 0, inf),
            (0, 1, T::real(f64::NAN)),
            (0, 2, T::real(1.0)),
            (0, 3, -inf),
        ],
    );
    for z in [T::zero(), -T::zero()] {
        let mut assigned = a.clone();
        assigned *= z;
        let mut assigned_by_reference = a.clone();
        assigned_by_reference *= &z;
        let spellings = [
            ("scale", a.scale(z)),
            ("&a * z", &a * z),
            ("&a * &z", &a * &z),
            ("a * z", a.clone() * z),
            ("a * &z", a.clone() * &z),
            ("a *= z", assigned),
            ("a *= &z", assigned_by_reference),
        ];
        let dense = &a.to_matrix() * z;
        for (name, b) in spellings {
            assert_csr(&b);
            assert_eq!(b.nnz(), 3, "{name}");
            assert_eq!(b.col_indices(), [0, 1, 3], "{name}");
            assert!(b.values().iter().all(|&x| is_nan(x)), "{name}");
            for j in 0..5 {
                let (x, y) = (b[(0, j)], dense[(0, j)]);
                assert!(x == y || (is_nan(x) && is_nan(y)), "{name}: cell {j}");
            }
        }
    }
    // A finite matrix times zero is the empty matrix.
    let mut r = Rng(0x5EED_221C);
    let f: SparseMatrix<T> = random_sparse(&mut r, 6, 7, 20);
    assert_eq!(&f * T::zero(), SparseMatrix::zeros(6, 7));
    assert_eq!(f.scale(-T::zero()), SparseMatrix::zeros(6, 7));
}

#[test]
fn scaling_by_zero_turns_the_stored_infinities_and_nans_into_nans_as_the_dense_matrix_does() {
    each_type!(zero_scaling_of);
}

#[test]
#[should_panic(expected = "the matrix is not square")]
fn rejects_the_trace_of_a_matrix_that_is_not_square_as_the_dense_trace_does() {
    let _ = SparseMatrix::<f64>::identity(3)
        .submatrix(0..3, 0..2)
        .trace();
}

#[test]
#[should_panic(expected = "the shapes don't fit")]
fn rejects_a_product_of_the_wrong_shapes() {
    let a = SparseMatrix::<f64>::zeros(3, 4);
    let b = SparseMatrix::<f64>::zeros(5, 2);
    let _ = &a * &b;
}

#[test]
#[should_panic(expected = "the shapes don't fit")]
fn rejects_a_product_with_a_dense_matrix_of_the_wrong_shape() {
    let a = SparseMatrix::<f64>::zeros(3, 4);
    let _ = &a * &Matrix::<f64>::zeros(3, 4);
}

#[test]
#[should_panic(expected = "the shapes don't fit")]
fn rejects_a_product_of_a_dense_matrix_of_the_wrong_shape() {
    let a = SparseMatrix::<f64>::zeros(3, 4);
    let _ = &Matrix::<f64>::zeros(3, 4) * &a;
}

#[test]
#[should_panic(expected = "same shape")]
fn rejects_a_sum_of_the_wrong_shapes() {
    let a = SparseMatrix::<f64>::zeros(3, 4);
    let b = SparseMatrix::<f64>::zeros(3, 5);
    let _ = &a + &b;
}

#[test]
#[should_panic(expected = "same shape")]
fn rejects_a_difference_of_the_wrong_shapes() {
    let a = SparseMatrix::<f64>::zeros(3, 4);
    let b = SparseMatrix::<f64>::zeros(4, 4);
    let _ = &a - &b;
}

#[test]
#[should_panic(expected = "the vector has the wrong length")]
fn rejects_a_vector_of_the_wrong_length() {
    let a = SparseMatrix::<f64>::zeros(3, 4);
    let _ = a.mul_vec(&[1.0, 2.0]);
}

#[test]
#[should_panic(expected = "the result has the wrong length")]
fn rejects_a_result_of_the_wrong_length() {
    let a = SparseMatrix::<f64>::zeros(3, 4);
    a.mul_vec_into(&[0.0; 4], &mut [0.0; 4]);
}

#[test]
#[should_panic(expected = "the vector has the wrong length")]
fn rejects_a_vector_of_the_wrong_length_for_the_transpose() {
    let a = SparseMatrix::<f64>::zeros(3, 4);
    let _ = a.mul_vec_transpose(&[0.0; 4]);
}

#[test]
#[should_panic(expected = "the result has the wrong length")]
fn rejects_a_result_of_the_wrong_length_for_the_adjoint() {
    let a = SparseMatrix::<f64>::zeros(3, 4);
    a.mul_vec_adjoint_into(&[0.0; 3], &mut [0.0; 3]);
}

#[test]
#[should_panic(expected = "the rows are outside the matrix")]
fn rejects_a_submatrix_of_rows_outside_the_matrix() {
    let _ = SparseMatrix::<f64>::zeros(3, 4).submatrix(1..4, 0..4);
}

#[test]
#[should_panic(expected = "the columns are outside the matrix")]
fn rejects_a_submatrix_of_columns_outside_the_matrix() {
    let _ = SparseMatrix::<f64>::zeros(3, 4).submatrix(0..3, 2..5);
}

#[test]
#[should_panic(expected = "the rows are outside the matrix")]
fn rejects_a_submatrix_whose_range_ends_before_it_begins() {
    #[allow(clippy::reversed_empty_ranges)]
    let _ = SparseMatrix::<f64>::zeros(3, 4).submatrix(2..1, 0..4);
}

#[test]
#[should_panic(expected = "a row is outside the matrix")]
fn rejects_a_selection_of_a_row_outside_the_matrix() {
    let _ = SparseMatrix::<f64>::zeros(3, 4).select(&[0, 3], &[0]);
}

#[test]
#[should_panic(expected = "a column is outside the matrix")]
fn rejects_a_selection_of_a_column_outside_the_matrix() {
    let _ = SparseMatrix::<f64>::zeros(3, 4).select(&[0], &[4]);
}

#[test]
#[should_panic(expected = "the permutation of the rows isn't of the right length")]
fn rejects_a_permutation_of_the_wrong_length() {
    let _ = SparseMatrix::<f64>::zeros(3, 2).permute(&[0, 1], &[0, 1]);
}

#[test]
#[should_panic(expected = "the permutation of the columns is out of range")]
fn rejects_a_permutation_out_of_range() {
    let _ = SparseMatrix::<f64>::zeros(3, 2).permute(&[0, 1, 2], &[0, 2]);
}

#[test]
#[should_panic(expected = "the permutation of the rows repeats an index")]
fn rejects_a_permutation_that_repeats_an_index() {
    let _ = SparseMatrix::<f64>::zeros(3, 2).permute(&[0, 1, 1], &[0, 1]);
}
