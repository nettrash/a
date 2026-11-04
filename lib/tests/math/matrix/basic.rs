//! Constructors, access, arithmetic elementwise, transposes, row and column operations, traces,
//! powers and formatting.

use a::math::complex::Complex;
use a::math::matrix::Matrix;

fn m(rows: &[[f64; 3]]) -> Matrix<f64> {
    Matrix::from_rows(rows)
}

#[test]
fn constructors_and_access() {
    let a = m(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    assert_eq!((a.rows(), a.cols(), a.shape()), (2, 3, (2, 3)));
    assert!(!a.is_square() && !a.is_empty());
    // Held column after column.
    assert_eq!(a.as_slice(), [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    assert_eq!(a.column(1), [2.0, 5.0]);
    assert_eq!(a.row(1).copied().collect::<Vec<_>>(), [4.0, 5.0, 6.0]);
    assert_eq!(
        (a[(0, 2)], a.get(1, 0), a.get(2, 0)),
        (3.0, Some(&4.0), None)
    );
    assert_eq!(
        Matrix::from_columns(&[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]),
        a
    );
    assert_eq!(
        Matrix::from_row_major(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        a
    );
    assert_eq!(Matrix::from_column_major(2, 3, a.as_slice().to_vec()), a);
    assert_eq!(Matrix::from_fn(2, 3, |i, j| (3 * i + j + 1) as f64), a);
    assert_eq!(a.diagonal(), [1.0, 5.0]);
    assert_eq!(a.map(|x| x * 2.0)[(1, 2)], 12.0);
    let mut b = a.clone();
    b[(1, 1)] = -1.0;
    *b.get_mut(0, 0).unwrap() = 7.0;
    b.column_mut(2)[0] = 9.0;
    assert_eq!(b.into_vec(), [7.0, 4.0, 2.0, -1.0, 9.0, 6.0]);
    assert_eq!(
        Matrix::<f64>::identity(2),
        Matrix::from_rows(&[[1.0, 0.0], [0.0, 1.0]])
    );
    assert_eq!(Matrix::<f64>::zeros(1, 2), Matrix::from_rows(&[[0.0, 0.0]]));
    assert_eq!(
        Matrix::from_diagonal(&[2.0, 3.0]),
        Matrix::from_rows(&[[2.0, 0.0], [0.0, 3.0]])
    );
    let empty: Matrix<f64> = Matrix::from_rows::<[f64; 0]>(&[]);
    assert_eq!((empty.shape(), empty.is_empty()), ((0, 0), true));
    assert_eq!(Matrix::<f64>::default(), empty);
}

#[test]
fn arithmetic() {
    let a = m(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let b = m(&[[0.5, -1.0, 2.0], [1.0, 0.0, -3.0]]);
    let sum = m(&[[1.5, 1.0, 5.0], [5.0, 5.0, 3.0]]);
    let difference = m(&[[0.5, 3.0, 1.0], [3.0, 5.0, 9.0]]);
    assert_eq!(&a + &b, sum);
    assert_eq!(a.clone() + &b, sum);
    assert_eq!(&a + b.clone(), sum);
    assert_eq!(a.clone() + b.clone(), sum);
    assert_eq!(&a - &b, difference);
    assert_eq!(&a - b.clone(), difference);
    let mut c = a.clone();
    c += &b;
    c -= b.clone();
    assert_eq!(c, a);
    assert_eq!(-&a, a.map(|x| -x));
    assert_eq!(&a * 2.0, a.map(|x| 2.0 * x));
    assert_eq!(2.0 * &a, a.map(|x| 2.0 * x));
    assert_eq!(&a / 2.0, a.map(|x| x / 2.0));
    let mut d = a.clone();
    d *= 3.0;
    d /= 3.0;
    assert_eq!(d, a);
    assert_eq!(a.hadamard(&b), m(&[[0.5, -2.0, 6.0], [4.0, 0.0, -18.0]]));
    let p = &a * &b.transpose();
    assert_eq!(p, Matrix::from_rows(&[[4.5, -8.0], [9.0, -14.0]]));
    let mut q = a.clone();
    q *= &b.transpose();
    assert_eq!(q, p);
}

#[test]
fn transposes_and_row_operations() {
    let z = |re, im| Complex::new(re, im);
    let a = Matrix::from_rows(&[[z(1.0, 2.0), z(3.0, -1.0)], [z(0.0, 1.0), z(5.0, 0.0)]]);
    assert_eq!(a.transpose()[(0, 1)], z(0.0, 1.0));
    assert_eq!(a.conjugate_transpose()[(0, 1)], z(0.0, -1.0));
    assert_eq!(a.conjugate_transpose()[(0, 0)], z(1.0, -2.0));
    let r = m(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    assert_eq!(r.transpose().transpose(), r);
    assert_eq!(r.conjugate_transpose(), r.transpose());
    let mut s = r.clone();
    s.swap_rows(0, 1);
    assert_eq!(s, m(&[[4.0, 5.0, 6.0], [1.0, 2.0, 3.0]]));
    s.scale_row(1, &2.0);
    assert_eq!(s, m(&[[4.0, 5.0, 6.0], [2.0, 4.0, 6.0]]));
    s.add_scaled_row(0, 1, &-2.0);
    assert_eq!(s, m(&[[0.0, -3.0, -6.0], [2.0, 4.0, 6.0]]));
    let mut c = r.clone();
    c.swap_columns(2, 0);
    assert_eq!(c, m(&[[3.0, 2.0, 1.0], [6.0, 5.0, 4.0]]));
    c.scale_column(1, &0.5);
    c.add_scaled_column(2, 0, &-1.0);
    assert_eq!(c, m(&[[3.0, 1.0, -2.0], [6.0, 2.5, -2.0]]));
    let sq = Matrix::from_rows(&[[1.0, 1.0], [1.0, 0.0]]);
    assert_eq!(sq.trace(), 1.0);
    // Fibonacci's numbers.
    assert_eq!(sq.pow(10), Matrix::from_rows(&[[89.0, 55.0], [55.0, 34.0]]));
    assert_eq!(sq.pow(0), Matrix::identity(2));
}

#[test]
fn formatting() {
    let a = Matrix::from_rows(&[[1.0, -2.5], [10.0, 0.25]]);
    assert_eq!(format!("{a}"), " 1  -2.5\n10  0.25");
    assert_eq!(format!("{a:.1}"), " 1.0  -2.5\n10.0   0.2");
    assert_eq!(format!("{a:?}"), "[[1.0, -2.5], [10.0, 0.25]]");
    let c = Matrix::from_rows(&[[Complex::new(1.0, -1.0)]]);
    assert_eq!(format!("{c}"), "1-1i");
}

#[test]
#[should_panic(expected = "the matrices are not of the same shape")]
fn adding_different_shapes_panics() {
    let _ = &Matrix::<f64>::zeros(2, 2) + &Matrix::zeros(2, 3);
}

#[test]
#[should_panic(
    expected = "the columns of the left matrix are not as many as the rows of the right one"
)]
fn multiplying_mismatched_shapes_panics() {
    let _ = &Matrix::<f64>::zeros(2, 3) * &Matrix::zeros(2, 3);
}

#[test]
#[should_panic(expected = "the rows are not all of the same length")]
fn ragged_rows_panic() {
    let _ = Matrix::from_rows(&[vec![1.0, 2.0], vec![3.0]]);
}

#[test]
#[should_panic(expected = "the index is outside the matrix")]
fn indexing_outside_panics() {
    let _ = Matrix::<f64>::zeros(2, 2)[(0, 2)];
}

#[test]
#[should_panic(expected = "the matrix is not square")]
fn trace_of_a_rectangle_panics() {
    let _ = Matrix::<f64>::zeros(2, 3).trace();
}
