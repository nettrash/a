//! Products of matrices of floats and complex numbers, `Matrix * Matrix`, against the textbook
//! sum computed here: every element within a bound on the rounding of a sum of `k` products, for
//! shapes on both sides of every edge of the blocks the product is computed by.

use super::common::{Num, Rng};
use a::math::complex::Complex;
use a::math::matrix::Matrix;

/// Checks `a · b` of an `m × k` and a `k × n` random matrix against the textbook sum in `f64`:
/// each element within `4 (k + 2) ε` of the sum of the absolute values of its products.
fn check<T: Num>(m: usize, k: usize, n: usize, seed: u64) {
    let mut r = Rng(seed | 1);
    let a = Matrix::from_fn(m, k, |_, _| T::random(&mut r));
    let b = Matrix::from_fn(k, n, |_, _| T::random(&mut r));
    let c = &a * &b;
    assert_eq!(c.shape(), (m, n));
    for j in 0..n {
        for i in 0..m {
            let (mut re, mut im, mut size) = (0.0, 0.0, 0.0);
            for p in 0..k {
                let (x, y) = (a[(i, p)].c64(), b[(p, j)].c64());
                let (pr, pi) = (x.re * y.re - x.im * y.im, x.re * y.im + x.im * y.re);
                re += pr;
                im += pi;
                size += pr.abs() + pi.abs();
            }
            let got = c[(i, j)].c64();
            let error = (got.re - re).abs() + (got.im - im).abs();
            let bound = 4.0 * (k as f64 + 2.0) * T::EPS * size;
            assert!(
                error <= bound,
                "{m} × {k} · {k} × {n}: element ({i}, {j}) is {got:?}, not ({re}, {im})"
            );
        }
    }
}

/// [`check`] on every type.
fn check_all(m: usize, k: usize, n: usize) {
    let seed = (m as u64) << 40 ^ (k as u64) << 20 ^ n as u64;
    check::<f64>(m, k, n, seed);
    check::<f32>(m, k, n, seed);
    check::<Complex<f64>>(m, k, n, seed);
    check::<Complex<f32>>(m, k, n, seed);
}

#[test]
fn small_shapes() {
    // Every shape up to 9, and some past the tiles of the direct products.
    let top = if cfg!(miri) { 5 } else { 9 };
    for m in 1..=top {
        for k in 1..=top {
            for n in 1..=top {
                check_all(m, k, n);
            }
        }
    }
    for &(m, k, n) in &[
        (16, 16, 16),
        (17, 3, 13),
        (33, 9, 5),
        (5, 31, 33),
        (64, 64, 64),
    ] {
        if !cfg!(miri) || m * k * n <= 4096 {
            check_all(m, k, n);
        }
    }
}

#[test]
fn blocked_edges() {
    // Large enough to be packed in blocks; rows on both sides of a tile's eight, columns of every
    // remainder of the panels' widths (2 to 12), and depths across a block's.
    let shapes: &[(usize, usize, usize)] = if cfg!(miri) {
        &[
            (50, 257, 1),
            (25, 600, 1),
            (200, 70, 1),
            (1, 9, 2100),
            (9, 9, 700),
        ]
    } else {
        &[
            (100, 100, 20),
            (127, 129, 31),
            (129, 255, 50),
            (257, 257, 7),
            (8, 600, 63),
            (300, 70, 300),
            (65, 513, 65),
            (130, 30, 1030),
            (4, 9, 2050),
        ]
    };
    for &(m, k, n) in shapes {
        check_all(m, k, n);
    }
    if !cfg!(miri) {
        // Every remainder of the columns, by the widest panels of each type.
        for n in 40..=52 {
            check_all(72, 90, n);
        }
    }
}

#[test]
fn empty_and_zero_depth() {
    for &(m, k, n) in &[(0, 3, 4), (3, 0, 4), (3, 4, 0), (0, 0, 0), (5, 0, 7)] {
        let a = Matrix::<f64>::from_fn(m, k, |i, j| (i + 2 * j) as f64);
        let b = Matrix::<f64>::from_fn(k, n, |i, j| (3 * i + j) as f64);
        assert_eq!(&a * &b, Matrix::zeros(m, n));
        let a = a.map(|x| Complex::new(*x as f32, 1.0));
        let b = b.map(|x| Complex::new(1.0, *x as f32));
        assert_eq!(&a * &b, Matrix::zeros(m, n));
    }
}

#[test]
fn identities_and_assignment() {
    let mut r = Rng(7);
    let a = Matrix::from_fn(37, 41, |_, _| r.unit());
    assert_eq!(&a * &Matrix::identity(41), a);
    assert_eq!(&Matrix::identity(37) * &a, a);
    let mut b = Matrix::from_fn(37, 37, |_, _| r.unit());
    let product = &b * &Matrix::identity(37);
    b *= Matrix::identity(37);
    assert_eq!(b, product);
    // Infinities and NaNs propagate as in the textbook sums.
    let x = Matrix::from_rows(&[[f64::INFINITY, 1.0], [2.0, f64::NAN]]);
    let y = Matrix::from_rows(&[[1.0, 0.0], [0.0, 1.0]]);
    let z = &x * &y;
    assert_eq!(z[(0, 0)], f64::INFINITY);
    assert!(z[(1, 1)].is_nan());
}
