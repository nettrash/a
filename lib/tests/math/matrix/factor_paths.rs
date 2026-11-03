//! The paths of the LU and Cholesky decompositions, the inverse and the triangular solves on
//! both sides of each of their thresholds: unrolled tiny orders, unblocked and recursive
//! factorizations, leaves of the triangular solves on the stack or allocated, and the fallbacks
//! for pivots whose reciprocals overflow, singular matrices and NaNs.

use super::common::{
    Num, Rng, assert_close, each_type, positive_definite, random, well_conditioned,
};
use a::math::complex::Complex;
use a::math::matrix::Matrix;

/// Orders on both sides of the thresholds: the unrolled orders up to 16, the unblocked LU up to
/// 24, the leaves of 16 and 32, the recursion's splits, and the working memory on the stack up to
/// 1024 elements.
fn orders() -> Vec<usize> {
    if cfg!(miri) {
        vec![0, 1, 2, 3, 4, 5, 8, 9, 12, 13, 16, 17, 25, 33]
    } else {
        let mut v: Vec<usize> = (0..=34).collect();
        v.extend([40, 47, 48, 49, 63, 64, 65, 96, 129]);
        v
    }
}

/// The LU decomposition by plain Gaussian elimination, element by element: the pivot of a
/// column is its first element of the largest `|re| + |im|`, as the scan from the diagonal with
/// `>` finds. Returns the permutation.
fn reference_permutation<T: Num>(a: &Matrix<T>) -> Vec<usize> {
    let (m, n) = a.shape();
    let mut lu: Vec<Complex<f64>> = a.as_slice().iter().map(|x| x.c64()).collect();
    let mut perm: Vec<usize> = (0..m).collect();
    let abs1 = |x: Complex<f64>| x.re.abs() + x.im.abs();
    for j in 0..m.min(n) {
        let mut p = j;
        let mut best = abs1(lu[j + j * m]);
        for i in j + 1..m {
            let v = abs1(lu[i + j * m]);
            if v > best {
                best = v;
                p = i;
            }
        }
        if p != j {
            perm.swap(j, p);
            for c in 0..n {
                lu.swap(j + c * m, p + c * m);
            }
        }
        let d = lu[j + j * m];
        if d == Complex::new(0.0, 0.0) {
            continue;
        }
        for i in j + 1..m {
            lu[i + j * m] /= d;
        }
        for c in j + 1..n {
            let u = lu[j + c * m];
            for i in j + 1..m {
                let l = lu[i + j * m];
                lu[i + c * m] -= l * u;
            }
        }
    }
    perm
}

fn lu_paths_of<T: Num>() {
    let mut r = Rng(0x5EED_0101);
    for n in orders() {
        for m in [n, n + 5, n / 2, 2 * n + 1] {
            let a = random::<T>(&mut r, m, n);
            let lu = a.lu();
            let (l, u) = (lu.l(), lu.u());
            assert_close(&(&lu.p() * &a), &(&l * &u), 16.0 * (m + n + 1) as f64);
            // The pivots of the elimination element by element: random elements leave no ties
            // that rounding could break the other way.
            if T::EPS < 1e-10 {
                assert_eq!(lu.permutation(), reference_permutation(&a), "{m} × {n}");
            }
            for i in 0..u.rows() {
                for j in 0..i {
                    assert_eq!(u[(i, j)], T::real(0.0));
                }
            }
        }
    }
}

#[test]
fn lu_paths() {
    each_type!(lu_paths_of);
}

/// Ties choose the first element: columns of ones swap nothing.
fn lu_ties_of<T: Num>() {
    for n in orders() {
        let a = Matrix::from_fn(n, n, |i, j| {
            if j == 0 || i == j {
                T::real(1.0)
            } else {
                T::real(0.0)
            }
        });
        let lu = a.lu();
        assert_eq!(lu.permutation(), (0..n).collect::<Vec<_>>(), "{n}");
        assert_eq!(a.det(), T::real(1.0), "{n}");
    }
}

#[test]
fn lu_ties() {
    each_type!(lu_ties_of);
}

fn inverse_paths_of<T: Num>() {
    let mut r = Rng(0x5EED_0102);
    for n in orders() {
        let a = well_conditioned::<T>(&mut r, n);
        let inv = a.inverse().unwrap();
        let tol = 32.0 * (n + 1) as f64;
        assert_close(&(&a * &inv), &Matrix::identity(n), tol);
        assert_close(&(&inv * &a), &Matrix::identity(n), tol);
        // From the decomposition as it is kept, by the cycles of the permutation.
        assert_close(&a.lu().inverse().unwrap(), &inv, tol);
        for cols in [1, 2, 3, 17, 40] {
            let b = random::<T>(&mut r, n, cols);
            let x = a.solve(&b).unwrap();
            assert_close(&(&a * &x), &b, tol);
        }
    }
}

#[test]
fn inverse_paths() {
    each_type!(inverse_paths_of);
}

fn singular_paths_of<T: Num>() {
    let mut r = Rng(0x5EED_0103);
    for n in orders().into_iter().filter(|&n| n > 0) {
        for k in [0, n / 2, n - 1] {
            // A zero column, and a zero row: an exact zero pivot, at the step of the column or
            // at the last. Scaled so that the products of the other pivots do not overflow.
            let mut a = well_conditioned::<T>(&mut r, n) * T::real(1.0 / (n + 1) as f64);
            let mut b = a.clone();
            for i in 0..n {
                a[(i, k)] = T::real(0.0);
                b[(k, i)] = T::real(0.0);
            }
            for a in [a, b] {
                assert!(a.inverse().is_none(), "{n} {k}");
                assert!(a.solve(&random::<T>(&mut r, n, 2)).is_none(), "{n} {k}");
                assert_eq!(a.det(), T::real(0.0), "{n} {k}");
                let lu = a.lu();
                assert!(lu.is_singular());
                assert!(lu.inverse().is_none());
                assert_close(&(&lu.p() * &a), &(&lu.l() * &lu.u()), 16.0 * (n + 1) as f64);
            }
        }
    }
}

#[test]
fn singular_paths() {
    each_type!(singular_paths_of);
}

/// A pivot so small that its reciprocal overflows: the decomposition and the triangular solves
/// divide instead, and the solution stays finite and exact where the products by the reciprocal
/// would give infinities and NaNs.
fn tiny_pivots_of<T: Num>() {
    // Subnormal, its reciprocal past the largest float.
    let tiny = if T::EPS < 1e-10 { 1e-310 } else { 1e-40 };
    for n in [1, 2, 3, 9, 13, 16, 17, 33] {
        if cfg!(miri) && n > 17 {
            continue;
        }
        for k in [0, n / 2, n - 1] {
            // Upper triangular, small integers above the diagonal but in row `k`, whose pivot is
            // `tiny`, and powers of two on it: the decomposition is the matrix itself, and
            // `U·1 = b` is solved exactly, the reciprocals of the other pivots being exact.
            let u = Matrix::from_fn(n, n, |i, j| {
                if i == j {
                    T::real(if i == k { tiny } else { [1.0, 2.0, 4.0][i % 3] })
                } else if i < j && i != k {
                    T::real(((i + 2 * j) % 5) as f64 - 2.0)
                } else {
                    T::real(0.0)
                }
            });
            let b = Matrix::from_fn(n, 1, |i, _| (0..n).fold(T::real(0.0), |s, j| s + u[(i, j)]));
            let ones = Matrix::from_fn(n, 1, |_, _| T::real(1.0));
            assert_eq!(u.solve(&b).unwrap(), ones, "{n} {k}");
            assert_eq!(u.lu().solve(&b).unwrap(), ones, "{n} {k}");
            // Its transpose, lower triangular: the multipliers below the tiny pivot are zeros
            // divided by it, not zeros times an infinity.
            let lu = u.transpose().lu();
            assert!(
                lu.l().as_slice().iter().all(|x| x.c64().is_finite()),
                "{n} {k}"
            );
        }
    }
}

#[test]
fn tiny_pivots() {
    each_type!(tiny_pivots_of);
}

/// NaNs and infinities go through without a panic; a NaN first in its column is its pivot, as
/// the scan from it finds.
fn non_finite_of<T: Num>() {
    let mut r = Rng(0x5EED_0104);
    for n in orders().into_iter().filter(|&n| n > 0) {
        for bad in [f64::NAN, f64::INFINITY] {
            let mut a = well_conditioned::<T>(&mut r, n);
            a[(0, 0)] = T::real(bad);
            let lu = a.lu();
            if bad.is_nan() {
                assert_eq!(lu.permutation()[0], 0, "{n}");
            }
            let _ = a.inverse();
            let _ = a.solve(&random::<T>(&mut r, n, 3));
            let _ = a.det();
            let mut b = well_conditioned::<T>(&mut r, n);
            b[(n - 1, n / 2)] = T::real(bad);
            let _ = b.lu();
            let _ = b.inverse();
        }
    }
}

#[test]
fn non_finite() {
    each_type!(non_finite_of);
}

fn cholesky_paths_of<T: Num>() {
    let mut r = Rng(0x5EED_0105);
    for n in orders() {
        let a = positive_definite::<T>(&mut r, n);
        let c = a.cholesky().unwrap();
        let l = c.l();
        for i in 0..n {
            assert!(l[(i, i)].c64().re > 0.0 && l[(i, i)].c64().im == 0.0);
            for j in i + 1..n {
                assert_eq!(l[(i, j)], T::real(0.0));
            }
        }
        let tol = 16.0 * (n + 1) as f64;
        assert_close(&(&l * &l.conjugate_transpose()), &a, tol);
        let b = random::<T>(&mut r, n, 3);
        assert_close(&(&a * &c.solve(&b)), &b, 8.0 * tol);
        // Only the lower triangle is read: NaNs above it change nothing.
        let mut nan = a.clone();
        for j in 0..n {
            for i in 0..j {
                nan[(i, j)] = T::real(f64::NAN);
            }
        }
        assert_eq!(nan.cholesky().unwrap().l(), l, "{n}");
        // Not positive definite at a column of each part of the recursion.
        for k in [0, n / 2, n.saturating_sub(1)] {
            if n == 0 {
                break;
            }
            let mut bad = a.clone();
            bad[(k, k)] = T::real(-1.0);
            assert!(bad.cholesky().is_none(), "{n} {k}");
            bad[(k, k)] = T::real(f64::NAN);
            assert!(bad.cholesky().is_none(), "{n} {k}");
        }
    }
}

#[test]
fn cholesky_paths() {
    each_type!(cholesky_paths_of);
}
