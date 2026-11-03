//! LU, Cholesky and QR decompositions, checked by what defines them.

use super::common::{
    Num, Rng, assert_close, assert_orthonormal, each_type, frobenius, positive_definite, random,
    size, well_conditioned,
};
use a::math::complex::{Complex, Float};
use a::math::matrix::Matrix;

/// Sizes around the recursions' and the panels' widths.
fn sizes() -> Vec<usize> {
    if cfg!(miri) {
        vec![0, 1, 2, 9, 17]
    } else {
        vec![0, 1, 2, 3, 7, 8, 9, 16, 17, 31, 32, 33, 64, 65, 100]
    }
}

fn lu_of<T: Num>() {
    let mut r = Rng(0x5EED_0001);
    for n in sizes() {
        for m in [n, n + 3, n / 2] {
            let a = random::<T>(&mut r, m, n);
            let lu = a.lu();
            let (l, u) = (lu.l(), lu.u());
            let k = m.min(n);
            assert_eq!((l.shape(), u.shape()), ((m, k), (k, n)));
            for i in 0..m {
                for j in 0..k {
                    if j > i {
                        assert_eq!(l[(i, j)], T::real(0.0));
                    }
                    if i == j {
                        assert_eq!(l[(i, j)], T::real(1.0));
                    }
                    // Partial pivoting by `|re| + |im|`: no multiplier above 1, or √2 for
                    // complex elements.
                    let bound = if core::any::type_name::<T>().contains("Complex") {
                        core::f64::consts::SQRT_2
                    } else {
                        1.0
                    };
                    assert!(l[(i, j)].c64().abs() <= bound * (1.0 + 4.0 * T::EPS));
                }
            }
            for i in 0..k {
                for j in 0..i {
                    assert_eq!(u[(i, j)], T::real(0.0));
                }
            }
            assert_close(&(&lu.p() * &a), &(&l * &u), 8.0 * (n + 1) as f64);
        }
    }
}

#[test]
fn lu() {
    each_type!(lu_of);
}

fn solve_of<T: Num>() {
    let mut r = Rng(0x5EED_0002);
    for n in sizes() {
        let a = well_conditioned::<T>(&mut r, n);
        let b = random::<T>(&mut r, n, 3);
        let x = a.solve(&b).unwrap();
        assert_close(&(&a * &x), &b, 16.0 * (n + 1) as f64);
        let inv = a.inverse().unwrap();
        assert_close(&(&a * &inv), &Matrix::identity(n), 16.0 * (n + 1) as f64);
        // `A⁻²·A² = I`.
        let p = a.powi(-2).unwrap();
        assert_close(
            &(&p * &a.pow(2)),
            &Matrix::identity(n),
            64.0 * (n + 1) as f64,
        );
    }
}

#[test]
fn solve_and_inverse() {
    each_type!(solve_of);
}

fn det_of<T: Num>() {
    let mut r = Rng(0x5EED_0003);
    for n in sizes() {
        // Scaled to determinants near 1, which `f32` holds for every size here.
        let s = T::real(1.0 / (n + 1) as f64);
        let a = well_conditioned::<T>(&mut r, n) * s;
        let b = well_conditioned::<T>(&mut r, n) * s;
        // `det(A·B) = det(A)·det(B)`, for matrices whose determinants are well conditioned.
        let (x, y) = ((&a * &b).det().c64(), (a.det() * b.det()).c64());
        assert!(
            (x - y).abs() <= 1e3 * (n + 1) as f64 * T::EPS * (y.abs() + T::EPS),
            "{n}"
        );
        // A triangular matrix's determinant is its diagonal's product.
        let t = Matrix::from_fn(n, n, |i, j| if i <= j { a[(i, j)] } else { T::real(0.0) });
        let want: Complex<f64> = t.diagonal().iter().map(|x| x.c64()).product();
        let got = t.det().c64();
        assert!(
            (got - want).abs() <= 8.0 * (n + 1) as f64 * T::EPS * want.abs(),
            "{n}"
        );
    }
    assert_eq!(Matrix::<T>::zeros(0, 0).det(), T::real(1.0));
}

#[test]
fn determinants() {
    each_type!(det_of);
}

fn singular_of<T: Num>() {
    let mut r = Rng(0x5EED_0004);
    let mut a = random::<T>(&mut r, 5, 5);
    // A zero column, and a column that repeats another.
    for i in 0..5 {
        a[(i, 2)] = T::real(0.0);
    }
    assert_eq!(a.det(), T::real(0.0));
    assert!(a.inverse().is_none());
    assert!(a.solve(&random::<T>(&mut r, 5, 1)).is_none());
    assert!(a.powi(-1).is_none());
    assert!(a.lu().is_singular());
    let lu = a.lu();
    assert_close(&(&lu.p() * &a), &(&lu.l() * &lu.u()), 50.0);
}

#[test]
fn singular_matrices() {
    each_type!(singular_of);
}

fn cholesky_of<T: Num>() {
    let mut r = Rng(0x5EED_0005);
    for n in sizes() {
        let a = positive_definite::<T>(&mut r, n);
        let c = a.cholesky().unwrap();
        let l = c.l();
        for i in 0..n {
            assert_eq!(l[(i, i)].c64().im, 0.0);
            assert!(l[(i, i)].c64().re > 0.0);
            for j in i + 1..n {
                assert_eq!(l[(i, j)], T::real(0.0));
            }
        }
        let tol = 8.0 * (n + 1) as f64;
        assert_close(&(&l * &l.conjugate_transpose()), &a, tol);
        let b = random::<T>(&mut r, n, 2);
        assert_close(&(&a * &c.solve(&b)), &b, 4.0 * tol);
        assert_close(&(&a * &c.inverse()), &Matrix::identity(n), 4.0 * tol);
        // As the LU decomposition's, unless both overflow, as `f32` does from about 30.
        let (d, got) = (a.det().c64(), c.det().to_f64());
        if d.is_finite() {
            assert!(
                (got - d.re).abs() <= 1e3 * (n + 1) as f64 * T::EPS * d.abs(),
                "{n}"
            );
        } else {
            assert!(got.is_infinite());
        }
    }
    // Indefinite, and only the lower triangle read.
    let mut a = positive_definite::<T>(&mut r, 6);
    a[(4, 4)] = T::real(-1.0);
    assert!(a.cholesky().is_none());
    let mut a = positive_definite::<T>(&mut r, 6);
    let l = a.cholesky().unwrap().l();
    a[(0, 5)] = T::real(1e6);
    assert_eq!(a.cholesky().unwrap().l(), l);
    let mut nan = positive_definite::<T>(&mut r, 3);
    nan[(1, 1)] = T::real(f64::NAN);
    assert!(nan.cholesky().is_none());
}

#[test]
fn cholesky() {
    each_type!(cholesky_of);
}

fn qr_of<T: Num>() {
    let mut r = Rng(0x5EED_0006);
    for n in sizes() {
        for m in [n, n + 5, n / 2, 2 * n + 1] {
            let a = random::<T>(&mut r, m, n);
            let qr = a.qr();
            let (q, rr) = (qr.q(), qr.r());
            let k = m.min(n);
            assert_eq!((q.shape(), rr.shape()), ((m, k), (k, n)));
            let tol = 8.0 * (m + n + 1) as f64;
            assert_orthonormal(&q, tol);
            assert_close(&(&q * &rr), &a, tol);
            for i in 0..k {
                assert_eq!(rr[(i, i)].c64().im, 0.0);
                for j in 0..i {
                    assert_eq!(rr[(i, j)], T::real(0.0));
                }
            }
            let full = qr.q_full();
            assert_eq!(full.shape(), (m, m));
            assert_orthonormal(&full, tol);
            assert_close(&Matrix::from_fn(m, k, |i, j| full[(i, j)]), &q, tol);
        }
    }
}

#[test]
fn qr() {
    each_type!(qr_of);
}

fn least_squares_of<T: Num>() {
    let mut r = Rng(0x5EED_0007);
    for n in sizes() {
        let m = 2 * n + 3;
        let a = random::<T>(&mut r, m, n);
        let b = random::<T>(&mut r, m, 2);
        let x = a.qr().solve(&b).unwrap();
        // The residual is orthogonal to the columns: `A*·(B - A·X) = 0`.
        let residual = &b - &(&a * &x);
        let g = &a.conjugate_transpose() * &residual;
        let scale = frobenius(&a) * frobenius(&b) + 1.0;
        assert!(
            frobenius(&g) <= 64.0 * (m + 1) as f64 * T::EPS * scale,
            "{n}"
        );
    }
    // Dependent columns.
    let mut a = random::<T>(&mut r, 6, 3);
    for i in 0..6 {
        a[(i, 1)] = T::real(0.0);
    }
    assert!(a.qr().solve(&random::<T>(&mut r, 6, 1)).is_none());
}

#[test]
fn least_squares() {
    each_type!(least_squares_of);
}

#[test]
fn exact_small_cases() {
    // Pivots that are powers of two keep everything exact.
    let a = Matrix::from_rows(&[[0.0, 2.0, 1.0], [4.0, 1.0, 0.0], [2.0, 0.5, 3.0]]);
    let lu = a.lu();
    assert_eq!(lu.permutation(), [1, 0, 2]);
    assert_eq!(&lu.p() * &a, &lu.l() * &lu.u());
    assert_eq!(a.det(), -24.0);
    let c = Matrix::from_rows(&[[4.0, 2.0, -2.0], [2.0, 10.0, 2.0], [-2.0, 2.0, 6.0]]);
    let l = c.cholesky().unwrap().l();
    assert_eq!(
        l,
        Matrix::from_rows(&[[2.0, 0.0, 0.0], [1.0, 3.0, 0.0], [-1.0, 1.0, 2.0]])
    );
    let _ = size(1, 1);
}
