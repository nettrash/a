//! Schur decompositions and eigenvalues, checked by what defines them.

use super::common::{Num, Rng, assert_close, assert_orthonormal, each_type, random};
use a::math::complex::Complex;
use a::math::matrix::Matrix;

/// Sizes around the exceptional shifts' period and a few larger ones.
fn sizes() -> Vec<usize> {
    if cfg!(miri) {
        vec![0, 1, 2, 3, 6]
    } else {
        vec![0, 1, 2, 3, 4, 5, 8, 11, 20, 33, 64, 100]
    }
}

/// Checks the Schur decomposition of `a`.
fn check_schur<T: Num>(a: &Matrix<T>) {
    let n = a.rows();
    let s = a.schur();
    let (q, t) = (s.q(), s.t());
    let tol = 32.0 * (n + 1) as f64;
    assert_orthonormal(&q, tol);
    assert_close(&(&(&q * &t) * &q.conjugate_transpose()), a, tol);
    let zero = T::real(0.0);
    let complex = core::any::type_name::<T>().contains("Complex");
    for j in 0..n {
        for i in j + 1..n {
            if complex || i > j + 1 {
                assert_eq!(t[(i, j)], zero, "{i} {j}");
            }
        }
    }
    let values = s.eigenvalues();
    assert_eq!(values.len(), n);
    let mut i = 0;
    while i < n {
        let d = t[(i, i)].c64();
        if !complex && i + 1 < n && t[(i + 1, i)] != zero {
            // A standard 2 × 2 block: equal diagonal, `b·c < 0`, eigenvalues `a ± i√(-bc)`.
            let (b, c) = (t[(i, i + 1)].c64().re, t[(i + 1, i)].c64().re);
            assert_eq!(t[(i + 1, i + 1)].c64(), d);
            assert!(b * c < 0.0);
            let im = b.abs().sqrt() * c.abs().sqrt();
            let (w1, w2) = (to_c64::<T>(values[i]), to_c64::<T>(values[i + 1]));
            assert!((w1 - Complex::new(d.re, im)).abs() <= 8.0 * T::EPS * (d.abs() + im));
            assert!((w2 - Complex::new(d.re, -im)).abs() <= 8.0 * T::EPS * (d.abs() + im));
            i += 2;
        } else {
            assert_eq!(to_c64::<T>(values[i]), d);
            i += 1;
        }
    }
    // Their sum is the trace.
    let sum: Complex<f64> = values.iter().map(|&w| to_c64::<T>(w)).sum();
    let trace = a.trace().c64();
    let scale: f64 =
        (0..n).map(|i| a[(i, i)].c64().abs()).sum::<f64>() + super::common::frobenius(a);
    assert!((sum - trace).abs() <= tol * T::EPS * (scale + 1.0));
    // And the eigenvalues without `Q` are the same ones.
    let mut alone: Vec<Complex<f64>> = a.eigenvalues().iter().map(|&w| to_c64::<T>(w)).collect();
    let mut with: Vec<Complex<f64>> = values.iter().map(|&w| to_c64::<T>(w)).collect();
    let key =
        |w: &Complex<f64>, v: &Complex<f64>| w.re.total_cmp(&v.re).then(w.im.total_cmp(&v.im));
    alone.sort_by(key);
    with.sort_by(key);
    for (x, y) in alone.iter().zip(&with) {
        assert!((x - y).abs() <= tol * T::EPS * (scale + 1.0), "{x} {y}");
    }
}

/// An eigenvalue of the type's precision as a `Complex<f64>`.
fn to_c64<T: Num>(w: Complex<T::Real>) -> Complex<f64> {
    use a::math::complex::Float;
    Complex::new(w.re.to_f64(), w.im.to_f64())
}

fn schur_of<T: Num>() {
    let mut r = Rng(0x5EED_0101);
    for n in sizes() {
        check_schur(&random::<T>(&mut r, n, n));
    }
    // Structured matrices: zero, diagonal, triangular, a Jordan block, a permutation.
    for n in [1, 4, 9] {
        check_schur(&Matrix::<T>::zeros(n, n));
        check_schur(&Matrix::from_fn(n, n, |i, j| {
            if i == j {
                T::real(i as f64 - 2.0)
            } else {
                T::real(0.0)
            }
        }));
        check_schur(&Matrix::from_fn(n, n, |i, j| {
            if i <= j {
                T::real((i + 2 * j) as f64)
            } else {
                T::real(0.0)
            }
        }));
        check_schur(&Matrix::from_fn(n, n, |i, j| {
            if i == j || i + 1 == j {
                T::real(1.0)
            } else {
                T::real(0.0)
            }
        }));
        check_schur(&Matrix::from_fn(n, n, |i, j| {
            if (i + 1) % n == j {
                T::real(1.0)
            } else {
                T::real(0.0)
            }
        }));
    }
}

#[test]
fn schur() {
    each_type!(schur_of);
}

#[test]
fn known_eigenvalues() {
    // The companion matrix of (x - 1)(x - 2)…(x - 6): eigenvalues 1 to 6.
    let coefficients = [720.0, -1764.0, 1624.0, -735.0, 175.0, -21.0];
    let c: Matrix<f64> = Matrix::from_fn(6, 6, |i, j| {
        if j == 5 {
            -coefficients[i]
        } else if i == j + 1 {
            1.0
        } else {
            0.0
        }
    });
    let mut w = c.eigenvalues();
    w.sort_by(|a, b| a.re.total_cmp(&b.re));
    for (k, w) in w.iter().enumerate() {
        assert!(
            (w.re - (k + 1) as f64).abs() < 1e-9 && w.im.abs() < 1e-9,
            "{w}"
        );
    }
    // A rotation by θ: e^(±iθ).
    let (s, c) = 0.3f64.sin_cos();
    let rot: Matrix<f64> = Matrix::from_rows(&[[c, -s], [s, c]]);
    let mut w = rot.eigenvalues();
    w.sort_by(|a, b| a.im.total_cmp(&b.im));
    assert!((w[0] - Complex::new(c, -s)).abs() < 1e-15);
    assert!((w[1] - Complex::new(c, s)).abs() < 1e-15);
    // A complex triangular matrix's diagonal.
    let t: Matrix<Complex<f64>> = Matrix::from_rows(&[
        [Complex::new(1.0, 2.0), Complex::new(3.0, 0.0)],
        [Complex::new(0.0, 0.0), Complex::new(-1.0, 0.5)],
    ]);
    let mut w = t.eigenvalues();
    w.sort_by(|a, b| a.re.total_cmp(&b.re));
    assert_eq!(w, [Complex::new(-1.0, 0.5), Complex::new(1.0, 2.0)]);
    // Not a number.
    let nan: Matrix<f64> = Matrix::from_rows(&[[1.0, f64::NAN], [0.5, 2.0]]);
    assert!(nan.eigenvalues().iter().all(|w| w.is_nan()));
}

/// The matrix in `Complex<f64>`.
fn c64<T: Num>(a: &Matrix<T>) -> Matrix<Complex<f64>> {
    a.map(|x| x.c64())
}

/// A complex matrix of the type's precision in `Complex<f64>`.
fn widen<T: Num>(a: &Matrix<Complex<T::Real>>) -> Matrix<Complex<f64>> {
    a.map(|&w| to_c64::<T>(w))
}

fn eigen_of<T: Num>() {
    let mut r = Rng(0x5EED_0102);
    for n in sizes() {
        let a = random::<T>(&mut r, n, n);
        let e = a.eigen();
        let (v, w) = (widen::<T>(e.vectors()), e.values());
        let ac = c64(&a);
        let scale = super::common::frobenius(&a) + 1.0;
        for (j, &wj) in w.iter().enumerate() {
            let x = Matrix::from_columns(&[v.column(j)]);
            let norm: f64 = x
                .as_slice()
                .iter()
                .map(|z| z.norm_sqr())
                .sum::<f64>()
                .sqrt();
            assert!(
                (norm - 1.0).abs() <= 8.0 * (n + 1) as f64 * T::EPS,
                "{n} {j}"
            );
            let d = &(&ac * &x) - &(&x * to_c64::<T>(wj));
            let d: f64 = d
                .as_slice()
                .iter()
                .map(|z| z.norm_sqr())
                .sum::<f64>()
                .sqrt();
            assert!(
                d <= 64.0 * (n + 1) as f64 * T::EPS * scale,
                "{n} {j}: {d:e}"
            );
        }
    }
}

#[test]
fn eigen() {
    each_type!(eigen_of);
}

fn hermitian_of<T: Num>() {
    let mut r = Rng(0x5EED_0103);
    for n in sizes() {
        let b = random::<T>(&mut r, n, n);
        let a = &b + &b.conjugate_transpose();
        let e = a.hermitian_eigen();
        let (v, w) = (e.vectors(), e.values());
        let tol = 32.0 * (n + 1) as f64;
        assert_orthonormal(v, tol);
        let lambda = Matrix::from_diagonal(&w.iter().map(|&x| T::real(f(x))).collect::<Vec<_>>());
        assert_close(&(&(v * &lambda) * &v.conjugate_transpose()), &a, tol);
        assert!(w.windows(2).all(|p| p[0] <= p[1]));
        let alone: Vec<f64> = a.hermitian_eigenvalues().into_iter().map(f).collect();
        let scale = super::common::frobenius(&a) + 1.0;
        for (x, y) in alone.iter().zip(w) {
            assert!((x - f(*y)).abs() <= tol * T::EPS * scale);
        }
        // The general algorithm's eigenvalues: real, and the same.
        let mut general: Vec<Complex<f64>> =
            a.eigenvalues().iter().map(|&z| to_c64::<T>(z)).collect();
        general.sort_by(|x, y| x.re.total_cmp(&y.re));
        for (g, h) in general.iter().zip(w) {
            assert!((g - Complex::new(f(*h), 0.0)).abs() <= 4.0 * tol * T::EPS * scale);
        }
        // Only the lower triangle is read.
        if n > 1 {
            let mut c = a.clone();
            c[(0, n - 1)] = T::real(1e3);
            assert_eq!(c.hermitian_eigenvalues(), a.hermitian_eigenvalues());
        }
    }
}

/// A real eigenvalue as an `f64`.
fn f<R: a::math::complex::Float>(x: R) -> f64 {
    x.to_f64()
}

#[test]
fn hermitian() {
    each_type!(hermitian_of);
}
