//! The nonsymmetric eigenproblem past the small sizes: the blocked Hessenberg reduction, the
//! multishift QR algorithm with aggressive early deflation, and the eigenvectors, on random and
//! hard matrices of every element type, at the sizes where the algorithms change.

use super::common::{Num, Rng, assert_close, assert_orthonormal, each_type, frobenius, random};
use a::math::complex::{Complex, Float};
use a::math::matrix::Matrix;

/// Sizes across the switches of the algorithms: to the multishift QR algorithm (at 120, for
/// complex elements with `Q` at 90), to fewer shifts than LAPACK's (150), to sweeps that gather
/// their reflections without `Q` too (200); past the blocks of the Hessenberg reduction and of
/// the eigenvectors (64), and to windows reduced by the multishift algorithm themselves (with
/// the stalling matrices of [`hard_matrices`]).
fn sizes<T: Num>() -> Vec<usize> {
    let mut sizes = if cfg!(miri) {
        vec![]
    } else if is_complex::<T>() {
        vec![89, 90, 119, 120, 150, 200]
    } else {
        vec![119, 120, 121, 150, 200]
    };
    // The largest only in double precision, whose code is the single's.
    if !cfg!(miri) && T::EPS < 1e-10 {
        sizes.push(300);
    }
    sizes
}

/// Whether the elements are complex.
fn is_complex<T: Num>() -> bool {
    core::any::type_name::<T>().contains("Complex")
}

/// A complex number of the type's precision as a `Complex<f64>`.
fn c64<R: Float>(w: Complex<R>) -> Complex<f64> {
    Complex::new(w.re.to_f64(), w.im.to_f64())
}

/// Checks the Schur decomposition of `a`, its eigenvalues alone, and its eigenvectors, with
/// tolerances `tol·n·ε` relative to the norm of `a`. The eigenvalues of a `defective` matrix,
/// sensitive to the roundings, are checked by their sum only.
fn check<T: Num>(a: &Matrix<T>, tol: f64, defective: bool) {
    let n = a.rows();
    let complex = is_complex::<T>();
    let scale = frobenius(a) + 1.0;
    let s = a.schur();
    let (q, t) = (s.q(), s.t());
    let tol = tol * (n + 1) as f64;
    assert_orthonormal(&q, tol);
    assert_close(&(&(&q * &t) * &q.conjugate_transpose()), a, tol);
    let zero = T::real(0.0);
    let values = s.eigenvalues();
    assert_eq!(values.len(), n);
    // The structure: triangular, or quasi-triangular with standard 2 × 2 blocks.
    let mut i = 0;
    while i < n {
        for r in i + 2..n {
            assert_eq!(t[(r, i)], zero, "below the subdiagonal at {r} {i}, n = {n}");
        }
        let d = t[(i, i)].c64();
        if !complex && i + 1 < n && t[(i + 1, i)] != zero {
            assert!(
                i + 2 >= n || t[(i + 2, i + 1)] == zero,
                "blocks overlap at {i}"
            );
            let (b, c) = (t[(i, i + 1)].c64().re, t[(i + 1, i)].c64().re);
            assert_eq!(
                t[(i + 1, i + 1)].c64(),
                d,
                "unequal diagonal in the block at {i}"
            );
            assert!(b * c < 0.0, "a block of real eigenvalues at {i}");
            let im = b.abs().sqrt() * c.abs().sqrt();
            let (w1, w2) = (c64(values[i]), c64(values[i + 1]));
            assert!((w1 - Complex::new(d.re, im)).abs() <= 8.0 * T::EPS * (d.abs() + im));
            assert!((w2 - Complex::new(d.re, -im)).abs() <= 8.0 * T::EPS * (d.abs() + im));
            i += 2;
        } else {
            if i + 1 < n {
                assert_eq!(t[(i + 1, i)], zero, "subdiagonal at {i}, n = {n}");
            }
            assert_eq!(c64(values[i]), d);
            i += 1;
        }
    }
    // Their sum is the trace.
    let sum: Complex<f64> = values.iter().map(|&w| c64(w)).sum();
    let trace = a.trace().c64();
    assert!(
        (sum - trace).abs() <= tol * T::EPS * scale,
        "trace {trace}, sum {sum}, n = {n}"
    );
    if !defective {
        same_values(a, values, tol, scale);
    }
    // The eigenvectors: of length 1, `A·v = λ·v` up to rounding, even for a defective matrix.
    let e = a.eigen();
    let v: Matrix<Complex<f64>> = e.vectors().map(|&x| c64(x));
    let ac = a.map(|x| x.c64());
    for (j, &w) in e.values().iter().enumerate() {
        let x = Matrix::from_columns(&[v.column(j)]);
        let norm: f64 = x
            .as_slice()
            .iter()
            .map(|z| z.norm_sqr())
            .sum::<f64>()
            .sqrt();
        assert!(
            (norm - 1.0).abs() <= tol * T::EPS,
            "norm {norm} of vector {j}, n = {n}"
        );
        let d = &(&ac * &x) - &(&x * c64(w));
        let d: f64 = d
            .as_slice()
            .iter()
            .map(|z| z.norm_sqr())
            .sum::<f64>()
            .sqrt();
        assert!(
            d <= 4.0 * tol * T::EPS * scale,
            "residual {d:e} of vector {j}, n = {n}"
        );
    }
}

/// Asserts that the eigenvalues alone are `values`, each matched to the nearest one left: the
/// order of nearly equal ones may differ.
fn same_values<T: Num>(a: &Matrix<T>, values: &[Complex<T::Real>], tol: f64, scale: f64) {
    let n = a.rows();
    let key =
        |w: &Complex<f64>, v: &Complex<f64>| w.re.total_cmp(&v.re).then(w.im.total_cmp(&v.im));
    let mut alone: Vec<Complex<f64>> = a.eigenvalues().into_iter().map(c64).collect();
    let mut with: Vec<Complex<f64>> = values.iter().map(|&w| c64(w)).collect();
    alone.sort_by(key);
    with.sort_by(key);
    let mut used = vec![false; n];
    for x in &alone {
        let (k, d) = with
            .iter()
            .enumerate()
            .filter(|(k, _)| !used[*k])
            .map(|(k, y)| (k, (x - y).abs()))
            .min_by(|p, q| p.1.total_cmp(&q.1))
            .unwrap();
        used[k] = true;
        let bound = (tol * T::EPS * scale).sqrt().max(tol * T::EPS * scale);
        assert!(
            d <= bound,
            "eigenvalue {x} alone, nearest {} with Q, n = {n}",
            with[k]
        );
    }
}

fn random_of<T: Num>() {
    let mut r = Rng(0x5EED_0201);
    for n in sizes::<T>() {
        check(&random::<T>(&mut r, n, n), 32.0, false);
    }
}

#[test]
fn random_matrices() {
    each_type!(random_of);
}

/// Matrices already upper Hessenberg, and ones with zeros on the subdiagonal.
fn hessenberg_of<T: Num>() {
    let mut r = Rng(0x5EED_0202);
    for n in sizes::<T>().into_iter().filter(|&n| n < 300) {
        let a = Matrix::from_fn(n, n, |i, j| {
            if i <= j + 1 {
                T::random(&mut r)
            } else {
                T::real(0.0)
            }
        });
        check(&a, 32.0, false);
        let split = Matrix::from_fn(n, n, |i, j| {
            if i == j + 1 && i % 7 == 0 {
                T::real(0.0)
            } else {
                a[(i, j)]
            }
        });
        check(&split, 32.0, false);
    }
}

#[test]
fn hessenberg_matrices() {
    each_type!(hessenberg_of);
}

/// The sizes of the hard cases: one each side of the switches to the multishift algorithm, none
/// under Miri, which the small ones of [`super::eigen`] serve.
fn hard_sizes() -> Vec<usize> {
    if cfg!(miri) { vec![] } else { vec![76, 150] }
}

/// A cyclic permutation, whose shifts stall without exceptional ones.
fn cyclic<T: Num>(n: usize) -> Matrix<T> {
    Matrix::from_fn(n, n, |i, j| {
        T::real(if (i + 1) % n == j { 1.0 } else { 0.0 })
    })
}

fn stalling_of<T: Num>() {
    for n in hard_sizes() {
        check(&cyclic::<T>(n), 32.0, false);
    }
    // Its stalls grow the deflation window past the switch to the multishift algorithm, 120
    // for real elements, 90 for complex ones with `Q`; in double precision only, and not under
    // Miri, where it takes hours.
    if !cfg!(miri) && T::EPS < 1e-10 {
        check(
            &cyclic::<T>(if is_complex::<T>() { 300 } else { 400 }),
            32.0,
            false,
        );
    }
}

#[test]
fn stalling_matrices() {
    each_type!(stalling_of);
}

/// Defective matrices: a Jordan block, and blocks of order 5, a few sharing an eigenvalue,
/// hidden by a similarity.
fn defective_of<T: Num>() {
    let mut r = Rng(0x5EED_0203);
    let zero = T::real(0.0);
    for n in hard_sizes() {
        let jordan = Matrix::from_fn(n, n, |i, j| {
            if i == j {
                T::real(2.0)
            } else if i + 1 == j {
                T::real(1.0)
            } else {
                zero
            }
        });
        check(&jordan, 32.0, true);
        let blocks = Matrix::from_fn(n, n, |i, j| {
            if i == j {
                T::real((i / 5 % 3) as f64)
            } else if i + 1 == j && i % 5 != 4 {
                T::real(1.0)
            } else {
                zero
            }
        });
        let p = super::common::well_conditioned::<T>(&mut r, n);
        let pinv = p.inverse().unwrap();
        check(&(&(&p * &blocks) * &pinv), 1e3, true);
    }
}

#[test]
fn defective_matrices() {
    each_type!(defective_of);
}

/// Hard cases: repeated and clustered eigenvalues; a graded matrix; a companion matrix; real
/// matrices with complex eigenvalues only; zero and a multiple of the identity.
fn hard_of<T: Num>() {
    let mut r = Rng(0x5EED_0204);
    let zero = T::real(0.0);
    for n in hard_sizes() {
        // Repeated and clustered eigenvalues of a normal matrix: Q·D·Q*.
        let q = random::<T>(&mut r, n, n).qr().q();
        let d = Matrix::from_diagonal(
            &(0..n)
                .map(|i| T::real(1.0 + (i % 4) as f64 + 1e-9 * (i % 3) as f64))
                .collect::<Vec<_>>(),
        );
        check(&(&(&q * &d) * &q.conjugate_transpose()), 32.0, false);
        // Graded: elements from 1 down to about 1e-12.
        let g = Matrix::from_fn(n, n, |i, j| {
            T::random(&mut r) * T::real(10f64.powf(-12.0 * (i + j) as f64 / (2 * n) as f64))
        });
        check(&g, 64.0, false);
        // The companion matrix of a polynomial with random coefficients.
        let coefficients: Vec<T> = (0..n).map(|_| T::random(&mut r)).collect();
        let c = Matrix::from_fn(n, n, |i, j| {
            if j == n - 1 {
                -coefficients[i]
            } else if i == j + 1 {
                T::real(1.0)
            } else {
                zero
            }
        });
        check(&c, 64.0, false);
        // Rotations by different angles on the diagonal, mixed by an orthogonal similarity: a
        // real matrix of complex pairs only.
        let mut rot = Matrix::<T>::zeros(n, n);
        for k in 0..n / 2 {
            let (s, cth) = (0.1 + k as f64).sin_cos();
            rot[(2 * k, 2 * k)] = T::real(cth);
            rot[(2 * k, 2 * k + 1)] = T::real(-s);
            rot[(2 * k + 1, 2 * k)] = T::real(s);
            rot[(2 * k + 1, 2 * k + 1)] = T::real(cth);
        }
        if n % 2 == 1 {
            rot[(n - 1, n - 1)] = T::real(1.0);
        }
        check(&(&(&q * &rot) * &q.conjugate_transpose()), 32.0, false);
        check(&Matrix::<T>::zeros(n, n), 32.0, false);
        check(
            &Matrix::from_fn(n, n, |i, j| if i == j { T::real(3.0) } else { zero }),
            32.0,
            false,
        );
    }
}

#[test]
fn hard_matrices() {
    each_type!(hard_of);
}

/// A matrix with a NaN or an infinity gives NaNs.
fn not_finite_of<T: Num>() {
    let mut r = Rng(0x5EED_0204);
    for n in hard_sizes() {
        let mut a = random::<T>(&mut r, n, n);
        a[(n / 2, n / 3)] = T::real(f64::INFINITY);
        assert!(a.eigenvalues().iter().all(|w| w.re.is_nan()));
        let e = a.eigen();
        assert!(e.values().iter().all(|w| w.re.is_nan()));
    }
}

#[test]
fn not_finite() {
    each_type!(not_finite_of);
}
