//! The Hermitian eigensolver on sizes across its thresholds and on the matrices that make divide
//! and conquer deflate: checked by orthonormality and `V·Λ·V* = A`, in `f64` arithmetic.

use super::common::{Num, Rng, each_type, random, size};
use a::math::complex::{Complex, Float};
use a::math::matrix::Matrix;

/// The matrix in `Complex<f64>`.
fn c64<T: Num>(a: &Matrix<T>) -> Matrix<Complex<f64>> {
    a.map(|x| x.c64())
}

/// The Frobenius norm of a `Complex<f64>` matrix.
fn norm(a: &Matrix<Complex<f64>>) -> f64 {
    a.as_slice()
        .iter()
        .map(|x| x.norm_sqr())
        .sum::<f64>()
        .sqrt()
}

/// The eigendecomposition of `a` and its errors in units of `n·ε`: `‖V*·V - I‖` and
/// `‖A·V - V·Λ‖ / ‖A‖`, and the largest difference of the eigenvalues computed alone, in units
/// of `n·ε·‖A‖`, all with `A` and `Λ` divided by `f` first. Checks that the eigenvalues ascend.
fn errors<T: Num>(a: &Matrix<T>, f: f64) -> (f64, f64, f64) {
    let n = a.rows();
    let e = a.hermitian_eigen();
    let (v, w) = (c64(e.vectors()), e.values());
    assert_eq!((v.shape(), w.len()), ((n, n), n));
    assert!(w.windows(2).all(|p| p[0] <= p[1]), "not ascending");
    let unit = (n.max(1) as f64) * T::EPS;
    let orth = norm(&(&(&v.conjugate_transpose() * &v) - &Matrix::identity(n))) / unit;
    // The whole matrix, from its lower triangle.
    let full = Matrix::from_fn(n, n, |i, j| {
        let x = if i >= j {
            a[(i, j)].c64()
        } else {
            a[(j, i)].c64().conj()
        };
        x / f
    });
    let vl = Matrix::from_fn(n, n, |i, j| v[(i, j)] * (w[j].to_f64() / f));
    let size = norm(&full);
    let scale = unit * if size > 0.0 { size } else { 1.0 };
    let resid = norm(&(&(&full * &v) - &vl)) / scale;
    let alone = a.hermitian_eigenvalues();
    let mut diff = 0f64;
    for (x, y) in alone.iter().zip(w) {
        diff = diff.max((x.to_f64() / f - y.to_f64() / f).abs());
    }
    (orth, resid, diff / scale)
}

/// Asserts the errors of [`errors`] are below `tol`.
#[track_caller]
fn check<T: Num>(a: &Matrix<T>, tol: f64, what: &str) {
    check_scaled(a, 1.0, tol, what);
}

/// [`check`] of `a` divided by `f`.
#[track_caller]
fn check_scaled<T: Num>(a: &Matrix<T>, f: f64, tol: f64, what: &str) {
    let (orth, resid, diff) = errors(a, f);
    let name = core::any::type_name::<T>();
    assert!(
        orth <= tol && resid <= tol && diff <= tol,
        "{what} of {} for {name}: ‖V*V - I‖ {orth:.2}, ‖VΛV* - A‖ {resid:.2}, values {diff:.2} (n·ε)",
        a.rows()
    );
}

/// A random unitary matrix, from a QR decomposition.
fn unitary<T: Num>(r: &mut Rng, n: usize) -> Matrix<T> {
    random::<T>(r, n, n).qr().q()
}

/// `Q·diag(λ)·Q*` for a random unitary `Q`.
fn with_eigenvalues<T: Num>(r: &mut Rng, lambda: &[f64]) -> Matrix<T> {
    let n = lambda.len();
    let q = unitary::<T>(r, n);
    let l: Vec<T> = lambda.iter().map(|&x| T::real(x)).collect();
    &(&q * &Matrix::from_diagonal(&l)) * &q.conjugate_transpose()
}

/// The symmetric tridiagonal matrix with the diagonal `d` and the off-diagonal `e`.
fn tridiagonal<T: Num>(d: &[f64], e: &[f64]) -> Matrix<T> {
    Matrix::from_fn(d.len(), d.len(), |i, j| {
        if i == j {
            T::real(d[i])
        } else if i == j + 1 || j == i + 1 {
            T::real(e[i.min(j)])
        } else {
            T::real(0.0)
        }
    })
}

/// Sizes around the divide and conquer's smallest problems (25) and the reduction's panels and
/// unblocked tail (32 and 64), and a few levels and panels more; 300 for `f64` alone, which is
/// enough to check slowly.
fn sizes<T: Num>() -> Vec<usize> {
    if cfg!(miri) {
        return vec![0, 1, 2, 24, 26];
    }
    let mut sizes = vec![
        0, 1, 2, 3, 24, 25, 26, 27, 50, 51, 63, 64, 65, 66, 96, 97, 100, 130, 200,
    ];
    if core::any::type_name::<T>() == "f64" {
        sizes.push(300);
    }
    sizes
}

fn random_of<T: Num>() {
    let mut r = Rng(0x5EED_0201);
    for n in sizes::<T>() {
        let b = random::<T>(&mut r, n, n);
        let a = &b + &b.conjugate_transpose();
        check(&a, 4.0, "random");
        // Only the lower triangle is read: anything above it changes nothing.
        let mut c = a.clone();
        for j in 1..n {
            for i in 0..j {
                c[(i, j)] = T::real(1e3 + i as f64);
            }
        }
        if n > 0 && n <= 100 {
            let (x, y) = (c.hermitian_eigen(), a.hermitian_eigen());
            assert_eq!((x.values(), x.vectors()), (y.values(), y.vectors()));
        }
    }
}

#[test]
fn random_sizes() {
    each_type!(random_of);
}

fn deflating_of<T: Num>() {
    let mut r = Rng(0x5EED_0202);
    let n = size(130, 30);
    deflating_at::<T>(&mut r, n);
    // Wilkinson's W21+ and a bigger one, whose eigenvalues come in pairs closer than any ulp.
    for m in [10, size(60, 13)] {
        wilkinson::<T>(&mut r, m);
    }
}

/// The matrices that make divide and conquer deflate, of order `n`: low rank changes of the
/// identity, repeated and clustered eigenvalues, the second difference matrix, and a
/// tridiagonal one that splits.
fn deflating_at<T: Num>(r: &mut Rng, n: usize) {
    // The identity plus rank one and rank two: all but one or two eigenvalues are 1.
    for rank in [1, 2] {
        let mut a = Matrix::<T>::identity(n);
        for _ in 0..rank {
            let u = random::<T>(r, n, 1);
            a = &a + &(&u * &u.conjugate_transpose());
        }
        check(&a, 4.0, "identity plus low rank");
    }
    // Eigenvalues repeated in a few big clusters, exactly and to within a few ulps.
    let repeated: Vec<f64> = (0..n).map(|i| (i % 3) as f64 - 1.0).collect();
    check(&with_eigenvalues::<T>(r, &repeated), 8.0, "repeated");
    let clustered: Vec<f64> = (0..n)
        .map(|i| 1.0 + (i % 7) as f64 * 4.0 * T::EPS)
        .collect();
    check(&with_eigenvalues::<T>(r, &clustered), 8.0, "clustered");
    let close: Vec<f64> = (0..n)
        .map(|i| 1.0 + i as f64 * 1e-3 * T::EPS.sqrt())
        .collect();
    check(&with_eigenvalues::<T>(r, &close), 8.0, "close");
    // The second difference matrix, with its known eigenvalues `2 - 2·cos(kπ/(n + 1))`.
    let t = tridiagonal::<T>(&vec![2.0; n], &vec![-1.0; n - 1]);
    check(&t, 4.0, "second difference");
    let w = t.hermitian_eigenvalues();
    for (k, w) in w.iter().enumerate() {
        let exact = 2.0 - 2.0 * ((k + 1) as f64 * core::f64::consts::PI / (n + 1) as f64).cos();
        assert!(
            (w.to_f64() - exact).abs() <= 8.0 * T::EPS * 4.0,
            "{k}: {w:?} {exact}"
        );
    }
    // Tridiagonal with negligible off-diagonals here and there: it splits.
    let d: Vec<f64> = (0..n).map(|i| ((i * 7) % 11) as f64).collect();
    let e: Vec<f64> = (0..n - 1)
        .map(|i| if i % 9 == 4 { 1e-30 } else { 0.5 })
        .collect();
    check(&tridiagonal::<T>(&d, &e), 4.0, "splitting");
}

/// Wilkinson's matrix `W_{2m+1}+`, whose eigenvalues come in pairs closer than any ulp, as it
/// is and reduced from a dense matrix.
fn wilkinson<T: Num>(r: &mut Rng, m: usize) {
    let d: Vec<f64> = (0..2 * m + 1)
        .map(|i| (i as f64 - m as f64).abs())
        .collect();
    let e = vec![1.0; 2 * m];
    check(&tridiagonal::<T>(&d, &e), 4.0, "Wilkinson");
    let q = unitary::<T>(r, 2 * m + 1);
    let t = tridiagonal::<T>(&d, &e);
    check(
        &(&(&q * &t) * &q.conjugate_transpose()),
        4.0,
        "dense Wilkinson",
    );
}

#[test]
fn deflating() {
    each_type!(deflating_of);
}

fn structured_of<T: Num>() {
    let mut r = Rng(0x5EED_0203);
    structured_at::<T>(&mut r, size(100, 30));
}

/// Matrices of order `n` with structure: diagonal, zero, graded, near the ends of the range,
/// not finite.
fn structured_at<T: Num>(r: &mut Rng, n: usize) {
    // Diagonal, with repeats: the eigenvalues are its elements, the vectors the identity's.
    let d: Vec<T> = (0..n)
        .map(|i| T::real(((i * 5) % 13) as f64 - 6.0))
        .collect();
    let a = Matrix::from_diagonal(&d);
    check(&a, 1.0, "diagonal");
    let mut sorted: Vec<f64> = d.iter().map(|x| x.c64().re).collect();
    sorted.sort_by(f64::total_cmp);
    let w: Vec<f64> = a
        .hermitian_eigenvalues()
        .iter()
        .map(|x| x.to_f64())
        .collect();
    assert_eq!(w, sorted);
    // Zero: zero eigenvalues and orthonormal vectors.
    let z = Matrix::<T>::zeros(n, n);
    check(&z, 1.0, "zero");
    assert!(
        z.hermitian_eigen()
            .values()
            .iter()
            .all(|&x| x == Float::ZERO)
    );
    // Graded: elements from 1 down to about 10⁻²⁰ (f64) or 10⁻¹⁰ (f32) across the matrix.
    let decades = if T::EPS < 1e-10 { 20.0 } else { 10.0 };
    let g = 10f64.powf(-decades / (2 * n) as f64);
    let b = random::<T>(r, n, n);
    let s = &b + &b.conjugate_transpose();
    let graded = Matrix::from_fn(n, n, |i, j| s[(i, j)] * T::real(g.powi((i + j) as i32)));
    check(&graded, 4.0, "graded");
    // Far from 1: scaled near the ends of the range.
    let (big, small) = if T::EPS < 1e-10 {
        (1e300, 1e-300)
    } else {
        (1e35, 1e-35)
    };
    for f in [big, small] {
        let scaled = s.map(|&x| x * T::real(f));
        check_scaled(&scaled, f, 4.0, "scaled");
    }
    // Not a number, or infinite: NaNs.
    let mut bad = s.clone();
    bad[(n - 1, 0)] = T::real(f64::INFINITY);
    let e = bad.hermitian_eigen();
    assert!(e.values().iter().all(|x| x.to_f64().is_nan()));
    assert!(e.vectors().as_slice().iter().all(|x| x.c64().re.is_nan()));
    assert!(
        bad.hermitian_eigenvalues()
            .iter()
            .all(|x| x.to_f64().is_nan())
    );
}

#[test]
fn structured() {
    each_type!(structured_of);
}

/// The orders of the small path, on both sides of the switches of its scratch at 16 and 48 and
/// of the crossover of the complex vectors at 80.
fn small_sizes() -> Vec<usize> {
    if cfg!(miri) {
        vec![2, 5, 17]
    } else {
        vec![2, 3, 5, 8, 16, 17, 40, 48, 49, 80, 81]
    }
}

/// The orders about the other crossovers of the small path: those of the vectors, 112 for `f64`
/// and 160 for `f32`, and that of the eigenvalues alone, 128.
fn crossovers<T: Num>() -> Vec<usize> {
    if cfg!(miri) {
        return Vec::new();
    }
    match core::any::type_name::<T>() {
        "f64" => vec![112, 113, 128, 129],
        "f32" => vec![128, 129, 160, 161],
        _ => vec![128, 129],
    }
}

fn small_of<T: Num>() {
    let mut r = Rng(0x5EED_0204);
    for n in small_sizes() {
        deflating_at::<T>(&mut r, n);
        structured_at::<T>(&mut r, n);
        wilkinson::<T>(&mut r, (n - 1) / 2);
        // Random, with anything finite written above the diagonal, which is not read.
        let b = random::<T>(&mut r, n, n);
        let a = &b + &b.conjugate_transpose();
        check(&a, 4.0, "random");
        let mut c = a.clone();
        for j in 1..n {
            for i in 0..j {
                c[(i, j)] = T::real(1e3 + i as f64);
            }
        }
        // Miri perturbs the roundings of some operations at random, run to run.
        if !cfg!(miri) {
            let (x, y) = (c.hermitian_eigen(), a.hermitian_eigen());
            assert_eq!((x.values(), x.vectors()), (y.values(), y.vectors()));
        }
    }
    for n in crossovers::<T>() {
        let b = random::<T>(&mut r, n, n);
        check(&(&b + &b.conjugate_transpose()), 4.0, "random");
        let clustered: Vec<f64> = (0..n)
            .map(|i| 1.0 + (i % 7) as f64 * 4.0 * T::EPS)
            .collect();
        check(&with_eigenvalues::<T>(&mut r, &clustered), 8.0, "clustered");
    }
}

#[test]
fn small_hard_cases() {
    each_type!(small_of);
}
