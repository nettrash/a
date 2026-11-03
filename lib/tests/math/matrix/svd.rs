//! Singular value decompositions, norms, condition numbers, ranks, null spaces and
//! characteristic polynomials.

use super::common::{
    Num, Rng, assert_close, assert_orthonormal, each_type, frobenius, random, well_conditioned,
};
use a::math::complex::Float;
use a::math::matrix::{Matrix, Norm};

/// Shapes around the panels' width, tall, wide and square.
fn shapes() -> Vec<(usize, usize)> {
    if cfg!(miri) {
        vec![(0, 0), (1, 1), (3, 2), (2, 3), (5, 5)]
    } else {
        let mut v = vec![(0, 0), (0, 3), (3, 0), (1, 1), (1, 5), (5, 1)];
        for n in [2, 3, 7, 16, 33, 64] {
            v.extend([(n, n), (n + 5, n), (n, n + 5), (2 * n + 1, n)]);
        }
        v
    }
}

/// A real value as an `f64`.
fn f<R: Float>(x: R) -> f64 {
    x.to_f64()
}

fn svd_of<T: Num>() {
    let mut r = Rng(0x5EED_0201);
    for (m, n) in shapes() {
        let a = random::<T>(&mut r, m, n);
        let svd = a.svd();
        let (u, v) = (svd.u(), svd.v());
        let s: Vec<f64> = svd.singular_values().iter().map(|&x| f(x)).collect();
        let k = m.min(n);
        assert_eq!((u.shape(), v.shape(), s.len()), ((m, k), (n, k), k));
        let tol = 16.0 * (m + n + 1) as f64;
        assert_orthonormal(&u, tol);
        assert_orthonormal(&v, tol);
        assert!(s.windows(2).all(|p| p[0] >= p[1]) && s.iter().all(|&x| x >= 0.0));
        let sigma = Matrix::from_diagonal(&s.iter().map(|&x| T::real(x)).collect::<Vec<_>>());
        assert_close(&(&(&u * &sigma) * &v.conjugate_transpose()), &a, tol);
        // Without the vectors, the same values.
        let alone: Vec<f64> = a.singular_values().iter().map(|&x| f(x)).collect();
        let scale = frobenius(&a) + 1.0;
        for (x, y) in alone.iter().zip(&s) {
            assert!((x - y).abs() <= tol * T::EPS * scale);
        }
        // The squares are the eigenvalues of `A*·A`, by the Hermitian algorithm.
        let g = &a.conjugate_transpose() * &a;
        let mut w: Vec<f64> = g.hermitian_eigenvalues().iter().map(|&x| f(x)).collect();
        w.reverse();
        for (x, w) in s.iter().zip(&w) {
            assert!(
                (x * x - w).abs() <= 4.0 * tol * T::EPS * scale * scale,
                "{m} {n}"
            );
        }
    }
}

#[test]
fn svd() {
    each_type!(svd_of);
}

/// A random `m × n` matrix of rank `r`.
fn of_rank<T: Num>(r: &mut Rng, m: usize, n: usize, rank: usize) -> Matrix<T> {
    &random::<T>(r, m, rank) * &random::<T>(r, rank, n)
}

fn rank_of<T: Num>() {
    let mut r = Rng(0x5EED_0202);
    for (m, n, rank) in [
        (5, 5, 5),
        (5, 5, 3),
        (8, 4, 2),
        (4, 8, 3),
        (20, 30, 7),
        (6, 6, 0),
    ] {
        let a = of_rank::<T>(&mut r, m, n, rank);
        assert_eq!(a.rank(), rank, "{m} {n} {rank}");
        let null = a.null_space();
        assert_eq!(null.shape(), (n, n - rank));
        assert_orthonormal(&null, 64.0);
        let an = &a * &null;
        assert!(
            frobenius(&an) <= 1e3 * T::EPS * (frobenius(&a) + 1.0),
            "{m} {n} {rank}"
        );
    }
    assert_eq!(Matrix::<T>::identity(4).rank(), 4);
    assert_eq!(Matrix::<T>::zeros(3, 4).rank(), 0);
    assert_eq!(Matrix::<T>::zeros(3, 4).null_space().shape(), (4, 4));
}

#[test]
fn rank_and_null_space() {
    each_type!(rank_of);
}

fn norms_of<T: Num>() {
    let mut r = Rng(0x5EED_0203);
    for (m, n) in shapes() {
        let a = random::<T>(&mut r, m, n);
        let s: Vec<f64> = a.singular_values().iter().map(|&x| f(x)).collect();
        let abs = |i: usize, j: usize| a[(i, j)].c64().abs();
        let one = (0..n)
            .map(|j| (0..m).map(|i| abs(i, j)).sum::<f64>())
            .fold(0.0, f64::max);
        let inf = (0..m)
            .map(|i| (0..n).map(|j| abs(i, j)).sum::<f64>())
            .fold(0.0, f64::max);
        let max = (0..m)
            .flat_map(|i| (0..n).map(move |j| (i, j)))
            .map(|(i, j)| abs(i, j));
        let max = max.fold(0.0, f64::max);
        let tol = 16.0 * (m + n + 1) as f64 * T::EPS;
        let close = |x: f64, y: f64| (x - y).abs() <= tol * (y.abs() + T::EPS);
        assert!(close(f(a.norm(Norm::One)), one));
        assert!(close(f(a.norm(Norm::Infinity)), inf));
        assert!(close(f(a.norm(Norm::Max)), max));
        assert!(close(f(a.norm(Norm::Frobenius)), frobenius(&a)));
        assert!(close(
            f(a.norm(Norm::Two)),
            s.first().copied().unwrap_or(0.0)
        ));
        assert!(close(f(a.norm(Norm::Nuclear)), s.iter().sum()));
        let fro: f64 = s.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(close(fro, frobenius(&a)));
    }
}

#[test]
fn norms() {
    each_type!(norms_of);
}

fn cond_of<T: Num>() {
    let mut r = Rng(0x5EED_0204);
    let i = Matrix::<T>::identity(5);
    for norm in [
        Norm::One,
        Norm::Two,
        Norm::Infinity,
        Norm::Max,
        Norm::Nuclear,
    ] {
        let c = f(i.cond(norm));
        let want = match norm {
            Norm::Nuclear => 25.0,
            _ => 1.0,
        };
        assert!((c - want).abs() <= 64.0 * T::EPS * want, "{norm:?} {c}");
    }
    assert!((f(i.cond(Norm::Frobenius)) - 5.0).abs() <= 64.0 * T::EPS * 5.0);
    let a = well_conditioned::<T>(&mut r, 12);
    let s: Vec<f64> = a.singular_values().iter().map(|&x| f(x)).collect();
    let k = s[0] / s[11];
    assert!((f(a.cond(Norm::Two)) - k).abs() <= 1e3 * T::EPS * k);
    let inv = a.inverse().unwrap();
    for norm in [Norm::One, Norm::Infinity, Norm::Frobenius] {
        let want = f(a.norm(norm)) * f(inv.norm(norm));
        assert!((f(a.cond(norm)) - want).abs() <= 1e3 * T::EPS * want);
    }
    let singular = Matrix::<T>::zeros(3, 3);
    for norm in [
        Norm::One,
        Norm::Two,
        Norm::Infinity,
        Norm::Frobenius,
        Norm::Nuclear,
    ] {
        assert!(f(singular.cond(norm)).is_infinite());
    }
    // In the 2-norm, of any shape.
    let tall = random::<T>(&mut r, 9, 4);
    let s: Vec<f64> = tall.singular_values().iter().map(|&x| f(x)).collect();
    assert!((f(tall.cond(Norm::Two)) - s[0] / s[3]).abs() <= 1e3 * T::EPS * s[0] / s[3]);
}

#[test]
fn condition_numbers() {
    each_type!(cond_of);
}

fn characteristic_of<T: Num>() {
    let mut r = Rng(0x5EED_0205);
    for n in [0, 1, 2, 3, 5, 8, 12] {
        // Scaled so that the powers stay near 1.
        let a = random::<T>(&mut r, n, n) * T::real(1.0 / (n as f64).sqrt().max(1.0));
        let p = a.characteristic_polynomial();
        assert_eq!(p.len(), n + 1);
        assert_eq!(p[0], T::real(1.0));
        // Cayley and Hamilton: `p(A) = 0`, by Horner's rule.
        let mut x = Matrix::<T>::zeros(n, n);
        for &c in &p {
            x = &x * &a;
            for i in 0..n {
                x[(i, i)] += c;
            }
        }
        assert!(
            frobenius(&x) <= 1e3 * (n + 1) as f64 * T::EPS,
            "{n}: {:e}",
            frobenius(&x)
        );
        if n > 0 {
            let tol = 1e3 * T::EPS * (frobenius(&a) + 1.0);
            assert!((p[1].c64() + a.trace().c64()).abs() <= tol);
            let det = a.det().c64() * if n % 2 == 1 { -1.0 } else { 1.0 };
            assert!((p[n].c64() - det).abs() <= tol);
        }
    }
}

#[test]
fn characteristic_polynomials() {
    each_type!(characteristic_of);
}
