//! Matrices at the ends of the floats' range: norms of subnormal elements, reflections near the
//! largest float, eigenvalues of matrices scaled up and down, complex elements with finite parts
//! whose absolute values overflow, and exponentials whose norms overflow.

use super::common::{Num, Rng, assert_orthonormal, each_type, random};
use a::math::complex::{Complex, Float};
use a::math::matrix::{Matrix, Norm};

#[test]
fn norms_of_subnormal_elements() {
    let tiny = 1e-310;
    assert_eq!(Matrix::from_rows(&[[tiny]]).norm(Norm::Frobenius), tiny);
    let a = Matrix::from_rows(&[[tiny, 0.0], [0.0, -tiny]]);
    let want = tiny * 2f64.sqrt();
    let ulp = f64::from_bits(1);
    assert!((a.norm(Norm::Frobenius) - want).abs() <= 2.0 * ulp);
    let c = Matrix::from_rows(&[[Complex::new(tiny, tiny)], [Complex::ZERO]]);
    assert!((c.norm(Norm::Frobenius) - want).abs() <= 2.0 * ulp);
}

/// The largest power of two whose multiples of elements below 1, a few times over, stay finite.
fn top<T: Num>() -> f64 {
    if T::EPS > 1e-10 {
        2f64.powi(126)
    } else {
        2f64.powi(1022)
    }
}

fn near_the_top_of<T: Num>() {
    // `alpha = s` and `x = [s/10]`, `1e308` and `1e307` in double precision: the norm is finite
    // but `alpha - β`, about `2s`, is not.
    let s = if T::EPS > 1e-10 { 3e38 } else { 1e308 };
    let a = Matrix::from_rows(&[[T::real(s)], [T::real(s / 10.0)]]);
    let qr = a.qr();
    let (q, r) = (qr.q(), qr.r());
    assert!(q.as_slice().iter().all(|x| x.c64().is_finite()));
    let norm = s.hypot(s / 10.0);
    assert!((r[(0, 0)].c64().abs() - norm).abs() <= 8.0 * T::EPS * norm);
    assert_orthonormal(&q, 16.0);
    let back = &q * &r;
    for (x, y) in back.as_slice().iter().zip(a.as_slice()) {
        assert!((x.c64() - y.c64()).abs() <= 16.0 * T::EPS * norm);
    }
}

#[test]
fn reflections_near_the_largest_float() {
    each_type!(near_the_top_of);
}

/// The eigenvalues of `b`, each matched with the nearest of `got`, scaled by `1/s`.
#[track_caller]
fn assert_scaled<T: Num>(b: &Matrix<T>, got: &[Complex<T::Real>], s: f64) {
    let want = b.eigenvalues();
    let scale = want.iter().fold(1.0f64, |m, w| m.max(c64::<T>(*w).abs()));
    for w in &want {
        let w = c64::<T>(*w);
        let d = got
            .iter()
            .map(|g| (c64::<T>(*g) * (1.0 / s) - w).abs())
            .fold(f64::INFINITY, f64::min);
        assert!(d <= 1e3 * T::EPS * scale, "{w}: {d:e}");
    }
}

/// A complex number of the element's real type as a `Complex<f64>`.
fn c64<T: Num>(z: Complex<T::Real>) -> Complex<f64> {
    Complex::new(z.re.to_f64(), z.im.to_f64())
}

fn eigenvalues_scaled_of<T: Num>() {
    // `s·B` for `B` with elements in `[-1, 1)`: near the top, its reductions' differences would
    // overflow, and near the bottom their products would underflow, unless the matrix is scaled
    // first. The eigenvalues are `s` times `B`'s.
    let mut r = Rng(0x5EED_0901);
    let n = if cfg!(miri) { 4 } else { 12 };
    let bottom = if T::EPS > 1e-10 {
        2f64.powi(-120)
    } else {
        2f64.powi(-1000)
    };
    for s in [top::<T>() / 4.0, bottom] {
        let b = random::<T>(&mut r, n, n);
        let a = b.map(|&x| x * T::real(s));
        let schur = a.schur();
        assert_orthonormal(&schur.q(), 32.0 * n as f64);
        assert!(schur.t().as_slice().iter().all(|x| x.c64().is_finite()));
        assert_scaled(&b, schur.eigenvalues(), s);
        assert_scaled(&b, &a.eigenvalues(), s);
        let e = a.eigen();
        assert!(
            e.vectors()
                .as_slice()
                .iter()
                .all(|x| c64::<T>(*x).is_finite())
        );
    }
}

#[test]
fn eigenvalues_near_the_ends_of_the_range() {
    each_type!(eigenvalues_scaled_of);
}

#[test]
fn complex_elements_whose_absolute_values_overflow() {
    // `|z| = √2·MAX` overflows although both parts are finite: the scaling of the singular
    // values and the Hermitian eigenvalues takes the largest part for it.
    let z = Complex::new(f64::MAX, f64::MAX);
    let one = Complex::<f64>::ONE;
    let mut a = Matrix::<Complex<f64>>::zeros(3, 3);
    (a[(0, 0)], a[(1, 1)], a[(2, 2)]) = (z, one, Complex::new(0.5, 0.0));
    let s = a.singular_values();
    assert_eq!(s[0], f64::INFINITY);
    assert!((s[1] - 1.0).abs() <= 4.0 * f64::EPSILON && (s[2] - 0.5).abs() <= 4.0 * f64::EPSILON);
    let svd = a.svd();
    assert_eq!(svd.singular_values()[0], f64::INFINITY);
    assert!(svd.singular_values()[1..].iter().all(|x| x.is_finite()));
    assert!(svd.u().as_slice().iter().all(|x| x.is_finite()));
    assert!(svd.v().as_slice().iter().all(|x| x.is_finite()));
    // `[[0, z*], [z, 0]]`, of eigenvalues `±|z|`, beside ones: on the small path and the blocked
    // one.
    for n in [3, if cfg!(miri) { 3 } else { 200 }] {
        let mut h = Matrix::<Complex<f64>>::identity(n);
        (h[(0, 0)], h[(1, 1)], h[(1, 0)], h[(0, 1)]) = (Complex::ZERO, Complex::ZERO, z, z.conj());
        let w = h.hermitian_eigenvalues();
        assert_eq!((w[0], w[n - 1]), (f64::NEG_INFINITY, f64::INFINITY), "{n}");
        assert!(w[1..n - 1].iter().all(|x| x.is_finite()), "{n}");
        let e = h.hermitian_eigen();
        assert_eq!(e.values()[0], f64::NEG_INFINITY, "{n}");
        assert!(!e.values().iter().any(|x| x.is_nan()), "{n}");
    }
}

fn exp_of_an_overflowing_norm<T: Num>() {
    // `A = -c·(I + J)` with `J` all ones, its elements finite and its 1-norm `5c` not: its
    // eigenvalues `-c` and `-5c` make `e^A` zero. The number of squarings stays near the type's
    // range rather than becoming `i32::MAX`, which never finished.
    let c = if T::EPS > 1e-10 { 8e37 } else { 4e307 };
    let a = Matrix::from_fn(4, 4, |i, j| T::real(if i == j { -2.0 * c } else { -c }));
    assert!(!a.norm(Norm::One).to_f64().is_finite());
    let e = a.exp();
    assert!(e.as_slice().iter().all(|x| x.c64().abs() <= 1e-30), "{e:?}");
}

#[test]
fn exponentials_of_norms_that_overflow() {
    each_type!(exp_of_an_overflowing_norm);
}
