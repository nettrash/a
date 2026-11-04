//! What the tests of matrices share: the element types under test, random matrices, and
//! comparisons up to rounding.

use a::math::complex::Complex;
use a::math::matrix::{Matrix, Scalar};

/// Scales a test down when it runs under Miri.
pub(super) fn size(normal: usize, miri: usize) -> usize {
    if cfg!(miri) { miri } else { normal }
}

/// Small deterministic generator.
pub(super) struct Rng(pub(super) u64);

impl Rng {
    /// The next 64 bits.
    pub(super) fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// A uniform float in `[-1, 1)`.
    pub(super) fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    }
}

/// An element type under test, seen through `Complex<f64>`.
pub(super) trait Num: Scalar + Copy + core::fmt::Debug {
    /// The unit roundoff of the type.
    const EPS: f64;
    /// A random element, its parts uniform in `[-1, 1)`.
    fn random(r: &mut Rng) -> Self;
    /// The real number `x`.
    fn real(x: f64) -> Self;
    /// The element, exactly, as a `Complex<f64>`.
    fn c64(self) -> Complex<f64>;
}

impl Num for f32 {
    const EPS: f64 = f32::EPSILON as f64;
    fn random(r: &mut Rng) -> Self {
        r.unit() as f32
    }
    fn real(x: f64) -> Self {
        x as f32
    }
    fn c64(self) -> Complex<f64> {
        Complex::new(self as f64, 0.0)
    }
}

impl Num for f64 {
    const EPS: f64 = f64::EPSILON;
    fn random(r: &mut Rng) -> Self {
        r.unit()
    }
    fn real(x: f64) -> Self {
        x
    }
    fn c64(self) -> Complex<f64> {
        Complex::new(self, 0.0)
    }
}

impl Num for Complex<f32> {
    const EPS: f64 = f32::EPSILON as f64;
    fn random(r: &mut Rng) -> Self {
        Complex::new(r.unit() as f32, r.unit() as f32)
    }
    fn real(x: f64) -> Self {
        Complex::new(x as f32, 0.0)
    }
    fn c64(self) -> Complex<f64> {
        Complex::new(self.re as f64, self.im as f64)
    }
}

impl Num for Complex<f64> {
    const EPS: f64 = f64::EPSILON;
    fn random(r: &mut Rng) -> Self {
        Complex::new(r.unit(), r.unit())
    }
    fn real(x: f64) -> Self {
        Complex::new(x, 0.0)
    }
    fn c64(self) -> Complex<f64> {
        self
    }
}

/// Runs `$f::<T>()` for each element type.
macro_rules! each_type {
    ($f:ident) => {
        $f::<f32>();
        $f::<f64>();
        $f::<a::math::complex::Complex<f32>>();
        $f::<a::math::complex::Complex<f64>>();
    };
}
pub(super) use each_type;

/// A module `$name` of four tests, `$f::<T>()` for each element type: for tests too large to
/// run all four types in one process under Miri on a 32-bit target, which never reuses stack
/// addresses and runs out of them.
macro_rules! test_each_type {
    ($name:ident, $f:ident) => {
        mod $name {
            #[test]
            fn f32() {
                super::$f::<f32>();
            }
            #[test]
            fn f64() {
                super::$f::<f64>();
            }
            #[test]
            fn complex_f32() {
                super::$f::<a::math::complex::Complex<f32>>();
            }
            #[test]
            fn complex_f64() {
                super::$f::<a::math::complex::Complex<f64>>();
            }
        }
    };
}
pub(super) use test_each_type;

/// A random `m × n` matrix.
pub(super) fn random<T: Num>(r: &mut Rng, m: usize, n: usize) -> Matrix<T> {
    Matrix::from_fn(m, n, |_, _| T::random(r))
}

/// A random matrix that is far from singular: random plus `n` on the diagonal.
pub(super) fn well_conditioned<T: Num>(r: &mut Rng, n: usize) -> Matrix<T> {
    let mut a = random::<T>(r, n, n);
    for i in 0..n {
        a[(i, i)] += T::real(n as f64);
    }
    a
}

/// A random Hermitian positive definite matrix: `B·B* + n·I`.
pub(super) fn positive_definite<T: Num>(r: &mut Rng, n: usize) -> Matrix<T> {
    let b = random::<T>(r, n, n);
    let mut a = &b * &b.conjugate_transpose();
    for i in 0..n {
        a[(i, i)] += T::real(n as f64);
    }
    a
}

/// The Frobenius norm, in `f64`.
pub(super) fn frobenius<T: Num>(a: &Matrix<T>) -> f64 {
    a.as_slice()
        .iter()
        .map(|x| x.c64().norm_sqr())
        .sum::<f64>()
        .sqrt()
}

/// Asserts `‖a - b‖ ≤ tol·ε·(‖b‖ + 1)` in the Frobenius norm, `ε` the unit roundoff of the type.
#[track_caller]
pub(super) fn assert_close<T: Num>(a: &Matrix<T>, b: &Matrix<T>, tol: f64) {
    assert_eq!(a.shape(), b.shape());
    let d = frobenius(&(a - b));
    let bound = tol * T::EPS * (frobenius(b) + 1.0);
    assert!(
        d <= bound,
        "‖a - b‖ = {d:e} > {bound:e} for {:?} of {}",
        a.shape(),
        core::any::type_name::<T>()
    );
}

/// Asserts that the columns of `q` are orthonormal: `Q*·Q = I`.
#[track_caller]
pub(super) fn assert_orthonormal<T: Num>(q: &Matrix<T>, tol: f64) {
    let n = q.cols();
    assert_close(
        &(&q.conjugate_transpose() * q),
        &Matrix::identity(n),
        tol * n as f64,
    );
}
