//! The types a matrix can hold.

use super::blas::{Op, gemm};
use crate::math::complex::{Complex, Float};
use crate::math::numbers::Number;
use crate::math::rational::Rational;
use core::fmt;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// What the arithmetic of a matrix needs of its elements, sealed.
pub(crate) mod sealed {
    use super::*;

    /// The ring operations, by reference so that a [`Number`] need not be copied.
    pub trait Element: Clone + PartialEq + fmt::Debug {
        /// 0.
        fn zero() -> Self;
        /// 1.
        fn one() -> Self;
        /// Whether it is 0.
        fn is_zero(&self) -> bool;
        /// The complex conjugate; the element itself when it is real.
        fn conj(&self) -> Self;
        /// `a + b`.
        fn add(a: &Self, b: &Self) -> Self;
        /// `a - b`.
        fn sub(a: &Self, b: &Self) -> Self;
        /// `a · b`.
        fn mul(a: &Self, b: &Self) -> Self;
        /// `-a`.
        fn neg(a: &Self) -> Self;
        /// `*a += b`.
        fn add_assign(a: &mut Self, b: &Self);
        /// `*a -= b`.
        fn sub_assign(a: &mut Self, b: &Self);
        /// `*a *= b`.
        fn mul_assign(a: &mut Self, b: &Self);
        /// `*c += a · b`.
        fn mul_add(c: &mut Self, a: &Self, b: &Self);

        /// `c = a · b` for column-major `a` of `m × k`, `b` of `k × n` and `c` of `m × n`, which
        /// holds zeros when it is called.
        fn product(c: &mut [Self], a: &[Self], b: &[Self], m: usize, k: usize, n: usize) {
            for j in 0..n {
                let c = &mut c[j * m..(j + 1) * m];
                for p in 0..k {
                    let x = &b[p + j * k];
                    if x.is_zero() {
                        continue;
                    }
                    for (c, a) in c.iter_mut().zip(&a[p * m..(p + 1) * m]) {
                        Self::mul_add(c, a, x);
                    }
                }
            }
        }
    }

    /// The field operations of floats and complex numbers, by value.
    pub trait Scalar:
        Element
        + Copy
        + Default
        + Add<Output = Self>
        + Sub<Output = Self>
        + Mul<Output = Self>
        + Div<Output = Self>
        + Neg<Output = Self>
        + AddAssign
        + SubAssign
        + MulAssign
        + DivAssign
        + Send
        + Sync
        + 'static
    {
        /// The real type of the absolute values, the public trait's `Real`.
        type R: Float;
        /// The complex type of the same precision, which eigenvalues of real matrices need.
        type C: super::Scalar<Real = Self::R>;
        /// The real type of the same precision as an element, for the real arithmetic inside
        /// complex algorithms.
        type Re: super::Scalar<Real = Self::R>;
        /// Whether the type is complex.
        const IS_COMPLEX: bool;
        /// The real part.
        fn re(self) -> Self::R;
        /// The imaginary part.
        fn im(self) -> Self::R;
        /// The number with the real part `re`.
        fn from_real(re: Self::R) -> Self;
        /// The number `re + im·i`, or `re` when the type is real.
        fn from_parts(re: Self::R, im: Self::R) -> Self;
        /// The absolute value.
        fn abs(self) -> Self::R;
        /// The square of the absolute value.
        fn abs2(self) -> Self::R;
        /// `|re| + |im|`, cheaper than the absolute value and within a factor √2 of it.
        fn abs1(self) -> Self::R;
        /// The product by a real number.
        fn scale(self, r: Self::R) -> Self;
        /// `self - a·b`, with fused multiply-adds: one for a real number, four for a complex one.
        fn mul_sub(self, a: Self, b: Self) -> Self;
        /// The principal square root.
        fn sqrt(self) -> Self;
        /// The complex number.
        fn to_complex(self) -> Complex<Self::R>;
        /// The element as its complex type.
        fn to_c(self) -> Self::C;
        /// The element of the complex type `c`, its real part for a real type.
        fn from_c(c: Self::C) -> Self;
        /// Whether both parts are finite.
        fn is_finite(self) -> bool;
        /// `c = β·c + α·a·b` by `gemm` on blocks of the real type `R`, which code generic over the
        /// element reaches no other way.
        #[allow(clippy::too_many_arguments)]
        fn gemm_real(
            m: usize,
            n: usize,
            k: usize,
            alpha: Self::R,
            a: &[Self::R],
            lda: usize,
            b: &[Self::R],
            ldb: usize,
            beta: Self::R,
            c: &mut [Self::R],
            ldc: usize,
        );
        /// [`gemm`](super::super::blas::gemm) on the type.
        #[allow(clippy::too_many_arguments)]
        fn gemm(
            m: usize,
            n: usize,
            k: usize,
            alpha: Self,
            a: &[Self],
            lda: usize,
            opa: Op,
            b: &[Self],
            ldb: usize,
            opb: Op,
            beta: Self,
            c: &mut [Self],
            ldc: usize,
        );
    }
}

/// A type a [`Matrix`](super::Matrix) can hold and compute with: `f32`, `f64`,
/// [`Complex<f32>`], [`Complex<f64>`], [`Number`], [`Rational`],
/// [`BigFloat`](crate::math::float::BigFloat) and [`BigComplex`](crate::math::float::BigComplex).
/// The trait is sealed.
pub trait Element: sealed::Element {}

/// An element whose matrices have the numerical algorithms, the decompositions, the eigenvalues
/// and the functions of matrices: `f32`, `f64`, [`Complex<f32>`] and [`Complex<f64>`]. The trait
/// is sealed.
pub trait Scalar: Element + sealed::Scalar<R = <Self as Scalar>::Real> {
    /// The float type of the parts and the absolute values, as of norms and singular values:
    /// `f32` for `f32` and `Complex<f32>`, `f64` for `f64` and `Complex<f64>`.
    type Real: Float;
}

impl Element for Number {}
impl Element for Rational {}

/// [`sealed::Element`] by the operators of a type of exact numbers, on references, with the
/// elements `$zero` and `$one`, and the matrix product `$product`.
macro_rules! exact_element {
    ($t:ty, $zero:expr, $one:expr, $product:path) => {
        impl sealed::Element for $t {
            #[inline]
            fn zero() -> Self {
                $zero
            }
            #[inline]
            fn one() -> Self {
                $one
            }
            #[inline]
            fn is_zero(&self) -> bool {
                <$t>::is_zero(self)
            }
            #[inline]
            fn conj(&self) -> Self {
                self.clone()
            }
            #[inline]
            fn add(a: &Self, b: &Self) -> Self {
                a + b
            }
            #[inline]
            fn sub(a: &Self, b: &Self) -> Self {
                a - b
            }
            #[inline]
            fn mul(a: &Self, b: &Self) -> Self {
                a * b
            }
            #[inline]
            fn neg(a: &Self) -> Self {
                -a
            }
            #[inline]
            fn add_assign(a: &mut Self, b: &Self) {
                *a += b;
            }
            #[inline]
            fn sub_assign(a: &mut Self, b: &Self) {
                *a -= b;
            }
            #[inline]
            fn mul_assign(a: &mut Self, b: &Self) {
                *a *= b;
            }
            #[inline]
            fn mul_add(c: &mut Self, a: &Self, b: &Self) {
                if !a.is_zero() {
                    *c += a * b;
                }
            }
            fn product(c: &mut [Self], a: &[Self], b: &[Self], m: usize, k: usize, n: usize) {
                $product(c, a, b, m, k, n);
            }
        }
    };
}

exact_element!(
    Rational,
    Rational::default(),
    Rational::from(1u8),
    super::rational::product
);

impl sealed::Element for Number {
    #[inline]
    fn zero() -> Self {
        Number::default()
    }
    #[inline]
    fn one() -> Self {
        Number::from(1u8)
    }
    #[inline]
    fn is_zero(&self) -> bool {
        Number::is_zero(self)
    }
    #[inline]
    fn conj(&self) -> Self {
        self.clone()
    }
    #[inline]
    fn add(a: &Self, b: &Self) -> Self {
        a + b
    }
    #[inline]
    fn sub(a: &Self, b: &Self) -> Self {
        a - b
    }
    #[inline]
    fn mul(a: &Self, b: &Self) -> Self {
        a * b
    }
    #[inline]
    fn neg(a: &Self) -> Self {
        -a
    }
    #[inline]
    fn add_assign(a: &mut Self, b: &Self) {
        *a += b;
    }
    #[inline]
    fn sub_assign(a: &mut Self, b: &Self) {
        *a -= b;
    }
    #[inline]
    fn mul_assign(a: &mut Self, b: &Self) {
        *a *= b;
    }
    #[inline]
    fn mul_add(c: &mut Self, a: &Self, b: &Self) {
        if !a.is_zero() {
            *c += a * b;
        }
    }
}

/// [`sealed::Scalar::gemm`] by the function `$f` of [`gemm`](super::gemm).
macro_rules! gemm {
    ($f:ident) => {
        #[inline]
        #[track_caller]
        fn gemm(
            m: usize,
            n: usize,
            k: usize,
            alpha: Self,
            a: &[Self],
            lda: usize,
            opa: Op,
            b: &[Self],
            ldb: usize,
            opb: Op,
            beta: Self,
            c: &mut [Self],
            ldc: usize,
        ) {
            super::gemm::$f(m, n, k, alpha, a, lda, opa, b, ldb, opb, beta, c, ldc);
        }
    };
}

/// [`sealed::Element`] by the operators of a `Copy` type.
macro_rules! copy_element {
    ($zero:expr, $one:expr, $conj:expr) => {
        #[inline]
        fn zero() -> Self {
            $zero
        }
        #[inline]
        fn one() -> Self {
            $one
        }
        #[inline]
        fn is_zero(&self) -> bool {
            *self == $zero
        }
        #[inline]
        fn conj(&self) -> Self {
            $conj(*self)
        }
        #[inline]
        fn add(a: &Self, b: &Self) -> Self {
            *a + *b
        }
        #[inline]
        fn sub(a: &Self, b: &Self) -> Self {
            *a - *b
        }
        #[inline]
        fn mul(a: &Self, b: &Self) -> Self {
            *a * *b
        }
        #[inline]
        fn neg(a: &Self) -> Self {
            -*a
        }
        #[inline]
        fn add_assign(a: &mut Self, b: &Self) {
            *a += *b;
        }
        #[inline]
        fn sub_assign(a: &mut Self, b: &Self) {
            *a -= *b;
        }
        #[inline]
        fn mul_assign(a: &mut Self, b: &Self) {
            *a *= *b;
        }
        #[inline]
        fn mul_add(c: &mut Self, a: &Self, b: &Self) {
            *c += *a * *b;
        }
        fn product(c: &mut [Self], a: &[Self], b: &[Self], m: usize, k: usize, n: usize) {
            gemm(m, n, k, $one, a, m, Op::N, b, k, Op::N, $zero, c, m);
        }
    };
}

macro_rules! real {
    ($t:ident) => {
        impl Element for $t {}
        impl Scalar for $t {
            type Real = $t;
        }

        impl sealed::Element for $t {
            copy_element!(0.0, 1.0, core::convert::identity);
        }

        impl sealed::Scalar for $t {
            type R = $t;
            type C = Complex<$t>;
            type Re = $t;
            const IS_COMPLEX: bool = false;
            #[inline]
            fn re(self) -> $t {
                self
            }
            #[inline]
            fn im(self) -> $t {
                0.0
            }
            #[inline]
            fn from_real(re: $t) -> Self {
                re
            }
            #[inline]
            fn from_parts(re: $t, _: $t) -> Self {
                re
            }
            #[inline]
            fn abs(self) -> $t {
                $t::abs(self)
            }
            #[inline]
            fn abs2(self) -> $t {
                self * self
            }
            #[inline]
            fn abs1(self) -> $t {
                $t::abs(self)
            }
            #[inline]
            fn scale(self, r: $t) -> Self {
                self * r
            }
            #[inline(always)]
            fn mul_sub(self, a: Self, b: Self) -> Self {
                $t::mul_add(-a, b, self)
            }
            #[inline]
            fn sqrt(self) -> Self {
                $t::sqrt(self)
            }
            #[inline]
            fn to_complex(self) -> Complex<$t> {
                Complex::new(self, 0.0)
            }
            #[inline]
            fn to_c(self) -> Complex<$t> {
                Complex::new(self, 0.0)
            }
            #[inline]
            fn from_c(c: Complex<$t>) -> Self {
                c.re
            }
            #[inline]
            fn is_finite(self) -> bool {
                $t::is_finite(self)
            }
            #[inline]
            fn gemm_real(
                m: usize,
                n: usize,
                k: usize,
                alpha: $t,
                a: &[$t],
                lda: usize,
                b: &[$t],
                ldb: usize,
                beta: $t,
                c: &mut [$t],
                ldc: usize,
            ) {
                gemm(m, n, k, alpha, a, lda, Op::N, b, ldb, Op::N, beta, c, ldc);
            }
            gemm!(real);
        }

        impl Element for Complex<$t> {}
        impl Scalar for Complex<$t> {
            type Real = $t;
        }

        impl sealed::Element for Complex<$t> {
            copy_element!(Complex::<$t>::ZERO, Complex::<$t>::ONE, Complex::conj);
        }

        impl sealed::Scalar for Complex<$t> {
            type R = $t;
            type C = Self;
            type Re = $t;
            const IS_COMPLEX: bool = true;
            #[inline]
            fn re(self) -> $t {
                self.re
            }
            #[inline]
            fn im(self) -> $t {
                self.im
            }
            #[inline]
            fn from_real(re: $t) -> Self {
                Complex::new(re, 0.0)
            }
            #[inline]
            fn from_parts(re: $t, im: $t) -> Self {
                Complex::new(re, im)
            }
            #[inline]
            fn abs(self) -> $t {
                Complex::abs(self)
            }
            #[inline]
            fn abs2(self) -> $t {
                self.norm_sqr()
            }
            #[inline]
            fn abs1(self) -> $t {
                self.re.abs() + self.im.abs()
            }
            #[inline]
            fn scale(self, r: $t) -> Self {
                self * r
            }
            #[inline(always)]
            fn mul_sub(self, a: Self, b: Self) -> Self {
                let re = $t::mul_add(a.im, b.im, $t::mul_add(-a.re, b.re, self.re));
                let im = $t::mul_add(-a.im, b.re, $t::mul_add(-a.re, b.im, self.im));
                Complex::new(re, im)
            }
            #[inline]
            fn sqrt(self) -> Self {
                Complex::sqrt(self)
            }
            #[inline]
            fn to_complex(self) -> Complex<$t> {
                self
            }
            #[inline]
            fn to_c(self) -> Self {
                self
            }
            #[inline]
            fn from_c(c: Self) -> Self {
                c
            }
            #[inline]
            fn is_finite(self) -> bool {
                Complex::is_finite(self)
            }
            #[inline]
            fn gemm_real(
                m: usize,
                n: usize,
                k: usize,
                alpha: $t,
                a: &[$t],
                lda: usize,
                b: &[$t],
                ldb: usize,
                beta: $t,
                c: &mut [$t],
                ldc: usize,
            ) {
                gemm(m, n, k, alpha, a, lda, Op::N, b, ldb, Op::N, beta, c, ldc);
            }
            gemm!(complex);
        }
    };
}

real!(f32);
real!(f64);
