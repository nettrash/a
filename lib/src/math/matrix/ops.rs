//! The operators of matrices.

use super::element::sealed::Element as Ring;
use super::{Element, Matrix, Scalar};
use crate::math::complex::Complex;
use crate::math::numbers::Number;
use crate::math::rational::Rational;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// Panics unless the two matrices have the same shape.
#[inline]
#[track_caller]
fn same_shape<T>(a: &Matrix<T>, b: &Matrix<T>) {
    if a.shape() != b.shape() {
        different_shapes();
    }
}

#[cold]
#[inline(never)]
#[track_caller]
fn different_shapes() -> ! {
    panic!("the matrices are not of the same shape")
}

/// `a · b`.
#[track_caller]
fn product<T: Element>(a: &Matrix<T>, b: &Matrix<T>) -> Matrix<T> {
    assert!(
        a.cols == b.rows,
        "the columns of the left matrix are not as many as the rows of the right one"
    );
    let mut c = Matrix::zeros(a.rows, b.cols);
    T::product(&mut c.data, &a.data, &b.data, a.rows, a.cols, b.cols);
    c
}

/// An elementwise operator between two matrices, owned or borrowed, and its assignment.
macro_rules! elementwise {
    ($op:ident, $method:ident, $assign:ident, $assign_method:ident, $f:ident, $f_assign:ident) => {
        impl<T: Element> $op<&Matrix<T>> for &Matrix<T> {
            type Output = Matrix<T>;
            #[track_caller]
            fn $method(self, rhs: &Matrix<T>) -> Matrix<T> {
                same_shape(self, rhs);
                Matrix {
                    rows: self.rows,
                    cols: self.cols,
                    data: self
                        .data
                        .iter()
                        .zip(&rhs.data)
                        .map(|(a, b)| T::$f(a, b))
                        .collect(),
                }
            }
        }
        impl<T: Element> $op<&Matrix<T>> for Matrix<T> {
            type Output = Matrix<T>;
            #[track_caller]
            fn $method(mut self, rhs: &Matrix<T>) -> Matrix<T> {
                self.$assign_method(rhs);
                self
            }
        }
        impl<T: Element> $op<Matrix<T>> for Matrix<T> {
            type Output = Matrix<T>;
            #[track_caller]
            fn $method(mut self, rhs: Matrix<T>) -> Matrix<T> {
                self.$assign_method(&rhs);
                self
            }
        }
        impl<T: Element> $op<Matrix<T>> for &Matrix<T> {
            type Output = Matrix<T>;
            #[track_caller]
            fn $method(self, mut rhs: Matrix<T>) -> Matrix<T> {
                same_shape(self, &rhs);
                for (b, a) in rhs.data.iter_mut().zip(&self.data) {
                    *b = T::$f(a, b);
                }
                rhs
            }
        }
        impl<T: Element> $assign<&Matrix<T>> for Matrix<T> {
            #[track_caller]
            fn $assign_method(&mut self, rhs: &Matrix<T>) {
                same_shape(self, rhs);
                for (a, b) in self.data.iter_mut().zip(&rhs.data) {
                    T::$f_assign(a, b);
                }
            }
        }
        impl<T: Element> $assign<Matrix<T>> for Matrix<T> {
            #[track_caller]
            fn $assign_method(&mut self, rhs: Matrix<T>) {
                self.$assign_method(&rhs);
            }
        }
    };
}

elementwise!(Add, add, AddAssign, add_assign, add, add_assign);
elementwise!(Sub, sub, SubAssign, sub_assign, sub, sub_assign);

impl<T: Element> Neg for &Matrix<T> {
    type Output = Matrix<T>;
    fn neg(self) -> Matrix<T> {
        self.map(T::neg)
    }
}

impl<T: Element> Neg for Matrix<T> {
    type Output = Matrix<T>;
    fn neg(mut self) -> Matrix<T> {
        for x in &mut self.data {
            *x = T::neg(x);
        }
        self
    }
}

impl<T: Element> Mul<&Matrix<T>> for &Matrix<T> {
    type Output = Matrix<T>;
    /// The matrix product.
    ///
    /// # Panics
    ///
    /// Panics if the columns of `self` are not as many as the rows of `rhs`.
    #[track_caller]
    fn mul(self, rhs: &Matrix<T>) -> Matrix<T> {
        product(self, rhs)
    }
}

impl<T: Element> Mul<Matrix<T>> for &Matrix<T> {
    type Output = Matrix<T>;
    #[track_caller]
    fn mul(self, rhs: Matrix<T>) -> Matrix<T> {
        product(self, &rhs)
    }
}

impl<T: Element> Mul<&Matrix<T>> for Matrix<T> {
    type Output = Matrix<T>;
    #[track_caller]
    fn mul(self, rhs: &Matrix<T>) -> Matrix<T> {
        product(&self, rhs)
    }
}

impl<T: Element> Mul<Matrix<T>> for Matrix<T> {
    type Output = Matrix<T>;
    #[track_caller]
    fn mul(self, rhs: Matrix<T>) -> Matrix<T> {
        product(&self, &rhs)
    }
}

impl<T: Element> MulAssign<&Matrix<T>> for Matrix<T> {
    #[track_caller]
    fn mul_assign(&mut self, rhs: &Matrix<T>) {
        *self = product(self, rhs);
    }
}

impl<T: Element> MulAssign<Matrix<T>> for Matrix<T> {
    #[track_caller]
    fn mul_assign(&mut self, rhs: Matrix<T>) {
        *self = product(self, &rhs);
    }
}

impl<T: Element> Mul<&T> for &Matrix<T> {
    type Output = Matrix<T>;
    /// The product of every element by `rhs`.
    fn mul(self, rhs: &T) -> Matrix<T> {
        self.map(|x| T::mul(x, rhs))
    }
}

impl<T: Element> Mul<T> for &Matrix<T> {
    type Output = Matrix<T>;
    fn mul(self, rhs: T) -> Matrix<T> {
        self * &rhs
    }
}

impl<T: Element> Mul<&T> for Matrix<T> {
    type Output = Matrix<T>;
    fn mul(mut self, rhs: &T) -> Matrix<T> {
        self *= rhs;
        self
    }
}

impl<T: Element> Mul<T> for Matrix<T> {
    type Output = Matrix<T>;
    fn mul(mut self, rhs: T) -> Matrix<T> {
        self *= &rhs;
        self
    }
}

impl<T: Element> MulAssign<&T> for Matrix<T> {
    fn mul_assign(&mut self, rhs: &T) {
        for x in &mut self.data {
            T::mul_assign(x, rhs);
        }
    }
}

impl<T: Element> MulAssign<T> for Matrix<T> {
    fn mul_assign(&mut self, rhs: T) {
        *self *= &rhs;
    }
}

impl<T: Scalar> Div<T> for &Matrix<T> {
    type Output = Matrix<T>;
    /// The quotient of every element by `rhs`.
    fn div(self, rhs: T) -> Matrix<T> {
        self.map(|&x| x / rhs)
    }
}

impl<T: Scalar> Div<T> for Matrix<T> {
    type Output = Matrix<T>;
    fn div(mut self, rhs: T) -> Matrix<T> {
        self /= rhs;
        self
    }
}

impl<T: Scalar> DivAssign<T> for Matrix<T> {
    fn div_assign(&mut self, rhs: T) {
        for x in &mut self.data {
            *x /= rhs;
        }
    }
}

impl Div<&Rational> for &Matrix<Rational> {
    type Output = Matrix<Rational>;
    /// The quotient of every element by `rhs`.
    ///
    /// # Panics
    ///
    /// Panics if `rhs` is zero and the matrix is not empty.
    #[track_caller]
    fn div(self, rhs: &Rational) -> Matrix<Rational> {
        self.map(|x| x / rhs)
    }
}

impl Div<Rational> for &Matrix<Rational> {
    type Output = Matrix<Rational>;
    #[track_caller]
    fn div(self, rhs: Rational) -> Matrix<Rational> {
        self / &rhs
    }
}

impl Div<&Rational> for Matrix<Rational> {
    type Output = Matrix<Rational>;
    #[track_caller]
    fn div(mut self, rhs: &Rational) -> Matrix<Rational> {
        self /= rhs;
        self
    }
}

impl Div<Rational> for Matrix<Rational> {
    type Output = Matrix<Rational>;
    #[track_caller]
    fn div(mut self, rhs: Rational) -> Matrix<Rational> {
        self /= &rhs;
        self
    }
}

impl DivAssign<&Rational> for Matrix<Rational> {
    /// Divides every element by `rhs`.
    ///
    /// # Panics
    ///
    /// Panics if `rhs` is zero and the matrix is not empty.
    #[track_caller]
    fn div_assign(&mut self, rhs: &Rational) {
        for x in &mut self.data {
            *x /= rhs;
        }
    }
}

impl DivAssign<Rational> for Matrix<Rational> {
    #[track_caller]
    fn div_assign(&mut self, rhs: Rational) {
        *self /= &rhs;
    }
}

/// A scalar on the left of a matrix, which the orphan rule leaves to one implementation for each
/// type of element.
macro_rules! left {
    ($t:ty) => {
        impl Mul<&Matrix<$t>> for &$t {
            type Output = Matrix<$t>;
            fn mul(self, rhs: &Matrix<$t>) -> Matrix<$t> {
                rhs.map(|x| <$t as Ring>::mul(self, x))
            }
        }
        impl Mul<Matrix<$t>> for &$t {
            type Output = Matrix<$t>;
            fn mul(self, mut rhs: Matrix<$t>) -> Matrix<$t> {
                for x in &mut rhs.data {
                    *x = <$t as Ring>::mul(self, x);
                }
                rhs
            }
        }
        impl Mul<&Matrix<$t>> for $t {
            type Output = Matrix<$t>;
            fn mul(self, rhs: &Matrix<$t>) -> Matrix<$t> {
                &self * rhs
            }
        }
        impl Mul<Matrix<$t>> for $t {
            type Output = Matrix<$t>;
            fn mul(self, rhs: Matrix<$t>) -> Matrix<$t> {
                &self * rhs
            }
        }
    };
}

left!(f32);
left!(f64);
left!(Complex<f32>);
left!(Complex<f64>);
left!(Number);
left!(Rational);
