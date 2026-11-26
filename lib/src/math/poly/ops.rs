//! `+`, `-`, `*` and their assignments, on polynomials and references to them.

use super::Polynomial;
use super::basic::trim;
use super::coeff::Coefficient;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

impl<T: Coefficient> AddAssign<&Polynomial<T>> for Polynomial<T> {
    fn add_assign(&mut self, rhs: &Polynomial<T>) {
        let a = &mut self.c;
        if a.len() < rhs.c.len() {
            a.resize(rhs.c.len(), T::zero());
        }
        for (x, y) in a.iter_mut().zip(&rhs.c) {
            T::add_assign(x, y);
        }
        trim(a);
    }
}

impl<T: Coefficient> SubAssign<&Polynomial<T>> for Polynomial<T> {
    fn sub_assign(&mut self, rhs: &Polynomial<T>) {
        let a = &mut self.c;
        if a.len() < rhs.c.len() {
            a.resize(rhs.c.len(), T::zero());
        }
        for (x, y) in a.iter_mut().zip(&rhs.c) {
            T::sub_assign(x, y);
        }
        trim(a);
    }
}

impl<T: Coefficient> MulAssign<&Polynomial<T>> for Polynomial<T> {
    fn mul_assign(&mut self, rhs: &Polynomial<T>) {
        *self = &*self * rhs;
    }
}

impl<T: Coefficient> Mul<&Polynomial<T>> for &Polynomial<T> {
    type Output = Polynomial<T>;
    /// The product, by the method the coefficients call for: over the integers by Kronecker
    /// substitution into the integers' multiplication (Karatsuba's, Toom's and the
    /// number-theoretic transforms by the lengths), or from 48 coefficients of at most 112 bits
    /// by the number-theoretic transforms on the coefficients with the Chinese remainder theorem
    /// for signed values; over the rationals the same on common
    /// denominators, each coefficient then in lowest terms; over the floats and the complex
    /// floats by the schoolbook method, with the componentwise error `γ_m Σ |a_i| |b_(k-i)|`
    /// (`γ_m = m ε / (1 − m ε)`, `ε = 2⁻⁵³`), while the shorter operand has fewer than 32
    /// coefficients or `la·lb ≤ c·N·log₂ N` (`c` = 19 for floats, 5 for complex floats, `N` the
    /// least power of 2 at least `la + lb − 1`: square products up to 441 coefficients, 101
    /// complex), and beyond by the crate's linear convolution
    /// ([`fft::convolve`](crate::math::fft::convolve)), by fast Fourier transforms within a few
    /// units of rounding times `‖a‖₂ ‖b‖₂ log₂ N` for a transform of length `N`, in norm rather
    /// than componentwise (an infinity or a NaN may then make every coefficient NaN).
    fn mul(self, rhs: &Polynomial<T>) -> Polynomial<T> {
        if self.c.is_empty() || rhs.c.is_empty() {
            return Polynomial::zero();
        }
        if core::ptr::eq(self, rhs) {
            return self.sqr();
        }
        Polynomial::new(T::poly_mul(&self.c, &rhs.c))
    }
}

impl<T: Coefficient> Add<&Polynomial<T>> for &Polynomial<T> {
    type Output = Polynomial<T>;
    fn add(self, rhs: &Polynomial<T>) -> Polynomial<T> {
        // Each coefficient of the sum built once, as the sum of two references: a copy added
        // into in place would grow (and move) about half of the integers' vectors.
        let (long, short) = if self.c.len() >= rhs.c.len() {
            (&self.c, &rhs.c)
        } else {
            (&rhs.c, &self.c)
        };
        let mut c = Vec::with_capacity(long.len());
        c.extend(long.iter().zip(short).map(|(x, y)| T::add(x, y)));
        c.extend_from_slice(&long[short.len()..]);
        trim(&mut c);
        Polynomial { c }
    }
}

impl<T: Coefficient> Sub<&Polynomial<T>> for &Polynomial<T> {
    type Output = Polynomial<T>;
    fn sub(self, rhs: &Polynomial<T>) -> Polynomial<T> {
        let (a, b) = (&self.c, &rhs.c);
        let n = a.len().min(b.len());
        let mut c = Vec::with_capacity(a.len().max(b.len()));
        c.extend(a.iter().zip(b).map(|(x, y)| T::sub(x, y)));
        c.extend_from_slice(&a[n..]);
        // 0 − b_i, as `-=` takes it (for the floats +0 from +0, where the negation is −0).
        c.extend(b[n..].iter().map(T::zero_sub));
        trim(&mut c);
        Polynomial { c }
    }
}

impl<T: Coefficient> Neg for &Polynomial<T> {
    type Output = Polynomial<T>;
    fn neg(self) -> Polynomial<T> {
        Polynomial {
            c: self.c.iter().map(T::neg).collect(),
        }
    }
}

impl<T: Coefficient> Neg for Polynomial<T> {
    type Output = Polynomial<T>;
    fn neg(mut self) -> Polynomial<T> {
        for x in &mut self.c {
            *x = T::neg(x);
        }
        self
    }
}

impl<T: Coefficient> AddAssign for Polynomial<T> {
    #[inline]
    fn add_assign(&mut self, rhs: Polynomial<T>) {
        *self += &rhs;
    }
}

impl<T: Coefficient> SubAssign for Polynomial<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: Polynomial<T>) {
        *self -= &rhs;
    }
}

impl<T: Coefficient> MulAssign for Polynomial<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: Polynomial<T>) {
        *self *= &rhs;
    }
}

impl<T: Coefficient> Add<&Polynomial<T>> for Polynomial<T> {
    type Output = Polynomial<T>;
    #[inline]
    fn add(mut self, rhs: &Polynomial<T>) -> Polynomial<T> {
        self += rhs;
        self
    }
}

impl<T: Coefficient> Add<Polynomial<T>> for &Polynomial<T> {
    type Output = Polynomial<T>;
    #[inline]
    fn add(self, mut rhs: Polynomial<T>) -> Polynomial<T> {
        rhs += self;
        rhs
    }
}

impl<T: Coefficient> Add for Polynomial<T> {
    type Output = Polynomial<T>;
    #[inline]
    fn add(self, rhs: Polynomial<T>) -> Polynomial<T> {
        if self.c.len() >= rhs.c.len() {
            self + &rhs
        } else {
            rhs + &self
        }
    }
}

impl<T: Coefficient> Sub<&Polynomial<T>> for Polynomial<T> {
    type Output = Polynomial<T>;
    #[inline]
    fn sub(mut self, rhs: &Polynomial<T>) -> Polynomial<T> {
        self -= rhs;
        self
    }
}

impl<T: Coefficient> Sub<Polynomial<T>> for &Polynomial<T> {
    type Output = Polynomial<T>;
    fn sub(self, rhs: Polynomial<T>) -> Polynomial<T> {
        // `-(rhs - self)`, in the vector of `rhs`.
        let mut r = rhs;
        r -= self;
        -r
    }
}

impl<T: Coefficient> Sub for Polynomial<T> {
    type Output = Polynomial<T>;
    #[inline]
    fn sub(self, rhs: Polynomial<T>) -> Polynomial<T> {
        self - &rhs
    }
}

impl<T: Coefficient> Mul<&Polynomial<T>> for Polynomial<T> {
    type Output = Polynomial<T>;
    #[inline]
    fn mul(self, rhs: &Polynomial<T>) -> Polynomial<T> {
        &self * rhs
    }
}

impl<T: Coefficient> Mul<Polynomial<T>> for &Polynomial<T> {
    type Output = Polynomial<T>;
    #[inline]
    fn mul(self, rhs: Polynomial<T>) -> Polynomial<T> {
        self * &rhs
    }
}

impl<T: Coefficient> Mul for Polynomial<T> {
    type Output = Polynomial<T>;
    #[inline]
    fn mul(self, rhs: Polynomial<T>) -> Polynomial<T> {
        &self * &rhs
    }
}
