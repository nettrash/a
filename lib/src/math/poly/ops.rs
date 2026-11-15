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
        // The longer one copied, the shorter added into it.
        let (mut r, s) = if self.c.len() >= rhs.c.len() {
            (self.clone(), rhs)
        } else {
            (rhs.clone(), self)
        };
        r += s;
        r
    }
}

impl<T: Coefficient> Sub<&Polynomial<T>> for &Polynomial<T> {
    type Output = Polynomial<T>;
    fn sub(self, rhs: &Polynomial<T>) -> Polynomial<T> {
        let mut r = self.clone();
        r -= rhs;
        r
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
