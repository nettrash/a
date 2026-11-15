//! Greatest common divisors: Euclid's algorithm over a field, and over the integers the primitive
//! remainder sequence.

use super::Polynomial;
use super::coeff::Field;
use crate::math::numbers::Number;

impl<T: Field> Polynomial<T> {
    /// The greatest common divisor, monic; the zero polynomial when both are zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::poly::Polynomial;
    /// use a::math::rational::Rational;
    ///
    /// let p = |c: &[i64]| Polynomial::new(c.iter().map(|&x| Rational::from(x)).collect());
    /// // gcd((x - 1)(x + 2), 3(x - 1)(x - 3)) = x - 1
    /// let a = p(&[-2, 1, 1]);
    /// let b = p(&[9, -12, 3]);
    /// assert_eq!(a.gcd(&b), p(&[-1, 1]));
    /// ```
    pub fn gcd(&self, other: &Self) -> Self {
        let (mut a, mut b) = (self.clone(), other.clone());
        while !b.is_zero() {
            let r = &a % &b;
            a = b;
            b = r;
        }
        a.to_monic()
    }

    /// `(g, s, t)` with `g` the monic greatest common divisor and `g = s·self + t·other`, the
    /// degrees of `s` and `t` below those of `other / g` and `self / g`; `(0, 0, 0)` when both are
    /// zero.
    pub fn xgcd(&self, other: &Self) -> (Self, Self, Self) {
        let (mut a, mut b) = (self.clone(), other.clone());
        let (mut sa, mut sb) = (Polynomial::one(), Polynomial::zero());
        let (mut ta, mut tb) = (Polynomial::zero(), Polynomial::one());
        while !b.is_zero() {
            let (q, r) = a.div_rem(&b);
            let s = &sa - &(&q * &sb);
            let t = &ta - &(&q * &tb);
            (a, b) = (b, r);
            (sa, sb) = (sb, s);
            (ta, tb) = (tb, t);
        }
        match a.leading() {
            None => (a, Polynomial::zero(), Polynomial::zero()),
            Some(l) => {
                let inv = T::inv(l);
                (a.to_monic(), sa.scale(&inv), ta.scale(&inv))
            }
        }
    }
}

impl Polynomial<Number> {
    /// The greatest common divisor over the integers: the greatest common divisor of the contents
    /// times that of the primitive parts, with a positive leading coefficient; the zero
    /// polynomial when both are zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::poly::Polynomial;
    /// use a::math::numbers::Number;
    ///
    /// let p = |c: &[i64]| Polynomial::new(c.iter().map(|&x| Number::from(x)).collect());
    /// // gcd(6(x + 1)(x - 2), 4(x + 1)) = 2(x + 1)
    /// assert_eq!(p(&[-12, -6, 6]).gcd(&p(&[4, 4])), p(&[2, 2]));
    /// ```
    pub fn gcd(&self, other: &Self) -> Self {
        if self.is_zero() {
            return other.primitive_part().scale(&other.content());
        }
        if other.is_zero() {
            return self.primitive_part().scale(&self.content());
        }
        let c = self.content().gcd(&other.content());
        let (mut a, mut b) = (self.primitive_part(), other.primitive_part());
        if a.length() < b.length() {
            core::mem::swap(&mut a, &mut b);
        }
        while !b.is_zero() {
            let (_, r) = a.pseudo_div_rem(&b);
            a = b;
            b = r.primitive_part();
        }
        a.primitive_part().scale(&c)
    }
}
