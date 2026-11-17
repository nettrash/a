//! Division with a remainder over a field, pseudo-division and exact division over the integers,
//! contents and primitive parts.

use super::Polynomial;
use super::basic::trim;
use super::coeff::Field;
use crate::math::numbers::Number;
use crate::math::rational::Rational;
use core::ops::{Div, DivAssign, Rem, RemAssign};

/// The quotient and the remainder of `a` by `d` of `n ≥ 1` coefficients with `inv` the inverse of
/// its leading coefficient, by the schoolbook method: `a.len() - n + 1` coefficients of the
/// quotient (none when `a` is shorter), and the remainder in `a`, cut to `n - 1` and trimmed.
pub(super) fn schoolbook_div_rem<T: Field>(a: &mut Vec<T>, d: &[T], inv: &T) -> Vec<T> {
    let n = d.len();
    if a.len() < n {
        trim(a);
        return Vec::new();
    }
    let m = a.len();
    let mut q = vec![T::zero(); m - n + 1];
    for i in (0..=m - n).rev() {
        let t = T::mul(&a[i + n - 1], inv);
        if !t.is_zero() {
            for (r, d) in a[i..i + n - 1].iter_mut().zip(d) {
                T::mul_sub(r, &t, d);
            }
        }
        q[i] = t;
    }
    a.truncate(n - 1);
    trim(a);
    q
}

impl<T: Field> Polynomial<T> {
    /// The quotient `q` and the remainder `r` of the division by `d`: `self = q·d + r` with `r` of
    /// a degree below that of `d`. Over the rationals, from a quotient of 16 coefficients, through
    /// the integers: both brought to common denominators and the pseudo-division of their
    /// numerators by Kronecker substitution, each coefficient then brought to lowest terms once;
    /// over the floats by the schoolbook method, and by Newton's method (the reversed divisor's
    /// inverse as a power series) from 512 coefficients in the quotient and the divisor.
    ///
    /// # Panics
    ///
    /// Panics if `d` is the zero polynomial.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::poly::Polynomial;
    ///
    /// // x³ + 2x + 1 = (x - 1)(x² + x + 3) + 4
    /// let a = Polynomial::new(vec![1.0, 2.0, 0.0, 1.0]);
    /// let d = Polynomial::new(vec![-1.0, 1.0]);
    /// let (q, r) = a.div_rem(&d);
    /// assert_eq!(q.coeffs(), [3.0, 1.0, 1.0]);
    /// assert_eq!(r.coeffs(), [4.0]);
    /// ```
    #[track_caller]
    pub fn div_rem(&self, d: &Self) -> (Self, Self) {
        let lead = d.c.last().expect("division by the zero polynomial");
        let inv = T::inv(lead);
        if let Some(qr) = T::poly_div_rem(self, d, &inv) {
            return qr;
        }
        let mut r = self.c.clone();
        let q = schoolbook_div_rem(&mut r, &d.c, &inv);
        (Polynomial::new(q), Polynomial { c: r })
    }

    /// The polynomial divided by its leading coefficient; the zero polynomial stays.
    pub fn to_monic(&self) -> Self {
        match self.c.last() {
            None => Polynomial::zero(),
            Some(l) if l.is_one() => self.clone(),
            Some(l) => {
                let inv = T::inv(l);
                let mut c: Vec<T> = self.c.iter().map(|x| T::mul(x, &inv)).collect();
                // Exactly 1 at the top, whatever a float's rounding made of it.
                *c.last_mut().expect("not empty") = T::one();
                Polynomial { c }
            }
        }
    }
}

/// `Div` and `Rem` on a field's polynomials and references to them.
macro_rules! division {
    ($tr:ident, $f:ident, $atr:ident, $af:ident, $part:tt) => {
        impl<T: Field> $tr<&Polynomial<T>> for &Polynomial<T> {
            type Output = Polynomial<T>;
            #[track_caller]
            fn $f(self, rhs: &Polynomial<T>) -> Polynomial<T> {
                self.div_rem(rhs).$part
            }
        }
        impl<T: Field> $tr<&Polynomial<T>> for Polynomial<T> {
            type Output = Polynomial<T>;
            #[track_caller]
            fn $f(self, rhs: &Polynomial<T>) -> Polynomial<T> {
                (&self).$f(rhs)
            }
        }
        impl<T: Field> $tr<Polynomial<T>> for &Polynomial<T> {
            type Output = Polynomial<T>;
            #[track_caller]
            fn $f(self, rhs: Polynomial<T>) -> Polynomial<T> {
                self.$f(&rhs)
            }
        }
        impl<T: Field> $tr for Polynomial<T> {
            type Output = Polynomial<T>;
            #[track_caller]
            fn $f(self, rhs: Polynomial<T>) -> Polynomial<T> {
                (&self).$f(&rhs)
            }
        }
        impl<T: Field> $atr<&Polynomial<T>> for Polynomial<T> {
            #[track_caller]
            fn $af(&mut self, rhs: &Polynomial<T>) {
                *self = (&*self).$f(rhs);
            }
        }
        impl<T: Field> $atr for Polynomial<T> {
            #[track_caller]
            fn $af(&mut self, rhs: Polynomial<T>) {
                *self = (&*self).$f(&rhs);
            }
        }
    };
}

division!(Div, div, DivAssign, div_assign, 0);
division!(Rem, rem, RemAssign, rem_assign, 1);

impl Polynomial<Number> {
    /// The pseudo-quotient `q` and pseudo-remainder `r` of the division by `d`, of the leading
    /// coefficient `l` and the degree `n`, for `self` of the degree `m`: `lᵏ·self = q·d + r` with
    /// `k = max(m - n + 1, 0)` and `r` of a degree below `n`, all with integer coefficients.
    ///
    /// By the schoolbook method on limbs, each coefficient multiplied by the powers of `l` it
    /// misses only when a step changes it, and from 384 coefficients in the quotient (32 for
    /// divisors of 1,000 bits) by the schoolbook method with the fewest factors of `l` (a step
    /// whose top coefficient `l` divides needs none, as in FLINT, so an exact or nearly exact
    /// division stays the size of its operands, its results multiplied by the missing powers of
    /// `l` at the end) while at most four steps need one, otherwise by Kronecker substitution: `lᵏ·self` and `d` evaluated at `2^s`
    /// for a field `s` that holds every coefficient of the quotient and of the remainder (bounded
    /// from `|q_j| ≤ (|lᵏ a| + n H |q|) / |l|`), the quotient's value the nearest integer to the
    /// quotient of the two values, read back, and the remainder by one product.
    ///
    /// # Panics
    ///
    /// Panics if `d` is the zero polynomial.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::poly::Polynomial;
    /// use a::math::numbers::Number;
    ///
    /// let p = |c: &[i64]| Polynomial::new(c.iter().map(|&x| Number::from(x)).collect());
    /// // 2²·(x² + 1) = (2x - 1)·(2x + 1) + 5
    /// let (q, r) = p(&[1, 0, 1]).pseudo_div_rem(&p(&[1, 2]));
    /// assert_eq!((q, r), (p(&[-1, 2]), p(&[5])));
    /// ```
    #[track_caller]
    pub fn pseudo_div_rem(&self, d: &Self) -> (Self, Self) {
        let l = d.c.last().expect("division by the zero polynomial");
        let n = d.c.len();
        let m = self.c.len();
        if m < n {
            return (Polynomial::zero(), self.clone());
        }
        if n == 1 {
            // lᵏ·self = (lᵏ⁻¹·self)·l: one power and a product a coefficient.
            return (
                Polynomial::new(super::arith::zdiv::pseudo_by_constant(&self.c, l)),
                Polynomial::zero(),
            );
        }
        if super::arith::pseudo_by_ks(self, d) {
            // The schoolbook method with the fewest factors of l first: an exact or nearly exact
            // division keeps the size of its operands there, where the substitution's field must
            // hold the largest coefficient of the pseudo-quotient.
            if let Some((q, r)) =
                super::arith::zdiv::pseudo_div_rem_lazy(&self.c, &d.c, super::arith::LAZY_FACTORS)
            {
                return (Polynomial::new(q), Polynomial::new(r));
            }
            return self.pseudo_div_rem_ks(d);
        }
        if super::arith::limb_school(m, n) {
            let (q, r) = super::arith::zdiv::pseudo_div_rem(&self.c, &d.c);
            return (Polynomial::new(q), Polynomial::new(r));
        }
        let mut r = self.c.clone();
        let mut q = vec![Number::default(); m - n + 1];
        let monic = crate::math::rational::is_one(l);
        for i in (0..=m - n).rev() {
            let t = core::mem::take(&mut r[i + n - 1]);
            if !monic {
                for x in &mut q[i + 1..] {
                    *x *= l;
                }
                for x in &mut r[..i + n - 1] {
                    *x *= l;
                }
            }
            if !t.is_zero() {
                for (x, y) in r[i..i + n - 1].iter_mut().zip(&d.c) {
                    *x -= &t * y;
                }
            }
            q[i] = t;
        }
        r.truncate(n - 1);
        trim(&mut r);
        (Polynomial::new(q), Polynomial { c: r })
    }

    /// The quotient `self / d` when it has integer coefficients and no remainder, `None`
    /// otherwise.
    ///
    /// By the schoolbook method on limbs, each quotient coefficient the top one divided by the
    /// leading coefficient of `d`, and from 12 coefficients in the quotient (48 for divisors of
    /// 512 bits) by Kronecker substitution: the values at `2^s` divide when the polynomials do, so
    /// a remainder of the integers says `None`; the quotient read back from theirs is checked by a
    /// product, and `s` doubled while the check fails, up to Mignotte's bound `2^(deg q)·‖self‖₂`
    /// on the coefficients of a divisor.
    ///
    /// # Panics
    ///
    /// Panics if `d` is the zero polynomial.
    #[track_caller]
    pub fn div_exact(&self, d: &Self) -> Option<Self> {
        let l = d.c.last().expect("division by the zero polynomial");
        let n = d.c.len();
        let m = self.c.len();
        if m == 0 {
            return Some(Polynomial::zero());
        }
        if m < n {
            return None;
        }
        if super::arith::exact_by_ks(self, d) {
            return self.div_exact_ks(d);
        }
        if super::arith::limb_school(m, n) {
            return super::arith::zdiv::div_exact(&self.c, &d.c).map(Polynomial::new);
        }
        let mut r = self.c.clone();
        let mut q = vec![Number::default(); m - n + 1];
        for i in (0..=m - n).rev() {
            let top = &r[i + n - 1];
            let t = if crate::math::rational::is_one(l) {
                top.clone()
            } else {
                let (t, rem) = top.div_rem(l);
                if !rem.is_zero() {
                    return None;
                }
                t
            };
            if !t.is_zero() {
                for (x, y) in r[i..i + n - 1].iter_mut().zip(&d.c) {
                    *x -= &t * y;
                }
            }
            q[i] = t;
        }
        r[..n - 1]
            .iter()
            .all(Number::is_zero)
            .then(|| Polynomial::new(q))
    }

    /// The content: the greatest common divisor of the coefficients, not negative; 0 for the zero
    /// polynomial.
    pub fn content(&self) -> Number {
        let mut g = Number::default();
        for c in &self.c {
            g = g.gcd(c);
            if crate::math::rational::is_one(&g) {
                break;
            }
        }
        g
    }

    /// The primitive part: the polynomial divided by its content, with a positive leading
    /// coefficient; the zero polynomial stays.
    pub fn primitive_part(&self) -> Self {
        let Some(l) = self.c.last() else {
            return Polynomial::zero();
        };
        let mut g = self.content();
        if l.is_negative() {
            g = -g;
        }
        if crate::math::rational::is_one(&g) {
            return self.clone();
        }
        Polynomial {
            c: self.c.iter().map(|c| c / &g).collect(),
        }
    }
}

impl Polynomial<Rational> {
    /// `(c, p)` with `self = c·p`, `p` primitive with integer coefficients and a positive leading
    /// coefficient, and `c` rational: `(0, 0)` for the zero polynomial.
    pub fn to_primitive(&self) -> (Rational, Polynomial<Number>) {
        if self.c.is_empty() {
            return (Rational::default(), Polynomial::zero());
        }
        let mut den = Number::from(1u8);
        for c in &self.c {
            let d = c.denom();
            let g = den.gcd(d);
            den = &den / &g * d;
        }
        let ints: Vec<Number> = self
            .c
            .iter()
            .map(|c| c.numer() * &(&den / c.denom()))
            .collect();
        let ints = Polynomial::new(ints);
        let p = ints.primitive_part();
        let mut g = ints.content();
        if ints.c.last().is_some_and(Number::is_negative) {
            g = -g;
        }
        (Rational::new(g, den), p)
    }
}

impl From<&Polynomial<Number>> for Polynomial<Rational> {
    /// The polynomial of the same coefficients, as rationals.
    fn from(p: &Polynomial<Number>) -> Self {
        Polynomial {
            c: p.c.iter().map(Rational::from).collect(),
        }
    }
}

impl From<Polynomial<Number>> for Polynomial<Rational> {
    /// The polynomial of the same coefficients, as rationals.
    fn from(p: Polynomial<Number>) -> Self {
        Polynomial {
            c: p.c.into_iter().map(Rational::from).collect(),
        }
    }
}
