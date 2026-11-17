//! Greatest common divisors, resultants and discriminants of [`ModPolynomial`]s: Euclid's
//! algorithm on the kernels of [`nmod`](super::nmod) and [`limbs`](super::limbs) below a degree,
//! the half gcd of [`hgcd`](super::hgcd) above.

use super::hgcd::{self, Engine};
use super::limbs::Limbs;
use super::nmod::Word;
use crate::math::numbers::Number;
use crate::math::poly::ModPolynomial;

impl ModPolynomial {
    /// The monic greatest common divisor; the zero polynomial when both are zero.
    ///
    /// Euclid's algorithm below a degree of a few hundred, with remainders that reduce once a
    /// coefficient for two steps of the division; above, the half gcd, which takes
    /// `O(M(n) log n)` operations.
    ///
    /// # Panics
    ///
    /// Panics if a leading coefficient that the algorithm divides by is not a unit modulo `n`
    /// (never for a prime `n`), or the moduli differ.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::poly::ModPolynomial;
    ///
    /// // Modulo 7: gcd((x + 1)(x + 2), (x + 1)(x + 3)) = x + 1
    /// let a = ModPolynomial::from_u64(&[2, 3, 1], 7);
    /// let b = ModPolynomial::from_u64(&[3, 4, 1], 7);
    /// assert_eq!(a.gcd(&b), ModPolynomial::from_u64(&[1, 1], 7));
    /// ```
    #[track_caller]
    pub fn gcd(&self, other: &ModPolynomial) -> ModPolynomial {
        self.same_modulus(other);
        let g = match self.m.word() {
            Some(n) => hgcd::gcd(&Word::new(n), &self.c, &other.c),
            None => hgcd::gcd(&Limbs::of(&self.m), &self.c, &other.c),
        };
        ModPolynomial::of(self.m.clone(), g).to_monic()
    }

    /// `(g, s, t)` with `g` the monic greatest common divisor and `g = s·self + t·other`, the
    /// degrees of `s` and `t` below those of `other / g` and `self / g`; `(0, 0, 0)` when both are
    /// zero.
    ///
    /// # Panics
    ///
    /// As [`gcd`](ModPolynomial::gcd).
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::poly::ModPolynomial;
    ///
    /// let a = ModPolynomial::from_u64(&[2, 3, 1], 7);
    /// let b = ModPolynomial::from_u64(&[3, 4, 1], 7);
    /// let (g, s, t) = a.xgcd(&b);
    /// assert_eq!(&(&s * &a) + &(&t * &b), g);
    /// assert_eq!(g, ModPolynomial::from_u64(&[1, 1], 7));
    /// ```
    #[track_caller]
    pub fn xgcd(&self, other: &ModPolynomial) -> (ModPolynomial, ModPolynomial, ModPolynomial) {
        self.same_modulus(other);
        let (g, s, t) = match self.m.word() {
            Some(n) => hgcd::xgcd(&Word::new(n), &self.c, &other.c),
            None => hgcd::xgcd(&Limbs::of(&self.m), &self.c, &other.c),
        };
        let of = |c| ModPolynomial::of(self.m.clone(), c);
        let (g, s, t) = (of(g), of(s), of(t));
        match g.degree() {
            None => (g, s, t),
            Some(d) => {
                let inv = self
                    .m
                    .inv(g.limbs(d))
                    .expect("the leading coefficient is not a unit modulo n");
                let inv = self.m.number(&inv);
                (g.scale(&inv), s.scale(&inv), t.scale(&inv))
            }
        }
    }

    /// The least common multiple, monic; the zero polynomial when either is zero.
    ///
    /// # Panics
    ///
    /// As [`gcd`](ModPolynomial::gcd).
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::poly::ModPolynomial;
    ///
    /// // Modulo 5: lcm(x² - 1, x - 1) = x² - 1
    /// let a = ModPolynomial::from_u64(&[4, 0, 1], 5);
    /// let b = ModPolynomial::from_u64(&[4, 1], 5);
    /// assert_eq!(a.lcm(&b), a);
    /// ```
    #[track_caller]
    pub fn lcm(&self, other: &ModPolynomial) -> ModPolynomial {
        self.same_modulus(other);
        if self.is_zero() || other.is_zero() {
            return self.zero_like();
        }
        let g = self.gcd(other);
        (&(self / &g) * other).to_monic()
    }

    /// The resultant `res(self, other)` modulo `n`, in `0..n`: the determinant of the Sylvester
    /// matrix, `lc(a)^deg b · Π b(α)` over the roots `α` of `a`; 0 when either is zero, and 1 for
    /// two non-zero constants.
    ///
    /// From the degrees and leading coefficients of the remainders of Euclid's sequence, which
    /// the half gcd records past a degree.
    ///
    /// # Panics
    ///
    /// Panics if a leading coefficient that the algorithm divides by is not a unit modulo `n`
    /// (never for a prime `n`), or the moduli differ.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::poly::ModPolynomial;
    /// use a::math::numbers::Number;
    ///
    /// // res(x² + 1, x - 2) = 2² + 1 = 5, modulo 7.
    /// let a = ModPolynomial::from_u64(&[1, 0, 1], 7);
    /// let b = ModPolynomial::from_u64(&[5, 1], 7);
    /// assert_eq!(a.resultant(&b), Number::from(5));
    /// ```
    #[track_caller]
    pub fn resultant(&self, other: &ModPolynomial) -> Number {
        self.same_modulus(other);
        match self.m.word() {
            Some(n) => {
                let e = Word::new(n);
                Number::from(resultant_in(&e, &self.c, &other.c))
            }
            None => {
                let e = Limbs::of(&self.m);
                self.m.number(&resultant_in(&e, &self.c, &other.c))
            }
        }
    }

    /// The discriminant modulo `n`, in `0..n`: `(-1)^(d(d-1)/2) res(f, f') / lc(f)` for `f` of
    /// the degree `d ≥ 1`, the resultant taken with `f'` of the formal degree `d - 1` (so that it is
    /// `lc(f)^(2d - 2) Π (αᵢ - αⱼ)²` over the pairs of roots, also when `n` divides `d`); 0 for a
    /// constant and for the zero polynomial.
    ///
    /// # Panics
    ///
    /// As [`resultant`](ModPolynomial::resultant), and if the leading coefficient is not a unit.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::poly::ModPolynomial;
    /// use a::math::numbers::Number;
    ///
    /// // disc(x² + x + 1) = 1 - 4 = -3 = 8 modulo 11.
    /// let f = ModPolynomial::from_u64(&[1, 1, 1], 11);
    /// assert_eq!(f.discriminant(), Number::from(8));
    /// ```
    #[track_caller]
    pub fn discriminant(&self) -> Number {
        let Some(d) = self.degree() else {
            return Number::default();
        };
        if d == 0 {
            return Number::default();
        }
        let df = self.derivative();
        let res = self.resultant(&df);
        if res.is_zero() {
            return res;
        }
        let lc = self.leading().expect("not zero");
        let n = self.modulus();
        let e = df.degree().expect("a non-zero resultant");
        // The formal degree d - 1 of f': lc^(d - 1 - deg f') more.
        let mut r = &res * &lc.pow_mod(&Number::from((d - 1 - e) as u64), n);
        let inv = lc
            .mod_inverse(n)
            .expect("the leading coefficient is not a unit modulo n");
        r = (&r * &inv).rem_euclid(n);
        if (d * (d - 1) / 2) % 2 == 1 && !r.is_zero() {
            r = n - &r;
        }
        r
    }
}

/// The resultant of the polynomials `a` and `b` of the engine `e`: the cases of zeros, constants
/// and degrees in the wrong order, and [`hgcd::resultant`].
pub(crate) fn resultant_in<E: Engine>(e: &E, a: &E::P, b: &E::P) -> E::C {
    let (la, lb) = (e.len(a), e.len(b));
    if la == 0 || lb == 0 {
        return e.czero();
    }
    if la == 1 {
        return e.cpow(&e.lead(a), (lb - 1) as u64);
    }
    if lb == 1 {
        return e.cpow(&e.lead(b), (la - 1) as u64);
    }
    if la < lb {
        let r = hgcd::resultant(e, b, a);
        return if (la - 1) & (lb - 1) & 1 == 1 {
            e.cneg(&r)
        } else {
            r
        };
    }
    hgcd::resultant(e, a, b)
}
