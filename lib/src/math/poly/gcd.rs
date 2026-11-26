//! Greatest common divisors, resultants and discriminants.
//!
//! Over a field, Euclid's algorithm, with the half gcd of [`hgcd`] for the coefficient types whose
//! products are fast; over the rationals and the integers, the heuristic gcd and the multimodular
//! algorithm of [`zz`] on the primitive parts; modulo `n`, Euclid's algorithm on the word and limb
//! kernels of [`nmod`] and [`limbs`] below a degree and the half gcd above ([`modpoly`]); modulo 2
//! on the packed words, Euclid's algorithm below a degree and the half gcd of the engine of
//! [`gf2`] above.

mod field;
mod gf2;
mod hgcd;
mod limbs;
mod modpoly;
mod nmod;
mod subres;
mod zz;

use super::Polynomial;
use super::coeff::Field;
use crate::math::numbers::Number;
use crate::math::rational::Rational;
use field::FieldEngine;

/// The greatest common divisor of the packed polynomials `a` and `b` over GF(2) by the half gcd
/// on the packed products ([`gf2`]'s engine), packed and trimmed: the method of
/// [`arith::gf2::gcd`](super::arith::gf2::gcd) for long operands.
pub(crate) fn gf2_gcd(a: &[u64], b: &[u64]) -> Vec<u64> {
    gf2::gcd(a, b)
}

/// The monic greatest common divisor of the word polynomials `a` and `b` (trimmed) modulo `n ≥ 2`:
/// the engine of [`ModPolynomial::gcd`](super::ModPolynomial::gcd) for the factor area's inner
/// gcds; empty when both are.
pub(crate) fn word_gcd(n: u64, a: &[u64], b: &[u64]) -> Vec<u64> {
    if n == 2 {
        return super::arith::gf2::gcd_words(a, b);
    }
    let e = nmod::Word::new(n);
    let mut g = hgcd::gcd(&e, &a.to_vec(), &b.to_vec());
    if let Some(&l) = g.last()
        && l != 1
    {
        let li = e.m.inv_lead(l);
        for x in &mut g {
            *x = e.m.mul(*x, li);
        }
    }
    g
}

/// The monic greatest common divisor of the packed polynomials `a` and `b` (trimmed) modulo the
/// `m` of several limbs: as [`word_gcd`], for the factor area's moduli of several limbs.
pub(crate) fn limbs_gcd(m: &super::modular::Modulus, a: &[u64], b: &[u64]) -> Vec<u64> {
    let e = limbs::Limbs::of(m);
    let g = hgcd::gcd(&e, &a.to_vec(), &b.to_vec());
    e.monic(g)
}

/// The monic greatest common divisor of the polynomials of the coefficients `a` and `b` over a
/// field, by Euclid's algorithm: the default of the coefficients' kernel.
pub(crate) fn field_gcd<T: Field>(a: &[T], b: &[T]) -> Vec<T> {
    let e = FieldEngine::<T>::euclid();
    let g = hgcd::gcd(&e, &a.to_vec(), &b.to_vec());
    monic(g)
}

/// `(g, s, t)` of [`Polynomial::xgcd`] on vectors, by Euclid's algorithm.
pub(crate) fn field_xgcd<T: Field>(a: &[T], b: &[T]) -> (Vec<T>, Vec<T>, Vec<T>) {
    let e = FieldEngine::<T>::euclid();
    let (g, s, t) = hgcd::xgcd(&e, &a.to_vec(), &b.to_vec());
    match g.last() {
        None => (g, s, t),
        Some(l) => {
            let inv = T::inv(l);
            let sc = |v: Vec<T>| -> Vec<T> {
                let mut r: Vec<T> = v.iter().map(|x| T::mul(x, &inv)).collect();
                super::basic::trim(&mut r);
                r
            };
            (monic(g), sc(s), sc(t))
        }
    }
}

/// The resultant of the polynomials of the coefficients `a` and `b` over a field, from the
/// remainders of Euclid's sequence.
pub(crate) fn field_resultant<T: Field>(a: &[T], b: &[T]) -> T {
    let e = FieldEngine::<T>::euclid();
    modpoly::resultant_in(&e, &a.to_vec(), &b.to_vec())
}

/// `c` divided by its leading coefficient, the top exactly 1.
fn monic<T: Field>(mut c: Vec<T>) -> Vec<T> {
    if let Some(l) = c.last()
        && !l.is_one()
    {
        let inv = T::inv(l);
        for x in &mut c {
            T::mul_assign(x, &inv);
        }
        *c.last_mut().expect("not empty") = T::one();
    }
    c
}

/// The monic gcd over the rationals: that of the primitive integer polynomials.
pub(crate) fn rational_gcd(a: &[Rational], b: &[Rational]) -> Vec<Rational> {
    let (_, pa) = Polynomial::new(a.to_vec()).to_primitive();
    let (_, pb) = Polynomial::new(b.to_vec()).to_primitive();
    let g = pa.gcd(&pb);
    let r: Polynomial<Rational> = g.into();
    r.to_monic().into_coeffs()
}

/// `(g, s, t)` of [`Polynomial::xgcd`] over the rationals: `g` by [`rational_gcd`], and the
/// cofactors of the coprime primitive integer polynomials `a/g` and `b/g` multimodular
/// ([`zz::xgcd_coprime`]), `s·a' + t·b' = res(a', b')`, scaled back.
pub(crate) fn rational_xgcd(
    a: &[Rational],
    b: &[Rational],
) -> (Vec<Rational>, Vec<Rational>, Vec<Rational>) {
    let (pa, pb) = (Polynomial::new(a.to_vec()), Polynomial::new(b.to_vec()));
    if pa.is_zero() || pb.is_zero() {
        return field_xgcd(a, b);
    }
    let g = Polynomial::new(rational_gcd(a, b));
    // a = ca·g·a', b = cb·g·b' with a', b' primitive integer polynomials.
    let (ca, ia) = (&pa / &g).to_primitive();
    let (cb, ib) = (&pb / &g).to_primitive();
    let (da, db) = (
        ia.degree().expect("not zero"),
        ib.degree().expect("not zero"),
    );
    let (s, t) = if da == 0 {
        // a is g times a constant: g = a / (ca·a').
        let c = &ca * &Rational::from(ia.coeffs()[0].clone());
        (Polynomial::constant(c.recip()), Polynomial::zero())
    } else if db == 0 {
        let c = &cb * &Rational::from(ib.coeffs()[0].clone());
        (Polynomial::zero(), Polynomial::constant(c.recip()))
    } else {
        let (s, t, r) = zz::xgcd_coprime(ia.coeffs(), ib.coeffs());
        let r = Rational::from(r);
        let fs = (&r * &ca).recip();
        let ft = (&r * &cb).recip();
        (
            Polynomial::new(s).map(|x| &Rational::from(x.clone()) * &fs),
            Polynomial::new(t).map(|x| &Rational::from(x.clone()) * &ft),
        )
    };
    (g.into_coeffs(), s.into_coeffs(), t.into_coeffs())
}

/// The resultant over the rationals: `res(c·p, d·q) = c^deg q · d^deg p · res(p, q)` for the
/// primitive integer polynomials `p` and `q`.
pub(crate) fn rational_resultant(a: &[Rational], b: &[Rational]) -> Rational {
    let (ca, pa) = Polynomial::new(a.to_vec()).to_primitive();
    let (cb, pb) = Polynomial::new(b.to_vec()).to_primitive();
    let (Some(da), Some(db)) = (pa.degree(), pb.degree()) else {
        return Rational::default();
    };
    let r = Rational::from(pa.resultant(&pb));
    if r.is_zero() {
        return r;
    }
    &(&r * &ca.pow(db as i32)) * &cb.pow(da as i32)
}

impl<T: Field> Polynomial<T> {
    /// The greatest common divisor, monic; the zero polynomial when both are zero.
    ///
    /// Over the rationals, the greatest common divisor of the primitive integer polynomials (the
    /// heuristic gcd or the multimodular algorithm, as [`Polynomial::<Number>::gcd`]), made monic;
    /// over the floats, Euclid's algorithm.
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
        Polynomial::new(T::poly_gcd(&self.c, &other.c))
    }

    /// `(g, s, t)` with `g` the monic greatest common divisor and `g = s·self + t·other`, the
    /// degrees of `s` and `t` below those of `other / g` and `self / g`; `(0, 0, 0)` when both are
    /// zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::poly::Polynomial;
    /// use a::math::rational::Rational;
    ///
    /// let p = |c: &[i64]| Polynomial::new(c.iter().map(|&x| Rational::from(x)).collect());
    /// let (a, b) = (p(&[-2, 1, 1]), p(&[9, -12, 3]));
    /// let (g, s, t) = a.xgcd(&b);
    /// assert_eq!(g, p(&[-1, 1]));
    /// assert_eq!(&(&s * &a) + &(&t * &b), g);
    /// ```
    pub fn xgcd(&self, other: &Self) -> (Self, Self, Self) {
        let (g, s, t) = T::poly_xgcd(&self.c, &other.c);
        (Polynomial::new(g), Polynomial::new(s), Polynomial::new(t))
    }

    /// The least common multiple, monic; the zero polynomial when either is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::poly::Polynomial;
    /// use a::math::rational::Rational;
    ///
    /// let p = |c: &[i64]| Polynomial::new(c.iter().map(|&x| Rational::from(x)).collect());
    /// // lcm(2(x - 1), (x - 1)(x + 1)) = x² - 1
    /// assert_eq!(p(&[-2, 2]).lcm(&p(&[-1, 0, 1])), p(&[-1, 0, 1]));
    /// ```
    pub fn lcm(&self, other: &Self) -> Self {
        if self.is_zero() || other.is_zero() {
            return Polynomial::zero();
        }
        let g = self.gcd(other);
        (&(self / &g) * other).to_monic()
    }

    /// The resultant `res(self, other)`: the determinant of the Sylvester matrix,
    /// `lc(a)^deg b · Π b(α)` over the roots `α` of `a`; 0 when either is zero, 1 for two non-zero
    /// constants.
    ///
    /// Over the rationals, from that of the primitive integer polynomials (multimodular, as
    /// [`Polynomial::<Number>::resultant`]); over the floats, from the remainders of Euclid's
    /// sequence.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::poly::Polynomial;
    /// use a::math::rational::Rational;
    ///
    /// let p = |c: &[i64]| Polynomial::new(c.iter().map(|&x| Rational::from(x)).collect());
    /// // res(x² - 2, 2x - 1) = 4·((1/2)² - 2) = -7
    /// assert_eq!(p(&[-2, 0, 1]).resultant(&p(&[-1, 2])), Rational::from(-7));
    /// ```
    pub fn resultant(&self, other: &Self) -> T {
        T::poly_resultant(&self.c, &other.c)
    }

    /// The discriminant `(-1)^(d(d-1)/2) res(f, f') / lc(f)` for `f` of the degree `d ≥ 1`,
    /// `lc(f)^(2d - 2) Π (αᵢ - αⱼ)²` over the pairs of roots; 0 for a constant and for the zero
    /// polynomial.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::poly::Polynomial;
    ///
    /// // disc(x² + 3x + 2) = 9 - 8 = 1
    /// assert_eq!(Polynomial::new(vec![2.0, 3.0, 1.0]).discriminant(), 1.0);
    /// ```
    pub fn discriminant(&self) -> T {
        let Some(d) = self.degree() else {
            return T::zero();
        };
        if d == 0 {
            return T::zero();
        }
        let r = self.resultant(&self.derivative());
        let r = T::div(&r, self.leading().expect("not zero"));
        if (d * (d - 1) / 2) % 2 == 1 {
            T::neg(&r)
        } else {
            r
        }
    }
}

/// Up to this many coefficients in the shorter polynomial (and for coefficients of at least
/// [`SUBRESULTANT_BITS`] bits) the resultant over the integers is the last subresultant of the
/// chain, not multimodular: measured (on the arithmetic area's pseudo-division), the crossover falls
/// near degree 7 for 64 and 1000 bits.
pub(crate) const SUBRESULTANT_RESULTANT: usize = 7;
/// Below this many bits in the coefficients the multimodular resultant, of few primes, is faster
/// from degree 4 (measured at 24 bits).
pub(crate) const SUBRESULTANT_BITS: u64 = 40;

impl Polynomial<Number> {
    /// The greatest common divisor over the integers: the greatest common divisor of the contents
    /// times that of the primitive parts, with a positive leading coefficient; the zero
    /// polynomial when both are zero.
    ///
    /// The primitive parts' greatest common divisor is the heuristic gcd (the integer gcd of the
    /// values at a power of two, read back as a polynomial) while those integers are short, and
    /// otherwise, or when it gives up, the multimodular algorithm (the gcds modulo word primes,
    /// joined by the Chinese remainder theorem until they stop changing); either way the
    /// candidate is kept only when it divides both.
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
        Polynomial::new(zz::gcd(&self.c, &other.c))
    }

    /// The least common multiple over the integers, with a positive leading coefficient: the
    /// product divided by the greatest common divisor; the zero polynomial when either is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::poly::Polynomial;
    /// use a::math::numbers::Number;
    ///
    /// let p = |c: &[i64]| Polynomial::new(c.iter().map(|&x| Number::from(x)).collect());
    /// // lcm(2(x + 1), 3(x + 1)(x - 1)) = 6(x² - 1)
    /// assert_eq!(p(&[2, 2]).lcm(&p(&[-3, 0, 3])), p(&[-6, 0, 6]));
    /// ```
    pub fn lcm(&self, other: &Self) -> Self {
        if self.is_zero() || other.is_zero() {
            return Polynomial::zero();
        }
        let g = zz::ints(&self.gcd(other).c);
        let q = zz::divides(&zz::ints(&self.c), &g).expect("the gcd divides");
        let r = &Polynomial::new(zz::numbers(&q)) * other;
        if r.leading().is_some_and(Number::is_negative) {
            -&r
        } else {
            r
        }
    }

    /// The resultant `res(self, other)` over the integers: the determinant of the Sylvester
    /// matrix; 0 when either is zero, 1 for two non-zero constants.
    ///
    /// Multimodular: the resultants modulo word primes (by the half gcd past a degree), as many as
    /// Hadamard's bound `|a|₂^deg b · |b|₂^deg a` asks, joined by the Chinese remainder theorem.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::poly::Polynomial;
    /// use a::math::numbers::Number;
    ///
    /// let p = |c: &[i64]| Polynomial::new(c.iter().map(|&x| Number::from(x)).collect());
    /// // res(x² + 1, x - 3) = 10
    /// assert_eq!(p(&[1, 0, 1]).resultant(&p(&[-3, 1])), Number::from(10));
    /// ```
    pub fn resultant(&self, other: &Self) -> Number {
        let (Some(da), Some(db)) = (self.degree(), other.degree()) else {
            return Number::default();
        };
        if da == 0 {
            return self.c[0].pow(db as u32);
        }
        if db == 0 {
            return other.c[0].pow(da as u32);
        }
        let bits = self
            .c
            .iter()
            .chain(&other.c)
            .map(Number::bits)
            .max()
            .unwrap_or(0);
        if self.length().min(other.length()) <= SUBRESULTANT_RESULTANT && bits >= SUBRESULTANT_BITS
        {
            // The chain's sign convention: res(p, q) with deg p ≥ deg q.
            return if self.length() >= other.length() {
                subres::resultant(self, other)
            } else {
                let r = subres::resultant(other, self);
                if da & db & 1 == 1 { -r } else { r }
            };
        }
        zz::resultant(&self.c, &other.c)
    }

    /// The non-zero subresultants of `self` and `other`, the lowest index first: for `p` and
    /// `q` of the degrees `m ≥ n` (the two taken in that order), the subresultant `Sⱼ` of index
    /// `j < n` is the polynomial of the determinants of the Sylvester matrix's rows
    /// `x^(n-j-1)p, …, p, x^(m-j-1)q, …, q` cut to their top `m + n - 2j - 1` columns and one more;
    /// `S₀` is the resultant, and the last non-zero one is the greatest common divisor up to a
    /// constant. Those that vanish (inside the gaps that defective steps leave) are left out, as
    /// Sage's `subresultants` leaves them. Empty when either is zero; `[qᵐ]` for a constant `q`.
    ///
    /// The subresultant remainder sequence with Lazard's optimisation of the defective steps (as
    /// Ducos presents it): each pseudo-remainder divided exactly by the factor the sequence knows,
    /// and the subresultant after a gap of `δ` degrees, `lc(b)^(δ-1)·b / s^(δ-1)`, by Lazard's
    /// powering, whose intermediate values stay below the result.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::poly::Polynomial;
    /// use a::math::numbers::Number;
    ///
    /// let p = |c: &[i64]| Polynomial::new(c.iter().map(|&x| Number::from(x)).collect());
    /// // Knuth's example: x⁸ + x⁶ - 3x⁴ - 3x³ + 8x² + 2x - 5 and 3x⁶ + 5x⁴ - 4x² - 9x + 21.
    /// let f = p(&[-5, 2, 8, -3, -3, 0, 1, 0, 1]);
    /// let g = p(&[21, -9, -4, 0, 5, 0, 3]);
    /// let s = f.subresultants(&g);
    /// assert_eq!(s[0], p(&[260708]));
    /// assert_eq!(s[1], p(&[-12300, 9326]));
    /// assert_eq!(s[5], p(&[9, 0, -3, 0, 15]));
    /// assert_eq!(s[0].coeffs()[0], f.resultant(&g));
    /// ```
    pub fn subresultants(&self, other: &Self) -> Vec<Self> {
        let (p, q) = if self.length() >= other.length() {
            (self, other)
        } else {
            (other, self)
        };
        let (Some(m), Some(n)) = (p.degree(), q.degree()) else {
            return Vec::new();
        };
        if n == 0 {
            return vec![Polynomial::constant(q.c[0].pow(m as u32))];
        }
        subres::subresultants(p, q)
    }

    /// The discriminant `(-1)^(d(d-1)/2) res(f, f') / lc(f)` over the integers for `f` of the
    /// degree `d ≥ 1`; 0 for a constant and for the zero polynomial.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::poly::Polynomial;
    /// use a::math::numbers::Number;
    ///
    /// // disc(x³ + x + 1) = -4 - 27 = -31
    /// let f = Polynomial::new(vec![Number::from(1), Number::from(1), Number::from(0), Number::from(1)]);
    /// assert_eq!(f.discriminant(), Number::from(-31));
    /// ```
    pub fn discriminant(&self) -> Number {
        let Some(d) = self.degree() else {
            return Number::default();
        };
        if d == 0 {
            return Number::default();
        }
        let r = self.resultant(&self.derivative());
        let r = r.div_exact(self.leading().expect("not zero"));
        if (d * (d - 1) / 2) % 2 == 1 { -r } else { r }
    }
}
