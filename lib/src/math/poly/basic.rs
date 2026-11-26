//! Construction, coefficients, evaluation by Horner's rule, the derivative and composition.

use super::Polynomial;
use super::coeff::Coefficient;

/// Drops the zero coefficients at the top of `c`.
#[inline]
pub(super) fn trim<T: Coefficient>(c: &mut Vec<T>) {
    while c.last().is_some_and(T::is_zero) {
        c.pop();
    }
}

impl<T: Coefficient> Default for Polynomial<T> {
    /// The zero polynomial.
    #[inline]
    fn default() -> Self {
        Polynomial::zero()
    }
}

impl<T: Coefficient> Polynomial<T> {
    /// The polynomial of the coefficients `coeffs`, the constant first: `Σ coeffs[i]·xⁱ`. Zeros at
    /// the top are dropped.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::poly::Polynomial;
    ///
    /// let p = Polynomial::new(vec![1.0, -2.0, 0.0]);
    /// assert_eq!(p.coeffs(), [1.0, -2.0]);
    /// assert_eq!(p.degree(), Some(1));
    /// ```
    #[inline]
    pub fn new(mut coeffs: Vec<T>) -> Self {
        trim(&mut coeffs);
        Polynomial { c: coeffs }
    }

    /// The polynomial of the coefficients `coeffs`, copied, the constant first.
    #[inline]
    pub fn from_slice(coeffs: &[T]) -> Self {
        Polynomial::new(coeffs.to_vec())
    }

    /// The zero polynomial, which has no coefficients and no degree.
    #[inline]
    pub fn zero() -> Self {
        Polynomial { c: Vec::new() }
    }

    /// The constant polynomial 1.
    #[inline]
    pub fn one() -> Self {
        Polynomial { c: vec![T::one()] }
    }

    /// The polynomial `x`.
    #[inline]
    pub fn x() -> Self {
        Polynomial {
            c: vec![T::zero(), T::one()],
        }
    }

    /// The constant polynomial `c`.
    #[inline]
    pub fn constant(c: T) -> Self {
        Polynomial::new(vec![c])
    }

    /// The monomial `c·xⁿ`.
    pub fn monomial(c: T, n: usize) -> Self {
        if c.is_zero() {
            return Polynomial::zero();
        }
        let mut v = vec![T::zero(); n + 1];
        v[n] = c;
        Polynomial { c: v }
    }

    /// The coefficients, the constant first, without zeros at the top; none for the zero
    /// polynomial.
    #[inline]
    pub fn coeffs(&self) -> &[T] {
        &self.c
    }

    /// The vector of the coefficients, the constant first.
    #[inline]
    pub fn into_coeffs(self) -> Vec<T> {
        self.c
    }

    /// The coefficient of `xⁱ`: 0 above the degree.
    #[inline]
    pub fn coeff(&self, i: usize) -> T {
        self.c.get(i).cloned().unwrap_or_else(T::zero)
    }

    /// The degree, `None` for the zero polynomial.
    #[inline]
    pub fn degree(&self) -> Option<usize> {
        self.c.len().checked_sub(1)
    }

    /// The number of coefficients up to the leading one, as FLINT's `fmpz_poly_length`: the degree
    /// plus one, 0 for the zero polynomial.
    #[inline]
    pub fn length(&self) -> usize {
        self.c.len()
    }

    /// Whether it is the zero polynomial.
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.c.is_empty()
    }

    /// Whether it is the constant 1.
    #[inline]
    pub fn is_one(&self) -> bool {
        self.c.len() == 1 && self.c[0].is_one()
    }

    /// Whether it is a constant, 0 included.
    #[inline]
    pub fn is_constant(&self) -> bool {
        self.c.len() <= 1
    }

    /// The leading coefficient, `None` for the zero polynomial.
    #[inline]
    pub fn leading(&self) -> Option<&T> {
        self.c.last()
    }

    /// Whether the leading coefficient is 1.
    #[inline]
    pub fn is_monic(&self) -> bool {
        self.c.last().is_some_and(T::is_one)
    }

    /// The value at `x`, by Horner's rule: a product and a sum for each coefficient below the
    /// leading one. Over the integers, from 32 coefficients, on limbs: Horner's rule on word
    /// blocks, and divide and conquer (`lo(x) + x^h·hi(x)` with the powers `x^(2^k)` squared once)
    /// when the value grows past a few thousand bits.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::poly::Polynomial;
    /// use a::math::rational::Rational;
    ///
    /// // 2x² - 3x + 1 at 1/2 is 0.
    /// let p = Polynomial::new(vec![Rational::from(1), Rational::from(-3), Rational::from(2)]);
    /// assert!(p.eval(&Rational::new(1.into(), 2.into())).is_zero());
    /// ```
    pub fn eval(&self, x: &T) -> T {
        T::poly_eval(&self.c, x)
    }

    /// The derivative.
    pub fn derivative(&self) -> Self {
        Polynomial::new(T::poly_derivative(&self.c))
    }

    /// The composition `self(g)`: by Horner's rule on polynomials for short ones, and by divide
    /// and conquer beyond, `self = lo + x^h hi` giving `lo(g) + g^h hi(g)` with the powers
    /// `g^(2^i)` computed once.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::poly::Polynomial;
    ///
    /// // (x² + 1)(x + 1) = x² + 2x + 2
    /// let p = Polynomial::new(vec![1.0, 0.0, 1.0]);
    /// let g = Polynomial::new(vec![1.0, 1.0]);
    /// assert_eq!(p.compose(&g).coeffs(), [2.0, 2.0, 1.0]);
    /// ```
    pub fn compose(&self, g: &Self) -> Self {
        if let Some(c) = T::poly_compose(&self.c, &g.c) {
            return Polynomial::new(c);
        }
        self.compose_dc(g)
    }

    /// Adds `c` to the constant coefficient.
    pub(super) fn add_constant(&mut self, c: &T) {
        if c.is_zero() {
            return;
        }
        match self.c.first_mut() {
            Some(c0) => T::add_assign(c0, c),
            None => self.c.push(c.clone()),
        }
        trim(&mut self.c);
    }

    /// The product by the constant `c`.
    pub fn scale(&self, c: &T) -> Self {
        if c.is_zero() {
            return Polynomial::zero();
        }
        Polynomial::new(self.c.iter().map(|x| T::mul(x, c)).collect())
    }

    /// The `n`-th power, by repeated squaring; `p⁰ = 1`.
    pub fn pow(&self, mut n: u32) -> Self {
        let mut r = Polynomial::one();
        let mut b = self.clone();
        while n > 0 {
            if n & 1 == 1 {
                r = &r * &b;
            }
            n >>= 1;
            if n > 0 {
                b = b.sqr();
            }
        }
        r
    }

    /// The square.
    pub fn sqr(&self) -> Self {
        if self.c.is_empty() {
            return Polynomial::zero();
        }
        Polynomial::new(T::poly_sqr(&self.c))
    }

    /// The product by `xⁿ`.
    pub fn shift_left(&self, n: usize) -> Self {
        if self.c.is_empty() {
            return Polynomial::zero();
        }
        let mut c = Vec::with_capacity(self.c.len() + n);
        c.resize(n, T::zero());
        c.extend_from_slice(&self.c);
        Polynomial { c }
    }

    /// The quotient by `xⁿ`: the coefficients of `xⁿ` and above, moved down.
    pub fn shift_right(&self, n: usize) -> Self {
        Polynomial {
            c: self.c.get(n..).map_or_else(Vec::new, <[T]>::to_vec),
        }
    }

    /// The remainder by `xⁿ`: the coefficients below `xⁿ`.
    pub fn truncate(&self, n: usize) -> Self {
        Polynomial::new(self.c[..n.min(self.c.len())].to_vec())
    }

    /// The reversal of the first `n` coefficients, `xⁿ⁻¹ · p(1/x)` for `p` of degree below `n`:
    /// the coefficients of `x⁰` to `xⁿ⁻¹` in the opposite order.
    pub fn reverse(&self, n: usize) -> Self {
        let mut c: Vec<T> = (0..n).map(|i| self.coeff(i)).collect();
        c.reverse();
        Polynomial::new(c)
    }

    /// The polynomial of the coefficients `f(c)` of each coefficient `c`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::poly::Polynomial;
    /// use a::math::numbers::Number;
    /// use a::math::rational::Rational;
    ///
    /// let p = Polynomial::new(vec![Number::from(2), Number::from(4)]);
    /// let q: Polynomial<Rational> = p.map(|c| Rational::new(c.clone(), 4.into()));
    /// assert_eq!(q.to_string(), "x + 1/2");
    /// ```
    pub fn map<U: Coefficient>(&self, f: impl FnMut(&T) -> U) -> Polynomial<U> {
        Polynomial::new(self.c.iter().map(f).collect())
    }
}
