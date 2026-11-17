//! Division of polynomials over a field by Newton's method, products, powers and compositions
//! modulo a polynomial, composition by divide and conquer, and the pseudo-division and exact
//! division over the integers by Kronecker substitution into the division of integers.

use super::super::Polynomial;
use super::super::basic::trim;
use super::super::coeff::{Coefficient, Field};
use super::zmul;
use crate::math::numbers::Number;

/// From this many coefficients in the quotient and in the divisor, a division over a field by
/// Newton's method; below, the schoolbook method. The rationals take the integers' division
/// instead (see `div_rem`), so this is for the floats.
pub(crate) const NEWTON_MIN: usize = if cfg!(miri) { 12 } else { 512 };

/// Up to this many coefficients, a power series over a field is inverted by the schoolbook
/// recurrence.
const INV_SCHOOL: usize = if cfg!(miri) { 6 } else { 32 };

impl<T: Coefficient> Polynomial<T> {
    /// The coefficients below `x^n`, as a polynomial.
    pub(crate) fn low(&self, n: usize) -> Self {
        self.truncate(n)
    }

    /// The composition `self(g)` by divide and conquer: `self = lo + x^h hi` gives `lo(g) + g^h
    /// hi(g)`, the powers `g^(2^i)` computed once, Horner's rule for short pieces.
    pub(crate) fn compose_dc(&self, g: &Self) -> Self {
        let n = self.c.len();
        if n <= 8 || g.c.len() <= 1 {
            return self.compose_horner(g);
        }
        let mut pows = vec![g.clone()];
        while (1usize << pows.len()) < n {
            let next = pows.last().expect("not empty").sqr();
            pows.push(next);
        }
        self.compose_part(0, n, &pows)
    }

    /// Horner's rule on polynomials.
    pub(crate) fn compose_horner(&self, g: &Self) -> Self {
        let mut it = self.c.iter().rev();
        let Some(top) = it.next() else {
            return Polynomial::zero();
        };
        let mut v = Polynomial::constant(top.clone());
        for c in it {
            v = &v * g;
            v.add_constant(c);
        }
        v
    }

    /// The composition of the coefficients `lo..lo + len` with `pows[0]`, `pows[i] = g^(2^i)`.
    fn compose_part(&self, lo: usize, len: usize, pows: &[Self]) -> Self {
        if len <= 8 {
            let piece = Polynomial::new(self.c[lo..lo + len].to_vec());
            return piece.compose_horner(&pows[0]);
        }
        let e = (usize::BITS - 1 - (len - 1).leading_zeros()) as usize;
        let h = 1usize << e;
        let a = self.compose_part(lo, h, pows);
        let b = self.compose_part(lo + h, len - h, pows);
        &a + &(&b * &pows[e])
    }
}

impl<T: Field> Polynomial<T> {
    /// The inverse of the power series modulo `x^n`: `g` with `self·g ≡ 1 (mod x^n)`, of at most `n`
    /// coefficients; the zero polynomial for `n = 0`. By the schoolbook recurrence for short
    /// series and Newton's iteration `g ← g + g (1 - self·g)` beyond, which doubles the correct
    /// coefficients: about three products of the length `n`.
    ///
    /// # Panics
    ///
    /// Panics if the constant coefficient is 0 (for `n > 0`).
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::poly::Polynomial;
    /// use a::math::rational::Rational;
    ///
    /// // 1 / (1 - x/2) = 1 + x/2 + x²/4 + …
    /// let p = Polynomial::new(vec![Rational::from(1), Rational::new((-1).into(), 2.into())]);
    /// let g = p.inv_series(3);
    /// assert_eq!(g.to_string(), "1/4*x^2 + 1/2*x + 1");
    /// ```
    #[track_caller]
    pub fn inv_series(&self, n: usize) -> Self {
        if n == 0 {
            return Polynomial::zero();
        }
        let c0 = self.c.first().filter(|c| !c.is_zero());
        let c0 = c0.expect("the constant coefficient is 0");
        if let Some(g) = T::poly_inv_series(self, n) {
            return g;
        }
        let inv = T::inv(c0);
        self.inv_newton(n, &inv)
    }

    /// The inverse modulo `x^n` by the schoolbook recurrence alone.
    pub(crate) fn inv_series_school(&self, n: usize) -> Self {
        let inv = T::inv(&self.c[0]);
        let mut g: Vec<T> = Vec::with_capacity(n);
        g.push(inv.clone());
        let ninv = T::neg(&inv);
        for j in 1..n {
            let mut s = T::zero();
            for i in 1..=j.min(self.c.len().saturating_sub(1)) {
                T::mul_add(&mut s, &self.c[i], &g[j - i]);
            }
            g.push(T::mul(&s, &ninv));
        }
        Polynomial::new(g)
    }

    /// [`inv_series`](Polynomial::inv_series) with `inv` the inverse of the constant.
    fn inv_newton(&self, n: usize, inv: &T) -> Self {
        if n <= INV_SCHOOL {
            // g_j = -inv Σ_(i=1..j) f_i g_(j-i).
            let mut g: Vec<T> = Vec::with_capacity(n);
            g.push(inv.clone());
            let ninv = T::neg(inv);
            for j in 1..n {
                let mut s = T::zero();
                for i in 1..=j.min(self.c.len().saturating_sub(1)) {
                    T::mul_add(&mut s, &self.c[i], &g[j - i]);
                }
                g.push(T::mul(&s, &ninv));
            }
            return Polynomial::new(g);
        }
        let h = n.div_ceil(2);
        let g = self.inv_newton(h, inv);
        let e = (&self.low(n) * &g).low(n);
        let t = e.shift_right(h);
        let u = (&t * &g).low(n - h);
        &g - &u.shift_left(h)
    }

    /// The quotient and the remainder by Newton's method: the reversed quotient is the reversed
    /// dividend times the inverse of the reversed divisor modulo `x^(la - ld + 1)`.
    pub(crate) fn div_rem_newton(&self, d: &Self, inv: &T) -> (Self, Self) {
        let (la, ld) = (self.c.len(), d.c.len());
        let lq = la - ld + 1;
        let dinv = d.reverse(ld).inv_newton(lq, inv);
        let q = (&self.reverse(la).low(lq) * &dinv).low(lq).reverse(lq);
        let r = &self.low(ld - 1) - &(&q * &d.low(ld - 1)).low(ld - 1);
        (q, r)
    }

    /// `self·b mod f`.
    ///
    /// # Panics
    ///
    /// Panics if `f` is the zero polynomial.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::poly::Polynomial;
    ///
    /// // x·x mod (x² + 1) = -1
    /// let x = Polynomial::new(vec![0.0, 1.0]);
    /// let f = Polynomial::new(vec![1.0, 0.0, 1.0]);
    /// assert_eq!(x.mul_mod(&x, &f).coeffs(), [-1.0]);
    /// ```
    #[track_caller]
    pub fn mul_mod(&self, b: &Self, f: &Self) -> Self {
        &(self * b) % f
    }

    /// `selfᵉ mod f`, by repeated squaring modulo `f`; `self⁰ mod f = 1 mod f`.
    ///
    /// # Panics
    ///
    /// Panics if `f` is the zero polynomial or `e` is negative.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::Number;
    /// use a::math::poly::Polynomial;
    /// use a::math::rational::Rational;
    ///
    /// // x^4 mod (x² + 1) = 1
    /// let p = |c: &[i64]| Polynomial::new(c.iter().map(|&x| Rational::from(x)).collect());
    /// assert_eq!(p(&[0, 1]).pow_mod(&Number::from(4), &p(&[1, 0, 1])), p(&[1]));
    /// ```
    #[track_caller]
    pub fn pow_mod(&self, e: &Number, f: &Self) -> Self {
        assert!(!f.is_zero(), "division by the zero polynomial");
        assert!(!e.is_negative(), "a negative exponent");
        if let Some(r) = T::poly_pow_mod(self, e, f) {
            return r;
        }
        let b = self % f;
        let mut r = &Polynomial::one() % f;
        for i in (0..e.bits()).rev() {
            r = r.sqr() % f;
            if e.bit(i) {
                r = (&r * &b) % f;
            }
        }
        r
    }

    /// The composition `self(g) mod f`, by Brent and Kung's method: with `m ≈ √(deg self)`, the
    /// powers `g⁰…gᵐ` modulo `f`, each block of `m` coefficients evaluated at `g` as a combination
    /// of them, and the blocks joined by Horner's rule in `gᵐ`.
    ///
    /// # Panics
    ///
    /// Panics if `f` is the zero polynomial.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::poly::Polynomial;
    ///
    /// // (x² + x)(x + 1) = x² + 3x + 2 ≡ 3x + 1 modulo x² + 1
    /// let p = Polynomial::new(vec![0.0, 1.0, 1.0]);
    /// let g = Polynomial::new(vec![1.0, 1.0]);
    /// let f = Polynomial::new(vec![1.0, 0.0, 1.0]);
    /// assert_eq!(p.compose_mod(&g, &f).coeffs(), [1.0, 3.0]);
    /// ```
    #[track_caller]
    pub fn compose_mod(&self, g: &Self, f: &Self) -> Self {
        assert!(!f.is_zero(), "division by the zero polynomial");
        if f.c.len() == 1 || self.is_zero() {
            return Polynomial::zero();
        }
        let la = self.c.len();
        let g = g % f;
        let m = ((la as f64).sqrt().ceil() as usize).max(1);
        let mut pows = vec![&Polynomial::one() % f];
        for i in 1..=m {
            let next = if i == 1 {
                g.clone()
            } else {
                (&pows[i - 1] * &g) % f
            };
            pows.push(next);
        }
        let gm = pows.pop().expect("m >= 1");
        let mut r = Polynomial::zero();
        for j in (0..la.div_ceil(m)).rev() {
            // The block's value at g: Σ c_i g^(i - jm).
            let width = f.c.len() - 1;
            let mut v = vec![T::zero(); width];
            for i in j * m..((j + 1) * m).min(la) {
                let c = &self.c[i];
                if c.is_zero() {
                    continue;
                }
                for (o, x) in v.iter_mut().zip(pows[i - j * m].coeffs()) {
                    T::mul_add(o, c, x);
                }
            }
            trim(&mut v);
            r = (&r * &gm) % f;
            r += &Polynomial { c: v };
        }
        r
    }
}

impl Polynomial<Number> {
    /// The bits of the field of Kronecker substitution that holds the pseudo-quotient and
    /// remainder of `a·l^k` (`a` scaled already, `bits_a` its largest coefficient's bits) by `d`
    /// of leading coefficient `l`: `|q_j| <= (|a| + n H |q|_max) / |l|` makes the quotient grow
    /// by at most `log2(1 + n H / |l|)` bits a coefficient from the top, and the remainder is at
    /// most `|a| + n H |q|`; two more bits for the signs and one for the rounding.
    fn pseudo_field(bits_a: usize, d: &[Number], lq: usize) -> usize {
        let n = d.len();
        let h = zmul::max_bits(d);
        let l = d.last().map_or(1, |l| l.bits() as usize);
        // log2(1 + n H / |l|) <= bits(n) + h - l + 2.
        let grow = (zmul::bits_of(n) + h + 2).saturating_sub(l.saturating_sub(1));
        let q = bits_a + lq * grow + 1;
        q + h + zmul::bits_of(n) + 4
    }

    /// The pseudo-quotient and pseudo-remainder by Kronecker substitution: `l^k a` and `d`
    /// evaluated at `2^s` for a field `s` that holds the quotient and the remainder with their
    /// signs, the quotient `q(2^s)` the nearest integer to the quotient of the two values (the
    /// remainder's value is below half the divisor's), read back, and `r = l^k a - q d`.
    pub(crate) fn pseudo_div_rem_ks(&self, d: &Self) -> (Self, Self) {
        let (m, n) = (self.c.len(), d.c.len());
        let lq = m - n + 1;
        let l = d.c.last().expect("not zero");
        let lk = l.pow(lq as u32);
        let a: Vec<Number> = self.c.iter().map(|c| c * &lk).collect();
        let s = Self::pseudo_field(zmul::max_bits(&a), &d.c, lq);
        let (q, r) = Self::divide_at(&a, &d.c, s);
        (q, r)
    }

    /// `(q, a - q d)` for the quotient `q` read from the nearest integer to `a(2^s) / d(2^s)` in
    /// fields of `s` bits.
    fn divide_at(a: &[Number], d: &[Number], s: usize) -> (Self, Self) {
        let (m, n) = (a.len(), d.len());
        let lq = m - n + 1;
        let mut x = zmul::value(a, s);
        let mut y = zmul::value(d, s);
        if y.is_negative() {
            x = -x;
            y = -y;
        }
        // round(x / y) = floor((2x + y) / 2y).
        let two = Number::from(2u8);
        let qv = (&(&x * &two) + &y).div_euclid(&(&y * &two));
        let q = Polynomial::new(zmul::digits(&qv, s, lq));
        let prod = &q * &Polynomial::new(d.to_vec());
        let mut r: Vec<Number> = a.to_vec();
        for (x, y) in r.iter_mut().zip(prod.c.iter()) {
            *x -= y;
        }
        (q, Polynomial::new(r))
    }

    /// The quotient by `d` when it is exact, `None` otherwise, by Kronecker substitution: with a
    /// field `s`, the values at `2^s` divide when the polynomials do; the quotient read back
    /// from the integers' quotient is checked by a product, and `s` doubled while it fails, up to
    /// Mignotte's bound on the coefficients of a divisor of `self` (`2^(deg q) ‖self‖₂`).
    pub(crate) fn div_exact_ks(&self, d: &Self) -> Option<Self> {
        let (m, n) = (self.c.len(), d.c.len());
        let lq = m - n + 1;
        let bits_a = zmul::max_bits(&self.c);
        let norm_bits = bits_a + zmul::bits_of(m) / 2 + 1;
        let limit = lq + norm_bits + 2;
        // The values need fields wider than every coefficient with its sign; the quotient's are
        // usually about those of self less those of the leading coefficient of d.
        let bits_d = zmul::max_bits(&d.c);
        // A divisor's coefficients are below Mignotte's bound too (2^(deg d)·‖self‖₂): a wider d
        // divides nothing.
        if bits_d > n + norm_bits + 2 {
            return None;
        }
        let wide = bits_a.max(bits_d) + 2;
        let guess = bits_a.saturating_sub(zmul::max_bits(&d.c[n - 1..])) + zmul::bits_of(m) + 34;
        let mut s = wide.max(guess);
        // The widest field tried: one that holds the quotient's bound and every coefficient of
        // both operands, so that the last check is conclusive.
        let cap = (limit + zmul::bits_of(lq) + 2).max(wide);
        let x = |s: usize| zmul::value(&self.c, s);
        loop {
            let s_now = s.min(cap);
            let (mut xv, mut yv) = (x(s_now), zmul::value(&d.c, s_now));
            if yv.is_negative() {
                xv = -xv;
                yv = -yv;
            }
            let (qv, rv) = xv.div_rem(&yv);
            if !rv.is_zero() {
                return None;
            }
            let q = Polynomial::new(zmul::digits(&qv, s_now, lq));
            if q.c.len() == lq && &q * d == *self {
                return Some(q);
            }
            if s_now >= cap {
                return None;
            }
            s *= 2;
        }
    }
}
