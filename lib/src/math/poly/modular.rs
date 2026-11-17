//! Polynomials with coefficients modulo an integer `n ≥ 2`, which they hold once.
//!
//! A coefficient is a residue below `n` of `k` limbs, `k` those of `n`, the least significant
//! first; the coefficients lie one after the other in one vector, the constant first, without
//! zero coefficients at the top. A modulus of one limb, `n < 2^64`, makes each coefficient a word.

use super::Polynomial;
use crate::math::numbers::{ByteOrder, Number, from_limbs, limbs_of, nat};
use core::cmp::Ordering;
use core::fmt;
use core::hash::{Hash, Hasher};
use core::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};
use std::sync::Arc;

/// The modulus of a [`ModPolynomial`], which its copies and the polynomials made from it share.
#[derive(Debug)]
pub(crate) struct Modulus {
    /// `n`, in little-endian order.
    pub(crate) n: Number,
    /// The `k` limbs of `n`, the least significant first, the top one not zero.
    pub(crate) d: Vec<u64>,
    /// The arithmetic modulo `n` when it is a word.
    pub(crate) w: Option<super::arith::nmod::WordMod>,
    /// The division by `n` when it has several limbs.
    pub(crate) l: Option<super::arith::lred::LimbRed>,
}

/// The working memory of the products of residues.
#[derive(Default)]
pub(crate) struct Work {
    /// A product of two residues.
    p: Vec<u64>,
    /// A quotient, thrown away.
    q: Vec<u64>,
    /// A remainder.
    r: Vec<u64>,
    /// The scratch of the products and divisions of limbs.
    s: Vec<u64>,
}

/// `a` compared with `b`, of the same length, from the top limb down.
#[inline]
fn cmp_limbs(a: &[u64], b: &[u64]) -> Ordering {
    for (x, y) in a.iter().rev().zip(b.iter().rev()) {
        match x.cmp(y) {
            Ordering::Equal => continue,
            o => return o,
        }
    }
    Ordering::Equal
}

impl Modulus {
    /// The modulus `n`.
    ///
    /// # Panics
    ///
    /// Panics if `n` is below 2.
    #[track_caller]
    pub(crate) fn new(n: &Number) -> Modulus {
        assert!(
            !n.is_negative() && n > &Number::from(1u8),
            "the modulus must be at least 2"
        );
        let d = limbs_of(n);
        Modulus {
            n: from_limbs(&d, ByteOrder::LittleEndian),
            w: (d.len() == 1).then(|| super::arith::nmod::WordMod::new(d[0])),
            l: (d.len() >= 2).then(|| super::arith::lred::LimbRed::new(&d)),
            d,
        }
    }

    /// The number of limbs of a residue.
    #[inline]
    pub(crate) fn k(&self) -> usize {
        self.d.len()
    }

    /// The arithmetic modulo `n` when it fits in a word.
    #[inline]
    pub(crate) fn word_mod(&self) -> Option<&super::arith::nmod::WordMod> {
        self.w.as_ref()
    }

    /// `n` when it fits in a word.
    #[inline]
    pub(crate) fn word(&self) -> Option<u64> {
        (self.d.len() == 1).then(|| self.d[0])
    }

    /// The residue of `x`, of `k` limbs.
    pub(crate) fn residue(&self, x: &Number) -> Vec<u64> {
        if let Some(l) = &self.l {
            let mut r = vec![0u64; self.k()];
            l.reduce(&limbs_of(x), &mut r);
            if x.is_negative() && r.iter().any(|&v| v != 0) {
                nat::sub_from(&mut r, &self.d);
            }
            return r;
        }
        let r = x.rem_euclid(&self.n);
        let mut l = limbs_of(&r);
        l.resize(self.k(), 0);
        l
    }

    /// The residue of the limbs `x`, of any length, without zeros at the top or with them, into
    /// `out` of `k` limbs.
    pub(crate) fn reduce(&self, out: &mut [u64], x: &[u64], w: &mut Work) {
        if let Some(l) = &self.l {
            l.reduce(x, out);
            return;
        }
        let n = x.iter().rposition(|&l| l != 0).map_or(0, |i| i + 1);
        out.fill(0);
        if n < self.k() || (n == self.k() && cmp_limbs(&x[..n], &self.d) == Ordering::Less) {
            out[..n].copy_from_slice(&x[..n]);
            return;
        }
        nat::div_rem_into(&mut w.q, &mut w.r, &x[..n], &self.d, &mut w.s);
        out[..w.r.len()].copy_from_slice(&w.r);
    }

    /// The number of the residue `r`.
    #[inline]
    pub(crate) fn number(&self, r: &[u64]) -> Number {
        from_limbs(r, ByteOrder::LittleEndian)
    }

    /// `r = r + a mod n`.
    #[inline]
    pub(crate) fn add(&self, r: &mut [u64], a: &[u64]) {
        if let Some(n) = self.word() {
            let (s, c) = r[0].overflowing_add(a[0]);
            r[0] = if c || s >= n { s.wrapping_sub(n) } else { s };
            return;
        }
        let carry = nat::add_assign(r, a);
        if carry || cmp_limbs(r, &self.d) != Ordering::Less {
            nat::sub_assign(r, &self.d);
        }
    }

    /// `r = r - a mod n`.
    #[inline]
    pub(crate) fn sub(&self, r: &mut [u64], a: &[u64]) {
        if let Some(n) = self.word() {
            let (s, b) = r[0].overflowing_sub(a[0]);
            r[0] = if b { s.wrapping_add(n) } else { s };
            return;
        }
        if nat::sub_assign(r, a) {
            nat::add_assign(r, &self.d);
        }
    }

    /// `r = -r mod n`.
    #[inline]
    pub(crate) fn neg(&self, r: &mut [u64]) {
        if r.iter().all(|&l| l == 0) {
            return;
        }
        nat::sub_from(r, &self.d);
    }

    /// `out = a·b mod n`.
    pub(crate) fn mul(&self, out: &mut [u64], a: &[u64], b: &[u64], w: &mut Work) {
        if let Some(w) = &self.w {
            out[0] = w.mul(a[0], b[0]);
            return;
        }
        let k = self.k();
        if let Some(l) = &self.l
            && k <= 8
        {
            let mut p = [0u64; 16];
            nat::mul_to(&mut p[..2 * k], a, b, &mut w.s);
            l.reduce(&p[..2 * k], out);
            return;
        }
        w.p.clear();
        w.p.resize(2 * k, 0);
        nat::mul_to(&mut w.p, a, b, &mut w.s);
        let p = core::mem::take(&mut w.p);
        self.reduce(out, &p, w);
        w.p = p;
    }

    /// The inverse of `a` modulo `n`, `None` when it has none.
    pub(crate) fn inv(&self, a: &[u64]) -> Option<Vec<u64>> {
        if let Some(w) = &self.w {
            return w.inv(a[0]).map(|v| vec![v]);
        }
        let x = self.number(a).mod_inverse(&self.n)?;
        Some(self.residue(&x))
    }
}

impl PartialEq for Modulus {
    fn eq(&self, other: &Modulus) -> bool {
        self.d == other.d
    }
}

impl Eq for Modulus {}

/// A polynomial in one variable with coefficients modulo an integer `n ≥ 2`, which it holds once,
/// shared by its copies: each coefficient a residue below `n`, as limbs (one word for `n < 2^64`),
/// densely, the constant first, without zeros at the top.
///
/// Operations on two polynomials need the same modulus. Division, greatest common divisors and
/// the methods built on them need the leading coefficients they divide by to be units modulo
/// `n`, as they all are for a prime `n`.
///
/// # Examples
///
/// ```
/// use a::math::poly::ModPolynomial;
///
/// // Modulo 7: (x + 3)(x + 4) = x² + 5
/// let p = ModPolynomial::from_u64(&[3, 1], 7);
/// let q = ModPolynomial::from_u64(&[4, 1], 7);
/// let r = &p * &q;
/// assert_eq!(r.word_coeffs(), Some(&[5, 0, 1][..]));
/// assert_eq!(r.to_string(), "x^2 + 5 (mod 7)");
/// ```
#[derive(Clone)]
pub struct ModPolynomial {
    /// The modulus.
    pub(crate) m: Arc<Modulus>,
    /// The coefficients, `k` limbs each, the constant first, without zero coefficients at the
    /// top.
    pub(crate) c: Vec<u64>,
}

impl ModPolynomial {
    /// Drops the zero coefficients at the top.
    #[inline]
    pub(crate) fn trim(&mut self) {
        let k = self.m.k();
        while self.c.len() >= k && self.c[self.c.len() - k..].iter().all(|&l| l == 0) {
            self.c.truncate(self.c.len() - k);
        }
    }

    /// The polynomial of the modulus `m` and the packed coefficients `c`, trimmed.
    #[inline]
    pub(crate) fn of(m: Arc<Modulus>, c: Vec<u64>) -> ModPolynomial {
        let mut p = ModPolynomial { m, c };
        p.trim();
        p
    }

    /// The zero polynomial of the same modulus.
    #[inline]
    pub(crate) fn zero_like(&self) -> ModPolynomial {
        ModPolynomial {
            m: self.m.clone(),
            c: Vec::new(),
        }
    }

    /// The coefficient of `xⁱ`, as limbs, for `i` below [`length`](ModPolynomial::length).
    #[inline]
    pub(crate) fn limbs(&self, i: usize) -> &[u64] {
        let k = self.m.k();
        &self.c[i * k..(i + 1) * k]
    }

    /// Panics unless `other` has the same modulus.
    #[inline]
    #[track_caller]
    pub(crate) fn same_modulus(&self, other: &ModPolynomial) {
        assert!(
            Arc::ptr_eq(&self.m, &other.m) || self.m == other.m,
            "the polynomials have different moduli"
        );
    }

    /// The polynomial of the coefficients `coeffs` modulo `modulus`, the constant first, each
    /// reduced to its residue in `0..modulus`.
    ///
    /// # Panics
    ///
    /// Panics if `modulus` is below 2.
    #[track_caller]
    pub fn new(coeffs: &[Number], modulus: &Number) -> ModPolynomial {
        let m = Arc::new(Modulus::new(modulus));
        let mut c = Vec::with_capacity(coeffs.len() * m.k());
        for x in coeffs {
            c.extend_from_slice(&m.residue(x));
        }
        ModPolynomial::of(m, c)
    }

    /// The polynomial of the coefficients `coeffs` modulo the word `modulus`, each reduced.
    ///
    /// # Panics
    ///
    /// Panics if `modulus` is below 2.
    #[track_caller]
    pub fn from_u64(coeffs: &[u64], modulus: u64) -> ModPolynomial {
        let m = Arc::new(Modulus::new(&Number::from(modulus)));
        let c = coeffs.iter().map(|&x| x % modulus).collect();
        ModPolynomial::of(m, c)
    }

    /// The polynomial `p` modulo `modulus`, each coefficient reduced to its residue.
    #[track_caller]
    pub fn from_polynomial(p: &Polynomial<Number>, modulus: &Number) -> ModPolynomial {
        ModPolynomial::new(p.coeffs(), modulus)
    }

    /// The polynomial of the coefficients `coeffs` with the modulus of `self`.
    pub fn with_coeffs(&self, coeffs: &[Number]) -> ModPolynomial {
        let mut c = Vec::with_capacity(coeffs.len() * self.m.k());
        for x in coeffs {
            c.extend_from_slice(&self.m.residue(x));
        }
        ModPolynomial::of(self.m.clone(), c)
    }

    /// The zero polynomial modulo `modulus`.
    #[track_caller]
    pub fn zero(modulus: &Number) -> ModPolynomial {
        ModPolynomial::new(&[], modulus)
    }

    /// The constant 1 modulo `modulus`.
    #[track_caller]
    pub fn one(modulus: &Number) -> ModPolynomial {
        ModPolynomial::new(&[Number::from(1u8)], modulus)
    }

    /// The polynomial `x` modulo `modulus`.
    #[track_caller]
    pub fn x(modulus: &Number) -> ModPolynomial {
        ModPolynomial::new(&[Number::default(), Number::from(1u8)], modulus)
    }

    /// The modulus.
    #[inline]
    pub fn modulus(&self) -> &Number {
        &self.m.n
    }

    /// The degree, `None` for the zero polynomial.
    #[inline]
    pub fn degree(&self) -> Option<usize> {
        self.length().checked_sub(1)
    }

    /// The number of coefficients up to the leading one: the degree plus one, 0 for the zero
    /// polynomial.
    #[inline]
    pub fn length(&self) -> usize {
        self.c.len() / self.m.k()
    }

    /// Whether it is the zero polynomial.
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.c.is_empty()
    }

    /// Whether it is the constant 1.
    #[inline]
    pub fn is_one(&self) -> bool {
        self.length() == 1 && self.c[0] == 1 && self.c[1..].iter().all(|&l| l == 0)
    }

    /// Whether the leading coefficient is 1.
    #[inline]
    pub fn is_monic(&self) -> bool {
        let n = self.length();
        n > 0 && {
            let l = self.limbs(n - 1);
            l[0] == 1 && l[1..].iter().all(|&x| x == 0)
        }
    }

    /// The coefficient of `xⁱ`, in `0..n`: 0 above the degree.
    pub fn coeff(&self, i: usize) -> Number {
        if i < self.length() {
            self.m.number(self.limbs(i))
        } else {
            Number::default()
        }
    }

    /// The coefficients, the constant first, each in `0..n`.
    pub fn coeffs(&self) -> Vec<Number> {
        (0..self.length()).map(|i| self.coeff(i)).collect()
    }

    /// The coefficients as words, the constant first, when the modulus is below `2^64`.
    #[inline]
    pub fn word_coeffs(&self) -> Option<&[u64]> {
        (self.m.k() == 1).then_some(&self.c[..])
    }

    /// The leading coefficient, `None` for the zero polynomial.
    pub fn leading(&self) -> Option<Number> {
        self.degree().map(|d| self.coeff(d))
    }

    /// The polynomial over the integers of the coefficients in `0..n`.
    pub fn to_polynomial(&self) -> Polynomial<Number> {
        Polynomial::new(self.coeffs())
    }

    /// The polynomial over the integers of the coefficients in `-n/2..n/2`, the symmetric
    /// residues: `r` when `2r ≤ n`, `r - n` otherwise.
    pub fn to_symmetric(&self) -> Polynomial<Number> {
        let half = &self.m.n >> 1usize;
        Polynomial::new(
            self.coeffs()
                .into_iter()
                .map(|r| if r > half { &r - &self.m.n } else { r })
                .collect(),
        )
    }

    /// The value at `x` modulo `n`, by Horner's rule: for a word modulus with the precomputed
    /// quotient of `x` (Shoup's product) in two chains, the even and the odd coefficients.
    pub fn eval(&self, x: &Number) -> Number {
        if let Some(w) = self.m.word_mod() {
            let x = self.m.residue(x)[0];
            return Number::from(super::arith::wmul::eval(&self.c, x, w));
        }
        let k = self.m.k();
        let x = self.m.residue(x);
        let mut v = vec![0u64; k];
        let mut t = vec![0u64; k];
        let mut w = Work::default();
        for i in (0..self.length()).rev() {
            self.m.mul(&mut t, &v, &x, &mut w);
            v.copy_from_slice(&t);
            self.m.add(&mut v, self.limbs(i));
        }
        self.m.number(&v)
    }

    /// The derivative.
    pub fn derivative(&self) -> ModPolynomial {
        let k = self.m.k();
        let n = self.length();
        if n <= 1 {
            return self.zero_like();
        }
        let mut c = vec![0u64; (n - 1) * k];
        let mut w = Work::default();
        for i in 1..n {
            let f = self.m.residue(&Number::from(i as u64));
            self.m
                .mul(&mut c[(i - 1) * k..i * k], self.limbs(i), &f, &mut w);
        }
        ModPolynomial::of(self.m.clone(), c)
    }

    /// The composition `self(g)`: by Horner's rule on polynomials for short ones, and by divide
    /// and conquer beyond, `self = lo + x^h hi` giving `lo(g) + g^h hi(g)` with the powers
    /// `g^(2^i)` computed once.
    #[track_caller]
    pub fn compose(&self, g: &ModPolynomial) -> ModPolynomial {
        self.compose_fast(g)
    }

    /// The product by the constant `c`, reduced modulo `n`.
    pub fn scale(&self, c: &Number) -> ModPolynomial {
        let k = self.m.k();
        let f = self.m.residue(c);
        let mut out = vec![0u64; self.c.len()];
        let mut w = Work::default();
        for i in 0..self.length() {
            self.m
                .mul(&mut out[i * k..(i + 1) * k], self.limbs(i), &f, &mut w);
        }
        ModPolynomial::of(self.m.clone(), out)
    }

    /// The `e`-th power, by repeated squaring; `p⁰ = 1`.
    pub fn pow(&self, mut e: u64) -> ModPolynomial {
        let mut r = self.one_like();
        let mut b = self.clone();
        while e > 0 {
            if e & 1 == 1 {
                r = &r * &b;
            }
            e >>= 1;
            if e > 0 {
                b = &b * &b;
            }
        }
        r
    }

    /// The constant 1 of the same modulus.
    #[inline]
    pub(crate) fn one_like(&self) -> ModPolynomial {
        let mut c = vec![0u64; self.m.k()];
        c[0] = 1;
        ModPolynomial {
            m: self.m.clone(),
            c,
        }
    }

    /// The polynomial divided by its leading coefficient; the zero polynomial stays.
    ///
    /// # Panics
    ///
    /// Panics if the leading coefficient is not a unit modulo `n`.
    #[track_caller]
    pub fn to_monic(&self) -> ModPolynomial {
        let Some(d) = self.degree() else {
            return self.clone();
        };
        if self.is_monic() {
            return self.clone();
        }
        let inv = self
            .m
            .inv(self.limbs(d))
            .expect("the leading coefficient is not a unit modulo n");
        self.scale(&self.m.number(&inv))
    }

    /// The quotient `q` and the remainder `r` of the division by `d`: `self = q·d + r` with `r` of
    /// a degree below that of `d`. By the schoolbook method for a short quotient or divisor (for a
    /// word modulus with the remainder's coefficients kept as unreduced sums, one reduction each),
    /// and by Newton's method beyond: the reversed quotient is the reversed dividend times the
    /// inverse of the reversed divisor as a power series.
    ///
    /// # Panics
    ///
    /// Panics if `d` is the zero polynomial, its leading coefficient is not a unit modulo `n`, or
    /// the moduli differ.
    #[track_caller]
    pub fn div_rem(&self, d: &ModPolynomial) -> (ModPolynomial, ModPolynomial) {
        self.div_rem_fast(d)
    }

    /// `self · b mod f`.
    ///
    /// # Panics
    ///
    /// As [`div_rem`](ModPolynomial::div_rem) by `f`.
    #[track_caller]
    pub fn mul_mod(&self, b: &ModPolynomial, f: &ModPolynomial) -> ModPolynomial {
        self.same_modulus(f);
        (self * b).rem_fast(f)
    }

    /// `selfᵉ mod f`: the inverse of the reversed `f` computed once, then squares and products
    /// modulo `f` by sliding windows of the exponent's bits (`x^e` by squares and shifts), each
    /// remainder two products by Newton's method, or the schoolbook division for a short `f`.
    ///
    /// # Panics
    ///
    /// As [`div_rem`](ModPolynomial::div_rem) by `f`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::poly::ModPolynomial;
    /// use a::math::numbers::Number;
    ///
    /// // Modulo 7 and x² + 1: x² ≡ -1, so x⁷ = x·(x²)³ ≡ -x = 6x.
    /// let f = ModPolynomial::from_u64(&[1, 0, 1], 7);
    /// let x = ModPolynomial::from_u64(&[0, 1], 7);
    /// assert_eq!(x.pow_mod(&Number::from(7), &f), ModPolynomial::from_u64(&[0, 6], 7));
    /// ```
    #[track_caller]
    pub fn pow_mod(&self, e: &Number, f: &ModPolynomial) -> ModPolynomial {
        self.same_modulus(f);
        super::arith::mdiv::ModInverse::new(f).pow_mod(self, e)
    }
}

impl PartialEq for ModPolynomial {
    fn eq(&self, other: &ModPolynomial) -> bool {
        (Arc::ptr_eq(&self.m, &other.m) || self.m == other.m) && self.c == other.c
    }
}

impl Eq for ModPolynomial {}

impl Hash for ModPolynomial {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.m.d.hash(state);
        self.c.hash(state);
    }
}

impl fmt::Debug for ModPolynomial {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ModPolynomial")
            .field("coeffs", &self.coeffs())
            .field("modulus", &self.m.n)
            .finish()
    }
}

impl fmt::Display for ModPolynomial {
    /// Writes the polynomial of the residues in `0..n` as [`Polynomial`]'s `Display` does, then
    /// the modulus: `x^2 + 5 (mod 7)`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} (mod {})", self.to_polynomial(), self.m.n)
    }
}

impl AddAssign<&ModPolynomial> for ModPolynomial {
    #[track_caller]
    fn add_assign(&mut self, rhs: &ModPolynomial) {
        self.same_modulus(rhs);
        let k = self.m.k();
        if self.c.len() < rhs.c.len() {
            self.c.resize(rhs.c.len(), 0);
        }
        let m = self.m.clone();
        for (r, a) in self.c.chunks_exact_mut(k).zip(rhs.c.chunks_exact(k)) {
            m.add(r, a);
        }
        self.trim();
    }
}

impl SubAssign<&ModPolynomial> for ModPolynomial {
    #[track_caller]
    fn sub_assign(&mut self, rhs: &ModPolynomial) {
        self.same_modulus(rhs);
        let k = self.m.k();
        if self.c.len() < rhs.c.len() {
            self.c.resize(rhs.c.len(), 0);
        }
        let m = self.m.clone();
        for (r, a) in self.c.chunks_exact_mut(k).zip(rhs.c.chunks_exact(k)) {
            m.sub(r, a);
        }
        self.trim();
    }
}

impl Mul<&ModPolynomial> for &ModPolynomial {
    type Output = ModPolynomial;
    /// The product, by the method the lengths and the modulus call for. For a word modulus:
    /// the schoolbook method with each coefficient a sum of products reduced once (in a word by
    /// the vector kernels below `2^32`, in two or three words above), and from about 50 to 100
    /// coefficients the number-theoretic transforms of the integers' multiplication applied to
    /// the coefficients, modulo as many of its primes below `2^29` as make the convolution exact,
    /// each coefficient brought back modulo `n` by the Chinese remainder theorem. For a modulus of
    /// several limbs: the schoolbook method below 6 coefficients, from 48 the same transforms on
    /// pieces of the coefficients of up to 112 bits (for 2 to 9 limbs and products of up to about
    /// 6,000 coefficients), otherwise Kronecker substitution into the integers' multiplication;
    /// one division by `n` a coefficient.
    #[track_caller]
    fn mul(self, rhs: &ModPolynomial) -> ModPolynomial {
        self.same_modulus(rhs);
        super::arith::mod_mul(self, rhs)
    }
}

impl Neg for &ModPolynomial {
    type Output = ModPolynomial;
    fn neg(self) -> ModPolynomial {
        let mut r = self.clone();
        let k = self.m.k();
        for x in r.c.chunks_exact_mut(k) {
            self.m.neg(x);
        }
        r
    }
}

impl Neg for ModPolynomial {
    type Output = ModPolynomial;
    #[inline]
    fn neg(self) -> ModPolynomial {
        -&self
    }
}

impl Add<&ModPolynomial> for &ModPolynomial {
    type Output = ModPolynomial;
    #[track_caller]
    fn add(self, rhs: &ModPolynomial) -> ModPolynomial {
        let mut r = self.clone();
        r += rhs;
        r
    }
}

impl Sub<&ModPolynomial> for &ModPolynomial {
    type Output = ModPolynomial;
    #[track_caller]
    fn sub(self, rhs: &ModPolynomial) -> ModPolynomial {
        let mut r = self.clone();
        r -= rhs;
        r
    }
}

impl Div<&ModPolynomial> for &ModPolynomial {
    type Output = ModPolynomial;
    #[track_caller]
    fn div(self, rhs: &ModPolynomial) -> ModPolynomial {
        self.div_rem(rhs).0
    }
}

impl Rem<&ModPolynomial> for &ModPolynomial {
    type Output = ModPolynomial;
    #[track_caller]
    fn rem(self, rhs: &ModPolynomial) -> ModPolynomial {
        self.rem_fast(rhs)
    }
}

/// The forms of an operator on owned polynomials and of its assignment, from the form on two
/// references.
macro_rules! forms {
    ($tr:ident, $f:ident, $atr:ident, $af:ident) => {
        impl $tr<&ModPolynomial> for ModPolynomial {
            type Output = ModPolynomial;
            #[inline]
            #[track_caller]
            fn $f(self, rhs: &ModPolynomial) -> ModPolynomial {
                (&self).$f(rhs)
            }
        }
        impl $tr<ModPolynomial> for &ModPolynomial {
            type Output = ModPolynomial;
            #[inline]
            #[track_caller]
            fn $f(self, rhs: ModPolynomial) -> ModPolynomial {
                self.$f(&rhs)
            }
        }
        impl $tr for ModPolynomial {
            type Output = ModPolynomial;
            #[inline]
            #[track_caller]
            fn $f(self, rhs: ModPolynomial) -> ModPolynomial {
                (&self).$f(&rhs)
            }
        }
        impl $atr for ModPolynomial {
            #[inline]
            #[track_caller]
            fn $af(&mut self, rhs: ModPolynomial) {
                *self = (&*self).$f(&rhs);
            }
        }
    };
}

forms!(Add, add, AddAssign, add_assign);
forms!(Sub, sub, SubAssign, sub_assign);
forms!(Mul, mul, MulAssign, mul_assign);
forms!(Div, div, DivAssign, div_assign);
forms!(Rem, rem, RemAssign, rem_assign);

impl MulAssign<&ModPolynomial> for ModPolynomial {
    #[inline]
    #[track_caller]
    fn mul_assign(&mut self, rhs: &ModPolynomial) {
        *self = &*self * rhs;
    }
}

impl DivAssign<&ModPolynomial> for ModPolynomial {
    #[inline]
    #[track_caller]
    fn div_assign(&mut self, rhs: &ModPolynomial) {
        *self = &*self / rhs;
    }
}

impl RemAssign<&ModPolynomial> for ModPolynomial {
    #[inline]
    #[track_caller]
    fn rem_assign(&mut self, rhs: &ModPolynomial) {
        *self = &*self % rhs;
    }
}
