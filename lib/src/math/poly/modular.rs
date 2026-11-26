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
    /// Montgomery's products modulo an odd `n` of several limbs, made the first time they serve.
    mont: std::sync::OnceLock<Option<super::eval::mont::MontCtx>>,
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
            mont: std::sync::OnceLock::new(),
        }
    }

    /// Montgomery's products modulo `n` when it is odd and of 2 to
    /// [`MAX_LIMBS`](super::eval::mont::MAX_LIMBS) limbs, made once for the modulus.
    #[inline]
    pub(crate) fn mont(&self) -> Option<&super::eval::mont::MontCtx> {
        self.mont
            .get_or_init(|| super::eval::mont::MontCtx::new(self))
            .as_ref()
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

    /// Whether `n` is 2, from the word arithmetic held in the modulus itself.
    #[inline]
    pub(crate) fn is_two(&self) -> bool {
        self.w.as_ref().is_some_and(|w| w.n == 2)
    }

    /// `n` when it fits in a word.
    #[inline]
    pub(crate) fn word(&self) -> Option<u64> {
        // The word arithmetic is there exactly for a modulus of one limb, with n itself.
        self.w.as_ref().map(|w| w.n)
    }

    /// The residue of `x`, of `k` limbs.
    pub(crate) fn residue(&self, x: &Number) -> Vec<u64> {
        if let Some(r) = self.word_residue(x) {
            return vec![r];
        }
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

    /// The residue of `x` modulo a word modulus when `|x|` fits in a word, without a division of
    /// numbers; `None` otherwise.
    #[inline]
    pub(crate) fn word_residue(&self, x: &Number) -> Option<u64> {
        let n = self.word_mod()?.n;
        let v = super::div::magnitude_u64(x)?;
        let r = if v >= n { v % n } else { v };
        Some(if x.is_negative() && r != 0 { n - r } else { r })
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

thread_local! {
    /// The modulus built last on this thread: a bounded cache of one entry, so that the
    /// polynomials made one after the other with the same modulus share it rather than each
    /// building its constants (a reciprocal, a division by `n` for several limbs) anew.
    static LAST: core::cell::RefCell<Option<Arc<Modulus>>> = const { core::cell::RefCell::new(None) };
}

/// The modulus `n` (at least 2), the one of this thread's cache when it is the same.
#[track_caller]
pub(crate) fn cached_modulus(n: &Number) -> Arc<Modulus> {
    if let Some(m) = LAST.with(|c| c.borrow().as_ref().filter(|m| m.n == *n).cloned()) {
        return m;
    }
    let m = Arc::new(Modulus::new(n));
    LAST.with(|c| *c.borrow_mut() = Some(m.clone()));
    m
}

/// The word modulus `n` (at least 2), as [`cached_modulus`].
#[track_caller]
pub(crate) fn cached_word_modulus(n: u64) -> Arc<Modulus> {
    if let Some(m) = LAST.with(|c| {
        c.borrow()
            .as_ref()
            .filter(|m| m.d.len() == 1 && m.d[0] == n)
            .cloned()
    }) {
        return m;
    }
    let m = Arc::new(Modulus::new(&Number::from(n)));
    LAST.with(|c| *c.borrow_mut() = Some(m.clone()));
    m
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
    pub(crate) c: Words,
}

/// The coefficients of a [`ModPolynomial`], read as the vector of their limbs (one word each for
/// a modulus below `2^64`). Modulo 2 they may be held packed, 64 coefficients a word, as the
/// products and divisions over GF(2) leave them, the words made the first time something reads
/// them; and the words of a polynomial modulo 2 keep their packed form once a product has made
/// it. Changing the words (through [`Words::words_mut`]) drops the packed form. Other moduli
/// carry nothing more than the words.
#[derive(Clone, Default)]
pub(crate) struct Words {
    /// The words, unless the polynomial was made packed.
    v: Vec<u64>,
    /// Modulo 2: the packed form, and the words of a polynomial made packed.
    two: Option<Box<Two>>,
}

/// The packed forms of a polynomial modulo 2.
#[derive(Clone, Default)]
struct Two {
    /// Whether the polynomial was made packed: `bits` then holds it, the words are made into
    /// `words` when read, and `Words::v` is empty.
    given: bool,
    /// The coefficients packed and trimmed (made from the words the first time a product asks).
    bits: std::sync::OnceLock<Vec<u64>>,
    /// The words of a polynomial made packed, once read.
    words: std::sync::OnceLock<Vec<u64>>,
}

impl Two {
    /// The packed coefficients of a polynomial made packed.
    #[inline]
    fn given_bits(&self) -> &[u64] {
        self.bits
            .get()
            .expect("a polynomial made packed has its bits")
    }

    /// The words of a polynomial made packed, made once (out of line, so that reading the words
    /// of every other polynomial stays a test of a pointer).
    #[cold]
    #[inline(never)]
    fn given_words(&self) -> &Vec<u64> {
        self.words
            .get_or_init(|| super::arith::gf2::unpack(self.given_bits()))
    }

    /// The number of coefficients of a polynomial made packed.
    #[cold]
    #[inline(never)]
    fn given_len(&self) -> usize {
        super::arith::gf2::degree(self.given_bits()).map_or(0, |d| d + 1)
    }
}

impl Words {
    /// The polynomial modulo 2 of the packed coefficients `bits`, trimmed.
    pub(crate) fn packed(bits: Vec<u64>) -> Words {
        debug_assert!(bits.last() != Some(&0));
        Words {
            v: Vec::new(),
            two: Some(Box::new(Two {
                given: true,
                bits: std::sync::OnceLock::from(bits),
                words: std::sync::OnceLock::new(),
            })),
        }
    }

    /// The words `v` of a polynomial modulo 2, with room for their packed form.
    pub(crate) fn modulo_two(v: Vec<u64>) -> Words {
        Words {
            v,
            two: Some(Box::default()),
        }
    }

    /// The number of words, without making them from a packed form.
    #[inline]
    pub(crate) fn len(&self) -> usize {
        match &self.two {
            Some(t) if t.given => t.given_len(),
            _ => self.v.len(),
        }
    }

    /// Whether there are no coefficients, without making the words from a packed form.
    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        match &self.two {
            Some(t) if t.given => t.given_bits().is_empty(),
            _ => self.v.is_empty(),
        }
    }

    /// Modulo 2 only: the coefficients packed and trimmed, made once when there is room for them.
    #[inline]
    pub(crate) fn bits(&self) -> std::borrow::Cow<'_, [u64]> {
        let pack = |v: &[u64]| {
            let mut b = super::arith::gf2::pack(v);
            super::arith::gf2::trim(&mut b);
            b
        };
        match &self.two {
            Some(t) => std::borrow::Cow::Borrowed(t.bits.get_or_init(|| pack(&self.v))),
            None => std::borrow::Cow::Owned(pack(&self.v)),
        }
    }

    /// The words to change; the packed form is dropped.
    pub(crate) fn words_mut(&mut self) -> &mut Vec<u64> {
        if let Some(t) = &mut self.two {
            if t.given {
                let words = core::mem::take(&mut t.words);
                self.v = words
                    .into_inner()
                    .unwrap_or_else(|| super::arith::gf2::unpack(t.given_bits()));
            }
            **t = Two::default();
        }
        &mut self.v
    }

    /// The words, by value.
    pub(crate) fn into_vec(mut self) -> Vec<u64> {
        match self.two.take() {
            Some(t) if t.given => {
                let Two { bits, words, .. } = *t;
                words.into_inner().unwrap_or_else(|| {
                    super::arith::gf2::unpack(&bits.into_inner().expect("given bits"))
                })
            }
            _ => core::mem::take(&mut self.v),
        }
    }

    /// The words of a polynomial without packed forms (any modulus but 2), straight from the
    /// vector: the fast path of the short operations.
    #[inline(always)]
    pub(crate) fn plain(&self) -> Option<&[u64]> {
        if self.two.is_none() {
            Some(&self.v)
        } else {
            None
        }
    }

    /// Whether the polynomial was made packed.
    #[inline]
    pub(crate) fn is_given(&self) -> bool {
        self.two.as_ref().is_some_and(|t| t.given)
    }
}

impl Drop for Words {
    /// The packed forms (modulo 2 only) dropped out of line, the common case a test of a pointer.
    #[inline]
    fn drop(&mut self) {
        if let Some(t) = self.two.take() {
            drop_two(t);
        }
    }
}

/// Drops the packed forms of a polynomial modulo 2.
#[inline(never)]
fn drop_two(t: Box<Two>) {
    drop(t);
}

impl From<Vec<u64>> for Words {
    #[inline]
    fn from(v: Vec<u64>) -> Words {
        Words { v, two: None }
    }
}

impl core::ops::Deref for Words {
    type Target = Vec<u64>;
    #[inline]
    fn deref(&self) -> &Vec<u64> {
        match &self.two {
            Some(t) if t.given => t.given_words(),
            _ => &self.v,
        }
    }
}

impl PartialEq for Words {
    fn eq(&self, other: &Words) -> bool {
        if self.is_given() && other.is_given() {
            return self.bits() == other.bits();
        }
        **self == **other
    }
}

impl Hash for Words {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (**self).hash(state);
    }
}

impl ModPolynomial {
    /// Drops the zero coefficients at the top.
    #[inline]
    pub(crate) fn trim(&mut self) {
        if self.c.is_given() {
            // Packed ones are made trimmed.
            return;
        }
        let k = self.m.k();
        let c = &mut self.c.v;
        while c.len() >= k && c[c.len() - k..].iter().all(|&l| l == 0) {
            c.truncate(c.len() - k);
        }
    }

    /// The polynomial of the modulus `m` and the packed coefficients `c`, trimmed.
    #[inline]
    pub(crate) fn of(m: Arc<Modulus>, mut c: Vec<u64>) -> ModPolynomial {
        if m.is_two() {
            return ModPolynomial::of_two(m, c);
        }
        let k = m.k();
        while c.len() >= k && c[c.len() - k..].iter().all(|&l| l == 0) {
            c.truncate(c.len() - k);
        }
        ModPolynomial { m, c: c.into() }
    }

    /// [`of`](ModPolynomial::of) modulo 2: the words with room for their packed form.
    #[inline(never)]
    fn of_two(m: Arc<Modulus>, c: Vec<u64>) -> ModPolynomial {
        let mut p = ModPolynomial {
            m,
            c: Words::modulo_two(c),
        };
        p.trim();
        p
    }

    /// The polynomial modulo 2 of the coefficients `bits` packed 64 a word (any zero words at
    /// the top dropped), its words made only when read.
    #[inline]
    pub(crate) fn of_bits(m: Arc<Modulus>, mut bits: Vec<u64>) -> ModPolynomial {
        debug_assert_eq!(m.word(), Some(2));
        super::arith::gf2::trim(&mut bits);
        ModPolynomial {
            m,
            c: Words::packed(bits),
        }
    }

    /// The zero polynomial of the same modulus.
    #[inline]
    pub(crate) fn zero_like(&self) -> ModPolynomial {
        ModPolynomial {
            m: self.m.clone(),
            c: Vec::new().into(),
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
        let m = cached_modulus(modulus);
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
        let m = cached_word_modulus(modulus);
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
        if self.m.is_two() {
            // The constant at 0, the parity of the number of coefficients 1 at 1.
            let b = self.c.bits();
            let v = if !x.bit(0) {
                b.first().map_or(0, |&w| w & 1)
            } else {
                u64::from(b.iter().map(|w| w.count_ones()).sum::<u32>() & 1)
            };
            return Number::from(v);
        }
        if let Some(w) = self.m.word_mod() {
            // A point of a word reduced without a number of its residue.
            let x = match super::div::magnitude_u64(x) {
                Some(v) => {
                    let r = v % w.n;
                    if x.is_negative() && r != 0 {
                        w.n - r
                    } else {
                        r
                    }
                }
                None => self.m.residue(x)[0],
            };
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
        if let (Some(x), Some(w)) = (self.c.plain(), self.m.word_mod())
            && w.n != 2
        {
            // A word modulus but 2: on the vector.
            if x.len() <= 1 {
                return self.zero_like();
            }
            return ModPolynomial::of(self.m.clone(), word_derivative(x, w));
        }
        let k = self.m.k();
        let n = self.length();
        if n <= 1 {
            return self.zero_like();
        }
        if self.m.is_two() {
            // i·cᵢ is cᵢ for an odd i and 0 for an even one: the odd bits moved down by one.
            let b = self.c.bits();
            let d: Vec<u64> = (0..b.len())
                .map(|i| ((b[i] >> 1) | (b.get(i + 1).map_or(0, |&w| w << 63))) & EVEN_BITS)
                .collect();
            return ModPolynomial::of_bits(self.m.clone(), d);
        }
        if let Some(w) = self.m.word_mod() {
            return ModPolynomial::of(self.m.clone(), word_derivative(&self.c, w));
        }
        let mut c = vec![0u64; (n - 1) * k];
        let mut w = Work::default();
        // The index as a residue, counted up rather than made from a number each time.
        let mut f = vec![0u64; k];
        let mut one = vec![0u64; k];
        one[0] = 1;
        for i in 1..n {
            self.m.add(&mut f, &one);
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
        if let (Some(x), Some(w)) = (self.c.plain(), self.m.word_mod())
            && w.n != 2
        {
            // A word modulus but 2: on the vector.
            let f = match self.m.word_residue(c) {
                Some(f) => f,
                None => self.m.residue(c)[0],
            };
            return ModPolynomial::of(self.m.clone(), word_scale(x, f, w));
        }
        if self.m.is_two() {
            return if !c.bit(0) {
                self.zero_like()
            } else {
                self.clone()
            };
        }
        if let Some(w) = self.m.word_mod() {
            let f = match self.m.word_residue(c) {
                Some(f) => f,
                None => self.m.residue(c)[0],
            };
            return ModPolynomial::of(self.m.clone(), word_scale(&self.c, f, w));
        }
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
            c: c.into(),
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
        if self.m.is_two() && f.length() >= 2 {
            // Packed from start to end: squares by spreading bits, remainders by words (by
            // Barrett's method on polynomials for a long `f`).
            assert!(!e.is_negative(), "a negative exponent");
            let d = super::arith::gf2::Div::new(&f.c.bits());
            return ModPolynomial::of_bits(self.m.clone(), d.pow(&self.c.bits(), &limbs_of(e)));
        }
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
        let rl = rhs.c.len();
        let c = self.c.words_mut();
        if c.len() < rl {
            c.resize(rl, 0);
        }
        if let Some(n) = self.m.word() {
            for (r, &a) in c.iter_mut().zip(rhs.c.iter()) {
                *r = add_word(*r, a, n);
            }
            self.trim();
            return;
        }
        let m = self.m.clone();
        for (r, a) in c.chunks_exact_mut(k).zip(rhs.c.chunks_exact(k)) {
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
        let rl = rhs.c.len();
        let c = self.c.words_mut();
        if c.len() < rl {
            c.resize(rl, 0);
        }
        if let Some(n) = self.m.word() {
            for (r, &a) in c.iter_mut().zip(rhs.c.iter()) {
                *r = sub_word(*r, a, n);
            }
            self.trim();
            return;
        }
        let m = self.m.clone();
        for (r, a) in c.chunks_exact_mut(k).zip(rhs.c.chunks_exact(k)) {
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
        if self.m.is_two() {
            // -1 = 1 modulo 2.
            return self.clone();
        }
        let mut r = self.clone();
        let k = self.m.k();
        for x in r.c.words_mut().chunks_exact_mut(k) {
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
        if let (Some(x), Some(y), Some(w)) = (self.c.plain(), rhs.c.plain(), self.m.word_mod()) {
            // A word modulus but 2: on the vectors, the longer one's top copied.
            self.same_modulus(rhs);
            let (long, short) = if x.len() >= y.len() { (x, y) } else { (y, x) };
            let mut c = Vec::with_capacity(long.len());
            c.extend(long.iter().zip(short).map(|(&x, &y)| add_word(x, y, w.n)));
            c.extend_from_slice(&long[short.len()..]);
            return ModPolynomial::of(self.m.clone(), c);
        }
        if self.m.is_two() {
            self.same_modulus(rhs);
            return ModPolynomial::of_bits(self.m.clone(), gf2_xor(&self.c, &rhs.c));
        }
        if let Some(n) = self.m.word() {
            // The sum written once into a new vector, the longer one's top copied.
            self.same_modulus(rhs);
            let (long, short) = if self.c.len() >= rhs.c.len() {
                (&self.c, &rhs.c)
            } else {
                (&rhs.c, &self.c)
            };
            let mut c = Vec::with_capacity(long.len());
            c.extend(
                long.iter()
                    .zip(short.iter())
                    .map(|(&x, &y)| add_word(x, y, n)),
            );
            c.extend_from_slice(&long[short.len()..]);
            return ModPolynomial::of(self.m.clone(), c);
        }
        let mut r = self.clone();
        r += rhs;
        r
    }
}

impl Sub<&ModPolynomial> for &ModPolynomial {
    type Output = ModPolynomial;
    #[track_caller]
    fn sub(self, rhs: &ModPolynomial) -> ModPolynomial {
        if let (Some(a), Some(b), Some(w)) = (self.c.plain(), rhs.c.plain(), self.m.word_mod()) {
            // A word modulus but 2: on the vectors.
            self.same_modulus(rhs);
            let n = w.n;
            let h = a.len().min(b.len());
            let mut c = Vec::with_capacity(a.len().max(b.len()));
            c.extend(a.iter().zip(b).map(|(&x, &y)| sub_word(x, y, n)));
            c.extend_from_slice(&a[h..]);
            c.extend(b[h..].iter().map(|&y| sub_word(0, y, n)));
            return ModPolynomial::of(self.m.clone(), c);
        }
        if self.m.is_two() {
            self.same_modulus(rhs);
            return ModPolynomial::of_bits(self.m.clone(), gf2_xor(&self.c, &rhs.c));
        }
        if let Some(n) = self.m.word() {
            self.same_modulus(rhs);
            let (a, b) = (&self.c, &rhs.c);
            let h = a.len().min(b.len());
            let mut c = Vec::with_capacity(a.len().max(b.len()));
            c.extend(a.iter().zip(b.iter()).map(|(&x, &y)| sub_word(x, y, n)));
            c.extend_from_slice(&a[h..]);
            c.extend(b[h..].iter().map(|&y| sub_word(0, y, n)));
            return ModPolynomial::of(self.m.clone(), c);
        }
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

/// The even bit positions of a word.
const EVEN_BITS: u64 = 0x5555_5555_5555_5555;

/// The sum modulo 2 of two polynomials' packed forms (made once each), not trimmed.
fn gf2_xor(a: &Words, b: &Words) -> Vec<u64> {
    let (x, y) = (a.bits(), b.bits());
    let (long, short) = if x.len() >= y.len() { (x, y) } else { (y, x) };
    let mut c = long.to_vec();
    for (o, &w) in c.iter_mut().zip(short.iter()) {
        *o ^= w;
    }
    c
}

/// `a + b mod n` for residues of a word modulus, without a branch.
#[inline(always)]
pub(crate) fn add_word(a: u64, b: u64, n: u64) -> u64 {
    let (s, c) = a.overflowing_add(b);
    let t = s.wrapping_sub(n);
    if c || s >= n { t } else { s }
}

/// `a - b mod n` for residues of a word modulus, without a branch.
#[inline(always)]
pub(crate) fn sub_word(a: u64, b: u64, n: u64) -> u64 {
    let (d, c) = a.overflowing_sub(b);
    d.wrapping_add(n & 0u64.wrapping_sub(u64::from(c)))
}

/// The derivative of the residues `a` (at least two) modulo a word: `i·aᵢ`, the index counted
/// up as a residue; one word's product and Barrett's reduction while `n·len` fits a word, the
/// two-word reduction otherwise.
fn word_derivative(a: &[u64], w: &super::arith::nmod::WordMod) -> Vec<u64> {
    let n = w.n;
    let len = a.len();
    let mut c = Vec::with_capacity(len - 1);
    let mut j = 1 % n;
    if (len as u128) * u128::from(n) <= u128::from(u64::MAX) {
        for &x in &a[1..] {
            c.push(w.reduce1(x * j));
            j += 1;
            if j == n {
                j = 0;
            }
        }
    } else {
        for &x in &a[1..] {
            c.push(w.reduce_wide(u128::from(x) * u128::from(j)));
            j += 1;
            if j == n {
                j = 0;
            }
        }
    }
    c
}

/// The residues `a` times the residue `f` modulo a word: Shoup's product with the quotient of
/// `f` precomputed below `2^63`, the reduction of the two-word product above.
fn word_scale(a: &[u64], f: u64, w: &super::arith::nmod::WordMod) -> Vec<u64> {
    let n = w.n;
    if n >> 63 == 0 {
        let fp = w.shoup(f);
        a.iter().map(|&x| w.mul_shoup(x, f, fp)).collect()
    } else if n & 1 == 1 {
        // Montgomery's product by f·2^64 mod n: x·f·2^64·2^-64 = x·f, one product and one
        // reduction of two words each, where Shoup's quotient would need n < 2^63.
        let ni = mont_neg_inv(n);
        let fr = w.reduce_wide(u128::from(f) << 64);
        a.iter()
            .map(|&x| {
                let t = u128::from(x) * u128::from(fr);
                let m = (t as u64).wrapping_mul(ni);
                let mn = u128::from(m) * u128::from(n);
                // t + m n is 0 below 2^64: its low words carry exactly when t's is not 0.
                let carry = u64::from(t as u64 != 0);
                let (hi, o1) = ((t >> 64) as u64).overflowing_add((mn >> 64) as u64);
                let (hi, o2) = hi.overflowing_add(carry);
                if o1 || o2 || hi >= n {
                    hi.wrapping_sub(n)
                } else {
                    hi
                }
            })
            .collect()
    } else {
        a.iter().map(|&x| w.mul(x, f)).collect()
    }
}

/// `-n^-1 mod 2^64` for an odd `n`, by Newton's iteration (each step doubles the correct bits).
fn mont_neg_inv(n: u64) -> u64 {
    let mut x = n; // correct to 3 bits: n·n = 1 mod 8 for odd n
    for _ in 0..5 {
        x = x.wrapping_mul(2u64.wrapping_sub(n.wrapping_mul(x)));
    }
    x.wrapping_neg()
}
