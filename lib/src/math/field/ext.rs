//! Extension fields `GF(pᵏ) = GF(p)[x] / g(x)` for a monic irreducible `g` of degree `k` over a
//! prime field, and their elements, the polynomials of degree below `k` in the class of `x`.
//!
//! Products are polynomial products reduced modulo `g` (schoolbook for small degrees, the
//! products and divisions of [`ModPolynomial`] for large ones); inverses come from the extended
//! Euclidean algorithm on polynomials; the Frobenius map `a ↦ aᵖ` is the matrix of the powers
//! `x^(p·j) mod g`, computed once; the norm is the resultant `res(g, a)` and the trace a sum
//! with the power sums of the roots of `g`, from Newton's identities (Lidl and Niederreiter,
//! *Finite Fields*, §2.3); square roots are Tonelli and Shanks's algorithm in `GF(pᵏ)` for odd
//! `p` and the inverse of the Frobenius map for `p = 2`, with the quadratic character through the
//! norm. The polynomial chosen for a given degree is the smallest monic irreducible one in the
//! order of its coefficients from the top down, binomials first, by Lidl and Niederreiter's
//! Theorem 3.75 for those. See [`ExtensionField`] for the details and the costs.

use super::prime_form::{Arith, Value, held, inline, with_form};
use super::{Fp, PrimeField, ext_arith, ext_find, ext_maps, ext_roots, ext_zech};
use crate::math::numbers::{Factorization, Number};
use crate::math::poly::ModPolynomial;
use core::fmt;
use core::hash::{Hash, Hasher};
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use std::sync::{Arc, OnceLock};

/// The finite field `GF(pᵏ) = GF(p)[x] / g(x)` for a monic irreducible polynomial `g` of degree
/// `k` over a [`PrimeField`] `GF(p)`, its elements the polynomials of degree below `k` in the
/// class of `x` (a root of `g`), held as their `k` coefficients in `GF(p)`, the constant first.
///
/// The field holds a copy of its base field (a [`PrimeField`] is its prime and the constants of
/// its arithmetic, cloned once), so it has no lifetime and can be kept and returned freely; its
/// elements, [`Fpk`], borrow it, as the elements of a [`PrimeField`] borrow theirs, and their
/// coefficients are elements of that copy, which mix with the elements of any [`PrimeField`] of
/// the same prime. What only some uses need is computed on the first use and kept: the matrix of
/// the Frobenius map, the traces of the powers of `x`, the data of the square roots, the
/// factorisation of `pᵏ − 1`, and the tables of logarithms of a field of at most `2¹⁶` elements
/// (`2¹⁴` for `k = 2` and `3`) over a prime of one word.
///
/// It is built from a given `g` ([`new`](ExtensionField::new), which checks that `g` is
/// irreducible) or from a degree ([`with_degree`](ExtensionField::with_degree), which takes the
/// polynomial [`first_irreducible`](ExtensionField::first_irreducible) gives). A degree of 1 gives
/// `GF(p)` itself.
///
/// Like the rest of the library this is not constant-time: it is for mathematics and for
/// verifying with public data, not for secret keys.
///
/// # Examples
///
/// ```
/// use a::math::field::{ExtensionField, PrimeField};
/// use a::math::numbers::Number;
/// use a::math::poly::ModPolynomial;
///
/// // GF(7²) = GF(7)[x] / (x² + 1): x is a square root of −1.
/// let base = PrimeField::new(&Number::from(7u8));
/// let f = ExtensionField::new(&base, &ModPolynomial::from_u64(&[1, 0, 1], 7));
/// let x = f.x();
/// assert_eq!(x.square(), -f.one());
/// assert_eq!(f.order(), &Number::from(49u8));
/// let a = f.from_u64(&[3, 2]); // 2x + 3
/// assert_eq!(&a * &a.inverse().unwrap(), f.one());
/// assert_eq!(a.frobenius(1), f.from_u64(&[3, 5])); // the conjugate, 3 − 2x
/// assert_eq!(a.norm().value(), Number::from(13u8 % 7)); // 3² + 2² = 13
/// ```
#[derive(Clone)]
pub struct ExtensionField {
    /// The base field, a copy of the one given.
    pub(super) base: PrimeField,
    /// The degree `k`.
    pub(super) k: usize,
    /// The polynomial `g`, monic and irreducible, of degree `k`.
    pub(super) g: ModPolynomial,
    /// The terms of `xᵏ mod g`: `(i, −gᵢ)` for each coefficient `gᵢ` of `g` below `xᵏ` that is not
    /// zero, `−gᵢ` in `0..p`.
    pub(super) low: Vec<(usize, Number)>,
    /// The number of elements, `pᵏ`.
    pub(super) q: Number,
    /// The degree from which products go through [`ModPolynomial`]
    /// ([`ext_arith::poly_min`]).
    pub(super) poly_min: usize,
    /// The matrix of the Frobenius map: column `j`, the `k` coefficients of `x^(p·j) mod g`, at
    /// `j·k..(j + 1)·k`.
    pub(super) frob: OnceLock<Vec<Number>>,
    /// The same matrix as the base field's words, for [`Kind::Words`].
    pub(super) frob_words: OnceLock<Vec<u64>>,
    /// The traces of `1, x, …, x^(k−1)`.
    pub(super) traces: OnceLock<Vec<Number>>,
    /// The terms of `xᵏ mod g`, as the base field's form holds them.
    pub(super) low_form: OnceLock<Vec<(usize, Value)>>,
    /// The traces of the powers of `x`, in the form.
    pub(super) traces_form: OnceLock<Vec<Value>>,
    /// For `p = 2`, the coefficients of the square root of `x`.
    pub(super) root_x: OnceLock<Vec<Number>>,
    /// For odd `p`, the data of Tonelli and Shanks's algorithm.
    pub(super) shanks: OnceLock<ext_roots::Shanks>,
    /// The factorisation of `pᵏ − 1`.
    pub(super) units: OnceLock<Factorization>,
    /// For at most [`ext_zech::max_order`]`(k)` elements over a prime of one word, the constants of
    /// the tables of logarithms and the tables, made on their first use and shared by the
    /// field's copies.
    pub(super) zech: Option<Arc<ext_zech::Zech>>,
    /// How the elements are held ([`Repr`]).
    pub(super) kind: Kind,
    /// For a prime of one word below `2⁶³` in Montgomery's form (the fields of [`Kind::Words`] and
    /// most of [`Kind::Log`]): the prime and `-1/p mod 2⁶⁴`; else `(0, 0)`.
    pub(super) word: (u64, u64),
}

/// How the elements of a field are held.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Kind {
    /// As their coefficients, elements of the base field.
    General,
    /// As the words of their coefficients in Montgomery's form of one word, for a prime below
    /// `2⁶³` (whose products sum in 128 bits before one reduction).
    Words,
    /// As their logarithms, for the fields of Zech's tables.
    Log,
}

impl ExtensionField {
    /// The field `GF(p)[x] / g(x)` over `base = GF(p)`, for a monic irreducible `g` modulo `p`
    /// of degree at least 1, checked by Ben-Or's test of irreducibility
    /// ([`ModPolynomial::is_irreducible`]).
    ///
    /// # Panics
    ///
    /// Panics if the modulus of `g` is not `p`, if `g` is constant or not monic, if it is
    /// reducible modulo `p`, or if its degree is `2³²` or more.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::{ExtensionField, PrimeField};
    /// use a::math::numbers::Number;
    /// use a::math::poly::ModPolynomial;
    ///
    /// // AES's GF(2⁸) by x⁸ + x⁴ + x³ + x + 1: {53}·{ca} = {01} (FIPS 197).
    /// let f = ExtensionField::new(
    ///     &PrimeField::new(&Number::from(2u8)),
    ///     &ModPolynomial::from_u64(&[1, 1, 0, 1, 1, 0, 0, 0, 1], 2),
    /// );
    /// let a = f.from_u64(&[1, 1, 0, 0, 1, 0, 1, 0]); // 0x53
    /// let b = f.from_u64(&[0, 1, 0, 1, 0, 0, 1, 1]); // 0xca
    /// assert!((&a * &b).is_one());
    /// ```
    #[track_caller]
    #[must_use]
    pub fn new(base: &PrimeField, g: &ModPolynomial) -> ExtensionField {
        assert!(
            g.modulus() == base.modulus(),
            "g is modulo {}, not modulo p = {}",
            g.modulus(),
            base.modulus()
        );
        let k = g.degree().unwrap_or(0);
        assert!(k >= 1, "g must have a degree of at least 1");
        assert!(g.is_monic(), "g must be monic");
        assert!(g.is_irreducible(), "g = {g} is reducible");
        ExtensionField::unchecked(base, g.clone())
    }

    /// The field `GF(pᵏ)` over `base = GF(p)` by the polynomial
    /// [`first_irreducible`](ExtensionField::first_irreducible) gives for `k`.
    ///
    /// # Panics
    ///
    /// Panics if `k` is 0 or `2³²` or more.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::{ExtensionField, PrimeField};
    /// use a::math::numbers::Number;
    ///
    /// let f = ExtensionField::with_degree(PrimeField::bls12_381_base(), 2);
    /// // p ≡ 3 (mod 4), so x² + 1 is the first irreducible polynomial: BLS12-381's F_p².
    /// assert_eq!(f.modulus().coeffs(), [Number::from(1u8), Number::default(), Number::from(1u8)]);
    /// assert_eq!(f.x().square(), -f.one());
    /// ```
    #[track_caller]
    #[must_use]
    pub fn with_degree(base: &PrimeField, k: usize) -> ExtensionField {
        ExtensionField::unchecked(base, ext_find::first_irreducible(base, k))
    }

    /// The smallest monic irreducible polynomial of degree `k` modulo `p`, in the order of the
    /// coefficients from `xᵏ⁻¹` down to the constant, each in `0..p` (the order of the integers
    /// `Σ gᵢ·pⁱ`): `x` for `k = 1`; else the binomial `xᵏ + c` of the smallest `c` when one is
    /// irreducible, which by Lidl and Niederreiter's Theorem 3.75 is when every prime factor `r`
    /// of `k` divides `p − 1` (and `p ≡ 1 (mod 4)` if `4 | k`), `−c` then being the first number
    /// that is not an `r`-th power modulo `p` for any of them; else the first of the candidates
    /// that follow, `xᵏ + x + c` for `c = 1, 2, …` first, by Ben-Or's test, after the test of the
    /// discriminant that Stickelberger's theorem gives for odd `p`, which sorts out about half of
    /// them. A random polynomial of degree `k` is irreducible with a probability of about `1/k`,
    /// so for large `p` the first irreducible `xᵏ + x + c` comes after about `k` candidates (37 for
    /// BN254's prime and `k = 64`, 253 for BLS12-381's); for small `p` every candidate is tried in
    /// order. For `p = 2` and `k = 8` this is AES's `x⁸ + x⁴ + x³ + x + 1`.
    ///
    /// PARI's `ffinit` makes another choice (its own construction, deterministic but not the
    /// smallest polynomial), and FLINT's `fq_ctx_init` a Conway polynomial from its table or a
    /// random one: the fields are isomorphic, but their elements are written in other bases.
    ///
    /// # Panics
    ///
    /// Panics if `k` is 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::{ExtensionField, PrimeField};
    /// use a::math::numbers::Number;
    ///
    /// let two = PrimeField::new(&Number::from(2u8));
    /// let g = ExtensionField::first_irreducible(&two, 8);
    /// assert_eq!(g.word_coeffs(), Some(&[1, 1, 0, 1, 1, 0, 0, 0, 1][..]));
    /// // Modulo 7, every prime factor of 3 divides 6: x³ + 2 (x³ − 5, 5 not a cube).
    /// let seven = PrimeField::new(&Number::from(7u8));
    /// assert_eq!(ExtensionField::first_irreducible(&seven, 3).word_coeffs(), Some(&[2, 0, 0, 1][..]));
    /// ```
    #[track_caller]
    #[must_use]
    pub fn first_irreducible(base: &PrimeField, k: usize) -> ModPolynomial {
        ext_find::first_irreducible(base, k)
    }

    /// The field of a `g` known to be monic and irreducible of degree at least 1.
    fn unchecked(base: &PrimeField, g: ModPolynomial) -> ExtensionField {
        let k = g.degree().expect("g is not zero");
        let p = base.modulus();
        let low: Vec<(usize, Number)> = (0..k)
            .filter_map(|i| {
                let c = g.coeff(i);
                (!c.is_zero()).then(|| (i, p - &c))
            })
            .collect();
        let q = p.pow(u32::try_from(k).expect("a degree below 2^32"));
        let poly_min = ext_arith::poly_min(p, low.len());
        let mut f = ExtensionField {
            base: base.clone(),
            k,
            g,
            low,
            q,
            poly_min,
            frob: OnceLock::new(),
            frob_words: OnceLock::new(),
            traces: OnceLock::new(),
            low_form: OnceLock::new(),
            traces_form: OnceLock::new(),
            root_x: OnceLock::new(),
            shanks: OnceLock::new(),
            units: OnceLock::new(),
            zech: None,
            kind: Kind::General,
            word: (0, 0),
        };
        f.zech = ext_zech::Zech::new(&f).map(Arc::new);
        let word = with_form!(f.base.form_ref(), g => g.word_prime());
        if let Some(w) = word {
            f.word = w;
        }
        if f.zech.is_some() {
            f.kind = Kind::Log;
        } else if word.is_some() {
            f.kind = Kind::Words;
        }
        f
    }

    /// The base field `GF(p)`: the copy the field holds.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::{ExtensionField, PrimeField};
    /// use a::math::numbers::Number;
    ///
    /// let f = ExtensionField::with_degree(PrimeField::bn254_base(), 2);
    /// assert_eq!(f.base().modulus(), PrimeField::bn254_base().modulus());
    /// assert_eq!(f.characteristic(), PrimeField::bn254_base().modulus());
    /// assert_eq!(f.degree(), 2);
    /// assert_eq!(f.order(), &f.characteristic().pow(2));
    /// assert_eq!(f.modulus().degree(), Some(2));
    /// ```
    #[must_use]
    pub fn base(&self) -> &PrimeField {
        &self.base
    }

    /// The characteristic `p`.
    #[must_use]
    pub fn characteristic(&self) -> &Number {
        self.base.modulus()
    }

    /// The degree `k` over `GF(p)`.
    #[must_use]
    pub fn degree(&self) -> usize {
        self.k
    }

    /// The polynomial `g`, monic and irreducible, of degree `k`.
    #[must_use]
    pub fn modulus(&self) -> &ModPolynomial {
        &self.g
    }

    /// The number of elements, `pᵏ`.
    #[must_use]
    pub fn order(&self) -> &Number {
        &self.q
    }

    /// The element of the polynomial `Σ cᵢ·xⁱ` of the coefficients `coeffs`, the constant first,
    /// of any sign, size and number, each reduced modulo `p` and the polynomial modulo `g`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::{ExtensionField, PrimeField};
    /// use a::math::numbers::Number;
    ///
    /// let f = ExtensionField::with_degree(&PrimeField::new(&Number::from(5u8)), 2); // x² + 2
    /// // x² ≡ −2 = 3, and −1 ≡ 4.
    /// let n = |v: i64| Number::from(v);
    /// assert_eq!(f.element(&[n(-1), n(0), n(1)]), f.from_u64(&[2]));
    /// ```
    #[must_use]
    pub fn element(&self, coeffs: &[Number]) -> Fpk<'_> {
        let c = coeffs.iter().map(|x| self.base.element(x)).collect();
        self.reduced(c)
    }

    /// The element of the polynomial of the coefficients `coeffs`, the constant first, as
    /// [`element`](ExtensionField::element).
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::{ExtensionField, PrimeField};
    /// use a::math::numbers::Number;
    ///
    /// let f = ExtensionField::with_degree(&PrimeField::new(&Number::from(7u8)), 3); // x³ + 2
    /// assert_eq!(f.from_u64(&[1, 2, 3, 1]), f.from_u64(&[6, 2, 3])); // x³ ≡ −2 = 5
    /// assert_eq!(f.from_u64(&[9]), f.from_u64(&[2]));
    /// ```
    #[must_use]
    pub fn from_u64(&self, coeffs: &[u64]) -> Fpk<'_> {
        let c = coeffs.iter().map(|&x| self.base.from_u64(x)).collect();
        self.reduced(c)
    }

    /// The element of the polynomial of the coefficients `coeffs`, elements of `GF(p)`, the
    /// constant first, of any number, reduced modulo `g`.
    ///
    /// # Panics
    ///
    /// Panics if a coefficient is of a field of another prime.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::{ExtensionField, PrimeField};
    /// use a::math::numbers::Number;
    ///
    /// let base = PrimeField::new(&Number::from(11u8));
    /// let f = ExtensionField::with_degree(&base, 2);
    /// let a = f.from_fp(&[base.from_u64(3), base.from_u64(4)]);
    /// assert_eq!(a, f.from_u64(&[3, 4]));
    /// assert_eq!(a.coeffs()[1], base.from_u64(4));
    /// ```
    #[track_caller]
    #[must_use]
    pub fn from_fp(&self, coeffs: &[Fp<'_>]) -> Fpk<'_> {
        let c = coeffs.iter().map(|x| self.lift(x)).collect();
        self.reduced(c)
    }

    /// The element `c` of `GF(p)` in `GF(pᵏ)`: the constant polynomial `c`.
    ///
    /// # Panics
    ///
    /// Panics if `c` is of a field of another prime.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::{ExtensionField, PrimeField};
    /// use a::math::numbers::Number;
    ///
    /// let base = PrimeField::new(&Number::from(11u8));
    /// let f = ExtensionField::with_degree(&base, 3);
    /// let c = f.from_base(&base.from_u64(5));
    /// assert_eq!(c.to_base(), Some(base.from_u64(5)));
    /// assert_eq!(c.norm(), base.from_u64(5 * 5 * 5));
    /// ```
    #[track_caller]
    #[must_use]
    pub fn from_base(&self, c: &Fp<'_>) -> Fpk<'_> {
        let mut v = vec![self.base.zero(); self.k];
        v[0] = self.lift(c);
        Fpk::of(self, v)
    }

    /// The element of the polynomial `a`, of any degree, reduced modulo `g`.
    ///
    /// # Panics
    ///
    /// Panics if the modulus of `a` is not `p`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::{ExtensionField, PrimeField};
    /// use a::math::numbers::Number;
    ///
    /// use a::math::poly::ModPolynomial;
    ///
    /// let f = ExtensionField::with_degree(&PrimeField::new(&Number::from(3u8)), 2); // x² + 1
    /// let a = f.from_polynomial(&ModPolynomial::from_u64(&[0, 0, 0, 1], 3)); // x³ = −x
    /// assert_eq!(a, -f.x());
    /// assert_eq!(a.to_polynomial(), ModPolynomial::from_u64(&[0, 2], 3));
    /// ```
    #[track_caller]
    #[must_use]
    pub fn from_polynomial(&self, a: &ModPolynomial) -> Fpk<'_> {
        assert!(
            a.modulus() == self.base.modulus(),
            "a polynomial modulo {} for a field of characteristic {}",
            a.modulus(),
            self.base.modulus()
        );
        self.element(&a.coeffs())
    }

    /// 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::{ExtensionField, PrimeField};
    /// use a::math::numbers::Number;
    ///
    /// let f = ExtensionField::with_degree(&PrimeField::new(&Number::from(3u8)), 2);
    /// assert!(f.zero().is_zero() && f.one().is_one());
    /// assert_eq!(&f.one() + &f.zero(), f.one());
    /// ```
    #[must_use]
    pub fn zero(&self) -> Fpk<'_> {
        match self.kind {
            Kind::Log => Fpk::of_log(self, ext_zech::ZERO),
            Kind::Words => Fpk::of_words(self, vec![0; self.k]),
            Kind::General => Fpk::of(self, vec![self.base.zero(); self.k]),
        }
    }

    /// 1.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::{ExtensionField, PrimeField};
    /// use a::math::numbers::Number;
    ///
    /// let f = ExtensionField::with_degree(&PrimeField::new(&Number::from(3u8)), 2);
    /// assert_eq!(f.one().pow_u64(100), f.one());
    /// ```
    #[must_use]
    pub fn one(&self) -> Fpk<'_> {
        match self.kind {
            Kind::Log => Fpk::of_log(self, 0),
            Kind::Words => {
                let mut w = vec![0; self.k];
                w[0] = inline::<1>(self.base.one().raw())[0];
                Fpk::of_words(self, w)
            }
            Kind::General => self.from_base(&self.base.one()),
        }
    }

    /// The class of `x`, a root of `g`, which generates the field over `GF(p)`: `x` itself for a
    /// degree of 2 or more, and the root of `g` in `GF(p)` for a degree of 1.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::{ExtensionField, PrimeField};
    /// use a::math::numbers::Number;
    ///
    /// let f = ExtensionField::with_degree(&PrimeField::new(&Number::from(2u8)), 8);
    /// // g(x) = x⁸ + x⁴ + x³ + x + 1 = 0.
    /// let x = f.x();
    /// assert!((x.pow_u64(8) + x.pow_u64(4) + x.pow_u64(3) + &x + f.one()).is_zero());
    /// ```
    #[must_use]
    pub fn x(&self) -> Fpk<'_> {
        self.from_u64(&[0, 1])
    }

    /// The inverses of `xs`, by Montgomery's trick: one inversion and `3(n − 1)` products. A zero
    /// stays zero, as it does in [`PrimeField::batch_invert`].
    ///
    /// # Panics
    ///
    /// Panics if an element is not of this field.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::{ExtensionField, PrimeField};
    /// use a::math::numbers::Number;
    ///
    /// let f = ExtensionField::with_degree(&PrimeField::new(&Number::from(3u8)), 4);
    /// let xs = [f.x(), f.zero(), f.from_u64(&[1, 2, 0, 1])];
    /// let inv = f.batch_invert(&xs);
    /// assert!((&inv[0] * &xs[0]).is_one());
    /// assert!(inv[1].is_zero());
    /// assert!((&inv[2] * &xs[2]).is_one());
    /// ```
    #[track_caller]
    #[must_use]
    pub fn batch_invert<'f>(&'f self, xs: &[Fpk<'f>]) -> Vec<Fpk<'f>> {
        for x in xs {
            self.check(x);
        }
        // prefix[i] = the product of the nonzero xs before i.
        let mut prefix = Vec::with_capacity(xs.len());
        let mut acc = self.one();
        for x in xs {
            prefix.push(acc.clone());
            if !x.is_zero() {
                acc = &acc * x;
            }
        }
        let mut inv = acc
            .inverse()
            .expect("a product of nonzero elements is nonzero");
        let mut out = vec![self.zero(); xs.len()];
        for i in (0..xs.len()).rev() {
            if xs[i].is_zero() {
                continue;
            }
            out[i] = &inv * &prefix[i];
            inv = &inv * &xs[i];
        }
        out
    }

    /// The factorisation of `pᵏ − 1`, the order of the multiplicative group, which
    /// [`Fpk::multiplicative_order`], [`Fpk::is_primitive`] and
    /// [`primitive_element`](ExtensionField::primitive_element) need: computed on the first call
    /// (unless given by [`with_factored_unit_order`](ExtensionField::with_factored_unit_order)) by
    /// [`Number::factor`] on the values `Φ_d(p)` of the cyclotomic polynomials for the divisors
    /// `d` of `k`, whose product it is. That takes the time of factoring numbers of up to
    /// `φ(k)·log₂ p` bits: seconds to hours beyond about 200 bits, depending on their second
    /// largest prime factor.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::{ExtensionField, PrimeField};
    /// use a::math::numbers::Number;
    ///
    /// // 2⁸ − 1 = 3·5·17.
    /// let f = ExtensionField::with_degree(&PrimeField::new(&Number::from(2u8)), 8);
    /// let primes: Vec<String> =
    ///     f.factored_unit_order().factors().iter().map(|(r, _)| r.to_string()).collect();
    /// assert_eq!(primes, ["3", "5", "17"]);
    /// ```
    #[must_use]
    pub fn factored_unit_order(&self) -> &Factorization {
        self.units
            .get_or_init(|| ext_roots::factor_unit_order(self.base.modulus(), self.k))
    }

    /// The field with the factorisation of `pᵏ − 1` given, for the methods that need it, when
    /// it is known and hard to compute. The primes are not tested: a composite one makes those
    /// methods wrong.
    ///
    /// # Panics
    ///
    /// Panics if the product of `factors` is not `pᵏ − 1`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::{ExtensionField, PrimeField};
    /// use a::math::numbers::{ByteOrder, Factorization, Number};
    ///
    /// // 7² − 1 = 48 = 2⁴·3.
    /// let f = ExtensionField::with_degree(&PrimeField::new(&Number::from(7u8)), 2)
    ///     .with_factored_unit_order(Factorization::new(
    ///         vec![(Number::from(2u8), 4), (Number::from(3u8), 1)],
    ///         ByteOrder::LittleEndian,
    ///     ));
    /// assert_eq!(f.x().multiplicative_order(), Some(Number::from(4u8))); // x² = −1
    /// ```
    #[track_caller]
    #[must_use]
    pub fn with_factored_unit_order(self, factors: Factorization) -> ExtensionField {
        let units = &self.q - &Number::from(1u8);
        assert!(
            factors.value() == units,
            "the factors multiply to {}, not to p^k - 1 = {units}",
            factors.value()
        );
        let mut f = self;
        f.units = OnceLock::from(factors);
        f
    }

    /// A primitive element, a generator of the multiplicative group: the first element that is
    /// one, in the order of the coefficients from the top down (`x`, `x + 1`, … for a degree of 2
    /// or more; `2, 3, …` modulo `p` for a degree of 1). It needs the factorisation of
    /// `pᵏ − 1` ([`factored_unit_order`](ExtensionField::factored_unit_order)) and tests each
    /// candidate by `a^((pᵏ − 1)/r) ≠ 1` for each prime `r`. A random element is primitive with
    /// the probability `φ(pᵏ − 1)/(pᵏ − 1)`, so if the candidates behaved like random elements
    /// about `(pᵏ − 1)/φ(pᵏ − 1)` of them, `O(log log pᵏ)`, would be tried on average; that is a
    /// heuristic, not a bound on the first one in a fixed order: it takes 39 candidates in
    /// `GF(37³)`, where the average would be 3.55 (as the least primitive root modulo a prime is
    /// not bounded by it either).
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::{ExtensionField, PrimeField};
    /// use a::math::numbers::Number;
    ///
    /// let f = ExtensionField::with_degree(&PrimeField::new(&Number::from(3u8)), 2); // x² + 1
    /// let a = f.primitive_element();
    /// assert_eq!(a, f.from_u64(&[1, 1])); // x has order 4, x + 1 order 8
    /// assert_eq!(a.multiplicative_order(), Some(Number::from(8u8)));
    /// ```
    #[must_use]
    pub fn primitive_element(&self) -> Fpk<'_> {
        ext_roots::primitive_element(self)
    }

    /// The element whose coefficients are the digits of `n` in base `p`, the constant the
    /// lowest: the `n`-th element in the order of the coefficients from the top down.
    pub(super) fn digits_element(&self, n: &Number) -> Fpk<'_> {
        let p = self.base.modulus();
        let mut c = Vec::with_capacity(self.k);
        let mut n = n.clone();
        for _ in 0..self.k {
            let (q, r) = n.div_rem(p);
            c.push(self.base.element(&r));
            n = q;
        }
        Fpk::of(self, c)
    }

    /// The element of the coefficients `c` of `GF(p)`, of any number, reduced modulo `g`.
    pub(super) fn reduced<'f>(&'f self, mut c: Vec<Fp<'f>>) -> Fpk<'f> {
        if c.len() > self.k {
            ext_arith::reduce(self, &mut c);
        } else {
            c.resize(self.k, self.base.zero());
        }
        Fpk::of(self, c)
    }

    /// The tables of the field, for [`Kind::Log`].
    #[inline(always)]
    pub(super) fn zech(&self) -> &ext_zech::Zech {
        self.zech.as_deref().expect("a field of Zech's tables")
    }

    /// `c` as an element of the base field this field holds.
    #[track_caller]
    fn lift(&self, c: &Fp<'_>) -> Fp<'_> {
        assert!(
            c.field().modulus() == self.base.modulus(),
            "elements of different fields"
        );
        self.base.element(&c.value())
    }

    /// Panics unless `x` is an element of this field.
    #[track_caller]
    pub(super) fn check(&self, x: &Fpk<'_>) {
        assert!(
            core::ptr::eq(self, x.f) || self == x.f,
            "elements of different fields"
        );
    }
}

impl PartialEq for ExtensionField {
    /// Two fields are equal when they have the same prime and the same polynomial.
    fn eq(&self, other: &ExtensionField) -> bool {
        self.base.modulus() == other.base.modulus() && self.g == other.g
    }
}

impl Eq for ExtensionField {}

impl Hash for ExtensionField {
    fn hash<H: Hasher>(&self, h: &mut H) {
        self.g.hash(h);
    }
}

impl fmt::Debug for ExtensionField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExtensionField")
            .field("p", self.base.modulus())
            .field("k", &self.k)
            .field("g", &self.g.to_polynomial().to_string())
            .finish()
    }
}

/// An element of an [`ExtensionField`] `GF(pᵏ)`, which it borrows: a polynomial of degree below
/// `k` in the class of `x`, held as its `k` coefficients in `GF(p)`.
///
/// # Examples
///
/// ```
/// use a::math::field::{ExtensionField, PrimeField};
/// use a::math::numbers::Number;
///
/// let f = ExtensionField::with_degree(&PrimeField::new(&Number::from(3u8)), 5);
/// let a = f.from_u64(&[1, 2, 0, 1, 1]);
/// assert_eq!(a.pow(f.order()), a); // a^(3⁵) = a
/// assert_eq!(a.frobenius(5), a);
/// assert_eq!(a.frobenius(2), a.pow_u64(9));
/// assert_eq!((&a + &a.frobenius(1)).trace(), a.trace().double());
/// ```
#[derive(Clone)]
pub struct Fpk<'f> {
    f: &'f ExtensionField,
    /// The element as its field holds it ([`Kind`]).
    pub(super) r: Repr<'f>,
}

/// An element as its field holds it; the compact forms keep the coefficients as elements of the
/// base field too, once something has asked for them.
#[derive(Clone, Debug)]
pub(super) enum Repr<'f> {
    /// Its `k` coefficients, the constant first.
    General(Vec<Fp<'f>>),
    /// The `k` words of its coefficients in the base field's form.
    Words(Vec<u64>, OnceLock<Vec<Fp<'f>>>),
    /// Its logarithm to the base of the primitive element of Zech's tables, or
    /// [`ext_zech::ZERO`] for 0.
    Log(u32, OnceLock<Vec<Fp<'f>>>),
}

impl PartialEq for Repr<'_> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Repr::General(a), Repr::General(b)) => a == b,
            (Repr::Words(a, _), Repr::Words(b, _)) => a == b,
            (Repr::Log(a, _), Repr::Log(b, _)) => a == b,
            _ => false,
        }
    }
}

impl Eq for Repr<'_> {}

impl Hash for Repr<'_> {
    fn hash<H: Hasher>(&self, h: &mut H) {
        match self {
            Repr::General(c) => c.hash(h),
            Repr::Words(w, _) => w.hash(h),
            Repr::Log(l, _) => l.hash(h),
        }
    }
}

impl<'f> Fpk<'f> {
    /// The element of the coefficients `c`, `k` of them, in its field's form.
    pub(super) fn of(f: &'f ExtensionField, c: Vec<Fp<'f>>) -> Fpk<'f> {
        debug_assert_eq!(c.len(), f.k);
        let r = match f.kind {
            Kind::General => Repr::General(c),
            Kind::Words => {
                let w = c.iter().map(|x| inline::<1>(x.raw())[0]).collect();
                Repr::Words(w, OnceLock::from(c))
            }
            Kind::Log => Repr::Log(f.zech().log_of_coeffs(&c), OnceLock::from(c)),
        };
        Fpk { f, r }
    }

    /// The element of the words `w` of its coefficients (for a field of a word prime).
    #[inline]
    pub(super) fn of_words(f: &'f ExtensionField, w: Vec<u64>) -> Fpk<'f> {
        debug_assert_eq!(w.len(), f.k);
        let r = match f.kind {
            Kind::Log => Repr::Log(f.zech().log_of_words(&w), OnceLock::new()),
            _ => Repr::Words(w, OnceLock::new()),
        };
        Fpk { f, r }
    }

    /// The element of the logarithm `l` (for a field of Zech's tables).
    #[inline]
    pub(super) fn of_log(f: &'f ExtensionField, l: u32) -> Fpk<'f> {
        Fpk {
            f,
            r: Repr::Log(l, OnceLock::new()),
        }
    }

    /// The coefficients, made from the compact forms on the first use.
    pub(super) fn c(&self) -> &[Fp<'f>] {
        match &self.r {
            Repr::General(c) => c,
            Repr::Words(w, cs) => cs.get_or_init(|| {
                w.iter()
                    .map(|&x| Fp::from_raw(&self.f.base, held(&[x])))
                    .collect()
            }),
            Repr::Log(l, cs) => cs.get_or_init(|| self.f.zech().coeffs_of_log(self.f, *l)),
        }
    }

    /// The field.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::{ExtensionField, PrimeField};
    /// use a::math::numbers::Number;
    ///
    /// let f = ExtensionField::with_degree(&PrimeField::new(&Number::from(5u8)), 3);
    /// assert_eq!(f.x().field(), &f);
    /// ```
    #[must_use]
    pub fn field(&self) -> &'f ExtensionField {
        self.f
    }

    /// The `k` coefficients in `GF(p)`, the constant first.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::{ExtensionField, PrimeField};
    /// use a::math::numbers::Number;
    ///
    /// let f = ExtensionField::with_degree(&PrimeField::new(&Number::from(5u8)), 3);
    /// let c: Vec<Number> = f.from_u64(&[4, 3]).coeffs().iter().map(|c| c.value()).collect();
    /// assert_eq!(c, [Number::from(4u8), Number::from(3u8), Number::default()]);
    /// ```
    #[must_use]
    pub fn coeffs(&self) -> &[Fp<'f>] {
        self.c()
    }

    /// The polynomial of degree below `k` that the element is, modulo `p`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::{ExtensionField, PrimeField};
    /// use a::math::numbers::Number;
    ///
    /// use a::math::poly::ModPolynomial;
    ///
    /// let f = ExtensionField::with_degree(&PrimeField::new(&Number::from(5u8)), 3);
    /// let a = f.from_u64(&[4, 3]);
    /// assert_eq!(a.to_polynomial(), ModPolynomial::from_u64(&[4, 3], 5));
    /// assert_eq!(f.from_polynomial(&a.to_polynomial()), a);
    /// ```
    #[must_use]
    pub fn to_polynomial(&self) -> ModPolynomial {
        ext_arith::to_polynomial(self.f, self.c())
    }

    /// The element as an element of `GF(p)`, when it is one (a constant), else `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::{ExtensionField, PrimeField};
    /// use a::math::numbers::Number;
    ///
    /// let f = ExtensionField::with_degree(&PrimeField::new(&Number::from(5u8)), 2);
    /// assert_eq!(f.from_u64(&[3]).to_base().unwrap().value(), Number::from(3u8));
    /// assert!(f.x().to_base().is_none());
    /// assert!(f.x().square().to_base().is_some()); // x² = −2
    /// ```
    #[must_use]
    pub fn to_base(&self) -> Option<Fp<'f>> {
        let c = self.c();
        c[1..].iter().all(Fp::is_zero).then(|| c[0].clone())
    }

    /// Whether it is 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::{ExtensionField, PrimeField};
    /// use a::math::numbers::Number;
    ///
    /// let f = ExtensionField::with_degree(&PrimeField::new(&Number::from(2u8)), 4);
    /// assert!((&f.x() + &f.x()).is_zero()); // characteristic 2
    /// assert!(!f.x().is_zero());
    /// ```
    #[must_use]
    pub fn is_zero(&self) -> bool {
        match &self.r {
            Repr::General(c) => c.iter().all(Fp::is_zero),
            Repr::Words(w, _) => w.iter().all(|&x| x == 0),
            Repr::Log(l, _) => *l == ext_zech::ZERO,
        }
    }

    /// Whether it is 1.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::{ExtensionField, PrimeField};
    /// use a::math::numbers::Number;
    ///
    /// let f = ExtensionField::with_degree(&PrimeField::new(&Number::from(2u8)), 4);
    /// assert!(f.x().pow_u64(15).is_one()); // x⁴ + x + 1 is primitive
    /// assert!(!f.x().is_one());
    /// ```
    #[must_use]
    pub fn is_one(&self) -> bool {
        match &self.r {
            Repr::General(c) => c[0].is_one() && c[1..].iter().all(Fp::is_zero),
            Repr::Words(w, _) => {
                w[0] == inline::<1>(self.f.base.one().raw())[0] && w[1..].iter().all(|&x| x == 0)
            }
            Repr::Log(l, _) => *l == 0,
        }
    }

    /// The square.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::{ExtensionField, PrimeField};
    /// use a::math::numbers::Number;
    ///
    /// let f = ExtensionField::with_degree(&PrimeField::new(&Number::from(7u8)), 2); // x² + 1
    /// assert_eq!(f.from_u64(&[1, 1]).square(), f.from_u64(&[0, 2])); // (1 + x)² = 2x
    /// ```
    #[must_use]
    pub fn square(&self) -> Fpk<'f> {
        ext_arith::square(self)
    }

    /// Twice the element.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::{ExtensionField, PrimeField};
    /// use a::math::numbers::Number;
    ///
    /// let f = ExtensionField::with_degree(&PrimeField::new(&Number::from(7u8)), 2);
    /// assert_eq!(f.from_u64(&[4, 5]).double(), f.from_u64(&[1, 3]));
    /// ```
    #[must_use]
    pub fn double(&self) -> Fpk<'f> {
        self + self
    }

    /// The `e`-th power, by squares and products from the top bit of `e` down, `e` first reduced
    /// modulo `pᵏ − 1` for an element that is not zero; a negative `e` is a power of the inverse,
    /// and `a⁰ = 1`, also for `a = 0`: at most `2·k·log₂ p` products.
    ///
    /// # Panics
    ///
    /// Panics if `e` is negative and the element is 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::{ExtensionField, PrimeField};
    /// use a::math::numbers::Number;
    ///
    /// let f = ExtensionField::with_degree(&PrimeField::new(&Number::from(7u8)), 2); // x² + 1
    /// let x = f.x();
    /// assert_eq!(x.pow(&Number::from(3u8)), -&x);
    /// assert_eq!(x.pow(&Number::from(-1)), -&x); // x⁻¹ = −x
    /// assert_eq!(x.pow(&Number::from(48u8)), f.one()); // 7² − 1 = 48
    /// ```
    #[track_caller]
    #[must_use]
    pub fn pow(&self, e: &Number) -> Fpk<'f> {
        if e.is_negative() {
            let inv = self.inverse().expect("0 has no inverse");
            return inv.pow(&-e);
        }
        ext_arith::pow(self, e)
    }

    /// The `e`-th power.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::{ExtensionField, PrimeField};
    /// use a::math::numbers::Number;
    ///
    /// let f = ExtensionField::with_degree(&PrimeField::new(&Number::from(3u8)), 3);
    /// let a = f.from_u64(&[1, 1, 2]);
    /// assert_eq!(a.pow_u64(26), f.one()); // 3³ − 1 = 26
    /// assert_eq!(a.pow_u64(3), a.frobenius(1));
    /// ```
    #[must_use]
    pub fn pow_u64(&self, e: u64) -> Fpk<'f> {
        self.pow(&Number::from(e))
    }

    /// The inverse, or `None` for 0: by the extended Euclidean algorithm on the element and `g`
    /// ([`ModPolynomial::xgcd`], half gcd for long polynomials), `s·a + t·g = 1` giving `s`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::{ExtensionField, PrimeField};
    /// use a::math::numbers::Number;
    ///
    /// let f = ExtensionField::with_degree(PrimeField::bls12_381_base(), 6);
    /// let a = f.from_u64(&[1, 2, 3, 4, 5, 6]);
    /// assert!((&a * &a.inverse().unwrap()).is_one());
    /// assert!(f.zero().inverse().is_none());
    /// ```
    #[must_use]
    pub fn inverse(&self) -> Option<Fpk<'f>> {
        ext_arith::inverse(self)
    }

    /// The `i`-th power of the Frobenius map, `a ↦ a^(pⁱ)`, `i` taken modulo `k`: `i` products by
    /// the matrix of `a ↦ aᵖ`, whose column `j` is `x^(p·j) mod g` (for `a = Σ aⱼxʲ`,
    /// `aᵖ = Σ aⱼ·x^(p·j)`, as `aⱼᵖ = aⱼ` in `GF(p)`), computed on the first call: `xᵖ mod g` by
    /// [`ModPolynomial::pow_mod`], `O(log p)` squares modulo `g`, then `k − 2` products modulo
    /// `g`. Then `O(i·k²)` operations in `GF(p)` a call, and `k²` elements of `GF(p)` kept.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::{ExtensionField, PrimeField};
    /// use a::math::numbers::Number;
    ///
    /// let f = ExtensionField::with_degree(PrimeField::bn254_base(), 6);
    /// let a = f.from_u64(&[5, 0, 7, 1]);
    /// assert_eq!(a.frobenius(1), a.pow(f.characteristic()));
    /// assert_eq!(a.frobenius(2).frobenius(4), a);
    /// ```
    #[must_use]
    pub fn frobenius(&self, i: usize) -> Fpk<'f> {
        ext_maps::frobenius(self, i % self.f.k)
    }

    /// The norm to `GF(p)`, `N(a) = a·aᵖ·…·a^(pᵏ⁻¹) = a^((pᵏ − 1)/(p − 1))`: the resultant
    /// `res(g, a)` ([`ModPolynomial::resultant`]), which is the product of `a` at the `k` roots
    /// of the monic `g`, `x, xᵖ, …`. `O(k²)` operations in `GF(p)`, fewer by the half gcd for
    /// long polynomials.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::{ExtensionField, PrimeField};
    /// use a::math::numbers::Number;
    ///
    /// let f = ExtensionField::with_degree(&PrimeField::new(&Number::from(11u8)), 3);
    /// let (a, b) = (f.from_u64(&[2, 3, 4]), f.from_u64(&[7, 0, 1]));
    /// assert_eq!((&a * &b).norm(), &a.norm() * &b.norm());
    /// assert_eq!(a.norm(), a.pow(&Number::from(1 + 11 + 121u32)).to_base().unwrap());
    /// ```
    #[must_use]
    pub fn norm(&self) -> Fp<'f> {
        ext_maps::norm(self)
    }

    /// The trace to `GF(p)`, `Tr(a) = a + aᵖ + … + a^(pᵏ⁻¹)`: `Σ aᵢ·Tr(xⁱ)` with the traces of
    /// the powers of `x`, the power sums of the roots of `g`, computed on the first call from the
    /// coefficients of `g` by Newton's identities in `O(k²)` operations; then `O(k)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::{ExtensionField, PrimeField};
    /// use a::math::numbers::Number;
    ///
    /// let f = ExtensionField::with_degree(&PrimeField::new(&Number::from(13u8)), 4);
    /// let a = f.from_u64(&[3, 1, 4, 1]);
    /// let sum = (0..4).fold(f.zero(), |s, i| s + a.frobenius(i));
    /// assert_eq!(sum.to_base().unwrap(), a.trace());
    /// ```
    #[must_use]
    pub fn trace(&self) -> Fp<'f> {
        ext_maps::trace(self)
    }

    /// The quadratic character: 1 for a nonzero square, −1 for a non-square, 0 for 0. For odd
    /// `p` it is the Legendre symbol of the norm, as `a^((pᵏ − 1)/2) = N(a)^((p − 1)/2)`; for
    /// `p = 2` every element is a square.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::{ExtensionField, PrimeField};
    /// use a::math::numbers::Number;
    ///
    /// let f = ExtensionField::with_degree(&PrimeField::new(&Number::from(3u8)), 2); // x² + 1
    /// assert_eq!(f.x().legendre(), 1); // N(x) = 1, a square: x = (x + x²)²
    /// assert_eq!(f.from_u64(&[1, 1]).legendre(), -1); // N(1 + x) = 2, not a square mod 3
    /// assert_eq!(f.zero().legendre(), 0);
    /// // Every element of GF(3) is a square in GF(9).
    /// assert_eq!(f.from_u64(&[2]).legendre(), 1);
    /// ```
    #[must_use]
    pub fn legendre(&self) -> i32 {
        ext_roots::legendre(self)
    }

    /// Whether the element is a square (0 is).
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::{ExtensionField, PrimeField};
    /// use a::math::numbers::Number;
    ///
    /// let f = ExtensionField::with_degree(&PrimeField::new(&Number::from(11u8)), 3);
    /// let a = f.from_u64(&[2, 7, 1]);
    /// assert!(a.square().is_square());
    /// assert_eq!(a.is_square(), a.sqrt().is_some());
    /// ```
    #[must_use]
    pub fn is_square(&self) -> bool {
        self.legendre() >= 0
    }

    /// A square root, or `None` if the element is not a square. Of the two roots `r` and `−r`
    /// the smaller in the order of the coefficients from the top down (that of the integers
    /// `Σ rᵢ·pⁱ`: the root whose highest nonzero coefficient is below `p/2`), so that for a degree
    /// of 1 it is [`Fp::sqrt`]'s; 0 for 0, and the only root for `p = 2`.
    ///
    /// For odd `p`: the quadratic character first, through the norm; then a square root in
    /// `GF(p)` for a constant that is a square there; else Tonelli and Shanks's algorithm in
    /// `GF(pᵏ)`, `pᵏ − 1 = 2ˢ·t` with `t` odd, by the power `a^((t − 1)/2)` and at most `s²`
    /// more squares and products, with `z^t` for the first non-square `z` (a constant for odd
    /// `k`, `x + c` and on for even `k`) found once. For `p = 2` the inverse of the Frobenius
    /// map: the even and the odd part, `√a = Σ a₂ᵢ·xⁱ + √x·Σ a₂ᵢ₊₁·xⁱ`, with `√x = x^(2ᵏ⁻¹)` computed
    /// once, one product.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::{ExtensionField, PrimeField};
    /// use a::math::numbers::Number;
    ///
    /// let f = ExtensionField::with_degree(PrimeField::bls12_381_base(), 2);
    /// let a = f.from_u64(&[2, 9]);
    /// let r = a.square().sqrt().unwrap();
    /// assert!(r == a || r == -&a);
    /// assert_eq!(r.square(), a.square());
    /// let n = f.x(); // the norm of x is 1, a square, so x is a square in F_p²
    /// assert_eq!(n.sqrt().unwrap().square(), n);
    /// ```
    #[must_use]
    pub fn sqrt(&self) -> Option<Fpk<'f>> {
        ext_roots::sqrt(self)
    }

    /// The multiplicative order, the least `n ≥ 1` with `aⁿ = 1`, or `None` for 0: from the
    /// factorisation of `pᵏ − 1` ([`ExtensionField::factored_unit_order`]): from `n = pᵏ − 1`,
    /// for each prime power `rᵉ` of it, `n` divided by `rᵉ` and multiplied by `r` again while
    /// `aⁿ ≠ 1`, one power of `a` and at most `e` powers by `r` a prime.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::{ExtensionField, PrimeField};
    /// use a::math::numbers::Number;
    ///
    /// let f = ExtensionField::with_degree(&PrimeField::new(&Number::from(2u8)), 4); // x⁴ + x + 1
    /// assert_eq!(f.x().multiplicative_order(), Some(Number::from(15u8)));
    /// assert_eq!(f.x().pow_u64(5).multiplicative_order(), Some(Number::from(3u8)));
    /// assert_eq!(f.zero().multiplicative_order(), None);
    /// ```
    #[must_use]
    pub fn multiplicative_order(&self) -> Option<Number> {
        ext_roots::multiplicative_order(self)
    }

    /// Whether the element generates the multiplicative group: `a^((pᵏ − 1)/r) ≠ 1` for each
    /// prime `r` of `pᵏ − 1`, from [`ExtensionField::factored_unit_order`].
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::{ExtensionField, PrimeField};
    /// use a::math::numbers::Number;
    ///
    /// let f = ExtensionField::with_degree(&PrimeField::new(&Number::from(2u8)), 4); // x⁴ + x + 1
    /// assert!(f.x().is_primitive());
    /// assert!(!f.x().pow_u64(3).is_primitive()); // of order 5
    /// ```
    #[must_use]
    pub fn is_primitive(&self) -> bool {
        ext_roots::is_primitive(self)
    }
}

impl PartialEq for Fpk<'_> {
    fn eq(&self, other: &Self) -> bool {
        (core::ptr::eq(self.f, other.f) || self.f == other.f) && self.r == other.r
    }
}

impl Eq for Fpk<'_> {}

impl Hash for Fpk<'_> {
    /// Hashes the coefficients. The tables the field computes on first use take no part, so
    /// clippy's `mutable_key_type` warning for a set of elements, which sees them, is a false
    /// positive.
    fn hash<H: Hasher>(&self, h: &mut H) {
        self.r.hash(h);
    }
}

impl fmt::Debug for Fpk<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} (mod {}, {})",
            self.to_polynomial().to_polynomial(),
            self.f.base.modulus(),
            self.f.g.to_polynomial()
        )
    }
}

impl fmt::Display for Fpk<'_> {
    /// Writes the polynomial of the coefficients in `0..p` as PARI/GP does: `3*x^2 + 1`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.to_polynomial().to_polynomial(), f)
    }
}

impl<'f> Neg for &Fpk<'f> {
    type Output = Fpk<'f>;
    fn neg(self) -> Fpk<'f> {
        ext_arith::neg(self)
    }
}

impl<'f> Neg for Fpk<'f> {
    type Output = Fpk<'f>;
    fn neg(self) -> Fpk<'f> {
        -&self
    }
}

/// The binary operators for the four combinations of owned and borrowed operands, and their
/// assignments, from the one on two borrows.
macro_rules! binary {
    ($tr:ident, $m:ident, $atr:ident, $am:ident, |$a:ident, $b:ident| $body:expr) => {
        impl<'f> $tr<&Fpk<'f>> for &Fpk<'f> {
            type Output = Fpk<'f>;
            #[track_caller]
            fn $m(self, rhs: &Fpk<'f>) -> Fpk<'f> {
                self.f.check(rhs);
                let ($a, $b) = (self, rhs);
                $body
            }
        }
        impl<'f> $tr<Fpk<'f>> for &Fpk<'f> {
            type Output = Fpk<'f>;
            #[track_caller]
            fn $m(self, rhs: Fpk<'f>) -> Fpk<'f> {
                self.$m(&rhs)
            }
        }
        impl<'f> $tr<&Fpk<'f>> for Fpk<'f> {
            type Output = Fpk<'f>;
            #[track_caller]
            fn $m(self, rhs: &Fpk<'f>) -> Fpk<'f> {
                (&self).$m(rhs)
            }
        }
        impl<'f> $tr<Fpk<'f>> for Fpk<'f> {
            type Output = Fpk<'f>;
            #[track_caller]
            fn $m(self, rhs: Fpk<'f>) -> Fpk<'f> {
                (&self).$m(&rhs)
            }
        }
        impl<'f> $atr<&Fpk<'f>> for Fpk<'f> {
            #[track_caller]
            fn $am(&mut self, rhs: &Fpk<'f>) {
                *self = (&*self).$m(rhs);
            }
        }
        impl<'f> $atr<Fpk<'f>> for Fpk<'f> {
            #[track_caller]
            fn $am(&mut self, rhs: Fpk<'f>) {
                *self = (&*self).$m(&rhs);
            }
        }
    };
}

binary!(Add, add, AddAssign, add_assign, |a, b| ext_arith::add(a, b));
binary!(Sub, sub, SubAssign, sub_assign, |a, b| ext_arith::sub(a, b));
binary!(Mul, mul, MulAssign, mul_assign, |a, b| ext_arith::mul(a, b));
binary!(Div, div, DivAssign, div_assign, |a, b| {
    let inv = b.inverse().expect("division by zero");
    ext_arith::mul(a, &inv)
});
