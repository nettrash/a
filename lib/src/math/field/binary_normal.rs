//! Binary fields `GF(2ᵐ)` in a Gaussian normal basis, and the conversions to and from polynomial
//! bases.
//!
//! A normal basis is `{θ, θ², θ⁴, …, θ^(2^(m−1))}` for a normal element `θ`; an element is the bit
//! string `(a₀ a₁ … a_{m−1})` of `Σ aᵢ·θ^(2ⁱ)`. Squaring is a rotation of the coordinates, the
//! trace their parity, and 1 is all ones. A Gaussian normal basis of type `T` (Ash, Blake and
//! Vanstone, "Low complexity normal bases", Discrete Applied Mathematics 25, 1989; as ANSI X9.62,
//! IEEE 1363-2000 §A.3 and FIPS 186-4 §D.3 define it) exists when `p = Tm + 1` is a prime and
//! `gcd(Tm/k, m) = 1` for the order `k` of 2 modulo `p`; then `θ = Σ_{j<T} γ^(uʲ)`, a Gauss period,
//! for a primitive `p`-th root of unity `γ` and a `u` of order `T` modulo `p`. Types 1 and 2 are
//! the optimal normal bases (Mullin, Onyszchuk, Vanstone and Wilson, 1989). One exists for every
//! `m` not divisible by 8, and the basis of the lowest type is the one the standards use.
//!
//! - Products by Massey and Omura's method, as FIPS 186-4 §D.3 has it: with `F(2ⁱuʲ mod p) = i`,
//!   `c₀ = Σ_{k=1}^{p−2} a_{F(k+1)}·b_{F(p−k)}` (plus `Σᵢ aᵢ·b_{i+m/2}` when `T` is odd), a bilinear
//!   form, and `cᵢ` the same form of `a` and `b` rotated by `i`. The form's terms, at most `Tm`,
//!   are found once; a product is then one pass over them with the coordinates in words, each term
//!   an `and` of two rotations, `O(Tm·m/64)` word operations.
//! - Inverses by Itoh and Tsujii's method ("A fast algorithm for computing multiplicative inverses
//!   in GF(2ᵐ) using normal bases", Information and Computation 78, 1988): `a⁻¹ = (a^(2^(m−1) − 1))²`
//!   by an addition chain of `m − 1`, `O(log m)` products, the squarings rotations.
//! - Conversions to a polynomial basis `GF(2)[x]/(f)`: a root `θ'` of the minimal polynomial of
//!   `θ` in that field, found by Berlekamp's trace algorithm ("Factoring polynomials over large
//!   finite fields", Math. Comp. 24, 1970): `gcd(h(y), Tr(u·y) mod h(y))` for random `u` splits the
//!   roots in two, and the parts are split again down to one root. The conversion maps `θ` to
//!   `θ'`; its matrix has the rows `θ'^(2ⁱ)` and the other direction is its inverse. Every
//!   conjugate of `θ'` gives a conversion (they differ by a rotation of the coordinates):
//!   [`NormalBasis::conversion`] takes the least as a number,
//!   [`NormalBasis::conversion_with`] the one given, as FIPS 186-4's tables do.
//!
//! Like the rest of the library this is not constant-time.

use super::binary::{BinaryField, F2m};
use crate::math::matrix::BitMatrix;
use core::fmt;
use core::hash::{Hash, Hasher};
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use std::sync::OnceLock;

/// The largest type looked for by [`NormalBasis::new`]; every `m` below 4096 that is not a multiple
/// of 8 has one of type at most 77.
const MAX_TYPE: u32 = 1 << 16;

/// The minimal polynomial of the normal element, found when a conversion first needs it.
#[derive(Clone)]
struct Minimal {
    /// Its coefficients, `m + 1` bits.
    poly: Vec<u64>,
    /// Row `i`: `y^(2ⁱ) mod g(y)`, the coordinates of `θ^(2ⁱ)` in the basis `1, θ, …, θ^(m−1)`.
    frobenius: Vec<u64>,
}

/// A Gaussian normal basis of `GF(2ᵐ)`, of type `T`.
///
/// # Examples
///
/// ```
/// use a::math::field::NormalBasis;
///
/// // The type 4 basis of GF(2⁷) of FIPS 186-4, D.3: (1010111)·(1100001) = (1011001).
/// let nb = NormalBasis::new(7);
/// assert_eq!(nb.gaussian_type(), 4);
/// // Coordinate aᵢ is bit i.
/// let a = nb.element(&[0b1110101]);
/// let b = nb.element(&[0b1000011]);
/// assert_eq!((&a * &b).words(), &[0b1001101]);
/// assert_eq!(a.square(), &a * &a);
/// ```
#[derive(Clone)]
pub struct NormalBasis {
    /// The degree `m`.
    m: u32,
    /// The type `T`.
    t: u32,
    /// The words of an element, `⌈m / 64⌉`.
    words: usize,
    /// Row `j` of the bilinear form `c₀ = Σ λⱼₗ aⱼ bₗ`: the `l` with `λⱼₗ = 1` are
    /// `cols[start[j]..start[j + 1]]`.
    start: Vec<usize>,
    /// The columns of the rows.
    cols: Vec<u32>,
    /// The minimal polynomial of `θ`, when computed.
    minimal: OnceLock<Minimal>,
}

impl NormalBasis {
    /// The Gaussian normal basis of `GF(2ᵐ)` of the lowest type, the one ANSI X9.62, IEEE 1363
    /// and FIPS 186-4 choose: type 4 for `m = 163`, 2 for 233, 6 for 283, 4 for 409 and 10 for 571.
    ///
    /// # Panics
    ///
    /// Panics if `m < 2` or `m` is a multiple of 8, for which there is no Gaussian normal basis.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::NormalBasis;
    ///
    /// assert_eq!(NormalBasis::new(163).gaussian_type(), 4);
    /// assert_eq!(NormalBasis::new(571).gaussian_type(), 10);
    /// assert_eq!(NormalBasis::new(4).gaussian_type(), 1); // 5 is prime and 2 generates GF(5)*
    /// ```
    #[track_caller]
    #[must_use]
    pub fn new(m: u32) -> NormalBasis {
        assert!(m >= 2, "a Gaussian normal basis needs m ≥ 2");
        assert!(
            !m.is_multiple_of(8),
            "GF(2^{m}) has no Gaussian normal basis: m is a multiple of 8"
        );
        let t = (1..=MAX_TYPE)
            .find(|&t| NormalBasis::exists(m, t))
            .expect("every m not divisible by 8 has a Gaussian normal basis");
        NormalBasis::with_type(m, t)
    }

    /// The Gaussian normal basis of `GF(2ᵐ)` of type `t`.
    ///
    /// # Panics
    ///
    /// Panics if there is none ([`exists`](NormalBasis::exists)).
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::NormalBasis;
    ///
    /// // GF(2¹⁶³) has bases of types 4, 16 and 36, and none of type 2.
    /// assert!(NormalBasis::exists(163, 16) && !NormalBasis::exists(163, 2));
    /// let nb = NormalBasis::with_type(163, 16);
    /// let a = nb.element(&[0x1234_5678, 0x9abc, 0x5]);
    /// assert_eq!(&a * &a.inverse().unwrap(), nb.one());
    /// ```
    #[track_caller]
    #[must_use]
    pub fn with_type(m: u32, t: u32) -> NormalBasis {
        let Some((p, u)) = parameters(m, t) else {
            panic!("GF(2^{m}) has no Gaussian normal basis of type {t}");
        };
        let (m_, p_) = (m as usize, p as usize);
        // F(2ⁱuʲ mod p) = i (FIPS 186-4, D.3, steps 3.1–3.2.3).
        let mut f = vec![u32::MAX; p_];
        let mut w = 1u64;
        for _ in 0..t {
            let mut n = w;
            for i in 0..m {
                f[n as usize] = i;
                n = 2 * n % p;
            }
            w = w * u % p;
        }
        assert!(
            f[1..].iter().all(|&i| i != u32::MAX),
            "the cosets of ⟨u⟩ cover GF(p)*"
        );
        // The terms of c₀, each pair counted modulo 2.
        let mut pairs: Vec<(u32, u32)> = (1..=p_ - 2).map(|k| (f[k + 1], f[p_ - k])).collect();
        if t % 2 == 1 {
            // −1 = 2^(m/2)·uʲ: the products aᵢ·b_{i+m/2} of the n with n + n' = 0.
            pairs.extend((0..m).map(|i| (i, (i + m / 2) % m)));
        }
        pairs.sort_unstable();
        let mut start = vec![0usize; m_ + 1];
        let mut cols = Vec::new();
        let mut i = 0;
        while i < pairs.len() {
            let mut j = i;
            while j < pairs.len() && pairs[j] == pairs[i] {
                j += 1;
            }
            if (j - i) % 2 == 1 {
                cols.push(pairs[i].1);
                start[pairs[i].0 as usize + 1] += 1;
            }
            i = j;
        }
        for j in 0..m_ {
            start[j + 1] += start[j];
        }
        NormalBasis {
            m,
            t,
            words: m_.div_ceil(64),
            start,
            cols,
            minimal: OnceLock::new(),
        }
    }

    /// Whether `GF(2ᵐ)` has a Gaussian normal basis of type `t`: `p = tm + 1` is a prime and
    /// `gcd(tm/k, m) = 1` for the order `k` of 2 modulo `p`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::NormalBasis;
    ///
    /// assert!(NormalBasis::exists(233, 2)); // 467 is prime
    /// assert!(!NormalBasis::exists(233, 1)); // 234 is not
    /// assert!(!NormalBasis::exists(8, 1) && !NormalBasis::exists(0, 1));
    /// ```
    #[must_use]
    pub fn exists(m: u32, t: u32) -> bool {
        parameters(m, t).is_some()
    }

    /// The degree `m`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::NormalBasis;
    ///
    /// assert_eq!(NormalBasis::new(233).degree(), 233);
    /// ```
    #[must_use]
    pub fn degree(&self) -> u32 {
        self.m
    }

    /// The type `T`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::NormalBasis;
    ///
    /// assert_eq!(NormalBasis::new(283).gaussian_type(), 6);
    /// ```
    #[must_use]
    pub fn gaussian_type(&self) -> u32 {
        self.t
    }

    /// The number of terms of the bilinear form of a coordinate of a product, the complexity of
    /// the basis: `2m − 1` for the optimal normal bases (types 1 and 2), at most `Tm − 1` for an
    /// even type.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::NormalBasis;
    ///
    /// assert_eq!(NormalBasis::new(233).complexity(), 2 * 233 - 1);
    /// assert_eq!(NormalBasis::new(7).complexity(), 21); // FIPS 186-4, D.3
    /// ```
    #[must_use]
    pub fn complexity(&self) -> usize {
        self.cols.len()
    }

    /// The matrix `λ` of the bilinear form of the first coordinate of a product,
    /// `c₀ = Σ λⱼₗ·aⱼ·bₗ`; coordinate `i` is the same form of the coordinates rotated by `i`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::NormalBasis;
    ///
    /// // FIPS 186-4, D.3: F(u, v) = u₀v₁ + u₁(v₀ + v₂ + v₅ + v₆) + …
    /// let l = NormalBasis::new(7).lambda_matrix();
    /// assert!(l.get(0, 1) && !l.get(0, 0));
    /// assert!(l.get(1, 0) && l.get(1, 2) && l.get(1, 5) && l.get(1, 6) && !l.get(1, 1));
    /// assert_eq!(l, l.transpose());
    /// ```
    #[must_use]
    pub fn lambda_matrix(&self) -> BitMatrix {
        let m = self.m as usize;
        let mut l = BitMatrix::zeros(m, m);
        for j in 0..m {
            for &c in &self.cols[self.start[j]..self.start[j + 1]] {
                l.set(j, c as usize, true);
            }
        }
        l
    }

    /// The element with the coordinates `words`: coordinate `aᵢ`, the coefficient of `θ^(2ⁱ)`,
    /// is bit `i mod 64` of word `i / 64`. (FIPS 186-4 writes the string `a₀ a₁ … a_{m−1}` from the
    /// left, so `a₀` is the leading bit of its hexadecimal numbers.)
    ///
    /// # Panics
    ///
    /// Panics if a bit from `m` on is set.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::NormalBasis;
    ///
    /// let nb = NormalBasis::new(7);
    /// assert_eq!(nb.element(&[0x7f]), nb.one());
    /// assert_eq!(nb.element(&[]), nb.zero());
    /// ```
    #[track_caller]
    #[must_use]
    pub fn element(&self, words: &[u64]) -> NormalElement<'_> {
        let m = self.m as usize;
        let mut w = words.to_vec();
        assert!(
            w.iter().enumerate().all(|(i, &x)| {
                let from = m.saturating_sub(64 * i);
                from >= 64 || x >> from == 0
            }),
            "a coordinate beyond the degree {m} is set"
        );
        w.resize(self.words, 0);
        NormalElement { b: self, w }
    }

    /// 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::NormalBasis;
    ///
    /// assert!(NormalBasis::new(7).zero().is_zero());
    /// ```
    #[must_use]
    pub fn zero(&self) -> NormalElement<'_> {
        NormalElement {
            b: self,
            w: vec![0; self.words],
        }
    }

    /// 1, the sum of the basis: all coordinates 1.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::NormalBasis;
    ///
    /// let nb = NormalBasis::new(10);
    /// assert_eq!(nb.one().words(), &[0x3ff]);
    /// assert!(nb.one().is_one());
    /// ```
    #[must_use]
    pub fn one(&self) -> NormalElement<'_> {
        NormalElement {
            b: self,
            w: ones(self.m as usize),
        }
    }

    /// The exponents of the terms of the minimal polynomial of `θ` over `GF(2)`, in decreasing
    /// order: the modulus of the polynomial basis in which `x` is `θ`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::{BinaryField, NormalBasis};
    ///
    /// let nb = NormalBasis::new(4); // type 1: θ is a primitive 5th root of unity
    /// assert_eq!(nb.minimal_polynomial(), vec![4, 3, 2, 1, 0]);
    /// let g = BinaryField::new(&NormalBasis::new(163).minimal_polynomial());
    /// assert_eq!(g.degree(), 163);
    /// ```
    #[must_use]
    pub fn minimal_polynomial(&self) -> Vec<u32> {
        let g = &self.minimal().poly;
        (0..=self.m)
            .rev()
            .filter(|&i| g[i as usize / 64] >> (i % 64) & 1 == 1)
            .collect()
    }

    /// The conversions between this basis and the polynomial basis of `field`, which maps `θ`
    /// to the least (as a number) of the roots in `field` of its minimal polynomial. It is
    /// computed once, by Berlekamp's trace algorithm and the inversion of an `m × m` matrix over
    /// `GF(2)`: `O(m²)` operations of the field and `O(m³/64)` of words.
    ///
    /// # Panics
    ///
    /// Panics if `field` is not of degree `m`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::{BinaryField, NormalBasis};
    ///
    /// let f = BinaryField::new(&[163, 7, 6, 3, 0]);
    /// let nb = NormalBasis::new(163);
    /// let c = nb.conversion(&f);
    /// let (a, b) = (f.element(&[0x1234_5678, 0x9abc]), f.element(&[0xfeed, 0, 0x7]));
    /// assert_eq!(c.to_normal(&(&a * &b)), &c.to_normal(&a) * &c.to_normal(&b));
    /// assert_eq!(c.to_polynomial(&c.to_normal(&a)), a);
    /// assert_eq!(c.to_normal(&f.one()), nb.one());
    /// ```
    #[track_caller]
    #[must_use]
    pub fn conversion<'n, 'g>(&'n self, field: &'g BinaryField) -> BasisConversion<'n, 'g> {
        self.check_degree(field);
        let root = self.find_root(field);
        // The least of the conjugates of the root, comparing the words from the top.
        let mut best = root.clone();
        let mut c = root;
        for _ in 1..self.m {
            c = c.square();
            if c.words().iter().rev().lt(best.words().iter().rev()) {
                best = c.clone();
            }
        }
        self.conversion_from(&best)
    }

    /// The conversions between this basis and the polynomial basis of `theta`'s field which map
    /// `θ` to `theta`, or `None` if `theta` is not a root of `θ`'s minimal polynomial (then no
    /// conversion maps `θ` to it). FIPS 186-4, §D.10, gives such a `theta` for each of its fields,
    /// the top row of its matrix.
    ///
    /// # Panics
    ///
    /// Panics if `theta`'s field is not of degree `m`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::{BinaryField, NormalBasis};
    ///
    /// let f = BinaryField::new(&[163, 7, 6, 3, 0]);
    /// let nb = NormalBasis::new(163);
    /// let theta = nb.conversion(&f).normal_element();
    /// // Its square gives the conversion rotated by one place; x itself is no root.
    /// let c2 = nb.conversion_with(&theta.square()).unwrap();
    /// assert_eq!(c2.to_normal(&theta.square()), nb.element(&[1]));
    /// assert!(nb.conversion_with(&f.element(&[2])).is_none());
    /// ```
    #[track_caller]
    #[must_use]
    pub fn conversion_with<'n, 'g>(&'n self, theta: &F2m<'g>) -> Option<BasisConversion<'n, 'g>> {
        self.check_degree(theta.field());
        let g = &self.minimal().poly;
        // g(θ') by Horner's rule.
        let field = theta.field();
        let mut v = field.zero();
        for i in (0..=self.m as usize).rev() {
            v = &v * theta;
            if g[i / 64] >> (i % 64) & 1 == 1 {
                v = &v + &field.one();
            }
        }
        v.is_zero().then(|| self.conversion_from(theta))
    }

    /// The conversion that maps `θ` to the root `theta` of its minimal polynomial.
    fn conversion_from<'n, 'g>(&'n self, theta: &F2m<'g>) -> BasisConversion<'n, 'g> {
        let m = self.m as usize;
        let n = self.words;
        let mut to_polynomial = Vec::with_capacity(m * n);
        let mut c = theta.clone();
        for _ in 0..m {
            to_polynomial.extend_from_slice(c.words());
            c = c.square();
        }
        let inverse = BitMatrix::from_words(m, m, to_polynomial.clone())
            .inverse()
            .expect("the conjugates of a root of an irreducible polynomial are a basis");
        BasisConversion {
            basis: self,
            field: theta.field(),
            to_polynomial,
            to_normal: inverse.as_words().to_vec(),
        }
    }

    /// Panics unless `field` is of degree `m`.
    #[track_caller]
    fn check_degree(&self, field: &BinaryField) {
        assert!(
            field.degree() == self.m,
            "a basis of GF(2^{}) and a field of degree {}",
            self.m,
            field.degree()
        );
    }

    /// The minimal polynomial of `θ` and the powers `y^(2ⁱ) mod g`: from the powers
    /// `θ⁰ … θᵐ`, by the inverse of the matrix of the first `m`.
    fn minimal(&self) -> &Minimal {
        self.minimal.get_or_init(|| {
            let m = self.m as usize;
            let n = self.words;
            let theta = self.element(&[1]);
            let mut p = Vec::with_capacity(m * n);
            let mut v = self.one();
            for _ in 0..m {
                p.extend_from_slice(&v.w);
                v = &v * &theta;
            }
            let inverse = BitMatrix::from_words(m, m, p)
                .inverse()
                .expect("a normal element generates the field");
            // θᵐ = Σ gᵢ θⁱ: g = (θᵐ's coordinates)·P⁻¹.
            let mut poly = vec![0u64; (m + 1).div_ceil(64)];
            for i in 0..m {
                if v.w[i / 64] >> (i % 64) & 1 == 1 {
                    for (g, r) in poly.iter_mut().zip(inverse.row_words(i)) {
                        *g ^= r;
                    }
                }
            }
            poly[m / 64] ^= 1 << (m % 64);
            Minimal {
                poly,
                frobenius: inverse.as_words().to_vec(),
            }
        })
    }

    /// A root in `field` of the minimal polynomial `g` of `θ`, by Berlekamp's trace algorithm:
    /// for `u` in the field, `T(y) = Σᵢ (u·y)^(2ⁱ) mod h(y)` takes the value `Tr(u·r)` at each
    /// root `r` of `h`, so `gcd(h, T)` keeps the roots of trace 0; a part of at most half the
    /// degree is kept, until one root is left. `T` mod `g` needs no products: its coefficients
    /// are sums of the conjugates `u^(2ⁱ)` by the bits of `y^(2ⁱ) mod g`.
    fn find_root<'g>(&self, field: &'g BinaryField) -> F2m<'g> {
        let m = self.m as usize;
        let n = self.words;
        let minimal = self.minimal();
        let mut h: Vec<F2m<'g>> = (0..=m)
            .map(|i| {
                if minimal.poly[i / 64] >> (i % 64) & 1 == 1 {
                    field.one()
                } else {
                    field.zero()
                }
            })
            .collect();
        // A fixed seed, so that the conversion is reproducible (it does not depend on the root
        // found, which is replaced by the least of its conjugates).
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            seed = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = seed;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        };
        while h.len() > 2 {
            let words: Vec<u64> = (0..n).map(|_| next()).collect();
            let mut u = field.element(&words);
            let mut t = vec![0u64; m * n];
            for i in 0..m {
                let row = &minimal.frobenius[i * n..(i + 1) * n];
                for j in 0..m {
                    if row[j / 64] >> (j % 64) & 1 == 1 {
                        for (t, x) in t[j * n..(j + 1) * n].iter_mut().zip(u.words()) {
                            *t ^= x;
                        }
                    }
                }
                u = u.square();
            }
            let mut t: Vec<F2m<'g>> = t.chunks(n).map(|c| field.element(c)).collect();
            poly_rem(&mut t, &h);
            let w = poly_gcd(h.clone(), t);
            let (dw, dh) = (w.len() - 1, h.len() - 1);
            if dw > 0 && dw < dh {
                h = if 2 * dw <= dh { w } else { poly_div(&h, &w) };
            }
        }
        // h = y + h₀: the root is h₀.
        h.swap_remove(0)
    }
}

impl PartialEq for NormalBasis {
    fn eq(&self, other: &NormalBasis) -> bool {
        self.m == other.m && self.t == other.t
    }
}

impl Eq for NormalBasis {}

impl Hash for NormalBasis {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.m.hash(state);
        self.t.hash(state);
    }
}

impl fmt::Debug for NormalBasis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NormalBasis")
            .field("degree", &self.m)
            .field("type", &self.t)
            .finish()
    }
}

/// `m` ones.
fn ones(m: usize) -> Vec<u64> {
    let mut w = vec![u64::MAX; m.div_ceil(64)];
    if !m.is_multiple_of(64) {
        w[m / 64] = (1 << (m % 64)) - 1;
    }
    w
}

/// `p = tm + 1` and a `u` of order `t` modulo `p`, if a Gaussian normal basis of type `t` exists.
fn parameters(m: u32, t: u32) -> Option<(u64, u64)> {
    if m < 2 || t == 0 {
        return None;
    }
    let tm = u64::from(t) * u64::from(m);
    let p = tm + 1;
    if !is_prime(p) {
        return None;
    }
    let factors = prime_factors(tm);
    let k = order(2, p, tm, &factors);
    if gcd(tm / k, u64::from(m)) != 1 {
        return None;
    }
    // u = c^((p−1)/t) has order dividing t; the first of order exactly t.
    let tf = prime_factors(u64::from(t));
    let u = (2..p)
        .map(|c| pow_mod(c, tm / u64::from(t), p))
        .find(|&u| tf.iter().all(|&q| pow_mod(u, u64::from(t) / q, p) != 1))
        .unwrap_or(1);
    Some((p, u))
}

/// The order of `a` modulo the prime `p`, from the prime factors of `p − 1 = n`.
fn order(a: u64, p: u64, n: u64, factors: &[u64]) -> u64 {
    let mut k = n;
    for &q in factors {
        while k.is_multiple_of(q) && pow_mod(a, k / q, p) == 1 {
            k /= q;
        }
    }
    k
}

/// The distinct prime factors of `n`.
fn prime_factors(mut n: u64) -> Vec<u64> {
    let mut f = Vec::new();
    let mut q = 2u64;
    while q * q <= n {
        if n.is_multiple_of(q) {
            f.push(q);
            while n.is_multiple_of(q) {
                n /= q;
            }
        }
        q += 1;
    }
    if n > 1 {
        f.push(n);
    }
    f
}

/// The greatest common divisor.
fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// `a^e mod p`.
fn pow_mod(a: u64, mut e: u64, p: u64) -> u64 {
    let mul = |x: u64, y: u64| (u128::from(x) * u128::from(y) % u128::from(p)) as u64;
    let mut r = 1 % p;
    let mut b = a % p;
    while e > 0 {
        if e & 1 == 1 {
            r = mul(r, b);
        }
        b = mul(b, b);
        e >>= 1;
    }
    r
}

/// Whether `n` is a prime: Miller and Rabin's test to the first twelve primes, which is exact for
/// every `n < 3.3·10²⁴` (Sorenson and Webster, 2015), so for every `u64`.
fn is_prime(n: u64) -> bool {
    const BASES: [u64; 12] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];
    if n < 2 {
        return false;
    }
    for p in BASES {
        if n.is_multiple_of(p) {
            return n == p;
        }
    }
    let s = (n - 1).trailing_zeros();
    let d = (n - 1) >> s;
    BASES.iter().all(|&a| {
        let mut x = pow_mod(a, d, n);
        if x == 1 || x == n - 1 {
            return true;
        }
        for _ in 1..s {
            x = (u128::from(x) * u128::from(x) % u128::from(n)) as u64;
            if x == n - 1 {
                return true;
            }
        }
        false
    })
}

/// Drops the zero coefficients at the top of `a`, leaving at least one.
fn trim<'g>(a: &mut Vec<F2m<'g>>, field: &'g BinaryField) {
    while a.last().is_some_and(F2m::is_zero) {
        a.pop();
    }
    if a.is_empty() {
        a.push(field.zero());
    }
}

/// `a mod b` in place, for `b` nonzero.
fn poly_rem<'g>(a: &mut Vec<F2m<'g>>, b: &[F2m<'g>]) {
    let field = b[0].field();
    let db = b.len() - 1;
    let lead = b[db]
        .inverse()
        .expect("a polynomial with a nonzero leading coefficient");
    trim(a, field);
    while a.len() > db && !(a.len() == 1 && a[0].is_zero()) {
        let top = a.len() - 1;
        let c = &a[top] * &lead;
        if !c.is_zero() {
            for j in 0..db {
                let t = &c * &b[j];
                a[top - db + j] += t;
            }
        }
        a.pop();
    }
    trim(a, field);
}

/// The monic greatest common divisor of `a` and `b`, by Euclid's algorithm.
fn poly_gcd<'g>(mut a: Vec<F2m<'g>>, mut b: Vec<F2m<'g>>) -> Vec<F2m<'g>> {
    let field = a[0].field();
    trim(&mut a, field);
    trim(&mut b, field);
    while !(b.len() == 1 && b[0].is_zero()) {
        poly_rem(&mut a, &b);
        core::mem::swap(&mut a, &mut b);
    }
    let lead = a[a.len() - 1].inverse().expect("a nonzero gcd");
    a.iter().map(|c| c * &lead).collect()
}

/// `a / b` for a monic `b` that divides `a`.
fn poly_div<'g>(a: &[F2m<'g>], b: &[F2m<'g>]) -> Vec<F2m<'g>> {
    let db = b.len() - 1;
    let mut r = a.to_vec();
    let mut q = vec![a[0].field().zero(); a.len() - db];
    for top in (db..a.len()).rev() {
        let c = r[top].clone();
        if !c.is_zero() {
            for j in 0..=db {
                let t = &c * &b[j];
                r[top - db + j] += t;
            }
        }
        q[top - db] = c;
    }
    q
}

/// The conversions between a [`NormalBasis`] and the polynomial basis of a [`BinaryField`] of the
/// same degree, two matrices over `GF(2)`: each conversion is `O(m²/64)` word operations.
///
/// # Examples
///
/// ```
/// use a::math::field::{BinaryField, NormalBasis};
///
/// let f = BinaryField::new(&[233, 74, 0]);
/// let nb = NormalBasis::new(233);
/// let c = nb.conversion(&f);
/// let a = f.element(&[0xdead_beef, 0, 1 << 33, 0x99]);
/// let na = c.to_normal(&a);
/// assert_eq!(na.inverse().unwrap(), c.to_normal(&a.inverse().unwrap()));
/// assert_eq!(na.trace(), a.trace());
/// assert_eq!(c.to_polynomial(&nb.element(&[1])), c.normal_element());
/// ```
#[derive(Clone)]
pub struct BasisConversion<'n, 'g> {
    /// The normal basis.
    basis: &'n NormalBasis,
    /// The polynomial basis.
    field: &'g BinaryField,
    /// Row `i`: `θ^(2ⁱ)` in the polynomial basis.
    to_polynomial: Vec<u64>,
    /// Row `i`: `xⁱ` in the normal basis.
    to_normal: Vec<u64>,
}

impl<'n, 'g> BasisConversion<'n, 'g> {
    /// The normal basis.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::{BinaryField, NormalBasis};
    ///
    /// let (f, nb) = (BinaryField::new(&[7, 1, 0]), NormalBasis::new(7));
    /// assert_eq!(nb.conversion(&f).basis(), &nb);
    /// ```
    #[must_use]
    pub fn basis(&self) -> &'n NormalBasis {
        self.basis
    }

    /// The polynomial basis.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::{BinaryField, NormalBasis};
    ///
    /// let (f, nb) = (BinaryField::new(&[7, 1, 0]), NormalBasis::new(7));
    /// assert_eq!(nb.conversion(&f).field(), &f);
    /// ```
    #[must_use]
    pub fn field(&self) -> &'g BinaryField {
        self.field
    }

    /// `θ` in the polynomial basis (FIPS 186-4, §D.10, the top row of its matrix).
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::{BinaryField, NormalBasis};
    ///
    /// let (f, nb) = (BinaryField::new(&[7, 1, 0]), NormalBasis::new(7));
    /// let c = nb.conversion(&f);
    /// assert_eq!(c.to_normal(&c.normal_element()), nb.element(&[1]));
    /// ```
    #[must_use]
    pub fn normal_element(&self) -> F2m<'g> {
        self.field.element(&self.to_polynomial[..self.basis.words])
    }

    /// `x` in the normal basis (FIPS 186-4, §D.9, the second row from the bottom of its matrix).
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::{BinaryField, NormalBasis};
    ///
    /// let (f, nb) = (BinaryField::new(&[7, 1, 0]), NormalBasis::new(7));
    /// let c = nb.conversion(&f);
    /// assert_eq!(c.to_polynomial(&c.polynomial_generator()), f.element(&[2]));
    /// ```
    #[must_use]
    pub fn polynomial_generator(&self) -> NormalElement<'n> {
        self.to_normal(&self.field.element(&[2]))
    }

    /// `a` in the normal basis.
    ///
    /// # Panics
    ///
    /// Panics if `a` is not of the conversion's field.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::{BinaryField, NormalBasis};
    ///
    /// let (f, nb) = (BinaryField::new(&[7, 1, 0]), NormalBasis::new(7));
    /// assert_eq!(nb.conversion(&f).to_normal(&f.zero()), nb.zero());
    /// ```
    #[track_caller]
    #[must_use]
    pub fn to_normal(&self, a: &F2m<'g>) -> NormalElement<'n> {
        assert!(a.field() == self.field, "an element of another field");
        NormalElement {
            b: self.basis,
            w: row_times(a.words(), &self.to_normal, self.basis.words),
        }
    }

    /// `a` in the polynomial basis.
    ///
    /// # Panics
    ///
    /// Panics if `a` is not of the conversion's basis.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::{BinaryField, NormalBasis};
    ///
    /// let (f, nb) = (BinaryField::new(&[7, 1, 0]), NormalBasis::new(7));
    /// assert_eq!(nb.conversion(&f).to_polynomial(&nb.one()), f.one());
    /// ```
    #[track_caller]
    #[must_use]
    pub fn to_polynomial(&self, a: &NormalElement<'n>) -> F2m<'g> {
        assert!(a.b == self.basis, "an element of another basis");
        self.field
            .element(&row_times(&a.w, &self.to_polynomial, self.basis.words))
    }
}

impl fmt::Debug for BasisConversion<'_, '_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BasisConversion")
            .field("basis", self.basis)
            .field("field", self.field)
            .field("normal_element", &self.normal_element())
            .finish()
    }
}

/// The row vector `v` times the matrix whose rows of `n` words are `rows`.
fn row_times(v: &[u64], rows: &[u64], n: usize) -> Vec<u64> {
    let mut r = vec![0u64; n];
    for (i, row) in rows.chunks(n).enumerate() {
        if v[i / 64] >> (i % 64) & 1 == 1 {
            for (r, x) in r.iter_mut().zip(row) {
                *r ^= x;
            }
        }
    }
    r
}

/// An element of `GF(2ᵐ)` in a [`NormalBasis`], which it borrows.
///
/// # Panics
///
/// The operators `+ - * /` and their assignments panic on elements of two different bases (of
/// another degree or type); `/` and `/=` panic on division by zero.
///
/// # Examples
///
/// ```
/// use a::math::field::NormalBasis;
///
/// let nb = NormalBasis::new(233);
/// let a = nb.element(&[0x0123_4567_89ab_cdef, 0x1111, 0, 1 << 40]);
/// // Squaring rotates the coordinates by one place.
/// assert_eq!(a.square().words()[0] & 1, a.words()[3] >> 40 & 1);
/// assert_eq!(&a.square() / &a, a);
/// assert_eq!(a.sqrt().square(), a);
/// ```
#[derive(Clone)]
pub struct NormalElement<'n> {
    b: &'n NormalBasis,
    /// The coordinates, `⌈m / 64⌉` words, the bits from `m` on zero.
    w: Vec<u64>,
}

impl<'n> NormalElement<'n> {
    /// The basis.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::NormalBasis;
    ///
    /// let nb = NormalBasis::new(7);
    /// assert_eq!(nb.one().basis().degree(), 7);
    /// ```
    #[must_use]
    pub fn basis(&self) -> &'n NormalBasis {
        self.b
    }

    /// The coordinates: `aᵢ`, the coefficient of `θ^(2ⁱ)`, is bit `i mod 64` of word `i / 64`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::NormalBasis;
    ///
    /// let nb = NormalBasis::new(7);
    /// assert_eq!(nb.element(&[0b11]).words(), &[0b11]);
    /// ```
    #[must_use]
    pub fn words(&self) -> &[u64] {
        &self.w
    }

    /// Whether it is 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::NormalBasis;
    ///
    /// let nb = NormalBasis::new(7);
    /// assert!((nb.one() + nb.one()).is_zero());
    /// ```
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.w.iter().all(|&x| x == 0)
    }

    /// Whether it is 1, all coordinates 1.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::NormalBasis;
    ///
    /// let nb = NormalBasis::new(7);
    /// assert!(nb.one().is_one() && !nb.element(&[1]).is_one());
    /// ```
    #[must_use]
    pub fn is_one(&self) -> bool {
        self.w == ones(self.b.m as usize)
    }

    /// `a^(2ᵏ)` (for `k` of either sign: `k = −1` is the square root): the coordinates rotated by
    /// `k` places, `aᵢ` to `a_{i+k}`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::NormalBasis;
    ///
    /// let nb = NormalBasis::new(7);
    /// let a = nb.element(&[0b0000011]);
    /// assert_eq!(a.frobenius(2).words(), &[0b0001100]);
    /// assert_eq!(a.frobenius(7), a);
    /// assert_eq!(a.frobenius(-1), a.sqrt());
    /// ```
    #[must_use]
    pub fn frobenius(&self, k: i64) -> NormalElement<'n> {
        let m = self.b.m as usize;
        let k = k.rem_euclid(m as i64) as usize;
        NormalElement {
            b: self.b,
            w: rotate_up(&self.w, m, k),
        }
    }

    /// The square: the coordinates rotated by one place, `(a₀ … a_{m−1})² = (a_{m−1} a₀ … a_{m−2})`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::NormalBasis;
    ///
    /// let nb = NormalBasis::new(7);
    /// assert_eq!(nb.element(&[0b1000000]).square().words(), &[0b0000001]);
    /// ```
    #[must_use]
    pub fn square(&self) -> NormalElement<'n> {
        self.frobenius(1)
    }

    /// The square root: the coordinates rotated back by one place.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::NormalBasis;
    ///
    /// let nb = NormalBasis::new(7);
    /// assert_eq!(nb.element(&[1]).sqrt().words(), &[0b1000000]);
    /// ```
    #[must_use]
    pub fn sqrt(&self) -> NormalElement<'n> {
        self.frobenius(-1)
    }

    /// The trace, the parity of the coordinates (`Tr(θ^(2ⁱ)) = Tr(θ) = 1`).
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::NormalBasis;
    ///
    /// let nb = NormalBasis::new(7);
    /// assert_eq!(nb.element(&[0b1011]).trace(), 1);
    /// assert_eq!(nb.one().trace(), 1); // m = 7 is odd
    /// ```
    #[must_use]
    pub fn trace(&self) -> u8 {
        (self.w.iter().map(|w| w.count_ones()).sum::<u32>() % 2) as u8
    }

    /// The `e`-th power, by squarings (rotations) and products; `0⁰ = 1`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::NormalBasis;
    ///
    /// let nb = NormalBasis::new(7);
    /// let a = nb.element(&[0b0110101]);
    /// assert!(a.pow_u64(127).is_one());
    /// assert_eq!(a.pow_u64(3), &(&a * &a) * &a);
    /// ```
    #[must_use]
    pub fn pow_u64(&self, e: u64) -> NormalElement<'n> {
        let mut r = self.b.one();
        for i in (0..u64::BITS - e.leading_zeros()).rev() {
            r = r.square();
            if e >> i & 1 == 1 {
                r = &r * self;
            }
        }
        r
    }

    /// The inverse by Itoh and Tsujii's method, `(a^(2^(m−1) − 1))²`, or `None` for 0: with
    /// `bₖ = a^(2ᵏ − 1)`, `b_{2k} = bₖ^(2ᵏ)·bₖ` and `b_{k+1} = bₖ²·a` along the bits of `m − 1`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::NormalBasis;
    ///
    /// let nb = NormalBasis::new(10); // type 1
    /// let a = nb.element(&[0b10_0110_1011]);
    /// assert_eq!(&a * &a.inverse().unwrap(), nb.one());
    /// assert!(nb.zero().inverse().is_none());
    /// ```
    #[must_use]
    pub fn inverse(&self) -> Option<NormalElement<'n>> {
        if self.is_zero() {
            return None;
        }
        let e = u64::from(self.b.m - 1);
        let mut b = self.clone();
        let mut k = 1u64;
        for i in (0..63 - e.leading_zeros()).rev() {
            b = &b.frobenius(k as i64) * &b;
            k *= 2;
            if e >> i & 1 == 1 {
                b = &b.square() * self;
                k += 1;
            }
        }
        debug_assert_eq!(k, e);
        Some(b.square())
    }
}

/// The `m` coordinates `w` rotated up by `k < m` places: bit `i` to bit `(i + k) mod m`.
fn rotate_up(w: &[u64], m: usize, k: usize) -> Vec<u64> {
    if k == 0 {
        return w.to_vec();
    }
    let n = w.len();
    let (q, r) = (k / 64, k % 64);
    let word = |i: usize, back: usize| i.checked_sub(back).map_or(0, |j| w[j]);
    // The bits below m − k move up by k (those pushed past m are masked off); those from m − k on
    // wrap around to the bottom.
    let mut out: Vec<u64> = (0..n)
        .map(|i| {
            if r == 0 {
                word(i, q)
            } else {
                (word(i, q) << r) | (word(i, q + 1) >> (64 - r))
            }
        })
        .collect();
    if !m.is_multiple_of(64) {
        out[n - 1] &= (1 << (m % 64)) - 1;
    }
    let mut wrapped = vec![0u64; n];
    super::binary_poly::shift_right_into(&mut wrapped, w, m - k);
    for (o, x) in out.iter_mut().zip(&wrapped) {
        *o ^= x;
    }
    out
}

/// The product by Massey and Omura's method: `c = Σⱼ rot(a, j) ∧ Σ_{λⱼₗ = 1} rot(b, l)`, for
/// `rot(v, j)` the coordinates `vᵢ` moved to `i − j`, which is coordinate `i` of the form applied to
/// `a` and `b` rotated by `i`, for every `i` at once.
fn product<'n>(a: &NormalElement<'n>, b: &NormalElement<'n>) -> NormalElement<'n> {
    let basis = a.b;
    let m = basis.m as usize;
    let n = basis.words;
    // rot(b, l) for every l: rot(v, j) is v rotated up by m − j.
    let mut rb = Vec::with_capacity(m * n);
    for l in 0..m {
        rb.extend(rotate_up(&b.w, m, (m - l) % m));
    }
    let mut c = vec![0u64; n];
    let mut v = vec![0u64; n];
    for j in 0..m {
        let row = &basis.cols[basis.start[j]..basis.start[j + 1]];
        if row.is_empty() {
            continue;
        }
        v.fill(0);
        for &l in row {
            for (v, x) in v.iter_mut().zip(&rb[l as usize * n..(l as usize + 1) * n]) {
                *v ^= x;
            }
        }
        let ra = rotate_up(&a.w, m, (m - j) % m);
        for ((c, x), y) in c.iter_mut().zip(&ra).zip(&v) {
            *c ^= x & y;
        }
    }
    NormalElement { b: basis, w: c }
}

impl PartialEq for NormalElement<'_> {
    fn eq(&self, other: &NormalElement<'_>) -> bool {
        (core::ptr::eq(self.b, other.b) || self.b == other.b) && self.w == other.w
    }
}

impl Eq for NormalElement<'_> {}

impl Hash for NormalElement<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.b.hash(state);
        self.w.hash(state);
    }
}

impl fmt::Debug for NormalElement<'_> {
    /// The coordinates as a number, `a₀` its least significant bit.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "0x")?;
        for w in self.w.iter().rev() {
            write!(f, "{w:016x}")?;
        }
        Ok(())
    }
}

impl<'n> Neg for &NormalElement<'n> {
    type Output = NormalElement<'n>;
    fn neg(self) -> NormalElement<'n> {
        self.clone()
    }
}

impl<'n> Neg for NormalElement<'n> {
    type Output = NormalElement<'n>;
    fn neg(self) -> NormalElement<'n> {
        self
    }
}

/// `a + b`, which is `a − b`.
fn sum<'n>(a: &NormalElement<'n>, b: &NormalElement<'n>) -> NormalElement<'n> {
    NormalElement {
        b: a.b,
        w: a.w.iter().zip(&b.w).map(|(x, y)| x ^ y).collect(),
    }
}

/// Panics unless `a` and `b` are of the same basis.
#[track_caller]
fn check(a: &NormalElement<'_>, b: &NormalElement<'_>) {
    assert!(
        core::ptr::eq(a.b, b.b) || a.b == b.b,
        "elements of different bases"
    );
}

/// The binary operators for the four combinations of owned and borrowed operands, and their
/// assignments, from the one on two borrows.
macro_rules! binary {
    ($tr:ident, $m:ident, $atr:ident, $am:ident, $f:expr) => {
        impl<'n> $tr<&NormalElement<'n>> for &NormalElement<'n> {
            type Output = NormalElement<'n>;
            #[track_caller]
            fn $m(self, rhs: &NormalElement<'n>) -> NormalElement<'n> {
                check(self, rhs);
                $f(self, rhs)
            }
        }
        impl<'n> $tr<NormalElement<'n>> for &NormalElement<'n> {
            type Output = NormalElement<'n>;
            #[track_caller]
            fn $m(self, rhs: NormalElement<'n>) -> NormalElement<'n> {
                self.$m(&rhs)
            }
        }
        impl<'n> $tr<&NormalElement<'n>> for NormalElement<'n> {
            type Output = NormalElement<'n>;
            #[track_caller]
            fn $m(self, rhs: &NormalElement<'n>) -> NormalElement<'n> {
                (&self).$m(rhs)
            }
        }
        impl<'n> $tr<NormalElement<'n>> for NormalElement<'n> {
            type Output = NormalElement<'n>;
            #[track_caller]
            fn $m(self, rhs: NormalElement<'n>) -> NormalElement<'n> {
                (&self).$m(&rhs)
            }
        }
        impl<'n> $atr<&NormalElement<'n>> for NormalElement<'n> {
            #[track_caller]
            fn $am(&mut self, rhs: &NormalElement<'n>) {
                *self = (&*self).$m(rhs);
            }
        }
        impl<'n> $atr<NormalElement<'n>> for NormalElement<'n> {
            #[track_caller]
            fn $am(&mut self, rhs: NormalElement<'n>) {
                *self = (&*self).$m(&rhs);
            }
        }
    };
}

binary!(Add, add, AddAssign, add_assign, sum);
binary!(Sub, sub, SubAssign, sub_assign, sum);
binary!(Mul, mul, MulAssign, mul_assign, product);
binary!(
    Div,
    div,
    DivAssign,
    div_assign,
    |a: &NormalElement<'n>, b: &NormalElement<'n>| {
        product(a, &b.inverse().expect("division by zero"))
    }
);
