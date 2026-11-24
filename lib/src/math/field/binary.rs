//! Binary fields `GF(2ᵐ) = GF(2)[x]/(f)` in a polynomial basis, and their elements.
//!
//! An element is a polynomial of degree below `m`, its coefficients the bits of `⌈m / 64⌉` words.
//! The field is built once from the terms of `f` and chooses there how it multiplies and reduces:
//!
//! - Products: carry-less, a pair of words by `PMULL` (aarch64) or `PCLMULQDQ` (x86_64), found at
//!   compile time where the target enables the instruction and at run time otherwise, or by a
//!   portable window method, and polynomials of several words by Karatsuba's method over those
//!   ([`Clmul`]); squares by spreading the bits, with no products.
//! - Reduction modulo `f`, one of two, whichever a rough count of word operations finds cheaper
//!   for `f` and the kernel: the sparse reduction, a chunk of at most `min(64, m − e₁)` bits at a
//!   time from the top (`e₁` the second exponent of `f`), each folded down by one shifted sum per
//!   term (Hankerson, Menezes and Vanstone, "Guide to Elliptic Curve Cryptography", 2004, §2.3.5),
//!   which the trinomials and pentanomials of the standards take; and Barrett's reduction, two
//!   products by the precomputed `⌊x²ᵐ / f⌋` (Barrett, CRYPTO 1986, over `GF(2)[x]`, where it is
//!   exact), which moduli of many terms or with terms close to `xᵐ` take, of one word or more.
//! - Inverses: the extended Euclidean algorithm over `GF(2)[x]` (Hankerson, Menezes and Vanstone,
//!   Algorithm 2.48), `O(m²/64)` word operations.
//! - Square roots: `√a = a_even + √x·a_odd`, the even and odd coefficients of `a` gathered and
//!   `√x = x^(2^(m−1))` computed once (Fong, Hankerson, López and Menezes, "Field inversion and
//!   point halving revisited", IEEE Trans. Computers 53, 2004): one product.
//! - Traces: linear, `Tr(a) = Σ aᵢ·Tr(xⁱ)`, the `Tr(xⁱ)` (power sums of the roots of `f`) found
//!   once by Newton's identities from the coefficients of `f` (Lidl and Niederreiter, "Finite
//!   Fields", Theorem 1.75): a masked parity.
//! - Irreducibility of `f`, checked when the field is built: Rabin's test ("Probabilistic
//!   algorithms in finite fields", SIAM J. Comput. 9, 1980), `f | x^(2^m) − x` and
//!   `gcd(x^(2^(m/q)) − x, f) = 1` for each prime `q | m`: `m` squarings and a gcd per prime.
//!
//! Like the rest of the library this is not constant-time.

use super::binary_clmul as clmul;
use super::binary_poly as poly;
use core::fmt;
use core::hash::{Hash, Hasher};
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// The carry-less multiplier of 64-bit words that a [`BinaryField`] (and [`Gf2_128`]) uses.
///
/// [`Clmul::detect`] picks the instruction of the CPU; the portable kernel can be chosen
/// explicitly with [`BinaryField::with_clmul`], to compare the two or to reproduce a result
/// without the instruction. Every kernel gives the same products.
///
/// [`Gf2_128`]: super::Gf2_128
///
/// # Examples
///
/// ```
/// use a::math::field::Clmul;
///
/// // (x + 1)·(x + 1) = x² + 1 over GF(2).
/// assert_eq!(Clmul::Portable.mul(0b11, 0b11), 0b101);
/// let k = Clmul::detect();
/// assert!(k.is_available());
/// assert_eq!(k.mul(u64::MAX, u64::MAX), Clmul::Portable.mul(u64::MAX, u64::MAX));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Clmul {
    /// Portable code: the 16 products of one word by the polynomials of degree below 4 in a table,
    /// the other word taken 4 bits at a time (a window method).
    Portable,
    /// `PMULL` (`vmull_p64`) of the aarch64 cryptographic extension, which the standard library
    /// reports as the feature `aes`.
    Pmull,
    /// `PCLMULQDQ` (`_mm_clmulepi64_si128`) of x86_64.
    Pclmulqdq,
}

impl Clmul {
    /// The fastest kernel this CPU has: [`Pmull`](Clmul::Pmull) or
    /// [`Pclmulqdq`](Clmul::Pclmulqdq) when the instruction is found (at compile time if the target
    /// enables it, otherwise at run time), [`Portable`](Clmul::Portable) otherwise and under Miri.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::Clmul;
    ///
    /// assert!(Clmul::detect().is_available());
    /// ```
    #[must_use]
    pub fn detect() -> Clmul {
        if Clmul::Pmull.is_available() {
            Clmul::Pmull
        } else if Clmul::Pclmulqdq.is_available() {
            Clmul::Pclmulqdq
        } else {
            Clmul::Portable
        }
    }

    /// Whether this CPU can run the kernel. The portable one runs everywhere.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::Clmul;
    ///
    /// assert!(Clmul::Portable.is_available());
    /// // At most one of the instructions exists on a given CPU.
    /// assert!(!(Clmul::Pmull.is_available() && Clmul::Pclmulqdq.is_available()));
    /// ```
    #[must_use]
    pub fn is_available(self) -> bool {
        match self {
            Clmul::Portable => true,
            Clmul::Pmull => pmull_available(),
            Clmul::Pclmulqdq => pclmulqdq_available(),
        }
    }

    /// The carry-less product of `a` and `b`, the polynomials over `GF(2)` whose coefficients are
    /// their bits, by this kernel.
    ///
    /// # Panics
    ///
    /// Panics if the CPU cannot run the kernel.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::Clmul;
    ///
    /// // x⁶³ · x⁶³ = x¹²⁶.
    /// assert_eq!(Clmul::detect().mul(1 << 63, 1 << 63), 1 << 126);
    /// ```
    #[track_caller]
    #[must_use]
    pub fn mul(self, a: u64, b: u64) -> u128 {
        let mut r = [0u64; 2];
        clmul::mul(self, &[a], &[b], &mut r, &mut []);
        u128::from(r[0]) | (u128::from(r[1]) << 64)
    }
}

/// Whether the CPU has `PMULL`.
fn pmull_available() -> bool {
    #[cfg(all(target_arch = "aarch64", target_feature = "aes", not(miri)))]
    {
        true
    }
    #[cfg(all(target_arch = "aarch64", not(target_feature = "aes"), not(miri)))]
    {
        std::arch::is_aarch64_feature_detected!("aes")
    }
    #[cfg(not(all(target_arch = "aarch64", not(miri))))]
    {
        false
    }
}

/// Whether the CPU has `PCLMULQDQ`.
fn pclmulqdq_available() -> bool {
    #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq", not(miri)))]
    {
        true
    }
    #[cfg(all(target_arch = "x86_64", not(target_feature = "pclmulqdq"), not(miri)))]
    {
        std::arch::is_x86_feature_detected!("pclmulqdq")
    }
    #[cfg(not(all(target_arch = "x86_64", not(miri))))]
    {
        false
    }
}

/// Runs `f` on a scratch buffer of `len` words, on the stack when it is small.
fn with_scratch<R>(len: usize, f: impl FnOnce(&mut [u64]) -> R) -> R {
    const STACK: usize = 192;
    if len <= STACK {
        let mut buf = [0u64; STACK];
        f(&mut buf[..len])
    } else {
        f(&mut vec![0u64; len])
    }
}

/// How a field reduces modulo `f`.
#[derive(Clone)]
enum Reduction {
    /// Chunks of at most `chunk ≤ m − e₁` bits from the top (`e₁` the second exponent of `f`),
    /// each within a word, folded down by one shifted sum per term of `f` below `xᵐ`.
    Sparse {
        /// The most bits a chunk holds.
        chunk: usize,
    },
    /// Barrett's reduction: `q = ⌊⌊c / xᵐ⌋·μ / xᵐ⌋` and `c mod f = c + q·f`, exact for `c` of
    /// degree below `2m`.
    Barrett {
        /// `μ − xᵐ` for `μ = ⌊x²ᵐ / f⌋`, `n` words.
        mu: Vec<u64>,
        /// `f − xᵐ`, `n` words.
        g: Vec<u64>,
    },
}

/// The binary field `GF(2ᵐ) = GF(2)[x] / f(x)` for an irreducible `f` of degree `m`, its
/// elements polynomials of degree below `m` held as words, the coefficient of `xⁱ` bit `i mod 64`
/// of word `i / 64`.
///
/// Two fields are equal when their moduli are: the multiplier ([`Clmul`]) is a matter of speed,
/// and elements of two equal fields mix.
///
/// # Examples
///
/// ```
/// use a::math::field::BinaryField;
///
/// // GF(2⁸) by AES's x⁸ + x⁴ + x³ + x + 1.
/// let f = BinaryField::new(&[8, 4, 3, 1, 0]);
/// let a = f.element(&[0x53]);
/// let b = f.element(&[0xca]);
/// assert_eq!((&a * &b).words(), &[1]); // FIPS 197: {53}·{ca} = {01}
/// ```
#[derive(Clone)]
pub struct BinaryField {
    /// The degree `m`.
    m: u32,
    /// The words of an element, `⌈m / 64⌉`.
    words: usize,
    /// The exponents of the terms of `f` below `xᵐ`, in decreasing order.
    low: Vec<u32>,
    /// `f` as words, bit `m` included.
    f: Vec<u64>,
    /// The carry-less multiplier.
    clmul: Clmul,
    /// The reduction modulo `f`.
    reduction: Reduction,
    /// The words of scratch a product and its reduction need beyond the product's `2n`.
    scratch: usize,
    /// `√x = x^(2^(m−1))`.
    sqrt_x: Vec<u64>,
    /// Bit `i`: `Tr(xⁱ)`.
    trace: Vec<u64>,
}

impl BinaryField {
    /// The field `GF(2)[x] / f(x)` for `f` given by the exponents of its terms, as `[233, 74, 0]`
    /// for `x²³³ + x⁷⁴ + 1`, in any order; the largest is the degree `m`. Its products use the
    /// carry-less multiplier of the CPU, [`Clmul::detect`].
    ///
    /// # Panics
    ///
    /// Panics if there are no exponents, a repeated one, `m` is 0, or `f` is reducible: by Rabin's
    /// test, and at once if `f` has no constant term or an even number of terms (then `x` or
    /// `x + 1` divides it), for `m > 1`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::BinaryField;
    ///
    /// let f = BinaryField::new(&[0, 74, 233]); // sect233k1's field
    /// assert_eq!(f.degree(), 233);
    /// assert_eq!(f.modulus(), vec![233, 74, 0]);
    /// ```
    #[track_caller]
    #[must_use]
    pub fn new(exponents: &[u32]) -> BinaryField {
        BinaryField::with_clmul(exponents, Clmul::detect())
    }

    /// The field of [`new`](BinaryField::new), its products by the carry-less multiplier `clmul`:
    /// [`Clmul::Portable`] runs everywhere and gives the same results.
    ///
    /// # Panics
    ///
    /// Panics as [`new`](BinaryField::new) does, and if the CPU cannot run `clmul`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::{BinaryField, Clmul};
    ///
    /// let fast = BinaryField::new(&[163, 7, 6, 3, 0]);
    /// let slow = BinaryField::with_clmul(&[163, 7, 6, 3, 0], Clmul::Portable);
    /// assert_eq!(slow.clmul(), Clmul::Portable);
    /// let (a, b) = ([0x0123_4567_89ab_cdef, 0xfedc_ba98_7654_3210, 5], [7, 8, 1]);
    /// assert_eq!(
    ///     (&fast.element(&a) * &fast.element(&b)).words(),
    ///     (&slow.element(&a) * &slow.element(&b)).words()
    /// );
    /// ```
    #[track_caller]
    #[must_use]
    pub fn with_clmul(exponents: &[u32], clmul: Clmul) -> BinaryField {
        assert!(
            clmul.is_available(),
            "{clmul:?} is not available on this CPU"
        );
        let (m, low) = parse(exponents);
        if m > 1 {
            assert!(
                *low.last().unwrap_or(&1) == 0,
                "{exponents:?} has no constant term, so x divides it"
            );
            assert!(
                low.len().is_multiple_of(2),
                "{exponents:?} has an even number of terms, so x + 1 divides it"
            );
        }
        let mut field = BinaryField::unchecked(m, low, clmul);
        match field.rabin() {
            Some(sqrt_x) => field.sqrt_x = sqrt_x,
            None => panic!("{exponents:?} is reducible"),
        }
        field.trace = trace_mask(m, &field.low, field.words);
        field
    }

    /// Whether the polynomial with the terms `exponents` (in any order) is irreducible over
    /// `GF(2)`, by Rabin's test.
    ///
    /// # Panics
    ///
    /// Panics if there are no exponents or a repeated one.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::BinaryField;
    ///
    /// assert!(BinaryField::is_irreducible(&[2, 1, 0])); // x² + x + 1
    /// assert!(!BinaryField::is_irreducible(&[2, 0])); // x² + 1 = (x + 1)²
    /// assert!(BinaryField::is_irreducible(&[571, 10, 5, 2, 0]));
    /// assert!(!BinaryField::is_irreducible(&[0])); // 1 is a unit
    /// ```
    #[track_caller]
    #[must_use]
    pub fn is_irreducible(exponents: &[u32]) -> bool {
        if exponents == [0] {
            return false;
        }
        let (m, low) = parse(exponents);
        if m > 1 && (low.last() != Some(&0) || low.len() % 2 == 1) {
            return false;
        }
        BinaryField::unchecked(m, low, Clmul::detect())
            .rabin()
            .is_some()
    }

    /// The field without its checks and its tables of square roots and traces.
    fn unchecked(m: u32, low: Vec<u32>, clmul: Clmul) -> BinaryField {
        let n = (m as usize).div_ceil(64);
        let mut f = vec![0u64; (m as usize + 1).div_ceil(64)];
        f[m as usize / 64] |= 1 << (m % 64);
        for &e in &low {
            f[e as usize / 64] |= 1 << (e % 64);
        }
        let reduction = choose_reduction(m, n, &low, &f, clmul);
        let scratch = match reduction {
            Reduction::Sparse { .. } => clmul::scratch_len(n),
            Reduction::Barrett { .. } => 4 * n + clmul::scratch_len(n),
        };
        BinaryField {
            m,
            words: n,
            low,
            f,
            clmul,
            reduction,
            scratch,
            sqrt_x: Vec::new(),
            trace: Vec::new(),
        }
    }

    /// Rabin's test of `f`: `Some(√x)` if `f` is irreducible, `None` otherwise.
    fn rabin(&self) -> Option<Vec<u64>> {
        let m = self.m as usize;
        let primes = prime_factors(self.m);
        let x = self.reduce_any(vec![2]);
        let mut s = x.clone();
        let mut sqrt_x = x.clone();
        for i in 1..=m {
            if i == m {
                sqrt_x.clone_from(&s);
            }
            s = self.square_words(&s);
            if primes.iter().any(|&q| (m / q as usize) == i) {
                let mut d = s.clone();
                for (d, x) in d.iter_mut().zip(&x) {
                    *d ^= x;
                }
                if poly::degree(&poly::gcd(&d, &self.f)) != Some(0) {
                    return None;
                }
            }
        }
        (s == x).then_some(sqrt_x)
    }

    /// The degree `m`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::BinaryField;
    ///
    /// assert_eq!(BinaryField::new(&[113, 9, 0]).degree(), 113);
    /// ```
    #[must_use]
    pub fn degree(&self) -> u32 {
        self.m
    }

    /// The exponents of the terms of `f`, in decreasing order.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::BinaryField;
    ///
    /// assert_eq!(BinaryField::new(&[0, 1, 3, 4, 8]).modulus(), vec![8, 4, 3, 1, 0]);
    /// ```
    #[must_use]
    pub fn modulus(&self) -> Vec<u32> {
        let mut e = vec![self.m];
        e.extend(&self.low);
        e
    }

    /// The carry-less multiplier of the field's products.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::{BinaryField, Clmul};
    ///
    /// assert_eq!(BinaryField::new(&[8, 4, 3, 1, 0]).clmul(), Clmul::detect());
    /// ```
    #[must_use]
    pub fn clmul(&self) -> Clmul {
        self.clmul
    }

    /// The element of the polynomial whose coefficients are the bits of `words` (the coefficient of
    /// `xⁱ` bit `i mod 64` of word `i / 64`), of any length, reduced modulo `f`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::BinaryField;
    ///
    /// let f = BinaryField::new(&[8, 4, 3, 1, 0]);
    /// assert_eq!(f.element(&[0x100]).words(), &[0x1b]); // x⁸ = x⁴ + x³ + x + 1
    /// assert_eq!(f.element(&[]), f.zero());
    /// ```
    #[must_use]
    pub fn element(&self, words: &[u64]) -> F2m<'_> {
        F2m {
            f: self,
            w: self.reduce_any(words.to_vec()),
        }
    }

    /// 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::BinaryField;
    ///
    /// assert!(BinaryField::new(&[8, 4, 3, 1, 0]).zero().is_zero());
    /// ```
    #[must_use]
    pub fn zero(&self) -> F2m<'_> {
        F2m {
            f: self,
            w: vec![0; self.words],
        }
    }

    /// 1.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::BinaryField;
    ///
    /// let f = BinaryField::new(&[8, 4, 3, 1, 0]);
    /// assert!(f.one().is_one());
    /// assert_eq!(f.one().words(), &[1]);
    /// ```
    #[must_use]
    pub fn one(&self) -> F2m<'_> {
        self.element(&[1])
    }

    /// `w` reduced modulo `f`, `n` words.
    fn reduce_any(&self, mut w: Vec<u64>) -> Vec<u64> {
        let n = self.words;
        let m = self.m as usize;
        match &self.reduction {
            Reduction::Sparse { chunk } => {
                if w.len() < n {
                    w.resize(n, 0);
                }
                let top = 64 * w.len();
                reduce_sparse(&mut w, m, &self.low, *chunk, top);
                w.truncate(n);
                w
            }
            Reduction::Barrett { .. } => {
                let fits = w.len() <= 2 * n && poly::degree(&w).is_none_or(|d| d < 2 * m);
                with_scratch(2 * n + self.scratch, |s| {
                    let (c, s) = s.split_at_mut(2 * n);
                    if fits {
                        c.fill(0);
                        c[..w.len()].copy_from_slice(&w);
                        self.reduce(c, s);
                        return c[..n].to_vec();
                    }
                    // Horner's rule from the top, k = min(m, 64) bits at a time: (acc·xᵏ + chunk)
                    // mod f, each of degree below m + k ≤ 2m.
                    let k = m.min(64);
                    let total = 64 * w.len();
                    let rest = total % k;
                    let mut pos = total - if rest == 0 { k } else { rest };
                    let mut width = total - pos;
                    let mut acc = vec![0u64; n];
                    loop {
                        c.fill(0);
                        poly::add_shifted(c, &acc, width);
                        c[0] ^= poly::bits(&w, pos, width);
                        self.reduce(c, s);
                        acc.copy_from_slice(&c[..n]);
                        if pos == 0 {
                            break;
                        }
                        pos -= k;
                        width = k;
                    }
                    acc
                })
            }
        }
    }

    /// Reduces `c`, `2n` words of degree below `2m`, into `c[..n]`, the words above left as they
    /// fall; `s` is the field's scratch.
    fn reduce(&self, c: &mut [u64], s: &mut [u64]) {
        let n = self.words;
        let m = self.m as usize;
        match &self.reduction {
            Reduction::Sparse { chunk } => reduce_sparse(c, m, &self.low, *chunk, 2 * m),
            Reduction::Barrett { mu, g } => {
                let (a, s) = s.split_at_mut(n);
                let (p, s) = s.split_at_mut(2 * n);
                let (q, s) = s.split_at_mut(n);
                // a = ⌊c / xᵐ⌋, q = ⌊a·μ / xᵐ⌋ = ⌊a·(μ − xᵐ) / xᵐ⌋ + a.
                poly::shift_right_into(a, c, m);
                clmul::mul(self.clmul, a, mu, p, s);
                poly::shift_right_into(q, p, m);
                for (q, a) in q.iter_mut().zip(a.iter()) {
                    *q ^= a;
                }
                // c mod f = the bits of c + q·(f − xᵐ) below m.
                clmul::mul(self.clmul, q, g, p, s);
                for (c, p) in c.iter_mut().zip(p.iter()).take(n) {
                    *c ^= p;
                }
                if !m.is_multiple_of(64) {
                    c[n - 1] &= (1 << (m % 64)) - 1;
                }
            }
        }
    }

    /// The product of two elements' words, reduced.
    fn mul_words(&self, a: &[u64], b: &[u64]) -> Vec<u64> {
        let n = self.words;
        with_scratch(2 * n + self.scratch, |s| {
            let (p, s) = s.split_at_mut(2 * n);
            clmul::mul(self.clmul, a, b, p, s);
            self.reduce(p, s);
            p[..n].to_vec()
        })
    }

    /// The square of an element's words, reduced.
    fn square_words(&self, a: &[u64]) -> Vec<u64> {
        let n = self.words;
        with_scratch(2 * n + self.scratch, |s| {
            let (p, s) = s.split_at_mut(2 * n);
            clmul::square(a, p);
            self.reduce(p, s);
            p[..n].to_vec()
        })
    }

    /// Panics unless `x` is an element of this field.
    #[track_caller]
    fn check(&self, x: &F2m<'_>) {
        assert!(
            core::ptr::eq(self, x.f) || self == x.f,
            "elements of different fields"
        );
    }
}

impl PartialEq for BinaryField {
    fn eq(&self, other: &BinaryField) -> bool {
        self.m == other.m && self.low == other.low
    }
}

impl Eq for BinaryField {}

impl Hash for BinaryField {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.m.hash(state);
        self.low.hash(state);
    }
}

impl fmt::Debug for BinaryField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let reduction = match self.reduction {
            Reduction::Sparse { .. } => "Sparse",
            Reduction::Barrett { .. } => "Barrett",
        };
        f.debug_struct("BinaryField")
            .field("modulus", &self.modulus())
            .field("clmul", &self.clmul)
            .field("reduction", &format_args!("{reduction}"))
            .finish()
    }
}

/// The degree and the lower exponents, decreasing, of the polynomial with the terms `exponents`.
#[track_caller]
fn parse(exponents: &[u32]) -> (u32, Vec<u32>) {
    let mut e = exponents.to_vec();
    e.sort_unstable_by(|a, b| b.cmp(a));
    assert!(!e.is_empty() && e[0] > 0, "the degree must be at least 1");
    assert!(
        e.windows(2).all(|w| w[0] != w[1]),
        "repeated exponent in {exponents:?}"
    );
    (e[0], e[1..].to_vec())
}

/// The distinct prime factors of `m`.
fn prime_factors(mut m: u32) -> Vec<u32> {
    let mut p = Vec::new();
    let mut q = 2;
    while u64::from(q) * u64::from(q) <= u64::from(m) {
        if m.is_multiple_of(q) {
            p.push(q);
            while m.is_multiple_of(q) {
                m /= q;
            }
        }
        q += 1;
    }
    if m > 1 {
        p.push(m);
    }
    p
}

/// The reduction for `f`, of `n` words, under the kernel `clmul`: the sparse one while its chunks
/// and terms cost no more than Barrett's two products, as for the trinomials and pentanomials of
/// the standards; Barrett's for moduli of many terms or with terms close to `xᵐ`, of one word or
/// more. The costs are rough counts of word operations, calibrated on one machine (Apple M4) for
/// `PMULL` and the portable kernel at one and two words, to be tuned when the fields are measured:
/// a chunk costs `3 + t` for `t` terms below `xᵐ`; a product of words 3 by an instruction and 20
/// by the portable kernel, and Karatsuba's sums `6n` a level.
fn choose_reduction(m: u32, n: usize, low: &[u32], f: &[u64], clmul: Clmul) -> Reduction {
    let e1 = low.first().copied().unwrap_or(0);
    let chunk = (m - e1).min(64) as usize;
    let sparse = sparse_chunks(m as usize, chunk) * (3 + low.len());
    let word = if clmul == Clmul::Portable { 20 } else { 3 };
    let levels = (usize::BITS - (n - 1).leading_zeros()) as usize;
    let product = word_products(n) * word + 6 * n * levels;
    let barrett = 2 * product + 4 * n;
    if sparse <= barrett {
        return Reduction::Sparse { chunk };
    }
    // μ = ⌊x²ᵐ / f⌋, of degree m.
    let m = m as usize;
    let mut x2m = vec![0u64; (2 * m + 1).div_ceil(64)];
    x2m[2 * m / 64] = 1 << (2 * m % 64);
    let (mut mu, _) = poly::div_rem(&x2m, f);
    mu[m / 64] ^= 1 << (m % 64);
    mu.truncate(n);
    let mut g = f.to_vec();
    g[m / 64] ^= 1 << (m % 64);
    g.truncate(n);
    Reduction::Barrett { mu, g }
}

/// The number of chunks [`reduce_sparse`] takes for a product, of degree below `2m − 1`.
fn sparse_chunks(m: usize, chunk: usize) -> usize {
    let mut top = 2 * m - 1;
    let mut count = 0;
    while top > m {
        top = m.max(top.saturating_sub(chunk)).max((top - 1) / 64 * 64);
        count += 1;
    }
    count
}

/// The products of words in a product of two polynomials of `n` words by Karatsuba's method as
/// `binary_clmul` splits it.
fn word_products(n: usize) -> usize {
    if n < 2 {
        n
    } else {
        let lo = n.div_ceil(2);
        2 * word_products(lo) + word_products(n - lo)
    }
}

/// Reduces `c`, of degree below `top`, modulo `f = xᵐ + Σ x^low`, in place: chunks of the bits from
/// `m` up, from the top, each within one word and at most `chunk ≤ m − e₁` bits, so that its
/// multiples by the terms below `xᵐ` land below it, each such multiple a sum of a shifted word.
fn reduce_sparse(c: &mut [u64], m: usize, low: &[u32], chunk: usize, mut top: usize) {
    while top > m {
        let hi = top - 1;
        let lo = m.max(top.saturating_sub(chunk)).max(hi / 64 * 64);
        let width = top - lo;
        let (wi, sh) = (lo / 64, lo % 64);
        let mask = if width == 64 {
            u64::MAX
        } else {
            (1u64 << width) - 1
        };
        let t = (c[wi] >> sh) & mask;
        if t != 0 {
            c[wi] &= !(mask << sh);
            for &e in low {
                poly::add_word_at(c, t, lo - m + e as usize);
            }
        }
        top = lo;
    }
}

/// The bits `Tr(xⁱ)`, `i < m`: the power sums `pₖ` of the roots of `f`, by Newton's identities
/// over `GF(2)`, `pₖ = Σ_{j<k} eⱼ·p_{k−j} + k·eₖ` with `eⱼ` the coefficient of `x^(m−j)` in `f`,
/// and `p₀ = m mod 2`: `O(m·t)` for `t` terms.
fn trace_mask(m: u32, low: &[u32], n: usize) -> Vec<u64> {
    let m = m as usize;
    let js: Vec<usize> = low.iter().map(|&e| m - e as usize).collect();
    let mut p = vec![false; m];
    p[0] = m % 2 == 1;
    for k in 1..m {
        let mut s = k % 2 == 1 && js.contains(&k);
        for &j in &js {
            if j < k {
                s ^= p[k - j];
            }
        }
        p[k] = s;
    }
    let mut mask = vec![0u64; n];
    for (i, &b) in p.iter().enumerate() {
        if b {
            mask[i / 64] |= 1 << (i % 64);
        }
    }
    mask
}

/// An element of a [`BinaryField`], which it borrows.
///
/// # Panics
///
/// The operators `+ - * /` and their assignments panic on elements of two different fields (fields
/// of different moduli); `/` and `/=` panic on division by zero.
///
/// # Examples
///
/// ```
/// use a::math::field::BinaryField;
///
/// let f = BinaryField::new(&[163, 7, 6, 3, 0]); // sect163k1's field
/// let x = f.element(&[0b1011]);
/// assert_eq!(&x + &x, f.zero());
/// assert_eq!(&x / &x, f.one());
/// assert_eq!(x.square(), &x * &x);
/// ```
#[derive(Clone)]
pub struct F2m<'f> {
    f: &'f BinaryField,
    /// The coefficients, `⌈m / 64⌉` words, the bits from `m` on zero.
    w: Vec<u64>,
}

impl<'f> F2m<'f> {
    /// The field.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::BinaryField;
    ///
    /// let f = BinaryField::new(&[8, 4, 3, 1, 0]);
    /// assert_eq!(f.one().field().degree(), 8);
    /// ```
    #[must_use]
    pub fn field(&self) -> &'f BinaryField {
        self.f
    }

    /// The coefficients: the coefficient of `xⁱ` is bit `i mod 64` of word `i / 64`, `⌈m / 64⌉`
    /// words.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::BinaryField;
    ///
    /// let f = BinaryField::new(&[131, 8, 3, 2, 0]);
    /// assert_eq!(f.element(&[5]).words(), &[5, 0, 0]);
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
    /// use a::math::field::BinaryField;
    ///
    /// let f = BinaryField::new(&[8, 4, 3, 1, 0]);
    /// assert!((f.one() + f.one()).is_zero());
    /// ```
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.w.iter().all(|&x| x == 0)
    }

    /// Whether it is 1.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::BinaryField;
    ///
    /// let f = BinaryField::new(&[8, 4, 3, 1, 0]);
    /// assert!(f.element(&[0x101]).is_one() == false);
    /// assert!((&f.element(&[0x53]) * &f.element(&[0xca])).is_one());
    /// ```
    #[must_use]
    pub fn is_one(&self) -> bool {
        self.w[0] == 1 && self.w[1..].iter().all(|&x| x == 0)
    }

    /// The square, by spreading the bits and reducing.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::BinaryField;
    ///
    /// let f = BinaryField::new(&[8, 4, 3, 1, 0]);
    /// assert_eq!(f.element(&[0x10]).square().words(), &[0x1b]); // (x⁴)² = x⁸
    /// ```
    #[must_use]
    pub fn square(&self) -> F2m<'f> {
        F2m {
            f: self.f,
            w: self.f.square_words(&self.w),
        }
    }

    /// The `e`-th power, by squarings and products from the top bit of `e`; `0⁰ = 1`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::BinaryField;
    ///
    /// let f = BinaryField::new(&[8, 4, 3, 1, 0]);
    /// let g = f.element(&[3]); // x + 1 generates GF(2⁸)*
    /// assert!(g.pow_u64(255).is_one());
    /// assert!(!g.pow_u64(85).is_one() && !g.pow_u64(51).is_one());
    /// ```
    #[must_use]
    pub fn pow_u64(&self, e: u64) -> F2m<'f> {
        let mut r = self.f.one();
        for i in (0..u64::BITS - e.leading_zeros()).rev() {
            r = r.square();
            if e >> i & 1 == 1 {
                r = &r * self;
            }
        }
        r
    }

    /// The inverse, by the extended Euclidean algorithm over `GF(2)[x]`, or `None` for 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::BinaryField;
    ///
    /// let f = BinaryField::new(&[8, 4, 3, 1, 0]);
    /// assert_eq!(f.element(&[0x53]).inverse().unwrap().words(), &[0xca]); // FIPS 197, 4.2
    /// assert!(f.zero().inverse().is_none());
    /// ```
    #[must_use]
    pub fn inverse(&self) -> Option<F2m<'f>> {
        poly::inverse(&self.w, &self.f.f, self.f.words).map(|w| F2m { f: self.f, w })
    }

    /// The square root, which every element has, and only one: `a_even + √x·a_odd`, for the
    /// polynomials `a_even` and `a_odd` of the even and the odd coefficients of `a`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::BinaryField;
    ///
    /// let f = BinaryField::new(&[233, 74, 0]);
    /// let a = f.element(&[0xdead_beef, 1 << 40, 7, 1 << 39]);
    /// assert_eq!(a.sqrt().square(), a);
    /// assert_eq!(f.element(&[0b100]).sqrt().words()[0], 0b10); // √(x²) = x
    /// ```
    #[must_use]
    pub fn sqrt(&self) -> F2m<'f> {
        let n = self.f.words;
        let mut even = vec![0u64; n];
        let mut odd = vec![0u64; n];
        for (i, &w) in self.w.iter().enumerate() {
            let shift = 32 * (i % 2);
            even[i / 2] |= u64::from(clmul::gather_even(w)) << shift;
            odd[i / 2] |= u64::from(clmul::gather_even(w >> 1)) << shift;
        }
        let mut r = self.f.mul_words(&odd, &self.f.sqrt_x);
        for (r, e) in r.iter_mut().zip(&even) {
            *r ^= e;
        }
        F2m { f: self.f, w: r }
    }

    /// The absolute trace `a + a² + a⁴ + … + a^(2ᵐ⁻¹)`, 0 or 1: the parity of the coefficients of
    /// `a` at the `xⁱ` of trace 1.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::BinaryField;
    ///
    /// let f = BinaryField::new(&[163, 7, 6, 3, 0]);
    /// assert_eq!(f.one().trace(), 1); // Tr(1) = m mod 2
    /// let a = f.element(&[0x1234, 0x5678]);
    /// assert_eq!(a.square().trace(), a.trace());
    /// ```
    #[must_use]
    pub fn trace(&self) -> u8 {
        let ones: u32 = self
            .w
            .iter()
            .zip(&self.f.trace)
            .map(|(a, t)| (a & t).count_ones())
            .sum();
        (ones % 2) as u8
    }

    /// The half-trace `a + a^(2²) + a^(2⁴) + … + a^(2^(m−1))`, for an odd degree `m`: when the
    /// trace of `c` is 0, `z = c.half_trace()` solves `z² + z = c` (the other root is `z + 1`), as
    /// the decompression of points on binary curves needs.
    ///
    /// # Panics
    ///
    /// Panics if the degree is even.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::BinaryField;
    ///
    /// let f = BinaryField::new(&[163, 7, 6, 3, 0]);
    /// let c = f.element(&[0x1234_5678, 0x9abc]);
    /// let c = if c.trace() == 0 { c } else { &c + &f.one() }; // m odd: Tr(1) = 1
    /// let z = c.half_trace();
    /// assert_eq!(&z.square() + &z, c);
    /// ```
    #[track_caller]
    #[must_use]
    pub fn half_trace(&self) -> F2m<'f> {
        assert!(self.f.m % 2 == 1, "the half-trace needs an odd degree");
        let mut h = self.clone();
        let mut s = self.clone();
        for _ in 0..(self.f.m - 1) / 2 {
            s = s.square().square();
            h = &h + &s;
        }
        h
    }

    /// A root `z` of `z² + z = c` for `c = self`, or `None` when there is none, which is when the
    /// trace of `c` is 1; the other root is `z + 1`. For an odd degree `z` is the half-trace; for an
    /// even one, `z = Σ_{i<m−1} θᵢ·c^(2ⁱ)` with `θᵢ = Σ_{i<j<m} δ^(2ʲ)` for a `δ` of trace 1 (then
    /// `z² + z = c + δ·Tr(c)`), `m − 1` products.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::BinaryField;
    ///
    /// let f = BinaryField::new(&[8, 4, 3, 1, 0]); // an even degree
    /// for v in 0..256u64 {
    ///     let c = f.element(&[v]);
    ///     match c.quadratic_root() {
    ///         Some(z) => assert_eq!(&z.square() + &z, c),
    ///         None => assert_eq!(c.trace(), 1),
    ///     }
    /// }
    /// ```
    #[must_use]
    pub fn quadratic_root(&self) -> Option<F2m<'f>> {
        if self.trace() == 1 {
            return None;
        }
        if self.f.m % 2 == 1 {
            return Some(self.half_trace());
        }
        // δ = the first xⁱ of trace 1, which exists as the trace is not 0.
        let i = (0..self.f.m as usize)
            .find(|&i| self.f.trace[i / 64] >> (i % 64) & 1 == 1)
            .expect("the trace is a nonzero map");
        let mut words = vec![0u64; self.f.words];
        words[i / 64] = 1 << (i % 64);
        let mut d = F2m {
            f: self.f,
            w: words,
        };
        let mut theta = &self.f.one() + &d;
        let mut c = self.clone();
        let mut z = self.f.zero();
        for _ in 0..self.f.m - 1 {
            z += &theta * &c;
            c = c.square();
            d = d.square();
            theta += &d;
        }
        Some(z)
    }
}

impl PartialEq for F2m<'_> {
    fn eq(&self, other: &F2m<'_>) -> bool {
        (core::ptr::eq(self.f, other.f) || self.f == other.f) && self.w == other.w
    }
}

impl Eq for F2m<'_> {}

impl Hash for F2m<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.f.m.hash(state);
        self.w.hash(state);
    }
}

impl fmt::Debug for F2m<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "0x")?;
        for w in self.w.iter().rev() {
            write!(f, "{w:016x}")?;
        }
        Ok(())
    }
}

impl<'f> Neg for &F2m<'f> {
    type Output = F2m<'f>;
    fn neg(self) -> F2m<'f> {
        self.clone()
    }
}

impl<'f> Neg for F2m<'f> {
    type Output = F2m<'f>;
    fn neg(self) -> F2m<'f> {
        self
    }
}

/// `a + b`, which is `a − b`.
fn sum<'f>(a: &F2m<'f>, b: &F2m<'f>) -> F2m<'f> {
    F2m {
        f: a.f,
        w: a.w.iter().zip(&b.w).map(|(x, y)| x ^ y).collect(),
    }
}

/// `a·b`: the carry-less product, then the reduction modulo `f`.
fn product<'f>(a: &F2m<'f>, b: &F2m<'f>) -> F2m<'f> {
    F2m {
        f: a.f,
        w: a.f.mul_words(&a.w, &b.w),
    }
}

/// The binary operators for the four combinations of owned and borrowed operands, and their
/// assignments, from the one on two borrows.
macro_rules! binary {
    ($tr:ident, $m:ident, $atr:ident, $am:ident, $f:expr) => {
        impl<'f> $tr<&F2m<'f>> for &F2m<'f> {
            type Output = F2m<'f>;
            #[track_caller]
            fn $m(self, rhs: &F2m<'f>) -> F2m<'f> {
                self.f.check(rhs);
                $f(self, rhs)
            }
        }
        impl<'f> $tr<F2m<'f>> for &F2m<'f> {
            type Output = F2m<'f>;
            #[track_caller]
            fn $m(self, rhs: F2m<'f>) -> F2m<'f> {
                self.$m(&rhs)
            }
        }
        impl<'f> $tr<&F2m<'f>> for F2m<'f> {
            type Output = F2m<'f>;
            #[track_caller]
            fn $m(self, rhs: &F2m<'f>) -> F2m<'f> {
                (&self).$m(rhs)
            }
        }
        impl<'f> $tr<F2m<'f>> for F2m<'f> {
            type Output = F2m<'f>;
            #[track_caller]
            fn $m(self, rhs: F2m<'f>) -> F2m<'f> {
                (&self).$m(&rhs)
            }
        }
        impl<'f> $atr<&F2m<'f>> for F2m<'f> {
            #[track_caller]
            fn $am(&mut self, rhs: &F2m<'f>) {
                *self = (&*self).$m(rhs);
            }
        }
        impl<'f> $atr<F2m<'f>> for F2m<'f> {
            #[track_caller]
            fn $am(&mut self, rhs: F2m<'f>) {
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
    |a: &F2m<'f>, b: &F2m<'f>| { product(a, &b.inverse().expect("division by zero")) }
);
