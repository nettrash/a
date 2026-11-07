//! Arithmetic functions: divisors, their number and the sums of their powers, Euler's totient,
//! Carmichael's function, Möbius's function, square-freeness and the radical, from a
//! [`Factorization`], and on [`Number`] by its factorisation.
//!
//! Numbers below `2^64` are factored in words, and their functions computed in words; the others,
//! and every [`Factorization`] of a number at least `2^64`, on numbers, whose products are taken
//! by balanced trees, so that a long result is built from operands of about equal lengths.

use super::modular::int::in_order;
use super::modular::ring::{magnitude_64, magnitude_128};
use super::modular::word;
use super::{ByteOrder, Factorization, Number};
use std::borrow::Cow;

impl Factorization {
    /// The divisors of the number, in increasing order, 1 and the number included, in the byte
    /// order of the factorisation: `d(n)` of them (see [`divisor_count`](Self::divisor_count)).
    ///
    /// They are generated in order, without sorting: the divisors of `n / p^e` (a sorted list
    /// `D`) give those of `n` as `S_e`, where `S_0 = D` and `S_j` is `D` merged with `p S_(j-1)`,
    /// both sorted. The primes of the largest exponents come first, while the list is short. In
    /// words when the number is below `2^64`, in pairs of words below `2^128`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Factorization, Number};
    ///
    /// let n = |x| Number::new_from_u64(x, ByteOrder::BigEndian);
    /// let f = Factorization::new(vec![(n(2), 2), (n(3), 1)], ByteOrder::BigEndian);
    /// let d: Vec<Number> = [1, 2, 3, 4, 6, 12].iter().map(|&x| n(x)).collect();
    /// assert_eq!(f.divisors(), d);
    /// ```
    pub fn divisors(&self) -> Vec<Number> {
        let order = self.byte_order();
        if let Some(w) = self.words() {
            return divisors_of(&w, 1u64, |d, p| d * p)
                .into_iter()
                .map(|d| Number::new_from_u64(d, order))
                .collect();
        }
        if let Some(w) = self.words_128() {
            return divisors_of(&w, 1u128, |d, p| d * p)
                .into_iter()
                .map(|d| Number::new_from_u128(d, order))
                .collect();
        }
        // Long divisors: each a product of one before it and a prime, then sorted, which takes
        // fewer products and copies than merging long numbers.
        let total = self.factors().iter().fold(1usize, |n, (_, e)| {
            n.saturating_mul((*e as usize).saturating_add(1))
        });
        let mut d = Vec::with_capacity(total);
        d.push(Number::new_from_u8(1, order));
        for (p, e) in self.factors() {
            let len = d.len();
            // Counted saturated: a count past `usize::MAX` fails to allocate, as it must.
            for j in 0..(*e as usize).saturating_mul(len) {
                let x = &d[j] * p;
                d.push(x);
            }
        }
        d.sort_unstable();
        d
    }

    /// The number of divisors, `d(n) = σ_0(n)`, the product of `e + 1` over the primes, in the
    /// byte order of the factorisation.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Factorization, Number};
    ///
    /// let n = |x| Number::new_from_u64(x, ByteOrder::BigEndian);
    /// // 720 = 2^4 3^2 5 has 5 * 3 * 2 = 30 divisors.
    /// let f = Factorization::new(vec![(n(2), 4), (n(3), 2), (n(5), 1)], ByteOrder::BigEndian);
    /// assert_eq!(f.divisor_count(), n(30));
    /// ```
    pub fn divisor_count(&self) -> Number {
        count_of(self.factors().iter().map(|(_, e)| *e), self.byte_order())
    }

    /// The sum of the `k`-th powers of the divisors, `σ_k(n)`, in the byte order of the
    /// factorisation: the product over the primes of `1 + p^k + p^(2k) + ... + p^(ek)`; for
    /// `k = 0` the number of divisors.
    ///
    /// Each factor is summed by Horner's rule, or as `(p^(k(e+1)) - 1) / (p^k - 1)` by a power and
    /// an exact division when it is long, in words while it fits in 128 bits; the factors are
    /// multiplied up a balanced tree.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Factorization, Number};
    ///
    /// let n = |x| Number::new_from_u64(x, ByteOrder::BigEndian);
    /// let f = Factorization::new(vec![(n(2), 2), (n(3), 1)], ByteOrder::BigEndian);
    /// // 1 + 2 + 3 + 4 + 6 + 12 = 28, and 1 + 4 + 9 + 16 + 36 + 144 = 210.
    /// assert_eq!(f.divisor_sigma(1), n(28));
    /// assert_eq!(f.divisor_sigma(2), n(210));
    /// ```
    pub fn divisor_sigma(&self, k: u32) -> Number {
        let order = self.byte_order();
        if k == 0 {
            return self.divisor_count();
        }
        let parts: Vec<Number> = self
            .factors()
            .iter()
            .map(|(p, e)| sigma_prime_power(p, *e, k))
            .collect();
        in_order(product_of(parts), order)
    }

    /// Euler's totient `φ(n)`, the number of integers from 1 to `n` prime to `n`, in the byte order
    /// of the factorisation: the product of `p^(e-1) (p - 1)` over the primes.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Factorization, Number};
    ///
    /// let n = |x| Number::new_from_u64(x, ByteOrder::BigEndian);
    /// let f = Factorization::new(vec![(n(2), 2), (n(3), 1)], ByteOrder::BigEndian);
    /// assert_eq!(f.totient(), n(4));
    /// ```
    pub fn totient(&self) -> Number {
        let order = self.byte_order();
        if let Some(w) = self.words() {
            return Number::new_from_u64(totient_of(&w), order);
        }
        let f = self.factors();
        let mut parts = Vec::with_capacity(2 * f.len());
        for (p, e) in f {
            parts.push(p - 1u8);
            if *e > 1 {
                parts.push(p.pow(*e - 1));
            }
        }
        in_order(product_of(parts), order)
    }

    /// Carmichael's function `λ(n)`, the exponent of the multiplicative group modulo `n`: the
    /// smallest `m >= 1` with `a^m = 1 (mod n)` for every `a` prime to `n`, in the byte order of the
    /// factorisation. The least common multiple of `λ(p^e)`, which is `p^(e-1) (p - 1)` for an odd
    /// prime, and 1, 2 and `2^(e-2)` for `2`, `4` and `2^e`, `e >= 3`. `λ(1) = 1`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Factorization, Number};
    ///
    /// let n = |x| Number::new_from_u64(x, ByteOrder::BigEndian);
    /// // 561 = 3 11 17, the first Carmichael number: λ = lcm(2, 10, 16) = 80, which divides 560.
    /// let f = Factorization::new(vec![(n(3), 1), (n(11), 1), (n(17), 1)], ByteOrder::BigEndian);
    /// assert_eq!(f.carmichael(), n(80));
    /// ```
    pub fn carmichael(&self) -> Number {
        let order = self.byte_order();
        if let Some(w) = self.words() {
            return Number::new_from_u64(carmichael_of(&w), order);
        }
        let parts: Vec<Number> = self
            .factors()
            .iter()
            .map(|(p, e)| lambda_prime_power(p, *e))
            .collect();
        in_order(lcm_of(parts), order)
    }

    /// Möbius's function `μ(n)`: 0 when the square of a prime divides `n`, otherwise `(-1)^k` for
    /// its `k` primes; `μ(1) = 1`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Factorization, Number};
    ///
    /// let n = |x| Number::new_from_u64(x, ByteOrder::BigEndian);
    /// let f = Factorization::new(vec![(n(2), 1), (n(3), 1), (n(5), 1)], ByteOrder::BigEndian);
    /// assert_eq!(f.mobius(), -1);
    /// ```
    pub fn mobius(&self) -> i32 {
        let f = self.factors();
        if f.iter().any(|(_, e)| *e > 1) {
            0
        } else if f.len().is_multiple_of(2) {
            1
        } else {
            -1
        }
    }

    /// Whether no square above 1 divides the number: every exponent is 1. 1 is square-free.
    pub fn is_squarefree(&self) -> bool {
        self.factors().iter().all(|(_, e)| *e == 1)
    }

    /// The radical, the product of the distinct primes, in the byte order of the factorisation;
    /// 1 for the factorisation of 1.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Factorization, Number};
    ///
    /// let n = |x| Number::new_from_u64(x, ByteOrder::BigEndian);
    /// let f = Factorization::new(vec![(n(2), 3), (n(3), 2)], ByteOrder::BigEndian);
    /// assert_eq!(f.radical(), n(6));
    /// ```
    pub fn radical(&self) -> Number {
        let order = self.byte_order();
        let f = self.factors();
        if f.len() <= LINEAR {
            let bytes: usize = f.iter().map(|(p, _)| p.value.len()).sum::<usize>() + 8;
            let mut v = Vec::with_capacity(bytes);
            v.push(1u8);
            let mut acc = Number {
                value: v,
                negative: false,
                byte_order: order,
            };
            for (p, _) in f {
                acc *= p;
            }
            return acc;
        }
        let parts: Vec<Number> = f.iter().map(|(p, _)| p.clone()).collect();
        in_order(product_of(parts), order)
    }

    /// The primes and exponents as words, when the number is below `2^64`.
    fn words(&self) -> Option<Vec<(u64, u32)>> {
        let mut out = Vec::with_capacity(self.factors().len());
        let mut value = 1u64;
        for (p, e) in self.factors() {
            let p = magnitude_64(p)?;
            value = value.checked_mul(p.checked_pow(*e)?)?;
            out.push((p, *e));
        }
        Some(out)
    }

    /// The primes and exponents as `u128`s, when the number is below `2^128`.
    fn words_128(&self) -> Option<Vec<(u128, u32)>> {
        let mut out = Vec::with_capacity(self.factors().len());
        let mut value = 1u128;
        for (p, e) in self.factors() {
            let p = magnitude_128(p)?;
            value = value.checked_mul(p.checked_pow(*e)?)?;
            out.push((p, *e));
        }
        Some(out)
    }
}

impl Number {
    /// The divisors of the absolute value of `self`, from its [factorisation](Number::factor), in
    /// increasing order, 1 and `|self|` included, in the byte order of `self`. As PARI's
    /// `divisors`, the positive divisors of a negative number.
    ///
    /// # Panics
    ///
    /// Panics if `self` is zero, which every number divides.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// let n = |x| Number::new_from_i64(x, ByteOrder::BigEndian);
    /// let d: Vec<Number> = [1, 2, 3, 4, 6, 12].iter().map(|&x| n(x)).collect();
    /// assert_eq!(n(-12).divisors(), d);
    /// ```
    #[track_caller]
    pub fn divisors(&self) -> Vec<Number> {
        let order = self.byte_order;
        if let Some(w) = magnitude_word(self) {
            let f: Vec<(u64, u32)> = factor_word(w).iter().collect();
            return divisors_of(&f, 1u64, |d, p| d * p)
                .into_iter()
                .map(|d| Number::new_from_u64(d, order))
                .collect();
        }
        self.factor().divisors()
    }

    /// The number of divisors of the absolute value of `self`, `d(n)`, in the byte order of `self`.
    ///
    /// # Panics
    ///
    /// Panics if `self` is zero.
    #[track_caller]
    pub fn divisor_count(&self) -> Number {
        if let Some(w) = magnitude_word(self) {
            let f = factor_word(w);
            return count_of(f.iter().map(|(_, e)| e), self.byte_order);
        }
        self.factor().divisor_count()
    }

    /// `σ_k`, the sum of the `k`-th powers of the divisors of the absolute value of `self`, in the
    /// byte order of `self`, as [`Factorization::divisor_sigma`] computes it.
    ///
    /// # Panics
    ///
    /// Panics if `self` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// let n = |x| Number::new_from_i64(x, ByteOrder::BigEndian);
    /// // 28 is perfect: its divisors add up to twice itself.
    /// assert_eq!(n(28).divisor_sigma(1), n(56));
    /// assert_eq!(n(12).divisor_sigma(0), n(6));
    /// ```
    #[track_caller]
    pub fn divisor_sigma(&self, k: u32) -> Number {
        if let Some(w) = magnitude_word(self) {
            let f = factor_word(w);
            if k == 0 {
                return count_of(f.iter().map(|(_, e)| e), self.byte_order);
            }
            let parts: Vec<Number> = f
                .iter()
                .map(|(p, e)| {
                    sigma_prime_power(&Number::new_from_u64(p, ByteOrder::LittleEndian), e, k)
                })
                .collect();
            return in_order(product_of(parts), self.byte_order);
        }
        self.factor().divisor_sigma(k)
    }

    /// Euler's totient of the absolute value of `self`, in the byte order of `self`.
    ///
    /// # Panics
    ///
    /// Panics if `self` is zero (PARI's `eulerphi(0)` is 2, the units of the integers).
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// let n = |x| Number::new_from_i64(x, ByteOrder::BigEndian);
    /// assert_eq!(n(36).totient(), n(12));
    /// assert_eq!(n(1).totient(), n(1));
    /// ```
    #[track_caller]
    pub fn totient(&self) -> Number {
        if let Some(w) = magnitude_word(self) {
            let f: Vec<(u64, u32)> = factor_word(w).iter().collect();
            return Number::new_from_u64(totient_of(&f), self.byte_order);
        }
        self.factor().totient()
    }

    /// Carmichael's function of the absolute value of `self`, in the byte order of `self`.
    ///
    /// # Panics
    ///
    /// Panics if `self` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// let n = |x| Number::new_from_i64(x, ByteOrder::BigEndian);
    /// assert_eq!(n(561).carmichael(), n(80));
    /// assert_eq!(n(8).carmichael(), n(2));
    /// ```
    #[track_caller]
    pub fn carmichael(&self) -> Number {
        if let Some(w) = magnitude_word(self) {
            let f: Vec<(u64, u32)> = factor_word(w).iter().collect();
            return Number::new_from_u64(carmichael_of(&f), self.byte_order);
        }
        self.factor().carmichael()
    }

    /// Möbius's function of the absolute value of `self`.
    ///
    /// # Panics
    ///
    /// Panics if `self` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// let n = |x| Number::new_from_i64(x, ByteOrder::BigEndian);
    /// assert_eq!(n(30).mobius(), -1);
    /// assert_eq!(n(-6).mobius(), 1);
    /// assert_eq!(n(12).mobius(), 0);
    /// ```
    #[track_caller]
    pub fn mobius(&self) -> i32 {
        if let Some(w) = magnitude_word(self) {
            let f = factor_word(w);
            return if f.iter().any(|(_, e)| e > 1) {
                0
            } else if f.len.is_multiple_of(2) {
                1
            } else {
                -1
            };
        }
        self.factor().mobius()
    }

    /// Whether no square above 1 divides `self`: `false` for 0, which every square divides, as
    /// PARI's `issquarefree` gives it.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// let n = |x| Number::new_from_i64(x, ByteOrder::BigEndian);
    /// assert!(n(-30).is_squarefree());
    /// assert!(!n(18).is_squarefree());
    /// assert!(!n(0).is_squarefree());
    /// ```
    pub fn is_squarefree(&self) -> bool {
        if self.is_zero() {
            return false;
        }
        if let Some(w) = magnitude_word(self) {
            return factor_word(w).iter().all(|(_, e)| e == 1);
        }
        self.factor().is_squarefree()
    }

    /// The product of the distinct primes dividing `self`, positive, in the byte order of `self`;
    /// 1 for 1 and -1.
    ///
    /// # Panics
    ///
    /// Panics if `self` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// let n = |x| Number::new_from_i64(x, ByteOrder::BigEndian);
    /// assert_eq!(n(-72).radical(), n(6));
    /// ```
    #[track_caller]
    pub fn radical(&self) -> Number {
        if let Some(w) = magnitude_word(self) {
            let r = factor_word(w).iter().fold(1u64, |r, (p, _)| r * p);
            return Number::new_from_u64(r, self.byte_order);
        }
        self.factor().radical()
    }
}

// Words

/// The magnitude of `x` when it is below `2^64`.
#[inline]
fn magnitude_word(x: &Number) -> Option<u64> {
    magnitude_64(x)
}

/// The factorisation of the word `w`, panicking for zero as [`Number::factor`] does.
#[track_caller]
fn factor_word(w: u64) -> word::Factors {
    if w == 0 {
        zero();
    }
    word::factor(w)
}

/// Panics for the arithmetic functions of zero.
#[cold]
#[inline(never)]
#[track_caller]
fn zero() -> ! {
    panic!("zero has no factorisation")
}

/// The divisors of `prod p^e`, sorted, from 1 by the product `times`: the primes of the largest
/// exponents first, each merged in as [`Factorization::divisors`] describes.
fn divisors_of<T: Clone + Ord, P: Clone>(
    f: &[(P, u32)],
    one: T,
    times: impl Fn(&T, &P) -> T,
) -> Vec<T> {
    let total = f.iter().fold(1usize, |n, (_, e)| {
        n.saturating_mul((*e as usize).saturating_add(1))
    });
    let mut by_exponent: Vec<&(P, u32)> = f.iter().collect();
    by_exponent.sort_by_key(|f| core::cmp::Reverse(f.1));
    let mut d = Vec::with_capacity(total);
    d.push(one);
    let mut scaled: Vec<T> = Vec::with_capacity(total);
    let mut merged: Vec<T> = Vec::with_capacity(total);
    for (p, e) in by_exponent {
        // `S_j = D merged with p S_(j-1)`, from `S_0 = D`.
        let mut s = d.clone();
        for _ in 0..*e {
            scaled.clear();
            scaled.extend(s.iter().map(|x| times(x, p)));
            merged.clear();
            let (mut i, mut j) = (0, 0);
            while i < d.len() && j < scaled.len() {
                if d[i] < scaled[j] {
                    merged.push(d[i].clone());
                    i += 1;
                } else {
                    merged.push(scaled[j].clone());
                    j += 1;
                }
            }
            merged.extend_from_slice(&d[i..]);
            merged.extend_from_slice(&scaled[j..]);
            core::mem::swap(&mut s, &mut merged);
        }
        d = s;
    }
    d
}

/// The number of divisors for the exponents `e`, in the byte order `order`.
fn count_of(e: impl Iterator<Item = u32>, order: ByteOrder) -> Number {
    let mut c = 1u128;
    let mut parts: Vec<Number> = Vec::new();
    for e in e {
        let f = u128::from(e) + 1;
        match c.checked_mul(f) {
            Some(x) => c = x,
            None => {
                parts.push(Number::new_from_u128(c, ByteOrder::LittleEndian));
                c = f;
            }
        }
    }
    if parts.is_empty() {
        return Number::new_from_u128(c, order);
    }
    parts.push(Number::new_from_u128(c, ByteOrder::LittleEndian));
    in_order(product_of(parts), order)
}

/// `φ(n)` of the primes `f` of `n < 2^64`.
fn totient_of(f: &[(u64, u32)]) -> u64 {
    f.iter().fold(1u64, |t, &(p, e)| t * (p - 1) * p.pow(e - 1))
}

/// `λ(n)` of the primes `f` of `n < 2^64`.
fn carmichael_of(f: &[(u64, u32)]) -> u64 {
    f.iter().fold(1u64, |l, &(p, e)| {
        let x = if p == 2 {
            match e {
                1 => 1,
                2 => 2,
                _ => 1 << (e - 2),
            }
        } else {
            (p - 1) * p.pow(e - 1)
        };
        l / word::gcd(l, x) * x
    })
}

// Numbers

/// `1 + p^k + p^(2k) + ... + p^(ek)`.
fn sigma_prime_power(p: &Number, e: u32, k: u32) -> Number {
    if e == 1 && k == 1 {
        return p + 1u8;
    }
    // In 128 bits when `p^(ke)` fits with room for the sum.
    if let Some(pw) = magnitude_128(p) {
        let bits = (128 - pw.leading_zeros()) as u64;
        if bits * u64::from(k) * u64::from(e) < 127 {
            let pk = pw.pow(k);
            let s = (0..e).fold(1u128, |s, _| s * pk + 1);
            return Number::new_from_u128(s, ByteOrder::LittleEndian);
        }
    }
    let pk: Cow<Number> = if k == 1 {
        Cow::Borrowed(p)
    } else {
        Cow::Owned(p.pow(k))
    };
    if e <= 8 {
        // Horner's rule: `p^k + 1`, then times `p^k` and plus 1 for each further power.
        let mut s = &*pk + 1u8;
        for _ in 1..e {
            s *= &*pk;
            s += 1u8;
        }
        return s;
    }
    // `(p^(k(e+1)) - 1) / (p^k - 1)`, exactly; `k (e + 1)` is counted in 64 bits, and a power too
    // long to allocate panics in `pow`.
    let top = u32::try_from(u64::from(k) * (u64::from(e) + 1)).unwrap_or(u32::MAX);
    let num = &p.pow(top) - 1u8;
    num.div_exact(&(&*pk - 1u8))
}

/// `λ(p^e)`.
fn lambda_prime_power(p: &Number, e: u32) -> Number {
    let p = in_order(p.clone(), ByteOrder::LittleEndian);
    if p == 2u8 {
        let one = Number::new_from_u8(1, ByteOrder::LittleEndian);
        return match e {
            1 => one,
            2 => Number::new_from_u8(2, ByteOrder::LittleEndian),
            _ => one << (e as usize - 2),
        };
    }
    let p1 = &p - 1u8;
    if e == 1 { p1 } else { &p1 * &p.pow(e - 1) }
}

/// Up to this many factors, a product is taken one factor at a time into one vector of the
/// product's length, allocated once; beyond, up a balanced tree, whose products have operands of
/// about equal lengths, which the fast multiplications want.
const LINEAR: usize = 16;

/// The product of the numbers, little-endian: 1 for none.
fn product_of(mut x: Vec<Number>) -> Number {
    if x.len() <= LINEAR {
        let bytes: usize = x.iter().map(|n| n.value.len()).sum::<usize>() + 8;
        let mut v = Vec::with_capacity(bytes);
        v.push(1u8);
        let mut acc = Number {
            value: v,
            negative: false,
            byte_order: ByteOrder::LittleEndian,
        };
        for n in &x {
            acc *= n;
        }
        return acc;
    }
    while x.len() > 1 {
        let mut next = Vec::with_capacity(x.len().div_ceil(2));
        let mut it = x.into_iter();
        while let Some(a) = it.next() {
            match it.next() {
                Some(b) => next.push(&a * &b),
                None => next.push(a),
            }
        }
        x = next;
    }
    x.pop().unwrap_or_default()
}

/// The least common multiple of the numbers, up a balanced tree: 1 for none.
fn lcm_of(mut x: Vec<Number>) -> Number {
    if x.is_empty() {
        return Number::new_from_u8(1, ByteOrder::LittleEndian);
    }
    while x.len() > 1 {
        let mut next = Vec::with_capacity(x.len().div_ceil(2));
        let mut it = x.into_iter();
        while let Some(a) = it.next() {
            match it.next() {
                Some(b) => next.push(a.lcm(&b)),
                None => next.push(a),
            }
        }
        x = next;
    }
    x.pop().unwrap_or_default()
}
