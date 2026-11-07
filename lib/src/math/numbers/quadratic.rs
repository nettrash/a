//! Squares modulo a number: the Jacobi and Kronecker symbols, square roots modulo any number, and
//! Cornacchia's algorithm for `x^2 + d y^2 = m`.

use super::modular::int::{Int, in_order};
use super::modular::ring::{magnitude_64, magnitude_128};
use super::modular::sqrt::{coefficients, roots_prime_power, smallest_sum, sqrt_prime_word, sums};
use super::modular::{jacobi, long_factors, positive, word, word_residue};
use super::{ByteOrder, Factorization, Number};

impl Number {
    /// The Jacobi symbol `(self / n)` for an odd positive `n`: 0 when the two have a common
    /// factor, otherwise 1 or -1. For a prime `n` it is the Legendre symbol, 1 for a non-zero
    /// square modulo `n` and -1 for a non-square; for a composite `n`, 1 does not make `self` a
    /// square, though -1 still rules it out.
    ///
    /// # Panics
    ///
    /// Panics if `n` is even, zero or negative.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// let n = Number::new_from_u8(7, ByteOrder::BigEndian);
    /// // 2 = 3^2 modulo 7; 3 is not a square modulo 7; 14 is a multiple of 7.
    /// assert_eq!(Number::new_from_u8(2, ByteOrder::BigEndian).jacobi(&n), 1);
    /// assert_eq!(Number::new_from_u8(3, ByteOrder::BigEndian).jacobi(&n), -1);
    /// assert_eq!(Number::new_from_u8(14, ByteOrder::BigEndian).jacobi(&n), 0);
    /// ```
    #[track_caller]
    pub fn jacobi(&self, n: &Number) -> i32 {
        if n.negative || !n.bit(0) {
            odd_positive();
        }
        if let (Some(a), Some(m)) = (magnitude_128(self), magnitude_128(n)) {
            let r = if (a | m) >> 64 == 0 {
                u128::from(a as u64 % m as u64)
            } else {
                a % m
            };
            let r = if self.negative && r != 0 { m - r } else { r };
            if m >> 64 == 0 {
                return jacobi::jacobi_word(r as u64, m as u64);
            }
            return jacobi::jacobi_128(r, m);
        }
        let nl = limbs(n);
        // `self` itself when it is already below `n`, its remainder otherwise.
        let mut a = Vec::new();
        if !self.negative && self.significant() <= 8 * nl.len() {
            a = limbs(self);
        }
        let below = !a.is_empty()
            && (a.len() < nl.len()
                || a.iter().rev().cmp(nl.iter().rev()) == core::cmp::Ordering::Less);
        if !below {
            a = limbs(&self.rem_euclid(n));
        }
        jacobi::jacobi(a, nl)
    }

    /// The smallest square root of `self` modulo `m`: the smallest `r >= 0` with `r^2 = self
    /// (mod m)`, or `None` when `self` is not a square modulo `m`. Zero is its own root, and modulo
    /// 1 every number is 0. In the byte order of `self`.
    ///
    /// `m` is factored (see [`Number::factor`]; [`Factorization::sqrt_mod`] takes a known
    /// factorisation and skips that): modulo an odd prime the root is by Tonelli and Shanks's
    /// method, or Cipolla's when `p - 1` has more than 24 factors of two; it is lifted to the
    /// prime's powers by Hensel's lifting, Newton's steps `x - (x^2 - a)/(2x)` that double the
    /// power each; modulo powers of two by the 2-adic steps; and the roots modulo the prime powers
    /// are joined by the Chinese remainder theorem, as [`sqrt_mod_all`](Self::sqrt_mod_all)
    /// describes, the smallest of every choice found by trying them all, or by meeting in the
    /// middle beyond 4096 of them. An odd `m` of Jacobi symbol −1 is answered without factoring.
    ///
    /// Like [`pow_mod`](Number::pow_mod), which it is built on, it is not constant-time.
    ///
    /// # Panics
    ///
    /// Panics if `m` is zero or negative.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// let n = |x| Number::new_from_i64(x, ByteOrder::BigEndian);
    /// // 6^2 = 36 = 10 modulo 13, and 7 = 13 - 6 is the other root.
    /// assert_eq!(n(10).sqrt_mod(&n(13)), Some(n(6)));
    /// assert_eq!(n(5).sqrt_mod(&n(13)), None);
    /// // Modulo 15 = 3 * 5, 4 has the roots 2, 7, 8 and 13; modulo 2^5, 17 has 7, 9, 23 and 25.
    /// assert_eq!(n(4).sqrt_mod(&n(15)), Some(n(2)));
    /// assert_eq!(n(17).sqrt_mod(&n(32)), Some(n(7)));
    /// ```
    #[track_caller]
    pub fn sqrt_mod(&self, m: &Number) -> Option<Number> {
        positive(m);
        let order = self.byte_order;
        if let Some(w) = magnitude_64(m) {
            let a = word_residue(self, w);
            return sqrt_word(a, w).map(|r| Number::new_from_u64(r, order));
        }
        let m = in_order(m.abs(), ByteOrder::LittleEndian);
        let a = self.rem_euclid(&m);
        if a.is_zero() {
            return Some(Number::new_from_u8(0, order));
        }
        if m.bit(0) && a.jacobi(&m) == -1 {
            return None;
        }
        let f = long_factors(&m);
        smallest_root(&a, &f).map(|r| in_order(r, order))
    }

    /// The Kronecker symbol `(self / n)` for any `n`, the Jacobi symbol's extension to even, zero
    /// and negative `n`: `(a / 2)` is 0 for an even `a`, 1 for `a = ±1 (mod 8)` and −1 for
    /// `a = ±3 (mod 8)`; `(a / -1)` is −1 for a negative `a` and 1 otherwise; `(a / 0)` is 1 for
    /// `a = ±1` and 0 otherwise. By Cohen's algorithm 1.4.10: the factors of two of `n` and its
    /// sign taken out, then the Jacobi symbol. For an odd positive `n` it is
    /// [`jacobi`](Self::jacobi).
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// let n = |x| Number::new_from_i64(x, ByteOrder::BigEndian);
    /// assert_eq!(n(5).kronecker(&n(8)), -1);
    /// assert_eq!(n(-3).kronecker(&n(-5)), 1);
    /// assert_eq!(n(1).kronecker(&n(0)), 1);
    /// assert_eq!(n(2).kronecker(&n(6)), 0);
    /// ```
    pub fn kronecker(&self, n: &Number) -> i32 {
        let a_negative = self.negative && !self.is_zero();
        let n_negative = n.negative && !n.is_zero();
        if let (Some(a), Some(nw)) = (magnitude_128(self), magnitude_128(n)) {
            if nw == 0 {
                return i32::from(a == 1);
            }
            return kronecker_128(a, a_negative, nw, n_negative);
        }
        if n.is_zero() {
            // `self` does not fit in 128 bits, so it is not ±1.
            return 0;
        }
        let v = n.trailing_zeros().unwrap_or(0);
        let a_odd = self.bit(0);
        if v > 0 && !a_odd {
            return 0;
        }
        let low = limbs(self).first().copied().unwrap_or(0);
        let mut k = 1;
        if v & 1 == 1 && matches!(low & 7, 3 | 5) {
            k = -k;
        }
        if n_negative && a_negative {
            k = -k;
        }
        // `v` may pass `usize::MAX` on 32-bit targets, for a number of 512 MiB: shifted in steps.
        let mut odd = in_order(n.abs(), ByteOrder::LittleEndian);
        let mut left = v;
        while left > 0 {
            let step = usize::try_from(left).unwrap_or(usize::MAX);
            odd >>= step;
            left -= step as u64;
        }
        k * self.jacobi(&odd)
    }

    /// Every square root of `self` modulo `m`, in increasing order, each in `0..m`: none when
    /// `self` is not a square modulo `m`. By Tonelli and Shanks modulo each prime of `m`, Hensel's
    /// lifting to its powers and the Chinese remainder theorem. In the byte order of `self`.
    ///
    /// The roots modulo `p^e` of `a = p^v u`, `u` prime to `p`: none for an odd `v`; for `v >= e`
    /// (`a = 0`) the multiples of `p^ceil(e/2)`; otherwise `p^(v/2)` times the roots of `u`
    /// modulo `p^(e-v)`, two for an odd `p` (Tonelli–Shanks, then Hensel), and for `p = 2` one
    /// modulo 2, two modulo 4 for `u = 1 (mod 4)` and four modulo `2^k`, `k >= 3`, for
    /// `u = 1 (mod 8)`, each with every multiple of `d = p^(e - v/2)` added. The roots modulo `m`
    /// are those of the Chinese remainder theorem for every choice of one root modulo each `d`,
    /// plus every multiple of the product of the `d`s: `2^k` of them for `k` odd primes that do
    /// not divide `a`, and far more for a square `a` that shares factors with `m` (`m / D` times
    /// as many), which this lists all of.
    ///
    /// # Panics
    ///
    /// Panics if `m` is zero or negative.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// let n = |x| Number::new_from_i64(x, ByteOrder::BigEndian);
    /// let roots = |a: i64, m: i64| -> Vec<i64> {
    ///     n(a).sqrt_mod_all(&n(m)).iter().map(|r| i64::try_from(r).unwrap()).collect()
    /// };
    /// assert_eq!(roots(4, 15), [2, 7, 8, 13]);
    /// assert_eq!(roots(0, 8), [0, 4]);
    /// assert_eq!(roots(-1, 65), [8, 18, 47, 57]);
    /// assert!(roots(3, 8).is_empty());
    /// ```
    #[track_caller]
    pub fn sqrt_mod_all(&self, m: &Number) -> Vec<Number> {
        positive(m);
        let order = self.byte_order;
        if let Some(w) = magnitude_64(m) {
            let a = word_residue(self, w);
            let f: Vec<(u64, u32)> = word::factor(w).iter().collect();
            return all_roots(&a, &f, &w)
                .into_iter()
                .map(|r| Number::new_from_u64(r, order))
                .collect();
        }
        let m = in_order(m.abs(), ByteOrder::LittleEndian);
        let a = self.rem_euclid(&m);
        if m.bit(0) && !a.is_zero() && a.jacobi(&m) == -1 {
            return Vec::new();
        }
        let f = long_factors(&m);
        all_roots(&a, &f, &m)
            .into_iter()
            .map(|r| in_order(r, order))
            .collect()
    }

    /// The primitive solutions of `x^2 + d y^2 = m` (`gcd(x, y) = 1`) with `x, y >= 0`, by
    /// Cornacchia's algorithm, in increasing order of `x`: none when there is none. In the byte
    /// order of `m`.
    ///
    /// Cornacchia (1908): a primitive solution has `x = r y (mod m)` for a square root `r` of `-d`
    /// modulo `m`; Euclid's algorithm on `m` and `r` stops at the first remainder `x` below
    /// `sqrt(m)`, and `(m - x^2) / d` is then `y^2` when there is a solution for `r` at all. Every
    /// root `r <= m / 2` is taken (`r` and `m - r` give the same solution), from
    /// [`sqrt_mod_all`](Self::sqrt_mod_all), so that a composite `m` has all its solutions; for
    /// `d = 1` the solutions come in pairs `(x, y)` and `(y, x)`. For a prime `m` that is one
    /// square root and one Euclid's algorithm on numbers of its length.
    ///
    /// # Panics
    ///
    /// Panics if `d` or `m` is zero or negative.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// let n = |x| Number::new_from_i64(x, ByteOrder::BigEndian);
    /// // 65 = 1 + 64 = 16 + 49, as a sum of two squares four ways in order.
    /// let s = Number::cornacchia(&n(1), &n(65));
    /// assert_eq!(s, [(n(1), n(8)), (n(4), n(7)), (n(7), n(4)), (n(8), n(1))]);
    /// // The prime 1009 = 19^2 + 2 * 18^2, in one way.
    /// assert_eq!(Number::cornacchia(&n(2), &n(1009)), [(n(19), n(18))]);
    /// assert!(Number::cornacchia(&n(1), &n(21)).is_empty());
    /// ```
    #[track_caller]
    pub fn cornacchia(d: &Number, m: &Number) -> Vec<(Number, Number)> {
        positive(m);
        if d.negative || d.is_zero() {
            d_not_positive();
        }
        let order = m.byte_order;
        if let (Some(dw), Some(mw)) = (magnitude_64(d), magnitude_64(m)) {
            let f: Vec<(u64, u32)> = word::factor(mw).iter().collect();
            return cornacchia_words(dw, mw, &f, order);
        }
        let d = in_order(d.abs(), ByteOrder::LittleEndian);
        let m = in_order(m.abs(), ByteOrder::LittleEndian);
        let roots = if m == 1u8 {
            Vec::new()
        } else {
            (-&d).sqrt_mod_all(&m)
        };
        finish_cornacchia(cornacchia_all(&d, &m, roots), order)
    }
}

/// The solutions sorted, without repeats, in the byte order `order`.
fn finish_cornacchia(mut out: Vec<(Number, Number)>, order: ByteOrder) -> Vec<(Number, Number)> {
    out.sort();
    out.dedup();
    out.into_iter()
        .map(|(x, y)| (in_order(x, order), in_order(y, order)))
        .collect()
}

/// Cornacchia's algorithm in words, for `d` and `m` below `2^64`, `m = prod p^e`.
fn cornacchia_words(d: u64, m: u64, f: &[(u64, u32)], order: ByteOrder) -> Vec<(Number, Number)> {
    let mut out: Vec<(u64, u64)> = Vec::new();
    if m == 1 {
        out.push((1, 0));
        if d == 1 {
            out.push((0, 1));
        }
    } else {
        let minus_d = (m - d % m) % m;
        let roots = all_roots(&minus_d, f, &m);
        let root = word::isqrt(m);
        for r in roots {
            if r > m / 2 {
                break;
            }
            let (mut a, mut b) = (m, r);
            while b > root {
                (a, b) = (b, a % b);
            }
            let x = b;
            let rest = m - x * x;
            if !rest.is_multiple_of(d) {
                continue;
            }
            let y2 = rest / d;
            let y = word::isqrt(y2);
            if y * y != y2 || y == 0 || word::gcd(x, y) != 1 {
                continue;
            }
            if d == 1 {
                out.push((y, x));
            }
            out.push((x, y));
        }
    }
    out.sort_unstable();
    out.dedup();
    out.into_iter()
        .map(|(x, y)| {
            (
                Number::new_from_u64(x, order),
                Number::new_from_u64(y, order),
            )
        })
        .collect()
}

impl Factorization {
    /// The smallest square root of `a` modulo the number factored, as [`Number::sqrt_mod`] gives
    /// it, without factoring: the primes are taken as given (a composite one gives wrong roots, or
    /// none). In the byte order of `a`. Modulo the factorisation of 1, every number is 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Factorization, Number};
    ///
    /// let n = |x| Number::new_from_i64(x, ByteOrder::BigEndian);
    /// // The prime of Curve25519.
    /// let p = (n(1) << 255usize) - 19u8;
    /// let f = Factorization::new(vec![(p.clone(), 1)], ByteOrder::BigEndian);
    /// let r = (&p - 1u8).clone();
    /// let root = f.sqrt_mod(&r).unwrap();
    /// assert_eq!((&root * &root).rem_euclid(&p), r);
    /// ```
    pub fn sqrt_mod(&self, a: &Number) -> Option<Number> {
        let order = a.byte_order;
        if let Some((f, m)) = self.modulus_words() {
            let a = word_residue(a, m);
            if a == 0 || m == 1 {
                return Some(Number::new_from_u8(0, order));
            }
            return smallest_root_word(a, &f).map(|r| Number::new_from_u64(r, order));
        }
        let m = self.value();
        let f: Vec<(Number, u32)> = self
            .factors()
            .iter()
            .map(|(p, e)| (in_order(p.clone(), ByteOrder::LittleEndian), *e))
            .collect();
        let a = in_order(a.rem_euclid(&m), ByteOrder::LittleEndian);
        smallest_root(&a, &f).map(|r| in_order(r, order))
    }

    /// Every square root of `a` modulo the number factored, in increasing order, as
    /// [`Number::sqrt_mod_all`] gives them, without factoring. In the byte order of `a`.
    pub fn sqrt_mod_all(&self, a: &Number) -> Vec<Number> {
        let order = a.byte_order;
        if let Some((f, m)) = self.modulus_words() {
            let a = word_residue(a, m);
            return all_roots(&a, &f, &m)
                .into_iter()
                .map(|r| Number::new_from_u64(r, order))
                .collect();
        }
        let m = in_order(self.value(), ByteOrder::LittleEndian);
        let f: Vec<(Number, u32)> = self
            .factors()
            .iter()
            .map(|(p, e)| (in_order(p.clone(), ByteOrder::LittleEndian), *e))
            .collect();
        let a = in_order(a.rem_euclid(&m), ByteOrder::LittleEndian);
        all_roots(&a, &f, &m)
            .into_iter()
            .map(|r| in_order(r, order))
            .collect()
    }
}

impl Factorization {
    /// The primitive solutions of `x^2 + d y^2 = m` with `x, y >= 0` for the number `m` factored,
    /// as [`Number::cornacchia`] gives them, without factoring: the primes are taken as given. In
    /// the byte order of the factorisation.
    ///
    /// # Panics
    ///
    /// Panics if `d` is zero or negative.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Factorization, Number};
    ///
    /// let n = |x| Number::new_from_i64(x, ByteOrder::BigEndian);
    /// // The prime of Curve25519 is 1 modulo 4: a sum of two squares, in one way up to order.
    /// let p = (n(1) << 255usize) - 19u8;
    /// let f = Factorization::new(vec![(p.clone(), 1)], ByteOrder::BigEndian);
    /// let s = f.cornacchia(&n(1));
    /// assert_eq!(s.len(), 2);
    /// let (x, y) = &s[0];
    /// assert_eq!(x * x + y * y, p);
    /// assert_eq!(s[1], (y.clone(), x.clone()));
    /// ```
    #[track_caller]
    pub fn cornacchia(&self, d: &Number) -> Vec<(Number, Number)> {
        if d.negative || d.is_zero() {
            d_not_positive();
        }
        let order = self.byte_order();
        if let (Some(dw), Some((f, m))) = (magnitude_64(d), self.modulus_words()) {
            return cornacchia_words(dw, m, &f, order);
        }
        let d = in_order(d.abs(), ByteOrder::LittleEndian);
        let m = in_order(self.value(), ByteOrder::LittleEndian);
        let roots = if m == 1u8 {
            Vec::new()
        } else {
            self.sqrt_mod_all(&-&d)
        };
        finish_cornacchia(cornacchia_all(&d, &m, roots), order)
    }

    /// The primes and exponents as words, and the number factored, when it is below `2^64`.
    fn modulus_words(&self) -> Option<(Vec<(u64, u32)>, u64)> {
        let mut out = Vec::with_capacity(self.factors().len());
        let mut value = 1u64;
        for (p, e) in self.factors() {
            let p = magnitude_64(p)?;
            value = value.checked_mul(p.checked_pow(*e)?)?;
            out.push((p, *e));
        }
        Some((out, value))
    }
}

/// Panics for a `d` of Cornacchia's equation that is zero or negative.
#[cold]
#[inline(never)]
#[track_caller]
fn d_not_positive() -> ! {
    panic!("d must be positive")
}

/// The Kronecker symbol `(±a / ±n)` for magnitudes of 128 bits, `n` not zero.
fn kronecker_128(a: u128, a_negative: bool, n: u128, n_negative: bool) -> i32 {
    if n >> 64 == 0 {
        return kronecker_word(a, a_negative, n as u64, n_negative);
    }
    let v = n.trailing_zeros();
    if v > 0 && a & 1 == 0 {
        return 0;
    }
    let mut k = 1;
    if v & 1 == 1 && matches!(a & 7, 3 | 5) {
        k = -k;
    }
    if n_negative && a_negative {
        k = -k;
    }
    let n = n >> v;
    let r = a % n;
    let r = if a_negative && r != 0 { n - r } else { r };
    k * jacobi::jacobi_128(r, n)
}

/// The Kronecker symbol `(±a / ±n)` for a magnitude `a` of 128 bits and a word `n`, not zero.
fn kronecker_word(a: u128, a_negative: bool, n: u64, n_negative: bool) -> i32 {
    let v = n.trailing_zeros();
    if v > 0 && a & 1 == 0 {
        return 0;
    }
    let mut k = 1;
    // `(a / 2)` depends on `a mod 8` up to its sign.
    if v & 1 == 1 && matches!(a & 7, 3 | 5) {
        k = -k;
    }
    if n_negative && a_negative {
        k = -k;
    }
    let n = n >> v;
    let r = if a >> 64 == 0 {
        a as u64 % n
    } else {
        (a % u128::from(n)) as u64
    };
    let r = if a_negative && r != 0 { n - r } else { r };
    k * jacobi::jacobi_word(r, n)
}

/// The smallest square root of `a` modulo the word `m`.
fn sqrt_word(a: u64, m: u64) -> Option<u64> {
    if a == 0 || m == 1 {
        return Some(0);
    }
    if m & 1 == 1 && jacobi::jacobi_word(a, m) == -1 {
        return None;
    }
    let f: Vec<(u64, u32)> = word::factor(m).iter().collect();
    smallest_root_word(a, &f)
}

/// [`smallest_root`] for a word modulus: a prime takes [`sqrt_prime_word`] alone.
fn smallest_root_word(a: u64, f: &[(u64, u32)]) -> Option<u64> {
    if let [(p, 1)] = f
        && *p != 2
    {
        return sqrt_prime_word(a, *p);
    }
    smallest_root(&a, f)
}

/// The roots modulo each prime power of `m = prod p^e` as `(d, b)`, or `None`.
fn parts<I: Int>(a: &I, f: &[(I, u32)]) -> Option<Vec<(I, Vec<I>)>> {
    f.iter()
        .map(|(p, e)| {
            let pe = p.pow(*e);
            roots_prime_power(&a.rem(&pe), p, *e)
        })
        .collect()
}

/// The smallest root of `a` modulo `prod p^e`.
fn smallest_root<I: Int>(a: &I, f: &[(I, u32)]) -> Option<I> {
    let parts = parts(a, f)?;
    let (d, c) = coefficients(&parts);
    Some(smallest_sum(&parts, &c, &d))
}

/// Every root of `a` modulo `m = prod p^e`, increasing.
fn all_roots<I: Int>(a: &I, f: &[(I, u32)], m: &I) -> Vec<I> {
    let Some(parts) = parts(a, f) else {
        return Vec::new();
    };
    let (d, c) = coefficients(&parts);
    let mut xs = sums(&parts, &c, &d);
    xs.sort();
    let mut out = Vec::with_capacity(xs.len());
    let mut base = I::small(0);
    while base < *m {
        for x in &xs {
            out.push(base.add(x));
        }
        if m.sub(&base) <= d {
            break;
        }
        base = base.add(&d);
    }
    out
}

/// Cornacchia's solutions for every root of `-d` modulo `m`, given in increasing order, unsorted.
fn cornacchia_all(d: &Number, m: &Number, roots: Vec<Number>) -> Vec<(Number, Number)> {
    let one = Number::new_from_u8(1, ByteOrder::LittleEndian);
    let zero = Number::new_from_u8(0, ByteOrder::LittleEndian);
    if *m == 1u8 {
        let mut out = vec![(one.clone(), zero)];
        if *d == 1u8 {
            out.push((Number::new_from_u8(0, ByteOrder::LittleEndian), one));
        }
        return out;
    }
    let mut out = Vec::new();
    let half = m >> 1usize;
    // The first remainder whose square is at most `m`.
    let root = m.isqrt();
    let bound = root.bits();
    let ml = limbs(m);
    for r in roots {
        if r > half {
            break;
        }
        // Lehmer's steps while the numbers are far above the bound, then Euclid's one at a time.
        let mut a = ml.clone();
        let mut b = limbs(&r);
        b.resize(a.len(), 0);
        jacobi::euclid_above(&mut a, &mut b, bound);
        let (mut a, mut b) = (
            super::modular::ring::from_limbs(&a, ByteOrder::LittleEndian),
            super::modular::ring::from_limbs(&b, ByteOrder::LittleEndian),
        );
        if a < b {
            core::mem::swap(&mut a, &mut b);
        }
        // Until `b^2 <= m`, that is `b <= isqrt(m)`.
        while b > root {
            let next = a.rem_euclid(&b);
            a = b;
            b = next;
        }
        let x = b;
        let rest = m - &(&x * &x);
        let (y2, left) = rest.div_rem(d);
        if !left.is_zero() {
            continue;
        }
        let y = y2.isqrt();
        if &y * &y != y2 || y.is_zero() || !x.gcd(&y).is_one() {
            continue;
        }
        if *d == 1u8 {
            out.push((y.clone(), x.clone()));
        }
        out.push((x, y));
    }
    out
}

/// Panics for a modulus the symbol is not defined for.
#[cold]
#[inline(never)]
#[track_caller]
fn odd_positive() -> ! {
    panic!("the modulus must be odd and positive")
}

/// The limbs of the magnitude of `x`, the least significant first, without zeros above it.
fn limbs(x: &Number) -> Vec<u64> {
    let mut l: Vec<u64> = if x.byte_order == ByteOrder::LittleEndian {
        x.value
            .chunks(8)
            .map(|c| {
                let mut b = [0; 8];
                b[..c.len()].copy_from_slice(c);
                u64::from_le_bytes(b)
            })
            .collect()
    } else {
        x.value
            .rchunks(8)
            .map(|c| {
                let mut b = [0; 8];
                b[8 - c.len()..].copy_from_slice(c);
                u64::from_be_bytes(b)
            })
            .collect()
    };
    trim(&mut l);
    l
}

/// Drops the zero limbs at the top.
fn trim(x: &mut Vec<u64>) {
    while x.last() == Some(&0) {
        x.pop();
    }
}

/// The Jacobi symbol `(a / n)` of limbs, the least significant first, for `a < n`, `n` odd.
pub(super) fn jacobi_of_limbs(a: Vec<u64>, n: Vec<u64>) -> i32 {
    jacobi::jacobi(a, n)
}
