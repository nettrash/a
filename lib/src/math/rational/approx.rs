//! Continued fractions and rational approximations: [`Rational::continued_fraction`],
//! [`Rational::from_continued_fraction`], [`Rational::convergents`],
//! [`Rational::limit_denominator`], [`Rational::simplest_between`], and rational reconstruction,
//! [`Rational::reconstruct`] and [`Rational::reconstruct_with_bounds`].
//!
//! Each is Euclid's algorithm on the parts. When they fit in 64 or 128 bits it runs in registers;
//! otherwise on the limbs of the greatest common divisor's own reductions (see
//! `numbers::gcd::euclid`): hgcd2's matrices from the top two limbs, and Möller's half gcd for the
//! long parts, so that a continued fraction, a best approximation and a reconstruction take
//! `O(M(n) log n)` for parts of `n` limbs, as the greatest common divisor does. The simplest
//! rational between two follows both ends' expansions together by hgcd2's matrices.

use super::small::number;
use super::{Rational, in_order, is_one, one, trim};
use crate::math::numbers::{
    Bound, ByteOrder, Magnitude, Number, common_prefix, partial, quotients,
};
use core::cmp::Ordering;

/// The magnitude of `n` when it fits in 64 bits.
#[inline(always)]
fn word(n: &Number) -> Option<u64> {
    let m = Magnitude::of(n);
    if m.bytes.len() <= 8 {
        Some(m.small() as u64)
    } else {
        None
    }
}

/// The magnitude of `n` when it fits in 128 bits.
#[inline(always)]
fn wide(n: &Number) -> Option<u128> {
    let m = Magnitude::of(n);
    if m.bytes.len() <= 16 {
        Some(m.small())
    } else {
        None
    }
}

/// A term of a continued fraction: the magnitude `q` in the byte order `order`, in one vector of
/// exactly its length.
#[inline(always)]
fn term(order: ByteOrder, q: u64) -> Number {
    let len = 8 - (q | 1).leading_zeros() as usize / 8;
    let mut v = Vec::with_capacity(len);
    match order {
        ByteOrder::BigEndian => v.extend_from_slice(&q.to_be_bytes()[8 - len..]),
        ByteOrder::LittleEndian => v.extend_from_slice(&q.to_le_bytes()[..len]),
    }
    Number {
        value: v,
        negative: false,
        byte_order: order,
    }
}

/// A rational from parts in lowest terms, the denominator positive, put in the byte order `order`
/// and the shortest form.
fn reduced(num: Number, den: Number, order: ByteOrder) -> Rational {
    let mut num = in_order(num, order);
    trim(&mut num);
    let mut den = in_order(den, order);
    trim(&mut den);
    Rational::from_reduced(num, den)
}

/// The rational `±m / d` from magnitudes, `d` positive, in lowest terms, in the byte order `order`.
#[inline]
fn small_rational(order: ByteOrder, negative: bool, m: u128, d: u128) -> Rational {
    Rational::from_reduced(number(order, negative, m), number(order, false, d))
}

/// The floor of `±n / d` for magnitudes, `d` not zero, `n / d` not an integer: its sign and
/// magnitude, and the fractional part's numerator, `0 < r < d`.
#[inline]
fn split_u128(negative: bool, n: u128, d: u128) -> (bool, u128, u128) {
    let (q, r) = if (n | d) >> 64 == 0 {
        let (n, d) = (n as u64, d as u64);
        (u128::from(n / d), u128::from(n % d))
    } else {
        (n / d, n % d)
    };
    if negative {
        (true, q + 1, d - r)
    } else {
        (false, q, r)
    }
}

/// Pushes the terms of Euclid's algorithm on `x > y > 0` to `out`.
#[inline]
fn push_terms(order: ByteOrder, mut x: u128, mut y: u128, out: &mut Vec<Number>) {
    while (x | y) >> 64 != 0 {
        let q = x / y;
        let r = x - q * y;
        out.push(number(order, false, q));
        if r == 0 {
            return;
        }
        (x, y) = (y, r);
    }
    let (mut x, mut y) = (x as u64, y as u64);
    loop {
        let (q, r) = div_rem(x, y);
        out.push(term(order, q));
        if r == 0 {
            return;
        }
        (x, y) = (y, r);
    }
}

/// The quotient and the remainder of `x` by `y`, `x >= y > 0`, by the hardware's division. The
/// three conditional subtractions that hgcd2 takes for quotients below 8 were measured 6 % slower
/// here, in the reconstruction of words on an Apple M4 Max.
#[inline(always)]
fn div_rem(x: u64, y: u64) -> (u64, u64) {
    debug_assert!(x >= y && y > 0);
    let q = x / y;
    (q, x - q * y)
}

/// The number of terms to plan for in a continued fraction of a denominator of `bits` bits: Lévy's
/// constant gives about `0.584 bits` for random fractions.
#[inline]
fn planned_terms(bits: u64) -> usize {
    (bits as usize * 3) / 5 + 2
}

impl Rational {
    /// The regular continued fraction `[a0; a1, …, an]` of the number, in its byte order: `a0 =
    /// ⌊x⌋`, which may be negative or zero, and `ai ≥ 1` for `i ≥ 1`, the last term at least 2 unless
    /// the number is an integer, which is the one term `[x]`. That is the canonical form, the one
    /// expansion of each rational with those properties.
    ///
    /// The terms are the quotients of Euclid's algorithm on the denominator and the numerator's
    /// remainder: in registers when the parts fit in 128 bits, and otherwise on the greatest
    /// common divisor's reductions, which keep the quotients that they find (hgcd2's from the top
    /// two limbs, the half gcd's for long parts). So it takes the time of a greatest common
    /// divisor of the parts, `O(M(n) log n)` for parts of `n` limbs, and that of writing the
    /// terms, about `0.58` a bit of the denominator for random numbers.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::rational::Rational;
    ///
    /// let r = Rational::from(415) / 93u32;
    /// let terms: Vec<String> = r.continued_fraction().iter().map(|t| t.to_string()).collect();
    /// assert_eq!(terms, ["4", "2", "6", "7"]);
    ///
    /// let r = Rational::from(-7) / 3u32;
    /// let terms: Vec<String> = r.continued_fraction().iter().map(|t| t.to_string()).collect();
    /// assert_eq!(terms, ["-3", "1", "2"]);
    /// assert_eq!(Rational::from_continued_fraction(&r.continued_fraction()), r);
    /// ```
    pub fn continued_fraction(&self) -> Vec<Number> {
        let order = self.byte_order();
        if is_one(&self.den) {
            return vec![self.num.clone()];
        }
        let negative = self.num.is_negative();
        if let (Some(n), Some(d)) = (wide(&self.num), wide(&self.den)) {
            let (neg0, a0, r) = split_u128(negative, n, d);
            let mut out = Vec::with_capacity(planned_terms(128 - u64::from(d.leading_zeros())));
            out.push(number(order, neg0, a0));
            push_terms(order, d, r, &mut out);
            return out;
        }
        let a0 = self.num.div_floor(&self.den);
        let r = self.num.rem_euclid(&self.den);
        let mut out = Vec::with_capacity(planned_terms(self.den.bits()));
        let mut a0 = in_order(a0, order);
        trim(&mut a0);
        out.push(a0);
        quotients(&self.den, &r, order, &mut out);
        out
    }

    /// The rational of the continued fraction `[a0; a1, …, an]`: `a0 + 1/(a1 + 1/(… + 1/an))`, in
    /// the byte order of `a0`. `a0` may be any integer and the other terms any positive ones; the
    /// expansion need not be the canonical one (`[1; 1]` is 2).
    ///
    /// The terms are taken as the matrices `(ai, 1; 1, 0)`, whose product is `(pn, pn-1; qn, qn-1)`
    /// with `x = pn / qn` in lowest terms: runs of short terms are multiplied in registers while
    /// the entries fit in 64 bits, and the matrices of the runs by a balanced tree of products, so
    /// that long expansions take `O(M(n) log n)` for a result of `n` limbs.
    ///
    /// # Panics
    ///
    /// When `terms` is empty, or a term after the first is zero or negative.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::Number;
    /// use a::math::rational::Rational;
    ///
    /// let terms = [4, 2, 6, 7].map(Number::from);
    /// assert_eq!(Rational::from_continued_fraction(&terms).to_string(), "415/93");
    /// let terms = [-3, 1, 2].map(Number::from);
    /// assert_eq!(Rational::from_continued_fraction(&terms).to_string(), "-7/3");
    /// ```
    #[track_caller]
    pub fn from_continued_fraction(terms: &[Number]) -> Rational {
        let Some(first) = terms.first() else {
            panic!("a continued fraction without terms");
        };
        let order = first.byte_order();
        let tail = &terms[1..];
        if tail.iter().any(|t| !t.is_positive()) {
            panic!("a term of a continued fraction after the first is not positive");
        }
        if tail.is_empty() {
            return Rational::from_integer(first.clone());
        }
        // The tail's matrix `(P, P'; Q, Q')`, `[a1; …, an] = P / Q`, and `x = a0 + Q / P`.
        let m = tail_matrix(tail);
        let [p, _, q, _] = m;
        if let (Some(a0), Some(pw), Some(qw)) = (wide(first), word(&p), word(&q)) {
            // x = (a0 P + Q) / P, in 128 bits when it fits; Q <= P.
            let (pw, qw) = (u128::from(pw), u128::from(qw));
            if let Some(ap) = a0.checked_mul(pw) {
                let value = if first.is_negative() {
                    Some((ap > qw, ap.abs_diff(qw)))
                } else {
                    ap.checked_add(qw).map(|m| (false, m))
                };
                if let Some((negative, m)) = value {
                    return small_rational(order, negative, m, pw);
                }
            }
        }
        general_value(first, &p, &q, order)
    }

    /// The convergents `p0/q0, p1/q1, …` of the continued fraction of the number, in its byte
    /// order: `pk/qk = [a0; a1, …, ak]`, by `pk = ak·pk-1 + pk-2` and `qk = ak·qk-1 + qk-2`, each in
    /// lowest terms. The last one is the number itself.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::rational::Rational;
    ///
    /// let r = Rational::from(415) / 93u32;
    /// let c: Vec<String> = r.convergents().iter().map(|c| c.to_string()).collect();
    /// assert_eq!(c, ["4", "9/2", "58/13", "415/93"]);
    /// ```
    pub fn convergents(&self) -> Vec<Rational> {
        let order = self.byte_order();
        if let (Some(n), Some(d)) = (word(&self.num), word(&self.den)) {
            return convergents_small(self.num.is_negative(), n, d, order);
        }
        let terms = self.continued_fraction();
        let mut out = Vec::with_capacity(terms.len());
        // (p1, q1) = (p_k-1, q_k-1) and (p2, q2) = (p_k-2, q_k-2), in registers while they fit.
        let (mut p1, mut q1, mut p2, mut q2) = (1i128, 0i128, 0i128, 1i128);
        let mut k = 0;
        while k < terms.len() {
            let t = &terms[k];
            let Some(a) = wide(t).filter(|&a| a >> 126 == 0) else {
                break;
            };
            let a = if t.is_negative() {
                -(a as i128)
            } else {
                a as i128
            };
            let p = a.checked_mul(p1).and_then(|x| x.checked_add(p2));
            let q = a.checked_mul(q1).and_then(|x| x.checked_add(q2));
            let (Some(p), Some(q)) = (p, q) else {
                break;
            };
            out.push(small_rational(order, p < 0, p.unsigned_abs(), q as u128));
            (p2, q2, p1, q1) = (p1, q1, p, q);
            k += 1;
        }
        if k < terms.len() {
            let num = |x: i128| number(order, x < 0, x.unsigned_abs());
            let (mut pa, mut qa, mut pb, mut qb) = (num(p1), num(q1), num(p2), num(q2));
            for t in &terms[k..] {
                let p = t * &pa + &pb;
                let q = t * &qa + &qb;
                out.push(reduced(p.clone(), q.clone(), order));
                (pb, qb, pa, qa) = (pa, qa, p, q);
            }
        }
        out
    }

    /// The closest rational to the number whose denominator is at most `max_denom`, in its byte
    /// order; of two equally close ones the one of the smaller denominator, and of two over 1, at
    /// a half between integers, the even one (as [`round_ties_even`](Rational::round_ties_even)).
    /// The number itself when its denominator is at most `max_denom`.
    ///
    /// It is a best approximation, found on the Stern–Brocot path of the number: the last
    /// convergent `pk/qk` with `qk ≤ max_denom`, or the semiconvergent `(pk-1 + j pk)/(qk-1 + j
    /// qk)` of the largest `j` that keeps the denominator at most `max_denom`, whichever is the
    /// closer, as Python's `Fraction.limit_denominator` chooses. Euclid's algorithm runs on the
    /// parts with the cofactors of the convergents' denominators until the next one would pass
    /// `max_denom`: in registers for parts of 64 bits, by the half gcd to the length the bound
    /// allows for long ones, `O(M(n) log n)` for parts of `n` limbs.
    ///
    /// # Panics
    ///
    /// When `max_denom` is zero or negative.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::Number;
    /// use a::math::rational::Rational;
    ///
    /// // π to 20 digits.
    /// let pi = Rational::new(
    ///     "314159265358979323846".parse().unwrap(),
    ///     "100000000000000000000".parse().unwrap(),
    /// );
    /// assert_eq!(pi.limit_denominator(&Number::from(10)).to_string(), "22/7");
    /// assert_eq!(pi.limit_denominator(&Number::from(1000)).to_string(), "355/113");
    /// assert_eq!((Rational::from(5) / 2u32).limit_denominator(&Number::from(1)), 2);
    /// ```
    #[track_caller]
    pub fn limit_denominator(&self, max_denom: &Number) -> Rational {
        if !max_denom.is_positive() {
            panic!("the largest denominator of an approximation is not positive");
        }
        if self.den <= *max_denom {
            return self.clone();
        }
        let order = self.byte_order();
        if is_one(max_denom) {
            return Rational::from_integer(self.round_ties_even());
        }
        if let (Some(n), Some(d)) = (word(&self.num), word(&self.den)) {
            // `max_denom < d` fits too.
            let dmax = word(max_denom).unwrap_or(u64::MAX);
            return limit_small(self.num.is_negative(), n, d, dmax, order);
        }
        if let (Some(n), Some(d)) = (wide(&self.num), wide(&self.den)) {
            let dmax = wide(max_denom).unwrap_or(u128::MAX);
            return limit_wide(self.num.is_negative(), n, d, dmax, order);
        }
        limit_general(self, max_denom, order)
    }

    /// The simplest rational in the closed interval between `a` and `b`, in either order: the one
    /// of the smallest denominator, and of those the one of the smallest magnitude, in the byte
    /// order of `a`. Only integers can tie on the denominator, so the rational is unique.
    ///
    /// It is found on the Stern–Brocot tree, by the continued fractions of the two ends: with `a ≤
    /// b`, the integer of the smallest magnitude between them when there is one, and otherwise
    /// the common terms `[c0; c1, …, ck-1]` of the two expansions followed by the term `ck` of
    /// the end that is the lower one at that depth (`a` at even depths, `b` at odd ones) when its
    /// expansion ends there, and by that term plus one when it does not. The expansions are
    /// followed together, in registers when the parts fit in 128 bits; otherwise the common terms
    /// are found by the reductions of the longer end, the half gcd's from 1400 limbs and then
    /// hgcd2's, each kept when the other end shares its steps, and the terms where they part one at
    /// a time. Past 600 limbs of common terms left to Lehmer's method, both expansions are computed
    /// in full and compared, so that it takes `O(M(n) log n)` for ends of `n` limbs.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::rational::Rational;
    ///
    /// let a = Rational::from(1) / 3u32;
    /// let b = Rational::from(2) / 3u32;
    /// assert_eq!(Rational::simplest_between(&a, &b).to_string(), "1/2");
    ///
    /// // 3.14159 and 3.1416 are both closer to 355/113 than to any simpler fraction.
    /// let lo = Rational::from(314159) / 100000u32;
    /// let hi = Rational::from(31416) / 10000u32;
    /// assert_eq!(Rational::simplest_between(&hi, &lo).to_string(), "355/113");
    ///
    /// let a = Rational::from(-5) / 2u32;
    /// assert_eq!(Rational::simplest_between(&a, &Rational::from(7)), 0);
    /// ```
    pub fn simplest_between(a: &Rational, b: &Rational) -> Rational {
        let order = a.byte_order();
        let fits = |x: &Rational| wide(&x.num).is_some() && wide(&x.den).is_some();
        if !(fits(a) && fits(b)) {
            return simplest_general(a, b, order);
        }
        let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
        if lo == hi {
            return reduced(lo.num.clone(), lo.den.clone(), order);
        }
        let c = lo.ceil();
        let f = hi.floor();
        if c <= f {
            return smallest_integer(c, f, order);
        }
        if let (Some(ln), Some(ld), Some(hn), Some(hd)) =
            (word(&lo.num), word(&lo.den), word(&hi.num), word(&hi.den))
        {
            return simplest_small(lo.num.is_negative(), ln, ld, hn, hd, order);
        }
        let w = |n: &Number| wide(n).unwrap_or(0);
        simplest_wide(
            lo.num.is_negative(),
            w(&lo.num),
            w(&lo.den),
            w(&hi.num),
            w(&hi.den),
            order,
        )
    }

    /// Rational reconstruction with the bounds `n = d = ⌊√((m - 1) / 2)⌋`:
    /// [`reconstruct_with_bounds`](Rational::reconstruct_with_bounds)`(r, m, n, n)`. As `2 n² < m`,
    /// the rational found, if any, is the only one in lowest terms with `|p| ≤ n`, `0 < q ≤ n` and
    /// `p ≡ r q (mod m)`.
    ///
    /// # Panics
    ///
    /// When `m` is zero or negative.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::Number;
    /// use a::math::rational::Rational;
    ///
    /// // 2/3 modulo 1009 is 2 · 673 = 337 (mod 1009).
    /// let m = Number::from(1009);
    /// let r = Rational::reconstruct(&Number::from(337), &m).unwrap();
    /// assert_eq!(r.to_string(), "2/3");
    /// ```
    #[track_caller]
    pub fn reconstruct(r: &Number, m: &Number) -> Option<Rational> {
        if !m.is_positive() {
            panic!("the modulus of a rational reconstruction is not positive");
        }
        if let (Some(mw), Some(rw)) = (word(m), word(r)) {
            let rw = if r.is_negative() {
                (mw - rw % mw) % mw
            } else {
                rw % mw
            };
            let b = ((mw - 1) / 2).isqrt();
            if b == 0 {
                return None;
            }
            return reconstruct_small(rw, mw, b, b, r.byte_order());
        }
        let b = if let Some(mw) = word(m) {
            Number::from(((mw - 1) / 2).isqrt())
        } else {
            let mut h = m - 1u8;
            h = h.div_floor(&Number::from(2u8));
            h.isqrt()
        };
        Rational::reconstruct_with_bounds(r, m, &b, &b)
    }

    /// Rational reconstruction: the `p/q` in lowest terms with `p ≡ r·q (mod m)`, `|p| ≤ n`, `0 <
    /// q ≤ d` and `gcd(q, m) = 1`, in the byte order of `r`, or `None`.
    ///
    /// When `2·n·d < m` there is at most one such rational, and this finds it, or tells that there
    /// is none: by Wang's method, Euclid's algorithm on `m` and `r mod m`, the remainders `ri ≡
    /// ti·r (mod m)`, stopped at the first remainder `ri ≤ n` (`r mod m` itself counting as the
    /// first); then `p/q = sgn(ti)·ri/|ti|` when `|ti| ≤ d` and `gcd(ri, ti) = 1`, which is
    /// `gcd(q, m) = 1`, and otherwise there is none. When `2·n·d ≥ m` there may be several; a
    /// rational returned then still has all the properties, but `None` does not tell that there
    /// is none, and the one found is not always the one of the smallest denominator.
    ///
    /// The remainders are reduced as the extended greatest common divisor reduces them, in
    /// registers for a modulus of 64 bits, and for long ones by the half gcd to the length of `n`,
    /// then by hgcd2's matrices and divisions to the exact stop: `O(M(n) log n)` for a modulus of
    /// `n` limbs. `n < 0` or `d < 1` allow no rational, and give `None`.
    ///
    /// # Panics
    ///
    /// When `m` is zero or negative.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::Number;
    /// use a::math::rational::Rational;
    ///
    /// let m = Number::from(10007);
    /// // -3/7 modulo 10007: 7 · 8577 = 60039 = -3 (mod 10007).
    /// let r = Number::from(8577);
    /// let found = Rational::reconstruct_with_bounds(&r, &m, &Number::from(10), &Number::from(500));
    /// assert_eq!(found.unwrap().to_string(), "-3/7");
    /// // No rational with |p| ≤ 2 and q ≤ 3.
    /// assert!(Rational::reconstruct_with_bounds(&r, &m, &Number::from(2), &Number::from(3)).is_none());
    /// ```
    #[track_caller]
    pub fn reconstruct_with_bounds(
        r: &Number,
        m: &Number,
        n: &Number,
        d: &Number,
    ) -> Option<Rational> {
        if !m.is_positive() {
            panic!("the modulus of a rational reconstruction is not positive");
        }
        if n.is_negative() || !d.is_positive() {
            return None;
        }
        let order = r.byte_order();
        if let (Some(mw), Some(rw)) = (word(m), word(r)) {
            let rw = if r.is_negative() {
                (mw - rw % mw) % mw
            } else {
                rw % mw
            };
            let nw = word(n).unwrap_or(u64::MAX);
            let dw = word(d).unwrap_or(u64::MAX);
            return reconstruct_small(rw, mw, nw, dw, order);
        }
        if let Some(mw) = wide(m) {
            let x = r.rem_euclid(m);
            let xw = wide(&x).unwrap_or(0);
            let nw = wide(n).unwrap_or(u128::MAX);
            let dw = wide(d).unwrap_or(u128::MAX);
            return reconstruct_wide(xw, mw, nw, dw, order);
        }
        let x = r.rem_euclid(m);
        if x <= *n {
            // The first remainder, of cofactor 1.
            let mut x = in_order(x, order);
            trim(&mut x);
            return Some(Rational::from_reduced(x, one(order)));
        }
        let s = partial(m, &x, Bound::Remainder(n), order);
        let (p, t) = (s.smaller, s.t_smaller);
        if t.abs() > *d || !is_one(&p.gcd(&t)) {
            return None;
        }
        let negative = t.is_negative();
        let mut p = p;
        p.negative = negative && !p.is_zero();
        Some(reduced(p, t.abs(), order))
    }
}

/// [`Rational::convergents`] of `±n/d` for parts of 64 bits: the terms by Euclid's algorithm in
/// registers, and the convergents, whose parts are at most the number's, in 128 bits, none of the
/// terms written.
fn convergents_small(negative: bool, n: u64, d: u64, order: ByteOrder) -> Vec<Rational> {
    let mut out = Vec::with_capacity(planned_terms(64 - u64::from(d.leading_zeros())));
    if d == 1 {
        out.push(small_rational(order, negative, u128::from(n), 1));
        return out;
    }
    let (neg0, a0, r) = split_u128(negative, u128::from(n), u128::from(d));
    let a0 = if neg0 { -(a0 as i128) } else { a0 as i128 };
    // (p1, q1) = (p_k-1, q_k-1) and (p2, q2) = (p_k-2, q_k-2).
    let (mut p1, mut q1, mut p2, mut q2) = (a0, 1i128, 1i128, 0i128);
    out.push(small_rational(order, a0 < 0, a0.unsigned_abs(), 1));
    let (mut x, mut y) = (d, r as u64);
    while y != 0 {
        let (a, r) = div_rem(x, y);
        (x, y) = (y, r);
        let a = i128::from(a);
        let (p, q) = (a * p1 + p2, a * q1 + q2);
        out.push(small_rational(order, p < 0, p.unsigned_abs(), q as u128));
        (p2, q2, p1, q1) = (p1, q1, p, q);
    }
    out
}

/// The matrix `(P, P'; Q, Q')` of the positive terms `tail`, `[a1; …, an] = P / Q`, as numbers in
/// the byte order of the first: runs of terms of 64 bits multiplied in registers while the entries
/// fit, and the runs' matrices by a balanced tree.
fn tail_matrix(tail: &[Number]) -> [Number; 4] {
    let order = tail[0].byte_order();
    let mut leaves: Vec<[Number; 4]> = Vec::new();
    // The run in registers: `(p, p'; q, q')`.
    let (mut p, mut p1, mut q, mut q1) = (1u64, 0u64, 0u64, 1u64);
    let mut open = false;
    let close = |leaves: &mut Vec<[Number; 4]>, m: [u64; 4]| {
        leaves.push(m.map(|x| term(order, x)));
    };
    for t in tail {
        if let Some(a) = word(t) {
            // (p, p'; q, q') (a, 1; 1, 0) = (a p + p', p; a q + q', q).
            let np = u128::from(a) * u128::from(p) + u128::from(p1);
            let nq = u128::from(a) * u128::from(q) + u128::from(q1);
            if (np | nq) >> 64 == 0 {
                (p, p1, q, q1) = (np as u64, p, nq as u64, q);
                open = true;
                continue;
            }
            if open {
                close(&mut leaves, [p, p1, q, q1]);
            }
            (p, p1, q, q1) = (a, 1, 1, 0);
            open = true;
            continue;
        }
        if open {
            close(&mut leaves, [p, p1, q, q1]);
            (p, p1, q, q1) = (1, 0, 0, 1);
            open = false;
        }
        let mut a = in_order(t.clone(), order);
        trim(&mut a);
        leaves.push([a, one(order), one(order), Number::from(0u8)]);
    }
    if open {
        close(&mut leaves, [p, p1, q, q1]);
    }
    product(&mut leaves)
}

/// The product of the matrices in order, by a balanced tree.
fn product(m: &mut [[Number; 4]]) -> [Number; 4] {
    if m.len() == 1 {
        return core::mem::take(&mut m[0]);
    }
    let h = m.len() / 2;
    let (l, r) = m.split_at_mut(h);
    let x = product(l);
    let y = product(r);
    // (a, b; c, d) (e, f; g, h).
    let [a, b, c, d] = &x;
    let [e, f, g, hh] = &y;
    [a * e + b * g, a * f + b * hh, c * e + d * g, c * f + d * hh]
}

/// `a0 + Q / P` as a rational in the byte order `order`.
fn general_value(a0: &Number, p: &Number, q: &Number, order: ByteOrder) -> Rational {
    let num = a0 * p + q;
    reduced(num, p.clone(), order)
}

/// [`Rational::limit_denominator`] for parts of 64 bits, `±n/d`, `1 < max < d`.
fn limit_small(negative: bool, n: u64, d: u64, max: u64, order: ByteOrder) -> Rational {
    let (neg0, a0, r) = split_u128(negative, u128::from(n), u128::from(d));
    let (a0, r) = (a0 as u64, r as u64);
    // Euclid on (L, S) = (d, r), the fractions P/Q of `r/d`: (P_L, Q_L) = (1, 0), (P_S, Q_S) =
    // (0, 1).
    let (mut l, mut s) = (d, r);
    let (mut pl, mut ql, mut ps, mut qs) = (1u64, 0u64, 0u64, 1u64);
    while s != 0 {
        let (a, r) = div_rem(l, s);
        let next = u128::from(ql) + u128::from(a) * u128::from(qs);
        if next > u128::from(max) {
            break;
        }
        let np = pl + a * ps;
        (pl, ql, ps, qs) = (ps, qs, np, next as u64);
        (l, s) = (s, r);
    }
    // The semiconvergent of the largest `j` that keeps the denominator at most `max`, against the
    // convergent: the convergent when `(2 j Q_S + Q_L) S <= Q_S L`.
    let j = (max - ql) / qs;
    let lhs = (2 * u128::from(j) * u128::from(qs) + u128::from(ql)).checked_mul(u128::from(s));
    let convergent = lhs.is_some_and(|x| x <= u128::from(qs) * u128::from(l));
    let (p, q) = if convergent {
        (ps, qs)
    } else {
        (pl + j * ps, ql + j * qs)
    };
    with_floor(neg0, a0, p, q, order)
}

/// `±a0 + p / q` for `0 <= p <= q`, `q >= 1`, `gcd(p, q) = 1`, as a rational.
fn with_floor(neg0: bool, a0: u64, p: u64, q: u64, order: ByteOrder) -> Rational {
    let aq = u128::from(a0) * u128::from(q);
    let (negative, m) = if neg0 {
        (aq > u128::from(p), aq - u128::from(p))
    } else {
        (false, aq + u128::from(p))
    };
    small_rational(order, negative, m, u128::from(q))
}

/// [`Rational::limit_denominator`] for long parts: Euclid's algorithm on the denominator and the
/// numerator's remainder with the cofactors, stopped by [`partial`] at the last one at most
/// `max`.
fn limit_general(x: &Rational, max: &Number, order: ByteOrder) -> Rational {
    let a0 = x.num.div_floor(&x.den);
    let r = x.num.rem_euclid(&x.den);
    let den = &x.den;
    let st = partial(den, &r, Bound::Cofactor(max), order);
    // The fraction `P/Q` of `r/den` of a remainder `R` of cofactor `t`: `R = s den + t r`, `P =
    // -s sgn(t)`, `Q = |t|`.
    let fraction = |rem: &Number, t: &Number| -> (Number, Number) {
        if t.is_zero() {
            return (Number::from(1u8), Number::from(0u8));
        }
        let s = (rem - &(t * &r)).div_exact(den);
        let p = if t.is_negative() { s } else { -s };
        (p, t.abs())
    };
    let (ps, qs) = fraction(&st.smaller, &st.t_smaller);
    let (pl, ql) = fraction(&st.larger, &st.t_larger);
    // The largest `j` that keeps `Q_L + j Q_S <= max`; negative when the larger is past a part of
    // its step whose cofactor is above `max`.
    let j = (max - &ql).div_floor(&qs);
    let lhs = (&(&j + &j) * &qs + &ql) * &st.smaller;
    let rhs = &qs * &st.larger;
    let (p, q) = if lhs <= rhs {
        (ps, qs)
    } else {
        (&pl + &(&j * &ps), &ql + &(&j * &qs))
    };
    let num = &a0 * &q + &p;
    reduced(num, q, order)
}

/// [`Rational::simplest_between`] for ends of 64-bit parts, `lo = ±ln/ld < hi = hn/hd` of the same
/// sign as `lo`, with no integer between them: the common terms by lockstep, the convergents in
/// 128 bits.
fn simplest_small(
    negative: bool,
    ln: u64,
    ld: u64,
    hn: u64,
    hd: u64,
    order: ByteOrder,
) -> Rational {
    // Both have the floor `a0`; the fractional parts `u = un/ud < v = vn/vd`.
    let (neg0, a0, lr) = split_u128(negative, u128::from(ln), u128::from(ld));
    let (_, _, hr) = split_u128(negative, u128::from(hn), u128::from(hd));
    let (a0, lr, hr) = (a0 as u64, lr as u64, hr as u64);
    // At depth 1 the interval is [1/v, 1/u] = [hd/hr, ld/lr]: lower (xn/xd), upper (yn/yd).
    let (mut xn, mut xd, mut yn, mut yd) = (hd, hr, ld, lr);
    // The convergents of the terms after a0: (p, q) and (p', q'), [c1; …] = p/q.
    let (mut p, mut q, mut p1, mut q1) = (1u128, 0u128, 0u128, 1u128);
    let last = loop {
        let (x, xr) = div_rem(xn, xd);
        if xr == 0 {
            break x;
        }
        let (y, yr) = div_rem(yn, yd);
        if y > x {
            break x + 1;
        }
        // The same term: on to [1/(upper - x), 1/(lower - x)].
        (p, p1) = (u128::from(x) * p + p1, p);
        (q, q1) = (u128::from(x) * q + q1, q);
        (xn, xd, yn, yd) = (yd, yr, xd, xr);
    };
    let (pn, qn) = (u128::from(last) * p + p1, u128::from(last) * q + q1);
    // x = a0 + 1 / (pn / qn) = a0 + qn / pn.
    let ap = u128::from(a0) * pn;
    let (neg, m) = if neg0 {
        (ap > qn, ap - qn)
    } else {
        (false, ap + qn)
    };
    small_rational(order, neg, m, pn)
}

/// [`Rational::simplest_between`] for long parts, by the expansions of both ends taken together
/// without ordering them first: the floors `x = q + r/d` by one division each, then the terms of
/// the fractional parts, two on numbers, then the steps they share by [`common_prefix`]
/// (Lehmer's method on both), then on numbers to where they part, where the end whose term is the
/// smaller, or whose expansion ends, is the lower; the result goes through the matrix of the shared
/// steps. Both expansions in full when more than a batch of terms are left to share.
fn simplest_general(a: &Rational, b: &Rational, order: ByteOrder) -> Rational {
    let split = |x: &Rational| {
        let (mut q, mut r) = x.num.div_rem(&x.den);
        if r.is_negative() {
            q -= 1u8;
            r += &x.den;
        }
        (q, r)
    };
    let (qa, ra) = split(a);
    let (qb, rb) = split(b);
    match qa.cmp(&qb) {
        Ordering::Less => {
            let c = if ra.is_zero() { qa } else { qa + 1u8 };
            return smallest_integer(c, qb, order);
        }
        Ordering::Greater => {
            let c = if rb.is_zero() { qb } else { qb + 1u8 };
            return smallest_integer(c, qa, order);
        }
        Ordering::Equal => {}
    }
    if ra.is_zero() || rb.is_zero() {
        // The floor is an end, the lower one, and the only integer.
        return smallest_integer(qa.clone(), qa, order);
    }
    let a0 = qa;
    // Euclid's pairs (d, r) of the fractional parts r/d; the tails at the first depth are d/r.
    let mut terms = vec![a0.clone()];
    if let Some(t) = part(
        a.den.clone(),
        ra.clone(),
        b.den.clone(),
        rb.clone(),
        2,
        &mut terms,
    ) {
        terms.push(t);
        return reduced_cf(&terms, order);
    }
    let c = common_prefix(&a.den, &ra, &b.den, &rb, order);
    if c.exhausted {
        return simplest_full(a, b, order);
    }
    // The next step reduces the larger number of the pairs, the same in both; the tails are that
    // over the other.
    let a_next = c.lo.0 > c.lo.1;
    let ((an, ad), (bn, bd)) = if a_next {
        (c.lo, c.hi)
    } else {
        ((c.lo.1, c.lo.0), (c.hi.1, c.hi.0))
    };
    let mut rest = Vec::new();
    let Some(t) = part(an, ad, bn, bd, 64, &mut rest) else {
        return simplest_full(a, b, order);
    };
    rest.push(t);
    // The rest `P/Q`, put in the pairs' places and taken through the matrix.
    let (p, q) = Rational::from_continued_fraction(&rest).into_parts();
    let (x, y) = if a_next { (p, q) } else { (q, p) };
    let [m00, m01, m10, m11] = &c.m;
    let d = &(m00 * &x) + &(m01 * &y);
    let r = &(m10 * &x) + &(m11 * &y);
    reduced(&(&a0 * &d) + &r, d, order)
}

/// The terms the tails `an/ad` and `bn/bd`, both above 1, share, pushed to `terms`, for up to
/// `steps` of them, and the last term of the simplest rational between them: the floor of the lower
/// tail, the one of the smaller floor or that is an integer, plus one unless it is an integer.
/// `None` when they share all `steps` terms.
fn part(
    mut an: Number,
    mut ad: Number,
    mut bn: Number,
    mut bd: Number,
    steps: usize,
    terms: &mut Vec<Number>,
) -> Option<Number> {
    for _ in 0..steps {
        let (x, xr) = an.div_rem(&ad);
        let (y, yr) = bn.div_rem(&bd);
        match x.cmp(&y) {
            Ordering::Less => return Some(if xr.is_zero() { x } else { x + 1u8 }),
            Ordering::Greater => return Some(if yr.is_zero() { y } else { y + 1u8 }),
            Ordering::Equal if xr.is_zero() || yr.is_zero() => return Some(x),
            Ordering::Equal => terms.push(x),
        }
        (an, ad, bn, bd) = (ad, xr, bd, yr);
    }
    None
}

/// The value of the terms, in the byte order `order`.
fn reduced_cf(terms: &[Number], order: ByteOrder) -> Rational {
    let r = Rational::from_continued_fraction(terms);
    if r.byte_order() == order {
        r
    } else {
        reduced(r.num, r.den, order)
    }
}

/// The integer of the smallest magnitude from `c` to `f`, `c <= f`, as a rational.
fn smallest_integer(c: Number, f: Number, order: ByteOrder) -> Rational {
    let n = if c.is_positive() {
        c
    } else if f.is_negative() {
        f
    } else {
        Number::default()
    };
    let mut n = in_order(n, order);
    trim(&mut n);
    Rational::from_reduced(n, one(order))
}

/// [`Rational::simplest_between`] by both expansions in full: the first term where they differ,
/// or where one of them ends, and the lower end's term there, plus one when it does not end.
fn simplest_full(a: &Rational, b: &Rational, order: ByteOrder) -> Rational {
    let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
    let a = lo.continued_fraction();
    let b = hi.continued_fraction();
    let mut k = 0;
    while a[k] == b[k] && k + 1 < a.len() && k + 1 < b.len() {
        k += 1;
    }
    let lower = if k % 2 == 0 { &a } else { &b };
    let mut out: Vec<Number> = lower[..k].to_vec();
    if k + 1 == lower.len() {
        out.push(lower[k].clone());
    } else {
        out.push(&lower[k] + 1u8);
    }
    let r = Rational::from_continued_fraction(&out);
    reduced(r.num, r.den, order)
}

/// [`Rational::reconstruct_with_bounds`] for a modulus of 64 bits, `0 <= r < m`.
fn reconstruct_small(r: u64, m: u64, n: u64, d: u64, order: ByteOrder) -> Option<Rational> {
    // (r0, r1) = (m, r), cofactors |t0| = 0, |t1| = 1, t1 > 0 and the signs alternating.
    let (mut r0, mut r1) = (m, r);
    let (mut t0, mut t1) = (0u64, 1u64);
    let mut positive = true;
    while r1 > n {
        let (q, r) = div_rem(r0, r1);
        (r0, r1) = (r1, r);
        // |t_i+1| = |t_i-1| + q |t_i|, at most m.
        (t0, t1) = (t1, t0 + q * t1);
        positive = !positive;
    }
    if t1 > d || super::small::gcd(r1, t1) != 1 {
        return None;
    }
    Some(small_rational(
        order,
        !positive && r1 != 0,
        u128::from(r1),
        u128::from(t1),
    ))
}

// Parts of up to 128 bits

/// `x · y` as its high and low halves.
#[inline(always)]
fn mul_wide(x: u128, y: u128) -> (u128, u128) {
    let (x0, x1) = (x as u64 as u128, x >> 64);
    let (y0, y1) = (y as u64 as u128, y >> 64);
    let (p00, p01, p10, p11) = (x0 * y0, x0 * y1, x1 * y0, x1 * y1);
    let mid = (p00 >> 64) + (p01 as u64 as u128) + (p10 as u64 as u128);
    let lo = (p00 as u64 as u128) | (mid << 64);
    let hi = p11 + (p01 >> 64) + (p10 >> 64) + (mid >> 64);
    (hi, lo)
}

/// `x / y` for `u128`s, by the hardware's division when both fit in 64 bits.
#[inline(always)]
fn div_wide(x: u128, y: u128) -> u128 {
    if (x | y) >> 64 == 0 {
        u128::from(x as u64 / y as u64)
    } else {
        x / y
    }
}

/// `±a0 + p / q` for `0 <= p <= q`, `q >= 1`, `gcd(p, q) = 1`, as a rational, in 128 bits when
/// the numerator fits.
fn with_floor_wide(neg0: bool, a0: u128, p: u128, q: u128, order: ByteOrder) -> Rational {
    let (hi, aq) = mul_wide(a0, q);
    if hi == 0 {
        let value = if neg0 {
            Some((aq > p, aq - p))
        } else {
            aq.checked_add(p).map(|m| (false, m))
        };
        if let Some((negative, m)) = value {
            return small_rational(order, negative, m, q);
        }
    }
    let a = number(order, neg0, a0);
    let num = &(&a * &number(order, false, q)) + &number(order, false, p);
    reduced(num, number(order, false, q), order)
}

/// [`Rational::limit_denominator`] for parts of 128 bits, `±n/d`, `1 < max < d`: as
/// [`limit_small`], the products of the comparison in 256 bits.
fn limit_wide(negative: bool, n: u128, d: u128, max: u128, order: ByteOrder) -> Rational {
    let (neg0, a0, r) = split_u128(negative, n, d);
    let (mut l, mut s) = (d, r);
    let (mut pl, mut ql, mut ps, mut qs) = (1u128, 0u128, 0u128, 1u128);
    while s != 0 {
        let a = div_wide(l, s);
        let next = a.checked_mul(qs).and_then(|x| x.checked_add(ql));
        let Some(next) = next.filter(|&x| x <= max) else {
            break;
        };
        let np = pl + a * ps;
        (pl, ql, ps, qs) = (ps, qs, np, next);
        (l, s) = (s, l - a * s);
    }
    // The convergent when `(j Q_S + Q_L) S <= Q_S (L - j S)`, `L > j S`.
    let j = (max - ql) / qs;
    let convergent = mul_wide(j * qs + ql, s) <= mul_wide(qs, l - j * s);
    let (p, q) = if convergent {
        (ps, qs)
    } else {
        (pl + j * ps, ql + j * qs)
    };
    with_floor_wide(neg0, a0, p, q, order)
}

/// [`Rational::simplest_between`] for ends of 128-bit parts: as [`simplest_small`].
fn simplest_wide(
    negative: bool,
    ln: u128,
    ld: u128,
    hn: u128,
    hd: u128,
    order: ByteOrder,
) -> Rational {
    let (neg0, a0, lr) = split_u128(negative, ln, ld);
    let (_, _, hr) = split_u128(negative, hn, hd);
    let (mut xn, mut xd, mut yn, mut yd) = (hd, hr, ld, lr);
    let (mut p, mut q, mut p1, mut q1) = (1u128, 0u128, 0u128, 1u128);
    let last = loop {
        let x = div_wide(xn, xd);
        let xr = xn - x * xd;
        if xr == 0 {
            break x;
        }
        let y = div_wide(yn, yd);
        if y > x {
            break x + 1;
        }
        (p, p1) = (x * p + p1, p);
        (q, q1) = (x * q + q1, q);
        let yr = yn - y * yd;
        (xn, xd, yn, yd) = (yd, yr, xd, xr);
    };
    let (pn, qn) = (last * p + p1, last * q + q1);
    // x = a0 + qn / pn.
    with_floor_wide(neg0, a0, qn, pn, order)
}

/// The greatest common divisor of two `u128`s by the binary method.
fn gcd_u128(mut x: u128, mut y: u128) -> u128 {
    if x == 0 || y == 0 {
        return x | y;
    }
    let shift = (x | y).trailing_zeros();
    x >>= x.trailing_zeros();
    loop {
        y >>= y.trailing_zeros();
        if x > y {
            core::mem::swap(&mut x, &mut y);
        }
        y -= x;
        if y == 0 {
            return x << shift;
        }
    }
}

/// [`Rational::reconstruct_with_bounds`] for a modulus of 128 bits, `0 <= r < m`.
fn reconstruct_wide(r: u128, m: u128, n: u128, d: u128, order: ByteOrder) -> Option<Rational> {
    let (mut r0, mut r1) = (m, r);
    let (mut t0, mut t1) = (0u128, 1u128);
    let mut positive = true;
    while r1 > n {
        let q = div_wide(r0, r1);
        (r0, r1) = (r1, r0 - q * r1);
        (t0, t1) = (t1, t0 + q * t1);
        positive = !positive;
    }
    if t1 > d || gcd_u128(r1, t1) != 1 {
        return None;
    }
    Some(small_rational(order, !positive && r1 != 0, r1, t1))
}
