//! Continued fractions and rational approximations: [`Rational::continued_fraction`],
//! [`Rational::from_continued_fraction`], [`Rational::convergents`],
//! [`Rational::limit_denominator`], [`Rational::simplest_between`], and rational reconstruction,
//! [`Rational::reconstruct`] and [`Rational::reconstruct_with_bounds`].
//!
//! Each is Euclid's algorithm on the parts. When they fit in 64 or 128 bits it runs in registers,
//! the parts of a rational in words read in place and a result in words written as words, with
//! nothing allocated but the vectors of the terms a continued fraction returns; otherwise on the
//! limbs of the greatest common divisor's own reductions (see `numbers::gcd::euclid`): hgcd2's
//! matrices from the top two limbs, and Möller's half gcd for the long parts, so that a continued
//! fraction, a best approximation and a reconstruction take `O(M(n) log n)` for parts of `n`
//! limbs, as the greatest common divisor does. The simplest rational between two follows both
//! ends' expansions together by hgcd2's matrices.

use super::sequences::limbs::{Limbs, add, mul, of_number, to_number};
use super::small::{Small, number};
use super::{Rational, in_order, is_one, one, trim};
use crate::math::numbers::nat::addmul_1;
use crate::math::numbers::{
    Bound, ByteOrder, Magnitude, Number, common_prefix, partial, quotients,
};
use core::cmp::Ordering;
use std::borrow::Cow;

/// The parts of `r`, sign apart, when they fit in 128 bits.
#[inline(always)]
fn wides(r: &Rational) -> Option<(u128, u128)> {
    let (n, d) = r.magnitudes();
    (n.bytes.len() <= 16 && d.bytes.len() <= 16).then(|| (n.small(), d.small()))
}

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

/// A term of a continued fraction: the magnitude `q` in the byte order `order`, in the shortest
/// form.
#[inline(always)]
fn term(order: ByteOrder, q: u64) -> Number {
    signed_term(order, false, q)
}

/// The integer `±q` as a number in the byte order `order`, in the shortest form, `-0` as 0: in a
/// vector of room for a limb, written by one store of the whole limb and cut to its significant
/// bytes. The allocator gives a vector of 8 bytes the block of one of fewer (16 bytes for the
/// system's allocators), and a store of 8 bytes costs less than a copy of a variable length.
#[inline(always)]
fn signed_term(order: ByteOrder, negative: bool, q: u64) -> Number {
    let len = 8 - (q | 1).leading_zeros() as usize / 8;
    let bytes = match order {
        // The significant bytes first: shifted to the top of the limb.
        ByteOrder::BigEndian => (q << (64 - 8 * len)).to_be_bytes(),
        ByteOrder::LittleEndian => q.to_le_bytes(),
    };
    let mut v = Vec::with_capacity(8);
    v.extend_from_slice(&bytes);
    v.truncate(len);
    Number {
        value: v,
        negative: negative && q != 0,
        byte_order: order,
    }
}

/// The integer `±q` as a number in the byte order `order`, in the shortest form: by
/// [`signed_term`] when it fits in 64 bits, as it nearly always does.
#[inline(always)]
fn wide_term(order: ByteOrder, negative: bool, q: u128) -> Number {
    if q >> 64 == 0 {
        signed_term(order, negative, q as u64)
    } else {
        number(order, negative, q)
    }
}

/// The integer `n` as a rational over 1 in its byte order, read in place when it fits in 64 bits.
#[inline]
fn integer(n: &Number) -> Rational {
    let m = Magnitude::of(n);
    if m.bytes.len() <= 8 {
        return Rational::from_word(m.negative, m.small() as u64, 1, n.byte_order());
    }
    Rational::from_integer(n.clone())
}

/// `r` in the byte order `order`: a copy when it is in it, and otherwise its parts turned around,
/// the words of a short one rewritten with nothing allocated.
fn in_order_of(r: &Rational, order: ByteOrder) -> Rational {
    if r.byte_order() == order {
        return r.clone();
    }
    match r.word() {
        Some(w) => Rational::from_word(w.negative, w.num, w.den, order),
        None => reduced(r.numer().clone(), r.denom().clone(), order),
    }
}

/// The parts of `r` as numbers: borrowed from a rational in numbers, and new ones, not kept, for
/// one in words, whose own would be made and kept for the rest of its life by
/// [`numer`](Rational::numer) and [`denom`](Rational::denom).
fn parts(r: &Rational) -> (Cow<'_, Number>, Cow<'_, Number>) {
    if let Some(w) = r.word() {
        let order = r.byte_order();
        return (
            Cow::Owned(signed_term(order, w.negative, w.num)),
            Cow::Owned(term(order, w.den)),
        );
    }
    // In numbers: `numer` and `denom` borrow them.
    (Cow::Borrowed(r.numer()), Cow::Borrowed(r.denom()))
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

/// The rational `±m / d` from magnitudes, `d` positive, in lowest terms, in the byte order `order`:
/// in words when both fit in 64 bits; `-0` is 0.
#[inline(always)]
fn small_rational(order: ByteOrder, negative: bool, m: u128, d: u128) -> Rational {
    Rational::from_u128(negative && m != 0, m, d, order)
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
        let (q, r) = div_rem_wide(x, y);
        out.push(wide_term(order, false, q));
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
    /// terms, about `0.58` a bit of the denominator for random numbers. Each term is a [`Number`]
    /// with a vector of its own, whose allocation and release take most of the time for parts of
    /// a limb and about half of it from 16 to 256 limbs: measured on an Apple M4 Max, 33 cycles a
    /// term, the same as a bare allocation and release of 8 bytes.
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
        if let Some(w) = self.word() {
            return continued_fraction_small(w.negative, w.num, w.den, order);
        }
        // In numbers from here: `numer` and `denom` borrow them.
        if self.is_integer() {
            return vec![self.numer().clone()];
        }
        let negative = self.is_negative();
        if let Some((n, d)) = wides(self) {
            let (neg0, a0, r) = split_u128(negative, n, d);
            let mut out = Vec::with_capacity(planned_terms(128 - u64::from(d.leading_zeros())));
            out.push(wide_term(order, neg0, a0));
            push_terms(order, d, r, &mut out);
            return out;
        }
        let a0 = self.numer().div_floor(self.denom());
        let r = self.numer().rem_euclid(self.denom());
        let mut out = Vec::with_capacity(planned_terms(self.denom().bits()));
        let mut a0 = in_order(a0, order);
        trim(&mut a0);
        out.push(a0);
        quotients(self.denom(), &r, order, &mut out);
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
        if tail.is_empty() {
            return integer(first);
        }
        // The tail's matrix `(P, P'; Q, Q')`, `[a1; …, an] = P / Q`, and `x = a0 + Q / P`: its first
        // run in registers, which is all of it for short terms.
        let (run, at) = first_run(tail);
        if at == tail.len() {
            let [p, _, q, _] = run;
            return value_small(first, p, q)
                .unwrap_or_else(|| general_value(first, &term(order, p), &term(order, q), order));
        }
        if tail[at..].iter().any(|t| !t.is_positive()) {
            panic!("a term of a continued fraction after the first is not positive");
        }
        let [p, _, q, _] = tail_matrix(tail, run, at);
        if let (Some(pw), Some(qw)) = (word(&p), word(&q))
            && let Some(r) = value_small(first, pw, qw)
        {
            return r;
        }
        general_value(first, &p, &q, order)
    }

    /// The convergents `p0/q0, p1/q1, …` of the continued fraction of the number, in its byte
    /// order: `pk/qk = [a0; a1, …, ak]`, by `pk = ak·pk-1 + pk-2` and `qk = ak·qk-1 + qk-2`, each in
    /// lowest terms. The last one is the number itself.
    ///
    /// For parts of up to 128 bits the terms are found and used in registers, never written, and
    /// the convergents in words are written as words; for longer ones the recurrence runs in
    /// registers while it fits in 127 bits and then on limbs in place, over the vectors of
    /// `pk-2`, so that each convergent costs the two vectors of its own parts.
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
        if let Some(w) = self.word() {
            return convergents_small(w.negative, w.num, w.den, order);
        }
        if let Some((n, d)) = wides(self)
            && let Some(out) = convergents_wide(self.is_negative(), n, d, order)
        {
            return out;
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
        if k == terms.len() {
            return out;
        }
        if k == 0 || (p1 < 0 && p2 > 0) || (p1 > 0 && p2 < 0) {
            // The first term, which may be negative, or numerators of two signs, which only the
            // first two can have: on numbers.
            let num = |x: i128| number(order, x < 0, x.unsigned_abs());
            let (mut pa, mut qa, mut pb, mut qb) = (num(p1), num(q1), num(p2), num(q2));
            for t in &terms[k..] {
                let p = t * &pa + &pb;
                let q = t * &qa + &qb;
                out.push(reduced(p.clone(), q.clone(), order));
                (pb, qb, pa, qa) = (pa, qa, p, q);
            }
            return out;
        }
        // The rest on limbs in place, `pk = ak·pk-1 + pk-2` over the vectors of `pk-2`, the
        // magnitudes adding as the numerators keep one sign: two vectors a convergent, its own.
        let negative = p1 < 0 || p2 < 0;
        let limbs = |x: u128| -> Limbs {
            let mut v = vec![x as u64, (x >> 64) as u64];
            while v.last() == Some(&0) {
                v.pop();
            }
            v
        };
        let (mut pa, mut qa) = (limbs(p1.unsigned_abs()), limbs(q1 as u128));
        let (mut pb, mut qb) = (limbs(p2.unsigned_abs()), limbs(q2 as u128));
        for t in &terms[k..] {
            match word(t) {
                Some(a) => {
                    add_times(&mut pb, &pa, a);
                    add_times(&mut qb, &qa, a);
                }
                None => {
                    let a = of_number(t);
                    add(&mut pb, &mul(&pa, &a));
                    add(&mut qb, &mul(&qa, &a));
                }
            }
            core::mem::swap(&mut pa, &mut pb);
            core::mem::swap(&mut qa, &mut qb);
            out.push(Rational::from_reduced(
                to_number(&pa, negative, order),
                to_number(&qa, false, order),
            ));
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
    /// `max_denom`: in registers for parts of up to 128 bits (a rational in words read in place,
    /// with nothing allocated), by the half gcd to the length the bound allows for long ones,
    /// `O(M(n) log n)` for parts of `n` limbs.
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
        let order = self.byte_order();
        if let Some(w) = self.word() {
            // A bound past 64 bits is above any denominator in words.
            let max = match word(max_denom) {
                Some(max) if max < w.den => max,
                _ => return self.clone(),
            };
            if max == 1 {
                return round_ties_even_small(w.negative, w.num, w.den, order);
            }
            return limit_small(w.negative, w.num, w.den, max, order);
        }
        // In numbers from here: `denom` borrows it.
        if *self.denom() <= *max_denom {
            return self.clone();
        }
        if is_one(max_denom) {
            return Rational::from_integer(self.round_ties_even());
        }
        if let Some((n, d)) = wides(self) {
            let dmax = wide(max_denom).unwrap_or(u128::MAX);
            return limit_wide(self.is_negative(), n, d, dmax, order);
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
        if let (Some(x), Some(y)) = (a.word(), b.word()) {
            return simplest_words(x, y, order);
        }
        let (Some(x), Some(y)) = (wides(a), wides(b)) else {
            return simplest_general(a, b, order);
        };
        let (lo, hi, x, y) = if a <= b { (a, b, x, y) } else { (b, a, y, x) };
        if lo == hi {
            return in_order_of(lo, order);
        }
        // The smallest integer at least `lo` and the largest at most `hi`, signed magnitudes.
        let c = ceil_wide(lo.is_negative(), x.0, x.1);
        let f = floor_wide(hi.is_negative(), y.0, y.1);
        if signed_le(c, f) {
            let (negative, m) = smallest_integer_wide(c, f);
            return small_rational(order, negative, m, 1);
        }
        simplest_wide(lo.is_negative(), x.0, x.1, y.0, y.1, order)
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
            let rw = residue(r.is_negative(), rw, mw);
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
    /// registers for a modulus of up to 128 bits, the result in words when it fits, and for long
    /// ones by the half gcd to the length of `n`,
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
        // The four read once, in place.
        let (mm, rm) = (Magnitude::of(m), Magnitude::of(r));
        let (nm, dm) = (Magnitude::of(n), Magnitude::of(d));
        if mm.negative || mm.bytes.is_empty() {
            panic!("the modulus of a rational reconstruction is not positive");
        }
        if nm.negative || dm.negative || dm.bytes.is_empty() {
            return None;
        }
        let order = r.byte_order();
        if mm.bytes.len() <= 8 && rm.bytes.len() <= 8 {
            let mw = mm.small() as u64;
            let rw = residue(rm.negative, rm.small() as u64, mw);
            // A bound past 64 bits is past any remainder or cofactor too.
            let fit = |x: Magnitude<'_>| {
                if x.bytes.len() <= 8 {
                    x.small() as u64
                } else {
                    u64::MAX
                }
            };
            return reconstruct_small(rw, mw, fit(nm), fit(dm), order);
        }
        if let Some(mw) = wide(m) {
            // `r mod m` in registers when `r` fits too.
            let xw = match wide(r) {
                Some(rw) if r.is_negative() => (mw - rw % mw) % mw,
                Some(rw) => rw % mw,
                None => wide(&r.rem_euclid(m)).unwrap_or(0),
            };
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

/// [`Rational::continued_fraction`] of `±n/d` for parts of 64 bits: the floor and Euclid's
/// algorithm on `d` and the floor's remainder in registers, each term written as it is found.
fn continued_fraction_small(negative: bool, n: u64, d: u64, order: ByteOrder) -> Vec<Number> {
    if d == 1 {
        return vec![signed_term(order, negative, n)];
    }
    // `n/d` is not an integer, so `0 < r < d`; the floor of a negative one is `-(q + 1)`, which
    // fits as `q < 2^63`.
    let (q, r) = div_rem_any(n, d);
    let (a0, mut y) = if negative { (q + 1, d - r) } else { (q, r) };
    let mut out = Vec::with_capacity(planned_terms(64 - u64::from(d.leading_zeros())));
    out.push(signed_term(order, negative, a0));
    let mut x = d;
    loop {
        let (q, r) = div_rem(x, y);
        out.push(term(order, q));
        if r == 0 {
            return out;
        }
        (x, y) = (y, r);
    }
}

/// The quotient and the remainder of `x` by `y > 0`, any `x`.
#[inline(always)]
fn div_rem_any(x: u64, y: u64) -> (u64, u64) {
    let q = x / y;
    (q, x - q * y)
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

/// The first run of the positive terms `tail` in registers: the matrix `(p, p'; q, q')` of the
/// terms before `at`, `[a1; …, a_at] = p / q`, multiplied while the terms and the entries fit in 64
/// bits, and `at`, where a term or an entry would not, or the length of the tail. A term that
/// is not positive in the run panics, as the caller's check of the rest does.
#[inline]
#[track_caller]
fn first_run(tail: &[Number]) -> ([u64; 4], usize) {
    let (mut p, mut p1, mut q, mut q1) = (1u64, 0u64, 0u64, 1u64);
    for (i, t) in tail.iter().enumerate() {
        let m = Magnitude::of(t);
        if m.bytes.len() > 8 {
            return ([p, p1, q, q1], i);
        }
        let a = m.small() as u64;
        if m.negative || a == 0 {
            panic!("a term of a continued fraction after the first is not positive");
        }
        // (p, p'; q, q') (a, 1; 1, 0) = (a p + p', p; a q + q', q).
        let np = u128::from(a) * u128::from(p) + u128::from(p1);
        let nq = u128::from(a) * u128::from(q) + u128::from(q1);
        if (np | nq) >> 64 != 0 {
            return ([p, p1, q, q1], i);
        }
        (p, p1, q, q1) = (np as u64, p, nq as u64, q);
    }
    ([p, p1, q, q1], tail.len())
}

/// `a0 + Q / P` for the first term `a0` and the tail's `P / Q` in 64 bits, `Q <= P` and `gcd(P, Q)
/// = 1`: `(a0 P + Q) / P` in registers when `a0` and the numerator fit in 128 bits, in words when
/// they fit in 64, and otherwise `None`.
#[inline]
fn value_small(first: &Number, p: u64, q: u64) -> Option<Rational> {
    let f = Magnitude::of(first);
    if f.bytes.len() > 16 {
        return None;
    }
    let (pw, qw) = (u128::from(p), u128::from(q));
    let ap = f.small().checked_mul(pw)?;
    let (negative, m) = if f.negative {
        (ap > qw, ap.abs_diff(qw))
    } else {
        (false, ap.checked_add(qw)?)
    };
    Some(small_rational(first.byte_order(), negative, m, pw))
}

/// `x += y·c` in place for magnitudes as limbs, the least significant first, without zero limbs at
/// the top; `c > 0`.
#[inline]
fn add_times(x: &mut Limbs, y: &[u64], c: u64) {
    let n = y.len();
    if n == 0 {
        return;
    }
    if x.len() < n {
        x.resize(n, 0);
    }
    // By the multiplication's kernel, then the carry through the rest of `x`.
    let mut carry = addmul_1(&mut x[..n], y, c);
    let mut i = n;
    while carry != 0 {
        if i == x.len() {
            x.push(carry);
            return;
        }
        let (s, c) = x[i].overflowing_add(carry);
        x[i] = s;
        carry = u64::from(c);
        i += 1;
    }
}

/// [`Rational::convergents`] of `±n/d` for parts of 128 bits, none of them in words: the terms by
/// Euclid's algorithm in registers and the convergents by the recurrence in `i128`s, none of the
/// terms written. `None` when an entry would pass 127 bits, which parts below `2^126` never let
/// happen (`|pk| <= |n| + d`).
fn convergents_wide(negative: bool, n: u128, d: u128, order: ByteOrder) -> Option<Vec<Rational>> {
    let mut out = Vec::with_capacity(planned_terms(128 - u64::from(d.leading_zeros())));
    if d == 1 {
        out.push(small_rational(order, negative, n, 1));
        return Some(out);
    }
    let (neg0, a0, r) = split_u128(negative, n, d);
    // a0 <= 2^127: -2^127 is an `i128`, +2^127 is not.
    let a0 = if neg0 {
        0i128.checked_sub_unsigned(a0)?
    } else {
        i128::try_from(a0).ok()?
    };
    let (mut p1, mut q1, mut p2, mut q2) = (a0, 1i128, 1i128, 0i128);
    out.push(small_rational(order, a0 < 0, a0.unsigned_abs(), 1));
    let (mut x, mut y) = (d, r);
    while y != 0 {
        let (a, r) = div_rem_wide(x, y);
        (x, y) = (y, r);
        let a = i128::try_from(a).ok()?;
        let p = a.checked_mul(p1)?.checked_add(p2)?;
        let q = a.checked_mul(q1)?.checked_add(q2)?;
        out.push(small_rational(order, p < 0, p.unsigned_abs(), q as u128));
        (p2, q2, p1, q1) = (p1, q1, p, q);
    }
    Some(out)
}

/// The matrix `(P, P'; Q, Q')` of the positive terms `tail`, `[a1; …, an] = P / Q`, as numbers in
/// the byte order of the first: runs of terms of 64 bits multiplied in registers while the entries
/// fit, and the runs' matrices by a balanced tree. `run` is the matrix of the terms before `at`,
/// from [`first_run`].
fn tail_matrix(tail: &[Number], run: [u64; 4], at: usize) -> [Number; 4] {
    let order = tail[0].byte_order();
    let mut leaves: Vec<[Number; 4]> = Vec::new();
    // The run in registers: `(p, p'; q, q')`.
    let [mut p, mut p1, mut q, mut q1] = run;
    let mut open = at > 0;
    let close = |leaves: &mut Vec<[Number; 4]>, m: [u64; 4]| {
        leaves.push(m.map(|x| term(order, x)));
    };
    for t in &tail[at..] {
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
    let a0 = x.numer().div_floor(x.denom());
    let r = x.numer().rem_euclid(x.denom());
    let den = x.denom();
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

/// [`Rational::simplest_between`] for ends in words, in registers.
fn simplest_words(a: Small, b: Small, order: ByteOrder) -> Rational {
    let (lo, hi) = if super::small::cmp(a, b) == Ordering::Greater {
        (b, a)
    } else {
        (a, b)
    };
    if (lo.negative, lo.num, lo.den) == (hi.negative, hi.num, hi.den) {
        return Rational::from_word(lo.negative, lo.num, lo.den, order);
    }
    let c = ceil_wide(lo.negative, u128::from(lo.num), u128::from(lo.den));
    let f = floor_wide(hi.negative, u128::from(hi.num), u128::from(hi.den));
    if signed_le(c, f) {
        let (negative, m) = smallest_integer_wide(c, f);
        return small_rational(order, negative, m, 1);
    }
    simplest_small(lo.negative, lo.num, lo.den, hi.num, hi.den, order)
}

/// The floor of `±n / d`, `d >= 1`, as a sign and a magnitude, 0 not negative.
#[inline]
fn floor_wide(negative: bool, n: u128, d: u128) -> (bool, u128) {
    let q = div_wide(n, d);
    if negative {
        // -⌈n / d⌉; `q + 1 <= 2^127` when `d >= 2`, and `d = 1` leaves no remainder.
        (true, q + u128::from(q * d != n))
    } else {
        (false, q)
    }
}

/// The ceiling of `±n / d`, `d >= 1`, as a sign and a magnitude, 0 not negative.
#[inline]
fn ceil_wide(negative: bool, n: u128, d: u128) -> (bool, u128) {
    let q = div_wide(n, d);
    if negative {
        (q != 0, q)
    } else {
        (false, q + u128::from(q * d != n))
    }
}

/// `x <= y` for integers as signs and magnitudes, 0 not negative.
#[inline]
fn signed_le(x: (bool, u128), y: (bool, u128)) -> bool {
    match (x.0, y.0) {
        (false, false) => x.1 <= y.1,
        (true, true) => x.1 >= y.1,
        (negative, _) => negative,
    }
}

/// The integer of the smallest magnitude from `c` to `f`, `c <= f`, signs and magnitudes.
#[inline]
fn smallest_integer_wide(c: (bool, u128), f: (bool, u128)) -> (bool, u128) {
    if !c.0 && c.1 != 0 {
        c
    } else if f.0 {
        f
    } else {
        (false, 0)
    }
}

/// `±n/d` rounded to the nearest integer, a half to the even one, as a rational, for `d >= 2`.
fn round_ties_even_small(negative: bool, n: u64, d: u64, order: ByteOrder) -> Rational {
    let (q, r) = div_rem_any(n, d);
    // `2 r` against `d` without passing 64 bits.
    let up = r > d - r || r == d - r && q & 1 == 1;
    let m = q + u64::from(up);
    Rational::from_word(negative && m != 0, m, 1, order)
}

/// [`Rational::simplest_between`] for long parts, by the expansions of both ends taken together
/// without ordering them first: the floors `x = q + r/d` by one division each, then the terms of
/// the fractional parts, two on numbers, then the steps they share by [`common_prefix`]
/// (Lehmer's method on both), then on numbers to where they part, where the end whose term is the
/// smaller, or whose expansion ends, is the lower; the result goes through the matrix of the shared
/// steps. Both expansions in full when more than a batch of terms are left to share.
fn simplest_general(a: &Rational, b: &Rational, order: ByteOrder) -> Rational {
    // One end may be in words: its parts as numbers here, not kept.
    let (an, ad) = parts(a);
    let (bn, bd) = parts(b);
    let split = |n: &Number, d: &Number| {
        let (mut q, mut r) = n.div_rem(d);
        if r.is_negative() {
            q -= 1u8;
            r += d;
        }
        (q, r)
    };
    let (qa, ra) = split(&an, &ad);
    let (qb, rb) = split(&bn, &bd);
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
        ad.as_ref().clone(),
        ra.clone(),
        bd.as_ref().clone(),
        rb.clone(),
        2,
        &mut terms,
    ) {
        terms.push(t);
        return in_order_of(&Rational::from_continued_fraction(&terms), order);
    }
    let c = common_prefix(&ad, &ra, &bd, &rb, order);
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
    in_order_of(&Rational::from_continued_fraction(&out), order)
}

/// `±r mod m` for `m > 0`, without a division when `r < m`, as residues usually are.
#[inline(always)]
fn residue(negative: bool, r: u64, m: u64) -> u64 {
    let r = if r < m { r } else { r % m };
    if negative && r != 0 { m - r } else { r }
}

/// [`Rational::reconstruct_with_bounds`] for a modulus of 64 bits, `0 <= r < m`.
fn reconstruct_small(r: u64, m: u64, n: u64, d: u64, order: ByteOrder) -> Option<Rational> {
    // (r0, r1) = (m, r), cofactors |t0| = 0, |t1| = 1, t1 > 0 and the signs alternating.
    let (mut r0, mut r1) = (m, r);
    let (mut t0, mut t1) = (0u64, 1u64);
    let mut positive = true;
    while r1 > n {
        let s = r0 - r1;
        let (q, r) = if s < r1 { (1, s) } else { div_rem(r0, r1) };
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

/// The quotient and the remainder of `x` by `y`, `x >= y > 0`, for `u128`s: by the hardware's
/// division when both fit in 64 bits, by at most three subtractions for the quotients up to 3,
/// which are two thirds of Euclid's (41.5 %, 17 % and 9.3 % by Gauss and Kuzmin's law), and by the
/// library's division of `u128`s, which takes some 40 cycles, for the others.
#[inline(always)]
fn div_rem_wide(x: u128, y: u128) -> (u128, u128) {
    debug_assert!(x >= y && y > 0);
    if (x | y) >> 64 == 0 {
        let (q, r) = div_rem(x as u64, y as u64);
        return (u128::from(q), u128::from(r));
    }
    let r = x - y;
    if r < y {
        return (1, r);
    }
    let r = r - y;
    if r < y {
        return (2, r);
    }
    let r = r - y;
    if r < y {
        return (3, r);
    }
    let q = x / y;
    (q, x - q * y)
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
        let (a, r) = div_rem_wide(l, s);
        let next = a.checked_mul(qs).and_then(|x| x.checked_add(ql));
        let Some(next) = next.filter(|&x| x <= max) else {
            break;
        };
        let np = pl + a * ps;
        (pl, ql, ps, qs) = (ps, qs, np, next);
        (l, s) = (s, r);
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
        let (x, xr) = div_rem_wide(xn, xd);
        if xr == 0 {
            break x;
        }
        let (y, yr) = div_rem_wide(yn, yd);
        if y > x {
            break x + 1;
        }
        (p, p1) = (x * p + p1, p);
        (q, q1) = (x * q + q1, q);
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
        let (q, r) = div_rem_wide(r0, r1);
        (r0, r1) = (r1, r);
        (t0, t1) = (t1, t0 + q * t1);
        positive = !positive;
    }
    if t1 > d || gcd_u128(r1, t1) != 1 {
        return None;
    }
    Some(small_rational(order, !positive && r1 != 0, r1, t1))
}
