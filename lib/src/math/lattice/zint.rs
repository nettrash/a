//! The integers of the exact parts of L²: in a machine word while they fit (fplll's choice of
//! `long`), and beyond as a sign and limbs that the updates write over, `y ← y − x·z` computed
//! by the limb kernels into working memory the caller keeps (a word times limbs in one pass of
//! `addmul_1` or `submul_1` over the limbs written), so that the size reductions, which are most
//! of L²'s exact work, allocate nothing once the rows have their lengths.

use crate::math::numbers::nat::{add_assign, addmul_1, mul_to, sub_assign, sub_from};
use crate::math::numbers::{ByteOrder, Magnitude, Number, from_limbs};
use core::cmp::Ordering;

/// An integer: `S` while its magnitude is below `2^63` (never `i64::MIN`), so that a product of
/// two fits an `i128` with room for a sum; `B` beyond, a sign and the limbs of the magnitude,
/// the least significant first, without zeros at the top.
#[derive(Clone, Debug)]
pub(super) enum Z {
    S(i64),
    B(bool, Vec<u64>),
}

impl Default for Z {
    fn default() -> Z {
        Z::S(0)
    }
}

/// The working memory of the products.
#[derive(Default)]
pub(super) struct Work {
    prod: Vec<u64>,
    scratch: Vec<u64>,
}

/// The magnitude of `x` as limbs: in place for `B`, in `buf` for `S`; empty for zero.
#[inline(always)]
fn mag<'a>(x: &'a Z, buf: &'a mut [u64; 1]) -> (bool, &'a [u64]) {
    match x {
        Z::S(v) => {
            buf[0] = v.unsigned_abs();
            (*v < 0, if *v == 0 { &buf[..0] } else { &buf[..] })
        }
        Z::B(neg, d) => (*neg, d),
    }
}

fn trim(d: &mut Vec<u64>) {
    while d.last() == Some(&0) {
        d.pop();
    }
}

/// `r ← r − x·y` over `x.len()` limbs, the borrow out returned.
#[inline]
fn submul_1(r: &mut [u64], x: &[u64], y: u64) -> u64 {
    let mut borrow = 0u64;
    for (ri, &xi) in r.iter_mut().zip(x) {
        let p = u128::from(xi) * u128::from(y) + u128::from(borrow);
        let (t, under) = ri.overflowing_sub(p as u64);
        *ri = t;
        borrow = (p >> 64) as u64 + u64::from(under);
    }
    borrow
}

/// `x` and `y` compared as magnitudes without zeros at the top.
fn cmp_mag(x: &[u64], y: &[u64]) -> Ordering {
    if x.len() != y.len() {
        return x.len().cmp(&y.len());
    }
    for (a, b) in x.iter().rev().zip(y.iter().rev()) {
        match a.cmp(b) {
            Ordering::Equal => {}
            o => return o,
        }
    }
    Ordering::Equal
}

impl Z {
    /// The integer of the sign `neg` and the magnitude `d`, in a word when it fits.
    fn of_limbs(neg: bool, mut d: Vec<u64>) -> Z {
        trim(&mut d);
        match d.len() {
            0 => Z::S(0),
            1 if d[0] < 1 << 63 => Z::S(if neg { -(d[0] as i64) } else { d[0] as i64 }),
            _ => Z::B(neg, d),
        }
    }

    pub(super) fn of_i128(v: i128) -> Z {
        if let Ok(s) = i64::try_from(v)
            && s != i64::MIN
        {
            return Z::S(s);
        }
        let m = v.unsigned_abs();
        Z::of_limbs(v < 0, vec![m as u64, (m >> 64) as u64])
    }

    /// The number `n`.
    pub(super) fn of(n: &Number) -> Z {
        let m = Magnitude::of(n);
        if m.bytes.len() <= 8 {
            let v = m.small() as u64;
            if v < 1 << 63 {
                return Z::S(if m.negative { -(v as i64) } else { v as i64 });
            }
        }
        let mut d = vec![0u64; m.bytes.len().div_ceil(8)];
        for (k, &byte) in m.bytes.iter().enumerate() {
            let i = if m.big_endian {
                m.bytes.len() - 1 - k
            } else {
                k
            };
            d[i / 8] |= (byte as u64) << (8 * (i % 8));
        }
        Z::of_limbs(m.negative, d)
    }

    /// The number, little-endian.
    pub(super) fn number(&self) -> Number {
        match self {
            Z::S(v) => Number::from(*v),
            Z::B(neg, d) => {
                let mut x = from_limbs(d, ByteOrder::LittleEndian);
                x.negative = *neg;
                x
            }
        }
    }

    pub(super) fn is_zero(&self) -> bool {
        matches!(self, Z::S(0))
    }

    /// The length in bits of the magnitude.
    pub(super) fn bits(&self) -> u64 {
        match self {
            Z::S(v) => 64 - v.unsigned_abs().leading_zeros() as u64,
            Z::B(_, d) => 64 * d.len() as u64 - d[d.len() - 1].leading_zeros() as u64,
        }
    }

    pub(super) fn is_negative(&self) -> bool {
        match self {
            Z::S(v) => *v < 0,
            Z::B(neg, _) => *neg,
        }
    }

    /// The order of two integers.
    pub(super) fn cmp(&self, o: &Z) -> Ordering {
        if let (Z::S(a), Z::S(b)) = (self, o) {
            return a.cmp(b);
        }
        let (mut ba, mut bb) = ([0u64], [0u64]);
        let (na, a) = mag(self, &mut ba);
        let (nb, b) = mag(o, &mut bb);
        let (na, nb) = (na && !a.is_empty(), nb && !b.is_empty());
        match (na, nb) {
            (false, true) => Ordering::Greater,
            (true, false) => Ordering::Less,
            (false, false) => cmp_mag(a, b),
            (true, true) => cmp_mag(b, a),
        }
    }

    /// The top bits of the magnitude as an `f64` and the power of two that scales them, as
    /// [`top_of`](super::l2_float) of the number takes them: the eight most significant bytes,
    /// then rounded once to 53 bits.
    pub(super) fn top(&self) -> (f64, i64) {
        match self {
            Z::S(v) => (v.unsigned_abs() as f64, 0),
            Z::B(_, d) => {
                let bytes = (self.bits() as usize).div_ceil(8);
                let shift = 8 * (bytes.saturating_sub(8));
                let (w, s) = (shift / 64, shift % 64);
                let mut x = d[w] >> s;
                if s > 0 && w + 1 < d.len() {
                    x |= d[w + 1] << (64 - s);
                }
                (x as f64, shift as i64)
            }
        }
    }

    /// `self ← self + (±p)`, `p` a magnitude without zeros at the top, its sign `neg_p`.
    fn add_magnitude(&mut self, neg_p: bool, p: &[u64]) {
        if p.is_empty() {
            return;
        }
        let (neg_a, mut d) = match core::mem::take(self) {
            Z::S(0) => {
                *self = Z::of_limbs(neg_p, p.to_vec());
                return;
            }
            Z::S(v) => (v < 0, vec![v.unsigned_abs()]),
            Z::B(neg, d) => (neg, d),
        };
        if neg_a == neg_p {
            if d.len() < p.len() {
                d.resize(p.len(), 0);
            }
            d.push(0);
            add_assign(&mut d, p);
            *self = Z::of_limbs(neg_a, d);
            return;
        }
        match cmp_mag(&d, p) {
            Ordering::Equal => *self = Z::S(0),
            Ordering::Greater => {
                sub_assign(&mut d, p);
                *self = Z::of_limbs(neg_a, d);
            }
            Ordering::Less => {
                d.resize(p.len(), 0);
                sub_from(&mut d, p);
                *self = Z::of_limbs(neg_p, d);
            }
        }
    }

    /// `self ← self + (±v)·o`, the sign of the term `neg_p`, for the magnitude `o` (limbs
    /// without zeros at the top) and the word `v`: one pass of `addmul_1` or `submul_1` over the
    /// limbs, the carry or the borrow taken on above them (a borrow out of the top is a result
    /// of the term's sign, the limbs then negated).
    fn add_scaled(&mut self, neg_p: bool, o: &[u64], v: u64) {
        let (neg_a, mut d) = match core::mem::take(self) {
            Z::S(0) => (neg_p, Vec::with_capacity(o.len() + 1)),
            Z::S(a) => (a < 0, vec![a.unsigned_abs()]),
            Z::B(neg, d) => (neg, d),
        };
        let l = o.len();
        if d.len() < l + 1 {
            d.resize(l + 1, 0);
        }
        if neg_a == neg_p {
            let mut c = addmul_1(&mut d[..l], o, v);
            let mut i = l;
            while c != 0 {
                if i == d.len() {
                    d.push(c);
                    break;
                }
                let (t, over) = d[i].overflowing_add(c);
                d[i] = t;
                c = u64::from(over);
                i += 1;
            }
            *self = Z::of_limbs(neg_a, d);
            return;
        }
        let mut b = submul_1(&mut d[..l], o, v);
        let mut i = l;
        while b != 0 && i < d.len() {
            let (t, under) = d[i].overflowing_sub(b);
            d[i] = t;
            b = u64::from(under);
            i += 1;
        }
        if b != 0 {
            // The term was the larger: its magnitude less the old one, the two's complement of
            // the limbs (the term is below `2^(64·len)` as `len > l`).
            let mut carry = 1u64;
            for x in &mut d {
                let (t, over) = (!*x).overflowing_add(carry);
                *x = t;
                carry = u64::from(over);
            }
            *self = Z::of_limbs(neg_p, d);
        } else {
            *self = Z::of_limbs(neg_a, d);
        }
    }

    /// `self ← self − x·y`, or `self ← self + x·y` when `add` is set.
    #[inline]
    fn fused(&mut self, x: &Z, y: &Z, add: bool, w: &mut Work) {
        // A word times limbs: one pass over the limbs.
        let single = match (x, y) {
            (Z::S(a), Z::B(nb, ob)) => Some((*a, *nb, ob.as_slice())),
            (Z::B(na, oa), Z::S(b)) => Some((*b, *na, oa.as_slice())),
            _ => None,
        };
        if let Some((sv, nb, o)) = single {
            if sv != 0 {
                // The term's sign: the product's, flipped for a subtraction.
                let neg = ((sv < 0) != nb) != !add;
                self.add_scaled(neg, o, sv.unsigned_abs());
            }
            return;
        }
        if let (Z::S(a), Z::S(b)) = (x, y) {
            let p = *a as i128 * *b as i128;
            if let Z::S(s) = self {
                // |s ± a·b| < 2^63 + 2^126: no overflow.
                *self = Z::of_i128(if add { *s as i128 + p } else { *s as i128 - p });
                return;
            }
            if p == 0 {
                return;
            }
            let m = p.unsigned_abs();
            let limbs = [m as u64, (m >> 64) as u64];
            let len = if limbs[1] == 0 { 1 } else { 2 };
            // The term added is p, or −p for a subtraction.
            self.add_magnitude((p < 0) == add, &limbs[..len]);
            return;
        }
        let (mut bx, mut by) = ([0u64], [0u64]);
        let (nx, a) = mag(x, &mut bx);
        let (ny, b) = mag(y, &mut by);
        if a.is_empty() || b.is_empty() {
            return;
        }
        let len = a.len() + b.len();
        if w.prod.len() < len {
            w.prod.resize(len, 0);
        }
        let prod = &mut w.prod[..len];
        mul_to(prod, a, b, &mut w.scratch);
        let top = prod.iter().rposition(|&l| l != 0).map_or(0, |i| i + 1);
        // The product's sign, then flipped for a subtraction.
        let neg = (nx != ny) != !add;
        let p = &w.prod[..top];
        self.add_magnitude(neg, p);
    }

    /// `|self|` against `|o|`.
    pub(super) fn cmp_abs(&self, o: &Z) -> Ordering {
        let (mut ba, mut bb) = ([0u64], [0u64]);
        let (_, a) = mag(self, &mut ba);
        let (_, b) = mag(o, &mut bb);
        cmp_mag(a, b)
    }

    /// `a / b` rounded to the nearest integer, halves away from zero, when the quotient is below
    /// `2^50` in magnitude (`None` otherwise, and for `b = 0`, or where the guess is not
    /// right within four steps): a guess from the top bits of both, then `r = a − t·b` exactly
    /// in `rem` and `t` moved by one while `2|r| > |b|`, the tie `2|r| = |b|` going away from
    /// zero; `two` holds `2r`, both kept by the caller for their memory while they are long.
    pub(super) fn nearest_quotient(
        a: &Z,
        b: &Z,
        rem: &mut Z,
        two: &mut Z,
        w: &mut Work,
    ) -> Option<i64> {
        if let (Z::S(x), Z::S(y)) = (a, b) {
            if *y == 0 {
                return None;
            }
            let (x, y) = (i128::from(*x), i128::from(*y));
            let (q, r) = (x / y, x % y);
            let q = if 2 * r.abs() >= y.abs() {
                if (x < 0) == (y < 0) { q + 1 } else { q - 1 }
            } else {
                q
            };
            return i64::try_from(q).ok().filter(|q| q.unsigned_abs() < 1 << 50);
        }
        if b.is_zero() {
            return None;
        }
        if a.is_zero() {
            return Some(0);
        }
        let (fa, ea) = a.top();
        let (fb, eb) = b.top();
        let e = ea - eb;
        // The tops are in [2^56, 2^64) for long integers: a quotient below 2^50 has e ≤ 58.
        if e > 60 {
            return None;
        }
        let ratio = fa / fb * f64::powi(2.0, e.max(-1100) as i32);
        // Below 2^50 (and so no overflow in the steps of `t` below), or given up.
        if ratio >= (1u64 << 50) as f64 {
            return None;
        }
        let neg = a.is_negative() != b.is_negative();
        let mut t = ratio.round() as i64;
        if neg {
            t = -t;
        }
        // r = a − t·b.
        rem.set_to(a);
        if t != 0 {
            rem.sub_mul(&Z::S(t), b, w);
        }
        for _ in 0..4 {
            two.set_to(rem);
            two.add_mul(&Z::S(1), rem, w);
            match two.cmp_abs(b) {
                Ordering::Less => return Some(t),
                Ordering::Equal => {
                    // a/b = t + r/b with |r/b| = 1/2: away from zero.
                    let up = rem.is_negative() == b.is_negative();
                    let positive = !neg;
                    return Some(match (up, positive) {
                        (true, true) => t + 1,
                        (false, false) => t - 1,
                        _ => t,
                    });
                }
                Ordering::Greater => {
                    if rem.is_negative() == b.is_negative() {
                        t += 1;
                        rem.sub_mul(&Z::S(1), b, w);
                    } else {
                        t -= 1;
                        rem.add_mul(&Z::S(1), b, w);
                    }
                }
            }
        }
        None
    }

    /// `self ← o`, the limbs copied into the memory `self` holds.
    pub(super) fn set_to(&mut self, o: &Z) {
        match (&mut *self, o) {
            (Z::B(n, d), Z::B(on, od)) => {
                *n = *on;
                d.clear();
                d.extend_from_slice(od);
            }
            _ => *self = o.clone(),
        }
    }

    /// `self ← self − x·y`.
    #[inline]
    pub(super) fn sub_mul(&mut self, x: &Z, y: &Z, w: &mut Work) {
        self.fused(x, y, false, w);
    }

    /// `self ← self + x·y`.
    #[inline]
    pub(super) fn add_mul(&mut self, x: &Z, y: &Z, w: &mut Work) {
        self.fused(x, y, true, w);
    }

    /// `2·self − x·y`.
    pub(super) fn twice_minus(&self, x: &Z, y: &Z, w: &mut Work) -> Z {
        let mut t = match self {
            Z::S(v) => Z::of_i128(2 * *v as i128),
            Z::B(..) => {
                let mut t = self.clone();
                t.add_mul(&Z::S(1), self, w);
                t
            }
        };
        t.sub_mul(x, y, w);
        t
    }
}

/// `⟨a, b⟩`: in an `i128` while every element is in a word and the sum does not overflow,
/// accumulated in place beyond.
pub(super) fn dot(a: &[Z], b: &[Z], w: &mut Work) -> Z {
    let mut s = 0i128;
    let mut i = 0;
    while i < a.len() {
        match (&a[i], &b[i]) {
            (Z::S(x), Z::S(y)) => match s.checked_add(*x as i128 * *y as i128) {
                Some(t) => s = t,
                None => break,
            },
            _ => break,
        }
        i += 1;
    }
    if i == a.len() {
        return Z::of_i128(s);
    }
    let mut acc = Z::of_i128(s);
    for (x, y) in a[i..].iter().zip(&b[i..]) {
        acc.add_mul(x, y, w);
    }
    acc
}

/// `dst ← dst − x·src`, element by element.
pub(super) fn sub_mul_row(dst: &mut [Z], x: &Z, src: &[Z], w: &mut Work) {
    for (d, s) in dst.iter_mut().zip(src) {
        if !s.is_zero() {
            d.sub_mul(x, s, w);
        }
    }
}
