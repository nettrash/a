//! The integers that orders, exponents and logarithms are counted in: machine words, for the
//! moduli below `2^64`, whose orders are below them too, and numbers for the rest, behind one trait
//! so that each algorithm is written once.

use super::super::{ByteOrder, Number};
use super::ring::{from_limbs, limbs_of};
use super::word;
use core::cmp::Ordering;

/// The limbs of an integer, the least significant first: one in place, or many.
pub(in super::super) enum Limbs {
    /// A word.
    One([u64; 1]),
    /// Any number of limbs.
    Many(Vec<u64>),
}

impl Limbs {
    /// The limbs.
    #[inline(always)]
    pub(in super::super) fn as_slice(&self) -> &[u64] {
        match self {
            Limbs::One(x) => x,
            Limbs::Many(v) => v,
        }
    }
}

/// A non-negative integer: a word or a number.
pub(in super::super) trait Int: Clone + Ord + core::fmt::Debug {
    /// The integer `x`.
    fn small(x: u64) -> Self;

    /// The integer of the limbs `x`, the least significant first, which fit for a word.
    fn of_limbs(x: &[u64]) -> Self;

    /// The integer of the limbs `x` reduced modulo `q`, not zero.
    fn of_limbs_reduced(x: &[u64], q: &Self) -> Self;

    /// The integer as a word, when it fits.
    fn word(&self) -> Option<u64>;

    /// The number of its bits.
    fn bits(&self) -> u64;

    /// Its limbs.
    fn limbs(&self) -> Limbs;

    /// `self + b`, which fits.
    fn add(&self, b: &Self) -> Self;

    /// `self - b`, for `self >= b`.
    fn sub(&self, b: &Self) -> Self;

    /// `self b`, which fits.
    fn mul(&self, b: &Self) -> Self;

    /// `self / b`, for `b` dividing `self`.
    fn div_exact(&self, b: &Self) -> Self;

    /// The remainder of `self` by `b`, not zero.
    fn rem(&self, b: &Self) -> Self;

    /// `self^e`, which fits.
    fn pow(&self, e: u32) -> Self;

    /// `a b mod m`.
    fn mul_mod(&self, b: &Self, m: &Self) -> Self;

    /// The inverse modulo `m >= 1`, or `None`.
    fn inverse(&self, m: &Self) -> Option<Self>;

    /// The greatest common divisor.
    fn gcd(&self, b: &Self) -> Self;

    /// Whether it is zero.
    fn is_zero(&self) -> bool;

    /// Whether it is one.
    fn is_one(&self) -> bool;

    /// The number, in the byte order `order`.
    fn number(&self, order: ByteOrder) -> Number;

    /// `self - b mod m` for `self, b < m`.
    fn sub_mod(&self, b: &Self, m: &Self) -> Self {
        if *self >= *b {
            self.sub(b)
        } else {
            m.sub(&b.sub(self))
        }
    }
}

impl Int for u64 {
    #[inline(always)]
    fn small(x: u64) -> u64 {
        x
    }

    #[inline(always)]
    fn of_limbs(x: &[u64]) -> u64 {
        debug_assert!(x.iter().skip(1).all(|&l| l == 0));
        x.first().copied().unwrap_or(0)
    }

    #[inline]
    fn of_limbs_reduced(x: &[u64], q: &u64) -> u64 {
        x.iter().rev().fold(0u64, |r, &l| {
            (((u128::from(r) << 64) | u128::from(l)) % u128::from(*q)) as u64
        })
    }

    #[inline(always)]
    fn word(&self) -> Option<u64> {
        Some(*self)
    }

    #[inline(always)]
    fn bits(&self) -> u64 {
        u64::from(64 - self.leading_zeros())
    }

    #[inline(always)]
    fn limbs(&self) -> Limbs {
        Limbs::One([*self])
    }

    #[inline(always)]
    fn add(&self, b: &u64) -> u64 {
        self + b
    }

    #[inline(always)]
    fn sub(&self, b: &u64) -> u64 {
        self - b
    }

    #[inline(always)]
    fn mul(&self, b: &u64) -> u64 {
        self * b
    }

    #[inline(always)]
    fn div_exact(&self, b: &u64) -> u64 {
        self / b
    }

    #[inline(always)]
    fn rem(&self, b: &u64) -> u64 {
        self % b
    }

    #[inline(always)]
    fn pow(&self, e: u32) -> u64 {
        u64::pow(*self, e)
    }

    #[inline(always)]
    fn mul_mod(&self, b: &u64, m: &u64) -> u64 {
        word::mul_mod(*self, *b, *m)
    }

    #[inline(always)]
    fn inverse(&self, m: &u64) -> Option<u64> {
        word::inverse(*self, *m)
    }

    #[inline(always)]
    fn gcd(&self, b: &u64) -> u64 {
        word::gcd(*self, *b)
    }

    #[inline(always)]
    fn is_zero(&self) -> bool {
        *self == 0
    }

    #[inline(always)]
    fn is_one(&self) -> bool {
        *self == 1
    }

    #[inline]
    fn number(&self, order: ByteOrder) -> Number {
        Number::new_from_u64(*self, order)
    }
}

impl Int for Number {
    fn small(x: u64) -> Number {
        Number::new_from_u64(x, ByteOrder::LittleEndian)
    }

    fn of_limbs(x: &[u64]) -> Number {
        from_limbs(x, ByteOrder::LittleEndian)
    }

    fn of_limbs_reduced(x: &[u64], q: &Number) -> Number {
        from_limbs(x, ByteOrder::LittleEndian).rem_euclid(q)
    }

    fn word(&self) -> Option<u64> {
        u64::try_from(self).ok()
    }

    fn bits(&self) -> u64 {
        Number::bits(self)
    }

    fn limbs(&self) -> Limbs {
        Limbs::Many(limbs_of(self))
    }

    fn add(&self, b: &Number) -> Number {
        self + b
    }

    fn sub(&self, b: &Number) -> Number {
        self - b
    }

    fn mul(&self, b: &Number) -> Number {
        self * b
    }

    fn div_exact(&self, b: &Number) -> Number {
        self.div_exact(b)
    }

    fn rem(&self, b: &Number) -> Number {
        self.rem_euclid(b)
    }

    fn pow(&self, e: u32) -> Number {
        Number::pow(self, e)
    }

    fn mul_mod(&self, b: &Number, m: &Number) -> Number {
        (self * b).rem_euclid(m)
    }

    fn inverse(&self, m: &Number) -> Option<Number> {
        if *m == 1u8 {
            return Some(Number::new_from_u8(0, ByteOrder::LittleEndian));
        }
        self.mod_inverse(m)
    }

    fn gcd(&self, b: &Number) -> Number {
        Number::gcd(self, b)
    }

    fn is_zero(&self) -> bool {
        Number::is_zero(self)
    }

    fn is_one(&self) -> bool {
        *self == 1u8
    }

    fn number(&self, order: ByteOrder) -> Number {
        in_order(self.clone(), order)
    }
}

/// `x` in the byte order `order`.
pub(in super::super) fn in_order(x: Number, order: ByteOrder) -> Number {
    if x.byte_order == order {
        return x;
    }
    let mut value = x.value;
    value.reverse();
    Number {
        value,
        negative: x.negative,
        byte_order: order,
    }
}

/// The congruence `x = r (mod m)` joined with `x = s (mod n)`, for `r < m` and `s < n`: the
/// solution modulo their least common multiple, or `None` when they contradict each other. By
/// `x = r + m t` with `t = (s - r) / g (m / g)^(-1) mod n / g`, `g = gcd(m, n)`.
pub(in super::super) fn join<I: Int>(r: &I, m: &I, s: &I, n: &I) -> Option<(I, I)> {
    if m.is_one() {
        return Some((s.clone(), n.clone()));
    }
    if n.is_one() {
        return Some((r.clone(), m.clone()));
    }
    // `s - r mod n`.
    let r_n = r.rem(n);
    let d = s.sub_mod(&r_n, n);
    // Coprime moduli, the common case, are told by the inverse alone.
    if let Some(inv) = m.rem(n).inverse(n) {
        let t = d.mul_mod(&inv, n);
        return Some((r.add(&m.mul(&t)), m.mul(n)));
    }
    let g = m.gcd(n);
    if !d.rem(&g).is_zero() {
        return None;
    }
    let n_g = n.div_exact(&g);
    let m_g = m.div_exact(&g).rem(&n_g);
    let t = if n_g.is_one() {
        I::small(0)
    } else {
        let inv = m_g.inverse(&n_g)?;
        d.div_exact(&g).rem(&n_g).mul_mod(&inv, &n_g)
    };
    let l = m.mul(&n_g);
    Some((r.add(&m.mul(&t)), l))
}

/// The prime factorisations `a` and `b`, primes increasing, joined into that of their least common
/// multiple: each prime with the larger exponent.
pub(in super::super) fn lcm_factored<I: Int>(a: &[(I, u32)], b: &[(I, u32)]) -> Vec<(I, u32)> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0, 0);
    while i < a.len() || j < b.len() {
        let order = match (a.get(i), b.get(j)) {
            (Some(x), Some(y)) => x.0.cmp(&y.0),
            (Some(_), None) => Ordering::Less,
            _ => Ordering::Greater,
        };
        match order {
            Ordering::Less => {
                out.push(a[i].clone());
                i += 1;
            }
            Ordering::Greater => {
                out.push(b[j].clone());
                j += 1;
            }
            Ordering::Equal => {
                out.push((a[i].0.clone(), a[i].1.max(b[j].1)));
                i += 1;
                j += 1;
            }
        }
    }
    out
}

/// The product of the prime powers.
pub(in super::super) fn product<I: Int>(f: &[(I, u32)]) -> I {
    match f.len() {
        0 => I::small(1),
        1 => f[0].0.pow(f[0].1),
        n => {
            let (a, b) = f.split_at(n / 2);
            product(a).mul(&product(b))
        }
    }
}
