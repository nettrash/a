//! Magnitudes as native limbs, the least significant first, without zero limbs at the top: the
//! integers that the arithmetic of [`BigFloat`](super::BigFloat) rounds.

use crate::math::numbers::{ByteOrder, Magnitude, Number, nat};
use core::cmp::Ordering;

/// Drops the zero limbs at the top of `v`.
#[inline]
pub(crate) fn trim(v: &mut Vec<u64>) {
    while v.last() == Some(&0) {
        v.pop();
    }
}

/// The number of bits of the magnitude `m`, 0 for zero; `m` without zero limbs at the top.
#[inline]
pub(crate) fn bit_len(m: &[u64]) -> u64 {
    match m.last() {
        None => 0,
        Some(&top) => {
            debug_assert!(top != 0);
            64 * (m.len() as u64 - 1) + u64::from(64 - top.leading_zeros())
        }
    }
}

/// Bit `i` of `m`.
#[inline]
pub(crate) fn bit(m: &[u64], i: u64) -> bool {
    let (k, b) = ((i / 64) as usize, i % 64);
    k < m.len() && m[k] >> b & 1 == 1
}

/// Whether the bits of `m` below bit `i` are all zero.
#[inline]
pub(crate) fn zero_below(m: &[u64], i: u64) -> bool {
    let (k, b) = ((i / 64) as usize, i % 64);
    let k = k.min(m.len());
    m[..k].iter().all(|&x| x == 0) && (k == m.len() || b == 0 || m[k] << (64 - b) == 0)
}

/// The number of zero bits at the bottom of the magnitude `m`, which is not zero.
#[inline]
pub(crate) fn trailing_zeros(m: &[u64]) -> u64 {
    let k = m
        .iter()
        .position(|&x| x != 0)
        .expect("a magnitude that is not zero");
    64 * k as u64 + u64::from(m[k].trailing_zeros())
}

/// `m·2^s`, without zero limbs at the top.
pub(crate) fn shl(m: &[u64], s: u64) -> Vec<u64> {
    if m.is_empty() {
        return Vec::new();
    }
    let (k, b) = ((s / 64) as usize, (s % 64) as u32);
    let mut r = Vec::with_capacity(m.len() + k + 1);
    r.resize(k, 0);
    if b == 0 {
        r.extend_from_slice(m);
    } else {
        let mut carry = 0u64;
        for &x in m {
            r.push(x << b | carry);
            carry = x >> (64 - b);
        }
        r.push(carry);
    }
    trim(&mut r);
    r
}

/// `floor(m·2^-s)`, without zero limbs at the top.
pub(crate) fn shr(m: &[u64], s: u64) -> Vec<u64> {
    let (k, b) = ((s / 64) as usize, (s % 64) as u32);
    if k >= m.len() {
        return Vec::new();
    }
    let m = &m[k..];
    let mut r = Vec::with_capacity(m.len());
    if b == 0 {
        r.extend_from_slice(m);
    } else {
        for i in 0..m.len() {
            let hi = m.get(i + 1).map_or(0, |&x| x << (64 - b));
            r.push(m[i] >> b | hi);
        }
    }
    trim(&mut r);
    r
}

/// The order of the magnitudes `a` and `b`, both without zero limbs at the top.
#[inline]
pub(crate) fn cmp(a: &[u64], b: &[u64]) -> Ordering {
    a.len()
        .cmp(&b.len())
        .then_with(|| a.iter().rev().cmp(b.iter().rev()))
}

/// `a + b`.
pub(crate) fn add(a: &[u64], b: &[u64]) -> Vec<u64> {
    let (a, b) = if a.len() >= b.len() { (a, b) } else { (b, a) };
    let mut r = Vec::with_capacity(a.len() + 1);
    r.extend_from_slice(a);
    if nat::add_assign(&mut r, b) {
        r.push(1);
    }
    r
}

/// `a + 1`.
pub(crate) fn increment(a: &mut Vec<u64>) {
    for x in a.iter_mut() {
        let (s, c) = x.overflowing_add(1);
        *x = s;
        if !c {
            return;
        }
    }
    a.push(1);
}

/// `a - b` for `a >= b`, without zero limbs at the top.
pub(crate) fn sub(a: &[u64], b: &[u64]) -> Vec<u64> {
    debug_assert!(cmp(a, b) != Ordering::Less);
    let mut r = a.to_vec();
    let borrow = nat::sub_assign(&mut r, b);
    debug_assert!(!borrow);
    trim(&mut r);
    r
}

/// `a·b`, without zero limbs at the top.
pub(crate) fn mul(a: &[u64], b: &[u64]) -> Vec<u64> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let mut r = Vec::new();
    let mut scratch = Vec::new();
    if core::ptr::eq(a, b) {
        nat::sqr_into(&mut r, a, &mut scratch);
    } else {
        nat::mul_into(&mut r, a, b, &mut scratch);
    }
    trim(&mut r);
    r
}

/// The magnitude of `n` as limbs.
pub(crate) fn of_number(n: &Number) -> Vec<u64> {
    let m = Magnitude::of(n);
    let mut r = vec![0u64; m.bytes.len().div_ceil(8)];
    for (i, &byte) in m.bytes.iter().enumerate() {
        let i = if m.big_endian {
            m.bytes.len() - 1 - i
        } else {
            i
        };
        r[i / 8] |= u64::from(byte) << (8 * (i % 8));
    }
    trim(&mut r);
    r
}

/// The number of the magnitude `m` and the sign `negative`, in the byte order `order`, in the
/// shortest form; zero is never negative.
pub(crate) fn to_number(m: &[u64], negative: bool, order: ByteOrder) -> Number {
    let mut v: Vec<u8> = m.iter().flat_map(|l| l.to_le_bytes()).collect();
    while v.last() == Some(&0) {
        v.pop();
    }
    if v.is_empty() {
        v.push(0);
    }
    if order == ByteOrder::BigEndian {
        v.reverse();
    }
    let negative = negative && !m.is_empty();
    Number {
        value: v,
        negative,
        byte_order: order,
    }
}
