//! Products of polynomials over the integers by Kronecker substitution: each polynomial is
//! evaluated at `2^s`, a field of `s` bits wide enough for any coefficient of the product with its
//! sign, the two integers are multiplied by the crate's multiplication (Karatsuba's, Toom's and
//! the number-theoretic transforms by their lengths), and the coefficients are read back from
//! their fields.
//!
//! Signs: a polynomial with a negative leading coefficient is negated first, so that its value
//! at `2^s` is positive, and that value is the sum of the positive coefficients' fields less that
//! of the negative ones', as `|c_i| < 2^(s-1)`. The product's coefficients `c_k`, `|c_k| < 2^(s-1)`,
//! come back from the bottom: a field `v` (plus the borrow from the field below) is `c_k = v` below
//! `2^(s-1)` and `c_k = v - 2^s`, a borrow of 1 from the field above, from it.

use super::pack;
use crate::math::numbers::{ByteOrder, Number, from_limbs, nat};

/// Up to this many coefficients in the shorter operand, a product by the schoolbook method.
const SCHOOL: usize = 1;

/// The value at `2^s` of the polynomial `a`, negated if its leading coefficient is negative, and
/// whether it was.
pub(crate) fn evaluate(a: &[Number], s: usize, buf: &mut Vec<u64>) -> (Vec<u64>, bool) {
    let flip = a.last().is_some_and(Number::is_negative);
    let n = (a.len() * s).div_ceil(64) + 1;
    let mut pos = vec![0u64; n];
    let mut neg: Option<Vec<u64>> = None;
    for (i, c) in a.iter().enumerate() {
        if c.is_zero() {
            continue;
        }
        pack::limbs(c, buf);
        if c.is_negative() != flip {
            pack::put(neg.get_or_insert_with(|| vec![0u64; n]), i * s, buf);
        } else {
            pack::put(&mut pos, i * s, buf);
        }
    }
    if let Some(neg) = neg {
        let borrow = nat::sub_assign(&mut pos, &neg);
        debug_assert!(!borrow, "the leading coefficient dominates");
    }
    while pos.last() == Some(&0) {
        pos.pop();
    }
    (pos, flip)
}

/// The largest bits of a coefficient of `a`.
pub(crate) fn max_bits(a: &[Number]) -> usize {
    a.iter().map(pack::bits).max().unwrap_or(0)
}

/// The bits of a number of `len` (at least 1).
#[inline]
pub(crate) fn bits_of(len: usize) -> usize {
    (usize::BITS - len.leading_zeros()) as usize
}

/// The `len` coefficients of the polynomial whose value at `2^s` is `x`, negated when `negate`.
pub(crate) fn read_back(x: &[u64], s: usize, len: usize, negate: bool) -> Vec<Number> {
    let w = s.div_ceil(64);
    let mut field = vec![0u64; w];
    let mut borrow = false;
    let mut out = Vec::with_capacity(len);
    let bit = |f: &[u64], i: usize| (f[i / 64] >> (i % 64)) & 1 == 1;
    for k in 0..len {
        pack::get(x, k * s, s, &mut field);
        if borrow {
            let carry = pack::increment(&mut field);
            if carry || (!s.is_multiple_of(64) && bit(&field, s)) {
                // v + 1 = 2^s: a coefficient of 0, and the borrow stays.
                out.push(Number::default());
                continue;
            }
        }
        let negative = if bit(&field, s - 1) {
            pack::negate(&mut field, s);
            borrow = true;
            !negate
        } else {
            borrow = false;
            negate
        };
        let mut c = from_limbs(&field, ByteOrder::LittleEndian);
        c.negative = negative && !c.is_zero();
        out.push(c);
    }
    out
}

/// The product of `a` and `b` (the square of `a` when `b` is `None`) by Kronecker substitution,
/// both not empty: `la + lb - 1` coefficients.
pub(crate) fn ks(a: &[Number], b: Option<&[Number]>) -> Vec<Number> {
    let la = a.len();
    let (lb, ba) = (b.map_or(la, <[Number]>::len), max_bits(a));
    let bb = b.map_or(ba, max_bits);
    let len = la + lb - 1;
    if ba == 0 || bb == 0 {
        return vec![Number::default(); len];
    }
    let s = ba + bb + bits_of(la.min(lb)) + 1;
    let mut buf = Vec::new();
    let mut scratch = Vec::new();
    let (xa, fa) = evaluate(a, s, &mut buf);
    let mut p = Vec::new();
    let flip = match b {
        None => {
            nat::sqr_into(&mut p, &xa, &mut scratch);
            false
        }
        Some(b) => {
            let (xb, fb) = evaluate(b, s, &mut buf);
            nat::mul_into(&mut p, &xa, &xb, &mut scratch);
            fa != fb
        }
    };
    read_back(&p, s, len, flip)
}

/// From this many coefficients in the shorter operand, a product of coefficients of at most 112
/// bits whose sums stay below `2^224` goes by the transforms directly (see [`multimodular`]).
const MULTIMODULAR: usize = if cfg!(miri) { 4 } else { 48 };

/// The coefficients as `i128`, each of at most 112 bits.
fn small(a: &[Number]) -> Vec<i128> {
    a.iter()
        .map(|c| {
            // The bytes from the most significant; zeros above the value shift out harmlessly.
            let m = if c.byte_order == ByteOrder::LittleEndian {
                c.value
                    .iter()
                    .rev()
                    .fold(0u128, |t, &b| (t << 8) | u128::from(b))
            } else {
                c.value.iter().fold(0u128, |t, &b| (t << 8) | u128::from(b))
            };
            if c.is_negative() {
                -(m as i128)
            } else {
                m as i128
            }
        })
        .collect()
}

/// The product of `a` and `b` (a square when `b` is `None`) by the number-theoretic transforms of
/// the multiplication applied to the coefficients: each coefficient, of at most 112 bits, cut
/// into digits of at most 28 bits (negated for a negative coefficient), the convolution modulo
/// the fewest of the primes below `2^29` whose product exceeds four times the largest sum, and
/// each coefficient brought back by the Chinese remainder theorem, its residues shifted by those
/// of half the primes' product so that the signed value comes out. `None` when the coefficients
/// are too large for it.
pub(crate) fn multimodular(a: &[Number], b: Option<&[Number]>) -> Option<Vec<Number>> {
    let la = a.len();
    let lb = b.map_or(la, <[Number]>::len);
    let (ba, bb) = (max_bits(a), b.map_or(max_bits(a), max_bits));
    let bound = ba + bb + bits_of(la.min(lb));
    if ba == 0 || bb == 0 || ba.max(bb) > 112 || bound > 224 {
        return None;
    }
    let p = nat::int_poly_plan(
        la,
        if b.is_some() { lb } else { 0 },
        ba.max(bb) as u32,
        bound as u32,
    )?;
    let x = small(a);
    let y = b.map(small).unwrap_or_default();
    let mut out = vec![([0u64; 4], false); la + lb - 1];
    nat::int_poly_mul(&mut out, &x, &y, p);
    Some(
        out.into_iter()
            .map(|(m, neg)| {
                let mut c = from_limbs(&m, ByteOrder::LittleEndian);
                c.negative = neg && !c.is_zero();
                c
            })
            .collect(),
    )
}

/// Up to this many coefficients in the shorter operand, a product of small coefficients by
/// [`words`].
const WORDS: usize = if cfg!(miri) { 4 } else { 12 };

/// The schoolbook product of `a` and `b` (a square when `b` is `None`) in machine integers,
/// when every sum of products fits an `i128`: the coefficients read once, each coefficient of
/// the product a sum of products of `i128`s, one number built for it. `None` for larger
/// coefficients or operands longer than [`WORDS`] times four.
fn words(a: &[Number], b: Option<&[Number]>) -> Option<Vec<Number>> {
    let la = a.len();
    let lb = b.map_or(la, <[Number]>::len);
    if la.max(lb) > 4 * WORDS {
        return None;
    }
    let (ba, bb) = (max_bits(a), b.map_or(max_bits(a), max_bits));
    if ba + bb + bits_of(la.min(lb)) > 126 {
        return if ba.max(bb) <= 64 { words3(a, b) } else { None };
    }
    let x = small(a);
    let y = b.map(small);
    let y = y.as_deref().unwrap_or(&x);
    let mut c = vec![0i128; la + lb - 1];
    for (i, &u) in x.iter().enumerate() {
        if u == 0 {
            continue;
        }
        for (o, &v) in c[i..].iter_mut().zip(y) {
            *o += u * v;
        }
    }
    Some(
        c.into_iter()
            .map(|v| {
                let m = v.unsigned_abs();
                let mut n = from_limbs(&[m as u64, (m >> 64) as u64], ByteOrder::LittleEndian);
                n.negative = v < 0;
                n
            })
            .collect(),
    )
}

/// Up to this many coefficients in the shorter operand, a product of large coefficients by the
/// schoolbook method on limbs ([`limbs_school`]), below Kronecker substitution's packing and
/// unpacking.
const LIMBS: usize = if cfg!(miri) { 3 } else { 4 };

/// Up to this many bits in a coefficient, [`LIMBS`] applies: measured on an Apple M4, 0.73 to
/// 0.9 of Kronecker substitution's time at 1,000 bits, 1.2 to 1.7 at 10,000, where one product
/// of the packed values by the integers' transforms beats 16 of the coefficients.
const LIMBS_BITS: usize = 2048;

/// The schoolbook product of `a` and `b` (a square when `b` is `None`) on limbs: each
/// coefficient's magnitude read once, each coefficient of the product the sum of its positive
/// products less that of its negative ones, in vectors of limbs reused from one coefficient to
/// the next, one number built for it.
fn limbs_school(a: &[Number], b: Option<&[Number]>) -> Vec<Number> {
    let read = |c: &[Number]| -> (Vec<Vec<u64>>, Vec<bool>) {
        let mut buf = Vec::new();
        let m = c
            .iter()
            .map(|x| {
                pack::limbs(x, &mut buf);
                buf.clone()
            })
            .collect();
        (m, c.iter().map(Number::is_negative).collect())
    };
    let (xa, sa) = read(a);
    let yb = b.map(read);
    let (xb, sb) = yb.as_ref().map_or((&xa, &sa), |(m, s)| (m, s));
    let (la, lb) = (xa.len(), xb.len());
    let wa = xa.iter().map(Vec::len).max().unwrap_or(0);
    let wb = xb.iter().map(Vec::len).max().unwrap_or(0);
    let w = wa + wb + 1;
    let (mut pos, mut neg, mut t) = (vec![0u64; w], vec![0u64; w], vec![0u64; wa + wb]);
    let mut scratch = Vec::new();
    let mut out = Vec::with_capacity(la + lb - 1);
    for k in 0..la + lb - 1 {
        pos.fill(0);
        neg.fill(0);
        let (mut any_pos, mut any_neg) = (false, false);
        for i in k.saturating_sub(lb - 1)..=k.min(la - 1) {
            let (x, y) = (&xa[i], &xb[k - i]);
            if x.is_empty() || y.is_empty() {
                continue;
            }
            let t = &mut t[..x.len() + y.len()];
            nat::mul_to(t, x, y, &mut scratch);
            if sa[i] != sb[k - i] {
                nat::add_assign(&mut neg, t);
                any_neg = true;
            } else {
                nat::add_assign(&mut pos, t);
                any_pos = true;
            }
        }
        let negative = if !any_neg {
            false
        } else if !any_pos {
            true
        } else if pos.iter().rev().cmp(neg.iter().rev()) == core::cmp::Ordering::Less {
            nat::sub_from(&mut pos, &neg);
            true
        } else {
            nat::sub_assign(&mut pos, &neg);
            false
        };
        let m = if any_pos || !any_neg { &pos } else { &neg };
        let mut c = from_limbs(m, ByteOrder::LittleEndian);
        c.negative = negative && !c.is_zero();
        out.push(c);
    }
    out
}

/// [`words`] for coefficients of up to 64 bits whose sums need more than an `i128`: the sums of
/// the positive and of the negative products apart, each in three words.
fn words3(a: &[Number], b: Option<&[Number]>) -> Option<Vec<Number>> {
    let la = a.len();
    let lb = b.map_or(la, <[Number]>::len);
    let x = small(a);
    let y = b.map(small);
    let y = y.as_deref().unwrap_or(&x);
    // (low 128 bits, high word) of the positive and the negative sums.
    let mut pos = vec![(0u128, 0u64); la + lb - 1];
    let mut neg = vec![(0u128, 0u64); la + lb - 1];
    for (i, &u) in x.iter().enumerate() {
        if u == 0 {
            continue;
        }
        let um = u.unsigned_abs();
        for (j, &v) in y.iter().enumerate() {
            let t = um * v.unsigned_abs();
            let s = if (u < 0) != (v < 0) {
                &mut neg[i + j]
            } else {
                &mut pos[i + j]
            };
            let (lo, c) = s.0.overflowing_add(t);
            *s = (lo, s.1 + u64::from(c));
        }
    }
    Some(
        pos.into_iter()
            .zip(neg)
            .map(|((pl, ph), (nl, nh))| {
                let negative = (ph, pl) < (nh, nl);
                let ((al, ah), (bl, bh)) = if negative {
                    ((nl, nh), (pl, ph))
                } else {
                    ((pl, ph), (nl, nh))
                };
                let (lo, b) = al.overflowing_sub(bl);
                let hi = ah - bh - u64::from(b);
                let mut n =
                    from_limbs(&[lo as u64, (lo >> 64) as u64, hi], ByteOrder::LittleEndian);
                n.negative = negative && !n.is_zero();
                n
            })
            .collect(),
    )
}

/// The product of `a` and `b`, both not empty, by the schoolbook method for a constant, the
/// transforms for long operands of small coefficients and Kronecker substitution otherwise.
pub(crate) fn mul(a: &[Number], b: &[Number]) -> Vec<Number> {
    let short = a.len().min(b.len());
    if short <= SCHOOL {
        return crate::math::poly::coeff::sealed::schoolbook(a, b);
    }
    if short <= WORDS
        && let Some(c) = words(a, Some(b))
    {
        return c;
    }
    if short <= LIMBS && max_bits(a).max(max_bits(b)) <= LIMBS_BITS {
        return limbs_school(a, Some(b));
    }
    if short >= MULTIMODULAR
        && let Some(c) = multimodular(a, Some(b))
    {
        return c;
    }
    ks(a, Some(b))
}

/// The square of `a`, not empty.
pub(crate) fn sqr(a: &[Number]) -> Vec<Number> {
    if a.len() <= SCHOOL {
        return crate::math::poly::coeff::sealed::schoolbook(a, a);
    }
    if a.len() <= WORDS
        && let Some(c) = words(a, None)
    {
        return c;
    }
    if a.len() <= LIMBS && max_bits(a) <= LIMBS_BITS {
        return limbs_school(a, None);
    }
    if a.len() >= MULTIMODULAR
        && let Some(c) = multimodular(a, None)
    {
        return c;
    }
    ks(a, None)
}

/// The value of `a` at `2^s`, a signed integer, for fields `s` wider than every coefficient
/// with its sign.
pub(crate) fn value(a: &[Number], s: usize) -> Number {
    let mut buf = Vec::new();
    let (x, flip) = evaluate(a, s, &mut buf);
    let mut v = from_limbs(&x, ByteOrder::LittleEndian);
    v.negative = flip && !v.is_zero();
    v
}

/// The `len` coefficients whose value at `2^s` is the signed `x`, each below `2^(s-1)` in
/// magnitude.
pub(crate) fn digits(x: &Number, s: usize, len: usize) -> Vec<Number> {
    let mut buf = Vec::new();
    pack::limbs(x, &mut buf);
    read_back(&buf, s, len, x.is_negative())
}
