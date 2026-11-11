//! Magnitudes as vectors of limbs, the least significant first, for the sequences: the
//! conversions from and to [`Number`]s and the linear operations their loops need. The products
//! are the numbers' own ([`times`], and the squares of `numbers::nat`): Karatsuba's and Toom-Cook's methods and the
//! transforms. A magnitude is normalised, with no zero limb at the top; zero is the empty vector.

use crate::math::numbers::{ByteOrder, Number, times};

/// A magnitude: limbs, the least significant first, with no zero limb at the top.
pub(super) type Limbs = Vec<u64>;

/// Drops the zero limbs at the top of `v`.
#[inline]
pub(super) fn normalize(v: &mut Limbs) {
    while v.last() == Some(&0) {
        v.pop();
    }
}

/// The magnitude of `n`, read in its byte order.
pub(super) fn of_number(n: &Number) -> Limbs {
    let bytes = &n.value;
    let mut v: Limbs = Vec::with_capacity(bytes.len().div_ceil(8));
    match n.byte_order {
        ByteOrder::LittleEndian => {
            let (chunks, rest) = bytes.as_chunks::<8>();
            v.extend(chunks.iter().map(|c| u64::from_le_bytes(*c)));
            if !rest.is_empty() {
                let mut b = [0u8; 8];
                b[..rest.len()].copy_from_slice(rest);
                v.push(u64::from_le_bytes(b));
            }
        }
        ByteOrder::BigEndian => {
            let (rest, chunks) = bytes.as_rchunks::<8>();
            v.extend(chunks.iter().rev().map(|c| u64::from_be_bytes(*c)));
            if !rest.is_empty() {
                let mut b = [0u8; 8];
                b[8 - rest.len()..].copy_from_slice(rest);
                v.push(u64::from_be_bytes(b));
            }
        }
    }
    normalize(&mut v);
    v
}

/// The number of the magnitude `v` with the sign `negative`, in the byte order `order`, in the
/// shortest form, in a new vector of exactly its length.
pub(super) fn to_number(v: &[u64], negative: bool, order: ByteOrder) -> Number {
    let len = v.len();
    if len == 0 {
        return Number {
            value: vec![0],
            negative: false,
            byte_order: order,
        };
    }
    let top = v[len - 1];
    let top_bytes = 8 - top.leading_zeros() as usize / 8;
    let n = 8 * (len - 1) + top_bytes;
    let mut value = Vec::with_capacity(n);
    match order {
        ByteOrder::LittleEndian => {
            for &l in &v[..len - 1] {
                value.extend_from_slice(&l.to_le_bytes());
            }
            value.extend_from_slice(&top.to_le_bytes()[..top_bytes]);
        }
        ByteOrder::BigEndian => {
            value.extend_from_slice(&top.to_be_bytes()[8 - top_bytes..]);
            for &l in v[..len - 1].iter().rev() {
                value.extend_from_slice(&l.to_be_bytes());
            }
        }
    }
    Number {
        value,
        negative,
        byte_order: order,
    }
}

/// The number of the magnitude `x < 2^128` with the sign `negative`, in the byte order `order`, in
/// the shortest form, in one new vector of exactly its length.
pub(super) fn number_u128(x: u128, negative: bool, order: ByteOrder) -> Number {
    let len = (16 - (x | 1).leading_zeros() as usize / 8).max(1);
    let value = match order {
        ByteOrder::LittleEndian => x.to_le_bytes()[..len].to_vec(),
        ByteOrder::BigEndian => x.to_be_bytes()[16 - len..].to_vec(),
    };
    Number {
        value,
        negative: negative && x != 0,
        byte_order: order,
    }
}

/// The bits of `v`: 0 for zero.
#[inline]
pub(super) fn bits(v: &[u64]) -> u64 {
    match v.last() {
        None => 0,
        Some(&t) => 64 * (v.len() as u64 - 1) + u64::from(64 - t.leading_zeros()),
    }
}

/// `x y`.
#[inline]
pub(super) fn mul(x: &[u64], y: &[u64]) -> Limbs {
    if x.is_empty() || y.is_empty() {
        return Vec::new();
    }
    if y.len() == 1 {
        let mut r = x.to_vec();
        mul_1(&mut r, y[0]);
        return r;
    }
    if x.len() == 1 {
        let mut r = y.to_vec();
        mul_1(&mut r, x[0]);
        return r;
    }
    let mut r = times(x, y);
    normalize(&mut r);
    r
}

/// `x c` in place.
#[inline]
pub(super) fn mul_1(x: &mut Limbs, c: u64) {
    if c == 0 {
        x.clear();
        return;
    }
    let mut carry = 0u64;
    for l in x.iter_mut() {
        let t = u128::from(*l) * u128::from(c) + u128::from(carry);
        *l = t as u64;
        carry = (t >> 64) as u64;
    }
    if carry != 0 {
        x.push(carry);
    }
}

/// `x + y` in place.
#[inline]
pub(super) fn add(x: &mut Limbs, y: &[u64]) {
    if x.len() < y.len() {
        x.resize(y.len(), 0);
    }
    let mut carry = false;
    let (low, high) = x.split_at_mut(y.len());
    for (a, &b) in low.iter_mut().zip(y) {
        let (s, c1) = a.overflowing_add(b);
        let (s, c2) = s.overflowing_add(u64::from(carry));
        *a = s;
        carry = c1 | c2;
    }
    if carry {
        for a in high.iter_mut() {
            let (s, c) = a.overflowing_add(1);
            *a = s;
            if !c {
                return;
            }
        }
        x.push(1);
    }
}

/// `x - y` in place, for `x >= y`.
#[inline]
pub(super) fn sub(x: &mut Limbs, y: &[u64]) {
    debug_assert!(x.len() >= y.len());
    let mut borrow = false;
    let (low, high) = x.split_at_mut(y.len());
    for (a, &b) in low.iter_mut().zip(y) {
        let (s, b1) = a.overflowing_sub(b);
        let (s, b2) = s.overflowing_sub(u64::from(borrow));
        *a = s;
        borrow = b1 | b2;
    }
    if borrow {
        for a in high.iter_mut() {
            let (s, b) = a.overflowing_sub(1);
            *a = s;
            if !b {
                break;
            }
        }
    }
    normalize(x);
}

/// `floor(x / 2^k)`.
pub(super) fn shr(x: &[u64], k: u64) -> Limbs {
    let words = (k / 64) as usize;
    if words >= x.len() {
        return Vec::new();
    }
    let s = (k % 64) as u32;
    let src = &x[words..];
    let mut r = Vec::with_capacity(src.len());
    if s == 0 {
        r.extend_from_slice(src);
    } else {
        for i in 0..src.len() {
            let hi = if i + 1 < src.len() {
                src[i + 1] << (64 - s)
            } else {
                0
            };
            r.push((src[i] >> s) | hi);
        }
    }
    normalize(&mut r);
    r
}

/// `floor(x / 2^k)` in place.
pub(super) fn shr_assign(x: &mut Limbs, k: u64) {
    let words = (k / 64) as usize;
    if words >= x.len() {
        x.clear();
        return;
    }
    let s = (k % 64) as u32;
    let len = x.len() - words;
    if s == 0 {
        x.copy_within(words.., 0);
    } else {
        for i in 0..len {
            let hi = if words + i + 1 < x.len() {
                x[words + i + 1] << (64 - s)
            } else {
                0
            };
            x[i] = (x[words + i] >> s) | hi;
        }
    }
    x.truncate(len);
    normalize(x);
}

/// `r = floor(x / 2^k)`, in the vector `r`.
pub(super) fn shr_into(r: &mut Limbs, x: &[u64], k: u64) {
    r.clear();
    let words = (k / 64) as usize;
    if words >= x.len() {
        return;
    }
    let s = (k % 64) as u32;
    let src = &x[words..];
    if s == 0 {
        r.extend_from_slice(src);
    } else {
        r.extend((0..src.len()).map(|i| {
            let hi = if i + 1 < src.len() {
                src[i + 1] << (64 - s)
            } else {
                0
            };
            (src[i] >> s) | hi
        }));
    }
    normalize(r);
}

/// `r = x 2^k`, in the vector `r`.
pub(super) fn shl_into(r: &mut Limbs, x: &[u64], k: u64) {
    r.clear();
    if x.is_empty() {
        return;
    }
    let words = usize::try_from(k / 64).expect("capacity overflow");
    let s = (k % 64) as u32;
    r.reserve(words + x.len() + 1);
    r.resize(words, 0);
    if s == 0 {
        r.extend_from_slice(x);
    } else {
        let mut carry = 0u64;
        for &l in x {
            r.push((l << s) | carry);
            carry = l >> (64 - s);
        }
        if carry != 0 {
            r.push(carry);
        }
    }
}

/// `x 2^k`.
pub(super) fn shl(x: &[u64], k: u64) -> Limbs {
    if x.is_empty() {
        return Vec::new();
    }
    let words = usize::try_from(k / 64).expect("capacity overflow");
    let s = (k % 64) as u32;
    let mut r = Vec::with_capacity(words + x.len() + 1);
    r.resize(words, 0);
    if s == 0 {
        r.extend_from_slice(x);
    } else {
        let mut carry = 0u64;
        for &l in x {
            r.push((l << s) | carry);
            carry = l >> (64 - s);
        }
        if carry != 0 {
            r.push(carry);
        }
    }
    r
}

/// `x mod 2^k` in place.
#[inline]
pub(super) fn truncate(x: &mut Limbs, k: u64) {
    let words = (k / 64) as usize;
    let s = (k % 64) as u32;
    if words >= x.len() {
        return;
    }
    if s == 0 {
        x.truncate(words);
    } else {
        x.truncate(words + 1);
        x[words] &= (1u64 << s) - 1;
    }
    normalize(x);
}

/// `x / d` in place for a non-zero `d`, and the remainder.
#[inline]
pub(super) fn div_1(x: &mut Limbs, d: u64) -> u64 {
    let mut r = 0u128;
    let d = u128::from(d);
    for l in x.iter_mut().rev() {
        let t = (r << 64) | u128::from(*l);
        *l = (t / d) as u64;
        r = t % d;
    }
    normalize(x);
    r as u64
}

/// The product of the non-zero limbs `f` by a balanced tree: pairs, then pairs of pairs.
pub(super) fn product(f: &[u64]) -> Limbs {
    match f.len() {
        0 => vec![1],
        1 => vec![f[0]],
        _ if f.len() <= 16 => {
            let mut r = vec![f[0]];
            for &x in &f[1..] {
                mul_1(&mut r, x);
            }
            r
        }
        len => {
            let h = len / 2;
            mul(&product(&f[..h]), &product(&f[h..]))
        }
    }
}

/// `x mod d` for a non-zero `d`.
#[inline]
pub(super) fn rem_1(x: &[u64], d: u64) -> u64 {
    let d = u128::from(d);
    let mut r = 0u128;
    for &l in x.iter().rev() {
        r = ((r << 64) | u128::from(l)) % d;
    }
    r as u64
}

/// The primes from 3 up to `limit`, by the sieve of Eratosthenes over the odd numbers.
pub(super) fn odd_primes(limit: u64) -> Vec<u64> {
    let limit = usize::try_from(limit).expect("capacity overflow");
    // `composite[i]` for `2i + 1`.
    let mut composite = vec![false; limit / 2 + 1];
    let mut out = Vec::new();
    let mut i = 1;
    while 2 * i < limit {
        if !composite[i] {
            let p = 2 * i + 1;
            out.push(p as u64);
            let mut j = p * p / 2;
            while j < composite.len() {
                composite[j] = true;
                j += p;
            }
        }
        i += 1;
    }
    out
}
