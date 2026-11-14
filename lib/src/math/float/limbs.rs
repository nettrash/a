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
#[allow(dead_code, reason = "the interface of the other modules of floats")]
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
#[allow(dead_code, reason = "the interface of the other modules of floats")]
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

/// Bits `pos..pos + 64` of the magnitude `m` as one limb, the bits outside `m` (below bit 0 for
/// a negative `pos`, or above its top) zero.
#[inline]
pub(crate) fn extract(m: &[u64], pos: i128) -> u64 {
    if pos <= -64 || pos >= 64 * m.len() as i128 {
        return 0;
    }
    let k = pos.div_euclid(64);
    let b = pos.rem_euclid(64) as u32;
    let at = |i: i128| -> u64 {
        if (0..m.len() as i128).contains(&i) {
            m[i as usize]
        } else {
            0
        }
    };
    if b == 0 {
        at(k)
    } else {
        at(k) >> b | at(k + 1) << (64 - b)
    }
}

/// Whether the bits of `m` below bit `pos` hold a one; none below 0.
#[inline]
pub(crate) fn any_below(m: &[u64], pos: i128) -> bool {
    if pos <= 0 {
        return false;
    }
    let p = pos.min(64 * m.len() as i128) as u64;
    !zero_below(m, p)
}

/// The order of the mantissas `0.a` and `0.b`, aligned at their top bits, of any lengths.
#[inline]
pub(crate) fn cmp_top(a: &[u64], b: &[u64]) -> Ordering {
    let (na, nb) = (a.len(), b.len());
    let k = na.min(nb);
    for i in 1..=k {
        let (x, y) = (a[na - i], b[nb - i]);
        if x != y {
            return x.cmp(&y);
        }
    }
    // The longer one is larger unless its other limbs are zero.
    if na > nb {
        if a[..na - k].iter().any(|&x| x != 0) {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    } else if nb > na {
        if b[..nb - k].iter().any(|&x| x != 0) {
            Ordering::Less
        } else {
            Ordering::Equal
        }
    } else {
        Ordering::Equal
    }
}

/// Sets `dst` to the top `dst.len()` limbs of the magnitude `0.a·2^-c` for the mantissa `a`,
/// whose top limb's top bit is set, and `c` in `0..64`: limb `i` of `dst` is bits `64(a.len() -
/// dst.len() + i) + c..` of `a` as an integer, zero below its lowest bit. Returns whether any bit
/// of `a` lies below those taken.
#[inline]
pub(crate) fn top_limbs(a: &[u64], c: u32, dst: &mut [u64]) -> bool {
    let (la, k) = (a.len() as i128, dst.len() as i128);
    let p0 = 64 * (la - k) + i128::from(c);
    if c == 0 && la >= k {
        let s = (la - k) as usize;
        dst.copy_from_slice(&a[s..]);
        return a[..s].iter().any(|&x| x != 0);
    }
    for (i, d) in dst.iter_mut().enumerate() {
        *d = extract(a, p0 + 64 * i as i128);
    }
    any_below(a, p0)
}

/// The number of zero limbs at the bottom of a mantissa, which is not zero.
#[inline]
pub(crate) fn low_zeros(m: &[u64]) -> usize {
    m.iter().position(|&x| x != 0).unwrap_or(0)
}

/// Working memory of the arithmetic: vectors that each thread keeps between operations, so that
/// an operation allocates nothing but its result. An operation that panics drops them; the next
/// takes new ones.
#[derive(Default)]
pub(crate) struct Work {
    pub(crate) a: Vec<u64>,
    pub(crate) b: Vec<u64>,
    pub(crate) c: Vec<u64>,
    pub(crate) scratch: Vec<u64>,
}

std::thread_local! {
    /// The thread's working memory.
    static WORK: core::cell::Cell<Option<Box<Work>>> = const { core::cell::Cell::new(None) };
    /// A vector kept for the result of an assigning operator, which hands the old mantissa back.
    static SPARE: core::cell::Cell<Vec<u64>> = const { core::cell::Cell::new(Vec::new()) };
}

/// The longest vectors, in limbs, that a thread keeps between operations: 1 MiB each.
const KEEP: usize = 1 << 17;

/// Runs `f` with the thread's working memory: a nested call gets memory of its own.
#[inline]
pub(crate) fn with_work<R>(f: impl FnOnce(&mut Work) -> R) -> R {
    let mut w = WORK.take().unwrap_or_default();
    let r = f(&mut w);
    for v in [&mut w.a, &mut w.b, &mut w.c, &mut w.scratch] {
        if v.capacity() > KEEP {
            *v = Vec::new();
        }
    }
    WORK.set(Some(w));
    r
}

/// The thread's spare vector, empty, for a result.
#[inline]
pub(crate) fn take_spare() -> Vec<u64> {
    SPARE.take()
}

/// Gives a vector back to the thread as its spare, unless it is too long to keep.
#[inline]
pub(crate) fn give_spare(v: Vec<u64>) {
    if v.capacity() <= KEEP {
        SPARE.set(v);
    }
}
