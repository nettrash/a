//! Discrete logarithms modulo a number, by method: [`bsgs()`], [`rho()`] and [`pohlig_hellman`].
//! [`Number::discrete_log`] chooses among them, and factors the modulus first; these take the
//! modulus as it is, and the order of the base, or a multiple of it, from the caller.
//!
//! Each works in the ring of residues that fits the modulus: Montgomery's form modulo an odd
//! number, in registers up to eight limbs, wrapping products modulo a power of two up to `2^128`,
//! and divisions modulo the others.

pub(super) mod bsgs;
pub(super) mod index;
pub(super) mod ph;
pub(super) mod rho;

use super::modular::int::{Int, in_order};
use super::modular::positive;
use super::modular::ring::{Ring, limbs_of};
use super::{ByteOrder, Factorization, Number};
use bsgs::{Bsgs, MOST_BABY_STEPS};
use ph::{isqrt_up, ph_in};
use rho::{Walk, rho_in};

/// [`with_ring!`] with the odd moduli of eight limbs in registers too ([`Many<8>`]), where the
/// ring on limbs would allocate each product: the walks and the steps of the logarithms take
/// millions of products.
///
/// [`with_ring!`]: super::modular::ring::with_ring
/// [`Many<8>`]: super::modular::ring::Many
macro_rules! with_ring_dlog {
    ($m:expr, |$r:ident| $body:expr) => {
        match $m {
            m => {
                let m: &[u64] = m;
                if m.len() == 8 && m[0] & 1 == 1 {
                    let $r = $crate::math::numbers::modular::ring::Many::<8>::new([
                        m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7],
                    ]);
                    $body
                } else {
                    $crate::math::numbers::modular::ring::with_ring!(m, |$r| $body)
                }
            }
        }
    };
}
pub(super) use with_ring_dlog;

/// [`with_odd!`](super::modular::ring::with_odd) with the moduli of eight limbs in registers, as
/// [`with_ring_dlog!`].
macro_rules! with_odd_dlog {
    ($m:expr, |$r:ident| $body:expr) => {
        match $m {
            m => {
                let m: &[u64] = m;
                if m.len() == 8 {
                    let $r = $crate::math::numbers::modular::ring::Many::<8>::new([
                        m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7],
                    ]);
                    $body
                } else {
                    $crate::math::numbers::modular::ring::with_odd!(m, |$r| $body)
                }
            }
        }
    };
}
pub(super) use with_odd_dlog;

/// The smallest `k` with `0 <= k < order` and `g^k = h (mod m)`, or `None`, by Shanks's
/// baby-step giant-step method, for `g` of order dividing `order` modulo `m`. In the byte order of
/// `h`.
///
/// The search covers every `k` below `order` whatever the order of `g`: the baby steps `g^j`,
/// `j < s = ceil(sqrt(order))` (at most `2^21` of them), go into a hash table, and the giant steps
/// `h g^(-s i)` are looked up in it; the first that is found gives the smallest `k = s i + j`. A
/// `g` that is not prime to `m` has no inverse: its powers are 0 modulo the part `m_1` of `m` made
/// of its primes from `k = t <= log2(m)`, and prime to the rest `m_2`; the `k < t` are tried one by
/// one, and the others searched modulo `m_2`. It takes about `2 sqrt(order)` products and as many
/// look-ups, and `sqrt(order)` words of memory.
///
/// # Panics
///
/// Panics if `m` or `order` is zero or negative.
///
/// # Examples
///
/// ```
/// use a::math::numbers::{ByteOrder, Number, dlog};
///
/// let n = |x| Number::new_from_i64(x, ByteOrder::BigEndian);
/// // 3 generates the units modulo 31, of order 30: 3^17 = 22.
/// assert_eq!(dlog::bsgs(&n(22), &n(3), &n(31), &n(30)), Some(n(17)));
/// // Only the k below the bound are searched.
/// assert_eq!(dlog::bsgs(&n(22), &n(3), &n(31), &n(17)), None);
/// ```
#[track_caller]
pub fn bsgs(h: &Number, g: &Number, m: &Number, order: &Number) -> Option<Number> {
    positive(m);
    positive_order(order);
    let out = h.byte_order;
    let n = saturated(order);
    let m = in_order(m.abs(), ByteOrder::LittleEndian);
    if m == 1u8 {
        return Some(Number::new_from_u8(0, out));
    }
    let h = h.rem_euclid(&m);
    let g = g.rem_euclid(&m);
    let k = if g.gcd(&m) == 1u8 {
        search(&h, &g, &m, n)
    } else {
        search_any(&h, &g, &m, n)
    };
    k.map(|k| Number::new_from_u128(k, out))
}

/// The order as a `u128`, all ones beyond: a search past `2^128` steps is out of reach anyway.
fn saturated(order: &Number) -> u128 {
    super::modular::ring::magnitude_128(order).unwrap_or(u128::MAX)
}

/// The smallest `k < n` with `g^k = h` modulo `m`, for `g` prime to `m`.
fn search(h: &Number, g: &Number, m: &Number, n: u128) -> Option<u128> {
    let ml = limbs_of(m);
    let inv = g.mod_inverse(m)?;
    with_ring_dlog!(&ml, |r| {
        let hr = r.residue(&limbs_of(h));
        let gr = r.residue(&limbs_of(g));
        let ir = r.residue(&limbs_of(&inv));
        let s = isqrt_up(u64::try_from(n).unwrap_or(u64::MAX)).min(MOST_BABY_STEPS);
        Bsgs::new(&r, &gr, &ir, s).log(&hr, n)
    })
}

/// The smallest `k < n` with `g^k = h` modulo `m`, for `g` with a factor in common with `m`.
fn search_any(h: &Number, g: &Number, m: &Number, n: u128) -> Option<u128> {
    // `m_1 = gcd(m, g^b)` for `b` at least every exponent of `m`.
    let b = Number::new_from_u64(m.bits(), ByteOrder::LittleEndian);
    let m1 = g.pow_mod(&b, m).gcd(m);
    let m1 = if m1.is_zero() { m.clone() } else { m1 };
    let m2 = m.div_exact(&m1);
    // The powers below `t`, the first that is 0 modulo `m_1`, one by one.
    let mut y = Number::new_from_u8(1, ByteOrder::LittleEndian).rem_euclid(m);
    let mut t = 0u128;
    loop {
        if t >= n {
            return None;
        }
        if y == *h {
            return Some(t);
        }
        if y.rem_euclid(&m1).is_zero() {
            break;
        }
        y = (&y * g).rem_euclid(m);
        t += 1;
    }
    // From `t` on, `g^k = 0 (mod m_1)`: `h` must be too, and `g^(k-t) = h g^(-t) (mod m_2)`.
    if !h.rem_euclid(&m1).is_zero() {
        return None;
    }
    if m2 == 1u8 {
        return Some(t);
    }
    let g2 = g.rem_euclid(&m2);
    let tn = Number::new_from_u128(t, ByteOrder::LittleEndian);
    let inv = g2.mod_inverse(&m2)?;
    let h2 = (h * &inv.pow_mod(&tn, &m2)).rem_euclid(&m2);
    search(&h2, &g2, &m2, n - t).map(|k| k + t)
}

/// A `k` with `0 <= k < order` and `g^k = h (mod m)`, or `None`, by Pollard's rho method for
/// logarithms, for `g` of prime order `order` modulo `m`. In the byte order of `h`.
///
/// Pollard (1978), with Teske's walk (2001): steps `x -> x M_i` on elements `g^a h^b`, the
/// multiplier `M_i = g^(u_i) h^(v_i)` one of 32 chosen by the element, until the walk comes back
/// to an element it met; `g^a h^b = g^(a') h^(b')` then gives `k = (a' - a) / (b - b') mod order`.
/// The walk recognises the elements it met by the distinguished ones, about one in `2^(bits/4)`,
/// which it keeps, so that eight walks run side by side and share them. About `1.25 sqrt(order)`
/// products and little memory, against the `sqrt(order)` words of [`bsgs()`]. The walk is the same
/// for every run. `None` when `g^order` or `h^order` is
/// not 1, and when `h` is not a power of `g`, which every collision then shows (they all have
/// `b = b'`): after a few of them the answer is settled by [`bsgs()`].
///
/// # Panics
///
/// Panics if `m` or `order` is zero or negative.
///
/// # Examples
///
/// ```
/// use a::math::numbers::{ByteOrder, Number, dlog};
///
/// let n = |x| Number::new_from_u64(x, ByteOrder::BigEndian);
/// // p = 2q + 1 with q prime: the squares modulo p are the group of order q, which 4 generates.
/// let (p, q) = (n(1_477_079_036_243), n(738_539_518_121));
/// let h = n(4).pow_mod(&n(123_456_789), &p);
/// assert_eq!(dlog::rho(&h, &n(4), &p, &q), Some(n(123_456_789)));
/// // -1 is not a square modulo p, which is 3 modulo 4.
/// assert_eq!(dlog::rho(&(&p - 1u8), &n(4), &p, &q), None);
/// ```
#[track_caller]
pub fn rho(h: &Number, g: &Number, m: &Number, order: &Number) -> Option<Number> {
    positive(m);
    positive_order(order);
    let out = h.byte_order;
    let m = in_order(m.abs(), ByteOrder::LittleEndian);
    if m == 1u8 {
        return Some(Number::new_from_u8(0, out));
    }
    let q = in_order(order.abs(), ByteOrder::LittleEndian);
    let ml = limbs_of(&m);
    let hl = limbs_of(&h.rem_euclid(&m));
    let gl = limbs_of(&g.rem_euclid(&m));
    let k = with_ring_dlog!(&ml, |r| {
        let hr = r.residue(&hl);
        let gr = r.residue(&gl);
        match q.word() {
            Some(w) if w < 1 << 63 => rho_word(&r, &hr, &gr, w).map(Number::small),
            _ => rho_long(&r, &hr, &gr, &q),
        }
    });
    k.map(|k| in_order(k, out))
}

/// [`rho()`] in the ring `r`, for an order below `2^63`.
fn rho_word<R: Ring>(r: &R, h: &R::V, g: &R::V, q: u64) -> Option<u64> {
    let ql = [q];
    if !r.is_one(&r.pow(g, &ql)) || !r.is_one(&r.pow(h, &ql)) {
        return None;
    }
    if r.is_one(h) {
        return Some(0);
    }
    if q < 1 << 16 {
        let inv = r.pow_u64(g, q - 1);
        return Bsgs::new(r, g, &inv, isqrt_up(q))
            .log(h, u128::from(q))
            .map(|k| k as u64);
    }
    match rho_in(r, h, g, &q, 4) {
        Walk::Found(k) => (r.pow_u64(g, k) == *h).then_some(k),
        Walk::Degenerate => {
            let inv = r.pow_u64(g, q - 1);
            Bsgs::new(r, g, &inv, isqrt_up(q).min(MOST_BABY_STEPS))
                .log(h, u128::from(q))
                .map(|k| k as u64)
        }
    }
}

/// [`rho()`] in the ring `r`, for a longer order.
fn rho_long<R: Ring>(r: &R, h: &R::V, g: &R::V, q: &Number) -> Option<Number> {
    let ql = limbs_of(q);
    if !r.is_one(&r.pow(g, &ql)) || !r.is_one(&r.pow(h, &ql)) {
        return None;
    }
    if r.is_one(h) {
        return Some(Number::small(0));
    }
    loop {
        if let Walk::Found(k) = rho_in(r, h, g, q, u32::MAX)
            && r.pow(g, &limbs_of(&k)) == *h
        {
            return Some(k);
        }
    }
}

/// The smallest `k` with `0 <= k < order` and `g^k = h (mod m)`, or `None`, by Pohlig and
/// Hellman's reduction to the prime powers of `order`, a multiple of the order of `g` given by its
/// factorisation. In the byte order of `h`.
///
/// Pohlig and Hellman (1978): for each prime power `q^f` of `order`, the logarithm of
/// `h^(order/q^f)` to `g^(order/q^f)`, whose order is `q^f'` with `f' <= f`, is found a digit in
/// base `q` at a time, each a logarithm in the group of order `q` by stepping through its powers
/// for `q < 16`, by baby steps and giant steps up to `2^21` modulo a word and `2^23` modulo longer
/// numbers (one table for all the digits), and by Pollard's rho beyond; the logarithms modulo the
/// `q^f'` are joined by the Chinese remainder theorem, which gives `k` modulo the order of `g`, the
/// smallest, and checked. `None` when `h` is not a power of `g`, and when `g^order` is not 1
/// (`order` is not a multiple of the order of `g`).
///
/// # Panics
///
/// Panics if `m` is zero or negative.
///
/// # Examples
///
/// ```
/// use a::math::numbers::{ByteOrder, Factorization, Number, dlog};
///
/// let n = |x| Number::new_from_u64(x, ByteOrder::BigEndian);
/// // p = 3 * 2^30 + 1 is prime, and 5 generates its units, of order 3 * 2^30.
/// let p = n(3 * (1 << 30) + 1);
/// let order = Factorization::new(vec![(n(2), 30), (n(3), 1)], ByteOrder::BigEndian);
/// let h = n(5).pow_mod(&n(1_000_000_000), &p);
/// assert_eq!(dlog::pohlig_hellman(&h, &n(5), &p, &order), Some(n(1_000_000_000)));
/// ```
#[track_caller]
pub fn pohlig_hellman(h: &Number, g: &Number, m: &Number, order: &Factorization) -> Option<Number> {
    positive(m);
    let out = h.byte_order;
    let m = in_order(m.abs(), ByteOrder::LittleEndian);
    if m == 1u8 {
        return Some(Number::new_from_u8(0, out));
    }
    let ml = limbs_of(&m);
    let hl = limbs_of(&h.rem_euclid(&m));
    let gl = limbs_of(&g.rem_euclid(&m));
    let n: Vec<(Number, u32)> = order
        .factors()
        .iter()
        .map(|(p, e)| (in_order(p.clone(), ByteOrder::LittleEndian), *e))
        .collect();
    let k = with_ring_dlog!(&ml, |r| {
        let hr = r.residue(&hl);
        let gr = r.residue(&gl);
        ph_in(&r, &hr, &gr, &n, None, None).map(|(k, _)| k)
    });
    k.map(|k| in_order(k, out))
}

/// Panics unless the order is positive.
#[inline(always)]
#[track_caller]
fn positive_order(order: &Number) {
    if order.negative || order.is_zero() {
        not_positive_order();
    }
}

/// Panics for an order that is zero or negative.
#[cold]
#[inline(never)]
#[track_caller]
fn not_positive_order() -> ! {
    panic!("the order must be positive")
}
