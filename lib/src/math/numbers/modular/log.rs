//! The discrete logarithm modulo any number, by its prime powers.

use super::super::dlog::ph::{PrimePower, ph_in};
use super::int::{Int, join, product};
use super::order::factor_below;
use super::ring::{Low, Low2, Plain, Ring, with_odd, with_ring};

/// What the congruence `g^k = h` modulo one prime power says of `k`.
enum Says<I> {
    /// `k = x` modulo the order of `g`, which is the given number.
    Modulo(I, I),
    /// `k` is at least the given number.
    AtLeast(u64),
    /// `k` is exactly the given number.
    Exactly(u64),
}

/// The valuation of the non-zero `x` at the prime `p`, at most `limit`.
fn valuation<I: Int>(x: &I, p: &I, limit: u32) -> u32 {
    let mut v = 0;
    let mut x = x.clone();
    while v < limit && x.rem(p).is_zero() {
        x = x.div_exact(p);
        v += 1;
    }
    v
}

/// What `g^k = h (mod p^e)` says of `k`, for `g` and `h` below `p^e`; `None` when it has no
/// solution.
fn says<I: Int>(h: &I, g: &I, p: &I, e: u32) -> Option<Says<I>> {
    let pe = p.pow(e);
    if g.rem(p).is_zero() {
        // `g^k` has the valuation `k v` until it reaches `e`, and is 0 from `t = ceil(e / v)`.
        let v = if g.is_zero() { e } else { valuation(g, p, e) };
        let t = e.div_ceil(v);
        if h.is_zero() {
            return Some(Says::AtLeast(u64::from(t)));
        }
        let w = valuation(h, p, e);
        if !w.is_multiple_of(v) {
            return None;
        }
        return Some(Says::Exactly(u64::from(w / v)));
    }
    if h.rem(p).is_zero() {
        return None;
    }
    let hl = h.limbs();
    let gl = g.limbs();
    let found = if *p == I::small(2) {
        let n = if e > 1 {
            vec![(p.clone(), e - 1)]
        } else {
            Vec::new()
        };
        match e {
            1..=64 => {
                let r = Low::new(e);
                ph_in(
                    &r,
                    &r.residue(hl.as_slice()),
                    &r.residue(gl.as_slice()),
                    &n,
                    None,
                )
            }
            65..=128 => {
                let r = Low2::new(e);
                ph_in(
                    &r,
                    &r.residue(hl.as_slice()),
                    &r.residue(gl.as_slice()),
                    &n,
                    None,
                )
            }
            _ => {
                let r = Plain::new(pe.limbs().as_slice());
                ph_in(
                    &r,
                    &r.residue(hl.as_slice()),
                    &r.residue(gl.as_slice()),
                    &n,
                    None,
                )
            }
        }
    } else {
        let mut n = factor_below(p);
        if e > 1 {
            n.push((p.clone(), e - 1));
        }
        let pp = (e > 1).then(|| PrimePower {
            p,
            pe1: p.pow(e - 1),
        });
        let pel = pe.limbs();
        with_odd!(pel.as_slice(), |r| {
            let hr = r.residue(hl.as_slice());
            let gr = r.residue(gl.as_slice());
            ph_in(&r, &hr, &gr, &n, pp.as_ref())
        })
    };
    let (x, orders) = found?;
    Some(Says::Modulo(x, product(&orders)))
}

/// The smallest `k >= 0` with `g^k = h (mod m)`, for `m = prod p^e` at least 2 and `g`, `h` below
/// it, or `None`.
///
/// Each prime power says what it can: modulo one that `g` is prime to, `k` is a logarithm modulo
/// the order of `g` there (Pohlig and Hellman, in Montgomery's form modulo `p^e`), or there is
/// none; modulo one that `p` divides `g` by `p^v`, `g^k` is 0 from `k = ceil(e / v)`, so `h = 0`
/// asks for `k` at least that, and otherwise `k = v_p(h) / v` exactly. The logarithms are joined
/// by the Chinese remainder theorem, their moduli the orders, which need not be coprime; an exact
/// `k` is checked by one power; and the least `k` of the joined class that is at least every bound
/// is the answer.
pub(in super::super) fn discrete_log<I: Int>(h: &I, g: &I, m: &I, mf: &[(I, u32)]) -> Option<I> {
    let (mut k, mut n) = (I::small(0), I::small(1));
    let mut least = 0u64;
    let mut exactly: Option<u64> = None;
    for (p, e) in mf {
        let pe = p.pow(*e);
        match says(&h.rem(&pe), &g.rem(&pe), p, *e)? {
            Says::Modulo(x, o) => (k, n) = join(&k, &n, &x, &o)?,
            Says::AtLeast(t) => least = least.max(t),
            Says::Exactly(c) => {
                if exactly.is_some_and(|x| x != c) {
                    return None;
                }
                exactly = Some(c);
            }
        }
    }
    if let Some(c) = exactly {
        let ml = m.limbs();
        let ok = with_ring!(ml.as_slice(), |r| {
            let hr = r.residue(h.limbs().as_slice());
            let gr = r.residue(g.limbs().as_slice());
            r.pow_u64(&gr, c) == hr
        });
        return ok.then(|| I::small(c));
    }
    if k >= I::small(least) {
        return Some(k);
    }
    // `k + n ceil((least - k) / n)`.
    let gap = I::small(least).sub(&k);
    let steps = gap.add(&n.sub(&I::small(1)));
    let q = steps.sub(&steps.rem(&n)).div_exact(&n);
    Some(k.add(&n.mul(&q)))
}
