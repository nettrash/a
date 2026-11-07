//! Square roots modulo primes, prime powers and products of them.
//!
//! Modulo an odd prime by Tonelli and Shanks's method, or Cipolla's when `p - 1` has many factors
//! of two; lifted to the prime's powers by Newton's (Hensel's) steps; modulo powers of two by the
//! 2-adic steps. The roots modulo `p^e` of `a = p^v u`, `u` prime to `p` and `v < e`, are those of
//! `u` modulo `p^(e - v)` times `p^(v/2)`, each with every multiple of `p^(e - v/2)` added: they are
//! the residues of a few numbers modulo `d = p^(e - v/2)`, and those modulo `m` are the solutions
//! of the Chinese remainder theorem for every choice of one root modulo each `d`, plus every
//! multiple of the product of the `d`s.

use super::int::Int;
use super::ring::{Ring, with_odd};
use super::root::jacobi_of_word;

/// Beyond this many factors of two in `p - 1`, Cipolla's method takes fewer products than Tonelli
/// and Shanks's, whose loop takes up to `s^2 / 2` of them.
const CIPOLLA: u32 = 24;

/// A square root of the residue `u`, not zero, modulo the odd prime `p` (of the limbs `pl`), in
/// the ring `r` modulo `p`, or `None` when `u` is not a square.
pub(in super::super) fn sqrt_prime<R: Ring, I: Int>(r: &R, u: &R::V, p: &I) -> Option<R::V> {
    let one = I::small(1);
    let pl = p.limbs();
    let p0 = pl.as_slice()[0];
    let root = if p0 & 3 == 3 {
        // `u^((p + 1) / 4)`.
        let e = p.add(&one).div_exact(&I::small(4));
        r.pow(u, e.limbs().as_slice())
    } else {
        let p1 = p.sub(&one);
        let s = trailing_zeros(&p1);
        if s > CIPOLLA {
            cipolla(r, u, p)?
        } else {
            tonelli_shanks(r, u, p, s)?
        }
    };
    (r.sqr(&root) == *u).then_some(root)
}

/// The smaller square root of `a`, below the odd prime `p` of a word and not zero, or `None` when
/// `a` is not a square: [`sqrt_prime`] in a register, for the moduli that are a prime.
pub(in super::super) fn sqrt_prime_word(a: u64, p: u64) -> Option<u64> {
    let f = super::word::Odd::new(p);
    let u = f.to(a);
    let root = if p & 3 == 3 {
        f.power(u, (p >> 2) + 1)
    } else {
        let s = (p - 1).trailing_zeros();
        if s > CIPOLLA {
            let r = super::word::Odd::new(p);
            return cipolla(&r, &u, &p).and_then(|x| {
                let x = r.from(x);
                (super::word::mul_mod(x, x, p) == a).then(|| x.min(p - x))
            });
        }
        tonelli_shanks(&f, &u, &p, s)?
    };
    if f.square(root) != u {
        return None;
    }
    let x = f.from(root);
    Some(x.min(p - x))
}

/// The number of zero bits at the low end of the non-zero `x`.
fn trailing_zeros<I: Int>(x: &I) -> u32 {
    let l = x.limbs();
    let mut z = 0;
    for &w in l.as_slice() {
        if w != 0 {
            return z + w.trailing_zeros();
        }
        z += 64;
    }
    z
}

/// `x / 2^s`.
fn shifted<I: Int>(x: &I, s: u32) -> I {
    let l = x.limbs();
    let l = l.as_slice();
    let (whole, bits) = ((s / 64) as usize, s % 64);
    let mut out = vec![0u64; l.len().saturating_sub(whole)];
    for i in 0..out.len() {
        let lo = l[i + whole] >> bits;
        let hi = if bits == 0 {
            0
        } else {
            l.get(i + whole + 1).map_or(0, |&h| h << (64 - bits))
        };
        out[i] = lo | hi;
    }
    I::of_limbs(&out)
}

/// The first word above 1 that is not a square modulo the odd prime `p`.
fn non_square(pl: &[u64]) -> u64 {
    let mut z = 2;
    while jacobi_of_word(z, pl) != -1 {
        z += 1;
    }
    z
}

/// Tonelli and Shanks's method for `p - 1 = q 2^s`, `s >= 1`: `x = u^((q + 1) / 2)` has
/// `x^2 = u b` for `b = u^q`, of order dividing `2^s`, and each step multiplies `x` and `b` by
/// powers of `c = z^q`, `z` not a square, of order `2^s`, until `b` is 1.
fn tonelli_shanks<R: Ring, I: Int>(r: &R, u: &R::V, p: &I, s: u32) -> Option<R::V> {
    let pl = p.limbs();
    let q = shifted(&p.sub(&I::small(1)), s);
    let half = shifted(&q.sub(&I::small(1)), 1);
    let y = r.pow(u, half.limbs().as_slice());
    let mut x = r.mul(u, &y);
    let mut b = r.mul(&x, &y);
    if r.is_one(&b) {
        return Some(x);
    }
    let z = non_square(pl.as_slice());
    let mut c = r.pow(&r.residue(&[z]), q.limbs().as_slice());
    let mut m = s;
    while !r.is_one(&b) {
        // The order of `b`, `2^t`.
        let mut t = 0;
        let mut b2 = b.clone();
        while !r.is_one(&b2) {
            b2 = r.sqr(&b2);
            t += 1;
            if t == m {
                return None;
            }
        }
        let mut g = c;
        for _ in 0..m - t - 1 {
            g = r.sqr(&g);
        }
        x = r.mul(&x, &g);
        c = r.sqr(&g);
        b = r.mul(&b, &c);
        m = t;
    }
    Some(x)
}

/// Cipolla's method (1903): with `w = t^2 - u` not a square, `(t + ω)^((p + 1) / 2)` in the field
/// of `a + b ω`, `ω^2 = w`, is a square root of `u`. About four and a half powers, whatever the
/// factors of two of `p - 1`.
fn cipolla<R: Ring, I: Int>(r: &R, u: &R::V, p: &I) -> Option<R::V> {
    let pl = p.limbs();
    let mut t = 1u64;
    let (tr, w) = loop {
        let tr = r.residue(&[t]);
        let w = r.sub(&r.sqr(&tr), u);
        let plain = r.plain(&w);
        let symbol = match plain.as_slice() {
            [] => 0,
            [x] => jacobi_of_word(*x, pl.as_slice()),
            _ => jacobi_limbs(&plain, pl.as_slice()),
        };
        if symbol == -1 {
            break (tr, w);
        }
        t += 1;
        if t > 1000 {
            // Not reached for a prime: half of all `t` give a non-square.
            return None;
        }
    };
    let e = p.add(&I::small(1));
    let e = shifted(&e, 1);
    let el = e.limbs();
    let el = el.as_slice();
    let bits = 64 * (el.len() - 1) as u32 + 64 - el[el.len() - 1].leading_zeros();
    // `(a, b)` for `a + b ω`, from the top bit.
    let (mut a, mut b) = (tr.clone(), r.one());
    for i in (0..bits - 1).rev() {
        // Square: `(a^2 + b^2 w) + 2ab ω`.
        let aa = r.sqr(&a);
        let bb = r.sqr(&b);
        let ab = r.mul(&a, &b);
        a = r.add(&aa, &r.mul(&bb, &w));
        b = r.add(&ab, &ab);
        if (el[(i / 64) as usize] >> (i % 64)) & 1 == 1 {
            // Times `t + ω`: `(a t + b w) + (a + b t) ω`.
            let na = r.add(&r.mul(&a, &tr), &r.mul(&b, &w));
            let nb = r.add(&a, &r.mul(&b, &tr));
            a = na;
            b = nb;
        }
    }
    Some(a)
}

/// The Jacobi symbol `(x / p)` of limbs, for the odd `p` above `x`.
fn jacobi_limbs(x: &[u64], p: &[u64]) -> i32 {
    super::super::quadratic::jacobi_of_limbs(x.to_vec(), p.to_vec())
}

/// The square root of the unit `u` modulo `p^f`, `u` given modulo it, from its root `x` modulo the
/// odd prime `p`, by Newton's steps `x - (x^2 - u) / (2x)`, each doubling the power of `p` it is
/// right modulo.
pub(in super::super) fn lift_odd<I: Int>(x: &I, u: &I, p: &I, f: u32) -> I {
    let mut x = x.clone();
    let mut k = 1;
    while k < f {
        let k2 = (2 * k).min(f);
        let pk = p.pow(k2);
        let u_k = u.rem(&pk);
        let d = x.mul_mod(&x, &pk).sub_mod(&u_k, &pk);
        let two_x = x.mul_mod(&I::small(2), &pk);
        let inv = two_x.inverse(&pk).unwrap_or_else(|| I::small(0));
        x = x.rem(&pk).sub_mod(&d.mul_mod(&inv, &pk), &pk);
        k = k2;
    }
    x
}

/// The square roots of the odd `u` modulo `2^f`, `f >= 1`, `u` given modulo it, or `None` when
/// there are none: `u = 1 (mod 4)` for `f = 2` and `u = 1 (mod 8)` from 3, then 2-adic Newton's
/// steps from 1, `x - ((x^2 - u) / 2) / x`, which take `x^2 = u (mod 2^k)` to `2^(2k - 2)`.
pub(in super::super) fn roots_two<I: Int>(u: &I, f: u32) -> Option<Vec<I>> {
    let low = u.limbs().as_slice()[0];
    match f {
        1 => return Some(vec![I::small(1)]),
        2 => {
            return (low & 3 == 1).then(|| vec![I::small(1), I::small(3)]);
        }
        _ => {}
    }
    if low & 7 != 1 {
        return None;
    }
    let x = if f <= 64 {
        // A bit at a time in a word: `x` with `x^2 = u (mod 2^k)` takes `2^(k - 1)` more when
        // that is not so modulo `2^(k + 1)`.
        let u = low;
        let mut x = 1u64;
        for k in 3..f {
            let mask = u64::MAX >> (63 - k);
            if x.wrapping_mul(x).wrapping_sub(u) & mask != 0 {
                x += 1 << (k - 1);
            }
        }
        I::small(x)
    } else {
        let two = I::small(2);
        let m = two.pow(f);
        let mm = m.mul(&two);
        let mut x = I::small(1);
        let mut k = 3;
        while k < f {
            // Modulo `2^(f + 1)`, so that the half of `x^2 - u` is right modulo `2^f`.
            let d = x.mul_mod(&x, &mm).sub_mod(&u.rem(&mm), &mm);
            let h = shifted(&d, 1).rem(&m);
            let inv = x.inverse(&m).unwrap_or_else(|| I::small(1));
            x = x.sub_mod(&h.mul_mod(&inv, &m), &m);
            k = 2 * k - 2;
        }
        x
    };
    let m = I::small(2).pow(f - 1).mul(&I::small(2));
    let m1 = I::small(2).pow(f - 1);
    let neg = m.sub(&x).rem(&m);
    let a = x.add(&m1).rem(&m);
    let b = neg.add(&m1).rem(&m);
    Some(vec![x, neg, a, b])
}

/// The roots of `a` modulo the prime power `p^e`, `a` given modulo it, as `(d, b)`: the roots are
/// the numbers below `p^e` congruent to one of the `b`s, each below `d`, modulo `d`; or `None` when
/// there are none.
pub(in super::super) fn roots_prime_power<I: Int>(a: &I, p: &I, e: u32) -> Option<(I, Vec<I>)> {
    if a.is_zero() {
        return Some((p.pow(e.div_ceil(2)), vec![I::small(0)]));
    }
    let mut v = 0;
    let mut u = a.clone();
    while u.rem(p).is_zero() {
        u = u.div_exact(p);
        v += 1;
    }
    if v % 2 == 1 {
        return None;
    }
    let f = e - v;
    let h = v / 2;
    let ys = if *p == I::small(2) {
        roots_two(&u, f)?
    } else {
        let pf = p.pow(f);
        let u_p = u.rem(p);
        let pl = p.limbs();
        let x = with_odd!(pl.as_slice(), |r| {
            let ur = r.residue(u_p.limbs().as_slice());
            sqrt_prime(&r, &ur, p).map(|x| I::of_limbs(&r.plain(&x)))
        })?;
        let x = lift_odd(&x, &u, p, f);
        let y = pf.sub(&x);
        vec![x, y]
    };
    let ph = p.pow(h);
    let d = p.pow(e - h);
    let mut bases: Vec<I> = ys.iter().map(|y| y.mul(&ph)).collect();
    bases.sort();
    bases.dedup();
    Some((d, bases))
}

/// The roots modulo the product `d` of the pairwise coprime `d_i` that are congruent to one of the
/// `b`s modulo each `d_i`: the coefficients `c_i = (d / d_i) ((d / d_i)^(-1) mod d_i)` of the
/// Chinese remainder theorem, by which a root is `sum b_i c_i mod d`.
pub(in super::super) fn coefficients<I: Int>(parts: &[(I, Vec<I>)]) -> (I, Vec<I>) {
    let d = parts.iter().fold(I::small(1), |acc, (di, _)| acc.mul(di));
    let c = parts
        .iter()
        .map(|(di, _)| {
            let rest = d.div_exact(di);
            let inv = rest.rem(di).inverse(di).unwrap_or_else(|| I::small(0));
            rest.mul_mod(&inv, &d)
        })
        .collect();
    (d, c)
}

/// The values `sum b_i c_i mod d` for every choice of one `b_i` for each part in `parts`, with the
/// coefficients `c`.
pub(in super::super) fn sums<I: Int>(parts: &[(I, Vec<I>)], c: &[I], d: &I) -> Vec<I> {
    let mut out = vec![I::small(0)];
    for ((_, b), ci) in parts.iter().zip(c) {
        let terms: Vec<I> = b.iter().map(|bi| bi.mul_mod(ci, d)).collect();
        let mut next = Vec::with_capacity(out.len() * terms.len());
        for x in &out {
            for t in &terms {
                // `x + t mod d`, without passing `d`, which may be near the top of a word.
                let room = d.sub(t);
                next.push(if *x >= room { x.sub(&room) } else { x.add(t) });
            }
        }
        out = next;
    }
    out
}

/// The smallest of `sum b_i c_i mod d` over every choice: by every sum when there are few, and by
/// meeting in the middle when there are many, the parts split in two halves whose sums `x` and `y`
/// are formed apart, the `y`s sorted, and for each `x` the smallest `x + y - d` with `y >= d - x`
/// or else `x + min y`.
pub(in super::super) fn smallest_sum<I: Int>(parts: &[(I, Vec<I>)], c: &[I], d: &I) -> I {
    let count: u128 = parts
        .iter()
        .map(|(_, b)| b.len() as u128)
        .fold(1, |a, b| a.saturating_mul(b));
    if count <= 1 << 12 || parts.len() < 2 {
        return sums(parts, c, d)
            .into_iter()
            .min()
            .unwrap_or_else(|| I::small(0));
    }
    // Split where the two halves' counts are nearest.
    let mut best = (u128::MAX, 1);
    let mut left = 1u128;
    for i in 1..parts.len() {
        left = left.saturating_mul(parts[i - 1].1.len() as u128);
        let right = count / left;
        let gap = left.abs_diff(right);
        if gap < best.0 {
            best = (gap, i);
        }
    }
    let k = best.1;
    let xs = sums(&parts[..k], &c[..k], d);
    let mut ys = sums(&parts[k..], &c[k..], d);
    ys.sort();
    let ymin = ys[0].clone();
    let mut min: Option<I> = None;
    for x in xs {
        let need = d.sub(&x);
        let i = ys.partition_point(|y| *y < need);
        let v = if i < ys.len() {
            ys[i].sub(&need)
        } else {
            x.add(&ymin)
        };
        if min.as_ref().is_none_or(|m| v < *m) {
            min = Some(v);
        }
    }
    min.unwrap_or_else(|| I::small(0))
}
