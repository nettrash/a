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
use super::word::Odd;

/// Beyond this many factors of two in `p - 1`, Cipolla's method takes fewer products than Tonelli
/// and Shanks's, whose loop takes up to `s^2 / 2` of them.
const CIPOLLA: u32 = 24;

/// A square root of the residue `u`, not zero, modulo the odd prime `p` (of the limbs `pl`), in
/// the ring `r` modulo `p`, or `None` when `u` is not a square: [`sqrt_prime_limbs`] on its limbs.
pub(in super::super) fn sqrt_prime<R: Ring, I: Int>(r: &R, u: &R::V, p: &I) -> Option<R::V> {
    let pl = p.limbs();
    sqrt_prime_limbs(r, u, pl.as_slice())
}

/// A square root of the residue `u`, not zero, modulo the odd prime of the limbs `p` (without zeros
/// at the top), in the ring `r` modulo it, or `None` when `u` is not a square, with its exponents
/// in limbs: `u^((p + 1) / 4)` for `p = 3 (mod 4)`; Atkin's `u v (i - 1)` for `p = 5 (mod 8)`, `v =
/// (2u)^((p - 5) / 8)` and `i = 2u v^2`, one power too; Tonelli and Shanks's method for `p = 1 (mod
/// 8)`, its powers of `u` and of a non-square side by side ([`Ring::pow_pair`]), or Cipolla's when
/// `p - 1` has more than [`CIPOLLA`] factors of two.
pub(in super::super) fn sqrt_prime_limbs<R: Ring>(r: &R, u: &R::V, p: &[u64]) -> Option<R::V> {
    let mut e = p.to_vec();
    let root = match p[0] & 7 {
        3 | 7 => {
            add_small(&mut e, 1);
            shift_right(&mut e, 2);
            r.pow(u, &e)
        }
        5 => {
            shift_right(&mut e, 3);
            let u2 = r.add(u, u);
            let v = r.pow(&u2, &e);
            let i = r.mul(&u2, &r.sqr(&v));
            r.mul(&r.mul(u, &v), &r.sub(&i, &r.one()))
        }
        _ => {
            // `p - 1 = q 2^s`: `p` is odd, so `p - 1` is `p` with its low bit cleared.
            e[0] &= !1;
            let s = trailing_zeros_limbs(&e);
            if s > CIPOLLA {
                cipolla(r, u, p)?
            } else {
                shift_right(&mut e, s);
                tonelli_shanks(r, u, p, &e, s)?
            }
        }
    };
    (r.sqr(&root) == *u).then_some(root)
}

/// `x += k` on limbs, which have room for the carry.
fn add_small(x: &mut Vec<u64>, k: u64) {
    let mut carry = k;
    for l in x.iter_mut() {
        let (s, c) = l.overflowing_add(carry);
        *l = s;
        carry = u64::from(c);
        if carry == 0 {
            return;
        }
    }
    if carry != 0 {
        x.push(carry);
    }
}

/// `x >>= s` on limbs, the zeros left at the top dropped.
fn shift_right(x: &mut Vec<u64>, s: u32) {
    let (whole, bits) = ((s / 64) as usize, s % 64);
    if whole > 0 {
        x.drain(..whole.min(x.len()));
    }
    if bits > 0 {
        for i in 0..x.len() {
            let hi = x.get(i + 1).map_or(0, |&h| h << (64 - bits));
            x[i] = (x[i] >> bits) | hi;
        }
    }
    while x.last() == Some(&0) {
        x.pop();
    }
}

/// The zero bits at the low end of limbs, not all zero.
fn trailing_zeros_limbs(x: &[u64]) -> u32 {
    let mut z = 0;
    for &w in x {
        if w != 0 {
            return z + w.trailing_zeros();
        }
        z += 64;
    }
    z
}

/// The smaller square root of `a`, below the odd prime `p` of a word and not zero, or `None` when
/// `a` is not a square: [`sqrt_prime`] in a register, for the moduli that are a prime.
///
/// For `p = 3 (mod 4)` the root is `a^((p + 1) / 4)`; for `p = 5 (mod 8)` Atkin's (1992), from one
/// power too: with `v = (2a)^((p - 5) / 8)` and `i = 2a v^2`, a square root of −1 when `a` is a
/// square, `a v (i - 1)`; for `p = 1 (mod 8)` Tonelli and Shanks's method, whose two powers, of `a`
/// and of a non-square, are taken side by side ([`Odd::powers`]).
pub(in super::super) fn sqrt_prime_word(a: u64, p: u64) -> Option<u64> {
    let f = Odd::new(p);
    let u = f.to(a);
    let root = match p & 7 {
        3 | 7 => f.power(u, (p >> 2) + 1),
        5 => {
            let u2 = f.plus(u, u);
            let v = f.power(u2, p >> 3);
            let i = f.times(u2, f.square(v));
            f.times(f.times(u, v), f.minus(i, f.unit()))
        }
        _ => {
            if (p - 1).trailing_zeros() > CIPOLLA {
                return cipolla(&f, &u, &[p]).and_then(|x| {
                    let x = f.from(x);
                    (super::word::mul_mod(x, x, p) == a).then(|| x.min(p - x))
                });
            }
            tonelli_word(&f, u, p)?
        }
    };
    if f.square(root) != u {
        return None;
    }
    let x = f.from(root);
    Some(x.min(p - x))
}

/// The odd primes below 64, each with the residues modulo it that are squares as bits (0 counted
/// among them).
const SMALL_SQUARES: [(u64, u64); 17] = {
    let primes = [
        3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61,
    ];
    let mut t = [(0, 0); 17];
    let mut i = 0;
    while i < 17 {
        let z = primes[i];
        let mut mask = 0u64;
        let mut x = 0;
        while x < z {
            mask |= 1 << (x * x % z);
            x += 1;
        }
        t[i] = (z, mask);
        i += 1;
    }
    t
};

/// The smallest prime that is not a square modulo the prime `p = 1 (mod 8)`: by reciprocity, `(z /
/// p) = (p / z)` for the odd primes `z`, which needs only `p mod z` and the squares modulo `z`;
/// past 61 by the Jacobi symbol of the odd numbers, which is −1 for a non-square.
fn non_square_word(p: u64) -> u64 {
    for &(z, squares) in &SMALL_SQUARES {
        if (squares >> (p % z)) & 1 == 0 {
            return z;
        }
    }
    let mut z = 67;
    while super::jacobi::jacobi_word(z, p) != -1 {
        z += 2;
    }
    z
}

/// Tonelli and Shanks's method in a register, for the prime `p = 1 (mod 8)`, `p - 1 = q 2^s`:
/// `u^((q - 1) / 2)` and `c = z^q` for the non-square `z` together, then the steps of
/// [`tonelli_shanks`].
fn tonelli_word(f: &Odd, u: u64, p: u64) -> Option<u64> {
    let s = (p - 1).trailing_zeros();
    let q = (p - 1) >> s;
    let z = f.to(non_square_word(p));
    let [y, mut c] = f.powers([u, z], [q >> 1, q]);
    let one = f.unit();
    let mut x = f.times(u, y);
    let mut b = f.times(x, y);
    let mut m = s;
    while b != one {
        // The order of `b`, `2^t`.
        let mut t = 0;
        let mut b2 = b;
        while b2 != one {
            b2 = f.square(b2);
            t += 1;
            if t == m {
                return None;
            }
        }
        let mut g = c;
        for _ in 0..m - t - 1 {
            g = f.square(g);
        }
        x = f.times(x, g);
        c = f.square(g);
        b = f.times(b, c);
        m = t;
    }
    Some(x)
}

/// `x / 2^s`.
fn shifted<I: Int>(x: &I, s: u32) -> I {
    let l = x.limbs();
    let mut v = l.as_slice().to_vec();
    shift_right(&mut v, s);
    I::of_limbs(&v)
}

/// The first word above 1 that is not a square modulo the odd prime `p`.
fn non_square(pl: &[u64]) -> u64 {
    let mut z = 2;
    while jacobi_of_word(z, pl) != -1 {
        z += 1;
    }
    z
}

/// Tonelli and Shanks's method for `p - 1 = q 2^s`, `s >= 1`, `q` given as limbs: `x = u^((q + 1) /
/// 2)` has `x^2 = u b` for `b = u^q`, of order dividing `2^s`, and each step multiplies `x` and `b`
/// by powers of `c = z^q`, `z` not a square, of order `2^s`, until `b` is 1. The powers of `u` and
/// of `z` to `(q - 1) / 2` are taken side by side, and `c = z (z^((q - 1) / 2))^2`.
fn tonelli_shanks<R: Ring>(r: &R, u: &R::V, p: &[u64], q: &[u64], s: u32) -> Option<R::V> {
    let mut half = q.to_vec();
    shift_right(&mut half, 1);
    let zr = r.residue(&[non_square(p)]);
    let (y, zh) = r.pow_pair(u, &zr, &half);
    let mut x = r.mul(u, &y);
    let mut b = r.mul(&x, &y);
    if r.is_one(&b) {
        return Some(x);
    }
    let mut c = r.mul(&zr, &r.sqr(&zh));
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
fn cipolla<R: Ring>(r: &R, u: &R::V, pl: &[u64]) -> Option<R::V> {
    let mut t = 1u64;
    let (tr, w) = loop {
        let tr = r.residue(&[t]);
        let w = r.sub(&r.sqr(&tr), u);
        let plain = r.plain(&w);
        let symbol = match plain.as_slice() {
            [] => 0,
            [x] => jacobi_of_word(*x, pl),
            _ => jacobi_limbs(&plain, pl),
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
    let mut el = pl.to_vec();
    add_small(&mut el, 1);
    shift_right(&mut el, 1);
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

// Words

/// The roots modulo one prime power `p^e` of a word modulus: the numbers below `p^e` congruent
/// modulo `d` to one of the first `n` of `b`, which are increasing.
#[derive(Clone, Copy, Default)]
pub(in super::super) struct Part {
    /// The modulus of the roots, `p^(e - v/2)` for `p^v` dividing the number.
    pub(in super::super) d: u64,
    /// The roots modulo `d`.
    pub(in super::super) b: [u64; 4],
    /// How many there are.
    pub(in super::super) n: usize,
}

/// [`roots_prime_power`] in words, for `a < p^e < 2^64`: no vector, and the root modulo an odd
/// prime by [`sqrt_prime_word`].
pub(in super::super) fn part_word(a: u64, p: u64, e: u32) -> Option<Part> {
    if a == 0 {
        let mut b = [0; 4];
        b[0] = 0;
        return Some(Part {
            d: p.pow(e.div_ceil(2)),
            b,
            n: 1,
        });
    }
    let (u, v) = if p == 2 {
        let v = a.trailing_zeros();
        (a >> v, v)
    } else {
        let (mut u, mut v) = (a, 0);
        while u % p == 0 {
            u /= p;
            v += 1;
        }
        (u, v)
    };
    if v % 2 == 1 {
        return None;
    }
    let f = e - v;
    let h = v / 2;
    let mut ys = [0u64; 4];
    let n = if p == 2 {
        match f {
            1 => {
                ys[0] = 1;
                1
            }
            2 => {
                if u & 3 != 1 {
                    return None;
                }
                ys[..2].copy_from_slice(&[1, 3]);
                2
            }
            _ => {
                if u & 7 != 1 {
                    return None;
                }
                // A bit at a time, as `roots_two` takes it in a word.
                let mut x = 1u64;
                for k in 3..f {
                    let mask = u64::MAX >> (63 - k);
                    if x.wrapping_mul(x).wrapping_sub(u) & mask != 0 {
                        x += 1 << (k - 1);
                    }
                }
                let m = 1u64 << f;
                let half = m >> 1;
                let neg = (m - x) & (m - 1);
                ys = [x, neg, (x + half) & (m - 1), (neg + half) & (m - 1)];
                4
            }
        }
    } else {
        let x = sqrt_prime_word(u % p, p)?;
        let x = lift_odd::<u64>(&x, &u, &p, f);
        ys[..2].copy_from_slice(&[x, p.pow(f) - x]);
        2
    };
    let ph = p.pow(h);
    let mut b = [0u64; 4];
    for (o, &y) in b.iter_mut().zip(&ys[..n]) {
        *o = y * ph;
    }
    // Sorted, without repeats (a few at most).
    let b4 = &mut b[..n];
    b4.sort_unstable();
    let mut k = 0;
    for i in 0..n {
        if i == 0 || b4[i] != b4[k - 1] {
            b4[k] = b4[i];
            k += 1;
        }
    }
    Some(Part {
        d: p.pow(e - h),
        b,
        n: k,
    })
}

/// The parts of `a` modulo `prod p^e` for the word factorisation `f`, into `out`: their number, or
/// `None` when `a` is not a square modulo one of them.
pub(in super::super) fn parts_word(
    a: u64,
    f: &[(u64, u32)],
    out: &mut [Part; 15],
) -> Option<usize> {
    for (o, &(p, e)) in out.iter_mut().zip(f) {
        let pe = p.pow(e);
        *o = part_word(a % pe, p, e)?;
    }
    Some(f.len())
}

/// The product `D` of the parts' moduli and the terms `b c_i mod D` of each part's roots, `c_i` the
/// Chinese remainder theorem's coefficient of the part (see [`coefficients`]), in place of its
/// roots.
pub(in super::super) fn terms_word(parts: &mut [Part]) -> u64 {
    let d = parts.iter().fold(1u64, |acc, p| acc * p.d);
    for p in parts.iter_mut() {
        let rest = d / p.d;
        let inv = super::word::inverse(rest % p.d, p.d).unwrap_or(0);
        let c = super::word::mul_mod(rest, inv, d);
        for b in &mut p.b[..p.n] {
            *b = super::word::mul_mod(*b, c, d);
        }
    }
    d
}

/// `x + t mod d` for `x, t < d`, without passing `d`, which may be near the top of a word.
#[inline(always)]
fn add_mod(x: u64, t: u64, d: u64) -> u64 {
    let room = d - t;
    if x >= room { x - room } else { x + t }
}

/// The number of choices of one root a part, saturated.
pub(in super::super) fn choices(parts: &[Part]) -> u128 {
    parts
        .iter()
        .fold(1u128, |c, p| c.saturating_mul(p.n as u128))
}

/// Calls `f` with every sum of one term of each part modulo `d`, depth first.
#[inline]
pub(in super::super) fn each_sum(parts: &[Part], d: u64, f: &mut impl FnMut(u64)) {
    fn go(parts: &[Part], d: u64, s: u64, f: &mut impl FnMut(u64)) {
        match parts {
            [] => f(s),
            [p, rest @ ..] => {
                for &t in &p.b[..p.n] {
                    go(rest, d, add_mod(s, t, d), f);
                }
            }
        }
    }
    go(parts, d, 0, f);
}

/// The smallest root of `a` modulo the word `m = prod p^e`, for its factorisation `f`, or `None`:
/// every sum of the terms when there are few choices, and by meeting in the middle beyond 4096 of
/// them ([`smallest_sum`]).
pub(in super::super) fn smallest_word(a: u64, f: &[(u64, u32)]) -> Option<u64> {
    let mut parts = [Part::default(); 15];
    let k = parts_word(a, f, &mut parts)?;
    let parts = &mut parts[..k];
    let d = terms_word(parts);
    if choices(parts) <= 1 << 12 {
        let mut min = u64::MAX;
        each_sum(parts, d, &mut |s| min = min.min(s));
        return Some(if min == u64::MAX { 0 } else { min });
    }
    // The terms are the roots times their coefficients already: coefficients of 1.
    let generic: Vec<(u64, Vec<u64>)> = parts.iter().map(|p| (p.d, p.b[..p.n].to_vec())).collect();
    let ones = vec![1u64; generic.len()];
    Some(smallest_sum(&generic, &ones, &d))
}

/// Every root of `a` modulo the word `m = prod p^e`, increasing, for its factorisation `f`.
pub(in super::super) fn all_word(a: u64, f: &[(u64, u32)], m: u64) -> Vec<u64> {
    let mut parts = [Part::default(); 15];
    let Some(k) = parts_word(a, f, &mut parts) else {
        return Vec::new();
    };
    let parts = &mut parts[..k];
    let d = terms_word(parts);
    // The roots below `d`, and their copies a multiple of `d` higher, up to `m`: counted saturated,
    // as on 32-bit targets they may pass `usize::MAX`, which fails to allocate, as it must.
    let below = usize::try_from(choices(parts)).unwrap_or(usize::MAX);
    let copies = usize::try_from(m / d).unwrap_or(usize::MAX);
    let mut xs = Vec::with_capacity(below.saturating_mul(copies));
    each_sum(parts, d, &mut |s| xs.push(s));
    xs.sort_unstable();
    for c in 1..copies {
        let base = c as u64 * d;
        for i in 0..below {
            xs.push(base + xs[i]);
        }
    }
    xs
}
