//! The Chinese remainder theorem: congruences modulo words joined one at a time on limbs, as
//! Garner's method does, and longer moduli, or the results of many words, joined in pairs up a
//! balanced tree.

use super::super::div::limbs::{div_2by1, invert_limb};
use super::super::{ByteOrder, Number};
use super::int::join;
use super::ring::{from_limbs, magnitude_64, magnitude_128};
use super::word;

/// Up to this many congruences modulo words are joined one at a time; more are joined in groups of
/// this many, whose results are joined in pairs up a tree, which takes subquadratic time in all.
pub(in super::super) const CHAIN: usize = 64;

/// A word prepared for remainders by it: shifted so that its top bit is set, with the reciprocal.
#[derive(Clone, Copy)]
struct Word {
    /// Its leading zeros.
    s: u32,
    /// `n << s`.
    d: u64,
    /// The reciprocal of `d`.
    v: u64,
}

impl Word {
    fn new(n: u64) -> Word {
        let s = n.leading_zeros();
        let d = n << s;
        Word {
            s,
            d,
            v: invert_limb(d),
        }
    }

    /// The remainders of the limbs `x` and `y`, of the same length, by the word, in one pass:
    /// both shifted left by `s` bits on the fly, divided from the top by the shifted word, and the
    /// remainders shifted back.
    #[inline]
    fn rem2(&self, x: &[u64], y: &[u64]) -> (u64, u64) {
        let (s, d, v) = (self.s, self.d, self.v);
        let n = x.len();
        let (mut rx, mut ry) = (0u64, 0u64);
        if s == 0 {
            for i in (0..n).rev() {
                rx = div_2by1(rx, x[i], d, v).1;
                ry = div_2by1(ry, y[i], d, v).1;
            }
            return (rx, ry);
        }
        // The limb above the top, the bits shifted out of it.
        rx = div_2by1(0, x[n - 1] >> (64 - s), d, v).1;
        ry = div_2by1(0, y[n - 1] >> (64 - s), d, v).1;
        for i in (0..n).rev() {
            let below_x = if i > 0 { x[i - 1] >> (64 - s) } else { 0 };
            let below_y = if i > 0 { y[i - 1] >> (64 - s) } else { 0 };
            rx = div_2by1(rx, (x[i] << s) | below_x, d, v).1;
            ry = div_2by1(ry, (y[i] << s) | below_y, d, v).1;
        }
        (rx >> s, ry >> s)
    }
}

/// `x += m t` on limbs, `x` with room for the carry.
fn add_times(x: &mut Vec<u64>, m: &[u64], t: u64) {
    if x.len() < m.len() + 1 {
        x.resize(m.len() + 1, 0);
    }
    let mut carry = 0u64;
    for i in 0..m.len() {
        let p = u128::from(m[i]) * u128::from(t) + u128::from(x[i]) + u128::from(carry);
        x[i] = p as u64;
        carry = (p >> 64) as u64;
    }
    let mut i = m.len();
    while carry != 0 {
        if i == x.len() {
            x.push(0);
        }
        let (s, c) = x[i].overflowing_add(carry);
        x[i] = s;
        carry = u64::from(c);
        i += 1;
    }
}

/// `m *= t` on limbs.
fn times(m: &mut Vec<u64>, t: u64) {
    let mut carry = 0u64;
    for l in m.iter_mut() {
        let p = u128::from(*l) * u128::from(t) + u128::from(carry);
        *l = p as u64;
        carry = (p >> 64) as u64;
    }
    if carry != 0 {
        m.push(carry);
    }
}

/// The congruences `x = r_i (mod m_i)` modulo words, `r_i < m_i`, joined one at a time: `(x, l)`
/// as limbs, `l` the least common multiple, `x < l`; or `None` when they contradict each other.
///
/// Each step brings in `x = s (mod n)`: with `g = gcd(l, n)`, which must divide `s - x`, and
/// `n' = n / g`, `x + l t` with `t = ((s - x) / g) ((l / g)^(-1)) mod n'` is the solution modulo
/// `l n'`. It needs `l` and `x` modulo `n`, by one pass over both.
pub(in super::super) fn chain(c: &[(u64, u64)]) -> Option<(Vec<u64>, Vec<u64>)> {
    // In registers while the modulus fits in two words.
    let (mut x, mut l) = (0u128, 1u128);
    let mut i = 0;
    while i < c.len() {
        let (s, n) = c[i];
        if n == 1 {
            i += 1;
            continue;
        }
        let w = Word::new(n);
        let (xr, lr) = w.rem2(&[x as u64, (x >> 64) as u64], &[l as u64, (l >> 64) as u64]);
        let Some((t, n1)) = step(s, n, xr, lr)? else {
            i += 1;
            continue;
        };
        let Some(next) = l.checked_mul(u128::from(n1)) else {
            break;
        };
        x += l * u128::from(t);
        l = next;
        i += 1;
    }
    let mut xl = vec![x as u64, (x >> 64) as u64];
    let mut ll = vec![l as u64, (l >> 64) as u64];
    for &(s, n) in &c[i..] {
        if n == 1 {
            continue;
        }
        let w = Word::new(n);
        xl.resize(ll.len(), 0);
        let (xr, lr) = w.rem2(&xl, &ll);
        let Some((t, n1)) = step(s, n, xr, lr)? else {
            continue;
        };
        add_times(&mut xl, &ll, t);
        times(&mut ll, n1);
        while xl.last() == Some(&0) && xl.len() > 1 {
            xl.pop();
        }
    }
    Some((xl, ll))
}

/// One step of [`chain`]: for `x = xr` and `l = lr` modulo `n`, the `t` and `n' = n / gcd(l, n)`
/// that make `x + l t` the solution modulo `l n'`; `Some(None)` when `n'` is 1 (nothing to add),
/// and `None` when the congruences contradict each other. The inverse comes first: it exists
/// exactly when `l` and `n` are coprime, the common case, which needs no greatest common divisor.
#[inline]
fn step(s: u64, n: u64, xr: u64, lr: u64) -> Option<Option<(u64, u64)>> {
    let d = if s >= xr { s - xr } else { n - (xr - s) };
    if let Some(inv) = word::inverse(lr, n) {
        return Some(Some((word::mul_mod(d, inv, n), n)));
    }
    let g = word::gcd(lr, n);
    if d % g != 0 {
        return None;
    }
    let n1 = n / g;
    if n1 == 1 {
        return Some(None);
    }
    let inv = word::inverse((lr / g) % n1, n1)?;
    Some(Some((word::mul_mod((d / g) % n1, inv, n1), n1)))
}

/// The congruences `(r_i, m_i)`, `r_i < m_i`, joined in pairs up a balanced tree.
pub(in super::super) fn tree(mut c: Vec<(Number, Number)>) -> Option<(Number, Number)> {
    if c.is_empty() {
        return Some((
            Number::new_from_u8(0, ByteOrder::LittleEndian),
            Number::new_from_u8(1, ByteOrder::LittleEndian),
        ));
    }
    while c.len() > 1 {
        let mut next = Vec::with_capacity(c.len().div_ceil(2));
        let mut it = c.into_iter();
        while let Some((r, m)) = it.next() {
            match it.next() {
                Some((s, n)) => next.push(join(&r, &m, &s, &n)?),
                None => next.push((r, m)),
            }
        }
        c = next;
    }
    c.pop()
}

/// The congruences `(r_i, m_i)`, `r_i < m_i`, for pairwise coprime moduli, by the product and
/// remainder trees (as PARI's `ZV_chinese` and FLINT's `fmpz_multi_CRT` take them): with `P` the
/// product of the moduli, `c_i = (P / m_i) mod m_i = (P mod m_i^2) / m_i` comes down the tree of
/// the remainders of `P` by the squares of the products, `t_i = r_i c_i^(-1) mod m_i` at each leaf,
/// and `x = sum t_i P / m_i` goes up the product tree as `x_l M_r + x_r M_l`, reduced by `P` at the
/// top. `None` when a `c_i` has no inverse: the moduli are not pairwise coprime.
pub(in super::super) fn fast(c: &[(Number, Number)]) -> Option<(Number, Number)> {
    let k = c.len();
    // The product tree, the moduli at the bottom.
    let mut levels: Vec<Vec<Number>> = vec![c.iter().map(|(_, m)| m.clone()).collect()];
    while levels.last().is_some_and(|l| l.len() > 1) {
        let below = levels.last().map(|l| l.as_slice()).unwrap_or(&[]);
        let up: Vec<Number> = below
            .chunks(2)
            .map(|p| {
                if p.len() == 2 {
                    &p[0] * &p[1]
                } else {
                    p[0].clone()
                }
            })
            .collect();
        levels.push(up);
    }
    let p = levels.last()?[0].clone();
    // Down: the remainders of `P` by the squares.
    let mut rems = vec![p.clone()];
    for level in levels.iter().rev().skip(1) {
        let mut next = Vec::with_capacity(level.len());
        for (i, m) in level.iter().enumerate() {
            let parent = &rems[i / 2];
            next.push(parent.rem_euclid(&(m * m)));
        }
        rems = next;
    }
    // The leaves, in words for the moduli of a word.
    let mut xs: Vec<Number> = Vec::with_capacity(k);
    for (i, (r, m)) in c.iter().enumerate() {
        if let (Some(mw), Some(rem)) = (magnitude_64(m), magnitude_128(&rems[i])) {
            let rw = magnitude_64(r).unwrap_or(0);
            let ci = ((rem / u128::from(mw)) % u128::from(mw)) as u64;
            let inv = word::inverse(ci, mw)?;
            xs.push(Number::new_from_u64(
                word::mul_mod(rw, inv, mw),
                ByteOrder::LittleEndian,
            ));
            continue;
        }
        let ci = rems[i].div_exact(m).rem_euclid(m);
        let inv = if *m == 1u8 {
            Number::new_from_u8(0, ByteOrder::LittleEndian)
        } else {
            ci.mod_inverse(m)?
        };
        xs.push((r * &inv).rem_euclid(m));
    }
    // Up: `x_l M_r + x_r M_l`.
    for level in &levels[..levels.len() - 1] {
        let mut next = Vec::with_capacity(xs.len().div_ceil(2));
        for (i, pair) in xs.chunks(2).enumerate() {
            if pair.len() == 2 {
                next.push(&(&pair[0] * &level[2 * i + 1]) + &(&pair[1] * &level[2 * i]));
            } else {
                next.push(pair[0].clone());
            }
        }
        xs = next;
    }
    let x = xs.pop()?.rem_euclid(&p);
    Some((x, p))
}

/// From this many congruences on, they are first tried as pairwise coprime by [`fast`] when a
/// modulus is longer than a word: measured on an Apple M4 Max, 8 primes of 1024 bits took 184,000
/// cycles so, against 220,000 for PARI's `ZV_chinese`.
pub(in super::super) const FAST: usize = 8;

/// From this many congruences modulo words on, they are tried by [`fast`]: below, the chain of
/// [`chain`] took less time, 198,000 cycles against 256,000 for 100 primes of 64 bits, and more
/// beyond, 587,000 for 200 against about 700,000.
pub(in super::super) const FAST_WORDS: usize = 150;

/// The congruences modulo words joined: one chain for few, groups joined up a tree for many.
pub(in super::super) fn words(c: &[(u64, u64)]) -> Option<(Number, Number)> {
    if c.len() <= CHAIN {
        let (x, l) = chain(c)?;
        return Some((
            from_limbs(&x, ByteOrder::LittleEndian),
            from_limbs(&l, ByteOrder::LittleEndian),
        ));
    }
    let mut parts = Vec::with_capacity(c.len().div_ceil(CHAIN));
    for group in c.chunks(CHAIN) {
        let (x, l) = chain(group)?;
        parts.push((
            from_limbs(&x, ByteOrder::LittleEndian),
            from_limbs(&l, ByteOrder::LittleEndian),
        ));
    }
    tree(parts)
}
