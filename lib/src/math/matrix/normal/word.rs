//! The Hermite normal form modulo a multiple of the determinant ([`super::hermite::hermite_mod`])
//! in words, for a multiple below `2^63`: the same steps, every element a residue in a word and
//! each product of two of them reduced in a register (in a word below `2^32`, and above it by
//! Barrett's reduction of the product in two words with the modulus's reciprocal). The form is
//! unique, so it is the one the numbers give.

use super::hermite::Rows;
use crate::math::matrix::Matrix;
use crate::math::numbers::{Divisor, Number};

/// The multiples of the determinant below which the form is computed in words.
pub(super) const BELOW: u64 = 1 << 63;

/// The high 128 bits of `a·b`.
#[inline(always)]
fn mul_hi(a: u128, b: u128) -> u128 {
    let (a0, a1) = (a as u64 as u128, a >> 64);
    let (b0, b1) = (b as u64 as u128, b >> 64);
    let (p00, p01, p10, p11) = (a0 * b0, a0 * b1, a1 * b0, a1 * b1);
    let mid = (p00 >> 64) + (p01 as u64 as u128) + (p10 as u64 as u128);
    p11 + (p01 >> 64) + (p10 >> 64) + (mid >> 64)
}

/// A modulus `m < 2^63` with `⌊(2^128 − 1)/m⌋`.
#[derive(Clone, Copy)]
struct Md {
    m: u64,
    inv: u128,
}

impl Md {
    fn new(m: u64) -> Md {
        Md {
            m,
            inv: u128::MAX / u128::from(m),
        }
    }

    /// `a·b mod m` for `a, b < m`: in a word below `2^32`; above, `q = ⌊p·inv / 2^128⌋` is at
    /// most `p/m` and at least `p/m − 2` (`p < 2^126`), so `p − q·m` is below `3m`.
    #[inline(always)]
    fn mul(self, a: u64, b: u64) -> u64 {
        let m = self.m;
        if m <= 1 << 32 {
            return a * b % m;
        }
        let p = u128::from(a) * u128::from(b);
        let q = mul_hi(p, self.inv);
        let mut r = (p - q * u128::from(m)) as u64;
        while r >= m {
            r -= m;
        }
        r
    }
}

/// `a·b mod m` for `a, b < m`.
#[inline(always)]
fn mul(a: u64, b: u64, m: Md) -> u64 {
    m.mul(a, b)
}

/// `a − b mod m` for `a, b < m`.
#[inline(always)]
fn sub(a: u64, b: u64, m: u64) -> u64 {
    if a >= b { a - b } else { a + (m - b) }
}

/// `a + b mod m` for `a, b < m < 2^63`.
#[inline(always)]
fn add(a: u64, b: u64, m: u64) -> u64 {
    let s = a + b;
    if s >= m { s - m } else { s }
}

/// `(g, x, y)` with `g = gcd(a, b) = x·a + y·b`, for `a, b` not both zero.
fn gcd_ext(a: u64, b: u64) -> (u64, i128, i128) {
    let (mut r0, mut r1) = (i128::from(a), i128::from(b));
    let (mut x0, mut x1) = (1i128, 0i128);
    let (mut y0, mut y1) = (0i128, 1i128);
    while r1 != 0 {
        let q = r0 / r1;
        (r0, r1) = (r1, r0 - q * r1);
        (x0, x1) = (x1, x0 - q * x1);
        (y0, y1) = (y1, y0 - q * y1);
    }
    (r0 as u64, x0, y0)
}

/// `x mod m` in `[0, m)`.
#[inline]
fn residue(x: i128, m: u64) -> u64 {
    x.rem_euclid(i128::from(m)) as u64
}

/// The residue of `x` modulo `d`, `by` its divisor: in a word for an `x` that fits one.
#[inline]
fn residue_of(x: &Number, d: u64, by: &Divisor) -> u64 {
    if x.is_zero() {
        return 0;
    }
    match i64::try_from(x) {
        Ok(v) => i128::from(v).rem_euclid(i128::from(d)) as u64,
        Err(_) => u64::try_from(&by.rem_euclid(x)).expect("a residue in a word"),
    }
}

/// [`super::hermite::hermite_mod`] in words: the Hermite normal form of the lattice the rows
/// `active` span, of rank `n`, given a positive multiple `det < 2^63` of its determinant.
pub(super) fn hermite_mod(active: &[Vec<Number>], n: usize, det: &Number) -> Rows {
    let d = u64::try_from(&det.abs()).expect("a modulus in a word");
    debug_assert!(d > 0 && d < BELOW);
    let by = Divisor::new(det);
    let active: Vec<Vec<u64>> = active
        .iter()
        .map(|row| row.iter().map(|x| residue_of(x, d, &by)).collect())
        .filter(|row: &Vec<u64>| row.iter().any(|&x| x != 0))
        .collect();
    hermite_words(active, n, d)
}

/// The same for the rows of `a` on the columns `cols` (`n` of them), read in place.
pub(super) fn hermite_mod_cols(a: &Matrix<Number>, cols: &[usize], det: &Number) -> Rows {
    let d = u64::try_from(&det.abs()).expect("a modulus in a word");
    debug_assert!(d > 0 && d < BELOW);
    let by = Divisor::new(det);
    let active: Vec<Vec<u64>> = (0..a.rows())
        .map(|i| {
            cols.iter()
                .map(|&c| residue_of(&a[(i, c)], d, &by))
                .collect()
        })
        .filter(|row: &Vec<u64>| row.iter().any(|&x| x != 0))
        .collect();
    hermite_words(active, cols.len(), d)
}

/// The steps of the form on the residues `active`, nonzero rows of `n` elements below `d`.
fn hermite_words(mut active: Vec<Vec<u64>>, n: usize, d: u64) -> Rows {
    let mut modulus = d;
    let mut h: Vec<Vec<u64>> = Vec::with_capacity(n);
    let mut moduli = Vec::with_capacity(n + 1);
    for k in 0..n {
        moduli.push(modulus);
        let md = Md::new(modulus);
        let mut g: Option<Vec<u64>> = None;
        let mut rest = Vec::with_capacity(active.len());
        for mut v in active.drain(..) {
            if v[k] == 0 {
                rest.push(v);
                continue;
            }
            let Some(g) = g.as_mut() else {
                g = Some(v);
                continue;
            };
            combine(g, &mut v, k, md);
            if v[k + 1..].iter().any(|&x| x != 0) {
                rest.push(v);
            }
        }
        let mut row = match g {
            None => {
                let mut row = vec![0u64; n];
                row[k] = modulus;
                row
            }
            Some(mut g) => {
                let (dk, u, _) = gcd_ext(g[k], modulus);
                let u = residue(u, modulus);
                if u != 1 {
                    for x in &mut g[k + 1..] {
                        *x = mul(u, *x, md);
                    }
                }
                g[k] = dk;
                g
            }
        };
        let next = modulus / row[k];
        for x in &mut row[k + 1..] {
            *x %= next;
        }
        for v in &mut rest {
            for x in &mut v[k + 1..] {
                *x %= next;
            }
        }
        rest.retain(|v| v[k + 1..].iter().any(|&x| x != 0));
        h.push(row);
        active = rest;
        modulus = next;
    }
    moduli.push(modulus);
    for l in 1..n {
        let (above, below) = h.split_at_mut(l);
        let pivot_row = &below[0];
        let pivot = pivot_row[l];
        let shrinks = moduli[l + 1] != moduli[l];
        let d = moduli[l + 1];
        let dm = Md::new(d);
        for row in above {
            let q = row[l] / pivot;
            if q != 0 {
                row[l] -= q * pivot;
                let qd = q % d;
                for (x, &y) in row[l + 1..].iter_mut().zip(&pivot_row[l + 1..]) {
                    if y != 0 {
                        *x = sub(*x % d, mul(qd, y, dm), d);
                    }
                }
            }
            if shrinks || q != 0 {
                for x in &mut row[l + 1..] {
                    *x %= d;
                }
            }
        }
    }
    h.into_iter()
        .map(|row| {
            row.into_iter()
                .map(|x| {
                    if x == 0 {
                        Number::default()
                    } else {
                        Number::from(x)
                    }
                })
                .collect()
        })
        .collect()
}

/// [`super::hermite`]'s step of two rows at column `k` in words, modulo `m`: `g` gets the greatest
/// common divisor there, `v` a 0.
fn combine(g: &mut [u64], v: &mut [u64], k: usize, md: Md) {
    let m = md.m;
    let (gk, vk) = (g[k], v[k]);
    if vk % gk == 0 {
        let q = vk / gk;
        for (x, &y) in v[k..].iter_mut().zip(&g[k..]) {
            if y != 0 {
                *x = sub(*x, mul(q, y, md), m);
            }
        }
        return;
    }
    let (e, x, y) = gcd_ext(gk, vk);
    let (a, b) = (gk / e, vk / e);
    let (x, y) = (residue(x, m), residue(y, m));
    for (gj, vj) in g[k..].iter_mut().zip(v[k..].iter_mut()) {
        if *gj == 0 && *vj == 0 {
            continue;
        }
        let ng = add(mul(x, *gj, md), mul(y, *vj, md), m);
        let nv = sub(mul(a, *vj, md), mul(b, *gj, md), m);
        *gj = ng;
        *vj = nv;
    }
}
