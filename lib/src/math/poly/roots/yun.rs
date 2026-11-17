//! The square-free factorisation the roots need, by Yun's algorithm over the integers with the
//! gcd area's greatest common divisor (one seam, `gcd`), after a test modulo the prime `2^61 − 1`
//! that settles the common case of a square-free polynomial without any gcd over the integers.

use super::super::Polynomial;
use crate::math::numbers::{Number, limbs_of};

/// The Mersenne prime `2^61 − 1` of the quick test.
const P: u64 = (1 << 61) - 1;

/// `a·b mod 2^61 − 1`, for `a, b < 2^61`.
#[inline]
fn mulmod(a: u64, b: u64) -> u64 {
    let t = u128::from(a) * u128::from(b);
    let r = (t as u64 & P) + (t >> 61) as u64;
    let r = (r & P) + (r >> 61);
    if r >= P { r - P } else { r }
}

/// `a − b mod 2^61 − 1`.
#[inline]
fn submod(a: u64, b: u64) -> u64 {
    if a >= b { a - b } else { a + P - b }
}

/// The coefficient `c` modulo `2^61 − 1`, from its limbs (`2^64 ≡ 8`).
fn residue(c: &Number) -> u64 {
    let mut r = 0u64;
    for l in limbs_of(c).into_iter().rev() {
        r = mulmod(r, 8) + (l & P) + (l >> 61);
        r = (r & P) + (r >> 61);
        if r >= P {
            r -= P;
        }
    }
    if c.is_negative() && r != 0 { P - r } else { r }
}

/// The degree of `gcd(a, b)` over `F_p`, `p = 2^61 − 1` (both not zero, without zeros at the
/// top), by Euclid's algorithm without divisions: `a ← lc(b)·a − lc(a)·x^k·b`, which keeps the
/// gcd up to a unit.
fn gcd_degree(mut a: Vec<u64>, mut b: Vec<u64>) -> usize {
    let trim = |v: &mut Vec<u64>| {
        while v.last() == Some(&0) {
            v.pop();
        }
    };
    loop {
        if a.len() < b.len() {
            core::mem::swap(&mut a, &mut b);
        }
        if b.is_empty() {
            return a.len() - 1;
        }
        if b.len() == 1 {
            return 0;
        }
        let m = b.len();
        let lb = b[m - 1];
        while a.len() >= m {
            let la = a[a.len() - 1];
            let off = a.len() - m;
            for x in a[..off].iter_mut() {
                *x = mulmod(*x, lb);
            }
            for (x, &y) in a[off..].iter_mut().zip(&b) {
                *x = submod(mulmod(*x, lb), mulmod(la, y));
            }
            a.pop();
            trim(&mut a);
        }
        if a.is_empty() {
            return b.len() - 1;
        }
    }
}

/// Whether `f` (primitive, of a degree at least 1) is square-free, if a test modulo the prime
/// `2^61 − 1` shows it: when the prime divides neither the leading coefficient nor `n·aₙ`,
/// `gcd(f mod p, f' mod p) = 1` implies `gcd(f, f') = 1`. `false` means "not shown", not "not
/// square-free".
fn squarefree_mod_p(f: &Polynomial<Number>) -> bool {
    let r: Vec<u64> = f.c.iter().map(residue).collect();
    let n = r.len() - 1;
    if r[n] == 0 || n as u64 >= P {
        return false;
    }
    let d: Vec<u64> = r
        .iter()
        .enumerate()
        .skip(1)
        .map(|(i, &x)| mulmod(x, i as u64))
        .collect();
    // The derivative keeps its degree: n·aₙ ≢ 0.
    gcd_degree(r, d) == 0
}

/// The primitive greatest common divisor of `a` and `b` over the integers, of a positive leading
/// coefficient: the one seam for the gcd area's gcd (multimodular, the heuristic gcd for the
/// shortest), which it calls. Yun's algorithm here, measured twice as fast as the factor area's `squarefree`
/// on the same gcd, calls nothing else of it.
fn gcd(a: &Polynomial<Number>, b: &Polynomial<Number>) -> Polynomial<Number> {
    a.gcd(b).primitive_part()
}

/// The square-free factorisation of the primitive polynomial `f` of a positive leading
/// coefficient and a degree at least 1, `f(0) ≠ 0` not needed: the pairs `(gᵢ, i)` with
/// `f = Π gᵢ^i`, each `gᵢ` square-free, primitive, of a positive leading coefficient and a degree
/// at least 1, pairwise coprime, by increasing `i`.
pub(crate) fn squarefree(f: &Polynomial<Number>) -> Vec<(Polynomial<Number>, u32)> {
    if f.degree().unwrap_or(0) <= 1 || squarefree_mod_p(f) {
        return vec![(f.clone(), 1)];
    }
    let exact = |a: &Polynomial<Number>, b: &Polynomial<Number>| {
        a.div_exact(b)
            .expect("an exact division of Yun's algorithm")
    };
    let df = f.derivative();
    let g = gcd(f, &df);
    if g.degree() == Some(0) {
        return vec![(f.clone(), 1)];
    }
    let mut c = exact(f, &g);
    let mut d = &exact(&df, &g) - &c.derivative();
    let mut out = Vec::new();
    let mut i = 1u32;
    while c.degree().unwrap_or(0) > 0 {
        let a = gcd(&c, &d);
        c = exact(&c, &a);
        d = &exact(&d, &a) - &c.derivative();
        if a.degree().unwrap_or(0) > 0 {
            out.push((a, i));
        }
        i += 1;
    }
    out
}

/// The primitive part of `f` with a positive leading coefficient, without the factor `x^k`, and
/// that `k`.
pub(crate) fn strip_zeros(f: &Polynomial<Number>) -> (Polynomial<Number>, usize) {
    let k = f.c.iter().position(|c| !c.is_zero()).unwrap_or(0);
    (f.shift_right(k).primitive_part(), k)
}
