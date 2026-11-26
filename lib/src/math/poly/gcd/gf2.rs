//! The engine of the half gcd on polynomials over GF(2), packed 64 coefficients a word
//! ([`arith::gf2`](crate::math::poly::arith::gf2)'s form): sums are exclusive ors of words,
//! products the packed products (Karatsuba's, the transforms for huge operands), divisions
//! shifted subtractions for a short quotient
//! and the packed division ([`Div`]) for a long one, and the base case Euclid's algorithm on the
//! packed words with the matrix of its steps.

use super::hgcd::{Engine, Mat, Steps};
use crate::math::poly::arith::gf2::{self, Div, degree, trim, xor_shifted};

/// Below this degree the half gcd modulo 2 takes Euclid's steps (with their matrix) on the packed
/// words: measured on an Apple M4 Max (128 to 4,096, gcds of 8,192 to 262,144 coefficients, dense,
/// planted, sparse and trinomial operands), 512 to 896 equally the fastest, 4 to 8% faster than
/// 1,024 (256 2% faster, 1,536 as fast), 128 and from 2,048 6 to 60% slower.
const HGCD_BASE: usize = if cfg!(miri) { 100 } else { 768 };

/// A quotient of up to this many coefficients is taken by shifted subtractions, a longer one by
/// the packed division (the steps of the half gcd have quotients of a coefficient or two).
/// Measured on an Apple M4 Max (16 to 4,096; gcds of 4,096 to 131,072 coefficients, dense, sparse,
/// trinomial, `x^a − 1` against `x^b − 1`, and dense ones whose first quotient has 16 to 3,000
/// coefficients): flat from 16 to 256 (within 0.5%); 512 and 1,024 up to 11% slower on long
/// dense quotients, 4,096 up to 17% faster for `x^a − 1` against `x^b − 1` (a few microseconds).
const SHORT_QUOTIENT: usize = 128;

/// The engine.
pub(crate) struct Gf2;

/// `a ^ b`, trimmed.
fn xor(a: &[u64], b: &[u64]) -> Vec<u64> {
    let (long, short) = if a.len() >= b.len() { (a, b) } else { (b, a) };
    let mut r = long.to_vec();
    for (x, &y) in r.iter_mut().zip(short) {
        *x ^= y;
    }
    trim(&mut r);
    r
}

impl Engine for Gf2 {
    type P = Vec<u64>;
    type C = u64;

    #[inline]
    fn len(&self, a: &Vec<u64>) -> usize {
        degree(a).map_or(0, |d| d + 1)
    }
    fn zero(&self) -> Vec<u64> {
        Vec::new()
    }
    fn one(&self) -> Vec<u64> {
        vec![1]
    }
    fn lead(&self, _a: &Vec<u64>) -> u64 {
        1
    }
    fn add(&self, a: &Vec<u64>, b: &Vec<u64>) -> Vec<u64> {
        xor(a, b)
    }
    fn sub(&self, a: &Vec<u64>, b: &Vec<u64>) -> Vec<u64> {
        xor(a, b)
    }
    fn neg(&self, a: &Vec<u64>) -> Vec<u64> {
        a.clone()
    }
    fn mul(&self, a: &Vec<u64>, b: &Vec<u64>) -> Vec<u64> {
        gf2::mul(a, b)
    }
    fn divrem(&self, a: &Vec<u64>, b: &Vec<u64>) -> (Vec<u64>, Vec<u64>) {
        let db = degree(b).expect("division by the zero polynomial");
        let Some(da) = degree(a) else {
            return (Vec::new(), Vec::new());
        };
        if da < db {
            return (Vec::new(), a.clone());
        }
        if da - db < SHORT_QUOTIENT {
            let mut q = vec![0u64; (da - db) / 64 + 1];
            let mut r = a.clone();
            gf2::reduce(&mut r, b, db, Some(&mut q));
            trim(&mut q);
            return (q, r);
        }
        Div::new(b).div_rem(a)
    }
    fn shr(&self, a: &Vec<u64>, k: usize) -> Vec<u64> {
        let l = self.len(a);
        if k >= l {
            return Vec::new();
        }
        gf2::extract(a, k, l - k)
    }
    fn low(&self, a: &Vec<u64>, k: usize) -> Vec<u64> {
        gf2::extract(a, 0, k)
    }
    fn shl_add(&self, hi: &Vec<u64>, k: usize, lo: &Vec<u64>) -> Vec<u64> {
        let Some(dh) = degree(hi) else {
            return lo.clone();
        };
        let mut r = lo.clone();
        r.resize(r.len().max((dh + k) / 64 + 1), 0);
        xor_shifted(&mut r, hi, k);
        trim(&mut r);
        r
    }
    fn cmul(&self, x: &u64, y: &u64) -> u64 {
        x & y
    }
    fn cneg(&self, x: &u64) -> u64 {
        *x
    }
    fn cpow(&self, x: &u64, e: u64) -> u64 {
        if e == 0 { 1 } else { *x }
    }
    fn cone(&self) -> u64 {
        1
    }
    fn czero(&self) -> u64 {
        0
    }
    fn euclid(
        &self,
        a: Vec<u64>,
        b: Vec<u64>,
        rec: Option<&mut Vec<(usize, u64)>>,
        shift: usize,
    ) -> Vec<u64> {
        gf2::euclid(&a, &b, rec, shift)
    }
    fn hgcd_steps(&self, a: &Vec<u64>, b: &Vec<u64>, m: usize) -> Option<Steps<Vec<u64>>> {
        Some(steps(a, b, m))
    }
    fn hgcd_cutoff(&self) -> usize {
        HGCD_BASE
    }
    fn gcd_cutoff(&self) -> usize {
        gf2::HGCD_GCD
    }
    fn xgcd_cutoff(&self) -> usize {
        usize::MAX
    }
}

/// `v` with room for the coefficients below `x^(d + 1)`.
#[inline]
fn room(v: &mut Vec<u64>, d: usize) {
    if v.len() <= d / 64 {
        v.resize(d / 64 + 1, 0);
    }
}

/// Euclid's steps from `(a, b)` (packed, trimmed, `deg a > deg b`) while the divisor has more
/// than `m` coefficients, on the packed words without a product: each quotient coefficient
/// `x^k` one shifted subtraction from the dividend and one from each column of the matrix
/// (`M·[[q, 1], [1, 0]] = (c₁q + c₂, c₁)` for the columns `c₁`, `c₂`). The matrix and the two
/// remainders, as the generic steps of the half gcd give them.
fn steps(a: &[u64], b: &[u64], m: usize) -> Steps<Vec<u64>> {
    let (mut x, mut y) = (a.to_vec(), b.to_vec());
    let (mut ma, mut mb, mut mc, mut md) = (vec![1u64], Vec::new(), Vec::new(), vec![1u64]);
    let mut neg = false;
    while let Some(dy) = degree(&y)
        && dy >= m
    {
        // x ← x mod y, (a, b) ← (a q + b, a), (c, d) ← (c q + d, c).
        let (mut na, mut nc) = (core::mem::take(&mut mb), core::mem::take(&mut md));
        let (da, dc) = (degree(&ma), degree(&mc));
        let mut dx = degree(&x);
        while let Some(d) = dx
            && d >= dy
        {
            let k = d - dy;
            xor_shifted(&mut x, &y, k);
            if let Some(da) = da {
                room(&mut na, da + k);
                xor_shifted(&mut na, &ma, k);
            }
            if let Some(dc) = dc {
                room(&mut nc, dc + k);
                xor_shifted(&mut nc, &mc, k);
            }
            dx = degree(&x[..d / 64 + 1]);
        }
        trim(&mut x);
        trim(&mut na);
        trim(&mut nc);
        mb = core::mem::replace(&mut ma, na);
        md = core::mem::replace(&mut mc, nc);
        neg = !neg;
        core::mem::swap(&mut x, &mut y);
    }
    let mat = Mat {
        a: ma,
        b: mb,
        c: mc,
        d: md,
        neg,
    };
    (mat, x, y)
}

/// The greatest common divisor of the packed `a` and `b` by the half gcd on the packed products,
/// packed and trimmed.
pub(crate) fn gcd(a: &[u64], b: &[u64]) -> Vec<u64> {
    let (mut x, mut y) = (a.to_vec(), b.to_vec());
    trim(&mut x);
    trim(&mut y);
    super::hgcd::gcd(&Gf2, &x, &y)
}
