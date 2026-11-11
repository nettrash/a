//! The multiplier and the factor base.
//!
//! Knuth and Schroeppel's function (Knuth, *The Art of Computer Programming*, vol. 2, 4.5.4,
//! exercise 4.5.4–12; Silverman 1987) scores each square-free multiplier `k` by how much the small
//! primes are expected to contribute to the values of the polynomials for `kN`, less half the
//! logarithm of `k`, by which they grow: 2 by `kN mod 8` (and a bit more when `kN ≡ 1 (mod 8)`, as
//! the polynomials `(2Ax + B)² − kN` are then half as large), each odd prime `p` by `2 ln p/(p − 1)`
//! when `kN` is a square modulo `p` (two roots) and `ln p/(p − 1)` when `p` divides `k` (one).
//! The factor base is then 2 and the odd primes modulo which `kN` is a square, with a square root
//! of `kN` modulo each.

use super::super::super::factorial::primes::Sieve;
use super::super::super::prime::jacobi;
use super::arith::{inverse_64, mul_mod, pow_mod, rem_1, sqrt_mod};

/// The multipliers tried: the square-free numbers below 100, as msieve and yafu take.
const MULTIPLIERS: [u32; 61] = [
    1, 2, 3, 5, 6, 7, 10, 11, 13, 14, 15, 17, 19, 21, 22, 23, 26, 29, 30, 31, 33, 34, 35, 37, 38,
    39, 41, 42, 43, 46, 47, 51, 53, 55, 57, 58, 59, 61, 62, 65, 66, 67, 69, 70, 71, 73, 74, 77, 78,
    79, 82, 83, 85, 86, 87, 89, 91, 93, 94, 95, 97,
];

/// The odd primes the multipliers are scored with: those below 2000 at most.
const SCORE_PRIMES_TO: u64 = 2000;

/// The primes of the multipliers.
const SMALL: [u32; 25] = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
];

/// The multiplier with the best score for the odd number `n` (limbs, least significant first),
/// which has no prime factor below 2000.
pub(super) fn multiplier(n: &[u64]) -> u32 {
    let n8 = (n[0] & 7) as u32;
    let mut scores = [0f64; MULTIPLIERS.len()];
    for (s, &k) in scores.iter_mut().zip(MULTIPLIERS.iter()) {
        let ln2 = core::f64::consts::LN_2;
        *s = -0.5 * f64::from(k).ln()
            + match (k * n8) & 7 {
                // Two bits of 2 on average, and a bit more from the polynomials (2Ax + B)².
                1 => 3.0 * ln2,
                5 => ln2,
                3 | 7 => 0.5 * ln2,
                _ => 0.0,
            };
    }
    // The symbol (kN/p) is (k/p)(N/p), and (k/p) the product of the symbols of the primes of the
    // square-free k: a mask of the primes of each multiplier, and for each p the masks of the
    // primes q with (q/p) = −1 and q = p.
    let masks: [u32; MULTIPLIERS.len()] = core::array::from_fn(|i| {
        let mut m = 0;
        for (b, &q) in SMALL.iter().enumerate() {
            if MULTIPLIERS[i].is_multiple_of(q) {
                m |= 1 << b;
            }
        }
        m
    });
    // Primes to about eight times the bits: the score is a sum over them, whose terms fall off
    // as 1/p, and for small numbers the choice matters less than the time it takes.
    let bound = (8 * u64::from(super::sieve::bits_of(n))).clamp(300, SCORE_PRIMES_TO);
    // The squares modulo each odd prime of the multipliers, a bit each: by reciprocity
    // (q/p) = (p/q), less when p and q are both 3 modulo 4, and (p/q) is the bit of p mod q.
    let squares: [u128; SMALL.len()] = core::array::from_fn(|b| {
        let q = u128::from(SMALL[b]);
        (1..q).fold(0, |m, x| m | 1 << (x * x % q))
    });
    let sieve = Sieve::new(bound);
    sieve.each(3, bound, |p| {
        let p = p as u32;
        let contribution = f64::from(p).ln() / f64::from(p - 1);
        let np = jacobi(u64::from(rem_1(n, p)), u64::from(p));
        let (mut minus, mut zero) = (0u32, 0u32);
        for (b, &q) in SMALL.iter().enumerate() {
            let negative = if q == 2 {
                // (2/p) = 1 for p ≡ ±1 (mod 8).
                !matches!(p & 7, 1 | 7)
            } else if q == p {
                zero |= 1 << b;
                false
            } else {
                (squares[b] >> (p % q) & 1 == 0) ^ (p & q & 3 == 3)
            };
            minus |= u32::from(negative) << b;
        }
        // (kN/p) = 1 when the primes of k with (q/p) = −1 are as many, modulo 2, as (N/p) asks
        // (none when p divides N).
        let odd = match np {
            1 => 0,
            -1 => 1,
            _ => 2,
        };
        for (s, &m) in scores.iter_mut().zip(masks.iter()) {
            let one = u32::from(m & zero != 0);
            let two = u32::from((m & minus).count_ones() & 1 == odd) & (1 - one);
            *s += contribution * f64::from(one + 2 * two);
        }
    });
    let mut best = 0;
    for i in 1..MULTIPLIERS.len() {
        if scores[i] > scores[best] {
            best = i;
        }
    }
    MULTIPLIERS[best]
}

/// The factor base: 2 and the odd primes modulo which `kN` is a square, increasing.
pub(super) struct FactorBase {
    /// The primes.
    pub(super) prime: Vec<u32>,
    /// A square root of `kN` modulo each prime (`kN mod 2` for 2, 0 for the primes of `k`).
    pub(super) sqrt: Vec<u32>,
    /// The inverse of each odd prime modulo `2^64` (1 for 2).
    pub(super) inv64: Vec<u64>,
    /// The index after the last prime that divides the multiplier (1 when it is 1 or 2).
    pub(super) k_end: usize,
}

/// What building the factor base gives.
pub(super) enum Built {
    /// The factor base.
    Base(FactorBase),
    /// A prime that divides `n`.
    Factor(u32),
}

/// The factor base of `size` primes for `kN`, the multiplier `k` and the odd `n` (limbs), or a prime
/// of the base that divides `n`.
pub(super) fn build(n: &[u64], k: u32, size: usize) -> Built {
    let size = size.max(8);
    // The primes modulo which kN is a square are about half of all: sieve to about the size of
    // the `2·size`-th prime, more if it is not enough.
    let mut limit = {
        let m = (2 * size) as f64;
        (m * (m.ln() + m.ln().ln()) * 1.1 + 100.0) as u64
    };
    loop {
        let mut fb = FactorBase {
            prime: Vec::with_capacity(size),
            sqrt: Vec::with_capacity(size),
            inv64: Vec::with_capacity(size),
            k_end: 1,
        };
        fb.prime.push(2);
        fb.sqrt.push(k & 1);
        fb.inv64.push(1);
        let mut found = None;
        let sieve = Sieve::new(limit);
        sieve.each(3, limit, |p| {
            if found.is_some() || fb.prime.len() >= size {
                return;
            }
            let p = p as u32;
            let rn = rem_1(n, p);
            if rn == 0 {
                found = Some(p);
                return;
            }
            let r = mul_mod(k % p, rn, p);
            let t = if r == 0 {
                fb.k_end = fb.prime.len() + 1;
                0
            } else if p & 3 == 3 {
                // The root, if any, is r^((p + 1)/4), and its square tells whether it is one.
                let t = pow_mod(r, (p + 1) / 4, p);
                if mul_mod(t, t, p) != r {
                    return;
                }
                t
            } else if jacobi(u64::from(r), u64::from(p)) == 1 {
                sqrt_mod(r, p)
            } else {
                return;
            };
            fb.prime.push(p);
            fb.sqrt.push(t);
            fb.inv64.push(inverse_64(u64::from(p)));
        });
        if let Some(p) = found {
            return Built::Factor(p);
        }
        if fb.prime.len() >= size || limit >= u64::from(u32::MAX) / 2 {
            return Built::Base(fb);
        }
        limit *= 2;
    }
}
