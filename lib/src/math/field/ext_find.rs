//! The polynomial [`ExtensionField::with_degree`](super::ExtensionField::with_degree) chooses:
//! the smallest monic irreducible polynomial of the degree in the order of the coefficients from
//! the top down.
//!
//! The binomials `xᵏ + c` come first in that order. Lidl and Niederreiter's Theorem 3.75 (*Finite
//! Fields*) says when one is irreducible: for `k ≥ 2`, `xᵏ − a` is irreducible over `GF(p)` if and
//! only if every prime factor `r` of `k` divides the order of `a` but not `(p − 1)/ord(a)`, and
//! `p ≡ 1 (mod 4)` when `4 | k`; the first condition is that each `r` divides `p − 1` and
//! `a^((p − 1)/r) ≠ 1`. That settles all `p` binomials without testing them, so that the search
//! goes on to `xᵏ + x + c`, `c = 1, 2, …`, and on, each tested by Ben-Or's test
//! ([`ModPolynomial::is_irreducible`]), the constant 0 skipped (`x` divides those). For odd `p`
//! the discriminant sorts out about half of the candidates first: by Stickelberger's theorem
//! (Swan, *Factorization of polynomials over finite fields*, Pacific J. Math. 12, 1962), a
//! square-free polynomial of degree `k` with `r` irreducible factors has a discriminant that is a
//! square modulo `p` exactly when `k − r` is even, so an irreducible one has the Legendre symbol
//! `(−1)^(k−1)`, and a candidate without it is reducible.

use super::PrimeField;
use crate::math::numbers::Number;
use crate::math::poly::ModPolynomial;

/// The prime factors of `k`, increasing.
fn prime_factors(mut k: usize) -> Vec<usize> {
    let mut v = Vec::new();
    let mut r = 2;
    while r * r <= k {
        if k.is_multiple_of(r) {
            v.push(r);
            while k.is_multiple_of(r) {
                k /= r;
            }
        }
        r += 1;
    }
    if k > 1 {
        v.push(k);
    }
    v
}

/// The smallest `c` with `xᵏ + c` irreducible modulo `p`, `k ≥ 2`, or `None` when no binomial of
/// degree `k` is.
fn binomial(base: &PrimeField, k: usize) -> Option<u64> {
    let p = base.modulus();
    let one = Number::from(1u8);
    let pm1 = p - &one;
    let primes = prime_factors(k);
    if primes
        .iter()
        .any(|&r| !pm1.rem_euclid(&Number::from(r as u64)).is_zero())
    {
        return None;
    }
    if k.is_multiple_of(4) && p.rem_euclid(&Number::from(4u8)) != one {
        return None;
    }
    let exps: Vec<Number> = primes
        .iter()
        .map(|&r| &pm1 / &Number::from(r as u64))
        .collect();
    // Each prime r divides p − 1, so non-r-th powers exist and a fraction Π(1 − 1/r) of the
    // residues will do: the search ends early.
    (1u64..).find(|&c| {
        let a = -base.from_u64(c);
        !a.is_zero() && exps.iter().all(|e| !a.pow(e).is_one())
    })
}

/// The smallest monic irreducible polynomial of degree `k` modulo `p`.
#[track_caller]
pub(super) fn first_irreducible(base: &PrimeField, k: usize) -> ModPolynomial {
    assert!(k >= 1, "the degree must be at least 1");
    let p = base.modulus();
    let zero = ModPolynomial::zero(p);
    let mut c = vec![Number::default(); k + 1];
    c[k] = Number::from(1u8);
    if k == 1 {
        return zero.with_coeffs(&c);
    }
    if let Some(b) = binomial(base, k) {
        c[0] = Number::from(b);
        return zero.with_coeffs(&c);
    }
    // The coefficients of x, …, x^(k−1) as an odometer, the lowest the fastest, from x; for each,
    // the constant from 1 up.
    c[1] = Number::from(1u8);
    let odd = *p != 2u8;
    let symbol = if k % 2 == 1 { 1 } else { -1 };
    loop {
        let mut c0 = Number::from(1u8);
        while &c0 < p {
            c[0] = c0.clone();
            let g = zero.with_coeffs(&c);
            if (!odd || base.element(&g.discriminant()).legendre() == symbol) && g.is_irreducible()
            {
                return g;
            }
            c0 += &Number::from(1u8);
        }
        let mut i = 1;
        loop {
            c[i] += &Number::from(1u8);
            if &c[i] < p {
                break;
            }
            c[i] = Number::default();
            i += 1;
            // Irreducible polynomials of every degree exist, so the odometer does not run past
            // x^(k−1) before one is found.
            assert!(i < k, "an irreducible polynomial of every degree exists");
        }
    }
}
