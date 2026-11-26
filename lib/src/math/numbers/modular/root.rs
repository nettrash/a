//! The smallest primitive root modulo `p^k` and `2 p^k`.

use super::int::{Int, Limbs};
use super::ring::{Ring, with_odd};

/// The Jacobi symbol `(g / p)` for a word `g` and an odd `p` of the limbs `pl`: the factors of two
/// of `g` by `p mod 8`, then reciprocity, which needs `p` only modulo the odd part of `g`.
pub(in super::super) fn jacobi_of_word(g: u64, pl: &[u64]) -> i32 {
    if g == 0 {
        return if pl == [1] { 1 } else { 0 };
    }
    let p0 = pl[0];
    let z = g.trailing_zeros();
    let h = g >> z;
    let mut t = 1;
    if z & 1 == 1 && matches!(p0 & 7, 3 | 5) {
        t = -t;
    }
    if h == 1 {
        return t;
    }
    // `(h / p) = (p / h)`, less when both are 3 modulo 4.
    if h & 3 == 3 && p0 & 3 == 3 {
        t = -t;
    }
    let r = pl.iter().rev().fold(0u64, |r, &l| {
        (((u128::from(r) << 64) | u128::from(l)) % u128::from(h)) as u64
    });
    t * super::jacobi::jacobi_word(r, h)
}

/// The smallest primitive root modulo `p^k`, `k >= 1`, or modulo `2 p^k` when `two`, for the odd
/// prime `p` and the primes `p1` of `p - 1`.
///
/// A primitive root modulo `p` is a `g` whose powers `g^((p - 1) / q)` are not 1 for any prime `q`
/// of `p - 1`; for `q = 2` that is Euler's criterion, which the Jacobi symbol gives without a
/// power. One modulo `p` is one modulo every `p^k` exactly when `g^(p - 1)` is not 1 modulo `p^2`.
/// Modulo `2 p^k` the roots are the odd roots modulo `p^k`. The candidates are tried in increasing
/// order, the cheap tests first; the smallest root is below `p^(1/4 + ε)` (Burgess), and in
/// practice small.
pub(in super::super) fn smallest<I: Int>(p: &I, k: u32, two: bool, p1: &[(I, u32)]) -> u64 {
    let pl = p.limbs();
    let pm1 = p.sub(&I::small(1));
    // The exponents `(p - 1) / q` for the odd primes `q`, the smallest first, which most
    // candidates fail first.
    let exps: Vec<Limbs> = p1
        .iter()
        .filter(|(q, _)| *q != I::small(2))
        .map(|(q, _)| pm1.div_exact(q).limbs())
        .collect();
    let p2 = (k >= 2).then(|| p.mul(p));
    let p_word = p.word();
    let lifts = |g: u64| -> bool {
        match &p2 {
            None => true,
            Some(p2) => with_odd!(p2.limbs().as_slice(), |r| {
                let x = r.residue(&[g]);
                !r.is_one(&r.pow(&x, pm1.limbs().as_slice()))
            }),
        }
    };
    with_odd!(pl.as_slice(), |r| {
        let (mut g, step) = if two { (3u64, 2) } else { (2u64, 1) };
        loop {
            let divisible = p_word.is_some_and(|p| g.is_multiple_of(p));
            if !divisible
                && jacobi_of_word(g, pl.as_slice()) == -1
                && {
                    let x = r.residue(&[g]);
                    exps.iter().all(|e| !r.is_one(&r.pow(&x, e.as_slice())))
                }
                && lifts(g)
            {
                return g;
            }
            g += step;
        }
    })
}
