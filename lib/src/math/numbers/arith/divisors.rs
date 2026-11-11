//! The divisors of a long number from its factorisation, in limbs.
//!
//! Each divisor is the product of one generated before it and a prime, written in limbs to the end
//! of one buffer, with its logarithm in base 2, the sum of its parent's and the prime's, as a key.
//! The keys of distinct divisors differ, and those that the floating point cannot tell apart are
//! far rarer than not: the divisors are sorted by the keys, which are floats compared as integers,
//! then each run of keys closer together than their rounding errors could be is sorted again by the
//! limbs. Each divisor is then written as a number once, in that order.

use super::super::modular::ring::{from_limbs, read_limbs};
use super::super::mul::limbs::mul;
use super::super::scratch::Scratch;
use super::super::{ByteOrder, Number};
use core::cmp::Ordering;

/// Below this many limbs a prime's products are the schoolbook kernels', which need no working
/// memory: Karatsuba's method takes over from 24.
pub(super) const PRIME_LIMBS: usize = 23;

/// From this many divisors on, they are made here; below, the fixed costs of the buffers and the
/// keys pass the products' savings: measured on an Apple M4 Max, the divisors of primes of 64 to
/// 200 bits took 1.24 to 1.84 times the time of the products of numbers for 4 to 24 of them,
/// 0.95 to 1.13 for 32, and 0.84 to 0.88 from 48.
pub(super) const LONG_MIN: usize = 40;

/// The limbs of `x`, not zero.
fn limbs_of(x: &Number) -> Vec<u64> {
    let n = x.significant().div_ceil(8);
    let mut v: Vec<u64> = Vec::with_capacity(n);
    read_limbs(x, &mut v.spare_capacity_mut()[..n]);
    // SAFETY: `read_limbs` wrote the `n` limbs.
    unsafe { v.set_len(n) };
    v
}

/// `log2` of the limbs `x`, the top one not zero, from their top 64 bits.
fn log2(x: &[u64]) -> f64 {
    let n = x.len();
    let top = x[n - 1];
    let z = top.leading_zeros();
    let bits = if z == 0 || n == 1 {
        top
    } else {
        (top << z) | (x[n - 2] >> (64 - z))
    };
    let shift = if n == 1 {
        0
    } else {
        64 * (n - 1) as u64 - u64::from(z)
    };
    (bits as f64).log2() + shift as f64
}

/// The divisors of `prod p^e`, every prime of fewer than [`PRIME_LIMBS`] limbs, in increasing
/// order, in the byte order `order`.
pub(super) fn divisors_long(f: &[(Number, u32)], order: ByteOrder) -> Vec<Number> {
    // Counted saturated: a count past `usize::MAX` fails to allocate, as it must.
    let total = f.iter().fold(1usize, |n, (_, e)| {
        n.saturating_mul((*e as usize).saturating_add(1))
    });
    let primes: Vec<(Vec<u64>, f64)> = f
        .iter()
        .map(|(p, _)| {
            let l = limbs_of(p);
            let lg = log2(&l);
            (l, lg)
        })
        .collect();
    // The limbs of the divisors, one after the other, where each starts and how long it is, and
    // its key.
    // About half the number's limbs a divisor, on average.
    let half: usize = primes
        .iter()
        .zip(f)
        .map(|((l, _), (_, e))| l.len().saturating_mul(*e as usize))
        .fold(0usize, usize::saturating_add)
        / 2;
    // Reserved up to a megabyte or so, then grown: the count may be past what memory holds.
    let mut limbs: Vec<u64> = Vec::with_capacity(total.saturating_mul(half + 1).min(1 << 17));
    limbs.push(1);
    let mut spans: Vec<(usize, usize)> = Vec::with_capacity(total);
    let mut keys: Vec<f64> = Vec::with_capacity(total);
    spans.push((0, 1));
    keys.push(0.0);
    let mut steps = 0u64;
    for ((pl, lg), (_, e)) in primes.iter().zip(f) {
        let len = spans.len();
        let k = pl.len();
        steps += u64::from(*e);
        for j in 0..(*e as usize).saturating_mul(len) {
            let (off, n) = spans[j];
            let start = limbs.len();
            limbs.resize(start + n + k, 0);
            let (old, new) = limbs.split_at_mut(start);
            let d = &old[off..off + n];
            let ((x, nx), (y, ny)) = if n >= k {
                ((d.as_ptr(), n), (pl.as_ptr(), k))
            } else {
                ((pl.as_ptr(), k), (d.as_ptr(), n))
            };
            // SAFETY: the prime has fewer than 24 limbs and is the shorter operand, or the divisor
            // is, so the product is the schoolbook kernels', which need no working memory; its
            // `n + k` limbs are the new ones, apart from both operands.
            unsafe { mul(new.as_mut_ptr(), x, nx, y, ny, Scratch::none()) };
            let m = if new[n + k - 1] == 0 {
                n + k - 1
            } else {
                n + k
            };
            limbs.truncate(start + m);
            spans.push((start, m));
            keys.push(keys[j] + lg);
        }
    }
    // Sorted by the keys, as their bits: positive floats are in the order of their bits.
    let mut sorted: Vec<(u64, usize)> = keys
        .iter()
        .enumerate()
        .map(|(i, k)| (k.to_bits(), i))
        .collect();
    sorted.sort_unstable();
    // Each key is a sum of at most `steps` logarithms of primes, each rounded, and of as many
    // additions, each off by half a unit of the last place of the largest key at most: keys closer
    // than this may be out of order, and their runs are sorted by the limbs.
    let largest = keys.iter().copied().fold(1.0f64, f64::max);
    let eps = (steps as f64 + 4.0) * largest * f64::EPSILON * 16.0;
    let cmp = |a: usize, b: usize| -> Ordering {
        let (oa, na) = spans[a];
        let (ob, nb) = spans[b];
        na.cmp(&nb).then_with(|| {
            limbs[oa..oa + na]
                .iter()
                .rev()
                .cmp(limbs[ob..ob + nb].iter().rev())
        })
    };
    let mut i = 0;
    while i + 1 < sorted.len() {
        let mut end = i + 1;
        while end < sorted.len()
            && f64::from_bits(sorted[end].0) - f64::from_bits(sorted[end - 1].0) <= eps
        {
            end += 1;
        }
        if end - i > 1 {
            sorted[i..end].sort_unstable_by(|x, y| cmp(x.1, y.1));
        }
        i = end;
    }
    sorted
        .iter()
        .map(|&(_, i)| {
            let (off, n) = spans[i];
            from_limbs(&limbs[off..off + n], order)
        })
        .collect()
}
