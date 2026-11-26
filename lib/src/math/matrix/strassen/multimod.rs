//! The product of matrices of long numbers by their residues: `A·B mod p` for word primes `p`
//! (each above `2^61`) by the product of [`ModMatrix`](super::super::ModMatrix)'s kernels, as many
//! primes as make their product pass twice the bound `k·max|a|·max|b|` on every element, then
//! each element rebuilt by the Chinese remainder theorem, one prime at a time (Garner's
//! incremental form: `x ← x + P·((r − x)·P⁻¹ mod p)`, `P ← P·p`), and taken as the residue of
//! least magnitude. Nothing is guessed: the primes stop at the bound, so every element is exact.
//!
//! For `m × k` by `k × n` of at most `L` limbs and `c ≈ 2·64·L/61` primes, the work is the
//! residues, `(m·k + k·n)·L·c` products of words, the products modulo the primes, `c·m·k·n`, and
//! the rebuilding, about `m·n·c²`, against `m·k·n·L²` for the sums of products of limbs: it pays
//! once the order is large beside the length ([`pays`]).

use super::super::modp::{Crt, Field, ProductMode, TopPrimes, product_mod};
use super::limbs::Limbs;
use crate::math::numbers::Number;

/// The bits each prime contributes for certain: each is above `2^61`.
const PRIME_BITS: u64 = 61;

/// Whether the residues cost less than the sums of products of limbs, for `m × k` by `k × n`
/// with magnitudes of at most `la` and `lb` limbs.
///
/// Measured on an Apple M4 Max in the release profile, the two methods forced, `n` from 4 to
/// 128 for 2 to 128 limbs, square, thin and wide shapes, and elements of mixed lengths: where
/// every element has the longest length, the residues pay from far smaller orders than these
/// weights say (from 8 rows for 2 limbs to 23 for 128; 0.26 to 0.60 of the time from 32 rows),
/// but the sums of products of limbs cost the elements' own lengths, which `pays` does not see:
/// with one long element or one long column among words (a knapsack basis), the residues took
/// 1.3 to 21 times the time from 6 limbs on, up to 96 rows. Every set of weights that moved the
/// choice towards the residues for the first made the second more than 2 % slower somewhere, so
/// the weights and the margin of 2 are kept.
fn pays(m: usize, k: usize, n: usize, la: usize, lb: usize, primes: usize) -> bool {
    let (m, k, n, la, lb, c) = (
        m as f64,
        k as f64,
        n as f64,
        la as f64,
        lb as f64,
        primes as f64,
    );
    let direct = m * k * n * la * lb;
    let residues = (m * k * la + k * n * lb) * c;
    let products = c * m * k * n;
    let rebuild = 2.0 * m * n * c * c;
    2.0 * (residues + products + rebuild) < direct
}

/// `c = a·b` for column-major `a` of `m × k`, `b` of `k × n` and `c` of `m × n` (which holds
/// zeros) by residues, when [`pays`] says so; `false`, with `c` untouched, otherwise.
pub(super) fn product_by_residues(
    c: &mut [Number],
    a: &[Number],
    b: &[Number],
    m: usize,
    k: usize,
    n: usize,
) -> bool {
    let la = Limbs::new(m * k, |e| &a[e]);
    let lb = Limbs::new(k * n, |e| &b[e]);
    if la.longest == 0 || lb.longest == 0 {
        return true;
    }
    // |c_ij| ≤ k·max|a|·max|b| < 2^(64·(la + lb) + bits(k)); the primes' product passes twice it.
    let bound = 64 * (la.longest + lb.longest) as u64 + u64::from(usize::BITS - k.leading_zeros());
    let count = (bound + 2).div_ceil(PRIME_BITS) as usize;
    if !pays(m, k, n, la.longest, lb.longest, count) {
        return false;
    }
    // The primes below 2^62, from the top down: each above 2^61.
    let primes: Vec<u64> = TopPrimes::default().take(count).collect();
    let len = la.longest.max(lb.longest);
    // Each element of the result as a residue modulo the product of the primes so far, built up
    // one prime at a time.
    let mn = m * n;
    let mut crt = Crt::new(mn, count);
    let (mut ra, mut rb, mut rc) = (vec![0u64; m * k], vec![0u64; k * n], vec![0u64; mn]);
    for &p in &primes {
        let f = Field::new(p);
        let powers = f.powers(len.max(crt.width()));
        for (e, r) in ra.iter_mut().enumerate() {
            let (neg, d) = la.get(e);
            *r = f.of_limbs_by(neg, d, &powers);
        }
        for (e, r) in rb.iter_mut().enumerate() {
            let (neg, d) = lb.get(e);
            *r = f.of_limbs_by(neg, d, &powers);
        }
        // Column-major `C = A·B` is row-major `Cᵀ = Bᵀ·Aᵀ`, the column-major operands as they are.
        product_mod(f, &mut rc, m, &rb, k, &ra, m, (n, k, m), ProductMode::Set);
        crt.add(f, p, &rc, &powers);
    }
    crt.finish_into(c);
    true
}
