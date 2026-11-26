//! The primes below `2^62`, from the top down, each found once for the whole program: the
//! residue methods of the matrices of integers (determinants, products, normal forms) take them
//! in that order, a few or a few hundred at a time, and a test of a word for primality costs a
//! few microseconds.

use crate::math::numbers::is_prime_u64;
use std::sync::Mutex;

/// The primes found so far, descending from the largest below `2^62`.
static FOUND: Mutex<Vec<u64>> = Mutex::new(Vec::new());

/// The `i`-th prime below `2^62` from the top (`i = 0` the largest), found once.
pub(in crate::math::matrix) fn top_prime(i: usize) -> u64 {
    let mut found = FOUND.lock().unwrap_or_else(|e| e.into_inner());
    while found.len() <= i {
        let mut q = found.last().map_or((1 << 62) + 1, |&p| p) - 2;
        while !is_prime_u64(q) {
            q -= 2;
        }
        found.push(q);
    }
    found[i]
}

/// The primes below `2^62` from the top down, as [`top_prime`] gives them.
#[derive(Clone, Debug, Default)]
pub(in crate::math::matrix) struct TopPrimes(usize);

impl Iterator for TopPrimes {
    type Item = u64;

    fn next(&mut self) -> Option<u64> {
        let p = top_prime(self.0);
        self.0 += 1;
        Some(p)
    }
}
