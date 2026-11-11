//! `π(x)`, the number of primes up to `x`.

mod dr;
mod hard;
mod phi;
pub(super) mod tables;

use tables::{PiTable, SMALL, SMALL_PI};

/// The `n`-th prime from the table, for `n` up to `π(2^16)`; `None` above.
pub(super) fn small_nth(n: u64) -> Option<u64> {
    if n <= 3 {
        return [None, Some(2), Some(3), Some(5)][n as usize];
    }
    // The last block whose count before it is below `n`.
    let k = SMALL_PI.partition_point(|&(c, _)| c < n).checked_sub(1)?;
    let (before, mut bits) = SMALL_PI[k];
    if before + u64::from(bits.count_ones()) < n {
        return None;
    }
    for _ in 1..n - before {
        bits &= bits - 1;
    }
    Some(240 * k as u64 + tables::BIT_VALUES[bits.trailing_zeros() as usize])
}

/// `π(x)` for every `u64`: from a table below `2^16`, and by Deléglise and Rivat's method above, which
/// takes its tables of the primes and of `π` from those below `2^16` while they reach (measured on
/// an Apple M4 Max: 0.24 µs just above `2^16`, where the sieve's count took 0.76 µs, and primecount's
/// Legendre's formula 0.46 µs).
pub(super) fn prime_pi(x: u64) -> u64 {
    if x < SMALL {
        return PiTable::small().pi(x);
    }
    dr::pi(x, prime_pi)
}
