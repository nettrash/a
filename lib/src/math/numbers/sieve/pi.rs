//! `π(x)`, the number of primes up to `x`.

mod dr;
mod hard;
mod phi;
pub(super) mod tables;

use core::sync::atomic::{AtomicU32, Ordering};
use std::sync::OnceLock;
use tables::{PiTable, SMALL, SMALL_PI, primes_to};

/// The `n`-th prime from the table, for `n` up to `π(2^16)`; `None` above; from the table of the
/// primes below `2^20` up to `π(2^20)` once it is there.
#[inline]
pub(super) fn small_nth(n: u64) -> Option<u64> {
    if n <= 3 {
        return [None, Some(2), Some(3), Some(5)][n as usize];
    }
    if let Some(p) = nth_in(&SMALL_PI, n) {
        return Some(p);
    }
    mid_nth(n)
}

/// The `n`-th prime from the table of the primes below [`MID`], for `n` above `π(2^16)`.
#[inline(never)]
fn mid_nth(n: u64) -> Option<u64> {
    if n <= MID_COUNT {
        return Some(u64::from(mid_primes()?[n as usize]));
    }
    None
}

/// The `n`-th prime, `n >= 4`, from the blocks of a [`PiTable`], or `None` past their end.
#[inline(always)]
fn nth_in(blocks: &[(u64, u64)], n: u64) -> Option<u64> {
    // The last block whose count before it is below `n`.
    let k = blocks.partition_point(|&(c, _)| c < n).checked_sub(1)?;
    let (before, mut bits) = blocks[k];
    if before + u64::from(bits.count_ones()) < n {
        return None;
    }
    for _ in 1..n - before {
        bits &= bits - 1;
    }
    Some(240 * k as u64 + tables::BIT_VALUES[bits.trailing_zeros() as usize])
}

/// The limit of the table [`mid`] builds, `2^20`, as PARI's default `primelimit`.
const MID: u64 = 1 << 20;

/// `π(2^20)`.
const MID_COUNT: u64 = 82_025;

/// The calls of [`prime_pi`] and `nth_prime` between `2^16` and `2^20` before [`mid`] builds its
/// table: each takes the method (about 0.9 µs for `prime_pi` and 2.7 µs for `nth_prime`), the
/// tables 0.25 to 0.3 ms once (0.16 ms the primes alone, all `nth_prime` needs) and then 2 to 4 ns
/// a call, so that a program which calls them a few times does not pay for it.
// Measured on an Apple M4 Max (random `x` in `[2^16, 2^20)` and `n` in `(π(2^16), π(2^20)]`): the
// tables pay for themselves after about 300 calls of `prime_pi` and 60 of `nth_prime`, so that a
// program which stops just after building them took at most 3.5 times as long as the better
// choice for it with 128 calls for `prime_pi` and 3.2 times for `nth_prime`; the least worst case
// of the two together is at 130 to 140 calls (3.3 to 3.4 times), about the same.
const MID_AFTER: u32 = 128;

/// The calls counted toward [`MID_AFTER`].
static MID_CALLS: AtomicU32 = AtomicU32::new(0);

/// The blocks of `π` up to [`MID`] (4370 of 16 bytes, 70 KB), built from [`MID_PRIMES`] the first
/// time [`mid`] is asked for them after [`MID_AFTER`] calls, and kept for the process.
static MID_PI: OnceLock<Box<[(u64, u64)]>> = OnceLock::new();

/// The table of `π` up to [`MID`], or `None` while fewer than [`MID_AFTER`] calls asked for it.
fn mid() -> Option<&'static [(u64, u64)]> {
    if let Some(t) = MID_PI.get() {
        return Some(t);
    }
    if !mid_due() {
        return None;
    }
    Some(MID_PI.get_or_init(|| PiTable::blocks(MID, mid_primes_now()).into_boxed_slice()))
}

/// The primes below [`MID`] and the one after, `p[i]` the `i`-th (`p[0] = 0`), 328 KB, built from
/// the sieve when `nth_prime` or [`mid`] first needs them after [`MID_AFTER`] calls, and kept.
static MID_PRIMES: OnceLock<Box<[u32]>> = OnceLock::new();

/// The table of the primes below [`MID`], or `None` while fewer than [`MID_AFTER`] calls asked.
fn mid_primes() -> Option<&'static [u32]> {
    if let Some(t) = MID_PRIMES.get() {
        return Some(t);
    }
    if !mid_due() {
        return None;
    }
    Some(mid_primes_now())
}

/// The table of the primes below [`MID`], built now if it is not there yet.
fn mid_primes_now() -> &'static [u32] {
    MID_PRIMES.get_or_init(|| primes_to(MID).into_boxed_slice())
}

/// Whether [`MID_AFTER`] calls have asked for the tables.
fn mid_due() -> bool {
    // Relaxed: the count only decides when to build; the tables are behind their `OnceLock`s.
    MID_CALLS.fetch_add(1, Ordering::Relaxed) >= MID_AFTER
}

/// `π(x)` for every `u64`: from a table below `2^16`, and by Deléglise and Rivat's method above, which
/// takes its tables of the primes and of `π` from those below `2^16` while they reach (measured on
/// an Apple M4 Max: 0.24 µs just above `2^16`, where the sieve's count took 0.76 µs, and primecount's
/// Legendre's formula 0.46 µs).
#[inline]
pub(super) fn prime_pi(x: u64) -> u64 {
    if x < SMALL {
        return PiTable::small().pi(x);
    }
    prime_pi_from_small(x)
}

/// [`prime_pi`] from [`SMALL`] up, apart so that the table's look-up below stays small.
#[inline(never)]
fn prime_pi_from_small(x: u64) -> u64 {
    if x < MID
        && let Some(t) = mid()
    {
        return PiTable::of(t).pi(x);
    }
    dr::pi(x, prime_pi)
}
