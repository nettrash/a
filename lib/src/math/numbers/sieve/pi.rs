//! `π(x)`, the number of primes up to `x`.

mod dr;
mod hard;
mod phi;
mod tables;

use tables::{BIT_OF, UP_TO};

/// The numbers below which `π` comes from [`SMALL_BITS`].
const TABLE: u64 = 1 << 16;

/// The numbers below which `π` comes from the sieve, above which from Deléglise and Rivat's
/// method: measured, both take about 2.5 µs at `2·10^5`, the sieve 1.2 µs at `10^5` and 4 µs
/// at `3·10^5`, the method 2.0 and 2.6 µs.
const SIEVE: u64 = 200_000;

/// The primes below [`TABLE`] by blocks of 240 numbers, a bit for each number prime to 30 (as the
/// tables of `pi` place them), computed at compile time.
static SMALL_BITS: [u64; (TABLE / 240 + 1) as usize] = {
    const N: usize = TABLE as usize;
    let mut composite = [false; N];
    let mut i = 2;
    while i * i < N {
        if !composite[i] {
            let mut j = i * i;
            while j < N {
                composite[j] = true;
                j += i;
            }
        }
        i += 1;
    }
    let mut t = [0u64; (TABLE / 240 + 1) as usize];
    let mut n = 7;
    while n < N {
        if !composite[n] && BIT_OF[n % 240] < 64 {
            t[n / 240] |= 1 << BIT_OF[n % 240];
        }
        n += 1;
    }
    t
};

/// The primes before each block of [`SMALL_BITS`], 2, 3 and 5 included.
static SMALL_COUNTS: [u16; (TABLE / 240 + 1) as usize] = {
    let mut t = [0u16; (TABLE / 240 + 1) as usize];
    let mut count = 3u16;
    let mut k = 0;
    while k < t.len() {
        t[k] = count;
        count += SMALL_BITS[k].count_ones() as u16;
        k += 1;
    }
    t
};

/// The `n`-th prime from the table, for `n` up to `π(2^16)`; `None` above.
pub(super) fn small_nth(n: u64) -> Option<u64> {
    if n <= 3 {
        return [None, Some(2), Some(3), Some(5)][n as usize];
    }
    // The last block whose count before it is below `n`.
    let k = SMALL_COUNTS
        .partition_point(|&c| u64::from(c) < n)
        .checked_sub(1)?;
    let mut bits = SMALL_BITS[k];
    let before = u64::from(SMALL_COUNTS[k]);
    if before + u64::from(bits.count_ones()) < n {
        return None;
    }
    for _ in 1..n - before {
        bits &= bits - 1;
    }
    Some(240 * k as u64 + tables::BIT_VALUES[bits.trailing_zeros() as usize])
}

/// `π(x)` for every `u64`: from a table below `2^16`, by the sieve below [`SIEVE`], and by
/// Deléglise and Rivat's method above.
pub(super) fn prime_pi(x: u64) -> u64 {
    if x < TABLE {
        if x < 6 {
            return [0, 0, 1, 2, 2, 3][x as usize];
        }
        let k = (x / 240) as usize;
        return u64::from(SMALL_COUNTS[k])
            + u64::from((SMALL_BITS[k] & UP_TO[(x % 240) as usize]).count_ones());
    }
    if x < SIEVE {
        return super::count(0, x);
    }
    dr::pi(x, prime_pi)
}
