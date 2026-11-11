//! The primes in an interval of `u64`s, for the methods' bounds: those of the crate's segmented
//! sieve, [`primes`](crate::math::numbers::primes), whose memory follows the square root of how far
//! it has gone, not of where it may end.

/// The primes `p` with `lo <= p <= hi`, in increasing order.
pub(super) fn primes_in(lo: u64, hi: u64) -> crate::math::numbers::Primes {
    crate::math::numbers::primes(lo..=hi)
}

/// The square root of `x`, rounded down.
pub(super) fn isqrt(x: u64) -> u64 {
    let mut r = (x as f64).sqrt() as u64;
    while u128::from(r) * u128::from(r) > u128::from(x) {
        r -= 1;
    }
    while u128::from(r + 1) * u128::from(r + 1) <= u128::from(x) {
        r += 1;
    }
    r
}
