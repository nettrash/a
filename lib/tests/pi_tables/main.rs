//! `prime_pi` and `nth_prime` from `2^16` to `2^20`, in a process of their own.
//!
//! There the two functions answer by the method until 128 calls in all have asked there, and from
//! the tables that the process then builds once and keeps (the call counter and the tables belong
//! to the process). In the `math` binary, other tests running in parallel make those calls too, so
//! which of the two answers a given call there depends on the order of the tests. This binary holds
//! one test only: its first calls are the method's and the later ones the tables', in every run.

use a::math::numbers::{nth_prime, prime_pi};

/// Whether each number below `n` is prime, by a sieve of Eratosthenes.
fn sieve(n: usize) -> Vec<bool> {
    let mut prime = vec![true; n];
    for p in prime.iter_mut().take(2) {
        *p = false;
    }
    let mut i = 2;
    while i * i < n {
        if prime[i] {
            let mut j = i * i;
            while j < n {
                prime[j] = false;
                j += i;
            }
        }
        i += 1;
    }
    prime
}

/// The calls between `2^16` and `2^20` that are answered by the method before the tables are built
/// (`MID_AFTER` in `sieve/pi.rs`): a call of `prime_pi` there counts once, a call of `nth_prime`
/// for an index from `π(2^16) + 1` to `π(2^20)` twice (its own, and that of the `prime_pi` of its
/// method). With a smaller count the first calls below still take the method, the others the tables.
const METHOD_CALLS: usize = 128;

/// The calls of `prime_pi` and of `nth_prime` that the method answers first.
const PI_CALLS: usize = 100;
const NTH_CALLS: usize = 14;

#[test]
fn prime_pi_and_nth_prime_up_to_2_20_match_a_sieve() {
    // Under Miri, the method's calls only, at a few points just above `2^16` (the tables, a sieve
    // of `2^20` numbers, would take too long).
    let limit: usize = if cfg!(miri) { (1 << 16) + 512 } else { 1 << 20 };
    let s = sieve(limit + 1);
    let mut pi = vec![0u64; limit + 1];
    let mut count = 0u64;
    let mut primes = vec![0u64];
    for (x, &is_p) in s.iter().enumerate() {
        if is_p {
            count += 1;
            primes.push(x as u64);
        }
        pi[x] = count;
    }
    // The first calls, the method's: `prime_pi` at the edges of the range and at primes and their
    // predecessors spread over it, then `nth_prime` at indices spread over its own range.
    let (pi_calls, nth_calls) = if cfg!(miri) {
        (3, 1)
    } else {
        (PI_CALLS, NTH_CALLS)
    };
    assert!(
        pi_calls + 2 * nth_calls <= METHOD_CALLS,
        "the first calls are the method's"
    );
    let lo = 1u64 << 16;
    let hi = limit as u64;
    let mut xs = vec![lo, lo + 1, hi - 1];
    let first = primes.partition_point(|&p| p < lo);
    let span = primes.len() - first;
    while xs.len() < pi_calls {
        let p = primes[first + (xs.len() * 7919) % span];
        xs.push(if xs.len() % 2 == 0 { p } else { p - 1 });
    }
    for &x in &xs[..pi_calls] {
        assert_eq!(prime_pi(x), pi[x as usize], "pi({x}), by the method");
    }
    for j in 0..nth_calls {
        let n = (first + 1 + (j * 5711 + 17) % (span - 1)) as u64;
        assert_eq!(
            nth_prime(n),
            primes[n as usize],
            "nth_prime({n}), by the method"
        );
    }
    if cfg!(miri) {
        return;
    }
    // Past the count: every later call reads the tables. At every prime and just below it, and at
    // its index.
    for (n, &p) in primes.iter().enumerate().skip(first) {
        assert_eq!(prime_pi(p), n as u64, "pi({p}), from the table");
        assert_eq!(
            prime_pi(p - 1),
            n as u64 - 1,
            "pi({}), from the table",
            p - 1
        );
        assert_eq!(nth_prime(n as u64), p, "nth_prime({n}), from the table");
    }
    assert_eq!(primes.len() - 1, 82_025);
    assert_eq!(prime_pi((1 << 20) - 1), 82_025);
    // `2^20` and the index past the table take the method always.
    assert_eq!(prime_pi(1 << 20), 82_025);
    assert_eq!(nth_prime(82_026), 1_048_583);
}
