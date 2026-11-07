//! Deléglise and Rivat's method (1996), in the form of Oliveira e Silva's "Computing π(x): the
//! combinatorial method" (2006) and of Walisch's primecount:
//!
//! `π(x) = S1 + S2 + π(y) - 1 - P2`, for `y = α x^(1/3)` and `a = π(y)`, where `S1 + S2 = φ(x, a)`,
//! the numbers up to `x` with no prime factor up to `y`, split by the leaves of Legendre's
//! recursion: `S1` the ordinary leaves, `S2` the special ones (trivial, easy and hard), and `P2`
//! the numbers up to `x` with exactly two prime factors above `y`.
//!
//! The sums are in 64 bits that wrap: each term is below `2^64`, the parts may not be, but the
//! result, below `2^63`, is right modulo `2^64`, so exact.

use super::super::segments::{Segments, count_bits, each_prime};
use super::super::wheel::{KEEP_TO, byte_remainder};
use super::hard::s2_hard;
use super::phi::{MAX_A, phi_tiny, tiny_a};
use super::tables::{FactorTable, PiTable, primes_to};

/// The integer cube root of `x`.
pub(super) fn icbrt(x: u64) -> u64 {
    let mut r = (x as f64).cbrt() as u64;
    while r.checked_pow(3).is_none_or(|c| c > x) {
        r -= 1;
    }
    while (r + 1).checked_pow(3).is_some_and(|c| c <= x) {
        r += 1;
    }
    r
}

/// primecount's tuning of `α` for Deléglise and Rivat's method, `y = α x^(1/3)`: a polynomial in
/// `ln x` fitted to its measurements, between 1 and `x^(1/6)`.
fn alpha(x: u64) -> f64 {
    let l = (x as f64).ln();
    let a = if x <= 1_000_000_000 {
        0.078173 * l + 1.0
    } else {
        0.00148918 * l * l * l - 0.0691909 * l * l + 1.00165 * l + 0.372253
    };
    let x16 = (x as f64).powf(1.0 / 6.0);
    a.clamp(1.0, x16.max(1.0))
}

/// `π(x)` by Deléglise and Rivat's method, for `x` from about `10^4`.
pub(super) fn pi(x: u64, pi_of: impl Fn(u64) -> u64) -> u64 {
    let x13 = icbrt(x);
    let y = ((x13 as f64 * alpha(x)) as u64).clamp(x13, x.isqrt());
    let z = x / y;
    let p = primes_to(y);
    let pi_y = (p.len() - 2) as u64;
    let c = tiny_a(y).min(MAX_A);
    let table = PiTable::new(y, &p);
    let factor = FactorTable::new(y, &p);
    let p2 = p2(x, y, pi_y, &pi_of);
    let s1 = s1(x, y, c, &p);
    let trivial = s2_trivial(x, y, z, c, &p, &table);
    let easy = s2_easy(x, y, z, c, &p, &table);
    let hard = s2_hard(x, y, z, c, &p, &table, &factor);
    s1.wrapping_add(trivial)
        .wrapping_add(easy)
        .wrapping_add(hard)
        .wrapping_add(pi_y)
        .wrapping_sub(1)
        .wrapping_sub(p2)
}

/// The ordinary leaves: `Σ μ(n) φ(x / n, c)` over the square-free `n <= y` whose prime factors are
/// all above the `c`-th prime, `n = 1` included.
#[inline(never)]
fn s1(x: u64, y: u64, c: u64, p: &[u32]) -> u64 {
    /// The leaves `n m` for the `m` of primes after the `b`-th, `mu` the sign of `n m`.
    fn leaves(x: u64, y: u64, c: u64, p: &[u32], b: usize, n: u64, mu: bool) -> u64 {
        let mut s = 0u64;
        for (i, &q) in p.iter().enumerate().skip(b + 1) {
            let m = n * u64::from(q);
            if m > y {
                break;
            }
            let leaf = phi_tiny(x / m, c);
            s = if mu {
                s.wrapping_add(leaf)
            } else {
                s.wrapping_sub(leaf)
            };
            s = s.wrapping_add(leaves(x, y, c, p, i, m, !mu));
        }
        s
    }
    let pi_y = p.len() - 2;
    let mut s = phi_tiny(x, c);
    for b in c as usize + 1..=pi_y {
        let q = u64::from(p[b]);
        s = s.wrapping_sub(phi_tiny(x / q, c));
        s = s.wrapping_add(leaves(x, y, c, p, b, q, true));
    }
    s
}

/// The trivial special leaves `p_b p_l`, `l > b`, whose `φ(x / (p_b p_l), b - 1)` is 1: for the
/// primes `p_b` from `sqrt(z)` up to `y`, `x / p_b^2 < p_l <= y`.
#[inline(never)]
fn s2_trivial(x: u64, y: u64, z: u64, c: u64, p: &[u32], pi: &PiTable) -> u64 {
    let pi_y = pi.pi(y);
    let start = (u64::from(p[c as usize])).max(z.isqrt()) + 1;
    if start >= y {
        return 0;
    }
    let mut s = 0u64;
    let mut b = pi.pi(start - 1) as usize + 1;
    while u64::from(p[b]) < y {
        let q = u64::from(p[b]);
        let xpp = x / (q * q);
        if xpp <= q {
            // From here on `x / p^2 <= p`: each prime `p_b < y` has `pi(y) - b` trivial leaves, an
            // arithmetic progression.
            let n = pi.pi(y - 1) - b as u64 + 1;
            let first = pi_y - b as u64;
            let last = pi_y - pi.pi(y - 1);
            return s.wrapping_add(((u128::from(n) * u128::from(first + last)) / 2) as u64);
        }
        s = s.wrapping_add(pi_y - pi.pi(xpp));
        b += 1;
    }
    s
}

/// The easy special leaves `p_b q`, `q` a prime above `p_b` with `x / (p_b q) <= y`, whose
/// `φ(x / (p_b q), b - 1)` is `π(x / (p_b q)) - b + 2`, for the primes `p_b` from `sqrt(y)` to
/// `x^(1/3)`.
///
/// The leaves `q` up to `sqrt(x / p_b)` (sparse: each has a value of its own) are summed one by
/// one. Those above (clustered: runs of `q` share a value), which primecount sums by runs, each
/// run's end found from its value, a chain of dependent divisions and look-ups, are summed by
/// swapping the order of the pairs `(q, r)` of primes with `q r <= x / p_b`:
/// `Σ_{Q < q <= T} π(N / q) = π(N / T) (π(T) - π(Q)) + Σ_{N / T < r <= N / Q} (π(N / r) - π(Q))`.
/// Its primes `r` run over the sparse leaves' `q` (both from `z / p_b` to `sqrt(N)`), so that one
/// division and one look-up serve both sums, and the terms are independent of one another.
#[inline(never)]
fn s2_easy(x: u64, y: u64, z: u64, c: u64, p: &[u32], pi: &PiTable) -> u64 {
    let x13 = icbrt(x);
    let pi_sqrty = pi.pi(y.isqrt());
    let pi_x13 = pi.pi(x13.min(y));
    let mut s = 0u64;
    for b in c.max(pi_sqrty) + 1..=pi_x13 {
        let prime = u64::from(p[b as usize]);
        let xp = x / prime;
        // The leaves `q` from `s_lo` (exclusive) to `q_lo` are sparse, those to `q_hi` clustered.
        let q_hi = (xp / prime).min(y);
        let s_lo = (z / prime).clamp(prime, y);
        let q_lo = xp.isqrt().clamp(prime, y).max(s_lo);
        let pi_q_lo = pi.pi(q_lo);
        let clustered = pi.pi(q_hi).saturating_sub(pi_q_lo);
        // The sparse leaves' indices, and the swapped sum's.
        let (a1, b1) = (pi.pi(s_lo), pi_q_lo);
        let (a2, b2) = if clustered > 0 {
            (pi.pi(xp / q_hi), pi.pi(xp / q_lo))
        } else {
            (0, 0)
        };
        let (lo, hi) = (a1.max(a2), b1.min(b2));
        let (both, only1, only2) = if lo < hi {
            (
                sum_pi(xp, p, pi, lo, hi),
                sum_pi(xp, p, pi, a1, lo).wrapping_add(sum_pi(xp, p, pi, hi, b1)),
                sum_pi(xp, p, pi, a2, lo).wrapping_add(sum_pi(xp, p, pi, hi, b2)),
            )
        } else {
            (0, sum_pi(xp, p, pi, a1, b1), sum_pi(xp, p, pi, a2, b2))
        };
        // Each leaf counts `π(x / (p_b q)) - b + 2`.
        let two_less_b = 2u64.wrapping_sub(b);
        let sparse = both
            .wrapping_add(only1)
            .wrapping_add((b1 - a1).wrapping_mul(two_less_b));
        s = s.wrapping_add(sparse);
        if clustered > 0 {
            let swapped = pi
                .pi(xp / q_hi)
                .wrapping_mul(clustered)
                .wrapping_add(both)
                .wrapping_add(only2)
                .wrapping_sub((b2 - a2).wrapping_mul(pi_q_lo));
            s = s
                .wrapping_add(swapped)
                .wrapping_add(clustered.wrapping_mul(two_less_b));
        }
    }
    s
}

/// `Σ π(n / p_i)` for the indices `i` from `lo` (exclusive) to `hi`, the quotients from 6 up.
#[inline(always)]
fn sum_pi(n: u64, p: &[u32], pi: &PiTable, lo: u64, hi: u64) -> u64 {
    if lo >= hi {
        return 0;
    }
    p[lo as usize + 1..=hi as usize]
        .iter()
        .fold(0u64, |s, &q| s.wrapping_add(pi.pi_from_6(n / u64::from(q))))
}

/// `P2(x, a)`: the numbers up to `x` with exactly two prime factors above `y = p_a`,
/// `Σ (π(x / p_b) - b + 1)` over `a < b <= π(sqrt(x))`, from a sieve of the numbers from
/// `x / p` for the largest prime `p <= sqrt(x)` up to `x / y`, counted at each `x / p`.
#[inline(never)]
fn p2(x: u64, y: u64, a: u64, pi_of: &impl Fn(u64) -> u64) -> u64 {
    let sqrtx = x.isqrt();
    if y >= sqrtx {
        return 0;
    }
    let b = pi_of(sqrtx);
    // Σ -(i - 1) for i from a + 1 to b.
    let mut s = ((u128::from(a) * u128::from(a.wrapping_sub(1)) / 2) as u64)
        .wrapping_sub((u128::from(b) * u128::from(b - 1) / 2) as u64);
    // The primes from y + 1 to sqrt(x), largest first, in chunks sieved from the top down.
    let mut counter: Option<(Counter, u64)> = None;
    let mut hi = sqrtx;
    let chunk = 1u64 << 22;
    while hi > y {
        let lo = hi.saturating_sub(chunk).max(y) + 1;
        let mut primes = Vec::new();
        if hi >= 7 {
            each_prime(lo.max(7), hi, |q| {
                primes.push(q);
                true
            });
        }
        for &q in primes.iter().rev() {
            let n = x / q;
            match &mut counter {
                None => {
                    let pi_n = pi_of(n);
                    s = s.wrapping_add(pi_n);
                    counter = Some((Counter::new(n + 1, x / y), pi_n));
                }
                Some((counter, pi_n)) => {
                    *pi_n += counter.count_to(n);
                    s = s.wrapping_add(*pi_n);
                }
            }
        }
        hi = lo - 1;
    }
    s
}

/// The primes of a range counted up to increasing numbers, by the sieve.
struct Counter {
    segments: Segments,
    /// Whether a segment is sieved; the word of it counted up to, and how many of that word's bits
    /// are counted.
    sieved: bool,
    word: usize,
    in_word: u64,
    /// The first number not counted yet.
    next: u64,
}

impl Counter {
    /// The primes from `start` to `stop`, `7 <= start`.
    fn new(start: u64, stop: u64) -> Self {
        Counter {
            segments: Segments::new(start, stop.max(start), u64::MAX),
            sieved: false,
            word: 0,
            in_word: 0,
            next: start,
        }
    }

    /// The primes from the first number not counted yet up to `n`, which are then counted.
    fn count_to(&mut self, n: u64) -> u64 {
        if n < self.next {
            return 0;
        }
        self.next = n + 1;
        let mut count = 0;
        loop {
            if !self.sieved {
                if !self.segments.next_segment() {
                    return count;
                }
                self.sieved = true;
                self.word = 0;
                self.in_word = 0;
            }
            let low = self.segments.low();
            let bytes = self.segments.bytes();
            if n >= low + 30 * bytes.len() as u64 + 6 {
                // Past the segment's bytes.
                count += count_bits(&bytes[8 * self.word..]) - self.in_word;
                self.sieved = false;
                continue;
            }
            if n < low + 7 {
                return count;
            }
            let j = ((n - low - 7) / 30) as usize;
            let w = j / 8;
            let k = 8 * (j % 8);
            let word = u64::from_le_bytes(bytes[8 * w..8 * w + 8].try_into().expect("a word"));
            let mask =
                ((1u64 << k) - 1) | u64::from(KEEP_TO[byte_remainder(n - low) as usize]) << k;
            let upto = u64::from((word & mask).count_ones());
            count += count_bits(&bytes[8 * self.word..8 * w]) + upto - self.in_word;
            self.word = w;
            self.in_word = upto;
            return count;
        }
    }
}
