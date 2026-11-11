//! Deléglise and Rivat's method (1996), in the form of Oliveira e Silva's "Computing π(x): the
//! combinatorial method" (2006) and of Walisch's primecount:
//!
//! `π(x) = S1 + S2 + π(y) - 1 - P2`, for `y = α x^(1/3)` and `a = π(y)`, where `S1 + S2 = φ(x, a)`,
//! the numbers up to `x` with no prime factor up to `y`, split by the leaves of Legendre's
//! recursion: `S1` the ordinary leaves, `S2` the special ones (trivial, easy and hard), and `P2`
//! the numbers up to `x` with exactly two prime factors above `y`.
//!
//! The special leaves are sorted as Gourdon's (2001) are: a leaf `p_b m` whose `x / (p_b m)` is
//! below `p_b^2` has `φ(x / (p_b m), b - 1) = π(x / (p_b m)) - b + 2`, so that only the leaves at
//! least `p_b^2`, whose primes are at most `x^(1/4)`, need the sieve ([`s2_hard`]); the leaves of a
//! prime `q` between `y` and `p_b^2` are easy ones above `y` ([`s2_above`]), and the primes above
//! `x^(1/4)` do not cross out the sieve. Below `2^16`, the primes, `π` and `P2`'s counts come from
//! tables computed at compile time, so that small `x` takes no sieve but the hard leaves'.
//!
//! The sums are in 64 bits that wrap: each term is below `2^64`, the parts may not be, but the
//! result, below `2^63`, is right modulo `2^64`, so exact.

use super::super::segments::{Segments, count_bits, each_prime, isqrt};
use super::super::wheel::{KEEP_TO, OFFSETS, byte_remainder};
use super::hard::s2_hard;
use super::phi::{MAX_A, phi_tiny, tiny_a};
use super::tables::{FactorTable, PiTable, SMALL, SMALL_PRIMES, primes_to, small_primes_to};

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

/// `π(x)` by Deléglise and Rivat's method, for `x` from about `10^4`, its special leaves sorted as
/// Gourdon's (2001): the hard ones, whose `φ` needs the sieve, only those with `x / (p_b m)` at
/// least `p_b^2`, the others by `π`.
pub(super) fn pi(x: u64, pi_of: impl Fn(u64) -> u64) -> u64 {
    let x13 = icbrt(x);
    let y = ((x13 as f64 * alpha(x)) as u64).clamp(x13, isqrt(x));
    let z = x / y;
    // Below `2^16` the primes and `π` come from the tables computed at compile time.
    let owned_p;
    let p: &[u32] = if y < SMALL {
        small_primes_to(y)
    } else {
        owned_p = primes_to(y);
        &owned_p
    };
    let pi_y = (p.len() - 2) as u64;
    let c = tiny_a(y).min(MAX_A);
    let owned_t;
    let table = if y < SMALL {
        PiTable::small()
    } else {
        owned_t = PiTable::blocks(y, p);
        PiTable::of(&owned_t)
    };
    let factor = FactorTable::new(y, p);
    let p2 = if z < SMALL {
        p2_small(x, y, pi_y)
    } else {
        p2(x, y, pi_y, &pi_of)
    };
    let s1 = s1(x, y, c, p);
    let trivial = s2_trivial(x, y, z, c, p, &table);
    let easy = s2_easy(x, y, z, c, p, &table);
    let above = s2_above(x, y, z, c, p, &table);
    let hard = s2_hard(x, y, z, c, p, &table, &factor);
    s1.wrapping_add(trivial)
        .wrapping_add(easy)
        .wrapping_add(above)
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
fn s2_trivial(x: u64, y: u64, z: u64, c: u64, p: &[u32], pi: &PiTable<'_>) -> u64 {
    let pi_y = pi.pi(y);
    let start = (u64::from(p[c as usize])).max(isqrt(z)) + 1;
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
///
/// The look-ups, `π(N / q)` for runs of `q`, are made by [`look_up`], in the order of their values
/// when the table of `π` is larger than the cache.
#[inline(never)]
fn s2_easy(x: u64, y: u64, z: u64, c: u64, p: &[u32], pi: &PiTable<'_>) -> u64 {
    let x13 = icbrt(x);
    let pi_sqrty = pi.pi(isqrt(y));
    let pi_x13 = pi.pi(x13.min(y));
    let mut s = 0u64;
    // The look-ups of a table that fits in the cache are made at once; those of a larger one are
    // gathered by batches, to be made by parts of the table.
    let direct = pi.blocks_len() <= 2 * LOOK_UP_PART;
    let mut runs: Vec<Run> = Vec::new();
    macro_rules! run {
        ($xp:expr, $lo:expr, $hi:expr, $w:expr) => {{
            let (xp, lo, hi, w) = ($xp, $lo, $hi, $w);
            if direct {
                s = s.wrapping_add(sum_pi(xp, p, pi, lo, hi).wrapping_mul(w));
            } else if lo < hi {
                runs.push(Run::new(xp, lo, hi, w));
            }
        }};
    }
    let first = c.max(pi_sqrty) + 1;
    let mut b = first;
    while b <= pi_x13 {
        // The primes by batches, whose runs are looked up together.
        let last = pi_x13.min(b + RUNS_OF);
        runs.clear();
        for b in b..=last {
            let prime = u64::from(p[b as usize]);
            let xp = x / prime;
            // The leaves `q` from `s_lo` (exclusive) to `q_lo` are sparse, those to `q_hi`
            // clustered.
            let q_hi = (xp / prime).min(y);
            let s_lo = (z / prime).clamp(prime, y);
            let q_lo = isqrt(xp).clamp(prime, y).max(s_lo);
            let pi_q_lo = pi.pi(q_lo);
            let clustered = pi.pi(q_hi).saturating_sub(pi_q_lo);
            // The sparse leaves' indices, and the swapped sum's.
            let (a1, b1) = (pi.pi(s_lo), pi_q_lo);
            let (a2, b2) = if clustered > 0 {
                (pi.pi(xp / q_hi), pi.pi(xp / q_lo))
            } else {
                (0, 0)
            };
            // Each leaf counts `π(x / (p_b q)) - b + 2`; the swapped sum takes `π(N / T)` for each
            // clustered leaf and `π(Q)` off each of its terms.
            let two_less_b = 2u64.wrapping_sub(b);
            s = s.wrapping_add((b1 - a1).wrapping_mul(two_less_b));
            if clustered > 0 {
                s = s
                    .wrapping_add(a2.wrapping_mul(clustered))
                    .wrapping_sub((b2 - a2).wrapping_mul(pi_q_lo))
                    .wrapping_add(clustered.wrapping_mul(two_less_b));
            }
            // The look-ups: the two ranges of indices, the part in both looked up once and
            // counted twice.
            let (lo, hi) = (a1.max(a2), b1.min(b2));
            if clustered > 0 && lo < hi {
                run!(xp, a1.min(a2), lo, 1);
                run!(xp, lo, hi, 2);
                run!(xp, hi, b1.max(b2), 1);
            } else {
                run!(xp, a1, b1, 1);
                if clustered > 0 {
                    run!(xp, a2, b2, 1);
                }
            }
        }
        if !direct {
            s = s.wrapping_add(look_up(&mut runs, p, pi));
        }
        b = last + 1;
    }
    s
}

/// The primes whose runs [`s2_easy`] looks up together, less one.
const RUNS_OF: u64 = 4095;

/// The look-ups `w Σ π(xp / p[i])` for the indices `i` from `lo` (exclusive) to `hi`.
#[derive(Clone, Copy)]
struct Run {
    xp: u64,
    lo: u64,
    hi: u64,
    w: u64,
}

impl Run {
    fn new(xp: u64, lo: u64, hi: u64, w: u64) -> Self {
        Run { xp, lo, hi, w }
    }
}

/// The blocks of the table of `π` that [`look_up`] reads at a time, 64 KiB.
const LOOK_UP_PART: usize = 4096;

/// The sum of the look-ups of `runs`, `π` of values from 6 up to the table's limit, a table larger
/// than a few parts: a run's look-ups, from its largest index down, come in increasing order of
/// their values, so they are made a part of the table after the other, each run taking the look-ups
/// whose values fall in the part, so that the part's blocks stay in the cache, where the look-ups in
/// turn would read the whole table from memory. The runs are emptied.
#[inline(never)]
fn look_up(runs: &mut Vec<Run>, p: &[u32], pi: &PiTable<'_>) -> u64 {
    let mut s = 0u64;
    let blocks = pi.blocks_len();
    // The first part: that of the smallest value.
    let least = runs
        .iter()
        .map(|r| r.xp / u64::from(p[r.hi as usize]))
        .min()
        .unwrap_or(u64::MAX);
    let mut k = (least / 240) as usize;
    while !runs.is_empty() {
        let end = (240 * (k + LOOK_UP_PART) as u64).min(240 * blocks as u64) - 1;
        for r in runs.iter_mut() {
            // The look-ups up to the part's end: `p[i] > xp / (end + 1)`.
            let t = r.xp / (end + 1);
            if t >= u64::from(p[r.hi as usize]) {
                continue;
            }
            let from = pi.pi(t).max(r.lo);
            s = s.wrapping_add(sum_pi(r.xp, p, pi, from, r.hi).wrapping_mul(r.w));
            r.hi = from;
        }
        runs.retain(|r| r.lo < r.hi);
        k += LOOK_UP_PART;
    }
    s
}

/// `Σ π(n / p_i)` for the indices `i` from `lo` (exclusive) to `hi`, the quotients from 6 up.
#[inline(always)]
fn sum_pi(n: u64, p: &[u32], pi: &PiTable<'_>, lo: u64, hi: u64) -> u64 {
    if lo >= hi {
        return 0;
    }
    pi.sum_quotients(n, &p[lo as usize + 1..=hi as usize])
}

/// The easy special leaves above `y`, by Gourdon's sorting of the leaves: `p_b q`, `q` a prime
/// above `p_b`, with `y <= x / (p_b q) < p_b^2`, whose `φ(x / (p_b q), b - 1)` is
/// `π(x / (p_b q)) - b + 2`, for the primes `p_b` from `sqrt(y)` to `sqrt(x / y)`. Deléglise and
/// Rivat's method takes them as hard leaves, by the sieve, with the leaves at least `p_b^2`, which
/// only those need: then the primes above `x^(1/4)` have none, and do not cross out the sieve's
/// numbers.
///
/// Their `π`, between `y` and `sqrt(x)`, comes from the table below `2^16` when `sqrt(x)` is below
/// it, and else from a sieve of the numbers from `y` up, in parts whose counts stay in the first
/// level of the cache: the leaves of a prime, its `q` from the largest down, come in the order of
/// their `x / (p_b q)`, so that each part takes the next leaves of each prime.
#[inline(never)]
fn s2_above(x: u64, y: u64, z: u64, c: u64, p: &[u32], pi: &PiTable<'_>) -> u64 {
    let first = c.max(pi.pi(isqrt(y))) + 1;
    let last = pi.pi(isqrt(z).min(y));
    if first > last {
        return 0;
    }
    let small = isqrt(x) < SMALL;
    let mut s = 0u64;
    // The leaves of each prime: the indices `q = p[i]`, `lo < i <= hi`.
    let mut leaves = Vec::with_capacity(if small {
        0
    } else {
        (last + 1 - first) as usize
    });
    for b in first..=last {
        let prime = u64::from(p[b as usize]);
        let xp = x / prime;
        let q_lo = (xp / (prime * prime)).max(prime);
        let q_hi = (z / prime).min(y);
        if q_lo < q_hi {
            let (lo, hi) = (pi.pi(q_lo), pi.pi(q_hi));
            let two_less_b = 2u64.wrapping_sub(b);
            if small {
                s = s
                    .wrapping_add(sum_pi(xp, p, &PiTable::small(), lo, hi))
                    .wrapping_add((hi - lo).wrapping_mul(two_less_b));
            } else if lo < hi {
                leaves.push(Above {
                    xp,
                    lo,
                    hi,
                    two_less_b,
                });
            }
        }
    }
    if leaves.is_empty() {
        return s;
    }
    // The largest `x / (p_b q)`, from each prime's smallest `q`.
    let top = leaves
        .iter()
        .map(|l| l.xp / u64::from(p[l.lo as usize + 1]))
        .max()
        .expect("leaves");
    let mut sieve = Segments::new(y, top, u64::MAX);
    // The primes below the part.
    let mut before = pi.pi(y - 1);
    let mut counts = vec![0u32; PART];
    while sieve.next_segment() {
        let low = sieve.low();
        let words = sieve.bytes().as_chunks::<8>().0;
        for (k, part) in words.chunks(PART).enumerate() {
            // The bits before each word of the part.
            let mut n = 0u32;
            for (c, w) in counts.iter_mut().zip(part) {
                *c = n;
                n += u64::from_le_bytes(*w).count_ones();
            }
            let start = low + 240 * (k * PART) as u64 + 7;
            let end = start + 240 * part.len() as u64 - 1;
            for l in &mut leaves {
                // The leaves up to the part's end: `q > x / (p_b (end + 1))`.
                let t = l.xp / (end + 1);
                if t >= u64::from(p[l.hi as usize]) {
                    continue;
                }
                let from = pi.pi(t).max(l.lo);
                let sum = part_sum(
                    l.xp,
                    &p[from as usize + 1..=l.hi as usize],
                    start,
                    part,
                    &counts,
                );
                let n = l.hi - from;
                s = s
                    .wrapping_add(sum)
                    .wrapping_add(n.wrapping_mul(before.wrapping_add(l.two_less_b)));
                l.hi = from;
            }
            leaves.retain(|l| l.hi > l.lo);
            before += u64::from(n);
        }
    }
    debug_assert!(leaves.is_empty());
    s
}

/// `Σ π(xp / q) - π(start - 1)` for the primes `q` of `qs`, whose quotients are in the part of the
/// sieve's words `part` from `start`, from its counts before each word: two at a time, the divisor
/// known not to be zero, the word's index kept to the part, the two words' bits counted together.
#[inline(always)]
fn part_sum(xp: u64, qs: &[u32], start: u64, part: &[[u8; 8]], counts: &[u32]) -> u64 {
    let last = part.len() - 1;
    let at = |q: u32| {
        let r = xp / core::num::NonZeroU64::from(core::num::NonZeroU32::new(q).expect("a prime"))
            - start;
        (
            ((r / 240) as usize).min(last),
            ERAT_UP_TO[(r % 240) as usize],
        )
    };
    let word = |w: usize| u64::from_le_bytes(part[w]);
    let (pairs, rest) = qs.as_chunks::<2>();
    let mut sum = 0u64;
    for &[q0, q1] in pairs {
        let ((w0, m0), (w1, m1)) = (at(q0), at(q1));
        let (b0, b1) = (word(w0) & m0, word(w1) & m1);
        sum += u64::from(counts[w0]) + u64::from(counts[w1]);
        sum += u64::from((u128::from(b0) | u128::from(b1) << 64).count_ones());
    }
    for &q in rest {
        let (w, m) = at(q);
        sum += u64::from(counts[w]) + u64::from((word(w) & m).count_ones());
    }
    sum
}

/// The leaves of a prime `p_b` for [`s2_above`]: `x / p_b`, the indices of its `q` left, from
/// `lo` (exclusive) to `hi`, and `2 - b`.
struct Above {
    xp: u64,
    lo: u64,
    hi: u64,
    two_less_b: u64,
}

/// The words of the sieve whose counts [`s2_above`] forms at once: 32 KiB of the sieve's bytes and
/// 16 KiB of counts, which stay in the first-level cache with the leaves' primes.
const PART: usize = 4096;

/// For the bits of a word of the sieve, the numbers from `30 j + 7` (see
/// [`wheel`](super::super::wheel)): the masks of the bits of the numbers up to the word's start
/// plus `7 + r`, for `r` from 0 to 239.
static ERAT_UP_TO: [u64; 240] = {
    let mut t = [0u64; 240];
    let mut r = 0;
    while r < 240 {
        let mut m = 0u64;
        let mut b = 0;
        while b < 64 {
            if 30 * (b as u64 / 8) + OFFSETS[b % 8] <= 7 + r as u64 {
                m |= 1 << b;
            }
            b += 1;
        }
        t[r] = m;
        r += 1;
    }
    t
};

/// `P2(x, a)` when `x / y` is below `2^16`: the sum of [`p2`] with each `π(x / p_b)` from the table.
#[inline(never)]
fn p2_small(x: u64, y: u64, a: u64) -> u64 {
    let sqrtx = isqrt(x);
    if y >= sqrtx {
        return 0;
    }
    let t = PiTable::small();
    let b = t.pi(sqrtx);
    let s = ((u128::from(a) * u128::from(a.wrapping_sub(1)) / 2) as u64)
        .wrapping_sub((u128::from(b) * u128::from(b - 1) / 2) as u64);
    SMALL_PRIMES[a as usize + 1..=b as usize]
        .iter()
        .fold(s, |s, &q| s.wrapping_add(t.pi_from_6(x / u64::from(q))))
}

/// `P2(x, a)`: the numbers up to `x` with exactly two prime factors above `y = p_a`,
/// `Σ (π(x / p_b) - b + 1)` over `a < b <= π(sqrt(x))`, from a sieve of the numbers from
/// `x / p` for the largest prime `p <= sqrt(x)` up to `x / y`, counted at each `x / p`.
#[inline(never)]
fn p2(x: u64, y: u64, a: u64, pi_of: &impl Fn(u64) -> u64) -> u64 {
    let sqrtx = isqrt(x);
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
