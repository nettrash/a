//! The `n`-th prime: Riemann's `R⁻¹(n)` close to it, `π` there, and the sieve from there to it, as
//! primecount and primesieve find it.

use super::pi::{prime_pi, small_nth};
use super::segments::{Segments, count_bits};
use super::wheel::BIT_VALUES;

/// `π(2^64 - 1)`, the number of primes below `2^64` (Gourdon's and primecount's computations).
pub(super) const PI_MAX: u64 = 425_656_284_035_217_743;

/// The largest prime below `2^64`.
const LAST: u64 = u64::MAX - 58;

/// The `n`-th prime, for `n` from 1 to [`PI_MAX`].
pub(super) fn nth_prime(n: u64) -> u64 {
    debug_assert!((1..=PI_MAX).contains(&n));
    if let Some(p) = small_nth(n) {
        return p;
    }
    if n == PI_MAX {
        return LAST;
    }
    // Up to about `π(2 10^5)`, where `π` itself is the sieve's count, by the sieve from the table's
    // end, `2^16`, whose `π` is 6542.
    if n <= 18_000 {
        return forward((1 << 16) + 1, n - 6542);
    }
    // Near the top, down from `2^64 - 1`, whose `π` is known, rather than `π` near `2^64`: up to
    // `10^7` primes, about `4.4 10^8` numbers, the sieve takes less time.
    if PI_MAX - n <= 10_000_000 {
        return backward(u64::MAX, PI_MAX - n + 1);
    }
    let x = estimate(n);
    let count = if x == u64::MAX { PI_MAX } else { prime_pi(x) };
    if count < n {
        forward(x + 1, n - count)
    } else {
        backward(x, count - n + 1)
    }
}

/// The `k`-th prime from `a` up, `k >= 1`, which is below `2^64`: windows of twice the expected
/// distance, at least 4096 numbers, each twice the last, each read once.
fn forward(a: u64, k: u64) -> u64 {
    let mut k = k;
    let mut a = a;
    let mut width = (k.saturating_mul(gap(a)).saturating_mul(2)).max(1 << 12);
    loop {
        let b = a.saturating_add(width);
        match scan(a, b, k) {
            Ok(p) => return p,
            Err(c) => k -= c,
        }
        a = b + 1;
        width = width.saturating_mul(2);
    }
}

/// The `k`-th prime from `b` down, `k >= 1`, which is at least 2: windows as [`forward`]'s, each
/// counted, and the one of the prime read again to it.
fn backward(b: u64, k: u64) -> u64 {
    let mut k = k;
    let mut b = b;
    let mut width = (k.saturating_mul(gap(b)).saturating_mul(2)).max(1 << 12);
    loop {
        let a = b.saturating_sub(width);
        let c = match scan(a, b, u64::MAX) {
            Ok(_) => unreachable!("no range holds u64::MAX primes"),
            Err(c) => c,
        };
        if c >= k {
            return scan(a, b, c - k + 1).expect("the prime of the window");
        }
        k -= c;
        b = a - 1;
        width = width.saturating_mul(2);
    }
}

/// The average gap between the primes near `x`, `ln x`, at least 2.
fn gap(x: u64) -> u64 {
    ((x as f64).ln() as u64).max(2)
}

/// The `k`-th prime from `a` to `b`, `k >= 1`, or the number of primes from `a` to `b` when they
/// are fewer: by the sieve's bits, a segment's at a time, or, for a short range high up, by the
/// tests of the numbers a sieve by the smaller primes leaves (see [`primes`](super::primes)).
fn scan(a: u64, b: u64, mut k: u64) -> Result<u64, u64> {
    let mut count = 0;
    if b.isqrt() / 90 > b - a {
        for p in super::primes(a..=b) {
            count += 1;
            if count == k {
                return Ok(p);
            }
        }
        return Err(count);
    }
    for q in [2, 3, 5] {
        if a <= q && q <= b {
            count += 1;
            if count == k {
                return Ok(q);
            }
        }
    }
    if b < 7 {
        return Err(count);
    }
    k -= count;
    let mut s = Segments::new(a.max(7), b, u64::MAX);
    while s.next_segment() {
        let bytes = s.bytes();
        let c = count_bits(bytes);
        if c < k {
            k -= c;
            count += c;
            continue;
        }
        // In this segment: the word, then the bit.
        let mut base = s.low();
        for word in bytes.as_chunks::<8>().0 {
            let mut bits = u64::from_le_bytes(*word);
            let c = u64::from(bits.count_ones());
            if c < k {
                k -= c;
                base += 240;
                continue;
            }
            for _ in 1..k {
                bits &= bits - 1;
            }
            return Ok(base + BIT_VALUES[bits.trailing_zeros() as usize]);
        }
    }
    Err(count)
}

/// `R⁻¹(n)`, the `x` with `R(x) = n`, rounded, by Newton's steps (`R'(x)` about `1 / ln x`) from
/// Cipolla's `n (ln n + ln ln n - 1 + (ln ln n - 2) / ln n)`, within about `10^-3` of it: within
/// a few times `sqrt(x) / ln x` of the `n`-th prime, after three or four steps, the last below the
/// floats' resolution.
fn estimate(n: u64) -> u64 {
    let n = n as f64;
    let (l, ll) = (n.ln(), n.ln().ln());
    let mut x = n * (l + ll - 1.0 + (ll - 2.0) / l);
    for _ in 0..8 {
        let step = (riemann_r(x) - n) * x.ln();
        x -= step;
        if step.abs() < 0.5f64.max(x * 1e-14) {
            break;
        }
    }
    if x >= u64::MAX as f64 {
        u64::MAX
    } else {
        x.max(2.0) as u64
    }
}

/// Riemann's `R(x) = Σ μ(k) li(x^(1/k)) / k`, by Gram's series
/// `R(x) = 1 + Σ_{k >= 1} (ln x)^k / (k k! ζ(k + 1))`, whose terms are all positive: about
/// `e ln x` of them, to `10^-17` of the sum.
fn riemann_r(x: f64) -> f64 {
    /// `ζ(k)` for `k` from 2 to 20; above, `1 + 2^-k + ... + 5^-k`, within `6^-k`.
    static ZETA: [f64; 19] = [
        1.644_934_066_848_226_4,
        1.202_056_903_159_594_2,
        1.082_323_233_711_138_2,
        1.036_927_755_143_37,
        1.017_343_061_984_449,
        1.008_349_277_381_922_8,
        1.004_077_356_197_944_3,
        1.002_008_392_826_082_2,
        1.000_994_575_127_818_1,
        1.000_494_188_604_119_5,
        1.000_246_086_553_308,
        1.000_122_713_347_578_5,
        1.000_061_248_135_058_7,
        1.000_030_588_236_307,
        1.000_015_282_259_408_7,
        1.000_007_637_297_640_4,
        1.000_003_817_293_265,
        1.000_001_908_212_716_6,
        1.000_000_953_962_033_9,
    ];
    let l = x.ln();
    let mut sum = 1.0;
    // `(ln x)^k / k!`.
    let mut power = 1.0;
    for k in 1..400 {
        let kf = f64::from(k);
        power *= l / kf;
        let s: i32 = k + 1;
        let zeta = if s <= 20 {
            ZETA[(s - 2) as usize]
        } else {
            1.0 + 2f64.powi(-s) + 3f64.powi(-s) + 4f64.powi(-s) + 5f64.powi(-s)
        };
        let term = power / (kf * zeta);
        sum += term;
        if term < 1e-17 * sum && kf > l {
            break;
        }
    }
    sum
}
