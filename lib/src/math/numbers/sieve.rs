//! Prime sieves and prime counting on `u64`: [`primes`], the primes in a range, by a segmented
//! sieve of Eratosthenes after Walisch's primesieve ([`erat`]); [`prime_pi`], `π(x)`, by
//! Deléglise and Rivat's method with Gourdon's sorting of the leaves ([`pi`]); and [`nth_prime`],
//! the n-th prime ([`nth`]).

mod erat;
mod nth;
mod pi;
mod presieve;
mod segments;
mod wheel;

use super::prime::is_prime_sieved_u64;
use core::ops::RangeBounds;
pub(in crate::math::numbers) use pi::tables::SMALL_PRIMES;
use segments::{Segments, count_bits, extract};

/// The places of the batch of primes of [`Primes`], a power of two.
const BUF: usize = 1024;

/// The primes in a range of `u64`s, in increasing order, from [`primes`].
///
/// The primes are formed a batch at a time, up to 1024, from the bits of a sieved segment, and
/// [`next`](Iterator::next) reads the batch; [`count`](Iterator::count) counts the bits of the sieve
/// without forming the primes, and [`fold`](Iterator::fold) (and so `sum`, `for_each` and
/// `collect`) reads the batches whole. The iterator is fused: after the last prime, `next` gives
/// `None` for ever.
#[derive(Clone)]
pub struct Primes {
    /// The primes of the current batch, the next at `i`, `n` of them.
    buf: Box<[u64; BUF]>,
    i: usize,
    n: usize,
    /// The sieve of the current stretch of the range, from 7, or `None` once it is done.
    sieve: Option<Box<Segments>>,
    /// The next word of the sieved segment to read, and the number its bits start from.
    w: usize,
    base: u64,
    /// The numbers the sieve leaves from this one up are tested: the sieve took out the multiples
    /// of the primes up to its square root only.
    tested: u64,
    /// The first number of the next stretch, and its width, when the current one is not the last.
    next: Option<u64>,
    width: u64,
    /// Whether all the primes left are to be taken, so that the next stretch is the rest.
    whole: bool,
    /// The last number of the range.
    last: u64,
}

impl core::fmt::Debug for Primes {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Primes")
            .field("next", &self.buf[self.i..self.n].first())
            .field("last", &self.last)
            .finish_non_exhaustive()
    }
}

impl Iterator for Primes {
    type Item = u64;

    #[inline]
    fn next(&mut self) -> Option<u64> {
        if self.i == self.n && !self.fill() {
            return None;
        }
        // `i < n <= BUF`: the mask leaves the index as it is, and spares the bounds check.
        let p = self.buf[self.i & (BUF - 1)];
        self.i += 1;
        Some(p)
    }

    /// At least the primes of the batch formed; exactly those once the range is sieved.
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.n - self.i;
        let done = self.sieve.is_none() && self.next.is_none();
        (n, done.then_some(n))
    }

    #[inline]
    fn fold<B, F: FnMut(B, u64) -> B>(mut self, init: B, mut f: F) -> B {
        // All the primes are taken: the rest is sieved as one stretch.
        self.whole = true;
        let mut acc = init;
        loop {
            for &p in &self.buf[self.i..self.n] {
                acc = f(acc, p);
            }
            self.i = self.n;
            if !self.fill() {
                return acc;
            }
        }
    }

    /// Counts the primes left by the bits of the sieve, without forming them, where it took out
    /// the multiples of the primes up to the square roots.
    ///
    /// # Panics
    ///
    /// Panics if the count passes `usize::MAX`, which it can only on targets of 32 bits.
    fn count(mut self) -> usize {
        let n = self.count_u64();
        usize::try_from(n).expect("the count of primes passes usize::MAX")
    }
}

impl core::iter::FusedIterator for Primes {}

/// A stretch of the range is sieved with all the primes up to the square root of its end when its
/// width is at least this many times that square root, as primesieve always does; below, the
/// primes up to a bound only, for fewer sieving primes to start, and the numbers left are tested
/// (see [`Primes::plan`]).
const FULL: f64 = 1.0 / 90.0;

impl Primes {
    /// The number of primes left, in 64 bits.
    fn count_u64(&mut self) -> u64 {
        self.whole = true;
        let mut n = (self.n - self.i) as u64;
        self.i = self.n;
        loop {
            let Some(sieve) = &mut self.sieve else {
                // After the primes from the table, the stretches of the sieve.
                match self.next.take() {
                    Some(a) => {
                        self.plan(a);
                        continue;
                    }
                    None => return n,
                }
            };
            if self.tested != u64::MAX {
                // The numbers left are tested: by batches.
                if !self.fill() {
                    return n;
                }
                n += (self.n - self.i) as u64;
                self.i = self.n;
                continue;
            }
            // The rest of the sieved segment, from the word the batch ended at.
            let words = sieve.bytes().len() / 8;
            if self.w < words {
                n += count_bits(&sieve.bytes()[8 * self.w..]);
            }
            while sieve.next_segment() {
                n += count_bits(sieve.bytes());
            }
            self.w = usize::MAX / 8;
            self.sieve = None;
            match self.next.take() {
                Some(a) => self.plan(a),
                None => return n,
            }
        }
    }

    /// Starts the sieve of the stretch of the range from `a`, at least 7: the rest of the range,
    /// with all the primes up to its square root, when the stretch is wide enough for their start
    /// (a division and a place for each), or a stretch of `width` numbers with the primes up to
    /// about a quarter of it, the numbers left tested by Baillie–PSW, with no trial division, and
    /// the next stretch four times as wide.
    ///
    /// The width at which the sieve with all the primes is the faster: their start is about
    /// `sqrt(a) / ln sqrt(a)` times 4 ns here, the tests `width / ln a` primes of 0.6 µs each and
    /// about as many other numbers of 70 ns, which make about `sqrt(a) / 90`.
    fn plan(&mut self, a: u64) {
        debug_assert!(a >= 7 && a <= self.last);
        if self.whole {
            self.width = (self.last - a).saturating_add(1);
        }
        let full = (a.isqrt() as f64 * FULL) as u64;
        if self.width >= full {
            // All the primes to the square root. The first segment of the whole rest is sized by
            // its end, 2 MiB for an end near `2^64`, 63 million numbers, about 3 ms, for the first
            // primes from 0 as well: so up to `2^40`, where a stretch's start (its primes to the
            // square root, at most 82,025) takes a third of a millisecond at most, the stretch is
            // its width while that is below a segment's, and the next four times as wide; past
            // that, the rest at once, as each new stretch starts its primes again.
            let end = a.saturating_add(self.width - 1);
            let (end, next) = if end < self.last && end <= 1 << 40 && self.width < 30 << 21 {
                (end, Some(end + 1))
            } else {
                (self.last, None)
            };
            self.sieve = Some(Box::new(Segments::new(a, end, u64::MAX)));
            self.tested = u64::MAX;
            self.next = next;
            self.width = self.width.saturating_mul(4);
            self.w = 0;
            return;
        }
        let end = a.saturating_add(self.width - 1).min(self.last);
        let bound = ((end - a) / 4).max(presieve_max()).min(end.isqrt());
        self.sieve = Some(Box::new(Segments::new(a, end, bound)));
        self.tested = if bound >= end.isqrt() {
            u64::MAX
        } else {
            (bound + 1).saturating_mul(bound + 1)
        };
        self.next = (end < self.last).then(|| end + 1);
        self.width = self.width.saturating_mul(4);
        self.w = 0;
    }

    /// Forms the next batch of primes, sieving the next segment when the current one is read and
    /// starting the next stretch when the current one is done; `false` at the end of the range.
    #[inline(never)]
    fn fill(&mut self) -> bool {
        loop {
            let Some(sieve) = self.sieve.as_mut() else {
                // After the primes from the table, the stretches of the sieve.
                if let Some(a) = self.next.take() {
                    self.plan(a);
                    continue;
                }
                self.i = 0;
                self.n = 0;
                return false;
            };
            if self.w * 8 >= sieve.bytes().len() {
                if !sieve.next_segment() {
                    self.sieve = None;
                    if let Some(a) = self.next.take() {
                        self.plan(a);
                    }
                    continue;
                }
                self.w = 0;
                self.base = sieve.low();
            }
            let mut n = extract(
                sieve.bytes(),
                &mut self.w,
                &mut self.base,
                &mut self.buf[..],
            );
            if self.tested != u64::MAX {
                // The numbers left from `tested` up, the last of the batch, are tested.
                let from = self.buf[..n].partition_point(|&q| q < self.tested);
                let mut k = from;
                for j in from..n {
                    let q = self.buf[j];
                    self.buf[k] = q;
                    k += usize::from(is_prime_sieved_u64(q));
                }
                n = k;
            }
            if n > 0 {
                self.i = 0;
                self.n = n;
                return true;
            }
        }
    }
}

/// The largest prime of the pre-sieve, the least bound of a sieve.
fn presieve_max() -> u64 {
    presieve::MAX
}

/// Returns the primes in `range`, in increasing order: `primes(10..=30)` gives 11, 13, 17, 19, 23,
/// 29.
///
/// Any range of `u64`s, to `u64::MAX`: an empty or reversed range gives no primes. The range is
/// sieved by Walisch's segmented sieve of Eratosthenes (a byte for 30 numbers, pre-sieved to 163,
/// small, medium and large primes each crossed out their own way), in stretches: the first of
/// 4096 numbers, then each four times as wide, a stretch narrower than about `√a / 90` from its
/// start `a` sieved by the primes to a quarter of its width only and the numbers left tested by
/// Baillie–PSW, the rest at once by all the primes to its square root (below `2^40`, in stretches
/// of the same widths while they are narrower than a segment, 63 million numbers). So the first
/// primes of a range come at once, where the start of the full sieve would take a division and a
/// place for each prime to the square root (203 million below `2^32`) and a first segment of 2 MiB,
/// and a long range is sieved as primesieve sieves it.
///
/// # Examples
///
/// ```
/// use a::math::numbers::primes;
///
/// assert_eq!(primes(10..=30).collect::<Vec<_>>(), [11, 13, 17, 19, 23, 29]);
/// assert_eq!(primes(..1_000_000).count(), 78_498);
/// assert_eq!(primes(1_000_000_000_000..).next(), Some(1_000_000_000_039));
/// assert_eq!(primes(u64::MAX - 100..).last(), Some(u64::MAX - 58));
/// ```
pub fn primes(range: impl RangeBounds<u64>) -> Primes {
    use core::ops::Bound::{Excluded, Included, Unbounded};
    let first = match range.start_bound() {
        Included(&a) => Some(a),
        Excluded(&a) => a.checked_add(1),
        Unbounded => Some(0),
    };
    let last = match range.end_bound() {
        Included(&b) => Some(b),
        Excluded(&b) => b.checked_sub(1),
        Unbounded => Some(u64::MAX),
    };
    let mut p = Primes {
        buf: Box::new([0; BUF]),
        i: 0,
        n: 0,
        sieve: None,
        w: 0,
        base: 0,
        tested: u64::MAX,
        next: None,
        width: 0,
        whole: false,
        last: 0,
    };
    let (Some(a), Some(b)) = (first, last) else {
        return p;
    };
    if a > b {
        return p;
    }
    p.last = b;
    // The first stretch of the sieve: 4096 numbers, about a hundred primes even near `2^64`.
    p.width = 1 << 12;
    if a < SMALL {
        // The first batch from the table of the primes below `2^16`, with no sieve to start: those
        // from `a` to `b`, the batch's places at most; the sieve from the next number on.
        // The primes below `2^16`: the table less its 0 before and the prime after.
        let table = &SMALL_PRIMES[1..SMALL_PRIMES.len() - 1];
        let from = table.partition_point(|&q| u64::from(q) < a);
        let to = table[from..]
            .partition_point(|&q| u64::from(q) <= b)
            .min(BUF);
        for (place, &q) in p.buf.iter_mut().zip(&table[from..from + to]) {
            *place = u64::from(q);
        }
        p.n = to;
        let next = if to == BUF { p.buf[BUF - 1] + 1 } else { SMALL };
        if next <= b {
            p.next = Some(next);
        }
        return p;
    }
    if b >= 7 {
        p.plan(a);
    }
    p
}

/// The numbers below which the first primes of a range come from the table of the primes.
const SMALL: u64 = 1 << 16;

/// Returns `π(x)`, the number of primes up to `x`, exactly for every `u64`.
///
/// Below `2^16` from a table computed at compile time, and above by Deléglise and Rivat's method
/// (1996) as Oliveira e Silva (2006) and Walisch's primecount carry it out, its special leaves
/// sorted as Gourdon's (2001), so that only those that need it are counted by the sieve, in
/// `O(x^(2/3) / log² x)` time and `O(x^(1/3) log³ x)` space: on an Apple M4 Max, `π(10^6)` takes
/// 1.04 µs, `π(10^16)` 988 ms and less than 10 MB, `π(2^64 - 1)` 99.0 s and 83 MB.
///
/// # Examples
///
/// ```
/// use a::math::numbers::prime_pi;
///
/// assert_eq!(prime_pi(100), 25);
/// assert_eq!(prime_pi(1_000_000_000_000), 37_607_912_018);
/// ```
pub fn prime_pi(x: u64) -> u64 {
    pi::prime_pi(x)
}

/// Returns the `n`-th prime, `nth_prime(1) = 2`, for `n` up to `π(2^64 - 1)`,
/// 425,656,284,035,217,743, whose prime is `2^64 - 59`.
///
/// From a table up to `π(2^16)`; above, from `x = R⁻¹(n)`, Riemann's `R` inverted by Newton's
/// steps, within a few times `√x / ln x` primes of it, [`prime_pi`] of `x`, and the sieve from `x`
/// to the prime, as primecount does; near the top, down from `2^64 - 1`, whose `π` is known.
///
/// # Panics
///
/// Panics if `n` is zero, or above `π(2^64 - 1)`, as the prime is then above `u64::MAX`.
///
/// # Examples
///
/// ```
/// use a::math::numbers::nth_prime;
///
/// assert_eq!(nth_prime(1), 2);
/// assert_eq!(nth_prime(1_000_000), 15_485_863);
/// assert_eq!(nth_prime(425_656_284_035_217_743), u64::MAX - 58);
/// ```
pub fn nth_prime(n: u64) -> u64 {
    if n == 0 {
        no_zeroth_prime();
    }
    if n > nth::PI_MAX {
        prime_above_u64_max(n);
    }
    nth::nth_prime(n)
}

#[cold]
#[track_caller]
fn no_zeroth_prime() -> ! {
    panic!("nth_prime: there is no 0th prime")
}

#[cold]
#[track_caller]
fn prime_above_u64_max(n: u64) -> ! {
    panic!("nth_prime: the prime of index {n} is above u64::MAX")
}
