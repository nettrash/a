//! The segments of a range, sieved one after the other, with the sieving primes they need, from a
//! sieve of their own.

use super::erat::Erat;
use super::pi::tables::{SMALL, SMALL_PRIMES};
use super::presieve;
use super::wheel::BIT_VALUES;

/// The primes a batch of [`extract`] gives at most, with 64 more of room.
pub(super) const BATCH: usize = 1024;

/// Writes into `out` the primes of the sieve's words from the word `*w` on, the numbers of that word
/// from the multiple of 30 `*base`, until fewer than 64 places are left in `out` or the words end;
/// returns how many it wrote and moves `*w` and `*base` past the words read.
///
/// Each word gives its primes from both ends at once, by the trailing zeros of the word and of its
/// bits reversed, the lowest and the highest left two at a time, into their places counted from
/// the word's first and last: two chains of `bits &= bits - 1` side by side, each half as long as
/// one, which took the forming of the primes from 3.3 to about 2.4 cycles a prime here. The steps
/// past the middle write junk in places of the 64 from the word's first that are past its last,
/// which the next word or the count leaves out.
#[inline(always)]
pub(super) fn extract(bytes: &[u8], w: &mut usize, base: &mut u64, out: &mut [u64]) -> usize {
    let words = bytes.as_chunks::<8>().0;
    let mut n = 0;
    while *w < words.len() && n + 64 <= out.len() {
        let bits = u64::from_le_bytes(words[*w]);
        let b = *base;
        let count = bits.count_ones() as usize;
        let window: &mut [u64; 64] = (&mut out[n..n + 64]).try_into().expect("64 places");
        let (mut lo, mut hi) = (bits, bits.reverse_bits());
        // The places from the last: `count - 1 - k`, kept in the window.
        let last = count.wrapping_sub(1);
        let mut k = 0;
        while 2 * k < count {
            window[k & 63] = b.wrapping_add(BIT_VALUES[lo.trailing_zeros() as usize]);
            lo &= lo.wrapping_sub(1);
            window[last.wrapping_sub(k) & 63] =
                b.wrapping_add(TOP_VALUES[hi.trailing_zeros() as usize]);
            hi &= hi.wrapping_sub(1);
            window[(k + 1) & 63] = b.wrapping_add(BIT_VALUES[lo.trailing_zeros() as usize]);
            lo &= lo.wrapping_sub(1);
            window[last.wrapping_sub(k + 1) & 63] =
                b.wrapping_add(TOP_VALUES[hi.trailing_zeros() as usize]);
            hi &= hi.wrapping_sub(1);
            k += 2;
        }
        n += count;
        *w += 1;
        *base = b.wrapping_add(240);
    }
    n
}

/// The values of [`BIT_VALUES`] by the trailing zeros of a word's bits reversed: bit `63 - t`, and
/// 0 for an empty word.
static TOP_VALUES: [u64; 65] = {
    let mut t = [0u64; 65];
    let mut z = 0;
    while z < 64 {
        t[z] = BIT_VALUES[63 - z];
        z += 1;
    }
    t
};

/// The integer square root of `x`: the floating-point root, within one of it, corrected; faster
/// than `u64::isqrt` here, which the counts take for each prime.
#[inline]
pub(super) fn isqrt(x: u64) -> u64 {
    // `x as f64` and the root are each within half a unit of the last place: the root is within
    // `1.5 2^-53 2^32` of the exact one, so its integer part is off by one at most.
    let r = ((x as f64).sqrt() as u64).min(u64::from(u32::MAX));
    if r * r > x {
        r - 1
    } else if r < u64::from(u32::MAX) && (r + 1) * (r + 1) <= x {
        r + 1
    } else {
        r
    }
}

/// The number of set bits of `bytes`.
#[inline]
pub(super) fn count_bits(bytes: &[u8]) -> u64 {
    let (words, rest) = bytes.as_chunks::<8>();
    let mut n = 0u64;
    for w in words {
        n += u64::from(u64::from_ne_bytes(*w).count_ones());
    }
    for b in rest {
        n += u64::from(b.count_ones());
    }
    n
}

/// Calls `f` with each prime from `a`, at least 7, to `b`, in increasing order, until it returns
/// `false`: from the sieve's batches, with no iterator.
pub(super) fn each_prime(a: u64, b: u64, mut f: impl FnMut(u64) -> bool) {
    let mut s = Segments::new(a, b, u64::MAX);
    let mut batch = [0u64; 256];
    while s.next_segment() {
        let (mut w, mut base) = (0, s.low());
        while w * 8 < s.bytes().len() {
            let n = extract(s.bytes(), &mut w, &mut base, &mut batch);
            for &q in &batch[..n] {
                if !f(q) {
                    return;
                }
            }
        }
    }
}

/// The primes above [`presieve::MAX`] up to a limit below `2^32`, in increasing order, then
/// `u64::MAX`: the sieving primes of a sieve.
#[derive(Clone, Debug)]
pub(super) struct SievingPrimes {
    /// The primes from the table, below `2^16`, the next at `s`: all of them when the limit is
    /// below `2^16`, and then there is no sieve.
    small: &'static [u32],
    s: usize,
    /// The sieve of the primes, or `None` when there are none or they come from the table.
    erat: Option<Erat>,
    /// Its own sieving primes, from the table, up to the square root of the limit.
    tiny: &'static [u32],
    /// The next of `tiny` to add.
    t: usize,
    /// The primes of the current batch, `i` the next and `n` their number.
    buf: Vec<u64>,
    i: usize,
    n: usize,
    /// The next word of the sieved segment to read, and the number its bits start from.
    w: usize,
    base: u64,
}

impl SievingPrimes {
    /// The primes from [`presieve::MAX`] up to `limit`.
    pub(super) fn new(limit: u64) -> Self {
        let first = presieve::MAX + 2;
        if limit < SMALL {
            // The primes from 167 to the limit, from the table.
            return SievingPrimes {
                small: tiny_primes(limit),
                s: 0,
                erat: None,
                tiny: &[],
                t: 0,
                buf: Vec::new(),
                i: 0,
                n: 0,
                w: 0,
                base: 0,
            };
        }
        let erat = (limit >= first).then(|| Erat::new(first, limit, limit.isqrt()));
        SievingPrimes {
            small: &[],
            s: 0,
            erat,
            tiny: tiny_primes(limit.isqrt()),
            t: 0,
            // Room for the primes to the limit, or a batch; and 64 more.
            buf: vec![0; (limit / 4).clamp(64, BATCH as u64) as usize + 64],
            i: 0,
            n: 0,
            w: 0,
            base: 0,
        }
    }

    /// The next prime, `u64::MAX` after the last.
    #[inline(always)]
    pub(super) fn next(&mut self) -> u64 {
        if let Some(&p) = self.small.get(self.s) {
            self.s += 1;
            return u64::from(p);
        }
        if self.i == self.n && !self.fill() {
            return u64::MAX;
        }
        let p = self.buf[self.i];
        self.i += 1;
        p
    }

    /// Fills the batch; `false` when there are no more primes.
    #[cold]
    fn fill(&mut self) -> bool {
        let Some(erat) = &mut self.erat else {
            return false;
        };
        self.i = 0;
        self.n = 0;
        while self.n == 0 {
            if self.w * 8 >= erat.bytes().len() || !erat.is_sieved() {
                let Some(high) = erat.begin() else {
                    return false;
                };
                let sqrt = high.isqrt();
                while self.t < self.tiny.len() && u64::from(self.tiny[self.t]) <= sqrt {
                    erat.add(u64::from(self.tiny[self.t]));
                    self.t += 1;
                }
                erat.sieve();
                self.w = 0;
                self.base = erat.low();
            }
            self.n = extract(erat.bytes(), &mut self.w, &mut self.base, &mut self.buf);
        }
        true
    }
}

/// The primes above [`presieve::MAX`] up to `limit`, at most `2^16`, from the table of the primes
/// below `2^16` (`SMALL_PRIMES[i]` is the i-th prime, and 167 the 39th).
fn tiny_primes(limit: u64) -> &'static [u32] {
    let end = SMALL_PRIMES.partition_point(|&q| u64::from(q) <= limit);
    &SMALL_PRIMES[39.min(end)..end]
}

/// The segments of the sieve of a range, from 7 up, each sieved by the primes up to the square
/// root of its end, or up to a bound below it.
#[derive(Clone, Debug)]
pub(super) struct Segments {
    erat: Erat,
    source: SievingPrimes,
    /// The next sieving prime.
    next: u64,
    /// The primes sieve up to this bound at most.
    bound: u64,
}

impl Segments {
    /// The segments of the numbers from `start` to `stop`, `7 <= start <= stop`, sieved by the
    /// primes up to `bound` at most, at most the square root of `stop`.
    pub(super) fn new(start: u64, stop: u64, bound: u64) -> Self {
        let bound = bound.min(stop.isqrt());
        let mut source = SievingPrimes::new(bound);
        let next = source.next();
        Segments {
            erat: Erat::new(start, stop, bound),
            source,
            next,
            bound,
        }
    }

    /// Sieves the next segment; `false` at the end.
    #[inline]
    pub(super) fn next_segment(&mut self) -> bool {
        let Some(high) = self.erat.begin() else {
            return false;
        };
        let sqrt = high.isqrt().min(self.bound);
        while self.next <= sqrt {
            self.erat.add(self.next);
            self.next = self.source.next();
        }
        self.erat.sieve();
        true
    }

    /// The sieved segment's bytes.
    #[inline(always)]
    pub(super) fn bytes(&self) -> &[u8] {
        self.erat.bytes()
    }

    /// The multiple of 30 the sieved segment's bytes start from.
    #[inline(always)]
    pub(super) fn low(&self) -> u64 {
        self.erat.low()
    }
}
