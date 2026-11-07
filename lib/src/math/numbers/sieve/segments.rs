//! The segments of a range, sieved one after the other, with the sieving primes they need, from a
//! sieve of their own.

use super::erat::Erat;
use super::presieve;
use super::wheel::BIT_VALUES;

/// The primes a batch of [`extract`] gives at most, with 64 more of room.
pub(super) const BATCH: usize = 1024;

/// Writes into `out` the primes of the sieve's words from the word `*w` on, the numbers of that word
/// from the multiple of 30 `*base`, until fewer than 64 places are left in `out` or the words end;
/// returns how many it wrote and moves `*w` and `*base` past the words read.
///
/// Each word gives its primes from its lowest bit up, by its trailing zeros, four at a time: the
/// places past its last prime get junk, which the next word or the count leaves out.
#[inline(always)]
pub(super) fn extract(bytes: &[u8], w: &mut usize, base: &mut u64, out: &mut [u64]) -> usize {
    let words = bytes.as_chunks::<8>().0;
    let mut n = 0;
    while *w < words.len() && n + 64 <= out.len() {
        let mut bits = u64::from_le_bytes(words[*w]);
        let b = *base;
        let count = bits.count_ones() as usize;
        for four in out[n..n + 64].as_chunks_mut::<4>().0 {
            if bits == 0 {
                break;
            }
            for place in four {
                *place = b.wrapping_add(BIT_VALUES[bits.trailing_zeros() as usize]);
                bits &= bits.wrapping_sub(1);
            }
        }
        n += count;
        *w += 1;
        *base = b.wrapping_add(240);
    }
    n
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
    /// The sieve of the primes, or `None` when there are none.
    erat: Option<Erat>,
    /// Its own sieving primes, from a sieve of Eratosthenes up to the square root of the limit.
    tiny: Vec<u32>,
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
        let erat = (limit >= first).then(|| Erat::new(first, limit, limit.isqrt()));
        SievingPrimes {
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

/// The primes above [`presieve::MAX`] up to `limit`, at most `2^16`, by the sieve of Eratosthenes
/// on the odd numbers.
fn tiny_primes(limit: u64) -> Vec<u32> {
    let limit = limit as usize;
    let mut composite = vec![false; limit / 2 + 1];
    let mut primes = Vec::new();
    let mut i = 1;
    while 2 * i < limit {
        if !composite[i] {
            let p = 2 * i + 1;
            if p as u64 > presieve::MAX {
                primes.push(p as u32);
            }
            let mut j = p * p / 2;
            while j < composite.len() {
                composite[j] = true;
                j += p;
            }
        }
        i += 1;
    }
    primes
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
