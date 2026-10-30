//! The odd primes up to a limit, by the sieve of Eratosthenes on the odd numbers.
//!
//! The multiples of 3, 5 and 7 come from a pattern of 105 words, which repeats, as 105 words hold
//! `105 * 64` odd numbers, a multiple of `3 * 5 * 7`; the multiples of each other prime up to the
//! square root of the limit are crossed out from its square up, those of the primes below 64,
//! several to a word, a word at a time in a register.

/// The odd numbers up to a limit, one bit each, set for those that are not prime: bit `i` of the
/// words is the number `2i + 1`.
pub(super) struct Sieve {
    /// The bits, 64 to a word, the lowest first.
    composite: Vec<u64>,
    /// The largest number the sieve tells about.
    limit: u64,
}

/// The period of [`PATTERN`], in words.
const PERIOD: usize = 3 * 5 * 7;

/// The odd multiples of 3, 5 and 7 among the odd numbers of 105 words: bit `i` of word `w` is set
/// when `2 (64 w + i) + 1` is a multiple of one of them.
const PATTERN: [u64; PERIOD] = {
    let mut t = [0u64; PERIOD];
    let mut j = 0;
    while j < 64 * PERIOD {
        let n = 2 * j + 1;
        if n % 3 == 0 || n % 5 == 0 || n % 7 == 0 {
            t[j / 64] |= 1 << (j % 64);
        }
        j += 1;
    }
    t
};

impl Sieve {
    /// The odd primes up to `limit`.
    pub(super) fn new(limit: u64) -> Sieve {
        // Bits `0..=(limit - 1) / 2`, for the odd numbers up to `limit`.
        let bits = usize::try_from(limit / 2 + 1).expect("capacity overflow");
        let words = bits.div_ceil(64);
        let mut composite = Vec::with_capacity(words);
        for w in 0..words {
            composite.push(PATTERN[w % PERIOD]);
        }
        // 1 is not prime, and 3, 5 and 7 are.
        composite[0] = (composite[0] | 1) & !0b1110;
        let mut p = 11u64;
        while p.saturating_mul(p) <= limit {
            let i = (p / 2) as usize;
            if composite[i / 64] >> (i % 64) & 1 == 0 {
                let step = p as usize;
                let mut j = (p * p / 2) as usize;
                if step < 64 {
                    // Several multiples to a word: the word's bits gathered in a register.
                    while j < bits {
                        let w = j / 64;
                        let end = (64 * w + 64).min(bits);
                        let mut m = 0u64;
                        while j < end {
                            m |= 1 << (j % 64);
                            j += step;
                        }
                        composite[w] |= m;
                    }
                } else {
                    while j < bits {
                        composite[j / 64] |= 1 << (j % 64);
                        j += step;
                    }
                }
            }
            p += 2;
        }
        Sieve { composite, limit }
    }

    /// Calls `f` with each odd prime from `lo` to `hi`, in increasing order, for `hi` not above
    /// the limit.
    #[inline(always)]
    pub(super) fn each(&self, lo: u64, hi: u64, mut f: impl FnMut(u64)) {
        debug_assert!(hi <= self.limit);
        if lo > hi {
            return;
        }
        // The bits of the odd numbers from `lo` to `hi`: `lo / 2` holds the first odd number not
        // below `lo` (`lo - 1` would be even), and `(hi - 1) / 2` the last not above `hi`.
        let (first, last) = ((lo / 2) as usize, (hi.max(1) - 1) as usize / 2);
        if first > last {
            return;
        }
        let (w0, w1) = (first / 64, last / 64);
        for w in w0..=w1 {
            let mut bits = !self.composite[w];
            if w == w0 {
                bits &= u64::MAX << (first % 64);
            }
            if w == w1 {
                bits &= u64::MAX >> (63 - last % 64);
            }
            while bits != 0 {
                let i = 64 * w + bits.trailing_zeros() as usize;
                f(2 * i as u64 + 1);
                bits &= bits - 1;
            }
        }
    }
}

/// The remainders of a number `x` by increasing divisors above `sqrt(x)`, with a division only
/// when the quotient changes: for divisors from `x / (q + 1)` to `x / q`, the quotient is `q`, and
/// the remainder `x - q d`.
pub(super) struct Remainders {
    /// The number divided.
    x: u64,
    /// The quotient of the last divisor.
    q: u64,
    /// The largest divisor with the quotient `q`.
    last: u64,
}

impl Remainders {
    /// The remainders of `x`, for divisors from 1 up.
    pub(super) fn of(x: u64) -> Remainders {
        Remainders { x, q: 0, last: 0 }
    }

    /// `x mod d`, for `d` not below the last divisor.
    #[inline(always)]
    pub(super) fn of_divisor(&mut self, d: u64) -> u64 {
        if d > self.last {
            self.q = self.x / d;
            self.last = self.x.checked_div(self.q).unwrap_or(u64::MAX);
        }
        self.x - self.q * d
    }
}
