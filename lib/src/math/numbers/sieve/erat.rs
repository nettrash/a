//! The segmented sieve of Eratosthenes, after Walisch's primesieve.
//!
//! The numbers from `start` to `stop` are sieved in segments of [`segment_bytes`] bytes, 30
//! numbers a byte (see [`wheel`](super::wheel)). A segment starts pre-sieved by the primes to 163
//! ([`presieve`]), and the multiples of the other sieving primes, up to the square root of its end,
//! are crossed out by three methods by their size:
//!
//! - Small primes, with many multiples in a segment, segment after segment in parts that stay in
//!   the first-level cache: the eight multiples of a turn of the wheel at once, at offsets fixed by
//!   the prime's class, from a loop specialised for each class and starting position.
//! - Medium primes, with a few multiples, one at a time, kept in 64 lists by their wheel index, so
//!   that each list runs one loop specialised for its class and position.
//! - Large primes, with fewer multiples than segments, by Oliveira e Silva's bucket sieve on the
//!   wheel of 210: each prime waits in the list of the segment of its next multiple, so that a
//!   segment touches only the primes that cross out one of its numbers.

use super::presieve::{self, presieve};
use super::wheel::{
    CLASS, CORR_CUM30, CUM30, D30, INIT30, INIT210, KEEP_FROM, KEEP_TO, R30, STEP30, STEP210,
    byte_remainder,
};

/// The bytes a part of a segment takes for the small primes: the first-level data cache.
const CHUNK: usize = 128 << 10;

/// The largest segment, in bytes.
const MAX_SEGMENT: usize = 2048 << 10;

/// A sieving prime: `p / 30`, and the byte of its next multiple with the wheel index of it,
/// `multiple + (index << 23)`, as primesieve packs them.
#[derive(Clone, Copy, Debug)]
pub(super) struct SievingPrime {
    /// `p / 30`.
    sp: u32,
    /// The byte of the next multiple, below `2^23`, and the wheel index above.
    packed: u32,
}

impl SievingPrime {
    #[inline(always)]
    fn new(sp: u64, multiple: usize, index: usize) -> Self {
        debug_assert!(multiple < 1 << 23 && index < 1 << 9);
        SievingPrime {
            sp: sp as u32,
            packed: (multiple | (index << 23)) as u32,
        }
    }

    #[inline(always)]
    fn multiple(self) -> usize {
        (self.packed & ((1 << 23) - 1)) as usize
    }

    #[inline(always)]
    fn index(self) -> usize {
        (self.packed >> 23) as usize
    }
}

/// The sieve of one range, segment by segment.
#[derive(Clone, Debug)]
pub(super) struct Erat {
    /// The multiple of 30 that byte 0 of the segment starts from.
    low: u64,
    /// The first number of the range.
    start: u64,
    /// The last number of the range.
    stop: u64,
    /// The bytes of the segment, `bytes` of a full one, a multiple of 8.
    sieve: Vec<u8>,
    /// The bytes of a full segment.
    bytes: usize,
    /// The bytes of the current segment's sieve that hold numbers of the range, a multiple of 8.
    len: usize,
    /// Whether the segment has been sieved and the next one not.
    sieved: bool,
    /// The primes up to which the small, then the medium primes go.
    max_small: u64,
    max_medium: u64,
    small: Vec<SievingPrime>,
    medium: Medium,
    big: Big,
}

/// The bytes of a full segment for the range up to `stop` whose sieving primes go up to `limit`:
/// as primesieve chooses them, twice the limit, from a part of the first-level cache to
/// [`MAX_SEGMENT`], a power of two once there are large primes.
fn segment_bytes(limit: u64) -> (usize, u64, u64) {
    let want = limit.saturating_mul(2);
    let mut bytes = if want > CHUNK as u64 {
        (want - want % CHUNK as u64).min(MAX_SEGMENT as u64) as usize
    } else {
        CHUNK
    };
    let mut max_medium = 3 * bytes as u64;
    if limit > max_medium {
        bytes = 1 << (usize::BITS - 1 - bytes.leading_zeros());
        max_medium = 3 * bytes as u64;
    }
    let max_small = (CHUNK as u64 / 5).min(bytes as u64 / 5);
    (bytes, max_small.min(limit), max_medium.min(limit))
}

impl Erat {
    /// A sieve of the numbers from `start` to `stop`, `7 <= start <= stop`, by primes up to `limit`
    /// at most the square root of `stop`.
    pub(super) fn new(start: u64, stop: u64, limit: u64) -> Erat {
        debug_assert!(7 <= start && start <= stop);
        let (mut bytes, max_small, max_medium) = segment_bytes(limit);
        let low = start - byte_remainder(start);
        // A range in one segment, and no large primes: a sieve of its length.
        let last = (stop - byte_remainder(stop) - low) / 30 + 1;
        if last <= bytes as u64 && limit <= max_medium {
            bytes = (last as usize).next_multiple_of(8);
        }
        Erat {
            low,
            start,
            stop,
            sieve: vec![0; bytes],
            bytes,
            len: 0,
            sieved: false,
            max_small,
            max_medium,
            small: Vec::new(),
            medium: Medium::new(),
            big: Big::new(bytes, limit, max_medium),
        }
    }

    /// The multiple of 30 that byte 0 of the segment starts from.
    #[inline(always)]
    pub(super) fn low(&self) -> u64 {
        self.low
    }

    /// The last number of the current segment.
    #[inline(always)]
    pub(super) fn high(&self) -> u64 {
        self.low
            .saturating_add(30 * self.bytes as u64 + 6)
            .min(self.stop)
    }

    /// Moves to the next segment when the current one has been sieved, and gives the last number
    /// of the segment to sieve, or `None` when the range is done.
    #[inline]
    pub(super) fn begin(&mut self) -> Option<u64> {
        if self.sieved {
            if self.high() >= self.stop {
                return None;
            }
            self.low += 30 * self.bytes as u64;
            self.sieved = false;
        }
        Some(self.high())
    }

    /// Whether the current segment has been sieved.
    #[inline(always)]
    pub(super) fn is_sieved(&self) -> bool {
        self.sieved
    }

    /// The bytes of the sieved segment, a multiple of 8, the bits of the numbers outside the range
    /// clear.
    #[inline(always)]
    pub(super) fn bytes(&self) -> &[u8] {
        &self.sieve[..self.len]
    }

    /// Adds the sieving prime `p`, above [`presieve::MAX`] and at most the limit, for the segment to
    /// sieve, after [`begin`](Self::begin).
    pub(super) fn add(&mut self, p: u64) {
        debug_assert!(p > presieve::MAX && !self.sieved);
        // The first multiple `p q`, `q >= p` prime to 30 (to 210 for large primes), above
        // `low + 6`, the first number of the segment, in 128 bits as it may pass `2^64`.
        let after = self.low + 6;
        let q = (after / p + 1).max(p);
        let big = p > self.max_medium;
        let (f, w) = if big {
            let (f, w) = INIT210[(q % 210) as usize];
            (f, w as usize)
        } else {
            let (f, w) = INIT30[(q % 30) as usize];
            (f, w as usize)
        };
        let multiple = u128::from(p) * u128::from(q + u64::from(f));
        if multiple > u128::from(self.stop) {
            return;
        }
        let i = ((multiple as u64 - after) / 30) as usize;
        let c = CLASS[(p % 30) as usize] as usize;
        if big {
            self.big.add(p / 30, i, 48 * c + w);
        } else {
            let prime = SievingPrime::new(p / 30, i, 8 * c + w);
            if p > self.max_small {
                self.medium.add(prime);
            } else {
                self.small.push(prime);
            }
        }
    }

    /// Sieves the segment, after [`begin`](Self::begin).
    pub(super) fn sieve(&mut self) {
        debug_assert!(!self.sieved);
        let high = self.high();
        self.len = if high == self.stop {
            (((self.stop - byte_remainder(self.stop) - self.low) / 30 + 1) as usize)
                .next_multiple_of(8)
        } else {
            self.bytes
        };
        let s = &mut self.sieve[..self.len];
        presieve(s, self.low);
        for part in s.chunks_mut(CHUNK) {
            cross_small(part, &mut self.small);
        }
        self.medium.cross(s);
        self.big.cross(&mut self.sieve);
        let s = &mut self.sieve[..self.len];
        if self.low < self.start {
            s[0] &= KEEP_FROM[byte_remainder(self.start) as usize];
        }
        if high == self.stop {
            let last = ((self.stop - byte_remainder(self.stop) - self.low) / 30) as usize;
            s[last] &= KEEP_TO[byte_remainder(self.stop) as usize];
            s[last + 1..].fill(0);
        }
        self.sieved = true;
    }
}

// Small primes

/// Crosses out the multiples of the small primes in the part `s` of a segment, and keeps for each
/// the byte of its next multiple from the part's end.
fn cross_small(s: &mut [u8], primes: &mut [SievingPrime]) {
    for prime in primes {
        let sp = prime.sp as usize;
        let i = prime.multiple();
        let (i, w) = match prime.index() {
            0 => small::<0, 0>(s, sp, i),
            1 => small::<0, 1>(s, sp, i),
            2 => small::<0, 2>(s, sp, i),
            3 => small::<0, 3>(s, sp, i),
            4 => small::<0, 4>(s, sp, i),
            5 => small::<0, 5>(s, sp, i),
            6 => small::<0, 6>(s, sp, i),
            7 => small::<0, 7>(s, sp, i),
            8 => small::<1, 0>(s, sp, i),
            9 => small::<1, 1>(s, sp, i),
            10 => small::<1, 2>(s, sp, i),
            11 => small::<1, 3>(s, sp, i),
            12 => small::<1, 4>(s, sp, i),
            13 => small::<1, 5>(s, sp, i),
            14 => small::<1, 6>(s, sp, i),
            15 => small::<1, 7>(s, sp, i),
            16 => small::<2, 0>(s, sp, i),
            17 => small::<2, 1>(s, sp, i),
            18 => small::<2, 2>(s, sp, i),
            19 => small::<2, 3>(s, sp, i),
            20 => small::<2, 4>(s, sp, i),
            21 => small::<2, 5>(s, sp, i),
            22 => small::<2, 6>(s, sp, i),
            23 => small::<2, 7>(s, sp, i),
            24 => small::<3, 0>(s, sp, i),
            25 => small::<3, 1>(s, sp, i),
            26 => small::<3, 2>(s, sp, i),
            27 => small::<3, 3>(s, sp, i),
            28 => small::<3, 4>(s, sp, i),
            29 => small::<3, 5>(s, sp, i),
            30 => small::<3, 6>(s, sp, i),
            31 => small::<3, 7>(s, sp, i),
            32 => small::<4, 0>(s, sp, i),
            33 => small::<4, 1>(s, sp, i),
            34 => small::<4, 2>(s, sp, i),
            35 => small::<4, 3>(s, sp, i),
            36 => small::<4, 4>(s, sp, i),
            37 => small::<4, 5>(s, sp, i),
            38 => small::<4, 6>(s, sp, i),
            39 => small::<4, 7>(s, sp, i),
            40 => small::<5, 0>(s, sp, i),
            41 => small::<5, 1>(s, sp, i),
            42 => small::<5, 2>(s, sp, i),
            43 => small::<5, 3>(s, sp, i),
            44 => small::<5, 4>(s, sp, i),
            45 => small::<5, 5>(s, sp, i),
            46 => small::<5, 6>(s, sp, i),
            47 => small::<5, 7>(s, sp, i),
            48 => small::<6, 0>(s, sp, i),
            49 => small::<6, 1>(s, sp, i),
            50 => small::<6, 2>(s, sp, i),
            51 => small::<6, 3>(s, sp, i),
            52 => small::<6, 4>(s, sp, i),
            53 => small::<6, 5>(s, sp, i),
            54 => small::<6, 6>(s, sp, i),
            55 => small::<6, 7>(s, sp, i),
            56 => small::<7, 0>(s, sp, i),
            57 => small::<7, 1>(s, sp, i),
            58 => small::<7, 2>(s, sp, i),
            59 => small::<7, 3>(s, sp, i),
            60 => small::<7, 4>(s, sp, i),
            61 => small::<7, 5>(s, sp, i),
            62 => small::<7, 6>(s, sp, i),
            _ => small::<7, 7>(s, sp, i),
        };
        *prime = SievingPrime::new(sp as u64, i, w);
    }
}

/// One step of a prime of class `C` at position `W` of the wheel: crosses out its multiple at byte
/// `i`, or returns the multiple's byte from the end of `s` and its wheel index when it lies beyond.
macro_rules! step {
    ($s:ident, $i:ident, $sp:ident, $C:ident, $w:expr) => {{
        let w = $w;
        if $i >= $s.len() {
            return ($i - $s.len(), 8 * $C + w);
        }
        $s[$i] &= STEP30[$C][w].0;
        $i += $sp * D30[w] as usize + STEP30[$C][w].1 as usize;
    }};
}

/// Crosses out the multiples in `s` of a small prime of class `C`, `sp` its `p / 30`, from the one
/// at byte `i` and position `W`: to the end of the wheel's turn one at a time, then the eight
/// multiples of each turn at once, then one at a time to the end; returns the byte of the next
/// multiple from the end of `s` and its wheel index.
#[inline(always)]
fn small<const C: usize, const W: usize>(s: &mut [u8], sp: usize, mut i: usize) -> (usize, usize) {
    // To the end of the turn.
    if W == 0 {
        step!(s, i, sp, C, 0);
    }
    if W <= 1 {
        step!(s, i, sp, C, 1);
    }
    if W <= 2 {
        step!(s, i, sp, C, 2);
    }
    if W <= 3 {
        step!(s, i, sp, C, 3);
    }
    if W <= 4 {
        step!(s, i, sp, C, 4);
    }
    if W <= 5 {
        step!(s, i, sp, C, 5);
    }
    if W <= 6 {
        step!(s, i, sp, C, 6);
    }
    step!(s, i, sp, C, 7);
    loop {
        // Whole turns: the last of the eight multiples is `28 sp + CORR_CUM30[C][7]` bytes on.
        let last = sp * CUM30[7] as usize + CORR_CUM30[C][7] as usize;
        let turn = sp * 30 + R30[C] as usize;
        let end = s.len().saturating_sub(last);
        let o =
            [0, 1, 2, 3, 4, 5, 6, 7].map(|w| sp * CUM30[w] as usize + CORR_CUM30[C][w] as usize);
        while i < end {
            // SAFETY: `i < s.len() - last`, and every offset is at most `last`.
            unsafe {
                *s.get_unchecked_mut(i + o[0]) &= STEP30[C][0].0;
                *s.get_unchecked_mut(i + o[1]) &= STEP30[C][1].0;
                *s.get_unchecked_mut(i + o[2]) &= STEP30[C][2].0;
                *s.get_unchecked_mut(i + o[3]) &= STEP30[C][3].0;
                *s.get_unchecked_mut(i + o[4]) &= STEP30[C][4].0;
                *s.get_unchecked_mut(i + o[5]) &= STEP30[C][5].0;
                *s.get_unchecked_mut(i + o[6]) &= STEP30[C][6].0;
                *s.get_unchecked_mut(i + o[7]) &= STEP30[C][7].0;
            }
            i += turn;
        }
        step!(s, i, sp, C, 0);
        step!(s, i, sp, C, 1);
        step!(s, i, sp, C, 2);
        step!(s, i, sp, C, 3);
        step!(s, i, sp, C, 4);
        step!(s, i, sp, C, 5);
        step!(s, i, sp, C, 6);
        step!(s, i, sp, C, 7);
    }
}

// Medium primes

/// The medium primes, in 64 lists by the wheel index of their next multiple, none until the
/// first.
#[derive(Clone, Debug)]
struct Medium {
    lists: Vec<Vec<SievingPrime>>,
    /// The lists of the next segment, filled while the current ones are crossed.
    next: Vec<Vec<SievingPrime>>,
}

impl Medium {
    fn new() -> Self {
        Medium {
            lists: Vec::new(),
            next: Vec::new(),
        }
    }

    fn add(&mut self, prime: SievingPrime) {
        if self.lists.is_empty() {
            self.lists = (0..64).map(|_| Vec::new()).collect();
            self.next = (0..64).map(|_| Vec::new()).collect();
        }
        self.lists[prime.index()].push(prime);
    }

    /// Crosses out the multiples of the medium primes in the segment `s`.
    fn cross(&mut self, s: &mut [u8]) {
        if self.lists.is_empty() {
            return;
        }
        for k in 0..64 {
            if self.lists[k].is_empty() {
                continue;
            }
            let list = core::mem::take(&mut self.lists[k]);
            let c = k / 8;
            let next: &mut [Vec<SievingPrime>; 8] = (&mut self.next[8 * c..8 * c + 8])
                .try_into()
                .expect("eight lists");
            match k {
                0 => medium::<0, 0>(s, &list, next),
                1 => medium::<0, 1>(s, &list, next),
                2 => medium::<0, 2>(s, &list, next),
                3 => medium::<0, 3>(s, &list, next),
                4 => medium::<0, 4>(s, &list, next),
                5 => medium::<0, 5>(s, &list, next),
                6 => medium::<0, 6>(s, &list, next),
                7 => medium::<0, 7>(s, &list, next),
                8 => medium::<1, 0>(s, &list, next),
                9 => medium::<1, 1>(s, &list, next),
                10 => medium::<1, 2>(s, &list, next),
                11 => medium::<1, 3>(s, &list, next),
                12 => medium::<1, 4>(s, &list, next),
                13 => medium::<1, 5>(s, &list, next),
                14 => medium::<1, 6>(s, &list, next),
                15 => medium::<1, 7>(s, &list, next),
                16 => medium::<2, 0>(s, &list, next),
                17 => medium::<2, 1>(s, &list, next),
                18 => medium::<2, 2>(s, &list, next),
                19 => medium::<2, 3>(s, &list, next),
                20 => medium::<2, 4>(s, &list, next),
                21 => medium::<2, 5>(s, &list, next),
                22 => medium::<2, 6>(s, &list, next),
                23 => medium::<2, 7>(s, &list, next),
                24 => medium::<3, 0>(s, &list, next),
                25 => medium::<3, 1>(s, &list, next),
                26 => medium::<3, 2>(s, &list, next),
                27 => medium::<3, 3>(s, &list, next),
                28 => medium::<3, 4>(s, &list, next),
                29 => medium::<3, 5>(s, &list, next),
                30 => medium::<3, 6>(s, &list, next),
                31 => medium::<3, 7>(s, &list, next),
                32 => medium::<4, 0>(s, &list, next),
                33 => medium::<4, 1>(s, &list, next),
                34 => medium::<4, 2>(s, &list, next),
                35 => medium::<4, 3>(s, &list, next),
                36 => medium::<4, 4>(s, &list, next),
                37 => medium::<4, 5>(s, &list, next),
                38 => medium::<4, 6>(s, &list, next),
                39 => medium::<4, 7>(s, &list, next),
                40 => medium::<5, 0>(s, &list, next),
                41 => medium::<5, 1>(s, &list, next),
                42 => medium::<5, 2>(s, &list, next),
                43 => medium::<5, 3>(s, &list, next),
                44 => medium::<5, 4>(s, &list, next),
                45 => medium::<5, 5>(s, &list, next),
                46 => medium::<5, 6>(s, &list, next),
                47 => medium::<5, 7>(s, &list, next),
                48 => medium::<6, 0>(s, &list, next),
                49 => medium::<6, 1>(s, &list, next),
                50 => medium::<6, 2>(s, &list, next),
                51 => medium::<6, 3>(s, &list, next),
                52 => medium::<6, 4>(s, &list, next),
                53 => medium::<6, 5>(s, &list, next),
                54 => medium::<6, 6>(s, &list, next),
                55 => medium::<6, 7>(s, &list, next),
                56 => medium::<7, 0>(s, &list, next),
                57 => medium::<7, 1>(s, &list, next),
                58 => medium::<7, 2>(s, &list, next),
                59 => medium::<7, 3>(s, &list, next),
                60 => medium::<7, 4>(s, &list, next),
                61 => medium::<7, 5>(s, &list, next),
                62 => medium::<7, 6>(s, &list, next),
                _ => medium::<7, 7>(s, &list, next),
            }
            // The emptied list keeps its memory for the segment after.
            let mut list = list;
            list.clear();
            self.lists[k] = list;
        }
        core::mem::swap(&mut self.lists, &mut self.next);
    }
}

/// Crosses out the multiples in `s` of the medium primes of `list`, all of class `C` and position
/// `W`, and files each in `next` by the position of its next multiple.
#[inline(always)]
fn medium<const C: usize, const W: usize>(
    s: &mut [u8],
    list: &[SievingPrime],
    next: &mut [Vec<SievingPrime>; 8],
) {
    for prime in list {
        let sp = prime.sp as usize;
        let (i, w) = medium_one::<C, W>(s, sp, prime.multiple());
        next[w % 8].push(SievingPrime::new(sp as u64, i, w));
    }
}

/// Crosses out the multiples in `s` of one medium prime of class `C`, `sp` its `p / 30`, from the
/// one at byte `i` and position `W`, one at a time; returns the byte of the next multiple from the
/// end of `s` and its wheel index.
#[inline(always)]
fn medium_one<const C: usize, const W: usize>(
    s: &mut [u8],
    sp: usize,
    mut i: usize,
) -> (usize, usize) {
    loop {
        step!(s, i, sp, C, W % 8);
        step!(s, i, sp, C, (W + 1) % 8);
        step!(s, i, sp, C, (W + 2) % 8);
        step!(s, i, sp, C, (W + 3) % 8);
        step!(s, i, sp, C, (W + 4) % 8);
        step!(s, i, sp, C, (W + 5) % 8);
        step!(s, i, sp, C, (W + 6) % 8);
        step!(s, i, sp, C, (W + 7) % 8);
    }
}

// Large primes

/// The primes of a bucket.
const BUCKET: usize = 1024;

/// The large primes, by Oliveira e Silva's bucket sieve: a list of buckets of primes for each
/// segment to come, each prime in the list of the segment of its next multiple, by the byte of it
/// in that segment and its wheel index on the wheel of 210.
#[derive(Clone, Debug)]
struct Big {
    /// The lists of the segments from the current one, in a ring of a power of two.
    lists: Vec<Vec<Vec<SievingPrime>>>,
    /// The ring's position of the current segment.
    head: usize,
    /// `log2` of the bytes of a segment, and the bytes less one.
    shift: u32,
    mask: usize,
    /// Empty buckets, for reuse.
    free: Vec<Vec<SievingPrime>>,
}

impl Big {
    /// The large primes of a sieve of segments of `bytes` bytes, a power of two when there are
    /// any, with sieving primes up to `limit`.
    fn new(bytes: usize, limit: u64, max_medium: u64) -> Self {
        // A prime's next multiple is at most `10 (p / 30) + 10` bytes on: the ring holds that many
        // segments and two more.
        let segments = if limit > max_medium {
            ((10 * (limit / 30) as usize + 10) >> bytes.trailing_zeros()) + 2
        } else {
            0
        };
        Big {
            lists: (0..segments.next_power_of_two() * usize::from(segments > 0))
                .map(|_| Vec::new())
                .collect(),
            head: 0,
            shift: bytes.trailing_zeros(),
            mask: bytes.wrapping_sub(1),
            free: Vec::new(),
        }
    }

    /// Files a large prime whose next multiple is at byte `i` from the start of the next segment.
    fn add(&mut self, sp: u64, i: usize, index: usize) {
        let segment = i >> self.shift;
        let mask = self.lists.len() - 1;
        let slot = (self.head + segment) & mask;
        push(
            &mut self.lists[slot],
            &mut self.free,
            SievingPrime::new(sp, i & self.mask, index),
        );
    }

    /// Crosses out the multiples of the large primes in the full segment `s`, and moves to the
    /// next segment.
    fn cross(&mut self, s: &mut [u8]) {
        if self.lists.is_empty() {
            return;
        }
        let ring = self.lists.len() - 1;
        loop {
            let list = core::mem::take(&mut self.lists[self.head]);
            if list.is_empty() {
                break;
            }
            for mut bucket in list {
                for prime in &bucket {
                    let sp = prime.sp as usize;
                    let mut i = prime.multiple();
                    let step = STEP210[prime.index()];
                    s[i] &= step.mask;
                    i += step.factor as usize * sp + step.corr as usize;
                    let slot = (self.head + (i >> self.shift)) & ring;
                    push(
                        &mut self.lists[slot],
                        &mut self.free,
                        SievingPrime::new(sp as u64, i & self.mask, step.next as usize),
                    );
                }
                bucket.clear();
                self.free.push(bucket);
            }
        }
        self.head = (self.head + 1) & ring;
    }
}

/// Pushes `prime` on the list of buckets `list`, taking a new bucket when the last one is full.
#[inline(always)]
fn push(list: &mut Vec<Vec<SievingPrime>>, free: &mut Vec<Vec<SievingPrime>>, prime: SievingPrime) {
    match list.last_mut() {
        Some(bucket) if bucket.len() < BUCKET => bucket.push(prime),
        _ => {
            let mut bucket = free.pop().unwrap_or_else(|| Vec::with_capacity(BUCKET));
            bucket.push(prime);
            list.push(bucket);
        }
    }
}
