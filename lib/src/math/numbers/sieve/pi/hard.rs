//! The hard special leaves of Deléglise and Rivat's method: `φ(x / n, b - 1)` for the leaves
//! `n = p_b m` whose `x / n` is above `y`, from a segmented sieve of the numbers from 0 to `z`
//! crossed out by the primes in order, `p_b` after the leaves of `b` are read.
//!
//! As in primecount, the sieve holds a bit for each number prime to 30 (`30 j + {1, 7, ..., 29}`
//! in byte `j`), and the numbers left up to a point are counted from counters of the bits left in
//! blocks of the segment, which follow the crossing out, instead of Oliveira e Silva's binary
//! indexed tree. primecount counts from the last leaf's point on, by its counters then by words;
//! here the count of a leaf is formed afresh, with no branches: the bits before its block of 64
//! bytes, from the blocks' counters summed once for each prime ([`Sieve::prepare`]), and those of
//! the words of the block up to it, masked ([`rank`]), so that the leaves of a prime are
//! independent of one another and overlap in the processor, where a count carried from leaf to leaf
//! mispredicts the ends of its loops. Measured at `x = 10^16`, this took the hard leaves from 2.6
//! to 1.0 s.

use super::tables::{FactorTable, PiTable, UP_TO};

/// The numbers of the bits of a byte from its multiple of 30.
const OFFSETS: [u64; 8] = [1, 7, 11, 13, 17, 19, 23, 29];

/// The multipliers prime to 30 by position on the wheel, and the steps from each to the next.
static M30: [u64; 8] = [1, 7, 11, 13, 17, 19, 23, 29];
static D30: [u64; 8] = [6, 4, 2, 4, 2, 4, 6, 2];

/// The residues modulo 30 of the classes of primes, the class of each residue.
static R30: [u64; 8] = [1, 7, 11, 13, 17, 19, 23, 29];
static CLASS: [u8; 30] = {
    let mut t = [0u8; 30];
    let mut c = 0;
    while c < 8 {
        t[R30[c] as usize] = c as u8;
        c += 1;
    }
    t
};

/// The bit of each residue modulo 30 prime to 30.
static BIT_OF: [u8; 30] = {
    let mut t = [8u8; 30];
    let mut k = 0;
    while k < 8 {
        t[OFFSETS[k] as usize] = k as u8;
        k += 1;
    }
    t
};

/// For a prime of class `c` and a multiplier at position `w`: the bit of the multiple in its byte,
/// and the bytes to the next multiple beyond `D30[w] (p / 30)`.
static STEP: [[(u8, u64); 8]; 8] = {
    let mut t = [[(0u8, 0u64); 8]; 8];
    let mut c = 0;
    while c < 8 {
        let mut w = 0;
        while w < 8 {
            let here = (R30[c] * M30[w]) % 30;
            let next = (R30[c] * M30[(w + 1) % 8]) % 30;
            let sum = here + R30[c] * D30[w];
            assert!((sum - next).is_multiple_of(30));
            t[c][w] = (BIT_OF[here as usize], (sum - next) / 30);
            w += 1;
        }
        c += 1;
    }
    t
};

/// The first multiplier prime to 30 from each residue modulo 30: the step to it and its position.
static INIT: [(u8, u8); 30] = {
    let mut t = [(0u8, 0u8); 30];
    let mut r = 0;
    while r < 30 {
        let mut f = 0;
        loop {
            let m = ((r + f) % 30) as u64;
            let mut w = 0;
            while w < 8 && M30[w] != m {
                w += 1;
            }
            if w < 8 {
                t[r] = (f as u8, w as u8);
                break;
            }
            f += 1;
        }
        r += 1;
    }
    t
};

/// The bytes of the sieve of `n` numbers with the multiples of the primes from 7 to 19 crossed
/// out, from the multiple of 30 `30 q`, by the products 7 11 13 and 17 19, after which they
/// repeat.
const fn pattern<const N: usize>(primes: [u64; 3]) -> [u8; N] {
    let mut t = [0u8; N];
    let mut j = 0;
    while j < N {
        let mut k = 0;
        while k < 8 {
            let n = 30 * j as u64 + OFFSETS[k];
            if !n.is_multiple_of(primes[0])
                && !n.is_multiple_of(primes[1])
                && !n.is_multiple_of(primes[2])
            {
                t[j] |= 1 << k;
            }
            k += 1;
        }
        j += 1;
    }
    t
}
static PATTERN_A: [u8; 1001] = pattern([7, 11, 13]);
static PATTERN_B: [u8; 323] = pattern([17, 19, 17]);

/// The state of a prime in the sieve: the byte of its next multiple from the segment's start, and
/// the multiple's wheel index, `8 c + w`.
#[derive(Clone, Copy)]
struct State {
    multiple: u32,
    index: u8,
}

/// `log2` of the bytes of a block, and its words.
const BLOCK: u32 = 6;
const WORDS: usize = 1 << (BLOCK - 3);

/// The primes below this are crossed out with no counts, and the counts formed again from the
/// bits: they cross out so many bits of each block that taking each off its counter, one after the
/// other, costs more.
const SMALL: u64 = 400;

/// The sieve of the hard leaves.
struct Sieve {
    /// The bytes of the segment, whole groups of four blocks, zero past its end.
    bytes: Vec<u8>,
    /// The multiple of 240 the segment starts from.
    low: u64,
    /// The state of each prime by its index `b`.
    states: Vec<State>,
    /// The bits left in each block, and in the whole segment.
    blocks: Vec<u16>,
    total: u64,
    /// Whether the counters of the blocks are out of date.
    stale: bool,
    /// The bits before each block, for the leaves of one prime.
    before: Vec<u32>,
}

/// How a prime is crossed out.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Keep {
    /// No counts: the counters are formed again after.
    Nothing,
    /// The total only: the counters are out of date after.
    Total,
    /// The counters and the total.
    Counters,
}

impl Sieve {
    fn new(primes: usize) -> Self {
        Sieve {
            bytes: Vec::new(),
            low: 0,
            states: Vec::with_capacity(primes),
            blocks: Vec::new(),
            total: 0,
            stale: true,
            before: Vec::new(),
        }
    }

    /// Starts the segment of the numbers from `low`, a multiple of 240, to `high` (exclusive), with
    /// the multiples of the primes up to 19 crossed out, and counts its bits.
    #[inline(never)]
    fn reset(&mut self, low: u64, high: u64) {
        self.low = low;
        let n = high - low;
        // Whole groups of four blocks, for `prepare`.
        let len = (n.div_ceil(30 << (BLOCK + 2)) << (BLOCK + 2)) as usize;
        self.bytes.resize(len, 0);
        let used = (n.div_ceil(240) * 8) as usize;
        let q = low / 30;
        let (mut a, mut b) = ((q % 1001) as usize, (q % 323) as usize);
        for byte in &mut self.bytes[..used] {
            *byte = PATTERN_A[a] & PATTERN_B[b];
            a += 1;
            if a == 1001 {
                a = 0;
            }
            b += 1;
            if b == 323 {
                b = 0;
            }
        }
        self.bytes[used..].fill(0);
        // The numbers from `high` on are not of the segment.
        let last = (n - 1) % 240;
        let w = used - 8;
        let word = u64::from_le_bytes(self.bytes[w..used].try_into().expect("a word"))
            & UP_TO[last as usize];
        self.bytes[w..used].copy_from_slice(&word.to_le_bytes());
        self.recount();
    }

    /// Forms the counters of the blocks and the total from the bits.
    #[inline(never)]
    fn recount(&mut self) {
        let blocks = self.bytes.len() >> BLOCK;
        self.blocks.resize(blocks, 0);
        let mut total = 0u64;
        for (c, block) in self
            .blocks
            .iter_mut()
            .zip(self.bytes.as_chunks::<{ 1 << BLOCK }>().0)
        {
            let n = count_bytes(block);
            *c = n as u16;
            total += n;
        }
        self.total = total;
        self.stale = false;
    }

    /// Prepares the counts of the leaves of a prime: the bits before each block, by four chains of
    /// sums, over the four quarters of the blocks, which the processor runs side by side, and the
    /// sums of the quarters before each added after.
    #[inline(never)]
    fn prepare(&mut self) {
        if self.stale {
            self.recount();
        }
        let blocks = self.blocks.len();
        self.before.resize(blocks, 0);
        let q = blocks / 4;
        let (b0, rest) = self.before.split_at_mut(q);
        let (b1, rest) = rest.split_at_mut(q);
        let (b2, b3) = rest.split_at_mut(q);
        let (c0, rest) = self.blocks.split_at(q);
        let (c1, rest) = rest.split_at(q);
        let (c2, c3) = rest.split_at(q);
        let mut n = [0u32; 4];
        for i in 0..q {
            b0[i] = n[0];
            b1[i] = n[1];
            b2[i] = n[2];
            b3[i] = n[3];
            n[0] += u32::from(c0[i]);
            n[1] += u32::from(c1[i]);
            n[2] += u32::from(c2[i]);
            n[3] += u32::from(c3[i]);
        }
        let mut carry = 0u32;
        for (k, part) in [b1, b2, b3].into_iter().enumerate() {
            carry += n[k];
            part.iter_mut().for_each(|b| *b += carry);
        }
        // The blocks past the four quarters, when they are not a multiple of 4: none, as a segment
        // is a whole number of groups of 4 blocks.
        debug_assert_eq!(blocks % 4, 0);
    }

    /// Crosses out the multiples of the prime `p`, of index `b`, in the segment, keeping the counts
    /// as `keep` says.
    fn cross(&mut self, p: u64, b: usize, keep: Keep) {
        if b >= self.states.len() {
            self.add(p, b);
        }
        let st = self.states[b];
        let sp = (p / 30) as usize;
        let c = (st.index / 8) as usize;
        let w = (st.index % 8) as usize;
        let m = st.multiple as usize;
        let (m, w) = match keep {
            Keep::Nothing => {
                let r = plain_class(&mut self.bytes, sp, c, w, m);
                self.recount();
                r
            }
            Keep::Total => {
                let (m, w, cleared) = count_class::<false>(&mut self.bytes, &mut [], sp, c, w, m);
                self.total -= cleared;
                self.stale = true;
                (m, w)
            }
            Keep::Counters => {
                if self.stale {
                    self.recount();
                }
                let (m, w, cleared) =
                    count_class::<true>(&mut self.bytes, &mut self.blocks, sp, c, w, m);
                self.total -= cleared;
                (m, w)
            }
        };
        self.states[b] = State {
            multiple: m as u32,
            index: (8 * c + w) as u8,
        };
    }

    /// Adds the state of the prime `p`, of index `b`, from its first multiple from the segment's
    /// start; the states of the primes before it are there.
    fn add(&mut self, p: u64, b: usize) {
        // The primes up to 19 are pre-sieved, and have no states.
        self.states.resize(
            b,
            State {
                multiple: 0,
                index: 0,
            },
        );
        let q = self.low / p + 1;
        let (f, w) = INIT[(q % 30) as usize];
        let multiple = p * (q + u64::from(f));
        self.states.push(State {
            multiple: ((multiple - self.low) / 30) as u32,
            index: 8 * CLASS[(p % 30) as usize] + w,
        });
    }
}

/// The set bits of `bytes`, a multiple of 8.
#[inline(always)]
fn count_bytes(bytes: &[u8]) -> u64 {
    bytes
        .as_chunks::<8>()
        .0
        .iter()
        .map(|w| u64::from(u64::from_ne_bytes(*w).count_ones()))
        .sum()
}

/// For each word of a block, the masks that keep the words before it.
static WORDS_BEFORE: [[u64; WORDS]; WORDS] = {
    let mut t = [[0u64; WORDS]; WORDS];
    let mut j = 0;
    while j < WORDS {
        let mut i = 0;
        while i < j {
            t[j][i] = u64::MAX;
            i += 1;
        }
        j += 1;
    }
    t
};

/// The bits of the numbers from the segment's start to `low + stop`: those before the block of
/// `stop`, those of the words of the block before its word, and those of its word up to it.
#[inline(always)]
fn rank(words: &[[u8; 8]], before: &[u32], stop: u64) -> u64 {
    let w = (stop / 240) as usize;
    let block = w >> (BLOCK - 3);
    let n = u64::from(before[block]);
    // The masks come from a table: a mask formed from the index, a select, is turned back into a
    // loop to the index, with a branch for each word.
    let first = block << (BLOCK - 3);
    let ws: &[[u8; 8]; WORDS] = words[first..first + WORDS].try_into().expect("a block");
    let masks = &WORDS_BEFORE[w - first];
    let bits: u32 = ws
        .iter()
        .zip(masks)
        .map(|(word, &m)| (u64::from_le_bytes(*word) & m).count_ones())
        .sum();
    let x = u64::from_le_bytes(words[w]) & UP_TO[(stop % 240) as usize];
    n + u64::from(bits) + u64::from(x.count_ones())
}

/// Dispatches [`plain`] by the class `c`.
#[inline(always)]
fn plain_class(s: &mut [u8], sp: usize, c: usize, w: usize, m: usize) -> (usize, usize) {
    match c {
        0 => plain::<0>(s, sp, w, m),
        1 => plain::<1>(s, sp, w, m),
        2 => plain::<2>(s, sp, w, m),
        3 => plain::<3>(s, sp, w, m),
        4 => plain::<4>(s, sp, w, m),
        5 => plain::<5>(s, sp, w, m),
        6 => plain::<6>(s, sp, w, m),
        _ => plain::<7>(s, sp, w, m),
    }
}

/// Crosses out, with no counts, the multiples of a prime of class `C`, `sp` its `p / 30`, from the
/// one at byte `m` and position `w`: to the end of the wheel's turn one at a time, then the eight
/// multiples of each turn at once; returns the byte of the next multiple from the end of `s` and
/// its position.
#[inline(never)]
fn plain<const C: usize>(s: &mut [u8], sp: usize, mut w: usize, mut m: usize) -> (usize, usize) {
    let len = s.len();
    let d: [usize; 8] = core::array::from_fn(|k| sp * D30[k] as usize + STEP[C][k].1 as usize);
    let mask: [u8; 8] = core::array::from_fn(|k| !(1u8 << STEP[C][k].0));
    while w < 8 {
        if m >= len {
            return (m - len, w);
        }
        s[m] &= mask[w];
        m += d[w];
        w += 1;
    }
    // The offsets of the eight multiples of a turn from its first, and the turn's bytes.
    let mut o = [0usize; 8];
    for k in 1..8 {
        o[k] = o[k - 1] + d[k - 1];
    }
    let turn = o[7] + d[7];
    let end = len.saturating_sub(o[7]);
    while m < end {
        // SAFETY: `m < len - o[7]`, and the offsets are at most `o[7]`.
        unsafe {
            *s.get_unchecked_mut(m) &= mask[0];
            *s.get_unchecked_mut(m + o[1]) &= mask[1];
            *s.get_unchecked_mut(m + o[2]) &= mask[2];
            *s.get_unchecked_mut(m + o[3]) &= mask[3];
            *s.get_unchecked_mut(m + o[4]) &= mask[4];
            *s.get_unchecked_mut(m + o[5]) &= mask[5];
            *s.get_unchecked_mut(m + o[6]) &= mask[6];
            *s.get_unchecked_mut(m + o[7]) &= mask[7];
        }
        m += turn;
    }
    for k in 0..8 {
        if m >= len {
            return (m - len, k);
        }
        s[m] &= mask[k];
        m += d[k];
    }
    // A turn is longer than its last offset: `m` is past the end.
    (m - len, 0)
}

/// Dispatches [`count`] by the class `c`.
#[inline(always)]
fn count_class<const COUNT: bool>(
    s: &mut [u8],
    blocks: &mut [u16],
    sp: usize,
    c: usize,
    w: usize,
    m: usize,
) -> (usize, usize, u64) {
    match c {
        0 => count::<0, COUNT>(s, blocks, sp, w, m),
        1 => count::<1, COUNT>(s, blocks, sp, w, m),
        2 => count::<2, COUNT>(s, blocks, sp, w, m),
        3 => count::<3, COUNT>(s, blocks, sp, w, m),
        4 => count::<4, COUNT>(s, blocks, sp, w, m),
        5 => count::<5, COUNT>(s, blocks, sp, w, m),
        6 => count::<6, COUNT>(s, blocks, sp, w, m),
        _ => count::<7, COUNT>(s, blocks, sp, w, m),
    }
}

/// Crosses out the multiples of a prime of class `C`, `sp` its `p / 30`, from the one at byte `m`
/// and position `w`, counting the bits cleared; with `COUNT`, takes each off the counter of its
/// block. Returns the byte of the next multiple from the end of `s`, its position, and
/// the bits cleared.
#[inline(never)]
fn count<const C: usize, const COUNT: bool>(
    s: &mut [u8],
    blocks: &mut [u16],
    sp: usize,
    mut w: usize,
    mut m: usize,
) -> (usize, usize, u64) {
    let len = s.len();
    let mut cleared = 0u64;
    let d: [usize; 8] = core::array::from_fn(|k| sp * D30[k] as usize + STEP[C][k].1 as usize);
    macro_rules! hit {
        ($k:expr) => {{
            let k = $k;
            let byte = s[m];
            let bit = (byte >> STEP[C][k].0) & 1;
            s[m] = byte & !(1 << STEP[C][k].0);
            if COUNT {
                blocks[m >> BLOCK] -= u16::from(bit);
            }
            cleared += u64::from(bit);
        }};
    }
    while w < 8 {
        if m >= len {
            return (m - len, w, cleared);
        }
        hit!(w);
        m += d[w];
        w += 1;
    }
    loop {
        macro_rules! step {
            ($k:expr) => {{
                if m >= len {
                    return (m - len, $k, cleared);
                }
                hit!($k);
                m += d[$k];
            }};
        }
        step!(0);
        step!(1);
        step!(2);
        step!(3);
        step!(4);
        step!(5);
        step!(6);
        step!(7);
    }
}

/// The leaves `p m` of the segment from `low`, for the `m` from `min_m` (exclusive) to `max_m`,
/// square-free with their prime factors above `p`: `Σ -μ(m) φ(x / (p m), b - 1)`, `phi_b` the
/// numbers left before the segment.
#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn m_leaves(
    sieve: &Sieve,
    factor: &FactorTable,
    xp: u64,
    prime: u64,
    min_m: u64,
    max_m: u64,
    low: u64,
    phi_b: u64,
) -> u64 {
    let first = FactorTable::index(min_m) + 1;
    let entries = factor.entries(first, FactorTable::index(max_m));
    let words = sieve.bytes.as_chunks::<8>().0;
    let mut s = 0u64;
    // The entries of the leaves are gathered first, with no branch on whether each is one, which
    // would be mispredicted for about one entry in three; the leaves after, in any order, as each
    // is counted afresh.
    let mut leaves = [0u16; 256];
    for (c, chunk) in entries.chunks(256).enumerate() {
        let mut n = 0;
        for (k, &e) in chunk.iter().enumerate() {
            leaves[n] = k as u16;
            n += usize::from(u64::from(e) > prime);
        }
        let base = first + 256 * c as u64;
        for &k in &leaves[..n] {
            let e = chunk[k as usize];
            let m = FactorTable::number(base + u64::from(k));
            let xpm = xp / m;
            let phi_xpm = phi_b + rank(words, &sieve.before, xpm - low);
            // `-μ(m) φ`: `μ(m) = -1` for an odd number of prime factors, so the leaf adds its `φ`
            // then, and subtracts it otherwise, by a mask rather than a branch.
            let minus = u64::from(e & 1) ^ 1;
            s = s.wrapping_add((phi_xpm ^ minus.wrapping_neg()).wrapping_add(minus));
        }
    }
    s
}

/// The leaves `p q` of the segment from `low`, for the primes `q` of `qs`: `Σ φ(x / (p q), b - 1)`.
#[inline(never)]
fn q_leaves(sieve: &Sieve, qs: &[u32], xp: u64, low: u64, phi_b: u64) -> u64 {
    let words = sieve.bytes.as_chunks::<8>().0;
    let mut s = 0u64;
    for &q in qs {
        let xpq = xp / u64::from(q);
        s = s.wrapping_add(phi_b + rank(words, &sieve.before, xpq - low));
    }
    s
}

/// The hard special leaves, `S2_hard(x, y)`.
#[inline(never)]
pub(super) fn s2_hard(
    x: u64,
    y: u64,
    z: u64,
    c: u64,
    p: &[u32],
    pi: &PiTable,
    factor: &FactorTable,
) -> u64 {
    debug_assert_eq!(c, 8);
    let c = c as usize;
    let pi_sqrty = pi.pi(y.isqrt()) as usize;
    let mut max_b = pi.pi(z.isqrt().min(y)) as usize;
    if max_b <= c {
        return 0;
    }
    // Segments of half the first-level data cache, in numbers: measured, 2 % less time than the
    // whole of it at `10^17`.
    let size = (64u64 << 10) * 30;
    let mut sieve = Sieve::new(max_b + 1);
    let mut phi = vec![0u64; max_b + 1];
    let mut s = 0u64;
    let mut low = 0u64;
    while low < z && max_b > c {
        let high = (low + size).min(z);
        let low1 = low.max(1);
        // The primes that may still have leaves, from here on, are crossed out in every segment, so
        // that `phi[b]` counts the numbers left before the segment; those with leaves in it are
        // from `min_b` to `top`.
        max_b = max_b.min(pi.pi((x / low1).isqrt().min(z.isqrt()).min(y)) as usize);
        if max_b <= c {
            break;
        }
        let top = if high <= y {
            pi_sqrty.min(max_b)
        } else {
            max_b
        };
        let min_b = (c.max(pi.pi((z / high).min(u64::from(p[top]))) as usize) + 1).min(top + 1);
        sieve.reset(low, high);
        // How the prime of index `b` is crossed out: the small ones with no counts, the others
        // with the counters while primes after them have leaves here.
        let keep = |b: usize| {
            if u64::from(p[b]) < SMALL {
                Keep::Nothing
            } else if b + 1 >= min_b && b < top {
                Keep::Counters
            } else {
                Keep::Total
            }
        };
        let mut b = c + 1;
        while b < min_b {
            phi[b] += sieve.total;
            sieve.cross(u64::from(p[b]), b, keep(b));
            b += 1;
        }
        'segment: {
            // The leaves `p_b m`, `m` square-free with its prime factors above `p_b`.
            while b <= pi_sqrty.min(top) {
                let prime = u64::from(p[b]);
                let xp = x / prime;
                let min_m = (xp / high).min(y).max(y / prime);
                let max_m = (xp / low1).min(y);
                if prime >= max_m {
                    // No leaves for this prime and the ones after, here and from here on.
                    max_b = b - 1;
                    break 'segment;
                }
                sieve.prepare();
                s = s.wrapping_add(m_leaves(
                    &sieve, factor, xp, prime, min_m, max_m, low, phi[b],
                ));
                phi[b] += sieve.total;
                sieve.cross(prime, b, keep(b));
                b += 1;
            }
            // The leaves `p_b q`, `q > p_b` prime.
            while b <= top {
                let prime = u64::from(p[b]);
                let xp = x / prime;
                let xp_low = (xp / low1).min(y);
                let xp_high = (xp / high).min(y);
                let l = pi.pi(xp_low.min(z / prime)) as usize;
                let min_hard = xp_high.max(prime);
                if prime >= u64::from(p[l]) {
                    max_b = b - 1;
                    break 'segment;
                }
                // The primes `q` from `min_hard` (exclusive) to `p_l`.
                let first = pi.pi(min_hard) as usize + 1;
                sieve.prepare();
                s = s.wrapping_add(q_leaves(&sieve, &p[first..=l], xp, low, phi[b]));
                phi[b] += sieve.total;
                sieve.cross(prime, b, keep(b));
                b += 1;
            }
            // The primes with no leaves here.
            while b <= max_b {
                phi[b] += sieve.total;
                sieve.cross(u64::from(p[b]), b, keep(b));
                b += 1;
            }
        }
        low = high;
    }
    s
}
