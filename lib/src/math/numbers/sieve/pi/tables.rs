//! The tables the counting methods share: the primes up to a limit, `π(n)` by a bit for each
//! number prime to 30, and the least prime factors and Möbius values of the numbers prime to 2310;
//! below [`SMALL`], the primes and `π` from tables computed at compile time.

use super::super::segments::each_prime;

/// The numbers below which the primes and `π` come from [`SMALL_PRIMES`] and [`SMALL_PI`].
pub(in crate::math::numbers::sieve) const SMALL: u64 = 1 << 16;

/// The number of primes below [`SMALL`].
const SMALL_COUNT: usize = 6542;

/// Whether each number below [`SMALL`] is composite, by a sieve of Eratosthenes at compile time.
const fn small_composite() -> [bool; SMALL as usize] {
    let mut composite = [false; SMALL as usize];
    composite[0] = true;
    composite[1] = true;
    let mut i = 2;
    while i * i < SMALL as usize {
        if !composite[i] {
            let mut j = i * i;
            while j < SMALL as usize {
                composite[j] = true;
                j += i;
            }
        }
        i += 1;
    }
    composite
}

/// The primes below [`SMALL`] as [`primes_to`] gives them: a 0 before them, so that
/// `SMALL_PRIMES[i]` is the `i`-th prime, and the first prime above, 65,537, after.
pub(in crate::math::numbers) static SMALL_PRIMES: [u32; SMALL_COUNT + 2] = {
    let composite = small_composite();
    let mut t = [0u32; SMALL_COUNT + 2];
    let (mut n, mut i) = (2, 1);
    while n < SMALL as usize {
        if !composite[n] {
            t[i] = n as u32;
            i += 1;
        }
        n += 1;
    }
    assert!(i == SMALL_COUNT + 1);
    t[i] = 65_537;
    t
};

/// `π` below [`SMALL`] in the blocks of [`PiTable`]: for each 240 numbers, the primes before them
/// (2, 3 and 5 included) and a bit for each of their numbers prime to 30, set for the primes.
pub(super) static SMALL_PI: [(u64, u64); (SMALL / 240 + 1) as usize] = {
    let composite = small_composite();
    let mut t = [(0u64, 0u64); (SMALL / 240 + 1) as usize];
    let mut n = 7;
    while n < SMALL as usize {
        if !composite[n] {
            t[n / 240].1 |= 1 << BIT_OF[n % 240];
        }
        n += 1;
    }
    let mut count = 3;
    let mut k = 0;
    while k < t.len() {
        t[k].0 = count;
        count += t[k].1.count_ones() as u64;
        k += 1;
    }
    t
};

/// The primes up to `limit`, below [`SMALL`], and the one after, as [`primes_to`] gives them, from
/// [`SMALL_PRIMES`].
pub(super) fn small_primes_to(limit: u64) -> &'static [u32] {
    debug_assert!(limit < SMALL);
    let n = PiTable::small().pi(limit) as usize;
    &SMALL_PRIMES[..n + 2]
}

/// The primes up to `limit` and the one after, with a 0 before them: `p[1] = 2`, `p[2] = 3`, and
/// so on, so that `p[i]` is the `i`-th prime; from the sieve's batches.
#[inline(never)]
pub(super) fn primes_to(limit: u64) -> Vec<u32> {
    let mut p = Vec::with_capacity(pi_upper(limit) as usize + 2);
    p.extend_from_slice(&[0, 2, 3, 5]);
    // The gaps between primes below `2^32` are below 400.
    each_prime(7, limit.saturating_add(400), |q| {
        p.push(q as u32);
        q <= limit
    });
    p
}

/// An upper bound of `π(x)`: `x / (ln x - 1.1) + 10` (Dusart: `π(x) < x / (ln x - 1.1)` for
/// `x >= 60184`, and a margin below).
pub(super) fn pi_upper(x: u64) -> u64 {
    if x < 100 {
        return 25;
    }
    let f = x as f64;
    (f / (f.ln() - 1.1)) as u64 + 10
}

/// The bits of the numbers prime to 30 in a block of 240, `30 j + {1, 7, 11, 13, 17, 19, 23, 29}`
/// at bit `8 j + k`, and the masks of the bits of the numbers up to each `r` of a block.
pub(super) static UP_TO: [u64; 240] = {
    let mut t = [0u64; 240];
    let mut r = 0;
    while r < 240 {
        let mut m = 0u64;
        let mut b = 0;
        while b < 64 {
            if BIT_VALUES[b] <= r as u64 {
                m |= 1 << b;
            }
            b += 1;
        }
        t[r] = m;
        r += 1;
    }
    t
};

/// The numbers of the 64 bits of a block of 240.
pub(super) static BIT_VALUES: [u64; 64] = {
    const OFFSETS: [u64; 8] = [1, 7, 11, 13, 17, 19, 23, 29];
    let mut t = [0u64; 64];
    let mut b = 0;
    while b < 64 {
        t[b] = 30 * (b as u64 / 8) + OFFSETS[b % 8];
        b += 1;
    }
    t
};

/// The bit of each residue modulo 240 of a number prime to 30, as [`BIT_VALUES`] places them, or 64
/// for the others.
pub(super) static BIT_OF: [u8; 240] = {
    let mut t = [64u8; 240];
    let mut b = 0;
    while b < 64 {
        t[BIT_VALUES[b] as usize] = b as u8;
        b += 1;
    }
    t
};

/// `π(n)` for `n` up to a limit: for each block of 240 numbers, the primes before it and a bit for
/// each of its numbers prime to 30, set for the primes; the blocks are [`SMALL_PI`] or those of
/// [`PiTable::blocks`].
#[derive(Clone, Copy)]
pub(super) struct PiTable<'a> {
    blocks: &'a [(u64, u64)],
}

impl PiTable<'static> {
    /// The table below [`SMALL`].
    #[inline(always)]
    pub(super) fn small() -> Self {
        PiTable { blocks: &SMALL_PI }
    }
}

impl<'a> PiTable<'a> {
    /// The table of the blocks `blocks`, from [`PiTable::blocks`].
    #[inline(always)]
    pub(super) fn of(blocks: &'a [(u64, u64)]) -> Self {
        PiTable { blocks }
    }

    /// The blocks of the table up to `limit` at least, from the primes `p` of [`primes_to`], which
    /// go beyond.
    #[inline(never)]
    pub(super) fn blocks(limit: u64, p: &[u32]) -> Vec<(u64, u64)> {
        let n = (limit / 240 + 1) as usize;
        let mut blocks = vec![(0u64, 0u64); n];
        for &q in &p[4..] {
            let q = u64::from(q);
            let k = (q / 240) as usize;
            if k >= n {
                break;
            }
            blocks[k].1 |= 1 << BIT_OF[(q % 240) as usize];
        }
        // 2, 3 and 5 have no bits: they are counted before the first block.
        let mut count = 3;
        for b in &mut blocks {
            b.0 = count;
            count += u64::from(b.1.count_ones());
        }
        blocks
    }

    /// The table's blocks of 240 numbers.
    #[inline(always)]
    pub(super) fn blocks_len(&self) -> usize {
        self.blocks.len()
    }

    /// `π(x)`, for `x` up to the table's limit.
    #[inline(always)]
    pub(super) fn pi(&self, x: u64) -> u64 {
        if x < 6 {
            return [0, 0, 1, 2, 2, 3][x as usize];
        }
        self.pi_from_6(x)
    }

    /// `Σ π(n / q)` for the primes `q` of `qs`, whose quotients are from 6 up to the table's limit:
    /// two at a time, the divisor known to be nonzero, the block's index kept to the last block (so
    /// that the look-up needs no check), the two words' bits counted together.
    #[inline(always)]
    pub(super) fn sum_quotients(&self, n: u64, qs: &[u32]) -> u64 {
        let last = self.blocks.len() - 1;
        let block = |v: u64| {
            // SAFETY: the index is at most the last block's.
            unsafe { *self.blocks.get_unchecked(((v / 240) as usize).min(last)) }
        };
        let quotient = |q: u32| {
            n / core::num::NonZeroU64::from(core::num::NonZeroU32::new(q).expect("a prime"))
        };
        let (pairs, rest) = qs.as_chunks::<2>();
        let (mut counts, mut bits) = (0u64, 0u64);
        for &[q0, q1] in pairs {
            let (v0, v1) = (quotient(q0), quotient(q1));
            let ((c0, b0), (c1, b1)) = (block(v0), block(v1));
            let w0 = b0 & UP_TO[(v0 % 240) as usize];
            let w1 = b1 & UP_TO[(v1 % 240) as usize];
            counts = counts.wrapping_add(c0).wrapping_add(c1);
            // One count of the two words, in a vector.
            bits += u64::from((u128::from(w0) | u128::from(w1) << 64).count_ones());
        }
        for &q in rest {
            let v = quotient(q);
            let (c, b) = block(v);
            counts = counts.wrapping_add(c);
            bits += u64::from((b & UP_TO[(v % 240) as usize]).count_ones());
        }
        counts.wrapping_add(bits)
    }

    /// `π(x)`, for `x` from 6 up to the table's limit.
    #[inline(always)]
    pub(super) fn pi_from_6(&self, x: u64) -> u64 {
        let (count, bits) = self.blocks[(x / 240) as usize];
        count + u64::from((bits & UP_TO[(x % 240) as usize]).count_ones())
    }
}

/// The product of the primes 2 to 11, whose multiples the factor table leaves out.
const WHEEL: u64 = 2310;

/// The numbers prime to 2310 below it, in order.
static COPRIME: [u16; 480] = {
    let mut t = [0u16; 480];
    let (mut n, mut i) = (1, 0);
    while n < WHEEL {
        if !n.is_multiple_of(2)
            && !n.is_multiple_of(3)
            && !n.is_multiple_of(5)
            && !n.is_multiple_of(7)
            && !n.is_multiple_of(11)
        {
            t[i] = n as u16;
            i += 1;
        }
        n += 1;
    }
    t
};

/// For each residue `r` modulo 2310, the index in [`COPRIME`] of the largest number prime to 2310
/// at most `r`, or -1 for `r = 0`.
static COPRIME_INDEX: [i16; 2310] = {
    let mut t = [0i16; 2310];
    let mut count = -1i16;
    let mut r = 0;
    while r < WHEEL as usize {
        let n = r as u64;
        if !n.is_multiple_of(2)
            && !n.is_multiple_of(3)
            && !n.is_multiple_of(5)
            && !n.is_multiple_of(7)
            && !n.is_multiple_of(11)
        {
            count += 1;
        }
        t[r] = count;
        r += 1;
    }
    t
};

/// The least prime factor and the Möbius value of each number prime to 2310 up to a limit, by its
/// index among them, as primecount packs them: 0 when the number has a square factor, else the
/// least prime factor (`u16::MAX` for 1 and the primes) with its lowest bit telling whether the
/// number of its prime factors is odd (set) or even (clear). A comparison `entry > p` tells for a
/// prime `p` that the number is square-free with all its prime factors above `p`.
pub(super) struct FactorTable {
    entries: Vec<u16>,
}

impl FactorTable {
    /// The index of the largest number prime to 2310 up to `n`, for `n >= 1`.
    #[inline(always)]
    pub(super) fn index(n: u64) -> u64 {
        (480 * (n / WHEEL)).wrapping_add_signed(i64::from(COPRIME_INDEX[(n % WHEEL) as usize]))
    }

    /// The number of index `i`.
    #[inline(always)]
    pub(super) fn number(i: u64) -> u64 {
        WHEEL * (i / 480) + u64::from(COPRIME[(i % 480) as usize])
    }

    /// The table of the numbers up to `y`, below `65,535^2`, from the primes `p` of [`primes_to`]
    /// to `y` at least.
    #[inline(never)]
    pub(super) fn new(y: u64, p: &[u32]) -> Self {
        let y = y.max(1);
        let size = Self::index(y) as usize + 1;
        let mut entries = vec![u16::MAX; size];
        // 1: no prime factors, an even number of them.
        entries[0] = u16::MAX ^ 1;
        let sqrt = y.isqrt();
        // The primes from 13 to y / 13, each crossing out its multiples by the numbers prime to
        // 2310 from 13 up.
        let first = p.partition_point(|&q| q < 13);
        for &p in p[first..].iter().take_while(|&&q| u64::from(q) <= y / 13) {
            let p = u64::from(p);
            let mut i = 1;
            loop {
                let m = p * Self::number(i);
                if m > y {
                    break;
                }
                let e = &mut entries[Self::index(m) as usize];
                if *e == u16::MAX {
                    *e = p as u16;
                } else if *e != 0 {
                    *e ^= 1;
                }
                i += 1;
            }
            if p <= sqrt {
                let square = p * p;
                let mut i = 0;
                loop {
                    let m = square * Self::number(i);
                    if m > y {
                        break;
                    }
                    entries[Self::index(m) as usize] = 0;
                    i += 1;
                }
            }
        }
        FactorTable { entries }
    }

    /// The entries of the indices from `first` to `last`.
    #[inline(always)]
    pub(super) fn entries(&self, first: u64, last: u64) -> &[u16] {
        &self.entries[first as usize..=last as usize]
    }
}
