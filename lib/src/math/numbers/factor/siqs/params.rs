//! The parameters of the sieve by the size of the number: the number of primes in the factor base,
//! the length of the interval sieved for each polynomial, the bound of the large primes, and whether
//! relations with two large primes are kept. They start from the tables of msieve and yafu and
//! were tuned by measurement (see `docs/math/siqs.md`); between two rows they are interpolated.

/// The parameters for one size of number.
#[derive(Clone, Copy, Debug)]
pub(super) struct Params {
    /// The number of primes in the factor base, 2 included.
    pub(super) fb_size: usize,
    /// The number of blocks of [`super::sieve::BLOCK`] bytes in the interval.
    pub(super) blocks: usize,
    /// The length of the interval when it is one block, at most a block.
    pub(super) interval: usize,
    /// The large primes go up to this times the largest prime of the factor base.
    pub(super) lp_mult: u32,
    /// Whether relations with two large primes are kept.
    pub(super) dlp: bool,
    /// The sieve threshold is this many bits below the logarithm of the largest polynomial value,
    /// in tenths of the bits of the large-prime bound.
    pub(super) slack: u32,
    /// The bits of the primes `A` is made of.
    pub(super) a_bits: u32,
}

/// Bits of the number, primes in the factor base, blocks, large-prime multiplier, two large primes,
/// slack (tenths of the large primes' bits), bits of the primes of `A`.
const TABLE: [(u32, u32, u32, u32, bool, u32, u32); 18] = [
    (60, 80, 1, 20, false, 10, 6),
    (80, 120, 1, 20, false, 11, 7),
    (100, 200, 1, 30, false, 12, 8),
    (120, 360, 1, 30, false, 13, 9),
    (140, 700, 1, 40, false, 13, 10),
    (150, 900, 1, 40, false, 13, 10),
    (166, 1260, 1, 50, false, 14, 10),
    (183, 2250, 1, 50, false, 16, 11),
    (199, 3460, 1, 60, false, 17, 11),
    (216, 7000, 2, 60, false, 17, 11),
    (233, 10000, 3, 60, true, 19, 11),
    (249, 20000, 5, 70, true, 19, 11),
    (266, 30000, 6, 80, true, 19, 11),
    (282, 50000, 8, 90, true, 19, 11),
    (299, 80000, 10, 100, true, 19, 11),
    (315, 110000, 11, 110, true, 19, 12),
    (332, 150000, 12, 120, true, 19, 12),
    (400, 300000, 16, 150, true, 19, 12),
];

/// The parameters for a number of `bits` bits.
pub(super) fn for_bits(bits: u32) -> Params {
    let mut params = table(bits);
    // Short intervals for small numbers, whose values are smooth often enough: measured, an eighth
    // of a block below 90 bits took a fifth less time than a block.
    params.interval = match bits {
        0..=90 => 8192,
        91..=120 => 16384,
        121..=166 => 32768,
        _ => super::sieve::BLOCK,
    };
    params
}

/// The parameters of the table for `bits` bits.
fn table(bits: u32) -> Params {
    let row = |r: (u32, u32, u32, u32, bool, u32, u32)| Params {
        fb_size: r.1 as usize,
        blocks: r.2 as usize,
        interval: super::sieve::BLOCK,
        lp_mult: r.3,
        dlp: r.4,
        slack: r.5,
        a_bits: r.6,
    };
    if bits <= TABLE[0].0 {
        return row(TABLE[0]);
    }
    for w in TABLE.windows(2) {
        let (lo, hi) = (w[0], w[1]);
        if bits <= hi.0 {
            let t = f64::from(bits - lo.0) / f64::from(hi.0 - lo.0);
            let mix =
                |a: u32, b: u32| (f64::from(a) + t * (f64::from(b) - f64::from(a))).round() as u32;
            return Params {
                fb_size: mix(lo.1, hi.1) as usize,
                blocks: mix(lo.2, hi.2).max(1) as usize,
                interval: super::sieve::BLOCK,
                lp_mult: mix(lo.3, hi.3),
                dlp: if t < 0.5 { lo.4 } else { hi.4 },
                slack: mix(lo.5, hi.5),
                a_bits: mix(lo.6, hi.6),
            };
        }
    }
    row(TABLE[TABLE.len() - 1])
}
