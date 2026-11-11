//! The parameters of the sieve by the size of the number: the number of primes in the factor base,
//! the length of the interval sieved for each polynomial, the bound of the large primes, whether
//! relations with two large primes are kept, the thresholds, the primes left out of the sieve and
//! the size of the primes of `A`. They started from the tables of msieve and yafu and were tuned
//! by measurement (see `docs/math/siqs.md`); between two rows they are interpolated.

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
    /// A candidate is tried when the logarithms of the primes that divide its value add up to
    /// this many bits below the logarithm of the largest polynomial value, in tenths of the bits
    /// of the large-prime bound.
    pub(super) slack: u32,
    /// The bits of the primes `A` is made of.
    pub(super) a_bits: u32,
    /// The primes below this are not sieved: their share of a candidate is found from their
    /// roots once the sieve has found it.
    pub(super) tiny: u32,
    /// The sieve's own threshold is this much lower than the candidates', in tenths of a bit, so
    /// that it finds those that the tiny primes make up for.
    pub(super) margin: u32,
}

/// Bits of the number, primes in the factor base, length of the interval in KiB, large-prime
/// multiplier, two large primes, slack (tenths of the large primes' bits), bits of the primes of
/// `A`, tiny primes' bound, margin (tenths of a bit).
type Row = (u32, u32, u32, u32, bool, u32, u32, u32, u32);

/// The rows, by bits.
const TABLE: [Row; 22] = [
    (60, 60, 8, 10, false, 8, 8, 256, 160),
    (80, 100, 16, 10, false, 8, 8, 256, 160),
    (87, 120, 16, 10, false, 8, 8, 256, 160),
    (92, 130, 16, 20, false, 10, 8, 256, 180),
    (99, 180, 16, 12, false, 10, 8, 192, 160),
    (105, 180, 16, 15, false, 11, 8, 256, 180),
    (118, 310, 16, 15, false, 11, 8, 256, 180),
    (132, 620, 28, 35, false, 11, 9, 384, 180),
    (145, 850, 32, 50, false, 11, 9, 512, 220),
    (149, 900, 32, 40, false, 12, 9, 512, 220),
    (165, 1500, 64, 35, false, 13, 10, 768, 220),
    (182, 2250, 64, 50, false, 13, 10, 512, 260),
    (199, 3800, 64, 70, false, 13, 10, 768, 280),
    (216, 8000, 128, 45, false, 15, 10, 1024, 280),
    (233, 10500, 192, 75, true, 17, 11, 1024, 260),
    (249, 20000, 288, 100, true, 18, 11, 1024, 260),
    (266, 30000, 384, 90, true, 20, 11, 1024, 260),
    (282, 50000, 512, 100, true, 20, 11, 1024, 260),
    (299, 80000, 640, 110, true, 20, 11, 1024, 260),
    (315, 110000, 704, 120, true, 20, 12, 1024, 260),
    (332, 150000, 768, 130, true, 20, 12, 1024, 260),
    (400, 300000, 1024, 150, true, 20, 12, 1024, 260),
];

/// The parameters for a number of `bits` bits.
pub(super) fn for_bits(bits: u32) -> Params {
    table(bits)
}

/// The parameters of a row.
fn of_row(r: Row) -> Params {
    let len = r.2 as usize * 1024;
    let block = super::sieve::BLOCK;
    Params {
        fb_size: r.1 as usize,
        blocks: len.div_ceil(block).max(1),
        interval: len.min(block),
        lp_mult: r.3,
        dlp: r.4,
        slack: r.5,
        a_bits: r.6,
        tiny: r.7,
        margin: r.8,
    }
}

/// The parameters of the table for `bits` bits.
fn table(bits: u32) -> Params {
    if bits <= TABLE[0].0 {
        return of_row(TABLE[0]);
    }
    for w in TABLE.windows(2) {
        let (lo, hi) = (w[0], w[1]);
        if bits <= hi.0 {
            let t = f64::from(bits - lo.0) / f64::from(hi.0 - lo.0);
            let mix =
                |a: u32, b: u32| (f64::from(a) + t * (f64::from(b) - f64::from(a))).round() as u32;
            // The length of the interval and the tiny primes' bound geometrically.
            let geo = |a: u32, b: u32| {
                (f64::from(a) * (f64::from(b) / f64::from(a)).powf(t)).round() as u32
            };
            return of_row((
                bits,
                mix(lo.1, hi.1),
                geo(lo.2, hi.2),
                mix(lo.3, hi.3),
                if t < 0.5 { lo.4 } else { hi.4 },
                mix(lo.5, hi.5),
                mix(lo.6, hi.6),
                geo(lo.7, hi.7),
                mix(lo.8, hi.8),
            ));
        }
    }
    of_row(TABLE[TABLE.len() - 1])
}
