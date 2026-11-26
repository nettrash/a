//! `φ(x, a)` for `a <= 8` in constant time, after primecount's `PhiTiny`: the numbers up to `x`
//! prime to the first `a` primes repeat with the period of their product `P`, so that
//! `φ(x, a) = (x / P) φ(P) + φ(x mod P, a)`, the last from a table of the numbers below `P`.

use super::tables::{BIT_OF, UP_TO};

/// The first eight primes.
static PRIMES: [u64; 8] = [2, 3, 5, 7, 11, 13, 17, 19];

/// The largest `a` that [`phi_tiny`] takes.
pub(super) const MAX_A: u64 = 8;

/// The products of the first `a` primes, for `a` from 0 to 7.
static PRODUCTS: [u64; 8] = [1, 2, 6, 30, 210, 2310, 30030, 510510];

/// Euler's `φ` of [`PRODUCTS`]: the numbers below each prime to it.
static TOTIENTS: [u64; 8] = [1, 1, 2, 8, 48, 480, 5760, 92160];

/// For `a` from 4 to 7, the numbers below the product `P` of the first `a` primes prime to it, by
/// blocks of 240: a bit for each number prime to 30 in a block, set for those prime to `P`, and the
/// count before the block, computed at compile time by crossing out the multiples of 7 to `p_a`.
const fn table<const N: usize>(a: usize) -> [(u32, u64); N] {
    let pp = PRODUCTS[a];
    let mut t = [(0u32, 0u64); N];
    // Every number prime to 30 below `P`.
    let mut m = 1;
    while m < pp {
        let b = BIT_OF[(m % 240) as usize];
        if b < 64 {
            t[(m / 240) as usize].1 |= 1 << b;
        }
        m += 1;
    }
    // The multiples of 7 to `p_a` crossed out.
    let mut i = 3;
    while i < a {
        let p = PRIMES[i];
        let mut m = p;
        while m < pp {
            let b = BIT_OF[(m % 240) as usize];
            if b < 64 {
                t[(m / 240) as usize].1 &= !(1 << b);
            }
            m += 2 * p;
        }
        i += 1;
    }
    let mut count = 0;
    let mut k = 0;
    while k < N {
        t[k].0 = count;
        count += t[k].1.count_ones();
        k += 1;
    }
    t
}

static TABLE4: [(u32, u64); 1] = table(4);
static TABLE5: [(u32, u64); 10] = table(5);
static TABLE6: [(u32, u64); 126] = table(6);
static TABLE7: [(u32, u64); 2128] = table(7);

/// The tables of [`table`] by `a - 4`.
static TABLES: [&[(u32, u64)]; 4] = [&TABLE4, &TABLE5, &TABLE6, &TABLE7];

/// `φ(x, a)`, the numbers from 1 to `x` that none of the first `a` primes divides, for `a <= 8`.
#[inline]
pub(super) fn phi_tiny(x: u64, a: u64) -> u64 {
    match a {
        0 => x,
        1 => x - x / 2,
        2 => x / 6 * 2 + [0, 1, 1, 1, 1, 2][(x % 6) as usize],
        3 => {
            const T: [u8; 30] = [
                0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 7,
                7, 8,
            ];
            x / 30 * 8 + u64::from(T[(x % 30) as usize])
        }
        8 => phi7(x) - phi7(x / 19),
        _ => {
            let a = a as usize;
            let t = TABLES[a - 4];
            let (q, r) = (x / PRODUCTS[a], x % PRODUCTS[a]);
            let (count, bits) = t[(r / 240) as usize];
            q * TOTIENTS[a]
                + u64::from(count)
                + u64::from((bits & UP_TO[(r % 240) as usize]).count_ones())
        }
    }
}

/// `φ(x, 7)`, by the table of 510,510.
#[inline(always)]
fn phi7(x: u64) -> u64 {
    let t = &TABLE7;
    let (q, r) = (x / 510510, x % 510510);
    let (count, bits) = t[(r / 240) as usize];
    q * 92160 + u64::from(count) + u64::from((bits & UP_TO[(r % 240) as usize]).count_ones())
}

/// The `a` of [`phi_tiny`] for the sieving bound `y`: `π(y)`, at most 8.
pub(super) fn tiny_a(y: u64) -> u64 {
    PRIMES.iter().filter(|&&p| p <= y).count() as u64
}
