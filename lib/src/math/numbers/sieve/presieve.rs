//! Pre-sieving: the multiples of the primes from 7 to [`MAX`] come from tables, which repeat,
//! instead of being crossed out one by one.
//!
//! Each table holds the bytes of the sieve of the numbers prime to 30 crossed out by two or three
//! of these primes, over their product, after which the bytes repeat, as 30 times the product is
//! a multiple of each. A segment starts as the AND of the tables at its position, four tables to a
//! pass, as Walisch's primesieve does (its tables of the primes to 163, after @zielaj).

use super::wheel::OFFSETS;

/// The largest prime pre-sieved; the sieving primes start above it.
pub(super) const MAX: u64 = 163;

/// The bytes of the sieve of the numbers prime to 30 with the multiples of `primes` crossed out,
/// over the `N` bytes of their product.
const fn table<const N: usize>(primes: &[u64]) -> [u8; N] {
    let mut t = [0xFFu8; N];
    let mut i = 0;
    while i < primes.len() {
        let p = primes[i];
        // The odd multiples of `p` prime to 3 and 5, `p m` for `m` from 1 by steps of 2, each in
        // its byte `(p m - 7) / 30` and bit.
        let mut m = 1;
        while p * m < 30 * N as u64 + 7 {
            let n = p * m;
            if !n.is_multiple_of(3) && !n.is_multiple_of(5) && n >= 7 {
                let j = ((n - 7) / 30) as usize;
                let r = n - 30 * j as u64;
                let mut k = 0;
                while OFFSETS[k] != r {
                    k += 1;
                }
                t[j] &= !(1 << k);
            }
            m += 2;
        }
        i += 1;
    }
    // The bytes wrap: the multiple of 30 N at offset 31 is bit 7 of the last byte, not of byte 0.
    t
}

static T0: [u8; 7 * 23 * 37] = table(&[7, 23, 37]);
static T1: [u8; 11 * 19 * 31] = table(&[11, 19, 31]);
static T2: [u8; 13 * 17 * 29] = table(&[13, 17, 29]);
static T3: [u8; 41 * 163] = table(&[41, 163]);
static T4: [u8; 43 * 157] = table(&[43, 157]);
static T5: [u8; 47 * 151] = table(&[47, 151]);
static T6: [u8; 53 * 149] = table(&[53, 149]);
static T7: [u8; 59 * 139] = table(&[59, 139]);
static T8: [u8; 61 * 137] = table(&[61, 137]);
static T9: [u8; 67 * 131] = table(&[67, 131]);
static T10: [u8; 71 * 127] = table(&[71, 127]);
static T11: [u8; 73 * 113] = table(&[73, 113]);
static T12: [u8; 79 * 109] = table(&[79, 109]);
static T13: [u8; 83 * 107] = table(&[83, 107]);
static T14: [u8; 89 * 103] = table(&[89, 103]);
static T15: [u8; 97 * 101] = table(&[97, 101]);

/// The tables, four to a pass.
static TABLES: [[&[u8]; 4]; 4] = [
    [&T0, &T1, &T2, &T3],
    [&T4, &T5, &T6, &T7],
    [&T8, &T9, &T10, &T11],
    [&T12, &T13, &T14, &T15],
];

/// The primes from 7 to [`MAX`], whose bits the tables cross out, as their own multiples.
static PRESIEVED: [u64; 35] = [
    7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101,
    103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163,
];

/// Writes into `sieve` the bytes from the multiple of 30 `low`: the numbers prime to 30 that none of
/// the primes from 7 to [`MAX`] divides, and those primes themselves.
pub(super) fn presieve(sieve: &mut [u8], low: u64) {
    for (pass, tables) in TABLES.iter().enumerate() {
        let mut pos = tables.map(|t| ((low / 30) % t.len() as u64) as usize);
        let mut at = 0;
        while at < sieve.len() {
            let mut n = sieve.len() - at;
            for k in 0..4 {
                n = n.min(tables[k].len() - pos[k]);
            }
            let s = &mut sieve[at..at + n];
            let [a, b, c, d] = [0, 1, 2, 3].map(|k| &tables[k][pos[k]..pos[k] + n]);
            if pass == 0 {
                and4(s, a, b, c, d);
            } else {
                and5(s, a, b, c, d);
            }
            at += n;
            for k in 0..4 {
                pos[k] += n;
                if pos[k] == tables[k].len() {
                    pos[k] = 0;
                }
            }
        }
    }
    if low <= MAX {
        // The pre-sieved primes are prime: their bits are set again.
        for &p in &PRESIEVED {
            if p >= low + 7 {
                let d = p - low;
                let j = ((d - 7) / 30) as usize;
                if j < sieve.len() {
                    let r = d - 30 * j as u64;
                    let k = OFFSETS.iter().position(|&o| o == r).unwrap_or(0);
                    sieve[j] |= 1 << k;
                }
            }
        }
    }
}

/// `s = a & b & c & d`, byte by byte; the slices have the length of `s`.
#[inline(always)]
fn and4(s: &mut [u8], a: &[u8], b: &[u8], c: &[u8], d: &[u8]) {
    let n = s.len();
    let (a, b, c, d) = (&a[..n], &b[..n], &c[..n], &d[..n]);
    for i in 0..n {
        s[i] = a[i] & b[i] & c[i] & d[i];
    }
}

/// `s &= a & b & c & d`, byte by byte; the slices have the length of `s`.
#[inline(always)]
fn and5(s: &mut [u8], a: &[u8], b: &[u8], c: &[u8], d: &[u8]) {
    let n = s.len();
    let (a, b, c, d) = (&a[..n], &b[..n], &c[..n], &d[..n]);
    for i in 0..n {
        s[i] &= a[i] & b[i] & c[i] & d[i];
    }
}
