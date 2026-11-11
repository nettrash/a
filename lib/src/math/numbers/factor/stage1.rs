//! The exponent of the first stage of `p - 1`, `p + 1` and the elliptic curve method: the product
//! `E` of the largest powers of the primes up to `B1` that are at most `B1`, so that `E` is a
//! multiple of every number up to `B1` whose prime powers are all at most `B1`.
//!
//! `E` has about `1.44 B1` bits. It is computed in chunks of consecutive primes whose products
//! have at most [`CHUNK_BITS`] bits, each by a balanced tree of products of limbs packed with as
//! many prime powers as fit, so that the methods can go through `E` a chunk at a time: the elliptic
//! curve method takes the point of one chunk to an affine one, with an inversion, before the next.
//! For bounds up to [`CACHE_B1`], the chunks are kept and serve every curve or starting value.

use super::super::{ByteOrder, Number};
use super::field::limbs_of;
use super::primes::primes_in;

/// The bits of a chunk of the exponent at most: 2^22, half a megabyte, `B1` to about 2.9 million
/// in one chunk.
const CHUNK_BITS: u64 = 1 << 22;

/// The bounds up to which the chunks are kept, about 18 MB of them.
const CACHE_B1: u64 = 100_000_000;

/// A chunk of the exponent: the product of the prime powers of the primes from `lo` to `hi`.
pub(super) struct Chunk {
    /// The first prime and the last.
    pub(super) lo: u64,
    pub(super) hi: u64,
    /// The product, its limbs the least significant first, the top one not zero.
    pub(super) e: Vec<u64>,
}

/// The exponent of the first stage for the bound `b1`.
pub(super) struct Stage1 {
    /// The bound.
    pub(super) b1: u64,
    /// The chunks, when they are kept.
    chunks: Option<Vec<Chunk>>,
}

impl Stage1 {
    /// The exponent for the bound `b1`, its chunks computed now when they are to be kept.
    pub(super) fn new(b1: u64) -> Stage1 {
        let chunks = if b1 <= CACHE_B1 {
            let mut v = Vec::new();
            chunks(b1, |c| {
                v.push(c);
                true
            });
            Some(v)
        } else {
            None
        };
        Stage1 { b1, chunks }
    }

    /// Calls `f` with each chunk in turn, while it returns `true`; whether every chunk was given.
    pub(super) fn each(&self, mut f: impl FnMut(&Chunk) -> bool) -> bool {
        match &self.chunks {
            Some(v) => v.iter().all(f),
            None => chunks(self.b1, |c| f(&c)),
        }
    }
}

/// The largest power of the prime `p` that is at most `b1`.
#[inline]
pub(super) fn prime_power(p: u64, b1: u64) -> u64 {
    let mut q = p;
    while q <= b1 / p {
        q *= p;
    }
    q
}

/// Computes the chunks of the exponent for `b1` and calls `f` with each, while it returns
/// `true`; whether every chunk was given.
fn chunks(b1: u64, mut f: impl FnMut(Chunk) -> bool) -> bool {
    if b1 < 2 {
        return true;
    }
    let mut primes = primes_in(2, b1).peekable();
    while let Some(&lo) = primes.peek() {
        // Prime powers while their bits fit, counted from the bits of each.
        let mut factors = Vec::new();
        let mut bits = 0f64;
        let mut hi = lo;
        while let Some(&p) = primes.peek() {
            let q = prime_power(p, b1);
            let b = (q as f64).log2();
            if !factors.is_empty() && bits + b > CHUNK_BITS as f64 {
                break;
            }
            bits += b;
            factors.push(q);
            hi = p;
            primes.next();
        }
        let e = product(&factors);
        if !f(Chunk { lo, hi, e }) {
            return false;
        }
    }
    true
}

/// The product of `factors`, as limbs, by a balanced tree of products of packed limbs.
pub(super) fn product(factors: &[u64]) -> Vec<u64> {
    let order = ByteOrder::LittleEndian;
    let mut level: Vec<Number> = Vec::new();
    let mut acc = 1u64;
    for &q in factors {
        match acc.checked_mul(q) {
            Some(m) => acc = m,
            None => {
                level.push(Number::new_from_u64(acc, order));
                acc = q;
            }
        }
    }
    level.push(Number::new_from_u64(acc, order));
    while level.len() > 1 {
        let mut next = Vec::with_capacity(level.len().div_ceil(2));
        let mut it = level.into_iter();
        while let Some(a) = it.next() {
            match it.next() {
                Some(b) => next.push(a * b),
                None => next.push(a),
            }
        }
        level = next;
    }
    limbs_of(&level[0])
}
