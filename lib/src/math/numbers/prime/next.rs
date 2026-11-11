//! The next probable prime, after GMP's `mpz_nextprime`: the odd numbers above the start are sieved
//! in intervals by the small primes, and those left are tested in order.

use super::super::mul::limbs::add_1;
use super::super::mul::{Operand, Product, STACK_LIMBS, on_heap, on_stack, to_limbs};
use super::super::pow::montgomery::even;
use super::super::primes;
use super::super::scratch::{Scratch, need, scratch};
use super::super::sieve::SMALL_PRIMES;
use super::super::{ByteOrder, Number};
use super::bpsw::{Bits, reciprocal, rem_by};
use super::tables::{GROUP, GROUPS, Group, PRIMES};
use super::{is_prime_u64, prime_itch, probable_prime_odd};
use core::mem::MaybeUninit;

/// The largest prime below `2^64`.
pub(super) const LAST_PRIME_64: u64 = u64::MAX - 58;

/// The next probable prime above `x`, in the byte order of `x`.
pub(super) fn next_prime(x: &Number) -> Number {
    let order = x.byte_order;
    let n = Operand::of(x);
    if x.negative || n.significant <= 8 {
        let v = if x.negative || n.significant == 0 {
            0
        } else {
            let mut v = [0u64];
            // SAFETY: the magnitude has at most 8 significant bytes, one limb.
            unsafe { to_limbs(n, v.as_mut_ptr()) };
            v[0]
        };
        if v < LAST_PRIME_64 {
            return number(&[next_u64(v)], order);
        }
    }
    let nn = n.significant.div_ceil(8);
    // The candidate, from `x + 1` made odd, with room for two limbs more, as it may grow by one,
    // and the test's memory for a candidate of either length: the plan is not monotone, a shorter
    // candidate can need more (on aarch64, 437 limbs 803 more than 438, whose low halves of the
    // reduction take the transforms instead of Toom-Cook's methods).
    // The sum saturates, as in `is_probable_prime`: on 32-bit targets it passes `usize::MAX` for
    // numbers of 1.5 GB.
    let need = even(nn + 2).saturating_add(prime_itch(nn).max(prime_itch(nn + 1)));
    let compute = |s: *mut MaybeUninit<u64>| {
        let s = scratch!(s.cast(), need);
        // SAFETY: the working memory holds the limbs counted.
        unsafe {
            // Read in before the rest is formed, which is formed at the call: formed here, it
            // would be held across the read.
            to_limbs(n, s.take(even(nn + 2)));
            let c = s.take(even(nn + 2));
            *c.add(nn) = 0;
            *c.add(nn + 1) = 0;
            add_1(c, nn + 1, if *c & 1 == 1 { 2 } else { 1 });
            let len = if *c.add(nn) != 0 { nn + 1 } else { nn };
            search(c, len, s.add(even(nn + 2)), order)
        }
    };
    if need <= STACK_LIMBS {
        on_stack(compute)
    } else {
        on_heap(need, compute)
    }
}

/// The non-negative number of the limbs `l`, the top one not zero, in the byte order `order`.
pub(super) fn number(l: &[u64], order: ByteOrder) -> Number {
    let top = l[l.len() - 1];
    let len = 8 * l.len() - top.leading_zeros() as usize / 8;
    let mut v = Vec::with_capacity(len);
    // SAFETY: the new vector has room for the `len` bytes, all written before the length is set;
    // the limbs hold them.
    unsafe {
        Product {
            r: l.as_ptr(),
            len,
            negative: false,
        }
        .write(v.as_mut_ptr(), order);
        v.set_len(len);
    }
    Number {
        value: v,
        negative: false,
        byte_order: order,
    }
}

/// The residues modulo 30 that are prime to 30, and the steps from each to the next.
const WHEEL: [u8; 8] = [6, 4, 2, 4, 2, 4, 6, 2];

/// The smallest prime above `n`, for `n` below the largest prime below `2^64`.
fn next_u64(n: u64) -> u64 {
    if n < 7 {
        return [2, 2, 3, 5, 5, 7, 7][n as usize];
    }
    // The odd numbers prime to 3 and 5, from the first above `n`: `30k + 1, 7, 11, 13, 17, 19, 23,
    // 29`.
    const START: [u8; 30] = [
        1, 7, 7, 7, 7, 7, 7, 11, 11, 11, 11, 13, 13, 17, 17, 17, 17, 19, 19, 23, 23, 23, 23, 29,
        29, 29, 29, 29, 29, 31,
    ];
    const INDEX: [u8; 32] = [
        0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 3, 0, 0, 0, 4, 0, 5, 0, 0, 0, 6, 0, 0, 0, 0, 0, 7,
        0, 0,
    ];
    let base = n - n % 30;
    let mut c = base + u64::from(START[(n % 30) as usize]);
    let mut i = INDEX[(c % 30) as usize] as usize;
    loop {
        if is_prime_u64(c) {
            return c;
        }
        c += u64::from(WHEEL[i]);
        i = (i + 1) & 7;
    }
}

/// The primes up to which the interval is sieved for a start of `bits` bits: GMP's estimate of where
/// sieving more costs more than the tests it saves, `bits^2.5 / 124`, from 500 to `2^24`, which it
/// reaches at about 5000 bits; the sieve that finds them takes a byte for each odd number below.
/// Measured on an Apple M4 Max, the next and the previous prime from random numbers, with the kernels
/// of Montgomery's products and without: the lower end at 500 instead of GMP's 1000 took 5 to 7 %
/// less time at 65 and 72 bits and 2 to 3 % at 80, and the divisors 62 to 496 were none faster than
/// 124 by more than 1.2 % from 128 to 2048 bits (62 and 496 up to 8 % slower).
pub(super) fn sieve_limit(bits: u64) -> u64 {
    let b = bits as f64;
    ((b * b * b.sqrt() / 124.0) as u64).clamp(500, 1 << 24)
}

/// The odd numbers in an interval: GMP's, five per bit, about five times the average gap, up to
/// `2^24`, which it reaches at about 3.4 million bits, so that the flags fit in memory on any
/// target.
pub(super) fn interval(bits: u64) -> usize {
    (5 * bits).min(1 << 24) as usize
}

/// The odd primes below `limit`, beyond those of the tables: from the table of the primes below
/// `2^16` up to there, and from the segmented sieve beyond.
pub(super) fn primes_beyond_tables(limit: u64) -> Vec<u32> {
    let first = PRIMES[PRIMES.len() - 1].p + 2;
    if limit <= first {
        return Vec::new();
    }
    if limit <= 1 << 16 {
        let from = SMALL_PRIMES.partition_point(|&q| u64::from(q) < first);
        let to = SMALL_PRIMES.partition_point(|&q| u64::from(q) < limit);
        return SMALL_PRIMES[from..to].to_vec();
    }
    primes(first..limit).map(|q| q as u32).collect()
}

/// The groups of [`GROUP`] with a prime below `limit`, which the interval is sieved by.
pub(super) fn groups_to(limit: u64) -> &'static [Group] {
    &GROUP[..GROUP[..GROUPS].partition_point(|g| PRIMES[g.first as usize].p < limit)]
}

/// The smallest probable prime from the odd start of `nn` limbs at `start`, at least `2^64`, in
/// steps of 2: the interval sieved, the numbers left tested in order, the candidate in limbs at
/// `start` itself, and the test's working memory at `s`.
///
/// # Safety
///
/// The limbs at `start` must be valid for reads and writes, with two more above them, zero, and `s`
/// for the [`prime_itch`] of each length the candidates take, `nn` and, once they grow, `nn + 1`.
unsafe fn search(start: *mut u64, nn: usize, s: Scratch, order: ByteOrder) -> Number {
    need!(s, prime_itch(nn));
    // SAFETY: as the caller guarantees.
    unsafe {
        let top = *start.add(nn - 1);
        let bits = 64 * nn as u64 - u64::from(top.leading_zeros());
        let limit = sieve_limit(bits);
        let odds = interval(bits);
        let more = primes_beyond_tables(limit);
        let mut composite = vec![false; odds];
        let mut nn = nn;
        loop {
            // The offsets of the first odd multiples of each prime from the start.
            composite.iter_mut().for_each(|c| *c = false);
            let mut mark = |p: u64, r: u64| {
                // `start + 2i = 0 mod p`: `2i = p - r`, halved modulo `p`.
                let t = if r == 0 { 0 } else { p - r };
                let mut i = if t & 1 == 0 { t / 2 } else { (t + p) / 2 } as usize;
                while i < odds {
                    composite[i] = true;
                    i += p as usize;
                }
            };
            for g in groups_to(limit) {
                let r = rem_by(start, nn, g.d, g.v);
                for p in &PRIMES[g.first as usize..g.end as usize] {
                    if p.p < limit {
                        mark(p.p, r % p.p);
                    }
                }
            }
            let mut i = 0;
            while i < more.len() {
                // A group of primes whose product fits in a limb.
                let mut product = 1u64;
                let mut j = i;
                while j < more.len() {
                    match product.checked_mul(u64::from(more[j])) {
                        Some(x) => product = x,
                        None => break,
                    }
                    j += 1;
                }
                let d = product << product.leading_zeros();
                let r = rem_by(start, nn, d, reciprocal(d));
                for &p in &more[i..j] {
                    mark(u64::from(p), r % u64::from(p));
                }
                i = j;
            }
            // The numbers left, in order.
            let mut at = 0;
            for (i, &c) in composite.iter().enumerate() {
                if c {
                    continue;
                }
                let step = 2 * (i - at) as u64;
                at = i;
                if add_1(start, nn + 1, step) != 0 || *start.add(nn) != 0 {
                    nn += 1;
                }
                let bits = Bits {
                    n: start,
                    len: 64 * nn - (*start.add(nn - 1)).leading_zeros() as usize,
                };
                let prime = if nn == 1 {
                    // Just below `2^64`, above its last prime: none.
                    is_prime_u64(*start)
                } else {
                    probable_prime_odd(start, nn, bits, s)
                };
                if prime {
                    return number(core::slice::from_raw_parts(start, nn), order);
                }
            }
            // The next interval.
            if add_1(start, nn + 1, 2 * (odds - at) as u64) != 0 || *start.add(nn) != 0 {
                nn += 1;
            }
        }
    }
}
