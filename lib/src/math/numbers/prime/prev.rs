//! The previous probable prime, as the next one is found ([`next`](super::next)), downwards: the
//! odd numbers below the start are sieved in intervals by the small primes, and those left are
//! tested in order, the largest first.

use super::super::Number;
use super::super::mul::limbs::sub_1;
use super::super::mul::{Operand, STACK_LIMBS, on_heap, on_stack, to_limbs};
use super::super::pow::montgomery::Mont1;
use super::super::pow::montgomery::even;
use super::super::scratch::{Scratch, need, scratch};
use super::bpsw::{Bits, reciprocal, rem_by};
use super::next::{LAST_PRIME_64, interval, number, primes_beyond_tables, sieve_limit};
use super::tables::{GROUP, GROUPS, PRIMES};
use super::{bpsw, is_prime_u64, prime_itch, probable_prime_odd, selfridge_u64};
use core::mem::MaybeUninit;

/// `2^64 + 13`, the first prime above `2^64`: from it down to `2^64` there is no prime.
const FIRST_PRIME_ABOVE_64: u128 = (1 << 64) + 13;

/// The largest probable prime below `x`, in the byte order of `x`, or `None` when `x` is at most 2.
pub(super) fn prev_prime(x: &Number) -> Option<Number> {
    let order = x.byte_order;
    let n = Operand::of(x);
    if x.negative || n.significant == 0 {
        return None;
    }
    if n.significant <= 16 {
        let mut v = [0u64; 2];
        // SAFETY: the magnitude has at most 16 significant bytes, two limbs.
        unsafe { to_limbs(n, v.as_mut_ptr()) };
        let v = u128::from(v[0]) | u128::from(v[1]) << 64;
        if v <= 2 {
            return None;
        }
        if v <= FIRST_PRIME_ABOVE_64 {
            let p = if v > 1 << 64 {
                LAST_PRIME_64
            } else {
                prev_u64((v - 1) as u64)
            };
            return Some(number(&[p], order));
        }
    }
    let nn = n.significant.div_ceil(8);
    // The candidate, from `x - 1` made odd, and the test's memory for a candidate of either length:
    // a candidate may be one limb shorter than the start, never two, as a prime is found (at the
    // latest `2^64 + 13`) long before. The plan of the test is not monotone in the length (see
    // `next`), so both lengths are planned for. The sum saturates, as in `is_probable_prime`.
    let need = even(nn).saturating_add(prime_itch(nn).max(prime_itch(nn - 1)));
    let compute = |s: *mut MaybeUninit<u64>| {
        let s = scratch!(s.cast(), need);
        // SAFETY: the working memory holds the limbs counted.
        unsafe {
            // Read in before the rest is formed, which is formed at the call: formed here, it would
            // be held across the read.
            to_limbs(n, s.take(even(nn)));
            let c = s.take(even(nn));
            sub_1(c, nn, if *c & 1 == 1 { 2 } else { 1 });
            let len = if *c.add(nn - 1) == 0 { nn - 1 } else { nn };
            search(c, len, s.add(even(nn)), order)
        }
    };
    Some(if need <= STACK_LIMBS {
        on_stack(compute)
    } else {
        on_heap(need, compute)
    })
}

/// Whether the odd `n`, which no prime up to 163 divides and at least `167^2`, is a probable prime,
/// exactly as below `2^64`: Baillie–PSW with no trial division, for the numbers a sieve leaves.
pub(in crate::math::numbers) fn is_prime_sieved_u64(n: u64) -> bool {
    debug_assert!(n & 1 == 1 && n >= 167 * 167);
    let f = Mont1::new(n);
    let bits = Bits {
        n: &n,
        len: 64 - n.leading_zeros() as usize,
    };
    // SAFETY: the bits are those of `n`, odd and without factors up to 163, so that none of the
    // first `D`s of Selfridge's search shares a factor with it but by being a multiple of one.
    unsafe { bpsw(&f, bits, || selfridge_u64(n)) }
}

/// The residues modulo 30 that are prime to 30, in order.
const WHEEL: [u8; 8] = [1, 7, 11, 13, 17, 19, 23, 29];

/// The largest prime at most `n`, for `n` from 2 up.
pub(in crate::math::numbers) fn prev_u64(n: u64) -> u64 {
    if n < 7 {
        return [0, 0, 2, 3, 3, 5, 5][n as usize];
    }
    // The numbers prime to 30 from the last at most `n` down, and the position of each in the
    // wheel.
    let base = n - n % 30;
    let r = (n % 30) as u8;
    let (mut c, mut i) = match WHEEL.iter().rposition(|&w| w <= r) {
        Some(i) => (base + u64::from(WHEEL[i]), i),
        // `n = 30k`, `k >= 1`: `30k - 1`.
        None => (base - 1, 7),
    };
    loop {
        if is_prime_u64(c) {
            return c;
        }
        // From `30k + 1` down to `30(k - 1) + 29`; `c >= 7` as 7 is prime.
        if i == 0 {
            i = 7;
            c -= 2;
        } else {
            c -= u64::from(WHEEL[i] - WHEEL[i - 1]);
            i -= 1;
        }
    }
}

/// The largest probable prime from the odd start of `nn` limbs at `start`, above `2^64 + 13`, in
/// steps of 2 downwards: the interval sieved, the numbers left tested in order, the candidate in
/// limbs at `start` itself, and the test's working memory at `s`.
///
/// # Safety
///
/// The limbs at `start` must be valid for reads and writes, the top one not zero, and `s` for the
/// [`prime_itch`] of each length the candidates take, `nn` and `nn - 1`.
unsafe fn search(start: *mut u64, nn: usize, s: Scratch, order: super::super::ByteOrder) -> Number {
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
            composite.iter_mut().for_each(|c| *c = false);
            // `start - 2i = 0 mod p`: `2i = r mod p` for the start's remainder `r`, halved modulo
            // `p`.
            let mut mark = |p: u64, r: u64| {
                let mut i = if r & 1 == 0 { r / 2 } else { (r + p) / 2 } as usize;
                while i < odds {
                    composite[i] = true;
                    i += p as usize;
                }
            };
            for g in &GROUP[..GROUPS] {
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
            // The numbers left, the largest first.
            let mut at = 0;
            for (i, &c) in composite.iter().enumerate() {
                if c {
                    continue;
                }
                sub_1(start, nn, 2 * (i - at) as u64);
                at = i;
                if *start.add(nn - 1) == 0 {
                    nn -= 1;
                }
                let prime = if nn == 1 {
                    // Not reached: `2^64 + 13` is prime, and stops the search above `2^64`.
                    is_prime_u64(*start)
                } else {
                    let bits = Bits {
                        n: start,
                        len: 64 * nn - (*start.add(nn - 1)).leading_zeros() as usize,
                    };
                    probable_prime_odd(start, nn, bits, s)
                };
                if prime {
                    return number(core::slice::from_raw_parts(start, nn), order);
                }
            }
            // The next interval down.
            sub_1(start, nn, 2 * (odds - at) as u64);
            if *start.add(nn - 1) == 0 {
                nn -= 1;
            }
        }
    }
}
