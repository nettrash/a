//! Trial division: the primes up to a bound that divide a number, with their exponents.
//!
//! The odd primes below 8192 come from the primality test's tables: a long number is reduced
//! modulo products of consecutive primes that fit in a limb, eight products a pass in independent
//! chains, and the primes of a product are tested on its remainder together, each by one
//! multiplication, as `x p^(-1) mod 2^64 <= (2^64 - 1) / p` exactly when `p` divides `x`; only a
//! product with a prime that divides is gone over a prime at a time. A prime found is divided out
//! by Hensel's exact division. Once the number fits in a limb, the primes are tested on it
//! directly, eight at a time while they are within the bound and their squares below it, the
//! quotient by `p` of a multiple of `p` being `x p^(-1) mod 2^64`, until the prime's square passes
//! the number, which is then 1 or prime. Primes beyond the tables come from a segmented sieve, in
//! products that fit in a limb for a long number.

use super::super::div::limbs::{div_2by1, invert_limb};
use super::super::prime::tables::{COUNT, GROUP, GROUPS, Group, PRIMES};
use super::field::{div_exact_1, inverse_limb, trim};
use super::primes::primes_in;
use std::sync::OnceLock;

/// The largest prime of the tables, 8191.
pub(super) const TABLE_MAX: u64 = 8191;

/// The primes up to `bound` that divide the limbs `x`, with their exponents, in increasing order,
/// and what is left of `x`: none and `x` itself for zero.
pub(super) fn trial_limbs(mut x: Vec<u64>, bound: u64) -> (Vec<(u64, u32)>, Vec<u64>) {
    trim(&mut x);
    let mut out = Vec::new();
    if x.is_empty() || bound < 2 {
        return (out, x);
    }
    // The factors of 2.
    let zeros = trailing_zeros(&x);
    if zeros > 0 {
        shift_right(&mut x, zeros);
        out.push((2, zeros as u32));
    }
    if x.len() == 1 {
        let rest = trial_u64(x[0], 0, bound, &mut out);
        return (out, vec![rest]);
    }
    // Groups of the tables, eight at a time in independent chains, while the number is longer than
    // a limb; a group's primes are tested on its remainder together, and gone over one at a time
    // only when one of them divides it.
    let mut g = 0;
    while g < GROUPS && x.len() > 1 {
        if PRIMES[GROUP[g].first as usize].p > bound {
            return (out, x);
        }
        let take = (GROUPS - g).min(8);
        let rems = rem_by_8(&x, &GROUP[g..g + take]);
        for k in 0..take {
            let grp = GROUP[g + k];
            let primes = &PRIMES[grp.first as usize..grp.end as usize];
            if !primes.iter().fold(false, |hit, p| hit | p.divides(rems[k])) {
                continue;
            }
            for p in primes {
                if p.p > bound {
                    return (out, x);
                }
                if p.divides(rems[k]) {
                    let mut e = 0;
                    while let Some(q) = div_exact_1(&x, p.p, p.inv) {
                        x = q;
                        e += 1;
                    }
                    out.push((p.p, e));
                }
            }
        }
        g += take;
        if x.len() == 1 {
            // The rest by the primes after the groups done.
            let next = if g < GROUPS {
                GROUP[g].first as usize
            } else {
                COUNT
            };
            let rest = trial_u64(x[0], next, bound, &mut out);
            return (out, vec![rest]);
        }
    }
    if bound <= TABLE_MAX || x.len() <= 1 {
        return (out, x);
    }
    // Beyond the tables: the table above them, made on first use, four groups a pass.
    let ext = ext();
    let mut g = 0;
    while g < ext.groups.len() && x.len() > 1 {
        let take = (ext.groups.len() - g).min(4);
        let groups = &ext.groups[g..g + take];
        let mut d = [0u64; 4];
        let mut v = [0u64; 4];
        for (k, grp) in groups.iter().enumerate() {
            d[k] = grp.d;
            v[k] = grp.v;
        }
        let rems = rem_by_n(&x, &d[..take], &v[..take]);
        for (grp, &r) in groups.iter().zip(&rems) {
            for p in &ext.primes[grp.first as usize..grp.end as usize] {
                if p.p > bound {
                    return (out, x);
                }
                if p.divides(r) {
                    let mut e = 0;
                    while let Some(q) = div_exact_1(&x, p.p, p.inv) {
                        x = q;
                        e += 1;
                    }
                    out.push((p.p, e));
                }
            }
        }
        g += take;
        if x.len() == 1 {
            let next = ext
                .groups
                .get(g)
                .map_or(EXT_LIMIT + 1, |grp| ext.primes[grp.first as usize].p);
            let rest = trial_u64_from(x[0], next, bound, &mut out);
            return (out, vec![rest]);
        }
    }
    if bound <= EXT_LIMIT || x.len() <= 1 {
        return (out, x);
    }
    // Beyond that table: products of sieved primes that fit in a limb.
    let mut group: Vec<u64> = Vec::with_capacity(8);
    let mut product = 1u64;
    let mut sieve = primes_in(EXT_LIMIT + 1, bound);
    loop {
        let p = sieve.next();
        if let Some(p) = p
            && let Some(m) = product.checked_mul(p)
        {
            product = m;
            group.push(p);
            continue;
        }
        // The group is full or the primes ran out: its remainder, and each of its primes.
        if !group.is_empty() {
            let s = product.leading_zeros();
            let d = product << s;
            let r = rem_by(&x, d, invert_limb(d));
            for &q in &group {
                if r.is_multiple_of(q) {
                    let inv = inverse_limb(q);
                    let mut e = 0;
                    while let Some(y) = div_exact_1(&x, q, inv) {
                        x = y;
                        e += 1;
                    }
                    out.push((q, e));
                }
            }
        }
        let Some(p) = p else {
            return (out, x);
        };
        if x.len() == 1 {
            let rest = trial_u64_from(x[0], p, bound, &mut out);
            return (out, vec![rest]);
        }
        group.clear();
        group.push(p);
        product = p;
    }
}

/// The primes above the tables up to which a table like theirs is made on first use of a bound
/// beyond them: 81,000 primes, 2.6 MB with their groups.
const EXT_LIMIT: u64 = 1 << 20;

/// A prime of the table above the tables, as [`Prime`](super::super::prime::tables::Prime).
struct ExtPrime {
    /// The prime.
    p: u64,
    /// Its inverse modulo `2^64`.
    inv: u64,
    /// `floor((2^64 - 1) / p)`.
    max: u64,
}

impl ExtPrime {
    /// Whether the prime divides `x`.
    #[inline(always)]
    fn divides(&self, x: u64) -> bool {
        x.wrapping_mul(self.inv) <= self.max
    }
}

/// A run of consecutive primes of the table whose product fits in a limb, as
/// [`Group`](super::super::prime::tables::Group).
struct ExtGroup {
    /// The index of the first prime and the index past the last.
    first: u32,
    end: u32,
    /// The product, normalized, and its reciprocal.
    d: u64,
    v: u64,
}

/// The table of the primes from 8192 to [`EXT_LIMIT`].
struct Ext {
    primes: Vec<ExtPrime>,
    groups: Vec<ExtGroup>,
}

/// The table above the tables, made on first use.
fn ext() -> &'static Ext {
    static EXT: OnceLock<Ext> = OnceLock::new();
    EXT.get_or_init(|| {
        let primes: Vec<ExtPrime> = primes_in(TABLE_MAX + 1, EXT_LIMIT)
            .map(|p| ExtPrime {
                p,
                inv: inverse_limb(p),
                max: u64::MAX / p,
            })
            .collect();
        let mut groups = Vec::new();
        let mut i = 0;
        while i < primes.len() {
            let first = i;
            let mut product = 1u64;
            while let Some(m) = primes.get(i).and_then(|p| product.checked_mul(p.p)) {
                product = m;
                i += 1;
            }
            let d = product << product.leading_zeros();
            groups.push(ExtGroup {
                first: first as u32,
                end: i as u32,
                d,
                v: invert_limb(d),
            });
        }
        Ext { primes, groups }
    })
}

/// The remainders of the limbs `x` by up to four normalized `d`s with their reciprocals `v`,
/// independent chains a limb.
#[inline]
fn rem_by_n(x: &[u64], d: &[u64], v: &[u64]) -> [u64; 4] {
    let mut r = [0u64; 4];
    if d.len() == 4 {
        for &l in x.iter().rev() {
            for k in 0..4 {
                r[k] = div_2by1(r[k], l, d[k], v[k]).1;
            }
        }
    } else {
        for k in 0..d.len() {
            r[k] = rem_by(x, d[k], v[k]);
        }
    }
    r
}

/// [`trial_limbs`] for a number of one limb, odd, from the prime of index `start` of the tables:
/// pushes the primes up to `bound` that divide `x` and returns what is left, which is 1 when it is
/// a prime up to `bound` (pushed too).
pub(super) fn trial_u64(mut x: u64, start: usize, bound: u64, out: &mut Vec<(u64, u32)>) -> u64 {
    // Eight primes at a time while all are within the bound and their squares below `x`: their
    // tests are independent, and only a block that holds a divisor is gone over again.
    let mut i = start;
    while i + 8 <= COUNT {
        let last = PRIMES[i + 7].p;
        if last > bound || last * last > x {
            break;
        }
        let block = &PRIMES[i..i + 8];
        let mut hit = false;
        for p in block {
            hit |= p.divides(x);
        }
        if hit {
            for p in block {
                if p.divides(x) {
                    let mut e = 0;
                    while p.divides(x) {
                        x = x.wrapping_mul(p.inv);
                        e += 1;
                    }
                    out.push((p.p, e));
                }
            }
        }
        i += 8;
    }
    for p in &PRIMES[i..] {
        if p.p > bound || x == 1 {
            return x;
        }
        if p.p * p.p > x {
            // `x` has no factor up to its square root: it is prime.
            if x <= bound {
                out.push((x, 1));
                return 1;
            }
            return x;
        }
        if p.divides(x) {
            let mut e = 0;
            while p.divides(x) {
                x = x.wrapping_mul(p.inv);
                e += 1;
            }
            out.push((p.p, e));
        }
    }
    if bound <= TABLE_MAX || x == 1 {
        return x;
    }
    trial_u64_from(x, TABLE_MAX + 1, bound, out)
}

/// [`trial_u64`] for the primes from `first` up, without factors below it: those of the table
/// above the tables, then sieved ones.
fn trial_u64_from(mut x: u64, first: u64, bound: u64, out: &mut Vec<(u64, u32)>) -> u64 {
    if x == 1 {
        return x;
    }
    // Only primes up to the square root are tried; beyond, `x` is prime.
    let end = bound.min(super::primes::isqrt(x));
    let mut next = first;
    if first <= EXT_LIMIT && first <= end {
        let ext = ext();
        let start = ext.primes.partition_point(|p| p.p < first);
        for p in &ext.primes[start..] {
            if p.p > end || p.p > x / p.p {
                next = u64::MAX;
                break;
            }
            if p.divides(x) {
                let mut e = 0;
                while p.divides(x) {
                    x = x.wrapping_mul(p.inv);
                    e += 1;
                }
                out.push((p.p, e));
            }
        }
        if next != u64::MAX {
            next = EXT_LIMIT + 1;
        }
    }
    if next <= end {
        for p in primes_in(next, end) {
            if p > x / p {
                break;
            }
            if x.is_multiple_of(p) {
                let mut e = 0;
                while x.is_multiple_of(p) {
                    x /= p;
                    e += 1;
                }
                out.push((p, e));
            }
        }
    }
    if x > 1 && x <= bound {
        out.push((x, 1));
        return 1;
    }
    x
}

/// The number of zero bits at the bottom of the limbs `x`, not all zero.
fn trailing_zeros(x: &[u64]) -> u64 {
    let mut z = 0u64;
    for &l in x {
        if l != 0 {
            return z + u64::from(l.trailing_zeros());
        }
        z += 64;
    }
    z
}

/// Shifts the limbs `x` right by `k` bits, and drops the zero limbs at the top.
pub(super) fn shift_right(x: &mut Vec<u64>, k: u64) {
    // A whole limb is at most `x.len()`, which bounds `k / 64`.
    let limbs = (k / 64).min(x.len() as u64) as usize;
    x.drain(..limbs);
    let s = (k % 64) as u32;
    if s != 0 && !x.is_empty() {
        for i in 0..x.len() - 1 {
            x[i] = (x[i] >> s) | (x[i + 1] << (64 - s));
        }
        let last = x.len() - 1;
        x[last] >>= s;
    }
    trim(x);
}

/// The remainder of the limbs `x` by the normalized `d` with its reciprocal `v`, which is a
/// remainder by any divisor of `d` too: limb by limb from the top.
#[inline]
pub(super) fn rem_by(x: &[u64], d: u64, v: u64) -> u64 {
    let mut r = 0;
    for &l in x.iter().rev() {
        r = div_2by1(r, l, d, v).1;
    }
    r
}

/// The remainders of the limbs `x`, at least two, by up to eight groups' `d`s at once, in
/// independent chains: for two limbs a division each, the high limb, below `2d` as `d` is
/// normalized, reduced first by a subtraction.
#[inline]
fn rem_by_8(x: &[u64], g: &[Group]) -> [u64; 8] {
    let mut r = [0u64; 8];
    if x.len() == 2 {
        for (rk, grp) in r.iter_mut().zip(g) {
            let h = if x[1] >= grp.d { x[1] - grp.d } else { x[1] };
            *rk = div_2by1(h, x[0], grp.d, grp.v).1;
        }
        return r;
    }
    if g.len() == 8 {
        for &l in x.iter().rev() {
            for k in 0..8 {
                r[k] = div_2by1(r[k], l, g[k].d, g[k].v).1;
            }
        }
    } else {
        for (rk, grp) in r.iter_mut().zip(g) {
            *rk = rem_by(x, grp.d, grp.v);
        }
    }
    r
}
