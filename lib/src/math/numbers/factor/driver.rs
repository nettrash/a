//! [`Number::factor`](super::super::Number::factor): the method for each cofactor chosen by its size,
//! as PARI, FLINT and yafu choose it.
//!
//! - Trial division by the primes of the tables, below 8192 (the cofactor is then 1 or prime when
//!   it is below `8192^2`).
//! - Each cofactor: the exact test below `2^64` or Baillie–PSW above, then perfect powers, then a
//!   factor by the methods below, and both parts again.
//! - One limb: Hart's one line method up to 40 bits, then the elliptic curve method from a bound
//!   for the size, a few curves on the three smaller bounds first, and rho if that ever failed.
//! - Two limbs and more: a short run of rho, `p - 1` from 45 digits, then the elliptic curve
//!   method: for two limbs by fine levels from a bound for the size (from a small bound when the
//!   quadratic sieve is to come), beyond by GMP-ECM's levels for factors of 10, 12, 15, 20, ...
//!   digits; the quadratic sieve for cofactors in its range, once the curves have made factors of
//!   [`ECM_BEFORE_SIQS`] of the digits unlikely. The curves go on with growing bounds until a
//!   factor is found, so every factorisation ends.
//!
//! The search for a factor of a cofactor runs in one arithmetic modulo it, for rho, `p - 1` and all
//! the levels of the curves, which are taken in turn from one sequence. The stages of the curves'
//! bounds, the exponent of the first and the plan of the second, are kept for the life of the
//! program for the bounds that recur (up to `B1 = 10^6`, about 3 MB at most, made on first use),
//! so that a factorisation of a small number costs its curves alone. The longer second stages of
//! the levels from 25 digits keep about 0.1 MB more in their plans, and would keep bitmaps of 0.14,
//! 1.2 and 9.6 MB more were they to take the baby-step giant-step continuation, which the
//! cofactors that reach them (from 84 digits) do not: they take the continuation by polynomials.
//!
//! The constants below were measured against the quadratic sieve and the arithmetic of the crate
//! as they were on 2026-11-10, on an Apple M4 Max; they are to be measured again
//! when those change: [`SIQS_MIN_DIGITS`], [`ECM_BEFORE_SIQS`] and [`SIQS_FIRST_FINE`] against the
//! sieve, the others against the curves' arithmetic.

use super::ecm::{Curves, FIRST_SIGMA, curves_from};
use super::field::{Arith, Job, Reg, gcd_u64, limbs_of, number_of, with_field};
use super::pm1::Pm1Job;
use super::primes::isqrt;
use super::rho::{Rho, RhoJob};
use super::{ECM_WEIGHT, Setup, exponent_overflow, is_prime_limbs, power, siqs, trial};
use crate::math::numbers::ByteOrder;
use crate::math::numbers::pow::montgomery::Mont1;
use std::sync::OnceLock;

/// The cofactors from this many digits up to [`SIQS_MAX_DIGITS`] go to the quadratic sieve, once
/// the curves have made factors of [`ECM_BEFORE_SIQS`] of their digits unlikely. Measured on an
/// Apple M4 Max with the sieve and the curves of 2026-11-10, interleaved, the mean (median) time
/// per number: 2000 balanced semiprimes of 24 digits 200 us (187) from 24 digits against 293 us
/// (216) from 29, of 26 digits 310 us (318) against 523 us (359), 400 random numbers of 30 digits
/// 95 us against 120 us, and of 25 digits 32.4 us against 31.9 us; from 23 digits the random ones
/// of 25 digits took 36 us, from 25 the semiprimes of 24 digits 284 us. The digits are those of the
/// bits: `n` of 94 bits counts as 29.
pub(super) const SIQS_MIN_DIGITS: u64 = 24;

/// The largest cofactors, in digits, that go to the quadratic sieve.
pub(super) const SIQS_MAX_DIGITS: u64 = 110;

/// The share of a cofactor's digits whose factors the curves look for before the quadratic sieve:
/// yafu's 4/13 or so. Measured on an Apple M4 Max, against the shares 0.25 and 0.35, it took
/// 6.0 ms against 6.2 and 5.3 on random numbers of 50 digits, 66 ms against 74 and 63 at 60 digits
/// and 247 ms against 415 and 224 at 70; on balanced semiprimes, where the curves are wasted, it
/// cost 4 to 32 % more than the sieve alone, against 0 to 8 % for 0.25 and up to 58 % for 0.35.
/// Measured again on 2026-11-10, with the sieve from 24 digits: 0.15 and 0.20 took 18 to 33 % less
/// on balanced semiprimes of 26 to 36 digits but 1.4 to 2 times as long on random numbers of 30 to
/// 40 digits; 0.40 and 0.50 took longer on both.
pub(super) const ECM_BEFORE_SIQS: f64 = 0.30;

/// While the quadratic sieve is to come, the curves of two limbs start from this fine level,
/// `B1 = 191`, for factors of about 7 digits, just above those the first run of rho finds, rather
/// than from the level for the size. Measured on an Apple M4 Max, random numbers of 36 digits took
/// 0.25 ms against 0.85, products of primes of 10 and 26 digits 0.21 ms against 3.2, and balanced
/// semiprimes of 28 to 38 digits 10 to 26 % more. Measured again on 2026-11-10: from level 8,
/// balanced semiprimes of 26 to 32 digits took 15 to 28 % less and random numbers of 25 and 30
/// digits 15 to 24 % more; from levels 3 and 4, the semiprimes 10 to 50 % more.
const SIQS_FIRST_FINE: usize = 6;

/// The primes up to which the factorisation divides first: the tables'.
const TRIAL_BOUND: u64 = super::trial::TABLE_MAX;

/// The steps of rho tried first on a cofactor of two limbs, for factors up to about `2^22`, and of
/// three limbs and more, up to about `2^24`. Measured on an Apple M4 Max, against 4096 steps for
/// two limbs (31 µs), 2048 took 15 % less on balanced semiprimes of 20 digits and 5 % less at 24
/// digits, with the same medians on random numbers of 25 and 30 digits; 1024 steps made the median
/// of 30 digits 1.6 times longer, and none that of 25 digits 2.7 times.
const RHO_STEPS_2: u64 = 1 << 11;
const RHO_STEPS: u64 = 1 << 12;

/// The curves of the first block of each level, before blocks of the arithmetic's width: two, so
/// that the curves of a level that one curve is likely to finish (a small factor on the lowest
/// levels) do not all go through stage 1 first. Measured on an Apple M4 Max against blocks of 4
/// from the start, products of primes of 20 and 44 bits took 8.4 µs against 10.1, random words
/// 6.3 against 7.0, and balanced semiprimes of 20 digits 0.94 times the time; balanced words of 32
/// bits and 32 bits 1.06 times.
const FIRST_BLOCK: usize = 2;

/// The cofactors from this many digits up take `p - 1` before the curves, with these bounds: about
/// the cost of one curve with `B1 = 15,000`.
const PM1_MIN_DIGITS: u64 = 45;
const PM1_B1: u64 = 100_000;
const PM1_B2: u64 = 10_000_000;

/// The levels of the elliptic curve method above two limbs: the digits of the factors each is
/// meant for, `B1`, `B2 / B1`, and the curves, after GMP-ECM's table (its README) from 15 digits
/// up, with two smaller levels below. `B2` is `100 B1` up to 20 digits, and from 25 digits grows
/// with `B1`, 200, 400 and 800 times it at 25, 30 and 35 digits and 800 times beyond, since the
/// continuation by polynomials makes a long second stage cheap against the first. Those levels
/// take GMP-ECM's curves times the ratio of one curve's chances at `100 B1` and at their `B2` for a
/// factor of their digits by GMP-ECM's model (libecm 7.0.7's `ecmprob` at `10^(d - 0.5)`): 251,
/// 501 and 1119 curves at 25 to 35 digits for 300, 700 and 1800, so that each finds its factors as
/// often as before (measured, one curve found a prime of 25 digits 1.27 times as often at `200 B1`
/// as at `100 B1`, the model says 1.20).
///
/// Measured on an Apple M4 Max on 2026-11-11: the curves each level took, traced through `factor`
/// for products of a prime of 25 digits with one of 33 (2100 of them) and of 30 digits with one of
/// 45 (300), costed at each size by the time of one curve there (its stages kept, the bounds
/// interleaved, the machine quiet), took this share of the time with `100 B1` on every level, the
/// mean (the median):
///
/// | limbs | 6 | 8 | 12 | 16 |
/// | --- | --- | --- | --- | --- |
/// | prime of 25 digits | 0.93 (0.98) | 0.92 (0.95) | 0.90 (0.94) | 0.88 (0.94) |
/// | prime of 30 digits | 0.95 (0.93) | 0.94 (0.92) | 0.90 (0.88) | 0.88 (0.87) |
///
/// (the 90 % intervals of the means +-3 % at 25 digits, +-8 % at 30). With `400 B1` at 25 digits
/// (800 and 1600 above) the 25-digit primes took 1.00, 0.97, 0.93 and 0.90; with `800 B1` at 30
/// digits the 30-digit ones 1.00, 0.96, 0.92 and 0.88. `factor` itself, both bounds at once on the
/// same numbers (load 5 to 40), took 0.82 [0.72, 0.94] of the time on 100 products of 12 limbs
/// with a prime of 25 digits, and 1.02 [0.94, 1.10] on 200 of 6 limbs and of 8, whose primes are
/// unlucky for these bounds: the traces of those same primes give 0.99 and 0.98, and 0.82 for the
/// primes of 12 limbs (a number's time moves by up to 10 times with the sequence of curves, so a
/// few hundred numbers decide to 8 % at best).
/// For each level alone, one curve's time over its chance was least at about 200, 400 and 800 to
/// 1600 times `B1` at 25, 30 and 35 digits, and at 800 to 3200 at 40 (0.66 to 0.73 of `100 B1`).
/// Factors of 20 digits take as long as before (0.98 to 1.00 on 3000), those of 15 the same time,
/// and the cofactors below 84 digits, which go to the quadratic sieve before the level of 25
/// digits, are untouched. A shorter `B2 = 50 B1` at 12 and 15 digits (more curves to match)
/// saved up to 2 % on the mean on 3000 factors of 15 and of 20 digits and moved random numbers of
/// 45 to 60 digits by 1.5 % at most, so those levels keep `100 B1`.
const LEVELS: [(u64, u64, u64, u64); 14] = [
    (10, 150, 100, 16),
    (12, 400, 100, 20),
    (15, 2_000, 100, 25),
    (20, 11_000, 100, 90),
    (25, 50_000, 200, 251),
    (30, 250_000, 400, 501),
    (35, 1_000_000, 800, 1119),
    (40, 3_000_000, 800, 3181),
    (45, 11_000_000, 800, 6679),
    (50, 43_000_000, 800, 12_298),
    (55, 110_000_000, 800, 31_281),
    (60, 260_000_000, 800, 79_263),
    (65, 850_000_000, 800, 135_241),
    (70, 2_900_000_000, 800, 220_651),
];

/// `B2 / B1` of the fine levels of two limbs. Measured on an Apple M4 Max on 2026-11-11 (load 12
/// to 30), `50 B1` with 1.15 times the curves from the size's level took 0.92 to 0.98 of the time
/// on 7420 to 10000 balanced semiprimes of 20 to 23 digits and 0.97 to 0.98 on random numbers of
/// 25 to 35, but 1.03 on products of a prime of 8 to 11 digits with one making 28 to 38 digits
/// (the curves before the sieve); `200 B1` took 1.08 to 1.14 on such sets.
const B2_RATIO: u64 = 100;

/// The levels of [`LEVELS`] whose stages are kept: those to 35 digits.
const CACHED_LEVELS: usize = 7;

/// The kept stages of the levels.
static LEVEL_SETUPS: [OnceLock<Setup>; CACHED_LEVELS] = [const { OnceLock::new() }; CACHED_LEVELS];

/// The levels of the curves for two limbs: `B1 = 50 * 1.25^k`, `max(8, B1 / 250)` curves, `B2`
/// [`B2_RATIO`] times `B1`, from a level for the size, after four curves on each of the three
/// below. Measured on an Apple M4 Max on balanced semiprimes of 22 to 34 digits, starting from
/// `B1 = 450 * 1.0583^(bits - 64)`, for a factor of half the bits, took least time, though
/// anything from 0.7 to 1.5 times that was within 10 %.
const FINE_LEVELS: usize = 48;

/// The kept stages of the levels of two limbs, up to `B1 = 50 * 1.25^41`, about 480,000.
const CACHED_FINE: usize = 42;

/// The kept stages of the levels of two limbs.
static FINE_SETUPS: [OnceLock<Setup>; CACHED_FINE] = [const { OnceLock::new() }; CACHED_FINE];

/// The kept stages of `p - 1`.
static PM1_SETUP: OnceLock<Setup> = OnceLock::new();

/// The primes below which a number of one limb is divided before the search for factors.
const TRIAL_U64: u64 = 1 << 10;

/// Up to this many bits, a factor of one limb is searched for first by Hart's one line method;
/// measured on 2026-11-11, for semiprimes of two primes of 16 to 20 bits, 0.82 to 4.2 µs against
/// 1.7 to 6.7 µs for FLINT's `n_factor`, and better than the curves up to 40 bits.
const HART_BITS: u64 = 40;

/// The steps of Hart's method before the curves take over.
const HART_ITERATIONS: u64 = 1 << 16;

/// The levels of the curves for one limb: `B1 = 50 * 1.25^k`, [`U64_CURVES`] curves each, and `B2`
/// [`U64_B2_RATIO`] times `B1`. Measured on an Apple M4 Max, semiprimes of two primes of 22 bits
/// took least time from `B1` about 64, of 26 bits about 125, of 32 bits about 300.
const U64_LEVELS: usize = 24;

/// The curves of a level of one limb, and of the three below its first.
const U64_CURVES: u64 = 8;
const U64_FEW: u64 = 2;

/// `B2 / B1` for one limb. Measured on an Apple M4 Max, with two curves on the lower levels, against
/// 50 and four, random words took 6.0 µs against 6.3, products of primes of 20 and 44 bits 8.3
/// against 8.4, of 24 and 24 bits 8.8 against 9.5, of 28 and 28 bits 18.4 against 20.1, of 32 and
/// 32 bits 29.5 against 31.1.
const U64_B2_RATIO: u64 = 35;

/// The kept stages of the levels of one limb.
static U64_SETUPS: [OnceLock<Setup>; U64_LEVELS] = [const { OnceLock::new() }; U64_LEVELS];

/// The setup kept in `cell`, made by `make` the first time.
fn cached(cell: &'static OnceLock<Setup>, make: impl FnOnce() -> Setup) -> &'static Setup {
    cell.get_or_init(make)
}

/// The factorisation of the limbs `x`, not zero: the primes below `2^64` as words and those above as
/// limbs, with exponents, in any order, a prime possibly more than once.
pub(super) fn factor_limbs(x: Vec<u64>, words: &mut Vec<(u64, u32)>) -> Vec<(Vec<u64>, u32)> {
    let mut out: Vec<(Vec<u64>, u32)> = Vec::new();
    if x == [1] {
        return out;
    }
    if x.len() == 1 {
        factor_u64(x[0], words);
        return out;
    }
    let (small, rest) = trial::trial_limbs(x, TRIAL_BOUND);
    words.extend(small);
    let mut stack: Vec<(Vec<u64>, u32)> = vec![(rest, 1)];
    while let Some((m, e)) = stack.pop() {
        if m == [1] {
            continue;
        }
        if m.len() == 1 {
            // No factor below the tables' bound: the cofactor goes on from there.
            let start = words.len();
            factor_odd_u64(m[0], TRIAL_BOUND + 1, words);
            for w in &mut words[start..] {
                w.1 = times(w.1, e);
            }
            continue;
        }
        if is_prime_limbs(&m) {
            out.push((m, e));
            continue;
        }
        if let Some((b, k)) = power::perfect_power_limbs(m.clone(), false) {
            stack.push((b, times(e, k)));
            continue;
        }
        let f = find_factor(&m);
        let q = divide(&m, &f);
        stack.push((f, e));
        stack.push((q, e));
    }
    out
}

/// `a b`, an exponent; past `u32::MAX`, for numbers of more than 512 MiB, it panics.
fn times(a: u32, b: u32) -> u32 {
    a.checked_mul(b).unwrap_or_else(|| exponent_overflow())
}

/// `a / b` for limbs `b` that divide `a`.
fn divide(a: &[u64], b: &[u64]) -> Vec<u64> {
    let order = ByteOrder::LittleEndian;
    limbs_of(&number_of(a, order).div_exact(&number_of(b, order)))
}

/// A factor of the odd composite `m` above `2^64`, without factors below 8192 and not a perfect
/// power, other than 1 and `m`.
fn find_factor(m: &[u64]) -> Vec<u64> {
    with_field(m, Search { m })
}

/// The search for a factor of [`find_factor`], in one arithmetic modulo `m` for all its methods,
/// the curves taken in turn from one sequence.
struct Search<'a> {
    /// The number.
    m: &'a [u64],
}

impl Job for Search<'_> {
    type Out = Vec<u64>;

    fn run<A: Arith>(self, f: &A) -> Vec<u64> {
        let m = self.m;
        let steps = if m.len() == 2 { RHO_STEPS_2 } else { RHO_STEPS };
        if let Rho::Found(g) = (RhoJob { steps }).run(f) {
            return g;
        }
        let bits = 64 * m.len() as u64 - u64::from(m[m.len() - 1].leading_zeros());
        let digits = (bits as f64 * core::f64::consts::LOG10_2) as u64 + 1;
        if digits >= PM1_MIN_DIGITS {
            let setup = cached(&PM1_SETUP, || Setup::new(PM1_B1, PM1_B2, 1));
            let job = Pm1Job {
                stage1: &setup.stage1,
                plan: setup.plan.as_ref(),
            };
            if let Some(g) = job.run(f) {
                return g;
            }
        }
        // The quadratic sieve, once, before the first level for factors above its share.
        let with_siqs = (SIQS_MIN_DIGITS..=SIQS_MAX_DIGITS).contains(&digits);
        let mut siqs_pending = with_siqs;
        let siqs_after = digits as f64 * ECM_BEFORE_SIQS;
        let mut sieve = |level_digits: f64| -> Option<Vec<u64>> {
            if siqs_pending && level_digits > siqs_after {
                siqs_pending = false;
                return siqs(&number_of(m, ByteOrder::LittleEndian)).map(|g| limbs_of(&g));
            }
            None
        };
        let mut curves = Curves::new(f, FIRST_SIGMA);
        let mut level = 0;
        if m.len() == 2 {
            let start = fine_first_level(bits);
            let mut first = start.saturating_sub(3);
            if with_siqs {
                first = first.min(SIQS_FIRST_FINE);
            }
            for k in first..FINE_LEVELS {
                let b1 = fine_b1(k);
                if let Some(g) = sieve(fine_digits(b1)) {
                    return g;
                }
                let count = if k < start { 4 } else { (b1 / 250).max(8) };
                let local;
                let setup = match FINE_SETUPS.get(k) {
                    Some(cell) => cached(cell, || Setup::new(b1, b1 * B2_RATIO, ECM_WEIGHT)),
                    None => {
                        local = Setup::new(b1, b1 * B2_RATIO, ECM_WEIGHT);
                        &local
                    }
                };
                let (stage1, plan) = (&setup.stage1, setup.plan.as_ref());
                if let Some(g) = curves_from(f, &mut curves, count, FIRST_BLOCK, stage1, plan) {
                    return g;
                }
            }
            // On with the levels from 20 digits.
            level = 3;
        }
        loop {
            let i = level.min(LEVELS.len() - 1);
            let (d, b1, ratio, count) = LEVELS[i];
            if let Some(g) = sieve(d as f64) {
                return g;
            }
            let local;
            let setup = match LEVEL_SETUPS.get(i) {
                Some(cell) => cached(cell, || Setup::new(b1, b1 * ratio, ECM_WEIGHT)),
                None => {
                    local = Setup::new(b1, b1 * ratio, ECM_WEIGHT);
                    &local
                }
            };
            let (stage1, plan) = (&setup.stage1, setup.plan.as_ref());
            if let Some(g) = curves_from(f, &mut curves, count, FIRST_BLOCK, stage1, plan) {
                return g;
            }
            level += 1;
        }
    }
}

/// `B1` of the fine level `k`.
fn fine_b1(k: usize) -> u64 {
    (50.0 * 1.25f64.powi(k as i32)).round() as u64
}

/// The first fine level for a cofactor of two limbs of `bits` bits: `B1` about
/// `450 * 1.0583^(bits - 64)`, for a factor of half the bits.
fn fine_first_level(bits: u64) -> usize {
    let b1 = 450.0 * 1.0583f64.powf(bits as f64 - 64.0);
    (((b1 / 50.0).ln() / 1.25f64.ln()).round().max(0.0) as usize).min(FINE_LEVELS - 1)
}

/// The digits of the factors a fine level of `B1` is meant for, by the same rule:
/// `(32 + ln(B1 / 450) / (2 ln 1.0583)) log10 2`.
fn fine_digits(b1: u64) -> f64 {
    (32.0 + (b1 as f64 / 450.0).ln() / (2.0 * 1.0583f64.ln())) * core::f64::consts::LOG10_2
}

/// Pushes the factorisation of `n >= 1` into `out`: trial division, then for each cofactor the
/// exact test, squares, and a factor in registers.
pub(super) fn factor_u64(n: u64, out: &mut Vec<(u64, u32)>) {
    let mut n = n;
    if n & 1 == 0 {
        let z = n.trailing_zeros();
        out.push((2, z));
        n >>= z;
    }
    let rest = trial::trial_u64(n, 0, TRIAL_U64, out);
    factor_odd_u64(rest, TRIAL_U64, out);
}

/// Pushes the factorisation of the odd `n >= 1` without prime factors below `below` into `out`:
/// for each cofactor the exact test, squares, and a factor in registers.
fn factor_odd_u64(n: u64, below: u64, out: &mut Vec<(u64, u32)>) {
    let square = below.saturating_mul(below);
    let mut stack = [(0u64, 0u32); 64];
    stack[0] = (n, 1);
    let mut depth = 1;
    while depth > 0 {
        depth -= 1;
        let (m, e) = stack[depth];
        if m == 1 {
            continue;
        }
        if m < square || super::super::prime::is_prime_u64(m) {
            out.push((m, e));
            continue;
        }
        let r = isqrt(m);
        if r * r == m {
            stack[depth] = (r, 2 * e);
            depth += 1;
            continue;
        }
        let g = find_factor_u64(m);
        // The numbers on the stack are odd, above 1 and their product divides `n`: at most 40.
        stack[depth] = (g, e);
        stack[depth + 1] = (m / g, e);
        depth += 2;
    }
}

/// A factor of the odd composite `m`, not a square, other than 1 and `m`.
pub(in crate::math::numbers) fn find_factor_u64(m: u64) -> u64 {
    let bits = 64 - u64::from(m.leading_zeros());
    if bits <= HART_BITS
        && let Some(g) = hart(m, HART_ITERATIONS)
    {
        return g;
    }
    // The curves of all the levels in one arithmetic, from one sequence.
    let f = Reg::new(Mont1::new(m));
    let start = u64_first_level(bits);
    let mut curves = Curves::new(&f, FIRST_SIGMA);
    for (k, cell) in U64_SETUPS.iter().enumerate().skip(start.saturating_sub(3)) {
        let b1 = fine_b1(k);
        let setup = cached(cell, || Setup::new(b1, b1 * U64_B2_RATIO, ECM_WEIGHT));
        let c = if k < start { U64_FEW } else { U64_CURVES };
        let (stage1, plan) = (&setup.stage1, setup.plan.as_ref());
        if let Some(g) = curves_from(&f, &mut curves, c, FIRST_BLOCK, stage1, plan) {
            return g[0];
        }
    }
    match with_field(&[m], RhoJob { steps: u64::MAX }) {
        Rho::Found(g) => g[0],
        Rho::Nothing => unreachable!("rho runs until it finds a factor"),
    }
}

/// The first level for a cofactor of one limb of `bits` bits: `B1` about `64 * 1.08^(bits - 44)`.
fn u64_first_level(bits: u64) -> usize {
    let k = (bits.saturating_sub(44) as f64 * 1.08f64.ln() / 1.25f64.ln()).round() as usize + 1;
    k.min(U64_LEVELS - 1)
}

/// Hart's "one line" factoring (2012), a variant of Lehman's: for `i = 1, 2, ...`, with
/// `s = ceil(sqrt(480 n i))`, a square `s^2 mod 480 n = t^2` gives the factor `gcd(s - t, n)`;
/// about `n^(1/3)` steps, each a square root in floating point and a test for a square.
fn hart(n: u64, iterations: u64) -> Option<u64> {
    let m = n.checked_mul(480)?;
    let mut x = m;
    for _ in 0..iterations.min(u64::MAX / m - 1) {
        let mut s = (x as f64).sqrt() as u64;
        while u128::from(s) * u128::from(s) < u128::from(x) {
            s += 1;
        }
        while s > 0 && u128::from(s - 1) * u128::from(s - 1) >= u128::from(x) {
            s -= 1;
        }
        let r = s.wrapping_mul(s).wrapping_sub(x);
        if let Some(t) = square_root(r) {
            let g = gcd_u64(n, s - t);
            if g != 1 && g != n {
                return Some(g);
            }
        }
        x += m;
    }
    None
}

/// The residues modulo 64 that are not squares, bit `r` set for each.
const NOT_SQUARE_64: u64 = {
    let mut m = u64::MAX;
    let mut x = 0;
    while x < 64 {
        m &= !(1 << (x * x % 64));
        x += 1;
    }
    m
};

/// The square root of `r` when it is a square: squares modulo 64 first.
#[inline]
fn square_root(r: u64) -> Option<u64> {
    if (NOT_SQUARE_64 >> (r & 63)) & 1 == 1 {
        return None;
    }
    let t = isqrt(r);
    if t * t == r { Some(t) } else { None }
}
