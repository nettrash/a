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
//!   quadratic sieve is to come), beyond by GMP-ECM's levels for factors of 10, 15, 20, ... digits;
//!   the quadratic sieve for cofactors in its range, once the curves have made factors of
//!   [`ECM_BEFORE_SIQS`] of the digits unlikely. The curves go on with growing bounds until a
//!   factor is found, so every factorisation ends.
//!
//! The stages of the curves' bounds, the exponent of the first and the plan of the second, are
//! kept for the life of the program for the bounds that recur (up to `B1 = 10^6`, about 3 MB at
//! most, made on first use), so that a factorisation of a small number costs its curves alone.

use super::ecm::FIRST_SIGMA;
use super::field::{gcd_u64, limbs_of, number_of, with_field};
use super::pm1::Pm1Job;
use super::primes::isqrt;
use super::rho::{Rho, RhoJob};
use super::{
    ECM_WEIGHT, Setup, ecm_limbs, ecm_with, exponent_overflow, is_prime_limbs, power, siqs, trial,
};
use crate::math::numbers::ByteOrder;
use std::sync::OnceLock;

/// The cofactors from this many digits up to [`SIQS_MAX_DIGITS`] go to the quadratic sieve, once
/// the curves have made factors of [`ECM_BEFORE_SIQS`] of their digits unlikely. Measured on an
/// Apple M4 Max on 128 balanced semiprimes of each size, the curves alone took 0.65 ms at 26
/// digits, 0.85 at 27, 1.11 at 28, 1.30 at 29 and 2.44 at 30, against 1.14, 1.12, 1.09, 1.10 and
/// 1.19 ms for the sieve.
pub(super) const SIQS_MIN_DIGITS: u64 = 28;

/// The largest cofactors, in digits, that go to the quadratic sieve.
pub(super) const SIQS_MAX_DIGITS: u64 = 110;

/// The share of a cofactor's digits whose factors the curves look for before the quadratic sieve:
/// yafu's 4/13 or so. Measured on an Apple M4 Max, against the shares 0.25 and 0.35, it took
/// 6.0 ms against 6.2 and 5.3 on random numbers of 50 digits, 66 ms against 74 and 63 at 60 digits
/// and 247 ms against 415 and 224 at 70; on balanced semiprimes, where the curves are wasted, it
/// cost 4 to 32 % more than the sieve alone, against 0 to 8 % for 0.25 and up to 58 % for 0.35.
pub(super) const ECM_BEFORE_SIQS: f64 = 0.30;

/// While the quadratic sieve is to come, the curves of two limbs start from this fine level,
/// `B1 = 191`, for factors of about 7 digits, just above those the first run of rho finds, rather
/// than from the level for the size. Measured on an Apple M4 Max, random numbers of 36 digits took
/// 0.25 ms against 0.85, products of primes of 10 and 26 digits 0.21 ms against 3.2, and balanced
/// semiprimes of 28 to 38 digits 10 to 26 % more.
const SIQS_FIRST_FINE: usize = 6;

/// The primes up to which the factorisation divides first: the tables'.
const TRIAL_BOUND: u64 = super::trial::TABLE_MAX;

/// The steps of rho tried first on a cofactor above `2^64`, for factors up to about `2^24`.
const RHO_STEPS: u64 = 1 << 12;

/// The cofactors from this many digits up take `p - 1` before the curves, with these bounds: about
/// the cost of one curve with `B1 = 15,000`.
const PM1_MIN_DIGITS: u64 = 45;
const PM1_B1: u64 = 100_000;
const PM1_B2: u64 = 10_000_000;

/// The levels of the elliptic curve method above two limbs: the digits of the factors each is
/// meant for, `B1`, and the curves, after GMP-ECM's table (its README) from 15 digits up, with two
/// smaller levels below; `B2` is [`B2_RATIO`] times `B1`.
const LEVELS: [(u64, u64, u64); 14] = [
    (10, 150, 16),
    (12, 400, 20),
    (15, 2_000, 25),
    (20, 11_000, 90),
    (25, 50_000, 300),
    (30, 250_000, 700),
    (35, 1_000_000, 1800),
    (40, 3_000_000, 5100),
    (45, 11_000_000, 10_600),
    (50, 43_000_000, 19_300),
    (55, 110_000_000, 49_000),
    (60, 260_000_000, 124_000),
    (65, 850_000_000, 210_000),
    (70, 2_900_000_000, 340_000),
];

/// `B2 / B1` of the levels of two limbs and more.
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
/// measured, for semiprimes of two primes of 16 to 20 bits, 1.1 to 4.5 µs against 1.8 to 7.0 µs
/// for FLINT's `n_factor`, and better than the curves up to 40 bits.
const HART_BITS: u64 = 40;

/// The steps of Hart's method before the curves take over.
const HART_ITERATIONS: u64 = 1 << 16;

/// The levels of the curves for one limb: `B1 = 50 * 1.25^k`, [`U64_CURVES`] curves each, and `B2`
/// [`U64_B2_RATIO`] times `B1`. Measured on an Apple M4 Max, semiprimes of two primes of 22 bits
/// took least time from `B1` about 64, of 26 bits about 125, of 32 bits about 300.
const U64_LEVELS: usize = 24;

/// The curves of a level of one limb, and of the three below its first.
const U64_CURVES: u64 = 8;
const U64_FEW: u64 = 4;

/// `B2 / B1` for one limb.
const U64_B2_RATIO: u64 = 50;

/// The kept stages of the levels of one limb.
static U64_SETUPS: [OnceLock<Setup>; U64_LEVELS] = [const { OnceLock::new() }; U64_LEVELS];

/// The setup kept in `cell`, made by `make` the first time.
fn cached(cell: &'static OnceLock<Setup>, make: impl FnOnce() -> Setup) -> &'static Setup {
    cell.get_or_init(make)
}

/// The factorisation of the limbs `x`, not zero: primes as limbs, with exponents, in any order.
pub(super) fn factor_limbs(x: Vec<u64>) -> Vec<(Vec<u64>, u32)> {
    let mut out: Vec<(Vec<u64>, u32)> = Vec::new();
    if x == [1] {
        return out;
    }
    if x.len() == 1 {
        let mut small = Vec::new();
        factor_u64(x[0], &mut small);
        return small.into_iter().map(|(p, e)| (vec![p], e)).collect();
    }
    let (small, rest) = trial::trial_limbs(x, TRIAL_BOUND);
    out.extend(small.into_iter().map(|(p, e)| (vec![p], e)));
    let mut stack: Vec<(Vec<u64>, u32)> = vec![(rest, 1)];
    while let Some((m, e)) = stack.pop() {
        if m == [1] {
            continue;
        }
        if m.len() == 1 {
            let mut small = Vec::new();
            factor_u64(m[0], &mut small);
            for (p, k) in small {
                out.push((vec![p], times(k, e)));
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
    if let Rho::Found(g) = with_field(m, RhoJob { steps: RHO_STEPS }) {
        return g;
    }
    let bits = 64 * m.len() as u64 - u64::from(m[m.len() - 1].leading_zeros());
    let digits = (bits as f64 * core::f64::consts::LOG10_2) as u64 + 1;
    if digits >= PM1_MIN_DIGITS {
        let setup = cached(&PM1_SETUP, || Setup::new(PM1_B1, PM1_B2, 1));
        if let Some(g) = with_field(
            m,
            Pm1Job {
                stage1: &setup.stage1,
                plan: setup.plan.as_ref(),
            },
        ) {
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
    let mut sigma = FIRST_SIGMA;
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
            let curves = if k < start { 4 } else { (b1 / 250).max(8) };
            let found = match FINE_SETUPS.get(k) {
                Some(cell) => {
                    let setup = cached(cell, || Setup::new(b1, b1 * B2_RATIO, ECM_WEIGHT));
                    ecm_with(m, setup, sigma, curves)
                }
                None => ecm_limbs(m, b1, b1 * B2_RATIO, sigma, curves),
            };
            if let Some(g) = found {
                return g;
            }
            sigma += curves;
        }
        // On with the levels from 20 digits.
        level = 3;
    }
    loop {
        let i = level.min(LEVELS.len() - 1);
        let (d, b1, curves) = LEVELS[i];
        if let Some(g) = sieve(d as f64) {
            return g;
        }
        let found = match LEVEL_SETUPS.get(i) {
            Some(cell) => {
                let setup = cached(cell, || Setup::new(b1, b1 * B2_RATIO, ECM_WEIGHT));
                ecm_with(m, setup, sigma, curves)
            }
            None => ecm_limbs(m, b1, b1 * B2_RATIO, sigma, curves),
        };
        if let Some(g) = found {
            return g;
        }
        sigma += curves;
        level += 1;
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
    let mut stack = vec![(rest, 1u32)];
    while let Some((m, e)) = stack.pop() {
        if m == 1 {
            continue;
        }
        if m < TRIAL_U64 * TRIAL_U64 || super::super::prime::is_prime_u64(m) {
            out.push((m, e));
            continue;
        }
        let r = isqrt(m);
        if r * r == m {
            stack.push((r, 2 * e));
            continue;
        }
        let g = find_factor_u64(m);
        stack.push((g, e));
        stack.push((m / g, e));
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
    let mut sigma = FIRST_SIGMA;
    let start = u64_first_level(bits);
    for (k, cell) in U64_SETUPS.iter().enumerate().skip(start.saturating_sub(3)) {
        let b1 = fine_b1(k);
        let setup = cached(cell, || Setup::new(b1, b1 * U64_B2_RATIO, ECM_WEIGHT));
        let c = if k < start { U64_FEW } else { U64_CURVES };
        if let Some(g) = ecm_with(&[m], setup, sigma, c) {
            return g[0];
        }
        sigma += c;
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
