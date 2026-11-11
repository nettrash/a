//! Pollard's `p - 1` method (1974) and Williams's `p + 1` method (1982).
//!
//! - `p - 1`: for a prime `p` of `n` with `p - 1` dividing the exponent `E` of the first stage,
//!   `3^E = 1` modulo `p`, so `gcd(3^E - 1, n)` is a multiple of `p`. `3^E` is computed from the
//!   top bit of `E` with a square for each bit and, for each one, a multiplication by 3, which is
//!   two additions; a later chunk of `E` takes a power by a sliding window. The second stage
//!   catches `p - 1 = s q` with `s | E` and one prime `q` up to `B2`, by the Lucas sequence of
//!   `V_1 = x + 1/x` for `x = 3^E`, whose `V_m - 2` is `x^(-m) (x^m - 1)^2`, or, for the longer
//!   ranges, by Montgomery and Kruppa's continuation on `x` ([`mk`](super::mk)), which hands the
//!   stage back to the Lucas sequence when it cannot tell the primes it caught apart.
//! - `p + 1`: the Lucas sequence `V_m(P_0)`, `V_0 = 2`, `V_1 = P_0`, is `a^m + a^(-m)` for a root
//!   `a` of `X^2 - P_0 X + 1`, which lies in the field of `p^2` elements and has an order dividing
//!   `p + 1` when `P_0^2 - 4` is not a square modulo `p` (and `p - 1` when it is): `V_E(P_0) - 2` is
//!   then 0 modulo `p` when that order divides `E`. `V_E` is computed by the ladder, a square and a
//!   product a bit, through the chunks of `E` as `V_(ab)(P) = V_b(V_a(P))`, or, from
//!   [`PRAC_LIMBS`] limbs and in memory, by Montgomery's PRAC chain for each prime; the second
//!   stage is the same as for `p - 1`, Montgomery and Kruppa's continuation in `Z_n[sqrt(Delta)]`
//!   ([`pp1_stage2`]) or the baby steps and giant steps. The starting values `2/7`, `6/5` and `6`,
//!   whose `P_0^2 - 4` are `-3`, `-1` and `2` times squares, are tried in turn, so that for a prime
//!   with `p + 1` smooth one of them takes `p + 1` with a chance of 7 in 8.
//!
//! When the greatest common divisor of a stage is `n`, every prime was caught at once: the chunk is
//! gone over again prime by prime from its start, with a greatest common divisor each.

use super::field::{Arith, Job, invert};
use super::mk::{pm1_stage2, pp1_stage2};
use super::primes::primes_in;
use super::rho::{Gcd, gcd};
use super::stage1::{Chunk, Stage1, prime_power};
use super::stage2::{Plan, lucas_ladder, lucas_prac, lucas_prac_with, lucas_stage2, prac_choice};
use std::sync::{Arc, Mutex};

/// From this many limbs, `p + 1`'s first stage takes PRAC's chains in registers too: measured on an
/// Apple M4 Max at `B1 = 10^5`, three starting values, with the multipliers kept, 6 limbs 27 ms
/// against 30 by the ladder and 8 limbs 44 against 53, while 4 limbs took 16 against 15.
const PRAC_LIMBS: usize = 6;

/// The bounds up to which the multipliers are kept: a byte a prime, 665 KB at `10^7`.
const PRAC_KEPT_B1: u64 = 10_000_000;

/// The multipliers of PRAC's chains of the last bound asked for, kept for the next calls: its
/// three starting values share them, and so do later calls with the bound.
static PRAC_CHOICES: Mutex<Option<(u64, Arc<[u8]>)>> = Mutex::new(None);

/// The index in [`PRAC_V`](super::stage2) of the multiplier of the cheapest chain for each prime up
/// to `b1`, in increasing order: computed once for the bound, about 0.4 us a prime, and kept, or
/// `None` past [`PRAC_KEPT_B1`].
fn prac_choices(b1: u64) -> Option<Arc<[u8]>> {
    if b1 > PRAC_KEPT_B1 {
        return None;
    }
    let mut kept = PRAC_CHOICES.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((b, c)) = kept.as_ref()
        && *b == b1
    {
        return Some(c.clone());
    }
    let c: Arc<[u8]> = primes_in(2, b1)
        .map(|p| if p < 3 { 0 } else { prac_choice(p) })
        .collect();
    *kept = Some((b1, c.clone()));
    Some(c)
}

/// The starting values of `p + 1`, as fractions.
pub(super) const PP1_SEEDS: [(u64, u64); 3] = [(2, 7), (6, 5), (6, 1)];

/// `p - 1` with the first stage's exponent and the second stage's plan.
pub(super) struct Pm1Job<'a> {
    /// The first stage.
    pub(super) stage1: &'a Stage1,
    /// The second stage, if any.
    pub(super) plan: Option<&'a Plan>,
}

impl Job for Pm1Job<'_> {
    type Out = Option<Vec<u64>>;

    fn run<A: Arith>(self, f: &A) -> Option<Vec<u64>> {
        let b1 = self.stage1.b1;
        let one = f.one();
        let mut x = f.small(3);
        let mut first = true;
        let mut result: Option<Option<Vec<u64>>> = None;
        let mut d = f.zero();
        self.stage1.each(|chunk| {
            let save = x.clone();
            x = if first {
                power_of_3(f, &chunk.e)
            } else {
                power(f, &x, &chunk.e)
            };
            first = false;
            f.sub(&mut d, &x, &one);
            match gcd(f, &d) {
                Gcd::One => true,
                Gcd::Factor(g) => {
                    result = Some(Some(g));
                    false
                }
                Gcd::Whole => {
                    result = Some(pm1_backtrack(f, save, chunk, b1));
                    false
                }
            }
        });
        if let Some(r) = result {
            return r;
        }
        let plan = self.plan?;
        // Montgomery and Kruppa's continuation when it costs less, unless it leaves the stage to
        // this one.
        if let Some(mk) = plan.mk(f.n().len(), false)
            && let Some(found) = pm1_stage2(f, &x, &mk)
        {
            return found;
        }
        // `V_1 = x + 1/x`.
        let xi = match invert(f, &x) {
            Ok(v) => v,
            Err(g) => return proper(f, g),
        };
        let mut v1 = f.zero();
        f.add(&mut v1, &x, &xi);
        lucas_stage2(f, &v1, plan)
    }
}

/// `p + 1` with the starting value `seed`, the first stage's exponent and the second stage's plan.
pub(super) struct Pp1Job<'a> {
    /// The starting value, a fraction.
    pub(super) seed: (u64, u64),
    /// The first stage.
    pub(super) stage1: &'a Stage1,
    /// The second stage, if any.
    pub(super) plan: Option<&'a Plan>,
}

impl Job for Pp1Job<'_> {
    type Out = Option<Vec<u64>>;

    fn run<A: Arith>(self, f: &A) -> Option<Vec<u64>> {
        let b1 = self.stage1.b1;
        let two = f.small(2);
        // `P_0 = a / b`.
        let (a, b) = self.seed;
        let mut v = f.small(a);
        if b != 1 {
            match invert(f, &f.small(b)) {
                Ok(bi) => f.mul_in(&mut v, &bi),
                Err(g) => return proper(f, g),
            }
        }
        let mut result: Option<Option<Vec<u64>>> = None;
        let mut d = f.zero();
        let mut t = [f.zero(), f.zero(), f.zero(), f.zero()];
        // In memory, from nine limbs, stage 1 takes Montgomery's PRAC chain for each prime, about
        // `1.52 log2 p` products against `2 log2 p` for the ladder on the exponent. Measured on an
        // Apple M4 Max at `B1 = 10^6`, three starting values, with the golden ratio alone: 2 limbs
        // 59 ms for the ladder against 71 ms for PRAC, 4 limbs 155 against 171, 8 limbs about the
        // same, 16 limbs 1.35 s against 1.20 s. In registers below [`PRAC_LIMBS`] limbs the
        // ladder's independent product and square a bit hide each other's latency; from there,
        // where the ladder's two products a bit cost twice one, and in memory, the chains take the
        // multipliers kept for the bound (see [`prac_choices`]), up to [`PRAC_KEPT_B1`]; beyond it
        // the ladder in registers and the golden ratio's chains in memory.
        let choices = if A::MEMORY || f.n().len() >= PRAC_LIMBS {
            prac_choices(b1)
        } else {
            None
        };
        let mut next = 0usize;
        self.stage1.each(|chunk| {
            let save = v.clone();
            if let Some(choices) = &choices {
                // A chain for each prime, as many times as its power, with the multiplier kept.
                for p in primes_in(chunk.lo, chunk.hi) {
                    let choice = choices[next];
                    next += 1;
                    let mut q = prime_power(p, b1);
                    while q > 1 {
                        if p == 2 {
                            f.sqr_in(&mut v);
                            f.sub_in(&mut v, &two);
                        } else {
                            lucas_prac_with(f, &mut v, p, &two, &mut t, choice);
                        }
                        q /= p;
                    }
                }
            } else if A::MEMORY {
                // A chain for each prime, as many times as its power.
                for p in primes_in(chunk.lo, chunk.hi) {
                    let mut q = prime_power(p, b1);
                    while q > 1 {
                        if p == 2 {
                            f.sqr_in(&mut v);
                            f.sub_in(&mut v, &two);
                        } else {
                            lucas_prac(f, &mut v, p, &two, &mut t);
                        }
                        q /= p;
                    }
                }
            } else {
                v = lucas_ladder(f, &v, &two, &chunk.e).0;
            }
            f.sub(&mut d, &v, &two);
            match gcd(f, &d) {
                Gcd::One => true,
                Gcd::Factor(g) => {
                    result = Some(Some(g));
                    false
                }
                Gcd::Whole => {
                    result = Some(pp1_backtrack(f, save, chunk, b1));
                    false
                }
            }
        });
        if let Some(r) = result {
            return r;
        }
        let plan = self.plan?;
        // Montgomery and Kruppa's continuation when it costs less, unless it leaves the stage to
        // this one.
        if let Some(mk) = plan.mk(f.n().len(), true)
            && let Some(found) = pp1_stage2(f, &v, &mk)
        {
            return found;
        }
        lucas_stage2(f, &v, plan)
    }
}

/// The factor `g` of a failed inversion when it is not `n`.
pub(super) fn proper<A: Arith>(f: &A, g: Vec<u64>) -> Option<Vec<u64>> {
    if g == f.n() { None } else { Some(g) }
}

/// `p - 1` over the chunk again from `x`, a prime at a time.
fn pm1_backtrack<A: Arith>(f: &A, mut x: A::V, chunk: &Chunk, b1: u64) -> Option<Vec<u64>> {
    let one = f.one();
    let mut d = f.zero();
    for p in primes_in(chunk.lo, chunk.hi) {
        let mut q = prime_power(p, b1);
        while q > 1 {
            x = power(f, &x, &[p]);
            q /= p;
            f.sub(&mut d, &x, &one);
            match gcd(f, &d) {
                Gcd::One => {}
                Gcd::Factor(g) => return Some(g),
                Gcd::Whole => return None,
            }
        }
    }
    None
}

/// `p + 1` over the chunk again from `v`, a prime at a time.
fn pp1_backtrack<A: Arith>(f: &A, mut v: A::V, chunk: &Chunk, b1: u64) -> Option<Vec<u64>> {
    let two = f.small(2);
    let mut d = f.zero();
    for p in primes_in(chunk.lo, chunk.hi) {
        let mut q = prime_power(p, b1);
        while q > 1 {
            v = lucas_ladder(f, &v, &two, &[p]).0;
            q /= p;
            f.sub(&mut d, &v, &two);
            match gcd(f, &d) {
                Gcd::One => {}
                Gcd::Factor(g) => return Some(g),
                Gcd::Whole => return None,
            }
        }
    }
    None
}

/// The number of bits of the limbs `e`, the top one not zero.
#[inline]
pub(super) fn bit_length(e: &[u64]) -> u64 {
    64 * e.len() as u64 - u64::from(e.last().map_or(64, |l| l.leading_zeros()))
}

/// Bit `i` of the limbs `e`.
#[inline(always)]
pub(super) fn bit(e: &[u64], i: u64) -> bool {
    (e[(i / 64) as usize] >> (i % 64)) & 1 == 1
}

/// `3^e` for the exponent of the limbs `e`, at least 1: a square a bit, and two additions for each
/// one.
fn power_of_3<A: Arith>(f: &A, e: &[u64]) -> A::V {
    let three = f.small(3);
    let mut x = three.clone();
    let mut t = f.zero();
    let bits = bit_length(e);
    for i in (0..bits - 1).rev() {
        f.sqr_in(&mut x);
        if bit(e, i) {
            f.add(&mut t, &x, &x);
            f.add_in(&mut x, &t);
        }
    }
    x
}

/// `b^e` for the exponent of the limbs `e`, at least 1, by a sliding window: the odd powers of `b`
/// up to `2^k`, a square for each bit, and a product for each window of up to `k` bits.
pub(super) fn power<A: Arith>(f: &A, b: &A::V, e: &[u64]) -> A::V {
    let bits = bit_length(e);
    let k: u64 = match bits {
        0..=24 => 1,
        25..=80 => 3,
        81..=240 => 4,
        241..=672 => 5,
        673..=1792 => 6,
        _ => 7,
    };
    // `table[i] = b^(2i + 1)`.
    let mut table: Vec<A::V> = vec![b.clone()];
    if k > 1 {
        let mut b2 = f.zero();
        f.sqr(&mut b2, b);
        for i in 1..1usize << (k - 1) {
            let mut t = f.zero();
            f.mul(&mut t, &table[i - 1], &b2);
            table.push(t);
        }
    }
    let mut x = f.one();
    let mut started = false;
    let mut i = bits;
    while i > 0 {
        if !bit(e, i - 1) {
            if started {
                f.sqr_in(&mut x);
            }
            i -= 1;
            continue;
        }
        // The window: the bits from `i - 1` down, up to `k` of them, ending with a one.
        let low = i.saturating_sub(k);
        let mut l = low;
        while !bit(e, l) {
            l += 1;
        }
        let mut w = 0u64;
        for j in (l..i).rev() {
            w = (w << 1) | u64::from(bit(e, j));
        }
        if started {
            for _ in l..i {
                f.sqr_in(&mut x);
            }
            f.mul_in(&mut x, &table[(w >> 1) as usize]);
        } else {
            x.clone_from(&table[(w >> 1) as usize]);
            started = true;
        }
        i = l;
    }
    x
}
