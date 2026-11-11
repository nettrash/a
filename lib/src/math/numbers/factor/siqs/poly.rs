//! The polynomials: `A` as a product of primes of the factor base, the `2^(s−1)` values of `B` with
//! `B² ≡ kN (mod A)`, taken in the order of a Gray code, and the roots of each polynomial modulo the
//! primes of the factor base (Alford and Pomerance 1993; Contini 1997).
//!
//! For `A = q_1 ⋯ q_s` and `t_l` a square root of `kN` modulo `q_l`, `B_l = (A/q_l)·γ_l` with
//! `γ_l = t_l·(A/q_l)^(−1) mod q_l` (taken at most `q_l/2`) is a square root of `kN` modulo `q_l` and
//! 0 modulo the other primes of `A`, so every `B = ±B_1 ± ⋯ ± B_s` has `B² ≡ kN (mod A)`; the sign of
//! `B_s` is kept, as `−B` gives the same values. Then `(Ax + B)² − kN = A·g(x)` with
//! `g(x) = Ax² + 2Bx + C` and `C = (B² − kN)/A`. Polynomial `v` takes `B_l` with the sign `−` when bit
//! `l` of `v ^ (v >> 1)` is set, so that from one to the next one `B_ν` changes sign, `ν` the
//! trailing zeros of `v + 1`, and each root modulo `p` moves by `±2·B_ν·A^(−1) mod p`, a number
//! computed once for each `A`.

use super::arith::{inverse, mul_mod};
use super::fb::FactorBase;
use std::collections::HashSet;

/// A root that is never reached: the roots of the primes of `A`, which do not sieve.
pub(super) const NO_ROOT: u32 = 1 << 30;

/// A generator of random numbers, SplitMix64 (Steele, Lea and Flood 2014), with a fixed seed so that
/// the sieve is the same from run to run.
pub(super) struct Random(u64);

impl Random {
    /// The generator with the seed `seed`.
    pub(super) fn new(seed: u64) -> Random {
        Random(seed)
    }

    /// The next number.
    pub(super) fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A number below `n`, for `n > 0`.
    pub(super) fn below(&mut self, n: usize) -> usize {
        ((u128::from(self.next()) * n as u128) >> 64) as usize
    }
}

/// The choice of the values of `A`: products of `s` primes of the factor base close to the target,
/// each product once.
pub(super) struct Chooser {
    /// The number of primes of `A`.
    pub(super) s: usize,
    /// The base-2 logarithm of the target.
    target: f64,
    /// The indices from which the first `s − 1` primes are drawn.
    lo: usize,
    /// The end of those indices.
    hi: usize,
    /// The primes of `A` lie in `[first, end)`.
    first: usize,
    /// The end of the indices of the primes of `A`.
    end: usize,
    /// How far from the target, in bits, an `A` may be.
    tolerance: f64,
    /// The sets of primes taken.
    used: HashSet<Vec<u32>>,
    /// The random numbers.
    random: Random,
    /// The base-2 logarithms of the primes of the factor base.
    log: Vec<f64>,
}

impl Chooser {
    /// The choice for a target of `target` bits, with primes of about `bits` bits among the primes
    /// of indices `first..end` of the factor base.
    pub(super) fn new(
        fb: &FactorBase,
        target: f64,
        bits: u32,
        first: usize,
        end: usize,
        seed: u64,
    ) -> Chooser {
        let log: Vec<f64> = fb.prime.iter().map(|&p| f64::from(p).log2()).collect();
        let end = end.max(first + 1).min(fb.prime.len());
        // The number of primes: the nearest to `target/bits` whose mean size is within the primes
        // allowed, half a bit from each end, so that the last prime can make up for the others.
        let (bottom, top) = (log[first], log[end - 1]);
        let ideal = target / f64::from(bits);
        let fits = |s: usize| {
            let mean = target / s as f64;
            mean >= bottom + 0.5 && mean <= top - 0.5
        };
        let s = (2..=20usize)
            .filter(|&s| fits(s))
            .min_by(|&a, &b| {
                (a as f64 - ideal)
                    .abs()
                    .partial_cmp(&(b as f64 - ideal).abs())
                    .expect("finite")
            })
            .unwrap_or_else(|| ((target / (0.5 * (bottom + top))).round() as usize).clamp(2, 20));
        let mut chooser = Chooser {
            s,
            target,
            lo: first,
            hi: end,
            first,
            end,
            tolerance: 0.1,
            used: HashSet::new(),
            random: Random::new(seed),
            log,
        };
        chooser.window(1.0);
        chooser
    }

    /// The indices from which the first primes are drawn: those within a factor `2^spread` of the
    /// mean size, at least `3s` primes when there are as many.
    fn window(&mut self, spread: f64) {
        let mean = self.target / (self.s as f64);
        let (first, end) = (self.first, self.end);
        let lo = self.log[first..end].partition_point(|&l| l < mean - spread) + first;
        let hi = self.log[first..end].partition_point(|&l| l <= mean + spread * 0.5) + first;
        let (mut lo, mut hi) = (lo.min(end - 1), hi.max(lo + 1).min(end));
        while hi - lo < 3 * self.s && (lo > first || hi < end) {
            lo = lo.saturating_sub(1).max(first);
            hi = (hi + 1).min(end);
        }
        self.lo = lo;
        self.hi = hi;
    }

    /// The indices of the primes of the next `A`, increasing, or `None` when no new one is found.
    pub(super) fn next(&mut self) -> Option<Vec<u32>> {
        let mut misses = 0u32;
        let mut spread = 1.0;
        loop {
            if let Some(a) = self.attempt() {
                return Some(a);
            }
            misses += 1;
            if misses.is_multiple_of(2000) {
                // Few products are near the target from these primes: draw from more of them and
                // accept products further from it.
                spread *= 1.5;
                self.tolerance *= 1.5;
                self.window(spread);
                if misses >= 40_000 {
                    return None;
                }
            }
        }
    }

    /// One attempt at an `A`: `s − 1` random primes from the window, and the prime that brings the
    /// product closest to the target.
    fn attempt(&mut self) -> Option<Vec<u32>> {
        let s = self.s;
        let mut chosen: Vec<u32> = Vec::with_capacity(s);
        let mut log = 0.0;
        let width = self.hi - self.lo;
        if width < s - 1 {
            return None;
        }
        while chosen.len() < s - 1 {
            let j = (self.lo + self.random.below(width)) as u32;
            if !chosen.contains(&j) {
                chosen.push(j);
                log += self.log[j as usize];
            }
        }
        let rest = self.target - log;
        let (first, end) = (self.first, self.end);
        let at = self.log[first..end].partition_point(|&l| l < rest) + first;
        let mut best = None;
        let mut best_d = f64::INFINITY;
        for j in at.saturating_sub(2).max(first)..(at + 2).min(end) {
            if chosen.contains(&(j as u32)) {
                continue;
            }
            let d = (self.log[j] - rest).abs();
            if d < best_d {
                best_d = d;
                best = Some(j as u32);
            }
        }
        let j = best?;
        if best_d > self.tolerance {
            return None;
        }
        chosen.push(j);
        chosen.sort_unstable();
        if self.used.insert(chosen.clone()) {
            Some(chosen)
        } else {
            None
        }
    }
}

/// `x · m` for the limbs `x`, least significant first, and a word `m`.
pub(super) fn mul_1(x: &[u64], m: u64) -> Vec<u64> {
    let mut r = Vec::with_capacity(x.len() + 1);
    let mut c = 0u64;
    for &l in x {
        let t = u128::from(l) * u128::from(m) + u128::from(c);
        r.push(t as u64);
        c = (t >> 64) as u64;
    }
    if c != 0 {
        r.push(c);
    }
    r
}

/// `A`, the values of `B_l` and the `γ_l` for the primes of `A` (indices `a_idx`), as limbs.
pub(super) fn b_values(fb: &FactorBase, a_idx: &[u32]) -> (Vec<u64>, Vec<Vec<u64>>, Vec<u32>) {
    let mut a = vec![1u64];
    for &j in a_idx {
        a = mul_1(&a, u64::from(fb.prime[j as usize]));
    }
    let mut b = Vec::with_capacity(a_idx.len());
    let mut gamma = Vec::with_capacity(a_idx.len());
    for (l, &j) in a_idx.iter().enumerate() {
        let q = fb.prime[j as usize];
        let mut al = vec![1u64];
        for (m, &i) in a_idx.iter().enumerate() {
            if m != l {
                al = mul_1(&al, u64::from(fb.prime[i as usize]));
            }
        }
        let alq = super::arith::rem_1(&al, q);
        let mut g = mul_mod(fb.sqrt[j as usize], inverse(alq, q), q);
        if g > q / 2 {
            g = q - g;
        }
        b.push(mul_1(&al, u64::from(g)));
        gamma.push(g);
    }
    (a, b, gamma)
}
