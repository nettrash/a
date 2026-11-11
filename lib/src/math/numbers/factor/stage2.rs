//! The second stage of `p - 1`, `p + 1` and the elliptic curve method: the primes `q` from `B1` to
//! `B2`, by Montgomery's baby-step giant-step continuation with prime pairing (1987).
//!
//! Each prime `q` is `k D + j` or `k D - j` for a multiple `k D` of a fixed `D`, the nearest, and an
//! odd `j < D/2` prime to `D`. With `x_m` a function of the multiple `m` of the point (or of the
//! power) that tells `m P` from `-m P` no more than `P` from `-P` does, `x_(kD) - x_j` is 0 modulo a
//! prime `p` when `(kD - j) P` or `(kD + j) P` is the identity modulo `p`: one product catches both
//! `kD - j` and `kD + j`, and serves the two primes when both are prime ("pairing"). The products of
//! all the differences are accumulated, and one greatest common divisor at the end of each block of
//! giant steps tells whether a prime was caught.
//!
//! The [`Plan`] holds, for each giant step `k`, which `j` are needed, as a bitmap: computed once by a
//! segmented sieve of `(B1, B2]` and kept for every curve or starting value, when it is not too
//! large, as sieving costs about as much as the products for small moduli. `D` is chosen among the
//! products of the first primes and their doubles by the cost of the baby steps, the giant steps
//! and the products, weighted for the method.
//!
//! For the Lucas sequences of `p - 1` and `p + 1`, `x_m = V_m = a^m + a^(-m)` for the element `a`
//! of the first stage: `V_(m+n) = V_m V_n - V_(m-n)` gives the baby steps and the giant steps with a
//! product each.
//!
//! The plan also chooses, per call from the bounds and the modulus's length, between this
//! continuation and those by fast polynomial arithmetic ([`mk`](super::mk) for `p - 1` and `p + 1`,
//! [`ecmfft`](super::ecmfft) for the curves) by their estimated times ([`Plan::mk`],
//! [`Plan::ecm_fft`]); those hand the stage back to this one when they cannot tell the primes
//! they caught apart.

use std::sync::OnceLock;

use super::ecm::rotate;
use super::ecmfft::FftPlan;
use super::field::Arith;
use super::mk::MkPlan;
use super::primes::primes_in;
use super::rho::{Gcd, gcd};

/// The candidates for `D`: the products of the first primes, and their doubles.
const DS: [u64; 9] = [6, 30, 60, 210, 420, 2310, 4620, 30030, 60060];

/// The bytes up to which the plan's bitmaps are kept for the whole of `(B1, B2]`.
const CACHE_BYTES: u64 = 32 << 20;

/// The giant steps between two greatest common divisors.
pub(super) const BLOCK: u64 = 128;

/// Which products the second stage takes, for `D`, `B1` and `B2`.
pub(super) struct Plan {
    /// `D`.
    pub(super) d: u64,
    /// The first stage's bound, at least `D / 2`, so that every prime of the second stage is above
    /// `D / 2`, with a giant step of at least 1.
    pub(super) b1: u64,
    /// The second stage's bound.
    pub(super) b2: u64,
    /// The `j`: the odd numbers below `D / 2` prime to `D`, increasing.
    pub(super) js: Vec<u64>,
    /// The position of each odd `j` below `D / 2` in `js`, by `j / 2`, or `u32::MAX`.
    index: Vec<u32>,
    /// The first giant step and the last.
    pub(super) k_first: u64,
    pub(super) k_last: u64,
    /// The words of a giant step's bitmap.
    pub(super) words: usize,
    /// The bitmaps of all the giant steps, made on the first use when they are to be kept.
    cached: OnceLock<Vec<u64>>,
    /// Whether they are to be kept.
    keep: bool,
    /// The parameters of the continuations by polynomials, made on first use.
    kept: OnceLock<Box<Kept>>,
}

/// The parameters of the continuations by polynomials a plan keeps, by the limbs of the modulus
/// up to [`KEPT_LIMBS`], each made on first use: about 8 KB, allocated when one is asked for.
struct Kept {
    /// Montgomery and Kruppa's, for `p - 1` and `p + 1`.
    mk: [[OnceLock<Option<MkPlan>>; 2]; KEPT_LIMBS],
    /// The elliptic curve method's.
    fft: [OnceLock<Option<FftPlan>>; KEPT_LIMBS],
}

/// The limbs of the moduli whose parameters of the continuations by polynomials a plan keeps.
const KEPT_LIMBS: usize = 32;

impl Plan {
    /// The plan for the bounds `b1 < b2`, `D` chosen for a method whose baby and giant steps cost
    /// `weight` products each; `None` when `b2 <= b1` or no prime lies between them above `D / 2`.
    pub(super) fn new(b1: u64, b2: u64, weight: u64) -> Option<Plan> {
        if b2 <= b1.max(3) {
            return None;
        }
        let d = choose_d(b1, b2, weight);
        let b1 = b1.max(d / 2);
        if b2 <= b1 {
            return None;
        }
        let half = d / 2;
        let mut js = Vec::new();
        let mut index = vec![u32::MAX; (half / 2 + 1) as usize];
        let mut j = 1;
        while j < half {
            if gcd_small(j, d) == 1 {
                index[(j / 2) as usize] = js.len() as u32;
                js.push(j);
            }
            j += 2;
        }
        let words = js.len().div_ceil(64);
        // Giant steps from that of the first prime above `b1` to that of the last below `b2`; the
        // nearest multiple `k D` of a prime `q > D/2` has `k >= 1`.
        let k_first = (b1 + 1 + half) / d;
        let k_last = ((u128::from(b2) + u128::from(half)) / u128::from(d)) as u64;
        let steps = k_last - k_first + 1;
        Some(Plan {
            d,
            b1,
            b2,
            js,
            index,
            k_first,
            k_last,
            words,
            cached: OnceLock::new(),
            keep: steps.saturating_mul(words as u64 * 8) <= CACHE_BYTES,
            kept: OnceLock::new(),
        })
    }

    /// Sets `out` to the bitmaps of the `count` giant steps from `k0`.
    pub(super) fn block(&self, k0: u64, count: u64, out: &mut Vec<u64>) {
        if !self.keep {
            return self.fill(k0, count, out);
        }
        let all = self.cached.get_or_init(|| {
            let mut all = Vec::new();
            self.fill(self.k_first, self.k_last - self.k_first + 1, &mut all);
            all
        });
        let w = self.words;
        let start = ((k0 - self.k_first) as usize) * w;
        out.clear();
        out.extend_from_slice(&all[start..start + count as usize * w]);
    }

    /// The kept parameters, allocated on first use.
    fn kept(&self) -> &Kept {
        self.kept.get_or_init(|| {
            Box::new(Kept {
                mk: [const { [const { OnceLock::new() }, const { OnceLock::new() }] }; KEPT_LIMBS],
                fft: [const { OnceLock::new() }; KEPT_LIMBS],
            })
        })
    }

    /// The parameters of Montgomery and Kruppa's continuation for `p - 1` (`pp1` false) or `p + 1`
    /// with a modulus of `k` limbs, when it costs less than this one: `None` at once when this one
    /// is too short for the other to pay.
    pub(super) fn mk(&self, k: usize, pp1: bool) -> Option<MkPlan> {
        let mm = super::mk::product_ns(k);
        let lucas = self.lucas_cost(k);
        // About the least the other costs, its fixed part (see `mk::cost`: 28 us and 350 products,
        // `f`, `h`, an evaluation of at least 16 points), so that its parameters are not searched
        // for when this one is clearly cheaper. Under Miri, whenever it applies: its cases are
        // small, and the unsafe code is there.
        if lucas < 600.0 * mm + 30_000.0 && !cfg!(miri) {
            return None;
        }
        let make =
            || MkPlan::new(self.b1, self.b2, k, pp1).filter(|p| p.cost < lucas || cfg!(miri));
        match self.kept().mk.get(k) {
            Some(cells) => cells[usize::from(pp1)].get_or_init(make).clone(),
            None => make(),
        }
    }

    /// The parameters of the elliptic curve method's continuation by polynomials for a modulus of
    /// `k` limbs, when it costs less than this one.
    pub(super) fn ecm_fft(&self, k: usize) -> Option<FftPlan> {
        let mm = super::mk::product_ns(k);
        let bsgs = self.ecm_cost(k);
        // As in [`Plan::mk`]: the other's fixed part (see `FftPlan::new`: 60 us, the baby steps, the
        // tree of `F` and its inverse) is about 600 products and 90 us at its least, with `D = 210`;
        // this threshold, above it, skips the search where only that least `D` could pay.
        if bsgs < 2000.0 * mm + 100_000.0 && !cfg!(miri) {
            return None;
        }
        let make = || FftPlan::new(self.b1, self.b2, k).filter(|p| p.cost < bsgs || cfg!(miri));
        match self.kept().fft.get(k) {
            Some(cell) => cell.get_or_init(make).clone(),
            None => make(),
        }
    }

    /// The time of this continuation for the elliptic curve method with a modulus of `k` limbs:
    /// about a product for each prime, the baby steps and the giant steps, an addition of points
    /// and its share of the normalisation each, and the sieve.
    pub(super) fn ecm_cost(&self, k: usize) -> f64 {
        let (b1, b2) = (self.b1 as f64, self.b2 as f64);
        let primes = (b2 / b2.ln() - b1 / b1.max(3.0).ln()).max(0.0);
        let giants = (self.k_last - self.k_first + 1) as f64;
        let mm = super::mk::product_ns(k);
        primes * (mm + 0.5)
            + (giants + self.js.len() as f64) * 9.0 * mm
            + self.d as f64 * 1.5 * mm
            + 0.05 * (b2 - b1)
    }

    /// The time of the continuation for `p - 1` and `p + 1` for a modulus of `k` limbs, in the
    /// nanoseconds of [`product_ns`](super::mk::product_ns): about a product for each prime of
    /// `(B1, B2]` (fewer as pairs serve two, but with the time of the sieve and the bitmaps), the
    /// steps, and the sieve, made once a plan, at about 0.05 ns a number.
    pub(super) fn lucas_cost(&self, k: usize) -> f64 {
        let (b1, b2) = (self.b1 as f64, self.b2 as f64);
        let primes = (b2 / b2.ln() - b1 / b1.max(3.0).ln()).max(0.0);
        let giants = (self.k_last - self.k_first + 1) as f64;
        let mm = super::mk::product_ns(k);
        // In one limb the four chains of products overlap.
        let per_prime = if k == 1 { 0.75 } else { 1.0 } * mm + 0.5;
        primes * per_prime + 0.5 * (giants + self.js.len() as f64) * mm + 0.05 * (b2 - b1)
    }

    /// Sieves the bitmaps of the `count` giant steps from `k0` into `out`.
    fn fill(&self, k0: u64, count: u64, out: &mut Vec<u64>) {
        let w = self.words;
        out.clear();
        out.resize(count as usize * w, 0);
        let half = self.d / 2;
        let lo = (k0 * self.d).saturating_sub(half).max(self.b1 + 1);
        let hi =
            ((k0 + count - 1) as u128 * self.d as u128 + half as u128).min(self.b2 as u128) as u64;
        if lo > hi {
            return;
        }
        // The giant step `k` of each prime, followed as the primes go up, without a division.
        let mut k = (lo + half) / self.d;
        let mut m = k * self.d;
        for q in primes_in(lo, hi) {
            while q > m + half {
                k += 1;
                m += self.d;
            }
            let j = q.abs_diff(m);
            let i = self.index[(j / 2) as usize];
            debug_assert!(i != u32::MAX);
            let s = (k - k0) as usize * w;
            out[s + (i / 64) as usize] |= 1 << (i % 64);
        }
    }
}

/// `D` for the bounds, by the costs of the baby steps (about `D/4` of the chain and `phi(D)/2`
/// kept), of the giant steps, both `weight` products each, of the first stage from `B1` to `D/2`
/// when `D/2` is larger, and of the products, fewer as more of them serve two primes.
fn choose_d(b1: u64, b2: u64, weight: u64) -> u64 {
    let range = (b2 - b1) as f64;
    let mid = ((b1 as f64 + b2 as f64) / 2.0).max(3.0);
    let primes = range / mid.ln();
    let mut best = (f64::MAX, DS[0]);
    for &d in &DS {
        let phi = totient(d) as f64;
        let babies = d as f64 / 4.0 + phi / 2.0;
        let giants = range / d as f64;
        let first = (d as f64 / 2.0 - b1 as f64).max(0.0) * 1.44;
        // The chance that the partner of a prime is prime too.
        let both = ((d as f64 / phi) / mid.ln()).min(1.0);
        let pairs = primes * (1.0 - both / 2.0);
        let cost = weight as f64 * (babies + giants + first) + pairs;
        if cost < best.0 {
            best = (cost, d);
        }
    }
    best.1
}

/// Euler's totient of the small `d`.
fn totient(d: u64) -> u64 {
    let (mut t, mut m, mut p) = (d, d, 2);
    while m > 1 {
        if m % p == 0 {
            t = t / p * (p - 1);
            while m % p == 0 {
                m /= p;
            }
        }
        p += 1;
    }
    t
}

/// The greatest common divisor of small numbers.
fn gcd_small(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// `(V_e, V_(e+1))` of the Lucas sequence `V_0 = 2`, `V_1 = p`, `V_(m+n) = V_m V_n - V_(m-n)`, by
/// the ladder on the bits of `e` (limbs, the least significant first): a square and a product a
/// bit. `two` is the residue of 2.
pub(super) fn lucas_ladder<A: Arith>(f: &A, p: &A::V, two: &A::V, e: &[u64]) -> (A::V, A::V) {
    let mut v0 = two.clone();
    let mut v1 = p.clone();
    let mut t = f.zero();
    let bits = 64 * e.len() as u64 - u64::from(e.last().map_or(64, |l| l.leading_zeros()));
    for i in (0..bits).rev() {
        let bit = (e[(i / 64) as usize] >> (i % 64)) & 1 == 1;
        // `V_(2k+1) = V_k V_(k+1) - P`, and `V_(2k) = V_k^2 - 2` or `V_(2k+2) = V_(k+1)^2 - 2`.
        f.mul(&mut t, &v0, &v1);
        f.sub_in(&mut t, p);
        f.cswap(bit, &mut v0, &mut v1);
        f.sqr_in(&mut v0);
        f.sub_in(&mut v0, two);
        // `v0` is the square; the product goes to `v1`, then both back in their places.
        core::mem::swap(&mut v1, &mut t);
        f.cswap(bit, &mut v0, &mut v1);
    }
    (v0, v1)
}

/// The multipliers of Montgomery's PRAC: the golden ratio's inverse and nearby values, the one
/// whose chain for `k` costs least taken.
const PRAC_V: [f64; 10] = [
    0.618_033_988_749_894_9,
    0.723_606_797_749_979,
    0.580_178_728_295_464_1,
    0.632_839_806_088_706_3,
    0.612_429_949_509_495,
    0.620_181_980_807_415_8,
    0.617_214_616_534_403_9,
    0.618_347_119_656_228,
    0.617_914_406_528_817_9,
    0.618_079_668_469_895_8,
];

/// The steps of PRAC's chain for `k` with the multiplier `v`, a product or a square each, or
/// `u64::MAX` when `v` does not suit `k`.
fn prac_cost(k: u64, v: f64) -> u64 {
    let r = (k as f64 * v + 0.5) as u64;
    if r >= k {
        return u64::MAX;
    }
    let (mut d, mut e) = (k - r, 2 * r - k);
    let mut c = 2;
    while d != e {
        if d < e {
            core::mem::swap(&mut d, &mut e);
        }
        if 4 * d <= 5 * e && (d + e) % 3 == 0 {
            (d, e) = (2 * d - e, 2 * e - d);
            c += 3;
        } else if 4 * d <= 5 * e && (d - e) % 6 == 0 {
            d = (d - e) / 2;
            c += 2;
        } else if d <= 4 * e {
            d -= e;
            c += 1;
        } else if (d + e) % 2 == 0 {
            d = (d - e) / 2;
            c += 2;
        } else if d % 2 == 0 {
            d /= 2;
            c += 2;
        } else if d % 3 == 0 {
            d = d / 3 - e;
            c += 4;
        } else if (d + e) % 3 == 0 {
            d = (d - 2 * e) / 3;
            c += 4;
        } else if (d - e) % 3 == 0 {
            d = (d - e) / 3;
            c += 4;
        } else {
            e /= 2;
            c += 2;
        }
    }
    c
}

/// `V_k(P)` for the residue `a` of `P` and an odd `k >= 3`, in `a`, by Montgomery's PRAC chain
/// ("Evaluating recurrences of form X_(m+n) = f(X_m, X_n, X_(m-n)) via Lucas chains", 1983):
/// about `1.5 log2 k` products and squares against `2 log2 k` for the ladder. `t` holds five
/// working residues, `two` the residue of 2.
pub(super) fn lucas_prac<A: Arith>(f: &A, a: &mut A::V, k: u64, two: &A::V, t: &mut [A::V; 4]) {
    lucas_prac_with(f, a, k, two, t, prac_choice(k));
}

/// The multiplier of [`PRAC_V`] whose chain for `k` costs least, by its index.
pub(super) fn prac_choice(k: u64) -> u8 {
    let mut best = (u64::MAX, 0u8);
    for (i, &v) in PRAC_V.iter().enumerate() {
        let c = prac_cost(k, v);
        if c < best.0 {
            best = (c, i as u8);
        }
    }
    best.1
}

/// [`lucas_prac`] with the multiplier of index `choice`, from [`prac_choice`].
pub(super) fn lucas_prac_with<A: Arith>(
    f: &A,
    a: &mut A::V,
    k: u64,
    two: &A::V,
    t: &mut [A::V; 4],
    choice: u8,
) {
    // `x = y z - w`, `V_(m+n) = V_m V_n - V_(m-n)`, and `x = y^2 - 2`.
    let add3 = |x: &mut A::V, y: &A::V, z: &A::V, w: &A::V| {
        f.mul(x, y, z);
        f.sub_in(x, w);
    };
    let dup = |x: &mut A::V, y: &A::V| {
        f.sqr(x, y);
        f.sub_in(x, two);
    };
    let r = (k as f64 * PRAC_V[usize::from(choice)] + 0.5) as u64;
    let (mut d, mut e) = (k - r, 2 * r - k);
    let [b, c, tt, t2] = t;
    b.clone_from(a);
    c.clone_from(a);
    dup(tt, a);
    core::mem::swap(a, tt);
    while d != e {
        if d < e {
            core::mem::swap(&mut d, &mut e);
            core::mem::swap(a, b);
        }
        if 4 * d <= 5 * e && (d + e) % 3 == 0 {
            (d, e) = (2 * d - e, 2 * e - d);
            add3(tt, a, b, c);
            add3(t2, tt, a, b);
            let mut nb = f.zero();
            add3(&mut nb, b, tt, a);
            *b = nb;
            core::mem::swap(a, t2);
        } else if 4 * d <= 5 * e && (d - e) % 6 == 0 {
            d = (d - e) / 2;
            add3(tt, a, b, c);
            core::mem::swap(b, tt);
            dup(tt, a);
            core::mem::swap(a, tt);
        } else if d <= 4 * e {
            d -= e;
            add3(tt, b, a, c);
            // `C = B`, `B = T`.
            core::mem::swap(c, b);
            core::mem::swap(b, tt);
        } else if (d + e) % 2 == 0 {
            d = (d - e) / 2;
            add3(tt, b, a, c);
            core::mem::swap(b, tt);
            dup(tt, a);
            core::mem::swap(a, tt);
        } else if d % 2 == 0 {
            d /= 2;
            add3(tt, c, a, b);
            core::mem::swap(c, tt);
            dup(tt, a);
            core::mem::swap(a, tt);
        } else if d % 3 == 0 {
            d = d / 3 - e;
            dup(tt, a);
            add3(t2, a, b, c);
            let mut na = f.zero();
            add3(&mut na, tt, a, a);
            *a = na;
            let mut nc = f.zero();
            add3(&mut nc, tt, t2, c);
            *c = nc;
            core::mem::swap(b, c);
        } else if (d + e) % 3 == 0 {
            d = (d - 2 * e) / 3;
            add3(tt, a, b, c);
            add3(t2, tt, a, b);
            core::mem::swap(b, t2);
            dup(tt, a);
            add3(t2, a, tt, a);
            core::mem::swap(a, t2);
        } else if (d - e) % 3 == 0 {
            d = (d - e) / 3;
            add3(tt, a, b, c);
            add3(t2, c, a, b);
            core::mem::swap(c, t2);
            core::mem::swap(b, tt);
            dup(tt, a);
            add3(t2, a, tt, a);
            core::mem::swap(a, t2);
        } else {
            e /= 2;
            add3(tt, c, b, a);
            core::mem::swap(c, tt);
            dup(tt, b);
            core::mem::swap(b, tt);
        }
    }
    add3(tt, a, b, c);
    core::mem::swap(a, tt);
}

/// The second stage for the Lucas sequence of `V_1 = p`: a factor of `n`, or `None`.
pub(super) fn lucas_stage2<A: Arith>(f: &A, p: &A::V, plan: &Plan) -> Option<Vec<u64>> {
    let two = f.small(2);
    let half = plan.d / 2;
    // The baby steps: `V_j` for the odd `j` up to `D/2 + 1`, `V_(j+2) = V_j V_2 - V_(j-2)`.
    let mut v2 = f.zero();
    f.sqr(&mut v2, p);
    f.sub_in(&mut v2, &two);
    let mut babies: Vec<A::V> = Vec::with_capacity(plan.js.len());
    let (mut prev, mut cur) = (p.clone(), p.clone());
    let mut t = f.zero();
    let mut j = 1;
    // `V_(D/2 - 1)` and `V_(D/2 + 1)` for `V_D` when `D/2` is even, `V_(D/2)` when it is odd.
    let (mut below, mut at) = (f.zero(), f.zero());
    loop {
        if j < half && plan.index_of(j) {
            babies.push(cur.clone());
        }
        if j + 1 == half {
            below.clone_from(&cur);
        }
        if j == half {
            at.clone_from(&cur);
        }
        if j > half {
            break;
        }
        f.mul(&mut t, &cur, &v2);
        f.sub_in(&mut t, &prev);
        // `prev, cur = cur, t`, the old `prev` left in `t`.
        core::mem::swap(&mut prev, &mut cur);
        core::mem::swap(&mut cur, &mut t);
        j += 2;
    }
    let mut vd = f.zero();
    if half % 2 == 1 {
        f.sqr(&mut vd, &at);
        f.sub_in(&mut vd, &two);
    } else {
        // `V_D = V_(D/2 + 1) V_(D/2 - 1) - V_2`, `cur` being `V_(D/2 + 1)`.
        f.mul(&mut vd, &cur, &below);
        f.sub_in(&mut vd, &v2);
    }
    // The giant steps: `V_(kD) = V_k(V_D)`, from `k_first` by the ladder, then the recurrence.
    let (mut g0, mut g1) = lucas_ladder(f, &vd, &two, &[plan.k_first]);
    // Four products accumulated in turn, so that their chains overlap.
    let mut acc = [f.one(), f.one(), f.one(), f.one()];
    let mut d = f.zero();
    let mut bitmap = Vec::new();
    let mut k = plan.k_first;
    while k <= plan.k_last {
        let count = BLOCK.min(plan.k_last - k + 1);
        plan.block(k, count, &mut bitmap);
        let checkpoint = (g0.clone(), g1.clone());
        for s in 0..count as usize {
            for w in 0..plan.words {
                let mut bits = bitmap[s * plan.words + w];
                while bits != 0 {
                    let i = 64 * w + bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    f.sub(&mut d, &g0, &babies[i]);
                    f.mul_in(&mut acc[0], &d);
                    rotate(&mut acc);
                }
            }
            // `V_((k+1)D) = V_(kD) V_D - V_((k-1)D)`, as `g1` holds the next.
            f.mul(&mut t, &g1, &vd);
            f.sub_in(&mut t, &g0);
            core::mem::swap(&mut g0, &mut g1);
            core::mem::swap(&mut g1, &mut t);
        }
        match gcd(f, super::ecm::combine(f, &mut acc)) {
            Gcd::One => {}
            Gcd::Factor(g) => return Some(g),
            Gcd::Whole => {
                // Over again, a greatest common divisor a product.
                let (mut h0, mut h1) = checkpoint;
                for s in 0..count as usize {
                    for w in 0..plan.words {
                        let mut bits = bitmap[s * plan.words + w];
                        while bits != 0 {
                            let i = 64 * w + bits.trailing_zeros() as usize;
                            bits &= bits - 1;
                            f.sub(&mut d, &h0, &babies[i]);
                            match gcd(f, &d) {
                                Gcd::One => {}
                                Gcd::Factor(g) => return Some(g),
                                Gcd::Whole => return None,
                            }
                        }
                    }
                    f.mul(&mut t, &h1, &vd);
                    f.sub_in(&mut t, &h0);
                    core::mem::swap(&mut h0, &mut h1);
                    core::mem::swap(&mut h1, &mut t);
                }
                return None;
            }
        }
        k += count;
    }
    None
}

impl Plan {
    /// Whether the odd `j < D/2` is one of the baby steps.
    #[inline]
    pub(super) fn index_of(&self, j: u64) -> bool {
        self.index[(j / 2) as usize] != u32::MAX
    }
}
